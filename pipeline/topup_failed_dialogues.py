"""Stage 2（対話生成・変換）で恒久スキップされたぶんを作り直し、target 件に収束させるループ。

★2026-11-20 旧 Stage3（passthrough 変換）は廃止して Stage 2 の最後に組み込んだ
（main.run_dialogue_stage）。以下の「変換」はその変換のこと。
skip ファイル名・破棄の規則は旧 Stage3 のときと同じ。TTS（旧 Stage4）はいま Stage3 と呼ぶ。

背景:
    generate_dialogues.py は、品質ゲート（伏せ字/自分の名前/文末句点 等）を
    dialogue.retry_max_attempts 回すべて通過できなかった item を
    `<dialogues>_skipped.jsonl` に記録して恒久スキップする（接続等の一時失敗は
    記録せず None を返して再試行）。原因はメタデータのシナリオ依存が多いため、
    「そのメタデータを捨てて別のを作り直す」方が件数を保てて同じ地雷も踏みにくい。

    Stage 1（run_metadata_generation）は target_dialogues までの不足分を追記する
    増分方式なので、metadata.jsonl から失敗分を削れば Stage 1 の再実行で同数の
    新規メタデータ（新しい uuid）が補充される。本スクリプトはそれをループで回す。

    変換（generate_voice_passthrough.py）も同じ規則で
    `<voice_dialogues>_skipped.jsonl` に恒久スキップを記録する（ターン0・空テキスト・
    話者不明・話者2人未満。後段の generate_tts.py が扱えない形）。こちらの原因は
    メタデータではなく Stage 2 の出力なので、**まず対話だけを捨てて Stage 2 に
    引き直させる**（メタデータは残す）。同じメタデータで2回目も変換で弾かれたら、
    そのメタデータ自体が地雷とみなしてメタデータごと捨てる。

各イテレーション:
    1. Stage 1: metadata を target 件まで補充（既に満杯なら何もしない）
    2. Stage 2: 対話生成。ゲート全滅の item は skip ファイルに記録される
    3. 変換（Stage 2 の仕上げ）: 音声対話スキーマへの変換。弾かれた item は skip ファイルに
       記録される（--no-voice で省ける）
    4. どちらの skip も無ければ収束として終了
    5. skip 分を退避のうえ破棄（Stage2 失敗＝メタデータを、変換で弾かれた＝対話を、
       変換で2回目＝両方）し、skip ファイルをクリア
       → 次イテレーションの Stage 1 / Stage 2 が同数を作り直す

使い方:
    python topup_failed_dialogues.py --config config_main_v1.yaml
    python topup_failed_dialogues.py --config <cfg> --max-iters 8 --max-cost 50
    python topup_failed_dialogues.py --config <cfg> --no-voice   # 従来どおり Stage1-2 のみ

注意:
    - コスト上限に達したら安全に停止する（cost_tracker を全イテレーションで共有）。
    - max-iters 到達時にまだ skip が残る場合は、その旨を報告して終了する
      （特定ジャンル等が繰り返しゲートを誘発するケースの無限ループ防止）。
    - config に limit が入っていると1回の Stage 2 が先頭 limit 件しか作らないので、
      target には収束しない（skip の解消だけを行う）。収束させたいときは外す。
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from cost_tracker import CostTracker
from generate_dialogues import run_dialogue_generation
from generate_metadata import run_metadata_generation
from generate_voice_passthrough import skip_file_for as _skip_file_for
from main import load_config, run_dialogue_stage, setup_logging

logger = logging.getLogger(__name__)


def _read_ids(path: Path) -> list[str]:
    ids = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)["id"])
    return ids


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for ln in lines:
            f.write(ln + "\n")
    os.replace(tmp, path)


def _retire_records(src: Path, ids: set[str], stamp: str) -> int:
    """src（jsonl）から ids のレコードを取り除き、`<stem>_discarded.jsonl` へ退避する。

    銀の弾丸で消さない・追跡可能にするため、消す前に必ず退避する。戻り値は退避件数。
    """
    if not ids or not src.exists():
        return 0
    kept, discarded = [], []
    with open(src) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            (discarded if json.loads(s).get("id") in ids else kept).append(s)
    if discarded:
        with open(src.parent / f"{src.stem}_discarded.jsonl", "a") as f:
            for s in discarded:
                rec = json.loads(s)
                rec["_discarded_at"] = stamp
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _atomic_write_lines(src, kept)
    return len(discarded)


def _archive_and_clear_skip(skip_file: Path, stamp: str) -> None:
    """skip ファイルの内容を `<stem>_history.jsonl` に退避してから空にする。"""
    if not skip_file.exists():
        return
    with open(skip_file) as f:
        content = f.read()
    if content.strip():
        with open(skip_file.parent / f"{skip_file.stem}_history.jsonl", "a") as f:
            for line in content.splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec["_reset_at"] = stamp
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _atomic_write_lines(skip_file, [])


def discard_and_reset(config: dict, stage2_ids: set[str], stage3_ids: set[str]) -> tuple[int, int]:
    """失敗分を破棄して skip ファイルをクリアする。戻り値: (破棄メタデータ数, 破棄対話数)。

    - Stage 2 の失敗 → メタデータを破棄（対話は元から存在しない）
    - 変換で弾かれた → 対話を破棄し、メタデータは残して Stage 2 に引き直させる
    - 変換で2回目の失敗（履歴に同じ id がある）→ メタデータも一緒に破棄
    """
    metadata_file = Path(config["metadata"]["output_file"])
    dialogue_file = Path(config["dialogue"]["output_file"])
    dialogue_skip = _skip_file_for(dialogue_file)
    voice_skip = _skip_file_for(Path(config["voice"]["output_file"]))
    stamp = datetime.now().isoformat(timespec="seconds")

    # 履歴の退避より先に「2回目か」を判定する（今回分はまだ履歴に入っていない）。
    repeat = stage3_ids & set(_read_ids(voice_skip.parent / f"{voice_skip.stem}_history.jsonl"))
    if repeat:
        logger.info(f"変換で2回目の失敗 {len(repeat)}件。メタデータごと作り直します。")

    n_meta = _retire_records(metadata_file, stage2_ids | repeat, stamp)
    n_dial = _retire_records(dialogue_file, stage3_ids, stamp)
    _archive_and_clear_skip(dialogue_skip, stamp)
    _archive_and_clear_skip(voice_skip, stamp)
    return n_meta, n_dial


async def run(config: dict, cost_tracker: CostTracker, max_iters: int,
              with_voice: bool = True) -> None:
    target = config["target_dialogues"]
    dialogue_skip = _skip_file_for(Path(config["dialogue"]["output_file"]))
    voice_skip = _skip_file_for(Path(config["voice"]["output_file"]))

    s2_ids, s3_ids = set(), set()
    for it in range(1, max_iters + 1):
        logger.info(f"===== top-up iteration {it}/{max_iters} =====")

        logger.info("--- Stage 1: メタデータ補充 ---")
        await run_metadata_generation(config, cost_tracker)
        if cost_tracker.exceeded():
            logger.warning("コスト上限に到達。停止します。")
            return

        voice_count, s3_ids = None, set()
        if with_voice:
            # 対話生成 → voice_dialogues.jsonl への変換（旧 Stage3。LLM を呼ばない）
            logger.info("--- Stage 2: 対話生成 → 音声対話スキーマへ変換 ---")
            dial_count, voice_count = await run_dialogue_stage(config, cost_tracker)
            s3_ids = set(_read_ids(voice_skip))
        else:
            logger.info("--- Stage 2: 対話生成（変換なし: --no-voice） ---")
            dial_count = await run_dialogue_generation(config, cost_tracker)
        s2_ids = set(_read_ids(dialogue_skip))

        logger.info(
            f"iteration {it} 結果: 対話 {dial_count}/{target}件"
            + (f"、音声対話 {voice_count}/{target}件" if voice_count is not None else "")
            + f"、今回のスキップ Stage2 {len(s2_ids)}件 / 変換 {len(s3_ids)}件")

        if not s2_ids and not s3_ids:
            logger.info("スキップ 0件。収束しました。完了。")
            done = voice_count if voice_count is not None else dial_count
            if done < target:
                logger.warning(
                    f"ただし {done}/{target}件で target に未達です。config の limit が"
                    "効いている可能性があります（外して再実行してください）。")
            return
        if cost_tracker.exceeded():
            logger.warning("コスト上限に到達。破棄・再生成せず停止します。")
            return

        n_meta, n_dial = discard_and_reset(config, s2_ids, s3_ids)
        logger.info(
            f"メタデータ {n_meta}件 / 対話 {n_dial}件 を破棄。次イテレーションの "
            f"Stage 1・Stage 2 が同数を作り直します。")

    # ここに来たら max_iters を使い切ってもまだ skip が残っている
    # （skip ファイル自体は直前の discard_and_reset でクリア済みなので、最後の
    #   イテレーションで観測した件数をそのまま報告する）
    logger.warning(
        f"max-iters={max_iters} 到達。最後のイテレーションで未解決のスキップが "
        f"Stage2 {len(s2_ids)}件 / 変換 {len(s3_ids)}件 出ています。"
        "特定条件が繰り返しゲートを誘発している可能性があります。"
        "--max-iters を増やすか、当該メタデータの傾向を確認してください。")


async def _amain() -> None:
    p = argparse.ArgumentParser(
        description="Stage2（対話生成・変換）の失敗分を破棄→作り直しを収束まで反復")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-iters", type=int, default=5,
                   help="破棄・再生成の最大反復回数（無限ループ防止、default 5）")
    p.add_argument("--target", type=int, help="目標対話数（config上書き）")
    p.add_argument("--max-cost", type=float, help="コスト上限USD（config上書き）")
    p.add_argument("--no-voice", action="store_true",
                   help="Stage 2 の最後の voice_dialogues.jsonl への変換（旧 Stage3）を省く")
    args = p.parse_args()

    config = load_config(args.config)
    if args.target is not None:
        config["target_dialogues"] = args.target
    if args.max_cost is not None:
        config["max_cost_usd"] = args.max_cost

    setup_logging(config["log_file"])
    cost_tracker = CostTracker(
        pricing=config["pricing"],
        max_cost_usd=config["max_cost_usd"],
        cost_log_file=config["cost_log_file"],
    )
    await run(config, cost_tracker, args.max_iters, with_voice=not args.no_voice)
    cost_tracker.save_log()


if __name__ == "__main__":
    asyncio.run(_amain())
