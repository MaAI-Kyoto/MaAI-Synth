"""対話データ自動生成パイプライン

ステージ（★2026-11-20 旧 Stage3 を廃止し、旧 Stage4 = TTS を Stage3 と呼ぶ）:
    Stage 1  metadata  メタデータ生成
    Stage 2  dialogue  対話生成 → 最後に音声対話スキーマへ変換して voice_dialogues.jsonl を書く
                       （旧 Stage3 の passthrough 変換。LLM は呼ばない・決定的・1万件で数秒）
    Stage 3  TTS       generate_tts.py / run_tts_pool.sh（このファイルでは回さない）

使い方:
    # Stage 1 + 2 を通しで実行（voice_dialogues.jsonl まで出る）
    python pipeline/main.py

    # Stage 1（メタデータ生成）のみ
    python pipeline/main.py --stage metadata

    # Stage 2（対話生成 → voice_dialogues.jsonl への変換）のみ（既存メタデータを使用）
    python pipeline/main.py --stage dialogue

    # 変換だけやり直す（audit_pipeline で voice_dialogues の行だけ捨てたあと等。互換用）
    python pipeline/main.py --stage voice

    # 設定ファイル指定
    python pipeline/main.py --config my_config.yaml

    # 目標数・コスト上限を上書き
    python pipeline/main.py --target 500 --max-cost 10.0
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from cost_tracker import CostTracker
from generate_dialogues import run_dialogue_generation
from generate_metadata import run_metadata_generation


def setup_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str, _raw: bool = False) -> dict:
    """`extends: <path>` があればそれをベースに deep-merge する（設定の継承）。
    最後に TTS 方式の名前（tts.engine: pocket-ja-nc-v1 など）を tts_presets/ の中身に展開する（tts_presets.apply・2026-11-15）。"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        from pathlib import Path
        cfg = _deep_merge(load_config(str(Path(config_path).parent / base), _raw=True), cfg)
    if _raw:
        return cfg
    import tts_presets
    return tts_presets.apply(cfg)


async def convert_voice_dialogues(config: dict, cost_tracker: CostTracker) -> int:
    """Stage 2 の仕上げ: dialogues.jsonl → voice_dialogues.jsonl（旧 Stage3 の passthrough）。

    ★2026-11-20 旧 Stage3 は全 config が voice.mode: passthrough（LLM を呼ばないスキーマ変換）
    だけになっていたので廃止し、Stage 2 の最後に必ずこれを回す。出力（voice_dialogues.jsonl・
    voice_dialogues_skipped.jsonl）の中身と規則は旧 Stage3 と同一（Stage3=TTS の入力は不変）。
    変換済みの id は飛ばすので何度呼んでもよい。三人対話（main_tri.py）は
    generate_voice_passthrough.build_voice_turns を差し替えてから呼ぶので、そのまま効く。
    """
    mode = (config.get("voice") or {}).get("mode")
    if mode not in (None, "passthrough"):
        # LLM で台本を後付けする旧モード（enrich / generate_voice）は廃止した。
        raise SystemExit(f"voice.mode: {mode} は廃止しました（旧 Stage3）。voice.mode を消すか"
                         " passthrough にしてください。")
    from generate_voice_passthrough import run_voice_generation
    return await run_voice_generation(config, cost_tracker)


async def run_dialogue_stage(config: dict, cost_tracker: CostTracker) -> tuple[int, int]:
    """Stage 2 の入口: 対話生成 → voice_dialogues.jsonl への変換。(対話数, 変換済み数) を返す。

    main.py（--stage dialogue）と topup_failed_dialogues.py の両方から使う。
    run_dialogue_generation はモジュール属性を実行時に引く（main_tri.py が三人用に差し替える）。
    """
    dial_count = await run_dialogue_generation(config, cost_tracker)
    logger = logging.getLogger(__name__)
    logger.info(f"--- Stage 2 仕上げ: 音声対話スキーマへ変換（{config['voice']['output_file']}） ---")
    voice_count = await convert_voice_dialogues(config, cost_tracker)
    return dial_count, voice_count


async def main():
    parser = argparse.ArgumentParser(description="日本語対話データ自動生成")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument("--stage", choices=["metadata", "dialogue", "voice", "all"], default="all",
                        help="実行ステージ (default: all = metadata + dialogue)。dialogue は "
                             "voice_dialogues.jsonl への変換まで行う。voice は変換だけ（互換用）")
    parser.add_argument("--target", type=int, help="目標対話数（config上書き）")
    parser.add_argument("--max-cost", type=float, help="コスト上限USD（config上書き）")
    parser.add_argument("--limit", type=int,
                        help="dialogue/voice ステージで処理する件数の上限（試行用。"
                             "metadata の順序は固定なので、同じ値なら別バリアント間で"
                             "同じシナリオが揃う）")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target:
        config["target_dialogues"] = args.target
    if args.max_cost:
        config["max_cost_usd"] = args.max_cost
    if args.limit:
        config["limit"] = args.limit

    setup_logging(config["log_file"])
    logger = logging.getLogger(__name__)

    cost_tracker = CostTracker(
        pricing=config["pricing"],
        max_cost_usd=config["max_cost_usd"],
        cost_log_file=config["cost_log_file"],
    )

    prev = cost_tracker.summary()
    logger.info(f"設定: model={config['model']}, target={config['target_dialogues']}, max_cost=${config['max_cost_usd']}")
    logger.info(f"前回までの累計: リクエスト={prev['request_count']}回, 入力={prev['total_input_tokens']:,}tok, 出力={prev['total_output_tokens']:,}tok, コスト=${prev['total_cost_usd']}")

    if args.stage in ("metadata", "all"):
        logger.info("=== Stage 1: メタデータ生成 ===")
        meta_count = await run_metadata_generation(config, cost_tracker)
        logger.info(f"メタデータ生成結果: {meta_count}件 | 累計コスト: ${cost_tracker.summary()['total_cost_usd']}")

        if cost_tracker.exceeded():
            logger.warning("コスト上限に到達。Stage 2 をスキップします。")
            print_summary(cost_tracker, meta_count, 0)
            return

    if args.stage in ("dialogue", "all"):
        metadata_file = Path(config["metadata"]["output_file"])
        if not metadata_file.exists():
            logger.error(f"メタデータファイルが見つかりません: {metadata_file}")
            logger.error("先に --stage metadata を実行してください。")
            sys.exit(1)

        logger.info("=== Stage 2: 対話生成 → 音声対話スキーマへ変換 ===")
        dial_count, voice_count = await run_dialogue_stage(config, cost_tracker)
        logger.info(f"対話生成結果: {dial_count}件 / voice_dialogues {voice_count}件 | "
                    f"累計コスト: ${cost_tracker.summary()['total_cost_usd']}")

    # 旧 Stage3（変換だけ）。Stage 2 が最後に同じ変換を回すので通常は不要。互換用に残す。
    if args.stage == "voice":
        dialogue_file = Path(config["dialogue"]["output_file"])
        if not dialogue_file.exists():
            logger.error(f"対話ファイルが見つかりません: {dialogue_file}")
            logger.error("先に --stage dialogue を実行してください。")
            sys.exit(1)

        logger.info("=== 音声対話スキーマへの変換のみ（旧 Stage3。通常は --stage dialogue に含まれる） ===")
        voice_count = await convert_voice_dialogues(config, cost_tracker)
        logger.info(f"変換結果: {voice_count}件")

    summary = cost_tracker.summary()
    print("\n" + "=" * 50)
    print("生成完了")
    print(f"  リクエスト数: {summary['request_count']}")
    print(f"  入力トークン: {summary['total_input_tokens']:,}")
    print(f"  出力トークン: {summary['total_output_tokens']:,}")
    print(f"  合計コスト:   ${summary['total_cost_usd']:.4f}")
    print("=" * 50)


def print_summary(cost_tracker, meta_count, dial_count):
    summary = cost_tracker.summary()
    print("\n" + "=" * 50)
    print("生成完了（コスト上限到達により中断）")
    print(f"  メタデータ: {meta_count}件")
    print(f"  対話:       {dial_count}件")
    print(f"  合計コスト: ${summary['total_cost_usd']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
