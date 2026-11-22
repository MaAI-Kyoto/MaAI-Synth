"""Stage 2 の仕上げ（旧 Stage 3 の passthrough モード）: LLM を呼ばず、対話を音声対話スキーマに変換する。

★2026-11-20 旧 Stage3 は廃止。main.run_dialogue_stage が Stage 2（対話生成）の直後に必ずこれを呼ぶ
（main.py --stage dialogue / topup_failed_dialogues.py）。出力の形と規則は旧 Stage3 のときと同一。
TTS（旧 Stage4）はいま Stage3 と呼ぶ。以下の説明の「Stage 3」「この工程」はこの変換のこと。

Stage 2 を enriched モード（`dialogue.mode: enriched`）で回すと、フィラー・言い淀み・
冒頭リアクション・終助詞・読点まで込みの台本がそのまま出てくる。つまり従来 Stage 3
(generate_voice_enrich.py) が LLM で後付けしていた仕事は済んでいるので、この工程に
残るのは後段 (generate_tts.py / retime / refine) が要求するスキーマへの変換だけになる。

やること（すべて決定的。LLM もコストも発生しない）:
  - utterance → text、type は全て speech
  - 同一話者の連続ターンに continued: true（retime が間を詰める）
  - start_offset / duration の見積り（本番は --retime の VAP が時刻を決めるので、
    ここで入れるのは retime を使わない場合と LLM フォールバック用の値）
  - speaker_profiles（性別）と speaking_rate は、enriched モードの Stage 2 が返した
    speakers / speaking_rate を使う。無ければ None のまま generate_tts の
    role/名前からの推定にフォールバックする。

後段が扱えない変換結果（ターン0・空テキスト・話者不明・話者2人未満）は書き出さず、
`<voice出力>_skipped.jsonl` に理由付きで記録して恒久スキップする（Stage 2 と同じ規則）。
再実行で焼き直さないためと、topup_failed_dialogues.py が「その対話を Stage 2 から
作り直す」ためのマーカーを兼ねる。

出力スキーマは旧 enrich モードと同一。voice.mode は無しか passthrough のみ（他は main.convert_voice_dialogues が止める）。
"""

import json
import logging
import re
from pathlib import Path

from cost_tracker import CostTracker
from generate_voice import (
    add_absolute_timeline,
    build_speaker_profiles,
    normalize_metadata_speakers,
    normalize_turn_speakers,
    parse_speaking_rate,
)

logger = logging.getLogger(__name__)

# 読点・句点・長音を除いた実質文字数（発話長の見積りに使う）
_CORE_RE = re.compile(r"[、。？ー\s]")

DEFAULT_START_OFFSET = 0.3     # 直前ターン終了からの間（秒）
CONTINUED_OFFSET = 0.3         # 同一話者の継ぎ足しの間（秒）
MORA_PER_SEC = 6.5             # 発話長の見積り（enrich と同じ係数）
MAX_DURATION = 8.0


def _core_len(s: str) -> int:
    return len(_CORE_RE.sub("", s or ""))


def estimate_duration(text: str) -> float:
    return round(min(max(0.4, _core_len(text) / MORA_PER_SEC), MAX_DURATION), 2)


def build_voice_turns(input_turns: list) -> list:
    """Stage 2 のターン列を voice_dialogue.turns に変換する。"""
    out = []
    prev_speaker = None
    prev_rp = False
    for t in input_turns:
        text = t["utterance"]
        turn = {
            "speaker": t["speaker"],
            "text": text,
            "type": "speech",
            "start_offset": DEFAULT_START_OFFSET,
            "duration": estimate_duration(text),
        }
        if t.get("yomi"):
            # 読みの注記（「方」のかた／ほう等）。fish に渡す直前だけ使う（tts_reading.apply_yomi）
            turn["yomi"] = t["yomi"]
        if t.get("ruby"):
            # 発話全体の読み（yomi_ruby）。本文（src）が変わっていれば Stage4 で付け直す
            turn["ruby"] = t["ruby"]
        # ★2026-11-10 重なりの場面（rp_v1。rp_plan.py）: 言いかけの残り（unsaid）と重ねの種類（overlap）を写す。
        #   Stage4 は unsaid 込みで合成してから言いかけの位置で切り（rp_audio.py）、overlap のターンを
        #   相手の音声に重ねて置く（retime_maai の rp-* 分岐）。重なりのターンは継ぎ足し（continued）にしない。
        rp = False
        for k in ("unsaid", "overlap"):
            if t.get(k):
                turn[k] = t[k]
                rp = True
        if rp:
            pass
        elif t.get("pause_before_sec") is not None:
            # ユーザの長い沈黙（Stage2 の dialogue.silence）。継ぎ足しではないので continued に
            # しない。秒数は retime_maai.MaaiRetimer.plan がそのまま置く（start_offset は
            # 旧来配置のフォールバック用に同じ値にしておく）。
            turn["pause_before_sec"] = float(t["pause_before_sec"])
            turn["start_offset"] = turn["pause_before_sec"]
        elif t["speaker"] == prev_speaker and not prev_rp:
            # 2文以上のターンを Stage 2 が機械分割した継ぎ足し。間を詰める。
            # ★2026-11-12 重なりの場面の言いかけ・重ねたターンのすぐ後は継ぎ足しにしない（相手が先の同時発話では
            #   イノウエの言いかけのあとにイノウエの返しが続く。継ぎ足しにすると refine の merge で言いかけの切り口が消える）
            turn["continued"] = True
            turn["start_offset"] = CONTINUED_OFFSET
        prev_rp = rp
        prev_speaker = t["speaker"]
        out.append(turn)
    add_absolute_timeline(out)
    return out


def convert_record(record: dict) -> dict:
    """dialogues.jsonl の1レコードを voice_dialogues.jsonl の1レコードにする。"""
    metadata = normalize_metadata_speakers(record["metadata"])
    dialogue = record["dialogue"]
    turns = normalize_turn_speakers(dialogue["turns"])
    return {
        "id": record["id"],
        "metadata": metadata,
        # enriched モードの Stage 2 が推定した性別・話速。無ければ後段が名前・役割から推定。
        "speaker_profiles": build_speaker_profiles(metadata, dialogue.get("speakers")),
        "speaking_rate": parse_speaking_rate(dialogue.get("speaking_rate")),
        "voice_dialogue": {"turns": build_voice_turns(turns)},
    }


def skip_file_for(output_file: Path) -> Path:
    """恒久スキップの記録先。Stage 2（`<dialogues>_skipped.jsonl`）と同じ規則。"""
    return output_file.parent / f"{output_file.stem}_skipped.jsonl"


def validate_voice_record(record: dict) -> str | None:
    """後段（generate_tts.py）が扱えない変換結果を弾く。理由を返す（問題なければ None）。

    Stage 2 のゲートを通っても、まれに壊れたターン（speaker が null で text が空 等）が
    混じる（実測 5,000件に1件）。素通しすると Stage 4 が黙って落とすか無音を合成するので、
    ここで弾いて topup_failed_dialogues.py に Stage 2 からやり直させる。
    判定はすべて決定的で、LLM は呼ばない。
    """
    turns = (record.get("voice_dialogue") or {}).get("turns") or []
    if not turns:
        return "ターン0"
    names = {p.get("name") for p in record.get("speaker_profiles") or []}
    for t in turns:
        if not (t.get("text") or "").strip():
            return "空テキストのターン"
        if t.get("speaker") not in names:
            return f"話者不明のターン({t.get('speaker')!r})"
    if len({t["speaker"] for t in turns}) < 2:
        # generate_tts.py が「話者が2人未満」として落とす条件を先取りする
        return "話者2人未満"
    return None


async def run_voice_generation(config: dict, cost_tracker: CostTracker) -> int:
    """LLM を使わないので cost_tracker は素通し（シグネチャを他モードとそろえるため受け取る）。"""
    input_file = Path(config["dialogue"]["output_file"])
    output_file = Path(config["voice"]["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        logger.error(f"入力の対話ファイルが見つかりません: {input_file}")
        logger.error("先に Stage 2（対話生成）を実行してください。")
        return 0

    dialogues = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    total = len(dialogues)

    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    skip_file = skip_file_for(output_file)
    skipped_ids = set()
    if skip_file.exists():
        with open(skip_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    skipped_ids.add(json.loads(line)["id"])

    pending = [d for d in dialogues
               if d.get("id") not in existing_ids and d.get("id") not in skipped_ids]
    logger.info(f"passthrough 変換開始: 全{total}件中、変換済み{len(existing_ids)}件、"
                f"恒久スキップ{len(skipped_ids)}件、未変換{len(pending)}件")

    limit = config.get("limit")
    if limit and len(pending) > limit:
        pending = pending[:limit]
        logger.info(f"--limit {limit}: 先頭{len(pending)}件のみ変換します。")

    if not pending:
        logger.info("未変換の対話はありません。")
        return len(existing_ids)

    converted = len(existing_ids)
    n_gender_missing = 0
    n_skipped = 0
    with open(output_file, "a") as f, open(skip_file, "a") as sf:
        for record in pending:
            try:
                out = convert_record(record)
            except (KeyError, TypeError) as e:
                reason = f"変換エラー: {type(e).__name__}: {e}"
            else:
                reason = validate_voice_record(out)
            if reason:
                # 恒久スキップに記録する（生成数は減る）。topup_failed_dialogues.py が
                # この対話を破棄して Stage 2 から作り直し、target 件に戻す。
                sf.write(json.dumps({"id": record.get("id"), "reason": reason},
                                    ensure_ascii=False) + "\n")
                sf.flush()
                n_skipped += 1
                logger.warning(f"[{record.get('id')}] 変換できない対話を恒久スキップに"
                               f"記録（理由: {reason}）。")
                continue
            if any(p.get("gender") not in ("male", "female")
                   for p in out["speaker_profiles"]):
                n_gender_missing += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1
        f.flush()

    if n_gender_missing:
        logger.info(f"性別が付いていない対話 {n_gender_missing}件（generate_tts が"
                    "名前・役割から推定します）。")
    logger.info(f"passthrough 変換完了: {converted}件"
                + (f"（今回の恒久スキップ {n_skipped}件）" if n_skipped else ""))
    return converted
