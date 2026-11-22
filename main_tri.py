"""三人対話データ自動生成パイプライン（Stage 1〜2）

二人用の main.py と同じ CLI・同じ処理の流れで、Stage1/2 だけ三人用
（generate_metadata_tri / generate_dialogues_tri）に差し替えて回す。
Stage2 の最後の voice_dialogues.jsonl への変換（★2026-11-20 旧 Stage3 を Stage2 に組み込んだ。
main.run_dialogue_stage）は二人用と同じ passthrough（2人以上なら話者数を問わない）。ターンの宛先
（addressee）を voice_dialogue.turns に写すため、build_voice_turns だけ包む。

使い方（config を省くと config_tri_ft_v1_gemma_inoue.yaml）:
    python main_tri.py --config config_tri_ft_v1_gemma_inoue.yaml --stage metadata
    python main_tri.py --config config_tri_ft_v1_gemma_inoue.yaml --stage dialogue   # voice_dialogues.jsonl まで
    python main_tri.py --config ... --stage all --target 20      # Stage1+2（voice_dialogues.jsonl まで）
    python main_tri.py --config ... --stage dialogue --limit 5   # 先頭5件だけ
Stage3（TTS。旧 Stage4）は generate_tts_tri.py。
"""

import asyncio
import sys

import main as m
import generate_dialogues_tri
import generate_metadata_tri

DEFAULT_CONFIG = "config_tri_ft_v1_gemma_inoue.yaml"

# Stage3（passthrough）が voice_dialogue.turns に写す、三人用のターンの項目。
# 二人用の build_voice_turns は yomi しか写さないので、宛先を写すよう包む。
VOICE_CARRY_KEYS = ("addressee", "addressee_src")
_ORIG: dict = {}


def build_voice_turns(input_turns: list) -> list:
    out = _ORIG["build_voice_turns"](input_turns)
    if len(out) != len(input_turns):     # 二人用は1ターン→1ターン。変わったらテストで分かる
        raise RuntimeError("build_voice_turns の入出力のターン数が違います（宛先を写せません）")
    for t, o in zip(input_turns, out):
        for k in VOICE_CARRY_KEYS:
            if t.get(k) is not None:
                o[k] = t[k]
    return out


def apply_voice_patches() -> None:
    import generate_voice_passthrough as gvp
    if not _ORIG:
        _ORIG["build_voice_turns"] = gvp.build_voice_turns
    gvp.build_voice_turns = build_voice_turns


def remove_voice_patches() -> None:
    import generate_voice_passthrough as gvp
    if _ORIG:
        gvp.build_voice_turns = _ORIG.pop("build_voice_turns")


def _config_path(argv: list[str]) -> str:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return DEFAULT_CONFIG


def check_config(config: dict, path: str) -> None:
    """二人用の config を三人用の入口に渡す取り違えを、生成を始める前に止める。"""
    md = config.get("metadata") or {}
    if md.get("num_speakers") != 3:
        raise SystemExit(f"{path}: 三人対話の config ではありません"
                         f"（metadata.num_speakers={md.get('num_speakers')}）")
    mode = (config.get("voice") or {}).get("mode")
    if mode != "passthrough":
        raise SystemExit(f"{path}: 三人対話の変換は voice.mode: passthrough だけ対応です（受領: {mode}）")


def run():
    argv = sys.argv[1:]
    path = _config_path(argv)
    if not any(a == "--config" or a.startswith("--config=") for a in argv):
        sys.argv = [sys.argv[0], "--config", path] + argv
    check_config(m.load_config(path), path)
    # main.main() が呼ぶ Stage1/2 の入口（main モジュールの属性）を三人用に差し替える。
    m.run_metadata_generation = generate_metadata_tri.run_metadata_generation
    m.run_dialogue_generation = generate_dialogues_tri.run_dialogue_generation
    apply_voice_patches()
    asyncio.run(m.main())


if __name__ == "__main__":
    run()
