"""Stage 4（三人対話）: TTS 音声合成（3チャネル）・リタイム・相槌の挿入（refine）

二人用の generate_tts.py / refine_maai.py を**そのまま使い**、二人を前提にした部分だけを
差し替えてから generate_tts.main() を呼ぶ。CLI は generate_tts.py と同じ
（--retime-refine / --refine / --limit / --claim / --shard / --recompose / --merge-parts /
--print-paths 等）。config を省くと config_tri_ft_v1_gemma_inoue.yaml。

  .venv-maai-cu12/bin/python generate_tts_tri.py --config config_tri_ft_v1_gemma_inoue.yaml --retime-refine --limit 5
  .venv-maai-cu12/bin/python generate_tts_tri.py --config ... --retime-refine --claim --claim-ttl-min 90 --gpu 0 --fish-url http://localhost:8888
  .venv-maai-cu12/bin/python generate_tts_tri.py --config ... --merge-parts

差し替えるもの（tri.retimer: vap ＝既定）:
  - record_speaker_order: 二人用は話者が2人のときしか主話者を ch0 に固定しない。
    三人版は主話者（metadata.main_speaker_index）を ch0、残りを出現順にする。
  - リタイマ（generate_tts / refine_maai の MaaiRetimer）→ retime_tri_vap.TriVapRetimer。
    二者 VAP を「注目する話者」と「ほかの全員の和」の2本に組み替えて使う（retime_tri_vap.py）。
  - refine の挿入候補の走査・タイムライン → TriInsertionScanner / tri_build_timeline
    （聞き手ごとに二者の相槌予測を当てる）。
  - refine_maai.SUPPORTED_SPEAKER_COUNTS → (2, 3)（このプロセスの中だけ）。
  - refine_maai.apply_ops: 挿入した相槌の宛先（addressee）を、相槌を打った相手の話者にする。
  - --dry-run のリタイマ → retime_tri.TriGapRetimer（モデルを使わず人間の鼎談の間合いで配置）、
    走査 → TriMockScanner。
tri.retimer: gap にすると、VAP を使わずに鼎談の間合いで配置する（GPU・maai 不要。refine は不可）。

使えないもの: run_tts_pool.sh（generate_tts.py を直に起動する）。各マシンで --claim 付きで起動する。
声の割り当て・fish の multiturn・取り違えと読み違いの引き直し・3ch の合成・無音の雑音埋め・
音量の均し・refine の LLM（ops の提案と検証）は、二人用が話者数を問わず動くのでそのまま使う。
"""

from __future__ import annotations

import logging
import sys
import types

import generate_tts as gt
import refine_maai as rm
import retime_tri
import retime_tri_vap as rtv

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config_tri_ft_v1_gemma_inoue.yaml"
RETIMERS = ("vap", "gap")


def record_speaker_order(record: dict) -> list[str]:
    """三人版: 主話者を ch0 に置き、残りは出現順。主話者が居なければ出現順のまま。"""
    order = gt.dialogue_speaker_order(record["voice_dialogue"]["turns"])
    main = gt._main_speaker_name(record)
    if main is None:
        return order
    if main not in order:
        logger.warning(f"[{record.get('id')}] 主話者「{main}」がターンに居ません"
                       f"(order={order})。出現順のままにします。")
        return order
    return [main] + [s for s in order if s != main]


def _config_path(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


def retimer_kind(config: dict) -> str:
    kind = str((config.get("tri") or {}).get("retimer", "vap"))
    if kind not in RETIMERS:
        raise SystemExit(f"tri.retimer は {RETIMERS} のどれか（受領: {kind}）")
    return kind


def check_config(config: dict, path: str) -> None:
    md = config.get("metadata") or {}
    if md.get("num_speakers") != 3:
        raise SystemExit(f"{path}: 三人対話の config ではありません"
                         f"（metadata.num_speakers={md.get('num_speakers')}）")
    rc = (config.get("tts") or {}).get("retime") or {}
    if retimer_kind(config) == "vap" and rc.get("backend", "maaibatch") not in ("maai", "vapbatch", "maaibatch"):
        raise SystemExit(f"{path}: tri.retimer: vap は tts.retime.backend が maai / vapbatch / maaibatch のとき"
                         f"だけ使えます（受領: {rc.get('backend')}）")


# 差し替える (モジュール, 属性名) と、元の値の控え
_TARGETS = [(gt, "record_speaker_order"), (gt, "MaaiRetimer"), (gt, "MockRetimer"),
            (rm, "MaaiRetimer"), (rm, "MockRetimer"), (rm, "InsertionScanner"),
            (rm, "MockScanner"), (rm, "build_timeline"), (rm, "SUPPORTED_SPEAKER_COUNTS"),
            (rm, "apply_ops")]
_ORIG: dict = {}


def apply_ops(record: dict, turns: list, ops: list, density):
    """refine の ops 適用（二人用）を包み、挿入した相槌の宛先を、相槌を打った相手の話者にする。

    既存ターンの宛先は二人用が dict を写すのでそのまま残る。挿入相槌は新しい dict なので宛先が無い。
    """
    new_record, index_map, changed, inserted, summary = \
        _ORIG[(rm, "apply_ops")](record, turns, ops, density)
    new_turns = new_record["voice_dialogue"]["turns"]
    for ni in inserted:
        for j in range(ni - 1, -1, -1):
            if new_turns[j].get("type", "speech") != "backchannel":
                if new_turns[j]["speaker"] != new_turns[ni]["speaker"]:
                    new_turns[ni]["addressee"] = new_turns[j]["speaker"]
                break
    return new_record, index_map, changed, inserted, summary


def apply_patches(config: dict) -> None:
    retime_tri.configure((config.get("tri") or {}).get("timing"))
    if not _ORIG:
        for mod, name in _TARGETS:
            _ORIG[(mod, name)] = getattr(mod, name)
    kind = retimer_kind(config)
    retimer = rtv.TriVapRetimer if kind == "vap" else retime_tri.TriGapRetimer
    gt.record_speaker_order = record_speaker_order   # refine_maai は関数内で import するので gt だけでよい
    gt.MaaiRetimer = retimer
    gt.MockRetimer = retime_tri.TriGapRetimer        # --dry-run はモデルなしの配置で配線を確かめる
    rm.MaaiRetimer = retimer
    rm.MockRetimer = retime_tri.TriGapRetimer
    rm.InsertionScanner = rtv.TriInsertionScanner
    rm.MockScanner = rtv.TriMockScanner
    rm.build_timeline = rtv.tri_build_timeline
    rm.SUPPORTED_SPEAKER_COUNTS = (2, 3)
    rm.apply_ops = apply_ops
    if kind == "gap":
        # generate_tts.run() は backend が vapbatch 以外だと maai パッケージの有無を確かめる。
        # 間合いの配置は maai を使わないので、無い環境（.venv-fish 等）では空のモジュールで通す。
        # ★import はしない（find_spec だけ）。pip の maai を読み込むと backend=maaibatch の
        #   リポジトリ版 maai と混ざって起動時に落ちる（retime_maaibatch.import_maai_from_repo）。
        import importlib.util
        if "maai" not in sys.modules and importlib.util.find_spec("maai") is None:
            sys.modules["maai"] = types.ModuleType("maai")


def remove_patches() -> None:
    """差し替えを元に戻す（テスト用）。"""
    for (mod, name), f in _ORIG.items():
        setattr(mod, name, f)
    _ORIG.clear()


def run():
    argv = sys.argv[1:]
    path = _config_path(argv)
    if path is None:
        path = DEFAULT_CONFIG
        sys.argv = [sys.argv[0], "--config", path] + argv
    config = gt.load_config(path)
    check_config(config, path)
    if retimer_kind(config) == "gap" and any(a in ("--refine", "--retime-refine") for a in argv):
        raise SystemExit("tri.retimer: gap では --refine / --retime-refine は使えません"
                         "（相槌の挿入候補は二者の相槌予測で求めるため）。tri.retimer: vap にしてください。")
    apply_patches(config)
    gt.main()


if __name__ == "__main__":
    run()
