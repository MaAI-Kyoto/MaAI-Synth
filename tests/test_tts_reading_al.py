"""読みの修正（al_v1 試走7の視聴。2026-11-07）: 点の促音便・動詞＋「方」→かた。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tts_reading as R  # noqa: E402


def test_ten_counter_uses_sokuon():
    assert R.to_tts_text("１点ずつ") == "いってんずつ"
    assert R.to_tts_text("8点") == "はってん" and R.to_tts_text("10点") == "じゅってん"
    assert R.to_tts_text("3点") == "さんてん" and R.to_tts_text("6点") == "ろくてん"
    assert R.to_tts_text("100点") == "ひゃくてん"


def test_verb_kata_only_after_verb_stems():
    assert R.verb_kata("世界の見え方が変わった") == "世界の見えかたが変わった"
    for w, r in (("考え方", "考えかた"), ("やり方", "やりかた"), ("言い方", "言いかた"), ("選び方", "選びかた")):
        assert R.verb_kata(w) == r
    for w in ("大きい方が", "したい方がいい", "同じ方を見て", "明け方に", "こちらの方", "あなた方", "方向", "先生方", "行った方が"):
        assert R.verb_kata(w) == w


def test_verb_kata_is_off_unless_configured():
    saved = dict(R._CFG)
    try:
        R.configure(True)
        assert R.apply_if_enabled("見え方") == "見え方"
        R.configure({"enabled": True, "verb_kata": True})
        assert R.apply_if_enabled("見え方") == "見えかた"
    finally:
        R._CFG.update(saved)
