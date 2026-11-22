"""thin_short_commas（短い句の直後の読点を外す・dialogue.enrich.comma_thin_seg）のテスト。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_dialogues import thin_short_commas                           # noqa: E402


def _thin(u, seg=5, run=24, names=("イノウエ", "ハルナ")):
    ts = [{"speaker": "イノウエ", "utterance": u}]
    thin_short_commas(ts, seg, run, set(names))
    return ts[0]["utterance"]


def test_thin_removes_short_particle_heads():
    assert _thin("私は、イノウエと申します。") == "私はイノウエと申します。"
    assert _thin("はい。合格して、家族に、自慢したいです。") == "はい。合格して、家族に自慢したいです。"
    assert _thin("例えば、今日は、この一ページだけ、完璧にする、という、感じです。") == \
        "例えば、今日はこの一ページだけ、完璧にするという感じです。"


def test_thin_keeps_fillers_reactions_conjunctions_vocatives():
    for u in ("まあ、そうですね。", "なるほど、面白いですね。", "でも、それは違います。",
              "私は、まあ、研究者です。", "ハルナさん、それは本当ですか？"):
        assert _thin(u) == u


def test_thin_off_and_run_limit():
    u = "私は、大学で会話するロボットの研究をしています。"
    assert _thin(u, seg=0) == u
    assert _thin(u, run=10) == u          # 外すと読点なしの区間が長くなりすぎる
