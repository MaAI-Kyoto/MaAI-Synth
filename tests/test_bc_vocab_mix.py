import collections
import unittest

from refine_maai import (
    QualityParams,
    RefineConfig,
    _continuer_only_inserts,
    _hee_fits,
    _rebalance_bc_vocab,
    _thin_hee_family,
)

MIX = {"うん": 27, "はい": 23, "あー": 20, "へえ": 8, "うんうん": 6.5,
       "ええ": 5, "うんうんうん": 4.5, "はいはい": 3, "はいはいはい": 3}


def _bc(speaker, text, rel=0.5):
    return {"speaker": speaker, "text": text, "type": "backchannel",
            "refine": {"op": "insert", "rel_pos": rel, "slot_p": 0.6}}


def _dialogue(n, bc_text, anchor="今日は京都の大学で会話ロボットの研究をしています。それで毎日忙しいです。",
              listener_text="そうなんですね。私は会社員です。", rel=0.55):
    turns = []
    for _ in range(n):
        turns.append({"speaker": "A", "text": anchor})
        turns.append(_bc("B", bc_text, rel))
        turns.append({"speaker": "B", "text": listener_text})
    return turns


class RebalanceTests(unittest.TestCase):
    def test_off_by_default(self):
        turns = _dialogue(10, "ええ。")
        self.assertEqual(_rebalance_bc_vocab(turns, set(), QualityParams(), 2, set(), "d"), 0)
        self.assertTrue(all(t["text"] == "ええ。" for t in turns if t.get("type") == "backchannel"))

    def test_reduces_overused_word_toward_mix(self):
        q = QualityParams(bc_vocab_mix=MIX)
        cnt = collections.Counter()
        for k in range(200):
            turns = _dialogue(10, "ええ。")
            changed = set()
            _rebalance_bc_vocab(turns, changed, q, 2, set(), f"d{k}")
            for i, t in enumerate(turns):
                if t.get("type") == "backchannel":
                    cnt[t["text"]] += 1
                    if t["text"] != "ええ。":
                        self.assertIn(i, changed)
                        self.assertEqual(t["refine"]["orig_text"], "ええ。")
                        self.assertEqual(t["refine"]["op"], "bc_mix")
        tot = sum(cnt.values())
        self.assertLess(cnt["ええ。"] / tot, 0.15)
        self.assertGreater(cnt["あー。"] / tot, 0.10)
        self.assertGreater(cnt["へえ。"], 0)
        self.assertTrue(all(k.endswith("。") for k in cnt))

    def test_deterministic_by_seed(self):
        q = QualityParams(bc_vocab_mix=MIX)
        a, b = _dialogue(10, "ええ。"), _dialogue(10, "ええ。")
        _rebalance_bc_vocab(a, set(), q, 2, set(), "same")
        _rebalance_bc_vocab(b, set(), q, 2, set(), "same")
        self.assertEqual([t["text"] for t in a], [t["text"] for t in b])

    def test_casual_listener_gets_no_hai_or_ee(self):
        q = QualityParams(bc_vocab_mix=MIX)
        for k in range(50):
            turns = _dialogue(8, "はい。", anchor="昨日さ、京都の古本屋でめっちゃ珍しい本見つけたんだよ。",
                              listener_text="まじか、俺も行きたいわ。")
            _rebalance_bc_vocab(turns, set(), q, 2, set(), f"c{k}")
            for t in turns:
                if t.get("type") == "backchannel":
                    self.assertNotIn(t["text"].rstrip("。"), {"はい", "はいはい", "はいはいはい", "ええ"})

    def test_protected_speaker_untouched(self):
        q = QualityParams(bc_vocab_mix=MIX)
        turns = _dialogue(10, "ええ。")
        _rebalance_bc_vocab(turns, set(), q, 2, {"B"}, "p")
        self.assertTrue(all(t["text"] == "ええ。" for t in turns if t.get("type") == "backchannel"))

    def test_words_outside_mix_untouched(self):
        q = QualityParams(bc_vocab_mix=MIX)
        turns = _dialogue(10, "なるほど。", rel=0.9)
        _rebalance_bc_vocab(turns, set(), q, 2, set(), "n")
        self.assertTrue(all(t["text"] == "なるほど。" for t in turns if t.get("type") == "backchannel"))

    def test_misplaced_hee_is_always_replaced(self):
        q = QualityParams(bc_vocab_mix=MIX)
        # 打点が文の途中（句点の直後でない）→ へえ は合わない
        turns = _dialogue(5, "へー。", anchor="今日は京都の大学で会話ロボットの研究をしていて毎日忙しいです。", rel=0.3)
        _rebalance_bc_vocab(turns, set(), q, 2, set(), "h")
        for t in turns:
            if t.get("type") == "backchannel":
                self.assertNotEqual(t["text"], "へー。")
                self.assertNotEqual(t["text"], "へえ。")


class HeeFitsTests(unittest.TestCase):
    def test_after_declarative_sentence(self):
        anchor = "今日は京都の大学で会話ロボットの研究をしています。それで毎日忙しいです。"
        rel = (anchor.index("。") + 1) / len(anchor)
        turns = [{"speaker": "A", "text": anchor}, _bc("B", "へえ。", rel)]
        self.assertTrue(_hee_fits(turns, 1))

    def test_not_after_greeting_or_question(self):
        for anchor in ["はじめまして、イノウエと申します。今日はよろしくお願いします。",
                       "京都の大学で研究をされているんですよね。そうですか。"]:
            rel = (anchor.index("。") + 1) / len(anchor)
            turns = [{"speaker": "A", "text": anchor}, _bc("B", "へえ。", rel)]
            self.assertFalse(_hee_fits(turns, 1), anchor)

    def test_not_mid_clause(self):
        anchor = "今日は京都の大学で会話ロボットの研究をしていて毎日忙しいです。"
        turns = [{"speaker": "A", "text": anchor}, _bc("B", "へえ。", 0.4)]
        self.assertFalse(_hee_fits(turns, 1))


class InteractionTests(unittest.TestCase):
    def test_mix_words_survive_continuer_enforcement(self):
        q = QualityParams(bc_vocab_mix=MIX)
        turns = [{"speaker": "A", "text": "今日は京都の大学で研究をしています。"},
                 _bc("B", "あー。", 0.3), _bc("B", "うんうんうん。", 0.4), _bc("B", "なるほど。", 0.5)]
        n = _continuer_only_inserts(turns, set(), q, 2)
        self.assertEqual(n, 1)
        self.assertEqual(turns[1]["text"], "あー。")
        self.assertEqual(turns[2]["text"], "うんうんうん。")

    def test_without_mix_old_behavior(self):
        turns = [{"speaker": "A", "text": "今日は京都の大学で研究をしています。"},
                 _bc("B", "へえ。", 0.3)]
        self.assertEqual(_continuer_only_inserts(turns, set(), QualityParams(), 2), 1)

    def test_hee_thin_disabled_when_mix_on(self):
        turns = [{"speaker": "A", "text": "京都の大学で研究をしています。"}] + \
                [_bc("B", "へえ。", 0.9) for _ in range(4)]
        self.assertEqual(_thin_hee_family(turns, set(), QualityParams(bc_vocab_mix=MIX), 0), 0)
        self.assertGreater(_thin_hee_family(turns, set(), QualityParams(), 0), 0)

    def test_config_parses_mix(self):
        cfg = RefineConfig.from_tts_cfg({"refine": {"quality": {"bc_vocab_mix": MIX}}})
        self.assertEqual(cfg.quality.bc_vocab_mix["あー"], 20)


if __name__ == "__main__":
    unittest.main()
