"""tts.fish_speech.main_gen（主話者(ch0)だけ生成パラメータを変える）の契約テスト。サーバは使わない。

    .venv-maai-cu12/bin/python -m pytest -q tests/test_main_gen.py
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

import generate_tts as G
from test_vllm_omni_client import _synth

MAIN_TEXT = "主話者の参照。"
REFS = [{"audio": b"RIFF0000", "text": MAIN_TEXT}, {"audio": b"RIFF1111", "text": "相手の参照。"}]


def _main_synth(**main_gen):
    s, sess = _synth()
    s.main_gen = main_gen or {"temperature": 0.5}
    s.main_ref_texts = frozenset({MAIN_TEXT})
    return s, sess


def _temps(sess):
    return [body["extra_params"]["temperature"] for _url, body in sess.calls]


class TestMainGen(unittest.TestCase):
    def test_off_by_default(self):
        s, sess = _synth()
        s.synth_dialogue("<|speaker:0|>はい。\n<|speaker:1|>うん。", REFS)
        self.assertEqual(_temps(sess), [0.8, 0.8])

    def test_only_main_speaker_turns_in_multiturn(self):
        s, sess = _main_synth()
        s.synth_dialogue("<|speaker:0|>はい。\n<|speaker:1|>うん。\n<|speaker:0|>そう。", REFS)
        self.assertEqual(_temps(sess), [0.5, 0.8, 0.5])
        # 相手が speaker:0 に来ても参照の書き起こしで見分ける
        s2, sess2 = _main_synth()
        s2.synth_dialogue("<|speaker:0|>はい。\n<|speaker:1|>うん。", REFS[::-1])
        self.assertEqual(_temps(sess2), [0.8, 0.5])
        self.assertEqual(s.gen_kwargs["temperature"], 0.8)          # 共有の既定値は書き換えない

    def test_per_turn_by_voice(self):
        s, sess = _main_synth()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f, np.zeros(4410, dtype=np.float32), 44100)
            path = f.name
        try:
            s.synth(["一。", "二。"], {"audio_path": path, "text": MAIN_TEXT})
            s.synth(["三。"], {"audio_path": path, "text": "相手の参照。"})
        finally:
            os.unlink(path)
        self.assertEqual(_temps(sess), [0.5, 0.5, 0.8])

    def test_explicit_override_wins_seed_does_not(self):
        """サー対策の温度指定は main_gen より優先、seed だけの引き直しは主話者の温度のまま。"""
        s, sess = _main_synth()
        with G.temperature_override(s, 0.9):
            G._synth_one_line(s, "はい。", REFS[0], None)
        G._synth_one_line(s, "はい。", REFS[0], None, seed=7)
        self.assertEqual(_temps(sess), [0.9, 0.5])
        self.assertFalse(s.temp_pinned)

    def test_other_keys_merge(self):
        s, sess = _main_synth(temperature=0.6, top_p=0.7)
        s.synth_dialogue("<|speaker:0|>はい。\n<|speaker:1|>うん。", REFS)
        ep = [body["extra_params"] for _u, body in sess.calls]
        self.assertEqual((ep[0]["temperature"], ep[0]["top_p"]), (0.6, 0.7))
        self.assertEqual((ep[1]["temperature"], ep[1]["top_p"]), (0.8, 0.8))
        self.assertEqual(ep[0]["eos_bias"], 4.0)


if __name__ == "__main__":
    unittest.main()
