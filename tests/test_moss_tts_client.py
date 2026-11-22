"""generate_tts.MossTtsSynth（tts.fish_speech.backend: moss_tts）のクライアント側の検査（2026-11-13）。
HTTP は偽物に差し替えるのでサーバは要らない。
    .venv-maai-cu12/bin/python -m unittest tests.test_moss_tts_client -v
"""
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_tts as G  # noqa: E402


class _Resp:
    def __init__(self, sr, sec=1.0):
        buf = io.BytesIO()
        t = np.arange(int(sr * sec)) / sr
        sf.write(buf, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
        self.content, self.status_code, self.text = buf.getvalue(), 200, ""


class _Session:
    def __init__(self, sr):
        self.sr, self.bodies = sr, []

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.bodies.append(json)
        return _Resp(self.sr)


def _synth(sr=24000, **fs):
    s = G.build_fish_speech_synth({"fish_speech": {"backend": "moss_tts", "base_url": "http://x", **fs}})
    s._local.session = _Session(sr)      # _session はスレッドごとの requests.Session を返すプロパティ
    return s


REF = {"audio": _Resp(24000).content, "text": "参照"}


class MossClientTest(unittest.TestCase):
    def test_sample_rate_fixed_and_resampled(self):
        s = _synth(24000)
        self.assertEqual(s.sample_rate, 48000)          # 作った直後から固定（パイプラインはこの値を使う）
        w = s.synth_dialogue("<|speaker:0|>こんにちは。", [REF])[0]
        self.assertAlmostEqual(len(w) / s.sample_rate, 1.0, delta=0.01)
        self.assertEqual(s.sample_rate, 48000)          # 応答の 24k で書き換えない

    def test_bwe_keeps_length(self):
        s = _synth(24000, bwe=True)
        w = s.synth_dialogue("<|speaker:0|>こんにちは。", [REF])[0]
        self.assertAlmostEqual(len(w) / 48000, 1.0, delta=0.01)

    def test_body_commas_mode_seed(self):
        s = _synth(48000, moss_commas="space", moss_mode="continuation")
        s.synth_one_seeded("えっと、そうですね、はい。", REF, seed=42)
        b = s._session.bodies[-1]
        self.assertEqual(b["input"], "えっと そうですね はい。")
        self.assertEqual(b["mode"], "continuation")
        self.assertEqual(b["seed"], 42)
        self.assertEqual(b["ref_text"], "参照")

    def test_speaker_tag_picks_reference(self):
        s = _synth(48000)
        r0, r1 = {"audio": REF["audio"], "text": "A"}, {"audio": REF["audio"], "text": "B"}
        s.synth_dialogue("<|speaker:1|>はい。<|speaker:0|>どうも。", [r0, r1])
        self.assertEqual([b["ref_text"] for b in s._session.bodies], ["B", "A"])
        self.assertNotIn("<|speaker", s._session.bodies[0]["input"])

    def test_rejects_multiturn_and_context(self):
        with self.assertRaises(ValueError):
            _synth(48000, multiturn=True)
        s = _synth(48000)
        with self.assertRaises(NotImplementedError):
            s.synth_dialogue("<|speaker:0|>はい。", [REF], context=[{"text": "x"}])
        with self.assertRaises(ValueError):
            _synth(48000, moss_commas="weird")

    def test_active_and_pauses(self):
        sr = 16000
        tone = 0.2 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
        w = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2), tone, np.zeros(sr // 2)]).astype(np.float32)
        act, pr = G._active_and_pauses(w, sr)
        self.assertAlmostEqual(act, 2.5, delta=0.05)    # 前後の無音を除いた区間
        self.assertAlmostEqual(pr, 0.2, delta=0.02)     # その中の 0.5秒の無音


class IrodoriVllmClientTest(unittest.TestCase):
    """backend: irodori_vllm（IrodoriVllmSynth）。長さと兄弟テイクを extra_params で送る。"""

    def test_extra_params_seconds_and_siblings(self):
        bank = Path(__file__).resolve().parent.parent / "voices/wav-inoue-sidon"
        f = bank / "inoue_t6.wav"
        if not f.exists():
            self.skipTest("声バンクが無い")
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 60, "num_steps": 40,
                                                                   "voices_dirs": [str(bank)]}}})
        s._local.session = _Session(48000)
        self.assertEqual(s.sample_rate, 48000)
        s.synth_one_seeded("こんにちは、テストです。", {"audio": f.read_bytes(), "text": "参照"}, seed=7)
        b = s._session.bodies[-1]
        ep = b["extra_params"]
        self.assertEqual(b["seed"], 7)
        self.assertEqual(ep["num_steps"], 40)
        self.assertEqual(ep["ref_wavs"][0], str(f))                 # 本人のファイルが先頭
        self.assertGreater(len(ep["ref_wavs"]), 1)                  # 兄弟テイクが続く
        self.assertTrue(all("inoue_t" in p for p in ep["ref_wavs"]))
        self.assertNotIn("ref_audio", b)                           # ファイルで渡すのでインラインは送らない（30秒制限）
        self.assertAlmostEqual(ep["seconds"], G.estimate_mora("こんにちは、テストです。")[0] / 7.3 / 0.94, places=3)
        # バンクに無い参照は兄弟を探さない（受け取った音声だけ）
        s.synth_one_seeded("はい。", REF, seed=1)
        self.assertNotIn("ref_wavs", s._session.bodies[-1]["extra_params"])
        self.assertIn("ref_audio", s._session.bodies[-1])

    def test_laugh_emoji_replaces_a_leading_laugh_only(self):
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0,
                                                                   "laugh_emoji": True}}})
        s._local.session = _Session(48000)
        s.synth_one_seeded("ふふ。それは危ないですね。", REF, seed=1)
        b = s._session.bodies[-1]
        self.assertEqual(b["input"], "🤭それは危ないですね。")
        # 長さは笑い込みの元の文で決める
        self.assertAlmostEqual(b["extra_params"]["seconds"],
                               G.estimate_mora("ふふ。それは危ないですね。")[0] / 7.3 / 0.94, places=3)
        for t in ("うふふ。", "はい、ふふ。", "へえ、そうなんだ。"):   # 笑いだけ・頭以外・笑いでない
            s.synth_one_seeded(t, REF, seed=1)
            self.assertEqual(s._session.bodies[-1]["input"], t)

    def test_laugh_split_makes_the_laugh_separately_and_picks_a_clear_one(self):
        import io as _io

        class _Laugh:
            """笑いの要求には seed ごとに有音の長さが違う 0.7 秒を、文の要求には 1 秒の音を返す。"""
            def __init__(self):
                self.bodies = []

            def post(self, url, json=None, timeout=None):  # noqa: A002
                self.bodies.append(json)
                sr = 48000
                if json["input"].startswith("🤭🤭"):
                    act = {1: 0.05, 2: 0.32, 3: 0.6, 4: 0.12}[(json["seed"] - 9) // 104729]
                    w = np.zeros(int(0.7 * sr), np.float32)
                    w[: int(act * sr)] = 0.3
                else:
                    w = np.full(sr, 0.1, np.float32)
                b = _io.BytesIO()
                sf.write(b, w, sr, format="WAV")
                return type("R", (), {"status_code": 200, "content": b.getvalue(), "text": ""})()

        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0,
                                                                   "laugh_emoji": "split"}}})
        sess = _Laugh()
        s._local.session = sess
        s._requests = type("R", (), {"Session": staticmethod(lambda: sess)})
        w = s.synth_one_seeded("ふふ。それは危ないですね。", REF, seed=9)
        laughs = [b for b in sess.bodies if b["input"] == "🤭🤭ふふっ。"]
        rest = [b for b in sess.bodies if b["input"] != "🤭🤭ふふっ。"]
        self.assertEqual(len(laughs), 4)
        self.assertTrue(all(b["extra_params"]["seconds"] == 0.7 for b in laughs))
        self.assertEqual([b["input"] for b in rest], ["それは危ないですね。"])   # 文は笑いを除いて 1 本
        self.assertEqual(rest[0]["seed"], 9)
        self.assertAlmostEqual(rest[0]["extra_params"]["seconds"],
                               G.estimate_mora("それは危ないですね。")[0] / 7.3 / 0.94, places=3)
        sr = 48000
        self.assertEqual(len(w), int(0.7 * sr) + int(0.15 * sr) + sr)            # 笑い＋0.15 秒＋文
        self.assertAlmostEqual(float((np.abs(w[: int(0.7 * sr)]) > 0.2).sum()) / sr, 0.32, places=2)  # 0.35 秒に近い笑い
        sess.bodies.clear()
        s.synth_one_seeded("うふふ。", REF, seed=9)                       # 笑いだけの発話は触らない
        self.assertEqual([b["input"] for b in sess.bodies], ["うふふ。"])

    def test_short_takes_draws_several_for_a_backchannel_only(self):
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0,
                                                                   "short_takes": 4}}})
        sess = _Session(48000)
        s._local.session = sess
        s._requests = type("R", (), {"Session": staticmethod(lambda: sess)})   # 別スレッドでも同じ偽物
        s.synth_one_seeded("はい。", REF, seed=10)
        b = sess.bodies
        self.assertEqual(len(b), 4)                                  # 相槌は 4 本
        self.assertEqual(sorted(x["seed"] for x in b), [10, 10 + 7919, 10 + 2 * 7919, 10 + 3 * 7919])
        s.synth_one_seeded("それは本当に面白いですね。", REF, seed=10)
        self.assertEqual(len(sess.bodies), 5)                        # 長い発話は 1 本のまま

    def test_a_lost_request_is_retried_with_the_same_seed(self):
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0}}})
        sess = _Session(48000)
        calls = []
        real = sess.post

        def flaky(url, json=None, timeout=None):  # noqa: A002
            calls.append((json["seed"], timeout))
            if len(calls) == 1:
                raise s._requests.exceptions.ReadTimeout("lost")
            return real(url, json=json, timeout=timeout)
        sess.post = flaky
        s._local.session = sess
        w = s.synth_one_seeded("それは本当に面白いですね。", REF, seed=3)
        self.assertGreater(len(w), 0)
        self.assertEqual([c[0] for c in calls], [3, 3])
        self.assertLessEqual(calls[0][1], 180)

    def test_text_replace_changes_only_the_tts_input(self):
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0,
                                                                   "text_replace": {"イノウエ": "井上"}}}})
        s._local.session = _Session(48000)
        s.synth_one_seeded("はじめまして。イノウエです。", REF, seed=1)
        b = s._session.bodies[-1]
        self.assertEqual(b["input"], "はじめまして。井上です。")
        self.assertAlmostEqual(b["extra_params"]["seconds"],
                               G.estimate_mora("はじめまして。イノウエです。")[0] / 7.3 / 0.94, places=3)


class ShortTakeScoreTest(unittest.TestCase):
    """short_take_score: 声が出て語尾が下がる相槌ほど小さい。"""

    @staticmethod
    def _glide(f_start, f_end, sr=48000, sec=0.4):
        t = np.arange(int(sr * sec)) / sr
        f = np.linspace(f_start, f_end, len(t))
        return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32), sr

    def test_falling_beats_rising_and_breath(self):
        fall = G.short_take_score(*self._glide(220, 170))
        rise = G.short_take_score(*self._glide(170, 220))
        rng = np.random.default_rng(0)
        breath = G.short_take_score((0.05 * rng.standard_normal(19200)).astype(np.float32), 48000)
        self.assertLess(fall, rise)
        self.assertLess(rise, breath)
        self.assertGreaterEqual(breath, 10.0)
        # 疑問の相槌（「はい？」）は上がるほうが良い
        self.assertLess(G.short_take_score(*self._glide(170, 220), question=True),
                        G.short_take_score(*self._glide(220, 170), question=True))


if __name__ == "__main__":
    unittest.main()


class EndClickTest(unittest.TestCase):
    def _synth_returning(self, w, **ir):
        class S:
            bodies = []

            def post(self, url, json=None, timeout=None):  # noqa: A002
                b = io.BytesIO()
                sf.write(b, w, 48000, format="WAV")
                return type("R", (), {"status_code": 200, "content": b.getvalue(), "text": ""})()
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "irodori_vllm", "base_url": "http://x",
                                                       "irodori": {"mora_rate": 7.3, "refs_sec": 0, **ir}}})
        s._local.session = S()
        return s

    def test_a_click_after_the_speech_is_muted(self):
        sr = 48000
        speech = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
        click = np.zeros(int(0.3 * sr), np.float32)
        click[-int(0.02 * sr):] = 0.3
        w = np.concatenate([speech, click])
        out = self._synth_returning(w).synth_one_seeded("こんにちは。", REF, seed=1)
        self.assertEqual(len(out), len(w))
        self.assertLess(float(np.abs(out[sr:]).max()), 1e-3)                  # クリックは消える
        self.assertTrue(np.allclose(out[: sr - 480], w[: sr - 480], atol=1e-4))  # 発話は無傷
        out = self._synth_returning(w, mute_end_click=False).synth_one_seeded("こんにちは。", REF, seed=1)
        self.assertGreater(float(np.abs(out[sr:]).max()), 0.2)
        # 無音の後でも 60ms 以上の音（最後の語）は残す
        tail = np.zeros(int(0.5 * sr), np.float32)
        tail[int(0.2 * sr): int(0.35 * sr)] = 0.3
        w2 = np.concatenate([speech, tail])
        out = self._synth_returning(w2).synth_one_seeded("こんにちは。", REF, seed=1)
        self.assertGreater(float(np.abs(out[sr:]).max()), 0.2)
