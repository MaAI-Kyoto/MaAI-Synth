"""tts.fish_speech.backend: vllm_omni（generate_tts.VllmOmniSynth）の契約テスト。サーバは使わない。

    .venv-maai-cu12/bin/python -m pytest -q tests/test_vllm_omni_client.py
"""

from __future__ import annotations

import io
import unittest

import numpy as np
import soundfile as sf

import generate_tts as G


class _Resp:
    status_code = 200

    def __init__(self, k=0):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(4410, dtype=np.float32), 44100, format="wav")
        self.content = buf.getvalue()
        codes = (np.arange(30, dtype="<i2").reshape(10, 3) + 100 * k)  # 呼び出しごとに違う音声トークン
        self.headers = {"X-Fish-Codes-Shape": "10,3",
                        "X-Fish-Codes": __import__("base64").b64encode(codes.tobytes()).decode()}


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None, **kw):
        import copy
        self.calls.append((url, copy.deepcopy(json)))
        return _Resp(len(self.calls))


def _synth(**gen):
    s = G.VllmOmniSynth(base_url="http://127.0.0.1:1", gen_kwargs={"temperature": 0.8, "top_p": 0.8,
                                                                   "eos_bias": 4.0, **gen})
    sess = _Session()
    s._local.session = sess
    return s, sess


class TestVllmOmniClient(unittest.TestCase):
    def test_one_body(self):
        s, sess = _synth(max_new_tokens=1024, repetition_penalty=1.1, chunk_length=200, normalize=True)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f, np.zeros(44100, dtype=np.float32), 44100)
            path = f.name
        try:
            w = s._one("こんにちは。", {"audio_path": path, "text": "参照。"})
        finally:
            os.unlink(path)
        self.assertEqual(len(w), 4410)
        url, body = sess.calls[0]
        self.assertTrue(url.endswith("/v1/audio/speech"))
        self.assertEqual(body["input"], "こんにちは。")           # /v1/tts と同じく話者タグなし
        self.assertTrue(body["ref_audio"].startswith("data:audio/wav;base64,"))
        self.assertEqual(body["ref_text"], "参照。")
        ep = body["extra_params"]
        self.assertEqual(ep, {"fish_native_sampling": True, "temperature": 0.8, "top_p": 0.8,
                              "top_k": 30, "eos_bias": 4.0})
        self.assertNotIn("seed", body)
        for k in ("repetition_penalty", "chunk_length", "normalize"):
            self.assertNotIn(k, body)
            self.assertNotIn(k, ep)

    def test_multiturn_follows_native_order_and_history(self):
        """fish 本体の /v1/tts_multiturn と同じ: 1ターンずつ・全話者の参照・生成済みターンを履歴に積む。"""
        s, sess = _synth()
        refs = [{"audio": b"RIFF0000", "text": "参照0"}, {"audio": b"RIFF1111", "text": "参照1"}]
        with G.seed_override(s, 1234):
            wavs, codes = s.synth_dialogue("<|speaker:0|>はい。\n<|speaker:1|>うん。\n<|speaker:0|>そう。", refs,
                                           return_codes=True)
        self.assertEqual(len(wavs), 3)
        self.assertEqual([c.shape for c in codes], [(10, 3)] * 3)
        bodies = [b for _, b in sess.calls]
        self.assertEqual([b["input"] for b in bodies], ["<|speaker:0|>はい。", "<|speaker:1|>うん。", "<|speaker:0|>そう。"])
        for b in bodies:
            ep = b["extra_params"]
            self.assertEqual([r["text"] for r in ep["fish_refs"]], ["参照0", "参照1"])
            self.assertTrue(ep["fish_return_codes"])
            self.assertEqual(b["seed"], 1234)
            self.assertNotIn("ref_audio", b)
        self.assertEqual(bodies[0]["extra_params"]["fish_history"], [])
        h = bodies[2]["extra_params"]["fish_history"]
        self.assertEqual([x["text"] for x in h], ["<|speaker:0|>はい。", "<|speaker:1|>うん。"])
        # 履歴の音声トークンはサーバが返したものそのまま（再エンコードしない）
        self.assertEqual(h[1]["codes"], _Resp(2).headers["X-Fish-Codes"])
        self.assertTrue(np.array_equal(codes[1], np.arange(30).reshape(10, 3) + 200))

    def test_context_is_rejected(self):
        s, _ = _synth()
        with self.assertRaises(NotImplementedError):
            s.synth_dialogue("<|speaker:0|>はい。", [{"audio": b"a", "text": "0"}], context=[{"text": "x", "audio": b"y"}])

    def test_factory_rejects_only_context_audio(self):
        with self.assertRaises(ValueError):
            G.build_fish_speech_synth({"fish_speech": {"backend": "vllm_omni", "context_audio": True}})
        s = G.build_fish_speech_synth({"fish_speech": {"backend": "vllm_omni", "multiturn": True, "window": 2,
                                                       "window_context": 1, "concurrency": 16,
                                                       "gen": {"eos_bias": 4.0}}})
        self.assertIsInstance(s, G.VllmOmniSynth)
        self.assertEqual((s.multiturn, s.window, s.window_context, s.concurrency), (True, 2, 1, 16))
        with self.assertRaises(ValueError):
            G.build_fish_speech_synth({"fish_speech": {"backend": "sglang"}})
        self.assertIs(type(G.build_fish_speech_synth({"fish_speech": {}})), G.FishSpeechSynth)
        self.assertFalse(getattr(G.FishSpeechSynth, "parallel_windows", False))  # fish 本体は今までどおり逐次


if __name__ == "__main__":
    unittest.main()
