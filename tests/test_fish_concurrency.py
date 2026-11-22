"""FishSpeechSynth.synth の同時実行の契約テスト。

fish は逐次の小さいカーネルでレイテンシ律速（実効帯域が Blackwell の1%未満）なので、
同時に投げると 1.7〜2.0倍になる。ただし「順序が入れ替わる」「例外の見え方が変わる」と
配置や無音代替の判定が黙って狂うので、そこを固定しておく。

`_one` を差し替えて HTTP を使わずに検証する（fish サーバ不要）。
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from generate_tts import FishSpeechSynth


def make_synth(concurrency: int) -> FishSpeechSynth:
    return FishSpeechSynth(base_url="http://127.0.0.1:1", concurrency=concurrency)


class RecordingSynth:
    """_one を差し替えて、呼び出しの並行度と順序を記録する。"""

    def __init__(self, synth: FishSpeechSynth, delay: float = 0.05,
                 fail_on: set[str] | None = None):
        self.synth = synth
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls: list[str] = []
        self.max_inflight = 0
        self._inflight = 0
        self._lock = threading.Lock()
        synth._one = self._one           # type: ignore[method-assign]

    def _one(self, text: str, voice: dict) -> np.ndarray:
        with self._lock:
            self.calls.append(text)
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            time.sleep(self.delay)
            if text in self.fail_on:
                raise RuntimeError(f"synth failed: {text}")
            # テキストの長さで見分けられる波形を返す
            return np.full(len(text), float(len(text)), dtype=np.float32)
        finally:
            with self._lock:
                self._inflight -= 1


TEXTS = [f"turn{i:02d}" + "あ" * i for i in range(8)]


class TestFishConcurrency(unittest.TestCase):
    def test_default_is_serial(self):
        """既定は 1（従来どおり逐次）。config に書いたときだけ同時化する。"""
        self.assertEqual(make_synth(1).concurrency, 1)
        self.assertEqual(FishSpeechSynth(base_url="http://127.0.0.1:1").concurrency, 1)

    def test_concurrency_is_clamped_to_at_least_one(self):
        for v in (0, -3, None):
            self.assertEqual(FishSpeechSynth(base_url="http://127.0.0.1:1",
                                             concurrency=v).concurrency, 1)

    def test_serial_path_runs_one_at_a_time(self):
        s = make_synth(1)
        rec = RecordingSynth(s)
        s.synth(TEXTS, {})
        self.assertEqual(rec.max_inflight, 1)
        self.assertEqual(rec.calls, TEXTS)          # 逐次なら呼び出し順もそのまま

    def test_concurrent_path_actually_overlaps(self):
        s = make_synth(4)
        rec = RecordingSynth(s)
        s.synth(TEXTS, {})
        self.assertGreater(rec.max_inflight, 1, "同時実行になっていない")
        self.assertLessEqual(rec.max_inflight, 4, "concurrency を超えて投げている")

    def test_result_order_matches_input_order(self):
        """★これが崩れると配置が別ターンの音声を掴む。"""
        for c in (1, 2, 4, 16):
            s = make_synth(c)
            RecordingSynth(s, delay=0.02)
            out = s.synth(TEXTS, {})
            self.assertEqual(len(out), len(TEXTS), c)
            for text, wav in zip(TEXTS, out):
                self.assertEqual(len(wav), len(text), f"concurrency={c}")
                self.assertTrue(np.allclose(wav, float(len(text))), f"concurrency={c}")

    def test_serial_and_concurrent_give_the_same_result(self):
        a = make_synth(1); RecordingSynth(a, delay=0.0)
        b = make_synth(4); RecordingSynth(b, delay=0.0)
        wa, wb = a.synth(TEXTS, {}), b.synth(TEXTS, {})
        self.assertEqual([len(w) for w in wa], [len(w) for w in wb])

    def test_exception_propagates_like_serial(self):
        """呼び出し側（synthesize_for_speaker）が個別合成へ落とせるよう、
        逐次時と同じく例外がそのまま上がること。"""
        for c in (1, 4):
            s = make_synth(c)
            RecordingSynth(s, delay=0.01, fail_on={TEXTS[3]})
            with self.assertRaises(RuntimeError, msg=f"concurrency={c}"):
                s.synth(TEXTS, {})

    def test_single_text_uses_serial_path(self):
        s = make_synth(4)
        rec = RecordingSynth(s)
        s.synth([TEXTS[0]], {})
        self.assertEqual(rec.max_inflight, 1)

    def test_empty_input(self):
        for c in (1, 4):
            s = make_synth(c)
            RecordingSynth(s)
            self.assertEqual(s.synth([], {}), [])

    def test_session_is_per_thread(self):
        """requests.Session はスレッドセーフを保証しないので、スレッドごとに別インスタンス。"""
        s = make_synth(4)
        seen: list[int] = []
        lock = threading.Lock()

        def grab():
            with lock:
                seen.append(id(s._session))

        ths = [threading.Thread(target=grab) for _ in range(4)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        self.assertEqual(len(set(seen)), 4, "スレッド間で Session を共有している")
        # 同じスレッドからは同じ Session が返る（毎回作り直さない）
        self.assertIs(s._session, s._session)

    def test_session_honours_use_proxy(self):
        self.assertFalse(make_synth(1)._session.trust_env)
        s = FishSpeechSynth(base_url="http://127.0.0.1:1", use_proxy=True)
        self.assertTrue(s._session.trust_env)


class TestConfigWiring(unittest.TestCase):
    def test_shipped_config_sets_concurrency(self):
        import yaml
        import main as M
        c = M.load_config("configs/main_v1.yaml")          # public release: concurrency comes from the TTS preset
        self.assertEqual(c["tts"]["fish_speech"].get("concurrency"), 32)

    def test_build_synth_passes_concurrency(self):
        """config の値が FishSpeechSynth まで届くこと。"""
        import generate_tts as G
        captured = {}
        orig = G.FishSpeechSynth

        class Spy(orig):
            def __init__(self, *a, **kw):
                captured.update(kw)
                super().__init__(*a, **kw)

        G.FishSpeechSynth = Spy
        try:
            G.build_synth({"engine": "fish_speech",
                           "fish_speech": {"base_url": "http://127.0.0.1:1",
                                           "concurrency": 3}},
                          "fish_speech", False, None)
        except Exception:
            pass       # 合成器の他の初期化で落ちても、渡った値だけ見たい
        finally:
            G.FishSpeechSynth = orig
        self.assertEqual(captured.get("concurrency"), 3)


if __name__ == "__main__":
    unittest.main()
