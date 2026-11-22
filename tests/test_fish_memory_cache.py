"""fish の参照音声キャッシュ（use_memory_cache）の契約テスト。

参照音声は1本30秒・2.6MB あり、fish の既定（`schema.py` の `use_memory_cache="off"`）
では **毎リクエスト VQ エンコードし直す**。"on" にすると sha256 をキーに
**同一の prompt_tokens** を返すだけなので出力は変わらない
（`fish-speech/fish_speech/inference_engine/reference_loader.py:99-126`。
"off" のときも辞書へ書き込んではいるので、"on" は読み出しを有効にするだけ）。

★出力が変わらないことは実サーバで検証済み（2026-09-10・node24）:
    seed を固定すると off/off が sha256 一致＝生成が決定的であることを確認した上で、
    off と on を突き合わせて **/v1/tts 12件・/v1/tts_multiturn 6件すべて一致**。
  速度は往復 4.17秒 → 3.83秒（-0.34秒/件）。fish 時間の 84〜91% は自己回帰デコード
  そのものなので、削れるのはここだけ。

ここで固定するのは「リクエストに必ず載ること」と「不正値を弾くこと」。
実サーバとの突き合わせは CI に置けないので、上記の実測を記録として残す。
"""

from __future__ import annotations

import unittest

from generate_tts import FishSpeechSynth


def make(**kw) -> FishSpeechSynth:
    return FishSpeechSynth(base_url="http://127.0.0.1:1", **kw)


class TestMemoryCacheFlag(unittest.TestCase):
    def test_default_is_on(self):
        """既定で有効。fish 側の既定 "off" をこちらで上書きする。"""
        self.assertEqual(make().use_memory_cache, "on")

    def test_invalid_value_is_rejected(self):
        for bad in ("true", "ON", "", "1", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                make(use_memory_cache=bad)

    def test_off_is_allowed(self):
        self.assertEqual(make(use_memory_cache="off").use_memory_cache, "off")

    def test_tts_body_carries_the_flag(self):
        """/v1/tts（base 合成）のボディに載ること。"""
        for value in ("on", "off"):
            s = make(use_memory_cache=value)
            captured = {}

            def fake_post(url, data=None, headers=None, timeout=None, _c=captured):
                _c["body"] = _unpack(data, headers)
                raise _Stop

            s._local.session = _FakeSession(fake_post)
            with self.assertRaises(_Stop):
                s._one("テスト", {"reference_id": "x"})
            self.assertEqual(captured["body"]["use_memory_cache"], value)

    def test_multiturn_body_carries_the_flag(self):
        """/v1/tts_multiturn（refine の Pass B）のボディにも載ること。

        Pass B は1対話の36〜40%を占めるので、こちらに載っていないと効果が半減する。
        """
        for value in ("on", "off"):
            s = make(use_memory_cache=value)
            captured = {}

            def fake_post(url, data=None, headers=None, timeout=None, _c=captured):
                _c["body"] = _unpack(data, headers)
                raise _Stop

            s._local.session = _FakeSession(fake_post)
            with self.assertRaises(_Stop):
                s.synth_dialogue("<|speaker:0|>あ", [{"audio": b"x", "text": "t"}])
            self.assertEqual(captured["body"]["use_memory_cache"], value)

    def test_build_synth_passes_it(self):
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
                                           "use_memory_cache": "off"}},
                          "fish_speech", False, None)
        except Exception:
            pass          # 他の初期化で落ちても、渡った値だけ見たい
        finally:
            G.FishSpeechSynth = orig
        self.assertEqual(captured.get("use_memory_cache"), "off")

    def test_shipped_config_enables_it(self):
        import yaml
        import main as M
        c = M.load_config("configs/main_v1.yaml")
        self.assertEqual(c["tts"]["fish_speech"].get("use_memory_cache"), "on")


class _Stop(Exception):
    """HTTP に出る前に止めるための番兵。"""


class _FakeSession:
    def __init__(self, post):
        self.post = post


def _unpack(data, headers):
    ctype = (headers or {}).get("Content-Type", "")
    if "msgpack" in ctype:
        import ormsgpack
        return ormsgpack.unpackb(data)
    import json
    return json.loads(data)


if __name__ == "__main__":
    unittest.main()
