"""Stage 3（passthrough）の恒久スキップと、その topup（破棄→作り直し）の台帳操作。

LLM は呼ばない。検査するのは
  - 後段が扱えない変換結果だけを弾いているか（正常な対話を落としていないか）
  - 弾いたものが skip ファイルに残り、再実行で焼き直されないか
  - topup が「Stage3 失敗＝対話を破棄・メタデータは温存」「2回目＝メタデータも破棄」
    という規則どおりに台帳を書き換えるか
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cost_tracker import CostTracker
from generate_voice_passthrough import (
    run_voice_generation,
    skip_file_for,
    validate_voice_record,
)
from topup_failed_dialogues import discard_and_reset, run


def _voice_record(turns, names=("あかり", "たける")):
    return {
        "speaker_profiles": [{"name": n, "gender": "female"} for n in names],
        "voice_dialogue": {"turns": turns},
    }


def _dialogue_record(did, turns):
    return {
        "id": did,
        "metadata": {
            "id": did,
            "speakers": [{"name": "あかり", "gender": "female"},
                         {"name": "たける", "gender": "male"}],
        },
        "dialogue": {"turns": turns, "speaking_rate": 1.1},
    }


_OK_TURNS = [{"speaker": "あかり", "utterance": "えっと、今日はどうも。"},
             {"speaker": "たける", "utterance": "こちらこそ、よろしくお願いします。"}]


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


class ValidateVoiceRecordTest(unittest.TestCase):
    def test_正常な対話は通す(self):
        rec = _voice_record([{"speaker": "あかり", "text": "そうですね。"},
                             {"speaker": "たける", "text": "ええ。"}])
        self.assertIsNone(validate_voice_record(rec))

    def test_ターン0は弾く(self):
        self.assertEqual(validate_voice_record(_voice_record([])), "ターン0")

    def test_空テキストのターンを弾く(self):
        rec = _voice_record([{"speaker": "あかり", "text": "そうですね。"},
                             {"speaker": "たける", "text": "  "}])
        self.assertEqual(validate_voice_record(rec), "空テキストのターン")

    def test_話者不明のターンを弾く(self):
        # 実測で出た形: speaker が null・text が空のターンが1つ混じる
        rec = _voice_record([{"speaker": "あかり", "text": "そうですね。"},
                             {"speaker": None, "text": "ええ。"}])
        self.assertIn("話者不明", validate_voice_record(rec))

    def test_話者2人未満を弾く(self):
        # generate_tts.py が落とす条件を Stage 3 で先取りする
        rec = _voice_record([{"speaker": "あかり", "text": "そうですね。"},
                             {"speaker": "あかり", "text": "ええ。"}])
        self.assertEqual(validate_voice_record(rec), "話者2人未満")


class PassthroughSkipFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.config = {
            "dialogue": {"output_file": str(d / "dialogues.jsonl")},
            "voice": {"output_file": str(d / "voice_dialogues.jsonl")},
            "metadata": {"output_file": str(d / "metadata.jsonl")},
        }
        self.dial = Path(self.config["dialogue"]["output_file"])
        self.voice = Path(self.config["voice"]["output_file"])
        self.skip = skip_file_for(self.voice)
        self.cost = CostTracker({"input_per_1m": 0.0, "output_per_1m": 0.0},
                                1.0, str(d / "cost_log.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        return asyncio.run(run_voice_generation(self.config, self.cost))

    def test_壊れた対話だけがskipに記録され再実行で焼き直されない(self):
        broken = _dialogue_record("b", [{"speaker": "あかり", "utterance": "はい。"},
                                        {"speaker": None, "utterance": ""}])
        _write_jsonl(self.dial, [_dialogue_record("a", _OK_TURNS), broken,
                                 _dialogue_record("c", _OK_TURNS)])

        self.assertEqual(self._run(), 2)
        self.assertEqual([r["id"] for r in _read_jsonl(self.voice)], ["a", "c"])
        skipped = _read_jsonl(self.skip)
        self.assertEqual([r["id"] for r in skipped], ["b"])
        self.assertEqual(skipped[0]["reason"], "空テキストのターン")

        # 再実行しても b は pending に入らない（skip は増えず、出力も重複しない）
        self.assertEqual(self._run(), 2)
        self.assertEqual(len(_read_jsonl(self.voice)), 2)
        self.assertEqual(len(_read_jsonl(self.skip)), 1)

    def test_変換できないレコードもskipに記録する(self):
        _write_jsonl(self.dial, [{"id": "x", "metadata": {"speakers": []}}])  # dialogue キー欠落
        self.assertEqual(self._run(), 0)
        self.assertEqual(_read_jsonl(self.skip)[0]["id"], "x")


class TopupDiscardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.config = {
            "metadata": {"output_file": str(d / "metadata.jsonl")},
            "dialogue": {"output_file": str(d / "dialogues.jsonl")},
            "voice": {"output_file": str(d / "voice_dialogues.jsonl")},
        }
        self.meta = Path(self.config["metadata"]["output_file"])
        self.dial = Path(self.config["dialogue"]["output_file"])
        self.voice_skip = skip_file_for(Path(self.config["voice"]["output_file"]))
        self.dial_skip = skip_file_for(self.dial)
        _write_jsonl(self.meta, [{"id": i} for i in ("a", "b", "c")])
        _write_jsonl(self.dial, [_dialogue_record(i, _OK_TURNS) for i in ("a", "b", "c")])

    def tearDown(self):
        self.tmp.cleanup()

    def _ids(self, path):
        return [r["id"] for r in _read_jsonl(path)]

    def test_Stage3の1回目は対話だけ捨ててメタデータは残す(self):
        _write_jsonl(self.voice_skip, [{"id": "b", "reason": "話者不明のターン(None)"}])
        n_meta, n_dial = discard_and_reset(self.config, set(), {"b"})

        self.assertEqual((n_meta, n_dial), (0, 1))
        self.assertEqual(self._ids(self.meta), ["a", "b", "c"])   # Stage2 が引き直す
        self.assertEqual(self._ids(self.dial), ["a", "c"])
        self.assertEqual(self._ids(self.dial.parent / "dialogues_discarded.jsonl"), ["b"])
        # skip はクリアされ、履歴に退避されている
        self.assertEqual(_read_jsonl(self.voice_skip), [])
        self.assertEqual(self._ids(self.voice_skip.parent /
                                   "voice_dialogues_skipped_history.jsonl"), ["b"])

    def test_Stage3の2回目はメタデータごと捨てる(self):
        _write_jsonl(self.voice_skip, [{"id": "b", "reason": "話者不明のターン(None)"}])
        discard_and_reset(self.config, set(), {"b"})
        # b が Stage2 で作り直され、また Stage3 に弾かれた
        _write_jsonl(self.dial, [_dialogue_record(i, _OK_TURNS) for i in ("a", "b", "c")])
        _write_jsonl(self.voice_skip, [{"id": "b", "reason": "話者不明のターン(None)"}])

        n_meta, n_dial = discard_and_reset(self.config, set(), {"b"})
        self.assertEqual((n_meta, n_dial), (1, 1))
        self.assertEqual(self._ids(self.meta), ["a", "c"])
        self.assertEqual(self._ids(self.dial), ["a", "c"])

    def test_Stage2の失敗は従来どおりメタデータを捨てる(self):
        _write_jsonl(self.dial, [_dialogue_record(i, _OK_TURNS) for i in ("a", "c")])
        _write_jsonl(self.dial_skip, [{"id": "b", "reason": "文末句点欠落"}])

        n_meta, n_dial = discard_and_reset(self.config, {"b"}, set())
        self.assertEqual((n_meta, n_dial), (1, 0))
        self.assertEqual(self._ids(self.meta), ["a", "c"])
        self.assertEqual(self._ids(self.meta.parent / "metadata_discarded.jsonl"), ["b"])
        self.assertEqual(_read_jsonl(self.dial_skip), [])
        self.assertEqual(self._ids(self.dial.parent /
                                   "dialogues_skipped_history.jsonl"), ["b"])


class TopupLoopTest(unittest.TestCase):
    """Stage1/Stage2 を差し替え、Stage3 は本物（決定的）でループ全体を回す。

    「Stage3 で弾かれた対話が、次のイテレーションで作り直されて target に届く」ことを見る。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.config = {
            "target_dialogues": 3,
            "metadata": {"output_file": str(d / "metadata.jsonl")},
            "dialogue": {"output_file": str(d / "dialogues.jsonl")},
            "voice": {"mode": "passthrough", "output_file": str(d / "voice_dialogues.jsonl")},
        }
        self.meta = Path(self.config["metadata"]["output_file"])
        self.dial = Path(self.config["dialogue"]["output_file"])
        self.voice = Path(self.config["voice"]["output_file"])
        self.cost = CostTracker({"input_per_1m": 0.0, "output_per_1m": 0.0},
                                1.0, str(d / "cost_log.jsonl"))
        self.stage2_calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    async def _fake_stage1(self, config, cost_tracker):
        """target 件に足りないぶんのメタデータを追記する（本物と同じ増分方式）。"""
        have = [r["id"] for r in _read_jsonl(self.meta)]
        n = config["target_dialogues"] - len(have)
        with open(self.meta, "a") as f:
            for i in range(n):
                f.write(json.dumps({"id": f"m{len(have) + i}"}) + "\n")
        return config["target_dialogues"]

    async def _fake_stage2(self, config, cost_tracker):
        """対話の無いメタデータぶんを生成する。初回だけ m1 を壊れた対話にする。"""
        self.stage2_calls += 1
        done = {r["id"] for r in _read_jsonl(self.dial)}
        with open(self.dial, "a") as f:
            for m in _read_jsonl(self.meta):
                if m["id"] in done:
                    continue
                turns = _OK_TURNS
                if m["id"] == "m1" and self.stage2_calls == 1:
                    turns = [{"speaker": "あかり", "utterance": "はい。"},
                             {"speaker": None, "utterance": ""}]
                f.write(json.dumps(_dialogue_record(m["id"], turns),
                                   ensure_ascii=False) + "\n")
        return len(_read_jsonl(self.dial))

    def test_Stage3で弾かれた対話が作り直されてtargetに届く(self):
        with mock.patch("topup_failed_dialogues.run_metadata_generation", self._fake_stage1), \
             mock.patch("topup_failed_dialogues.run_dialogue_generation", self._fake_stage2), \
             mock.patch("main.run_dialogue_generation", self._fake_stage2):   # run_dialogue_stage (main.py) calls this
            asyncio.run(run(self.config, self.cost, max_iters=3))

        self.assertEqual(sorted(r["id"] for r in _read_jsonl(self.voice)),
                         ["m0", "m1", "m2"])
        self.assertEqual(self.stage2_calls, 2)          # 2周目で作り直した
        self.assertEqual(_read_jsonl(skip_file_for(self.voice)), [])
        # メタデータは温存され、壊れた対話だけが退避されている
        self.assertEqual(sorted(r["id"] for r in _read_jsonl(self.meta)), ["m0", "m1", "m2"])
        self.assertEqual([r["id"] for r in
                          _read_jsonl(self.dial.parent / "dialogues_discarded.jsonl")], ["m1"])


if __name__ == "__main__":
    unittest.main()
