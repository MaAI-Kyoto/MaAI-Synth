import collections
import json
import random
import tempfile
import unicodedata
import unittest
from pathlib import Path

import numpy as np

from generate_tts import save_turn_cache
from refine_maai import (
    DensityParams,
    LimitParams,
    OpenAIRefiner,
    QualityParams,
    RefineConfig,
    apply_ops,
    run_refine,
    validate_ops,
)


class OpenAIRefinerEndpointTests(unittest.TestCase):
    """refine の投げ先を base_urls（複数）でランダム分散できること。"""

    def _refiner(self, **refine):
        refine.setdefault("provider", "openai")
        refine.setdefault("text_model", "m")
        refine["ops_dir"] = tempfile.mkdtemp()
        cfg = RefineConfig.from_tts_cfg({"refine": refine})
        return OpenAIRefiner(cfg, "v")

    def test_base_urls_is_parsed_from_config(self):
        r = self._refiner(base_urls=["http://a:1/v1", "http://b:2/v1"])
        self.assertEqual(r._endpoints, ["http://a:1/v1", "http://b:2/v1"])

    def test_requests_are_spread_randomly_across_endpoints(self):
        urls = ["http://a:1/v1", "http://b:2/v1", "http://c:3/v1"]
        r = self._refiner(base_urls=urls)
        random.seed(0)
        picks = collections.Counter(
            str(r.client().base_url).rstrip("/") for _ in range(3000))
        self.assertEqual(set(picks), set(urls))          # 全エンドポイントが選ばれる
        self.assertTrue(all(c > 700 for c in picks.values()))  # おおむね均等
        # クライアントはエンドポイントごとに1つだけ生成・再利用される
        self.assertEqual(len(r._clients), len(urls))

    def test_single_base_url_is_unchanged(self):
        r = self._refiner(base_url="http://solo:9/v1")
        self.assertEqual(r._endpoints, ["http://solo:9/v1"])
        self.assertEqual(str(r.client().base_url).rstrip("/"), "http://solo:9/v1")

    def test_base_urls_takes_precedence_over_base_url(self):
        r = self._refiner(base_url="http://solo:9/v1",
                          base_urls=["http://a:1/v1", "http://b:2/v1"])
        self.assertEqual(r._endpoints, ["http://a:1/v1", "http://b:2/v1"])


class RefineQualityGuardTests(unittest.TestCase):
    def make_config(self):
        return RefineConfig(
            density=DensityParams(
                min_bc_per_min=0.0,
                max_bc_per_min=20.0,
                tolerance=20.0,
            ),
            limits=LimitParams(
                max_reword_frac=1.0,
                max_rewrite_frac=1.0,
                max_delete_frac=1.0,
            ),
            quality=QualityParams(
                min_rewrite_chars=12,
                min_content_retention=0.75,
                max_rewrite_length_ratio=1.8,
                duplicate_bc_window=2,
                auto_merge_deleted_splits=True,
            ),
        )

    def validate(self, raw, turns, slots=None):
        return validate_ops(
            raw,
            turns,
            slots or [],
            duration_sec=60.0,
            annotations={},
            cfg=self.make_config(),
            speaker_names=["A", "B"],
        )

    def test_rewrite_accepts_filler_insertion_with_content_preserved(self):
        turns = [{
            "speaker": "A",
            "text": "明日の会議は十時から第二会議室で始まります。",
            "type": "speech",
        }]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 0.0},
            "ops": [{
                "op": "rewrite_text",
                "index": 0,
                "new_text": "えーと、明日の会議は、あの、十時から第二会議室で始まります。",
            }],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual(["rewrite_text"], [op["op"] for op in clean])
        self.assertEqual([], dropped)

    def test_rewrite_text_is_composed_to_nfc(self):
        turns = [{
            "speaker": "A",
            "text": "明日の会議は十時から第二会議室で始まります。",
            "type": "speech",
        }]
        expected = "えーと、明日の会議は、あの、十時から第二会議室で始まります。"
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 0.0},
            "ops": [{
                "op": "rewrite_text",
                "index": 0,
                "new_text": unicodedata.normalize("NFD", expected),
            }],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual([], dropped)
        self.assertEqual([expected], [op["new_text"] for op in clean])

    def test_rewrite_differing_only_in_normalization_is_not_an_edit(self):
        text = "明日の会議は十時から第二会議室で始まります。"
        turns = [{"speaker": "A", "text": text, "type": "speech"}]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 0.0},
            "ops": [{
                "op": "rewrite_text",
                "index": 0,
                "new_text": unicodedata.normalize("NFD", text),
            }],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual([], clean)
        self.assertEqual([], dropped)

    def test_rewrite_rejects_semantic_drift(self):
        turns = [{
            "speaker": "A",
            "text": "明日の会議は十時から第二会議室で始まります。",
            "type": "speech",
        }]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 0.0},
            "ops": [{
                "op": "rewrite_text",
                "index": 0,
                "new_text": "来週の旅行は午後から駅前で始まります。",
            }],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual([], clean)
        self.assertTrue(any("保持率" in reason or "欠落" in reason
                            for reason in dropped))

    def test_rewrite_rejects_polarity_change(self):
        turns = [{
            "speaker": "A",
            "text": "この資料は今日中に提出しなくても問題ありません。",
            "type": "speech",
        }]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 0.0},
            "ops": [{
                "op": "rewrite_text",
                "index": 0,
                "new_text": "あの、この資料は今日中に提出しても問題あります。",
            }],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual([], clean)
        self.assertTrue(any("否定表現" in reason for reason in dropped))

    def test_duplicate_reword_is_rejected(self):
        turns = [
            {"speaker": "A", "text": "最初の説明です。", "type": "speech"},
            {"speaker": "B", "text": "うん。", "type": "backchannel"},
            {"speaker": "A", "text": "次の説明です。", "type": "speech"},
            {"speaker": "B", "text": "そう。", "type": "backchannel"},
        ]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 2.0},
            "ops": [{"op": "reword", "index": 3, "new_text": "うん"}],
        }

        clean, _, dropped, fatal = self.validate(raw, turns)

        self.assertIsNone(fatal)
        self.assertEqual([], clean)
        self.assertTrue(any("近傍相槌と重複" in reason for reason in dropped))

    def test_duplicate_insert_is_rejected(self):
        turns = [
            {"speaker": "A", "text": "最初の説明です。", "type": "speech"},
            {"speaker": "B", "text": "うん。", "type": "backchannel"},
            {"speaker": "A", "text": "長い説明が続きます。", "type": "speech"},
        ]
        slots = [{
            "slot_id": 1,
            "anchor_index": 2,
            "listener": "B",
            "rel_pos": 0.5,
            "p": 0.8,
        }]
        raw = {
            "density": {"label": "normal", "target_bc_per_min": 2.0},
            "ops": [{"op": "insert", "slot": 1, "new_text": "うん"}],
        }

        clean, _, dropped, fatal = self.validate(raw, turns, slots)

        self.assertIsNone(fatal)
        self.assertEqual([], clean)
        self.assertTrue(any("近傍相槌と重複" in reason for reason in dropped))

    def test_deleted_embedded_backchannel_is_merged_automatically(self):
        turns = [
            {
                "speaker": "A",
                "text": "明日の会議は、",
                "type": "speech",
                "duration": 1.0,
            },
            {
                "speaker": "B",
                "text": "はい。",
                "type": "backchannel",
                "duration": 0.4,
            },
            {
                "speaker": "A",
                "text": "十時からです。",
                "type": "speech",
                "continued": True,
                "duration": 1.0,
            },
        ]
        raw = {
            "density": {"label": "sparse", "target_bc_per_min": 0.0},
            "ops": [{"op": "delete", "index": 1}],
        }

        clean, density, dropped, fatal = self.validate(raw, turns)
        record = {"id": "d1", "voice_dialogue": {"turns": turns}}
        refined, _, changed, _, summary = apply_ops(
            record, turns, clean, density)

        self.assertIsNone(fatal)
        self.assertEqual([], dropped)
        self.assertEqual(["delete", "merge"], [op["op"] for op in clean])
        self.assertEqual(1, len(refined["voice_dialogue"]["turns"]))
        self.assertEqual(
            "明日の会議は、十時からです。",
            refined["voice_dialogue"]["turns"][0]["text"],
        )
        self.assertEqual({0}, changed)
        self.assertEqual(1, summary["auto_merge"])

    def test_multiple_deleted_splits_are_merged_as_one_chain(self):
        turns = [
            {"speaker": "A", "text": "最初に、", "type": "speech", "duration": 0.8},
            {"speaker": "B", "text": "はい。", "type": "backchannel", "duration": 0.4},
            {"speaker": "A", "text": "次に、", "type": "speech", "continued": True,
             "duration": 0.8},
            {"speaker": "B", "text": "ええ。", "type": "backchannel", "duration": 0.4},
            {"speaker": "A", "text": "最後です。", "type": "speech", "continued": True,
             "duration": 0.8},
        ]
        raw = {
            "density": {"label": "sparse", "target_bc_per_min": 0.0},
            "ops": [{"op": "delete", "index": 1},
                    {"op": "delete", "index": 3},
                    {"op": "merge", "index": 0, "with": 2}],
        }

        clean, density, _, fatal = self.validate(raw, turns)
        record = {"id": "d1", "voice_dialogue": {"turns": turns}}
        refined, _, _, _, summary = apply_ops(record, turns, clean, density)

        self.assertIsNone(fatal)
        self.assertEqual(1, len(refined["voice_dialogue"]["turns"]))
        self.assertEqual("最初に、次に、最後です。",
                         refined["voice_dialogue"]["turns"][0]["text"])
        self.assertEqual(1, summary["auto_merge"])

    def test_dry_run_pipeline_writes_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_file = root / "voice_dialogues.jsonl"
            record = {
                "id": "dialogue-1",
                "metadata": {
                    "relationship": "同僚",
                    "speech_style": "敬語",
                    "emotional_tone": "落ち着いている",
                    "speakers": [
                        {"name": "A", "age_range": "30代"},
                        {"name": "B", "age_range": "30代"},
                    ],
                },
                "speaker_profiles": [
                    {"name": "A", "gender": "female", "age_range": "30代"},
                    {"name": "B", "gender": "male", "age_range": "30代"},
                ],
                "voice_dialogue": {"turns": [
                    {"speaker": "A", "text": "明日の会議について詳しい予定を確認します。",
                     "type": "speech", "start_offset": 0.0, "duration": 4.0},
                    {"speaker": "B", "text": "はい。", "type": "backchannel",
                     "start_offset": -0.5, "duration": 0.4},
                    {"speaker": "A", "text": "開始時刻は十時で、第二会議室を使います。",
                     "type": "speech", "continued": True,
                     "start_offset": -0.1, "duration": 4.0},
                    {"speaker": "B", "text": "承知しました。", "type": "speech",
                     "start_offset": 0.2, "duration": 1.0},
                ]},
            }
            input_file.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            config = {
                "voice": {"output_file": str(input_file)},
                "tts": {
                    "engine": "google_cloud",
                    "sample_rate": 8000,
                    "audio_dir": str(root / "audio"),
                    "manifest_file": str(root / "audio_manifest.jsonl"),
                    "google_cloud": {"voices": [
                        {"id": "voice-a", "voice_name": "voice-a", "gender": "female"},
                        {"id": "voice-b", "voice_name": "voice-b", "gender": "male"},
                    ]},
                    "retime": {"tag": "maai"},
                    "refine": {
                        "tag": "ref",
                        "ops_dir": str(root / "ops"),
                        "refined_records_file": str(root / "refined.jsonl"),
                    },
                },
            }

            done = run_refine(config, dry_run=True, limit=1)
            variant = "maai_ref_dryrun"
            manifest = root / f"audio_manifest_{variant}.jsonl"
            refined = root / f"refined_{variant}.jsonl"
            audio = root / "audio" / variant / "dialogue-1.wav"

            self.assertEqual(1, done)
            self.assertTrue(audio.exists())
            self.assertTrue(manifest.exists())
            self.assertTrue(refined.exists())
            manifest_record = json.loads(manifest.read_text().splitlines()[0])
            refined_record = json.loads(refined.read_text().splitlines()[0])
            self.assertEqual(len(manifest_record["turns"]),
                             len(refined_record["voice_dialogue"]["turns"]))
            self.assertIn("refine_config", manifest_record)


class FusedStage4Test(unittest.TestCase):
    """融合モード（--retime-refine）: 1対話ごとに retime→refine を連続実行する。"""

    @staticmethod
    def _record(dialogue_id: str = "dialogue-1") -> dict:
        return {
            "id": dialogue_id,
            "metadata": {
                "relationship": "同僚",
                "speech_style": "敬語",
                "emotional_tone": "落ち着いている",
                "speakers": [
                    {"name": "A", "age_range": "30代"},
                    {"name": "B", "age_range": "30代"},
                ],
            },
            "speaker_profiles": [
                {"name": "A", "gender": "female", "age_range": "30代"},
                {"name": "B", "gender": "male", "age_range": "30代"},
            ],
            "voice_dialogue": {"turns": [
                {"speaker": "A", "text": "明日の会議について詳しい予定を確認します。",
                 "type": "speech", "start_offset": 0.0, "duration": 4.0},
                {"speaker": "B", "text": "はい。", "type": "backchannel",
                 "start_offset": -0.5, "duration": 0.4},
                {"speaker": "A", "text": "開始時刻は十時で、第二会議室を使います。",
                 "type": "speech", "continued": True,
                 "start_offset": -0.1, "duration": 4.0},
                {"speaker": "B", "text": "承知しました。", "type": "speech",
                 "start_offset": 0.2, "duration": 1.0},
            ]},
        }

    @staticmethod
    def _config(root: Path, input_file: Path) -> dict:
        return {
            "voice": {"output_file": str(input_file)},
            "tts": {
                "engine": "google_cloud",
                "sample_rate": 8000,
                "audio_dir": str(root / "audio"),
                "manifest_file": str(root / "audio_manifest.jsonl"),
                "google_cloud": {"voices": [
                    {"id": "voice-a", "voice_name": "voice-a", "gender": "female"},
                    {"id": "voice-b", "voice_name": "voice-b", "gender": "male"},
                ]},
                "retime": {"tag": "maai"},
                "refine": {
                    "tag": "ref",
                    "ops_dir": str(root / "ops"),
                    "refined_records_file": str(root / "refined.jsonl"),
                },
            },
        }

    def test_fused_writes_both_stage_outputs(self):
        """1段目（base）と2段目（ref）の成果物が1回の実行で揃う。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_file = root / "voice_dialogues.jsonl"
            input_file.write_text(
                json.dumps(self._record(), ensure_ascii=False) + "\n")

            done = run_refine(self._config(root, input_file), dry_run=True,
                              limit=1, fused=True)

            self.assertEqual(1, done)
            base = root / "audio" / "maai_dryrun" / "dialogue-1.wav"
            ref = root / "audio" / "maai_ref_dryrun" / "dialogue-1.wav"
            self.assertTrue(base.exists(), "1段目の音声が書かれていない")
            self.assertTrue(ref.exists(), "2段目の音声が書かれていない")
            base_mf = root / "audio_manifest_maai_dryrun.jsonl"
            ref_mf = root / "audio_manifest_maai_ref_dryrun.jsonl"
            self.assertTrue(base_mf.exists())
            self.assertTrue(ref_mf.exists())
            # 1段目の manifest は --retime と同じ形（retime_config 付き・原ターン数）
            base_rec = json.loads(base_mf.read_text().splitlines()[0])
            self.assertIn("retime_config", base_rec)
            self.assertEqual(4, len(base_rec["turns"]))

    def test_fused_skips_retime_when_turn_cache_exists(self):
        """retime 済み（発話wavキャッシュあり）の対話は1段目を回さず refine だけ走る。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_file = root / "voice_dialogues.jsonl"
            record = self._record()
            input_file.write_text(json.dumps(record, ensure_ascii=False) + "\n")

            # --retime が書いたキャッシュを模す（dry-run はキャッシュを書かないため手で置く）
            sr = 8000
            n = len(record["voice_dialogue"]["turns"])
            wavs = {i: np.zeros(sr // 2, dtype=np.float32) for i in range(n)}
            save_turn_cache(root / "audio" / "plain" / "turns",   # ★2026-11-22 旧 audio/<engine>/turns
                            record["id"], wavs, set(), sr)

            done = run_refine(self._config(root, input_file), dry_run=True,
                              limit=1, fused=True)

            self.assertEqual(1, done)
            base = root / "audio" / "maai_dryrun" / "dialogue-1.wav"
            ref = root / "audio" / "maai_ref_dryrun" / "dialogue-1.wav"
            self.assertFalse(base.exists(), "キャッシュがあるのに1段目が走っている")
            self.assertTrue(ref.exists(), "2段目の音声が書かれていない")


if __name__ == "__main__":
    unittest.main()


class MainListenerInsertTests(unittest.TestCase):
    """insert.main_listener: 聞き手が主話者になる位置だけ候補の条件を緩める（傾聴 al_v1。2026-11-07）。"""

    SR = 16000

    class _Retimer:
        """p_bc が 0.5 秒おきに 0.5 の山（その間 0.1）。12.5Hz のフレーム。"""

        class _Scorer:
            def __init__(self, dur):
                self.dur = dur

            def score_long_many(self, pairs):
                frames = []
                for k in range(int(self.dur * 12.5)):
                    t = (k + 1) / 12.5
                    p = 0.5 if k % 6 == 3 else 0.1
                    frames.append((int(t * 16000), {"p_bc": [p]}))
                return [frames, frames]

        def __init__(self, dur):
            self.scorer = self._Scorer(dur)

        def _bc_scorer(self):
            return self.scorer

        def _bc_signal(self, fr):
            return fr["p_bc"][0]

    class _Timeline:
        def __init__(self, dur):
            self._len = int(dur * 16000)

        def slice(self, t0, t1):
            n = int((t1 - t0) * 16000)
            return np.zeros(n, np.float32), np.zeros(n, np.float32)

    def _scan(self, main_ch, **main_listener):
        from refine_maai import InsertionScanner, InsertParams
        # 主話者 A（ch0）の短い問い → 相手 B（ch1）の3秒の話 → A の3秒の発話
        turns = [{"speaker": "A", "text": "最近は何をされてるんですか？", "type": "speech"},
                 {"speaker": "B", "text": "最近は、近所の公園を、毎朝ゆっくり歩いているんです。", "type": "speech"},
                 {"speaker": "A", "text": "それは、気持ちよさそうですね、朝の公園って、いいですよね。", "type": "speech"}]
        durs = [2.0, 3.0, 3.0]
        plans, t = [], 0.0
        for i, d in enumerate(durs):
            plans.append({"index": i, "start_sec": t, "retime": {}})
            t += d + 0.3
        wav = {i: np.zeros(int(d * self.SR), np.float32) for i, d in enumerate(durs)}
        cfg = RefineConfig(insert=InsertParams(main_listener=main_listener))
        sc = InsertionScanner(self._Retimer(t), cfg)
        return sc.scan(self._Timeline(t), turns, plans, wav, self.SR, {"A": 0, "B": 1}, set(),
                       **({"main_ch": main_ch} if main_ch is not None else {}))

    def test_relaxed_only_where_the_main_speaker_listens(self):
        # 既定（4秒以上のターンだけ）では3秒のターンに候補が出ない
        self.assertEqual(self._scan(None), [])
        slots = self._scan(0, min_turn_sec=2.5, threshold=0.3, min_separation_sec=0.4)
        self.assertTrue(slots)
        # 候補は相手の話（index 1）にだけ立ち、聞き手は主話者 A。A の3秒の発話には相手の相槌を立てない
        self.assertEqual({s["anchor_index"] for s in slots}, {1})
        self.assertEqual({s["listener"] for s in slots}, {"A"})
        # main_ch を渡さなければ（三人用・主話者の無いコーパス）上書きは当てない
        self.assertEqual(self._scan(None, min_turn_sec=2.5, threshold=0.3, min_separation_sec=0.4), [])

    def test_unknown_main_listener_key_is_rejected(self):
        from refine_maai import InsertParams
        with self.assertRaises(ValueError):
            InsertParams(main_listener={"min_turn": 2.5})
        p = InsertParams(main_listener={"min_turn_sec": 2.5, "edge_margin_sec": [0.6, 0.3]})
        m = p.for_main_listener()
        self.assertEqual((m.min_turn_sec, m.edge_margin_sec, m.main_listener), (2.5, (0.6, 0.3), {}))
        self.assertEqual(p.min_turn_sec, 4.0)
        self.assertIs(InsertParams().for_main_listener().__class__, InsertParams)

    def test_main_speaker_channel(self):
        from refine_maai import main_speaker_channel
        rec = {"metadata": {"main_speaker_index": 1, "speakers": [{"name": "B"}, {"name": "A"}]}}
        self.assertEqual(main_speaker_channel(rec, {"A": 0, "B": 1}), 0)
        self.assertIsNone(main_speaker_channel({"metadata": {"speakers": [{"name": "A"}]}}, {"A": 0}))


class CountHintPromptTests(unittest.TestCase):
    """density.count_hint: 相槌の総数の目安をプロンプトに書く（傾聴 al_v1。既定は従来どおり）。"""

    def _prompt(self, **density):
        from refine_maai import build_prompt
        rec = {"metadata": {"speakers": [{"name": "A"}, {"name": "B"}]}}
        turns = [{"speaker": "A", "text": "こんにちは。", "type": "speech"},
                 {"speaker": "B", "text": "最近は、公園を、歩いています。", "type": "speech"}]
        slots = [{"slot_id": 1, "anchor_index": 1, "rel_pos": 0.5, "t_peak": 1.0, "p": 0.4,
                  "char_pos": 5, "listener": "A"}]
        cfg = RefineConfig(density=DensityParams(10.0, 16.0, 0.8, **density))
        return build_prompt(rec, turns, {}, slots, 120.0, 0.0, cfg)

    def test_default_keeps_the_weak_candidate_rule(self):
        p = self._prompt()
        self.assertIn("目標密度を満たすためだけに弱い候補を採用しない", p)
        self.assertNotIn("総数ではありません", p)

    def test_count_hint_gives_the_total(self):
        p = self._prompt(count_hint=True)
        self.assertIn("合計約20個", p)          # 10回/分 × 2.0分
        self.assertIn("約32個", p)              # 16回/分 × 2.0分
        self.assertIn("挿入候補〔◆〕は 1 個", p)
        self.assertNotIn("目標密度を満たすためだけに弱い候補を採用しない", p)


class ListenerVocabAndLeakTests(unittest.TestCase):
    """傾聴（al_v1）の挿入相槌: 語の案内（bc_vocab_style）と、声が相手に寄った相槌の引き直し（2026-11-07）。"""

    def _prompt(self, style):
        from refine_maai import build_prompt
        rec = {"metadata": {"speakers": [{"name": "A"}, {"name": "B"}]}}
        turns = [{"speaker": "A", "text": "こんにちは。", "type": "speech"},
                 {"speaker": "B", "text": "最近は、公園を、歩いています。", "type": "speech"}]
        slots = [{"slot_id": 1, "anchor_index": 1, "rel_pos": 0.5, "t_peak": 1.0, "p": 0.6,
                  "char_pos": 5, "listener": "A"}]
        q = QualityParams(bc_vocab_mix={"うん": 22, "はい": 16, "あー": 15, "へえ": 12, "おー": 8},
                          bc_vocab_style=style)
        return build_prompt(rec, turns, {}, slots, 120.0, 0.0, RefineConfig(quality=q))

    def test_listener_style_defaults_to_continuers_and_asks_for_a_cue(self):
        # ★2026-11-08 試走8の視聴「あー・おー・へー はピンポイントで。迷ったら うん・はい」→ 割合の目標をやめ、
        #   反応の語は条件を満たす位置だけ・きっかけを cue に写させる。
        p = self._prompt("listener")
        self.assertNotIn("目安の割合", p)
        self.assertIn("基本は聞いていることを示す「うん」「はい」で", p)
        self.assertIn("「あー」「へえ」「おー」は相手の話の中身にピンポイントで反応する語", p)
        self.assertIn("少しでも迷ったら「うん」「はい」にする", p)
        self.assertIn('"cue"', p)
        self.assertIn("挿入の2割まで", p)
        self.assertIn("相槌は「うん」「はい」等の短い反応なので", p)
        d = self._prompt("")
        self.assertNotIn('"cue"', d)
        self.assertIn("実際の会話に近い配分で使い分け", d)
        self.assertIn("相槌は「うん」「へえ」等の短い反応なので", d)   # 既定の文面は変えない

    def test_end_mark_only_after_a_sentence_with_content(self):
        # ★2026-11-08 「そうなんです。｜」の直後に へえ が来た → 中身のある文の後だけ「文末」
        from refine_maai import build_prompt
        rec = {"metadata": {"speakers": [{"name": "A"}, {"name": "B"}]}}
        text = "そうなんです。先週、初めてフルマラソンを走ったんです。とても疲れました"
        turns = [{"speaker": "A", "text": "こんにちは。", "type": "speech"},
                 {"speaker": "B", "text": text, "type": "speech"}]
        cp1, cp2 = text.index("先週"), text.index("とても")
        slots = [{"slot_id": 1, "anchor_index": 1, "rel_pos": 0.2, "t_peak": 1.0, "p": 0.6, "char_pos": cp1, "listener": "A"},
                 {"slot_id": 2, "anchor_index": 1, "rel_pos": 0.7, "t_peak": 2.0, "p": 0.6, "char_pos": cp2, "listener": "A"}]
        q = QualityParams(bc_vocab_mix={"うん": 1, "へえ": 1}, bc_vocab_style="listener")
        p = build_prompt(rec, turns, {}, slots, 60.0, 0.0, RefineConfig(quality=q))
        self.assertIn("〔◆1 p=0.60〕先週", p)
        self.assertIn("〔◆2 p=0.60 文末〕とても", p)

    def test_reactive_inserts_are_checked_by_the_llm(self):
        # ★2026-11-08 反応の語の insert だけを短いプロンプトで LLM に確かめさせ、合わないものを LLM の選んだ語に替える
        from refine_maai import verify_reactive_inserts, apply_ops

        class Fake:
            def __init__(self, res):
                self.res, self.prompts = res, []
            def propose(self, did, prompt, **kw):
                self.prompts.append((did, prompt))
                if isinstance(self.res, Exception):
                    raise self.res
                return self.res

        rec = {"metadata": {"speech_style": "丁寧語"}, "voice_dialogue": {"turns": []}}
        turns = [{"speaker": "A", "text": "最近どうですか？", "type": "speech"},
                 {"speaker": "B", "text": "花粉がひどくて。でも、新しい薬が効いたんです。", "type": "speech"}]
        t1 = turns[1]["text"]
        s1 = {"slot_id": 1, "anchor_index": 1, "rel_pos": 0.3, "t_peak": 1.0, "p": 0.6,
              "char_pos": t1.index("でも"), "listener": "A"}
        s2 = {"slot_id": 2, "anchor_index": 1, "rel_pos": 0.9, "t_peak": 2.0, "p": 0.6,
              "char_pos": len(t1), "listener": "A"}
        def ops():
            return [{"op": "insert", "_slot": s1, "new_text": "へえ。", "cue": "花粉がひどくて"},
                    {"op": "insert", "_slot": s2, "new_text": "おー。"},
                    {"op": "insert", "_slot": s2, "new_text": "うん。"}]
        o = ops()
        f = Fake({"items": [{"id": 1, "ok": False, "new_text": "うん。"}, {"id": 2, "ok": True}]})
        self.assertEqual(verify_reactive_inserts(f, "d", rec, turns, o), 1)
        self.assertEqual((o[0]["new_text"], o[0]["react_orig"]), ("うん。", "へえ。"))
        self.assertEqual(o[1]["new_text"], "おー。")
        did, prompt = f.prompts[0]
        self.assertEqual(did, "d.react")
        self.assertIn("「花粉がひどくて。｜」 → 聞き手の相槌: へえ。", prompt)
        self.assertIn("（直前の聞き手: 「最近どうですか？」）", prompt)
        self.assertNotIn("うん。\n", prompt.split("# 確かめる相槌")[1].split("# 出力形式")[0])  # 継続の語は確かめない
        # 替えた先が うん・はい 系でなければ「うん」、確認に失敗したら元のまま
        o = ops()
        verify_reactive_inserts(Fake({"items": [{"id": 1, "ok": False, "new_text": "なるほど。"}]}), "d", rec, turns, o)
        self.assertEqual(o[0]["new_text"], "うん。")
        o = ops()
        self.assertEqual(verify_reactive_inserts(Fake(ValueError("bad json")), "d", rec, turns, o), 0)
        self.assertEqual(o[0]["new_text"], "へえ。")
        # 来歴: 挿入相槌の refine に cue と react_orig が残る
        o = ops()[:1]
        o[0].update(new_text="うん。", react_orig="へえ。")
        r = {"voice_dialogue": {"turns": turns}}
        new_rec, *_ = apply_ops(r, turns, o, None)
        bc = [t for t in new_rec["voice_dialogue"]["turns"] if t["type"] == "backchannel"][0]
        self.assertEqual(bc["refine"]["react_orig"], "へえ。")
        self.assertEqual(bc["refine"]["cue"], "花粉がひどくて")

    def test_oo_is_kept_mid_turn_when_in_the_mix(self):
        from refine_maai import _continuer_only_inserts
        q = QualityParams(bc_vocab_mix={"うん": 1, "おー": 1})
        turns = [{"speaker": "B", "text": "三年かけて、やっと完成したんです。", "type": "speech"},
                 {"speaker": "A", "text": "おー。", "type": "backchannel", "refine": {"rel_pos": 0.4}}]
        self.assertEqual(_continuer_only_inserts(turns, set(), q, 2), 0)
        self.assertEqual(turns[1]["text"], "おー。")
        q2 = QualityParams(bc_vocab_mix={"うん": 1})
        self.assertEqual(_continuer_only_inserts(turns, set(), q2, 2), 1)   # 配分に無いと替えられる

    def test_leaky_backchannel_is_resynthesized_without_context(self):
        import sys, types
        import refine_maai as RM

        class Ver:
            # wav の先頭値で「誰の声か」を決める: 1.0 = 本人、-1.0 = 相手
            def embed(self, w, sr):
                return np.array([1.0, 0.0]) if float(w[0]) > 0 else np.array([0.0, 1.0])

        sc = types.ModuleType("speaker_check")
        sc.get_verifier = lambda model, device: Ver()
        sc.is_leak = lambda own, other, own_max, margin: own < own_max and other - own >= margin
        calls = []

        def synth_one(synth, texts, voice, ests, did, sr, short_threshold=4, speed=1.0, prompts=None):
            calls.append(texts[0])
            return [np.ones(8000, np.float32)], [False]

        gt = types.ModuleType("generate_tts")
        gt.synthesize_for_speaker = synth_one
        gt.normalize_tts_text = lambda s: s
        sf = types.ModuleType("soundfile")
        sf.read = lambda path, dtype=None: (np.ones(16000, np.float32) * (1 if "own" in path else -1), 16000)
        saved = {k: sys.modules.get(k) for k in ("speaker_check", "generate_tts", "soundfile")}
        sys.modules.update({"speaker_check": sc, "generate_tts": gt, "soundfile": sf})
        RM._LEAK_REF_EMB.clear()
        try:
            turns = [{"speaker": "A", "text": "はい。", "type": "backchannel"},
                     {"speaker": "A", "text": "うん。", "type": "backchannel"}]
            wavs = {0: -np.ones(8000, np.float32), 1: np.ones(8000, np.float32)}
            n = RM.regen_leaky_bcs(None, turns, wavs, 16000, {0, 1},
                                   {"A": {"audio_path": "own.wav"}, "B": {"audio_path": "other.wav"}},
                                   {"enabled": True}, "d" * 8, 4, 1.0)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            RM._LEAK_REF_EMB.clear()
        self.assertEqual(n, 1)                     # 相手の声の「はい。」だけ引き直す
        self.assertEqual(calls, ["はい。"])
        self.assertGreater(float(wavs[0][0]), 0)
