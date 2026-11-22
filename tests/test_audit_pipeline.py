"""audit_pipeline の検査と、Stage 別の破棄。

LLM も音声合成も呼ばない。合成コーパスを temp に組んで確かめるのは

  - 検査が「refine が黙って落とす条件」（話者がちょうど2人でない）を拾うか
  - 健全な対話を巻き込まないか
  - 同じ id に複数の所見があるとき、いちばん手前の Stage を採るか
  - 一度 stage2 に差し戻して再発した id を stage1 に格上げするか
  - 破棄が Stage 別に正しい台帳だけを書き換えるか
    （stage4 = 音声だけ / stage2 = 対話も / stage1 = メタデータも）
  - Stage4 の成果物を canonical と per-shard part の両方から抜くか
    （抜き損ねると resume がファイル存在で素通しして永久に埋まらない）
"""

import json
import tempfile
import unittest
from pathlib import Path

from audit_pipeline import audit, plan_stages, repair, resolve_layout, variant_files

GOOD, BAD3, EMPTY = "id-good", "id-3speakers", "id-empty-text"


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_ids(path: Path):
    if not path.exists():
        return []
    return [json.loads(ln)["id"] for ln in open(path) if ln.strip()]


def _voice_record(did, turn_speakers, texts=None):
    """voice_dialogues.jsonl の1レコード。metadata の話者は常に あかり/たける の2人。"""
    texts = texts or ["こんにちは。"] * len(turn_speakers)
    return {
        "id": did,
        "metadata": {"id": did, "speakers": [{"name": "あかり", "gender": "female"},
                                             {"name": "たける", "gender": "male"}]},
        "speaker_profiles": [{"name": "あかり", "gender": "female"},
                             {"name": "たける", "gender": "male"}],
        "voice_dialogue": {"turns": [{"speaker": s, "text": t, "type": "speech"}
                                     for s, t in zip(turn_speakers, texts)]},
    }


def _config(root: Path) -> dict:
    return {
        "target_dialogues": 3,
        "metadata": {"output_file": str(root / "metadata.jsonl")},
        "dialogue": {"output_file": str(root / "dialogues.jsonl")},
        "voice": {"output_file": str(root / "voice_dialogues.jsonl")},
        "tts": {
            "engine": "fish_speech",
            "audio_dir": str(root / "audio"),
            "manifest_file": str(root / "audio_manifest.jsonl"),
            "refine": {"ops_dir": str(root / "refine_ops"),
                       "refined_records_file": str(root / "voice_dialogues_refined.jsonl")},
        },
    }


def _build_corpus(root: Path):
    """3対話ぶんの台帳と Stage4 成果物を作る。

    id-good        すべて健全
    id-3speakers   話者が3人（refine が黙って落とす）＋ base が 3ch ＋ refine 成果物なし
    id-empty-text  空テキストのターンがあるが、音声は一通り出来ている
    """
    ids = [GOOD, BAD3, EMPTY]
    _write_jsonl(root / "metadata.jsonl", [{"id": i} for i in ids])
    _write_jsonl(root / "dialogues.jsonl", [{"id": i} for i in ids])
    _write_jsonl(root / "voice_dialogues.jsonl", [
        _voice_record(GOOD, ["あかり", "たける", "あかり"]),
        _voice_record(BAD3, ["あかり", "たける", "たけし"]),   # ★3人目は打ち間違い
        _voice_record(EMPTY, ["あかり", "たける"], texts=["こんにちは。", ""]),
    ])

    base_dir = root / "audio" / "maai"
    ref_dir = root / "audio" / "maai_ref"
    for d in (base_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    # base manifest は per-shard part のまま（マージ前の状態を再現する）
    _write_jsonl(root / "audio_manifest_maai.part-1of2.jsonl", [
        {"id": GOOD, "channels": 2, "duration_sec": 10.0},
        {"id": BAD3, "channels": 3, "duration_sec": 12.0,
         "channel_speakers": {"0": "あかり", "1": "たける", "2": "たけし"}},
    ])
    _write_jsonl(root / "audio_manifest_maai.part-2of2.jsonl", [
        {"id": EMPTY, "channels": 2, "duration_sec": 9.0},
    ])
    # refine 側はマージ済み（canonical）。BAD3 だけ行が無い＝スキップされた状態
    _write_jsonl(root / "audio_manifest_maai_ref.jsonl", [
        {"id": GOOD, "channels": 2, "duration_sec": 11.0},
        {"id": EMPTY, "channels": 2, "duration_sec": 9.5},
    ])
    _write_jsonl(root / "voice_dialogues_refined_maai_ref.jsonl", [
        {"id": GOOD, "voice_dialogue": {"turns": [{"speaker": "あかり", "text": "はい。"}]}},
        {"id": EMPTY, "voice_dialogue": {"turns": [{"speaker": "あかり", "text": "はい。"}]}},
    ])

    for i in ids:
        (base_dir / f"{i}.wav").write_bytes(b"RIFF-base")
        (root / "audio" / "plain" / "turns" / i).mkdir(parents=True, exist_ok=True)
        (root / "audio" / "plain" / "turns" / i / "0000.wav").write_bytes(b"RIFF-turn")
        (root / "refine_ops" / "maai_ref").mkdir(parents=True, exist_ok=True)
        (root / "refine_ops" / "maai_ref" / f"{i}.deadbeef.json").write_text("{}")
        if i != BAD3:                       # BAD3 は refine が落ちたので ref wav が無い
            (ref_dir / f"{i}.wav").write_bytes(b"RIFF-ref")


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _build_corpus(self.root)
        self.config = _config(self.root)
        self.layout = resolve_layout(self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def _audit(self, **kw):
        return audit(self.layout, self.config, **kw)

    def _codes(self, findings, did):
        return {f.code for f in findings if f.id == did}

    # --- 検査 -------------------------------------------------------------

    def test_layout_matches_stage4_naming(self):
        """パスは Stage4 の実装と同じ命名規則で導けている。★2026-11-22 バリアント名からエンジン名を外した（plain / maai / maai_ref）。"""
        self.assertEqual(self.layout.base_audio_dir.name, "maai")
        self.assertEqual(self.layout.ref_audio_dir.name, "maai_ref")
        self.assertEqual(self.layout.refined_records.name,
                         "voice_dialogues_refined_maai_ref.jsonl")
        self.assertEqual(self.layout.ops_dir.name, "maai_ref")
        self.assertEqual(self.layout.turn_cache_dir,
                         self.root / "audio" / "plain" / "turns")

    def test_healthy_dialogue_is_not_flagged(self):
        self.assertEqual(self._codes(self._audit(), GOOD), set())

    def test_three_speakers_is_flagged(self):
        """★これが今回の取りこぼし。refine の `!= 2` を検査側で先取りする。"""
        codes = self._codes(self._audit(), BAD3)
        self.assertIn("speaker_count", codes)
        self.assertIn("speaker_unknown", codes)     # metadata.speakers に無い名前
        self.assertIn("bad_voice_record", codes)    # Stage3 のゲートにも掛かる
        self.assertIn("bad_base_channels", codes)   # 3ch の wav が出来ている
        self.assertIn("missing_refined", codes)     # refine が落ちた
        self.assertIn("missing_ref", codes)

    def test_empty_text_is_flagged(self):
        self.assertIn("bad_voice_record", self._codes(self._audit(), EMPTY))

    def test_pre_mode_skips_stage4_checks(self):
        """--pre は Stage4 を回す前でも使えるよう、成果物の所見を出さない。"""
        codes = self._codes(self._audit(pre=True), BAD3)
        self.assertIn("speaker_count", codes)
        self.assertNotIn("bad_base_channels", codes)
        self.assertNotIn("missing_refined", codes)

    def test_backlog_is_not_a_finding(self):
        """metadata にあるが未生成なのは「積み残し」であって欠損ではない。"""
        _write_jsonl(self.root / "metadata.jsonl",
                     [{"id": i} for i in (GOOD, BAD3, EMPTY, "id-not-yet")])
        self.assertEqual(self._codes(self._audit(), "id-not-yet"), set())
        self.assertEqual(self._codes(self._audit(check_backlog=True), "id-not-yet"),
                         {"missing_dialogue"})

    def test_only_ids_narrows_scope(self):
        findings = self._audit(only_ids={EMPTY})
        self.assertEqual({f.id for f in findings}, {EMPTY})

    # --- 戻るべき Stage ---------------------------------------------------

    def test_plan_takes_earliest_stage(self):
        """BAD3 は stage2 と stage4 の所見を持つが、手前の stage2 を採る。"""
        plan = plan_stages(self._audit(), self.layout)
        self.assertEqual(plan[BAD3], "stage2")
        self.assertNotIn(GOOD, plan)

    def test_repeat_failure_escalates_to_stage1(self):
        """同じメタデータで再発したら、そのメタデータ自体を地雷とみなす。"""
        self.layout.repair_history.write_text(
            json.dumps({"id": BAD3, "stage": "stage2", "at": "2026-01-01"}) + "\n")
        self.assertEqual(plan_stages(self._audit(), self.layout)[BAD3], "stage1")

    def test_force_stage_overrides(self):
        plan = plan_stages(self._audit(), self.layout, force_stage="stage4")
        self.assertEqual(set(plan.values()), {"stage4"})

    # --- 破棄 -------------------------------------------------------------

    def test_repair_stage4_only_keeps_text_ledgers(self):
        """stage4 差し戻しは音声だけ捨てる。対話・メタデータには触らない。"""
        repair(self.layout, {BAD3: "stage4"}, purge="move")
        self.assertEqual(_read_ids(self.layout.metadata_file), [GOOD, BAD3, EMPTY])
        self.assertEqual(_read_ids(self.layout.dialogue_file), [GOOD, BAD3, EMPTY])
        self.assertEqual(_read_ids(self.layout.voice_file), [GOOD, BAD3, EMPTY])

    def test_repair_stage2_retires_dialogue_but_keeps_metadata(self):
        repair(self.layout, {BAD3: "stage2"}, purge="move")
        self.assertEqual(_read_ids(self.layout.metadata_file), [GOOD, BAD3, EMPTY])
        self.assertEqual(_read_ids(self.layout.dialogue_file), [GOOD, EMPTY])
        self.assertEqual(_read_ids(self.layout.voice_file), [GOOD, EMPTY])
        # 捨てたものは消さずに退避してある
        self.assertEqual(_read_ids(self.root / "dialogues_discarded.jsonl"), [BAD3])

    def test_repair_stage1_retires_metadata_too(self):
        repair(self.layout, {BAD3: "stage1"}, purge="move")
        self.assertEqual(_read_ids(self.layout.metadata_file), [GOOD, EMPTY])
        self.assertEqual(_read_ids(self.root / "metadata_discarded.jsonl"), [BAD3])

    def test_repair_purges_stage4_artifacts_from_parts_and_canonical(self):
        """manifest は canonical と per-shard part の両方から抜く必要がある。"""
        repair(self.layout, {BAD3: "stage2", EMPTY: "stage2"}, purge="move")

        self.assertFalse((self.layout.base_audio_dir / f"{BAD3}.wav").exists())
        self.assertFalse((self.layout.ref_audio_dir / f"{EMPTY}.wav").exists())
        self.assertFalse((self.layout.turn_cache_dir / BAD3).exists())
        self.assertFalse(list(self.layout.ops_dir.glob(f"{BAD3}.*.json")))

        # part（BAD3 は part-1、EMPTY は part-2）と canonical の両方から消えている
        self.assertEqual(
            _read_ids(self.root / "audio_manifest_maai.part-1of2.jsonl"), [GOOD])
        self.assertEqual(
            _read_ids(self.root / "audio_manifest_maai.part-2of2.jsonl"), [])
        self.assertEqual(
            _read_ids(self.root / "audio_manifest_maai_ref.jsonl"), [GOOD])
        self.assertEqual(_read_ids(self.layout.refined_records), [GOOD])

        # 健全な対話の成果物は残っている
        self.assertTrue((self.layout.base_audio_dir / f"{GOOD}.wav").exists())
        self.assertTrue((self.layout.ref_audio_dir / f"{GOOD}.wav").exists())
        self.assertTrue((self.layout.turn_cache_dir / GOOD).exists())

    def test_purged_audio_is_quarantined_not_deleted(self):
        repair(self.layout, {BAD3: "stage2"}, purge="move")
        moved = list((self.root / "_audit_quarantine").rglob(f"{BAD3}.wav"))
        self.assertTrue(moved, "退避先に wav が見つからない")

    def test_purge_delete_removes_files(self):
        repair(self.layout, {BAD3: "stage2"}, purge="delete")
        self.assertFalse((self.root / "_audit_quarantine").exists())
        self.assertFalse((self.layout.base_audio_dir / f"{BAD3}.wav").exists())

    def test_repair_clears_skip_records_so_regeneration_is_not_blocked(self):
        """恒久スキップの記録が残っていると作り直しても素通しされる。"""
        _write_jsonl(self.layout.voice_skip, [{"id": BAD3, "reason": "話者不明のターン"}])
        repair(self.layout, {BAD3: "stage2"}, purge="move")
        self.assertEqual(_read_ids(self.layout.voice_skip), [])
        self.assertEqual(_read_ids(self.root / "voice_dialogues_skipped_history.jsonl"),
                         [BAD3])

    def test_discarded_sidecar_is_not_mistaken_for_a_shard_part(self):
        """_retire_records が作る `<stem>.part-1of2_discarded.jsonl` は part ではない。

        glob だけで part を拾うと、破棄した行が次の検査で生き返って
        「消したはずの対話がまだ manifest にある」ことになる。
        """
        # stage4 差し戻しなら voice_dialogues に残るので、2回目の検査対象になる
        repair(self.layout, {BAD3: "stage4"}, purge="move")
        sidecar = self.root / "audio_manifest_maai.part-1of2_discarded.jsonl"
        self.assertTrue(sidecar.exists(), "退避先が出来ていない（前提が崩れている）")
        self.assertNotIn(sidecar, variant_files(self.layout.base_manifest))
        # 2回目の検査でも BAD3 は base manifest から消えたままに見える
        self.assertIn("missing_base", self._codes(self._audit(), BAD3))

    def test_repair_is_recorded_in_history(self):
        repair(self.layout, {BAD3: "stage2"}, purge="move")
        hist = [json.loads(ln) for ln in open(self.layout.repair_history) if ln.strip()]
        self.assertEqual([(h["id"], h["stage"]) for h in hist], [(BAD3, "stage2")])

    def test_audit_is_clean_after_repair_and_regeneration(self):
        """破棄 → 作り直し（を模した書き戻し）で所見が消えること。"""
        repair(self.layout, {BAD3: "stage2"}, purge="move")
        # Stage2/3 が同じ id で正しい台本を作り直した状態を作る
        with open(self.layout.dialogue_file, "a") as f:
            f.write(json.dumps({"id": BAD3}, ensure_ascii=False) + "\n")
        with open(self.layout.voice_file, "a") as f:
            f.write(json.dumps(_voice_record(BAD3, ["あかり", "たける"]),
                               ensure_ascii=False) + "\n")
        # Stage4 前なので --pre で見る（Stage4 成果物はまだ無い）
        self.assertEqual(self._codes(self._audit(pre=True), BAD3), set())


if __name__ == "__main__":
    unittest.main()
