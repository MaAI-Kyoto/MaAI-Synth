"""合成失敗→無音代替が出た対話は音声を書き出さないことのテスト（Stage4）。

fish が落ちている等で個別合成が失敗すると、その発話は推定長ぶんの無音で代替される。
無音入りの音声が「成功」として出てくると使えないデータが混ざるため、1発話でも無音代替が
出た対話は wav / manifest / transcript / 発話wavキャッシュを一切作らずに落とす。
wav が無ければ次回実行の resume 対象として残るので、ログを見て直してから再実行できる。
"""
import json
from pathlib import Path

import numpy as np
import pytest

import generate_tts
from generate_tts import DummySynth, drop_if_synth_failed, run

SR = 8000
FAIL_TEXT = "承知しました。"


class FlakySynth(DummySynth):
    """特定のテキストだけ必ず失敗する合成器（バッチでも個別でも落ちる）。"""

    def synth(self, texts, voice, prompts=None):
        if any(FAIL_TEXT in t for t in texts):
            raise RuntimeError("synth failed")
        return super().synth(texts, voice, prompts=prompts)


def _record(dialogue_id: str = "dialogue-1") -> dict:
    return {
        "id": dialogue_id,
        "metadata": {
            "relationship": "同僚",
            "speech_style": "敬語",
            "speakers": [{"name": "A", "gender": "female", "age_range": "30代"},
                         {"name": "B", "gender": "male", "age_range": "30代"}],
        },
        "voice_dialogue": {"turns": [
            {"speaker": "A", "text": "明日の会議について予定を確認します。",
             "type": "speech", "start_offset": 0.0, "duration": 4.0},
            {"speaker": "B", "text": "はい。", "type": "backchannel",
             "start_offset": -0.5, "duration": 0.4},
            {"speaker": "B", "text": FAIL_TEXT, "type": "speech",
             "start_offset": 0.2, "duration": 1.0},
        ]},
    }


def _config(root: Path, input_file: Path) -> dict:
    return {
        "voice": {"output_file": str(input_file)},
        "tts": {
            "engine": "google_cloud",
            "sample_rate": SR,
            "audio_dir": str(root / "audio"),
            "manifest_file": str(root / "audio_manifest.jsonl"),
            "google_cloud": {"voices": [
                {"id": "voice-a", "voice_name": "voice-a", "gender": "female"},
                {"id": "voice-b", "voice_name": "voice-b", "gender": "male"},
            ]},
            "retime": {"enabled": False},
        },
    }


def _run(tmp_path: Path, monkeypatch, synth_cls) -> tuple[int, Path]:
    input_file = tmp_path / "voice_dialogues.jsonl"
    input_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n")
    monkeypatch.setattr(generate_tts, "DummySynth", synth_cls)
    done = run(_config(tmp_path, input_file), dry_run=True, limit=1, retime=False)
    return done, tmp_path / "audio" / "plain"


def test_drop_if_synth_failed_flags_only_failed_dialogues():
    ok = [{"index": 0, "synth_failed": False}, {"index": 1, "synth_failed": False}]
    assert drop_if_synth_failed("d1", ok) is False
    assert drop_if_synth_failed("d1", ok + [{"index": 2, "synth_failed": True}]) is True


def test_failed_dialogue_writes_nothing(tmp_path, monkeypatch, caplog):
    with caplog.at_level("ERROR", logger="generate_tts"):
        done, audio_dir = _run(tmp_path, monkeypatch, FlakySynth)

    assert done == 0
    assert any("無音代替のため対話を破棄" in r.message for r in caplog.records)
    assert not (audio_dir / "dialogue-1.wav").exists()
    assert not (audio_dir / "dialogue-1.json").exists()      # 対話ごとの書き起こし
    manifest = tmp_path / "audio_manifest_plain.jsonl"
    assert not manifest.exists() or manifest.read_text().strip() == ""
    transcripts = tmp_path / "transcripts_plain.jsonl"
    assert not transcripts.exists() or transcripts.read_text().strip() == ""


def test_failed_dialogue_leaves_no_turn_cache(tmp_path, monkeypatch):
    """無音入りの発話wavキャッシュは残さない（refine / --recompose が拾わないように）。"""
    cache = tmp_path / "cache"
    record = _record()
    voices = [{"id": "voice-a", "gender": "female", "age_range": "30代"},
              {"id": "voice-b", "gender": "male", "age_range": "30代"}]

    r = generate_tts.synthesize_dialogue(FlakySynth(SR), record, voices, SR,
                                         turn_cache_dir=cache)
    assert r is not None and drop_if_synth_failed("dialogue-1", r[1]) is True
    assert not (cache / "dialogue-1" / "meta.json").exists()

    # 正常な対話ではこれまで通りキャッシュを書く
    generate_tts.synthesize_dialogue(DummySynth(SR), record, voices, SR,
                                     turn_cache_dir=cache)
    assert (cache / "dialogue-1" / "meta.json").exists()


def test_healthy_dialogue_still_written(tmp_path, monkeypatch):
    done, audio_dir = _run(tmp_path, monkeypatch, DummySynth)

    assert done == 1
    assert (audio_dir / "dialogue-1.wav").exists()
    manifest = tmp_path / "audio_manifest_plain.jsonl"
    line = json.loads(manifest.read_text().splitlines()[0])
    assert line["id"] == "dialogue-1"
    assert not any(t["synth_failed"] for t in line["turns"])


def test_second_run_retries_dropped_dialogue(tmp_path, monkeypatch):
    """落とした対話は wav が無いので、同じコマンドの再実行で再合成される。"""
    input_file = tmp_path / "voice_dialogues.jsonl"
    input_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n")
    cfg = _config(tmp_path, input_file)

    monkeypatch.setattr(generate_tts, "DummySynth", FlakySynth)
    assert run(cfg, dry_run=True, limit=1, retime=False) == 0

    monkeypatch.setattr(generate_tts, "DummySynth", DummySynth)  # fish 復旧後の再実行
    assert run(cfg, dry_run=True, limit=1, retime=False) == 1
    assert (tmp_path / "audio" / "plain" / "dialogue-1.wav").exists()


def test_refine_drops_failed_dialogue(tmp_path, monkeypatch, caplog):
    """--refine（refine_maai.run_refine）でも同じく何も書き出さない。"""
    from refine_maai import run_refine

    input_file = tmp_path / "voice_dialogues.jsonl"
    input_file.write_text(json.dumps(_record(), ensure_ascii=False) + "\n")
    cfg = _config(tmp_path, input_file)
    cfg["tts"]["retime"] = {"tag": "maai"}
    cfg["tts"]["refine"] = {"tag": "ref", "ops_dir": str(tmp_path / "ops"),
                            "refined_records_file": str(tmp_path / "refined.jsonl")}
    monkeypatch.setattr(generate_tts, "DummySynth", FlakySynth)

    with caplog.at_level("ERROR"):
        done = run_refine(cfg, dry_run=True, limit=1)

    variant = "maai_ref_dryrun"   # ★2026-11-22 バリアント名からエンジン名を外した
    assert done == 0
    # 例外や別要因のスキップではなく「無音破棄」で落ちたこと（集計にも出ること）
    assert any("無音代替のため対話を破棄" in r.message for r in caplog.records)
    assert any("無音代替が出た 1件を破棄" in r.message for r in caplog.records)
    assert not (tmp_path / "audio" / variant / "dialogue-1.wav").exists()
    manifest = tmp_path / f"audio_manifest_{variant}.jsonl"
    assert not manifest.exists() or manifest.read_text().strip() == ""
    refined = tmp_path / f"refined_{variant}.jsonl"
    assert not refined.exists() or refined.read_text().strip() == ""


def test_silence_turn_is_marked_failed_in_manifest():
    """compose_dialogue は無音代替ターンに synth_failed を立てる（破棄判定の入力）。"""
    record = _record()
    turns = record["voice_dialogue"]["turns"]
    voices = [{"id": "voice-a", "gender": "female", "age_range": "30代"},
              {"id": "voice-b", "gender": "male", "age_range": "30代"}]
    wav_by_turn = {i: np.zeros(int(SR * 0.5), dtype=np.float32)
                   for i in range(len(turns))}
    _stereo, manifest_turns = generate_tts.compose_dialogue(
        record, voices, SR, wav_by_turn, {2})
    assert [t["synth_failed"] for t in manifest_turns] == [False, False, True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
