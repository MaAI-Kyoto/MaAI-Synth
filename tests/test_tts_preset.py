"""TTS method names (tts.engine: irodori-yomi-v4) are expanded from tts_presets/ (tts_presets.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tts_presets  # noqa: E402


def test_preset_overrides_fish_settings_and_keeps_dataset_ones():
    cfg = {"tts": {"engine": "irodori-yomi-v4", "voices_file": "voices/x.csv",
                   "fish_speech": {"backend": "vllm_omni", "multiturn": True, "concurrency": 4,
                                   "reading_normalize": {"enabled": True, "verb_kata": True}},
                   "refine": {"density": {"max_bc_per_min": 11.0}}}}
    t = tts_presets.apply(cfg)["tts"]
    assert t["method"] == "irodori-yomi-v4"
    assert t["engine"] == "fish_speech"                       # the synthesizer that actually runs
    f = t["fish_speech"]
    assert f["backend"] == "irodori_vllm" and f["checkpoint"] == "Irodori-TTS-v4.1-Small-Yomi"
    assert f["concurrency"] == 32                              # the method wins
    assert f["multiturn"] is True                              # keys the method does not set are kept
    assert f["irodori"]["num_steps"] == 24 and f["irodori"]["refs_sec"] == 60
    assert f["best_of"]["criterion"] == "quality" and f["best_of"]["staged"]["first"] == 4
    assert f["reading_normalize"] == {"enabled": True, "verb_kata": True}
    assert t["voices_file"] == "voices/x.csv"
    assert t["refine"]["density"]["max_bc_per_min"] == 11.0
    assert t["loudness"]["bc_gain_db"] == -5.0 and t["audio_format"] == "flac"


def test_unknown_engine_is_left_alone():
    cfg = {"tts": {"engine": "fish_speech", "fish_speech": {"backend": "fish"}}}
    assert tts_presets.apply(cfg) == cfg
    assert tts_presets.apply({"llm": {}}) == {"llm": {}}


def test_shipped_configs_resolve_to_the_default_method():
    import generate_tts as G
    import main as M
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        if p.name == "base.yaml":
            continue
        t = G.load_config(str(p))["tts"]
        assert t.get("method") == "irodori-yomi-v4", p.name
        assert t["fish_speech"]["backend"] == "irodori_vllm", p.name
        assert t["retime"]["backend"] == "maaibatch", p.name
        assert M.load_config(str(p))["tts"] == t, p.name       # main.py loads the same
