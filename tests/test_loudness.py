"""音量均し（generate_tts.loudness_gains / speech_level_db）の単体テスト。

合成もLLMも使わない。ターンごとの音量ばらつきを潰しつつ、
「許容範囲内のゆらぎは残す」「上げすぎない」「相槌は少し低い目標」を確かめる。
"""
import numpy as np
import pytest

from generate_tts import loudness_gains, speech_level_db

SR = 24000


def tone(db: float, sec: float = 1.0, sr: int = SR) -> np.ndarray:
    """指定 dBFS の RMS を持つ帯域雑音（発話の代わり）。"""
    rng = np.random.default_rng(0)
    w = rng.standard_normal(int(sr * sec)).astype(np.float32)
    w /= np.sqrt((w ** 2).mean())
    return (w * (10.0 ** (db / 20.0))).astype(np.float32)


def applied(gains, i, wav, sr=SR):
    return speech_level_db(wav * gains.get(i, 1.0), sr)


def test_speech_level_db_matches_rms():
    assert speech_level_db(tone(-20.0), SR) == pytest.approx(-20.0, abs=0.5)


def test_pulls_outliers_to_target_and_keeps_small_variation():
    turns = [{"type": "speech"} for _ in range(4)]
    wavs = {0: tone(-30.0), 1: tone(-21.0), 2: tone(-12.0), 3: tone(-19.0)}
    cfg = {"target_dbfs": -20.0, "deadband_db": 2.0, "max_gain_db": 12.0,
           "peak_ceiling_dbfs": -1.0}
    g = loudness_gains(turns, wavs, SR, cfg)
    # 目標から2dB以内(1,3)は無補正、外側(0,2)は境界まで引き戻す
    assert 1 not in g and 3 not in g
    assert applied(g, 0, wavs[0]) == pytest.approx(-22.0, abs=0.6)
    assert applied(g, 2, wavs[2]) == pytest.approx(-18.0, abs=0.6)


def test_max_gain_caps_correction():
    turns = [{"type": "speech"}]
    wavs = {0: tone(-45.0)}
    g = loudness_gains(turns, wavs, SR,
                       {"target_dbfs": -20.0, "deadband_db": 2.0, "max_gain_db": 6.0})
    assert 20 * np.log10(g[0]) == pytest.approx(6.0, abs=0.01)


def test_peak_ceiling_prevents_clipping():
    turns = [{"type": "speech"}]
    w = tone(-30.0)
    w[100] = 0.5                      # 突出したピークを持つ小音量ターン
    g = loudness_gains(turns, {0: w}, SR,
                       {"target_dbfs": -20.0, "deadband_db": 2.0, "max_gain_db": 12.0,
                        "peak_ceiling_dbfs": -1.0})
    # 音量だけ見れば +8dB 欲しいが、ピークが天井に当たるぶん手前で止める
    assert 20 * np.log10(g[0]) < 8.0
    assert float(np.abs(w * g[0]).max()) <= 10.0 ** (-1.0 / 20.0) + 1e-6


def test_ceiling_does_not_attenuate_untouched_loud_turn():
    """既に大きいターンを天井のために下げたりはしない（リミッタではない）。"""
    turns = [{"type": "speech"}]
    w = tone(-21.0)
    w[100] = 0.95                     # 目標レンジ内だがピークは天井超え
    g = loudness_gains(turns, {0: w}, SR,
                       {"target_dbfs": -20.0, "deadband_db": 2.0,
                        "peak_ceiling_dbfs": -1.0})
    assert g == {}


def test_backchannel_target_is_lower():
    turns = [{"type": "speech"}, {"type": "backchannel"}]
    wavs = {0: tone(-30.0), 1: tone(-30.0)}
    g = loudness_gains(turns, wavs, SR,
                       {"target_dbfs": -20.0, "deadband_db": 2.0, "max_gain_db": 12.0,
                        "bc_offset_db": -3.0})
    assert 20 * np.log10(g[0]) - 20 * np.log10(g[1]) == pytest.approx(3.0, abs=0.1)


def test_skips_failed_and_too_short_turns():
    turns = [{"type": "speech"}, {"type": "speech"}]
    wavs = {0: tone(-30.0), 1: tone(-30.0, sec=0.05)}
    g = loudness_gains(turns, wavs, SR,
                       {"target_dbfs": -20.0, "deadband_db": 2.0, "min_sec": 0.2},
                       failed_turns={0})
    assert g == {}
