"""話速の均し（generate_tts.speed_factors / estimate_mora / time_stretch_wsola）の単体テスト。

合成もLLMも使わない。「速すぎるターンだけを遅くする」「遅い側とターン内のポーズは
潰さない」「読み長が推定できないターンも1本だけ浮かない」ことを確かめる。
"""
import numpy as np
import pytest

from generate_tts import (estimate_mora, speed_factors, time_stretch_wsola,
                          _num_mora)

SR = 24000


def voiced(sec: float, sr: int = SR, f0: float = 120.0) -> np.ndarray:
    """有声音の代わりのパルス列（WSOLA が波形を繋げる程度の周期性がある）。"""
    t = np.arange(int(sr * sec)) / sr
    w = (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.2 * np.sin(2 * np.pi * 3 * f0 * t))
    return (w * 0.3).astype(np.float32)


def turn(text: str, speaker: str = "A", type_: str = "speech") -> dict:
    return {"text": text, "speaker": speaker, "type": type_}


# --- モーラ推定 -------------------------------------------------------------

def test_counts_kana_and_treats_small_kana_as_zero():
    # ありがとうございます(10) + ミナ(2) + さん(2)
    assert estimate_mora("ありがとうございます、ミナさん。")[0] == pytest.approx(14.0)
    # しゅっちょう = し ゅ(0) っ ち ょ(0) う = 4
    assert estimate_mora("しゅっちょう")[0] == pytest.approx(4.0)


def test_long_vowel_is_one_mora():
    assert estimate_mora("ラーメン")[0] == pytest.approx(4.0)


def test_kanji_counts_as_two_mora():
    assert estimate_mora("会議")[0] == pytest.approx(4.0)


@pytest.mark.parametrize("digits,expected", [
    ("10", 2.0),        # じゅう
    ("100", 2.0),       # ひゃく
    ("1000", 2.0),      # せん
    ("35", 5.0),        # さんじゅうご
    ("2026", 8.0),      # にせんにじゅうろく
    ("3.5", 5.0),       # さんてんご
])
def test_reads_numbers_by_their_japanese_reading(digits, expected):
    """桁数に比例させると 1000（せん=2モーラ）を8モーラと誤るので、実際の読みで数える。"""
    assert _num_mora(digits) == pytest.approx(expected)


def test_confidence_drops_when_the_reading_is_not_determined():
    assert estimate_mora("今日は雨ですね。")[1] == pytest.approx(1.0)
    assert estimate_mora("OSはWindows10でいいのかな？")[1] < 0.6


# --- 補正係数 ---------------------------------------------------------------

def _wav_for(text: str, rate: float) -> np.ndarray:
    """指定の話速（モーラ/秒）になる長さの波形を作る。"""
    return voiced(estimate_mora(text)[0] / rate)


CFG = {"enabled": True, "target_mora_per_sec": 7.3, "deadband": 0.3,
       "ratio": 0.75, "max_slower": 0.80, "max_faster": 1.0, "min_mora": 8.0,
       "min_sec": 0.5, "min_confidence": 0.6}

TEXT = "きょうはあめですねかさをもってきましたか"          # 20モーラ


def test_slows_a_fast_turn_toward_the_target():
    turns = [turn(TEXT)]
    wavs = {0: _wav_for(TEXT, 10.0)}
    f = speed_factors(turns, wavs, SR, CFG)
    # 10.0 → 10.0 - 0.75*(10.0-7.6) = 8.2 モーラ/秒
    assert f[0] == pytest.approx(8.2 / 10.0, abs=0.01)


def test_leaves_a_turn_inside_the_deadband_alone():
    turns = [turn(TEXT)]
    wavs = {0: _wav_for(TEXT, 7.5)}
    assert speed_factors(turns, wavs, SR, CFG) == {}


def test_never_speeds_up_a_slow_turn():
    """ポーズを入れた発話は遅い側に出る。ここを詰めると Stage2 が入れた間が消える。"""
    turns = [turn(TEXT)]
    wavs = {0: _wav_for(TEXT, 5.0)}
    assert speed_factors(turns, wavs, SR, CFG) == {}


def test_caps_how_far_a_turn_is_stretched():
    turns = [turn(TEXT)]
    wavs = {0: _wav_for(TEXT, 20.0)}
    assert speed_factors(turns, wavs, SR, CFG)[0] == pytest.approx(0.80)


def test_backchannels_are_left_alone():
    turns = [turn(TEXT, type_="backchannel")]
    wavs = {0: _wav_for(TEXT, 12.0)}
    assert speed_factors(turns, wavs, SR, CFG) == {}


def test_short_turns_are_left_alone():
    turns = [turn("うんそうだね")]                      # 6モーラ < min_mora
    wavs = {0: voiced(0.6)}
    assert speed_factors(turns, wavs, SR, CFG) == {}


def test_unreadable_turns_borrow_the_speakers_median_factor():
    """数字・英字だらけのターンは読み長が読めない。放っておくと1本だけ速いまま残る。"""
    unsure = "Windows10とOneDriveのSLAはOKですか"
    turns = [turn(TEXT, "A"), turn(TEXT, "A"), turn(unsure, "A")]
    wavs = {0: _wav_for(TEXT, 10.0), 1: _wav_for(TEXT, 10.0),
            2: voiced(2.0)}
    f = speed_factors(turns, wavs, SR, CFG)
    assert estimate_mora(unsure)[1] < 0.6
    assert f[2] == pytest.approx(f[0], abs=1e-6)


def test_unreadable_turn_is_skipped_when_the_speaker_has_no_measurable_turn():
    unsure = "Windows10とOneDriveのSLAはOKですか"
    turns = [turn(TEXT, "A"), turn(unsure, "B")]
    wavs = {0: _wav_for(TEXT, 10.0), 1: voiced(2.0)}
    assert 1 not in speed_factors(turns, wavs, SR, CFG)


def test_failed_turns_are_left_alone():
    turns = [turn(TEXT)]
    wavs = {0: _wav_for(TEXT, 12.0)}
    assert speed_factors(turns, wavs, SR, CFG, failed_turns={0}) == {}


# --- WSOLA ------------------------------------------------------------------

@pytest.mark.parametrize("rate", [0.72, 0.85, 1.15])
def test_wsola_hits_the_requested_length(rate):
    w = voiced(2.0)
    out = time_stretch_wsola(w, rate, SR)
    assert len(out) / len(w) == pytest.approx(1.0 / rate, rel=0.05)


def test_wsola_preserves_pitch():
    w = voiced(2.0, f0=150.0)
    out = time_stretch_wsola(w, 0.8, SR)
    def f0(x):
        spec = np.abs(np.fft.rfft(x[:SR] * np.hanning(min(len(x), SR))))
        return float(np.fft.rfftfreq(min(len(x), SR), 1 / SR)[int(np.argmax(spec))])
    assert f0(out) == pytest.approx(f0(w), abs=3.0)


def test_wsola_keeps_the_signal_level():
    w = voiced(2.0)
    out = time_stretch_wsola(w, 0.8, SR)
    rms = lambda x: float(np.sqrt((x ** 2).mean()))
    assert rms(out) == pytest.approx(rms(w), rel=0.15)


def test_wsola_returns_very_short_input_untouched():
    w = voiced(0.02)
    assert time_stretch_wsola(w, 0.8, SR) is w
