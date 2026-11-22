"""テイク選択（generate_tts.score_takes / take_features / voiced_f0）の単体テスト。

合成もLLMも使わない。「外れテイクを落として中庸を採る」「話速だけは固定目標にもできる」
「ヒスの乗ったテイクは避ける」「測れないテイクは採らない」を確かめる。
"""
import numpy as np
import pytest

from generate_tts import (score_takes, take_features, voiced_f0, f0_sd_semitones,
                          estimate_mora)

SR = 24000


def voiced(sec: float, f0: float = 120.0, amp: float = 0.3,
           vibrato: float = 0.0, sr: int = SR) -> np.ndarray:
    """有声音の代わりの倍音つき信号。vibrato で F0 を揺らす（＝抑揚を作る）。"""
    t = np.arange(int(sr * sec)) / sr
    inst = f0 * (1.0 + vibrato * np.sin(2 * np.pi * 1.5 * t))
    ph = 2 * np.pi * np.cumsum(inst) / sr
    w = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return (w * amp).astype(np.float32)


def hiss(sec: float, level: float = 0.01, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(sr * sec)) * level).astype(np.float32)


# --- F0 -------------------------------------------------------------------

def test_voiced_f0_finds_the_pitch():
    f0 = voiced_f0(voiced(1.0, f0=150.0), SR, fmin=60, fmax=260)
    assert np.median(f0) == pytest.approx(150.0, rel=0.03)


def test_f0_sd_grows_with_vibrato():
    flat = f0_sd_semitones(voiced(1.5, f0=150.0), SR, "male")
    wobbly = f0_sd_semitones(voiced(1.5, f0=150.0, vibrato=0.12), SR, "male")
    assert wobbly > flat + 0.5


def test_f0_range_is_narrowed_by_gender():
    """男声の範囲で女声を測ると倍音を掴んでオクターブを誤る。範囲は性別で切る。"""
    w = voiced(1.0, f0=220.0)
    assert np.median(voiced_f0(w, SR, *(110.0, 450.0))) == pytest.approx(220.0, rel=0.03)


# --- 特徴量 ---------------------------------------------------------------

def test_take_features_measures_rate_from_the_trimmed_length():
    w = voiced(2.0)
    f = take_features(w, SR, mora=20.0, gender="male")
    assert f["rate"] == pytest.approx(10.0, rel=0.05)


def test_take_features_returns_none_for_a_too_short_take():
    assert take_features(voiced(0.05), SR, mora=20.0, gender="male") is None


# --- スコア ---------------------------------------------------------------

CFG = {"enabled": True, "takes": 5, "w_rate": 1.0, "w_f0sd": 0.5, "w_level": 0.5,
       "rate_scale": 1.0, "f0sd_scale": 1.0, "level_scale": 3.0,
       "hiss_sec": 0.6, "hiss_penalty": 2.0}


def feat(rate, f0sd=3.0, level=-20.0, hiss=0.0):
    return {"rate": rate, "f0sd": f0sd, "level": level, "hiss": hiss, "dur": 2.0}


def test_picks_the_median_take_and_drops_the_outliers():
    feats = [feat(12.0), feat(8.0), feat(7.9), feat(8.1), feat(6.0)]
    assert int(np.argmin(score_takes(feats, CFG))) == 1     # 中央値 8.0 にいちばん近い


def test_a_fixed_rate_target_pulls_the_whole_group_down():
    """テイクがそろって速いとき、中央値基準では速いまま。目標を数値で置くと引き戻せる。"""
    feats = [feat(10.0), feat(10.2), feat(9.0), feat(10.4), feat(10.1)]
    assert int(np.argmin(score_takes(feats, CFG))) == 4                  # 中央値=10.1
    cfg = dict(CFG, rate_target=7.3)
    assert int(np.argmin(score_takes(feats, cfg))) == 2                  # いちばん遅い


def test_avoids_a_take_with_hiss_even_if_its_rate_is_perfect():
    feats = [feat(8.0, hiss=1.6), feat(8.4), feat(7.6), feat(8.0, hiss=0.0)]
    assert int(np.argmin(score_takes(feats, CFG))) == 3


def test_unmeasurable_takes_are_never_selected():
    feats = [None, feat(9.0), None]
    s = score_takes(feats, CFG)
    assert s[0] == float("inf") and s[2] == float("inf")
    assert int(np.argmin(s)) == 1


def test_all_unmeasurable_keeps_the_first_take():
    s = score_takes([None, None], CFG)
    assert s == [float("inf"), float("inf")]
    assert int(np.argmin(s)) == 0


def test_intonation_outlier_is_dropped_when_rates_tie():
    feats = [feat(8.0, f0sd=0.5), feat(8.0, f0sd=3.0), feat(8.0, f0sd=3.1)]
    assert int(np.argmin(score_takes(feats, CFG))) in (1, 2)


def test_weights_can_disable_a_criterion():
    # 0番は話速こそ中央値ちょうどだが、音量だけ20dB外れている
    feats = [feat(8.0, level=-40.0), feat(8.6, level=-20.0), feat(7.4, level=-20.2)]
    assert int(np.argmin(score_takes(feats, CFG))) == 2        # 音量の外れを避ける
    cfg = dict(CFG, w_level=0.0)
    assert int(np.argmin(score_takes(feats, cfg))) == 0        # 音量を見なければ話速で選ぶ


def test_the_reference_mora_estimate_is_shared_with_the_rate_normalizer():
    """テイク選択の話速も speed_factors と同じモーラ推定を使う（基準がずれない）。"""
    assert estimate_mora("ありがとうございます")[0] == pytest.approx(10.0)


# --- 窓ごとのテイク選択 -----------------------------------------------------

class _FakeSynth:
    """synth_dialogue が呼ばれるたびに、決められた話速の窓を返す偽の合成器。"""

    def __init__(self, rates_per_call, mora=20.0, sr=SR):
        self.rates_per_call = list(rates_per_call)   # [[窓内ターンの話速…], …]
        self.mora = mora
        self.sample_rate = sr
        self.calls = 0

    def synth_dialogue(self, tagged, references, context=None, return_codes=False):
        rates = self.rates_per_call[self.calls]
        self.calls += 1
        wavs = [voiced(self.mora / r) for r in rates]
        return (wavs, [None] * len(wavs)) if return_codes else wavs


def _metas(n, mora=20.0):
    return [{"mora": mora, "conf": 1.0, "gender": "male", "bc": False}
            for _ in range(n)]


def test_best_window_take_picks_the_median_window():
    from generate_tts import best_window_take
    # 3テイク: 速い窓 / 中庸な窓 / 遅い窓
    synth = _FakeSynth([[12.0, 12.0], [8.0, 8.0], [5.0, 5.0]])
    cfg = dict(CFG, takes=3)
    wavs, _codes = best_window_take(synth, "t", [], None, SR, _metas(2), cfg)
    assert synth.calls == 3
    assert take_features(wavs[0], SR, 20.0, "male")["rate"] == pytest.approx(8.0, rel=0.1)


def test_best_window_take_scores_the_whole_window_not_one_turn():
    """1ターンだけ良い窓より、窓全体が中庸な窓を採る。"""
    from generate_tts import best_window_take
    synth = _FakeSynth([[8.0, 14.0, 14.0],     # 先頭だけ中庸、あとは外れ
                        [9.0, 9.0, 9.0],       # 全体が中庸
                        [5.0, 9.0, 9.0]])
    cfg = dict(CFG, takes=3)
    wavs, _ = best_window_take(synth, "t", [], None, SR, _metas(3), cfg)
    rates = [take_features(w, SR, 20.0, "male")["rate"] for w in wavs]
    assert all(r == pytest.approx(9.0, rel=0.1) for r in rates)


def test_best_window_take_ignores_short_and_backchannel_turns():
    from generate_tts import best_window_take
    synth = _FakeSynth([[8.0, 20.0], [12.0, 6.0]])
    metas = [{"mora": 20.0, "conf": 1.0, "gender": "male", "bc": False},
             {"mora": 3.0, "conf": 1.0, "gender": "male", "bc": True}]
    cfg = dict(CFG, takes=2)
    wavs, _ = best_window_take(synth, "t", [], None, SR, metas, cfg)
    # 2本しか無いので中央値は中点。相槌を無視すれば1本目の 8.0 と 12.0 の差は同じだが、
    # 相槌を数えていたら 20.0 の外れで1本目が必ず負ける。ここでは同点＝先頭が残る。
    assert take_features(wavs[0], SR, 20.0, "male")["rate"] == pytest.approx(8.0, rel=0.1)


def test_best_window_take_falls_back_when_segment_counts_disagree():
    from generate_tts import best_window_take
    synth = _FakeSynth([[8.0, 8.0], [9.0]])     # 2本目はセグメント数が合わない
    cfg = dict(CFG, takes=2)
    wavs, _ = best_window_take(synth, "t", [], None, SR, _metas(2), cfg)
    assert len(wavs) == 2


# --- テイクごとに temperature を振る ----------------------------------------

def test_temperature_schedule_leaves_the_first_take_at_the_default():
    from generate_tts import take_temperatures
    # 1本目は必ず None（＝config の gen.temperature のまま）。per_turn 経路では
    # 1本目が「合成済みのもの」なので、ここを振ると引き直しが1本増えてしまう。
    assert take_temperatures({"temperatures": [0.3, 0.9]}, 5) == [None, 0.3, 0.9, 0.3, 0.9]


def test_temperature_schedule_of_nine_covers_each_value_once_for_ten_takes():
    from generate_tts import take_temperatures
    sweep = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    got = take_temperatures({"temperatures": sweep}, 10)
    assert got[0] is None and sorted(got[1:]) == sweep


def test_without_a_schedule_the_old_single_temperature_still_applies():
    from generate_tts import take_temperatures
    assert take_temperatures({"temperature": 0.9}, 3) == [None, 0.9, 0.9]
    assert take_temperatures({}, 3) == [None, None, None]


class _TempSynth(_FakeSynth):
    """呼ばれたときの gen_kwargs['temperature'] を記録する偽の合成器。"""

    def __init__(self, rates_per_call, **kw):
        super().__init__(rates_per_call, **kw)
        self.gen_kwargs = {"temperature": 0.8, "top_p": 0.8}
        self.seen = []

    def synth_dialogue(self, tagged, references, context=None, return_codes=False):
        self.seen.append(self.gen_kwargs["temperature"])
        return super().synth_dialogue(tagged, references, context, return_codes)


def test_each_window_take_is_generated_at_its_own_temperature():
    from generate_tts import best_window_take
    synth = _TempSynth([[12.0, 12.0], [8.0, 8.0], [5.0, 5.0]])
    cfg = dict(CFG, takes=3, temperatures=[0.3, 1.0])
    best_window_take(synth, "t", [], None, SR, _metas(2), cfg)
    assert synth.seen == [0.8, 0.3, 1.0]          # 1本目は既定のまま
    assert synth.gen_kwargs == {"temperature": 0.8, "top_p": 0.8}   # 後始末される


def test_the_temperature_is_restored_even_if_generation_raises():
    from generate_tts import temperature_override
    synth = _TempSynth([[8.0]])
    with pytest.raises(RuntimeError):
        with temperature_override(synth, 0.2):
            assert synth.gen_kwargs["temperature"] == 0.2
            raise RuntimeError("合成が落ちた")
    assert synth.gen_kwargs["temperature"] == 0.8


def test_the_chosen_takes_temperature_is_recorded():
    """どの温度が採用されたかを数える（温度を振る意味があったかの直接の証拠）。"""
    from generate_tts import best_window_take
    synth = _TempSynth([[12.0, 12.0], [8.0, 8.0], [5.0, 5.0]])
    synth.take_temp_hist = {}
    cfg = dict(CFG, takes=3, temperatures=[0.3, 1.0])
    best_window_take(synth, "t", [], None, SR, _metas(2), cfg)
    assert synth.take_temp_hist == {0.3: 1}       # 中庸だった2本目＝t=0.3
