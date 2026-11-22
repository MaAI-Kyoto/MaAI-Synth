"""generate_tts._select_quality_takes（best_of.criterion: quality）の検査（2026-11-14）。
ASR・x-vector は偽物に差し替える。テイクがそろったターンから採点する（合成と重ねる）作りでも、
1本ずつ回したときと同じテイクを選ぶこと、採点がテイクをまとめて呼ぶことを確かめる。
    .venv-maai-cu12/bin/python -m pytest tests/test_quality_select.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_tts as G  # noqa: E402
import reading_check as RC  # noqa: E402
import speaker_check as SC  # noqa: E402

SR = 24000


def _wav(seed: int) -> np.ndarray:
    """テイクの「良さ」を振幅に埋める（偽 ASR が振幅から CER を返す）。"""
    t = np.arange(SR) / SR
    amp = 0.05 + (seed * 37 % 101) / 1000.0
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class _Synth:
    def __init__(self, conc):
        self.concurrency = conc

    def synth_one_seeded(self, text, ref, seed):
        return _wav(seed)


def _setup(monkeypatch, calls):
    monkeypatch.setattr(RC, "get_asr", lambda *a, **k: object())
    monkeypatch.setattr(SC, "get_verifier", lambda *a, **k: None)

    def fake_score(asr, wavs, sr, texts, hints=None, **kw):
        calls.append(len(wavs))
        return [{"cer": float(np.abs(w).max())} for w in wavs]
    monkeypatch.setattr(RC, "score", fake_score)
    monkeypatch.setattr(G, "_asr_text_nll_batch", lambda asr, ws, sr, text, **kw: [0.0] * len(ws))


def _run(conc, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    turns = [{"speaker": "A", "text": "こんにちは、元気ですか"},
             {"speaker": "B", "text": "うん", "type": "backchannel"},
             {"speaker": "B", "text": "はい、元気です"},
             {"speaker": "A", "text": "よかった、また明日"}]
    wav_by_turn = {i: _wav(1000 + i) for i in range(len(turns))}
    inp = (turns, None, [{"audio": b"", "text": ""}] * 2, {"A": 0, "B": 1}, [t["text"] for t in turns], None)
    cfg = {"rate_kps": 7.0, "w_kps": 0.0, "w_pause": 0.0, "w_hiss": 0.0, "w_nat": 10.0}
    n = G._select_quality_takes(_Synth(conc), {"id": "x"}, wav_by_turn, SR, cfg, inp, 8)
    return n, wav_by_turn, calls


def test_overlapped_selection_equals_sequential(monkeypatch):
    n1, w1, _ = _run(1, monkeypatch)
    n8, w8, _ = _run(8, monkeypatch)
    assert n1 == n8
    for i in w1:
        assert np.array_equal(w1[i], w8[i])


def test_picks_the_lowest_score_and_skips_backchannels(monkeypatch):
    n, w, calls = _run(8, monkeypatch)
    assert calls == [8, 8, 8]                 # 3 ターン × (元の 1 本 + 追加 7 本) をまとめて採点
    assert np.array_equal(w[1], _wav(1001))   # 相槌は触らない
    for i in (0, 2, 3):
        cands = [_wav(1000 + i)] + [_wav(7001 + 7919 * j + i) for j in range(1, 8)]
        best = min(cands, key=lambda x: float(np.abs(x).max()))
        assert np.array_equal(w[i], best)


def _speech(tail=None, sr=SR):
    """1 秒の「発話」＋0.3 秒の無音＋（tail 秒の短い音）＋0.2 秒の無音。"""
    t = np.arange(sr) / sr
    parts = [0.3 * np.sin(2 * np.pi * 200 * t), np.zeros(int(0.3 * sr))]
    if tail:
        parts.append(0.2 * np.sin(2 * np.pi * 300 * np.arange(int(tail * sr)) / sr))
    parts.append(np.zeros(int(0.2 * sr)))
    return np.concatenate(parts).astype(np.float32)


def test_tail_blob_finds_a_short_sound_after_silence_only():
    assert G.tail_blob(_speech(), SR) is None
    a, b = G.tail_blob(_speech(0.1), SR)
    assert abs(a / SR - 1.3) < 0.02 and abs((b - a) / SR - 0.1) < 0.02
    assert G.tail_blob(_speech(1.0), SR) is None              # 長い音は末尾の余計な音ではない（次の文）
    w = _speech(0.1)
    m = G.mute_span(w, a, b, SR)
    assert len(m) == len(w) and G.tail_blob(m, SR) is None
    assert np.array_equal(m[: int(1.2 * SR)], w[: int(1.2 * SR)])   # 発話は無傷


def test_a_take_with_an_extra_tail_sound_is_avoided_or_muted(monkeypatch):
    monkeypatch.setattr(RC, "get_asr", lambda *a, **k: object())
    monkeypatch.setattr(SC, "get_verifier", lambda *a, **k: None)
    # 偽 ASR: 末尾の音（1.25 秒より後の有音）は台本に無い → 切っても CER は 0 のまま
    monkeypatch.setattr(RC, "score", lambda asr, wavs, sr, texts, hints=None, **kw: [{"cer": 0.0} for _ in wavs])
    monkeypatch.setattr(G, "_asr_text_nll_batch", lambda asr, ws, sr, text, **kw: [0.0] * len(ws))

    class S:
        concurrency = 4

        def __init__(self, clean_seed):
            self.clean = clean_seed

        def synth_one_seeded(self, text, ref, seed):
            return _speech(None if seed == self.clean else 0.2)

    turns = [{"speaker": "A", "text": "こんにちは、元気ですか"}]
    inp = (turns, None, [{"audio": b"", "text": ""}], {"A": 0}, [turns[0]["text"]], None)
    cfg = {"rate_kps": 7.0, "w_kps": 0.0, "w_pause": 0.0, "w_hiss": 0.0, "w_tail": 3.0}
    # 3 本目だけ末尾がきれい → それを採る
    clean = 7001 + 7919 * 3
    wb = {0: _speech(0.2)}
    G._select_quality_takes(S(clean), {"id": "x"}, wb, SR, cfg, inp, 4)
    assert G.tail_blob(wb[0], SR) is None and len(wb[0]) == len(_speech(None))
    # どれにも末尾の音がある → 選んだテイクの末尾の音だけ無音に（長さは同じ）
    wb = {0: _speech(0.2)}
    G._select_quality_takes(S(-1), {"id": "x"}, wb, SR, cfg, inp, 4)
    assert G.tail_blob(wb[0], SR) is None and len(wb[0]) == len(_speech(0.2))
    assert np.array_equal(wb[0][: SR], _speech(0.2)[: SR])
    # w_tail 0（既定）なら何もしない
    wb = {0: _speech(0.2)}
    G._select_quality_takes(S(-1), {"id": "x"}, wb, SR, {**cfg, "w_tail": 0.0}, inp, 4)
    assert G.tail_blob(wb[0], SR) is not None
