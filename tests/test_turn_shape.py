"""ターンの形（短い発話・問いかけ・受け止め語）と、対話の先頭の無音のテスト。

2026-10-28 追加。人間コーパス maai-f2f-jp-2025 の**先頭33ターン**と 010 の差から入れた
対処（generate_dialogues.S2Params の short_turn_* / question_ratio_cap / ack_* /
react_vocab_human、generate_tts.lead_in_sec、refine_maai の挿入相槌の長さの歯止め）。
いずれも config に新しいキーが無ければプロンプト・音声とも従来どおりであることを確かめる。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_dialogues import (S2Params, _question_norm, _short_turn_norm,   # noqa: E402
                                build_dialogue_prompt, rebalance_ack_heads,
                                turn_shape_ratios)
from generate_tts import lead_in_sec                                          # noqa: E402
from refine_maai import (QualityParams, _bc_dur_limit, drop_turns,            # noqa: E402
                         overlong_bc_indices)

META = {"id": "x", "scenario": "イノウエとオカダが初めて会う",
        "speakers": [{"name": "イノウエ", "gender": "male", "role": "大学の研究者",
                      "personality": "穏やか"},
                     {"name": "オカダ", "gender": "female", "role": "スーパーの店員",
                      "personality": "控えめ", "age_range": "40代"}],
        "target_turns": 32, "main_speaker_index": 0, "speech_style": "敬語"}


def _turns(*utts):
    return [{"speaker": "イノウエ" if i % 2 == 0 else "オカダ", "utterance": u}
            for i, u in enumerate(utts)]


# --- 受け止めの定型語 -------------------------------------------------------

def test_ack_cap_drops_heads_and_keeps_content():
    ts = _turns("なるほど。私も自転車をよく使っています。",
                "いいですね。朝に走ると気持ちいいんですよね。",
                "そうですね。続けるのが大事だと思います。",
                "たしかに。無理のない範囲がいちばんです。",
                "なるほど。それは楽しそうです。")
    n = rebalance_ack_heads(ts, cap=3, phrase_cap=2, seed="t")
    heads = [t["utterance"] for t in ts]
    assert n >= 2
    # 上限を超えた分は冒頭の語だけが外れ、中身は残る
    assert "私も自転車をよく使っています。" in heads[0] or heads[0].startswith("なるほど")
    assert any(h == "それは楽しそうです。" for h in heads)
    # どの発話も中身が消えていない
    assert all(len(h) >= 8 for h in heads)


def test_ack_phrase_cap_swaps_repeats():
    ts = _turns("なるほど。1つめの話です。", "なるほど。2つめの話です。",
                "なるほど。3つめの話です。", "なるほど。4つめの話です。")
    rebalance_ack_heads(ts, cap=99, phrase_cap=2, seed="t")
    kept = [t["utterance"] for t in ts if t["utterance"].startswith("なるほど")]
    assert len(kept) == 2
    # 置き換え先は人間に多い感動詞で、中身が続くときは読点でつなぐ
    swapped = [t["utterance"] for t in ts if not t["utterance"].startswith("なるほど")]
    assert all(s.split("、")[0] in ("あー", "ああ", "すごい", "えっ") for s in swapped)
    assert all("つめの話です。" in s for s in swapped)


def test_ack_cap_keeps_standalone_turn():
    """その語だけのターンは外すと空になるので残す。"""
    ts = _turns("なるほど。", "そうですね。", "いいですね。")
    rebalance_ack_heads(ts, cap=1, phrase_cap=2, seed="t")
    assert all(t["utterance"] for t in ts)


def test_ack_cap_off_by_default():
    ts = _turns("なるほど。私も使っています。", "なるほど。そうなんですね。")
    before = [t["utterance"] for t in ts]
    assert rebalance_ack_heads(ts, cap=0, phrase_cap=0) == 0
    assert [t["utterance"] for t in ts] == before


# --- 指標 -------------------------------------------------------------------

def test_turn_shape_ratios():
    ts = _turns("はい。", "そうなんですか？", "あのですね、少し長めに話すとこうなります。")
    short, q = turn_shape_ratios(ts)
    assert short == pytest.approx(2 / 3)
    assert q == pytest.approx(1 / 3)


# --- プロンプト -------------------------------------------------------------

def test_norms_absent_without_keys():
    sp = S2Params()
    assert _short_turn_norm(30, 34, sp) == ""
    assert _question_norm(30, 34, sp) == ""
    assert _short_turn_norm(30, 34, None) == ""


def test_prompt_unchanged_without_new_keys():
    """新しいキーを設定しなければ、ユーザプロンプトは1バイトも変わらない。"""
    base = build_dialogue_prompt(META, sp=S2Params())
    assert "7〜15文字の短い発話" not in base
    assert "問いかけ（「？」で終わる発話）" not in base
    assert "いちばん多く使うのは" not in base


def test_prompt_has_norms_with_keys():
    sp = S2Params(short_turn_ratio=0.35, long_turn_ratio=0.10,
                  question_ratio_cap=0.12, react_vocab_human=True)
    p = build_dialogue_prompt(META, sp=sp)
    assert "7〜15文字の短い発話を12ターン以上" in p
    assert "31〜55文字の長めの発話を3ターン" in p
    assert "問いかけ（「？」で終わる発話）はこの対話全体で4ターンまで" in p
    # 決まり文句の回数は語を並べて明示する（「あー」も一族に入れる）
    cap = p.split("次の決まり文句")[1][:300]
    for w in ("なるほど", "いいですね", "私も"):
        assert w in cap
    # 「あー」は機械側（rebalance_ack_heads）で散らす。プロンプトで禁じると冒頭
    # リアクションごと避けて react_floor を割る（1回目の作り直しで実測）。
    assert "「あー」" not in cap


def test_ack_swap_spreads_across_words():
    """置き換え先を1語に寄せない（1回目の作り直しで「あー」が人間の3倍出た）。"""
    ts = _turns(*[f"あー、{i}番目の話です。" for i in range(6)])
    rebalance_ack_heads(ts, cap=99, phrase_cap=2, seed="s")
    heads = [t["utterance"].split("、")[0] for t in ts]
    assert heads.count("あー") == 2
    assert len(set(heads)) >= 3


# --- 対話の先頭の無音 -------------------------------------------------------

def test_lead_in_off_by_default():
    assert lead_in_sec("abc", None) == 0.0
    assert lead_in_sec("abc", {"enabled": False, "sec": [1, 2]}) == 0.0


def test_lead_in_reproducible_and_in_range():
    cfg = {"sec": [0.4, 6.0]}
    a = lead_in_sec("dialog-1", cfg)
    assert a == lead_in_sec("dialog-1", cfg)          # id から決まるので再現する
    vals = [lead_in_sec(f"id{i}", cfg) for i in range(500)]
    assert all(0.4 <= v <= 6.0 for v in vals)
    assert all(v > 0 for v in vals)                   # 「0秒ちょうど」は作らない
    assert len(set(vals)) > 400                       # ばらつく


# --- 挿入相槌の長さの歯止め -------------------------------------------------

def test_bc_dur_limit_separates_normal_from_broken():
    q = QualityParams()
    # 実測: 正常な相槌は字数によらず 1.07秒以下、壊れたものは 1.6秒以上
    assert _bc_dur_limit("はい。", q) == pytest.approx(1.3)
    assert _bc_dur_limit("うんうん。", q) == pytest.approx(1.8)
    assert _bc_dur_limit("はい。", QualityParams(bc_dur_base=0.0)) == 0.0


def test_overlong_bc_indices_and_drop():
    q = QualityParams()
    turns = [{"type": "speech", "text": "長い発話です。"},
             {"type": "backchannel", "text": "はい。"},
             {"type": "backchannel", "text": "ええ。"}]
    wavs = {0: np.zeros(16000), 1: np.zeros(16000 * 5), 2: np.zeros(8000)}
    assert overlong_bc_indices(turns, wavs, 16000, q) == [1]
    failed = {2}
    drop_turns(turns, wavs, failed, {1})
    assert [t["text"] for t in turns] == ["長い発話です。", "ええ。"]
    assert sorted(wavs) == [0, 1]
    assert len(wavs[1]) == 8000          # 番号が詰められても波形は対応したまま
    assert failed == {1}


def test_compose_shifts_timeline_by_lead_in():
    """lead_in を渡すと、全ターンが同じ量だけ後ろへずれる（配置の相対関係は不変）。"""
    import generate_tts
    sr = 16000
    record = {
        "id": "dialogue-1",
        "metadata": {"relationship": "同僚", "speech_style": "敬語",
                     "speakers": [{"name": "A", "gender": "female", "age_range": "30代"},
                                  {"name": "B", "gender": "male", "age_range": "30代"}]},
        "voice_dialogue": {"turns": [
            {"speaker": "A", "text": "こんにちは。", "type": "speech",
             "start_offset": 0.0, "duration": 1.0},
            {"speaker": "B", "text": "はい。", "type": "speech",
             "start_offset": 0.2, "duration": 0.5}]}}
    voices = [{"id": "voice-a", "gender": "female", "age_range": "30代"},
              {"id": "voice-b", "gender": "male", "age_range": "30代"}]
    wavs = {i: np.zeros(int(sr * 0.5), dtype=np.float32) for i in range(2)}

    _s0, m0 = generate_tts.compose_dialogue(record, voices, sr, dict(wavs))
    lead = lead_in_sec("dialogue-1", {"sec": [0.4, 6.0]})
    s1, m1 = generate_tts.compose_dialogue(record, voices, sr, dict(wavs),
                                           lead_in={"sec": [0.4, 6.0]})
    assert m0[0]["start_sec"] == 0.0
    assert m1[0]["start_sec"] == pytest.approx(lead, abs=0.002)
    # 相対関係（ターン間の間）は変わらない
    assert (m1[1]["start_sec"] - m1[0]["start_sec"]) == pytest.approx(
        m0[1]["start_sec"] - m0[0]["start_sec"], abs=0.002)
    assert len(s1) == pytest.approx(int(round(lead * sr)) + len(_s0), abs=2)


def test_overlong_bc_off_when_disabled():
    q = QualityParams(bc_dur_base=0.0)
    turns = [{"type": "backchannel", "text": "はい。"}]
    assert overlong_bc_indices(turns, {0: np.zeros(16000 * 9)}, 16000, q) == []
