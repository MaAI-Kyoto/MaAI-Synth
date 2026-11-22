"""ユーザ側の長い沈黙（dialogue.silence。2026-10-23 ft_sil_v1）のテスト。

Stage2 で LLM が書いた pause_before_sec を検査して残し、Stage3 が写し、Stage4 の
retime がその秒数をそのまま置く（VAP で決めない）。config に silence が無いときは
すべて従来どおり、というのが前提。
"""

from pathlib import Path

import numpy as np

import generate_dialogues as G
from generate_voice_passthrough import build_voice_turns

ROOT = Path(__file__).resolve().parent.parent
SIL = G.SilenceParams()
META = {"speakers": [{"name": "イノウエ"}, {"name": "ミサキ"}], "main_speaker_index": 0}
MAIN, USER = "イノウエ", "ミサキ"


def _t(sp, u="発話です。", pause=None):
    t = {"speaker": sp, "utterance": u}
    if pause is not None:
        t["pause_before_sec"] = pause
    return t


# ── Stage2 ────────────────────────────────────────────────────────────────
def test_load_silence_default_off_and_values():
    assert G.load_silence({"dialogue": {}}) is None
    assert G.load_silence({"dialogue": {"silence": {"enabled": False}}}) is None
    s = G.load_silence({"dialogue": {"silence": {"count": [1, 4], "sec": [1.5, 6],
                                                 "max_prompt": 2}}})
    assert (s.count_min, s.count_max, s.sec_min, s.sec_max, s.max_prompt) == (1, 4, 1.5, 6.0, 2)


def test_silence_roles():
    assert G.silence_roles(META) == (MAIN, USER)
    assert G.silence_roles({**META, "main_speaker_index": 1}) == (USER, MAIN)
    assert G.silence_roles({"speakers": META["speakers"]}) == (None, None)


def test_normalize_pauses_kinds_and_clamp():
    turns = [_t(MAIN), _t(USER, "えっと、その。"), _t(USER, "料理です。", 3.24),   # A
             _t(MAIN, "休日は何を？"), _t(USER, "うーん、寝てます。", 12),          # B（上限へ）
             _t(MAIN), _t(USER, "それが。"), _t(MAIN, "ゆっくりで。", 0.5)]         # C（下限へ）
    kept = G.normalize_pauses(turns, MAIN, USER, SIL)
    assert kept == [(2, "A"), (4, "B"), (7, "C")]
    assert [t.get("pause_before_sec") for t in turns] == [None, None, 3.2, None, 8.0,
                                                          None, None, 2.0]


def test_normalize_pauses_drops_invalid():
    turns = [_t(MAIN, pause=3),                 # 先頭
             _t(USER), _t(MAIN), _t(MAIN, pause=3),   # 主話者 → 主話者
             _t(USER, pause="abc"), _t(MAIN, pause=0),
             _t(USER, "それが。"), _t(MAIN, pause=3),
             _t(USER, "でも。"), _t(MAIN, pause=3)]                    # C は1回まで
    kept = G.normalize_pauses(turns, MAIN, USER, SIL)
    assert kept == [(7, "C")]
    assert sum("pause_before_sec" in t for t in turns) == 1


def test_normalize_pauses_count_max_drops_later():
    turns = [_t(MAIN)]
    for _ in range(5):
        turns += [_t(USER), _t(USER, pause=3), _t(MAIN)]
    kept = G.normalize_pauses(turns, MAIN, USER, SIL)
    assert len(kept) == SIL.count_max
    assert [i for i, _ in kept] == [2, 5, 8]


def test_merge_keeps_pause_boundary_only():
    turns = [_t(MAIN), _t(USER, "えっと。"), _t(USER, "料理です。", 3.0),
             _t(MAIN, "なるほど。"), _t(MAIN, "いいですね。")]
    out = G.merge_consecutive_same_speaker(turns)
    assert [t["speaker"] for t in out] == [MAIN, USER, USER, MAIN]
    assert out[2]["pause_before_sec"] == 3.0
    assert out[3]["utterance"] == "なるほど。いいですね。"


def test_echo_head_ignores_own_continuation():
    turns = [_t(MAIN, "京都に住んでます。"), _t(MAIN, "京都ですか？大変です。")]
    assert G._echo_head(turns, 1) is None
    turns[0]["speaker"] = USER
    assert G._echo_head(turns, 1) is not None


def test_prompt_alternation_line_replaced_only_with_silence():
    alt = "2人の話者は必ず交互に発話し、同じ話者を続けて2ターン置かないでください。"
    meta = {**META, "target_turns": 30}
    off = G.build_dialogue_prompt(meta, None, None, "gemma", None, "")
    on = G.build_dialogue_prompt(meta, None, None, "gemma", None, "", SIL)
    assert alt in off and "pause_before_sec" not in off
    assert alt not in on and "pause_before_sec" in on and "ミサキが長く黙る" in on


def test_proofread_message_marks_pause():
    turns = [_t(MAIN), _t(USER, "えっと。"), _t(USER, "料理です。", 3.0)]
    msg = G._proofread_user_message(turns, META)
    assert "2 ミサキ: （3秒の沈黙のあと）料理です。" in msg
    assert G._PAUSE_NOTE_RE.sub("", "（3.5秒の沈黙のあと）料理です。") == "料理です。"


def test_sil_pack_loads_and_builds():
    pack = G.load_prompt_pack(str(ROOT / "prompts/stage2_first_meeting_sil.yaml"))
    gates = {**G.ALL_GATES, "long_utterance": False}
    s = G.build_system_prompt(None, gates, 1, "gemma", pack)
    assert "# 最重要: ユーザの長い沈黙" in s and "pause_before_sec" in s


# ── Stage3 ────────────────────────────────────────────────────────────────
def test_passthrough_copies_pause_not_continued():
    out = build_voice_turns([_t(MAIN), _t(USER, "えっと。"), _t(USER, "料理です。", 3.5),
                             _t(MAIN, "あ。"), _t(MAIN, "なるほど。")])
    assert out[2]["pause_before_sec"] == 3.5 and out[2]["start_offset"] == 3.5
    assert not out[2].get("continued")
    assert out[4].get("continued") and "pause_before_sec" not in out[4]


# ── Stage4 ────────────────────────────────────────────────────────────────
class _FakeScorer:
    """入力音声から決定的に予測値を作る偽スコアラ（tests/test_tri.py と同じ作り）。"""

    def score(self, ch1, ch2, scan_from_sample=0):
        step = 1600
        n = max(len(ch1), len(ch2))
        out = []
        for end in range(step, n + step, step):
            e1 = float(np.abs(ch1[max(0, end - step):end]).sum())
            e2 = float(np.abs(ch2[max(0, end - step):end]).sum())
            u = [(np.sin(e1 * 12.9898 + e2 * 78.233 + j * 3.1) * 43758.5453) % 1.0
                 for j in range(3)]
            out.append((end, {"p_now": [u[0], u[1]], "p_future": [u[1], u[0]],
                              "p_bc": [u[2]], "p_bc_react": [u[2]], "p_bc_emo": [u[0]]}))
        return out


def _plan(turns, durs):
    import generate_tts as gt
    from retime_maai import MaaiRetimer
    tts = gt.load_config(str(ROOT / "configs/ft_sil_v1.yaml"))["tts"]
    r = MaaiRetimer(gt.RetimeConfig.from_tts_cfg(tts))
    r._vap, r._bc = _FakeScorer(), _FakeScorer()
    sr = 24000
    rng = np.random.default_rng(0)
    wavs = {i: (rng.standard_normal(int(d * sr)) * 0.1).astype(np.float32)
            for i, d in enumerate(durs)}
    rec = {"id": "sil", "voice_dialogue": {"turns": turns}, "metadata": META}
    plans = r.plan(rec, turns, wavs, sr, {MAIN: 0, USER: 1}, set())
    return {p["index"]: p for p in plans}, durs


def _sp(sp, text="これはテストのための少し長めの発話で、途中に読点も入ります。", **kw):
    return {"speaker": sp, "type": "speech", "text": text, "start_offset": 0.3, **kw}


def test_retime_places_fixed_pause_after_frontier():
    turns = [_sp(MAIN), _sp(USER), _sp(USER, pause_before_sec=3.5), _sp(MAIN),
             _sp(USER, pause_before_sec=6.0)]
    plans, durs = _plan(turns, [3.0, 2.5, 4.0, 3.5, 2.0])
    ends = {i: plans[i]["start_sec"] + durs[i] for i in plans}
    for i, sec in ((2, 3.5), (4, 6.0)):
        frontier = max(ends[j] for j in range(i))
        assert plans[i]["retime"]["method"] == "fixed-pause"
        assert abs(plans[i]["start_sec"] - (frontier + sec)) < 2e-3


def test_retime_bc_before_pause_stays_inside_anchor():
    turns = [_sp(MAIN), _sp(USER),
             {"speaker": MAIN, "type": "backchannel", "text": "うん", "start_offset": -0.5,
              "refine": {"rel_pos": 0.95}},
             _sp(USER, pause_before_sec=4.0)]
    plans, durs = _plan(turns, [3.0, 4.0, 0.4, 3.0])
    user_end = plans[1]["start_sec"] + durs[1]
    assert plans[2]["start_sec"] + durs[2] <= user_end + 2e-3
    assert abs(plans[3]["start_sec"] - (user_end + 4.0)) < 2e-3


# ── 2026-10-24 試走1回目を受けた修正 ─────────────────────────────────────
def test_normalize_pauses_reports_drops():
    turns = [_t(MAIN), _t(MAIN, pause=3), _t(USER, pause="x")]
    dropped = []
    G.normalize_pauses(turns, MAIN, USER, SIL, dropped)
    assert dropped == ["1:主話者の沈黙", "2:数値でない('x')"]


def test_pre_pause_indices():
    turns = [_t(MAIN), _t(USER, "あの、実は、その、"), _t(USER, "料理です。", 3)]
    assert G.pre_pause_indices(turns) == {1}


def test_hesitation_interrupts():
    turns = [_t(MAIN, "休日は？"), _t(USER, "えっと。"), _t(MAIN, "何か趣味ですか？"),
             _t(USER, "はい。"), _t(MAIN, "なるほど。"), _t(USER, "それが。"),
             _t(MAIN, "ゆっくりで大丈夫ですよ。", 4.0), _t(USER, "あー。"), _t(MAIN, "ですよね。")]
    assert G.hesitation_interrupts(turns, MAIN, USER) == [2]
    assert G.is_hesitation("うーん、えっと。") and G.is_hesitation("でも、ただ。")
    assert not G.is_hesitation("あー。") and not G.is_hesitation("でも、それは料理です。")


def test_wants_prompt_ratio_and_prompt_text():
    import dataclasses
    on = dataclasses.replace(SIL, prompt_ratio=0.3)
    picks = [G.wants_prompt({"id": f"d{k}"}, on) for k in range(2000)]
    assert 0.25 < sum(picks) / len(picks) < 0.35
    assert not any(G.wants_prompt({"id": f"d{k}"}, SIL) for k in range(50))
    k = picks.index(True)
    meta = {**META, "id": f"d{k}", "target_turns": 30}
    assert "1回は「ミサキが途切れて黙ったあと" in G.build_dialogue_prompt(
        meta, None, None, "gemma", None, "", on)
    meta["id"] = f"d{picks.index(False)}"
    txt = G.build_dialogue_prompt(meta, None, None, "gemma", None, "", on)
    assert "促す型は使わないでください" in txt and "ちょうど3回" in txt


def test_trailing_off_keeps_fillers():
    assert G.clean_enriched_utterance("あの、実は、その、", 1, trailing_off=True) == "あの、実は、その。"
    assert G.clean_enriched_utterance("あの、実は、その、", 1) == "あの、実は。"


# ── 2026-10-24 試走2回目を受けた修正 ─────────────────────────────────────
def test_pause_moved_to_prompt_and_c_needs_trailing():
    turns = [_t(MAIN, "おすすめは？"), _t(USER, "えっと。"), _t(MAIN, "山ですか？"),
             _t(USER, "はい、大文字山です。", 4.0)]
    dropped = []
    kept = G.normalize_pauses(turns, MAIN, USER, G.SilenceParams(prompt_sec_min=3.0), dropped)
    assert kept == [(2, "C")] and turns[2]["pause_before_sec"] == 4.0
    assert "pause_before_sec" not in turns[3] and dropped == ["3:促しへ移動"]
    # 直前のユーザが言いさしでない → 主話者自身の考え込み
    turns = [_t(USER, "コツとかあるかな？"), _t(MAIN, "うーん、どうでしょうね。", 5.2)]
    dropped = []
    assert G.normalize_pauses(turns, MAIN, USER, SIL, dropped) == []
    assert dropped == ["1:主話者の沈黙(促しでない)"]


def test_adjacent_pauses_keep_first_and_c_min():
    turns = [_t(MAIN), _t(USER, "えっと。"), _t(MAIN, "どうされました？", 2.0),
             _t(USER, "あ、いえ。", 5.0)]
    dropped = []
    kept = G.normalize_pauses(turns, MAIN, USER, G.SilenceParams(prompt_sec_min=3.0), dropped)
    assert kept == [(2, "C")] and turns[2]["pause_before_sec"] == 3.0
    assert dropped == ["3:隣の沈黙と連続"]


def test_hesitation_last_sentence():
    assert G.is_hesitation("いいよ。あ、そういえば。")
    assert not G.is_hesitation("実は筋トレを始めたんです。地味ですけど。")
    assert G.is_trailing("いいですね。私は、その。")


def test_fix_missing_utterance_key():
    import json
    t = '{"turns": [{"speaker": "ハヤシ", "えっと。"}, {"speaker": "A", "utterance": "x"}]}'
    d = json.loads(G._fix_missing_utterance_key(t))
    assert d["turns"][0] == {"speaker": "ハヤシ", "utterance": "えっと。"}
    ok = '{"turns": [{"speaker": "A", "utterance": "x", "pause_before_sec": 3}]}'
    assert G._fix_missing_utterance_key(ok) == ok


# ── 2026-10-24 試走3回目を受けた修正 ─────────────────────────────────────
def test_drop_closing_pauses():
    turns = [_t(MAIN) for _ in range(4)] + [
        _t(MAIN, "それでは、失礼いたします。"), _t(USER, "はい。失礼いたします。", 3.0)]
    turns[2] = _t(USER, "えっと。")
    turns[3] = _t(USER, "料理です。", 4.0)
    assert G.drop_closing_pauses(turns) == 1
    assert turns[3]["pause_before_sec"] == 4.0 and "pause_before_sec" not in turns[5]


def test_jitter_pauses_deterministic_and_bounded():
    import dataclasses
    sil = dataclasses.replace(SIL, jitter_sec=0.5, prompt_sec_min=3.0)
    base = [_t(MAIN), _t(USER, "えっと。"), _t(USER, "a。", 3.5), _t(MAIN, "b。"),
            _t(USER, "その。"), _t(MAIN, "c？", 3.0), _t(USER, "d。"), _t(MAIN, "e？"),
            _t(USER, "f。", 7.9)]
    vals = set()
    for k in range(30):
        turns = [dict(t) for t in base]
        G.jitter_pauses(turns, MAIN, USER, sil, f"id{k}")
        again = [dict(t) for t in base]
        G.jitter_pauses(again, MAIN, USER, sil, f"id{k}")
        assert turns == again
        assert 3.0 <= turns[2]["pause_before_sec"] <= 4.0
        assert turns[5]["pause_before_sec"] >= 3.0          # C の下限
        assert turns[8]["pause_before_sec"] <= 8.0
        vals.add(turns[2]["pause_before_sec"])
    assert len(vals) > 5
    turns = [dict(t) for t in base]
    G.jitter_pauses(turns, MAIN, USER, SIL, "x")            # jitter_sec 0 なら変えない
    assert turns == base


def test_trailing_fragment_counts_as_hesitation():
    # 試走4回目: 「でも、家で。」の直後に主話者が即座に引き取った
    for u in ["でも、家で。", "大型犬ですね。でも、家で。", "実は、私も。", "その、夜に。"]:
        assert G.is_hesitation(u), u
    for u in ["でも、勉強が結構大変で。", "え、イノウエさんも。", "それ、いいかも。", "忙しいので。"]:
        assert not G.is_hesitation(u), u
    turns = [_t(MAIN, "犬は家でも元気ですか？"), _t(USER, "でも、家で。"),
             _t(MAIN, "家での様子は、どうなんですか？")]
    assert G.hesitation_interrupts(turns, MAIN, USER) == [2]


def test_self_answers_detects_question_then_own_answer():
    turns = [_t(MAIN, "私はミニマリストです。"),
             _t(USER, "はい。イノウエさんは、どういうバランスが、理想だと思いますか？"),
             _t(USER, "うーん。まあ、家族の時間が大事です。", 6.0),
             _t(MAIN, "なるほど。"),
             _t(USER, "モンステラですか？"),
             _t(USER, "いい選択ですね。", 3.6),
             _t(MAIN, "そうなんですね。"),
             _t(USER, "あ、なるほど。具体的に、どんなことを。"),
             _t(USER, "スクワットとかです。", 3.2),
             _t(MAIN, "いいですね。"),
             _t(USER, "どうしようかな？"),
             _t(USER, "やっぱり行きます。", 3.0)]
    assert G.self_answers(turns, MAIN, USER) == [2, 8]


def test_limit_echoes_no_wh_question_before_own_continuation():
    # 置き換えの「どんな感じなんですか？」が沈黙を挟む自分の続きの前に入らない
    seen = set()
    for k in range(12):
        turns = [_t(MAIN, "カレー屋もいいですよ。"), _t(USER, "カレー屋ですか？"),
                 _t(MAIN, f"京都のルミナスパイス{k}が、品揃えが豊富です。"),
                 _t(USER, f"ルミナスパイス{k}ですか？"),
                 _t(USER, "今度、絶対に行ってみます。", 3.2), _t(MAIN, "ぜひ。")]
        assert G.limit_echoes(turns, cap=1) == 1
        seen.add(turns[3]["utterance"])
        assert "どんな" not in turns[3]["utterance"]
    assert len(seen) == 2          # 置き換えは起きていて、候補は「どんな」以外の2つ


def test_b_requires_main_question():
    turns = [_t(USER, "コツとかあるんですかね。"), _t(MAIN, "うーん。"),
             _t(USER, "やっぱり、根気かなって。", 4.2), _t(MAIN, "ぜひ一緒に行きましょう。"),
             _t(USER, "うーん。ぜひ。", 5.5), _t(MAIN, "おすすめはありますか？"),
             _t(USER, "うーん。槍ヶ岳かな。", 5.0)]
    dropped = []
    kept = G.normalize_pauses(turns, MAIN, USER, SIL, dropped)
    assert kept == [(6, "B")]
    assert dropped == ["2:問いでない主話者のあと", "4:問いでない主話者のあと"]


def test_fix_stray_pause_key():
    import json
    raw = ('{"turns": [{"speaker": "ハヤシ", "utterance": "えっと、あの。"}, "pause_before_sec": 4.2},\n'
           '    {"speaker": "ハヤシ", "utterance": "今度、また。"}]}')
    d = json.loads(G._fix_missing_utterance_key(raw))
    assert d["turns"][0] == {"speaker": "ハヤシ", "utterance": "えっと、あの。", "pause_before_sec": 4.2}
    assert len(d["turns"]) == 2


def test_fix_echo_fragments_keeps_mo():
    # 2026-10-24 聴取: 「何枚も。」が「何枚ですか。」になっていた
    turns = [_t(USER, "やっぱり、カイロを何枚も貼るのが、一番手っ取り早いです。"),
             _t(MAIN, "何枚も。かなり、気合が入ってます。")]
    G.fix_echo_fragments(turns)
    assert turns[1]["utterance"] == "何枚もですか。かなり、気合が入ってます。"
    turns = [_t(USER, "京都のスタジアムに行きました。"), _t(MAIN, "スタジアムに。いいですね。")]
    G.fix_echo_fragments(turns)
    assert turns[1]["utterance"] == "スタジアムですか。いいですね。"


def test_reading_kokoromotonai():
    import tts_reading
    assert tts_reading.apply_readings("外だとちょっと心許ないですね。") == "外だとちょっとこころもとないですね。"
    assert tts_reading.apply_readings("心許せる友人。") == "心許せる友人。"
