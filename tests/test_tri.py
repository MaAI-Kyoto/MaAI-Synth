"""三人対話（main_tri / generate_*_tri / retime_tri / turn_order）のテスト。

三人用は二人用のモジュールを import して関数を差し替える作りなので、
  - 差し替えたテストは必ず元に戻す（同じ pytest プロセスで二人用のテストも動くため）
  - 置き換え元の文言・関数が二人用に残っていることを確かめる（二人用が変わったら落ちる）
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dialogues as gd                    # noqa: E402
import generate_dialogues_tri as gdt               # noqa: E402
import generate_metadata as gm                     # noqa: E402
import generate_metadata_tri as gmt                # noqa: E402
import retime_tri                                  # noqa: E402
from main import load_config                       # noqa: E402
from turn_order import (TurnOrderLimits, merged_speakers, penalty,  # noqa: E402
                        turn_order_stats, violations)

CFG = "configs/tri_ft_v1.yaml"


def _turns(seq: str, addressee: bool = True) -> list[dict]:
    """seq の各文字が話者。宛先は直前の話者（1ターン目は both）。"""
    names = {"A": "イノウエ", "B": "ミサキ", "C": "ゴロウ"}
    out = []
    for i, c in enumerate(seq):
        t = {"speaker": names[c], "utterance": "はい。"}
        if addressee:
            t["addressee"] = names[seq[i - 1]] if i and seq[i - 1] != c else "both"
        out.append(t)
    return out


SPEAKERS = ["イノウエ", "ミサキ", "ゴロウ"]


# ── turn_order ────────────────────────────────────────────────────────────
def test_rotation_is_detected():
    st = turn_order_stats(_turns("ABC" * 10), SPEAKERS)
    assert st.aba_ratio == 0.0
    assert st.max_cycle_run == 28
    assert violations(st, TurnOrderLimits())


def test_human_like_order_passes():
    # 二人の往復に三人目が入り、中心の組が入れ替わる
    seq = "ABABACACBCBCABABCACA"
    st = turn_order_stats(_turns(seq), SPEAKERS)
    assert st.aba_ratio > 0.5
    assert st.max_cycle_run <= 3
    assert not violations(st, TurnOrderLimits())
    assert penalty(st, TurnOrderLimits()) == 0.0


def test_silent_speaker_counts_zero_share():
    st = turn_order_stats(_turns("ABABABAB"), SPEAKERS)
    assert st.min_share == 0.0
    assert any("最少話者" in v for v in violations(st, TurnOrderLimits()))


def test_same_speaker_runs_are_merged():
    assert merged_speakers(_turns("AABBBCA")) == ["イノウエ", "ミサキ", "ゴロウ", "イノウエ"]


def test_limits_reject_unknown_keys():
    TurnOrderLimits.from_config({"min_aba": 0.3, "retry": 2})
    with pytest.raises(ValueError):
        TurnOrderLimits.from_config({"min_abba": 0.3})


# ── Stage1 ────────────────────────────────────────────────────────────────
def _meta3(names=("サトウ", "ミサキ", "ゴロウ"), roles=("会社員", "大学の研究者", "保育士"),
           genders=("male", "male", "female")):
    return {"scenario": f"{names[0]}と{names[1]}と{names[2]}が話す",
            "conversation_flow": [f"{names[1]}が挨拶する"],
            "speakers": [{"name": n, "gender": g, "role": r, "age_range": "30代"}
                         for n, g, r in zip(names, genders, roles)]}


SPEC = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "大学の研究者"}


def test_enforce_main_speaker_moves_main_first_and_keeps_order():
    out, n_swap, _, n_name, n_drop = gmt.enforce_main_speaker([_meta3()], SPEC)
    assert (n_swap, n_name, n_drop) == (1, 1, 0)
    m = out[0]
    assert [s["name"] for s in m["speakers"]] == ["イノウエ", "サトウ", "ゴロウ"]
    assert m["main_speaker_index"] == 0
    assert "イノウエが挨拶する" in m["conversation_flow"][0]   # 本文の旧名も置換


def test_enforce_main_speaker_drops_duplicate_role_and_name_clash():
    dup = _meta3(roles=("大学の研究者", "大学の研究者", "保育士"))
    clash = _meta3(names=("イノウエダ", "ミサキ", "ゴロウ"))
    others = _meta3(names=("サトウ", "ミサキ", "ミサキコ"))
    two = {"speakers": _meta3()["speakers"][:2]}
    out, *_, n_drop = gmt.enforce_main_speaker([dup, clash, others, two], SPEC)
    assert out == [] and n_drop == 4


def test_tracker_excludes_compositions_without_main_gender():
    tr = gmt.DiversityTracker(None, main_speaker={"gender": "male", "age_range": "30代"},
                              trio_mix={"女性3人": 0.5, "男性3人": 0.5})
    assert set(tr.trio_mix) == {"男性3人"}
    with pytest.raises(ValueError):
        gmt.DiversityTracker(None, trio_mix={"男女": 1.0})


def test_tracker_starters_are_balanced_and_trio_counted():
    tr = gmt.DiversityTracker(None)
    starts = [tr.next_starter() for _ in range(30)]
    assert sorted(set(starts)) == [1, 2, 3]
    assert max(starts.count(i) for i in (1, 2, 3)) == 10
    q = tr.trio_quota(1)
    label = next(iter(q))
    assert tr.trio_pending[label] == 1
    genders = {"男性3人": ["male"] * 3, "男性2人と女性1人": ["male", "male", "female"],
               "男性1人と女性2人": ["male", "female", "female"], "女性3人": ["female"] * 3}[label]
    tr.add({"speakers": [{"name": f"N{i}", "gender": g} for i, g in enumerate(genders)]})
    assert tr.trio_counts[label] == 1 and tr.trio_pending[label] == 0


def test_stage1_prompt_is_three_party():
    cfg = load_config(str(ROOT / CFG))
    md = cfg["metadata"]
    tr = gmt.DiversityTracker(gm.load_scenario(md["scenario_file"]), main_speaker=md["main_speaker"],
                              partner_roles=md["partner_roles"], style_mix=md["style_mix"],
                              trio_mix=md["trio_mix"], group_shapes=cfg["tri"]["group_shapes"])
    p = gmt.build_user_prompt(1, 30, 44, tr, 3)
    assert "必ず3人" in p and "2人目と3人目" in p and "3人目の role" in p
    assert "三人の関係は「" in p and "中心になって話す人" in p
    assert "【1人目の人物設定】" in p                       # ペルソナは二人用と同じ段落
    with pytest.raises(ValueError):
        gmt.build_user_prompt(1, 30, 44, tr, 2)


def test_stage1_patches_are_reversible():
    orig = {n: getattr(gm, n) for n in gmt._PATCH_NAMES}
    gmt.apply_patches()
    try:
        assert gm.enforce_main_speaker is gmt.enforce_main_speaker
        assert gm.build_user_prompt is gmt.build_user_prompt
    finally:
        gmt.remove_patches()
    assert {n: getattr(gm, n) for n in gmt._PATCH_NAMES} == orig


def test_tri_scenario_loads():
    sc = gm.load_scenario("scenarios/tri/first_meeting.yaml")
    assert sc.system_prompt.count('"name": "話者名"') == 3
    assert len(sc.genres) > 100 and sc.relationships


# ── Stage2 ────────────────────────────────────────────────────────────────
@pytest.fixture
def stage2_patched():
    gdt.apply_patches()
    yield
    gdt.remove_patches()


def _meta_s2(style="敬語"):
    m = _meta3(names=("イノウエ", "ミサキ", "ゴロウ"))
    m.update({"speech_style": style, "target_turns": 34, "main_speaker_index": 0})
    return m


def test_two_party_texts_still_exist_in_stage2():
    """置き換え元の文言が二人用に残っていること（二人用が変わったら三人用を直す合図）。"""
    sp = gd.S2Params.from_config(load_config(str(ROOT / CFG)))
    p = gd.build_dialogue_prompt(_meta_s2("混合"), sp)
    assert gdt.TWO_PARTY_ALTERNATION in p
    assert gdt._STYLE_MIXED_TWO in p
    fq, fcap = gdt._filler_bounds(34, 39, sp)
    assert gdt._SPLIT_TWO.format(half=gd._half_range(fq, fcap)) in gd._enrich_norm(34, 39, sp)


def test_stage2_prompt_is_three_party(stage2_patched):
    cfg = load_config(str(ROOT / CFG))
    sp = gd.S2Params.from_config(cfg)
    assert sp.filler_split
    pack = gd.load_prompt_pack(cfg["dialogue"]["prompt_file"])
    p = gd.build_dialogue_prompt(_meta_s2("混合"), sp, gd.ALL_GATES, "gemma", pack)
    assert gdt.TWO_PARTY_ALTERNATION not in p and gdt.TRI_ORDER_NORM in p
    assert "3人の話者それぞれ" in p and "2人の話者それぞれ" not in p
    assert gdt._STYLE_MIXED_THREE in p and gdt._STYLE_MIXED_TWO not in p


def test_stage2_patches_are_reversible():
    orig = {n: getattr(gd, n) for n in gdt._PATCH_NAMES}
    gdt.apply_patches()
    gdt.apply_patches()           # 二重に呼んでも元を指し続ける
    gdt.remove_patches()
    assert {n: getattr(gd, n) for n in gdt._PATCH_NAMES} == orig


def test_tri_pack_loads_with_both_profiles():
    pack = gd.load_prompt_pack("prompts/tri/stage2_first_meeting.yaml")
    assert pack.schema_enriched.count('"name": "話者名"') == 3
    for prof in ("qwen", "gemma"):
        s = gd.build_system_prompt(gd.S2Params(), gd.ALL_GATES, 1, prof, pack)
        assert "三人の会話の順番" in s
        assert "交互" not in s and "2人の話者" not in s
    assert "三人" in pack.proofread_prompt


def _fake_result(seq):
    return {"id": "x", "metadata": _meta_s2(), "dialogue": {"turns": _turns(seq)}}


def _run_gate(results, retry=2):
    calls = []

    async def fake(client, model, metadata, idx, *a, **k):
        calls.append(idx)
        return results[len(calls) - 1]

    gdt._ORIG["generate_single_dialogue"] = fake
    gdt._GATE.update(limits=TurnOrderLimits(), retry=retry)
    try:
        out = asyncio.run(gdt.generate_single_dialogue(None, "m", _meta_s2(), "x"))
    finally:
        gdt._ORIG.clear()
    return out, len(calls)


def test_gate_retries_rotation_then_accepts():
    good = "ABABACACBCBCABABCACA"
    out, n = _run_gate([_fake_result("ABC" * 8), _fake_result(good)])
    assert n == 2 and out["dialogue"]["turn_order"]["attempt"] == 2


def test_gate_keeps_best_when_all_fail():
    worse, better = "ABC" * 8, "ABC" * 4 + "ABAB"      # better も閾値は外れる（連続10）
    out, n = _run_gate([_fake_result(worse), _fake_result(better), _fake_result(worse)])
    assert n == 3
    assert [t["speaker"] for t in out["dialogue"]["turns"]] == [t["speaker"] for t in _turns(better)]


def test_gate_skips_when_one_speaker_is_silent():
    out, n = _run_gate([_fake_result("ABABABABABAB")] * 3)
    assert n == 3 and out.get("__skip__")


def test_gate_passes_through_skip_and_none():
    out, n = _run_gate([{"__skip__": True, "id": "x", "reason": "r"}])
    assert n == 1 and out["__skip__"]
    out, n = _run_gate([None])
    assert n == 1 and out is None


# ── Stage4 ────────────────────────────────────────────────────────────────
def _record(seq="ABABCACBBA", rid="d1"):
    turns = [{"speaker": t["speaker"], "text": "はい"} for t in _turns(seq)]
    return {"id": rid, "voice_dialogue": {"turns": turns},
            "metadata": {"speakers": [{"name": n} for n in SPEAKERS], "main_speaker_index": 0}}


def _plan(rec, durs, params=None):
    sr = 100
    turns = rec["voice_dialogue"]["turns"]
    wavs = {i: np.zeros(int(d * sr)) for i, d in enumerate(durs)}
    ch = {n: i for i, n in enumerate(SPEAKERS)}
    r = retime_tri.TriGapRetimer(None, params=params or retime_tri.TriTimingParams())
    return r.plan(rec, turns, wavs, sr, ch), ch, turns


def test_tri_retimer_no_same_channel_overlap_and_bounded_overlap():
    rec = _record("ABABCACBBACBCAB")
    durs = [0.4, 2.0, 0.3, 3.0, 1.0, 0.5, 2.5, 0.8, 1.2, 0.6, 1.5, 0.4, 2.2, 0.9, 1.1]
    plans, ch, turns = _plan(rec, durs)
    p = retime_tri.TriTimingParams()
    last_end = {}
    floor = 0.0
    for pl, t, d in zip(plans, turns, durs):
        c = ch[t["speaker"]]
        s = pl["start_sec"]
        assert s >= last_end.get(c, -1) + p.min_same_channel_gap - 1e-3
        if pl["retime"]["method"] == "tri_fto":
            assert s >= floor - p.max_overlap - 1e-3
        last_end[c] = s + d
        floor = max(floor, s + d)


def test_tri_retimer_is_deterministic_per_dialogue():
    durs = [1.0] * 10
    a, *_ = _plan(_record(rid="d1"), durs)
    b, *_ = _plan(_record(rid="d1"), durs)
    c, *_ = _plan(_record(rid="d2"), durs)
    assert a == b
    assert [x["start_sec"] for x in a] != [x["start_sec"] for x in c]


def test_tri_timing_rejects_unknown_keys():
    with pytest.raises(ValueError):
        retime_tri.TriTimingParams.from_config({"max_overlapp": 0.1})


def test_record_speaker_order_puts_main_first():
    import generate_tts_tri as gtt
    rec = _record("BCABAC")
    assert gtt.record_speaker_order(rec) == ["イノウエ", "ミサキ", "ゴロウ"]
    rec["metadata"]["main_speaker_index"] = 2
    assert gtt.record_speaker_order(rec) == ["ゴロウ", "ミサキ", "イノウエ"]
    del rec["metadata"]["main_speaker_index"]
    assert gtt.record_speaker_order(rec) == ["ミサキ", "ゴロウ", "イノウエ"]


def test_tts_patches_are_reversible_and_config_uses_vap():
    import generate_tts as gt
    import generate_tts_tri as gtt
    import refine_maai as rm
    import retime_tri_vap as rtv
    cfg = gt.load_config(str(ROOT / CFG))
    gtt.check_config(cfg, CFG)
    rc = gt.RetimeConfig.from_tts_cfg(cfg["tts"])       # 未知キー検査を通る
    assert rc.backend in ("maai", "vapbatch", "maaibatch") and gtt.retimer_kind(cfg) == "vap"
    rm.RefineConfig.from_tts_cfg(cfg["tts"])            # refine の未知キー検査も通る
    orig = {(m, n): getattr(m, n) for m, n in gtt._TARGETS}
    assert rm.SUPPORTED_SPEAKER_COUNTS == (2,)           # 二人用の既定は2人だけ
    gtt.apply_patches(cfg)
    try:
        assert gt.MaaiRetimer is rtv.TriVapRetimer and rm.MaaiRetimer is rtv.TriVapRetimer
        assert rm.InsertionScanner is rtv.TriInsertionScanner
        assert rm.SUPPORTED_SPEAKER_COUNTS == (2, 3)
    finally:
        gtt.remove_patches()
    assert {(m, n): getattr(m, n) for m, n in gtt._TARGETS} == orig


def test_tts_gap_mode_rejects_refine(monkeypatch, tmp_path):
    import generate_tts_tri as gtt
    c = tmp_path / "c.yaml"
    c.write_text(f"extends: {ROOT / CFG}\ntri:\n  retimer: gap\n")
    monkeypatch.setattr(sys, "argv", ["generate_tts_tri.py", "--config", str(c), "--refine"])
    with pytest.raises(SystemExit):
        gtt.run()


# ── Stage4: 二者モデルを話者ごとに組み替える（retime_tri_vap）───────────────
class _FakeScorer:
    """入力音声から決定的に予測値を作る偽スコアラ（入力が1サンプルでも違えば出力が変わる）。"""

    def __init__(self):
        self.calls = []

    def score(self, ch1, ch2, scan_from_sample=0):
        self.calls.append((np.asarray(ch1).copy(), np.asarray(ch2).copy()))
        step = 1600
        n = max(len(ch1), len(ch2))
        out = []
        for k, end in enumerate(range(step, n + step, step)):
            e1 = float(np.abs(ch1[max(0, end - step):end]).sum())
            e2 = float(np.abs(ch2[max(0, end - step):end]).sum())
            u = [(np.sin(e1 * 12.9898 + e2 * 78.233 + j * 3.1) * 43758.5453) % 1.0
                 for j in range(3)]
            out.append((end, {"p_now": [u[0], u[1]], "p_future": [u[1], u[0]],
                              "p_bc": [u[2]], "p_bc_react": [u[2]], "p_bc_emo": [u[0]]}))
        return out


def _stage4_cfgs():
    import generate_tts as gt
    import refine_maai as rm
    tts = gt.load_config(str(ROOT / CFG))["tts"]
    return gt.RetimeConfig.from_tts_cfg(tts), rm.RefineConfig.from_tts_cfg(tts)


def _rec_with_audio(names, seq, bc_after=(), seed=0):
    """seq の各文字が話者。bc_after のターンの後ろに、次の話者の相槌（rel_pos 付き）を入れる。"""
    rng = np.random.default_rng(seed)
    sr = 24000
    turns, wavs = [], {}
    for k, c in enumerate(seq):
        sp = names["ABC".index(c)]
        turns.append({"speaker": sp, "type": "speech",
                      "text": "これはテストのための少し長めの発話で、途中に読点も入ります。",
                      "start_offset": 0.3})
        if k in bc_after and k + 1 < len(seq):
            nxt = names["ABC".index(seq[k + 1])]
            turns.append({"speaker": nxt, "type": "backchannel", "text": "うん",
                          "start_offset": -0.5, "refine": {"rel_pos": 0.5}})
    for i, t in enumerate(turns):
        sec = 0.4 if t["type"] == "backchannel" else float(rng.uniform(3.0, 6.0))
        wavs[i] = (rng.standard_normal(int(sec * sr)) * 0.1).astype(np.float32)
    rec = {"id": f"rec{seed}", "voice_dialogue": {"turns": turns},
           "metadata": {"speakers": [{"name": n} for n in names], "main_speaker_index": 0}}
    return rec, turns, wavs, sr


def _retimer(cls, rcfg):
    r = cls(rcfg)
    r._vap, r._bc = _FakeScorer(), _FakeScorer()
    return r


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tri_vap_plan_equals_two_party_for_two_speakers(seed):
    """2話者では、組み替えても二人用 MaaiRetimer.plan と1件残らず同じ配置になる。"""
    from retime_maai import MaaiRetimer
    from retime_tri_vap import TriVapRetimer
    rcfg, _ = _stage4_cfgs()
    names = ["イノウエ", "ミサキ"]
    rec, turns, wavs, sr = _rec_with_audio(names, "ABABBABAAB", bc_after=(0, 3, 6), seed=seed)
    ch = {"イノウエ": 0, "ミサキ": 1}
    a = _retimer(MaaiRetimer, rcfg).plan(rec, turns, wavs, sr, ch, set())
    b = _retimer(TriVapRetimer, rcfg).plan(rec, turns, wavs, sr, ch, set())
    assert a == b
    assert {p["retime"]["method"] for p in a} - {"first"}   # 判断が実際に走っている


def test_tri_scanner_equals_two_party_for_two_speakers():
    from refine_maai import InsertionScanner, build_timeline
    from retime_maai import MaaiRetimer
    from retime_tri_vap import TriInsertionScanner, tri_build_timeline
    rcfg, fcfg = _stage4_cfgs()
    names = ["イノウエ", "ミサキ"]
    rec, turns, wavs, sr = _rec_with_audio(names, "ABABABAB", seed=3)
    ch = {"イノウエ": 0, "ミサキ": 1}
    r = _retimer(MaaiRetimer, rcfg)
    plans = r.plan(rec, turns, wavs, sr, ch, set())
    tl, _ = build_timeline(turns, plans, wavs, sr, ch)
    a = InsertionScanner(r, fcfg).scan(tl, turns, plans, wavs, sr, ch, set())
    tl3, _ = tri_build_timeline(turns, plans, wavs, sr, ch)
    b = TriInsertionScanner(r, fcfg).scan(tl3, turns, plans, wavs, sr, ch, set())
    assert a == b and a                                   # 候補が実際に出ている


def test_tri_timeline_pairs_focus_against_the_rest():
    from retime_tri_vap import TriScoreTimeline
    tl = TriScoreTimeline(num_ch=3, sr_engine=16000)
    for c, v in enumerate((1.0, 2.0, 4.0)):
        tl.place(c, 0.0, np.full(1600, v, dtype=np.float32))
    out = tl.slice_all(0.0, 0.1)
    for focus, (x0, x1) in {0: (1, 6), 1: (5, 2), 2: (3, 4)}.items():
        a, b = tl.view(focus).slice(0.0, 0.1)
        assert np.allclose(a, x0) and np.allclose(b, x1), focus
    assert out.shape == (1600, 3)


def test_tri_vap_plan_three_speakers():
    from retime_tri_vap import TriVapRetimer
    rcfg, _ = _stage4_cfgs()
    rec, turns, wavs, sr = _rec_with_audio(SPEAKERS, "ABABCACBBACB", bc_after=(1, 4, 8), seed=5)
    ch = {n: i for i, n in enumerate(SPEAKERS)}
    r = _retimer(TriVapRetimer, rcfg)
    plans = r.plan(rec, turns, wavs, sr, ch, set())
    assert len(plans) == len(turns)
    ends = {}
    for p in sorted(plans, key=lambda p: p["start_sec"]):
        c = ch[turns[p["index"]]["speaker"]]
        assert p["start_sec"] >= ends.get(c, 0.0) - 1e-3       # 同じチャネルは重ならない
        ends[c] = p["start_sec"] + len(wavs[p["index"]]) / sr
    methods = {p["retime"]["method"] for p in plans}
    assert methods & {"vap", "vap-hazard", "vap-hazard-timeout", "fallback-latency"}
    # 次の話者の判断では、その話者の音声が「注目」側に入っている（3本目の ch2 も単独で渡る）
    assert any(np.array_equal(x1, np.zeros_like(x1)) is False for _, x1 in r._vap.calls)


def test_tri_scanner_three_speakers_uses_both_listeners():
    from retime_tri_vap import TriInsertionScanner, TriVapRetimer, tri_build_timeline
    rcfg, fcfg = _stage4_cfgs()
    rec, turns, wavs, sr = _rec_with_audio(SPEAKERS, "ABACABCBCACB", seed=7)
    ch = {n: i for i, n in enumerate(SPEAKERS)}
    r = _retimer(TriVapRetimer, rcfg)
    plans = r.plan(rec, turns, wavs, sr, ch, set())
    tl, _ = tri_build_timeline(turns, plans, wavs, sr, ch)
    slots = TriInsertionScanner(r, fcfg).scan(tl, turns, plans, wavs, sr, ch, set())
    assert len(r._bc.calls) == 3                               # 聞き手ごとに1方向
    first_turn = {n: min(i for i, t in enumerate(turns) if t["speaker"] == n) for n in SPEAKERS}
    for s in slots:
        assert s["listener"] != turns[s["anchor_index"]]["speaker"]
        assert first_turn[s["listener"]] < s["anchor_index"]   # まだ話していない人は打たない
    by_anchor = {}
    for s in slots:
        by_anchor.setdefault(s["anchor_index"], set()).add(s["listener"])
    assert any(len(v) == 2 for v in by_anchor.values())        # 2人の聞き手が両方打つ発話がある


def test_tri_configs_are_three_party_and_two_party_is_not():
    import main_tri
    for name in ("configs/tri_ft_v1.yaml", CFG):
        main_tri.check_config(load_config(str(ROOT / name)), name)
    with pytest.raises(SystemExit):
        main_tri.check_config(load_config(str(ROOT / "configs/ft_v1.yaml")), "2party")


# ── 発話の宛先（addressee）─────────────────────────────────────────────────
from turn_order import BOTH, addressee_stats, normalize_addressee   # noqa: E402


def test_normalize_addressee():
    n = SPEAKERS
    assert normalize_addressee("ミサキ", "イノウエ", n) == "ミサキ"
    assert normalize_addressee("ミサキさん", "イノウエ", n) == "ミサキ"
    assert normalize_addressee("Both", "イノウエ", n) == BOTH
    assert normalize_addressee("二人", "イノウエ", n) == BOTH
    assert normalize_addressee(["ミサキ", "ゴロウ"], "イノウエ", n) == BOTH
    assert normalize_addressee(["ミサキ"], "イノウエ", n) == "ミサキ"
    assert normalize_addressee("イノウエ", "イノウエ", n) is None        # 自分自身
    assert normalize_addressee("タナカ", "イノウエ", n) is None          # 話者にいない
    assert normalize_addressee("", "イノウエ", n) is None


def test_addressee_stats_counts_next_speaker():
    t = _turns("ABACB")
    t[2]["addressee"] = "ゴロウ"           # A→C の後に C が話す
    t[3]["addressee"] = "イノウエ"         # C→A の後に B が口を挟む
    st = addressee_stats(t, SPEAKERS)
    assert st["single"] == 4 and st["both"] == 1
    assert st["next_is_addressee"] == round(2 / 3, 3)   # B→A の次 A、A→C の次 C、C→A の次 B


def test_apply_addressees_fills_unreadable_with_inference():
    t = _turns("ABCA")
    t[1]["addressee"] = "ミサキさん"                    # 自分自身 → 読めない
    t[2]["addressee"] = "誰か"
    t[2]["utterance"] = "イノウエさん、どうですか？"
    n = gdt.apply_addressees(t, SPEAKERS)
    assert n == 2
    assert t[1]["addressee"] == "イノウエ" and t[1]["addressee_src"] == "inferred"   # 直前の話者
    assert t[2]["addressee"] == "イノウエ" and t[2]["addressee_src"] == "inferred"   # 呼びかけ
    assert "addressee_src" not in t[3]


def test_merge_takes_addressee_of_last_part(stage2_patched):
    t = [{"speaker": "イノウエ", "addressee": "ミサキ", "utterance": "そうなんですね"},
         {"speaker": "イノウエ", "addressee": "ゴロウ", "utterance": "ゴロウさんはどうですか？"},
         {"speaker": "ゴロウ", "addressee": "イノウエ", "utterance": "僕は釣りです。"},
         {"speaker": "ミサキ", "addressee": "ゴロウ", "utterance": "釣り。"},
         {"speaker": "ミサキ", "utterance": "いいですね。"}]
    out = gd.merge_consecutive_same_speaker(t)
    ref = gdt._ORIG["merge_consecutive_same_speaker"](t)
    assert [x["utterance"] for x in out] == [x["utterance"] for x in ref]   # 本文は二人用のまま
    assert [x["addressee"] for x in out] == ["ゴロウ", "イノウエ", "ゴロウ"]  # 宛先の無い部分は前を残す


def test_proofread_message_shows_addressee(stage2_patched):
    t = _turns("ABA")
    t[2]["addressee"] = "both"
    msg = gd._proofread_user_message(t, _meta_s2())
    assert "0 イノウエ→both: はい。" in msg and "1 ミサキ→イノウエ: はい。" in msg
    assert "2 イノウエ→both: はい。" in msg


def test_gate_retries_when_addressees_are_unreadable():
    bad = _fake_result("ABABACACBCBCABABCACA")
    for x in bad["dialogue"]["turns"]:
        x.pop("addressee")
    out, n = _run_gate([bad, _fake_result("ABABACACBCBCABABCACA")])
    assert n == 2 and out["dialogue"]["addressee_stats"]["inferred"] == 0


def test_tri_pack_schema_has_addressee():
    pack = gd.load_prompt_pack("prompts/tri/stage2_first_meeting.yaml")
    assert '"addressee"' in pack.schema_plain and '"addressee"' in pack.schema_enriched
    for prof in ("qwen", "gemma"):
        s = gd.build_system_prompt(gd.S2Params(), gd.ALL_GATES, 1, prof, pack)
        assert '"addressee"' in s and "宛先の人が次に話すとは限らない" in s
    assert "→" in pack.proofread_prompt


def test_stage3_carries_addressee():
    import generate_voice_passthrough as gvp
    import main_tri
    orig = gvp.build_voice_turns
    main_tri.apply_voice_patches()
    try:
        t = _turns("ABCA")
        t[2]["addressee_src"] = "inferred"
        out = gvp.build_voice_turns(t)
        assert [o["addressee"] for o in out] == [x["addressee"] for x in t]
        assert out[2]["addressee_src"] == "inferred" and "addressee_src" not in out[0]
    finally:
        main_tri.remove_voice_patches()
    assert gvp.build_voice_turns is orig


def test_refine_inserted_backchannel_gets_addressee():
    import generate_tts as gt
    import generate_tts_tri as gtt
    import refine_maai as rm
    cfg = gt.load_config(str(ROOT / CFG))
    gtt.apply_patches(cfg)
    try:
        turns = [{"speaker": "イノウエ", "text": "これは長めの発話です。", "type": "speech",
                  "addressee": "ミサキ", "duration": 3.0},
                 {"speaker": "ミサキ", "text": "そうなんですね。", "type": "speech",
                  "addressee": "イノウエ", "duration": 1.0}]
        rec = {"id": "x", "voice_dialogue": {"turns": turns}}
        ops = [{"op": "insert", "new_text": "うん",
                "_slot": {"anchor_index": 0, "rel_pos": 0.5, "p": 0.6, "listener": "ゴロウ"}}]
        new_rec, _, _, inserted, _ = rm.apply_ops(rec, turns, ops, None)
        nt = new_rec["voice_dialogue"]["turns"]
        (ni,) = inserted
        assert nt[ni]["speaker"] == "ゴロウ" and nt[ni]["addressee"] == "イノウエ"
        assert nt[0]["addressee"] == "ミサキ"                  # 既存ターンの宛先は残る
    finally:
        gtt.remove_patches()
