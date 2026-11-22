"""rp_v1（重なりの場面）の台本の検査・音声の切り方・配置のテスト。"""

import copy

import numpy as np
import pytest

from rp_plan import (RPDeck, rp_edit_problem, RP_NOTE_RE, normalize_rp, rp_events, rp_scene_prompt, rp_shape_problem, rp_summary,
                     scene_candidates, scene_window, stage2_rp_block, strip_rp_keys, validate_scene)
import rp_audio
import rp_plan

M, P = "イノウエ", "サトウ"


def _t(sp, u, **kw):
    return {"speaker": sp, "utterance": u, **kw}


def _base_script():
    return [
        _t(P, "よろしくお願いします。"),
        _t(M, "よろしくお願いします。"),
        # 割り込まれて応じる
        _t(M, "最近は週末に近所の川沿いを走っていて、そのあと", unsaid="パン屋に寄るのが楽しみなんです。"),
        _t(P, "え、川沿いって、あの桜並木のところですか？", overlap="interrupt"),
        _t(M, "そうです、駅の裏の。"),
        # 反応が重なる
        _t(M, "この前は雨で滑って転びました。"),
        _t(P, "あはは。", overlap="react"),
        _t(M, "それ以来、雨の日は歩くことにしてます。"),
        # 同時に話し始めてイノウエが譲る
        _t(P, "あの、", unsaid="サトウさんはどこを走ってるんですか？"),
        _t(M, "そういえば", unsaid="サトウさんは運動されますか？", overlap="collision"),
        _t(M, "あ、すみません、どうぞ。"),
        _t(P, "いえ、私は全然走らないんですけど。"),
        # 重ねて応じる
        _t(P, "駅前の、あの、なんでしたっけ、大きい本屋さんの。"),
        _t(M, "ああ、丸い看板の。", overlap="coop"),
        _t(P, "そうそう、そこです。"),
        # 割り込む
        _t(P, "イノウエさんも東京のご出身だから、たぶん", unsaid="あのお店は知ってますよね。"),
        _t(M, "あ、すみません、私は九州の出身なんです。", overlap="interrupt"),
        _t(P, "あ、そうだったんですね。"),
    ]


def _fix_alternation(turns):
    # テスト用の台本は交互でなくてよい（normalize_rp は交互を前提にしない）
    return turns


def _vs(turns, data, kind, a, **kw):
    """場面の並び＋次のターンの言い直し（1ターン）で validate_scene を呼ぶ（範囲は並びから決める）。"""
    pat = rp_plan.full_pattern(kind)
    last = M if pat[-1] == "M" else P
    span = 1 if pat[0] == pat[-1] else 2
    if span == 1 and len(pat) > 1:
        span = 3
    d = data if isinstance(data, dict) else {"turns": data}
    d = {"turns": list(d["turns"]) + [_t(last, "それはいいですね、私もそう思います。")]}
    return validate_scene(turns, d, kind, M, P, a, a + span - 1, **kw)


PLAN = {"style": "rp", "opening": "話題から", "topics": ["ランニング", "本"],
        "overlaps": ["割り込まれて応じる", "反応が重なる", "同時に話し始めて譲る", "重ねて応じる", "割り込む"]}


def test_normalize_keeps_valid_script_and_counts_events():
    turns = _base_script()
    dropped = []
    normalize_rp(turns, dropped)
    assert dropped == []
    shapes = [s for s, _ in rp_events(turns, M)]
    assert shapes == ["cut_main", "react", "coll_main", "coop", "cut_user"]
    assert rp_shape_problem(turns, M, PLAN) is None
    s = rp_summary(turns, M)
    assert [x["turn"] for x in s] == [3, 6, 9, 13, 16]


def test_unsaid_without_interrupt_is_joined_back():
    turns = [_t(M, "それで、そのあと", unsaid="パン屋に寄るんです。"), _t(P, "いいですね。")]
    dropped = []
    normalize_rp(turns, dropped)
    assert "unsaid" not in turns[0]
    assert turns[0]["utterance"] == "それで、そのあとパン屋に寄るんです。"
    assert dropped


def test_interrupt_without_cut_becomes_coop_and_coop_after_cut_becomes_interrupt():
    turns = [_t(P, "駅前の大きい本屋さんで。"), _t(M, "ああ、あそこ。", overlap="interrupt")]
    normalize_rp(turns)
    assert turns[1]["overlap"] == "coop"
    turns = [_t(P, "駅前の大きい本屋さんで、", unsaid="買いました。"), _t(M, "ああ、あそこ。", overlap="coop")]
    normalize_rp(turns)
    assert turns[1]["overlap"] == "interrupt"
    assert turns[0]["utterance"] == "駅前の大きい本屋さんで、"   # 句点を付けない・読点は残す


def test_collision_needs_at_least_one_cut():
    turns = [_t(P, "あの、"), _t(M, "そういえば運動はされますか？", overlap="collision")]
    normalize_rp(turns)
    assert "overlap" not in turns[1]
    # 相手が先に言い終え、イノウエが言いかけで黙って引く（相手が先の型）は残す
    turns = [_t(P, "そういえば最近、駅前に新しいパン屋ができたんですよ。"),
             _t(M, "あの、", unsaid="運動はされますか？", overlap="collision")]
    normalize_rp(turns)
    assert turns[1]["overlap"] == "collision" and turns[1]["unsaid"]
    assert [s for s, _ in rp_events(turns, M)] == ["coll_implicit"]


def test_react_must_be_followed_by_the_speaker_continuing():
    turns = [_t(M, "雨で滑って転びました。"), _t(P, "あはは。", overlap="react"), _t(P, "大丈夫でしたか？")]
    normalize_rp(turns)
    assert "overlap" not in turns[1]


def test_shape_problem_reports_missing_and_short_cut():
    turns = _base_script()
    normalize_rp(turns)
    plan = dict(PLAN, overlaps=PLAN["overlaps"] + ["割り込まれて戻る"])
    why = rp_shape_problem(turns, M, plan)
    assert why and "割り込まれる 1/2" in why
    turns = _base_script()
    turns[2]["utterance"] = "最近は"
    normalize_rp(turns)
    assert "短い" in (rp_shape_problem(turns, M, PLAN) or "")


def test_strip_rp_keys_for_non_rp_dialogues():
    turns = _base_script()
    strip_rp_keys(turns)
    assert not any("unsaid" in t or "overlap" in t for t in turns)
    assert turns[2]["utterance"].endswith("楽しみなんです。")


def test_deck_gives_distinct_kinds_and_at_most_one_collision():
    deck = RPDeck({"seed": 1, "rp_count": {4: 1.0}}, ["読書", "ランニング", "料理", "旅行", "映画"])
    for _ in range(50):
        p = deck.next_plan({"age": 30})
        assert p["style"] == "rp" and len(p["overlaps"]) == 4
        assert len(set(p["overlaps"])) == 4
        assert sum(k.startswith("同時に") for k in p["overlaps"]) <= 1


def test_stage2_block_is_plain_chat_and_scene_prompt_has_steps():
    txt = stage2_rp_block(PLAN, M, P)
    assert "【話題の流れ】" in txt and "unsaid" not in txt
    base = [_t(P if i % 2 == 0 else M, f"発話{i}です。") for i in range(20)]
    pr = rp_scene_prompt(base, {"speech_style": "敬語"}, "割り込まれて戻る", 5, 5, M, P)
    assert "turn 5（" in pr and "【入れる場面】割り込まれて戻る" in pr and pr.count("\n3. ") == 1
    pc = rp_scene_prompt(base, {}, "重ねて応じる", 6, 6, M, P, variant=("言葉が出てこずに詰まる", "探している言葉を言う"))
    assert "言葉が出てこずに詰まる" in pc and "探している言葉を言う" in pc
    assert pr.count("←置き換える") == 2          # 書き方の説明と、対話の中の印
    assert "人物設定" not in pr
    assert "人物設定" in rp_scene_prompt(base, {}, "割り込む", 5, 10, M, P, persona="九州出身。")


def _chat(n=20):
    return [_t(P if i % 2 == 0 else M, f"これは{i}番目の発話です。") for i in range(n)]


def test_scene_window_spreads_over_body():
    ws = [scene_window(32, j, 4) for j in range(4)]
    assert ws[0][0] == 3 and ws[-1][1] <= 32 - 1 - 3
    assert all(a[1] < b[0] for a, b in zip(ws, ws[1:]))


def test_scene_candidates_respect_speaker_pattern_and_existing_scenes():
    turns = _chat(24)                       # 偶数番が P、奇数番が M
    turns[9]["utterance"] = "最近は週末になると、近所の川沿いを一時間くらいかけてゆっくり走っています。"
    c = scene_candidates(turns, "割り込まれて応じる", 3, 12, M, P, seed="x")
    assert c[0] == (9, 10) and all(turns[a]["speaker"] == M and b == a + 1 for a, b in c)
    c = scene_candidates(turns, "同時に話し始めて譲られる", 3, 12, M, P, seed="x")
    assert all(turns[a]["speaker"] == P and b == a + 2 for a, b in c)
    turns[10]["overlap"] = "react"          # 既存の場面の隣は選ばない
    c = scene_candidates(turns, "割り込まれて応じる", 3, 12, M, P, seed="x")
    assert all(not (8 <= a <= 12) for a, _ in c if a in (9, 11))


def test_validate_scene_accepts_interrupt_and_rejects_leak():
    turns = _chat()
    data = {"turns": [
        _t(M, "最近は週末に近所の川沿いをよく走っていて、そのあと", unsaid="駅前のパン屋に寄るんです。"),
        _t(P, "え、川沿いって、どのあたりですか？", overlap="interrupt"),
        _t(M, "駅の裏の、桜並木のところです。")]}
    new, why = _vs(turns, data, "割り込まれて応じる", 7)
    assert why == "" and new[0]["unsaid"] == "駅前のパン屋に寄るんです。" and new[1]["overlap"] == "interrupt"
    assert all(t["rp_scene"] == "割り込まれて応じる" for t in new)
    leak = {"turns": [data["turns"][0], _t(P, "パン屋って、どこのパン屋ですか？", overlap="interrupt"),
                      data["turns"][2]]}
    new, why = _vs(turns, leak, "割り込まれて応じる", 7)
    assert new is None and "聞こえていない" in why
    q = {"turns": [_t(M, "最近は週末に近所の川沿いをよく走っているんですか？", unsaid="私はそうです。"),
                   data["turns"][1], data["turns"][2]]}
    assert "問いで終わっている" in _vs(turns, q, "割り込まれて応じる", 7)[1]


def test_validate_scene_rejects_wrong_shape_and_speaker_order():
    turns = _chat()
    bad = {"turns": [_t(M, "そうなんですね。"), _t(P, "はい。"), _t(M, "なるほど。")]}
    assert "形が違う" in _vs(turns, bad, "割り込まれて応じる", 7)[1]
    order = {"turns": [_t(P, "あ。"), _t(M, "え。"), _t(P, "はい。")]}
    assert "並びが違う" in _vs(turns, order, "割り込まれて応じる", 7)[1]
    assert "ターン数" in _vs(turns, {"turns": bad["turns"][:2]}, "割り込まれて応じる", 7)[1]


def test_validate_scene_collision_and_return():
    turns = _chat()
    # turn 7（M）〜8（P）を同時発話に。イノウエが先に言いかけて、譲る
    data = {"turns": [
        _t(M, "あの、", unsaid="ご出身はどちらですか？"),
        _t(P, "そういえば", unsaid="最近寒いですね。", overlap="collision"),
        _t(M, "あ、すみません、どうぞ。"),
        _t(P, "いえ、最近急に寒くなりましたね。")]}
    new, why = _vs(turns, data, "同時に話し始めて譲る", 7)
    assert why == "" and len(new) == 5          # 並び4ターン＋次のターンの言い直し
    assert "並びが違う" in _vs(turns, data, "同時に話し始めて譲られる", 7)[1]
    # 割り込まれて戻る（3ターン）: 答えて、同じターンで言いかけたところから続ける
    ret = {"turns": [
        _t(M, "週末は近所の川沿いをよく走っていて、そのあと", unsaid="駅前のパン屋に寄るんです。"),
        _t(P, "川沿いって、どこの川ですか？", overlap="interrupt"),
        _t(M, "駅の裏の川です。それで、走ったあとはだいたい家に帰ります。")]}
    new, why = _vs(turns, ret, "割り込まれて戻る", 7)
    assert new is None and "戻っていない" in why
    ret["turns"][2] = _t(M, "駅の裏の川です。で、そのあと駅前のパン屋に寄るんです。")
    assert _vs(turns, ret, "割り込まれて戻る", 7)[1] == ""
    ret["turns"][2] = _t(M, "駅の裏です。週末は近所の川沿いをよく走っていて、駅前のパン屋に寄るんです。")
    assert "言い直し" in _vs(turns, ret, "割り込まれて戻る", 7)[1]
    ret["turns"][1] = _t(P, "川沿いを走る、ということですか？", overlap="interrupt")
    assert "決まった聞き返し" in _vs(turns, ret, "割り込まれて戻る", 7)[1]


def test_implicit_yield_collision_and_new_checks():
    turns = _chat()
    turns[6]["utterance"] = "最近は近所でおいしいパン屋を探してるんです。"   # 引いたあとの相手の話は直前とつながる
    data = {"turns": [
        _t(M, "あの、", unsaid="ご出身はどちらですか？"),
        _t(P, "そういえば最近、近所に新しいパン屋ができたんですよ。", overlap="collision"),
        _t(M, "へえ、どんなパン屋さんですか？")]}
    new, why = _vs(turns, data, "同時に話し始めて引く", 7)
    assert why == "" and "unsaid" not in new[1] and new[1]["overlap"] == "collision"
    ctx = turns[:7] + new + turns[9:]
    assert [s for s, _ in rp_events(ctx, M)] == ["coll_implicit"]
    bad = dict(data, turns=data["turns"][:2] + [_t(M, "あ、どうぞ。")])
    assert "譲る" in _vs(turns, bad, "同時に話し始めて引く", 7)[1]
    off = dict(data, turns=[data["turns"][0], _t(P, "そういえば最近、サブスクで古い映画を観ています。", overlap="collision"),
                            data["turns"][2]])
    assert "つながらない" in _vs(turns, off, "同時に話し始めて引く", 7)[1]
    sorry = dict(data, turns=[data["turns"][0], _t(P, "すみません、つい話し始めてしまいました。", overlap="collision"),
                              data["turns"][2]])
    assert "重なりに触れている" in _vs(turns, sorry, "同時に話し始めて引く", 7)[1]
    # 割り込まれて応じる: くり返しだけの返し
    sc = {"turns": [_t(M, "最近は週末に近所の川沿いをよく走っていて、そのあと", unsaid="駅前のパン屋に寄るんです。"),
                    _t(P, "あ、私も最近はスポーツから歌に変わったんです。", overlap="interrupt"),
                    _t(M, "スポーツから歌に変わったんですね。")]}
    assert "くり返し" in _vs(turns, sc, "割り込まれて応じる", 7)[1]
    # 反応が重なる: 続きが一般論
    rc = {"turns": [_t(M, "この前、雨の日に滑って転んでしまいました。"), _t(P, "あはは。", overlap="react"),
                    _t(M, "心身ともに、すっかりリラックスできました。")]}
    assert "一般論" in _vs(turns, rc, "反応が重なる", 7)[1]
    # 重ねて応じる（言葉探し）: 相手が言い終えていたら捨てる・くり返しだけの重ねも捨てる
    v = rp_plan.SCENE_VARIANTS["重ねて応じる"][0]
    co = {"turns": [_t(P, "最近は動画配信で昔の映画を観ているんです。"), _t(M, "映画ですね。", overlap="coop"),
                    _t(P, "そうそう。")]}
    assert "言い終えている" in _vs(turns, co, "重ねて応じる", 6, variant=v)[1]
    co["turns"][0] = _t(P, "最近は動画配信で、あの、昔の白黒の、なんでしたっけ、")
    co["turns"][1] = _t(M, "白黒のですね。", overlap="coop")
    assert "くり返し" in _vs(turns, co, "重ねて応じる", 6, variant=v)[1]
    co["turns"][1] = _t(M, "ローマの休日とか？", overlap="coop")
    assert _vs(turns, co, "重ねて応じる", 6, variant=v)[1] == ""
    co["turns"][2] = _t(P, "そうそう、共感していただけて嬉しいです。")
    assert "お礼" in _vs(turns, co, "重ねて応じる", 6, variant=v)[1]


def test_validate_scene_strips_copied_neighbor_turns():
    turns = _chat()
    scene = [_t(M, "最近は週末に近所の川沿いをよく走っていて、そのあと", unsaid="駅前のパン屋に寄るんです。"),
             _t(P, "え、川沿いって、どのあたりですか？", overlap="interrupt"),
             _t(M, "駅の裏の、桜並木のところです。"), _t(P, "ああ、あそこですか。")]
    data = {"turns": [dict(turns[6])] + scene + [dict(turns[9])]}     # 前後の元のターンを写してきた
    new, why = validate_scene(turns, data, "割り込まれて応じる", M, P, 7, 8)
    assert why == "" and len(new) == 4 and new[0]["unsaid"]


def test_validate_scene_tolerates_whole_dialogue_extras_and_coop_wordsearch():
    turns = _chat()
    scene = [_t(M, "最近は週末に近所の川沿いをよく走っていて、そのあと", unsaid="駅前のパン屋に寄るんです。"),
             _t(P, "え、川沿いって、どのあたりですか？", overlap="interrupt"),
             _t(M, "駅の裏の、桜並木のところです。")]
    whole = {"turns": [dict(t) for t in turns[:7]] + scene + [_t(P, "それはいいですね。")] + [dict(t) for t in turns[9:]]}
    new, why = validate_scene(turns, whole, "割り込まれて応じる", M, P, 7, 8)
    assert why == "" and len(new) == 4
    extra = {"turns": scene + [_t(P, "ああ、あそこですか。"), _t(M, "そうです。")]}
    new, why = _vs(turns, extra, "割り込まれて応じる", 7)
    assert why == "" and len(new) == 6
    odd = {"turns": scene + [_t(P, "ああ、あそこですか。")]}
    assert "ターン数" in _vs(turns, odd, "割り込まれて応じる", 7)[1]
    coop = {"turns": [_t(P, "駅前の、あの、なんでしたっけ、大きい", unsaid="本屋さんです。"),
                      _t(M, "丸善ですか？", overlap="interrupt"), _t(P, "そうそう、そこです。")]}
    new, why = _vs(turns, coop, "重ねて応じる", 6)
    assert why == "" and "unsaid" not in new[0] and new[1]["overlap"] == "coop"


def test_yield_then_silence_scene_sets_pause_and_shape():
    turns = _chat()
    data = {"turns": [
        _t(M, "あの、", unsaid="ご出身はどちらですか？"),
        _t(P, "そういえば", unsaid="最近寒いですね。", overlap="collision"),
        _t(M, "あ、すみません、どうぞ。"),
        _t(M, "じゃあ、私からいいですか、ご出身はどちらなんですか？")]}
    new, why = _vs(turns, data, "同時に話し始めて譲り、相手が黙る", 7)
    assert why == "" and 1.2 <= new[3]["pause_before_sec"] <= 2.2
    ctx = turns[:7] + new + turns[9:]
    assert [s for s, _ in rp_events(ctx, M)] == ["coll_silent"]
    bad = dict(data, turns=data["turns"][:3] + [_t(M, "どうぞ。")])
    assert "フォロー" in _vs(turns, bad, "同時に話し始めて譲り、相手が黙る", 7)[1]
    c = scene_candidates(turns, "同時に話し始めて譲り、相手が黙る", 3, 12, M, P)
    assert c and all(b == a + 1 and turns[a]["speaker"] == M for a, b in c)


def test_partner_first_collisions_and_meaning_checks():
    turns = _chat()
    turns[6]["utterance"] = "最近は駅前のパン屋によく行くんです。"
    # 相手が先に言い終え、イノウエが言いかけで黙って引く（turn 6 から: P M M ＋ 言い直し P）
    data = {"turns": [_t(P, "そういえば駅前のパン屋で、新しいメロンパンが出てたんですよ。"),
                      _t(M, "あの、", unsaid="パンはよく食べますか？", overlap="collision"),
                      _t(M, "メロンパン、いいですね、焼きたてだと香りがすごそうです。")]}
    new, why = _vs(turns, data, "同時に話し始めて引く（相手が先）", 6)
    assert why == "" and new[1]["unsaid"] and new[1]["overlap"] == "collision"
    # 相手が先に言いかけて2人とも止まり、イノウエが名前を呼んで譲る
    data2 = {"turns": [_t(P, "あ、それで", unsaid="パン屋の話なんですけど。"),
                       _t(M, "あの、", unsaid="パンはよく食べますか？", overlap="collision"),
                       _t(M, "サトウさんから、すみません。"),
                       _t(P, "パン屋のメロンパンがおいしかったんですよ。")]}
    new, why = _vs(turns, data2, "同時に話し始めて譲る（相手が先）", 6)
    assert why == "", why
    # 割り込まれて戻る（意味を聞く型）: ふつうの言葉の意味を聞いたら捨てる
    v0 = rp_plan.SCENE_VARIANTS["割り込まれて戻る"][0]
    ret = {"turns": [_t(M, "旅先で景色を眺めていると、だんだん視界が", unsaid="広がる感じがするんです。"),
                     _t(P, "視界って、どういう意味ですか？", overlap="interrupt"),
                     _t(M, "見える範囲のことです。それで、だんだん広がる感じがするんです。")]}
    assert "ふつうの言葉" in _vs(turns, ret, "割り込まれて戻る", 7, variant=v0)[1]
    ret["turns"][0] = _t(M, "旅先で景色を眺めていると、いわゆるマインドフルネスな", unsaid="状態になって落ち着くんです。")
    ret["turns"][1] = _t(P, "マインドフルネスって何ですか？", overlap="interrupt")
    ret["turns"][2] = _t(M, "今の瞬間に意識を向けることです。で、そういう状態になって落ち着くんです。")
    assert _vs(turns, ret, "割り込まれて戻る", 7, variant=v0)[1] == ""
    # 反応が評価のひと言
    rc = {"turns": [_t(M, "この前、雨の日に滑って転んでしまいました。"), _t(P, "いいですね。", overlap="react"),
                    _t(M, "それで、ズボンが泥だらけになって帰ったんです。")]}
    assert "評価" in _vs(turns, rc, "反応が重なる", 7)[1]


def test_rp_edit_problem_keeps_scene_roles():
    assert rp_edit_problem({"utterance": "x"}, "a", "b") == ""
    assert "言いかけ" in rp_edit_problem(_t(M, "それで、", unsaid="残り。"), "それで、", "それで。")
    assert "譲る" in rp_edit_problem(_t(M, "あ、どうぞ。", rp_scene="同時に話し始めて譲る"), "あ、どうぞ。", "あ、いいですね。")
    assert "直さない" in rp_edit_problem(_t(P, "あはは。", overlap="react"), "あはは。", "えっ、本当ですか？")
    assert rp_edit_problem(_t(M, "ご家族との時間が大切です。", rp_scene="反応が重なる"),
                           "ご家族との時間が大切です。", "家族との時間が大切です。") == ""
    assert RP_NOTE_RE.sub("", "（相手の話をさえぎって）え、そうなんですか？") == "え、そうなんですか？"


# ── Stage4: 言いかけの切り方 ──────────────────────────────────────────

def _fake_speech(sr, segs):
    """(秒, 有声か) の並びから、有声区間に正弦波を置いた波形。"""
    out = []
    for sec, voiced in segs:
        n = int(sec * sr)
        t = np.arange(n) / sr
        out.append((0.3 * np.sin(2 * np.pi * 180 * t)) if voiced else np.zeros(n))
    return np.concatenate(out).astype(np.float32)


def test_prepare_and_apply_cuts_round_trip():
    sr = 16000
    rec = {"id": "x", "voice_dialogue": {"turns": [
        {"speaker": M, "text": "あいうえおかきくけこ", "unsaid": "さしすせそたちつてと。"},
        {"speaker": P, "text": "はい。", "overlap": "interrupt"}]}}
    assert rp_audio.prepare_cut_turns(rec) == 1
    t0 = rec["voice_dialogue"]["turns"][0]
    assert t0["text"] == "あいうえおかきくけこさしすせそたちつてと。"
    wavs = {0: _fake_speech(sr, [(0.1, False), (2.0, True), (0.1, False)]),
            1: _fake_speech(sr, [(0.5, True)])}
    assert rp_audio.apply_cuts(rec, wavs, sr, asr=None) == 1
    assert t0["text"] == "あいうえおかきくけこ" and "cut_text" not in t0
    # 20モーラ中10モーラ → 声のある 2.0秒の半分あたり（＋前の無音 0.1秒）で切る
    assert abs(len(wavs[0]) / sr - 1.1) < 0.12
    assert t0["cut_sec"] == pytest.approx(len(wavs[0]) / sr, abs=1e-3)
    # 2回目は何もしない
    assert rp_audio.apply_cuts(rec, wavs, sr, asr=None) == 0
    assert rp_audio.prepare_cut_turns(rec) == 0


def test_snap_to_quiet_moves_cut_into_a_gap():
    sr = 16000
    w = _fake_speech(sr, [(0.5, True), (0.05, False), (0.5, True)])
    n = rp_audio.snap_to_quiet(w, sr, int(0.52 * sr))        # 無音（0.50〜0.55）の少し手前から
    assert 0.5 <= n / sr <= 0.56


def test_cut_snaps_to_pause_after_comma():
    sr = 16000
    # 「あいうえお、」で切る。読点のあとのポーズ（0.3秒）の頭に寄せる
    w = _fake_speech(sr, [(1.0, True), (0.3, False), (1.0, True)])
    n, info = rp_audio.cut_sample(w, sr, "あいうえお、", "あいうえお、かきくけこ。")
    assert info.get("snap") == "pause"
    assert 0.98 <= n / sr <= 1.08


def test_refine_cut_with_asr_moves_toward_spoken_end(monkeypatch):
    sr = 16000
    w = _fake_speech(sr, [(0.1, False), (2.0, True), (0.1, False)])
    import reading_check
    monkeypatch.setattr(reading_check, "kana", lambda t: t)       # テストは読みをそのまま使う
    full = "アイウエオカキクケコサシスセソタチツテト"
    # 偽の ASR: 切った長さに比例して全文の先頭を返す（声のある 2.0秒で20文字）
    def fake_asr(ws, _sr):
        n = int(round((len(ws[0]) / _sr - 0.1) / 2.0 * 20))
        return [full[:max(0, n)]]
    n0 = int(0.1 * sr + 0.6 * sr)                  # 6文字ぶん（早すぎ）。言いかけは10文字
    n, info = rp_audio.refine_cut_with_asr(w, sr, n0, full[:10], full, fake_asr)
    assert info["asr_fix"][0]["d"] == 4
    assert abs(n / sr - 1.1) < 0.1


# ── Stage4: 配置 ─────────────────────────────────────────────────────

def test_retime_rp_placement_helpers():
    from retime_maai import MaaiRetimer, RetimeConfig
    r = MaaiRetimer.__new__(MaaiRetimer)
    r.cfg = RetimeConfig()
    r._rng = np.random.default_rng(0)
    turns = [{"speaker": M, "text": "a", "type": "speech", "unsaid": "b"},
             {"speaker": P, "text": "c", "type": "speech", "overlap": "interrupt"},
             {"speaker": M, "text": "d", "type": "speech"},
             {"speaker": P, "text": "え", "type": "speech", "overlap": "react"},
             {"speaker": M, "text": "e", "type": "speech"},
             {"speaker": P, "text": "あの", "type": "speech", "unsaid": "x"},
             {"speaker": M, "text": "その", "type": "speech", "unsaid": "y", "overlap": "collision"}]
    placed = {0: (1.0, 3.0)}
    assert r._rp_target(1, turns, placed, P) == 0
    start, method, anchor, det = r._decide_rp("interrupt", placed[0])
    assert method == "rp-interrupt" and 3.0 - 1.2 - 1e-9 <= start <= 3.0 - 0.12 + 1e-9
    # 2人とも言いかけで止まった直後のターン: 2人の末尾から after_collision_sec の間をおく
    placed2 = {5: (10.0, 10.4), 6: (10.05, 10.5)}
    turns2 = turns + [{"speaker": M, "text": "どうぞ", "type": "speech"}]
    assert r._after_collision(7, turns2, placed2) == 10.5
    assert r._after_collision(4, turns, {0: (0, 1)}) is None
    vals = [r._rp_uniform((0.1, 0.9), log=True) for _ in range(400)]
    assert 0.1 <= min(vals) and max(vals) <= 0.9 and 0.22 < sorted(vals)[200] < 0.4   # 中央値 ≈ 0.3
    assert r._react_through(4, turns, M) and not r._react_through(2, turns, M)
    assert r._collision_firsts(turns) == {5}
    start, method, anchor, det = r._decide_rp("collision", (10.0, 10.5))
    assert method == "rp-collision" and 10.0 <= start <= 10.2
    # 短い相手の発話には frac までしか重ねない
    start, _, _, det = r._decide_rp("coop", (5.0, 5.4))
    assert det["rp_overlap"] <= 0.2 + 1e-9


def test_scene_system_carries_the_spoken_style_section():
    """場面を入れる呼び出しに、本生成の「話し言葉」の段（書き言葉の禁止と書き換え例）を渡す（2026-11-12）。"""
    import generate_dialogues as g
    sysmsg = rp_plan.rp_scene_system(g.GENERAL_S2.system_prompt)
    assert "# 最重要" in sysmsg and "書き換えの例" in sysmsg
    assert "# 最重要: 登場する話者" not in sysmsg          # 次の段は含めない
    assert rp_plan.rp_scene_system("話し言葉の段の無いプロンプト") == ""


def test_correction_variant_is_rarer_and_must_fix_partners_own_words():
    """正す型は人の割り込み（訂正8%）に合わせて少なめ。正す中身は相手の前の発話に出た事柄（2026-11-13）。"""
    from collections import Counter
    c = Counter("取り違え" in str(rp_plan.scene_variant("割り込まれて応じる", str(k))) for k in range(4000))
    assert 0.08 < c[True] / 4000 < 0.18
    v = next(x for x in rp_plan.SCENE_VARIANTS["割り込まれて応じる"] if "取り違え" in x)
    M, P = "イノウエ", "シマダ"
    turns = [{"speaker": M if i % 2 == 0 else P, "utterance": u} for i, u in enumerate(
        ["はじめまして。", "はじめまして、シマダです。", "ご趣味は何ですか？", "週末は山に登ってます。", "いいですね。", "近所の低い山ですけど。",
         "なるほど。", "ええ。", "そうなんですね。", "はい。"])]
    def scene(interrupt):
        return {"turns": [
            {"speaker": M, "utterance": "シマダさんは高い山に登るのが好きで、毎週のように遠くまで", "unsaid": "行かれてるんですよね。"},
            {"speaker": P, "utterance": interrupt, "overlap": "interrupt"},
            {"speaker": M, "utterance": "あ、そうでしたか。近場なら気軽に行けますね。"},
            {"speaker": P, "utterance": "ええ、朝だけ歩いて帰ってくるんです。"}]}
    new, why = rp_plan.validate_scene(turns, scene("あ、いえ、近所の低い山なんですよ。"), "割り込まれて応じる", M, P, 6, 7, variant=v)
    assert new is not None, why
    new, why = rp_plan.validate_scene(turns, scene("いや、気合っていうか、情熱ですかね。"), "割り込まれて応じる", M, P, 6, 7, variant=v)
    assert new is None and "前の話に無い" in why


def test_resume_scene_has_a_non_question_variant():
    """割り込まれて戻るに、問いでない短いひと言で割り込まれて話を戻す型（2026-11-13。人の割り込みで問いはまれ）。"""
    vs = rp_plan.SCENE_VARIANTS["割り込まれて戻る"]
    v = vs[2]
    assert "問いにしない" in v
    M, P = "イノウエ", "シマダ"
    turns = [{"speaker": M if i % 2 == 0 else P, "utterance": u} for i, u in enumerate(
        ["はじめまして。", "はじめまして、シマダです。", "ご趣味は何ですか？", "週末は山に登ってます。", "いいですね。", "近所の低い山ですけど。",
         "なるほど。", "ええ。", "そうなんですね。", "はい。"])]
    def scene(interrupt):
        return {"turns": [
            {"speaker": M, "utterance": "私は週末に、えっと、近所の川沿いを走ってて、そこから見える", "unsaid": "夕日がすごくきれいなんです。"},
            {"speaker": P, "utterance": interrupt, "overlap": "interrupt"},
            {"speaker": M, "utterance": "そうなんです。で、そこから見える夕日が、すごくきれいなんですよ。"},
            {"speaker": P, "utterance": "へえ、いいですね。"}]}
    new, why = rp_plan.validate_scene(turns, scene("あ、あの川沿い、桜きれいですよね。"), "割り込まれて戻る", M, P, 6, 7, variant=v)
    assert new is not None, why
    new, why = rp_plan.validate_scene(turns, scene("川沿いって、どこの川ですか？"), "割り込まれて戻る", M, P, 6, 7, variant=v)
    assert new is None and "問い" in why


def test_key_name_fragment_is_stripped_from_utterance_head():
    """JSON のキー名の切れ端（utterance / erance）が発話の頭に残ったら外す（2026-11-13）。英単語の発話は残す。"""
    assert rp_plan._strip_key_leak("eranceあの、地元の魚って") == "あの、地元の魚って"
    assert rp_plan._strip_key_leak("utterance直接、形にする仕事") == "直接、形にする仕事"
    assert rp_plan._strip_key_leak("iPhoneを買いました") == "iPhoneを買いました"
    assert rp_plan._strip_key_leak("cameraが好きです") == "cameraが好きです"
