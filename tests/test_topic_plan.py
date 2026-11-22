"""metadata.topic_plan（1対話で扱う話題の組と順番）のテスト。

指定が無い（config に無い・メタデータに無い）ときは Stage1 / Stage2 のプロンプトを変えない。
"""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_dialogues import build_dialogue_prompt                      # noqa: E402
from generate_metadata import (DiversityTracker, attach_topic_plan,       # noqa: E402
                               build_user_prompt)
from topic_plan import (OPENING_TEXT, TopicDeck, fits_age, stage1_topic_text,  # noqa: E402
                        stage2_topic_block, topic_group, topic_plan_of)

MS = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "大学の研究者"}
POOL = ["読書", "好きな作家", "ランニング", "筋トレ", "ペット", "家庭菜園", "旅行",
        "仕事の内容", "転職", "出身地", "天気の話", "睡眠", "料理", "ラーメン"]


def _meta(plan=None):
    m = {"id": "x", "scenario": "イノウエと岡田が初めて会う",
         "speakers": [{"name": "イノウエ", "gender": "male", "role": "大学の研究者", "personality": "穏やか"},
                      {"name": "オカダ", "gender": "female", "role": "スーパーの店員",
                       "personality": "控えめ", "age_range": "40代"}],
         "target_turns": 20, "main_speaker_index": 0}
    if plan:
        m["topic_plan"] = plan
    return m


# --- 話題の組の引き方 -------------------------------------------------------

def test_deck_follows_opening_and_count_weights():
    deck = TopicDeck({"opening_mix": {"話題から": 1.0, "近況から": 1.0},
                      "topics": {2: 1.0}}, POOL)
    plans = [deck.next_plan() for _ in range(10)]
    assert Counter(p["opening"] for p in plans) == {"話題から": 5, "近況から": 5}
    assert all(len(p["topics"]) == 2 for p in plans)


def test_deck_never_repeats_a_group_within_a_dialogue():
    deck = TopicDeck({"topics": {3: 1.0}}, POOL)
    for _ in range(20):
        groups = [topic_group(t) for t in deck.next_plan()["topics"]]
        assert len(groups) == len(set(groups)), groups


def test_deck_skips_the_group_the_opening_already_covers():
    deck = TopicDeck({"opening_mix": {"仕事・学校から": 1.0}, "topics": {3: 1.0}}, POOL)
    for _ in range(10):
        assert "仕事" not in [topic_group(t) for t in deck.next_plan()["topics"]]


def test_work_and_school_are_never_the_first_topic():
    deck = TopicDeck({"topics": {2: 0.5, 3: 0.5}}, POOL)
    firsts = [deck.next_plan()["topics"][0] for _ in range(30)]
    assert not any(topic_group(t) in {"仕事", "学校"} for t in firsts)
    # 1つめにならないだけで、2つめ以降には出る
    later = [t for _ in range(30) for t in deck.next_plan()["topics"][1:]]
    assert any(topic_group(t) == "仕事" for t in later)


@pytest.mark.parametrize("topic,age,want", [
    ("通学", 77, False), ("通学", 22, True), ("通学", None, True),
    ("孫の話", 30, False), ("孫の話", 68, True),
    ("アルバイト", 45, False), ("読書", 77, True),
])
def test_fits_age(topic, age, want):
    assert fits_age(topic, age) is want


def test_deck_drops_topics_that_do_not_fit_the_card_age():
    pool = POOL + ["通学", "孫の話", "学生生活"]
    deck = TopicDeck({"topics": {3: 1.0}}, pool)
    old = {"gender": "female", "age": 77, "hobbies": []}
    got = {t for _ in range(20) for t in deck.next_plan(old)["topics"]}
    assert not ({"通学", "学生生活"} & got)
    young = TopicDeck({"topics": {3: 1.0}}, pool)
    got = {t for _ in range(20) for t in young.next_plan({"age": 22, "hobbies": []})["topics"]}
    assert "孫の話" not in got


def test_deck_spreads_topics_over_the_pool():
    deck = TopicDeck({"topics": {2: 1.0}, "seed": 3}, POOL)
    used = Counter(t for _ in range(20) for t in deck.next_plan()["topics"])
    assert len(used) >= len(POOL) - 2           # ほぼ全部の話題が出る
    assert max(used.values()) - min(used.values()) <= 2


def test_deck_can_take_a_topic_from_the_partner_card():
    card = {"hobbies": ["ドローン撮影", "指圧"]}
    always = TopicDeck({"topics": {2: 1.0}, "card_hobby": 1.0}, POOL)
    assert any(t in card["hobbies"] for t in always.next_plan(card)["topics"])
    never = TopicDeck({"topics": {2: 1.0}, "card_hobby": 0.0}, POOL)
    assert not any(t in card["hobbies"] for t in never.next_plan(card)["topics"])


def test_deck_add_used_keeps_the_ratio_when_appending():
    cfg = {"opening_mix": {"話題から": 1.0, "近況から": 1.0}, "topics": {2: 1.0}}
    deck = TopicDeck(cfg, POOL)
    for _ in range(4):
        deck.add_used({"opening": "話題から", "topics": ["読書", "旅行"]})
    assert [deck.next_plan()["opening"] for _ in range(4)] == ["近況から"] * 4


def test_deck_rejects_an_unknown_opening_label():
    with pytest.raises(ValueError):
        TopicDeck({"opening_mix": {"雑談から": 1.0}}, POOL)


# --- Stage1 -----------------------------------------------------------------

def _stage1(topic_plan=None, seed=0):
    random.seed(seed)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS,
                          partner_roles={"会社員": {"weight": 1, "examples": "営業職"}},
                          topic_plan=topic_plan)
    if topic_plan:
        tr.topic_deck.pool = list(POOL)
    plans = []
    return build_user_prompt(1, 16, 24, tr, num_speakers=2, plan_out=plans), plans


def test_stage1_unchanged_without_topic_plan():
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS,
                          partner_roles={"会社員": {"weight": 1, "examples": "営業職"}})
    old = build_user_prompt(1, 16, 24, tr, num_speakers=2)
    new, plans = _stage1()
    assert new == old and plans == []
    assert "この対話で扱う話題" not in new


def test_stage1_lists_the_topics_in_order():
    p, plans = _stage1({"topics": {3: 1.0}})
    assert len(plans) == 1 and len(plans[0]["topics"]) == 3
    t = plans[0]["topics"]
    assert f"- 話題: 1) {t[0]} → 2) {t[1]} → 3) {t[2]}" in p
    assert OPENING_TEXT[plans[0]["opening"]] in p
    assert "conversation_flow は5項目" in p
    assert f"genre には「{t[0]}」" in p


def test_attach_topic_plan_marks_every_record():
    plan = {"opening": "話題から", "topics": ["読書", "旅行"]}
    metas = [_meta(), _meta()]
    attach_topic_plan(metas, plan)
    assert all(m["topic_plan"] is plan for m in metas)


def test_tracker_counts_the_planned_topics_not_the_written_genre():
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS,
                          topic_plan={"topics": {2: 1.0}})
    m = _meta({"opening": "話題から", "topics": ["読書", "旅行"]})
    m["genre"] = "仕事の内容"
    tr.add(m)
    assert tr.genre_counts == Counter({"読書": 1, "旅行": 1})
    assert tr.topic_deck.topic_counts == Counter({"読書": 1, "旅行": 1})


# --- Stage2 -----------------------------------------------------------------

def test_stage2_block_without_raw_plan():
    plan = {"opening": "話題から", "topics": ["読書", "旅行"]}
    p = build_dialogue_prompt(_meta(plan), persona="- 性格・人柄: 穏やか")
    assert "topic_plan" not in p and "opening" not in p
    assert "【話題の流れ】" in p and "1) 読書 → 2) 旅行" in p
    assert "仕事・職業・学校の話は" in p          # 仕事から入らない対話では釘を刺す
    # 段落を外した残りは、指定の無いメタデータのプロンプトと同じ
    base = build_dialogue_prompt(_meta(), persona="- 性格・人柄: 穏やか")
    i0 = p.index("\n\n【話題の流れ】")
    i1 = p.index("\n\n発話は1ターン1文だけにし")
    assert p[:i0] + p[i1:] == base


def test_stage2_block_is_quiet_about_work_when_the_opening_is_work():
    plan = {"opening": "仕事・学校から", "topics": ["読書", "旅行"]}
    assert "仕事・職業・学校の話は" not in stage2_topic_block(plan)


def test_stage2_does_not_mutate_metadata():
    m = _meta({"opening": "話題から", "topics": ["読書"]})
    build_dialogue_prompt(m)
    assert "topic_plan" in m


def test_topic_plan_of_ignores_broken_values():
    assert topic_plan_of({}) is None
    assert topic_plan_of({"topic_plan": {"opening": "話題から"}}) is None
    assert topic_plan_of({"topic_plan": {"opening": "謎", "topics": ["読書"]}}) is None


def test_stage1_text_matches_the_topic_count():
    txt = stage1_topic_text({"opening": "近況から", "topics": ["読書", "旅行"]})
    assert "conversation_flow は4項目" in txt and "続く2項目" in txt


# --- 2026-10-28 追加: 話題の切り替えの前置きと、主話者の職業の話 -------------

def test_stage2_block_limits_shift_cue_and_suppresses_main_job():
    b = stage2_topic_block({"opening": "話題から", "topics": ["天気の話", "ランニング"]},
                           "イノウエ")
    # 2026-10-29: 前置きを「置け」から「1回まで」に（人間の先頭33ターンは 0.02回/対話）
    assert "対話全体で1回まで" in b and "前置きを置いてください" not in b
    assert "主話者（イノウエ）の研究・大学・仕事の話" in b
    assert "自分からは持ち出さない" in b


def test_stage2_block_no_shift_cue_for_single_topic():
    b = stage2_topic_block({"opening": "話題から", "topics": ["天気の話"]}, "イノウエ")
    assert "そういえば" not in b


def test_stage2_block_work_opening_lets_the_partner_go_first():
    b = stage2_topic_block({"opening": "仕事・学校から", "topics": ["野球", "カラオケ"]},
                           "イノウエ")
    assert "研究・大学・仕事の話" not in b
    assert "主話者（イノウエ）は自分から先に職業を言わず" in b


def test_stage1_text_bans_mutual_job_intro_unless_work_opening():
    t = stage1_topic_text({"opening": "話題から", "topics": ["天気の話"]})
    assert "互いの職業を紹介し合うやり取りを書かないでください" in t
    t2 = stage1_topic_text({"opening": "仕事・学校から", "topics": ["野球"]})
    assert "互いの職業を紹介し合う" not in t2


# --- 2026-10-29 追加: 挨拶の形・近況の中身・締め方、序盤の職業の自己言及 -------

from topic_plan import (CLOSING_TEXT, GREETING_TEXT, KINKYO_TEXT,  # noqa: E402
                        early_self_job)
from generate_dialogues import (cap_shift_marks, swap_stock_words,  # noqa: E402
                                _partner_name)

EXTRA = {"greeting_mix": {"はじめまして": 1.0, "名乗らない": 1.0},
         "kinkyo_mix": {"道のり": 1.0, "天気": 1.0},
         "closing_mix": {"締めない": 1.0, "短く": 1.0}}


def test_deck_without_extra_mix_keeps_the_old_plan_shape():
    deck = TopicDeck({"topics": {2: 1.0}}, POOL)
    for _ in range(5):
        assert set(deck.next_plan()) == {"opening", "topics"}


def test_deck_issues_greeting_closing_and_kinkyo_by_weight():
    deck = TopicDeck({**EXTRA, "opening_mix": {"近況から": 1.0, "話題から": 1.0}}, POOL)
    plans = [deck.next_plan() for _ in range(8)]
    assert Counter(p["greeting"] for p in plans) == {"はじめまして": 4, "名乗らない": 4}
    assert Counter(p["closing"] for p in plans) == {"締めない": 4, "短く": 4}
    # 近況の中身は「近況から」の対話にだけ付く
    assert all(("kinkyo" in p) == (p["opening"] == "近況から") for p in plans)
    assert Counter(p["kinkyo"] for p in plans if "kinkyo" in p) == {"道のり": 2, "天気": 2}


def test_deck_rejects_unknown_extra_labels():
    with pytest.raises(ValueError):
        TopicDeck({"greeting_mix": {"おはよう": 1.0}}, POOL)


def test_deck_add_used_counts_extra_labels():
    deck = TopicDeck(EXTRA, POOL)
    for _ in range(3):
        deck.add_used({"opening": "話題から", "topics": ["読書"], "greeting": "名乗らない",
                       "closing": "短く"})
    p = [deck.next_plan() for _ in range(3)]
    assert [x["greeting"] for x in p] == ["はじめまして"] * 3
    assert [x["closing"] for x in p] == ["締めない"] * 3


def test_stage1_text_uses_greeting_kinkyo_and_closing():
    plan = {"opening": "近況から", "topics": ["読書", "旅行"], "greeting": "よろしく",
            "kinkyo": "道のり", "closing": "締めない"}
    t = stage1_topic_text(plan)
    assert "「よろしくお願いします」と言い合い、それぞれ名前だけを名乗る" in t and KINKYO_TEXT["道のり"] in t
    assert "季節のことや最近あったこと" not in t
    assert "締めの挨拶は書かない" in t and "軽い着地" not in t
    t2 = stage1_topic_text({**plan, "closing": "短く"})
    assert "相手（speakers の2人目）がそれまでの話を受けたひと言" in t2
    assert "褒め合う" in t2


def test_stage1_text_without_extras_is_unchanged_in_wording():
    t = stage1_topic_text({"opening": "近況から", "topics": ["読書", "旅行"]})
    assert OPENING_TEXT["近況から"] in t and "軽い着地" in t


def test_stage2_block_names_the_partner_for_closing():
    plan = {"opening": "話題から", "topics": ["読書"], "greeting": "こんにちは",
            "closing": "話題に触れて"}
    b = stage2_topic_block(plan, "イノウエ", "オカダ")
    assert "軽い挨拶をして、それぞれ名前だけを名乗る" in b and "名乗りに職業や所属を付けない" in b
    assert "オカダが最後の話題に出たこと" in b


def test_build_prompt_drops_vary_closing_when_plan_has_closing():
    from generate_dialogues import S2Params
    sp = S2Params(vary_closing=True)
    with_c = build_dialogue_prompt(_meta({"opening": "話題から", "topics": ["読書"],
                                          "closing": "締めない"}), sp=sp)
    without = build_dialogue_prompt(_meta({"opening": "話題から", "topics": ["読書"]}), sp=sp)
    assert "締めの言い方は対話ごとに変えてください" not in with_c
    assert "締めの言い方は対話ごとに変えてください" in without
    assert "オカダ" in with_c and _partner_name(_meta(), 0) == "オカダ"


def _t(*pairs):
    return [{"speaker": s, "utterance": u} for s, u in pairs]


def test_early_self_job_catches_an_unprompted_intro():
    turns = _t(("イノウエ", "はじめまして。大学で研究をしております、イノウエです。"),
               ("オカダ", "オカダです。"))
    assert early_self_job(turns, "イノウエ") == 0


def test_early_self_job_allows_after_the_partner_brings_up_work():
    asked = _t(("イノウエ", "はじめまして。"), ("オカダ", "普段は何をされてるんですか？"),
               ("イノウエ", "大学で、えっと、AIの研究をしています。"))
    assert early_self_job(asked, "イノウエ") is None
    told = _t(("オカダ", "卸売の会社で営業をしています。"), ("イノウエ", "私は大学で研究を。"))
    assert early_self_job(told, "イノウエ") is None


def test_early_self_job_ignores_late_mentions_and_school_memories():
    late = _t(*[("オカダ" if i % 2 else "イノウエ", "そうですね。") for i in range(14)],
              ("イノウエ", "仕事でAIの研究をしているので。"))
    assert early_self_job(late, "イノウエ") is None
    school = _t(("イノウエ", "大学生のころは、よく山に登ってました。"))
    assert early_self_job(school, "イノウエ") is None


def test_cap_shift_marks_keeps_the_first_and_strips_the_rest():
    turns = _t(("A", "あ、そういえば、最近ランニングを始めたんです。"),
               ("B", "そうなんですね。ところで夜はゆっくり眠れてますか？"),
               ("A", "あ、そういえば、昔フルートを習っていました。"),
               ("B", "ところで、はい。"))
    assert cap_shift_marks(turns, 1) == 2
    assert turns[0]["utterance"].startswith("あ、そういえば、")
    assert turns[1]["utterance"] == "そうなんですね。夜はゆっくり眠れてますか？"
    assert turns[2]["utterance"] == "昔フルートを習っていました。"
    assert turns[3]["utterance"] == "ところで、はい。"      # 外すと短すぎるものは残す
    assert cap_shift_marks(_t(("A", "そういえばそうですね。")), 0) == 0


def test_swap_stock_words():
    turns = _t(("A", "景色を眺めるのが心地いいんですよ。"), ("B", "心地よさそうですね。"),
               ("A", "居心地がいいお店です。"), ("B", "不思議な心地良さがあります。"))
    assert swap_stock_words(turns) == 3
    assert [t["utterance"] for t in turns] == [
        "景色を眺めるのが気持ちいいんですよ。", "気持ちよさそうですね。",
        "居心地がいいお店です。", "不思議な気持ちよさがあります。"]


def test_short_closing_text_quotes_no_line():
    # 台詞（「そろそろ」）を書くと、締めがすべて同じ3ターンになった（2回目の試走）
    assert "「" not in CLOSING_TEXT["短く"]


def test_deck_never_pairs_work_and_school():
    pool = ["仕事の内容", "専攻", "転職", "部活", "読書", "旅行"]
    deck = TopicDeck({"topics": {3: 1.0}}, pool)
    for _ in range(20):
        groups = [topic_group(t) for t in deck.next_plan()["topics"]]
        assert not {"仕事", "学校"} <= set(groups), groups


def test_limit_echoes_replace_once():
    from generate_dialogues import limit_echoes
    turns = []
    for w in ("外食", "魚", "煮物", "肉", "野菜", "健康"):
        turns += [{"speaker": "A", "utterance": f"最近は{w}が多いです。"},
                  {"speaker": "B", "utterance": f"{w}ですか？"}]
    limit_echoes(turns, 1, replace_once=True)
    b = [t["utterance"] for t in turns if t["speaker"] == "B"]
    stock = [u for u in b if not u.endswith("ですか？") or u in ("そうなんですか？", "どんな感じなんですか？")]
    assert len(stock) == len(set(stock))           # 置き換え語はそれぞれ1回まで
    assert len(stock) <= 3                          # 使い切ったら聞き返しのまま残す
    assert sum(u.endswith("ですか？") and u not in stock for u in b) >= 3


def test_early_self_job_allows_reply_to_a_plain_job_statement():
    turns = _t(("オカダ", "はじめまして。"), ("イノウエ", "はじめまして。"),
               ("オカダ", "私はスーパーの店員をしています。"), ("イノウエ", "私は大学で研究をしています。"))
    assert early_self_job(turns, "イノウエ") is None


def test_partner_knows_job_before_being_told():
    from topic_plan import partner_knows_job
    leak = _t(("セキ", "編み物を始めました。"), ("イノウエ", "いいですね。"),
              ("セキ", "イノウエさんは、研究のお仕事を、されているそうですね。"))
    assert partner_knows_job(leak, "イノウエ") == 2
    told = _t(("イノウエ", "私は大学で研究をしています。"), ("セキ", "研究のお仕事ですか。"))
    assert partner_knows_job(told, "イノウエ") is None


def test_stage2_block_keeps_work_topics_on_the_partner():
    b = stage2_topic_block({"opening": "話題から", "topics": ["食べ歩き", "専攻"]},
                           "イノウエ", "ヒグチ")
    assert "「専攻」の話題はヒグチのことを中心に話し" in b
    assert "聞き役" not in stage2_topic_block({"opening": "話題から", "topics": ["読書"]}, "イノウエ")


def test_swap_stock_words_plain_form():
    turns = _t(("A", "心地よい静寂です。"))
    swap_stock_words(turns)
    assert turns[0]["utterance"] == "気持ちいい静寂です。"


def test_major_is_only_for_student_age():
    assert fits_age("専攻", 21) and not fits_age("専攻", 27)


# ── 2026-10-29 再評価の直し（名乗り方・聞き返しの畳み込み・指摘箇所の書き直し・主話者の締め）──

def test_greeting_text_fills_naming():
    from topic_plan import _greeting_text, naming_of
    t = _greeting_text({"greeting": "はじめまして", "naming": "片方だけ"})
    assert "どちらか1人だけが名前を名乗る" in t and "{naming}" not in t
    assert "照れくさい" not in _greeting_text({"greeting": "ひと言"})
    assert naming_of({"greeting": "名乗らない", "naming": "両方"}) == "なし"
    assert naming_of({"greeting": "よろしく"}) is None


def test_deck_assigns_naming_except_for_no_name_greeting():
    deck = TopicDeck({"greeting_mix": {"名乗らない": 0.5, "はじめまして": 0.5},
                      "naming_mix": {"両方": 0.5, "なし": 0.5}}, POOL)
    plans = [deck.next_plan() for _ in range(8)]
    for p in plans:
        assert ("naming" in p) == (p["greeting"] != "名乗らない")
    assert Counter(p["naming"] for p in plans if "naming" in p) == {"両方": 2, "なし": 2}


def _t(*pairs):
    return [{"speaker": s, "utterance": u} for s, u in pairs]


def test_apply_naming_drops_second_intro_and_keeps_addressed_names():
    from generate_dialogues import apply_naming
    meta = {"speakers": [{"name": "イノウエ"}, {"name": "シミズ"}]}
    turns = _t(("シミズ", "はじめまして。シミズです。"), ("イノウエ", "はじめまして。イノウエです。"),
               ("シミズ", "最近、釣りを始めました。"))
    assert apply_naming(turns, meta, "片方だけ") == 1
    assert turns[0]["utterance"] == "はじめまして。シミズです。"
    assert turns[1]["utterance"] == "はじめまして。"
    turns = _t(("シミズ", "はじめまして。シミズです。"), ("イノウエ", "イノウエです。"),
               ("シミズ", "イノウエさんは、どちらから？"))
    assert apply_naming(turns, meta, "なし") == 1          # 呼ばれたイノウエ・中身の残らない発話は残す
    assert turns[0]["utterance"] == "はじめまして。" and turns[1]["utterance"] == "イノウエです。"
    turns = _t(("イワタ", "私はイワタよ。よろしくね。"), ("イノウエ", "よろしくお願いします。"))
    assert apply_naming(turns, {"speakers": [{"name": "イワタ"}]}, "なし") == 1
    assert turns[0]["utterance"] == "よろしくね。"
    assert apply_naming(_t(("シミズ", "シミズです。どうも。")), meta, "両方") == 0


def test_fold_main_echoes_joins_the_partner_turns():
    from generate_dialogues import fold_main_echoes
    turns = _t(("オカダ", "最近カラオケにハマってるんです。"), ("イノウエ", "カラオケですか？"),
               ("オカダ", "はい。週に一度は行きます。"), ("イノウエ", "いいですね。"),
               ("オカダ", "演歌をよく歌うんです。"), ("イノウエ", "演歌ですか？"),
               ("オカダ", "ええ、昔の歌謡曲も歌います。"), ("イノウエ", "あー。"))
    out, n = fold_main_echoes(turns, "イノウエ", 1)
    assert n == 1 and len(out) == 6
    assert out[0]["utterance"] == "最近カラオケにハマってるんです。" and out[1]["utterance"] == "カラオケですか？"
    assert out[4]["utterance"] == "演歌をよく歌うんです。昔の歌謡曲も歌います。"
    assert fold_main_echoes(turns, "イノウエ", -1)[1] == 0
    # 沈黙を挟む箇所は畳まない
    turns[6]["pause_before_sec"] = 3.0
    assert fold_main_echoes(turns, "イノウエ", 0)[1] == 1


def test_flag_turns_counts_calm_eval_and_time():
    from generate_dialogues import S2Params, flag_turns
    sp = S2Params(flag_rewrite=True, calm_word_cap=1, main_eval_cap=0, time_close_rewrite=True)
    meta = {"topic_plan": {"opening": "話題から", "topics": ["読書"], "closing": "短く"}}
    turns = _t(("オカダ", "静かな場所が好きです。"), ("イノウエ", "落ち着きますね。"),
               ("オカダ", "本を読みます。"), ("イノウエ", "読書の時間は大切ですね。"),
               ("オカダ", "そろそろ時間ですね。"), ("イノウエ", "はい。"))
    f = flag_turns(turns, meta, "イノウエ", sp)
    assert f == {1: ["calm"], 3: ["eval"], 4: ["time"]}
    assert 1 not in flag_turns(turns, meta, "イノウエ", sp, skip={1})


def test_validate_rewrites_rejects_leftovers_and_new_questions():
    from generate_dialogues import validate_rewrites
    turns = _t(("オカダ", "静かな場所が好きです。"), ("イノウエ", "読書の時間は大切ですね。"))
    flags = {0: ["calm"], 1: ["eval"]}
    ok, rej = validate_rewrites(turns, flags, [{"i": 0, "text": "落ち着いた場所が好きです。"},
                                               {"i": 1, "text": "どんな本ですか？"}])
    assert ok == {} and len(rej) == 2
    ok, _ = validate_rewrites(turns, flags, [{"i": 0, "text": "川沿いの道が好きです。"},
                                             {"i": 1, "text": "私も寝る前に少し読みます。"},
                                             {"i": 5, "text": "x"}])
    assert ok == {0: "川沿いの道が好きです。", 1: "私も寝る前に少し読みます。"}


def test_main_close_start_ignores_replies():
    from generate_dialogues import main_close_start
    turns = _t(("オカダ", "今度、おすすめの店を教えますね。"), ("イノウエ", "はい。よろしくお願いします。"),
               ("オカダ", "今日は楽しかったです。"), ("イノウエ", "それでは、失礼します。"))
    assert main_close_start(turns, "イノウエ") is None
    turns = _t(("オカダ", "最近の流行りの曲を練習してます。"), ("イノウエ", "ありがとうございます。失礼します。"))
    assert main_close_start(turns, "イノウエ") == 1


def test_fold_main_reactions_only_where_it_makes_a_long_turn():
    from generate_dialogues import fold_main_reactions
    turns = _t(("オカダ", "去年から、近所の川沿いで朝に釣りをしています。"), ("イノウエ", "あー。"),
               ("オカダ", "はい、始めたきっかけは父の道具です。"), ("イノウエ", "そうなんですね。"),
               ("オカダ", "短いです。"), ("イノウエ", "へえ。"), ("オカダ", "はい？"),
               ("イノウエ", "釣りですか？"), ("オカダ", "どこで釣るんですか？"), ("イノウエ", "はい。"),
               ("オカダ", "海です。"))
    out, n = fold_main_reactions(turns, "イノウエ", 3)
    assert n == 1                                   # 31字以上になる最初の1か所だけ（問いへの返事は畳まない）
    assert out[0]["utterance"] == "去年から、近所の川沿いで朝に釣りをしています。始めたきっかけは父の道具です。"
    assert fold_main_reactions(turns, "イノウエ", 0)[1] == 0


def test_flag_turns_always_flags_kokochi_and_skips_time_by_default():
    from generate_dialogues import S2Params, flag_turns
    sp = S2Params(flag_rewrite=True, calm_word_cap=2, main_eval_cap=-1)
    meta = {"topic_plan": {"opening": "話題から", "topics": ["読書"], "closing": "短く"}}
    turns = _t(("オカダ", "心地いい場所です。"), ("イノウエ", "居心地がいいですね。"),
               ("オカダ", "そろそろ時間ですね。"))
    assert flag_turns(turns, meta, "イノウエ", sp) == {0: ["calm"]}


def test_unprompted_job_statement_anywhere():
    from topic_plan import unprompted_job_statement
    turns = _t(("オカダ", "最近ジムに通い始めました。"), ("イノウエ", "いいですね。"),
               ("オカダ", "体を動かすとすっきりします。"), ("イノウエ", "私は大学でロボットの研究をしてます。"))
    assert unprompted_job_statement(turns, "イノウエ") == 3
    turns[2]["utterance"] = "お仕事は何をされてるんですか？"
    assert unprompted_job_statement(turns, "イノウエ") is None
    assert unprompted_job_statement(_t(("オカダ", "座りっぱなしです。"),
                                       ("イノウエ", "私も研究室にいると座りっぱなしです。")), "イノウエ") is None
