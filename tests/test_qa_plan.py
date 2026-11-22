"""質問と答えの対話（metadata.topic_plan の style: qa。qa_plan.py）と、
主話者のペルソナを対話ごとに引く仕組み（metadata.main_speaker.persona_files。persona.py）のテスト。

persona_file（1つだけ）の config は、プロンプトもメタデータも従来と一字も変わらない。
"""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_dialogues import ALL_GATES, GATE_NAMES, build_dialogue_prompt  # noqa: E402
from generate_metadata import DiversityTracker, attach_persona, build_user_prompt  # noqa: E402
from main import load_config                                                # noqa: E402
from persona import (META_KEY, PersonaPicker, load_persona, persona_files,  # noqa: E402
                     persona_of)
from qa_plan import (QA_KIND_TEXT, QADeck, qa_counts, qa_shape_problem,     # noqa: E402
                     stage1_qa_text, stage2_qa_block)
from topic_plan import make_deck, stage1_topic_text, stage2_topic_block, topic_plan_of  # noqa: E402

MS = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "大学の研究者"}
POOL = ["読書", "ランニング", "筋トレ", "ペット", "家庭菜園", "旅行", "仕事の内容", "転職",
        "出身地", "天気の話", "睡眠", "料理", "ラーメン", "家族の話", "孫の話", "ちょっとした失敗談",
        "英語の勉強", "家電選び", "通学"]
QA = {"style": "qa", "seed": 0}
P1 = "personality: 落ち着いていて面倒見がよい。\n"
P2 = "personality: せっかちで早口。\nspeech_style: 結論から言う。\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _meta(plan=None, persona=None):
    m = {"id": "x", "scenario": "オカダがイノウエにいろいろ聞く",
         "speakers": [{"name": "イノウエ", "gender": "male", "role": "大学の研究者", "personality": "穏やか"},
                      {"name": "オカダ", "gender": "female", "role": "スーパーの店員",
                       "personality": "控えめ", "age_range": "40代"}],
         "target_turns": 24, "main_speaker_index": 0}
    if plan:
        m["topic_plan"] = plan
    if persona:
        m[META_KEY] = persona
    return m


# --- persona_files（対話ごとに1つ引く）--------------------------------------

def test_persona_files_list_and_weights(tmp_path):
    a, b = _write(tmp_path, "a.yaml", P1), _write(tmp_path, "b.yaml", P2)
    assert persona_files({"persona_files": [a, b]}) == {a: 1.0, b: 1.0}
    assert persona_files({"persona_files": {a: 2, b: 0}}) == {a: 2.0}
    assert persona_files({"persona_file": a}) == {}


def test_persona_file_and_files_together_is_an_error(tmp_path):
    a = _write(tmp_path, "a.yaml", P1)
    with pytest.raises(ValueError):
        persona_files({"persona_file": a, "persona_files": [a]})
    # 継承元の persona_file は null で消せる
    assert persona_files({"persona_file": None, "persona_files": [a]}) == {a: 1.0}
    # persona_files のときは config 全体のペルソナは無し（対話ごとに引く）
    assert load_persona({"persona_file": None, "persona_files": [a]}) == ""


def test_picker_is_random_but_balanced_by_weight(tmp_path):
    a, b, c = (_write(tmp_path, f"{n}.yaml", P1) for n in "abc")
    pk = PersonaPicker({"persona_files": {a: 1, b: 1, c: 2}})
    picks = [pk.next() for _ in range(40)]
    assert Counter(picks) == {a: 10, b: 10, c: 20}
    # 同数のあいだの選び方は乱数（毎回同じ順の巡回ではない）
    firsts = {tuple(PersonaPicker({"persona_files": [a, b, c], "persona_seed": s}).next()
                    for _ in range(1)) for s in range(20)}
    assert len(firsts) > 1


def test_picker_rejects_an_empty_persona(tmp_path):
    a, e = _write(tmp_path, "a.yaml", P1), _write(tmp_path, "e.yaml", "personality:\n")
    with pytest.raises(ValueError):
        PersonaPicker({"persona_files": [a, e]})


def test_tracker_counts_existing_metadata(tmp_path):
    a, b = _write(tmp_path, "a.yaml", P1), _write(tmp_path, "b.yaml", P2)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=dict(MS, persona_files=[a, b]))
    for _ in range(3):
        tr.add(_meta(persona=a))
    out = []
    for _ in range(3):
        tr.next_persona(out)
    assert out == [b, b, b]


def test_stage1_uses_the_picked_persona_and_reports_it(tmp_path):
    a, b = _write(tmp_path, "a.yaml", P1), _write(tmp_path, "b.yaml", P2)
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=dict(MS, persona_files=[a, b]))
    seen = {}
    for _ in range(4):
        out = []
        p = build_user_prompt(1, 16, 24, tr, num_speakers=2, persona_out=out)
        assert len(out) == 1
        seen.setdefault(out[0], p)
    assert set(seen) == {a, b}
    assert "落ち着いていて面倒見がよい" in seen[a] and "せっかち" not in seen[a]
    assert "せっかちで早口" in seen[b]
    metas = [_meta()]
    attach_persona(metas, b)
    assert metas[0][META_KEY] == b


def test_stage1_single_persona_file_is_unchanged(tmp_path):
    a = _write(tmp_path, "a.yaml", P1)
    out = []
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=dict(MS, persona_file=a))
    p1 = build_user_prompt(1, 16, 24, tr, num_speakers=2, persona_out=out)
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=dict(MS, persona_file=a))
    p2 = build_user_prompt(1, 16, 24, tr, num_speakers=2)
    assert p1 == p2 and out == []          # 1つだけのときはメタデータに書かない


def test_stage2_reads_the_persona_from_metadata(tmp_path):
    a, b = _write(tmp_path, "a.yaml", P1), _write(tmp_path, "b.yaml", P2)
    pa = build_dialogue_prompt(_meta(persona=a), persona="- 性格・人柄: 既定")
    pb = build_dialogue_prompt(_meta(persona=b), persona="- 性格・人柄: 既定")
    assert "落ち着いていて面倒見がよい" in pa and "既定" not in pa
    assert "せっかちで早口" in pb
    assert META_KEY not in pa and str(tmp_path) not in pa      # ファイル名は台本に出さない
    # キーが無ければ config 全体のペルソナ（従来どおり）
    assert build_dialogue_prompt(_meta(), persona="- 性格・人柄: 既定") == \
        build_dialogue_prompt(_meta(), persona="- 性格・人柄: 既定")
    assert "既定" in build_dialogue_prompt(_meta(), persona="- 性格・人柄: 既定")
    assert persona_of(_meta(), "dflt") == "dflt"


def test_shipped_qa_personas_load():
    for i in (1, 2, 3):
        assert load_persona({"persona_file": str(ROOT / f"personas/inoue_qa_{i}.yaml")})


# --- QADeck（質問の組と順番）------------------------------------------------

def test_make_deck_picks_qa_deck():
    assert isinstance(make_deck(QA, POOL), QADeck)
    with pytest.raises(ValueError):
        make_deck({"style": "nope"}, POOL)


def test_deck_follows_count_kind_and_askback_weights():
    deck = QADeck(dict(QA, questions={2: 1.0, 4: 1.0}, ask_back={0: 1.0, 2: 1.0}), POOL)
    plans = [deck.next_plan() for _ in range(40)]
    assert Counter(len(p["questions"]) for p in plans) == {2: 20, 4: 20}
    assert Counter(p["ask_back"] for p in plans) == {0: 20, 2: 20}
    kinds = Counter(q["kind"] for p in plans for q in p["questions"])
    assert set(kinds) == set(QA_KIND_TEXT)
    for p in plans:
        assert p["style"] == "qa" and p["topics"] == [q["topic"] for q in p["questions"]]
        assert topic_plan_of({"topic_plan": p}) is p
        ks = [q["kind"] for q in p["questions"]]
        assert len(set(ks)) == len(ks)                        # 1対話の中で種類を重ねない（5種まで）


def test_deck_skips_unfit_kind_topic_pairs():
    deck = QADeck(dict(QA, questions={4: 1.0}), POOL)
    for _ in range(60):
        p = deck.next_plan({"age": 70})
        for q in p["questions"]:
            if q["kind"] == "本人のこと":
                # 主話者の家族・出身・経歴を作らせない
                assert q["topic"] not in ("家族の話", "孫の話", "転職")
            if q["topic"] == "ちょっとした失敗談":
                assert q["kind"] == "本人のこと"
            if q["kind"] in ("知識", "おすすめ", "相談"):
                assert q["topic"] != "通学"                      # 70歳の相手の状況に合わない
            if q["kind"] == "知識":
                assert q["topic"] != "出身地"                    # 本人への質問に化ける
        assert p["questions"][0]["topic"] not in ("仕事の内容", "転職")


def test_deck_always_has_greeting_and_closing_and_kinkyo_only_for_chat_opening():
    deck = QADeck(QA, POOL)
    for _ in range(30):
        p = deck.next_plan()
        assert p["greeting"] and p["closing"]
        assert ("kinkyo" in p) == (p["opening"] == "雑談から")
        assert ("naming" in p) == (p["greeting"] != "名乗らない")


def test_deck_card_hobby_goes_only_to_partner_side_kinds():
    deck = QADeck(dict(QA, card_hobby=1.0), POOL)
    n = 0
    for _ in range(40):
        p = deck.next_plan({"hobbies": ["和菓子作り"]})
        for q in p["questions"]:
            if q["topic"] == "和菓子作り":
                assert q["kind"] in ("相談", "意見")
                n += 1
    assert n > 0


def test_deck_unsure_is_never_about_the_main_speaker():
    deck = QADeck(dict(QA, unsure=1.0), POOL)
    for _ in range(30):
        p = deck.next_plan()
        if "unsure" in p:
            assert p["questions"][p["unsure"]]["kind"] in ("知識", "おすすめ")
            assert p["questions"][p["unsure"]]["topic"] not in ("家族の話", "孫の話", "出身地")


def test_deck_add_used_keeps_the_ratio():
    a = QADeck(dict(QA, ask_back={0: 1.0, 1: 1.0}), POOL)
    old = [a.next_plan() for _ in range(5)]
    b = QADeck(dict(QA, ask_back={0: 1.0, 1: 1.0}), POOL)
    for p in old:
        b.add_used(p)
    new = [b.next_plan() for _ in range(5)]
    assert Counter(p["ask_back"] for p in old + new) == {0: 5, 1: 5}


def test_deck_rejects_unknown_labels():
    with pytest.raises(ValueError):
        QADeck(dict(QA, kind_mix={"雑学": 1.0}), POOL)
    with pytest.raises(ValueError):
        QADeck(dict(QA, qa_opening_mix={"話題から": 1.0}), POOL)


# --- 文面 -------------------------------------------------------------------

def _plan(**kw):
    p = {"style": "qa", "opening": "前置きして", "topics": ["睡眠", "旅行"],
         "questions": [{"kind": "相談", "topic": "睡眠"}, {"kind": "本人のこと", "topic": "旅行"}],
         "ask_back": 1, "greeting": "よろしく", "naming": "片方だけ", "closing": "短く"}
    p.update(kw)
    return p


def test_stage1_text_lists_questions_in_order():
    t = stage1_topic_text(_plan(unsure=0))   # 文面だけの確認（unsure の種類の制限は QADeck 側）
    assert t == stage1_qa_text(_plan(unsure=0))
    assert t.index("「睡眠」について") < t.index("「旅行」について")
    assert "1番目の質問は、1人目があまり詳しくない" in t
    assert "conversation_flow は4項目" in t
    assert "対話全体で1回まで" in t
    assert "genre には「睡眠」" in t
    assert "2人目は1人目の職業・趣味・経歴・家族を知りません" in t
    assert "1人目の人物設定に無い趣味・習い事・特技" in t
    assert "1人目の家族・住まい・出身は、人物設定に書いてあることだけ" in t


def test_stage2_block_uses_names_and_settings():
    b = stage2_topic_block(_plan(ask_back=0), "イノウエ", "オカダ")
    assert b == stage2_qa_block(_plan(ask_back=0), "イノウエ", "オカダ")
    assert b.startswith("\n\n【質問と答えの流れ】この対話は、オカダがイノウエに質問し")
    assert "イノウエからは質問しないでください" in b
    assert "オカダが自分の困りごとや迷いを話して" in b
    assert "あまり詳しくない" not in b
    assert "どちらか1人だけが名前を名乗る" in b
    assert "対話全体で2回までにし" in b and "いま見ているように話さず" in b
    assert "{" not in b and "}" not in b                                  # 埋め忘れが無い
    assert "オカダはイノウエの仕事・趣味・経歴・家族をまだ何も知りません" in b and "性格を表す言葉" in b
    assert "具体的なやり方（手順・道具・目安）を少なくとも1つ" in b
    assert "人物設定に無い趣味・習い事・特技を自分がしていることにはしない" in b
    assert "無理にこじつけてつながなくてかまいません" in b and "季節や最近の天気" in b
    assert "「試行錯誤」「直感」" in b and "家族・住まい・出身は、人物設定に書いてあることだけ" in b
    assert "お役に立てましたか" in b
    assert "類推して具体的に答えてください" in b and "対話全体で1回までにしてください。" in b
    assert "ぼかすか、分からないと言って" not in b
    u = stage2_qa_block(_plan(unsure=0), "イノウエ", "オカダ")
    assert "あまり詳しくない質問への答えがその1回です" in u


def test_stage2_prompt_has_the_qa_block():
    p = build_dialogue_prompt(_meta(_plan()))
    assert "【質問と答えの流れ】" in p and '"topic_plan"' not in p


# --- ゲート（qa_shape）------------------------------------------------------

def _turns(rows):
    return [{"speaker": s, "utterance": u} for s, u in rows]


GOOD = _turns([
    ("オカダ", "あの、最近なかなか寝つけないんですけど、どうされてます？"),
    ("イノウエ", "私は、寝る前の一時間はスマホを見ないようにしてます。"),
    ("オカダ", "へえ、それだけで変わるものですか？"),
    ("イノウエ", "まあ、最初の一週間くらいは、正直あんまり変わらなかったですね。"),
    ("オカダ", "そうなんですね。"),
    ("イノウエ", "でも、その、続けてたら、布団に入ってからが早くなった気がします。"),
    ("オカダ", "旅行とかって、よく行かれるんですか？"),
    ("イノウエ", "年に二回くらいですかね、近場の温泉に行くことが多いです。"),
])


def test_qa_shape_passes_a_good_dialogue():
    assert "qa_shape" in GATE_NAMES and ALL_GATES["qa_shape"]
    c = qa_counts(GOOD, "イノウエ")
    assert c["partner_q"] == 3 and c["main_q"] == 0 and c["share"] > 0.5
    assert qa_shape_problem(GOOD, "イノウエ", _plan()) is None


def test_qa_shape_flags_too_few_partner_questions():
    t = [dict(x) for x in GOOD]
    t[6]["utterance"] = "旅行は、私も好きでよく行きます。"
    t[2]["utterance"] = "へえ、それだけで変わるものなんですね。"
    assert "相手の質問" in qa_shape_problem(t, "イノウエ", _plan())


def test_qa_shape_flags_a_questioning_main_speaker():
    t = [dict(x) for x in GOOD]
    t[1]["utterance"] = "寝る前って、だいたい何をされてることが多いですか？"
    t[3]["utterance"] = "スマホは、寝室まで持っていかれてるんですか？"
    assert "主話者の問いかけ" in qa_shape_problem(t, "イノウエ", _plan(ask_back=0))
    # 短い聞き返し（実質8字未満）は数えない
    t = [dict(x) for x in GOOD]
    t[1]["utterance"] = "え、寝つけない？"
    assert qa_counts(t, "イノウエ")["main_q"] == 0


def test_qa_shape_flags_a_listening_main_speaker():
    t = _turns([("オカダ", "最近ずっと寝つけなくて、夜中に何度も目が覚めちゃうんですけど、どうしたらいいですか？"),
                ("イノウエ", "そうなんですね。"),
                ("オカダ", "仕事も忙しくて、帰るのが毎日遅くて、寝る時間もばらばらなんですけど、何かありますか？"),
                ("イノウエ", "スマホを置くといいですよ。"),
                ("オカダ", "あと、休みの日は昼まで寝ちゃうので、それもよくないのかなと思ってて、どうですか？"),
                ("イノウエ", "少し早めに起きましょうか。")])
    assert "発話量" in qa_shape_problem(t, "イノウエ", _plan(questions=[{"kind": "相談", "topic": "睡眠"}]))


def test_qa_shape_counts_consultation_statements_as_questions():
    t = _turns([("オカダ", "方言を直すべきかなって、迷ってるんです。"),
                ("イノウエ", "無理に消さなくても、自然な形で残していいと思いますよ。")])
    assert qa_counts(t, "イノウエ")["partner_q"] == 1
    # 主話者の側は「？」だけを数える
    t = _turns([("イノウエ", "私も、どうしたらいいか迷ってるんです。")])
    assert qa_counts(t, "イノウエ")["main_q"] == 0


def test_qa_shape_flags_too_many_vague_answers():
    t = [dict(x) for x in GOOD]
    t[1]["utterance"] = "うーん、寝つきのことは、あまり詳しくないんですけど。"
    t[3]["utterance"] = "正直、効いたかどうかは、はっきり覚えてないです。"
    assert qa_counts(t, "イノウエ")["vague"] == 2
    assert qa_shape_problem(t, "イノウエ", _plan()) is None            # 2回までは通す（1回分の余裕）
    t[5]["utterance"] = "まあ、そのへんは人によると思います。"
    assert "ぼかした答え3回" in qa_shape_problem(t, "イノウエ", _plan())


def test_qa_shape_ignores_chat_plans():
    assert qa_shape_problem(GOOD, "イノウエ", {"opening": "話題から", "topics": ["睡眠"]}) is None


# --- config -----------------------------------------------------------------

def test_qa_config_loads_and_builds_a_deck():
    cfg = load_config(str(ROOT / "configs/qa_v1.yaml"))
    md = cfg["metadata"]
    assert md["main_speaker"]["persona_file"] is None
    assert list(persona_files(md["main_speaker"])) == [f"personas/inoue_ft_v6_00{i}.yaml" for i in (2, 3, 4, 5)]
    assert isinstance(make_deck(md["topic_plan"], POOL), QADeck)
    assert cfg["dialogue"]["prompt_file"] == "prompts/stage2_qa.yaml"
    assert cfg["dialogue"]["enrich"]["question_ratio_cap"] == 0
    deck = make_deck(md["topic_plan"], POOL)
    assert "天気" not in deck.extra_mix["kinkyo"] and deck.topics_mix[4] == 0.15


def test_deck_mixes_are_not_locked_together():
    """入り方・聞き返しの回数・質問数が組で固定されない（比率の組が同じでも周期がそろわない）。"""
    import math
    from collections import Counter
    from qa_plan import QADeck
    deck = QADeck({"style": "qa", "seed": 3,
                   "qa_opening_mix": {"すぐ質問": 0.30, "前置きして": 0.45, "雑談から": 0.25},
                   "ask_back": {0: 0.25, 1: 0.45, 2: 0.30}}, ["料理", "旅行", "映画", "読書", "散歩", "音楽"])
    plans = [deck.next_plan({"age": 40}) for _ in range(1500)]

    def cramers_v(a, b):
        n = len(a); ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
        chi = sum((cab[(x, y)] - ca[x] * cb[y] / n) ** 2 / (ca[x] * cb[y] / n) for x in ca for y in cb)
        return math.sqrt(chi / (n * (min(len(ca), len(cb)) - 1)))
    assert cramers_v([p["opening"] for p in plans], [p["ask_back"] for p in plans]) < 0.25
    shares = Counter(p["ask_back"] for p in plans)
    assert abs(shares[1] / 1500 - 0.45) < 0.01


def test_no_family_topic_for_recommendation():
    from qa_plan import QADeck
    deck = QADeck({"style": "qa", "seed": 0}, ["兄弟姉妹", "料理"])
    assert not deck._fits("おすすめ", "兄弟姉妹", 40)


def test_no_habit_marks_activities_missing_from_persona():
    from qa_plan import mark_habits
    persona = "家族で大文字山に登る。京都サンガのスタジアムへ応援に行く。子どもと粘土でコマ撮りをする。"
    plan = {"questions": [{"kind": "本人のこと", "topic": "サウナ"},
                          {"kind": "本人のこと", "topic": "登山"},
                          {"kind": "おすすめ", "topic": "応援しているチーム"},
                          {"kind": "相談", "topic": "手芸"},
                          {"kind": "本人のこと", "topic": "睡眠"},
                          {"kind": "知識", "topic": "朝ごはん"}]}
    got = {q["topic"] for q in mark_habits(plan, persona)["questions"] if q.get("no_habit")}
    assert got == {"サウナ", "手芸"}
    # ペルソナが無ければ何も付けない
    plan2 = {"questions": [{"kind": "本人のこと", "topic": "サウナ"}]}
    assert not mark_habits(plan2, "")["questions"][0].get("no_habit")


def test_no_habit_note_reaches_stage1_and_stage2():
    from qa_plan import stage1_qa_text, stage2_qa_block
    plan = {"style": "qa", "opening": "すぐ質問", "ask_back": 1, "topics": ["サウナ", "読書"],
            "questions": [{"kind": "本人のこと", "topic": "サウナ", "no_habit": True},
                          {"kind": "意見", "topic": "読書"}]}
    s1, s2 = stage1_qa_text(plan), stage2_qa_block(plan, "イノウエ", "サトウ")
    assert "1人目は「サウナ」をふだんはしていません" in s1
    assert "イノウエは「サウナ」をふだんはしていません" in s2
    assert "「読書」をふだんは" not in s2
    assert "数回" not in s2


def test_deck_marks_habits_with_persona():
    from qa_plan import QADeck
    deck = QADeck({"style": "qa", "seed": 1, "kind_mix": {"本人のこと": 1.0}, "questions": {2: 1.0}},
                  ["サウナ", "キャンプ", "睡眠", "方言"])
    plans = [deck.next_plan({"age": 40}, persona="九州出身で京都に住む。") for _ in range(6)]
    for p in plans:
        for q in p["questions"]:
            assert bool(q.get("no_habit")) == (q["topic"] in {"サウナ", "キャンプ"})


def test_opinion_skips_family_and_knowledge_text_has_no_shikumi():
    from qa_plan import QADeck, QA_KIND_TEXT
    deck = QADeck({"style": "qa", "seed": 0}, ["子どもの成長", "孫の話", "料理"])
    assert not deck._fits("意見", "子どもの成長", 40)
    assert not deck._fits("意見", "孫の話", 70)
    assert "仕組み" not in QA_KIND_TEXT["知識"]


def test_stage2_block_new_rules():
    from qa_plan import stage2_qa_block
    plan = {"style": "qa", "opening": "すぐ質問", "ask_back": 0, "topics": ["カメラ", "読書"], "unsure": 0,
            "questions": [{"kind": "知識", "topic": "カメラ"}, {"kind": "相談", "topic": "読書"}]}
    s = stage2_qa_block(plan, "イノウエ", "サトウ")
    assert "目標の立て方や続け方を聞かれたときだけ" in s
    assert "他に気になることは" in s and "お役に立てれば" in s
    assert "決まった言い方にしないでください" in s


def test_main_questions_count_sentences_not_echo_or_suggestion():
    """主話者の短い聞き返し＋答え・提案の形は、主話者の問いかけに数えない。"""
    from qa_plan import qa_counts
    turns = [{"speaker": "イノウエ", "utterance": "筋トレ、ですか？うーん、ジムには一度だけ行ったことがあります。"},
             {"speaker": "イノウエ", "utterance": "まずは好きな食材を、一つだけ決めて、それを回してはどうですか？"},
             {"speaker": "イノウエ", "utterance": "時間帯を早朝か夕方に、変えてみるのは、どうでしょうか？"},
             {"speaker": "イノウエ", "utterance": "そうなんですね。サトウさんは、どうやって選んでるんですか？"}]
    assert qa_counts(turns, "イノウエ")["main_q"] == 1
