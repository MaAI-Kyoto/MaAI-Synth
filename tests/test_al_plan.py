"""傾聴の対話（metadata.topic_plan の style: al。al_plan.py）のテスト。"""

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from al_plan import (AL_KIND_TEXT, ALDeck, al_counts, al_shape_problem,     # noqa: E402
                     is_reecho, stage1_al_text, stage2_al_block)
from generate_dialogues import ALL_GATES, GATE_NAMES, build_dialogue_prompt  # noqa: E402
from generate_metadata import load_scenario                                 # noqa: E402
from main import load_config                                                # noqa: E402
from persona import persona_files                                           # noqa: E402
from topic_plan import (make_deck, stage1_topic_text, stage2_topic_block,   # noqa: E402
                        topic_group, topic_plan_of)

POOL = ["読書", "ランニング", "筋トレ", "ペット", "家庭菜園", "旅行", "仕事の内容", "天気の話",
        "睡眠", "料理", "ラーメン", "家族の話", "孫の話", "子どもの成長", "子育て", "部活",
        "学生生活", "ちょっとした失敗談", "若い頃の話", "出身地", "温泉", "釣り", "習い事", "映画"]
AL = {"style": "al", "seed": 0}
PERSONA = "background: 休日は家族でハイキングに出かけ、スタジアムでサッカーを応援する。\n"


def _plan(**kw):
    p = {"style": "al", "opening": "相手から", "topics": ["旅行", "ペット"],
         "episodes": [{"kind": "最近あったこと", "topic": "旅行"},
                      {"kind": "続けていること", "topic": "ペット"}],
         "koekake": "掘り下げ", "self_talk": 1, "greeting": "はじめまして", "naming": "片方だけ",
         "closing": "話題に触れて"}
    p.update(kw)
    return p


def _meta(plan=None):
    m = {"id": "x", "scenario": "オカダが最近の旅行の話をし、イノウエが聞く",
         "speakers": [{"name": "イノウエ", "gender": "male", "role": "大学の研究者", "personality": "穏やか"},
                      {"name": "オカダ", "gender": "female", "role": "スーパーの店員",
                       "personality": "話し好き", "age_range": "40代"}],
         "target_turns": 24, "main_speaker_index": 0}
    if plan:
        m["topic_plan"] = plan
    return m


def _card(age=45, marital="既婚 (子供あり)", hobbies=None):
    return {"age": age, "marital": marital, "hobbies": hobbies or []}


# --- 配り方（ALDeck）--------------------------------------------------------

def test_make_deck_picks_al_deck():
    assert isinstance(make_deck(AL, POOL), ALDeck)
    with pytest.raises(ValueError, match="al"):
        make_deck({"style": "xx"}, POOL)


def test_deck_follows_the_weights():
    deck = ALDeck(dict(AL, episodes={1: 1, 2: 1}, koekake_mix={"なし": 1, "掘り下げ": 1},
                       self_talk={0: 1, 2: 1}), POOL)
    plans = [deck.next_plan(_card()) for _ in range(200)]
    n = Counter(len(p["episodes"]) for p in plans)
    assert abs(n[1] - 100) <= 2 and abs(n[2] - 100) <= 2
    assert set(Counter(p["koekake"] for p in plans)) == {"なし", "掘り下げ"}
    st = Counter(p["self_talk"] for p in plans)
    assert abs(st[0] - 100) <= 2 and abs(st[2] - 100) <= 2
    op = Counter(p["opening"] for p in plans)
    assert abs(op["聞き手から"] - 80) <= 3 and abs(op["相手から"] - 70) <= 3


def test_next_topic_koekake_needs_two_episodes_and_self_story_needs_self_talk():
    deck = ALDeck(dict(AL, koekake_mix={"次の話へ": 1, "自分の話から": 1}, self_talk={0: 1}), POOL)
    for _ in range(100):
        p = deck.next_plan(_card())
        if p["koekake"] == "次の話へ":
            assert len(p["episodes"]) >= 2 and 1 <= p["koekake_at"] < len(p["episodes"])
        else:
            assert "koekake_at" not in p
        if len(p["episodes"]) == 1:
            assert p["koekake"] != "次の話へ"
        if p["koekake"] == "自分の話から":
            assert p["self_talk"] >= 1


def test_deck_skips_unfit_kind_topic_pairs():
    deck = ALDeck(AL, POOL)
    for _ in range(300):
        p = deck.next_plan(_card(age=72, marital="既婚"))
        for e in p["episodes"]:
            g = topic_group(e["topic"])
            if e["kind"] != "思い出":
                assert g != "思い出"
                assert e["topic"] not in ("部活", "学生生活")      # 72歳の今の話にしない
            if e["kind"] == "続けていること":
                assert g not in ("天気", "家族", "出身地")
            assert e["topic"] not in ("孫の話", "子どもの成長", "子育て")   # 子供なし
    assert ALDeck._fits("思い出", "部活", _card(age=72))
    assert ALDeck._fits("最近あったこと", "孫の話", _card(age=72, marital="既婚 (子供あり)"))
    assert not ALDeck._fits("最近あったこと", "孫の話", _card(age=40, marital="既婚 (子供あり)"))
    assert ALDeck._fits("気がかり", "子育て", None)                   # カードが無ければ年齢・家族は見ない


def test_kinds_take_only_topics_that_make_sense():
    # ★2026-11-07 試走6の「気がかり:カレー」「気がかり:好きな食べ物」。気がかりは一覧の話題だけ、
    #   楽しみ・続けていること・思い出は合わない話題を外す。
    assert not ALDeck._fits("気がかり", "カレー", None)
    assert not ALDeck._fits("気がかり", "好きな食べ物", None)
    assert not ALDeck._fits("気がかり", "出身地", None)
    assert ALDeck._fits("気がかり", "暑さ対策", None) and ALDeck._fits("気がかり", "お土産", None)
    assert not ALDeck._fits("楽しみ", "花粉症", None) and not ALDeck._fits("楽しみ", "職業のあるある", None)
    assert ALDeck._fits("楽しみ", "温泉", None)
    assert not ALDeck._fits("続けていること", "誕生日", None)
    assert not ALDeck._fits("思い出", "連休の予定", None)
    assert ALDeck._fits("最近あったこと", "天気の話", None)
    sc = load_scenario("scenarios/al_v1.yaml")
    deck = ALDeck(AL, sc.genres)
    seen = Counter()
    for _ in range(400):
        p = deck.next_plan(_card(age=45))
        for e in p["episodes"]:
            assert ALDeck._fits(e["kind"], e["topic"], _card(age=45)), e
            seen[e["kind"]] += 1
    assert seen["気がかり"] > 50


def test_card_hobby_goes_only_to_hobby_kinds():
    deck = ALDeck(dict(AL, card_hobby=1.0), POOL)
    hits = 0
    for _ in range(100):
        p = deck.next_plan(_card(hobbies=["盆栽"]))
        for e in p["episodes"]:
            if e["topic"] == "盆栽":
                hits += 1
                assert e["kind"] in ("最近あったこと", "続けていること", "楽しみ")
    assert hits > 50


def test_add_used_keeps_the_ratio():
    a, b = ALDeck(AL, POOL), ALDeck(AL, POOL)
    for _ in range(30):
        b.add_used(a.next_plan(_card()))
    assert b.opening_issued == a.opening_issued and b.koekake_issued == a.koekake_issued
    assert b.self_issued == a.self_issued and b.kind_issued == a.kind_issued
    b.add_used({"style": "qa", "opening": "相手から"})                  # 別の型は数えない
    assert b.opening_issued == a.opening_issued


def test_deck_rejects_unknown_labels():
    for key in ("al_opening_mix", "kind_mix", "koekake_mix"):
        with pytest.raises(ValueError, match=key):
            ALDeck(dict(AL, **{key: {"ない": 1}}), POOL)


def test_no_habit_marks_activities_missing_from_persona():
    deck = ALDeck(dict(AL, card_hobby=0), ["ランニング", "釣り", "サッカー"])
    marks = {}
    for _ in range(20):
        for e in deck.next_plan(_card(), persona=PERSONA)["episodes"]:
            marks[e["topic"]] = bool(e.get("no_habit"))
    assert marks.get("ランニング") is True and marks.get("釣り") is True
    assert marks.get("サッカー") is False


def test_mixes_are_not_locked_together():
    deck = ALDeck(AL, POOL)
    pairs = Counter()
    for _ in range(600):
        p = deck.next_plan(_card())
        pairs[(p["opening"], p["self_talk"])] += 1
    # 入り方ごとに、自分の話の回数がどれか1つに固まっていない
    for op in ("相手から", "聞き手から", "近況から"):
        row = [pairs[(op, k)] for k in (0, 1, 2)]
        assert max(row) < 0.8 * sum(row)


# --- 文面 ------------------------------------------------------------------

def test_stage1_text_lists_episodes_in_order():
    t = stage1_topic_text(_plan())
    assert t == stage1_al_text(_plan())
    assert "1) 「旅行」について、2人目に最近あった出来事" in t
    assert "2) 「ペット」について、2人目が続けていること" in t
    assert "conversation_flow は4項目" in t and "genre には「旅行」" in t
    assert "自分の経験を話すのは対話全体で1回まで" in t and "助言・評価・説教をしない" in t
    assert "同じ話の中で気になったこと" in t and "きっかけ" not in t and "具体的なこと" not in t    # 試走10: 「きっかけ・気持ち」が面接の問いに
    assert "scenario と conversation_flow に1人目の職業を書かないでください" in t
    assert "教えてもらった店" not in t and "これからやってみたいこと" in t
    assert "{" not in t and "}" not in t


def test_stage1_koekake_variants():
    t = stage1_al_text(_plan(koekake="次の話へ", koekake_at=1))
    assert "1つめの話を2人目が話し終えて言葉が途切れた" in t
    t = stage1_al_text(_plan(koekake="なし"))
    assert "2人目は自分から話を続け" in t and "途切れる場面" not in t


def test_stage2_block_uses_names_and_settings():
    b = stage2_topic_block(_plan(), "イノウエ", "オカダ")
    assert b == stage2_al_block(_plan(), "イノウエ", "オカダ")
    assert b.startswith("\n\n【傾聴の流れ】この対話は、オカダが自分のことを話し、イノウエが聞き役")
    assert "どちらか1人だけが名前を名乗る" in b
    assert "（目安は2回）" in b and "2〜3割" in b and "「？」で終わるターンは対話全体で4回まで" in b
    assert "「〜ね」で終わる応答は5回くらいまで" in b and "「〜ですか？」も問いに数え" in b
    assert "いちばん多く使うのは短い受けで" in b and "「そうなんですね」「そうなんですか」「そうでしたか」「なるほど」「へえ」" in b
    # ★2026-11-09 試走10: 型の一覧と目安の回数を並べると型を順番に回す機械的な応答になった → 心が動いたところへのひと言
    assert "決まった型を順番に回すのではなく" in b and "心が動いたところ" in b
    assert "次の4つから選び" not in b and "焦点のくり返し2回まで" not in b
    assert "どんな気分になりますか・どう感じましたか・どうして始めたんですか" in b
    assert "知りたくなった具体的なこと" not in b and "「具体的に」で始める問い" in b   # 版c: 「具体的」が問いに写った
    assert "「いいですね」「〇〇、いいですね」で受けるのは対話全体で2回まで" in b
    assert "「そうでしたか」" in b and "短い受けだけで流さない" in b
    assert "はい・いいえで終わる問い" not in b                                     # 人間の聞き手はよく使う
    assert "話を聞いていない機械的な相づちに聞こえます" in b
    assert "、ですか" not in b                     # 試走8: 例の「数グラム単位で、ですか」が写った
    assert "助言・説教・一般論でまとめたりしない" in b
    assert "イノウエが自分の経験をひと言話すのは、オカダの話とよく似た経験が人物設定にあるときだけにし、対話全体で1回まで" in b
    assert "オカダからイノウエに質問するのは" in b
    assert "オカダはイノウエの仕事・趣味・経歴・家族をまだ何も知らず" in b and "仕事や研究のことは尋ねません" in b
    # ★2026-11-07 試走7の視聴: 話し手はひと区切りまで1ターン・聞き手は区切りで受ける・受けは焦点のくり返し
    assert "ひと区切り（言いたいことを言い終えるところ）まで1つのターンで" in b
    assert "オカダの話の途中に割り込みません" in b
    # ★2026-11-08 試走8の評価: 応答は相手が直前に伝えたことから選ぶ・受けは焦点の語句だけ・問いは具体的なことを話し言葉で・
    #   つらい話の直後は気持ちを先に・話のまとめに一般論を言わない。例は形だけ（「数グラム単位」が話し手の話まで写った）
    assert "面接やカウンセリングのように聞こえる" in b and "問いより先に気持ちを受ける" in b
    assert "「〜っていいですよね」" in b
    assert "数グラム" not in b and "そのときどう思ったか" not in b
    assert "話題も変えません" in b                 # _plan() の声かけは「次の話へ」ではない
    n = stage2_al_block(_plan(koekake="次の話へ", koekake_at=1), "イノウエ", "オカダ")
    assert "下の声かけのとき以外は話題を変えません" in n and "締めくくりを短く受けてから" in n
    assert "前の話と無理につなげたり" in n and "同じ場所・人・時期・気持ち" not in n   # 試走10: こじつけの問い
    assert "「あ、そういえば」と別の話を始めない" in b and "「話は変わるんですけど」のようにひと言断って" in b
    one = stage2_al_block(_plan(episodes=[{"kind": "思い出", "topic": "ランニング"}], topics=["ランニング"]),
                          "イノウエ", "オカダ")
    assert "次の話に移るのは" not in one
    assert "問いのターンを2回続けず" in b and "答えに詰まったときも問いを重ねない" in b
    assert "何ターンにも分けて" not in b and "いちばん多いのは〜でしたね" not in b   # 試走7: 「〜ですね」を最多にしない（試走10で最多は短い受け）
    assert "オカダの気持ちをイノウエ自身の気持ちのように「〜ました。」で言い切る形にもしない" in b and "それはかなり焦りました" not in b   # 版d: 悪い例が写った
    assert "教えてもらった店" not in b
    assert "{" not in b and "}" not in b
    z = stage2_al_block(_plan(self_talk=0), "イノウエ", "オカダ")
    assert "イノウエは自分の経験や意見を話さず" in z


def test_no_habit_note_only_when_main_may_talk_about_self():
    ep = [{"kind": "思い出", "topic": "ランニング", "no_habit": True}]
    b = stage2_al_block(_plan(episodes=ep, topics=["ランニング"]), "イノウエ", "オカダ")
    assert "イノウエは「ランニング」をふだんはしていません" in b
    b0 = stage2_al_block(_plan(episodes=ep, topics=["ランニング"], self_talk=0), "イノウエ", "オカダ")
    assert "ふだんはしていません" not in b0
    assert "ふだんはしていません" not in stage1_al_text(_plan(episodes=ep, topics=["ランニング"]))


def test_stage2_prompt_has_the_al_block():
    p = build_dialogue_prompt(_meta(_plan()))
    assert "【傾聴の流れ】" in p and '"topic_plan"' not in p


def test_topic_plan_of_accepts_al_plans():
    assert topic_plan_of(_meta(_plan()))["style"] == "al"
    assert topic_plan_of(_meta(_plan(episodes=[]))) is None


# --- ゲート（al_shape）------------------------------------------------------

def _turns(rows):
    return [{"speaker": s, "utterance": u} for s, u in rows]


GOOD = _turns([
    ("イノウエ", "最近、何か楽しかったこととかありました？"),
    ("オカダ", "あの、先週、娘と二人で、久しぶりに温泉に行ってきたんです。"),
    ("イノウエ", "へえ、娘さんと。"),
    ("オカダ", "そうなんです、なんか、娘が仕事で忙しくて、ゆっくり話すのも久しぶりで。"),
    ("イノウエ", "それはうれしいですね。"),
    ("オカダ", "露天風呂に入りながら、まあ、昔の話とかいろいろしちゃって。"),
    ("イノウエ", "どんな話が出たんですか？"),
    ("オカダ", "娘が小さいころ、家族で行った海の話とか、もう全然覚えてないと思ってたんですけど。"),
    ("イノウエ", "覚えてらしたんですね。"),     # 問いを3ターン続けない（2026-11-07 の AL_Q_RUN_MAX）
    ("オカダ", "そうなんです、細かいことまで覚えてて、ちょっと泣きそうになりました。"),
    ("イノウエ", "そのとき、オカダさんはどう思われました？"),
    ("オカダ", "なんか、ちゃんと楽しかったんだなあって、ほっとしましたね。"),
])


def test_al_shape_passes_a_good_dialogue():
    assert "al_shape" in GATE_NAMES and ALL_GATES["al_shape"]
    c = al_counts(GOOD, "イノウエ")
    assert c["share"] < 0.35 and c["main_q"] == 3 and c["partner_q"] == 0 and c["main_q_run"] == 1
    assert al_shape_problem(GOOD, "イノウエ", _plan()) is None


def test_al_shape_flags_a_talkative_main_speaker():
    talk = GOOD + _turns([
        ("イノウエ", "私も、子どもたちと、家族でハイキングに行くんですけど、なんか、帰りの車でいろいろ話すんですよね。"),
        ("オカダ", "へえ、いいですね。"),
        ("イノウエ", "大文字山とか、伏見稲荷の山頂まで登ったりして、まあ、子どもはすぐ疲れたって言うんですけど。"),
        ("オカダ", "そうなんですね。"),
        ("イノウエ", "でも、上に着くと、すごく喜んで、その、写真とかいっぱい撮って、また来たいって言うんですよ。"),
        ("オカダ", "ふふ。"),
    ])
    why = al_shape_problem(talk, "イノウエ", _plan(self_talk=1))
    assert why and ("発話量" in why or "長い発話" in why)


def test_al_shape_flags_too_few_questions_and_ignores_bare_echoes():
    rows = [(s, u) for s, u in [(t["speaker"], t["utterance"]) for t in GOOD]]
    rows[6] = ("イノウエ", "露天風呂ですか？")       # くり返しだけの聞き返しは掘り下げに数えない
    t = _turns(rows)
    assert al_counts(t, "イノウエ")["main_q"] == 2
    rows[0] = ("イノウエ", "こんにちは。")
    why = al_shape_problem(_turns(rows), "イノウエ", _plan())
    assert why and "掘り下げの問い" in why


def test_al_shape_flags_a_questioning_partner():
    t = GOOD + _turns([
        ("イノウエ", "そうなんですね。"),
        ("オカダ", "イノウエさんは、温泉とかよく行かれるんですか？"),
        ("イノウエ", "たまに、家族で行きますね。"),
        ("オカダ", "どのあたりの温泉に行かれることが多いんですか？"),
    ])
    assert al_shape_problem(t, "イノウエ", _plan(self_talk=1)) is None
    why = al_shape_problem(t, "イノウエ", _plan(self_talk=0))
    assert why and "相手の質問" in why


def test_reecho_counts_restatements_but_not_questions_or_greetings():
    prev = "えっと、クミンとか、コリアンダーをベースにして、気分で足してます。"
    assert is_reecho("気分で足すんですね。", "気分で足してます。")
    assert is_reecho("クミンとコリアンダーなんですね。", prev)
    assert not is_reecho("どんなスパイスを足すんですか？", prev)          # 掘り下げの問い
    assert not is_reecho("えー、それは楽しそうですね。", prev)            # 気持ちを受けるひと言
    assert not is_reecho("こちらこそ、ありがとうございました。", "ありがとうございました。")
    assert not is_reecho("へえ。", prev) and not is_reecho("気分で。", "")
    assert al_counts(_turns([("オカダ", "カレーを作ってるんです。"), ("イノウエ", "そうなんですか？")]),
                     "イノウエ")["main_q"] == 0          # 「なんですか」は疑問詞の「なんで」ではない


def test_al_shape_flags_too_many_restatements():
    rows = [("オカダ", "先週、娘と二人で、久しぶりに温泉に行ってきたんです。"), ("イノウエ", "温泉に行ってきたんですね。"),
            ("オカダ", "露天風呂に入りながら、昔の話をいろいろしちゃって。"), ("イノウエ", "昔の話をされたんですね。"),
            ("オカダ", "娘が小さいころに行った海の話とかを、覚えてたんですよ。"), ("イノウエ", "海の話を覚えてたんですね。"),
            ("オカダ", "細かいことまで覚えてて、ちょっと泣きそうになりました。"), ("イノウエ", "泣きそうになったんですね。"),
            ("オカダ", "なんか、ちゃんと楽しかったんだなあって、ほっとしました。"), ("イノウエ", "ほっとしたんですね。"),
            ("オカダ", "そうなんです。帰りの電車でも、ずっと話してました。"), ("イノウエ", "どんな話をされたんですか？"),
            ("オカダ", "次はどこに行こうかって話を、二人で考えてました。"), ("イノウエ", "それは楽しみですね。"),
            ("オカダ", "そうなんです、次は紅葉を見に行きたいねって。"), ("イノウエ", "そのとき、どう思われました？")]
    assert al_counts(_turns(rows), "イノウエ")["main_reecho"] == 5
    # ★2026-11-08 上限 6→4（試走9で「〜んですね」のくり返しが聞き手のターンの46%）
    four = rows[:6] + rows[10:]
    assert al_counts(_turns(four), "イノウエ")["main_reecho"] == 3
    assert al_shape_problem(_turns(four), "イノウエ", _plan()) is None
    rows[13] = ("イノウエ", "二人で考えてたんですね。")
    assert al_counts(_turns(rows), "イノウエ")["main_reecho"] == 6
    why = al_shape_problem(_turns(rows), "イノウエ", _plan())
    assert why and "くり返し・言い直し" in why


def test_al_shape_ignores_other_plans():
    assert al_shape_problem(GOOD, "イノウエ", {"style": "qa"}) is None
    assert al_shape_problem(GOOD, "", _plan()) is None


# --- config ----------------------------------------------------------------

def test_al_config_loads_and_builds_a_deck():
    cfg = load_config(str(ROOT / "configs/al_v1.yaml"))
    md = cfg["metadata"]
    assert md["scenario_file"] == "scenarios/al_v1.yaml"
    # ft_v2・qa_v1 と同じ一覧（2026-11-03/04 にユーザが 001 を外して 002 を2回にした）
    ft = load_config(str(ROOT / "configs/ft_v2.yaml"))["metadata"]["main_speaker"]
    assert list(persona_files(md["main_speaker"])) == list(persona_files(ft))
    sc = load_scenario(md["scenario_file"])
    deck = make_deck(md["topic_plan"], sc.genres)
    assert isinstance(deck, ALDeck) and "天気" not in deck.extra_mix["kinkyo"]
    assert set(deck.kind_mix) == set(AL_KIND_TEXT)
    d = cfg["dialogue"]
    assert d["prompt_file"] == "prompts/stage2_al.yaml"
    e = d["enrich"]
    assert e["question_ratio_cap"] == 0 and e["main_eval_cap"] == -1 and e["main_echo_cap"] == 2
    assert d["output_file"] == "data/al_v1/dialogues.jsonl"
    assert cfg["tts"]["audio_dir"] == "data/al_v1/audio"
    # Stage4: 聞き手（イノウエ）の挿入相槌を増やす（2026-11-07）
    from refine_maai import RefineConfig
    rf = RefineConfig.from_tts_cfg(cfg["tts"])
    assert rf.density.min_bc_per_min == 7.0 and rf.density.max_bc_per_min == 11.0 and rf.density.count_hint
    assert rf.quality.bc_vocab_style == "listener" and rf.quality.bc_leak_check.get("enabled")
    assert "おー" in rf.quality.bc_vocab_mix
    assert cfg["tts"]["fish_speech"]["reading_normalize"]["verb_kata"] is True
    assert cfg["dialogue"]["min_turns"] == 20 and cfg["dialogue"]["max_turns"] == 26
    m = rf.insert.for_main_listener()
    assert m.min_turn_sec < rf.insert.min_turn_sec and m.min_separation_sec < rf.insert.min_separation_sec
    assert rf.insert.min_turn_sec == 4.0          # 相手が聞き手の位置は継承のまま


# --- 試走2のあとの変更（傾聴のときだけ効く）-----------------------------------

def test_short_reaction_wording_changes_only_for_al():
    from generate_dialogues import S2Params
    sp = S2Params(react_vocab_human=True)
    al = build_dialogue_prompt(_meta(_plan()), sp=sp)
    chat = build_dialogue_prompt(_meta({"opening": "話題から", "topics": ["旅行", "ペット"]}), sp=sp)
    assert "直前の発話への驚き・感心・共感のひと言にし" in al and "直前の発話に出てきた語を使った聞き返し" not in al
    assert "相手の出来事や気持ちへの具体的なひと言にする" in al and "自分の経験をそのまま言う" not in al
    assert "直前の発話に出てきた語を使った聞き返し" in chat and "自分の経験をそのまま言う" in chat
    assert "締めのお礼や挨拶はオカダから切り出し" in al


def test_job_gates_are_lenient_only_for_al():
    from topic_plan import early_self_job, partner_knows_job
    own = _turns([("イノウエ", "最近どうですか？"), ("オカダ", "実は昔、化学メーカーで研究職をしていたんです。")])
    assert partner_knows_job(own, "イノウエ") == 1                      # 雑談・qa は従来どおり
    assert partner_knows_job(own, "イノウエ", need_name=True) is None
    asks = _turns([("イノウエ", "へえ。"), ("オカダ", "イノウエさんは、研究のお仕事をされてるんですか？")])
    assert partner_knows_job(asks, "イノウエ", need_name=True) == 1
    echo = _turns([("オカダ", "最近、お掃除ロボットを買ったんですよ。"), ("イノウエ", "お掃除ロボットですか？")])
    assert early_self_job(echo, "イノウエ") == 1
    assert early_self_job(echo, "イノウエ", skip_echo=True) is None


def test_listener_empathy_keeps_ne_only_for_al():
    from generate_dialogues import fix_ne_misuse, fix_yone_misuse
    t = _turns([("オカダ", "道に迷っちゃって。"), ("イノウエ", "それは焦りましたね。"), ("オカダ", "参考になりましたね。")])
    assert fix_ne_misuse(t, skip_speaker="イノウエ") == 1
    assert [x["utterance"] for x in t] == ["道に迷っちゃって。", "それは焦りましたね。", "参考になりました。"]
    y = _turns([("オカダ", "道に迷っちゃって。"), ("イノウエ", "それは焦りましたよね。")])
    assert fix_yone_misuse(y, skip_speaker="イノウエ") == 0 and y[1]["utterance"] == "それは焦りましたよね。"
    assert fix_yone_misuse(y) == 1                                          # 雑談・qa は従来どおり


# --- 試走7の視聴のあと（2026-11-07）------------------------------------------

def test_al_shape_flags_question_runs_and_too_many_ne_receipts():
    from al_plan import AL_NE_REC_MAX, AL_Q_RUN_MAX
    assert AL_Q_RUN_MAX == 3 and AL_NE_REC_MAX == 6
    q3 = _turns([("イノウエ", "はじめまして。"), ("オカダ", "はじめまして。オカダです。"),
                 ("イノウエ", "最近はどんなことをされてるんですか？"), ("オカダ", "最近は、山歩きを始めたんです。"),
                 ("イノウエ", "どこの山に行かれたんですか？"), ("オカダ", "うーん。"),
                 ("イノウエ", "そのときの気持ちはどうでしたか？"), ("オカダ", "頂上で、すごく感動しました。"),
                 ("イノウエ", "感動されたんですね。"), ("オカダ", "はい。また行きたいです。"),
                 ("イノウエ", "ありがとうございます。"), ("オカダ", "こちらこそ、ありがとうございました。")])
    c = al_counts(q3, "イノウエ")
    assert c["main_q_run"] == 3
    long = [dict(t, utterance=t["utterance"] * 3) if t["speaker"] == "オカダ" else t for t in q3]
    assert "問いが3ターン続く" in al_shape_problem(long, "イノウエ", _plan())
    rows = [("イノウエ", "はじめまして。"), ("オカダ", "はじめまして。オカダです。")]
    for k in range(8):
        rows += [("オカダ" if False else "イノウエ", "どんな山なんですか？" if k in (0, 4) else f"{k}回目の山なんですね。"),
                 ("オカダ", f"そうなんです、{k}回目の山で、とても気持ちよかったんです。")]
    rows += [("イノウエ", "ありがとうございました。"), ("オカダ", "失礼します。"), ("イノウエ", "失礼します。")]
    ne = _turns(rows)
    assert al_counts(ne, "イノウエ")["main_ne_rec"] == 6
    many = _turns(rows[:2] + [("イノウエ", "どんな山ですか？" if k in (0, 3) else f"{k}回目なんですね。") if i % 2 == 0
                             else ("オカダ", "そうなんです、本当に気持ちのいい山で、景色もすごくよかったんですよ。")
                             for k in range(10) for i in range(2)] + rows[-3:])
    assert al_counts(many, "イノウエ")["main_ne_rec"] == 8
    assert "「〜ね」の受け8回>6回" in al_shape_problem(many, "イノウエ", _plan())


def test_listener_prompt_lets_the_speaker_finish_only_for_al():
    from generate_dialogues import S2Params
    cfg = load_config(str(ROOT / "configs/al_v1.yaml"))
    sp = S2Params.from_config(cfg)
    al = build_dialogue_prompt(_meta(_plan()), sp, None, "gemma")
    assert "聞き役の発話は1ターン1〜2文の短いもの、話し手の発話は1ターン1〜3文" in al
    assert "聞き役の短い受け・驚きや共感のひと言・問い" in al
    assert "2文以上入ったターンを作らないでください" not in al
    assert "極短の反応を全体の1割" in al                      # Gemma 向けの2割にしない
    assert "言いさし・体言止め" not in al
    chat = build_dialogue_prompt(_meta(), sp, None, "gemma")
    assert "発話は1ターン1文だけにし" in chat and "2文以上入ったターンを作らないでください" in chat
    assert "極短の反応を全体の2割" in chat


def test_listener_sentence_final_particles_are_not_thinned():
    from generate_dialogues import diversify_sfp_runs, sentence_ids, thin_ne, thin_sfp
    turns = [{"speaker": "オカダ", "utterance": "履歴書の控えを、家に置いてきてしまったんですけど、なんとかなりました。"},
             {"speaker": "イノウエ", "utterance": "それは、本当に焦ったでしょうね。"},
             {"speaker": "オカダ", "utterance": "はい、もう本当に、どうしようかと思いましたね。"},
             {"speaker": "イノウエ", "utterance": "勇気がいる決断でしたね。"},
             {"speaker": "オカダ", "utterance": "そのあと、受付の方に相談してみたんですよね。"},
             {"speaker": "イノウエ", "utterance": "受付の方に相談されたんですね。"}]
    import copy
    a = copy.deepcopy(turns)
    only = sentence_ids(a, {"オカダ"})
    thin_ne(a, 0.5, only=only, keep_realize=True)
    assert a[3]["utterance"] == "勇気がいる決断でしたね。"            # 聞き手の「ね」は残る
    b = copy.deepcopy(turns)
    thin_ne(b, 0.5, keep_realize=True)
    assert b[3]["utterance"] == "勇気がいる決断でした。"              # 外さないと落ちる（試走7の原因）
    c = copy.deepcopy(turns)
    thin_sfp(c, 0.0, skip_speaker="イノウエ")
    assert all(c[i]["utterance"] == turns[i]["utterance"] for i in (1, 3, 5))
    d = [{"speaker": s, "utterance": u} for s, u in (("オカダ", "楽しかったですね。"), ("イノウエ", "楽しそうですね。"),
                                                    ("オカダ", "また行きたいですね。"), ("イノウエ", "いいですね。"))]
    diversify_sfp_runs(d, polite=True, skip_speaker="イノウエ")
    assert d[1]["utterance"] == "楽しそうですね。" and d[3]["utterance"] == "いいですね。"


def test_comma_echo_counts_as_echo_and_question_turns_are_capped():
    from al_plan import AL_Q_TURNS_MAX, _is_bare_echo
    assert _is_bare_echo("効率が上がった、ですか？", "走ると、仕事の効率が上がった気がします。")
    assert not _is_bare_echo("どんな効率ですか？", "走ると、仕事の効率が上がった気がします。")
    rows = [("イノウエ", "はじめまして。"), ("オカダ", "はじめまして。オカダです。")]
    for k in range(AL_Q_TURNS_MAX + 1):
        rows += [("イノウエ", f"{k}回目の山、ですか？" if k % 2 else "それはうれしいことでしたね。"),
                 ("オカダ", f"そうなんです、{k}回目の山で、景色もすごくよくて、本当に気持ちよかったんですよ。")]
        rows += [("イノウエ", "どんな景色だったんですか？" if k % 2 == 0 else "楽しそうですね。"),
                 ("オカダ", "山の上から海まで見えて、本当に感動しました、また行きたいです。")]
    rows += [("イノウエ", "ありがとうございました。"), ("オカダ", "失礼します。"), ("イノウエ", "失礼します。")]
    c = al_counts(_turns(rows), "イノウエ")
    assert c["main_q_turns"] > AL_Q_TURNS_MAX


def test_unknown_speaker_names_are_caught():
    # ★2026-11-08 試走9: 話者名「イノウ\u17c1」（最後の1文字がクメール文字）がゲートを抜けて3人目になった
    from generate_dialogues import unknown_speakers
    meta = {"speakers": [{"name": "イノウエ"}, {"name": "ウチダ"}]}
    turns = [{"speaker": "イノウエ", "utterance": "こんにちは。"}, {"speaker": "ウチダ", "utterance": "どうも。"},
             {"speaker": "イノウ\u17c1", "utterance": "そうなんですね。"}]
    assert unknown_speakers(turns, meta) == ["イノウ\u17c1"]
    assert unknown_speakers(turns[:2], meta) == []
    assert unknown_speakers(turns, {"speakers": [{"name": "イノウエ"}]}) == []
    assert "speaker_names" in GATE_NAMES and ALL_GATES["speaker_names"]


def test_al_shape_flags_too_many_ndesune():
    # ★2026-11-08 試走9b: 「しんどかったんですね」「不安だったんですね」…と「〜んですね」が聞き手のターンの35%
    from al_plan import AL_NDESUNE_MAX
    rows = [("オカダ", "こんにちは。"), ("イノウエ", "こんにちは。よろしくお願いします。")]
    for k in range(AL_NDESUNE_MAX + 1):
        rows += [("オカダ", f"それで{k}回目は、人前で話す場面があって、手が震えるくらいすごく緊張したんです。"),
                 ("イノウエ", f"いよいよ本番{k}だったんですね。")]
        if k in (1, 3):
            rows += [("オカダ", "でも、なんとか最後までやりました。"), ("イノウエ", "どんなふうに乗り切ったんですか？")]
    rows += [("オカダ", "どこに行ったんですか？"), ("イノウエ", "公園です。"), ("オカダ", "そうなんです。"),
             ("イノウエ", "そうなんですね。"), ("オカダ", "ありがとうございました。")]
    c = al_counts(_turns(rows), "イノウエ")
    assert c["main_ndesune"] == AL_NDESUNE_MAX + 1          # 「そうなんですね」・最後の3ターンは数えない
    assert "「〜んですね」" in al_shape_problem(_turns(rows), "イノウエ", _plan(self_talk=2))
    rows[3] = ("イノウエ", "あー、そうなんですね。")
    assert al_counts(_turns(rows), "イノウエ")["main_ndesune"] == AL_NDESUNE_MAX
    b = stage2_al_block(_plan(), "イノウエ", "オカダ")
    assert "「そうなんですね」を除いて対話全体で2回まで" in b and "気持ちの語をそのまま言い直したり" in b
