"""1対話で扱う話題の組と、その順番（config: metadata.topic_plan）。

v1 の初対面コーパスは、シナリオの conversation_flow が「挨拶 → 自己紹介（仕事）→ 趣味 →
着地」の型に固定されていた。実測（output_ft_v1_gemma_inoue_010 の1000対話）:

  - 仕事の話の初出が2ターンめ（中央値）。96% が最初の6ターン以内。
  - 1対話が3ターン以上使う話題は平均1.8個（1個だけが34%）。
  - 最初の話題は「研究・大学」37%・「仕事」21%。genre 欄は137種あるのに、
    実際の中身は序盤が職業の確認で埋まる。

このため v1 で学習した J-Moshi は、何を言っても会話の序盤で職業の話に入る。
ここでは対話ごとに「話の入り方」と「扱う話題2〜3個の順番」を引いて Stage1 に渡し、
Stage2 には同じ順番を段落で渡す（pair_mix / style_mix / partner_persona と同じ方式）。

config に topic_plan が無い・メタデータに topic_plan が無いときは、どのプロンプトも
一字も変えない。Stage3/4 は読まない。

★2026-10-30 topic_plan.style: qa（質問と答えが中心の対話。config_qa_v1_gemma_inoue.yaml）を追加。
中身は qa_plan.py（QADeck・文面・ゲート）。メタデータの topic_plan に style: "qa" が入り、
挨拶・名乗り・締め方の指定と、それに付く後処理・ゲート（apply_naming / main_closes /
self_job_early / 時間を口実にした締めの書き直し）は雑談の topic_plan と共通で効く。
★2026-11-02 topic_plan.style: al（傾聴。config_al_v1_gemma_inoue.yaml）を追加。中身は al_plan.py。
qa と同じく、挨拶・名乗り・締め方の指定とそれに付く後処理・ゲートは共通で効く。
★2026-11-10 topic_plan.style: rp（雑談＋声が重なる場面。config_rp_v1_gemma_inoue.yaml）を追加。中身は rp_plan.py。
雑談の plan（話題・入り方・挨拶・締め方）に overlaps（重なりの場面の種類）を足したもの。
"""

import random
import re
from collections import Counter

# 挨拶のあと、何から話に入るか。v1 は実質この最後の1つだけだった。
OPENING_TEXT = {
    "話題から": "挨拶と短い名乗りを交わしたら、お互いの職業を確かめ合うやり取りはせず、"
                "すぐに下の1つめの話題に入る",
    "近況から": "挨拶と短い名乗りを交わしたら、季節のことや最近あったことなど"
                "軽い雑談をひと往復し、そこから1つめの話題に移る",
    "住まい・出身から": "挨拶と短い名乗りを交わしたら、住んでいるあたりや出身の話を"
                        "ひと往復し、そこから1つめの話題に移る",
    "仕事・学校から": "挨拶と短い名乗りを交わしたら、お互いの仕事や学校のことを"
                      "ひと言ずつ交わし、そこから1つめの話題に移る",
}
OPENING_DEFAULT = {"話題から": 0.35, "近況から": 0.20,
                   "住まい・出身から": 0.15, "仕事・学校から": 0.30}
TOPICS_DEFAULT = {2: 0.45, 3: 0.55}

# ── 2026-10-29 挨拶の形・近況の中身・締め方も1対話ずつ指定する ───────────────
# config に greeting_mix / kinkyo_mix / closing_mix があるときだけ plan に入る
# （無ければ従来の plan と同じ。下の OPENING_TEXT の文面も変わらない）。
# 試走 output_ft_v2_gemma_inoue_001（20対話）の実測で、話題を散らした代わりに
# 次の型が固まっていた（人間 = maai-f2f-jp-2025 の先頭33ターン、54対話）:
#   第1発話が「はじめまして。〜です。よろしくお願いします。」 90%（人間 11%・v1 53%）
#   「近況から」の7件中6件が天気（「急に冷え込んできました」）。冒頭6ターンに天気 30%（人間 4%）
#   最後3ターンに「ゆっくりお話し」45%・最後6ターンに「似て/共通/共感/嬉しい」25%（人間 4%）
#   締めはほぼ主話者が切り出す（学習したモデルが2分前後で自分から会話を畳む）
# 挨拶の後半（名乗り）を指定するので、plan に greeting があるときは OPENING_AFTER を使う。
# {naming} は NAMING_TEXT（plan に naming が無ければ「両方」＝従来の文面）。
# ★2026-10-29 再評価: 「ひと言」に例（照れくさい・何から話そうか）を書いたら、
#   「いきなり…緊張します」→「わかります。私も…迷って」→「そう言っていただけると、少し安心します」
#   の3ターンがほぼそのまま繰り返された。例を外し、中身をその人から考えさせる。
GREETING_TEXT = {
    "はじめまして": "「はじめまして」と挨拶して、{naming}",
    "よろしく": "「はじめまして」は言わずに「よろしくお願いします」と言い合い、{naming}",
    "こんにちは": "「こんにちは」「どうも」のような軽い挨拶をして、{naming}",
    # ★2026-10-30 「この場や今の自分の様子について」→「初対面で話し始めることについて」（ユーザ承認）。
    #   「この部屋の空気に、なんだか新鮮な気持ちになります」「この空間の静けさが、いいですね」と、場所を
    #   設定しない設計なのにその場に触れていた（ft_v2 本番分でこの型の 64/815・qa_v1 試走3 で 1/3。
    #   数え方は控えめ）。例は書かない（上の注記）。それより前の ft_v2 の対話は旧文面。
    "ひと言": "挨拶して{naming}、そのあと話し始める前に、初対面で話し始めることについて"
              "その人らしいひと言を交わす（中身と言い方はその人の性格から考える。"
              "その場の様子や周りの物には触れない）",
    "名乗らない": "軽い挨拶だけして、名乗らずにそのまま話に入る（この対話では名前を出さない）",
}
GREETING_DEFAULT = {"はじめまして": 0.30, "よろしく": 0.25, "こんにちは": 0.15,
                    "ひと言": 0.15, "名乗らない": 0.15}
# 名乗り方（★2026-10-29 再評価: 最初の2ターンで両者が「〇〇です」「イノウエです」と名乗り合う
#   形が 17/20。挨拶の言葉を散らしても名乗りの鏡写しは残った）。指示は破られる前提で、
#   generate_dialogues.apply_naming が名乗りを機械で外す。挨拶が「名乗らない」の対話には付けない。
NAMING_TEXT = {
    "両方": "それぞれ名前だけを名乗る",
    "片方だけ": "どちらか1人だけが名前を名乗る（もう1人は名乗らずに挨拶を返す）",
    "なし": "どちらも名乗らない",
}
NAMING_DEFAULT = {"両方": 0.45, "片方だけ": 0.30, "なし": 0.25}
OPENING_AFTER = {
    "話題から": "お互いの職業を確かめ合うやり取りはせず、すぐに下の1つめの話題に入る",
    "近況から": "{kinkyo}の軽い雑談をひと往復し、そこから1つめの話題に移る",
    "住まい・出身から": "住んでいるあたりや出身の話をひと往復し、そこから1つめの話題に移る",
    "仕事・学校から": "相手（speakers の2人目）が自分の仕事や学校のことをひと言話すか、"
                      "1人目に尋ねるやり取りをひと往復し、そこから1つめの話題に移る",
}
# 「近況から」の中身。指定しないと Gemma は季節の変わり目の天気にほぼ決め打ちする。
KINKYO_TEXT = {
    "道のり": "今日ここまで来る道のり（どうやって来たか・遠かったか）",
    "週末": "この週末や最近の休みの日にしたこと",
    "忙しさ": "最近忙しいかどうか",
    "食べたもの": "最近食べておいしかったもの",
    "買ったもの": "最近買ったものや新しく使い始めたもの",
    "天気": "季節や天気のこと",
}
KINKYO_DEFAULT = {"道のり": 1.0, "週末": 1.0, "忙しさ": 1.0, "食べたもの": 1.0,
                  "買ったもの": 1.0, "天気": 0.5}
# 締め方。「締めない」は人間コーパスの切り出しと同じく、話の途中で終わる。
# 締めるときは相手（ユーザ側）が切り出す（主話者が自分から畳む型を学ばせない）。
CLOSING_TEXT = {
    "締めない": "締めの挨拶は書かない。最後の話題のやり取りが続いているところで対話を終える"
                "（「また話しましょう」「ありがとうございました」のような締めやまとめを入れない）",
    # ★2026-10-29 2回目の試走: ここに「そろそろ」と書いたら、6件すべてが「あ、そろそろ、
    #   時間ですね。」「そうですね。」「ありがとうございました。」の同じ3ターンになった。台詞は書かない。
    # ★3回目: 台詞を外しても、Gemma は「もうこんな時間。そろそろ行かないと」→「そうですね」→
    #   「ありがとうございました」に寄った（短く・話題に触れて 12件中7件）。時間を口実にしない形を書く。
    "短く": "最後の話題の流れのまま、{partner}がそれまでの話を受けたひと言（感想・お礼・"
            "次にやってみたいこと など）で切り上げ、ひと言ずつの挨拶で終える",
    "話題に触れて": "{partner}が最後の話題に出たこと（教えてもらった店・やってみたいこと など）"
                    "に触れて締めを切り出し、短い挨拶で終える",
}
CLOSING_DEFAULT = {"締めない": 0.4, "短く": 0.3, "話題に触れて": 0.3}
# ★2026-10-29 再評価: 時間を口実にした締め（「そろそろ時間が」）が 15〜20%（人間 4%）。
CLOSING_COMMON = ("共通点や価値観が似ていることを確かめ合うまとめや、お互いを褒め合う"
                  "締めにはしない。時間や次の予定を理由に切り上げる形にもしない")
POOL_WINDOW = 24          # 発注の少ない話題の上位この数から引く（偏らせずに散らす）
QA_STYLE = "qa"           # topic_plan.style: 質問と答えが中心の対話（qa_plan.py）
AL_STYLE = "al"           # topic_plan.style: 傾聴の対話（al_plan.py）
RP_STYLE = "rp"           # topic_plan.style: 雑談＋声が重なる場面（rp_plan.py。2026-11-10）

# 話題のまとまり。1つの対話に同じまとまりから2つ入れない（「読書」と「好きな作家」等）。
_GROUPS = [
    (r"^(趣味|最近始めた趣味|長く続けている趣味|集めているもの)$", "趣味全般"),
    (r"読書|作家|^本$|漫画", "本"),
    (r"アニメ|映画|ドラマ|配信", "映像"),
    (r"音楽|アーティスト|ライブ|楽器|カラオケ", "音楽"),
    (r"ゲーム", "ゲーム"),
    (r"お笑い|舞台|演劇", "演芸"),
    (r"写真|カメラ|絵|手芸|編み物|プラモデル|DIY|陶芸|書道", "手仕事"),
    (r"園芸|家庭菜園|観葉植物", "園芸"),
    (r"料理|自炊|お菓子|パン|弁当|朝ごはん", "料理"),
    (r"コーヒー|紅茶|お酒|日本酒", "飲み物"),
    (r"食べ歩き|ラーメン|カレー|甘いもの|外食|食べ物|食生活", "食べ物"),
    (r"運動|ランニング|ウォーキング|筋トレ|ジム|ヨガ|水泳|自転車", "運動"),
    (r"登山|キャンプ|釣り", "アウトドア"),
    (r"サッカー|野球|テニス|バドミントン|スポーツ観戦|チーム", "スポーツ観戦"),
    (r"サウナ|温泉", "温泉"),
    (r"旅行|旅先|行ってみたい|お土産", "旅行"),
    (r"仕事|職業|働き方|在宅|転職|副業|資格|定年", "仕事"),
    (r"学生生活|専攻|ゼミ|部活|アルバイト|就職", "学校"),
    (r"出身|方言|地元", "出身地"),
    (r"住んで|引っ越|一人暮らし|実家暮らし", "住まい"),
    (r"通勤|通学|運転", "移動"),
    (r"家族|兄弟|姉妹|子ども|孫", "家族"),
    (r"ペット|^犬$|^猫$", "ペット"),
    (r"休日|連休|年末年始|行事", "休日"),
    (r"天気|季節|暑さ|寒さ|花粉", "天気"),
    (r"睡眠|朝型|健康|肩こり|腰痛", "健康"),
    (r"節約|ポイント", "節約"),
    (r"スマホ|アプリ|SNS|動画|通販|ネット", "ネット"),
    (r"模様替え|掃除|片付け|収納|部屋", "家のこと"),
    (r"流行", "流行"),
    (r"思い出|若い頃|初めての|失敗|うっかり", "思い出"),
]
_GROUPS = [(re.compile(p), g) for p, g in _GROUPS]
# 「仕事・学校から」で入る対話に、話題としての仕事・学校を重ねない。
_OPENING_SKIP = {"仕事・学校から": {"仕事", "学校"}, "住まい・出身から": {"出身地", "住まい"}}
# 仕事・学校は1つめの話題にしない（序盤が職業の話になるのを避けるのが topic_plan の目的）。
_NOT_FIRST = {"仕事", "学校"}
# 相手の年齢に合わない話題（人物カードがあるときだけ効く。77歳に「通学」が当たった）。
_AGE_RULES = [
    (re.compile(r"学生生活|専攻|ゼミ|部活|就職活動|通学"), 0, 29),
    (re.compile(r"アルバイト"), 0, 39),
    # ★2026-10-29 学生でない相手に「専攻」を振ると、主話者が自分の研究の話として広げる
    #   （試走で27歳の会社員に当たった2回とも、後半がイノウエの研究の説明になった）
    (re.compile(r"専攻|ゼミ"), 0, 24),
    (re.compile(r"孫|定年"), 55, 200),
    (re.compile(r"子どもの成長"), 25, 64),
]


def fits_age(topic: str, age: int | None) -> bool:
    """その話題を相手に振ってよいか（年齢が分からなければ常に True）。"""
    if age is None:
        return True
    return all(lo <= age <= hi for rx, lo, hi in _AGE_RULES if rx.search(topic))


def topic_group(topic: str) -> str:
    """話題のまとまりの名前（どれにも当たらなければ話題そのもの）。"""
    for rx, g in _GROUPS:
        if rx.search(topic):
            return g
    return topic


def _weights(src, default: dict, key=lambda x: x) -> dict:
    """config の重み（{ラベル: 重み}）。空なら default。"""
    out = {}
    for k, v in (src or default).items():
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if w > 0:
            out[key(k)] = w
    return out or dict(default)


class TopicDeck:
    """対話ごとに「話の入り方」と「話題2〜3個の順番」を配る。

    入り方と話題数は、発注の少ないものから目標比率に寄せて配る
    （DiversityTracker.next_style / PartnerPool.next_card と同じ方式）。
    話題はシナリオの genres から、発注の少ない順の上位 POOL_WINDOW 件を母数にして引く。
    """

    def __init__(self, cfg: dict, genres: list[str]):
        cfg = cfg or {}
        self.pool = [g for g in (genres or []) if g]
        if not self.pool:
            raise ValueError("metadata.topic_plan はシナリオの genres が要ります")
        self.rng = random.Random(int(cfg.get("seed", 0)))
        self.opening_mix = _weights(cfg.get("opening_mix"), OPENING_DEFAULT)
        unknown = set(self.opening_mix) - set(OPENING_TEXT)
        if unknown:
            raise ValueError(f"topic_plan.opening_mix の未知のラベル: {sorted(unknown)}"
                             f"（使えるのは {sorted(OPENING_TEXT)}）")
        self.topics_mix = _weights(cfg.get("topics"), TOPICS_DEFAULT, key=int)
        # 話題の1つを相手の人物カードの趣味にする割合（partner_persona と併用するとき）。
        self.card_hobby = float(cfg.get("card_hobby", 0.0) or 0.0)
        # 挨拶の形・近況の中身・締め方（config に書いたものだけ plan に入れる）
        self.extra_mix = {}
        for key, cfg_key, default, texts in (
                ("greeting", "greeting_mix", GREETING_DEFAULT, GREETING_TEXT),
                ("kinkyo", "kinkyo_mix", KINKYO_DEFAULT, KINKYO_TEXT),
                ("closing", "closing_mix", CLOSING_DEFAULT, CLOSING_TEXT),
                ("naming", "naming_mix", NAMING_DEFAULT, NAMING_TEXT)):
            if cfg_key not in cfg:
                continue
            mix = _weights(cfg.get(cfg_key), default)
            unknown = set(mix) - set(texts)
            if unknown:
                raise ValueError(f"topic_plan.{cfg_key} の未知のラベル: {sorted(unknown)}"
                                 f"（使えるのは {sorted(texts)}）")
            self.extra_mix[key] = mix
        self.extra_issued = {k: Counter() for k in self.extra_mix}
        self.opening_issued = Counter()
        self.count_issued = Counter()
        self.topic_counts = Counter()

    def describe(self) -> str:
        """起動時のログに出す設定の要約。"""
        return (f"話題 {len(self.pool)}種から{self.topics_mix}個、入り方の重み {self.opening_mix}"
                + (f"、カードの趣味を使う割合 {self.card_hobby}" if self.card_hobby else ""))

    def _least(self, issued: Counter, mix: dict):
        """目標比率に対していちばん不足しているラベル。"""
        return min(mix, key=lambda k: ((issued[k] + 1) / mix[k], self.rng.random()))

    def add_used(self, plan: dict):
        """既存メタデータの plan を発注数に数える（追記しても比率を保つ）。"""
        if not plan:
            return
        if plan.get("opening") in self.opening_mix:
            self.opening_issued[plan["opening"]] += 1
        n = len(plan.get("topics") or [])
        if n in self.topics_mix:
            self.count_issued[n] += 1
        for k, mix in self.extra_mix.items():
            if plan.get(k) in mix:
                self.extra_issued[k][plan[k]] += 1
        for t in plan.get("topics") or []:
            self.topic_counts[t] += 1

    def next_plan(self, card: dict | None = None, persona: str = "") -> dict:
        """persona は QA の版（qa_plan.QADeck）だけが使う。呼び出しをそろえるために受け取る。"""
        opening = self._least(self.opening_issued, self.opening_mix)
        self.opening_issued[opening] += 1
        n = self._least(self.count_issued, self.topics_mix)
        self.count_issued[n] += 1

        skip = _OPENING_SKIP.get(opening, set())
        age = (card or {}).get("age")
        cands = sorted(self.pool, key=lambda t: (self.topic_counts[t], self.rng.random()))
        # 発注の少ない順（同数のあいだは毎回ばらばら）に、まとまりが重ならないよう取る。
        cands = [t for t in cands
                 if topic_group(t) not in skip and fits_age(t, age)][:POOL_WINDOW]
        topics, groups = [], set()
        for t in cands:
            g = topic_group(t)
            # 仕事と学校は同じ対話に2つ入れない（★2026-10-29 試走で「仕事の内容」「専攻」の
            #   組が当たり、後半が主話者の研究の説明になった）
            if g in groups or (g in _NOT_FIRST and groups & _NOT_FIRST):
                continue
            topics.append(t)
            groups.add(g)
            if len(topics) == n:
                break
        if len(topics) < n:                      # 母数が足りないときはまとまりの重複を許す
            for t in cands:
                if t not in topics:
                    topics.append(t)
                if len(topics) == n:
                    break
        for _ in range(len(topics) - 1):
            if topic_group(topics[0]) not in _NOT_FIRST:
                break
            topics.append(topics.pop(0))
        hobbies = [h for h in ((card or {}).get("hobbies") or [])
                   if topic_group(h) not in groups and topic_group(h) not in skip]
        if hobbies and self.rng.random() < self.card_hobby:
            topics[self.rng.randrange(len(topics))] = self.rng.choice(hobbies)
        for t in topics:
            self.topic_counts[t] += 1
        plan = {"opening": opening, "topics": topics}
        for k, mix in self.extra_mix.items():
            if k == "kinkyo" and opening != "近況から":
                continue
            if k == "naming" and plan.get("greeting") == "名乗らない":
                continue
            v = self._least(self.extra_issued[k], mix)
            self.extra_issued[k][v] += 1
            plan[k] = v
        return plan


def _opening_line(plan: dict) -> str:
    """「話の入り方」の1行。plan に greeting があれば挨拶の形と入り方を分けて書く。"""
    g = plan.get("greeting")
    if g not in GREETING_TEXT:
        return OPENING_TEXT[plan["opening"]]
    kinkyo = KINKYO_TEXT.get(plan.get("kinkyo"), "季節のことや最近あったことなど")
    return (f"{_greeting_text(plan)}。そのあと、"
            + OPENING_AFTER[plan["opening"]].format(kinkyo=kinkyo))


def _greeting_text(plan: dict) -> str:
    """挨拶の形の文面（名乗り方を埋めたもの）。"""
    naming = NAMING_TEXT.get(plan.get("naming"), NAMING_TEXT["両方"])
    return GREETING_TEXT[plan["greeting"]].format(naming=naming)


def naming_of(plan: dict | None) -> str | None:
    """名乗り方のラベル（挨拶が「名乗らない」なら "なし"。指定が無ければ None）。"""
    if not plan:
        return None
    if plan.get("greeting") == "名乗らない":
        return "なし"
    return plan.get("naming") if plan.get("naming") in NAMING_TEXT else None


def _closing_line(plan: dict, partner: str) -> str:
    c = plan.get("closing")
    if c not in CLOSING_TEXT:
        return ""
    return CLOSING_TEXT[c].format(partner=partner)


def stage1_topic_text(plan: dict) -> str:
    """Stage1 プロンプトに入れる「話題と順番」の段落。"""
    if plan.get("style") == QA_STYLE:
        from qa_plan import stage1_qa_text
        return stage1_qa_text(plan)
    if plan.get("style") == AL_STYLE:
        from al_plan import stage1_al_text
        return stage1_al_text(plan)
    if plan.get("style") == RP_STYLE:
        from rp_plan import stage1_rp_text
        return stage1_rp_text(plan)
    topics = plan["topics"]
    n = len(topics)
    order = " → ".join(f"{i}) {t}" for i, t in enumerate(topics, 1))
    close = _closing_line(plan, "相手（speakers の2人目）")
    if close:
        close_item = (f"最後の項目で終わり方（{close}。{CLOSING_COMMON}）を書いてください。\n"
                      if plan["closing"] != "締めない" else
                      f"最後の項目は最後の話題の続き（{close}）にしてください。\n")
    else:
        close_item = "最後の項目で軽い着地を書いてください。\n"
    return (
        "この対話で扱う話題と、その順番を指定します:\n"
        f"- 話の入り方: {_opening_line(plan)}\n"
        f"- 話題: {order}\n"
        f"- conversation_flow は{n + 2}項目にしてください。1項目めに挨拶と上の話の入り方、"
        f"続く{n}項目でこの話題を指定の順に1つずつ、" + close_item +
        "- どの話題も同じくらいの長さで話します。1つの話題だけで会話の大半を使わないでください。\n"
        "- 指定した話題以外を会話の中心にしないでください。相手の趣味や本人のことは、"
        "上の話題の中で触れるぶんにはかまいません。\n"
        f"- genre には「{topics[0]}」と書いてください。\n"
        + ("" if plan["opening"] == "仕事・学校から" else
           "- 1項目めに、二人が互いの職業を紹介し合うやり取りを書かないでください"
           "（名乗ったら上の話の入り方でそのまま話題に入る）。\n"))


def stage2_topic_block(plan: dict, main_name: str = "", partner_name: str = "") -> str:
    """Stage2 プロンプトに足す「話題の流れ」の段落（人物設定の段落の直後）。

    main_name: 主話者（システム側）の名前。渡すと「自分の仕事の話を自分から
    持ち出さない」の一文にその名前を入れる。partner_name は締めを切り出す側の名前。
    """
    if plan.get("style") == QA_STYLE:
        from qa_plan import stage2_qa_block
        return stage2_qa_block(plan, main_name, partner_name)
    if plan.get("style") == AL_STYLE:
        from al_plan import stage2_al_block
        return stage2_al_block(plan, main_name, partner_name)
    if plan.get("style") == RP_STYLE:
        from rp_plan import stage2_rp_block
        return stage2_rp_block(plan, main_name, partner_name)
    topics = plan["topics"]
    order = " → ".join(f"{i}) {t}" for i, t in enumerate(topics, 1))
    work = ("" if plan["opening"] == "仕事・学校から" else
            "仕事・職業・学校の話は、尋ねるとしてもひと言ふた言で切り上げ、"
            "会話の中心にしないでください。")
    # 話題の切り替えの前置き。★2026-10-29 以前は「あ、そういえば」等を置かせていた
    # （人間 1.50回/対話という根拠）が、それは人間の1セッション（中央値437ターン）全体の
    # 数で、先頭33ターンでは 0.02回/対話（33ターン換算 0.07回）しかない。指示どおり
    # 置いた試走は「そういえば」0.8・「ところで」0.8回/対話で、話題が変わるたびに同じ
    # 前置きが付く型になった。前置きは1回までにし、残りは連想でつながせる
    # （超えた分は generate_dialogues.cap_shift_marks が機械で外す）。
    shift = ("話題を移るときは前置きに頼らず、直前の話に出た言葉からの連想で、"
             "自分の話としてつなげてください（「そういえば」「ところで」は対話全体で1回まで）。"
             if len(topics) >= 2 else "")
    # 主話者（システム側）が自分の職業の話を序盤に出すと、学習したモデルが会話の頭で
    # 必ず職業の話をするようになる（ユーザ報告）。人間コーパスの先頭33ターンでは、
    # 研究・大学・学校の語が3ターン以内に出るのは2%・まったく出ないのが42%。
    # 仕事から入る対話でも、主話者が先に自分の職業を言う型にしない（試走で 3/3 件が
    # 主話者から「大学で研究を」と切り出した）。検査は early_self_job。
    who = f"主話者（{main_name}）" if main_name else "主話者"
    self_job = (f"{who}の研究・大学・仕事の話は、相手から尋ねられたときに"
                "ひと言答えるだけにし、自分からは持ち出さないでください。"
                if plan["opening"] != "仕事・学校から" else
                f"{who}は自分から先に職業を言わず、相手が自分の仕事を話すか"
                f"{main_name or '主話者'}に尋ねてから、ひと言で答えてください。")
    # 仕事・学校の話題（「専攻」「仕事の内容」等）は相手の話として扱う（★3回目: 「専攻」の対話で
    #   主話者が自分の研究を説明し続けた）。
    ws = [t for t in topics if topic_group(t) in _NOT_FIRST]
    ws_txt = (f"「{ws[0]}」の話題は{partner_name or '相手'}のことを中心に話し、"
              f"{who}は聞き役に回ってください。" if ws else "")
    greet = (f"対話の始まりは、{_greeting_text(plan)}形にしてください"
             "（名乗りに職業や所属を付けない）。" if plan.get("greeting") in GREETING_TEXT else "")
    close = _closing_line(plan, partner_name or "相手")
    if close:
        close = f"終わり方: {close}。" + ("" if plan["closing"] == "締めない"
                                       else f"{CLOSING_COMMON}。")
    return (
        f"\n\n【話題の流れ】{greet}この対話は次の順に話題が移ります: {order}。"
        f"それぞれの話題に同じくらいのターン数を使い、1つの話題で対話の大半を使わないでください。"
        f"次の話題に移るときは、直前のやり取りに出た言葉を受けて自然につなげ、"
        f"唐突に話を変えないでください。{shift}{work}{self_job}{ws_txt}{close}")


# ── 主話者が聞かれずに自分の職業を言う（Stage2 のゲート。2026-10-29）──────────
# 指示（上の self_job）は破られる前提で検査する。試走20対話で、主話者が聞かれずに研究・大学の
# 話をしたのが 30%（v1 の1万対話は 58%）。うち1件は第1発話の名乗り「大学で研究をして
# おります、イノウエです」。序盤（EARLY_WINDOW ターン）だけを見る。後半に話の流れで
# 触れるのは自然なので通す。
EARLY_WINDOW = 12
_SELF_JOB = re.compile(r"研究|ＡＩ|AI|ロボット|学会|論文|大学の(?:先生|教員|仕事)"
                       r"|大学で(?:働|勤|教)|大学に勤")
# 相手が尋ねた・自分の仕事を話した（＝主話者が答える・返すのは自然）
# ★2026-10-29 「スーパーの店員をしています」を拾えず、仕事から入る対話が7回空回りした
#   （相手が先に職業を言い、主話者が返すのは狙いどおりの形）。「〜をしています」型も数える。
_PARTNER_WORK = re.compile(r"仕事|働い|勤め|勤務|職場|会社|ご職業|何をされて|何されて|何してる"
                           r"|学生|研究|店員|社員|職員|(?:を|で)(?:し|やっ)て(?:い|おり)?ます")


# 相手が、主話者から聞いていないのに主話者の職業を知っている（メタデータの role の漏れ）。
# ★3回目: 「イノウエさんは、研究のお仕事を、されているそうですね。」
_KNOWS_JOB = re.compile(r"研究(?:の)?(?:お仕事|職)|研究(?:を)?(?:されて|なさって)|研究者(?:さん|の方)"
                        r"|大学の先生|大学に(?:お)?勤め")


def partner_knows_job(turns: list, main_name: str, need_name: bool = False) -> int | None:
    """主話者が職業を言う前に、相手が主話者の職業を言い当てたターンの index（無ければ None）。

    need_name: 主話者の名前を含む発話だけを見る（傾聴 al。★2026-11-02 試走2: 相手が自分の昔の仕事を
    「化学メーカーで研究職をしていた」と話したのを拾い、7回空回りした。相手が自分の話をたくさんする型）。
    """
    if not main_name:
        return None
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        if str(t.get("speaker", "")) == main_name:
            if _SELF_JOB.search(u):
                return None
            continue
        if _KNOWS_JOB.search(u) and (not need_name or main_name in u):
            return i
    return None


def early_self_job(turns: list, main_name: str, window: int = EARLY_WINDOW,
                   skip_echo: bool = False) -> int | None:
    """序盤に主話者が、相手に促されずに自分の研究・職業の話をしたターンの index。

    相手が先に仕事の話をした（尋ねた・自分の職業を言った）あとなら通す。無ければ None。
    skip_echo: 当たった語が相手の直前の発話にもあれば通す（傾聴 al。★2026-11-02 試走2: 相手の
    「お掃除ロボット」を聞き返した「お掃除ロボットですか？」を拾った）。
    """
    if not main_name:
        return None
    prev = ""
    for i, t in enumerate(turns[:window]):
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        if str(t.get("speaker", "")) != main_name:
            if _PARTNER_WORK.search(u):
                return None
            prev = u
            continue
        m = _SELF_JOB.search(u)
        if m and not (skip_echo and m.group(0) in prev):
            return i
    return None


# 主話者が自分の職業をはっきり言う文（「私は大学でロボットの研究をしてます」）。v1 の固定化そのもの。
# ★2026-10-29 再評価: 序盤（EARLY_WINDOW）の外で、ジムの話の途中に唐突に言った例があった。
_JOB_STATEMENT = re.compile(r"大学で[^。？]{0,15}研究(?:を|して|しま|して)|研究者(?:です|をして)"
                            r"|(?:ロボット|ＡＩ|AI|会話する)[^。？]{0,12}研究(?:を|して|しま)")


def unprompted_job_statement(turns: list, main_name: str) -> int | None:
    """主話者が、直前2ターンの相手の発話に仕事の話が無いのに自分の職業をはっきり言ったターンの index。"""
    if not main_name:
        return None
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or str(t.get("speaker", "")) != main_name:
            continue
        if not _JOB_STATEMENT.search(str(t.get("utterance", ""))):
            continue
        prev = [str(x.get("utterance", "")) for x in turns[max(0, i - 3):i]
                if isinstance(x, dict) and str(x.get("speaker", "")) != main_name][-2:]
        if not any(_PARTNER_WORK.search(u) for u in prev):
            return i
    return None


def topic_plan_of(metadata: dict) -> dict | None:
    """メタデータの話題の指定（無ければ None）。質問と答え（style: qa）・傾聴（style: al）の指定も返す。"""
    plan = (metadata or {}).get("topic_plan")
    if not isinstance(plan, dict) or not plan.get("topics"):
        return None
    if plan.get("style") == QA_STYLE:
        return plan if plan.get("questions") else None
    if plan.get("style") == AL_STYLE:
        return plan if plan.get("episodes") else None
    # 雑談＋重なりの場面（style: rp）は雑談の plan に overlaps を足したもの（雑談の後処理・ゲートがそのまま効く）
    if plan.get("style") == RP_STYLE:
        return plan if plan.get("opening") in OPENING_TEXT else None
    if plan.get("opening") in OPENING_TEXT:
        return plan
    return None


def make_deck(cfg: dict, genres: list[str]):
    """config の metadata.topic_plan から配り手を作る。style: qa なら qa_plan.QADeck、al なら al_plan.ALDeck。"""
    style = (cfg or {}).get("style")
    if style == QA_STYLE:
        from qa_plan import QADeck
        return QADeck(cfg, genres)
    if style == AL_STYLE:
        from al_plan import ALDeck
        return ALDeck(cfg, genres)
    if style == RP_STYLE:
        from rp_plan import RPDeck
        return RPDeck(cfg, genres)
    if style not in (None, "", "chat"):
        raise ValueError(f"metadata.topic_plan.style の未知の値: {style!r}"
                         f"（使えるのは chat / {QA_STYLE} / {AL_STYLE} / {RP_STYLE}）")
    return TopicDeck(cfg, genres)
