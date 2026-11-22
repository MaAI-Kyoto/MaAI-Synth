"""質問と答えが中心の対話（qa）の、質問の組と順番（config: metadata.topic_plan に style: qa）。

★2026-10-30 追加（ユーザ依頼「ft_v2 をベースに qa_v1 を作る。ユーザ側の質問にシステム（inoue）が
答えるのがメイン。流れが自然になるよう、挨拶や井上からの聞き返しも織り交ぜる。ただしシステムが
質問に答えるのがメインであることを忘れない」）。config_qa_v1_gemma_inoue.yaml。

ft_v2 の topic_plan（話の入り方＋話題2〜3個の順番）と同じく、対話ごとに code 側で引いて
Stage1 プロンプトの末尾に書き、Stage2 には同じものを【質問と答えの流れ】段落で渡す。
引くもの:
  - 相手（ユーザ側。speakers の2人目）がする質問の数（2〜4）と、各質問の種類×話題の順番
    種類 = 本人のこと / 知識 / おすすめ / 相談 / 意見（QA_KIND_TEXT）
  - 挨拶のあと、どう質問に入るか（QA_OPENING_TEXT）
  - 主話者からの聞き返しの回数（0〜2。ASK_BACK_DEFAULT）
  - 質問の1つを「主話者があまり詳しくないこと」にするか（unsure。知ったかぶりをしない答え方）
  - 挨拶・名乗り・締め方（topic_plan の GREETING_TEXT / NAMING_TEXT / CLOSING_TEXT をそのまま使う）
メタデータの topic_plan に {"style": "qa", "questions": [...], "topics": [...], ...} で残る。
topics（質問の話題の並び）は雑談の topic_plan と同じキーなので、DiversityTracker の話題の
数え方・genre の指定はそのまま効く。

「主話者が答えるのが中心」は指示だけに頼らず、Stage2 のゲート qa_shape（qa_shape_problem）で
数えて作り直す: 相手の質問の数・主話者の問いかけの数・主話者の発話量の割合。
"""

import random
import re
from collections import Counter

from topic_plan import (CLOSING_COMMON, CLOSING_DEFAULT, CLOSING_TEXT, GREETING_DEFAULT,
                        GREETING_TEXT, KINKYO_DEFAULT, KINKYO_TEXT, NAMING_DEFAULT, NAMING_TEXT,
                        POOL_WINDOW, QA_STYLE, _NOT_FIRST, _closing_line, _greeting_text,
                        _weights, fits_age, topic_group)

# 挨拶のあと、どう質問に入るか。{partner} は質問する側（相手）、{kinkyo} は近況の中身。
# ★台詞の例は書かない（topic_plan の試走で、例に書いた言い回しがそのまま繰り返された）。
QA_OPENING_TEXT = {
    "すぐ質問": "{partner}が前置きなしで1つめの質問をする",
    "前置きして": "{partner}が、それを聞いてみたい理由（最近気になっている・迷っている など）を"
                  "ひと言添えてから1つめの質問をする",
    "雑談から": "{kinkyo}の軽い雑談をひと往復し、その流れで{partner}が1つめの質問をする",
}
QA_OPENING_DEFAULT = {"すぐ質問": 0.30, "前置きして": 0.45, "雑談から": 0.25}

# 質問の種類。{main} は答える側（主話者）、{partner} は質問する側。
QA_KIND_TEXT = {
    "本人のこと": "{main}自身の好みややり方、経験を尋ねる",
    # ★2026-10-31 50対話: 「コツとか、仕組みみたいなの、あるかな」「効率的な対策の仕組みとか、知ってる？」と、
    #   この文面の「仕組み」が相手の質問にそのまま写った（12/50）。
    "知識": "{main}が知っていそうなこと（違い・やり方・コツなど）を尋ねる",
    "おすすめ": "おすすめや選び方を尋ねる",
    "相談": "{partner}が自分の困りごとや迷いを話して、どうしたらいいか{main}の考えを求める",
    "意見": "そのことを{main}がどう思うか、考えを尋ねる",
}
QA_KIND_DEFAULT = {"本人のこと": 0.25, "知識": 0.20, "おすすめ": 0.20, "相談": 0.20, "意見": 0.15}
QUESTIONS_DEFAULT = {2: 0.25, 3: 0.50, 4: 0.25}     # 1対話の質問の数（聞き足しは数えない）
ASK_BACK_DEFAULT = {0: 0.25, 1: 0.45, 2: 0.30}      # 主話者から相手への聞き返しの回数の上限
UNSURE_DEFAULT = 0.3                                # 質問の1つを「主話者があまり詳しくない」にする割合

# 種類と話題の組み合わせで外すもの（話題のまとまり＝topic_plan.topic_group）。
#   本人のこと×家族: 主話者の家族のことは人物設定に無い。答えると事実を作ることになり、
#     その事実を学習したモデルが「自分のこと」として話す（main-speaker-persona の所見）。
#   思い出（失敗談・若い頃の話など）は本人のことにしか合わない。
#     ★2026-10-30 ペルソナを inoue_ft_v6_001〜005 にしたあとも外したまま（子どもの人数は書いてあるが、
#     年齢や性別を聞かれると対話ごとに違う答えを作る）。
#   出身地・住まいは、初回試走の inoue_qa_3 に書いていなかったので外していたが、inoue_ft_v6_001〜005 は
#   5つとも「九州出身・京都に長く住む」と書いてあるので配る（2026-10-30）。
#   知識×住まい・出身地: 「知識:実家暮らし」が「イノウエさんは実家暮らしですか？」という本人への質問になり、
#     「はい、実家暮らしです」と作り話をした（初回試走 29a607a7）。知識として聞く中身が無い。
_KIND_GROUP_SKIP = {
    "本人のこと": {"家族"},
    "知識": {"思い出", "家族", "住まい", "出身地"},
    "おすすめ": {"思い出", "家族"},      # ★2026-10-31 試走5「おすすめ:兄弟姉妹」→「兄弟とか、いらっしゃいますか？」
    "相談": {"思い出"},
    # ★2026-10-31 50対話: 「意見:子どもの成長」→「お子さんの成長について、親としてどう思われますか？」と、
    #   相手が知らないはずの子どもを前提に聞いた（「意見:孫の話」も同じ型）。家族は相談（相手の家族の話）だけに残す。
    "意見": {"思い出", "家族"},
}
# 本人のことに振らない話題（30代の主話者に孫・定年・子育て・副業・転職を聞くと経歴を作らせる）
_SELF_SKIP = re.compile(r"孫|子ども|定年|家族|兄弟|副業|転職")
# 相手の年齢で外す（topic_plan.fits_age）のは、相手の状況が前提になる種類だけ。
# 本人のこと・意見は主話者の話なので、77歳の相手が「学生生活」を聞いてもよい。
_AGE_KINDS = {"知識", "おすすめ", "相談"}
# 相手の人物カードの趣味を差し込んでよい種類（next_plan の card_hobby）
_HOBBY_KINDS = {"相談", "意見"}
# 「主話者があまり詳しくない」にしてよい種類。相談・意見に当てると「読書については詳しくないんですけど」
# のように、詳しさと関係ない問いに前置きだけが付いた（初回試走 e170652f）。
_UNSURE_KINDS = {"知識", "おすすめ"}
# ★2026-10-30 試走4: おすすめ:兄弟姉妹 に「詳しくない」を当てたら「実は私は兄弟がいないので」と家族の事実を作った。
_UNSURE_SKIP_GROUPS = {"家族", "住まい", "出身地"}
# ★2026-10-31 50対話: ペルソナに無い活動を、主話者が趣味や習慣にしていることにした（4/50。サウナに月に数回・
#   庭で野菜を育てている・手芸を自分でする・キャンプにたまに行く）。「人物設定に無い趣味を持たせない」の一般的な
#   指示だけでは、「よく行かれるんですか？」と聞かれると経験を作る。そこで、活動の話題がペルソナの本文に
#   出てこなければ、質問ごとに no_habit を付けて「イノウエは○○を趣味や習慣にしていない」と書く。
#   対象は活動のまとまりだけ（睡眠・甘いもの・方言のような本人の性質は、類推で答えてよいので付けない）。
_HABIT_GROUPS = {"本", "映像", "音楽", "ゲーム", "演芸", "手仕事", "園芸", "料理", "飲み物",
                 "運動", "アウトドア", "スポーツ観戦", "温泉"}
# 活動のまとまりでも、誰でもふだんすること・状態の話題には付けない
_HABIT_EXEMPT = re.compile(r"朝ごはん|お弁当|運動不足")
# 話題 → ペルソナの本文にこれがあれば「している」とみなす（無い話題は話題の語そのもの）
_HABIT_EVIDENCE = [
    (r"登山", r"登山|山に登|山へ|ハイキング|大文字山|伏見稲荷"),
    (r"サッカー|スポーツ観戦|応援しているチーム", r"サッカー|観戦|スタジアム|サンガ|サガン|応援"),
    (r"運動習慣", r"運動|ハイキング|ランニング|ジョギング|自転車|ジム|散歩"),
    (r"ウォーキング", r"ウォーキング|散歩|ハイキング"),
    (r"ジム通い", r"ジム"),
    (r"写真|カメラ", r"写真|カメラ"),
    (r"絵を描く", r"絵を描|スケッチ"),
    (r"お菓子作り", r"お菓子を?作"),
    (r"パン作り", r"パンを?(?:作|焼)"),
    (r"料理|自炊", r"料理|自炊|食材"),
    (r"家庭菜園", r"菜園|野菜を育|畑"),
    (r"園芸", r"園芸|ガーデニング|植物を育|庭"),
    (r"観葉植物", r"観葉植物"),
    (r"お酒|日本酒", r"酒|ビール|ワイン"),
    (r"読書|好きな作家", r"読書|本を読|小説|作家"),
    (r"好きなアーティスト", r"アーティスト|バンド|歌手"),
    (r"配信サービス|動画配信", r"配信"),
    (r"舞台・演劇", r"舞台|演劇|芝居"),
]
_HABIT_EVIDENCE = [(re.compile(t), re.compile(e)) for t, e in _HABIT_EVIDENCE]


def persona_has_habit(topic: str, persona: str) -> bool:
    """活動の話題が主話者のペルソナの本文に出てくるか（出てこなければ、主話者はそれをしていない）。"""
    for rx, ev in _HABIT_EVIDENCE:
        if rx.search(topic):
            return bool(ev.search(persona))
    return topic in persona


def mark_habits(plan: dict, persona: str) -> dict:
    """ペルソナに無い活動の質問に no_habit を付ける（ペルソナが無ければ何もしない）。"""
    if not persona:
        return plan
    for q in plan.get("questions") or []:
        t = q.get("topic", "")
        if (topic_group(t) in _HABIT_GROUPS and not _HABIT_EXEMPT.search(t)
                and not persona_has_habit(t, persona)):
            q["no_habit"] = True
    return plan

# ── ゲート（qa_shape）の閾値 ────────────────────────────────────────────
# 主話者の発話量（実質文字数）の割合の下限。雑談の ft_v2 試走（output_ft_v2_gemma_inoue_001 の20対話）は
# 中央値 0.45（0.31〜0.73）・主話者の問いかけ 中央値1回（最大3）・相手の質問 中央値0回（同じ数え方）。
QA_MAIN_SHARE_MIN = 0.5
QA_ASK_SLACK = 1          # 主話者の問いかけは ask_back＋これまで（話を確かめる問いの分）
QA_Q_MIN_CORE = 8         # 実質この文字数未満の「？」は数えない（「ジムですか？」「本当ですか？」は
                          # 聞き返し・驚きで、質問・聞き返しの回数には入れない）
# 主話者のぼかした答え（分からない・覚えていない・人による・詳しくない）。★2026-10-30 ユーザ指定で
# 「1対話につき1回程度まで」。指示は QA_VAGUE_MAX 回、ゲートはそれを1回超えたもの（＝3回目）から落とす
# （正規表現は「人によって効く成分が違う」のような中身のある答えも拾うので、1回分の余裕を持たせる）。
# 試走1・2（ぼかしを促す指示があった版）の分布: 0回 24・1回 13・2回 3（40対話）。
QA_VAGUE_MAX = 1
_VAGUE_RE = re.compile(r"詳しくな|分からな(?:い|く)|わからな(?:い|く)|覚えて(?:い)?な|人によ(?:る|って|りけり)"
                       r"|一概に|何とも言え|はっきり(?:とは)?(?:言え|分から|覚え)|ケースバイケース")
# 相手の「？」の無い頼みごと・打ち明けも質問に数える。★2026-10-30 試走2: 相談は「方言を直すべきかなって、
# 迷ってるんです。」の形で切り出されることが多く、「？」だけ数えていたら1対話で7回作り直した。
# ★2026-10-31 試走6: 主話者の問いかけを発話全体の長さで数えていたので、「筋トレ、ですか？うーん、数回…」
#   （短い聞き返しのあとに答える）と、相談への提案「食材を一つ決めて、回してはどうですか？」が問いかけに
#   入っていた（試走6の30対話で主話者の8字以上の問い24件のうち提案7件）。聞き返し0回の対話はこれで
#   7回作り直して上限に達した。主話者は「？」で終わる文ごとに数え、提案の形は数えない。
_SUGGEST_RE = re.compile(r"(?:たら|ては|では|のは|のも|とか|って)、?(?:どう|いかが)(?:ですか|でしょうか|でしょう)？$")
_ASK_RE = re.compile(r"迷って|悩んで|困って|どうしたら|どうすれば|知りたく|聞きたく"
                     r"|教えて(?:ください|もらえ|いただけ|ほしい)")


def _core(u: str) -> int:
    return len(re.sub(r"[、。？！!?\s]", "", u or ""))


class QADeck:
    """対話ごとに「質問の組と順番」を配る（topic_plan.TopicDeck と同じ配り方）。

    種類・数・入り方・聞き返しの回数は、発注の少ないものから目標比率に寄せて配る。
    話題はシナリオの genres から、発注の少ない順の上位 POOL_WINDOW 件を母数にして引く。
    """

    def __init__(self, cfg: dict, genres: list[str]):
        cfg = cfg or {}
        self.pool = [g for g in (genres or []) if g]
        if not self.pool:
            raise ValueError("metadata.topic_plan（style: qa）はシナリオの genres が要ります")
        self.rng = random.Random(int(cfg.get("seed", 0)))
        # ★キー名は qa_opening_mix（雑談の opening_mix と分ける）。qa の config は ft_v2 を継承するので、
        #   同じキーだと deep-merge で雑談のラベル（話題から 等）と混ざる。挨拶・近況・締め方・名乗り方は
        #   雑談と同じラベルなので、同じキーのまま継承させる。
        self.opening_mix = self._labels(cfg, "qa_opening_mix", QA_OPENING_DEFAULT, QA_OPENING_TEXT)
        self.kind_mix = self._labels(cfg, "kind_mix", QA_KIND_DEFAULT, QA_KIND_TEXT)
        self.topics_mix = _weights(cfg.get("questions"), QUESTIONS_DEFAULT, key=int)
        self.ask_back_mix = _weights(cfg.get("ask_back"), ASK_BACK_DEFAULT, key=int)
        self.unsure = float(cfg.get("unsure", UNSURE_DEFAULT) or 0.0)
        self.card_hobby = float(cfg.get("card_hobby", 0.0) or 0.0)
        # 挨拶・近況・締め方・名乗り方は雑談の topic_plan と同じ文面。qa では常に配る。
        self.extra_mix = {
            "greeting": self._labels(cfg, "greeting_mix", GREETING_DEFAULT, GREETING_TEXT),
            "kinkyo": self._labels(cfg, "kinkyo_mix", KINKYO_DEFAULT, KINKYO_TEXT),
            "closing": self._labels(cfg, "closing_mix", CLOSING_DEFAULT, CLOSING_TEXT),
            "naming": self._labels(cfg, "naming_mix", NAMING_DEFAULT, NAMING_TEXT),
        }
        self.extra_issued = {k: Counter() for k in self.extra_mix}
        self.opening_issued = Counter()
        self.count_issued = Counter()
        self.kind_issued = Counter()
        self.ask_issued = Counter()
        self.topic_counts = Counter()

    @staticmethod
    def _labels(cfg: dict, key: str, default: dict, texts: dict) -> dict:
        mix = _weights(cfg.get(key), default)
        unknown = set(mix) - set(texts)
        if unknown:
            raise ValueError(f"topic_plan.{key} の未知のラベル: {sorted(unknown)}"
                             f"（使えるのは {sorted(texts)}）")
        return mix

    def describe(self) -> str:
        """起動時のログに出す設定の要約。"""
        return (f"質問と答え（style: qa）。話題 {len(self.pool)}種、質問の数 {self.topics_mix}、"
                f"種類 {self.kind_mix}、入り方 {self.opening_mix}、主話者の聞き返し {self.ask_back_mix}、"
                f"詳しくない質問 {self.unsure}"
                + (f"、カードの趣味を使う割合 {self.card_hobby}" if self.card_hobby else ""))

    def _least(self, issued: Counter, mix: dict, exclude=()):
        """目標比率に対していちばん不足しているラベル（exclude はなるべく避ける）。

        ★2026-10-31 不足の比べ方に乱数を混ぜた（issued+1 → issued+一様乱数）。決まった順に配ると、比率の
        組が同じ項目どうしが同じ周期で回り、入り方と聞き返しの回数が組で固定されていた（3000対話の模擬で
        すぐ質問→聞き返し2・前置き→1・雑談から→0 が87%、Cramér's V 0.79。試走5の20対話では20/20）。
        乱数入りでも各ラベルの数は目標から1つ程度しかずれない（V 0.79→0.12、20本の配分は変わらず）。"""
        cands = [k for k in mix if k not in exclude] or list(mix)
        return min(cands, key=lambda k: (issued[k] + self.rng.random()) / mix[k])

    def add_used(self, plan: dict):
        """既存メタデータの plan を発注数に数える（追記しても比率を保つ）。"""
        if not plan or plan.get("style") != QA_STYLE:
            return
        if plan.get("opening") in self.opening_mix:
            self.opening_issued[plan["opening"]] += 1
        qs = plan.get("questions") or []
        if len(qs) in self.topics_mix:
            self.count_issued[len(qs)] += 1
        for q in qs:
            if q.get("kind") in self.kind_mix:
                self.kind_issued[q["kind"]] += 1
        if plan.get("ask_back") in self.ask_back_mix:
            self.ask_issued[plan["ask_back"]] += 1
        for k, mix in self.extra_mix.items():
            if plan.get(k) in mix:
                self.extra_issued[k][plan[k]] += 1
        for t in plan.get("topics") or []:
            self.topic_counts[t] += 1

    def _fits(self, kind: str, topic: str, age) -> bool:
        g = topic_group(topic)
        if g in _KIND_GROUP_SKIP.get(kind, ()):
            return False
        if kind == "本人のこと" and _SELF_SKIP.search(topic):
            return False
        if kind in _AGE_KINDS and not fits_age(topic, age):
            return False
        return True

    def next_plan(self, card: dict | None = None, persona: str = "") -> dict:
        """persona（主話者のペルソナの本文）を渡すと、ペルソナに無い活動の質問に no_habit を付ける。"""
        opening = self._least(self.opening_issued, self.opening_mix)
        self.opening_issued[opening] += 1
        n = self._least(self.count_issued, self.topics_mix)
        self.count_issued[n] += 1

        # 種類: 1対話の中ではなるべく重ねない
        kinds = []
        for _ in range(n):
            k = self._least(self.kind_issued, self.kind_mix, exclude=kinds)
            self.kind_issued[k] += 1
            kinds.append(k)

        age = (card or {}).get("age")
        cands = sorted(self.pool, key=lambda t: (self.topic_counts[t], self.rng.random()))
        window = cands[:POOL_WINDOW]
        topics, groups = [], set()
        for k in kinds:
            pick = None
            for src in (window, cands):          # 母数が足りないときだけ全体から探す
                for t in src:
                    g = topic_group(t)
                    if t in topics or g in groups or not self._fits(k, t, age):
                        continue
                    # 仕事と学校は同じ対話に2つ入れない（topic_plan.next_plan と同じ）
                    if g in _NOT_FIRST and groups & _NOT_FIRST:
                        continue
                    pick = t
                    break
                if pick:
                    break
            if pick is None:                     # それでも無ければまとまりの重複を許す
                pick = next(t for t in cands if t not in topics)
            topics.append(pick)
            groups.add(topic_group(pick))
        questions = [{"kind": k, "topic": t} for k, t in zip(kinds, topics)]
        # 仕事・学校の質問は1つめにしない（序盤が職業の話になる型を作らない。topic_plan と同じ）
        for _ in range(len(questions) - 1):
            if topic_group(questions[0]["topic"]) not in _NOT_FIRST:
                break
            questions.append(questions.pop(0))
        # 話題の1つを相手の人物カードの趣味にする（種類はそのまま）。相手の趣味なので、相手の側の
        # 話になる種類（相談・意見）の質問にだけ差し込む。本人のこと・知識・おすすめに当てると
        # 「イノウエさんの和菓子レシピ研究のやり方は？」のように、主話者が相手の趣味の経験を作り話する。
        hobbies = [h for h in ((card or {}).get("hobbies") or [])
                   if topic_group(h) not in {topic_group(q["topic"]) for q in questions}]
        slots = [i for i, q in enumerate(questions) if q["kind"] in _HOBBY_KINDS]
        if hobbies and slots and self.rng.random() < self.card_hobby:
            i = self.rng.choice(slots)
            questions[i] = {"kind": questions[i]["kind"], "topic": self.rng.choice(hobbies)}
        for q in questions:
            self.topic_counts[q["topic"]] += 1

        ask_back = self._least(self.ask_issued, self.ask_back_mix)
        self.ask_issued[ask_back] += 1
        plan = {"style": QA_STYLE, "opening": opening,
                "topics": [q["topic"] for q in questions],
                "questions": questions, "ask_back": ask_back}
        # 主話者があまり詳しくない質問（知識・おすすめだけ）
        cand = [i for i, q in enumerate(questions) if q["kind"] in _UNSURE_KINDS
                and topic_group(q["topic"]) not in _UNSURE_SKIP_GROUPS]
        if cand and self.rng.random() < self.unsure:
            plan["unsure"] = self.rng.choice(cand)
        for k, mix in self.extra_mix.items():
            if k == "kinkyo" and opening != "雑談から":
                continue
            if k == "naming" and plan.get("greeting") == "名乗らない":
                continue
            v = self._least(self.extra_issued[k], mix)
            self.extra_issued[k][v] += 1
            plan[k] = v
        mark_habits(plan, persona)
        return plan


# ── 文面 ───────────────────────────────────────────────────────────────────

def _question_list(plan: dict, main: str, partner: str) -> str:
    items = []
    for i, q in enumerate(plan["questions"], 1):
        kind = QA_KIND_TEXT.get(q.get("kind"), "尋ねる").format(main=main, partner=partner)
        # ★2026-10-31 試走6: 初めの文面「経験はあっても数回程度」「趣味や習慣にしていません」がそのまま写り、
        #   主話者の「数回、行ったことはあります」が18/30対話・相手の「数回だけですか？」が5/30対話になった
        #   （作り話は0になった）。言い方を指定しない文面にする。
        habit = (f"（{main}は「{q['topic']}」をふだんはしていません。よくする・続けている・自分でも作る、"
                 "とは言わないでください。したことがあるかどうかと、その言い方は話題に合わせて自由に決め、"
                 "好みや考えは類推で答えてかまいません）"
                 if q.get("no_habit") else "")
        items.append(f"{i}) 「{q['topic']}」について、{kind}{habit}")
    return " → ".join(items)


def _opening_text(plan: dict, partner: str) -> str:
    kinkyo = KINKYO_TEXT.get(plan.get("kinkyo"), "季節のことや最近あったことなど")
    after = QA_OPENING_TEXT[plan["opening"]].format(partner=partner, kinkyo=kinkyo)
    if plan.get("greeting") in GREETING_TEXT:
        return f"{_greeting_text(plan)}。そのあと、{after}"
    return f"挨拶と短い名乗りを交わしたら、{after}"


def _ask_back_text(n: int, main: str, partner: str) -> str:
    if n <= 0:
        return (f"{main}からは質問しないでください（{partner}の質問への答えと、"
                f"{partner}の話への短い受け止めだけにする）。")
    return (f"{main}から{partner}に聞き返す（同じことを{partner}にも尋ねる・相談の事情を確かめる など）"
            f"のは対話全体で{n}回までにし、{partner}の答えを短く受けたら、また{partner}が質問する"
            f"流れに戻してください。{main}が質問攻めにしたり、聞き役に回ったりしないでください。")


def stage1_qa_text(plan: dict) -> str:
    """Stage1 プロンプトに入れる「質問の組と順番」の段落。"""
    main, partner = "1人目", "2人目"
    n = len(plan["questions"])
    unsure = plan.get("unsure")
    close = _closing_line(plan, "2人目")
    if close:
        close_item = (f"最後の項目で終わり方（{close}。{CLOSING_COMMON}）を書いてください。\n"
                      if plan.get("closing") != "締めない" else
                      f"最後の項目は最後の質問の続き（{close}）にしてください。\n")
    else:
        close_item = "最後の項目で軽い着地を書いてください。\n"
    return (
        "この対話は、speakers の2人目が1人目にいろいろ質問し、1人目が答えるのが中心です。"
        "質問の組と順番を指定します:\n"
        f"- 話の入り方: {_opening_text(plan, '2人目')}\n"
        f"- 2人目の質問: {_question_list(plan, main, partner)}\n"
        + (f"- {unsure + 1}番目の質問は、1人目があまり詳しくないことにしてください"
           "（1人目は知ったかぶりをせず、知っている範囲で答える）。\n" if unsure is not None else "")
        + f"- conversation_flow は{n + 2}項目にしてください。1項目めに挨拶と上の話の入り方、"
        f"続く{n}項目で上の質問を指定の順に1つずつ（2人目が何を聞き、1人目がどう答えるか。"
        "質問の中身と答えの中身を具体的に書く）、" + close_item
        + f"- 1人目から2人目に聞き返すのは、対話全体で{plan.get('ask_back', 0)}回まで。"
        "1人目が聞き役に回る展開・2人目の身の上話が中心になる展開にしないでください。\n"
        f"- genre には「{plan['topics'][0]}」と書いてください。\n"
        "- 1項目めに、二人が互いの職業を紹介し合うやり取りを書かないでください。\n"
        # ★2026-10-30 試走2: 「研究者の視点から見てどう思うか」「研究職の働き方を質問する」と、相手が
        #   主話者の職業を最初から知っている流れを Stage1 が書き、partner_knows_job で17回作り直した。
        # ★2026-10-30 試走3: 職業以外でも「イノウエさんは、お写真も撮られるんですよね？」が出た。
        "- 2人目は1人目の職業・趣味・経歴・家族を知りません。1人目の仕事に触れる質問は、2人目がまず何の仕事を"
        "しているかを尋ねる形にし、研究者であることを知っている前提の質問（研究者の目から見て など）に"
        "しないでください。1人目の趣味や経験を、2人目が知っている前提で尋ねないでください。\n"
        # ★2026-10-30 試走3: 本人のことの質問で、ペルソナに無い趣味（模型作り・野球観戦・陶芸・写真）を
        #   主話者が持っていることにした（20対話中4）。
        "- 1人目の人物設定に無い趣味・習い事・特技を、1人目がしていることにしないでください"
        "（聞かれたら、やったことは無いと答えたうえで、人物設定にある近い経験の話や、自分ならどうするかを話す）。\n"
        # ★2026-10-31 50対話: Stage1 の答えの中身に「目標は途中で変わってもいい」が書かれ、Stage2 がそのまま使う。
        "- 1人目の答えの中身に「目標ややり方は途中で変わってもいい」を使うのは、目標の立て方や続け方の"
        "質問のときだけにしてください。\n"
        # ★2026-10-30 試走4: 「兄弟がいない」「学生の頃からずっと一人暮らし」（ペルソナは妻と子どもと暮らす）。
        "- 1人目の家族・住まい・出身は、人物設定に書いてあることだけを事実として扱い、"
        "書いていないこと（兄弟の有無など）を新しく決めないでください。\n")


def stage2_qa_block(plan: dict, main_name: str = "", partner_name: str = "") -> str:
    """Stage2 プロンプトに足す【質問と答えの流れ】段落（人物設定の段落の直後）。"""
    main = main_name or "主話者"
    partner = partner_name or "相手"
    unsure = plan.get("unsure")
    greet = (f"対話の始まりは、{_greeting_text(plan)}形にしてください"
             "（名乗りに職業や所属を付けない）。" if plan.get("greeting") in GREETING_TEXT else "")
    after = QA_OPENING_TEXT[plan["opening"]].format(
        partner=partner, kinkyo=KINKYO_TEXT.get(plan.get("kinkyo"), "季節のことや最近あったことなど"))
    # ★2026-10-31 50対話: 「あまり詳しくないんです」がほぼ同じ言い方で12対話に出た。言い方の例を散らす。
    unsure_txt = (f"{unsure + 1}番目の質問は{main}があまり詳しくないことです。知ったかぶりをせず、"
                  "詳しくないと正直に言ったうえで、知っている範囲のことや自分ならどうするかを話してください。"
                  "詳しくないことの言い方は話題に合わせて選び（やったことがない・聞いたことはあるくらい・"
                  "人から聞いた話ですけど・自分は試したことがなくて など）、「あまり詳しくないんです」の"
                  "決まった言い方にしないでください。"
                  if unsure is not None else "")
    close = _closing_line(plan, partner)
    if close:
        close = f"終わり方: {close}。" + ("" if plan.get("closing") == "締めない"
                                       else f"{CLOSING_COMMON}。")
    return (
        f"\n\n【質問と答えの流れ】この対話は、{partner}が{main}に質問し、{main}が答えるのが中心です。"
        f"{greet}挨拶のあと、{after}。"
        f"{partner}は次の順に質問します: {_question_list(plan, main, partner)}。"
        f"{main}は質問にまず短く要点から答え、理由・自分のやり方・身近な例は、{partner}の反応や"
        f"聞き足しを挟みながら2〜3ターンに分けて話してください（1ターンで説明しきらない）。"
        "答えは具体的にし、一般論や教訓でまとめないでください。"
        # ★2026-10-30 試走3: 相談12件の約半分が「完璧を目指さず、気楽に楽しめばいい」の気の持ちようで終わった
        #   （主話者1000ターンあたり あえて27・気楽/無理せず33・楽しむ18。ft_v2 は 2・3・3）。
        "相談や「どうすれば」への答えは、気の持ちよう（完璧を目指さない・気楽に・楽しめばいい など）だけで"
        "終わらせず、すぐ試せる具体的なやり方（手順・道具・目安）を少なくとも1つ言ってください。"
        # ★2026-10-30 試走4: 試行錯誤・直感 が 15→35/1000ターンに戻った（仕事・学生時代の答えに集中）。
        "「あえて」「無理せず」「気楽に」「楽しむ」「試行錯誤」「直感」は、それぞれ対話全体で1回までにしてください。"
        # ★2026-10-31 50対話: 1対話1回は守られたが、書道・転職・子どもの成長など関係の薄い相談にも出て
        #   8/50（対話をまたぐと決まり文句）。
        "目標ややり方が途中で変わってもいい、という話は、目標の立て方や続け方を聞かれたときだけにし"
        "（ほかの相談や意見の答えには使わない）、使っても対話全体で1回までにしてください。"
        # ★2026-10-30 ユーザ指定: 「ぼかすか、分からないと言う」がぼかした答えを誘うので、作らないのは
        #   細かい数字・固有名だけにした（ペルソナの avoid の「確かでない事実を作らない」もこの意味で読ませる）。
        "細かい数字・日付・値段・店や商品の固有名は出さずに、だいたいの言い方で答えてください"
        "（人物設定の「確かでない数字や事実を作らない」は、この細かい数字や固有名のことです）。"
        f"{unsure_txt}"
        f"{partner}は答えを聞いて、気になったところを短く聞き足したり、自分の場合をひと言話したり"
        "します（質問だけを並べない）。"
        # ★2026-10-30 初回試走: 相手の聞き足しが「距離を広げる、ですか？」「得意じゃない、ですか？」の
        #   くり返しばかり（1対話の中央値4回・ft_v2 は1回）。回数を実数で書く。
        f"{partner}が{main}の言葉をそのままくり返して「〜ですか？」と確かめるのは対話全体で2回までにし、"
        "ほかは知りたいことを具体的に聞く（どうやって・どれくらい・なぜ など）か、驚きや納得、"
        "自分の場合のひと言で受けてください。"
        f"{_ask_back_text(int(plan.get('ask_back', 0)), main, partner)}"
        # ★2026-10-30 試走3: 「直前の答えの言葉からつなげる」に従って、こじつけのつなぎが出た（「そういう突き詰める
        #   感じって、食べ歩きの店選びにも関係ありますか？」）。前の話が一区切りしたら話を変えてよいことにする。
        "次の質問に移るときは、前の話と無理にこじつけてつながなくてかまいません。自然につながるときはつなげ、"
        "つながらないときは、前の話がひと区切りしたところで、話を変えることを短く断ってから次を聞いてください"
        "（断り方は毎回変える。「そういえば」「ところで」は対話全体で1回まで）。"
        f"{main}の研究・大学・仕事の話は、{partner}から尋ねられたときに答えるだけにし、"
        "答えの例として自分から持ち出さないでください。"
        f"{partner}は{main}の仕事・趣味・経歴・家族をまだ何も知りません。{main}が自分で言うまでは、"
        f"研究・大学のことや、{main}の趣味・経験を、知っている前提で話したり尋ねたりしないでください。"
        # ★2026-10-30 試走2: 自分のことを聞かれた答えが、人物設定の性格の言葉の言い換えになった
        #   （「目標を柔軟に変えながら試行錯誤」「直感や価値観を重視」。主話者1000ターンあたり 直感26・
        #   試行錯誤23・大切に23。ft_v2 は 1・6・1）。
        f"{main}が自分のことを答えるときは、性格を表す言葉（試行錯誤・直感・価値観・大切にしている など）で"
        "まとめず、実際にしていること・したことを具体的に言ってください。"
        # ★2026-10-30 ユーザ指定「ペルソナにないような質問には、できるだけ類推して答える。曖昧な答えにしても
        #   大丈夫だが、曖昧な答え方は1対話につき1回程度まで」。
        "人物設定に書いていないことを聞かれても、経歴・暮らしぶり・性格から、この人ならこうだろうと"
        "類推して具体的に答えてください。"
        # ★2026-10-30 試走3: 「類推して答える」が「その趣味をやっていることにする」になった（20対話中4）。
        "ただし、人物設定に無い趣味・習い事・特技を自分がしていることにはしないでください"
        "（やったことは無いと言ったうえで、人物設定にある近い経験の話につなげるか、自分ならどうするかを答える）。"
        # ★2026-10-30 試走4: 「実は私は兄弟がいないので」「一人暮らしを、学生の頃から、ずっとしています」。
        f"{main}の家族・住まい・出身は、人物設定に書いてあることだけを事実として話し、書いていないこと"
        "（兄弟の有無など）を新しく決めないでください（聞かれたら、人物設定にある家族や暮らしの話につなげる）。"
        # ★2026-10-30 試走4: 「何か、お役に立てましたでしょうか？」「お役に立てたなら、よかったです」。
        # ★2026-10-31 50対話: 「何か、他に気になることは、ありますか？」「お役に立てれば嬉しいです」。
        f"{main}は相談員や店員ではないので、「お役に立てましたか」「お役に立てれば」「他に気になることは"
        "ありますか」のような窓口の言い方はしないでください。"
        f"分からない・覚えていない・人による、のようにはっきり答えない返しは、対話全体で{QA_VAGUE_MAX}回まで"
        + ("にしてください（上の、あまり詳しくない質問への答えがその1回です）。" if unsure is not None
           else "にしてください。")
        +
        # ★2026-10-30 初回試走: 「本人のこと:天気の話」が「外の天気、悪いですね」「結構降っていますね」になった
        #   （場所を設定しない設計なのに、その場の天気を見て話している）。
        # ★2026-10-30 試走3: 「最近、本当に暑いじゃないですか？」「この猛暑に」と季節を決めつけた。
        "その場の天気や周りの様子を、いま見ているように話さず、今の季節や最近の天気（暑い・寒い・雨が続く など）も"
        "決めつけないでください（暑さ・寒さや天気の話は、ふだんの過ごし方や好みとして話す）。"
        f"{close}")


# ── ゲート（Stage2。generate_dialogues の qa_shape）─────────────────────────

def qa_counts(turns: list, main: str) -> dict:
    """主話者・相手の実質文字数と、「？」で終わる・含む発話の数（短い聞き返しは除く）。"""
    c = {"main_chars": 0, "partner_chars": 0, "main_q": 0, "partner_q": 0, "vague": 0}
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        who = "main" if str(t.get("speaker", "")) == main else "partner"
        c[f"{who}_chars"] += _core(u)
        if who == "main":
            if any(_core(s) >= QA_Q_MIN_CORE and not _SUGGEST_RE.search(s)
                   for s in re.findall(r"[^。！？!?]*[？?]", u)):
                c["main_q"] += 1
        elif "？" in u and _core(u) >= QA_Q_MIN_CORE:
            c["partner_q"] += 1
        elif who == "partner" and _ASK_RE.search(u):
            c["partner_q"] += 1
        if who == "main" and _VAGUE_RE.search(u):
            c["vague"] += 1
    tot = c["main_chars"] + c["partner_chars"]
    c["share"] = c["main_chars"] / tot if tot else 0.0
    return c


def qa_shape_problem(turns: list, main: str, plan: dict) -> str | None:
    """質問と答えの形が崩れていれば理由の文字列（無ければ None）。

    相手の質問が指定の数に届かない／主話者の問いかけが聞き返しの上限＋QA_ASK_SLACK を超える／
    主話者の発話量の割合が QA_MAIN_SHARE_MIN 未満／主話者のぼかした答えが QA_VAGUE_MAX＋1 を超える、のどれか。
    """
    if not main or not plan or plan.get("style") != QA_STYLE:
        return None
    c = qa_counts(turns, main)
    need = len(plan.get("questions") or [])
    if c["partner_q"] < need:
        return f"相手の質問{c['partner_q']}回<{need}回"
    cap = int(plan.get("ask_back", 0)) + QA_ASK_SLACK
    if c["main_q"] > cap:
        return f"主話者の問いかけ{c['main_q']}回>{cap}回"
    if c["share"] < QA_MAIN_SHARE_MIN:
        return f"主話者の発話量{c['share'] * 100:.0f}%<{QA_MAIN_SHARE_MIN * 100:.0f}%"
    if c["vague"] > QA_VAGUE_MAX + 1:
        return f"主話者のぼかした答え{c['vague']}回>{QA_VAGUE_MAX + 1}回"
    return None
