"""傾聴（attentive listening）の対話（al）の、話の組と聞き手の動き（config: metadata.topic_plan に style: al）。

★2026-11-02 追加（ユーザ依頼「ft_v2 をベースに al_v1 を作る。傾聴タスクで、システム（inoue）は
ユーザの話を傾聴する。聞き手応答（相槌・オウム返し・共感応答・掘り下げ質問など）で理解・関心・
興味を示し、話を引き出す。自分の話も時折してよいがしすぎない。相手の話題が尽きたら、話が続く
ように適切な声かけをする」）。config_al_v1_gemma_inoue.yaml。

qa_plan.py（質問と答え）と同じく、対話ごとに code 側で引いて Stage1 プロンプトの末尾に書き、
Stage2 には同じものを【傾聴の流れ】段落で渡す。引くもの:
  - 相手（ユーザ側。話し手）が話すことの数（1〜3）と、それぞれの「種類×話題」の順番
    種類 = 最近あったこと / 続けていること / 思い出 / 気がかり / 楽しみ（AL_KIND_TEXT）
  - 挨拶のあと、どう相手の話に入るか（AL_OPENING_TEXT）
  - 相手の話が途切れる場面と、主話者の声かけの形（AL_KOEKAKE_TEXT。途切れない対話もある）
  - 主話者が自分の経験を話す回数（0〜2。SELF_TALK_DEFAULT）
  - 挨拶・名乗り・締め方（topic_plan の GREETING_TEXT / NAMING_TEXT / CLOSING_TEXT をそのまま使う）
メタデータの topic_plan に {"style": "al", "episodes": [...], "topics": [...], ...} で残る。

「主話者が聞き役」は指示だけに頼らず、Stage2 のゲート al_shape（al_shape_problem）で数えて作り直す:
主話者の発話量の割合・主話者の掘り下げの問いの数・相手から主話者への質問の数・主話者の長い発話の数。

人間の基準（ERICA 傾聴コーパスのうち人間のオペレータが聞き手を務めた49セッション。1セッション約7.7分。
聞き手＝オペレータ。相槌だけの発話はターンに数えない）:
  聞き手の発話量の割合 中央値 0.16（p10 0.10・p90 0.25）、聞き手のターンの長さ 中央値 6.5字、
  話し手のターン 中央値 28字、聞き手の相槌 15.6回/分、聞き手のターンのうち問い 24%・
  共感の語（大変・素敵・おいしそう 等）24%・自分の話（私も 等）0%・31字以上 5%。
  聞き手のターンで多い語: そうなんですね・なるほど・いいですね・そうですか・素敵ですね。
  ★ERICA の傾聴は高齢者がロボットに話す設定で、話題もオペレータが決まった言い方で振る。
    数字は「聞き手の割合はこのくらいまで下がる」目安で、そのまま目標にはしない。
"""

import random
import re
from collections import Counter

from qa_plan import mark_habits
from topic_plan import (AL_STYLE, CLOSING_COMMON, CLOSING_DEFAULT, CLOSING_TEXT, GREETING_DEFAULT,
                        GREETING_TEXT, KINKYO_DEFAULT, KINKYO_TEXT, NAMING_DEFAULT, NAMING_TEXT,
                        POOL_WINDOW, _NOT_FIRST, _greeting_text, _weights, fits_age, topic_group)

# 挨拶のあと、どう相手の話に入るか。{main} は聞き手（主話者）、{partner} は話し手（相手）。
# ★台詞の例は書かない（topic_plan の試走で、例に書いた言い回しがそのまま繰り返された）。
AL_OPENING_TEXT = {
    "相手から": "{partner}が自分から、話したいこと（下の1つめの話）を切り出す",
    "聞き手から": "{main}が、はい・いいえで終わらない聞き方で{partner}の最近のことを尋ね、"
                  "{partner}が下の1つめの話を始める",
    "近況から": "{kinkyo}の軽いやり取りをひと往復し、その流れで{partner}が下の1つめの話を始める",
}
AL_OPENING_DEFAULT = {"相手から": 0.35, "聞き手から": 0.40, "近況から": 0.25}

# 相手が話すことの種類。{partner} は話し手。
AL_KIND_TEXT = {
    "最近あったこと": "{partner}に最近あった出来事（うれしかった・驚いた・ちょっと大変だった など）を話す",
    "続けていること": "{partner}が続けていること・はまっていることを話す",
    "思い出": "{partner}の昔の思い出を話す",
    "気がかり": "{partner}がいま少し気になっていること・迷っていることを話す（深刻なものにしない）",
    "楽しみ": "{partner}がこれから楽しみにしていること・やってみたいことを話す",
}
AL_KIND_DEFAULT = {"最近あったこと": 0.30, "続けていること": 0.25, "思い出": 0.15,
                   "気がかり": 0.15, "楽しみ": 0.15}
EPISODES_DEFAULT = {1: 0.30, 2: 0.55, 3: 0.15}     # 1対話で相手が話すことの数

# 相手の話が途切れる場面と、主話者の声かけ。{main} は聞き手、{partner} は話し手。
# 「次の話へ」は話が2つ以上の対話だけ（{at} は何番めの話のあとか）。
AL_KOEKAKE_TEXT = {
    "なし": "",
    # ★2026-11-09 試走10: 例の「きっかけ・そのときの気持ち」がそのまま「そもそもどうして始めたんですか？」
    #   「泳ぎ終わった後は、どんな気分になりますか？」のような面接の問いになった（100対話の掘り下げの約半分）。
    "掘り下げ": "{partner}の話がひと区切りして言葉が途切れたところで、{main}が同じ話の中で気になったこと"
                "（そのあとどうなったか・誰と・どれくらい など、まだ聞いていないこと）を軽く尋ね、{partner}がまた話し始める",
    # ★2026-11-08 試走8で、聞き手が前の話を受けずに「休日の楽しみは、何かありますか？」「最近、体調や、気になって
    #   いることとか、ありますか？」「普段の移動は、どうされてるんですか？」と面接のように話題を変えた（20対話中10）。
    # ★2026-11-09 試走10: 「つながること（同じ場所・人・時期・気持ち）を手がかりに」が、こじつけの問いになった
    #   （「今の、迷う感じに似たこと、昔にもありましたか？」「茶道みたいに、こだわりがある食べ物とか」）。
    "次の話へ": "{at}つめの話を{partner}が話し終えて言葉が途切れたところで、{main}がその話の締めくくりを短く受けてから、"
                "次の話の入口になることを、ほかに最近あったことを聞くように広く軽く尋ね、{partner}が次の話を始める"
                "（前の話と無理につなげたり、似た気持ちの経験を尋ねたりしない。受けずにいきなり尋ねない）",
    "自分の話から": "{partner}の話がひと区切りして言葉が途切れたところで、{main}が自分の似た経験を"
                    "ひと言だけ話してから{partner}に問いを返し、{partner}がまた話し始める",
}
KOEKAKE_DEFAULT = {"なし": 0.30, "掘り下げ": 0.30, "次の話へ": 0.30, "自分の話から": 0.10}
# 締め方は雑談と同じラベル・同じ配り方。「話題に触れて」だけ、雑談の例（教えてもらった店）が傾聴に
# 合わない（相手は話した側で、何も教わっていない）ので例を替える。
AL_CLOSING_TEXT = dict(CLOSING_TEXT, 話題に触れて=(
    "{partner}が最後の話に出たこと（これからやってみたいこと・楽しみにしていること など）"
    "に触れて締めを切り出し、短い挨拶で終える"))
SELF_TALK_DEFAULT = {0: 0.35, 1: 0.45, 2: 0.20}     # 主話者が自分の経験を話す回数（上限）

# 種類と話題の組み合わせで外すもの（話題のまとまり＝topic_plan.topic_group）。
#   思い出（失敗談・若い頃の話など）のまとまりは「思い出」の種類にしか合わない。
#   続けていること×天気・住まい・出身地・家族: 「続けている天気の話」のような組になる。
_KIND_GROUP_SKIP = {
    "最近あったこと": {"思い出"},
    "続けていること": {"思い出", "天気", "住まい", "出身地", "家族", "移動", "休日", "流行"},
    "思い出": {"天気", "流行"},
    "気がかり": {"思い出"},
    "楽しみ": {"思い出"},
}
# ★2026-11-07 話題ごとの組み合わせ。試走6で「気がかり:カレー」「気がかり:好きな食べ物」（Stage1 が「好きな食べ物の悩み」と
#   ぼかした）が出た。2000プランを引くと、気がかりは132種の話題に散り、上位が 好きな食べ物・出身地・天気の話・職業のあるある。
#   気がかりにできる話題は少ないので一覧から選ぶ（深刻にしない日常の気がかり・迷い）。
_WORRY = re.compile(
    r"運動不足|運動習慣|ジム通い|食生活|甘いもの|働き方|在宅勤務|転職|副業|資格の勉強|定年後|就職活動|アルバイト|"
    r"学生生活|部活|専攻|暑さ対策|寒さ対策|花粉症|睡眠|朝型か夜型か|^健康$|肩こり|引っ越し|一人暮らし|実家暮らし|"
    r"家族の話|子どもの成長|孫の話|子育て|節約|ポイント活動|スマホとアプリ|SNS|ネット通販|収納|掃除・片付け|大掃除|"
    r"部屋の模様替え|休日の過ごし方|連休の予定|年末年始|^旅行$|国内旅行|海外旅行|お土産|近所づきあい|習い事|誕生日|"
    r"最近の買い物|車の運転|通勤|ペット|^犬$|^猫$|最近始めた趣味|家庭菜園|観葉植物|^園芸$|ボランティア|地域の行事")
#   ほかの種類は合わない話題だけ外す（楽しみ:花粉症・続けていること:誕生日・思い出:連休の予定）。
_KIND_TOPIC_SKIP = {
    "楽しみ": re.compile(
        r"天気の話|暑さ対策|寒さ対策|花粉症|睡眠|朝型か夜型か|^健康$|肩こり|運動不足|仕事の内容|職業のあるある|"
        r"働き方|在宅勤務|資格の勉強|通勤|通学|節約|苦手な食べ物|旅先での失敗|住んでいるあたり|実家暮らし|一人暮らし|"
        r"^出身地$|方言|専攻|ゼミ|就職活動|学生生活|アルバイト|近所づきあい|大掃除|掃除・片付け|収納"),
    "続けていること": re.compile(
        r"誕生日|外食の好み|苦手な食べ物|好きな食べ物|お土産|旅先での失敗|行ってみたい場所|職業のあるある|最近の買い物"),
    "思い出": re.compile(
        r"連休の予定|行ってみたい場所|最近の買い物|最近始めた趣味|定年後の過ごし方|外食の好み|朝型か夜型か|"
        r"職業のあるある|運動不足|肩こり|節約|睡眠|^健康$|ポイント活動|副業"),
}
# 相手の昔の話なら、学生時代の話題は年齢を問わない（70代の「部活」の思い出）。
_SCHOOL = re.compile(r"学生生活|専攻|ゼミ|部活|就職活動|通学|アルバイト")
# 子ども・孫の話題は、相手のカードの家族欄が「子供あり」のときだけ（未婚の相手に子育ての話をさせない）。
_KIDS = re.compile(r"子ども|子育て|孫")
# 相手の人物カードの趣味を差し込んでよい種類（相手が自分の趣味を話す）
_HOBBY_KINDS = {"最近あったこと", "続けていること", "楽しみ"}

# ── ゲート（al_shape）の閾値 ────────────────────────────────────────────
# 主話者の発話量（実質文字数）の割合の上限。人間の傾聴 0.16（p90 0.25）、雑談の ft_v2 試走 0.45。
# ★2026-11-02 0.35→0.38。試走1〜5の採用分は中央値 0.31〜0.33 に集まり、0.35 では 35〜37% の出力の作り直しが
#   1回の試走で21〜42回（試走5）。指示は「2〜3割」のまま、ゲートは外れ値（40%超など）を落とす役にする。
AL_MAIN_SHARE_MAX = 0.38
# 主話者の掘り下げの問いの数（対話全体）。人間は聞き手のターンの 24% が問い（16ターンなら約4）。
AL_MAIN_Q_MIN = 2
# ★2026-11-07 7→5。試走7の視聴「全体的にやや質問が多い」「質問を畳みかけすぎ」（指示は 3〜5回→2〜4回）。
AL_MAIN_Q_MAX = 5
# ★2026-11-07 主話者の問いのターン（「？」を含む）がこの数だけ続いたら作り直す（試走7で3〜4ターン続く対話が3本。
#   答えに詰まった相手に問いを重ねる型。指示は「問いのターンを2回続けない」）。
AL_Q_RUN_MAX = 3
# ★2026-11-07 主話者の問いのターン（「？」を含む。聞き返しも含む）の総数の上限。試走8で聞き返し「〜、ですか？」が
#   掘り下げの問いの代わりに増えた（指示は問い合わせて2〜3回。開きの問い・声かけの問いもここに入る）。
# ★2026-11-09 6→5。版e（100対話）で問いのターンが中央値4〜5回・聞き手のターンの39%（人間 28%）。指示は4回まで。
AL_Q_TURNS_MAX = 5
# ★2026-11-07 主話者の「〜ね。」で終わる受け（問いでないターン。冒頭2ターンと末尾3ターンは挨拶なので数えない）の上限。
#   試走7で中央値7回/対話（聞き手のターンの約半分）・視聴「発話の途中で〜ですねと割り込みすぎ」。指示は3〜4回。
AL_NE_REC_MAX = 6
# ★2026-11-08 「〜んですね」「〜なんですね」で終わる主話者のターン（「そうなんですね」を除く・最初の2ターンと最後の3ターンは
#   数えない）の上限。試走9b で聞き手のターンの35%（人間の傾聴 ERICA は約7%）。指示は3回まで、ゲートは余裕を見て5。
AL_NDESUNE_MAX = 5
_NDESUNE = re.compile(r"(?:ん|な)ですね[。！]*$")
_SOUNANDESUNE = re.compile(r"^(?:(?:へえ|あー|ああ|あ|なるほど)[、。]?)?そうなんですね[。！]*$")
# 主話者の長い発話（実質この字数以上）は、自分の話の回数＋1まで（人間の聞き手は 31字以上 5%）。
AL_LONG_CORE = 31
# 相手から主話者への質問は、主話者が自分の話をする回数＋これまで（「〇〇さんは？」の聞き返し）。
AL_PARTNER_Q_SLACK = 1
# 主話者が相手の直前の言葉をくり返す・言い直すだけで受けるターン（「気分で足すんですね。」）の上限。
# ★2026-11-02 試走1: 主話者のターンの40%がこの型（中央値7回/対話）で、応答がオウム返しに偏った。
#   同じ数え方で ft_v2 本番200対話 11%・人間の傾聴（ERICA）10%。指示は「3回まで」、ゲートは1回の余裕を持たせる。
# ★2026-11-02 試走2: 4 では作り直し40回（中央値は6→3.5回に下がった）。極端なものだけ落とす5に。
# ★2026-11-07 5→6。視聴「〜ですね は相手の発話の焦点をオウム返しに」で、受けを焦点のくり返しにしたため
#   （試走8で中央値 4.5回・この上限で作り直し12回）。
# ★2026-11-08 6→4。試走9で「〜んですね」のくり返しが聞き手のターンの46%になった（指示で2回までにし、ゲートは余裕を見て4）。
AL_REECHO_MAX = 4
_REECHO_FILL = re.compile(r"えっと|ええと|あの|その|なんか|まあ|うーん|こう|ちょっと")
_REECHO_END = re.compile(r"(?:ということですね|という感じ|という気持ち|なんですね|んですね|ですね|んですか|ですか"
                         r"|なんだ|んだ|ね|か|って)?[。？！]*$")
_REECHO_HEAD = re.compile(r"^(?:へえ|へー|あー|ああ|えっ|え|そう|そっか|なるほど)")
_NE_REC = re.compile(r"ね[。！]*$")
_GREETING = re.compile(r"ありがと|失礼|よろしく|はじめまして|初めまして|こちらこそ|こんにちは|どうも|お疲れ")
# ★「なんで」は「そうなんですね」「なんでしょう」に当たらないよう後ろを見る（試走1の数え直しで見つけた）
_Q_WORD = re.compile(r"どんな|どう|何|なに|なんで(?!す|しょ)|どこ|いつ|誰|だれ|どれ|どの|どちら|なぜ|きっかけ")
_SENT_Q = re.compile(r"[^。！？!?]*[？?]")


def _core(u: str) -> int:
    return len(re.sub(r"[、。？！!?\s]", "", u or ""))


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def is_reecho(utt: str, prev: str) -> bool:
    """主話者の発話が、相手の直前の発話の言葉をくり返す・言い直すだけの受けか。

    文末（〜んですね・〜ですか）・頭の感動詞・フィラーを除いた残りの2文字組の半分以上が相手の直前の発話に
    あれば当たり。疑問詞のある問い（掘り下げ）と、挨拶・お礼・締めの言い返しは数えない。
    """
    if not prev or _Q_WORD.search(utt) or _GREETING.search(utt):
        return False
    core = re.sub(r"[、。？！!?\s]", "", _REECHO_FILL.sub("", _REECHO_END.sub("", utt.strip())))
    core = _REECHO_HEAD.sub("", core)
    if len(core) < 3:
        return False
    b = _bigrams(core)
    p = _bigrams(re.sub(r"[、。？！!?\s]", "", _REECHO_FILL.sub("", prev)))
    return len(b & p) / len(b) >= 0.5


class ALDeck:
    """対話ごとに「相手が話すことの組と順番」と「聞き手の動き」を配る（qa_plan.QADeck と同じ配り方）。

    種類・数・入り方・声かけ・自分の話の回数は、発注の少ないものから目標比率に寄せて配る（乱数入り）。
    話題はシナリオの genres から、発注の少ない順の上位 POOL_WINDOW 件を母数にして引く。
    """

    def __init__(self, cfg: dict, genres: list[str]):
        cfg = cfg or {}
        self.pool = [g for g in (genres or []) if g]
        if not self.pool:
            raise ValueError("metadata.topic_plan（style: al）はシナリオの genres が要ります")
        self.rng = random.Random(int(cfg.get("seed", 0)))
        # ★キー名は al_opening_mix（雑談の opening_mix・qa の qa_opening_mix と分ける。deep-merge で混ざる）。
        self.opening_mix = self._labels(cfg, "al_opening_mix", AL_OPENING_DEFAULT, AL_OPENING_TEXT)
        self.kind_mix = self._labels(cfg, "kind_mix", AL_KIND_DEFAULT, AL_KIND_TEXT)
        self.koekake_mix = self._labels(cfg, "koekake_mix", KOEKAKE_DEFAULT, AL_KOEKAKE_TEXT)
        self.count_mix = _weights(cfg.get("episodes"), EPISODES_DEFAULT, key=int)
        self.self_mix = _weights(cfg.get("self_talk"), SELF_TALK_DEFAULT, key=int)
        self.card_hobby = float(cfg.get("card_hobby", 0.0) or 0.0)
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
        self.koekake_issued = Counter()
        self.self_issued = Counter()
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
        return (f"傾聴（style: al）。話題 {len(self.pool)}種、相手の話の数 {self.count_mix}、"
                f"種類 {self.kind_mix}、入り方 {self.opening_mix}、声かけ {self.koekake_mix}、"
                f"主話者の自分の話 {self.self_mix}"
                + (f"、カードの趣味を使う割合 {self.card_hobby}" if self.card_hobby else ""))

    def _least(self, issued: Counter, mix: dict, exclude=()):
        """目標比率に対していちばん不足しているラベル（乱数入り。qa_plan.QADeck._least の注記）。"""
        cands = [k for k in mix if k not in exclude] or list(mix)
        return min(cands, key=lambda k: (issued[k] + self.rng.random()) / mix[k])

    def add_used(self, plan: dict):
        """既存メタデータの plan を発注数に数える（追記しても比率を保つ）。"""
        if not plan or plan.get("style") != AL_STYLE:
            return
        if plan.get("opening") in self.opening_mix:
            self.opening_issued[plan["opening"]] += 1
        eps = plan.get("episodes") or []
        if len(eps) in self.count_mix:
            self.count_issued[len(eps)] += 1
        for e in eps:
            if e.get("kind") in self.kind_mix:
                self.kind_issued[e["kind"]] += 1
        if plan.get("koekake") in self.koekake_mix:
            self.koekake_issued[plan["koekake"]] += 1
        if plan.get("self_talk") in self.self_mix:
            self.self_issued[plan["self_talk"]] += 1
        for k, mix in self.extra_mix.items():
            if plan.get(k) in mix:
                self.extra_issued[k][plan[k]] += 1
        for t in plan.get("topics") or []:
            self.topic_counts[t] += 1

    @staticmethod
    def _fits(kind: str, topic: str, card: dict | None) -> bool:
        if topic_group(topic) in _KIND_GROUP_SKIP.get(kind, ()):
            return False
        if kind == "気がかり" and not _WORRY.search(topic):
            return False
        skip = _KIND_TOPIC_SKIP.get(kind)
        if skip and skip.search(topic):
            return False
        card = card or {}
        age = card.get("age")
        if kind == "思い出" and _SCHOOL.search(topic):
            ok_age = age is None or age >= 22
        else:
            ok_age = fits_age(topic, age)
        if not ok_age:
            return False
        if _KIDS.search(topic) and card and "子供あり" not in str(card.get("marital", "")):
            return False
        return True

    def next_plan(self, card: dict | None = None, persona: str = "") -> dict:
        """persona（主話者のペルソナの本文）を渡すと、主話者がしていない活動の話に no_habit を付ける
        （主話者が自分の話をするとき、その活動の経験を作らせない。qa_plan.mark_habits）。"""
        opening = self._least(self.opening_issued, self.opening_mix)
        self.opening_issued[opening] += 1
        n = self._least(self.count_issued, self.count_mix)
        self.count_issued[n] += 1

        kinds = []
        for _ in range(n):
            k = self._least(self.kind_issued, self.kind_mix, exclude=kinds)
            self.kind_issued[k] += 1
            kinds.append(k)

        cands = sorted(self.pool, key=lambda t: (self.topic_counts[t], self.rng.random()))
        window = cands[:POOL_WINDOW]
        topics, groups = [], set()
        for k in kinds:
            pick = None
            for src in (window, cands):          # 母数が足りないときだけ全体から探す
                for t in src:
                    g = topic_group(t)
                    if t in topics or g in groups or not self._fits(k, t, card):
                        continue
                    if g in _NOT_FIRST and groups & _NOT_FIRST:
                        continue
                    pick = t
                    break
                if pick:
                    break
            if pick is None:
                pick = next(t for t in cands if t not in topics)
            topics.append(pick)
            groups.add(topic_group(pick))
        episodes = [{"kind": k, "topic": t} for k, t in zip(kinds, topics)]
        # 仕事・学校の話は1つめにしない（序盤が職業の話になる型を作らない。topic_plan と同じ）
        for _ in range(len(episodes) - 1):
            if topic_group(episodes[0]["topic"]) not in _NOT_FIRST:
                break
            episodes.append(episodes.pop(0))
        # 話の1つを相手の人物カードの趣味にする（相手が自分の趣味を話す。種類はそのまま）
        hobbies = [h for h in ((card or {}).get("hobbies") or [])
                   if topic_group(h) not in {topic_group(e["topic"]) for e in episodes}]
        slots = [i for i, e in enumerate(episodes) if e["kind"] in _HOBBY_KINDS]
        if hobbies and slots and self.rng.random() < self.card_hobby:
            i = self.rng.choice(slots)
            episodes[i] = {"kind": episodes[i]["kind"], "topic": self.rng.choice(hobbies)}
        for e in episodes:
            self.topic_counts[e["topic"]] += 1

        koekake = self._least(self.koekake_issued, self.koekake_mix,
                              exclude=() if n >= 2 else ("次の話へ",))
        self.koekake_issued[koekake] += 1
        # 「自分の話から」は声かけそのものが自分の話なので、0回は配らない
        self_talk = self._least(self.self_issued, self.self_mix,
                                exclude=(0,) if koekake == "自分の話から" else ())
        if koekake == "自分の話から":
            self_talk = max(self_talk, 1)        # self_talk に 0 しか無い config でも
        self.self_issued[self_talk] += 1
        plan = {"style": AL_STYLE, "opening": opening,
                "topics": [e["topic"] for e in episodes],
                "episodes": episodes, "koekake": koekake, "self_talk": self_talk}
        if koekake == "次の話へ":
            plan["koekake_at"] = self.rng.randrange(1, n)   # 何番めの話のあとか（1始まり）
        for k, mix in self.extra_mix.items():
            if k == "kinkyo" and opening != "近況から":
                continue
            if k == "naming" and plan.get("greeting") == "名乗らない":
                continue
            v = self._least(self.extra_issued[k], mix)
            self.extra_issued[k][v] += 1
            plan[k] = v
        # 主話者がしていない活動の印（qa の questions と同じキーで付けるため、一時的に渡す）
        marked = mark_habits({"questions": [dict(e) for e in episodes]}, persona)
        for e, m in zip(episodes, marked["questions"]):
            if m.get("no_habit"):
                e["no_habit"] = True
        return plan


# ── 文面 ───────────────────────────────────────────────────────────────────

def _closing_line(plan: dict, partner: str) -> str:
    c = plan.get("closing")
    return AL_CLOSING_TEXT[c].format(partner=partner) if c in AL_CLOSING_TEXT else ""


def _episode_list(plan: dict, main: str, partner: str, habit_note: bool = False) -> str:
    items = []
    for i, e in enumerate(plan["episodes"], 1):
        kind = AL_KIND_TEXT.get(e.get("kind"), "話す").format(main=main, partner=partner)
        habit = (f"（{main}は「{e['topic']}」をふだんはしていません。自分の話をするときも、"
                 "しているとは言わないでください）"
                 if habit_note and e.get("no_habit") and plan.get("self_talk") else "")
        items.append(f"{i}) 「{e['topic']}」について、{kind}{habit}")
    return " → ".join(items)


def _opening_text(plan: dict, main: str, partner: str) -> str:
    kinkyo = KINKYO_TEXT.get(plan.get("kinkyo"), "季節のことや最近あったことなど")
    after = AL_OPENING_TEXT[plan["opening"]].format(main=main, partner=partner, kinkyo=kinkyo)
    if plan.get("greeting") in GREETING_TEXT:
        return f"{_greeting_text(plan)}。そのあと、{after}"
    return f"挨拶と短い名乗りを交わしたら、{after}"


def _koekake_text(plan: dict, main: str, partner: str) -> str:
    k = plan.get("koekake")
    txt = AL_KOEKAKE_TEXT.get(k, "")
    if not txt:
        return (f"{partner}は自分から話を続け、話を移るときも{partner}が自分で次の話に進む"
                f"（{main}が話題を変えない）")
    return (f"{partner}の話が途切れる場面を対話の中ほどに1回入れる: "
            + txt.format(main=main, partner=partner, at=plan.get("koekake_at", 1))
            + f"。途切れる場面は、{partner}がひとつの話を言い終えて言うことが尽きたところにし"
            f"（{main}の問いに答えられずに詰まる形にはしない）、"
            f"{main}の声かけで{partner}がまた具体的に話し始めるようにする")


def _self_talk_text(n: int, main: str, partner: str) -> str:
    if n <= 0:
        return (f"{main}は自分の経験や意見を話さず、聞き手の応答だけにしてください"
                f"（{partner}に聞かれたら、ひと言で答えてすぐ{partner}の話に戻す）。")
    # ★2026-11-09 試走10: 回数を配ると、人物設定から無理に引いた自分の話（「私も子供と屋外で工作をするのが、好きなんです」）が
    #   相手の話と関係なく入り、判定で「話を奪う」と指摘された（ノルマとして読まれる）。
    return (f"{main}が自分の経験をひと言話すのは、{partner}の話とよく似た経験が人物設定にあるときだけにし、"
            f"対話全体で{n}回までにしてください（似た経験が無ければ話さない。1〜2文で済ませ、すぐ{partner}の話に戻す。"
            f"{partner}より長く話さない）。")


def stage1_al_text(plan: dict) -> str:
    """Stage1 プロンプトに入れる「相手の話の組と、聞き手の動き」の段落。"""
    main, partner = "1人目", "2人目"
    n = len(plan["episodes"])
    close = _closing_line(plan, "2人目")
    if close:
        close_item = (f"最後の項目で終わり方（{close}。{CLOSING_COMMON}）を書いてください。\n"
                      if plan.get("closing") != "締めない" else
                      f"最後の項目は最後の話の続き（{close}）にしてください。\n")
    else:
        close_item = "最後の項目で軽い着地を書いてください。\n"
    return (
        "この対話は、speakers の2人目が自分のことを話し、1人目が聞き役になって話を引き出す会話（傾聴）です。"
        "話の組と順番を指定します:\n"
        f"- 話の入り方: {_opening_text(plan, main, partner)}\n"
        f"- 2人目が話すこと: {_episode_list(plan, main, partner)}\n"
        f"- 話の途切れと声かけ: {_koekake_text(plan, main, partner)}\n"
        f"- 1人目が自分の経験を話すのは対話全体で{plan.get('self_talk', 0)}回まで。"
        "1人目は相談員やカウンセラーではないので、助言・評価・説教をしないでください。\n"
        f"- conversation_flow は{n + 2}項目にしてください。1項目めに挨拶と上の話の入り方、"
        f"続く{n}項目で2人目が話すことを指定の順に1つずつ（2人目が何を話し、1人目がどんな問いや"
        "ひと言で話を引き出すか。話の中身を具体的に書く）、" + close_item
        + f"- genre には「{plan['topics'][0]}」と書いてください。\n"
        "- 1項目めに、二人が互いの職業を紹介し合うやり取りを書かないでください。\n"
        # ★2026-11-02 試走1: scenario に「30代の大学研究者のイノウエ」と書かれ、Stage2 で相手が研究のことを尋ねた。
        "- scenario と conversation_flow に1人目の職業を書かないでください。1人目の仕事の話はこの対話には出しません。\n"
        "- 2人目は1人目の職業・趣味・経歴・家族を知りません。2人目の話の中身は、2人目の人物"
        "（下の人物設定）の暮らしや趣味に合うものにしてください。\n"
        "- 1人目の人物設定に無い趣味・習い事・経験を、1人目がしていることにしないでください。\n")


def stage2_al_block(plan: dict, main_name: str = "", partner_name: str = "") -> str:
    """Stage2 プロンプトに足す【傾聴の流れ】段落（人物設定の段落の直後）。"""
    main = main_name or "主話者"
    partner = partner_name or "相手"
    greet = (f"対話の始まりは、{_greeting_text(plan)}形にしてください"
             "（名乗りに職業や所属を付けない）。" if plan.get("greeting") in GREETING_TEXT else "")
    after = AL_OPENING_TEXT[plan["opening"]].format(
        main=main, partner=partner,
        kinkyo=KINKYO_TEXT.get(plan.get("kinkyo"), "季節のことや最近あったことなど"))
    close = _closing_line(plan, partner)
    if close:
        close = f"終わり方: {close}。" + ("" if plan.get("closing") == "締めない"
                                       else f"{CLOSING_COMMON}。")
    return (
        f"\n\n【傾聴の流れ】この対話は、{partner}が自分のことを話し、{main}が聞き役になって"
        f"{partner}の話を引き出す会話です。{greet}挨拶のあと、{after}。"
        f"{partner}は次の順に話します: {_episode_list(plan, main, partner, habit_note=True)}。"
        # ★2026-11-07 試走7の視聴「発話の途中で〜ですねと割り込みすぎ」「〜ですねは相手の発話の焦点をオウム返しに」
        #   「質問がやや多い」。数えると聞き手の「〜ね。」の受けが1対話7回（聞き手のターンの約半分）で、
        #   相手の言いかけ（〜て。〜けど。）の直後が43回。以前の「相手の話は聞き手の応答を挟みながら何ターンにも分けて」
        #   「いちばん多いのは〜でしたね・〜ですねのひと言」が相手の話を細切れにしていた。途中の相槌は Stage4 が入れるので、
        #   相手はひと区切りまで1ターンで話し、聞き手は区切りで受ける。受けは焦点のくり返し（推測・評価にしない。
        #   「勇気がいる決断でしたね」「遠かったですね」が不自然と指摘）。
        f"{partner}は、ひとつの話をひと区切り（言いたいことを言い終えるところ）まで1つのターンで話してください"
        "（2〜3文になってもよい。「〜て。」「〜けど。」のような言いかけで切って相手に渡さない）。"
        f"{partner}が話している途中の相槌（うん・へえ など）は音声にするときに入るので、台本には書きません。"
        "出来事の具体的な中身（いつ・誰と・何が起きたか）や、そのときの気持ちを話してください。"
        f"長めの発話（31字以上）は、ほとんどを{partner}が話します。"
        f"{main}が話すのは{partner}の話がひと区切りしたところだけで、{partner}の話の途中に割り込みません。"
        # ★2026-11-08 試走8で、話し手が答えの直後に「あ、そういえば」で別の話を始める（同じターンの中を含む）・
        #   「薬の話をしていたら、学生時代の就職活動を思い出しました」のようなこじつけが20対話中6。
        + (f"{partner}が次の話に移るのは、前の話を言い終えて{main}が受けたあとの{partner}のターンの頭にしてください。"
           "前の話から自然に思い出したつながりがあれば短く言ってから始め、つながりが無ければ"
           "「話は変わるんですけど」のようにひと言断って切り替える（「〜の話をしていたら〜を思い出しました」のような"
           "こじつけをしない。話を終えた同じターンの中で、すぐ「あ、そういえば」と別の話を始めない）。"
           if len(plan.get("episodes") or []) >= 2 else "")
        + f""
        # ★2026-11-07 試走8（Stage2 のみ）: 例に書いた「数グラム単位で、ですか」が写り、「効率が上がった、ですか？」
        #   「迷い込んだ、ですか？」が1対話に5回（問いの連続が最大7ターン・問いの上限超えで作り直し66回）。
        #   受けは「〜んですね」だけにし、聞き返しの「〜ですか？」は問いに数えると書く。応答の内訳も数で示す。
        # ★2026-11-08 試走8の評価（20対話の聞き手のターン約220を1つずつ読んだ。001/eval_trial9_20261108.md）:
        #   問いの3割が面接・カウンセリング調（「どのような点に」「具体的に」「どんな心境で」「どんな感覚」）。
        #   (3) の例「そのときどう思ったか」が心境の問いを呼んでいた。受けは文をまるごとくり返す長い「〜んですね」
        #   （「低糖質のスイーツに挑戦しようか迷っているんですね」）や、相手がうれしい・つらいことを言った直後の
        #   事実のくり返し（「10キロ完走したんですね」）・理由の問いただし（「どうして枯らしてしまったんですか？」）。
        #   話のまとめに一般論・評価（「泥臭い経験って、いいですよね」「その慎重さは、大切だと思います」）が16件。
        #   例の「数グラム単位」は話し手の話にまで写った（「ミリ単位で整理」「1センチ単位の調整」）。
        #   → 応答は相手が直前に伝えたこと（うれしい・つらい・意外・ふつうの説明）から選ぶ、受けは焦点の語句だけ、
        #   問いは話に出た具体的なことを話し言葉で、つらい話の直後は気持ちを先に受ける。例は形だけにする。
        # ★2026-11-08 試走9（上の版）: 「ふつうの説明なら受けで返す」が、受け（＝焦点のくり返し）を既定の応答にした。
        #   「〜んですね」で終わる聞き手のターンが 29%→46%（05 は9回続けてくり返し「香りが変わるんですね」
        #   「満足感があるんですね」…）。人間の傾聴の聞き手（ERICA）は「そうなんですね・そうですか・なるほど」の
        #   短い受けが多く、くり返しは1割ほど。ふつうの説明は短い受けで返し、くり返しは大事な語句だけ・2回までにする。
        # ★2026-11-09 試走10（100対話を読んだ。001/eval_trial10_20261109.md）: 4つの型と目安の回数を並べると、型を順番に
        #   回す機械的な応答になった（受けと問いの交互・「それは焦りますね」「それは不安になりますね」の連続・
        #   「リズムよく動かすのが、いいんですね」のような文の後半の言い直し）。問いは「どんな気分になりますか」
        #   「どうして始めたんですか」「どこに惹かれたんですか」が残った（「掘り下げ」の例の「きっかけ・気持ち」から）。
        #   人間の傾聴（ERICA）の聞き手は短い受けが37%でいちばん多く、くり返しは心が動いた語だけ（「熊に会われたんですね」）。
        #   → 型の一覧をやめ、「心が動いたところに向けたひと言」を軸にし、言い直しの受けが機械的に聞こえる理由を書く。
        f"{main}の応答は、決まった型を順番に回すのではなく、{partner}が直前に話したことのうち{main}の心が動いたところ"
        "（いちばん新しいこと・意外なこと・気持ちが出たところ）に向けて、その場で口から出るひと言にしてください。"
        # ★2026-11-09 版c: 同じ短い受けのくり返し（「そうなんですね」3回）と、気持ちを話したところへの「へえ。」だけの受け。
        f"いちばん多く使うのは短い受けで、ふつうの説明や、{partner}が話を続けそうなところはこれで十分です（目安は対話全体で4〜5回）。"
        "短い受けは同じ言い方を続けず、「そうなんですね」「そうなんですか」「そうでしたか」「なるほど」「へえ」「あー」などを"
        f"散らしてください。ただし{partner}が気持ち（うれしい・つらい・不安・懐かしい）を話したところは、短い受けだけで流さないでください。"
        f"心が動いたところでは、その出来事への{main}の反応をひと言で返してください（目安は3〜4回）: "
        "うれしい・うまくいった話には喜びや感心、失敗・つらい・不安な話には気持ちに寄りそうひと言、"
        "意外な話には驚き、おかしい話には笑い（短い感動詞から始めてもよい）。"
        f"{partner}の気持ちの語をそのまま言い直したり（「不安だったんですね」）、「それは〜ますね」「それは〜でしたね」の"
        f"同じ形を続けたり、{partner}が困っていない話に「大変でしたね」と言ったりしないでください。"
        # ★2026-11-09 版d: ここに悪い例として書いた「それはかなり焦りました。」がそのまま写った（40対話で3回）。例は書かない。
        f"{partner}の気持ちを{main}自身の気持ちのように「〜ました。」で言い切る形にもしない。"
        # ★2026-11-08 試走9: 問いが減り、掘り下げの問いの下限（AL_MAIN_Q_MIN=2）で作り直し8回。問いは目安に残す。
        # ★2026-11-09 版c: この段落の「知りたくなった具体的なこと」の「具体的」が「具体的にどんな〜ですか？」の問いに写った。
        f"問いは、{partner}の話に出たことで{main}が本当に知りたくなったこと（そのあとどうなったか・誰と・どこで・"
        "どれくらい・何を・初めてだったか）を、ふだんの話し言葉で短く尋ねてください（目安は2回）。"
        "気分・感じ方・きっかけ・理由・魅力を尋ねる問い（どんな気分になりますか・どう感じましたか・どうして始めたんですか・"
        "どんなところに惹かれたんですか・一番〜だったことは何ですか）や「具体的に」で始める問いは、面接やカウンセリングのように"
        f"聞こえるので使わないでください。{partner}がいま話したことや、もう言った気持ちを聞き直さない（「どうでしたか？」）。"
        f"{partner}が失敗・後悔・つらいことを話した直後は、問いより先に気持ちを受ける。"
        # ★2026-11-09 「はい・いいえで終わる問いにしない」を外す。人間の傾聴の聞き手は「バンクーバーは初めてですか」
        #   「怖くなかったですか」のような問いをよく使い、禁じると「どんな〜ですか」の抽象的な問いに寄る（試走10）。
        f"{partner}の語をくり返して聞き返す「〜ですか？」も問いに数え、挨拶のあとの最初の問いや声かけの問いも合わせて、"
        # ★2026-11-09 版e（100対話）: 問いのターンが1対話4.2回（聞き手のターンの39%・人間の傾聴 28%）。5回以上が46%。
        #   試走7の視聴「全体的にやや質問が多い」。校閲が問いを言い直しに替えるのを止めた分と、短い受けでターンが増えた分。
        f"{main}の「？」で終わるターンは対話全体で4回までにしてください。問いを2つ重ねた聞き方にしない。"
        f"{main}の問いのターンを2回続けず、{partner}が答えに詰まったときも問いを重ねないでください。"
        f"{partner}の文の後半をそのまま「〜んですね」「〜なんですね」で言い直す受け（「〇〇するのが好きなんですね」）は、"
        "話を聞いていない機械的な相づちに聞こえます。くり返すのは、驚いたり感心したりした語句だけを短く言うときにし、"
        "「〜んですね」「〜なんですね」で終わる応答は「そうなんですね」を除いて対話全体で2回までにしてください。"
        # ★2026-11-08 「〜ね」の受けのゲート（AL_NE_REC_MAX=6）と合わせる。共感のひと言も「ね」で終わりやすい。
        "受け・ひと言を合わせても「〜ね」で終わる応答は5回くらいまでにしてください"
        "（驚き・共感は「ね」で終わらない形も混ぜる）。"
        # ★2026-11-09 版c の判定: 「いいですね。」「散歩、いいですね。」のような評価で受けるのが1対話0.7回、浅い評価と指摘された。
        "「いいですね」「〇〇、いいですね」で受けるのは対話全体で2回までにしてください。"
        f"{main}の発話はほとんどを15字以内の短いものにし（受けにひと言を続けるときも20字くらいまで）、"
        f"{main}の話す量は対話全体の2〜3割にとどめてください。"
        f"{main}は{partner}を評価・称賛したり、助言・説教・一般論でまとめたりしないでください"
        f"（{partner}に意見を求められたときだけ、ひと言で）。"
        f"{partner}が話を締めくくったところでも、「〜っていいですよね」「〜のは大切ですね」「きっと〜」「〜が一番ですよね」"
        # ★2026-11-09 「気持ちの語を受ける」が締めの言い直し（「心に残ったんですね」「挑戦してみたいんですね」）になった。
        f"のようにまとめず、短い受けか短い共感で返してください。"
        f"{main}は{partner}の話を自分の話に持っていかず、"
        + ("下の声かけのとき以外は話題を変えません。" if plan.get("koekake") == "次の話へ" else "話題も変えません。")
        + f""
        f"{_self_talk_text(int(plan.get('self_talk', 0)), main, partner)}"
        f"{partner}から{main}に質問するのは、{main}が自分の話をする回数までにしてください"
        f"（{partner}が{main}を質問攻めにしない）。"
        f"話の途切れと声かけ: {_koekake_text(plan, main, partner)}。"
        # ★2026-11-02 試走2: 主話者が先に「お話、聞かせてくれてありがとうございます」と締めて作り直し22回。
        f"締めのお礼や挨拶は{partner}から切り出し、{main}はそれに返すだけにしてください"
        f"（{main}から話を終わらせない）。"
        "声かけの言い方は、その場の話の中身から考えてください（決まった言い方にしない）。"
        # ★2026-11-02 試走1: 終盤に相手が「イノウエさんは、大学で何を研究されてるんですか？」と聞く出力が8回
        #   （partner_knows_job で作り直し）。傾聴では主話者の仕事の話は要らないので、尋ねさせない。
        f"{partner}は{main}の仕事・趣味・経歴・家族をまだ何も知らず、{main}の仕事や研究のことは尋ねません"
        f"（{main}の仕事の話はこの対話には出さない）。"
        f"{main}が自分の話をするときは、人物設定にあることだけを話し、人物設定に無い趣味・経験・家族のことを"
        "新しく作らないでください。"
        "その場の天気や周りの様子を、いま見ているように話さず、今の季節も決めつけないでください。"
        f"{close}")


# ── ゲート（Stage2。generate_dialogues の al_shape）─────────────────────────

def _is_bare_echo(sentence: str, prev: str) -> bool:
    """相手の直前の発話の語をくり返しただけの聞き返し（「ふろふき大根ですか？」）。"""
    s = re.sub(r"^(?:え[っ]?|へえ|あー|あ)、?", "", sentence.strip())
    m = re.match(r"^(.{1,14}?)、?(?:ですか|ですね|なんですか|か|って)？$", s)
    # ★2026-11-07 「効率が上がった、ですか？」（試走8）も、読点を除いて相手の語なら聞き返し
    return bool(m) and m.group(1).replace("、", "") in (prev or "").replace("、", "")


def al_counts(turns: list, main: str) -> dict:
    """主話者・相手の実質文字数と、問いの数・主話者の長い発話の数。"""
    c = {"main_chars": 0, "partner_chars": 0, "main_q": 0, "partner_q": 0, "main_long": 0, "main_reecho": 0,
         "main_q_run": 0, "main_ne_rec": 0, "main_q_turns": 0, "main_ndesune": 0}
    prev_partner = ""
    run = 0
    n = len(turns)
    for k, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        who = "main" if str(t.get("speaker", "")) == main else "partner"
        c[f"{who}_chars"] += _core(u)
        sents = _SENT_Q.findall(u)
        if who == "main":
            is_q = "？" in u or "?" in u
            c["main_q_turns"] += is_q
            run = run + 1 if is_q else 0
            c["main_q_run"] = max(c["main_q_run"], run)
            if not is_q and 2 <= k < n - 3 and _NE_REC.search(u):
                c["main_ne_rec"] += 1
            if 2 <= k < n - 3 and _NDESUNE.search(u) and not _SOUNANDESUNE.match(u.strip()):
                c["main_ndesune"] += 1
            # 掘り下げの問い: 疑問詞を含むか、実質8字以上の問い。くり返しただけの聞き返しは数えない。
            if any((_Q_WORD.search(s) or _core(s) >= 8) and not _is_bare_echo(s, prev_partner)
                   for s in sents):
                c["main_q"] += 1
            if _core(u) >= AL_LONG_CORE:
                c["main_long"] += 1
            if is_reecho(u, prev_partner):
                c["main_reecho"] += 1
        else:
            if any(_core(s) >= 8 for s in sents):
                c["partner_q"] += 1
            prev_partner = u
    tot = c["main_chars"] + c["partner_chars"]
    c["share"] = c["main_chars"] / tot if tot else 0.0
    return c


def al_shape_problem(turns: list, main: str, plan: dict) -> str | None:
    """傾聴の形が崩れていれば理由の文字列（無ければ None）。

    主話者の発話量の割合が AL_MAIN_SHARE_MAX を超える／主話者の掘り下げの問いが AL_MAIN_Q_MIN 未満か
    AL_MAIN_Q_MAX 超／相手から主話者への質問が自分の話の回数＋AL_PARTNER_Q_SLACK を超える／
    主話者の長い発話（実質 AL_LONG_CORE 字以上）が自分の話の回数＋1 を超える／
    主話者のくり返し・言い直しだけの受けが AL_REECHO_MAX を超える／主話者の問いのターンが AL_Q_TURNS_MAX を超える／
    主話者の問いのターンが AL_Q_RUN_MAX 続く／
    主話者の「〜ね。」の受けが AL_NE_REC_MAX を超える／「〜んですね」が AL_NDESUNE_MAX を超える、のどれか。
    """
    if not main or not plan or plan.get("style") != AL_STYLE:
        return None
    c = al_counts(turns, main)
    st = int(plan.get("self_talk", 0))
    if c["share"] > AL_MAIN_SHARE_MAX:
        return f"主話者の発話量{c['share'] * 100:.0f}%>{AL_MAIN_SHARE_MAX * 100:.0f}%"
    if c["main_q"] < AL_MAIN_Q_MIN:
        return f"主話者の掘り下げの問い{c['main_q']}回<{AL_MAIN_Q_MIN}回"
    if c["main_q"] > AL_MAIN_Q_MAX:
        return f"主話者の問い{c['main_q']}回>{AL_MAIN_Q_MAX}回"
    if c["partner_q"] > st + AL_PARTNER_Q_SLACK:
        return f"相手の質問{c['partner_q']}回>{st + AL_PARTNER_Q_SLACK}回"
    if c["main_long"] > st + 1:
        return f"主話者の長い発話{c['main_long']}回>{st + 1}回"
    if c["main_reecho"] > AL_REECHO_MAX:
        return f"主話者のくり返し・言い直し{c['main_reecho']}回>{AL_REECHO_MAX}回"
    if c["main_q_turns"] > AL_Q_TURNS_MAX:
        return f"主話者の問いのターン{c['main_q_turns']}回>{AL_Q_TURNS_MAX}回"
    if c["main_q_run"] >= AL_Q_RUN_MAX:
        return f"主話者の問いが{c['main_q_run']}ターン続く"
    if c["main_ne_rec"] > AL_NE_REC_MAX:
        return f"主話者の「〜ね」の受け{c['main_ne_rec']}回>{AL_NE_REC_MAX}回"
    if c["main_ndesune"] > AL_NDESUNE_MAX:
        return f"主話者の「〜んですね」{c['main_ndesune']}回>{AL_NDESUNE_MAX}回"
    return None
