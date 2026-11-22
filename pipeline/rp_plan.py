"""rp_v1（声が重なる場面の修復・調整。topic_plan.style: rp）。

★2026-11-10 作成。ユーザ依頼「Full-Duplex の研究動向を踏まえた会話能力のうち A1〜A6 について新しいタスク rp_v1 を作る」:
  A1 ユーザの割り込みで止まる          → 割り込まれて応じる / 割り込まれて戻る
  A2 止まったあとに続けるか応えるかを選ぶ → 同上（応じる＝元の話に戻らない、戻る＝短く答えて言いかけた話に戻る）
  A3 相槌・短い反応は割り込みと見分けて話し続ける → 反応が重なる（＋Stage4 の挿入相槌）
  A4 同時に話し始めたときの修復          → 同時に話し始めて譲る / 同時に話し始めて譲られる
  A5 協調的な重なり（先取り・言葉探しの手助け・強い同意）→ 重ねて応じる
  A6 モデルから適切に割り込む            → 割り込む
  参考: Full-Duplex-Bench v1/v1.5（割り込み・ユーザの相槌）、FLEXI（モデル側の割り込み）。

雑談（ft_v2 の topic_plan）をそのまま土台にし、1対話に3〜4個の「重なりの場面」を足す。話題・挨拶・名乗り・
締め方の配り方と、それに付く後処理・ゲートは雑談と共通で効く（topic_plan_of が style: rp の plan も返す）。

台本の形（Stage2 の JSON）。ふつうのターンは {"speaker", "utterance"} のまま。
  - 言いかけて止まった発話: "utterance" に実際に口に出したところまで（文の途中）、"unsaid" に言うつもりだった残り。
  - 相手の声に重ねて話し始めたターン: "overlap" に次のどれか。
      interrupt … 相手の発話をさえぎった（直前の相手のターンは "unsaid" 付き）
      collision … 相手とほぼ同時に話し始めた（直前の相手のターンも自分のターンも "unsaid" 付きの短い言いかけ）
      coop      … 相手の話の終わりぎわに重ねて応じた（相手は言い終える）
      react     … 相手が話している途中に短く反応した（相手はそのまま話し続ける＝次は相手の続きのターン）
Stage3（passthrough）はこの2つのキーを写し、Stage4 は "unsaid" 込みの文を合成してから言いかけの位置で音声を切り
（rp_audio.py）、"overlap" のターンを相手の音声に重ねて置く（retime_maai.MaaiRetimer.plan の rp-* 分岐）。
"""

import random
import re
from collections import Counter

from topic_plan import TopicDeck, _weights

RP_STYLE = "rp"
UNSAID_KEY = "unsaid"
PAUSE_KEY = "pause_before_sec"   # generate_dialogues.PAUSE_KEY と同じ（長い間。retime が fixed-pause で置く）
OVERLAP_KEY = "overlap"
OVERLAP_KINDS = ("interrupt", "collision", "coop", "react")
SCENE_KEY = "rp_scene"      # 場面として入れたターン（重ねていないターンも）。校閲・後処理で触らない印。Stage3 には写さない
RP_KEYS = (UNSAID_KEY, OVERLAP_KEY, SCENE_KEY)

# 場面の種類（config: topic_plan.rp_mix のラベル）。値は Stage2 の【話が重なる場面】に書く説明。
# {main} = 主話者（システム側・ch0）、{partner} = ユーザ側。
# ★例文は書かない（al_v1 の教訓: 指示文の言葉・例はそのまま写る）。譲る言い方だけは決まった言い回しが
#   無いと何を書けばよいか伝わらないので、言い方を変えるように書く。
RP_KIND_TEXT = {
    "割り込まれて応じる": (
        "{main}が少し長めに話している途中で、{partner}が割り込む。{main}のそのターンは文の途中で止まる"
        "（\"unsaid\" 付き）。{partner}の割り込みのターン（\"overlap\": \"interrupt\"）は、{main}がそこまでに"
        "言ったことの中で今すぐ聞きたくなったこと、{main}の思い違いを正すこと、急に思い出して言いたくなったこと"
        "のどれか。{main}は次のターンで、言いかけたことには戻らず、{partner}の言ったことにそのまま応じる"),
    "割り込まれて戻る": (
        "{main}が少し長めに話している途中で、{partner}が短い確かめの問い（知らない言葉・聞き取れなかった"
        "ところ・誰のことか など）で割り込む。{main}のそのターンは文の途中で止まる（\"unsaid\" 付き）。"
        "{main}は{partner}の問いに短く答え、{partner}が短く受けたあと、言いかけていた話に戻って、"
        "言うつもりだったこと（\"unsaid\" の中身）を自分の言葉で言い終える（戻るときの言い方は毎回変える）"),
    "反応が重なる": (
        "{main}が話している途中で、{partner}が笑い・驚き・感心の短いひと言（問いではないもの）を挟む。"
        "{main}はそれに応じず、止まらずにそのまま話を続ける。台本では{main}の話を文の切れ目で2つのターンに"
        "分け、その間に{partner}の短い反応のターン（\"overlap\": \"react\"）を置く。{main}の2つめのターンは"
        "1つめの続きで、{partner}の反応への返事にしない"),
    "同時に話し始めて譲る": (
        "話がひと区切りして少し間があいたところで、{main}と{partner}が同時に話し始めてしまう。2人とも最初の"
        "ひと言で止まる（2人の短い言いかけのターンを続けて置く。どちらが先でもよく、どちらも \"unsaid\" 付き、"
        "2つめに \"overlap\": \"collision\"）。次に{main}が{partner}に先を譲り（譲る言い方は場面に合わせて"
        "変える）、{partner}が言いかけたことを話す。そのやり取りが落ち着いたら、{main}は自分が言いかけた"
        "ことに戻ってもよい"),
    "同時に話し始めて引く": (
        "話がひと区切りしたところで、{main}と{partner}が同時に話し始めてしまう。{main}は最初のひと言で黙って引き、"
        "{partner}はそのまま言いたいことを話し続ける。{main}はそれに応じる"),
    "同時に話し始めて引く（相手が先）": (
        "{partner}が話し始めたのとほぼ同時に{main}も話し始めてしまい、{main}は最初のひと言で黙って引く。{partner}はそのまま"
        "話し続け、{main}はそれに応じる"),
    "同時に話し始めて譲る（相手が先）": (
        "{partner}が話し始めたのとほぼ同時に{main}も話し始めてしまい、2人とも止まる。{main}が先を譲り、{partner}が話す"),
    "同時に話し始めて譲り、相手が黙る": (
        "話がひと区切りして少し間があいたところで、{main}と{partner}が同時に話し始めてしまう。{main}が先を譲るが、"
        "{partner}は黙ったままで、少し間があいてから{main}が続ける"),
    "同時に話し始めて譲られる": (
        "話がひと区切りして少し間があいたところで、{main}と{partner}が同時に話し始めてしまう。2人とも最初の"
        "ひと言で止まる（2人の短い言いかけのターンを続けて置く。どちらが先でもよく、どちらも \"unsaid\" 付き、"
        "2つめに \"overlap\": \"collision\"）。次に{partner}が{main}に先を譲り、{main}は短く受けてから、"
        "言いかけたことを話す"),
    "重ねて応じる": (
        "{partner}が話している終わりぎわに、{main}が声を重ねて短く応じる（{main}のターンに "
        "\"overlap\": \"coop\"）。形は次のどれか: {partner}が言葉を思い出せずにいるところへ{main}が"
        "その言葉を言う／{partner}が言おうとしている文の残りを{main}が先回りして言う／{partner}の話に"
        "{main}が思わず強く同意する。{partner}のそのターンは止めずに最後まで言い終え（\"unsaid\" は付けない）、"
        "次のターンで{main}の言葉を受ける"),
    "割り込む": (
        "{partner}が話している途中で、{main}が割り込む（{partner}のそのターンは \"unsaid\" 付きの文の途中で"
        "止まる言いかけ、{main}のターンに \"overlap\": \"interrupt\"）。{main}が割り込むのは、そのほうが"
        "{partner}のためになるときだけ: {partner}が{main}のことを思い違いしたまま話を進めそうなとき"
        "（{main}の人物設定と違うことを言いかけた）／{partner}が必要以上に謝ったり遠慮したりしているとき／"
        "{partner}が言いかけたことの答えを{main}がすでに知っていて、先に言ったほうが早いとき。{main}は短い"
        "断りを添えて手短に言い、必要ならそのあと{partner}に話を返す"),
}
# ★2026-11-11 ユーザ指定「同時発話は基本的にシステムが譲る（ユーザが黙ったらシステムがフォローする場面もあってよい）。
#   割り込みも基本的にはユーザ側が割り込む側」→ 相手が譲る・主話者が割り込むは既定 0（config で戻せる）。
# ★2026-11-11 評価（人の対面コーパス maai-f2f-jp-2025 54セッション）: 同時に話し始めたとき、66% は一方が黙って引き
#   もう一方がそのまま話し続ける。「どうぞ」と口に出す発話は 0件。→ 黙って引く型を主にし、「どうぞ」の型は少なめに。
# ★2026-11-12 多様さの評価: 同時発話で先に言いかけたのが18件すべてイノウエ → 相手が先の型を足した。人の割り込みは
#   問いが少ない（自分の話・答え58%・同意34%・訂正8%）のに試走12は問い55% → 割り込まれて戻る（問いで割り込む）を下げた。
RP_MIX_DEFAULT = {"割り込まれて応じる": 1.2, "割り込まれて戻る": 0.7, "反応が重なる": 0.8,
                  "同時に話し始めて引く": 0.5, "同時に話し始めて引く（相手が先）": 0.4,
                  "同時に話し始めて譲る": 0.2, "同時に話し始めて譲る（相手が先）": 0.15,
                  "同時に話し始めて譲り、相手が黙る": 0.25,
                  "同時に話し始めて譲られる": 0.0, "重ねて応じる": 0.6, "割り込む": 0.0}
RP_COUNT_DEFAULT = {3: 0.5, 4: 0.5}
# 同時に話し始める場面は1対話に1回まで（2回あると「また同時に」の不自然な繰り返しになる）。
_COLLISION_KINDS = {"同時に話し始めて引く", "同時に話し始めて引く（相手が先）", "同時に話し始めて譲る",
                    "同時に話し始めて譲る（相手が先）", "同時に話し始めて譲り、相手が黙る", "同時に話し始めて譲られる"}

# 場面の種類 → 台本上の形（rp_events が数えるラベル）。
#   cut_main  = 主話者が割り込まれる（主話者の言いかけ＋相手の interrupt）
#   react     = 相手の react（主話者の話の間）
#   coll_main = 同時に話し始めて主話者が譲る、coll_user = 相手が譲る
#   coop      = 主話者の coop
#   cut_user  = 主話者が割り込む（相手の言いかけ＋主話者の interrupt）
KIND_SHAPE = {"割り込まれて応じる": "cut_main", "割り込まれて戻る": "cut_main",
              "反応が重なる": "react", "同時に話し始めて譲る": "coll_main",
              "同時に話し始めて譲り、相手が黙る": "coll_silent", "同時に話し始めて引く": "coll_implicit",
              "同時に話し始めて引く（相手が先）": "coll_implicit", "同時に話し始めて譲る（相手が先）": "coll_main",
              "同時に話し始めて譲られる": "coll_user", "重ねて応じる": "coop", "割り込む": "cut_user"}
SHAPE_LABEL = {"cut_main": "割り込まれる", "react": "相手の反応", "coll_main": "同時→主話者が譲る",
               "coll_silent": "同時→主話者が譲る→相手が黙る", "coll_implicit": "同時→主話者が黙って引く",
               "coll_user": "同時→相手が譲る", "coop": "重ねて応じる", "cut_user": "主話者が割り込む",
               "coll_none": "同時→譲りなし", "other": "その他の重なり"}

# 長さの決まり（実質文字数）。
CUT_MIN_CORE = 8            # 割り込まれた言いかけの下限（割り込む側がその中身に反応できる長さ）
# ★2026-11-11 試走8の言いかけは中央値0.96秒（人の同時発話で止まる側は中央値0.56秒・p90 1.09）→ 12→7字
COLLISION_MAX_CORE = 7      # 同時に話し始めた言いかけの上限（最初のひと言で止まる）
REACT_MAX_CORE = 10         # 重なる反応の上限（2026-11-12 14→10。人の重なる反応は短い）
UNSAID_MIN_CORE = 3
# ★2026-11-12 譲るひと言の型を散らした（謝るだけ・名前を呼ぶ・「お先に」）ので、それも譲るひと言として数える
_YIELD_RE = re.compile(r"どうぞ|お先|先に(?:どうぞ|言って|話して)|お話しください|すみません|すいません|ごめんなさい|"
                       r"さんから|そちらから")


def _core(u: str) -> int:
    return len(re.sub(r"[、。？！?!\s…ー]", "", str(u or "")))


def is_rp_plan(plan: dict | None) -> bool:
    return bool(plan) and plan.get("style") == RP_STYLE


def is_rp_turn(t) -> bool:
    """重なりの場面に関わるターン（言いかけ・重ねたターン）か。"""
    return isinstance(t, dict) and any(t.get(k) not in (None, "") for k in RP_KEYS)


def rp_indices(turns: list) -> set:
    """重なりの場面に関わるターンの番号（後処理・校閲・書き直しで触らない）。"""
    return {i for i, t in enumerate(turns) if is_rp_turn(t)}


def strip_rp_keys(turns: list) -> int:
    """重なりのキーを外す（style: rp でない対話）。言いかけは残りをつないで1文に戻す。"""
    n = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        if t.get(UNSAID_KEY):
            t["utterance"] = join_unsaid(str(t.get("utterance", "")), str(t[UNSAID_KEY]))
        for k in RP_KEYS:
            if k in t:
                t.pop(k)
                n += 1
    return n


def join_unsaid(spoken: str, unsaid: str) -> str:
    """言いかけと残りをつないだ1文（言いかけを取り消すとき）。"""
    s = spoken.rstrip().rstrip("。")
    u = unsaid.strip()
    out = s + u
    return out if out.endswith(("。", "？")) else out.rstrip("、") + "。"


class RPDeck(TopicDeck):
    """雑談の配り方（TopicDeck）に、1対話ぶんの「重なりの場面」の種類と数を足す。

    種類は発注の少ないものから目標比率に寄せて配る（乱数入り。qa_plan.QADeck._least の注記: 決定的に配ると
    比率の組が同じ項目どうしが同じ周期で回って組が固定される）。同じ対話に同じ種類は入れない。
    """

    def __init__(self, cfg: dict, genres: list[str]):
        super().__init__(cfg, genres)
        cfg = cfg or {}
        self.rp_mix = _weights(cfg.get("rp_mix"), RP_MIX_DEFAULT)
        unknown = set(self.rp_mix) - set(RP_KIND_TEXT)
        if unknown:
            raise ValueError(f"topic_plan.rp_mix の未知のラベル: {sorted(unknown)}"
                             f"（使えるのは {sorted(RP_KIND_TEXT)}）")
        self.rp_count_mix = _weights(cfg.get("rp_count"), RP_COUNT_DEFAULT, key=int)
        self.rp_rng = random.Random(int(cfg.get("seed", 0)) + 7919)
        self.rp_issued = Counter()
        self.rp_count_issued = Counter()

    def describe(self) -> str:
        return (f"重なりの場面（style: rp）。{super().describe()}、場面の数 {self.rp_count_mix}、"
                f"場面の種類 {self.rp_mix}")

    def _rp_least(self, issued: Counter, mix: dict, exclude=()):
        cands = [k for k in mix if k not in exclude] or list(mix)
        return min(cands, key=lambda k: (issued[k] + self.rp_rng.random()) / mix[k])

    def add_used(self, plan: dict):
        super().add_used(plan)
        if not is_rp_plan(plan):
            return
        kinds = plan.get("overlaps") or []
        if len(kinds) in self.rp_count_mix:
            self.rp_count_issued[len(kinds)] += 1
        for k in kinds:
            self.rp_issued[k] += 1

    def next_plan(self, card: dict | None = None, persona: str = "") -> dict:
        plan = super().next_plan(card, persona)
        n = self._rp_least(self.rp_count_issued, self.rp_count_mix)
        self.rp_count_issued[n] += 1
        kinds: list[str] = []
        for _ in range(min(n, len(self.rp_mix))):
            ex = set(kinds) | (_COLLISION_KINDS if _COLLISION_KINDS & set(kinds) else set())
            k = self._rp_least(self.rp_issued, self.rp_mix, exclude=ex)
            if k in ex:
                break
            kinds.append(k)
            self.rp_issued[k] += 1
        self.rp_rng.shuffle(kinds)
        return {**plan, "style": RP_STYLE, "overlaps": kinds}


def stage1_rp_text(plan: dict) -> str:
    """Stage1 の「話題と順番」の段落（雑談と同じ。重なりの場面は Stage2 だけで指定する）。"""
    from topic_plan import stage1_topic_text
    return stage1_topic_text({**plan, "style": "chat"})


def stage2_rp_block(plan: dict, main_name: str = "", partner_name: str = "") -> str:
    """Stage2 の本生成に足す段落（雑談の【話題の流れ】と同じ）。

    ★2026-11-10 試走1: 本生成のプロンプトに【話が重なる場面】（4場面の説明と書き方）を足して1回で書かせたら、
    20対話中5対話が作り直しの上限で落ち、通った15対話も場面の形が崩れていた（"unsaid" の付け忘れ・割り込む側が
    聞こえていない残りに反応・割り込みに応じない返事・問いの形の反応）。Gemma には対話全体と場面を同時に
    書かせるのが重い。本生成はふつうの雑談にし、場面はあとから1つずつ LLM に入れさせる（rp_scene_prompt）。
    """
    from topic_plan import stage2_topic_block
    return stage2_topic_block({**plan, "style": "chat"}, main_name, partner_name)


def rp_norm(plan: dict) -> str:
    """本生成の末尾の念押し（2段にしたので空。場面は rp_scene_prompt で入れる）。"""
    return ""


# ── 場面を1つずつ入れる（Stage2 の本生成のあと）─────────────────────────
# 場面ごとの手順（新しいターンの並び）。{main} = 主話者、{partner} = ユーザ側。話者の並びは SCENE_SPEAKERS で固定する。
# ★2026-11-10 試走2: 置き換える範囲を LLM に選ばせたら、不採用の大半が「直後と同じ話者で終わる」だった（場面の
#   ターン数と範囲の偶奇を合わせられない）。範囲はこちらで選び（scene_candidates）、LLM には並びどおりに書かせる。
#   同時に話し始める場面は、譲る側が先に言いかける並びにする（言いかけた本人が続けて譲ると話者が続くため）。
# ★例文は書かない（指示文の言葉はそのまま写る）。
RP_SCENE_STEPS = {
    "割り込まれて応じる": [
        "{main}: 少し長めに話し始めるが、文の途中で止まる（\"utterance\" は口に出したところまで、\"unsaid\" に言うつもりだった残り）",
        "{partner}（\"overlap\": \"interrupt\"）: {main}がそこまでに口に出したことについて、{v}"
        "（{main}の言葉をくり返すだけの問いにしない。第一声は{o}）",
        "{main}: 言いかけたことには戻らず、{partner}の言ったことに自分の言葉で応じる（{partner}の言葉をくり返して受けるだけにせず、"
        "答える・自分のことを足す・問い返す など。笑うのは{partner}の言ったことがおかしいときだけ）",
    ],
    "割り込まれて戻る": [
        # ★2026-11-13 試走15: 残りが述語だけだと、戻った発話が言いかけの言い直しにとどまる（「…変な味に――」→「あー、ありますよね。」
        #   →「で、結局、変な味になっちゃうんです。」）。語で検査すると、ひらがなの多い自然な残りまで半分以上弾くので文面で頼む
        "{main}: 少し長めに話し始めるが、文の途中で止まる（\"utterance\" は口に出したところまで、\"unsaid\" に言うつもりだった残り。"
        "残りには、まだ言っていない話の中身が入るところで止める）",
        "{partner}（\"overlap\": \"interrupt\"）: {main}がそこまでに口に出したことについての短いひと言。{v}"
        "（{main}の言葉をくり返すだけの問いや、「具体的に」「どういうこと」と聞き返す形にしない。第一声は{o}）",
        "{main}: {partner}のひと言に短く応じてから、同じターンの中で、言いかけたところから話を続けて言い終える（言いかけの頭から"
        "言い直さない。このターンだけは2文でよい）",
    ],
    "反応が重なる": [
        "{main}: 自分のことについて、{partner}が思わず{v}の声をもらすような話をする（{w}。問いかけにしない。文の切れ目で終える）",
        "{partner}（\"overlap\": \"react\"）: {main}の話への{v}の短いひと言（問いにしない。10字まで）",
        "{main}: 1つめの話の続きをそのまま話す（{partner}のひと言への返事にしない。受けのひと言から始めない。"
        "1つめで言ったことをくり返さず、そのとき何があったか・どうしたか など、まだ言っていない出来事を続ける。"
        "気持ちや一般論のまとめにしない）",
    ],
    "同時に話し始めて引く": [
        "{main}: 話がひと区切りしたところで話し始めるが、最初のひと言（2〜5文字ほど）で黙る（\"unsaid\" に言おうとしたこと）",
        "{partner}（\"overlap\": \"collision\"）: ほぼ同時に話し始め、止まらずにそのまま、直前の話の続きとして言いたかった"
        "ことを言い終える（重なったことには触れない。\"unsaid\" は付けない）",
        "{main}: {partner}の話に応じる（譲るひと言や断りは言わない。相づちだけで済ませず、ひと言中身を足す）",
    ],
    "同時に話し始めて譲る": [
        "{main}: 話がひと区切りしたところで話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる（\"unsaid\" に言おうとしたこと）",
        "{partner}（\"overlap\": \"collision\"）: ほぼ同時に話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる"
        "（\"unsaid\" に言おうとしたこと。{main}とは別の言葉にする）",
        "{main}: {partner}に先を譲るひと言（{y}）",
        "{partner}: お礼は言わずに、言おうとしていたことを話す",
    ],
    "同時に話し始めて引く（相手が先）": [
        "{partner}: 話がひと区切りしたところで話し始め、止まらずに言いたいことを言い終える",
        "{main}（\"overlap\": \"collision\"）: ほぼ同時に話し始めるが、{partner}の声に気づいて最初のひと言（2〜5文字ほど）で黙る"
        "（\"unsaid\" に言おうとしたこと）",
        "{main}: {partner}の話に応じる（譲るひと言や断りは言わない。相づちだけで済ませず、ひと言中身を足す）",
    ],
    "同時に話し始めて譲る（相手が先）": [
        "{partner}: 話がひと区切りしたところで話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる（\"unsaid\" に言おうとしたこと）",
        "{main}（\"overlap\": \"collision\"）: ほぼ同時に話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる"
        "（\"unsaid\" に言おうとしたこと。{partner}とは別の言葉にする）",
        "{main}: {partner}に先を譲るひと言（{y}）",
        "{partner}: お礼は言わずに、言おうとしていたことを話す",
    ],
    "同時に話し始めて譲り、相手が黙る": [
        "{main}: 話がひと区切りしたところで話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる（\"unsaid\" に言おうとしたこと）",
        "{partner}（\"overlap\": \"collision\"）: ほぼ同時に話し始めるが、最初のひと言（2〜5文字ほど）だけで止まる"
        "（\"unsaid\" に言おうとしたこと。{main}とは別の言葉にする）",
        "{main}: {partner}に先を譲るひと言（{y}）",
        "{main}: {partner}が話しださずに少し間があいたのを受けて、{v}",
    ],
    "同時に話し始めて譲られる": [
        "{partner}: 話がひと区切りしたところで話し始めるが、最初のひと言だけで止まる（\"unsaid\" に言おうとしたこと）",
        "{main}（\"overlap\": \"collision\"）: ほぼ同時に話し始めるが、最初のひと言だけで止まる"
        "（\"unsaid\" に言おうとしたこと。{partner}とは別の言葉にする）",
        "{partner}: {main}に先を譲るひと言",
        "{main}: 短く受けてから、言おうとしていたことを話す",
    ],
    "重ねて応じる": [
        "{partner}: {v}",
        "{main}（\"overlap\": \"coop\"）: その終わりぎわに声を重ねて、{w}（短く）",
        "{partner}: {main}の言葉を受けて話を続ける（お礼は言わない）",
    ],
    "割り込む": [
        "{partner}: 話し始めるが、文の途中で止まる（\"utterance\" は口に出したところまで、\"unsaid\" に言うつもりだった残り）。"
        "{v}",
        "{main}（\"overlap\": \"interrupt\"）: 短い断りを添えて、手短に言う（決まり文句にせず、{partner}の言いかけた中身に触れて言う）",
        "{partner}: 受ける",
    ],
}
# 場面ごとの中身の型（{v} に入る）。対話 id と場面の番号で決める（scene_variant）。
# ★試走6: 型を任せると、割り込むは 11中8が「相手が謝る→謝らなくて大丈夫ですよ」（謝る理由のない発話に謝罪を足す）、
#   割り込まれて戻るの問いは「Xって、Yのことですか？」に寄った。型を割り当てて散らす。
SCENE_VARIANTS = {
    # ★2026-11-11 試走8の判定: 「急に思い出した自分のこと」は、相手の言葉にだけ反応して自分の昔話をかぶせる横取りになり、
    #   きっかけが不自然とされた → 「問いかけの途中で答えが分かって先に答える」に替えた（試走8で自然だった型）。
    # ★2026-11-12 人の割り込みの働き（自分の話・答え58%・同意34%・訂正8%）に寄せて、問いは5つに1つ
    "割り込まれて応じる": [
        "{main}が{partner}に何かを尋ねかけている途中で答えが分かり、先回りして答える"
        "（そのため{main}の言いかけは{partner}への問いかけの途中にする）",
        "思わず強く共感・同意したことを言う",
        "自分も同じだという短い自分の話をひと言して、{main}に話を返す（話を横取りしない）",
        "{main}が言いかけの中で{partner}が前に話した{partner}自身のこと（経験・好み・事実）を少し取り違えているのに気づいて、"
        "その事実を正す（そのため{main}の言いかけには、{partner}が前に話したことの取り違えを入れる。{main}自身の気持ちや"
        "経験の言い表し方は正さない）",
        "話についていくのに今すぐ確かめたくなったことを尋ねる",
    ],
    "割り込まれて戻る": [
        "{main}が自分の趣味や仕事の話の中で自然に使った、{partner}が知らなそうな言葉（専門の言葉・外来語・店や作品の名前）が"
        "何かを尋ねる（はい・いいえで答えられる確かめにしない。言葉を問うために不自然な言い回しを作らない）",
        "{main}の言いかけの終わりのほうがよく聞き取れなかったので、短く聞き返す（聞き取れた語だけをくり返すか、何と言ったかを"
        "尋ねる。聞き違えた語を作らない。{main}は答えで言い直してから続ける）",
        # ★2026-11-13 試走14: 割り込みの27中15が問い（戻るの型が問いだけ。人の割り込みは自分の話・答え58%・同意34%で、問いはまれ）
        "{main}の言いかけに出てきた事柄について、自分も知っている・同じだという短いひと言（問いにしない。話を取らない）",
    ],
    # ★2026-11-12 反応の語が「いいですよね」「いいですね」「そうなんです」に寄った → 種類と語の例を割り当てる
    # ★2026-11-13 試走13: 反応の種類だけ割り当てたら、話の中身と合わない反応（「準備からこだわりがあるんですね」に「えっ。」、
    #   イノウエの問いかけに「へえー。」）→ 反応を引き出す話の中身も組で割り当てる（{v}＝反応、{w}＝主話者の話の中身）
    "反応が重なる": [("笑い（あはは・ふふ など）", "ちょっとした失敗や、思わず笑ってしまう出来事"),
                     ("驚き（えー・えっ など）", "相手が予想しないような意外な出来事や数字"),
                     ("感心（へえ・すごい など）", "手間のかかることや、ふつうはしないことを続けている話"),
                     ("納得（なるほど・あー など）", "そうしている理由やわけ"),
                     ("共感（わかる・ありますね など）", "誰にでもありそうな困りごとや、よくあること")],
    "同時に話し始めて譲り、相手が黙る": [
        "前置きを付けずに、自分が言いかけたことを話す",
        "相手が言いかけたことを、短くくだけた言い方で促す（かしこまった言い方にしない。急かさない）",
    ],
    # 重ねて応じる: {v}＝相手の1つめの話し方、{w}＝主話者の重ね方（同じ番号どうしで組む。scene_variant が組で返す）
    "重ねて応じる": [
        ("話している途中で、言いたい言葉（名前・地名・料理名 など）が出てこずに詰まる（言い終えずに、言葉を探すところで止める。"
         "\"unsaid\" は付けない）", "{partner}が探している言葉を言う"),
        ("話の最後の言葉を言いかけて、ゆっくり言いよどむ（文の残りが聞き手にも見当がつく形で、言い終えずに止める。"
         "\"unsaid\" は付けない）", "文の残りを先回りして言う"),
        # ★2026-11-13 試走13: 強い同意が「ほんとそうですよね」「まじでそうですよね」「最高ですよね」と「〜ですよね」に寄った（8中5）
        ("{main}も思わず強く同意したくなることを言う（言い終える）",
         "「〜ですよね」の形にせず、ごく短い言葉で強く同意する"),
    ],
    "割り込む": [
        "中身は、{partner}が{main}について人物設定と違う思い違い（出身・住まい・仕事・家族・好み など）をしたまま話を進めかけるもの。"
        "{main}はそれを正す",
        "中身は、{partner}が話の流れの中でつい必要以上に謝ったり遠慮したりしかけるもの（謝る理由が前の話にあること）。"
        "{main}は気にしなくていいと伝える",
        "中身は、{partner}が言いたい言葉（名前・地名・作品名・料理名 など）を思い出せずに言いよどむもの（\"utterance\" は"
        "言いよどんだところまで）。{main}はその言葉を知っていて先に言う",
    ],
}


# 割り込みの第一声（人の割り込みは なし69%・あ/あー14%・いや3%・へー2%・うん/そう4%・え2% …）。正す型は「いや」「いえ」も。
_OPENERS = [("なし（いきなり中身から）", 11), ("「あ」か「あー」", 3), ("「え」", 1), ("「うん」か「そう」", 1), ("「へえ」", 1)]
# 譲るひと言の型（2026-11-12: 7件すべて「どうぞ」で始まった。人の雑談では「どうぞ」は 0件）
YIELD_VARIANTS = ["「どうぞ」を使った短いひと言", "謝るひと言だけで（「どうぞ」は使わない）",
                  "{partner}の名前を呼んで先を促す（「どうぞ」は使わない）", "「お先に」のような短いひと言（「どうぞ」は使わない）"]


def scene_opener(kind: str, seed: str, variant="") -> str:
    """割り込む側の第一声の型。"""
    import zlib
    if "正す" in str(variant):
        return "「いや」「いえ」「あ」のどれか"
    pool = [o for o, w in _OPENERS for _ in range(w)]
    return pool[zlib.crc32(f"{seed}:{kind}:open".encode()) % len(pool)]


def scene_yield(kind: str, seed: str) -> str:
    import zlib
    return YIELD_VARIANTS[zlib.crc32(f"{seed}:{kind}:yield".encode()) % len(YIELD_VARIANTS)]


# 型の重み（無い種類は等分）。★2026-11-13 試走13: 割り込まれて応じるの型を等分にしていたら、正す型が 18中6（人の割り込みの
#   訂正は 8%。tools/human_interrupt_lex.py: 自分の話・答え58%・同意34%・訂正8%、問いはまれ）→ 先回りの答え・同意・自分の話を2、
#   正す・尋ねるを1。
# ★2026-11-13 試走13・14の判定: 戻るの聞き返し・意味を聞く型の流れが割り込みの中で最も低い（4.25・4.00）。聞き返しは
#   「顔が見えるんですか？」→「あ、はい。そういうのがいいところですよね。」と答えにならない返しが目立つ → 戻るは 意味2・聞き返し1・問いでない4
SCENE_VARIANT_WEIGHTS = {"割り込まれて応じる": [2, 2, 2, 1, 1], "割り込まれて戻る": [2, 1, 4]}


def scene_variant(kind: str, seed: str):
    """場面の中身の型（SCENE_VARIANTS）。対話 id と場面の種類で決める（無い種類は空）。重ねて応じるは (v, w) の組。"""
    vs = SCENE_VARIANTS.get(kind)
    if not vs:
        return ""
    import zlib
    ws = SCENE_VARIANT_WEIGHTS.get(kind) or [1] * len(vs)
    pool = [v for v, w in zip(vs, ws) for _ in range(w)]
    return pool[zlib.crc32(f"{seed}:{kind}".encode()) % len(pool)]


# 新しいターンの話者の並び（M = 主話者、P = 相手）。重ねたターンはどれも2つめ。
SCENE_SPEAKERS = {"割り込まれて応じる": "MPM", "割り込まれて戻る": "MPM", "反応が重なる": "MPM",
                  "同時に話し始めて引く": "MPM", "同時に話し始めて引く（相手が先）": "PMM",
                  "同時に話し始めて譲る（相手が先）": "PMMP",
                  "同時に話し始めて譲る": "MPMP", "同時に話し始めて譲り、相手が黙る": "MPMM",
                  "同時に話し始めて譲られる": "PMPM",
                  "重ねて応じる": "PMP", "割り込む": "PMP"}
# ★2026-11-12 試走10の判定: 流れが途切れる一番の原因は、場面のすぐ後の元のターンが置き換える前の発話への返事のまま
#   残ること（「…スマホを遠くに置くんですか？」ともう聞いたことを聞く）。場面の並びのあとに1ターン足し、元の次のターンを
#   場面の流れに合わせて言い直させる（置き換える範囲もその1ターンを含める）。
def full_pattern(kind: str) -> str:
    pat = SCENE_SPEAKERS[kind]
    return pat + ("P" if pat[-1] == "M" else "M")


# 置き換える元のターンの中身を優先して選ぶ場面（言いかけ・反応をはさむ側の発話が長いほうが書きやすい）
_PREFER_LONG = {"割り込まれて応じる", "割り込まれて戻る", "反応が重なる", "重ねて応じる", "割り込む"}
SCENE_EDGE = 3            # 冒頭・末尾のこのターン数には場面を入れない（挨拶・締め）
_CONTENT_WORD = re.compile(r"[一-龥々ァ-ヶー]{2,}")
# どの話にも出る語（つながりの判定に使わない）
_GENERIC_WORDS = {"最近", "今日", "昨日", "今度", "本当", "自分", "結構", "一番", "時間", "普段", "毎日", "以前", "前回",
                  "今回", "全然", "大体", "一緒", "気持", "感じ"}
_ACK_HEAD = re.compile(r"^(?:ええ、|はい、|うん、|まあ、)?(?:そうなんです|そうですね|そうですよね|ですよね|本当に|ほんとに|そうそう)")
# Gemma の決まった聞き返し（試走3で割り込みの問いの半分近く）
# ★2026-11-11 試走8: 分かりきったことの言い直しの確かめ（「集中できる環境がいい、ということですか？」）も弾く
_ASK_BACK = re.compile(r"具体的に|どういうこと|ということ(?:です)?か|ってこと(?:です)?か")
# 反応のあとの続きが気分・一般論のまとめ（試走8:「心身ともに、すっかりリラックスできます」「心にゆとりが生まれる気がします」）
_PLATITUDE = re.compile(r"心地|リラックス|リフレッシュ|癒や?さ|癒し|ゆとり|喜びを感じ|大切(?:です|だと)|いいこと(?:です|だと)"
                        r"|気がします。?$|素敵(?:です|だと)")
# 言い終えていない話し方（重ねて応じるの言葉探し・先回り。句点を足す前の形で見る）
_TRAILING_OPEN = re.compile(r"(?:の|が|を|に|へ|で|と|は|も|や|て|って|けど|から|ので|のに|たり|ような|みたいな|、|…|ー)[。]?$")


def scene_window(n_turns: int, j: int, n_scenes: int) -> tuple[int, int]:
    """j 番目（0始まり）の場面を入れる範囲 [lo, hi]（対話の本体を場面の数で等分した j 番目）。"""
    lo0, hi0 = SCENE_EDGE, max(SCENE_EDGE + 4, n_turns - 1 - SCENE_EDGE)
    w = (hi0 - lo0 + 1) / max(1, n_scenes)
    lo = lo0 + int(round(j * w))
    hi = min(hi0, lo0 + int(round((j + 1) * w)) - 1)
    return lo, max(hi, min(hi0, lo + 4))


def scene_candidates(turns: list, kind: str, lo: int, hi: int, main: str, partner: str,
                     seed: str = "") -> list[tuple[int, int]]:
    """置き換える範囲 (a, b) の候補（よい順）。

    並びの最初の話者で始まるターン a から、並びが偶数なら2ターン・奇数なら1ターンを置き換える
    （前後の話者と交互になる）。前後に既存の場面（重なりのターン）があるところ・締めの挨拶は避ける。
    """
    pat = full_pattern(kind)
    first = main if pat[0] == "M" else partner
    span = 1 if pat[0] == pat[-1] else 2          # 元の並びは交互なので、最初と最後の話者で範囲の偶奇が決まる
    if span == 1 and len(pat) > 1:
        span = 3                                  # 次のターンの言い直しを含めると、最初と最後が同じ話者なら3ターン
    n = len(turns)
    rng = random.Random(f"{seed}:{kind}:{lo}")
    cands = []
    for a in range(lo, hi + 1):
        b = a + span - 1
        if b >= n - 1 or a < 1 or _speaker(turns[a]) != first:
            continue
        if any(is_rp_turn(t) for t in turns[max(0, a - 1):b + 2]):
            continue
        if any(_speaker(x) == _speaker(y) for x, y in zip(turns[a - 1:b + 2], turns[a:b + 2])):
            continue                     # 元の並びが交互でない箇所（沈黙の続きなど）は避ける
        u = str(turns[a].get("utterance", ""))
        score = rng.random()
        if kind in _PREFER_LONG:
            score += min(_core(u), 40) / 10.0 - (2.0 if "？" in u else 0.0)
        elif kind in _COLLISION_KINDS:   # ひと区切りのあと（直前が問いでない）
            score += 0.0 if "？" in str(turns[a - 1].get("utterance", "")) else 2.0
        cands.append((score, a, b))
    return [(a, b) for _, a, b in sorted(cands, reverse=True)]


# ★2026-11-12 日本語の判定（tools/rp_ja_judge.py。試走12）: 場面のターンは100発話あたりの不自然さが 10.4件で、それ以外
#   （本生成のターン）の 7.8件より多く、いちばん多いのが「硬い」（「向上心があって素晴らしいですね」「趣がありますね」）。
#   評価・気分のまとめの言い方も 場面 11.2%・それ以外 4.8%。場面を入れる呼び出しには system プロンプトが無く、本生成の
#   「最重要: 徹底した話し言葉」の段（書き言葉の禁止と書き換え例）が渡っていなかった → 同じ段を切り出して渡す。
_SPOKEN_HEAD = re.compile(r"^# 最重要: [^\n]*話し言葉[^\n]*$", re.M)


def spoken_style_section(system_prompt: str) -> str:
    """本生成の system プロンプトから「話し言葉」の段（見出しから次の「# 」見出しの前まで）を切り出す。無ければ ""。"""
    s = str(system_prompt or "")
    m = _SPOKEN_HEAD.search(s)
    if not m:
        return ""
    nxt = re.search(r"^# ", s[m.end():], re.M)
    return s[m.start():m.end() + (nxt.start() if nxt else len(s) - m.end())].strip()


def rp_scene_system(system_prompt: str) -> str:
    """場面を入れる呼び出しの system メッセージ（本生成の話し言葉の段つき。段が無ければ ""）。"""
    sec = spoken_style_section(system_prompt)
    return ("あなたは音声対話の台本の書き手です。台本はそのまま音声合成され、話し言葉の音声対話のデータになります。\n\n"
            + sec) if sec else ""


def rp_scene_prompt(turns: list, metadata: dict, kind: str, a: int, b: int,
                    main: str, partner: str, persona: str = "", variant="", opener: str = "なし（いきなり中身から）",
                    yield_style: str = "「どうぞ」を使った短いひと言", style_norm: str = "") -> str:
    """場面を1つ入れる書き換えの依頼文（user メッセージ）。turn a〜b を並びどおりの新しいターンで置き換える。"""
    vv, ww = (variant if isinstance(variant, tuple) else (variant, ""))
    v, w = vv.format(main=main, partner=partner), ww.format(main=main, partner=partner)
    last = main if full_pattern(kind)[-1] == "M" else partner
    y = yield_style.format(main=main, partner=partner)
    step_list = [s.format(main=main, partner=partner, v=v, w=w, o=opener, y=y) for s in RP_SCENE_STEPS[kind]]
    step_list.append(f"{last}: 元の turn {b} の発話を、ここまでの場面の流れにつながるように言い直す（元の発話の話の中身は"
                     f"なるべく残し、そのあとの turn {b + 1} へもつながるようにする。場面ですでに言われたことを聞き直さない）")
    steps = "\n".join(f"{i}. {x}" for i, x in enumerate(step_list, 1))
    lines = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        mark = " ←置き換える" if a <= i <= b else ""
        lines.append(f"{i} {t.get('speaker', '')}: "
                     f"{rp_render(t) if is_rp_turn(t) else t.get('utterance', '')}{mark}")
    style = metadata.get("speech_style", "")
    rng = f"turn {a}" if a == b else f"turn {a}〜{b}"
    # ★2026-11-12 試走13: 本生成の話し言葉の段（タメ口の例「まじで」「てかさ」を含む）を渡したら、敬語の対話の場面に
    #   「まじで」「てかさ」が 274中14 混ざった（本生成のターンは 0。本生成は _style_norm の念押しを付けている）→ 同じ念押し
    norm = ("\n- " + str(style_norm).strip()) if str(style_norm or "").strip() else ""
    per = (f"\n\n【{main}の人物設定】\n{persona.strip()}" if persona and kind in ("割り込む", "割り込まれて応じる")
           else "")
    return (
        "次の対話の一部を書き換えて、2人の声が重なる場面を1つ入れてください。この対話は音声対話の台本で、"
        "重なりの場面はあとで音声を重ねて作ります。\n\n"
        f"【入れる場面】{kind}（新しいターンの並び。全部で{len(step_list)}ターン）\n{steps}\n\n"
        "【書き方】\n"
        f"- {rng}（「←置き換える」の印）を、上の並びのとおりの新しいターンで置き換えてください。元の発話の中身は"
        f"使っても変えてもかまいません。turn {a - 1} からつながり、turn {b + 1} へ自然につながるようにしてください"
        "（ほかのターンは変えません）。\n"
        "- 言いかけて止まった発話は、\"utterance\" に実際に口に出したところまで（文の途中。句点を付けない）、"
        "\"unsaid\" に言うつもりだった残りを書きます。言いかけは語の切れ目で止め（語の途中で切らない）、"
        "文を言い終えたところ（「？」「。」の後）では止めないでください。割り込まれる言いかけは15字以上にし、"
        "割り込む側が反応するきっかけの言葉は、言いかけの終わり近くに置いてください。\n"
        "- 声を重ねたターンには、並びに書いた \"overlap\" を付けます。並びに書いていないターンには "
        "\"unsaid\" も \"overlap\" も付けないでください。\n"
        "- 割り込む側は、相手が口に出したところまでにだけ反応します（\"unsaid\" の中身は聞こえていません）。"
        "割り込まれた側は、割り込みを責めたり、言いかけたことを押し通したりしません。\n"
        f"- 口調（{style}）とそれぞれの話し方は前後の発話にそろえ、1ターン1文の話し言葉で書いてください。"
        "実在の人や作品の名前を作り変えないでください（よく知らなければ一般的な言い方にする）。\n"
        "- 場面の発話は、前後の元の発話と同じくらいくだけた短い言い方にします。相手や相手の話をほめる・評価する言い方や、"
        "気分・一般論のまとめで発話を締めず、何があったか・何をしたか・どう思ったかを、ふだんの言葉で具体的に言ってください"
        "（漢語の多い書き言葉や、文学的な言い回しにしない）。"
        f"{norm}"
        f"{per}\n\n"
        "【対話】\n" + "\n".join(lines) + "\n\n"
        "出力は次の JSON だけにしてください:\n"
        "{\"turns\": [{\"speaker\": \"話者名\", \"utterance\": \"発話内容\"}, ...]}")


def _words(u: str) -> set:
    return set(_CONTENT_WORD.findall(str(u or "")))


def _topic_link(u, ctx) -> bool:
    """u が ctx の話とつながっているか（漢字・カタカナの語か、漢字・カタカナを含む2文字の重なりが2つ以上）。
    ★2026-11-12 試走11: 語だけで見ると「毎日バタバタしています」が直前の「バタバタしてます」とつながらない扱いになった。"""
    if (_words(u) - _GENERIC_WORDS) & _words(ctx):
        return True
    def bg(x):
        x = re.sub(r"[、。？！?!\s]", "", str(x or ""))
        return {x[i:i + 2] for i in range(len(x) - 1) if re.search(r"[一-龥々ァ-ヶ]", x[i:i + 2])}
    return len(bg(u) & bg(ctx)) >= 2


def _echo_only(resp, said) -> bool:
    """resp が said の言葉をくり返して受けるだけの短い返しか（「スポーツから歌に変わったんですね」）。"""
    r = str(resp or "")
    rw = _words(r)
    return (_core(r) <= 18 and bool(rw) and rw <= _words(said)
            and bool(re.search(r"(?:んですね|ですね|ですか|んですか)[。？]?$", r)))


def _bigram_overlap(a, b) -> float:
    """b の文字2-gram のうち a にもあるものの割合（句読点・空白は除く）。"""
    def bg(u):
        u = re.sub(r"[、。？！?!\s]", "", str(u or ""))
        return {u[i:i + 2] for i in range(len(u) - 1)}
    ga, gb = bg(a), bg(b)
    return len(ga & gb) / len(gb) if gb else 0.0


def _same_turn(x, y) -> bool:
    """LLM が写してきた元のターンか（同じ話者で、発話の2文字の重なりが7割以上）。"""
    if not (isinstance(x, dict) and isinstance(y, dict)) or _speaker(x) != _speaker(y):
        return False
    return _bigram_overlap(y.get("utterance"), x.get("utterance")) >= 0.7


def _extract_block(turns: list, new: list, a: int, b: int):
    """LLM が対話全体を返したとき、置き換えた部分（turn a〜b の代わり）だけを取り出す。

    前（turn 0〜a-1）と後ろ（turn b+1〜）が話者と発話でほぼ元のままなら、その間を返す。取り出せなければ new のまま。
    """
    pre, post = turns[:a], turns[b + 1:]
    if len(new) <= len(pre) + len(post):
        return new

    def same(x, y):
        return (isinstance(x, dict) and isinstance(y, dict) and _speaker(x) == _speaker(y)
                and str(x.get("utterance", ""))[:12] == str(y.get("utterance", ""))[:12])
    hp = sum(same(x, y) for x, y in zip(pre, new[:len(pre)]))
    tp = sum(same(x, y) for x, y in zip(post, new[len(new) - len(post):])) if post else 0
    if hp >= 0.8 * len(pre) and (not post or tp >= 0.8 * len(post)):
        return new[len(pre):len(new) - len(post)]
    return new


# JSON のキー名の切れ端が発話の頭に残ったもの（試走13「utteranceあの、地元の魚って」・試走15「erance実際に使ってる人の」）
_KEY_LEAK = re.compile(r"^([a-z]{3,9})\s*[:：]?\s*(?=[^\x00-\x7f])")


def _strip_key_leak(u: str) -> str:
    m = _KEY_LEAK.match(u)
    if m and any(k.endswith(m.group(1)) for k in ("utterance", "unsaid", "speaker", "overlap")):
        return u[m.end():]
    return u


def validate_scene(turns: list, data, kind: str, main: str, partner: str, a: int, b: int,
                   clean=lambda u: u, variant=None) -> tuple:
    """LLM が返した場面の書き換え（turn a〜b を置き換える新しいターン）を検査する。(新しいターン or None, 理由)。

    clean は発話の整形（generate_dialogues の sanitize_utterance 等）。新しいターンは normalize_rp を
    前後のターンと一緒にかけたもの（句点の付け外し・キーの整え）を返す。
    """
    new = data.get("turns") if isinstance(data, dict) else data
    pat = full_pattern(kind)
    # ★2026-11-12 試走11: 次のターンの言い直しを足したら「話者の並びが違う」が増えた。言い直しのターンが抜けていて
    #   場面の並びどおりなら、元の次のターン（turn b）をそのまま残して受け入れる（試走10までと同じ形）。
    if isinstance(new, list) and len(new) == len(pat) - 1 and isinstance(turns[b], dict) and \
            all(isinstance(t, dict) and t.get("speaker") == (main if w_ == "M" else partner)
                for t, w_ in zip(new, pat[:-1])):
        new = list(new) + [{"speaker": turns[b].get("speaker"), "utterance": turns[b].get("utterance", "")}]
    if isinstance(new, list) and len(new) > len(pat) + 2:
        new = _extract_block(turns, new, a, b)      # ★試走3: 対話全体を返してくる（不採用の約3割）
    # ★2026-11-12 試走12: 不採用の「話者の並びが違う」は、置き換える範囲の前後の元のターン（turn a-1・b+1）を写して前後に
    #   付けたもの（期待 MPMP… に対して PMPMPM）。元のターンと中身が同じなら外す。
    if isinstance(new, list) and new and a > 0 and _same_turn(new[0], turns[a - 1]):
        new = new[1:]
    if isinstance(new, list) and new and b + 1 < len(turns) and _same_turn(new[-1], turns[b + 1]):
        new = new[:-1]
    if not isinstance(new, list) or not len(pat) <= len(new) <= len(pat) + 2:
        return None, f"ターン数 {len(new) if isinstance(new, list) else '?'}（{len(pat)}）"
    # 並びのあとに足したターン（2つまで）は、交互が続くなら認める（試走3: 同時発話の6ターン・重ねての5ターン）
    if (len(new) - len(pat)) % 2:
        return None, f"ターン数 {len(new)}（{len(pat)}）"
    other = {"M": "P", "P": "M"}
    full = pat + "".join(other[pat[-1]] if k % 2 == 0 else pat[-1] for k in range(len(new) - len(pat)))
    out = []
    for t, who in zip(new, full):
        if not isinstance(t, dict) or t.get("speaker") != (main if who == "M" else partner):
            got = "".join("M" if isinstance(x, dict) and x.get("speaker") == main else
                          "P" if isinstance(x, dict) and x.get("speaker") == partner else "?" for x in new)
            return None, f"話者の並びが違う（{got}→{full}）"
        u = clean(_strip_key_leak(str(t.get("utterance", ""))))
        if not u:
            return None, "空の発話"
        x = {"speaker": t["speaker"], "utterance": u, "_raw": u}
        if t.get(UNSAID_KEY):
            x[UNSAID_KEY] = clean(_strip_key_leak(str(t[UNSAID_KEY])))
        if t.get(OVERLAP_KEY):
            x[OVERLAP_KEY] = str(t[OVERLAP_KEY]).strip()
        if not x.get(UNSAID_KEY) and not u.endswith(("。", "？")):
            x["utterance"] = u.rstrip("、") + "。"
        out.append(x)
    # ★試走3: 重ねて応じる場面で、相手の言葉探し（言い終えずに詰まる発話）を言いかけ（"unsaid" 付き）で書いてくる
    #   （不採用の約2割。形が「主話者が割り込む」になる）。言葉探しは言いかけの印を外して、詰まったまま言い終えた
    #   発話として扱う（相手は止められていない＝重ねて応じる）。
    if kind == "重ねて応じる" and out[0].get(UNSAID_KEY):
        out[0].pop(UNSAID_KEY)
        out[0]["utterance"] = out[0]["utterance"].rstrip().rstrip("、") + "。"
        if out[1].get(OVERLAP_KEY) == "interrupt":
            out[1][OVERLAP_KEY] = "coop"
    # 重ねて応じる: 言葉探し・先回りなら相手の1つめは言い終えていない形（試走8:「…昔の映画を観ているんです。」→「映画ですね。」）。
    #   主話者の重ねが相手の言葉のくり返しだけなら捨てる。
    if kind == "重ねて応じる":
        vi = SCENE_VARIANTS["重ねて応じる"].index(variant) if variant in SCENE_VARIANTS["重ねて応じる"] else None
        if vi in (0, 1) and not _TRAILING_OPEN.search(out[0]["_raw"]):
            return None, f"相手が言い終えている（{out[0]['_raw']}）"
        cw, pw = _words(out[1]["utterance"]), _words(out[0]["utterance"])
        if cw and cw <= pw:
            return None, f"重ねが相手の言葉のくり返し（{out[1]['utterance']}）"
    # ★試走12: 重ねて応じたあと相手がお礼で受ける（「共感していただけて嬉しいです。」）、場面のターンに「心地」
    #   （場面のターンは後処理の calm_word_cap を通らないので、Gemma の癖の「心地いい」が残る）
    if kind == "重ねて応じる" and re.search(r"ありがと|嬉しいです|うれしいです", str(out[2].get("utterance", ""))):
        return None, f"重ねられた相手がお礼で受ける（{out[2]['utterance']}）"
    for x in out:
        if "心地" in str(x.get("utterance", "")) or "心地" in str(x.get(UNSAID_KEY, "")):
            return None, f"場面に「心地」（{x['utterance']}）"
    # 譲ったのに相手が黙る場面: 4つめ（主話者のフォロー）の前に長い間を置く（retime の fixed-pause。1.5〜3.0秒）
    if kind == "同時に話し始めて譲り、相手が黙る":
        follow = str(out[3].get("utterance", ""))
        if "さっきの続き" in follow:
            return None, f"フォローが決まった前置き（{follow}）"
        if _YIELD_RE.search(follow) or _core(follow) < 8 or re.match(r"^(?:あ、)?(?:すみません|すいません)。?$", follow):
            return None, f"相手が黙ったあとのフォローになっていない（{follow}）"
        import zlib
        # ★2026-11-11 試走8の判定で 1.5〜3.0秒は長すぎるとされた → 1.2〜2.2秒（人のふつうの間は p90 0.86秒）
        out[3][PAUSE_KEY] = round(1.2 + (zlib.crc32(follow.encode()) % 11) / 10.0, 1)
    for x in out:
        x.pop("_raw", None)
    prev = turns[a - 1] if a > 0 else None
    nxt = turns[b + 1] if b + 1 < len(turns) else None
    ctx = ([dict(prev)] if prev is not None else []) + [dict(x) for x in out] \
        + ([dict(nxt)] if nxt is not None else [])
    off = 1 if prev is not None else 0
    normalize_rp(ctx)
    ev = [(s, i - off) for s, i in rp_events(ctx, main) if off <= i < off + len(out)]
    want = KIND_SHAPE[kind]
    if len(ev) != 1 or ev[0][0] != want or ev[0][1] != 1:
        got = "、".join(f"{SHAPE_LABEL[s]}@{i}" for s, i in ev) or "なし"
        return None, f"場面の形が違う（{got}）"
    new_turns = ctx[off:off + len(out)]
    cut, ovt = new_turns[0], new_turns[1]
    if cut.get(UNSAID_KEY) and str(cut.get("utterance", "")).rstrip().endswith(("？", "?")):
        return None, f"言いかけが問いで終わっている（{cut.get('utterance')}）"
    # ★試走6: 「あ、そうなんですね」＋残り「。シンプルでいい作品ですよ。」＝文の切れ目で止めた（言いかけになっていない）
    for t in new_turns[:2]:
        if t.get(UNSAID_KEY) and str(t[UNSAID_KEY]).lstrip()[:1] in ("。", "？", "?", "！"):
            return None, f"文の切れ目で止めている（{t.get('utterance')}／{t[UNSAID_KEY]}）"
    if want in ("cut_main", "cut_user"):
        if _core(cut.get("utterance")) < CUT_MIN_CORE:
            return None, f"言いかけが短い（{cut.get('utterance')}）"
        heard = "".join(str(t.get("utterance", "")) for t in turns[:a]) + str(cut.get("utterance", ""))
        leak = {w for w in _words(ovt.get("utterance")) & _words(cut.get(UNSAID_KEY))
                if w not in heard}
        if leak:
            return None, f"聞こえていない残りに反応（{'・'.join(sorted(leak))}）"
    if kind == "同時に話し始めて引く（相手が先）":
        frag, p_turn = new_turns[1], new_turns[0]
        if _core(frag.get("utterance")) > COLLISION_MAX_CORE:
            return None, "同時の言いかけが長い"
        if _core(p_turn.get("utterance")) < 8:
            return None, "相手が言い終えていない"
        if _YIELD_RE.search(str(new_turns[2].get("utterance", ""))):
            return None, "黙って引く場面で譲るひと言を言っている"
        if _core(new_turns[2].get("utterance")) < 8:
            return None, f"引いたあとの応じ方が短すぎる（{new_turns[2].get('utterance')}）"
    elif want == "coll_implicit":
        if _core(cut.get("utterance")) > COLLISION_MAX_CORE:
            return None, "同時の言いかけが長い"
        if _core(ovt.get("utterance")) < 8:
            return None, "引いたあと相手が言い終えていない"
        # ★試走9の判定: 相手が重なりに謝るだけ（「すみません。つい話し始めてしまいました。」）、直前とつながらない話題転換
        #   （「あ、そういえば、最近はサブスクで…」）は、きっかけ・流れとも低い
        ou = str(ovt.get("utterance", ""))
        if re.match(r"^(?:あ、)?(?:すみません|すいません|ごめん)", ou) or _YIELD_RE.search(ou):
            return None, f"引いたあと相手が重なりに触れている（{ou}）"
        # 置き換える前の元のターン（a〜b）も含める: 相手が元の次のターンで言うはずだったことを先に話すのは自然
        ctx_text = "".join(str(t.get("utterance", "")) for t in turns[max(0, a - 2):b + 1]) + str(cut.get("unsaid", ""))
        if not _topic_link(ou, ctx_text):
            return None, f"引いたあとの相手の話が直前とつながらない（{ou}）"
        if _YIELD_RE.search(str(new_turns[2].get("utterance", ""))):
            return None, "黙って引く場面で譲るひと言を言っている"
        if _core(new_turns[2].get("utterance")) < 8:     # ★試走9: 「そうですね。」「確かにそうですね。」だけの返し
            return None, f"引いたあとの応じ方が短すぎる（{new_turns[2].get('utterance')}）"
    elif want.startswith("coll") and max(_core(cut.get("utterance")),
                                         _core(ovt.get("utterance"))) > COLLISION_MAX_CORE:
        return None, "同時の言いかけが長い"
    if kind in ("同時に話し始めて譲る", "同時に話し始めて譲る（相手が先）") and \
            re.match(r"^(?:あ、|いえ、)?ありがと", str(new_turns[3].get("utterance", ""))):
        return None, "譲られた相手がお礼から始める"
    if want == "react" and (_core(ovt.get("utterance")) > REACT_MAX_CORE
                            or "？" in str(ovt.get("utterance", ""))):
        return None, f"重なる反応が長いか問い（{ovt.get('utterance')}）"
    # ★試走3: 反応のあとの主話者が「そうなんです。」「そういう作品はいいですよね。」と反応に返事をしていた
    if want == "react":
        if "？" in str(cut.get("utterance", "")):      # ★試走13: イノウエの問いかけに「へえー。」と反応していた
            return None, f"反応の前が問いかけ（{cut.get('utterance')}）"
        cont = str(new_turns[2].get("utterance", ""))
        if _core(cont) < 10 or _ACK_HEAD.match(cont):
            return None, f"反応のあとが話の続きでない（{cont}）"
        # ★試走6: 「私も旅先で珍しい食材を試すのが好きなんです。」→（いいですね）→「試すのが好きなんです。」のくり返し
        if _bigram_overlap(cut.get("utterance"), cont) > 0.4:
            return None, f"反応のあとが同じ話のくり返し（{cont}）"
        if _PLATITUDE.search(cont):
            return None, f"反応のあとが気分・一般論のまとめ（{cont}）"
        if re.fullmatch(r"(?:本当に、?)?いい(?:です)?(?:よ)?ね[。！]?", str(ovt.get("utterance", "")).strip()):
            return None, f"反応が評価のひと言（{ovt.get('utterance')}）"
    if want in ("cut_main", "cut_user") and _ASK_BACK.search(str(ovt.get("utterance", ""))):
        return None, f"割り込みが決まった聞き返し（{ovt.get('utterance')}）"
    if kind == "割り込まれて戻る":
        back = str(new_turns[2].get("utterance", ""))
        q = str(ovt.get("utterance", ""))
        said = str(cut.get("utterance", ""))
        vs = SCENE_VARIANTS["割り込まれて戻る"]
        if variant == vs[0]:
            # ★2026-11-12 試走12: ふつうの言葉の意味を聞く（「視界が、とはどういう意味でしょうか？」）→ 尋ねる言葉はカタカナ語か
            #   漢字3字以上の言葉に限る。答えが言葉のくり返しだけ（「試行錯誤、ですか？」→「はい、試行錯誤です。」）も弾く。
            asked = _words(q) & _words(said)
            if asked and not any(re.fullmatch(r"[ァ-ヶー]{3,}", w_) or re.fullmatch(r"[一-龥々]{3,}", w_) for w_ in asked):
                return None, f"ふつうの言葉の意味を聞いている（{q}）"
            first_s = re.split(r"(?<=[。？])", back)[0]
            if _core(first_s) <= 10 and (_words(first_s) <= _words(q)):
                return None, f"答えが言葉のくり返しだけ（{first_s}）"
        elif variant == vs[1]:
            made = {w_ for w_ in _words(q) if re.fullmatch(r"[ァ-ヶー]{3,}", w_)} - _words(said)
            if made:
                return None, f"聞き返しで言っていない語を作っている（{q}）"
        elif variant == vs[2] and "？" in q:
            return None, f"問いでないはずのひと言が問い（{q}）"
        wu = _words(cut.get(UNSAID_KEY))
        if wu and not (wu & _words(back)):
            return None, "言いかけた話に戻っていない"
        # ★試走8: 戻るときに言いかけの頭から言い直す（「そういう風に、細かく調整して…」）→ 言いかけたところから続ける
        if _bigram_overlap(back, cut.get("utterance")) > 0.5:
            return None, f"言いかけを頭から言い直している（{back}）"
    if want == "cut_main" and kind == "割り込まれて応じる" and "取り違え" in str(variant):
        # ★2026-11-13 試走13: 正す型で、相手がイノウエ自身の経験の言い方を「いや、Xっていうか、Y」と正していた（6中4。
        #   「いや、絶望っていうか、かなり心細かったんじゃないですか？」）→ 正す中身が相手の前の発話に出た事柄か確かめる。
        before = "".join(str(t.get("utterance", "")) for t in turns[:a] if isinstance(t, dict) and t.get("speaker") == partner)
        if not (_words(ovt.get("utterance")) & _words(before)):
            return None, f"正す中身が相手の前の話に無い（{ovt.get('utterance')}）"
    if want == "cut_main" and kind == "割り込まれて応じる":
        resp = str(new_turns[2].get("utterance", ""))
        if _echo_only(resp, ovt.get("utterance")):
            return None, f"割り込みへの返しが相手の言葉のくり返しだけ（{resp}）"
    # ★試走5: 場面のうち重ねていないターン（譲るひと言・反応のあとの続き など）を、あとの校閲・後処理が書き換えていた
    #   （「本当に、気をつけないと…」）。場面のターンは全部印を付けて守る。
    for t in new_turns:
        t[SCENE_KEY] = kind
    return new_turns, ""


# ── Stage2 の後処理・検査 ─────────────────────────────────────────────

def _speaker(t) -> str:
    return str(t.get("speaker", "")) if isinstance(t, dict) else ""


def normalize_rp(turns: list, dropped: list | None = None) -> None:
    """LLM が書いた "unsaid" / "overlap" を検査して整える（その場で書き換える）。

    - 値の崩れ（未知の overlap・空の unsaid）は外す。言いかけの句点は外す（文の途中なので）。
    - "unsaid" は、次が相手の interrupt / collision のとき、または自分が collision の2つめのときだけ残す。
      それ以外は残りをつないで1文に戻す（割り込まれていない言いかけ）。
    - "overlap" は直前が相手のターンのときだけ残す。interrupt なのに直前が言いかけでなければ coop に、
      coop なのに直前が言いかけなら interrupt に直す。collision は2人とも言いかけのときだけ。
      react は次が直前の話者（話し続ける側）のときだけ。
    """
    drop = dropped.append if dropped is not None else (lambda _: None)
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        ov = t.get(OVERLAP_KEY)
        if ov is not None and ov not in OVERLAP_KINDS:
            t.pop(OVERLAP_KEY)
            drop(f"{i}:overlap値({ov!r:.12})")
        if UNSAID_KEY in t:
            un = str(t.get(UNSAID_KEY) or "").strip()
            if _core(un) < UNSAID_MIN_CORE:
                t.pop(UNSAID_KEY)
                if un:
                    t["utterance"] = join_unsaid(str(t.get("utterance", "")), un)
                drop(f"{i}:unsaid短い")
                continue
            t[UNSAID_KEY] = un if un.endswith(("。", "？")) else un.rstrip("、") + "。"
            t["utterance"] = str(t.get("utterance", "")).rstrip().rstrip("。！")
    n = len(turns)
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or UNSAID_KEY not in t:
            continue
        nxt = turns[i + 1] if i + 1 < n and isinstance(turns[i + 1], dict) else None
        if (nxt is not None and _speaker(nxt) != _speaker(t) and nxt.get(OVERLAP_KEY) == "coop"
                and t.get(OVERLAP_KEY) != "collision"):
            nxt[OVERLAP_KEY] = "interrupt"          # 言いかけのあとに重ねた＝さえぎった
            drop(f"{i + 1}:coop→interrupt")
        ok = (t.get(OVERLAP_KEY) == "collision"
              or (nxt is not None and _speaker(nxt) != _speaker(t)
                  and nxt.get(OVERLAP_KEY) in ("interrupt", "collision")))
        if not ok:
            t["utterance"] = join_unsaid(str(t.get("utterance", "")), str(t.pop(UNSAID_KEY)))
            drop(f"{i}:さえぎられない言いかけ")
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or OVERLAP_KEY not in t:
            continue
        prev = turns[i - 1] if i > 0 and isinstance(turns[i - 1], dict) else None
        nxt = turns[i + 1] if i + 1 < n and isinstance(turns[i + 1], dict) else None
        ov = t[OVERLAP_KEY]
        if prev is None or _speaker(prev) == _speaker(t):
            t.pop(OVERLAP_KEY)
            drop(f"{i}:重ねる相手がいない")
            if UNSAID_KEY in t:
                t["utterance"] = join_unsaid(str(t.get("utterance", "")), str(t.pop(UNSAID_KEY)))
            continue
        if ov == "interrupt" and UNSAID_KEY not in prev:
            t[OVERLAP_KEY] = "coop"
            drop(f"{i}:interrupt→coop")
        elif ov == "coop" and UNSAID_KEY in prev:
            t[OVERLAP_KEY] = "interrupt"
            drop(f"{i}:coop→interrupt")
        elif ov == "collision" and UNSAID_KEY not in prev and UNSAID_KEY not in t:   # どちらか1人は言いかけ（黙って引く型）
            t.pop(OVERLAP_KEY)
            for x in (prev, t):
                if UNSAID_KEY in x:
                    x["utterance"] = join_unsaid(str(x.get("utterance", "")), str(x.pop(UNSAID_KEY)))
            drop(f"{i}:collisionの言いかけ不足")
        elif ov == "react" and (nxt is None or _speaker(nxt) != _speaker(prev)):
            t.pop(OVERLAP_KEY)
            drop(f"{i}:reactの後が続きでない")
        if t.get(OVERLAP_KEY) != "collision" and UNSAID_KEY in t:
            nn = turns[i + 1] if i + 1 < n and isinstance(turns[i + 1], dict) else None
            if not (nn is not None and _speaker(nn) != _speaker(t)
                    and nn.get(OVERLAP_KEY) in ("interrupt", "collision")):
                t["utterance"] = join_unsaid(str(t.get("utterance", "")), str(t.pop(UNSAID_KEY)))


def rp_events(turns: list, main: str) -> list[tuple[str, int]]:
    """台本の重なりの場面を (形, 重ねたターンの番号) で列挙する（形は KIND_SHAPE の値）。"""
    out = []
    n = len(turns)
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or not t.get(OVERLAP_KEY):
            continue
        ov, is_main = t[OVERLAP_KEY], _speaker(t) == main
        if ov == "interrupt":
            out.append(("cut_user" if is_main else "cut_main", i))
        elif ov == "collision":
            nxt = turns[i + 1] if i + 1 < n and isinstance(turns[i + 1], dict) else None
            y = (_speaker(nxt) if nxt is not None and _YIELD_RE.search(str(nxt.get("utterance", "")))
                 else "")
            shape = "coll_main" if y == main else "coll_user" if y else "coll_none"
            prev_t = turns[i - 1] if i > 0 and isinstance(turns[i - 1], dict) else {}
            if UNSAID_KEY not in t:          # 2人目が言い終えた＝1人目が黙って引いた（2026-11-11）
                shape = "coll_implicit" if _speaker(prev_t) == main else "other"
            elif UNSAID_KEY not in prev_t:   # 1人目が言い終えた＝2人目が黙って引いた（相手が先。2026-11-12）
                shape = "coll_implicit" if _speaker(t) == main else "other"
            # 主話者が譲ったのに相手が黙り、間をおいて主話者が続ける（2026-11-11）
            nn = turns[i + 2] if i + 2 < n and isinstance(turns[i + 2], dict) else None
            if shape == "coll_main" and nn is not None and _speaker(nn) == main and nn.get(PAUSE_KEY):
                shape = "coll_silent"
            out.append((shape, i))
        elif ov == "coop":
            out.append(("coop" if is_main else "other", i))
        elif ov == "react":
            out.append(("react" if not is_main else "other", i))
    return out


def rp_shape_problem(turns: list, main: str, plan: dict) -> str | None:
    """指定した重なりの場面が台本に揃っていなければ、その理由（揃っていれば None）。"""
    if not is_rp_plan(plan) or not main:
        return None
    want = Counter(KIND_SHAPE[k] for k in plan.get("overlaps") or [] if k in KIND_SHAPE)
    ev = rp_events(turns, main)
    have = Counter(s for s, _ in ev)
    miss = [f"{SHAPE_LABEL[s]} {have[s]}/{c}" for s, c in want.items() if have[s] < c]
    if miss:
        return "場面の不足: " + "、".join(miss)
    if len(ev) > sum(want.values()) + 2:
        return f"重なりが多すぎる（{len(ev)}回・指定{sum(want.values())}回）"
    for s, i in ev:
        t, prev = turns[i], turns[i - 1]
        if s in ("cut_main", "cut_user") and _core(prev.get("utterance")) < CUT_MIN_CORE:
            return f"turn{i - 1} の割り込まれた言いかけが短い（「{prev.get('utterance')}」）"
        if s.startswith("coll") and max(_core(prev.get("utterance")),
                                        _core(t.get("utterance"))) > COLLISION_MAX_CORE:
            return f"turn{i - 1}〜{i} の同時の言いかけが長い"
        if s == "react" and (_core(t.get("utterance")) > REACT_MAX_CORE
                             or "？" in str(t.get("utterance", ""))):
            return f"turn{i} の重なる反応が長いか問いになっている（「{t.get('utterance')}」）"
    return None


def rp_summary(turns: list, main: str) -> list[dict]:
    """確定した台本の重なりの場面の一覧（dialogue.overlaps に書く。検証用）。"""
    return [{"turn": i, "shape": s, "label": SHAPE_LABEL[s], "speaker": _speaker(turns[i])}
            for s, i in rp_events(turns, main)]


def rp_edit_problem(turn: dict, old: str, new: str) -> str:
    """校閲が重なりの場面のターンを直すとき、場面の形を壊す直しなら理由（壊さなければ ""）。

    ★2026-11-11 ユーザ指摘「たまに日本語が変」。場面のターンは校閲から外していたので、場面の日本語は誰も見ていなかった。
    言いかけ（"unsaid"）と同時の言いかけは直させない（残りとつながらなくなる・最初のひと言だけの形が崩れる）。
    ほかは、場面での役割（短い反応・譲るひと言・話の続き）を保つ直しだけ通す。
    """
    if not is_rp_turn(turn):
        return ""
    if turn.get(UNSAID_KEY):
        return "言いかけは直さない"
    if turn.get(OVERLAP_KEY) == "collision" and turn.get(UNSAID_KEY):
        return "同時の言いかけは直さない"
    # ★試走9: 重ねた短い発話を校閲が「掃除をされたんですね」のような相手の言葉のくり返しに書き換えていた → 直させない
    if turn.get(OVERLAP_KEY) in ("react", "coop"):
        return "重ねた短い発話は直さない"
    if _YIELD_RE.search(old) and not _YIELD_RE.search(new):
        return "譲るひと言でなくなる"
    if _ACK_HEAD.match(new) and not _ACK_HEAD.match(old):
        return "受けのひと言から始まる"
    if not 0.5 <= (_core(new) + 1) / (_core(old) + 1) <= 1.6:
        return "長さが大きく変わる"
    return ""


# 校閲（proofread_prompt）に、重なりの場面のある対話でだけ足す注意（2026-11-11）
RP_PROOFREAD_NOTE = """

# この対話について（声が重なる場面を含む台本）
- 「（相手の話をさえぎって）」などの印と「――（ここで相手の声と重なって言いやめた）」は、声が重なる場面の説明です。
  印の付いた発話も日本語として壊れていれば直してかまいませんが、印そのものは書かないでください。言いかけて止まった発話
  （――の付いた発話）は直さないでください。相手に先を譲るひと言は、譲るひと言のまま直してください。
- 敬語の向きの誤りも「1. 壊れた日本語」として直してください: 自分や自分の家族・身内のことに「ご」「お」や尊敬語を付ける、
  相手のことに謙譲語を使う。
- 声が重なる場面のすぐ後の発話が、場面の中身とつながっているかを特に確かめ、つながっていなければ「2. 噛み合わない応答」
  として直してください。
"""


_NOTE_HEAD = {"interrupt": "（相手の話をさえぎって）", "collision": "（相手と同時に話し始めて）",
              "coop": "（相手の話の終わりぎわに重ねて）", "react": "（相手が話している途中に）"}
# 校閲が印を写してきたら消す
RP_NOTE_RE = re.compile(r"（(?:相手の話をさえぎって|相手と同時に話し始めて|相手の話の終わりぎわに重ねて|相手が話している途中に)）"
                        r"|――?（ここで相手の声と重なって言いやめた）")


def rp_render(t: dict) -> str:
    """校閲・書き直しのプロンプトに出す1ターンの本文（重なりの注記つき）。"""
    u = str(t.get("utterance", ""))
    head = _NOTE_HEAD.get(t.get(OVERLAP_KEY), "")
    tail = "――（ここで相手の声と重なって言いやめた）" if t.get(UNSAID_KEY) else ""
    return f"{head}{u}{tail}"
