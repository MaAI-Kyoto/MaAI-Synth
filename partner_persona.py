"""ユーザ側（主話者でない方）の人物カード（config: metadata.partner_persona）。

主話者（ch0。persona.py）は固定人物だが、相手側は対話ごとに別の人物カードを引いて
Stage1（メタデータ）と Stage2（台本）に渡す。カードは build_partner_personas.py が
2つの公開データから前もって作った母集団（personas/partner_pool.jsonl）から引く。

  - Nemotron-Personas-Japan（CC BY 4.0。国勢調査に沿った合成の人物）
      性別・年齢・住まい・婚姻・業種・趣味（リストから2〜3個）・姓
      ★紹介文（persona / professional_persona 等）は渡さない。「〜を体系的に進める実務家
        である」のような硬い書き言葉で、台本の口調に写る。
  - RealPersonaChat（CC BY-SA 4.0。実在 233人の自己記述と性格検査）
      性格検査の点数（同じ1人分）→ 話し方の傾向の記述（数値は渡さない）
      同じ人のペルソナ文から2文（出身・職業・家族・年齢を含む文は外す）
      ★「特定の話者へのなりすましに用いない」の注意書きに沿い、1人の10文と点数を
        まとめて写さない。属性は Nemotron 側のものと組み合わせる。

カードは metadata.speakers[相手].persona_card に入り、Stage3/4 は読まない（声の割り当ては
今どおり gender / age_range）。config に partner_persona が無い・メタデータにカードが
無いときは、どのプロンプトも一字も変えない。
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

DECADES = ("20代", "30代", "40代", "50代", "60代", "70代")

# 話し方の傾向に使う性格検査の項目と、低い／高いときの記述。
# 点数は RPC の 233人の中での順位（0〜1）で見る（尺度が項目ごとに違うため）。
# ★記述は「台本に出る話し方」に寄せる。数値で渡しても Gemma はほとんど反応しない
#   （stage2-enriched-s2b: フィラー・笑いの数値指示が効かなかった）。
TRAIT_TEXT = {
    "BigFive_Extraversion": (
        "口数は少なめで、自分から話題を広げることはあまりない。答えは短めになりがち",
        "話し好きで、自分から話題を出したり話を広げたりする"),
    "BigFive_Neuroticism": (
        "落ち着いていて、迷わずはっきり言い切る",
        "少し心配性で自信なさげ。「たぶん」「〜かもしれないです」と言葉を濁したり、言いよどんだりしやすい"),
    "BigFive_Agreeableness": (
        "お世辞や大げさな共感は言わず、思ったことを率直に言う（失礼にはならない）",
        "人当たりがやわらかく、相手の話によく共感する"),
    "BigFive_Openness": (
        "なじみのない話題には深入りせず、身近な話を好む",
        "好奇心が強く、知らない話題にも興味を示して質問する"),
    "BigFive_Conscientiousness": (
        "話があちこちに飛びやすく、細かいことはあいまいに言う",
        "話が順序立っていて、具体的な数や手順で話す"),
    "KiSS18_BasicSkill": (
        "会話のやりとりが少しぎこちなく、返事や切り出しに少し時間がかかることがある",
        "会話の受け渡しがうまく、相手に話を振るのが自然"),
}
TRAIT_EXTREME = 0.25      # 順位が下位・上位この割合に入る項目だけ記述にする
TRAIT_MAX = 3             # 記述は多くても3つ（多いと矛盾した人物像になる）
TRAIT_PLAIN = "話し方に目立った癖はなく、ふつうに受け答えする"
EXEC_MIN_AGE = 40         # これより若いカードの「経営」ラベルは外す（_occupation_line）

# ペルソナ文のうち、カードの属性（Nemotron 側）とぶつかる・個人に寄りすぎる文を外す。
_FACT_DROP = re.compile(
    r"出身|在住|住んで|住み|[都道府県]|[市町村区]|地元|学生|大学|学校|高校生|中学生|授業|サークル|部活"
    r"|受験|就活|就職|バイト|会社|勤め|職|仕事|働|職場|上司|同僚|定年|年金|主婦|主夫|家事|無職|転職"
    r"|[0-9０-９]+歳|[0-9０-９]+才|年齢|男性|女性|子供|子ども|息子|娘|夫|妻|主人|旦那|嫁|結婚|独身|彼氏|彼女"
    r"|孫|家族|両親|父|母|兄|姉|弟|妹|名前"
    # 2回目（2026-10-27 見直し）: 未婚のカードに「子育て」、住まいとぶつかる「大阪生まれ」、職業・学生の文
    r"|子育て|育児|保育|幼稚園|ママ|パパ|恋人|生まれ|育ち|実家|研究室|志望|在宅|起業|事務|デザイナー|講師|教員"
    r"|20[0-9]{2}|２０[０-９]{2}|年内")


def trait_percentiles(people: list[dict]) -> dict[str, list[float]]:
    """項目ごとの全員の点数（昇順）。順位の計算に使う。"""
    return {k: sorted(float(p["personality"][k]) for p in people) for k in TRAIT_TEXT}


def _rank(sorted_vals: list[float], v: float) -> float:
    """v の順位（0〜1。同点は中央）。"""
    lo = sum(1 for x in sorted_vals if x < v)
    hi = sum(1 for x in sorted_vals if x <= v)
    return (lo + hi) / 2 / max(1, len(sorted_vals))


def talk_style(personality: dict, pct: dict[str, list[float]]) -> tuple[list[str], dict]:
    """性格検査の点数 → 話し方の傾向の記述（極端な項目から最大 TRAIT_MAX 個）と各項目の水準。"""
    ranks = {k: _rank(pct[k], float(personality[k])) for k in TRAIT_TEXT}
    levels = {k: ("low" if r <= TRAIT_EXTREME else "high" if r >= 1 - TRAIT_EXTREME else "mid")
              for k, r in ranks.items()}
    picked = sorted((k for k in TRAIT_TEXT if levels[k] != "mid"),
                    key=lambda k: -abs(ranks[k] - 0.5))[:TRAIT_MAX]
    lines = [TRAIT_TEXT[k][0 if levels[k] == "low" else 1] for k in picked]
    return (lines or [TRAIT_PLAIN]), levels


def usable_facts(sentences: list[str]) -> list[str]:
    """RPC のペルソナ文から、カードに入れてよい文だけを残す。"""
    return [s.strip() for s in sentences if s.strip() and not _FACT_DROP.search(s)]


def age_range_of(age: int) -> str:
    return f"{int(age) // 10 * 10}代"


# ── 母集団の読み込みと、対話ごとのカードの引き方（Stage1） ─────────────────


class PartnerPool:
    """partner_pool.jsonl を性別×年代のセルに分けて持ち、対話ごとに1枚ずつ配る。

    年代は config の age_mix（性別ごとの重み）に対して発注の少ないものから配る
    （DiversityTracker.next_partner_role と同じ方式）。同じカードは二度配らない
    （既存メタデータのカードも add_used で使用済みにする＝追記しても重複しない）。
    """

    def __init__(self, cfg: dict):
        path = Path(cfg.get("pool_file") or "personas/partner_pool.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"metadata.partner_persona.pool_file が見つかりません: {path}"
                                    "（build_partner_personas.py で作る）")
        self.use_surname = bool(cfg.get("use_surname", True))
        self.rng = random.Random(int(cfg.get("seed", 0)))
        self.cells: dict[tuple[str, str], list[dict]] = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    self.cells.setdefault((c["gender"], c["age_range"]), []).append(c)
        for k in sorted(self.cells):
            self.rng.shuffle(self.cells[k])
        self.age_mix = {}
        mix = cfg.get("age_mix") or {}
        for g in ("male", "female"):
            w = mix.get(g, mix) if isinstance(mix.get(g, mix), dict) else {}
            w = {d: float(v) for d, v in (w or {d: 1.0 for d in DECADES}).items()
                 if d in DECADES and v and float(v) > 0}
            self.age_mix[g] = w or {d: 1.0 for d in DECADES}
        self.issued = Counter()       # (gender, 年代) ごとの発注数
        self.used: set[str] = set()

    def add_used(self, card: dict):
        """既存メタデータのカードを使用済みにし、年代の発注数にも数える。"""
        if card.get("card_id") and card["card_id"] not in self.used:
            self.used.add(card["card_id"])
            self.issued[(card.get("gender"), card.get("age_range"))] += 1

    def next_card(self, gender: str) -> dict:
        w = self.age_mix[gender]
        decades = [d for d in w if self.cells.get((gender, d))]
        if not decades:
            raise ValueError(f"partner_pool に {gender} のカードがありません")
        dec = min(decades, key=lambda d: ((self.issued[(gender, d)] + 1) / w[d], self.rng.random()))
        self.issued[(gender, dec)] += 1
        cell = self.cells[(gender, dec)]
        while cell:
            c = cell.pop()
            if c["card_id"] not in self.used:
                self.used.add(c["card_id"])
                return c
        raise ValueError(f"partner_pool の {gender}・{dec} を使い切りました"
                         "（build_partner_personas.py の --per-cell を増やして作り直す）")


def _occupation_line(occ: str, age: int | None = None) -> str:
    """Nemotron の業種ラベル（「小売業 中堅 (現在は引退)」「病院 中堅 経営」等）を文にする。

    「経営」は経営層のラベルで、20代にも他の年代と同じ約4%付いている（21歳の卸売業経営者に
    なった）。EXEC_MIN_AGE 歳未満では外して勤め人として扱う。
    """
    s = str(occ or "").strip()
    status = ""
    m = re.search(r"\s*\(現在は(引退|離職)\)$", s)
    if m:
        status = m.group(1)
        s = s[:m.start()].strip()
    if s == "学生":
        return "学生"
    exec_ = False
    m = re.match(r"^(.*?)\s+経営$", s)
    if m:
        s = m.group(1)
        exec_ = age is None or age >= EXEC_MIN_AGE
    m = re.match(r"^(.*?)\s+(大手|中堅|中小)$", s)
    base, size = (m.group(1), m.group(2)) if m else (s, "")
    if exec_ and size:
        now = f"{base}の{size}企業の経営に携わっている"
    elif exec_:
        now = f"{base}の事業を経営している"
    elif size:
        now = f"{base}の{size}企業で働いている"
    else:
        now = f"{base}の仕事をしている"
    past = now[:-2] + "いた"          # 「〜している」→「〜していた」
    if status == "引退":
        return f"今は引退している（以前は{past}）"
    if status == "離職":
        return f"今は仕事をしていない（以前は{past}）"
    return now


def stage1_card_text(card: dict, use_surname: bool = True) -> str:
    """Stage1 プロンプトに入れる「2人目の人物」の段落。"""
    g = "男性" if card["gender"] == "male" else "女性"
    lines = [f"- 性別・年齢: {g}、{card['age']}歳（gender は \"{card['gender']}\"、"
             f"age_range は \"{card['age_range']}\" と書く）"]
    if use_surname and card.get("surname"):
        lines.append(f"- 名前: name は姓「{card['surname']}」の読みをカタカナで書く（例: 佐藤→サトウ）")
    lines.append(f"- 住まい: {card['prefecture']}（ただし会話は標準語）")
    if card.get("marital"):
        lines.append(f"- 家族: {card['marital'].replace(' (子供あり)', '・子どもあり')}")
    lines.append(f"- 仕事: {_occupation_line(card.get('occupation'), card.get('age'))}")
    if card.get("hobbies"):
        lines.append(f"- 趣味: {'、'.join(card['hobbies'])}")
    lines.append(f"- 話し方の傾向: {'。'.join(card['talk_style'])}")
    if card.get("facts"):
        lines.append(f"- 本人のこと: {''.join('「' + f + '」' for f in card['facts'])}")
    return (
        "2人目は次の人物にしてください:\n" + "\n".join(lines) + "\n"
        "2人目の role には、この仕事の具体的な職業・立場を1つだけ書いてください"
        "（例: 小売業→スーパーの店員、介護福祉業→介護福祉士。引退・離職していれば「元〇〇」"
        "や「専業主婦」など今の立場）。店長・工場長・現場監督・社長などの肩書きにはせず、"
        "仕事の種類で書いてください（例: 化学工業→化学メーカーの工場勤務）。2人目の personality には、この話し方の傾向に沿った"
        "性格と話し方を書いてください。scenario と conversation_flow は、この人物の趣味や"
        "本人のことから自然に広がる話題にしてかまいません（全部を使う必要はありません）。\n")


def stage2_card_block(name: str, card: dict) -> str:
    """Stage2 プロンプトに足す相手側の人物設定の段落（主話者の段落の直後）。"""
    lines = [f"- 話し方の傾向: {'。'.join(card['talk_style'])}"]
    if card.get("hobbies"):
        lines.append(f"- 趣味: {'、'.join(card['hobbies'])}")
    if card.get("facts"):
        lines.append(f"- 本人のこと: {''.join('「' + f + '」' for f in card['facts'])}")
    return (f"\n\n【{name}の人物設定】{name}は次のような人です。{name}のセリフの話す量・"
            f"言葉選び・反応のしかたは、この人物設定に合わせてください。人物設定の内容は、"
            f"話の流れで自然に出せるものだけを使い、無理に盛り込まないでください"
            f"（全部を使う必要はありません）。\n" + "\n".join(lines))


def partner_card_of(metadata: dict) -> tuple[str, dict] | None:
    """メタデータの相手側の (名前, カード)。カードが無ければ None。"""
    for sp in metadata.get("speakers") or []:
        if isinstance(sp, dict) and isinstance(sp.get("persona_card"), dict) and sp.get("name"):
            return str(sp["name"]), sp["persona_card"]
    return None


def strip_cards(speakers: list) -> list:
    """speakers のコピーから persona_card を外す（Stage2 のメタデータ JSON に生のカードを出さない）。"""
    return [{k: v for k, v in s.items() if k != "persona_card"} if isinstance(s, dict) else s
            for s in speakers]
