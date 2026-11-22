"""話者の性別（male / female）の正規化と、書かれていない場合の推定。

性別は Stage 1（generate_metadata）が metadata.speakers[].gender に明示するのが
本筋で、ここにあるのはその取りこぼしを埋めるフォールバック。Stage 4 の
assign_voices は性別が None の話者に性別ペナルティを課さない（＝男女どちらの声も
同点になる）ため、None を残すと年代だけで声が決まってしまう。

Stage 1 / Stage 4 の両方から使うので、重い依存（numpy・torch・MaAI）を持つ
generate_tts.py ではなくこの小さなモジュールに置く。
"""

from __future__ import annotations

GENDERS = ("male", "female")

# 「男性」「女」など日本語で返ってきた場合の受け皿。LLM は英語で返すよう
# 指示してあるが、たまに日本語になる。
_GENDER_ALIASES = {
    "male": "male", "m": "male", "man": "male", "男": "male", "男性": "male",
    "female": "female", "f": "female", "woman": "female",
    "女": "female", "女性": "female",
}


def normalize_gender(value) -> str | None:
    """'male'/'female' に正規化する。判定できなければ None。"""
    if not isinstance(value, str):
        return None
    return _GENDER_ALIASES.get(value.strip().lower())


# --- 役割・肩書きからの推定 -------------------------------------------------
# 性別がほぼ一意に決まる語だけを入れる（「先生」「店員」などは入れない）。
_FEMALE_ROLE_KW = ("母", "妻", "嫁", "娘", "姉", "妹", "祖母", "女性", "彼女",
                   "ママ", "おば", "主婦", "女子", "シングルマザー", "婦人")
_MALE_ROLE_KW = ("父", "夫", "婿", "息子", "兄", "弟", "祖父", "男性", "彼氏",
                 "パパ", "おじ", "主夫", "男子", "シングルファザー")

# --- 名前からの推定 ---------------------------------------------------------
# このパイプラインの話者名はカタカナ（Stage 1 の指示）なので、漢字の名前リスト
# だけでは一件も当たらない。カタカナの語尾で見る。
# 語尾は「性別がほぼ一意に決まるもの」だけを入れ、両性に散らばる語尾
# （キ: ミキ/ユウキ、ト: ミト/ユウト、シ: ヨシ/タカシ、ヤ: サヤ/カツヤ、
#   マ: エマ/タクマ、ノ: アヤノ/シノ）は入れずに None を返させる。
# 長い語尾から先に照合する（「コウ」は男性寄りだが「〜コ」は女性）。
_FEMALE_NAME_SUFFIX = ("コ", "ミ", "ナ", "カ", "エ", "ヨ", "サ", "ホ", "リ",
                       "ネ", "ハ", "ラ", "ユ", "セ", "ワ")
_MALE_NAME_SUFFIX = ("ロウ", "スケ", "ヘイ", "ゾウ", "イチ", "ジロウ", "タロウ",
                     "ヒコ", "タ", "ジ", "オ", "ゴ", "ム", "ダイ", "ジン")
# 2文字の名前では効かせない語尾（「〜オ」は3文字以上なら男性寄り（ミツオ・カズオ）
# だが、2文字だと女性名になる（ミオ・リオ・ナオ））。
_LONG_ONLY_SUFFIX = ("オ",)
# 語尾より優先する完全一致（語尾規則だと取り違えるもの）。
_NAME_EXACT = {
    "ヨシ": None, "ユウ": None, "カイ": "male", "レオ": "male", "ケン": "male",
    "ソウ": "male", "リク": "male", "ハル": None, "アオイ": None, "ヒカル": None,
    "ツバサ": None, "カオル": None, "マコト": None, "イズミ": None, "チヒロ": None,
    # 語尾が逆の性別を指してしまう名前
    "タクミ": None, "イサミ": None, "ツグミ": "female", "メグミ": "female",
    "ナオミ": "female", "アユミ": "female",
}

# 漢字表記の名前（他のデータソース由来のメタデータ用）。
_FEMALE_NAMES = ("麻衣", "由美子", "美咲", "さくら", "陽子", "恵子", "智子", "裕子", "久美子",
                 "あや", "ゆき", "みき", "なお", "えり", "かおり", "彩", "舞", "結衣", "花")
_MALE_NAMES = ("健太", "翔太", "大輔", "拓也", "達也", "亮", "誠", "聡", "学", "浩",
               "翔", "蓮", "大樹", "直樹", "和也", "雄太", "竜也", "隆", "豊", "剛")


def infer_gender(name: str | None, role: str | None = None) -> str | None:
    """名前・役割から性別を推定する。決められなければ None。

    確信が持てないときに None を返すことを優先している（誤った性別を入れると
    声が確実に外れるが、None なら年代だけで選ばれるので半分は当たる）。
    """
    text = role or ""
    f_hit = any(kw in text for kw in _FEMALE_ROLE_KW)
    m_hit = any(kw in text for kw in _MALE_ROLE_KW)
    # 両方に当たる役割（「祖父母」「両親」「夫妻」）は決められない。片方だけ見て
    # 返すと「祖父母」が母にも当たって female になる（実測でサトシを女性と判定）。
    if f_hit != m_hit:
        return "female" if f_hit else "male"
    if f_hit and m_hit:
        return None

    nm = (name or "").strip()
    if not nm:
        return None
    if nm in _NAME_EXACT:
        return _NAME_EXACT[nm]
    for n in _FEMALE_NAMES:
        if n in nm:
            return "female"
    for n in _MALE_NAMES:
        if n in nm:
            return "male"
    # カタカナ語尾（長いものから）
    for suf in sorted(_MALE_NAME_SUFFIX + _FEMALE_NAME_SUFFIX, key=len, reverse=True):
        if not nm.endswith(suf):
            continue
        if suf in _LONG_ONLY_SUFFIX and len(nm) <= 2:
            return None
        return "male" if suf in _MALE_NAME_SUFFIX else "female"
    return None


def resolve_gender(explicit, name: str | None = None, role: str | None = None) -> str | None:
    """明示された性別を優先し、無ければ名前・役割から推定する。"""
    return normalize_gender(explicit) or infer_gender(name, role)
