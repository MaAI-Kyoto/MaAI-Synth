#!/usr/bin/env python3
"""fish に渡す直前のテキストで、読み間違えやすい語と数字の読みを明示する。

fish（OpenAudio）はテキストから直接読みを決めるので、頻度の低い漢語や数字をたまに
読み違える。人手評価と ASR の突き合わせ（check_tts_readings.py）で見つかった実例:
  表書き→ひょうがき / 内祝→うちしゅく / 内のし→からだのし・うたいのし / 一足→ひとわく /
  焼成（「小生」に聞こえる）/ 番号の桁抜け（「1234、5678」→「1234578」）・「8」→「ぱち」
対策は2つで、どちらも**合成入力だけ**を書き換える（台本・manifest・字幕は原文のまま）:
  1. 読み辞書（tts_readings.yaml）: 見つかった語をかなに開く。長い語から順に当てる。
  2. 数字: 番号・電話番号は1桁ずつ、数量（円・時・分・人・個…）は助数詞の音便込みで
     かなに開く。

依存は標準ライブラリと PyYAML だけ（TTS の venv に新しい依存を入れないため）。
読み辞書の置き場所は既定でこのファイルと同じディレクトリの tts_readings.yaml。

  python tts_reading.py "表書きは内祝で、1500円です。注文番号は10582番、090、1234、5678です。"
"""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

_DIGIT = "〇一二三四五六七八九"
_Z2H = str.maketrans("０１２３４５６７８９，", "0123456789,")

# ── 数を読む ─────────────────────────────────────────────────────────────
_ONES = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]
_HYAKU = {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく", 4: "よんひゃく", 5: "ごひゃく",
          6: "ろっぴゃく", 7: "ななひゃく", 8: "はっぴゃく", 9: "きゅうひゃく"}
_SEN = {1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん", 5: "ごせん",
        6: "ろくせん", 7: "ななせん", 8: "はっせん", 9: "きゅうせん"}
_JUU = {1: "じゅう", 2: "にじゅう", 3: "さんじゅう", 4: "よんじゅう", 5: "ごじゅう",
        6: "ろくじゅう", 7: "ななじゅう", 8: "はちじゅう", 9: "きゅうじゅう"}


def _under_10000(n: int) -> str:
    s = ""
    th, n = divmod(n, 1000)
    if th:
        s += _SEN[th]
    h, n = divmod(n, 100)
    if h:
        s += _HYAKU[h]
    t, o = divmod(n, 10)
    if t:
        s += _JUU[t]
    if o:
        s += _ONES[o]
    return s


def read_number(n: int) -> str:
    """整数をかなで読む（0〜9999億）。「いっせんまん」ではなく「せんまん」。

    それより大きい数（電話番号・番号が数字でつながったもの）は1桁ずつ読む。
    """
    if n == 0:
        return "ぜろ"
    if n >= 10 ** 12:
        return read_digits(str(n))
    parts = []
    oku, n = divmod(n, 10 ** 8)
    man, n = divmod(n, 10 ** 4)
    if oku:
        parts.append(_under_10000(oku) + "おく")
    if man:
        parts.append(_under_10000(man) + "まん")
    if n:
        parts.append(_under_10000(n))
    return "".join(parts)


# 1桁ずつ読むときの読み（電話番号・注文番号の読み方。2・5 は伸ばす）
_DIGIT_WISE = {"0": "ぜろ", "1": "いち", "2": "にい", "3": "さん", "4": "よん",
               "5": "ごお", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう"}


def read_digits(s: str) -> str:
    return "".join(_DIGIT_WISE[c] for c in s if c in _DIGIT_WISE)


# ── 助数詞の音便 ─────────────────────────────────────────────────────────
# 末尾の数（1〜10 と 100 の位のまるめ）で読みが変わる助数詞だけ個別に持つ。
# 値は {末尾の形: (数の読みの末尾を置き換える文字列, 助数詞の読み)}。
def _tail_kind(n: int) -> str:
    """音便を決める末尾の形。1〜9 の一の位 / 10 / 100 / 1000 / 10000 など。"""
    if n % 10:
        return str(n % 10)
    if n % 100:
        return "10"
    if n % 1000:
        return "100"
    return "1000"


# 促音化する助数詞（本・分・個・件・回・杯・泊・匹・歩・パーセント）
_SOKUON = {
    # kind: (数の読みの末尾 → 置換, 助数詞の頭子音)
    "1": ("いち", "いっ"), "6": ("ろく", "ろっ"), "8": ("はち", "はっ"),
    "10": ("じゅう", "じゅっ"), "100": ("ひゃく", "ひゃっ"),
}


def _sokuon_read(n: int, head_plain: str, head_voiced: str | None,
                 head_semi: str | None, six: bool = True) -> str:
    """促音便する助数詞の読み。head_plain=「ほん」「ふん」「こ」…、
    head_voiced=3 のときの濁音（「ぼん」）、head_semi=促音後の半濁音（「ぽん」「ぷん」）。
    six=False は 6 を促音にしない助数詞（さ行・た行: 「ろくさい」「ろくさつ」「ろくとう」）。"""
    num = read_number(n)
    kind = _tail_kind(n)
    if kind in _SOKUON and (six or kind != "6"):
        old, new = _SOKUON[kind]
        if num.endswith(old):
            return num[:-len(old)] + new + (head_semi or head_plain)
    if kind == "3" and head_voiced:
        return num + head_voiced
    if kind == "4" and head_semi and head_plain in ("ふん",):
        return num + head_semi                           # よんぷん
    return num + head_plain


def _counter_reading(n: int, counter: str) -> str | None:
    """数＋助数詞のかな。知らない助数詞は None（数だけ開く）。"""
    kind = _tail_kind(n)
    num = read_number(n)
    if counter == "円":
        return (num[:-2] + "よ" if kind == "4" and num.endswith("よん") else num) + "えん"
    if counter == "時":
        if kind == "4" and num.endswith("よん"):
            return num[:-2] + "よじ"
        if kind == "7" and num.endswith("なな"):
            return num[:-2] + "しちじ"
        if kind == "9" and num.endswith("きゅう"):
            return num[:-3] + "くじ"
        return num + "じ"
    if counter == "分":
        return _sokuon_read(n, "ふん", "ぷん", "ぷん")        # 3分＝さんぷん
    if counter == "本":
        return _sokuon_read(n, "ほん", "ぼん", "ぽん")
    if counter == "杯":
        return _sokuon_read(n, "はい", "ばい", "ぱい")
    if counter == "泊":
        return _sokuon_read(n, "はく", "ぱく", "ぱく")
    if counter == "足":
        return _sokuon_read(n, "そく", "ぞく", None, six=False)
    # ★2026-09-26 日常会話（汎用・初対面）向けに追加。知らない助数詞のまま数だけ開くと
    #   「いち匹」「いちつ」「さん軒」のように音便が消える（下の残りの数の扱いも参照）。
    if counter == "つ":
        return {1: "ひとつ", 2: "ふたつ", 3: "みっつ", 4: "よっつ", 5: "いつつ", 6: "むっつ",
                7: "ななつ", 8: "やっつ", 9: "ここのつ", 10: "とお"}.get(n)
    if counter == "匹":
        return _sokuon_read(n, "ひき", "びき", "ぴき")
    if counter in ("品", "敗"):
        return _sokuon_read(n, {"品": "ひん", "敗": "はい"}[counter],
                            {"品": "ぴん", "敗": "ぱい"}[counter], {"品": "ぴん", "敗": "ぱい"}[counter])
    if counter in ("軒", "曲", "缶", "か所", "ヶ所", "カ所", "箇所"):
        head = {"軒": "けん", "曲": "きょく", "缶": "かん"}.get(counter, "かしょ")
        return _sokuon_read(n, head, "げん" if counter == "軒" else None, None)
    if counter in ("冊", "着", "周", "勝", "席", "戦", "試合", "級", "頭", "通"):
        head = {"冊": "さつ", "着": "ちゃく", "周": "しゅう", "勝": "しょう", "席": "せき",
                "戦": "せん", "試合": "しあい", "級": "きゅう", "頭": "とう", "通": "つう"}[counter]
        return _sokuon_read(n, head, None, None, six=False)
    if counter in ("個", "件", "回", "階", "ヶ月", "か月", "ヵ月", "カ月"):
        head = {"個": "こ", "件": "けん", "回": "かい", "階": "かい"}.get(counter, "かげつ")
        return _sokuon_read(n, head, "がい" if counter == "階" else None, None)
    if counter in ("人", "名"):
        if counter == "人" and n == 1:
            return "ひとり"
        if counter == "人" and n == 2:
            return "ふたり"
        if counter == "人" and kind == "4" and num.endswith("よん"):
            return num[:-2] + "よにん"
        return num + ("にん" if counter == "人" else "めい")
    # ★2026-11-07 点は 1・8・10 で促音便（いってん・はってん・じゅってん。100点は ひゃくてん のまま）。
    #   以前は下の一覧に入っていて「１点」を「いちてん」と読ませていた（al_v1 試走7でユーザ指摘）。
    if counter == "点":
        if kind in ("1", "8", "10"):
            return _sokuon_read(n, "てん", None, None, six=False)
        return num + "てん"
    if counter in ("枚", "歳", "才", "番", "号", "センチ", "ミリ", "キロ", "メートル",
                   "ポイント", "パーセント", "年", "週", "週間", "時間", "日間", "名様", "部屋",
                   "番地", "丁目", "番目", "位"):
        if counter in ("歳", "才"):
            if n == 20:
                return "はたち"
            return _sokuon_read(n, "さい", None, None, six=False)
        if counter == "パーセント":
            return _sokuon_read(n, "ぱーせんと", None, None)
        if counter in ("センチ", "キロ"):
            return _sokuon_read(n, {"センチ": "せんち", "キロ": "きろ"}[counter], None, None,
                                six=counter == "キロ")
        head = {"枚": "まい", "番": "ばん", "号": "ごう", "ミリ": "みり",
                "メートル": "めーとる", "ポイント": "ぽいんと", "年": "ねん", "週": "しゅう",
                "週間": "しゅうかん", "時間": "じかん", "日間": "にちかん", "名様": "めいさま",
                "部屋": "へや", "番地": "ばんち", "丁目": "ちょうめ", "番目": "ばんめ", "位": "い"}[counter]
        if counter in ("週", "週間") and kind in ("1", "8", "10"):
            return _sokuon_read(n, head, None, None)
        if counter == "年" and kind == "4" and num.endswith("よん"):
            return num[:-2] + "よねん"
        if counter in ("時間",) and kind == "4" and num.endswith("よん"):
            return num[:-2] + "よじかん"
        if counter == "部屋" and n in (1, 2):
            return {1: "ひとへや", 2: "ふたへや"}[n]
        return num + head
    return None


_DATE_DAY = {1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか", 6: "むいか",
             7: "なのか", 8: "ようか", 9: "ここのか", 10: "とおか", 14: "じゅうよっか",
             20: "はつか", 24: "にじゅうよっか"}
_MONTH = {4: "しがつ", 7: "しちがつ", 9: "くがつ"}

# 助数詞（長いものから）
_COUNTERS = sorted(["円", "時", "分", "本", "杯", "泊", "足", "個", "件", "回", "階", "ヶ月", "か月",
                    "ヵ月", "カ月", "人", "名", "枚", "歳", "才", "番", "号", "点", "センチ",
                    "ミリ", "キロ", "メートル", "ポイント", "パーセント", "年", "週", "週間",
                    "時間", "日間", "名様", "部屋", "番地", "丁目", "番目", "位",
                    "つ", "匹", "品", "敗", "軒", "曲", "缶", "か所", "ヶ所", "カ所", "箇所",
                    "冊", "着", "周", "勝", "席", "戦", "試合", "級", "頭", "通"],
                   key=len, reverse=True)
# 助数詞表に無い語が続く数は開かない（fish に任せる）。開いてよいのは、どの数でも音便が
# 起きない語が続くときだけ（「20代」「3割」「2倍」「第1話」）。表に無い漢字・カタカナ・「つ」
# が続く数を開くと「1日→いち日」「1匹→いち匹」のように読みを固定して間違える。英字が続く
# 「5G」「3D」も英語読みがあるので開かない（2026-09-26）。
_SAFE_AFTER_NUM = "代度割台話段倍限部次列型系"
# 数に掛かる位の字（「2千5百円」「3百人」）。数と一緒に読む
_MULT = {"万": 10 ** 4, "千": 10 ** 3, "百": 100}
_NUM = r"[0-9][0-9,]*"
# 電話番号・番号の区切り（「090、1234、5678」「090-1234-5678」）
_PHONE = re.compile(r"(?<![0-9])([0-9]{2,5})(?:[、,\-ー－]|の、?)([0-9]{2,5})(?:(?:[、,\-ー－]|の、?)([0-9]{2,5}))?(?![0-9])")
# 4桁以上の番号（金額・年・数量の助数詞が続かないもの）
# ★2026-09-26: 日常会話で「8000歩→はちぜろぜろぜろ歩」「予算は3500くらい」を1桁ずつ読んでいた。
#   漢字・カタカナ（表に無い助数詞）が続くものと、きりのいい数（100の倍数）・西暦らしい数は
#   番号とみなさない（_long_id_ok）。
_LONG_ID = re.compile(r"(?<![0-9])([0-9]{4,})(?![0-9,])(?!\s*(?:円|年|人|名|個|本|枚|ポイント|万|千|点|件|回|分|時|パーセント|センチ|キロ|メートル|ミリ|[一-龥々ァ-ヴA-Za-z]))")


def _long_id_ok(s: str) -> bool:
    """4桁以上の数を番号（1桁ずつ読む）とみなすか。きりのいい数・西暦らしい数は数として読む。"""
    n = int(s)
    if s.startswith("0"):
        return True
    if n % 100 == 0:
        return False
    return not (len(s) == 4 and 1900 <= n <= 2099)
_COUNTED = re.compile(r"(?<![0-9])(" + _NUM + r")(万|千|百)?(" + "|".join(map(re.escape, _COUNTERS)) + r")")
_MONTH_DAY = re.compile(r"(?<![0-9])([0-9]{1,2})月(?:([0-9]{1,2})日)?")
_DAY_AFTER = re.compile(r"(?<=[月の])([0-9]{1,2})日(?!間)")


def _int(s: str) -> int | None:
    s = s.replace(",", "")
    return int(s) if s.isdigit() else None


def normalize_numbers(text: str) -> str:
    """数字を読みのかなに開く（番号は1桁ずつ、数量は助数詞込み）。"""
    text = text.translate(_Z2H)

    def phone(m: "re.Match") -> str:
        groups = [g for g in m.groups() if g]
        return "、".join(read_digits(g) for g in groups)
    text = _PHONE.sub(phone, text)

    def month_day(m: "re.Match") -> str:
        mo = int(m.group(1))
        out = (_MONTH.get(mo) or read_number(mo) + "がつ") if 1 <= mo <= 12 else m.group(0)
        if m.group(2):
            d = int(m.group(2))
            out += _DATE_DAY.get(d, read_number(d) + "にち")
        return out
    text = _MONTH_DAY.sub(month_day, text)
    text = _DAY_AFTER.sub(lambda m: _DATE_DAY.get(int(m.group(1)),
                                                  read_number(int(m.group(1))) + "にち"), text)
    # 単独の「N日」（「2泊3日」「14日と24日」）。1日は日付（ついたち）か期間（いちにち）か
    # 決められないので開かない
    text = re.sub(r"(?<![0-9])([0-9]{1,2})日(?!間)",
                  lambda m: m.group(0) if m.group(1) == "1" else
                  _DATE_DAY.get(int(m.group(1)), read_number(int(m.group(1))) + "にち"), text)

    def counted(m: "re.Match") -> str:
        n = _int(m.group(1))
        if n is None:
            return m.group(0)
        # 9桁以上の数（電話番号などがつながったもの）は数としては読まない
        if len(m.group(1).replace(",", "")) >= 9:
            return read_digits(m.group(1)) + m.group(3)
        # 4桁以上の番号（「10582番」「1024号」）は1桁ずつ
        if m.group(3) in ("番", "号", "番地") and len(m.group(1).replace(",", "")) >= 4:
            return read_digits(m.group(1)) + {"番": "ばん", "号": "ごう", "番地": "ばんち"}[m.group(3)]
        if m.group(3) == "日間":
            return ("いちにち" if n == 1 else _DATE_DAY.get(n, read_number(n) + "にち")) + "かん"
        mult = _MULT.get(m.group(2) or "", 1)
        r = _counter_reading(n * mult, m.group(3))
        return r if r else m.group(0)
    text = _COUNTED.sub(counted, text)
    text = _LONG_ID.sub(lambda m: read_digits(m.group(1)) if _long_id_ok(m.group(1))
                        else m.group(0), text)

    # 残った数（助数詞なし）は数として開く。表に無い助数詞らしい語が続くものは開かない
    def rest(m: "re.Match") -> str:
        v = _int(m.group(1))
        if v is None:
            return m.group(0)
        nxt = text[m.end():m.end() + 1]
        if re.match(r"[一-龥々ァ-ヴつA-Za-zＡ-Ｚａ-ｚ]", nxt) and nxt not in _SAFE_AFTER_NUM:
            return m.group(0)
        return read_number(v * _MULT.get(m.group(2) or "", 1))
    text = re.sub(r"(?<![0-9])([0-9][0-9,]*)(万|千|百)?", rest, text)
    return text


# ── 読み辞書 ────────────────────────────────────────────────────────────
_DEFAULT_DICT = Path(__file__).with_name("tts_readings.yaml")


@functools.lru_cache(maxsize=4)
def load_readings(path: str | None = None) -> list[tuple["re.Pattern", str]]:
    """tts_readings.yaml を読む。形式は「表層: 読み」または「表層: {reading: 読み, not_before: 正規表現}」。"""
    import yaml
    p = Path(path) if path else _DEFAULT_DICT
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    items = []
    for surf, v in (raw.get("readings") or {}).items():
        if isinstance(v, dict):
            reading, nb, na = v.get("reading", ""), v.get("not_before"), v.get("not_after")
        else:
            reading, nb, na = str(v), None, None
        pat = (f"(?<!{na})" if na else "") + re.escape(str(surf)) + (f"(?!{nb})" if nb else "")
        items.append((len(str(surf)), re.compile(pat), reading))
    items.sort(key=lambda x: -x[0])                       # 長い語から当てる
    return [(rx, r) for _, rx, r in items]


def apply_readings(text: str, path: str | None = None) -> str:
    for rx, reading in load_readings(path):
        text = rx.sub(reading, text)
    return text


# ★2026-11-07 動詞の連用形に付く「方」（やり方・見え方・考え方）を「かた」に開く。fish は「世界の見え方」を
#   「みえほう」と読んだ（al_v1 試走7でユーザ指摘）。直前が動詞の連用形の語尾（い段・え段のかな）のときだけ。
#   ・「い」は外す（「大きい方」「したい方」は ほう）。「い方」で かた の語は _KATA_I に並べる。
#   ・方向・方面・方法・方角・方々・方針・方式・方言・方程式 は別の語。明け方（あけがた）・同じ方（ほう）は外す。
#   推定の規則なので既定は無効（tts.fish_speech.reading_normalize.verb_kata で有効にする）。
_KATA = re.compile(r"(?<=[きしちにみりえけせてねべめれぎじびげぜで])方(?![向面法角々針式言程])")
_KATA_NG = re.compile(r"同じ方|明け方")
_KATA_I = re.compile(r"(言い|使い|買い|扱い|付き合い|向き合い|払い|洗い|習い|誘い|戦い|笑い|吸い|会い|伺い)方"
                     r"(?![向面法角々針式言程])")


def verb_kata(text: str) -> str:
    """動詞の連用形＋「方」の「方」を「かた」にする（明け方・同じ方などは触らない）。"""
    if "方" not in text:
        return text
    keep = [(m.start(), m.end()) for m in _KATA_NG.finditer(text)]

    def sub(m):
        i = m.start() if m.re is _KATA else m.end(1)
        if any(a <= i < b for a, b in keep):
            return m.group(0)
        return m.group(0)[:-1] + "かた" if m.re is _KATA else m.group(1) + "かた"
    text = _KATA_I.sub(sub, text)
    return _KATA.sub(sub, text)


def to_tts_text(text: str, readings_path: str | None = None, numbers: bool = True,
                kata: bool = False) -> str:
    """合成入力用のテキスト（読み辞書 → 動詞＋方 → 数字の順）。"""
    text = apply_readings(text, readings_path)
    if kata:
        text = verb_kata(text)
    if numbers:
        text = normalize_numbers(text)
    return text


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["表書きは内祝で、1500円です。注文番号は10582番、電話は090、1234、5678です。"
                                 "10月1日の8時に、3本と6個、1泊で8000円。"]:
        print(arg)
        print("→", to_tts_text(arg))


# ── 発話ごとの読みの注記（Stage2 の annotate_readings が付ける turn["yomi"]）──────────
# 「何」の注記が後ろの助数詞・漢字と1語になるとき、その語ごと開く読み。注記は「何」だけを
# 「なん」にするので、「何時くらい」が「なん時くらい」になり fish が「なんとき」と読んでいた
# （2026-10-04 output_ft_v1_gemma_inoue_cmp_new 4bf21312）。「人」「本」「日」も同じく読みが変わる。
_NAN_COMPOUND = {
    "時": "なんじ", "人": "なんにん", "本": "なんぼん", "日": "なんにち", "名": "なんめい",
    "杯": "なんばい", "階": "なんがい", "分": "なんぷん", "年": "なんねん", "軒": "なんげん",
    "匹": "なんびき", "着": "なんちゃく", "歳": "なんさい", "回": "なんかい", "度": "なんど",
    "個": "なんこ", "枚": "なんまい", "件": "なんけん", "十": "なんじゅう", "曜": "なんよう",
    "色": "なにいろ", "気": "なにげ", "物": "なにもの", "事": "なにごと",
}


def apply_yomi(text: str, yomi) -> str:
    """注記 [{"phrase": "上司の方", "surface": "方", "reading": "かた"}, ...] を当てる。

    語句（phrase）を本文の前から順に探し、その中の語（surface）だけを読みに替える。
    Stage4 の refine が本文を書き換えて語句が見つからなくなった注記は飛ばす。
    """
    if not yomi or not text:
        return text
    pos = 0
    for it in yomi:
        ph, sf, rd = it.get("phrase", ""), it.get("surface", ""), it.get("reading", "")
        if not ph or not sf or not rd or sf not in ph:
            continue
        if sf == "何":
            k = ph.find("何")
            nxt = ph[k + 1:k + 2]
            if nxt in _NAN_COMPOUND:
                sf, rd = "何" + nxt, _NAN_COMPOUND[nxt]
        k = text.find(ph, pos)
        if k < 0:
            k = text.find(ph)                     # 順序が崩れていても1回は探す
            if k < 0:
                continue
        new_ph = ph.replace(sf, rd, 1)
        text = text[:k] + new_ph + text[k + len(ph):]
        pos = k + len(new_ph)
    return text


# ── generate_tts から使う入口 ─────────────────────────────────────────────
# generate_tts は __main__ として動き、refine_maai は同じファイルを別モジュール
# 「generate_tts」として読み込む。設定をこのモジュールに持たせれば両方から同じものが見える。
_CFG: dict = {"enabled": False, "path": None, "numbers": True, "verb_kata": False}


def configure(cfg) -> None:
    """config の tts.fish_speech.reading_normalize を渡す（true / false / dict）。

    dict のときのキー: enabled（既定 true）, readings（辞書 yaml のパス）, numbers（既定 true）,
    verb_kata（動詞＋「方」を「かた」に開く。既定 false）。
    """
    if isinstance(cfg, dict):
        _CFG.update(enabled=bool(cfg.get("enabled", True)),
                    path=cfg.get("readings") or None,
                    numbers=bool(cfg.get("numbers", True)),
                    verb_kata=bool(cfg.get("verb_kata", False)))
    else:
        _CFG.update(enabled=bool(cfg), path=None, numbers=True, verb_kata=False)


def apply_if_enabled(text: str) -> str:
    if not _CFG["enabled"] or not text:
        return text
    return to_tts_text(text, _CFG["path"], _CFG["numbers"], _CFG["verb_kata"])


def turn_text(turn: dict) -> str:
    """ターンの合成用の本文（読みの正規化が有効なら注記を当てた本文、無効なら本文そのまま）。"""
    text = turn.get("text") or ""
    return apply_yomi(text, turn.get("yomi")) if _CFG["enabled"] else text
