"""主話者(ch0)のペルソナ（config: metadata.main_speaker.persona_file / persona_files）。

シナリオのプロンプトパックとは別に、固定人物（inoue など）の人物像を
Stage1（build_user_prompt）と Stage2（build_dialogue_prompt）の両方に差し込む。
ファイルは personas/<name>.yaml。項目がすべて空ならペルソナ無しと同じ扱いにし、
プロンプトは一字も変えない（enrich の数値は現行プロンプトに対して調整済みのため）。

★2026-10-30 persona_files（複数のペルソナから対話ごとに1つを引く）を追加。ユーザ依頼
「システムのプロンプト（inoue のプロンプト）を複数用意し、各対話でランダムに1つ選ぶ。
ft や他のシナリオと共通に」。Stage1 が引いたファイルをメタデータの main_speaker_persona に
書き、Stage2 はそれを読む（Stage3/4 には metadata ごと運ばれるので、どの対話がどの
ペルソナで作られたかは後から分かる）。persona_file（1つだけ）の config はプロンプトも
メタデータも従来と一字も変わらない。
"""

import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

# 対話ごとに引いたペルソナのファイル（メタデータのキー）。Stage1 が書き、Stage2 が読む。
META_KEY = "main_speaker_persona"

# (YAML のキー, プロンプトでの見出し)。この順でプロンプトに並べる。
PERSONA_FIELDS = (
    ("personality", "性格・人柄"),
    ("speech_style", "話し方・口調"),
    ("background", "経歴・背景"),
    ("habits", "よく使う言い回し・癖"),
    ("avoid", "この人物がしないこと"),
    ("disfluency", "言いよどみ・フィラー"),
)
# Stage2（台本）にだけ渡す項目。Stage1（メタデータ）に渡すと、LLM がその文を speakers[].personality
# に書き写し、Stage2 で人物設定と二重にフィラーを促すことになる（2026-10-03 実測: 002 の100件すべてで
# 性格欄に「えー」「あのー」が書き写され、イノウエ 2.16・相手 0.38 個per100字に偏った）。
STAGE2_ONLY_FIELDS = {"disfluency"}


def _join_lines(text: str) -> str:
    """YAML の | ブロックの折り返しをつなぐ。日本語は詰め、英数字どうしの間だけ空白にする。"""
    out = ""
    for line in (x.strip() for x in text.splitlines()):
        if not line:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() and line[0].isascii() and line[0].isalnum():
            out += " "
        out += line
    return out


def render_persona(data: dict, stage2: bool = False) -> str:
    """ペルソナの dict をプロンプト用の箇条書きにする。空の項目は出さない。

    stage2=False（Stage1）では STAGE2_ONLY_FIELDS を出さない。
    """
    lines = []
    for key, label in PERSONA_FIELDS:
        if key in STAGE2_ONLY_FIELDS and not stage2:
            continue
        v = data.get(key)
        if isinstance(v, list):
            v = "、".join(str(x).strip() for x in v if str(x).strip())
        v = _join_lines(str(v or ""))
        if v:
            lines.append(f"- {label}: {v}")
    unknown = set(data) - {k for k, _ in PERSONA_FIELDS}
    if unknown:
        raise ValueError(f"ペルソナに未知の項目があります: {sorted(unknown)}"
                         f"（使えるのは {[k for k, _ in PERSONA_FIELDS]}）")
    return "\n".join(lines)


def load_persona_path(path: str, stage2: bool = False) -> str:
    """ペルソナファイル1つを読み、プロンプト用の文字列を返す。空なら ""。

    stage2=True は Stage2（台本）用で、STAGE2_ONLY_FIELDS（言いよどみ）も含める。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ペルソナファイルが見つかりません: {p}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"ペルソナファイルは項目: 値 の形で書いてください: {p}")
    return render_persona(data, stage2=stage2)


def load_persona(main_speaker: dict | None, stage2: bool = False) -> str:
    """main_speaker.persona_file を読み、プロンプト用の文字列を返す。無い・空なら ""。

    stage2=True は Stage2（台本）用で、STAGE2_ONLY_FIELDS（言いよどみ）も含める。
    persona_files（対話ごとに引く）を使う config では "" を返す（対話ごとの文面は
    PersonaPicker / persona_of で得る）。
    """
    files = persona_files(main_speaker)        # persona_file との併記はここで弾く
    path = (main_speaker or {}).get("persona_file")
    if files or not path:
        return ""
    if not Path(path).exists():
        raise FileNotFoundError(f"metadata.main_speaker.persona_file が見つかりません: {path}")
    return load_persona_path(path, stage2=stage2)


def persona_files(main_speaker: dict | None) -> dict[str, float]:
    """main_speaker.persona_files（対話ごとに1つ引く候補）を {パス: 重み} で返す。無ければ {}。

    リストなら等しい重み、{パス: 重み} なら重みどおり。重み 0 以下は外す。
    persona_file（1つだけ）と同時に書くのは誤り（どちらが効くか分からなくなる）。
    継承元の persona_file を消すときは persona_file: null と書く。
    """
    ms = main_speaker or {}
    src = ms.get("persona_files")
    if not src:
        return {}
    if ms.get("persona_file"):
        raise ValueError("metadata.main_speaker に persona_file と persona_files の両方があります。"
                         "どちらか一方にしてください（継承元の persona_file は persona_file: null で消せます）")
    if isinstance(src, (list, tuple)):
        out = {str(p): 1.0 for p in src if p}
    elif isinstance(src, dict):
        out = {str(p): float(w) for p, w in src.items() if p and w is not None and float(w) > 0}
    else:
        raise ValueError(f"metadata.main_speaker.persona_files はリストか {{パス: 重み}} で書いてください: {src!r}")
    if not out:
        raise ValueError(f"metadata.main_speaker.persona_files に使えるファイルがありません: {src!r}")
    return out


@lru_cache(maxsize=None)
def _cached_persona(path: str, stage2: bool) -> str:
    return load_persona_path(path, stage2=stage2)


def persona_of(metadata: dict | None, default: str = "", stage2: bool = True) -> str:
    """その対話の主話者ペルソナの文面。メタデータに main_speaker_persona が無ければ default。

    Stage2 は config 全体のペルソナ（persona_file）を default に渡す。persona_files の
    config ではメタデータに書かれたファイルを読む（同じファイルは1回だけ読む）。
    """
    path = (metadata or {}).get(META_KEY)
    if not path:
        return default
    return _cached_persona(str(path), stage2)


class PersonaPicker:
    """persona_files から対話ごとに1つを引く（Stage1）。

    引き方はランダム。ただし本数は重みどおりに揃える（発注の少ないものから配り、
    同数のあいだは乱数で選ぶ＝等しい重みなら候補数ごとに並びを毎回シャッフルするのと同じ）。
    DiversityTracker.next_style / TopicDeck と同じ方式で、20対話の試走でも全部のペルソナが
    ほぼ同数ずつ出る。既存メタデータの分は add_used で数える（追記しても比率を保つ）。
    """

    def __init__(self, main_speaker: dict):
        self.weights = persona_files(main_speaker)
        if not self.weights:
            raise ValueError("metadata.main_speaker.persona_files がありません")
        self.rng = random.Random(int((main_speaker or {}).get("persona_seed", 0)))
        self.issued: Counter = Counter()
        # 読めること・中身があることを最初に確かめる（途中の対話で落ちないように）
        for p in self.weights:
            if not load_persona_path(p):
                raise ValueError(f"persona_files のペルソナが空です: {p}")

    def next(self) -> str:
        path = min(self.weights, key=lambda k: ((self.issued[k] + 1) / self.weights[k],
                                                 self.rng.random()))
        self.issued[path] += 1
        return path

    def add_used(self, path) -> None:
        if path in self.weights:
            self.issued[path] += 1

    def text(self, path: str, stage2: bool = False) -> str:
        return _cached_persona(path, stage2)
