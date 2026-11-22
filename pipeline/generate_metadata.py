"""Stage 1: 対話メタデータの生成"""

import asyncio
import functools
import json
import logging
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from cost_tracker import CostTracker
from partner_persona import PartnerPool, stage1_card_text
from topic_plan import make_deck, stage1_topic_text, topic_plan_of
from persona import META_KEY as PERSONA_KEY, PersonaPicker, load_persona, persona_files
from speaker_gender import infer_gender, normalize_gender
from llm_client import (
    build_async_llm_client,
    build_async_llm_clients,
    endpoint_concurrency,
    extract_json_text,
    run_endpoint_worker_pool,
)

logger = logging.getLogger(__name__)

# ── シナリオパック ────────────────────────────────────────────────────────
# シナリオ（どんな場面の対話を作るか）に依存するのは、システムプロンプトと
# 3つのプール（ジャンル・場面・関係性）だけ。それを scenarios/*.yaml に外出しし、
# config の metadata.scenario_file で選ぶ。多様性追跡・性別補完・ペア比率の配分・
# 複数エンドポイント分散といった仕組みはシナリオに依らないので、この先のコードは
# 1本のまま全シナリオで共有する（コピーを作ると改良が片方にしか入らない）。
DEFAULT_SCENARIO_FILE = "scenarios/general.yaml"


@dataclass(frozen=True)
class ScenarioPack:
    name: str
    system_prompt: str
    genres: list[str]
    settings: list[str]
    relationships: list[str]


@functools.lru_cache(maxsize=None)
def load_scenario(path: str | None = None) -> ScenarioPack:
    """シナリオパック(YAML)を読む。相対パスはリポジトリのルート基準（モジュールは pipeline/ にある）。"""
    p = Path(path or DEFAULT_SCENARIO_FILE)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    with open(p) as f:
        d = yaml.safe_load(f) or {}
    if not d.get("system_prompt"):
        raise ValueError(f"シナリオパック {p} に system_prompt がありません")
    # 3つのプールは任意。省略した軸は多様性コンテキストから丸ごと落ちる
    # （例: 場所を設定しないシナリオでは settings を書かない）。
    return ScenarioPack(
        name=str(d.get("name") or p.stem),
        system_prompt=str(d["system_prompt"]),
        genres=list(d.get("genres") or []),
        settings=list(d.get("settings") or []),
        relationships=list(d.get("relationships") or []),
    )


GENERAL = load_scenario()

# 既定シナリオの中身は従来どおりモジュール変数としても公開する（テストや
# 他スクリプトからの import 互換のため）。実行時に使うのは ScenarioPack の側。
SYSTEM_PROMPT = GENERAL.system_prompt
GENRE_POOL = GENERAL.genres
SETTING_POOL = GENERAL.settings
RELATIONSHIP_POOL = GENERAL.relationships


def fill_speaker_genders(metadata_list: list[dict]) -> tuple[int, int]:
    """speakers[].gender を 'male'/'female' に正規化し、抜けを名前・役割から埋める。

    性別は Stage 4 の声の割り当て（assign_voices）が使う唯一の性別情報なので、
    ここで確実に残す。None のまま下流に流すと男女どちらの声も同点になり、
    年代だけで声が決まってしまう。

    返り値: (LLM が正しく書かなかった話者数, 推定でも埋まらなかった話者数)
    """
    n_bad = n_unknown = 0
    for meta in metadata_list:
        if not isinstance(meta, dict):
            continue
        for sp in meta.get("speakers") or []:
            if not isinstance(sp, dict):
                continue
            g = normalize_gender(sp.get("gender"))
            if g is None:
                n_bad += 1
                g = infer_gender(sp.get("name"), sp.get("role"))
                if g is None:
                    n_unknown += 1
            sp["gender"] = g
    return n_bad, n_unknown


def _age_decade(age_range) -> int | None:
    """'30代' / '30代前半' / 35 などから代表年代(10の位)を取り出す。"""
    if age_range is None:
        return None
    s = str(age_range)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits[:2]) // 10 * 10
    except ValueError:
        return None


def _replace_name_deep(obj, old: str, new: str):
    """metadata 中の文字列を再帰的に置換する（名前の差し替え用）。"""
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_replace_name_deep(v, old, new) for v in obj]
    if isinstance(obj, dict):
        return {k: (v if k == "id" else _replace_name_deep(v, old, new)) for k, v in obj.items()}
    return obj


def enforce_main_speaker(metadata_list: list[dict], spec: dict) -> tuple[list, int, int, int, int]:
    """speakers[0] を必ず spec (gender / age_range) の話者にし、main_speaker_index を書く。

    LLM の指示遵守は当てにしない。ここで機械的に是正する。
      1) spec に合致する話者が居れば index 0 に移す
      2) 性別だけ合う話者しか居なければ、その話者の age_range を spec の値に書き換える
         （性別と違い年代は名前や本文に現れにくいので、後から差し替えても破綻しにくい）
      3) その性別の話者が1人も居なければ破棄
    3) を極力起こさないために pair_mix でその性別を含む組だけを発注しておくこと。

    役割(spec["role"])が指定されていれば（接客の「担当者」など、役割が固定の
    シナリオ）、候補はその role を持つ話者に限る。役割は本文（scenario /
    conversation_flow）と結び付いているので、性別のために別の話者へ乗り換えたり
    role を書き換えたりはしない: role が一致する話者がちょうど1人いてその性別が
    合えば採用（年代だけ違えば 2) と同じく書き換える）、それ以外は破棄。

    名前(spec["name"])が指定されていれば、speakers[0].name をその名前に固定し、
    scenario / conversation_flow の本文中の旧名も同時に置換する。相手と同名・部分一致に
    なる対話は、置換が相手側まで壊すので破棄する。

    返り値: (是正後のリスト, 入替件数, 年代書換件数, 名前差替件数, 破棄件数)
    """
    want_g = normalize_gender(spec.get("gender"))
    want_d = _age_decade(spec.get("age_range"))
    want_n = spec.get("name")
    want_r = str(spec.get("role") or "").strip() or None
    out, n_swap, n_age, n_name, n_drop = [], 0, 0, 0, 0
    for meta in metadata_list:
        if not isinstance(meta, dict):
            continue
        sp = meta.get("speakers") or []
        if len(sp) != 2:
            out.append(meta)
            continue
        if want_r:
            by_role = [i for i, s in enumerate(sp)
                       if str(s.get("role") or "").strip() == want_r]
            if len(by_role) != 1 or sp[by_role[0]].get("gender") != want_g:
                n_drop += 1
                continue
            i = by_role[0]
            if want_d is not None and _age_decade(sp[i].get("age_range")) != want_d:
                sp[i]["age_range"] = spec.get("age_range")
                n_age += 1
        else:
            exact = [i for i, s in enumerate(sp)
                     if s.get("gender") == want_g
                     and (want_d is None or _age_decade(s.get("age_range")) == want_d)]
            if exact:
                i = exact[0]
            else:
                same_g = [i for i, s in enumerate(sp) if s.get("gender") == want_g]
                if not same_g:
                    n_drop += 1
                    continue
                i = same_g[0]
                sp[i]["age_range"] = spec.get("age_range")
                n_age += 1
        if i != 0:
            meta["speakers"] = [sp[i], sp[1 - i]]
            n_swap += 1

        # 名前の固定。LLM が従っていれば何もしない。従っていなければ
        # scenario / conversation_flow の本文まで含めて機械的に差し替える。
        # ★本文を直さずに speakers[].name だけ変えると、Stage2 が本文側の旧名で
        #   台本を書いてしまう（conversation_flow が開始者を決めていたのと同じ罠）。
        if want_n:
            old = (meta["speakers"][0].get("name") or "").strip()
            other = (meta["speakers"][1].get("name") or "").strip()
            if (other == want_n or (old and (old in other or other in old))
                    or (other and (want_n in other or other in want_n))):
                # 相手と同名／部分一致（ユキ と ユキコ 等）。置換すると相手側まで
                # 壊れるので、この対話は捨てる。固定名との部分一致（イノウエ と
                # イノウエダ）も、Stage2 の自分の名前ゲートが誤反応するので捨てる。
                n_drop += 1
                continue
            if old and old != want_n:
                meta = _replace_name_deep(meta, old, want_n)
                n_name += 1
            meta["speakers"][0]["name"] = want_n

        meta["main_speaker_index"] = 0    # Stage4 が ch0 を固定するのに使う
        out.append(meta)
    return out, n_swap, n_age, n_name, n_drop


def attach_partner_card(metadata_list: list[dict], card: dict) -> tuple[list, int, int]:
    """主話者でない方（speakers[1]）に人物カードを付ける（enforce_main_speaker の後に呼ぶ）。

    性別がカードと違う対話は捨てる（名前・本文がその性別で書かれているので書き換えない）。
    年代は声の割り当てに使うのでカードの値に揃える（違えば書き換えて数える）。
    返り値: (残ったリスト, 年代書換件数, 破棄件数)
    """
    out, n_age, n_drop = [], 0, 0
    for meta in metadata_list:
        sp = meta.get("speakers") or []
        if meta.get("main_speaker_index") != 0 or len(sp) != 2 or sp[1].get("gender") != card["gender"]:
            n_drop += 1
            continue
        if sp[1].get("age_range") != card["age_range"]:
            sp[1]["age_range"] = card["age_range"]
            n_age += 1
        sp[1]["persona_card"] = card
        out.append(meta)
    return out, n_age, n_drop


def attach_topic_plan(metadata_list: list[dict], plan: dict):
    """このプロンプトで指定した話題の組をメタデータに残す（Stage2 が同じ順番で書く）。"""
    for meta in metadata_list:
        meta["topic_plan"] = plan


def attach_persona(metadata_list: list[dict], path: str):
    """このプロンプトで引いた主話者のペルソナ（persona_files の1つ）をメタデータに残す。

    Stage2 はこのファイルを読んで【名前の人物設定】段落を書く（persona.persona_of）。
    """
    for meta in metadata_list:
        meta[PERSONA_KEY] = path


# 話者2人の性別の組み合わせの目標比率。
# speakers に gender を書かせると、モデルは2人の例をテンプレートとして読み、
# ほぼ全対話を「男女1人ずつ」にしてしまう（実測: 指示なしで 24/24、
# 「同性同士も作れ」と書き足しても 23/24 が男女）。プロンプトの言い聞かせでは
# 直らないので、build_user_prompt が1件ごとに組み合わせを指定して散らす。
# 男女を半分にすると話者全体の男女比はちょうど 50:50 になる。
PAIR_MIX_TARGET = {"男女": 0.50, "男性同士": 0.25, "女性同士": 0.25}
_PAIR_LABELS = {("female", "male"): "男女", ("male", "male"): "男性同士",
                ("female", "female"): "女性同士"}


class DiversityTracker:
    """既存メタデータの分布を追跡し、重複回避と多様性促進を行う"""

    def __init__(self, scenario: ScenarioPack | None = None,
                 pair_mix: dict | None = None, main_speaker: dict | None = None,
                 partner_roles: dict | None = None, style_mix: dict | None = None,
                 partner_persona: dict | None = None, topic_plan: dict | None = None):
        self.scenario = scenario or GENERAL
        # 性別ペアの目標比率。既定は PAIR_MIX_TARGET。主話者を固定するコーパスでは
        # その性別を含む組だけに絞る（config: metadata.pair_mix）。
        self.pair_mix = dict(pair_mix) if pair_mix else dict(PAIR_MIX_TARGET)
        # 主話者(ch0)の固定属性。{"gender": "male", "age_range": "30代"} 等。
        # role を足すと役割も固定になる（接客の "担当者" 等。enforce_main_speaker 参照）。
        self.main_speaker = dict(main_speaker) if main_speaker else None
        # 主話者のペルソナ（main_speaker.persona_file）。空なら ""＝プロンプトを変えない。
        self.persona = load_persona(self.main_speaker)
        # ★2026-10-30 persona_files（複数）なら対話ごとに1つ引く（persona.PersonaPicker）。
        self.persona_picker = (PersonaPicker(self.main_speaker)
                               if persona_files(self.main_speaker) else None)
        # 「どちらから話し始めるか」を半々に振るためのカウンタ（starter_quota 参照）
        self.starter_issued = 0
        # 2人目の役割の系統 {系統: {"weight": w, "examples": "例1、例2"}}（config: metadata.partner_roles）。
        # 空なら従来どおり LLM に任せる。★2026-10-04 初対面 inoue（002 の 3,000件）で相手の role が
        # 大学生 46%・会社員 32% に偏った（001 でも 38%・32%）。性別の組と同じく末尾で指定して散らす。
        self.partner_roles = {k: v for k, v in (partner_roles or {}).items()
                              if float((v or {}).get("weight", 0)) > 0}
        self.role_issued = Counter()
        # speech_style の目標比率 {"敬語": 0.65, "混合": 0.35}（config: metadata.style_mix）。空なら LLM 任せ。
        # ★2026-10-04 partner_roles で学生が減ると「混合」が 41%→12% に落ちた（試走100件）。
        self.style_mix = {k: float(v) for k, v in (style_mix or {}).items() if v and float(v) > 0}
        self.style_issued = Counter()
        # 2人目の人物カード（config: metadata.partner_persona。partner_persona.py）。
        # 無ければ None＝プロンプトは従来どおり。あれば partner_roles の行の代わりに使う。
        # ★2026-10-27 追加。010 の相手側は 20代 53%・30代 0.3%・70代以上 0、性格欄は
        #   「明るい」「好奇心」「穏やか」「気さく」が大半で、無口・慎重な人がほぼいなかった。
        self.partner_pool = PartnerPool(partner_persona) if partner_persona else None
        if self.partner_pool and not self.main_speaker:
            raise ValueError("metadata.partner_persona は main_speaker（主話者の固定）と一緒に使ってください")
        # 1対話で扱う話題の組と順番（config: metadata.topic_plan。topic_plan.py）。
        # 無ければ None＝プロンプトは従来どおり（conversation_flow の型はシナリオ任せ）。
        # ★2026-10-28 追加。010 は仕事の話の初出が2ターンめ（96% が6ターン以内）で、
        #   1対話が3ターン以上使う話題は平均1.8個しかなかった。
        # ★2026-10-30 topic_plan.style: qa なら質問の組と順番（qa_plan.QADeck）。
        self.topic_deck = make_deck(topic_plan, self.scenario.genres) if topic_plan else None
        self.genre_counts = Counter()
        self.setting_counts = Counter()
        self.relationship_counts = Counter()
        self.name_counts = Counter()
        self.pair_counts = Counter()
        self.pair_pending = Counter()      # 発注済み・未着の指定（pair_quota 参照）
        self.scenario_set: set[str] = set()

    def pair_quota(self, batch_size: int) -> Counter:
        """このバッチで作らせる性別の組み合わせ。累計を目標比率に近づける向きに配る。

        発注した時点で pair_pending に積む。そうしないと、並行して飛んでいる
        リクエストが全部おなじ組み合わせを指定されてしまう（batch_size=1・
        並行度10 の本番構成では10件連続で同じ組になる）。
        """
        quota = Counter()
        total = sum(self.pair_counts.values()) + sum(self.pair_pending.values())
        # 比率 0 のラベルは候補から外す。残すと累計が目標に追いついた時点で
        # 全ラベルのスコアが 0 で並び、dict の順序次第で 0 指定のはずの組が選ばれる。
        labels = [k for k, v in self.pair_mix.items() if v and v > 0]
        if not labels:
            raise ValueError(f"metadata.pair_mix に正の比率がありません: {self.pair_mix}")
        for _ in range(batch_size):
            total += 1
            # 「目標本数 − 現状（発注済みを含む）」が最も大きいものから
            label = max(labels,
                        key=lambda k: (self.pair_mix[k] * total
                                       - self.pair_counts[k] - self.pair_pending[k]
                                       - quota[k]))
            quota[label] += 1
        self.pair_pending.update(quota)
        return quota

    def next_partner_role(self) -> tuple[str, str] | None:
        """次の対話の2人目の役割の系統と例。重みに対して発注の少ない系統から配る。

        実績（生成された role）は見ない。系統の名前と生成された role の対応を機械で
        取るのは脆いので、「指示を重みどおりに出す」までを保証する（next_starter と同じ）。
        """
        if not self.partner_roles:
            return None
        tot = sum(self.role_issued.values()) + 1
        cat = min(self.partner_roles, key=lambda k: (
            (self.role_issued[k] + 1) / float(self.partner_roles[k]["weight"])
            - random.random() * 1e-6 * tot))
        self.role_issued[cat] += 1
        return cat, str(self.partner_roles[cat].get("examples") or "")

    def next_style(self) -> str | None:
        """次の対話の speech_style。目標比率に対して発注の少ないものから配る。"""
        if not self.style_mix:
            return None
        st = min(self.style_mix, key=lambda k: (self.style_issued[k] + 1) / self.style_mix[k])
        self.style_issued[st] += 1
        return st

    def next_persona(self, out: list | None = None) -> str:
        """この対話の主話者ペルソナ（Stage1 用の文面）。

        persona_files があれば1つ引いて、そのパスを out に積む（メタデータに残すため）。
        無ければ config 全体のペルソナ（persona_file。無ければ ""）。
        """
        if not self.persona_picker:
            return self.persona
        path = self.persona_picker.next()
        if out is not None:
            out.append(path)
        return self.persona_picker.text(path)

    def next_starter(self) -> int:
        """次の対話で「何人目から話し始めるか」(1 or 2)。交互に返して半々にする。

        ★誰が先に話すかを決めているのは speakers[] の並び順ではなく
          conversation_flow[0] の本文（パイロット実測: conversation_flow[0] が示す
          開始者 == turns[0].speaker が 19/20）。Stage2 は flow に従って台本を書くので、
          Stage1 の時点で flow の書き出しを指定しないと制御できない。
          主話者を speakers[1] へ動かしても flow の本文は変わらないため無効だった。

        実績のフィードバックは持たない。実際に誰が話し始めたかは Stage2 を通すまで
        分からず、conversation_flow の本文から開始者を機械判定するのは脆いため、
        ここでは「指示を半々に出す」までを保証し、遵守率はパイロットで実測する。
        """
        self.starter_issued += 1
        return 2 if self.starter_issued % 2 == 0 else 1

    def add(self, meta: dict):
        genre = meta.get("genre", "")
        setting = meta.get("setting", "")
        relationship = meta.get("relationship", "")
        scenario = meta.get("scenario", "")
        plan = topic_plan_of(meta)
        if self.topic_deck and plan:
            # 話題は指定して配っているので、LLM が書いた genre ではなく指定のほうを数える。
            self.topic_deck.add_used(plan)
            for t in plan["topics"]:
                self.genre_counts[t] += 1
        elif genre:
            self.genre_counts[genre] += 1
        if setting:
            self.setting_counts[setting] += 1
        if relationship:
            self.relationship_counts[relationship] += 1
        if scenario:
            self.scenario_set.add(scenario)
        speakers = [sp for sp in (meta.get("speakers") or []) if isinstance(sp, dict)]
        for sp in speakers:
            name = sp.get("name")
            if name:
                self.name_counts[str(name)] += 1
            if self.partner_pool and isinstance(sp.get("persona_card"), dict):
                self.partner_pool.add_used(sp["persona_card"])
        if self.persona_picker:
            self.persona_picker.add_used(meta.get(PERSONA_KEY))
        if len(speakers) == 2:
            key = tuple(sorted(str(sp.get("gender")) for sp in speakers))
            if key in _PAIR_LABELS:
                label = _PAIR_LABELS[key]
                self.pair_counts[label] += 1
                if self.pair_pending[label] > 0:   # 発注ぶんの引き当て
                    self.pair_pending[label] -= 1

    def _least_used(self, counter: Counter, pool: list[str], n: int) -> list[str]:
        """プール内の候補を使用頻度の低い順にn件返す。未使用のものを優先。"""
        pool_set = set(pool)
        unused = [x for x in pool if counter[x] == 0]
        random.shuffle(unused)
        used_sorted = sorted(
            [x for x in pool if counter[x] > 0],
            key=lambda x: counter[x],
        )
        combined = unused + used_sorted
        # プール外で使用頻度が高いものも「避けるべき」として認識
        return combined[:n]

    def _overused(self, counter: Counter, top_n: int = 10) -> list[str]:
        """最も使用頻度が高いものをtop_n件返す"""
        return [item for item, _ in counter.most_common(top_n)]

    def build_diversity_context(self, batch_size: int) -> str:
        genre_pool = self.scenario.genres
        setting_pool = self.scenario.settings
        relationship_pool = self.scenario.relationships
        total = sum(self.genre_counts.values()) or 0
        if total == 0:
            # まだデータがない場合はランダムな候補のみ
            text = "以下の候補を参考にしてください:\n"
            for label, pool in (("ジャンル", genre_pool), ("場面", setting_pool),
                                ("関係性", relationship_pool)):
                if not pool:
                    continue
                picks = random.sample(pool, min(batch_size * 2, len(pool)))
                text += f"- {label}候補: {', '.join(picks)}\n"
            return text

        n = max(batch_size * 2, 10)
        recommended_genres = self._least_used(self.genre_counts, genre_pool, n)
        recommended_settings = self._least_used(self.setting_counts, setting_pool, n)
        recommended_rels = self._least_used(self.relationship_counts, relationship_pool, n)

        overused_genres = self._overused(self.genre_counts, 5)
        overused_settings = self._overused(self.setting_counts, 5)

        text = f"これまでに{total}件のメタデータが生成済みです。\n\n"
        text += "【推奨】以下はまだ少ない or 未使用の候補です。優先的に使ってください:\n"
        if genre_pool:
            text += f"- ジャンル（少ない順）: {', '.join(recommended_genres[:n])}\n"
        if setting_pool:
            text += f"- 場面（少ない順）: {', '.join(recommended_settings[:n])}\n"
        if relationship_pool:
            text += f"- 関係性（少ない順）: {', '.join(recommended_rels[:n])}\n"
        text += "\n"
        text += "【避けるべき】以下は既に多いので避けてください:\n"
        if genre_pool:
            text += f"- ジャンル: {', '.join(overused_genres)}\n"
        if setting_pool:
            text += f"- 場面: {', '.join(overused_settings)}\n"
        overused_names = self._overused(self.name_counts, 10)
        # 人物カードが姓を決めるときは「既出の名前を避ける」を出さない（姓の指定とぶつかる）。
        if overused_names and not (self.partner_pool and self.partner_pool.use_surname):
            text += f"- 話者名（既出。別の名前を新しく考える）: {', '.join(overused_names)}\n"
        text += "\n"

        # 既存シナリオからランダムサンプルも提示（大量でも最大20件）
        if self.scenario_set:
            sample_n = min(20, len(self.scenario_set))
            sample = random.sample(sorted(self.scenario_set), sample_n)
            text += "【既出シナリオ例（一部）】以下と同じ・類似は避けてください:\n"
            for s in sample:
                text += f"  - {s}\n"

        return text


def _persona_lines(persona: str, vary: str) -> str:
    """ペルソナ指定時に「1人目の人物像は毎回変える」の行と差し替える文。

    シナリオパック本文の「性格・話し方の癖は対話ごとに変える」（接客パック等）より
    優先させる。人物像は固定しても、vary に挙げたものは毎回変えさせて多様性を残す。
    vary の既定は role の有無で決まる（接客向け／役割も変える向け）。初対面で role を
    固定するときなどは config の main_speaker.persona_vary で上書きする。
    """
    return ("1人目は毎回同じ人物で、性格・話し方は次の人物設定に合わせてください"
            "（この指定は、上の「性格・話し方を対話ごとに変える」という指示より優先します）。"
            "speakers の1人目の personality も、この人物設定に沿って書いてください。\n"
            f"【1人目の人物設定】\n{persona}\n"
            f"ただし{vary}は、毎回まったく違うものにしてください"
            "（同じ場面・同じ用件を繰り返さない）。\n")


_STYLE_HINT = {
    "敬語": "2人とも敬語で話す",
    "混合": "年齢や立場の差、相手の人柄などから、どちらか一方（または両方が部分的に）くだけた話し方をする",
    "タメ口": "2人ともくだけた話し方をする",
}


def _same_generation(main_speaker: dict | None, card: dict) -> bool:
    """主話者と人物カードの年代の差が20歳未満か（年代が読めなければ False＝従来どおり）。"""
    a, b = _age_decade((main_speaker or {}).get("age_range")), _age_decade(card.get("age_range"))
    return a is not None and b is not None and abs(a - b) < 20


def _partner_role_line(tracker: "DiversityTracker | None", card: dict | None = None) -> str:
    """2人目の役割の系統と speech_style の指定（metadata.partner_roles / style_mix が無ければ空行だけ）。

    card（人物カード）があれば、役割の系統の代わりにカードの段落を置く。
    """
    if card is not None and _same_generation(tracker.main_speaker, card):
        # ★2026-10-27 カードで 30代の相手が出るようになり、「混合」で同年代の初対面の相手だけが
        #   イノウエにタメ口を使う対話ができた（試走20対話中 30代の2件）。年の差が20歳未満なら
        #   敬語に固定し、style_mix の比率は年の差のある対話の中で守る（発注数に数えない）。
        style = "敬語"
    else:
        style = tracker.next_style() if tracker else None
    line = ""
    if style:
        hint = _STYLE_HINT.get(style)
        line = (f"この対話の speech_style は「{style}」にしてください"
                + (f"（{hint}）" if hint else "") + "。\n")
    if card is not None:
        return line + stage1_card_text(card, tracker.partner_pool.use_surname) + "\n"
    pick = tracker.next_partner_role() if tracker else None
    if not pick:
        return line + "\n"
    cat, ex = pick
    return line + (f"2人目の role は「{cat}」の人にしてください"
            + (f"（例: {ex}。例にない職業・立場でもよい）" if ex else "")
            + "。role には具体的な職業・立場を1つだけ書き、年代はその立場に合うものにしてください。\n\n")


def build_user_prompt(
    batch_size: int,
    min_turns: int,
    max_turns: int,
    diversity_tracker: DiversityTracker | None = None,
    num_speakers: int | None = None,
    card_out: list | None = None,
    plan_out: list | None = None,
    persona_out: list | None = None,
) -> str:
    """card_out を渡すと、このプロンプトで指定した2人目の人物カードをそこへ追加する（partner_persona）。

    plan_out も同じで、このプロンプトで指定した話題の組を追加する（topic_plan）。
    persona_out には、このプロンプトで引いた主話者のペルソナのパスを追加する（persona_files）。
    """
    prompt = f"{batch_size}個の対話メタデータを生成してください。\n"
    prompt += f"target_turnsは{min_turns}〜{max_turns}の範囲でばらつかせてください。\n"
    if num_speakers:
        # 音声合成パイプライン（2声・2ch前提）に合わせた話者数の固定。
        prompt += f"各対話の話者(speakers)は必ず{num_speakers}人にしてください。\n"
    prompt += "\n"

    if diversity_tracker:
        prompt += diversity_tracker.build_diversity_context(batch_size)
        prompt += "\n"

    # 性別の組み合わせの指定はプロンプト末尾（最も従いやすい位置）に実数で置く。
    pair_label = None
    if num_speakers == 2 and diversity_tracker:
        quota = diversity_tracker.pair_quota(batch_size)
        if batch_size == 1:
            pair_label = next(iter(quota))
            label = next(iter(quota))
            prompt += f"この対話の2人は{label}にしてください。\n\n"
        else:
            spec = "、".join(f"{n}件を{label}" for label, n in quota.items())
            prompt += f"{batch_size}件のうち{spec}にしてください。\n\n"

    # 主話者(ch0)の属性固定も、性別の組み合わせと同じく末尾に置く。
    ms = diversity_tracker.main_speaker if diversity_tracker else None
    card = None
    if ms and diversity_tracker.partner_pool:
        if pair_label is None:
            raise ValueError("metadata.partner_persona は batch_size: 1・num_speakers: 2 のときだけ使えます")
        g = {"男女": "female" if normalize_gender(ms.get("gender")) == "male" else "male",
             "男性同士": "male", "女性同士": "female"}[pair_label]
        card = diversity_tracker.partner_pool.next_card(g)
        if card_out is not None:
            card_out.append(card)
    partner_free = ("2人目の性別は上の組み合わせの指定に従い、年代・性格は自由に多様にしてください。\n"
                    if card is None else "")
    if ms:
        persona = diversity_tracker.next_persona(persona_out)
        name_line = ""
        if ms.get("name"):
            name_line = (
                f"speakers の1人目の name は必ず \"{ms['name']}\" にしてください"
                f"（毎回同じ固定の人物名です）。"
                f"scenario と conversation_flow の本文でも、その人物は必ず"
                f"「{ms['name']}」と書いてください。\n"
                f"2人目には「{ms['name']}」を含む名前を絶対に使わないでください。\n")
        if ms.get("role"):
            # 役割が固定のシナリオ（接客の「担当者」等）。役割は毎回変えさせず、
            # 変えるのは性格・話し方・場面（setting）・関係性だけにする。
            prompt += (
                f"speakers の1人目は必ず role を \"{ms['role']}\"、"
                f"gender を \"{ms.get('gender')}\"、"
                f"age_range を \"{ms.get('age_range')}\" にしてください。\n"
                + name_line
                + (_persona_lines(persona, ms.get("persona_vary") or "勤め先（setting）・相手との関係性・用件")
                   if persona else
                   "ただし1人目の性格・話し方・勤め先（setting）・相手との関係性は、"
                   "これらの制約とは無関係に毎回まったく違うものにしてください"
                   "（同じような人物像を繰り返さない）。\n")
                + partner_free)
        else:
            prompt += (
                f"speakers の1人目は必ず gender を \"{ms.get('gender')}\"、"
                f"age_range を \"{ms.get('age_range')}\" にしてください。\n"
                + name_line
                + (_persona_lines(persona, ms.get("persona_vary") or "役割・場面・相手との関係性")
                   if persona else
                   "ただし1人目の役割・性格・関係性・話し方は、これらの制約とは無関係に"
                   "毎回まったく違うものにしてください（同じような人物像を繰り返さない）。\n")
                + (partner_free.replace("年代・性格", "年代・役割・性格")))
        prompt += _partner_role_line(diversity_tracker, card)
        # 話題の組と順番（topic_plan）。シナリオの conversation_flow の型より後ろに置く。
        if diversity_tracker.topic_deck:
            plan = diversity_tracker.topic_deck.next_plan(card, persona=persona or "")
            if plan_out is not None:
                plan_out.append(plan)
            prompt += stage1_topic_text(plan)
        # ★どちらから話し始めるかは conversation_flow の本文が決める（Stage2 はそれに従う）。
        #   speakers[] の並び替えでは制御できないので、ここで本文の書き出しを指定する。
        st = diversity_tracker.next_starter()
        prompt += (
            f"この対話は speakers の{st}人目から話し始めてください。\n"
            f"conversation_flow の最初の項目は、必ず{st}人目の発言・行動から書き始めてください"
            f"（{st}人目の名前で始まる文にする）。\n\n")

    prompt += "JSON配列のみを出力してください（マークダウンのコードブロックは不要です）。"
    return prompt


async def generate_metadata_batch(
    client: AsyncOpenAI,
    model: str,
    batch_size: int,
    min_turns: int,
    max_turns: int,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    diversity_tracker: DiversityTracker | None = None,
    retry_max: int = 3,
    retry_delay: float = 1.0,
    request_interval: float = 0.0,
    extra_body: dict | None = None,
    num_speakers: int | None = None,
    scenario: ScenarioPack | None = None,
) -> list[dict]:
    scenario = scenario or (diversity_tracker.scenario if diversity_tracker else GENERAL)
    async with semaphore:
        if request_interval > 0:
            await asyncio.sleep(request_interval)
        for attempt in range(retry_max):
            if cost_tracker.exceeded():
                logger.warning("コスト上限に達したため生成を中止")
                return []
            try:
                cards: list[dict] = []   # このプロンプトで指定した2人目の人物カード（partner_persona）
                plans: list[dict] = []   # このプロンプトで指定した話題の組（topic_plan）
                personas: list[str] = [] # このプロンプトで引いた主話者のペルソナ（persona_files）
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": scenario.system_prompt},
                        {"role": "user", "content": build_user_prompt(
                            batch_size, min_turns, max_turns, diversity_tracker,
                            num_speakers, card_out=cards, plan_out=plans,
                            persona_out=personas,
                        )},
                    ],
                    temperature=1.0,
                    # scenario 2〜3文 + conversation_flow 3〜4項目で1件あたりの
                    # 出力が従来の約2倍になったため下限を引き上げ
                    max_completion_tokens=max(4096, 8192 * batch_size // 10),
                    extra_body=extra_body or {},
                )
                usage = response.usage
                cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)

                text = extract_json_text(response.choices[0].message.content or "")

                # strict=False: 文字列中の生の制御文字（改行等）を許容
                metadata_list = json.loads(text, strict=False)
                if isinstance(metadata_list, list):
                    if num_speakers:
                        # 話者数が合わないメタデータは下流（2声TTS・2chリタイム）で
                        # 合成不能になるため、ここで捨てる。
                        ok = [m for m in metadata_list
                              if isinstance(m, dict)
                              and len(m.get("speakers") or []) == num_speakers]
                        if len(ok) < len(metadata_list):
                            logger.warning(f"話者数≠{num_speakers}のメタデータ "
                                           f"{len(metadata_list) - len(ok)}件を破棄")
                        metadata_list = ok
                        if not metadata_list:
                            continue  # 全滅ならリトライ
                    n_bad, n_unknown = fill_speaker_genders(metadata_list)
                    if n_bad:
                        logger.warning(
                            f"gender が male/female でない話者 {n_bad}人を名前・役割から補完"
                            f"（うち{n_unknown}人は不明のまま）")
                    # 主話者の属性固定は LLM 任せにせず機械的に是正する
                    # （fill_speaker_genders の後＝gender が正規化済みであることが前提）
                    if diversity_tracker and diversity_tracker.main_speaker:
                        metadata_list, n_swap, n_age, n_name, n_drop = enforce_main_speaker(
                            metadata_list, diversity_tracker.main_speaker)
                        if n_swap or n_age or n_name or n_drop:
                            logger.warning(f"主話者の是正: 入替{n_swap}件 / 年代書換{n_age}件 / "
                                           f"名前差替{n_name}件 / 破棄{n_drop}件")
                        if not metadata_list:
                            continue  # 全滅ならリトライ
                    if cards:
                        metadata_list, n_age, n_drop = attach_partner_card(metadata_list, cards[0])
                        if n_age or n_drop:
                            logger.warning(f"人物カードの是正: 年代書換{n_age}件 / 性別違いで破棄{n_drop}件")
                        if not metadata_list:
                            continue  # 全滅ならリトライ（次の試行は別のカード）
                    if plans:
                        attach_topic_plan(metadata_list, plans[0])
                    if personas:
                        attach_persona(metadata_list, personas[0])
                    logger.info(f"バッチ生成成功: {len(metadata_list)}件")
                    return metadata_list
                else:
                    logger.warning("返却値がリストではありません。リトライします。")
            except (json.JSONDecodeError, Exception) as e:
                # 401(認証)・429(レートリミット)は長めに待つ
                err_str = str(e)
                if "401" in err_str or "429" in err_str:
                    delay = max(10, retry_delay * (3 ** attempt))
                    logger.warning(f"バッチ生成失敗 (attempt {attempt+1}/{retry_max}, 認証/レート制限エラー、{delay:.0f}秒待機): {e}")
                else:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(f"バッチ生成失敗 (attempt {attempt+1}/{retry_max}, {delay:.0f}秒待機): {e}")
                if attempt < retry_max - 1:
                    await asyncio.sleep(delay)
        return []


async def run_metadata_generation(config: dict, cost_tracker: CostTracker) -> int:
    # 複数エンドポイント（同一モデルを別GPUに複数インスタンス）に対応。
    # llm.base_urls を指定するとラウンドロビンで負荷分散する（単数 base_url なら1台）。
    clients = build_async_llm_clients(config)
    num_endpoints = len(clients)
    # シナリオはここで1度だけ読む。出力先（metadata.output_file）をシナリオごとに
    # 分けておくこと。同じファイルに混ぜると DiversityTracker の分布も混ざる。
    scenario = load_scenario(config["metadata"].get("scenario_file"))
    target = config["target_dialogues"]
    batch_size = config["metadata"]["batch_size"]
    output_file = Path(config["metadata"]["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    min_turns = config["dialogue"]["min_turns"]
    max_turns = config["dialogue"]["max_turns"]
    retry_max = config["retry_max_attempts"]
    retry_delay = config["retry_base_delay"]
    # max_concurrent_requests は「1エンドポイントあたりの並行度」と解釈し、
    # 全体の同時リクエスト数はエンドポイント数倍にする（各インスタンスを遊ばせない）。
    # llm.ctx_per_request があれば台ごとにサーバの /props から決める（endpoint_concurrency）。
    per_endpoint = endpoint_concurrency(config, clients)
    total_concurrency = sum(per_endpoint)
    semaphore = asyncio.Semaphore(total_concurrency)
    if num_endpoints > 1:
        logger.info(f"エンドポイント {num_endpoints}台でラウンドロビン分散 "
                    f"(各{min(per_endpoint)}〜{max(per_endpoint)}並行 / 合計{total_concurrency}並行): "
                    f"{[str(c[0].base_url) for c in clients]}")

    num_batches = (target + batch_size - 1) // batch_size
    logger.info(f"シナリオ: {scenario.name} "
                f"(ジャンル{len(scenario.genres)}種, 場面{len(scenario.settings)}種, "
                f"関係性{len(scenario.relationships)}種)")
    logger.info(f"メタデータ生成開始: 目標{target}件, {num_batches}バッチ (各{batch_size}件)")

    # 既存メタデータを読み込み（追記モード + 重複回避用の分布追跡）
    diversity_tracker = DiversityTracker(
        scenario,
        pair_mix=config["metadata"].get("pair_mix"),
        main_speaker=config["metadata"].get("main_speaker"),
        partner_roles=config["metadata"].get("partner_roles"),
        style_mix=config["metadata"].get("style_mix"),
        partner_persona=config["metadata"].get("partner_persona"),
        topic_plan=config["metadata"].get("topic_plan"),
    )
    if diversity_tracker.style_mix:
        logger.info(f"speech_style の比率を指定（config の metadata.style_mix）: {diversity_tracker.style_mix}")
    if diversity_tracker.partner_pool:
        pp = diversity_tracker.partner_pool
        logger.info("2人目は人物カードで指定（config の metadata.partner_persona。partner_roles は使わない）: "
                    f"{sum(len(v) for v in pp.cells.values())}枚、年代の重み {pp.age_mix}")
    elif diversity_tracker.partner_roles:
        logger.info("2人目の役割の系統を指定して散らす（config の metadata.partner_roles）: "
                    + "、".join(f"{k}={v['weight']}" for k, v in diversity_tracker.partner_roles.items()))
    if diversity_tracker.topic_deck:
        logger.info("話題の組と順番を1対話ずつ指定（config の metadata.topic_plan）: "
                    + diversity_tracker.topic_deck.describe())
    if diversity_tracker.persona:
        logger.info("主話者のペルソナを Stage1 に追加"
                    f"（config の metadata.main_speaker.persona_file）:\n{diversity_tracker.persona}")
    if diversity_tracker.persona_picker:
        logger.info("主話者のペルソナを対話ごとに1つ引く（config の metadata.main_speaker.persona_files。"
                    f"メタデータの {PERSONA_KEY} に記録）: "
                    + "、".join(f"{p}={w:g}" for p, w in diversity_tracker.persona_picker.weights.items()))
    existing_count = 0
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_count += 1
                    diversity_tracker.add(json.loads(line))
    if existing_count:
        logger.info(f"既存メタデータ: {existing_count}件。追記します。（ジャンル{len(diversity_tracker.genre_counts)}種, 場面{len(diversity_tracker.setting_counts)}種, 関係性{len(diversity_tracker.relationship_counts)}種を追跡中）")

    remaining_target = target - existing_count
    if remaining_target <= 0:
        logger.info(f"既に目標数({target}件)に達しています。スキップします。")
        return existing_count

    num_batches = (remaining_target + batch_size - 1) // batch_size
    logger.info(f"メタデータ生成開始: 残り{remaining_target}件, {num_batches}バッチ (各{batch_size}件)")

    generated = 0
    done = 0  # 完了処理したバッチ数（進捗ログ間引き用）
    # グループバリア（total_concurrency バッチを gather で待つ）を廃し、ワーカーが
    # 空き次第すぐ次のバッチを引く pull 型に変更。遅いバッチが他エンドポイント（GPU）を
    # 待たせない。diversity_tracker は完了バッチごとに（ロック下で）即時更新するので、
    # 従来のグループ単位更新より新鮮な分布が次バッチのプロンプトに反映される。
    with open(output_file, "a") as f:
        async def process(client, model, extra_body, _batch, idx):
            return await generate_metadata_batch(
                client, model, batch_size, min_turns, max_turns,
                cost_tracker, semaphore,
                diversity_tracker=diversity_tracker,
                retry_max=retry_max,
                retry_delay=retry_delay,
                request_interval=config.get("request_interval", 0.0),
                extra_body=extra_body,
                num_speakers=config["metadata"].get("num_speakers"),
                scenario=scenario,
            )

        async def on_result(batch_result, _batch, idx):
            nonlocal generated, done
            for meta in batch_result:
                if generated >= remaining_target:
                    break
                meta["id"] = str(uuid.uuid4())
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
                # 分布を即時更新（次バッチの重複回避に反映）
                diversity_tracker.add(meta)
            done += 1
            if done % total_concurrency == 0:
                cost_tracker.save_log()
                s = cost_tracker.summary()
                logger.info(f"進捗: {existing_count + generated}/{target}件 | 入力={s['total_input_tokens']:,}tok 出力={s['total_output_tokens']:,}tok | コスト: ${s['total_cost_usd']}")

        # 1 item = 1バッチリクエスト。target 到達で新規取得を止める（実行中は完走）。
        await run_endpoint_worker_pool(
            clients, range(num_batches), per_endpoint, process, on_result,
            should_stop=lambda: cost_tracker.exceeded() or generated >= remaining_target,
        )
        if cost_tracker.exceeded():
            logger.warning(f"コスト上限 (${cost_tracker.max_cost_usd}) に到達。停止しました。")
        cost_tracker.save_log()

    total_count = existing_count + generated
    logger.info(f"メタデータ生成完了: 新規{generated}件 (合計{total_count}件)")
    return total_count
