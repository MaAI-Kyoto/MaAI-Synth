"""Stage 1（三人対話）: 対話メタデータの生成

二人用の generate_metadata.py を**そのまま使い**、二人を前提にした部分だけを
モジュール属性として差し替えてから gm.run_metadata_generation を呼ぶ
（apply_patches）。差し替えは三人用のプロセス（main_tri.py）の中でしか起きないので、
二人対話の実行には影響しない。コピーにしないのは、旧三人対話（2026-04〜07）が
コピーだったために、後から二人用に入った改良（主話者・ペルソナ・役割の散らし等）を
取りこぼしたため（RUNBOOK_main_v1.md の「コピーの取りこぼし」）。

差し替えるもの（いずれも二人用は「話者がちょうど2人」を前提にしている）:
  - enforce_main_speaker: 二人用は len(speakers) != 2 だと素通しして main_speaker_index を
    書かない。三人版は主話者を index 0 に置き、残り2人は順序を保つ。
  - DiversityTracker: 性別の組（男女/男性同士/女性同士）→ 三人の性別の構成（trio_mix）、
    2人目の役割 → 2人目と3人目の役割、話し始める人 1/2 の交互 → 1〜3人目を均等に、
    さらに「三人の関係の形」（tri.group_shapes。誰と誰が以前からの知り合いか）を配る。
  - build_user_prompt: 「1人目／2人目」の指示文を三人用に書き直したもの。
    ペルソナの段落（gm._persona_lines）はそのまま使う。

config（三人用で増えたキー）:
  metadata.num_speakers: 3
  metadata.trio_mix:   {男性3人: w, 男性2人と女性1人: w, 男性1人と女性2人: w, 女性3人: w}
  tri.group_shapes:    {名前: {weight: w, text: "1人目と2人目は以前からの知り合いで…"}}
                       （本文の「1人目／2人目／3人目」は speakers の並び順を指す）
"""

from __future__ import annotations

import logging
import random
from collections import Counter

import generate_metadata as gm
from generate_metadata import _age_decade, _replace_name_deep, normalize_gender

logger = logging.getLogger(__name__)

NUM_SPEAKERS = 3

# 三人の性別の構成。二人用の PAIR_MIX_TARGET と同じく、指定しないと LLM は
# テンプレートの並び（男・女・男）を写しがちなので、1件ごとに構成を指定して散らす。
TRIO_LABELS = {
    ("male", "male", "male"): "男性3人",
    ("female", "male", "male"): "男性2人と女性1人",
    ("female", "female", "male"): "男性1人と女性2人",
    ("female", "female", "female"): "女性3人",
}
TRIO_MIX_TARGET = {"男性3人": 0.15, "男性2人と女性1人": 0.35,
                   "男性1人と女性2人": 0.35, "女性3人": 0.15}
_LABEL_COUNTS = {v: Counter(k) for k, v in TRIO_LABELS.items()}

# 三人の関係の形。全員が初対面とは限らない（知り合い2人＋新顔1人のほうが日常的）。
# 誰と誰が知り合いかの非対称があると、話す組に自然な偏りが出る。
# 「1人目／2人目／3人目」は speakers の並び（主話者を固定するときは1人目が主話者）。
GROUP_SHAPES_DEFAULT = {
    "全員初対面": {"weight": 40, "text": "三人とも今日が初対面"},
    "1人目が新顔": {"weight": 20, "text": "2人目と3人目は以前からの知り合いで、1人目だけが今日初めて二人に会う"},
    "3人目が新顔": {"weight": 20, "text": "1人目と2人目は以前からの知り合いで、3人目が今日初めて加わる"},
    "2人目が新顔": {"weight": 10, "text": "1人目と3人目は以前からの知り合いで、2人目が今日初めて加わる"},
    "紹介役": {"weight": 10, "text": "2人目が、互いに初対面の1人目と3人目を引き合わせる（2人目は二人の共通の知り合い）"},
}


def trio_label(speakers: list[dict]) -> str | None:
    key = tuple(sorted(str(s.get("gender")) for s in speakers))
    return TRIO_LABELS.get(key)


def enforce_main_speaker(metadata_list: list[dict], spec: dict) -> tuple[list, int, int, int, int]:
    """三人版。speakers[0] を必ず spec の話者にし、main_speaker_index = 0 を書く。

    規則は二人用（gm.enforce_main_speaker）と同じ:
      role 指定あり → role が完全一致する話者がちょうど1人で、性別も合うときだけ採用
                      （年代だけ違えば書き換える）。それ以外は破棄。
      role 指定なし → 性別・年代が合う話者、なければ性別だけ合う話者の年代を書き換える。
                      その性別の話者が居なければ破棄。
      name 指定あり → speakers[0].name を固定し、本文中の旧名も置換。ほかの2人と
                      同名・部分一致になる対話は破棄。
    主話者を先頭に移すとき、残り2人の順序は保つ。
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
        if len(sp) != NUM_SPEAKERS:
            n_drop += 1
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
            meta["speakers"] = [sp[i]] + [s for j, s in enumerate(sp) if j != i]
            n_swap += 1

        if want_n:
            old = (meta["speakers"][0].get("name") or "").strip()
            others = [(s.get("name") or "").strip() for s in meta["speakers"][1:]]
            clash = False
            for other in others:
                if (other == want_n or (old and other and (old in other or other in old))
                        or (other and (want_n in other or other in want_n))):
                    clash = True
            # ほかの2人どうしの同名・部分一致も、Stage2 の話者判定を壊すので捨てる
            if others[0] and others[1] and (others[0] in others[1] or others[1] in others[0]):
                clash = True
            if clash:
                n_drop += 1
                continue
            if old and old != want_n:
                meta = _replace_name_deep(meta, old, want_n)
                n_name += 1
            meta["speakers"][0]["name"] = want_n

        meta["main_speaker_index"] = 0    # Stage4 が ch0 を固定するのに使う
        out.append(meta)
    return out, n_swap, n_age, n_name, n_drop


class DiversityTracker(gm.DiversityTracker):
    """二人用の DiversityTracker に、三人用の配分（性別の構成・関係の形・話し始め）を足したもの。

    ジャンル・関係性・名前の偏りの追跡（build_diversity_context）、役割の系統（partner_roles）、
    口調（style_mix）の配分は二人用をそのまま使う。
    """

    def __init__(self, scenario=None, pair_mix=None, main_speaker=None,
                 partner_roles=None, style_mix=None, trio_mix=None, group_shapes=None):
        super().__init__(scenario, pair_mix=None, main_speaker=main_speaker,
                         partner_roles=partner_roles, style_mix=style_mix)
        self.trio_mix = {k: float(v) for k, v in (trio_mix or TRIO_MIX_TARGET).items()
                         if v and float(v) > 0}
        unknown = sorted(set(self.trio_mix) - set(_LABEL_COUNTS))
        if unknown:
            raise ValueError(f"metadata.trio_mix に未知のラベル: {unknown}（使えるのは {list(_LABEL_COUNTS)}）")
        # 主話者の性別を含まない構成は enforce_main_speaker で必ず破棄されるので発注しない
        g = normalize_gender((self.main_speaker or {}).get("gender"))
        if g:
            self.trio_mix = {k: v for k, v in self.trio_mix.items() if _LABEL_COUNTS[k][g] > 0}
        if not self.trio_mix:
            raise ValueError("metadata.trio_mix に、主話者の性別を含む正の比率の構成がありません")
        self.trio_counts = Counter()
        self.trio_pending = Counter()
        shapes = group_shapes if group_shapes is not None else GROUP_SHAPES_DEFAULT
        self.group_shapes = {k: v for k, v in shapes.items()
                             if float((v or {}).get("weight", 0)) > 0 and (v or {}).get("text")}
        self.shape_issued = Counter()
        self.starter_counts = Counter()

    # ── 性別の構成 ────────────────────────────────────────────────────────
    def trio_quota(self, batch_size: int) -> Counter:
        """二人用の pair_quota と同じ配り方（発注済み・未着も数える）。"""
        quota = Counter()
        total = sum(self.trio_counts.values()) + sum(self.trio_pending.values())
        labels = list(self.trio_mix)
        for _ in range(batch_size):
            total += 1
            label = max(labels, key=lambda k: (self.trio_mix[k] * total - self.trio_counts[k]
                                               - self.trio_pending[k] - quota[k]))
            quota[label] += 1
        self.trio_pending.update(quota)
        return quota

    # ── 関係の形 ──────────────────────────────────────────────────────────
    def next_group_shape(self) -> tuple[str, str] | None:
        if not self.group_shapes:
            return None
        tot = sum(self.shape_issued.values()) + 1
        key = min(self.group_shapes, key=lambda k: (
            (self.shape_issued[k] + 1) / float(self.group_shapes[k]["weight"])
            - random.random() * 1e-6 * tot))
        self.shape_issued[key] += 1
        return key, str(self.group_shapes[key]["text"])

    # ── 話し始める人 ──────────────────────────────────────────────────────
    def next_starter(self) -> int:
        """1〜3人目を均等に。発注の少ない順（同数なら無作為）。

        二人用は 1/2 の交互。三人で交互（1,2,3,1,2,3…）にすると、バッチの並びで
        関係の形と話し始めの組み合わせが固定されるので、同数の中から無作為に選ぶ。
        """
        least = min(self.starter_counts[i] for i in (1, 2, 3))
        st = random.choice([i for i in (1, 2, 3) if self.starter_counts[i] == least])
        self.starter_counts[st] += 1
        self.starter_issued += 1
        return st

    def add(self, meta: dict):
        super().add(meta)   # 二人用の性別の組は len(speakers)==2 のときだけ数えるので、ここでは何もしない
        speakers = [s for s in (meta.get("speakers") or []) if isinstance(s, dict)]
        if len(speakers) == NUM_SPEAKERS:
            label = trio_label(speakers)
            if label:
                self.trio_counts[label] += 1
                if self.trio_pending[label] > 0:
                    self.trio_pending[label] -= 1


_STYLE_HINT = {
    "敬語": "3人とも敬語で話す",
    "混合": "年齢や立場の差、以前からの知り合いかどうかなどから、3人のうち一部の人が"
            "（または一部の相手に対してだけ）くだけた話し方をする",
    "タメ口": "3人ともくだけた話し方をする",
}


def _gender_jp(g: str | None) -> str:
    return {"male": "男性", "female": "女性"}.get(str(g), str(g))


def _trio_line(tracker: DiversityTracker, batch_size: int) -> str:
    quota = tracker.trio_quota(batch_size)
    if batch_size != 1:
        spec = "、".join(f"{n}件を{label}" for label, n in quota.items())
        return f"{batch_size}件のうち{spec}の三人にしてください。\n\n"
    label = next(iter(quota))
    line = f"この対話の3人は{label}にしてください。"
    g = normalize_gender((tracker.main_speaker or {}).get("gender"))
    if g:
        rest = _LABEL_COUNTS[label].copy()
        rest[g] -= 1
        others = [_gender_jp(x) for x in ("male", "female") for _ in range(rest[x])]
        line += f"（1人目は{_gender_jp(g)}なので、2人目と3人目は{'と'.join(others)}）"
    return line + "\n\n"


def _style_and_roles_line(tracker: DiversityTracker | None) -> str:
    """speech_style と、2人目・3人目の役割の系統（config が無ければ空行だけ）。"""
    style = tracker.next_style() if tracker else None
    line = ""
    if style:
        hint = _STYLE_HINT.get(style)
        line = (f"この対話の speech_style は「{style}」にしてください"
                + (f"（{hint}）" if hint else "") + "。\n")
    picks = [tracker.next_partner_role() for _ in range(2)] if tracker else [None, None]
    picks = [p for p in picks if p]
    if not picks:
        return line + "\n"
    for nth, (cat, ex) in zip(("2人目", "3人目"), picks):
        line += (f"{nth}の role は「{cat}」の人にしてください"
                 + (f"（例: {ex}。例にない職業・立場でもよい）" if ex else "") + "。")
    return line + ("role には具体的な職業・立場を1つだけ書き、年代はその立場に合うものにしてください"
                   "（2人目と3人目が同じ系統になったときも、職業そのものは別にしてください）。\n\n")


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
    """三人版の Stage1 ユーザプロンプト（二人用 gm.build_user_prompt を書き直したもの）。

    card_out / plan_out は二人用の呼び出し（gm.generate_metadata_batch）と引数を揃えるためだけに
    受ける（三人用は人物カード・話題の指定を使わないので何も積まない）。★2026-10-30 まで受けて
    いなかったため、二人用が card_out を渡すようになった 10-27 以降は TypeError で全バッチが
    失敗していた。persona_out には persona_files で引いたペルソナのパスを積む。

    指定は二人用と同じくプロンプト末尾（最も従いやすい位置）に置く。
    """
    if num_speakers not in (None, NUM_SPEAKERS):
        raise ValueError(f"三人用の Stage1 に num_speakers={num_speakers} が渡されました")
    prompt = f"{batch_size}個の対話メタデータを生成してください。\n"
    prompt += f"target_turnsは{min_turns}〜{max_turns}の範囲でばらつかせてください。\n"
    prompt += f"各対話の話者(speakers)は必ず{NUM_SPEAKERS}人にしてください。\n\n"

    if diversity_tracker:
        prompt += diversity_tracker.build_diversity_context(batch_size)
        prompt += "\n"
        prompt += _trio_line(diversity_tracker, batch_size)

    ms = diversity_tracker.main_speaker if diversity_tracker else None
    if ms:
        persona = diversity_tracker.next_persona(persona_out)
        if ms.get("name"):
            prompt += (
                f"speakers の1人目の name は必ず \"{ms['name']}\" にしてください"
                f"（毎回同じ固定の人物名です）。"
                f"scenario と conversation_flow の本文でも、その人物は必ず"
                f"「{ms['name']}」と書いてください。\n"
                f"2人目と3人目には「{ms['name']}」を含む名前を絶対に使わないでください。\n")
        head = (f"speakers の1人目は必ず "
                + (f"role を \"{ms['role']}\"、" if ms.get("role") else "")
                + f"gender を \"{ms.get('gender')}\"、"
                f"age_range を \"{ms.get('age_range')}\" にしてください。\n")
        if persona:
            vary = ms.get("persona_vary") or ("ほかの2人・話題" if ms.get("role")
                                              else "役割・ほかの2人・話題")
            body = gm._persona_lines(persona, vary)
        else:
            body = ("ただし1人目の"
                    + ("" if ms.get("role") else "役割・")
                    + "性格・話し方・ほかの2人との関係は、これらの制約とは無関係に毎回まったく"
                    "違うものにしてください（同じような人物像を繰り返さない）。\n")
        prompt += head + body
        prompt += "2人目と3人目の性別は上の構成の指定に従い、年代・役割・性格は自由に多様にしてください。\n"
    if diversity_tracker:
        prompt += _style_and_roles_line(diversity_tracker)
        shape = diversity_tracker.next_group_shape()
        if shape:
            prompt += (f"三人の関係は「{shape[1]}」にしてください。"
                       "relationship にはその関係を、名前を使って具体的に書いてください。\n")
        # ★話し始めは conversation_flow の本文が決める（二人用 next_starter の注記と同じ）。
        st = diversity_tracker.next_starter()
        prompt += (
            f"この対話は speakers の{st}人目から話し始めてください。\n"
            f"conversation_flow の最初の項目は、必ず{st}人目の発言・行動から書き始めてください"
            f"（{st}人目の名前で始まる文にする）。\n"
            "conversation_flow の各項目には、その段で中心になって話す人を名前で書き、"
            "項目ごとに中心の組が入れ替わるようにしてください"
            "（同じ二人がずっと話し続ける流れ・三人が順番に一言ずつ話す流れにしない）。\n\n")

    prompt += "JSON配列のみを出力してください（マークダウンのコードブロックは不要です）。"
    return prompt


# 元の関数・クラスは apply_patches で控える（二重に差し替えても元を指し続けるように）。
_ORIG: dict = {}
_PATCH_NAMES = ("enforce_main_speaker", "DiversityTracker", "build_user_prompt")


def apply_patches() -> None:
    """二人用モジュール generate_metadata の該当関数を三人用に差し替える（何度呼んでもよい）。

    DiversityTracker は config の値を束ねる必要があるので run_metadata_generation で差し替える。
    """
    if _ORIG:
        return
    for name in _PATCH_NAMES:
        if not hasattr(gm, name):
            raise RuntimeError(f"generate_metadata.{name} が見つかりません（二人用が変わった？）")
        _ORIG[name] = getattr(gm, name)
    gm.enforce_main_speaker = enforce_main_speaker
    gm.build_user_prompt = build_user_prompt


def remove_patches() -> None:
    """差し替えを元に戻す（テスト用）。"""
    for name, f in _ORIG.items():
        setattr(gm, name, f)
    _ORIG.clear()


async def run_metadata_generation(config: dict, cost_tracker) -> int:
    """三人用の Stage1。config を検査し、差し替えてから二人用の本体を回す。"""
    md = config.get("metadata") or {}
    if md.get("num_speakers") != NUM_SPEAKERS:
        raise ValueError(f"三人対話の config は metadata.num_speakers: {NUM_SPEAKERS} にしてください"
                         f"（受領: {md.get('num_speakers')}）")
    apply_patches()
    tri = config.get("tri") or {}
    trio_mix = md.get("trio_mix")
    group_shapes = tri.get("group_shapes")

    # DiversityTracker は run_metadata_generation の中で、二人用の引数
    # （pair_mix 等）で作られる。三人用の配分を渡すため、その場で束ねたクラスに差し替える。
    class _Tracker(DiversityTracker):
        # ★2026-10-30 partner_persona / topic_plan も受ける（二人用が 10-27/28 から渡すようになり、
        #   受けないと起動時に TypeError で止まっていた）。三人用はどちらにも未対応なので、書いてあれば止める。
        def __init__(self, scenario=None, pair_mix=None, main_speaker=None,
                     partner_roles=None, style_mix=None, partner_persona=None, topic_plan=None):
            if partner_persona or topic_plan:
                raise ValueError("三人対話の Stage1 は metadata.partner_persona / topic_plan に未対応です")
            super().__init__(scenario, main_speaker=main_speaker, partner_roles=partner_roles,
                             style_mix=style_mix, trio_mix=trio_mix, group_shapes=group_shapes)

    gm.DiversityTracker = _Tracker
    t = _Tracker(None, main_speaker=md.get("main_speaker"))
    logger.info(f"三人対話の Stage1: 性別の構成 {t.trio_mix}（metadata.trio_mix）、"
                f"関係の形 {[(k, v['weight']) for k, v in t.group_shapes.items()]}（tri.group_shapes）")
    return await gm.run_metadata_generation(config, cost_tracker)
