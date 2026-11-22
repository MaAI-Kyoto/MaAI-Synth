"""三人対話の発話順の指標（Stage2 のゲート・検査スクリプト・テストで共用）。

旧三人対話（2026-04〜07）は「A→B→C→A…」と順番に回る台本が多かった
（旧出力20対話: 三人目に回る遷移 68%）。人間の鼎談では、二人が数往復やりとりして
三人目が言いたいときに入るので、直前の話者に戻る遷移（A→B→A）のほうが多い。

基準値（reference_corpus_tri.py が鼎談コーパスから測る。相槌・笑いだけの発話を除き、
同じ話者の連続を1ターンに併合。40ターンの窓 91個）:
  A→B→A の割合         全体 0.57 / 窓の5パーセンタイル 0.39
  順に回る遷移の最長連続  窓の中央値 4 / 95パーセンタイル 7
  最少話者の占有率        窓の5パーセンタイル 0.15（平均の占有率 42 / 35 / 23%）

指標は同じ話者の連続を併合した列で測る（Stage2 は merge_consecutive_same_speaker で
併合済みだが、検査スクリプトは任意の turns を受けるので、ここでも併合する）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnOrderStats:
    n_turns: int            # 併合後のターン数
    aba_ratio: float        # A→B→A（直前の話者に戻る）遷移の割合。遷移が無ければ 0
    max_cycle_run: int      # 三人が順に回る遷移（s[i] != s[i-2]）の最長連続
    min_share: float        # 最も話さない人の占有率（speakers を渡せば0回の人も数える）
    shares: dict            # 話者ごとの占有率


@dataclass(frozen=True)
class TurnOrderLimits:
    """ゲートの閾値。既定は鼎談の窓の値より少し緩くしてある（config の tri.turn_order）。"""
    min_aba: float = 0.30
    max_cycle_run: int = 8
    min_share: float = 0.12

    @classmethod
    def from_config(cls, d: dict | None) -> "TurnOrderLimits":
        d = d or {}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(k for k in d if k not in known and k != "retry")
        if unknown:
            raise ValueError(f"tri.turn_order に未知のキー: {unknown}（使えるのは "
                             f"{sorted(known | {'retry'})}）")
        return cls(**{k: d[k] for k in known if k in d})


def merged_speakers(turns: list) -> list[str]:
    """turns（dict の speaker、または話者名の列）から、同じ話者の連続を併合した列を返す。"""
    seq: list[str] = []
    for t in turns:
        s = t.get("speaker") if isinstance(t, dict) else t
        if s is None:
            continue
        s = str(s)
        if not seq or seq[-1] != s:
            seq.append(s)
    return seq


def turn_order_stats(turns: list, speakers: list[str] | None = None) -> TurnOrderStats:
    seq = merged_speakers(turns)
    n = len(seq)
    aba = [seq[i] == seq[i - 2] for i in range(2, n)]
    run = best = 0
    for x in aba:
        run = 0 if x else run + 1
        best = max(best, run)
    names = list(speakers) if speakers else sorted(set(seq))
    shares = {s: (seq.count(s) / n if n else 0.0) for s in names}
    return TurnOrderStats(
        n_turns=n,
        aba_ratio=(sum(aba) / len(aba)) if aba else 0.0,
        max_cycle_run=best,
        min_share=min(shares.values()) if shares else 0.0,
        shares=shares,
    )


def violations(st: TurnOrderStats, lim: TurnOrderLimits) -> list[str]:
    """閾値を外れた項目の説明（空なら合格）。"""
    out = []
    if st.aba_ratio < lim.min_aba:
        out.append(f"A→B→A {st.aba_ratio:.2f}<{lim.min_aba:.2f}")
    if st.max_cycle_run > lim.max_cycle_run:
        out.append(f"順に回る連続 {st.max_cycle_run}>{lim.max_cycle_run}")
    if st.min_share < lim.min_share:
        out.append(f"最少話者 {st.min_share:.2f}<{lim.min_share:.2f}")
    return out


def penalty(st: TurnOrderStats, lim: TurnOrderLimits) -> float:
    """閾値からの不足量（0なら合格）。作り直しの候補どうしの優劣付けに使う。"""
    return (max(0.0, lim.min_aba - st.aba_ratio) / max(lim.min_aba, 1e-9)
            + max(0, st.max_cycle_run - lim.max_cycle_run) / max(lim.max_cycle_run, 1)
            + max(0.0, lim.min_share - st.min_share) / max(lim.min_share, 1e-9))


# ── 発話の宛先（addressee）────────────────────────────────────────────────
# Stage2 が各ターンに書く addressee（ほかの話者1人の名前 / "both"）。宛先は「誰に向けたか」で、
# 次の話者の指定ではない（宛先でない三人目が先に口を挟むこともある）。人間の鼎談
# （非公開の鼎談コーパス）は視線の層が空で宛先の基準値が取れないため、ゲートは「書けているか」
# だけで、宛先と次の話者の関係は数字を出して見るだけにしている。
BOTH = "both"
_BOTH_WORDS = {"both", "all", "everyone", "両方", "全員", "二人", "2人", "ふたり", "お二人",
               "みんな", "皆", "皆さん", "みなさん"}


def normalize_addressee(value, speaker: str, names: list[str]) -> str | None:
    """addressee を「ほかの話者の名前」か "both" にそろえる。読めなければ None。

    「タナカさん」「タナカくん」のような敬称付き、全角・大文字の both、「両方」「二人」等も受ける。
    自分自身・話者一覧に無い名前は None。話者が2人だけの対話では both も相手1人と同じ。
    """
    if value is None:
        return None
    if isinstance(value, list):            # ["タナカ", "ミサキ"] のように並べた形
        vals = {normalize_addressee(v, speaker, names) for v in value}
        vals.discard(None)
        if not vals:
            return None
        return next(iter(vals)) if len(vals) == 1 else BOTH
    s = str(value).strip().strip("「」\"'")
    if not s:
        return None
    low = s.lower().replace("ｂｏｔｈ", "both")
    if low in _BOTH_WORDS or s in _BOTH_WORDS:
        return BOTH
    for suf in ("さん", "くん", "君", "ちゃん", "様"):
        if s.endswith(suf) and s[:-len(suf)] in names:
            s = s[:-len(suf)]
    if s in names and s != speaker:
        return s
    return None


def addressee_stats(turns: list, names: list[str]) -> dict:
    """宛先の内訳と、宛先と次の話者の関係（同じ話者の連続は併合済みの turns を想定）。

    next_is_addressee: 宛先が1人のターンのうち、次のターンをその人が話した割合
    both_next:         宛先が both のターンの次の話者が、直前の話者（応答相手）か三人目か
    """
    single = both = missing = nxt_hit = nxt_n = 0
    both_prev = both_third = 0
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        a = t.get("addressee")
        if a is None:
            missing += 1
            continue
        if a == BOTH:
            both += 1
            if i + 1 < len(turns) and i >= 1:
                nxt = turns[i + 1].get("speaker")
                if nxt == turns[i - 1].get("speaker"):
                    both_prev += 1
                else:
                    both_third += 1
            continue
        single += 1
        if i + 1 < len(turns):
            nxt_n += 1
            nxt_hit += turns[i + 1].get("speaker") == a
    n = single + both + missing
    return {"n": n, "single": single, "both": both, "missing": missing,
            "both_ratio": round(both / max(single + both, 1), 3),
            "next_is_addressee": round(nxt_hit / nxt_n, 3) if nxt_n else None,
            "both_next_prev_speaker": both_prev, "both_next_third": both_third}
