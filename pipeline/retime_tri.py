"""三人対話の配置（モデルなし）: 人間の鼎談の間合いから標本化する。

三人対話の本番の配置は retime_tri_vap.TriVapRetimer（二者 VAP を話者ごとに組み替えて使う）。
こちらは VAP を使わない配置で、generate_tts_tri.py の --dry-run と、config の tri.retimer: gap
（GPU・maai が無いとき）で使う。人間の鼎談（reference/teidan_tri.json。reference_corpus_tri.py が
作る）の話者交替の間（FTO）と同じ話者の切れ目の分布から、実際の wav 長に合わせて配置する。

generate_tts_tri.py が generate_tts.MaaiRetimer / MockRetimer をこのクラスに差し替えて使う
（generate_tts.run() が RetimeConfig だけを渡して作るので、三人用の設定は
configure() で先にモジュールへ入れておく）。

配置の規則:
  - 話者交替: 「それまでに話した全員の発話の終わり（床が空いた時刻）」＋ FTO。
    FTO は参照の標本（max_gap 以下）から引く。max_overlap より深い重なりの標本は捨てずに
    fold_range の一様乱数（既定 0〜0.2秒の間）に畳む。人間の深い重なりは「前の人が言い終わる
    前に次の人が話し始めた」素早い交替なので、捨てると交替が遅い側に偏る（捨てた版の試算:
    中央値 +0.51秒。畳むと中央値 +0.15秒・0.5秒超 22%・1秒超 11%。二人用の参照 maai-f2f は
    0.5秒超 17%・1秒超 8.8%）。
    重なりは直前の発話長の overlap_frac 倍までに抑える。
  - 同じ話者が続く（continued）: その話者の直前の発話の終わり＋同じ話者の切れ目の標本。
  - 同じチャネルは重ねない（その話者の直前の発話の終わり＋min_same_channel_gap 以降）。
  - 乱数は対話IDをシードにする（--recompose で同じ配置が再現する）。
二人用の既定（retime.hazard、かぶり0%）に合わせて、重なりの既定は小さく（0.2秒）してある。
人間の鼎談は話者交替の53%が重なる（中央値 -0.04秒、10パーセンタイル -1.42秒）が、
台本と無関係に長く重ねると聞き取れない発話になるため。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE = "reference/teidan_tri.json"


@dataclass(frozen=True)
class TriTimingParams:
    reference_file: str = DEFAULT_REFERENCE
    max_overlap: float = 0.2         # 話者交替の重なりの上限（秒）
    overlap_frac: float = 0.3        # 重なりは直前の発話長のこの割合まで
    max_gap: float = 3.0             # 話者交替の間の上限（秒）
    within_min: float = 0.15         # 同じ話者の切れ目の下限・上限（秒）
    within_max: float = 1.2
    min_same_channel_gap: float = 0.1
    fold_range: tuple = (0.0, 0.2)   # max_overlap より深い重なりを畳む先（秒）
    seed_salt: str = "tri_gap_v1"

    @classmethod
    def from_config(cls, d: dict | None) -> "TriTimingParams":
        d = d or {}
        unknown = sorted(k for k in d if k not in cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"tri.timing に未知のキー: {unknown}"
                             f"（使えるのは {sorted(cls.__dataclass_fields__)}）")
        if "fold_range" in d:
            lo, hi = (float(x) for x in d["fold_range"])
            if not lo <= hi:
                raise ValueError(f"tri.timing.fold_range は [下限, 上限] で書いてください: {d['fold_range']}")
            d = {**d, "fold_range": (lo, hi)}
        return cls(**d)


_PARAMS = TriTimingParams()


def configure(d: dict | None) -> TriTimingParams:
    """config の tri.timing を入れる（TriGapRetimer を作る前に呼ぶ）。"""
    global _PARAMS
    _PARAMS = TriTimingParams.from_config(d)
    return _PARAMS


def _load_pools(p: TriTimingParams) -> tuple[np.ndarray, np.ndarray]:
    path = Path(p.reference_file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"三人用の参照値がありません: {path}"
                                "（python reference_corpus_tri.py で作る）")
    ref = json.loads(path.read_text(encoding="utf-8"))
    fto = np.asarray(ref["fto"], dtype=float)
    fto = fto[fto <= p.max_gap]
    within = np.asarray(ref["within_pause"], dtype=float)
    within = within[(within >= p.within_min) & (within <= p.within_max)]
    if len(fto) < 50 or len(within) < 50:
        raise ValueError(f"参照値の標本が少なすぎます（範囲内 FTO {len(fto)} / 切れ目 {len(within)}）。"
                         "tri.timing の範囲を見直してください")
    return fto, within


class TriGapRetimer:
    """generate_tts の retimer として使う（plan() / log_summary() だけを持つ）。

    コンストラクタは generate_tts.run() の都合で RetimeConfig を受け取るが、使わない。
    """

    def __init__(self, cfg=None, params: TriTimingParams | None = None):
        self.cfg = cfg
        self.params = params or _PARAMS
        self.fto, self.within = _load_pools(self.params)
        self.method_counts: dict[str, int] = {}
        self.offsets: list[float] = []
        logger.info(f"三人用の配置（VAP なし・人間の鼎談の間合い）: {asdict(self.params)}、"
                    f"標本 FTO {len(self.fto)} / 同じ話者の切れ目 {len(self.within)}")

    def _rng(self, record: dict) -> np.random.Generator:
        h = hashlib.sha256(f"{self.params.seed_salt}:{record.get('id')}".encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def plan(self, record, turns, wav_by_turn, sr, channel_of,
             failed_turns=None) -> list[dict] | None:
        if len(channel_of) < 2:
            return None
        p = self.params
        rng = self._rng(record)
        plans = []
        floor_end = 0.0                       # それまでの全発話の終わり（床が空いた時刻）
        last_end: dict[int, float] = {}       # チャネルごとの直前の発話の終わり
        prev_dur = 0.0
        prev_spk = None
        for i, t in enumerate(turns):
            ch = channel_of[t["speaker"]]
            dur = len(wav_by_turn[i]) / sr
            clamped = False
            if i == 0:
                start, method, sampled, anchor = 0.0, "start", 0.0, 0.0
            elif t["speaker"] == prev_spk:
                sampled = float(rng.choice(self.within))
                anchor = last_end.get(ch, floor_end)
                start, method = anchor + sampled, "within"
            else:
                sampled = float(rng.choice(self.fto))
                anchor = floor_end
                off = sampled
                if off < -p.max_overlap:
                    off = float(rng.uniform(*p.fold_range))
                    self.method_counts["folded"] = self.method_counts.get("folded", 0) + 1
                if off < 0 and -off > p.overlap_frac * prev_dur:
                    off, clamped = -p.overlap_frac * prev_dur, True
                start, method = anchor + off, "fto"
            floor_ch = last_end.get(ch)
            if floor_ch is not None and start < floor_ch + p.min_same_channel_gap:
                start, clamped = floor_ch + p.min_same_channel_gap, True
            start = max(0.0, start)
            end = start + dur
            plans.append({
                "index": i,
                "start_sec": round(start, 3),
                "retime": {"method": f"tri_{method}", "anchor_sec": round(anchor, 3),
                           "sampled_offset": round(sampled, 3),
                           "decided_offset": round(start - anchor, 3),
                           "p_peak": None, "clamped": clamped},
            })
            self.method_counts[method] = self.method_counts.get(method, 0) + 1
            if method == "fto":
                self.offsets.append(start - anchor)
            last_end[ch] = max(last_end.get(ch, 0.0), end)
            floor_end = max(floor_end, end)
            prev_dur, prev_spk = dur, t["speaker"]
        return plans

    def log_summary(self):
        if self.offsets:
            o = np.asarray(self.offsets)
            q = np.percentile(o, [10, 50, 90])
            logger.info(f"三人用の配置の集計: {self.method_counts}、話者交替の間 "
                        f"p10 {q[0]:+.2f} / 中央値 {q[1]:+.2f} / p90 {q[2]:+.2f} 秒、"
                        f"重なり {np.mean(o < 0) * 100:.0f}%")
        else:
            logger.info(f"三人用の配置の集計: {self.method_counts}")
