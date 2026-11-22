"""MaAI によるターンテイキング・相槌タイミングのリタイミング（Stage 4 用）

Stage 3 の LLM が推定した start_offset の代わりに、MaAI
(https://github.com/MaAI-Kyoto/MaAI) のターンテイキング予測（VAP）と
相槌タイミング予測（bc）モデルを、合成済みの実音声に適用して
発話の配置タイミングを決め直す。

方針:
  - LLM の生成内容（相槌の語・発話分割位置・ターン順序）は変えない。
  - compose_dialogue() の「prev_end + LLM offset」連鎖を、MaAI 予測に
    基づく逐次配置（MaaiRetimer.plan）で置き換える。
  - 判断ごとに fresh forward: 確定済みタイムラインの直近 context 秒 +
    無音プローブをモデルに流し、閾値クロス/ピークで開始時刻を決める。
  - 決められない場合は LLM オフセットにフォールバック（method に記録）。

MaAI 側の前提（pip maai / スモークテストで確認）:
  - 入力: 16kHz mono float32、1チャンク = 16000/frame_rate サンプル
  - Maai.process(x1, x2) を直接呼ぶと結果が result_dict_queue に1件入る
    （スレッド/リアルタイムペーシング不要）
  - vap:  p_now[2], p_future[2], vad[2]（index = MaAI ch1/ch2）
  - bc:   p_bc, p_bc_detect / bc_2type: p_bc_react, p_bc_emo
  - bc のチャネル規約: ch1=聞かれている側（話し手）, ch2=相槌を打つ側

依存（lazy import）: maai, torch は必要時のみ。リサンプリングは soxr 優先、
scipy.signal.resample_poly フォールバック。maai 未導入環境でも本モジュールの
import 自体は成功する（MockRetimer は maai 不要）。
"""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAAI_SR = 16000  # MaAI の入力サンプルレート

# backend="vapbatch" の重みの既定パス（vapmodel/export_state_dicts.py の出力先）
_VAP_MODELS = str(Path(__file__).resolve().parent / "models" / "vap")
DEFAULT_VAP_CKPT = f"{_VAP_MODELS}/vap_jp_12.5hz_20s.pt"
DEFAULT_BC_CKPT = f"{_VAP_MODELS}/vap_bc_jp_12.5hz_20s.pt"


# --------------------------------------------------------------------------
# スコアラの実測（backend の比較用）
# --------------------------------------------------------------------------

@dataclass
class ScorerStats:
    """1スコアラぶんの呼び出し実測。

    Stage4 の推論コストは今まで一度も切り出して測られていない
    （TIMING_1dialogue.md の値は fish 合成と混ざった外部計測）。backend を
    足す前後で「MaAI 推論が Stage4 の何割か」を言えるようにするための計測。
    """
    calls: int = 0
    frames: int = 0          # モデルに投入したフレーム総数
    seconds: float = 0.0     # score() の実時間合計
    load_seconds: float = 0.0

    def add(self, frames: int, seconds: float) -> None:
        self.calls += 1
        self.frames += int(frames)
        self.seconds += float(seconds)

    def summary(self) -> str:
        if not self.calls:
            return f"呼び出し0件（ロード {self.load_seconds:.1f}s）"
        rate = self.frames / self.seconds if self.seconds > 0 else float("nan")
        return (f"{self.calls}回 / {self.frames}フレーム / {self.seconds:.1f}s "
                f"({rate:.0f} frame/s) / ロード {self.load_seconds:.1f}s")


def _weight_fingerprint(path: str) -> str | None:
    """重みファイルの sha256 先頭8桁。manifest に残して出所を追えるようにする。

    export_state_dicts.py が隣に置く .json から読む（重み本体を毎回
    ハッシュし直さないため）。無ければ None を返すだけで落とさない。
    """
    if not path:
        return None
    import json
    from pathlib import Path
    side = Path(path).with_suffix(".json")
    try:
        return json.loads(side.read_text())["sha256"][:8]
    except Exception:
        return None


def _reject_unknown_keys(block: str, d: dict, cls, subs: dict) -> None:
    """config の未知キーを見つけたら落とす。

    `from_tts_cfg` は dataclass のフィールド名で白リストするので、綴り間違いや
    「あるはずだと思って書いた設定」は**黙って捨てられる**。とくに
    `tts.refine.backend` は書きたくなるが実際には `tts.retime.backend` が
    refine まで含めて効くので（refine は RetimeConfig を1つ作って
    plan() と InsertionScanner の両方に使う）、無視されたまま
    「切り替えたつもりの maai で焼けた」状態になりやすい。
    設定漏れは黙って通さない。
    """
    d = d or {}
    known = set(cls.__dataclass_fields__)
    bad = {}
    unknown = sorted(k for k in d if k not in known)
    if unknown:
        bad[block] = unknown
    for key, sub_cls in subs.items():
        sd = d.get(key) or {}
        if not isinstance(sd, dict):
            continue
        u = sorted(k for k in sd if k not in sub_cls.__dataclass_fields__)
        if u:
            bad[f"{block}.{key}"] = u
    if not bad:
        return
    lines = [f"  {where}: {', '.join(keys)}" for where, keys in bad.items()]
    hint = ""
    if any("backend" in v or "vapbatch" in v for v in bad.values()):
        hint = ("\n  ★推論バックエンドは `tts.retime.backend` だけで指定します"
                "（refine もそれを使うので `tts.refine.backend` はありません）。")
    raise ValueError(
        f"config に知らないキーがあります（綴り間違い？ 黙って無視すると"
        f"設定したつもりの値で動かないので落とします）:\n"
        + "\n".join(lines) + hint)


def _core_len(text: str) -> int:
    """句読点・記号を除いた実質文字数（短フィラー・相槌の判定用）。"""
    return len((text or "").strip().rstrip("。！？.!?、 ").strip())


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

@dataclass
class VapParams:
    mode: str = "vap"              # "vap" / "vap_mc"（雑音頑健）
    context_len_sec: int = 5       # jp/10Hz: 3/5/10/20
    signal: str = "p_now"          # "p_now" / "p_future"
    threshold: float = 0.60
    min_consecutive_frames: int = 2
    max_overlap: float = 0.8       # 許容する最大かぶせ（秒）
    max_gap: float = 2.0
    min_gap: float = 0.10
    hold_enabled: bool = False     # 同一話者連続(hold)で p_now 降下待ちを使うか
    hold_threshold: float = 0.70   # 自chの p_now がこれを下回るまで間を置く


@dataclass
class BcParams:
    mode: str = "bc"               # "bc"(p_bc, ctx 20s固定) / "bc_2type"(react/emo)
    context_len_sec: int = 20
    signal: str = "p_bc"           # bc_2type: "sum" / "react" / "emo" / "max"
    threshold: float = 0.35
    strategy: str = "peak"         # "peak" / "first_crossing"
    search_before: float = 2.5     # アンカー（相手セグメント末）より前の探索幅
    search_after: float = 1.0      # アンカー後の探索幅（無音プローブ側）
    max_overlap: float = 1.0       # 相槌が相手発話に食い込める最大時間（秒）。
                                   # 相槌自体がこれより長い場合、開始をアンカー末の
                                   # max_overlap 秒前以降に制限（長文相槌の全面二重発話を防ぐ）
    min_into_turn: float = 0.3     # 相手発話の頭から最低この秒数は経過してから
    insert_window_sec: float = 1.0 # refine挿入相槌: rel_pos ヒント中心の探索窓 ±秒


@dataclass
class LatencyParams:
    """応答潜時モデル（R2）: VAP のクロス時刻を「交替が予測可能になった時刻」と
    解釈し、そこに人間の応答潜時 L を足して開始時刻を決める。

    p_now は「今後およそ600ms 誰が話しているか」の予測なので、そのクロス時刻を
    そのまま開始時刻にすると系統的に早くなる（実測: 最頻値 -0.02秒・sd 0.13秒の
    離散スパイクで、人間の裾が全く出ない）。既定値は参照コーパス
    maai-f2f-jp-2025 に対する fit_gap_latency.py の推定結果（条件 D: 深いかぶりを
    0.5秒で浅いかぶりに畳み込み、max_gap 3.5秒で切断。Wasserstein-1 = 0.012秒）:

        python3 fit_gap_latency.py <manifest> \\
            --max-overlap 0.5 --fold-overlap --max-gap 3.5

    max_overlap / max_gap を変えたら必ず再フィットすること（切断範囲が変われば
    最適な L も変わる）。

    3成分は refine の遷移クラスに対応する:
        fast    = 投射的な即応 / overlap = 食い気味 / delayed = 考える・言いにくい
    """
    enabled: bool = False
    subframe: bool = True          # 閾値クロスを副フレーム補間する（R1。単体でも有効）
    seed: int = 0                  # 対話IDと合わせて決定的な乱数を作る
    w_fast: float = 0.4088
    w_overlap: float = 0.2970
    w_delayed: float = 0.2942
    m: float = -0.1144             # 3成分共通の基点（秒）
    sd_fast: float = 0.3037
    mu_overlap: float = -1.0442
    sd_overlap: float = 1.1657
    mu_delayed: float = -0.2471
    sd_delayed: float = 1.0074
    max_overlap: float = 0.5       # 結果のかぶり上限（棄却サンプリング）
    max_gap: float = 3.5           # 結果のポーズ上限（同上）
    apply_to_fallback: bool = True  # R4: 閾値クロス無しのときも LLM 値でなく L を引く


@dataclass
class HazardParams:
    """R2 の別案（ハザード方式）: 閾値クロス後、毎フレームくじを引いて開始点を決める。

    LatencyParams（3成分混合の潜時 L を足す方式）の代わりに使う。有効にすると
    `_decide_vap` の**クロスした場合の枝だけ**が置き換わる。クロスしなかった場合
    （R4 の fallback）は LatencyParams 側の設定がそのまま効くので、両者を比べるとき
    latency.enabled は true のままにしておくと差分がクロス枝だけになる。

    手順:
      1. p_now が threshold を min_consecutive_frames 連続で超えた地点で待機開始
      2. そのフレームから毎フレーム、確率 clip(scale * p_now, min_prob, max_prob)
         でくじを引く。最初に当たったフレームの時刻を発火時刻とする
      3. 発火時刻に offset ~ N(offset_mean, offset_sd)（[offset_min, offset_max] で
         棄却サンプリング）を足したものを開始時刻にする

    注意（実測に基づく）: クロス時点の p_now は本番実測で中央値 0.864 あるため、
    scale=1.0 だと 84% が最初のフレームで当たる（平均待ち 0.017秒）。待ち時間に
    幅を持たせたい場合は scale を下げる（例 0.12 で毎フレーム約10%）。
    """
    enabled: bool = False
    scale: float = 1.0             # くじの確率 = clip(scale * p_now, min_prob, max_prob)
    min_prob: float = 0.0
    max_prob: float = 1.0
    offset_mean: float = 0.3       # 発火後に足すオフセット（秒）
    offset_sd: float = 0.1
    offset_min: float = 0.0
    offset_max: float = 0.6


@dataclass
class RepairParams:
    """明らかな発話衝突（実質発話同士の大きな被り）の後処理修復。

    実質文字数 > min_speech_len の発話（speech と長い相槌の両方）をブロッカーとし、
    それに大きく被る speech を後ろへずらす。長い相槌自体はずらさない
    （MaAI が決めた食い込みタイミングを保つ）。短いフィラー・相槌のかぶせは残す。
    """
    enabled: bool = True
    max_speech_overlap: float = 0.3  # 実質発話同士の許容かぶせ上限（秒）。超過分を後ろへずらす
    min_speech_len: int = 5          # 実質文字数がこれ以下の発話は対象外（フィラー等はかぶせOK）


@dataclass
class OverlapParams:
    """重なりの場面（rp_v1。rp_plan.py）の置き方。台本の "overlap" を持つターンだけに効く（無ければ何も変わらない）。

    interrupt: 割り込まれた側（言いかけで切ってある）が、割り込んだ側の話し始めからこの秒数だけ話し続けて止まる
               （相手の言いかけの末尾 − この秒数で話し始める）。言いかけの長さの max_frac までに抑える。
    coop:      相手の話の終わりぎわに重ねる秒数（相手は言い終える）。
    react:     相手の文の終わりぎわに重ねる秒数（短い反応）。相手はそのまま話し続ける（through_gap）。
    collision: 2人の言いかけの話し始めの差。1人目は、それまでの音の最後から lull_sec あいてから話し始める。
    """
    # ★2026-11-11 人の対面会話（maai-f2f-jp-2025）で、割り込まれた側が相手の話し始めから止まるまで 中央値0.38秒
    #   （p10 0.11・p90 1.31）。一様 0.25〜0.65 では幅が狭い → 対数一様 0.12〜1.2（中央値 約0.38）。
    interrupt_sec: tuple = (0.12, 1.2)
    interrupt_max_frac: float = 0.6
    log_uniform: bool = True        # interrupt_sec / after_collision_sec を対数一様で引く
    coop_sec: tuple = (0.15, 0.6)
    coop_max_frac: float = 0.5
    react_sec: tuple = (0.0, 0.4)
    react_max_frac: float = 0.5
    through_gap: tuple = (0.10, 0.35)
    through_max_overlap: float = 0.6   # 話し続ける側が反応と重なる長さの上限（反応の終わりの何秒前から話してよいか）
    collision_offset: tuple = (0.0, 0.20)
    lull_sec: tuple = (0.5, 1.1)
    # 2人とも言いかけで止まったあと、次の発話（譲るひと言など）までの間。人は中央値0.31秒（p10 0.04・p90 0.86）。
    # ★試走8 は VAP 任せで 中央値0.00秒（間髪いれずに「どうぞ」）だった。
    after_collision_sec: tuple = (0.1, 0.9)


RP_OVERLAP_KINDS = ("interrupt", "collision", "coop", "react")


@dataclass
class VapBatchParams:
    """backend="vapbatch"（学習コードの一括推論）の設定。

    重みは vapmodel/export_state_dicts.py で ckpt から作る。MaAI 配布の
    state_dict とビット一致であることは同スクリプトが検証する。
    """
    # 既定は export_state_dicts.py の既定出力先。config で上書きできる。
    vap_ckpt: str = DEFAULT_VAP_CKPT
    bc_ckpt: str = DEFAULT_BC_CKPT
    batch_size: int = 8            # 長尺スキャンで一度に回す AR チャンク数
    scan_step_sec: float = 20.0    # 長尺スキャンの折りたたみステップ
    fold_context_mult: int = 4     # 折りたたみの前置き = context_len × これ
                                   # （attention の層数ぶん要る。retime_vapbatch 参照）
    mimi_align: str = "maai"       # "maai"(先頭ゼロ) / "train"(末尾 flush)
    dtype: str = "float32"         # fp16/bf16 は使わない（TF32 でさえ判定が動く）


@dataclass
class MaaiBatchParams:
    """backend="maaibatch"（MaAI 公式リポジトリの MaaiBatch）の設定。retime_maaibatch.py 参照。

    重みは MaAI と同じく HF の maai-kyoto/* から読む（ワーカーは HF_HUB_OFFLINE=1 なので
    HF_HOME のキャッシュに入っている必要がある）。
    """
    repo_src: str = ""             # <MaAI の clone>/src。空ならこのファイルの隣の MaAI/src
    mc: bool = False               # True で雑音頑健版 VAP（vap_mc）。vap.mode: vap_mc でも同じ
    batch_size: int = 8            # 一度にエンコードする本数 / AR の窓数
    encoder_chunk_frames: int = 300  # エンコーダ1回あたりのフレーム数（結果は変わらない。50→300 で1.4倍速）
    vap_local_model: str | None = None   # HF の代わりに使うローカル重み（任意）
    bc_local_model: str | None = None


@dataclass
class RetimeConfig:
    enabled: bool = False
    tag: str = "maai"              # 出力分離: audio/<engine>_<tag>/
    # 推論バックエンド。★2026-11-22 から既定は maaibatch（MaAI 公式の一括推論。RUNBOOK 6.5.1）。
    #   maai     … maai pip パッケージ（1フレームずつ forward）
    #   vapbatch … 学習コード(vap-kth/vap-bc)の一括推論。窓ごとに forward 1回
    # 重みは同じ（配布 state_dict と ckpt はビット一致）。差はエンコーダ実装
    # （ONNX Mimi vs PyTorch Mimi）だけなので、値は完全一致しない。
    #   maaibatch … MaAI 公式リポジトリの MaaiBatch（2026-11-22〜）。重みは MaAI と同じ HF から
    backend: str = "maaibatch"
    vapbatch: VapBatchParams = field(default_factory=VapBatchParams)
    maaibatch: MaaiBatchParams = field(default_factory=MaaiBatchParams)
    device: str = "cpu"
    lang: str = "jp"
    frame_rate: int = 10
    model_type: str = "normal"     # normal=CPC / normal-ver2=Mimi
    torch_num_threads: int = 4
    cache_dir: str | None = None
    probe_sec: float = 2.0         # 発話末尾後に与える無音プローブ長
    vap: VapParams = field(default_factory=VapParams)
    bc: BcParams = field(default_factory=BcParams)
    repair: RepairParams = field(default_factory=RepairParams)
    latency: LatencyParams = field(default_factory=LatencyParams)
    hazard: HazardParams = field(default_factory=HazardParams)
    overlap: OverlapParams = field(default_factory=OverlapParams)   # 重なりの場面（rp_v1）
    continued_gap: float = 0.15    # continued: 同一話者直前セグメント末からの間
    fallback: str = "llm"          # "llm" / "default_gap"
    default_gap: float = 0.4
    cache_turn_wavs: bool = True

    @classmethod
    def from_tts_cfg(cls, tts_cfg: dict) -> "RetimeConfig":
        rc = (tts_cfg or {}).get("retime") or {}
        vap = VapParams(**{k: v for k, v in (rc.get("vap") or {}).items()
                           if k in VapParams.__dataclass_fields__})
        bc = BcParams(**{k: v for k, v in (rc.get("bc") or {}).items()
                         if k in BcParams.__dataclass_fields__})
        repair = RepairParams(**{k: v for k, v in (rc.get("repair") or {}).items()
                                 if k in RepairParams.__dataclass_fields__})
        latency = LatencyParams(**{k: v for k, v in (rc.get("latency") or {}).items()
                                   if k in LatencyParams.__dataclass_fields__})
        hazard = HazardParams(**{k: v for k, v in (rc.get("hazard") or {}).items()
                                 if k in HazardParams.__dataclass_fields__})
        vb = VapBatchParams(**{k: v for k, v in (rc.get("vapbatch") or {}).items()
                               if k in VapBatchParams.__dataclass_fields__})
        mb = MaaiBatchParams(**{k: v for k, v in (rc.get("maaibatch") or {}).items()
                                 if k in MaaiBatchParams.__dataclass_fields__})
        ov = OverlapParams(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in (rc.get("overlap") or {}).items()
                              if k in OverlapParams.__dataclass_fields__})
        _reject_unknown_keys("tts.retime", rc, cls,
                             {"vap": VapParams, "bc": BcParams,
                              "repair": RepairParams, "latency": LatencyParams,
                              "hazard": HazardParams, "vapbatch": VapBatchParams,
                              "maaibatch": MaaiBatchParams,
                              "overlap": OverlapParams})
        keys = set(cls.__dataclass_fields__) - {"vap", "bc", "repair", "latency",
                                                "hazard", "vapbatch", "maaibatch",
                                                "overlap"}
        kwargs = {k: v for k, v in rc.items() if k in keys}
        return cls(vap=vap, bc=bc, repair=repair, latency=latency, hazard=hazard,
                   vapbatch=vb, maaibatch=mb, overlap=ov, **kwargs)

    def summary(self) -> dict:
        """manifest 行に埋める実行時設定サマリ。

        backend と重みの指紋を必ず残す。どちらの実装で焼いた音声かを
        後から manifest だけで判別できるようにするため（ハザード方式の
        切り替えのときと同じ運用）。
        """
        out = {
            "tag": self.tag,
            "backend": self.backend,
            "vap": {"mode": self.vap.mode, "signal": self.vap.signal,
                    "threshold": self.vap.threshold,
                    "context_len_sec": self.vap.context_len_sec,
                    "hold_enabled": self.vap.hold_enabled,
                    "hold_threshold": self.vap.hold_threshold},
            "bc": {"mode": self.bc.mode, "signal": self.bc.signal,
                   "threshold": self.bc.threshold,
                   "strategy": self.bc.strategy,
                   "context_len_sec": self.bc.context_len_sec,
                   "max_overlap": self.bc.max_overlap},
            "probe_sec": self.probe_sec,
            "continued_gap": self.continued_gap,
            "fallback": self.fallback,
            "repair": {"enabled": self.repair.enabled,
                       "max_speech_overlap": self.repair.max_speech_overlap,
                       "min_speech_len": self.repair.min_speech_len},
            "latency": asdict(self.latency),
            "hazard": asdict(self.hazard),
        }
        if self.backend == "vapbatch":
            out["vapbatch"] = {
                "vap_ckpt": self.vapbatch.vap_ckpt,
                "bc_ckpt": self.vapbatch.bc_ckpt,
                "vap_sha8": _weight_fingerprint(self.vapbatch.vap_ckpt),
                "bc_sha8": _weight_fingerprint(self.vapbatch.bc_ckpt),
                "mimi_align": self.vapbatch.mimi_align,
                "scan_step_sec": self.vapbatch.scan_step_sec,
            }
        if self.backend == "maaibatch":
            from retime_maaibatch import DEFAULT_REPO_SRC, maai_repo_commit
            src = self.maaibatch.repo_src or DEFAULT_REPO_SRC
            out["maaibatch"] = {
                "repo_src": src,
                "repo_commit": maai_repo_commit(src),
                "vap_mc": bool(self.maaibatch.mc or self.vap.mode == "vap_mc"),
                "vap_local_model": self.maaibatch.vap_local_model,
                "bc_local_model": self.maaibatch.bc_local_model,
            }
        return out


# --------------------------------------------------------------------------
# リサンプリング（スコアリング用 16kHz 変換）
# --------------------------------------------------------------------------

_resampler = None


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """エンジンSR → 16kHz mono float32。soxr 優先、scipy フォールバック。"""
    global _resampler
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr == MAAI_SR:
        return wav
    if _resampler is None:
        try:
            import soxr
            _resampler = ("soxr", soxr)
        except ImportError:
            from scipy.signal import resample_poly
            _resampler = ("scipy", resample_poly)
            logger.info("soxr が無いため scipy.signal.resample_poly を使います。")
    kind, mod = _resampler
    if kind == "soxr":
        return np.asarray(mod.resample(wav, sr, MAAI_SR), dtype=np.float32)
    from math import gcd
    g = gcd(MAAI_SR, sr)
    return np.asarray(mod(wav, MAAI_SR // g, sr // g), dtype=np.float32)


# --------------------------------------------------------------------------
# MaaiScorer: 1モデル分のオフライン一括スコアリング
# --------------------------------------------------------------------------

def _as_floats(v):
    """結果フィールド（list / np / torch tensor）を float のリストに正規化。"""
    if v is None:
        return None
    if hasattr(v, "detach"):        # torch tensor
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr]


class MaaiScorer:
    """Maai を直接 process() 呼びでオフライン一括実行するラッパ。

    スレッド（start()）やリアルタイムペーシングは使わない。
    1600サンプル（@10Hz）を process() に渡すごとに結果1件を回収する。
    フレーム k は供給サンプル [k*chunk, (k+1)*chunk) に対応し、
    予測時刻は 供給開始時刻 + (k+1)/frame_rate 秒。
    """

    def __init__(self, mode: str, lang: str, frame_rate: int,
                 context_len_sec: int, device: str,
                 model_type: str = "normal", cache_dir: str | None = None,
                 num_threads: int | None = None):
        import torch
        if num_threads:
            torch.set_num_threads(int(num_threads))
        from maai import Maai, MaaiInput
        self.mode = mode
        self.stats = ScorerStats()
        self.frame_rate = float(frame_rate)
        # 1フレーム分のサンプル数（= maai の audio_frame_size - frame_contxt_padding）。
        # frame_rate=12.5 のような小数でも round で正しく 1280 になる（int切り捨ては不可:
        # 16000//12=1333 だと ONNX Mimi の固定長 1920 期待とズレて shape mismatch になる）。
        self.chunk = int(round(MAAI_SR / self.frame_rate))
        kwargs = dict(mode=mode, lang=lang, frame_rate=frame_rate,
                      context_len_sec=context_len_sec,
                      audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(),
                      device=device)
        if model_type and model_type != "normal":
            kwargs["model_type"] = model_type
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        t_load = time.perf_counter()
        try:
            self.maai = Maai(**kwargs)
        except TypeError:
            # 古い/新しい pip 版でオプション引数が無い場合は落として再試行
            for opt in ("cache_dir", "model_type"):
                kwargs.pop(opt, None)
            self.maai = Maai(**kwargs)
        self.stats.load_seconds = time.perf_counter() - t_load
        logger.info(f"MaAI({mode}) ロード完了: {self.stats.load_seconds:.1f}s")
        self._can_reset = hasattr(self.maai, "reset_runtime_state")
        if not self._can_reset:
            logger.warning(f"Maai(mode={mode}) に reset_runtime_state がありません。"
                           "判断ごとにインスタンスを再生成します（低速）。")
            self._ctor_kwargs = kwargs

    def _reset(self):
        if self._can_reset:
            self.maai.reset_runtime_state()
        else:
            from maai import Maai
            self.maai = Maai(**self._ctor_kwargs)
        self._drain()  # 残留結果を捨てる

    def _drain(self) -> list[dict]:
        out = []
        q = self.maai.result_dict_queue
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                break
        return out

    def score(self, ch1_16k: np.ndarray, ch2_16k: np.ndarray,
              scan_from_sample: int = 0) -> list[tuple[int, dict]]:
        """等長にゼロ埋めした2chの16kHz音声をスコアし、フレーム列を返す。

        maai.process() は「ちょうど1フレーム分（self.chunk サンプル）」を1回ずつ受け取る
        ストリーミングAPI。ONNX Mimi エンコーダ(model_type=normal-ver2)は固定長チャンク
        しか受け付けないため、コンテキスト部分も含めて全区間をフレーム単位で流す
        （旧CPCエンコーダのような可変長の一括投入はできない）。冒頭 context_len_sec 分の
        結果は呼び出し側が時刻窓 (t_k < w_lo) で捨てるため、ここでは全フレームを返す。

        scan_from_sample は後方互換のため受け取るが未使用（結果の取捨は呼び出し側の時刻窓）。

        Returns: [(end_sample, result_dict)] — end_sample は「供給音声の先頭から
        何サンプルまでを見た時点の予測か」。時刻 = 供給開始時刻 + end_sample/16000。
        """
        t0 = time.perf_counter()
        n = max(len(ch1_16k), len(ch2_16k))
        n_pad = ((n + self.chunk - 1) // self.chunk) * self.chunk
        x1 = np.zeros(n_pad, dtype=np.float32)
        x2 = np.zeros(n_pad, dtype=np.float32)
        x1[:len(ch1_16k)] = ch1_16k
        x2[:len(ch2_16k)] = ch2_16k
        self._reset()
        results: list[tuple[int, dict]] = []
        # pip版 maai は bc モデルが毎フレーム print する（vap_bc.py の消し忘れ）ため
        # stdout を吸収する。それ以外の出力は WARNING でログに回す。
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pos = 0
            while pos < n_pad:
                self.maai.process(x1[pos:pos + self.chunk], x2[pos:pos + self.chunk])
                pos += self.chunk
                for r in self._drain():
                    results.append((pos, r))
            for r in self._drain():
                results.append((pos, r))
        for line in buf.getvalue().splitlines():
            if not line or line.startswith("p_bc:"):
                continue  # bcモデルの毎フレームprint（pip版の消し忘れ）
            if "Average processing time" in line:
                continue  # 周期的な性能モニタ出力
            logger.warning(f"maai: {line}")
        self.stats.add(n_pad // self.chunk, time.perf_counter() - t0)
        return results


# --------------------------------------------------------------------------
# スコアラの factory（backend の唯一の分岐点）
# --------------------------------------------------------------------------

def make_scorer(cfg: "RetimeConfig", mode_key: str):
    """mode_key は "vap" / "bc"。cfg.backend でバックエンドを選ぶ。

    MaaiRetimer だけでなく診断ツール（retime_debug / eval_vap_readout /
    plot_vap_response）からも呼ぶ。両バックエンドを同じ入口で作れないと
    old/new の比較ができないため、ここに集約する。
    """
    params = cfg.vap if mode_key == "vap" else cfg.bc
    label = "VAP" if mode_key == "vap" else "相槌"

    if cfg.backend == "vapbatch":
        from retime_vapbatch import BatchVapScorer
        ckpt = (cfg.vapbatch.vap_ckpt if mode_key == "vap"
                else cfg.vapbatch.bc_ckpt)
        logger.info(f"{label} モデルをロード中…（backend=vapbatch, "
                    f"ckpt={ckpt}, ctx={params.context_len_sec}s, device={cfg.device}）")
        return BatchVapScorer(
            mode=params.mode, ckpt_path=ckpt, frame_rate=cfg.frame_rate,
            context_len_sec=params.context_len_sec, device=cfg.device,
            batch_size=cfg.vapbatch.batch_size,
            scan_step_sec=cfg.vapbatch.scan_step_sec,
            fold_context_mult=cfg.vapbatch.fold_context_mult,
            mimi_align=cfg.vapbatch.mimi_align,
            torch_num_threads=cfg.torch_num_threads,
        )

    if cfg.backend == "maaibatch":
        from retime_maaibatch import DEFAULT_REPO_SRC, MaaiBatchScorer
        mb = cfg.maaibatch
        local = mb.vap_local_model if mode_key == "vap" else mb.bc_local_model
        logger.info(f"{label} モデルをロード中…（backend=maaibatch, mode={params.mode}, "
                    f"mc={mb.mc}, ctx={params.context_len_sec}s, device={cfg.device}）")
        return MaaiBatchScorer(
            mode=params.mode, lang=cfg.lang, frame_rate=cfg.frame_rate,
            context_len_sec=params.context_len_sec, device=cfg.device,
            model_type=cfg.model_type,
            repo_src=mb.repo_src or DEFAULT_REPO_SRC, mc=mb.mc,
            batch_size=mb.batch_size, encoder_chunk_frames=mb.encoder_chunk_frames,
            local_model=local, cache_dir=cfg.cache_dir,
            torch_num_threads=cfg.torch_num_threads,
        )

    if cfg.backend != "maai":
        raise ValueError(
            f"tts.retime.backend は 'maai' / 'vapbatch' / 'maaibatch'（受領: {cfg.backend!r}）")

    logger.info(f"MaAI {label} モデルをロード中…（mode={params.mode}, "
                f"lang={cfg.lang}, ctx={params.context_len_sec}s, device={cfg.device}）")
    return MaaiScorer(params.mode, cfg.lang, cfg.frame_rate,
                      params.context_len_sec, cfg.device,
                      cfg.model_type, cfg.cache_dir, cfg.torch_num_threads)


# --------------------------------------------------------------------------
# ScoreTimeline: スコアリング用の16kHzタイムラインミラー
# --------------------------------------------------------------------------

class ScoreTimeline:
    """確定済み配置の 16kHz ミラー（スコアリング専用。最終ミックスとは別物）。

    place() でターン波形を1回だけ 16k にリサンプルして加算配置し、
    slice() で任意区間の2chを取り出す（末尾越えはゼロ埋め＝無音プローブを兼ねる）。
    """

    def __init__(self, num_ch: int, sr_engine: int):
        self.num_ch = num_ch
        self.sr_engine = sr_engine
        self._buf = np.zeros((MAAI_SR * 60, num_ch), dtype=np.float32)  # 60秒から成長
        self._len = 0  # 有効サンプル数（配置済み最遠端）

    def place(self, ch: int, start_sec: float, wav_engine: np.ndarray):
        w = resample_to_16k(wav_engine, self.sr_engine)
        start = int(round(start_sec * MAAI_SR))
        end = start + len(w)
        if end > len(self._buf):
            grow = np.zeros((max(end, len(self._buf) * 2), self.num_ch),
                            dtype=np.float32)
            grow[:len(self._buf)] = self._buf
            self._buf = grow
        self._buf[start:end, ch] += w
        self._len = max(self._len, end)

    def slice(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """[t0, t1) 秒の (ch0, ch1) を返す。確定済み末尾より先はゼロ（無音）。"""
        s0 = max(0, int(round(t0 * MAAI_SR)))
        s1 = max(s0, int(round(t1 * MAAI_SR)))
        out = np.zeros((s1 - s0, self.num_ch), dtype=np.float32)
        avail = min(self._len, s1)
        if avail > s0:
            out[:avail - s0] = self._buf[s0:avail]
        return out[:, 0], out[:, 1]


# --------------------------------------------------------------------------
# MaaiRetimer: 逐次配置プランナー
# --------------------------------------------------------------------------

class MaaiRetimer:
    """MaAI 予測に基づいてターン配置（start_sec）を決めるプランナー。

    compose_dialogue() から plan() を呼ぶ。None を返したらレガシー連鎖に
    フォールバックする（3話者以上など）。モデルは初回使用時にロードし、
    以後全対話で再利用する。
    """

    def __init__(self, cfg: RetimeConfig):
        self.cfg = cfg
        self._vap: MaaiScorer | None = None
        self._bc: MaaiScorer | None = None
        self.method_counts: dict[str, int] = {}  # ラン全体の集計（ログ用）
        self._rng = np.random.default_rng(cfg.latency.seed)
        # _decide_vap が最後の判断の内訳を置く場所（plan() が retime dict に混ぜる）。
        # 再フィットのため tc（潜時を足す前）と L を別々に残す。
        self._vap_detail: dict = {}

    # --- 応答潜時 L（R2） ---

    def _seed_rng(self, dialogue_id: str):
        """対話ごとに決定的な乱数列にする（--recompose で結果が再現するように）。"""
        h = int.from_bytes(str(dialogue_id).encode("utf-8")[-8:].ljust(8, b"\0"), "big")
        self._rng = np.random.default_rng((self.cfg.latency.seed * 1_000_003 + h) % (2**63))

    def _sample_latency(self, tc_off: float) -> tuple[float, float]:
        """L を引き、(オフセット, L) を返す。

        結果が [-max_overlap, max_gap] に収まるまで引き直す（棄却サンプリング）。
        クランプにすると境界に人工的なスパイクが立つため。
        """
        p = self.cfg.latency
        w = np.array([p.w_fast, p.w_overlap, p.w_delayed], dtype=float)
        w = w / w.sum()
        for _ in range(64):
            c = self._rng.choice(3, p=w)
            if c == 0:
                lat = self._rng.normal(p.m, p.sd_fast)
            elif c == 1:
                lat = p.m - np.exp(self._rng.normal(p.mu_overlap, p.sd_overlap))
            else:
                lat = p.m + np.exp(self._rng.normal(p.mu_delayed, p.sd_delayed))
            off = tc_off + float(lat)
            if -p.max_overlap <= off <= p.max_gap:
                return off, float(lat)
        off = min(max(tc_off + float(lat), -p.max_overlap), p.max_gap)
        return off, float(lat)

    # --- ハザード方式（R2 別案） ---

    def _hazard_fire(self, seq: list, start_idx: int) -> tuple[float | None, int]:
        """start_idx から毎フレームくじを引き、当たったフレームの時刻を返す。

        Returns: (発火時刻 or None, 引いた回数)。窓の右端まで当たらなければ None。
        """
        h = self.cfg.hazard
        rolls = 0
        for j in range(start_idx, len(seq)):
            t_k, p = seq[j]
            if p is None:
                continue
            rolls += 1
            pr = min(max(float(p) * h.scale, h.min_prob), h.max_prob)
            if self._rng.random() < pr:
                return t_k, rolls
        return None, rolls

    def _sample_start_offset(self) -> float:
        """発火時刻に足すオフセット。[offset_min, offset_max] で棄却サンプリング。"""
        h = self.cfg.hazard
        x = h.offset_mean
        for _ in range(64):
            x = float(self._rng.normal(h.offset_mean, h.offset_sd))
            if h.offset_min <= x <= h.offset_max:
                return x
        return float(min(max(x, h.offset_min), h.offset_max))

    # --- scorers（lazy） ---

    def _vap_scorer(self):
        if self._vap is None:
            self._vap = make_scorer(self.cfg, "vap")
        return self._vap

    def _bc_scorer(self):
        if self._bc is None:
            self._bc = make_scorer(self.cfg, "bc")
        return self._bc

    # --- シグナル抽出 ---

    def _vap_signal(self, frame: dict, ch: int) -> float | None:
        v = _as_floats(frame.get(self.cfg.vap.signal))
        if v is None or len(v) <= ch:
            return None
        return v[ch]

    def _bc_signal(self, frame: dict) -> float | None:
        if self.cfg.bc.mode == "bc_2type":
            react = _as_floats(frame.get("p_bc_react")) or [0.0]
            emo = _as_floats(frame.get("p_bc_emo")) or [0.0]
            r, e = react[0], emo[0]
            sig = self.cfg.bc.signal
            if sig == "react":
                return r
            if sig == "emo":
                return e
            if sig == "max":
                return max(r, e)
            return r + e  # "sum"
        v = _as_floats(frame.get("p_bc"))
        return v[0] if v else None

    # --- メイン ---

    def plan(self, record: dict, turns: list, wav_by_turn: dict, sr: int,
             channel_of: dict, failed_turns: set | None = None) -> list[dict] | None:
        """全ターンの配置を決める。Returns: [{index, start_sec, retime}] or None。"""
        if len(channel_of) != 2:
            logger.info(f"[{record['id']}] 話者{len(channel_of)}人はリタイム対象外"
                        "（2人のみ）。レガシー連鎖で配置します。")
            return None
        failed_turns = failed_turns or set()
        cfg = self.cfg
        self._seed_rng(record.get("id", ""))
        frame_dur = 1.0 / cfg.frame_rate
        timeline = ScoreTimeline(num_ch=2, sr_engine=sr)

        plans: list[dict] = []
        placed: dict[int, tuple[float, float]] = {}   # index -> (start, end)
        frontier = 0.0            # 配置済み音声の最遠端（秒）
        frontier_idx = -1         # 最遠端を作ったターン index
        last_end_on_channel = {0: 0.0, 1: 0.0}
        last_longbc_end = {0: 0.0, 1: 0.0}  # ch -> 長い相槌（実質発話）の末尾
        last_speech_end_by_speaker: dict[str, float] = {}
        prev_speech_speaker: str | None = None
        prev_placed_end = 0.0     # 直前（リスト順）ターンの配置末尾
        legacy_prev_end = 0.0     # 純LLM連鎖（manifest 記録用）
        # 重なりの場面（rp_v1）: 同時に話し始める2人の1人目（2人目が collision を持つ）
        coll_first = self._collision_firsts(turns)

        for i, t in enumerate(turns):
            sp = t["speaker"]
            ch = channel_of[sp]
            wav = wav_by_turn[i]
            dur = len(wav) / sr
            ttype = t.get("type", "speech")
            if ttype not in ("speech", "backchannel"):
                ttype = "speech"  # type 異常は speech 扱い
            llm_off = 0.0 if i == 0 else float(t.get("start_offset", 0.0) or 0.0)
            llm_start = max(0.0, legacy_prev_end + llm_off)

            p_peak = None
            anchor_sec = prev_placed_end
            vap_detail: dict = {}
            if i == 0:
                start, method = 0.0, "first"
            elif i in failed_turns:
                # 自身が無音代替: 予測根拠なし → LLMオフセットを直前配置末尾に適用
                start, method = self._fallback_start(prev_placed_end, llm_off), "llm-synthfailed"
            elif ttype == "backchannel":
                start, method, p_peak, anchor_sec = self._decide_bc(
                    i, t, turns, placed, timeline, failed_turns,
                    prev_placed_end, llm_off, channel_of, frame_dur, dur)
                # 次の speech が長い沈黙のあとなら、相槌はアンカー発話の末尾までに収める
                # （沈黙に食い込むと frontier が伸びて、沈黙の始まりがずれる）。
                if _next_speech_has_pause(turns, i) and start + dur > anchor_sec:
                    start = max(0.0, anchor_sec - dur)
                    vap_detail = {"pause_clamped": True}
            elif ttype == "speech" and t.get("overlap") in RP_OVERLAP_KINDS \
                    and self._rp_target(i, turns, placed, sp) is not None:
                # 重なりの場面（rp_v1）: 相手の発話に重ねて置く（VAP では決めない）
                k = self._rp_target(i, turns, placed, sp)
                start, method, anchor_sec, vap_detail = self._decide_rp(t["overlap"], placed[k])
            elif ttype == "speech" and self._after_collision(i, turns, placed) is not None:
                # 2人とも言いかけで止まった直後: 少し間をおいてから（譲るひと言など）
                anchor_sec = self._after_collision(i, turns, placed)
                start, method = anchor_sec + self._rp_uniform(cfg.overlap.after_collision_sec, log=True), \
                    "rp-after-collision"
            elif ttype == "speech" and i in coll_first:
                # 同時に話し始める1人目: 少し間があいてから話し始める（2人目はこの開始に重ねる）
                anchor_sec = frontier
                start, method = frontier + self._rp_uniform(cfg.overlap.lull_sec), "rp-lull"
            elif ttype == "speech" and self._react_through(i, turns, sp) \
                    and sp in last_speech_end_by_speaker:
                # 相手の短い反応（react）が重なっても止まらずに話し続ける: 自分の直前の発話の末尾から。
                # ★試走3: 長めの反応（「すごいですね。」）と 1.5秒重なった → 反応の終わりの through_max_overlap 秒前より前には始めない
                anchor_sec = last_speech_end_by_speaker[sp]
                start, method = anchor_sec + self._rp_uniform(cfg.overlap.through_gap), "rp-through"
                j = self._prev_speech(i, turns)
                if j is not None and j in placed:
                    start = max(start, placed[j][1] - cfg.overlap.through_max_overlap)
            elif t.get("pause_before_sec") is not None:
                # ユーザの長い沈黙（Stage2 の dialogue.silence）: VAP で決めず、両 ch で
                # 置いた音の最後（frontier）から指定の秒数だけ空けて置く。沈黙の間は
                # どちらの声も入らない。
                anchor_sec = frontier
                start, method = frontier + float(t["pause_before_sec"]), "fixed-pause"
            elif t.get("continued") and sp in last_speech_end_by_speaker:
                # 相槌分割の続き: 同一話者の直前セグメント末を基準（相槌位置に非依存）
                anchor_sec = last_speech_end_by_speaker[sp]
                start, method = anchor_sec + cfg.continued_gap, "continued"
            elif sp == prev_speech_speaker:
                # 同一話者の連続 speech (hold): 自chの p_now が閾値を下回るまで
                # 間を置く（無効時は LLM オフセットを [0, max_gap] にクランプ）。
                hold_anchor = last_speech_end_by_speaker.get(sp, prev_placed_end)
                start, method, p_peak, anchor_sec = self._decide_hold(
                    ch, timeline, hold_anchor, i - 1, failed_turns,
                    prev_placed_end, llm_off, frame_dur)
            else:
                self._vap_detail = {}
                start, method, p_peak, anchor_sec = self._decide_vap(
                    ch, timeline, frontier, frontier_idx, failed_turns,
                    prev_placed_end, llm_off, frame_dur)
                vap_detail = self._vap_detail

            # 相手chで再生中の長い相槌を実質 speech が踏み潰さない:
            # 相槌末 - max_speech_overlap より前には開始しない（配置時に確定させる
            # ことで、以降の MaAI 判断も修正済みタイムラインを見る。repair は
            # 時間順の前方パスなので「後から始まる相槌」との衝突を直せない）。
            bc_waited = False
            if (cfg.repair.enabled and ttype == "speech" and method != "fixed-pause"
                    and not method.startswith("rp-")
                    and _core_len(t.get("text", "")) > cfg.repair.min_speech_len):
                lim = last_longbc_end[1 - ch] - cfg.repair.max_speech_overlap
                if start < lim - 1e-6:
                    start = lim
                    bc_waited = True

            # 同一チャネルの自己重なり（二重声）禁止
            clamped = False
            if start < last_end_on_channel[ch] - 1e-6:
                start = last_end_on_channel[ch]
                clamped = True
            start = max(0.0, start)

            end = start + dur
            plans.append({
                "index": i,
                "start_sec": round(start, 3),
                "retime": {
                    "method": method,
                    "anchor_sec": round(anchor_sec, 3),
                    "llm_start_sec": round(llm_start, 3),
                    "decided_offset": round(start - anchor_sec, 3),
                    "p_peak": round(p_peak, 3) if p_peak is not None else None,
                    "clamped": clamped,
                    # tc_offset / latency は潜時モデルの再フィット用（VAP を回さずに
                    # fit_gap_latency.py を回し直せるよう、足す前と足した分を残す）
                    **vap_detail,
                },
            })
            if bc_waited:  # 相槌待ちで開始を遅らせた（集計・デバッグ用）
                plans[-1]["retime"]["bc_waited"] = True
                self.method_counts["bc-waited"] = \
                    self.method_counts.get("bc-waited", 0) + 1
            self.method_counts[method] = self.method_counts.get(method, 0) + 1

            # トラッカー更新 + タイムラインに確定配置
            timeline.place(ch, start, wav)
            placed[i] = (start, end)
            if end > frontier:
                frontier, frontier_idx = end, i
            last_end_on_channel[ch] = max(last_end_on_channel[ch], end)
            if (ttype == "backchannel"
                    and _core_len(t.get("text", "")) > cfg.repair.min_speech_len):
                last_longbc_end[ch] = max(last_longbc_end[ch], end)
            if ttype == "speech":
                last_speech_end_by_speaker[sp] = end
                prev_speech_speaker = sp
            prev_placed_end = end
            legacy_prev_end = llm_start + dur

        # 明らかな発話衝突（speech同士の大きな被り）を修復（相槌・短フィラーのかぶせは残す）
        self._repair_collisions(plans, turns, wav_by_turn, sr, channel_of)
        return plans

    def _repair_collisions(self, plans: list, turns: list, wav_by_turn: dict,
                           sr: int, channel_of: dict):
        """実質発話同士が max_speech_overlap を超えて被る箇所を後ろへずらす。

        実質発話（実質文字数 > min_speech_len。speech と長い相槌の両方）を
        ブロッカーとし、それに大きく被る speech を修復する。当該ターンと、元の
        時間順でそれ以降の全ターンを同じ delta だけずらす（相対タイミングを
        保ったまま隙間を空ける）。長い相槌はブロッカーにはなるが修復対象には
        しない（_decide_bc が max_overlap で食い込みを制限済み）。極短の
        フィラー・相槌はどちらでもない＝かぶせを残す。
        """
        rp = self.cfg.repair
        if not rp.enabled or not plans:
            return
        starts = {p["index"]: float(p["start_sec"]) for p in plans}
        durs = {p["index"]: len(wav_by_turn[p["index"]]) / sr for p in plans}

        order = sorted((p["index"] for p in plans), key=lambda i: starts[i])
        last_sub_end = {0: -1e9, 1: -1e9}  # ch -> 実質発話（speech/長い相槌）の末尾
        repaired: dict[int, dict] = {}
        # 重なりの場面（rp_v1）で意図して重ねたターン・言いかけのターンはずらさない（ブロッカーにはなる）
        rp_fixed = {p["index"] for p in plans
                    if str(p["retime"].get("method", "")).startswith("rp-")
                    or turns[p["index"]].get("unsaid")}
        for pos, i in enumerate(order):
            t = turns[i]
            ch = channel_of[t["speaker"]]
            if _core_len(t.get("text", "")) <= rp.min_speech_len:
                continue  # 極短のフィラー・相槌はブロッカーにも修復対象にもしない
            s = starts[i]
            if t.get("type", "speech") == "speech" and i not in rp_fixed:
                other = 1 - ch
                allowed = last_sub_end[other] - rp.max_speech_overlap
                if s < allowed - 1e-6:
                    delta = allowed - s
                    repaired[i] = {"shift_sec": round(delta, 3),
                                   "overlap_before": round(last_sub_end[other] - s, 3)}
                    for j in order[pos:]:  # 自分と、時間的に後ろの全ターンをずらす
                        starts[j] += delta
                    s = starts[i]
            last_sub_end[ch] = max(last_sub_end[ch], s + durs[i])

        if repaired:
            logger.info(f"衝突修復: {len(repaired)}件の speech 大被りを後ろにずらしました。")
        for p in plans:  # start_sec を書き戻し、修復情報を retime に付与
            i = p["index"]
            p["start_sec"] = round(starts[i], 3)
            if i in repaired:
                p["retime"]["repaired"] = repaired[i]
                self.method_counts["repaired"] = self.method_counts.get("repaired", 0) + 1

    # --- 重なりの場面（rp_v1。rp_plan.py）---

    @staticmethod
    def _prev_speech(i: int, turns: list) -> int | None:
        for j in range(i - 1, -1, -1):
            if turns[j].get("type", "speech") != "backchannel":
                return j
        return None

    def _collision_firsts(self, turns: list) -> set:
        out = set()
        for i, t in enumerate(turns):
            if t.get("overlap") == "collision" and t.get("type", "speech") == "speech":
                j = self._prev_speech(i, turns)
                if j is not None and turns[j]["speaker"] != t["speaker"]:
                    out.add(j)
        return out

    def _rp_target(self, i: int, turns: list, placed: dict, sp: str) -> int | None:
        """重ねる相手のターン（直前の相手の speech。配置済みのもの）。"""
        for j in range(i - 1, -1, -1):
            if turns[j].get("type", "speech") == "backchannel" or turns[j]["speaker"] == sp:
                if turns[j].get("type", "speech") != "backchannel":
                    return None          # 自分の発話を挟むなら重ねる相手はいない
                continue
            return j if j in placed else None
        return None

    def _react_through(self, i: int, turns: list, sp: str) -> bool:
        """直前の speech が相手の react で、その前が自分の発話（＝反応をはさんで話し続ける）か。"""
        j = self._prev_speech(i, turns)
        if j is None or turns[j].get("overlap") != "react" or turns[j]["speaker"] == sp:
            return False
        k = self._prev_speech(j, turns)
        return k is not None and turns[k]["speaker"] == sp

    def _rp_uniform(self, lo_hi, log: bool = False) -> float:
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if hi <= lo:
            return lo
        if log and self.cfg.overlap.log_uniform and lo > 0:
            return float(np.exp(self._rng.uniform(np.log(lo), np.log(hi))))
        return float(self._rng.uniform(lo, hi))

    def _after_collision(self, i: int, turns: list, placed: dict):
        """直前の speech が「2人とも言いかけで止まった同時発話」の2人目なら、2人の言いかけの末尾（無ければ None）。"""
        j = self._prev_speech(i, turns)
        if j is None or turns[j].get("overlap") != "collision" or not turns[j].get("unsaid"):
            return None
        k = self._prev_speech(j, turns)
        ends = [placed[x][1] for x in (j, k) if x is not None and x in placed]
        return max(ends) if ends else None

    def _decide_rp(self, kind: str, target: tuple) -> tuple:
        """(start, method, anchor_sec, detail)。target は重ねる相手の (start, end)。"""
        o = self.cfg.overlap
        ks, ke = target
        dur = max(0.0, ke - ks)
        if kind == "collision":
            off = self._rp_uniform(o.collision_offset)
            return ks + off, "rp-collision", ks, {"rp_offset": round(off, 3)}
        sec, frac = {"interrupt": (o.interrupt_sec, o.interrupt_max_frac),
                     "coop": (o.coop_sec, o.coop_max_frac),
                     "react": (o.react_sec, o.react_max_frac)}[kind]
        ov = min(self._rp_uniform(sec, log=(kind == "interrupt")), frac * dur)
        return ke - ov, f"rp-{kind}", ke, {"rp_overlap": round(ov, 3)}

    # --- 個別判断 ---

    def _fallback_start(self, base_end: float, llm_off: float) -> float:
        if self.cfg.fallback == "default_gap":
            return base_end + self.cfg.default_gap
        return max(0.0, base_end + llm_off)

    def _decide_vap(self, next_ch: int, timeline: ScoreTimeline,
                    frontier: float, frontier_idx: int, failed_turns: set,
                    prev_placed_end: float, llm_off: float, frame_dur: float):
        """話者交替: p_now[次話者ch] の閾値クロスで開始時刻を決める。"""
        v = self.cfg.vap
        anchor = frontier
        if frontier_idx in failed_turns:
            # 直前の音声が無音代替 → 予測根拠なし
            return (self._fallback_start(prev_placed_end, llm_off),
                    "llm-synthfailed", None, anchor)
        w_lo = max(0.0, anchor - v.max_overlap)
        w_hi = anchor + self.cfg.probe_sec
        feed_start = max(0.0, w_lo - v.context_len_sec)
        x0, x1 = timeline.slice(feed_start, w_hi)
        scan_from = max(0, int((w_lo - feed_start) * MAAI_SR)
                        - MAAI_SR // int(self.cfg.frame_rate))
        frames = self._vap_scorer().score(x0, x1, scan_from)  # 対話ch0→MaAI ch1, ch1→ch2

        lat = self.cfg.latency
        hz = self.cfg.hazard
        self._vap_detail = {}
        # 窓内のフレーム列を先に取り出す（ハザード方式がクロス以降を走査するため）。
        # p=None のフレームも run_len のリセット判定に要るのでそのまま残す。
        seq: list[tuple[float, float | None]] = []
        for end_sample, fr in frames:
            t_k = feed_start + end_sample / MAAI_SR
            if t_k < w_lo:
                continue
            if t_k > w_hi:
                break
            seq.append((t_k, self._vap_signal(fr, next_ch)))

        run_len = 0
        t_first = p_first = None      # クロスが始まったフレーム
        t_below = p_below = None      # その直前の閾値未満フレーム（副フレーム補間用）
        for idx, (t_k, p) in enumerate(seq):
            if p is not None and p >= v.threshold:
                run_len += 1
                if run_len == 1:
                    t_first, p_first = t_k, p
                if run_len >= v.min_consecutive_frames:
                    tc = t_first
                    # R1: 閾値をまたぐ2フレームを線形補間して連続値にする。
                    # そのままだとフレーム格子（frame_rate の逆数）の櫛になる。
                    if (lat.subframe and p_below is not None
                            and p_first > p_below):
                        frac = (v.threshold - p_below) / (p_first - p_below)
                        tc = t_below + frac * (t_first - t_below)
                    tc_off = min(max(tc - anchor, -v.max_overlap), v.max_gap)
                    if hz.enabled:
                        # R2 別案: クロスを確認したこのフレームから毎フレームくじを
                        # 引き、当たった時刻 + N(offset_mean, offset_sd) を開始点にする。
                        fire_t, rolls = self._hazard_fire(seq, idx)
                        timed_out = fire_t is None
                        if timed_out:                 # 窓内で当たらなかった
                            fire_t = seq[-1][0]
                        add = self._sample_start_offset()
                        off = min(max(fire_t + add - anchor, -v.max_overlap),
                                  v.max_gap)
                        self._vap_detail = {
                            # tc_offset は旧方式と比較するための参考値（副フレーム
                            # 補間後のクロス時刻）。ハザードの時計はクロスを確認した
                            # フレーム t_k から回るので、実配置は
                            #   fire_offset + start_offset
                            # で再現できる（= anchor からの秒）。
                            "tc_offset": round(tc_off, 4),
                            "fire_offset": round(fire_t - anchor, 4),
                            "hazard_wait": round(fire_t - t_k, 4),
                            "hazard_rolls": rolls,
                            "start_offset": round(add, 4),
                            "subframe": False,
                        }
                        return (anchor + off,
                                "vap-hazard-timeout" if timed_out else "vap-hazard",
                                p, anchor)
                    if lat.enabled:
                        # R2: p_now は「今後約600ms」の予測。クロス時刻は交替が
                        # 予測可能になった点であって開始点ではないので潜時を足す。
                        off, lat_val = self._sample_latency(tc_off)
                    else:
                        off, lat_val = tc_off, None
                        if 0.0 <= off < v.min_gap:   # R3: 潜時モデル使用時は掛けない
                            off = v.min_gap          # （+min_gap に人工スパイクが立つ）
                    self._vap_detail = {"tc_offset": round(tc_off, 4),
                                        "latency": None if lat_val is None else round(lat_val, 4),
                                        "subframe": bool(lat.subframe and tc != t_first)}
                    return anchor + off, "vap", p, anchor
            else:
                run_len = 0
                if p is not None:
                    t_below, p_below = t_k, p
        # 窓内で閾値クロス無し
        if lat.enabled and lat.apply_to_fallback:
            # R4: LLM の定型オフセット（1.2秒に集中）を使わず同じ分布から引く
            off, lat_val = self._sample_latency(0.0)
            self._vap_detail = {"tc_offset": 0.0, "latency": round(lat_val, 4),
                                "subframe": False}
            return anchor + off, "fallback-latency", None, anchor
        return (self._fallback_start(prev_placed_end, llm_off),
                "fallback-llm", None, anchor)

    def _decide_hold(self, ch: int, timeline: ScoreTimeline, anchor: float,
                     prev_idx: int, failed_turns: set,
                     prev_placed_end: float, llm_off: float, frame_dur: float):
        """同一話者の連続 speech (hold): p_now[自ch] が hold_threshold を
        下回る最初の点まで間を置く。話者交替(_decide_vap)の下降版。"""
        v = self.cfg.vap

        def _llm():
            off = min(max(llm_off, 0.0), v.max_gap)
            return prev_placed_end + off, "llm-samespeaker", None, anchor

        if not v.hold_enabled or prev_idx in failed_turns:
            return _llm()
        w_lo = anchor
        w_hi = anchor + self.cfg.probe_sec
        feed_start = max(0.0, w_lo - v.context_len_sec)
        x0, x1 = timeline.slice(feed_start, w_hi)
        scan_from = max(0, int((w_lo - feed_start) * MAAI_SR)
                        - MAAI_SR // int(self.cfg.frame_rate))
        frames = self._vap_scorer().score(x0, x1, scan_from)

        run_len = 0
        for end_sample, fr in frames:
            t_k = feed_start + end_sample / MAAI_SR
            if t_k < w_lo:
                continue
            if t_k > w_hi:
                break
            p = self._vap_signal(fr, ch)
            if p is not None and p < v.hold_threshold:
                run_len += 1
                if run_len >= v.min_consecutive_frames:
                    tc = t_k - (run_len - 1) * frame_dur  # 降下開始フレームの時刻
                    off = min(max(tc - anchor, v.min_gap), v.max_gap)
                    return anchor + off, "vap-hold", p, anchor
            else:
                run_len = 0
        # probe 窓内で閾値を下回らなかった → LLM フォールバック
        return _llm()

    def _decide_bc(self, i: int, t: dict, turns: list, placed: dict,
                   timeline: ScoreTimeline, failed_turns: set,
                   prev_placed_end: float, llm_off: float,
                   channel_of: dict, frame_dur: float, dur: float):
        """相槌: 相手発話セグメント末をアンカーに p_bc のピーク/クロスで打点を決める。

        dur は相槌自体の音声長（秒）。max_overlap より長い相槌は開始を
        アンカー末 - max_overlap 以降に制限し、長文相槌が相手発話に
        丸ごと重なる（プロローグ的な二重発話になる）のを防ぐ。
        """
        b = self.cfg.bc
        sp = t["speaker"]
        # アンカー = 直前に配置された別話者の speech ターン
        anchor_idx = None
        for j in range(i - 1, -1, -1):
            if j in placed and turns[j]["speaker"] != sp \
                    and turns[j].get("type", "speech") != "backchannel":
                anchor_idx = j
                break
        if anchor_idx is None:
            return (self._fallback_start(prev_placed_end, llm_off),
                    "fallback-llm", None, prev_placed_end)
        if anchor_idx in failed_turns:
            return (self._fallback_start(prev_placed_end, llm_off),
                    "llm-synthfailed", None, placed[anchor_idx][1])
        a_start, a_end = placed[anchor_idx]
        # refine で挿入された相槌: アンカー発話内の相対位置(rel_pos)を中心に探索
        hint = (t.get("refine") or {}).get("rel_pos")
        if hint is not None:
            center = a_start + float(hint) * (a_end - a_start)
            w_lo = max(a_start + b.min_into_turn, center - b.insert_window_sec)
            w_hi = min(a_end + b.search_after, center + b.insert_window_sec)
            method = "bc-inserted"
        else:
            w_lo = max(a_start + b.min_into_turn, a_end - b.search_before)
            w_hi = a_end + b.search_after
            method = "bc"
        # 長い相槌の食い込み制限: 相槌が max_overlap 秒より長いなら、相手発話との
        # 重なりが max_overlap 秒以下になる位置（アンカー末 - max_overlap 以降）に限定。
        # 短い相槌（うん・はい等）は従来どおりターン中の任意位置に置ける。
        if dur > b.max_overlap:
            lo = a_end - b.max_overlap
            if lo > w_lo:
                w_lo = lo
                w_hi = max(w_hi, a_end + b.search_after)  # rel_pos 窓が退化しないよう確保
        if w_lo >= w_hi:  # 窓が退化（アンカーが極端に短い等）
            return (self._fallback_start(prev_placed_end, llm_off),
                    "fallback-llm", None, a_end)
        feed_start = max(0.0, w_lo - b.context_len_sec)
        x0, x1 = timeline.slice(feed_start, w_hi)
        # bc の規約: ch1=聞かれている側（話し手）, ch2=相槌を打つ側
        sp_ch = channel_of[sp]
        speaker_ch = 1 - sp_ch
        xs = (x0, x1) if speaker_ch == 0 else (x1, x0)
        scan_from = max(0, int((w_lo - feed_start) * MAAI_SR)
                        - MAAI_SR // int(self.cfg.frame_rate))
        frames = self._bc_scorer().score(*xs, scan_from)

        best_t, best_p = None, -1.0        # 窓全体のピーク
        best_in_t, best_in_p = None, -1.0  # 相手発話の内側（t_k <= a_end）のピーク
        run_len = 0
        for end_sample, fr in frames:
            t_k = feed_start + end_sample / MAAI_SR
            if t_k < w_lo:
                continue
            if t_k > w_hi:
                break
            p = self._bc_signal(fr)
            if p is None:
                continue
            if b.strategy == "first_crossing":
                if p >= b.threshold:
                    run_len += 1
                    if run_len >= 2:
                        tc = t_k - (run_len - 1) * frame_dur
                        return tc, method, p, a_end
                else:
                    run_len = 0
            else:
                if p > best_p:
                    best_t, best_p = t_k, p
                if t_k <= a_end and p > best_in_p:
                    best_in_t, best_in_p = t_k, p
        if b.strategy == "peak":
            # 「相手の発話の合間に打つ」感を出すため、相手発話の内側に閾値超えの
            # ピークがあればそちらを優先する（発話後の無音区間のピークは次点）。
            if best_in_t is not None and best_in_p >= b.threshold:
                return best_in_t, method, best_in_p, a_end
            if best_t is not None and best_p >= b.threshold:
                return best_t, method, best_p, a_end
        return (self._fallback_start(prev_placed_end, llm_off),
                "fallback-llm", None, a_end)

    def log_summary(self):
        self.log_timing()
        total = sum(self.method_counts.values())
        if not total:
            return
        parts = ", ".join(f"{k}={v}" for k, v in
                          sorted(self.method_counts.items(), key=lambda x: -x[1]))
        fb = sum(v for k, v in self.method_counts.items() if k.startswith("fallback"))
        logger.info(f"リタイム集計: 全{total}ターン（{parts}） "
                    f"フォールバック率 {fb / total:.1%}")

    def log_timing(self):
        """スコアラの実測を出す（backend=maai/vapbatch のどちらでも）。"""
        for label, sc in (("vap", self._vap), ("bc", self._bc)):
            if sc is not None:
                logger.info(f"スコアラ実測[{self.cfg.backend}/{label}]: "
                            f"{sc.stats.summary()}")


# --------------------------------------------------------------------------
# MockRetimer: --dry-run 用（maai 不要）
# --------------------------------------------------------------------------

def _next_speech_has_pause(turns: list, i: int) -> bool:
    """turns[i] の後ろで最初の speech ターンが長い沈黙のあと（pause_before_sec）か。"""
    for t in turns[i + 1:]:
        if t.get("type", "speech") != "backchannel":
            return t.get("pause_before_sec") is not None
    return False


class MockRetimer:
    """--dry-run 用のダミープランナー。正弦波に MaAI を掛けても無意味なので、
    クランプ済み LLM オフセットで配置し、配線・manifest・出力分離だけ検証する。
    """

    def __init__(self, cfg: RetimeConfig):
        self.cfg = cfg
        self.method_counts: dict[str, int] = {}

    def plan(self, record, turns, wav_by_turn, sr, channel_of,
             failed_turns=None) -> list[dict] | None:
        if len(channel_of) != 2:
            return None
        v = self.cfg.vap
        plans = []
        prev_end = 0.0
        last_end_on_channel = {0: 0.0, 1: 0.0}
        for i, t in enumerate(turns):
            ch = channel_of[t["speaker"]]
            dur = len(wav_by_turn[i]) / sr
            off = 0.0 if i == 0 else float(t.get("start_offset", 0.0) or 0.0)
            if t.get("pause_before_sec") is not None and i > 0:
                off = float(t["pause_before_sec"])      # 長い沈黙は max_gap で切らない
            else:
                off = min(max(off, -v.max_overlap), v.max_gap)
            start = max(0.0, prev_end + off)
            clamped = False
            if start < last_end_on_channel[ch] - 1e-6:
                start, clamped = last_end_on_channel[ch], True
            plans.append({
                "index": i,
                "start_sec": round(start, 3),
                "retime": {"method": "mock", "anchor_sec": round(prev_end, 3),
                           "llm_start_sec": round(max(0.0, prev_end + off), 3),
                           "decided_offset": round(start - prev_end, 3),
                           "p_peak": None, "clamped": clamped},
            })
            self.method_counts["mock"] = self.method_counts.get("mock", 0) + 1
            end = start + dur
            prev_end = end
            last_end_on_channel[ch] = max(last_end_on_channel[ch], end)
        return plans

    def log_summary(self):
        logger.info(f"リタイム集計(mock): {self.method_counts.get('mock', 0)}ターン")
