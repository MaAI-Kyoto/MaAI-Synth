"""MaaiBatchScorer — MaAI 公式の一括推論 MaaiBatch を使うスコアラ（backend="maaibatch"）

★2026-11-22 追加。MaAI 公式リポジトリ（MaAI-Kyoto/MaAI, readme/batch.md）に入った
`maai.MaaiBatch` を、`MaaiScorer` / `BatchVapScorer` と同じ入出力契約で包む。
自前の vapbatch（retime_vapbatch.py）と中身はほぼ同じ設計
（PyTorch Mimi を状態を持ち越しながらチャンクで回す → 帯マスクつき AR を窓まるごと1回）で、
違いは:
  - 重みを MaAI と同じ経路（HF の maai-kyoto/*）で読む。export_state_dicts.py が要らず、
    MaAI 側でモデルが更新されればそのまま使える。
  - 雑音頑健版 VAP（vap_mc）を選べる（tts.retime.maaibatch.mc）。
  - 先頭フレーム落とし・24kHz 先頭ゼロ・BC のチャネル入れ替え・帯マスク・長尺の折りたたみは
    MaaiBatch の内部で閉じている（retime_vapbatch.py の docstring の (a)(b)(c) に相当）。

★maai はリポジトリの src から import する（pip は 0.2.11 のまま触らない）。
  新しい Maai は mode="vap" の既定が mc=True（雑音頑健版）に変わっていて、pip を上げると
  backend=maai を使う config の挙動が黙って変わるため。同じプロセスで pip 版の maai が
  先に import されていたら混ざるので、その場合は起動時に落とす。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from retime_maai import MAAI_SR, ScorerStats

logger = logging.getLogger(__name__)

DEFAULT_REPO_SRC = str(Path(__file__).resolve().parent / "MaAI" / "src")


def _stub_pyaudio_if_missing() -> None:
    """pyaudio が読めない機械（libportaudio.so.2 が無い node18・node25。2026-11-22 時点）では空の代役を入れる。

    maai/input.py がモジュールの先頭で pyaudio を import するので、無いと `import maai` ごと落ちる。
    pyaudio を使うのはマイク入力の関数の中だけで、MaaiBatch（録音済み音声の一括推論）は一切使わない。
    """
    try:
        import pyaudio  # noqa: F401
        return
    except Exception as e:      # ImportError（libportaudio.so.2 が無い）など
        reason = f"{type(e).__name__}: {e}"
    import types

    def _unavailable(*_a, **_k):
        raise RuntimeError(f"pyaudio はこの機械では使えません（{reason}）。MaaiBatch には不要")

    stub = types.ModuleType("pyaudio")
    stub.PyAudio = _unavailable
    stub.paFloat32, stub.paInt16 = 1, 8
    stub.__maai_stub__ = True
    sys.modules["pyaudio"] = stub
    logger.info(f"pyaudio が読めないので代役を入れて maai を import します（{reason}）")


def import_maai_from_repo(repo_src: str):
    """repo_src/maai を import して返す。別の場所の maai が既に import 済みなら落とす。"""
    src = Path(repo_src).resolve()
    if not (src / "maai" / "batch.py").exists():
        raise FileNotFoundError(
            f"MaAI リポジトリが見つかりません: {src}/maai/batch.py "
            "（git clone https://github.com/MaAI-Kyoto/MaAI.git して "
            "tts.retime.maaibatch.repo_src に <clone>/src を書く）")
    loaded = sys.modules.get("maai")
    if loaded is not None:
        where = Path(getattr(loaded, "__file__", "") or "").resolve()
        if src not in where.parents:
            raise RuntimeError(
                f"maai が別の場所から既に import されています: {where}。"
                f"backend=maaibatch は {src} の maai を使うので、同じプロセスで "
                "backend=maai と混ぜないでください")
        return loaded
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _stub_pyaudio_if_missing()
    import maai
    where = Path(maai.__file__).resolve()
    if src not in where.parents:
        raise RuntimeError(f"maai の import 先が想定と違います: {where}（期待: {src}）")
    return maai


def maai_repo_commit(repo_src: str) -> str | None:
    """manifest に残すリポジトリの commit（.git が無ければ None）。"""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", str(repo_src), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


class MaaiBatchScorer:
    """`MaaiScorer` と同じ入出力契約を持つ、MaaiBatch のラッパ。

    score(ch1_16k, ch2_16k, scan_from_sample=0) -> [(end_sample, result_dict)]
      mode="vap" → {"p_now": [a,b], "p_future": [a,b], "vad": [a,b]}
      mode="bc"  → {"p_bc": float, "p_bc_detect": float}
    引数の順は MaaiScorer と同じ（Maai.process(audio1, audio2) に渡す順）。
    BC のチャネル入れ替えは MaaiBatch が Maai と同じ規則で内部でやる。
    """

    def __init__(self, mode: str, lang: str, frame_rate: float,
                 context_len_sec: float, device: str = "cuda", *,
                 model_type: str = "normal-ver2", repo_src: str = DEFAULT_REPO_SRC, mc: bool = False,
                 batch_size: int = 8, encoder_chunk_frames: int = 300,
                 local_model: str | None = None, cache_dir: str | None = None,
                 torch_num_threads: int | None = None):
        import torch
        maai = import_maai_from_repo(repo_src)

        if torch_num_threads:
            torch.set_num_threads(int(torch_num_threads))
        # TF32 は相槌の挿入位置を動かす（config_main_v1.yaml）。vapbatch と同じく明示的に切る。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True

        # 旧来の mode 名 "vap_mc" は新しい MaAI では廃止（mode="vap", mc=True）。
        if mode == "vap_mc":
            mode, mc = "vap", True
        if mode not in ("vap", "bc", "bc_2type"):
            raise ValueError(f"backend=maaibatch は mode=vap / vap_mc / bc / bc_2type のみ（受領: {mode!r}）")

        self.mode = mode
        self.mc = bool(mc) if mode == "vap" else False
        self.device = device
        self.repo_src = str(Path(repo_src).resolve())
        self.stats = ScorerStats()
        self.frame_rate = float(frame_rate)
        self.chunk = int(round(MAAI_SR / self.frame_rate))   # 1280 @12.5Hz

        t0 = time.perf_counter()
        self.batch = maai.MaaiBatch(
            mode=mode, lang=lang, frame_rate=self.frame_rate,
            context_len_sec=context_len_sec, device=device,
            model_type=model_type, mc=self.mc, local_model=local_model,
            cache_dir=cache_dir, batch_size=int(batch_size),
            encoder_chunk_frames=int(encoder_chunk_frames))
        self.variant = self.batch.models[0].variant      # 'vap' / 'vap_mc' / 'bc'
        if mode == "vap" and self.mc and "mc" not in self.variant:
            raise RuntimeError(f"mc=True なのに雑音頑健版が読めませんでした（variant={self.variant}）")
        self.stats.load_seconds = time.perf_counter() - t0
        logger.info(
            f"maaibatch({mode}) ロード完了: {self.stats.load_seconds:.1f}s "
            f"(variant={self.variant}, ctx={context_len_sec}s, "
            f"fold_context={self.batch.fold_context_frames}, repo={self.repo_src})")

    # ------------------------------------------------------------------

    def _pad(self, a: np.ndarray, n_pad: int) -> np.ndarray:
        x = np.zeros(n_pad, dtype=np.float32)
        x[:len(a)] = a
        return x

    def _to_frames(self, res: dict) -> list[tuple[int, dict]]:
        es = res["end_sample"]
        keys = [k for k, v in res.items()
                if k not in ("end_sample", "time_sec") and isinstance(v, np.ndarray)]
        cols = {k: res[k].tolist() for k in keys}
        return [(int(es[t]), {k: cols[k][t] for k in keys}) for t in range(len(es))]

    def _sync(self):
        import torch
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 公開API（MaaiScorer 互換）
    # ------------------------------------------------------------------

    def score(self, ch1_16k: np.ndarray, ch2_16k: np.ndarray,
              scan_from_sample: int = 0) -> list[tuple[int, dict]]:
        """2本を等長にゼロ埋めしてスコアする（MaaiScorer と同じく末尾は1フレームに切り上げ）。

        MaaiBatch は不完全な末尾フレームを捨てるので、MaaiScorer と同じフレーム数に
        なるようにこちらで切り上げてから渡す。長い入力は MaaiBatch が折りたたむ。
        scan_from_sample は MaaiScorer と同じく受け取って無視する。
        """
        return self.score_long_many([(ch1_16k, ch2_16k)])[0]

    def score_long(self, ch1_16k: np.ndarray, ch2_16k: np.ndarray,
                   scan_from_sample: int = 0) -> list[tuple[int, dict]]:
        return self.score_long_many([(ch1_16k, ch2_16k)])[0]

    def score_long_many(self, pairs) -> list[list[tuple[int, dict]]]:
        """複数の (ch1, ch2) をまとめてスコアする（refine の InsertionScanner が2方向で呼ぶ）。"""
        t0 = time.perf_counter()
        items, n_frames = [], 0
        for a, b in pairs:
            n = max(len(a), len(b))
            n_pad = ((n + self.chunk - 1) // self.chunk) * self.chunk
            items.append((self._pad(a, n_pad), self._pad(b, n_pad)))
            n_frames += n_pad // self.chunk
        results = self.batch.process_many(items)
        self._sync()
        out = [self._to_frames(r) for r in results]
        self.stats.add(n_frames, time.perf_counter() - t0)
        return out
