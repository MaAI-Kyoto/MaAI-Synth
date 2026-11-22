# Helper module loaded by the vLLM-Omni Irodori pipeline (config.json "irodori.helpers" points to this directory).
# It is the same file as irodori_fast.py at the repository root; it lives in its own directory so that the
# TTS server does not put the whole repository on sys.path.
"""Irodori-TTS を速く・同じ結果のまま回すための部品（2026-11-14）。ext_tts_server.py と vLLM-Omni の
IrodoriPipeline（vllm-omni-irodori/vllm_omni/diffusion/models/irodori/）の両方から使う。

- install_ref_latent_cache(runtime): 参照音声 → DACVAE 潜在 の符号化を (ファイル・mtime・設定) で使い回す。
  上流の InferenceRuntime は毎リクエスト参照を符号化し直す（60秒の参照なら 1発話ごとに 60秒ぶんの encoder）。
  同じ入力なら同じ潜在（codec_deterministic_encode=True）なので結果は変わらない。
- RefIndex: 受け取った参照音声（バイト列）から、バンク上の同じファイルと同じ話者の兄弟テイク（*_t<N>.wav）を探す。
- seconds_for(text, mora_rate): 「モーラ数 ÷ mora_rate ÷ 0.94」秒（generate_tts.IrodoriSynth の mode=mora と同じ式）。
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger("irodori_fast")


def install_ref_latent_cache(runtime, max_items: int = 512) -> None:
    """runtime._encode_reference_wavs を LRU キャッシュ付きに差し替える（同じ入力なら同じ潜在を返す）。"""
    if getattr(runtime, "_ref_cache_installed", False):
        return
    orig = runtime._encode_reference_wavs
    cache: OrderedDict = OrderedDict()
    lock = threading.Lock()

    def cached(*, req, wav_paths, max_ref_seconds, max_ref_latent_steps):
        key = (tuple((p, os.path.getmtime(p)) for p in wav_paths), float(max_ref_seconds), max_ref_latent_steps,
               req.ref_normalize_db, bool(req.ref_ensure_max))
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit[0].clone(), list(hit[1])
        lat, msgs = orig(req=req, wav_paths=wav_paths, max_ref_seconds=max_ref_seconds,
                         max_ref_latent_steps=max_ref_latent_steps)
        with lock:
            cache[key] = (lat.clone(), list(msgs))
            while len(cache) > max_items:
                cache.popitem(last=False)
        return lat, msgs

    runtime._encode_reference_wavs = cached
    runtime._ref_cache_installed = True


class RefIndex:
    """受け取った参照音声の中身から、バンク上のファイルと同じ話者の兄弟テイクを探す。"""

    def __init__(self, dirs: list[str]):
        self.by_size: dict = {}
        for d in dirs:
            for p in Path(d).glob("*.wav"):
                self.by_size.setdefault(p.stat().st_size, []).append(p)
        self._h: dict = {}
        self._lock = threading.Lock()
        self.tmp = Path(tempfile.mkdtemp(prefix="irodori_ref_"))
        log.info(f"参照の照合用に {sum(len(v) for v in self.by_size.values())} 本を登録")

    def _sha(self, p: Path) -> str:
        if p not in self._h:
            self._h[p] = hashlib.sha1(p.read_bytes()).hexdigest()
        return self._h[p]

    def find(self, b: bytes) -> Path | None:
        h = hashlib.sha1(b).hexdigest()
        with self._lock:
            return next((p for p in self.by_size.get(len(b), []) if self._sha(p) == h), None)

    def paths(self, b: bytes, multi: bool) -> list[str]:
        """multi なら [本人のファイル, 兄弟テイク...]。バンクに無ければ一時ファイル 1 本。"""
        hit = self.find(b)
        if hit is None:
            f = self.tmp / f"{hashlib.sha1(b).hexdigest()}.wav"
            if not f.exists():
                f.write_bytes(b)
            return [str(f)]
        if not multi:
            return [str(hit)]
        return [str(hit)] + siblings(hit)


def siblings(p: Path) -> list[str]:
    stem = p.stem.rsplit("_t", 1)[0]
    return [str(x) for x in sorted(p.parent.glob(stem + "_t*.wav")) if x != p]


def seconds_for(text: str, mora_rate: float, lo: float = 0.5, hi: float = 30.0) -> float:
    from generate_tts import estimate_mora
    return float(min(hi, max(lo, estimate_mora(text)[0] / mora_rate / 0.94)))
