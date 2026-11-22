"""合成音声の話者を x-vector（WavLM）で照合する共通部品（generate_tts の取り違え対策と check_speaker_leak.py）。

fish の2話者まとめての生成（tts.fish_speech.multiturn）や、相手の文脈つきの再合成では、
ある発話の声が相手に入れ替わることがある（2026-10-05 cmp_mt 20対話で 667発話中2件）。
発話の x-vector と各話者の参照音声の x-vector のコサイン類似度を比べ、本人より相手に
近い発話を見つける。

実測（cmp_mt / cmp_pt・約1,300発話）: 正常な発話の本人との類似度は平均 0.95。はっきりした
取り違えは 本人 0.66〜0.83・相手 0.93〜0.97。男性同士など声の近い組では本人と相手の差が
0.01〜0.03 しかない発話が多く、差が小さいものは取り違えとは見なさない（判定は own_max と margin）。

依存: torch / transformers / librosa。無ければ get_verifier が None を返す（呼び出し側は検査を飛ばす）。
"""
from __future__ import annotations

import functools
import threading
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

SV_SR = 16000


class Verifier:
    def __init__(self, model: str, device: str):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.fe = AutoFeatureExtractor.from_pretrained(model)
        self.m = WavLMForXVector.from_pretrained(model).to(self.device).eval()

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        w = np.asarray(wav, dtype=np.float32)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != SV_SR:
            w = librosa.resample(w, orig_sr=sr, target_sr=SV_SR)
        x = self.fe(w, sampling_rate=SV_SR, return_tensors="pt")
        x = {k: v.to(self.device) for k, v in x.items()}
        with self.torch.no_grad():
            e = self.m(**x).embeddings[0]
        e = self.torch.nn.functional.normalize(e, dim=-1)
        return e.detach().cpu().numpy()

    def embed_batch(self, wavs: list, sr: int, batch: int = 16, max_ratio: float = 1.10) -> list:
        """embed をまとめて（長さの違いは詰め物＋attention_mask）。1 本ずつだと WavLM の相対位置の計算が呼ぶたびに走り、
        テイク選択の採点の 1/4 を占めていた（2026-11-14）。"""
        import librosa
        xs = []
        for w in wavs:
            w = np.asarray(w, dtype=np.float32)
            if w.ndim > 1:
                w = w.mean(axis=1)
            xs.append(librosa.resample(w, orig_sr=sr, target_sr=SV_SR) if sr != SV_SR else w)
        # 長さの近いものだけを束ねる（詰め物が多いと WavLM の正規化がずれて類似度が最大 0.05 動く）。
        # 長さ順に並べ、束の最長が最短の max_ratio 倍以内に収まるところで区切る。
        order = sorted(range(len(xs)), key=lambda k: len(xs[k]))
        out: list = [None] * len(xs)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < batch and len(xs[order[j]]) <= max_ratio * len(xs[order[i]]):
                j += 1
            idx = order[i:j]
            ch = [xs[k] for k in idx]
            x = self.fe(ch, sampling_rate=SV_SR, return_tensors="pt", padding=True, return_attention_mask=True)
            x = {k: v.to(self.device) for k, v in x.items()}
            with self.torch.no_grad():
                e = self.m(**x).embeddings
            e = self.torch.nn.functional.normalize(e, dim=-1).detach().cpu().numpy()
            for k, v in zip(idx, e):
                out[k] = v
            i = j
        return out

    def embed_bytes(self, audio: bytes) -> np.ndarray:
        import soundfile as sf
        w, sr = sf.read(io.BytesIO(audio), dtype="float32")
        return self.embed(w, sr)


_VER_LOCK = threading.Lock()
_VER: dict = {}


def get_verifier(model: str = "microsoft/wavlm-base-plus-sv", device: str = "cuda"):
    """プロセスで1回だけ読み込む。依存が無い・モデルを取得できないなら None。
    ★2026-11-15 ロックを掛けて 1 スレッドずつ読む: pocket の文脈つき合成は 2 人の列を並行に回し、それぞれが声の確認で同時に
    ここを呼ぶ。transformers の遅延 import が同時に走ると片方が「cannot import name 'AutoFeatureExtractor'」で失敗し、
    以前の lru_cache はその None を覚えて、そのワーカーでは最後まで声の確認が切れていた（本番 36 本全部）。"""
    key = (model, device)
    with _VER_LOCK:
        if key not in _VER:
            try:
                _VER[key] = Verifier(model, device)
            except Exception as e:
                logger.warning(f"話者の照合を無効化（モデルを用意できない）: {e}")
                _VER[key] = None
        return _VER[key]


get_verifier.cache_clear = _VER.clear      # 以前の lru_cache と同じ呼び方（テスト用）


def is_leak(own: float, other: float, own_max: float, margin: float) -> bool:
    """本人との類似度が own_max 未満で、相手のほうが margin 以上近いなら取り違え。"""
    return own < own_max and other - own >= margin
