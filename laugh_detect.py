"""笑いらしさ（AudioSet で学習した AST 分類器の「笑い」系クラスの確率）。2026-11-14。
generate_tts の best_of（w_laugh）が、笑いで始まる発話のテイクを選ぶのに使う。人の笑い（maai-f2f の笑い IPU）でも平均 0.09・
普通の発話 0.006 程度と値は小さいが、順位付けには使える（_moss_tune/laughness.py・laugh_sel_test.py）。
"""
import threading

import numpy as np

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
LAUGH = ("Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle")
_inst = None
_lock = threading.Lock()


class Laughness:
    def __init__(self, device="cuda"):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(MODEL)
        self.m = ASTForAudioClassification.from_pretrained(MODEL).to(device).eval()
        self.dev = device
        self.idx = [i for i, n in self.m.config.id2label.items() if n in LAUGH]

    def head(self, w, sr, head=1.5, win=1.0, hop=0.1) -> float:
        """先頭 head 秒の中の 1 秒窓（0.1 秒刻み）での笑い確率の最大。"""
        import librosa
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
        L = int(win * 16000)
        w = w[: max(L, int(head * 16000))]
        if len(w) < L:
            w = np.pad(w, (0, L - len(w)))
        segs = [w[s:s + L] for s in range(0, len(w) - L + 1, int(hop * 16000))] or [w[:L]]
        x = self.fe(segs, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.m(x).logits).cpu().numpy()
        return float(p[:, self.idx].max())


def get(device="cuda") -> Laughness:
    global _inst
    with _lock:
        if _inst is None:
            _inst = Laughness(device)
        return _inst
