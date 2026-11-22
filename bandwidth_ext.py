"""帯域拡張（2026-11-13）: 24kHz で生成する TTS（MOSS-TTS-v1.5 の codec は 24kHz＝12kHz より上が無い）の
出力を 48kHz にし、12〜24kHz を信号処理で補う。

やり方（生成型の復元はしない＝声そのものは作り直さない）:
  1. 48kHz にアップサンプル。
  2. 元の上側の帯域（src_lo〜12kHz）を取り出し、単側波帯変調で shift Hz だけ上へずらす（12kHz〜）。
  3. ずらした成分に、上へ行くほど下がる傾き（tilt_db_per_oct）を掛け、
     「12〜16kHz が 8〜12kHz より gap_db 小さい」になるよう全体の量を合わせて足す。
既定の gap_db 8.4 は人間の参照音声（ボイスバンクの録音）で測った値。他の TTS の出力は使っていない。
短時間フーリエ変換の上で、フレームごとに元の 8〜12kHz のエネルギーに比例させる（無音・子音の時間形を保つ）。

usage（試聴・評価用）: python bandwidth_ext.py in.wav out.wav
"""
from __future__ import annotations

import numpy as np


def extend(w: np.ndarray, sr: int, out_sr: int = 48000, src_lo: float = 6000.0, top: float = 12000.0,
           gap_db: float = 8.4, tilt_db_per_oct: float = 6.0, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, int]:
    """w（モノラル float）を out_sr にし、top Hz より上を補って返す。sr が top*2 を十分超えていれば何もしない。"""
    import scipy.signal as ss
    w = np.asarray(w if w.ndim == 1 else w.mean(axis=1), dtype=np.float64)
    if sr != out_sr:
        from math import gcd
        g = gcd(int(sr), int(out_sr))
        w = ss.resample_poly(w, out_sr // g, sr // g)
    if sr >= 2 * top * 1.3:          # 元から帯域がある
        return w.astype(np.float32), out_sr
    f, t, Z = ss.stft(w, fs=out_sr, nperseg=n_fft, noverlap=n_fft - hop)
    shift = top - src_lo                      # src_lo〜top を top〜top+shift へ
    k_shift = int(round(shift / (f[1] - f[0])))
    src = (f >= src_lo) & (f < top)
    ref_band = (f >= top - 4000) & (f < top)  # 8〜12kHz（量の基準）
    H = np.zeros_like(Z)
    idx = np.nonzero(src)[0]
    dst = idx + k_shift
    ok = dst < len(f)
    H[dst[ok], :] = Z[idx[ok], :]
    # 上へ行くほど下げる（オクターブあたり tilt_db）
    fh = np.maximum(f, top)
    tilt = 10 ** (-tilt_db_per_oct * np.log2(fh / top) / 20)
    H *= tilt[:, None]
    H[f < top, :] = 0
    # 量の合わせ込み: 12〜16kHz のエネルギーを、元の 8〜12kHz より gap_db 小さく（全体で一律）
    e_ref = np.sum(np.abs(Z[ref_band]) ** 2)
    hb = (f >= top) & (f < top + 4000)
    e_h = np.sum(np.abs(H[hb]) ** 2)
    if e_h > 0 and e_ref > 0:
        H *= np.sqrt(e_ref / e_h * 10 ** (-gap_db / 10))
    # 元の帯域の上端付近の不連続を避けるため、元は top のすぐ下まで残し、補った帯は top から立ち上げる
    ramp = np.clip((f - top) / 300.0, 0, 1)
    H *= ramp[:, None]
    _, y = ss.istft(Z + H, fs=out_sr, nperseg=n_fft, noverlap=n_fft - hop)
    y = y[: len(w)]
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    return y.astype(np.float32), out_sr


if __name__ == "__main__":
    import sys
    import soundfile as sf
    x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
    y, osr = extend(x.mean(1), sr)
    sf.write(sys.argv[2], y, osr)
