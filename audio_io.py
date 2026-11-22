"""対話単位の音声ファイル（audio/<variant>/<id>.wav|.flac・audio_raw）の形式をそろえる小道具（2026-11-17）。

config の tts.audio_format（wav / flac・既定 wav）で書く形式を選ぶ。読む側・「出来ているか」の判定は
どちらの拡張子でもよい（途中で形式を変えても二重に作らない）。FLAC は可逆で、1 対話の大きさは wav の約 1/4
（24kHz・2ch・Sidon 後の実測 14.1→3.6MB）。中身は wav と同じ PCM16 の値（いったん wav と同じ量子化をしてから書く）。
発話単位のキャッシュ（turns/）はこれまでどおり wav。重い import をしないこと（.venv-fish・ファイルサーバ からも使う）。
"""
from __future__ import annotations

import io
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac")


def audio_ext(tts_cfg: dict | None) -> str:
    """書く拡張子（".wav" / ".flac"）。"""
    f = str((tts_cfg or {}).get("audio_format") or "wav").lower().lstrip(".")
    if f not in ("wav", "flac"):
        raise ValueError(f"tts.audio_format は wav か flac: {f}")
    return "." + f


def is_audio_name(name: str) -> bool:
    return name.endswith(AUDIO_EXTS)


def find_audio(d: Path, did: str) -> Path | None:
    """<d>/<did>.wav か .flac のうち、ある方（両方あれば wav）。"""
    for e in AUDIO_EXTS:
        p = Path(d) / f"{did}{e}"
        if p.exists():
            return p
    return None


def has_audio(d: Path, did: str) -> bool:
    return find_audio(d, did) is not None


def audio_index(d: Path) -> dict[str, Path]:
    """<d> を 1 回だけ読んで {id: 音声のパス}（find_audio と同じ選び方: 両方あれば wav・無い dir は空）。
    1 万対話を 1 本ずつ find_audio すると、NFS 越しで ファイルサーバ の回線が混んでいるとき 1 回 0.1 秒・flac は wav を探して
    外してから見るので 2 回 → 1 dir 約 35 分かかる（2026-11-17・audit_pipeline）。多くの id を見るときはこちら。"""
    import os
    d = Path(d)
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return {}
    out: dict[str, Path] = {}
    for n in names:
        stem, ext = os.path.splitext(n)
        if ext in AUDIO_EXTS and (stem not in out or ext == AUDIO_EXTS[0]):
            out[stem] = d / n
    return out


def pcm16_bytes(data, sr: int) -> bytes:
    """今まで wav に書いていたのと同じ PCM16 の wav のバイト列。"""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_audio(path: Path, data, sr: int, fmt: str | None = None) -> None:
    """data（float / int16）を path に PCM16 で書く。形式は fmt（"WAV"/"FLAC"）か path の拡張子。
    FLAC も wav と同じ量子化（wav に書いて読み戻した int16）なので、wav で作ったものと値が一致する。"""
    import soundfile as sf
    fmt = (fmt or Path(path).suffix.lstrip(".")).upper()
    if fmt == "WAV":
        sf.write(str(path), data, sr, format="WAV", subtype="PCM_16")
        return
    write_wav_bytes_as(path, pcm16_bytes(data, sr), fmt)


def write_wav_bytes_as(path: Path, wav_bytes: bytes, fmt: str | None = None) -> None:
    """PCM16 の wav のバイト列を、path の形式（fmt）で書く（wav ならそのまま・flac なら可逆に詰め直す）。"""
    import soundfile as sf
    fmt = (fmt or Path(path).suffix.lstrip(".")).upper()
    if fmt == "WAV":
        Path(path).write_bytes(wav_bytes)
        return
    pcm, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    sf.write(str(path), pcm, sr, format=fmt, subtype="PCM_16")
