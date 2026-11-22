#!/usr/bin/env python3
"""Make a voice bank (voices/<bank>/voices.csv + voices/<bank>/wav/*.wav) from recordings of ONE speaker.

    python tools/make_voice_bank.py --bank alice --gender female --age 20代 rec1.wav rec2.wav
    python tools/make_voice_bank.py --bank alice --gender female --age 20代 --sidon --asr rec*.wav

Steps
  1. energy-based voice activity detection on each recording (relative to its noise floor)
  2. pack consecutive speech regions into takes of --min-sec..--max-sec seconds (target --target-sec, default 30 s),
     joined with --gap seconds of silence. Takes are named <bank>_t1.wav, <bank>_t2.wav, ...: the Irodori TTS
     method uses all takes of the same speaker (same prefix) as one ~60 s reference.
  3. optional speech restoration with Sidon (--sidon; sarulab-speech/sidon-v0.1, output 48 kHz)
  4. loudness: peak -1 dBFS, speech RMS at most -23 dBFS
  5. optional transcription with kotoba-whisper (--asr). The text column is not used by the default TTS method
     (Irodori does not need the reference transcript), but other engines may need it.
  6. write voices/<bank>/voices.csv (id, gender, age_range, audio_path, text, restore, speech_db)

Input: mono or stereo WAV/FLAC of a single speaker (a conversation track is fine if the other person is not
audible). Use only recordings you have the right to use, with the speaker's consent.
Then add the CSV to a config: tts.voices_file (user side) or tts.main_voices_file (system side).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.02  # seconds


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * FRAME)
    m = len(x) // n
    r = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(r)


def vad(x: np.ndarray, sr: int, hi_db: float, lo_db: float, min_seg: float, merge: float) -> list[tuple[float, float]]:
    """Speech regions [(start, end)] in seconds: frames above floor+lo_db that contain a core above floor+hi_db."""
    db = frame_db(x, sr)
    if not len(db):
        return []
    floor = float(np.percentile(db, 10))
    above = db > floor + lo_db
    core = db > floor + hi_db
    segs, i = [], 0
    while i < len(db):
        if above[i]:
            j = i
            while j < len(db) and above[j]:
                j += 1
            if core[i:j].sum() >= 3:
                segs.append([i * FRAME, j * FRAME])
            i = j
        else:
            i += 1
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(max(0.0, s - 0.05), e + 0.05) for s, e in merged if e - s >= min_seg]


def pack(regions: list[tuple[float, float]], target: float, lo: float, hi: float, gap: float):
    """Group consecutive regions into takes whose total length (with gaps) is in [lo, hi], close to target."""
    takes, cur, cur_len = [], [], 0.0
    for s, e in regions:
        d = e - s
        if d > hi:                       # very long region: cut it into target-sized pieces
            t = s
            while e - t > lo:
                takes.append([(t, min(e, t + target))])
                t += target
            continue
        add = d + (gap if cur else 0.0)
        if cur and cur_len + add > hi:
            if cur_len >= lo:
                takes.append(cur)
            cur, cur_len = [], 0.0
            add = d
        cur.append((s, e))
        cur_len += add
        if cur_len >= target:
            takes.append(cur)
            cur, cur_len = [], 0.0
    if cur and cur_len >= lo:
        takes.append(cur)
    return takes


def speech_level(x: np.ndarray, sr: int) -> float | None:
    db = frame_db(x, sr)
    if len(db) < 10:
        return None
    lin = 10 ** (db / 20)
    loud = lin[lin >= 0.25 * lin.max()]
    return float(20 * np.log10(np.sqrt((loud ** 2).mean()) + 1e-12)) if len(loud) else None


def normalize(x: np.ndarray, sr: int, peak_dbfs: float = -1.0, speech_dbfs: float = -23.0):
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 0:
        x = x * (10 ** (peak_dbfs / 20) / peak)
    sp = speech_level(x, sr)
    if sp is not None and sp > speech_dbfs:
        x = x * 10 ** ((speech_dbfs - sp) / 20)
        sp = speech_dbfs
    return x.astype(np.float32), sp


class Sidon:
    """Speech restoration (sarulab-speech/sidon-v0.1; same steps as the official demo). Output: 48 kHz."""

    def __init__(self, device: str | None = None):
        import torch
        import torchaudio
        import transformers
        from huggingface_hub import hf_hub_download
        self.torch, self.ta = torch, torchaudio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tag = "cuda" if self.device.startswith("cuda") else "cpu"
        repo = "sarulab-speech/sidon-v0.1"
        self.fe = torch.jit.load(hf_hub_download(repo, f"feature_extractor_{tag}.pt"), map_location=self.device)
        self.de = torch.jit.load(hf_hub_download(repo, f"decoder_{tag}.pt"), map_location=self.device)
        self.pp = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def restore(self, wav: np.ndarray, sr: int):
        torch = self.torch
        with torch.inference_mode():
            w = 0.9 * (wav / (np.abs(wav).max() + 1e-12))
            target_n = int(48000 / sr * len(w))
            t = torch.tensor(w, dtype=torch.float32).view(1, -1)
            t = self.ta.functional.highpass_biquad(t, sr, 50)
            t16 = torch.nn.functional.pad(self.ta.functional.resample(t, sr, 16000), (0, 24000))
            outs, cache = [], None
            for chunk in t16.view(-1).split(16000 * 96):
                inp = self.pp(torch.nn.functional.pad(chunk, (160, 160)), sampling_rate=16000, return_tensors="pt")
                f = self.fe(inp["input_features"].to(self.device))["last_hidden_state"]
                if cache is not None:
                    f = torch.cat([cache, f], dim=1)
                outs.append(self.de(f.transpose(1, 2)).view(-1)[:-960])
                cache = f[:, -1:]
            y = torch.cat(outs, 0).float().cpu().numpy()[:target_n]
        return y.astype(np.float32), 48000


def transcriber(device: str | None):
    import torch
    from transformers import pipeline
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    asr = pipeline("automatic-speech-recognition", model="kotoba-tech/kotoba-whisper-v2.0", device=dev,
                   torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32)

    def run(x: np.ndarray, sr: int) -> str:
        if sr != 16000:
            import soxr
            x = soxr.resample(x, sr, 16000)
        out = asr({"raw": x, "sampling_rate": 16000}, return_timestamps=True,
                  generate_kwargs={"language": "ja", "task": "transcribe"})
        return str(out.get("text", "")).strip()
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="recordings of one speaker")
    ap.add_argument("--bank", required=True, help="bank name = id prefix (letters, digits, -)")
    ap.add_argument("--gender", required=True, choices=["male", "female"])
    ap.add_argument("--age", default="", help="age range as used in the scenarios: 20代, 30代, ... (empty = any)")
    ap.add_argument("--root", default="voices", help="voices directory (default voices/)")
    ap.add_argument("--target-sec", type=float, default=30.0)
    ap.add_argument("--min-sec", type=float, default=25.0)
    ap.add_argument("--max-sec", type=float, default=33.0)
    ap.add_argument("--gap", type=float, default=0.3, help="silence between joined regions")
    ap.add_argument("--hi-db", type=float, default=22.0)
    ap.add_argument("--lo-db", type=float, default=12.0)
    ap.add_argument("--min-seg", type=float, default=0.5, help="drop regions shorter than this")
    ap.add_argument("--merge", type=float, default=0.15, help="merge regions closer than this")
    ap.add_argument("--max-takes", type=int, default=10)
    ap.add_argument("--sidon", action="store_true", help="restore with Sidon (GPU recommended)")
    ap.add_argument("--asr", action="store_true", help="transcribe with kotoba-whisper")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="only print how many takes would be made")
    args = ap.parse_args()

    clips = []
    for path in args.inputs:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        regions = vad(x, sr, args.hi_db, args.lo_db, args.min_seg, args.merge)
        for take in pack(regions, args.target_sec, args.min_sec, args.max_sec, args.gap):
            g = np.zeros(int(args.gap * sr), np.float32)
            parts = []
            for k, (s, e) in enumerate(take):
                if k:
                    parts.append(g)
                parts.append(x[int(s * sr): int(e * sr)])
            clips.append((np.concatenate(parts), sr, path))
        print(f"{path}: {sum(e - s for s, e in regions):.0f} s of speech in {len(regions)} regions")
    clips = clips[: args.max_takes]
    print(f"-> {len(clips)} takes" + (" (need at least 1; 3 or more recommended)" if len(clips) < 3 else ""))
    if args.dry_run or not clips:
        return

    restorer = Sidon(args.device) if args.sidon else None
    asr = transcriber(args.device) if args.asr else None
    out_dir = Path(args.root) / args.bank
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (x, sr, src) in enumerate(clips, 1):
        if restorer is not None:
            x, sr = restorer.restore(x, sr)
        x, sp = normalize(x, sr)
        vid = f"{args.bank}_t{i}"
        wav = out_dir / "wav" / f"{vid}.wav"
        sf.write(str(wav), x, sr, subtype="PCM_16")
        text = asr(x, sr) if asr is not None else ""
        rows.append({"id": vid, "gender": args.gender, "age_range": args.age, "audio_path": wav.as_posix(),
                     "text": text, "restore": "sidon" if restorer is not None else "",
                     "speech_db": "" if sp is None else round(sp, 1)})
        print(f"  {wav}  {len(x) / sr:.1f} s  {text[:40]}")
    with open(out_dir / "voices.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / 'voices.csv'} ({len(rows)} takes)")
    print(f"use it in a config: tts.voices_file: [{out_dir.as_posix()}/voices.csv]  (or tts.main_voices_file)")


if __name__ == "__main__":
    main()
