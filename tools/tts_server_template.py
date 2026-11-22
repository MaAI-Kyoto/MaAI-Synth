#!/usr/bin/env python3
"""Template of an HTTP TTS server that Stage 3 can use (`tts.fish_speech.backend: moss_tts`).

Wrap any voice-cloning TTS by implementing `synthesize()` below, start this server, and select a TTS method whose
backend is `moss_tts` (see tts_presets/example-http-tts.yaml and readme/extending.md). Stage 3 then uses the same
take selection, reading check, MaAI placement and backchannel refinement as with the default Irodori method.

API (one utterance per request, standard library only):
  POST /v1/audio/speech
       {"model": str, "input": str (text to speak), "ref_audio": "data:audio/wav;base64,...",
        "ref_text": str (transcript of the reference, may be empty), "seed": int (optional),
        "max_new_tokens": int (hint), "response_format": "wav", "mode": "clone"}
       -> 200, body = WAV (any sample rate; Stage 3 resamples to 48 kHz)
  GET  /health -> 200

    python tools/tts_server_template.py --port 8990          # default engine: a test tone (no TTS)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

LOCK = threading.Lock()   # most TTS models are not thread-safe: one request at a time


def synthesize(text: str, ref_wav: np.ndarray, ref_sr: int, ref_text: str, seed: int | None) -> tuple[np.ndarray, int]:
    """Return (mono float32 waveform, sample rate) of `text` spoken in the voice of `ref_wav`.

    Replace this with your TTS. This placeholder returns a quiet tone whose length follows the text
    (about 8 characters per second), which is enough to test the wiring of the pipeline.
    """
    sr = 24000
    dur = max(0.5, len(text) / 8.0)
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(seed)
    f0 = 120.0 + 40.0 * rng.random()
    return (0.1 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), sr


def decode_data_url(url: str) -> tuple[np.ndarray, int]:
    b64 = url.split(",", 1)[1] if url.startswith("data:") else url
    x, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/speech":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            ref, ref_sr = decode_data_url(req["ref_audio"])
            with LOCK:
                wav, sr = synthesize(str(req["input"]), ref, ref_sr, str(req.get("ref_text") or ""), req.get("seed"))
            buf = io.BytesIO()
            sf.write(buf, np.asarray(wav, np.float32), int(sr), format="WAV", subtype="PCM_16")
            body = buf.getvalue()
        except Exception as e:  # report the error to the client (Stage 3 logs it and retries)
            msg = f"{type(e).__name__}: {e}".encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8990)
    args = ap.parse_args()
    print(f"listening on http://{args.host}:{args.port}  (POST /v1/audio/speech, GET /health)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
