#!/usr/bin/env python3
"""Create the vLLM-Omni "model directory" for Irodori-TTS (a single config.json).

vLLM-Omni is started with `serve <model_dir>`; the patched IrodoriPipeline reads <model_dir>/config.json
to find the upstream Irodori-TTS source tree, the checkpoint and the codec.

    python third_party/irodori_vllm/make_model_dir.py \
        --irodori-src third_party/Irodori-TTS \
        --checkpoint  models/Irodori-TTS-v4.1-Small-Yomi/model.safetensors \
        --out         models/irodori-v4.1-small-yomi-vllm

The checkpoint is https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi (download it first,
e.g. `huggingface-cli download takuma104/Irodori-TTS-v4.1-Small-Yomi --local-dir models/Irodori-TTS-v4.1-Small-Yomi`).
The codec (Aratako/Semantic-DACVAE-Japanese-32dim) is fetched from the Hugging Face Hub on first start.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--irodori-src", required=True, help="clone of https://github.com/Aratako/Irodori-TTS")
    ap.add_argument("--checkpoint", required=True, help="model.safetensors of Irodori-TTS-v4.1-Small-Yomi")
    ap.add_argument("--out", required=True, help="directory to create (holds only config.json)")
    ap.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"], help="DiT precision (default bf16)")
    ap.add_argument("--codec-precision", default="fp32", choices=["bf16", "fp32"])
    ap.add_argument("--empty-cache-every", type=int, default=16,
                    help="call torch.cuda.empty_cache() once every N batches (1 = every batch)")
    args = ap.parse_args()

    src = Path(args.irodori_src).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not (src / "irodori_tts").is_dir():
        raise SystemExit(f"{src} does not look like an Irodori-TTS clone (no irodori_tts/ package)")
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    cfg = {
        "architectures": ["IrodoriPipeline"],
        "model_type": "irodori_tts",
        "irodori": {
            "src": str(src),
            "helpers": str(HERE / "helpers"),
            "checkpoint": str(ckpt),
            "codec_repo": args.codec_repo,
            "precision": args.precision,
            "codec_precision": args.codec_precision,
            "compile": False,
            "empty_cache_every": args.empty_cache_every,
        },
    }
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {out / 'config.json'}")


if __name__ == "__main__":
    main()
