#!/usr/bin/env bash
# Start the default TTS server (Irodori-TTS-v4.1-Small-Yomi on vLLM-Omni) on http://localhost:8980.
# Stage 3 sends POST /v1/audio/speech to tts.fish_speech.base_url (configs/base.yaml).
# The server reads reference WAVs by path, so run it on the same machine (or the same file system) as Stage 3.
# One GPU with ~8 GB is enough. For several GPUs, start one server per GPU on different ports and list them
# comma-separated in tts.fish_speech.base_url (or pass --fish-url to generate_tts.py).
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/serve_irodori.sh
#   CUDA_VISIBLE_DEVICES=2 PORT=8981 bash scripts/serve_irodori.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR=${MODEL_DIR:-models/irodori-v4.1-small-yomi-vllm}
DEPLOY=${DEPLOY:-third_party/vllm-omni/vllm_omni/deploy/irodori_tts.yaml}
PORT=${PORT:-8980}
test -f "$MODEL_DIR/config.json" || { echo "$MODEL_DIR/config.json not found: run scripts/setup_irodori_server.sh"; exit 1; }
exec python -m vllm.entrypoints.cli.main serve "$MODEL_DIR" --omni \
  --host "${HOST:-127.0.0.1}" --port "$PORT" \
  --served-model-name irodori \
  --init-timeout 2400 --stage-init-timeout 1800 \
  --deploy-config "$DEPLOY" \
  "$@"
