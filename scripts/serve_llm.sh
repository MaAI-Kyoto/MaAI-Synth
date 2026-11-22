#!/usr/bin/env bash
# Serve the default LLM (Gemma 4 31B, FP8 weights by RedHatAI) with vLLM's OpenAI-compatible server.
# Stage 1-2 and the Stage 3 backchannel refinement talk to it at http://localhost:8001/v1
# (configs/base.yaml: llm.base_urls / tts.refine.base_urls). Any OpenAI-compatible server works;
# see readme/extending.md to use another model.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_llm.sh
#   MODEL=google/gemma-4-31B-it PORT=8002 bash scripts/serve_llm.sh
# One 48 GB GPU is enough for the FP8 weights (~31 GB) with a 32k context.
set -euo pipefail
MODEL=${MODEL:-RedHatAI/gemma-4-31B-it-FP8-dynamic}
NAME=${NAME:-google/gemma-4-31B-it}     # must equal `model:` in the config
PORT=${PORT:-8001}
CTX=${CTX:-32768}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec vllm serve "$MODEL" \
  --served-model-name "$NAME" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" \
  --max-model-len "$CTX" \
  --language-model-only \
  --gpu-memory-utilization "${GPU_MEM:-0.93}" \
  --enable-prefix-caching \
  --override-generation-config '{"top_k": 40, "top_p": 0.95, "min_p": 0.05}' \
  "$@"
