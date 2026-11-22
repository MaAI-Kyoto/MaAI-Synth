#!/usr/bin/env bash
# Build the default TTS server: Irodori-TTS-v4.1-Small-Yomi on vLLM-Omni (+ the Irodori pipeline patch).
# Creates third_party/vllm-omni, third_party/Irodori-TTS, models/Irodori-TTS-v4.1-Small-Yomi and
# models/irodori-v4.1-small-yomi-vllm/config.json. Run it inside the Python environment you want to use for
# the TTS server (tested: Python 3.12, vllm 0.30.0, torch 2.13, CUDA 12.x). See third_party/irodori_vllm/README.md.
#
#   python3.12 -m venv .venv-tts-server && source .venv-tts-server/bin/activate
#   bash scripts/setup_irodori_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_OMNI_COMMIT=${VLLM_OMNI_COMMIT:-a8576cc}
IRODORI_COMMIT=${IRODORI_COMMIT:-3822056}
mkdir -p third_party models

# 1) vLLM-Omni at the tested commit + the Irodori pipeline
if [ ! -d third_party/vllm-omni/.git ]; then
  git clone https://github.com/vllm-project/vllm-omni.git third_party/vllm-omni
fi
git -C third_party/vllm-omni checkout -q "$VLLM_OMNI_COMMIT"
if git -C third_party/vllm-omni apply --check ../irodori_vllm/vllm-omni-irodori.patch 2>/dev/null; then
  git -C third_party/vllm-omni apply ../irodori_vllm/vllm-omni-irodori.patch
else
  echo "(patch already applied or does not apply cleanly; skipping)"
fi
pip install "vllm==${VLLM_VERSION:-0.30.0}"
pip install -e third_party/vllm-omni

# 2) Irodori-TTS (imported by the pipeline from its source tree)
if [ ! -d third_party/Irodori-TTS/.git ]; then
  git clone https://github.com/Aratako/Irodori-TTS.git third_party/Irodori-TTS
fi
git -C third_party/Irodori-TTS checkout -q "$IRODORI_COMMIT"
pip install dacvae "peft>=0.18" "sentencepiece>=0.1.99,<0.2" "safetensors>=0.7" "huggingface-hub>=0.34" soundfile pyyaml

# 3) Weights + model directory for `vllm serve`
if command -v hf >/dev/null; then hf download takuma104/Irodori-TTS-v4.1-Small-Yomi --local-dir models/Irodori-TTS-v4.1-Small-Yomi; else huggingface-cli download takuma104/Irodori-TTS-v4.1-Small-Yomi --local-dir models/Irodori-TTS-v4.1-Small-Yomi; fi
python third_party/irodori_vllm/make_model_dir.py \
  --irodori-src third_party/Irodori-TTS \
  --checkpoint models/Irodori-TTS-v4.1-Small-Yomi/model.safetensors \
  --out models/irodori-v4.1-small-yomi-vllm
echo "done. Start the server with: bash scripts/serve_irodori.sh"
