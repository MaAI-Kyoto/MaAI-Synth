#!/usr/bin/env bash
# Fetch MaAI (https://github.com/MaAI-Kyoto/MaAI). Stage 3 uses its batch inference API `maai.MaaiBatch`
# (turn-taking with VAP + backchannel timing) to decide when each utterance starts.
# The pipeline imports MaAI from ./MaAI/src (tts.retime.maaibatch.repo_src; empty = this default).
#
#   bash scripts/setup_maai.sh            # clone into ./MaAI at the tested commit
#   MAAI_COMMIT=main bash scripts/setup_maai.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MAAI_COMMIT=${MAAI_COMMIT:-95ba113}   # tested: "Align nod_para outputs with Maai ..." (includes MaaiBatch)
if [ ! -d MaAI/.git ]; then
  git clone --filter=blob:none https://github.com/MaAI-Kyoto/MaAI.git MaAI   # fetch file contents on demand (smaller)
fi
git -C MaAI fetch -q origin
git -C MaAI checkout -q "$MAAI_COMMIT"
test -f MaAI/src/maai/batch.py || { echo "MaAI/src/maai/batch.py not found (MaaiBatch is required)"; exit 1; }
echo "MaAI ready at $(git -C MaAI rev-parse --short HEAD) (MaAI/src)."
echo "Model weights (maai-kyoto/* on Hugging Face) are downloaded on first use."
