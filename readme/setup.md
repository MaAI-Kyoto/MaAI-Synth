# Setup

📄 [English](setup.md) | [日本語](setup_JP.md)

The pipeline is a set of Python scripts run from the repository root, plus two servers:
an **LLM server** (Stage 1–3) and a **TTS server** (Stage 3). They can run on the same machine or on different ones.

| Component | Default | Hardware | Environment |
|---|---|---|---|
| Stage 1–2 (`main.py`) | — | CPU | `requirements.txt` |
| Stage 3 (`generate_tts.py`) | MaAI (MaaiBatch), ASR / speaker checks | 1 GPU, ~6 GB | `requirements-tts.txt` + `./MaAI` |
| LLM server | Gemma 4 31B (FP8), vLLM, `:8001` | 1 GPU, 48 GB | your vLLM install |
| TTS server | Irodori-TTS-v4.1-Small-Yomi, vLLM-Omni, `:8980` | 1 GPU, ~8 GB | separate venv (Python 3.12) |

Tested with: Python 3.10 (pipeline), Python 3.12 + vllm 0.30.0 + vllm-omni `a8576cc` (TTS server),
torch 2.13 / CUDA 12.6, transformers 5.5.3, MaAI `95ba113`, on RTX 3090 / A6000 / L40S / RTX 6000 Ada.

## 1. Pipeline environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the build for your CUDA
pip install -r requirements-tts.txt
bash scripts/setup_maai.sh            # clones MaAI into ./MaAI (Stage 3 imports MaAI/src)
```

Stage 3 downloads these models from the Hugging Face Hub on first use (set `HF_HOME` to choose the cache):
`maai-kyoto/*` (MaAI VAP / backchannel models), `kyutai/mimi` (MaAI encoder),
`kotoba-tech/kotoba-whisper-v2.0` (reading check), `microsoft/wavlm-base-plus-sv` (speaker checks).

Always run the scripts **from the repository root** (configs use relative paths such as `voices/...`).

## 2. LLM server (Gemma 4)

```bash
pip install vllm          # in any environment with a GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_llm.sh
curl -s http://localhost:8001/v1/models      # -> "google/gemma-4-31B-it"
```

`scripts/serve_llm.sh` serves `RedHatAI/gemma-4-31B-it-FP8-dynamic` under the name `google/gemma-4-31B-it`
(the `model:` of the configs) with a 32k context. Gemma's terms of use apply.
Several servers (e.g. one per GPU) can be listed in `llm.base_urls` and `tts.refine.base_urls`; requests are spread over them.
To use another model or the OpenAI API, see [extending.md](extending.md).

## 3. TTS server (Irodori-TTS on vLLM-Omni)

```bash
python3.12 -m venv .venv-tts-server && source .venv-tts-server/bin/activate
bash scripts/setup_irodori_server.sh        # vLLM-Omni + Irodori patch, Irodori-TTS, weights, model dir
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_irodori.sh
curl -s http://localhost:8980/health        # -> 200 when ready (first start takes a few minutes)
```

What the setup does is described in [third_party/irodori_vllm/README.md](../third_party/irodori_vllm/README.md).
The server opens the reference WAV files **by path**: run it on the same machine as Stage 3, or make sure both see
the same file system at the same paths.
More servers: start one per GPU (`PORT=8981 ...`) and write `tts.fish_speech.base_url: http://localhost:8980,http://localhost:8981`.

## 4. Persona pool (only for some scenarios)

`ft_v2`, `ft_sil_v2`, `al_v1`, `rp_v1` and `qa_v1` give the user side a persona card drawn from
`personas/partner_pool.jsonl`, which is built from two public datasets: see [personas.md](personas.md).

## 5. Run

```bash
python main.py        --config configs/ft_v1.yaml --target 2          # Stage 1+2
python generate_tts.py --config configs/ft_v1.yaml --retime-refine --gpu 0   # Stage 3
```

Useful options of `generate_tts.py`:

| Option | Meaning |
|---|---|
| `--retime-refine` | full Stage 3 (TTS → MaAI placement → backchannel refinement → MaAI placement) |
| `--retime` | TTS + MaAI placement only (no LLM refinement; output `audio/maai/`) |
| `--limit N` | process at most N dialogues |
| `--gpu 0` | GPU for MaAI / ASR / speaker checks (`cpu` also works, slowly) |
| `--fish-url URL[,URL]` | override `tts.fish_speech.base_url` |
| `--claim` | several workers on the same config share the work (one process per GPU) |
| `--merge-parts` | after parallel workers: merge their `*.part-*.jsonl` files |
| `--dry-run` | no TTS server: sine waves instead of speech (wiring test) |

Stage 1–2 skip dialogues that already exist, so a run can be resumed by running the same command again.

## Parallel runs

Start several Stage 3 workers on the same config with `--claim` (each claims one dialogue at a time):

```bash
for g in 0 1 2 3; do python generate_tts.py --config configs/ft_v1.yaml --retime-refine --claim --gpu $g & done; wait
python generate_tts.py --config configs/ft_v1.yaml --merge-parts
```

## Troubleshooting

- **`maai` import error** — run `bash scripts/setup_maai.sh`; `MaAI/src/maai/batch.py` must exist. Do not mix a
  pip-installed `maai` and the repository version in one process.
- **The LLM returns 404** — `model:` in the config must equal the server's `--served-model-name`.
- **Requests go through a proxy** — add the server hosts to `no_proxy` / `NO_PROXY`.
- **TTS server "connection refused"** — wait for `/health` to return 200; the first start compiles kernels.
- **Backchannels at odd positions** — keep `NVIDIA_TF32_OVERRIDE=0` (set automatically by `generate_tts.py`);
  MaAI's decisions change under TF32.
