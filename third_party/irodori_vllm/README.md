# Irodori-TTS server on vLLM-Omni

📄 [English](README.md) | [日本語](README_JP.md)

The default TTS method (`tts_presets/irodori-yomi-v4.yaml`) talks to
[Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi)
(a fine-tune of [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1 Small that fixes kanji readings; RF-DiT +
DACVAE, 48 kHz, Japanese) served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni). vLLM-Omni groups concurrent
requests into batches, which makes Stage 3 several times faster than calling the model one utterance at a time.

`scripts/setup_irodori_server.sh` does the following; you can also run the steps by hand.

| Step | What |
|---|---|
| 1 | clone vLLM-Omni at commit `a8576cc` and apply [`vllm-omni-irodori.patch`](vllm-omni-irodori.patch); `pip install vllm==0.30.0` and `pip install -e` vLLM-Omni |
| 2 | clone Irodori-TTS at commit `3822056` (the pipeline imports `irodori_tts` from this source tree) and install its runtime packages |
| 3 | download the weights and write the model directory with [`make_model_dir.py`](make_model_dir.py) |

Then `scripts/serve_irodori.sh` runs
`python -m vllm.entrypoints.cli.main serve <model_dir> --omni --port 8980 --served-model-name irodori --deploy-config vllm_omni/deploy/irodori_tts.yaml`.

## What the patch adds

| File | Change |
|---|---|
| `vllm_omni/diffusion/models/irodori/` | `IrodoriPipeline`: runs Irodori's `InferenceRuntime.synthesize_batch` on the vLLM-Omni diffusion engine; per-request `extra_params` (`seconds`, `ref_wavs`, `max_ref_seconds`, `num_steps`, `cfg_scale_speaker`, `seed`, ...) |
| `vllm_omni/deploy/irodori_tts.yaml` | one diffusion stage, up to 16 requests per batch |
| `vllm_omni/diffusion/registry.py`, `config/pipeline_registry.py` | register the pipeline |
| `vllm_omni/entrypoints/openai/serving_speech.py` | read the sample rate of diffusion outputs; shorter request logging; `VLLM_OMNI_REF_AUDIO_MAX_SEC` |
| `vllm_omni/diffusion/executor/multiproc_executor.py` | fix: a finished batch could be left uncollected (requests timed out after 3600 s) |

`helpers/irodori_fast.py` caches the encoded reference audio per file, so that the 60 s reference is not re-encoded
for every utterance (identical output).

## Notes

- **Shared file system.** Stage 3 sends the voice-bank files as paths (`extra_params.ref_wavs`); the server must be able
  to open them (same machine, or the same paths on a shared file system). Temporary files for backchannels are written
  to `data/_tmp/irodori_prevref/` and passed the same way.
- **GPU memory.** About 6–8 GB with the default batch size (bf16 DiT). One server per GPU; list several in
  `tts.fish_speech.base_url` (comma-separated).
- **License.** Irodori-TTS-v4.1-Small-Yomi is MIT-licensed; its model card (and that of the base model) prohibits
  impersonation and misleading deepfakes. Do not clone a voice without the speaker's consent. vLLM-Omni is Apache-2.0; the patch is
  distributed under the same license as vLLM-Omni.
- **Faster variant.** Production runs used a locally optimized copy of Irodori-TTS (folded weight norm, GPU caches,
  length-sorted batches; ~1.4x faster on an RTX 3090). It is not included; upstream Irodori-TTS gives the same output
  up to batching noise.
