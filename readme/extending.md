# Using other LLMs and TTS engines

📄 [English](extending.md) | [日本語](extending_JP.md)

## LLM

All text generation (Stage 1, Stage 2, the Stage 2 proofreading pass and the Stage 3 backchannel refinement) goes
through the OpenAI Python SDK (`pipeline/llm_client.py`, `pipeline/refine_maai.py`). Any server that speaks the OpenAI
`chat.completions` API works: vLLM, SGLang, llama.cpp (`llama-server`), Ollama, or the OpenAI API itself.

Change these keys (in `configs/base.yaml`, or override them in a scenario config):

```yaml
model: Qwen/Qwen3-32B                    # = the served model name
llm:
  provider: local                        # local = OpenAI-compatible server, openai = OpenAI API
  model: Qwen/Qwen3-32B
  base_urls: [http://gpu1:8001/v1, http://gpu2:8001/v1]   # requests are spread over all servers
  api_key_env: OPENAI_API_KEY            # name of the environment variable with the key (local: any / none)
  extra_body:                            # passed as-is to the server (e.g. disable "thinking")
    chat_template_kwargs: {enable_thinking: false}
tts:
  refine:
    text_model: Qwen/Qwen3-32B
    fallback_text_model: Qwen/Qwen3-32B
    base_urls: [http://gpu1:8001/v1]
```

For the OpenAI API: `provider: openai`, remove `base_urls`, set `model` / `text_model` to e.g. `gpt-5-mini`,
and export `OPENAI_API_KEY`. Costs are tracked with `pricing` (USD per 1M tokens) and stopped at `max_cost_usd`.

Notes:

- `dialogue.prompt_profile` selects prompt variants tuned per model family (`gemma` in the shipped configs;
  `auto` guesses from the model name). With a new model, start with `auto` and check the gate statistics in
  `data/<scenario>/generation.log`: the quality gates retry dialogues that miss the targets, so a model that follows
  the instructions less well mainly costs more retries.
- Reasoning ("thinking") models should run with thinking disabled; the parsers expect plain JSON / text.
- The shipped settings were tuned with Gemma 4 31B; other models may need different `dialogue.enrich` targets.

## TTS

### How a TTS method is chosen

`tts.engine` names a file in `tts_presets/`. After all `extends:` are resolved, that YAML is deep-merged into
`tts` (`pipeline/tts_presets.py`) — so a method is just a reusable bundle of TTS settings. The default
`tts_presets/irodori-yomi-v4.yaml` selects the reference-audio TTS path (`engine: fish_speech`) with the
`irodori_vllm` backend. (The block is called `fish_speech` for historical reasons; it holds every reference-audio backend.)

To change settings of the default method, copy it (`cp tts_presets/irodori-yomi-v4.yaml tts_presets/my-irodori.yaml`),
edit it, and set `tts.engine: my-irodori`. Keep old presets unchanged so that existing datasets stay reproducible.

### Adding a new TTS engine as an HTTP server (recommended)

The `moss_tts` backend (`generate_tts.MossTtsSynth`) is a generic client: one HTTP request per utterance with the text
and one reference clip of the speaker. Wrap your TTS in a small server with this API and every other part of Stage 3
(take selection for short utterances, the ASR reading check, speaker checks, MaAI placement, backchannel refinement)
works unchanged.

```
POST <base_url>/v1/audio/speech
  {"model": "...", "input": "<text>", "ref_audio": "data:audio/wav;base64,...", "ref_text": "<transcript or empty>",
   "seed": 123, "max_new_tokens": 300, "response_format": "wav", "mode": "clone"}
-> 200 audio/wav   (any sample rate; resampled to 48 kHz)
GET <base_url>/health -> 200
```

1. Implement `synthesize()` in [`tools/tts_server_template.py`](../tools/tts_server_template.py) (standard library HTTP
   server; replace the test tone with your model) and start it: `python tools/tts_server_template.py --port 8990`.
2. Copy [`tts_presets/example-http-tts.yaml`](../tts_presets/example-http-tts.yaml) to `tts_presets/<your-engine>.yaml`.
3. In a config: `tts: {engine: <your-engine>, fish_speech: {base_url: http://localhost:8990}}`
   (several servers: comma-separated URLs; requests are distributed and dead servers are skipped).
4. Try `python pipeline/generate_tts.py --config ... --retime-refine --limit 1` and listen to `audio/maai_ref/`.

Useful options of the `moss_tts` backend (under `tts.fish_speech`): `concurrency`, `moss_out_sr` (keep the server's
rate instead of 48 kHz), `moss_tempo` (time-stretch every take), `short_best_of` (several takes for short utterances),
`best_of` (several takes for every utterance, scored by naturalness / rate / pauses), `regen_on_misread`
(re-draw takes the ASR reads differently).

### Adding a synthesizer class

For engines that need more than one reference clip, context audio, or batching, add a class in `pipeline/generate_tts.py`
next to `MossTtsSynth` / `IrodoriVllmSynth` (they show how to implement `_post_speech`, take selection and the
context paths) and register it in `build_fish_speech_synth()` under a new `backend` name. `IrodoriVllmSynth` is the
reference implementation used by the default method: it sends `extra_params` (target length, sibling takes of the
voice bank as `ref_wavs`, diffusion steps, CFG) to the vLLM-Omni Irodori pipeline.

Other engines already present in the code (not configured or tested in this release): fish-speech S2 (`backend: fish`
/ `vllm_omni`), Qwen3-TTS (`engine: qwen3`), Irodori-TTS in-process (`engine: irodori`), Sarashina2.2-TTS,
Gemini / Google Cloud TTS (preset voices).

## Timing model

`tts.retime.backend` selects how utterances are placed:

| Backend | What it is |
|---|---|
| `maaibatch` (default) | MaAI's official batch API `maai.MaaiBatch` (VAP + backchannel models, imported from `./MaAI/src`) |
| `maai` | MaAI's real-time `Maai` class driven offline (pip `maai`; slower, kept for compatibility) |
| `vapbatch` | an earlier in-house batch re-implementation (not included in this release) |

The VAP model, thresholds and the hazard / latency parameters are under `tts.retime` (`vap`, `bc`, `hazard`, `latency`).
