# Third-party software, models and data

MaAI-Synth does not bundle the following; they are downloaded by the setup scripts or on first use.
Each is governed by its own license.

**Generated data follows these licenses.** The MIT license of this repository covers only its source code.
Dialogue scripts, audio and transcripts generated with the pipeline are subject to the licenses and terms of use of
the components that produced them — at least the LLM, the TTS model and codec, the voice banks and, for the
scenarios with persona cards, the persona datasets. Check them before using or distributing generated data,
especially for commercial use or for training models, and again whenever you switch to another LLM, TTS or voice.

| Component | Used for | License / terms |
|---|---|---|
| [MaAI](https://github.com/MaAI-Kyoto/MaAI) (`maai.MaaiBatch`, models `maai-kyoto/*`) | Stage 3 turn-taking and backchannel timing | MIT (code); see each model's card |
| [Mimi](https://huggingface.co/kyutai/mimi) | audio encoder inside MaAI | CC BY 4.0 |
| [Gemma 4](https://ai.google.dev/gemma) (`google/gemma-4-31B-it`, FP8 weights `RedHatAI/gemma-4-31B-it-FP8-dynamic`) | default LLM | Gemma terms of use |
| [vLLM](https://github.com/vllm-project/vllm) | LLM server | Apache-2.0 |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) (+ `third_party/irodori_vllm/vllm-omni-irodori.patch`) | TTS server | Apache-2.0 |
| [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) | TTS inference code | MIT |
| [Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi) | default TTS model | MIT + ethical restrictions of the model card (no impersonation, no misleading deepfakes) |
| [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) | TTS codec | see model card |
| [kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) | ASR reading check | see model card |
| [wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | speaker checks, backchannel continuity | see model card |
| [Sidon](https://huggingface.co/sarulab-speech/sidon-v0.1) (optional, `tools/make_voice_bank.py --sidon`) | speech restoration of voice banks | see model card |
| [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) | readings of Japanese text | see repository |
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) | persona pool (built locally) | CC BY 4.0 |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) | persona pool (built locally) | CC BY-SA 4.0 |

The quality targets of Stage 2 and the timing parameters of Stage 3 were measured on human conversation corpora
that are not distributed; only aggregate numbers appear in the configs (and `reference/teidan_tri.json` for the
three-party scenario).
