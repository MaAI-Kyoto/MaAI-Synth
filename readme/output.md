# Output format

📄 [English](output.md) | [日本語](output_JP.md)

Everything a scenario produces goes to the directory set in its config (`data/<scenario>/` by default).

| File | Stage | Content |
|---|---|---|
| `metadata.jsonl` | 1 | one line per dialogue: `id`, `scenario`, `conversation_flow`, `speakers`, `relationship`, `speech_style`, `topic_plan`, ... |
| `dialogues.jsonl` | 2 | the script: `turns` with `speaker`, `text`, `type` (`speech` / `backchannel`) and reading annotations |
| `dialogues_skipped.jsonl` | 2 | ids that failed the quality gates after all retries |
| `voice_dialogues.jsonl` | 2 | input of Stage 3 (same turns + rough timing, scenario extras such as `pause_before_sec`, `overlap`) |
| `audio/plain/turns/<id>/` | 3 | cache of the synthesized utterances (before placement) |
| `audio/maai/<id>.flac` | 3 | after MaAI placement, before refinement |
| `audio/maai_ref/<id>.flac` | 3 | **final audio** |
| `transcripts_maai_ref.jsonl` | 3 | **final time-aligned transcript** |
| `audio_manifest_maai_ref.jsonl` | 3 | per-turn details: voice id, timing, how the start was decided (`retime.method`), refinement history |
| `voice_dialogues_refined_maai_ref.jsonl` | 3 | the script after refinement; its turns match the audio one-to-one |
| `refine_ops/` | 3 | cached LLM refinement operations (re-runs do not query the LLM again) |
| `generation.log`, `tts.log`, `cost_log.jsonl` | 1–3 | logs and token usage |

## License of generated data

Everything in `data/<scenario>/` is derived from the LLM, the TTS model and the voice banks you used (and, for some
scenarios, the persona datasets). It is subject to their licenses and terms of use, not to the MIT license of the code.
See [THIRD_PARTY.md](../THIRD_PARTY.md) and [voices/LICENSE.md](../voices/LICENSE.md).

## Audio

- FLAC, 48 kHz, 16-bit, **2 channels: ch0 (left) = system side, ch1 (right) = user side**.
- Each channel contains only that speaker's voice (no crosstalk), with very low pink noise (−80 dBFS) in the gaps.
- A random silence of 0.4–6 s before the first and 1–6 s after the last utterance (`tts.lead_in`, `tts.tail_pad`).

## Transcript (`transcripts_maai_ref.jsonl`)

One JSON object per dialogue:

```json
{"id": "6630d4cd-36ff-43df-8fff-bfa421064240", "audio": "6630d4cd-36ff-43df-8fff-bfa421064240.flac",
 "sample_rate": 48000, "channels": 2, "duration_sec": 131.5,
 "channel_speakers": {"0": "井上", "1": "森"},
 "voice_prompts": {"森": {"voice_id": "mori_t2", "gender": "male", ...}, "井上": {"voice_id": "inoue_t4", ...}},
 "segments": [
   {"start": 0.435, "end": 4.315, "speaker": "森", "channel": 1, "type": "speech",
    "text": "どうも。俺、森って言います。よろしくお願いします。"},
   {"start": 4.353, "end": 8.293, "speaker": "井上", "channel": 0, "type": "speech",
    "text": "はじめまして。井上です。こちらこそ、よろしくお願いします。"},
   ...,
   {"start": 26.474, "end": 26.974, "speaker": "井上", "channel": 0, "type": "backchannel", "text": "うん。"},
   ...]}
```

`start` / `end` are seconds in the audio file. Segments are in order of start time and may overlap (overlapping speech,
backchannels during the other speaker's turn). Some segments carry reading annotations (`ruby`, `yomi`) for kanji.
