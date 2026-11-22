# Pipeline

📄 [English](pipeline.md) | [日本語](pipeline_JP.md)

The prompts, settings and models shipped in this release are for Japanese dialogues; see [Dialogue language](extending.md#dialogue-language).

```
configs/<scenario>.yaml
   │
   ├─ Stage 1  metadata   pipeline/generate_metadata.py   LLM ─▶ data/<s>/metadata.jsonl
   ├─ Stage 2  script     pipeline/generate_dialogues.py  LLM ─▶ data/<s>/dialogues.jsonl
   │                      pipeline/generate_voice_passthrough.py ─▶ data/<s>/voice_dialogues.jsonl
   └─ Stage 3  audio      pipeline/generate_tts.py  (TTS server + MaAI + LLM)
                          ─▶ data/<s>/audio/maai_ref/*.flac, transcripts_maai_ref.jsonl
```

## Stage 1 — metadata (`pipeline/generate_metadata.py`)

One LLM request per dialogue designs the situation: a scenario sentence, the two speakers
(name, gender, age, role, personality), the relationship, speech style, emotional tone, topic keywords
and a 3–5 step conversation flow. The scenario file (`scenarios/*.yaml`) gives the genre lists and rules;
the config adds distributions (`metadata.style_mix`, `partner_roles`, `pair_mix`, `topic_plan`) and fixes
the system-side speaker (`metadata.main_speaker`, optionally with a persona file from `personas/`).
In the v2 scenarios the user side receives a **persona card** sampled from `personas/partner_pool.jsonl`
(attributes from Nemotron-Personas-Japan, personality and two facts from RealPersonaChat; see [personas.md](personas.md)).

## Stage 2 — script (`pipeline/generate_dialogues.py`)

`dialogue.mode: enriched` asks the LLM to write the dialogue **as spoken language from the start**:
fillers, sentence-final particles, commas as pauses, short reactions, split turns. The targets
(e.g. fillers per 100 characters, share of short turns) were measured on a corpus of human face-to-face
conversations and are set in `dialogue.enrich`. After generation:

- deterministic post-processing (thinning repeated reactions, echo questions, particle fixes, reading
  annotations for the TTS: `dialogue.yomi`, `yomi_ruby`),
- an optional LLM proofreading pass (`dialogue.proofread`),
- **quality gates** (spoken-language density, turn shape, scenario-specific gates such as `al_shape`,
  `qa_shape`, silences). A dialogue that fails is regenerated, up to `dialogue.retry_max_attempts`
  times, and skipped otherwise (`dialogues_skipped.jsonl`).

Finally the script is converted to the input schema of Stage 3 (`voice_dialogues.jsonl`: turns with speaker,
text, type `speech` / `backchannel`, a rough start offset, and scenario extras such as long pauses or overlaps).
This conversion is deterministic and uses no LLM.

## Stage 3 — audio (`pipeline/generate_tts.py`, `pipeline/retime_maai.py`, `pipeline/retime_maaibatch.py`, `pipeline/refine_maai.py`)

`--retime-refine` runs the three steps below for each dialogue.

### 1. TTS

Voices are assigned per dialogue (`assign_voices`): the system side from `tts.main_voices_file`, the user side from
`tts.voices_file`, matched by gender and age. Each utterance is synthesized separately by the TTS server with a
reference of ~60 s of the speaker's voice. With the default method (`tts_presets/irodori-yomi-v4.yaml`):
length from the mora count, several takes per utterance scored for naturalness, speaking rate, pauses, tail noise
and cut-off voice, an ASR reading check (kotoba-whisper) that re-draws misread takes, and a speaker check.

### 2. Placement with MaAI (turn-taking)

[MaAI](https://github.com/MaAI-Kyoto/MaAI) is a real-time, light-weight software that predicts non-linguistic
behaviors of conversation (turn-taking, backchannels, nodding) from two-channel audio. This pipeline uses its
**Voice Activity Projection (VAP)** models through the offline batch API `maai.MaaiBatch`
(`tts.retime.backend: maaibatch`, the default; MaAI is imported from `./MaAI/src`).

The synthesized utterances are placed one after another (`MaaiRetimer.plan`):

- At every boundary the confirmed timeline of the last `context_len_sec` (20 s, both channels) plus a short
  silence probe is fed to the VAP model (`lang: jp` in the shipped configs, `frame_rate: 12.5`).
- The start of the next utterance is drawn from a hazard function over the predicted turn-shift probability
  `p_now` (`tts.retime.hazard`), and a human response-latency model (`tts.retime.latency`; the shipped
  parameters are fitted to Japanese conversation) shapes gaps and short overlaps. Same-speaker continuations use shorter pauses.
- Backchannels already in the script are placed with the **backchannel model** (`mode: bc`, signal `p_bc`):
  at the peak of the predicted backchannel probability near their intended position.
- If MaAI cannot decide (e.g. no clear peak), the position from the script is used (`fallback`), and the choice is
  recorded per turn in the manifest (`method`).

`MaaiBatch` runs the same models as the real-time `Maai` (same weights from `maai-kyoto/*`, same encoder)
but over whole windows on the GPU; its outputs match the real-time version within float32 rounding.

### 3. Backchannel refinement (MaAI + LLM)

- **Candidates (MaAI)** — the backchannel model scans the listener's channel over the whole dialogue and finds
  peaks of `p_bc` inside the other speaker's long turns (`tts.refine.insert`).
- **Decision (LLM)** — the LLM (`tts.refine`, default Gemma 4) receives the metadata and the dialogue annotated with
  timing and candidates, and returns JSON operations: insert a backchannel, reword or delete an existing one,
  rewrite an utterance with fillers. Deterministic guards validate the operations (density per minute, vocabulary
  mix close to human data `quality.bc_vocab_mix`, no backchannels inside questions, etc.).
- **Re-synthesis and re-placement** — only changed or new turns are synthesized (backchannels with the speaker's
  previous utterance as an extra reference, so that they continue the voice), and MaAI places everything again.

The result is mixed to a stereo file (ch0 = system, ch1 = user) with low-level noise in the gaps, short lead-in and
tail silences (`tts.lead_in`, `tts.tail_pad`), and written as FLAC together with the transcript and a refined script
that matches the audio exactly (see [output.md](output.md)).

## Scenario-specific steps

- **Long silences** (`ft_sil_*`): Stage 2 marks pauses (`pause_before_sec`), Stage 3 keeps them as fixed pauses.
- **Overlaps** (`rp_v1`): Stage 2 inserts overlap scenes (`overlap`, `unsaid` fields); Stage 3 cuts the abandoned
  part of an utterance and overlaps the voices (`pipeline/rp_audio.py`).
- **Three-party** (`tri_ft_v1`): `pipeline/main_tri.py` / `pipeline/generate_tts_tri.py` reuse the two-party code with a third speaker;
  MaAI's two-party models are applied as "focus speaker vs. the sum of the others" (experimental).
