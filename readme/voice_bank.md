# Voice banks

📄 [English](voice_bank.md) | [日本語](voice_bank_JP.md)

The TTS clones voices from reference recordings. A **voice bank** is a folder with ~30-second clips of one speaker
and a CSV listing them:

```
voices/<bank>/
├── voices.csv
└── wav/
    ├── <bank>_t1.wav
    ├── <bank>_t2.wav
    └── ...
```

```csv
id,gender,age_range,audio_path,text,restore,speech_db,sd_snr_db,f0_span
mori_t1,male,,voices/mori/wav/mori_t1.wav,その、そういうの作っていて、…,sidon,-23.0,42.5,10.4
```

| Column | Required | Meaning |
|---|---|---|
| `id` | yes | unique id, `<speaker>_t<N>` (takes of the same speaker share the prefix) |
| `gender` | yes | `male` / `female` — matched with the speaker's gender in the script |
| `age_range` | no | `20代`, `30代`, ... (the age-range labels Stage 1 writes); empty = matches any age |
| `audio_path` | yes | path relative to the repository root |
| `text` | no | transcript of the clip (not used by the default Irodori method; used by some other engines) |
| other columns | no | free metadata (`restore`, `speech_db`, `sd_snr_db`, `f0_span` are informational) |

## Included banks

| Bank | Speaker | Clips | Notes |
|---|---|---|---|
| `voices/inoue/` | Inoue, male, 30s | 10 × 27–32 s | restored with Sidon; 48 kHz |
| `voices/mori/` | Mori, male | 5 × 32–33 s | taken from a recorded conversation; restored with Sidon; 48 kHz |

Their terms of use are in [voices/LICENSE.md](../voices/LICENSE.md).

## How banks are used

- `tts.main_voices_file` — the bank(s) reserved for the **system side** (ch0, the fixed main speaker).
- `tts.voices_file` — the bank(s) for the **user side** (ch1). Both accept one path or a list:

  ```yaml
  tts:
    main_voices_file: [voices/inoue/voices.csv]
    voices_file: [voices/mori/voices.csv, voices/alice/voices.csv, voices/bob/voices.csv]
  ```

- For every dialogue, each speaker gets one clip whose `gender` matches (and whose `age_range` is within
  `tts.voice_age_tolerance` years, if set); the choice is deterministic per dialogue id. If no voice of the right
  gender is left, another one is used and a warning is logged — so add voices of both genders if your scenarios
  have female speakers (the shipped configs set `metadata.pair_mix` to male-only pairs because only male user-side
  voices are bundled).
- The default Irodori method gives the TTS ~60 s of reference: the chosen clip plus the other takes of the same speaker
  (`<speaker>_t*.wav` in the same folder). Banks are found by scanning every folder under `voices/` that contains WAV
  files (or `tts.fish_speech.irodori.voices_dirs`).

## Making a bank from your recordings

```bash
python tools/make_voice_bank.py --bank alice --gender female --age 20代 alice_rec1.wav alice_rec2.wav
# optional: --sidon (speech restoration, GPU recommended), --asr (transcripts with kotoba-whisper)
```

The tool detects speech, packs it into 25–33 s takes, normalizes loudness (peak −1 dBFS, speech ≤ −23 dBFS) and writes
`voices/alice/voices.csv`. Then add the CSV to `tts.voices_file` (or `tts.main_voices_file`).

Tips from building the included banks:

- **Clean, single-speaker audio.** Other voices in the clip (e.g. the partner's backchannels leaking into a
  conversation track) are cloned too. Per-speaker tracks of a conversation work if crosstalk is low (> 30 dB down).
- **Natural conversational speech** gives livelier synthetic dialogue than read speech; the speaking style of the
  references carries over (pitch range, tempo, fillers).
- **3–10 takes** per speaker. More takes = more variety between dialogues; ~60 s in total is used per utterance.
- **Sidon** (`--sidon`) removes noise and reverberation and outputs 48 kHz, at some cost in speaker similarity.
- Check the result with `python tools/manage_voices.py validate --voices voices/alice/voices.csv`.

**Consent.** Only use recordings of people who agreed to have their voice cloned, and follow the license of the TTS model
(the Irodori-TTS model cards prohibit impersonation and misleading deepfakes).
