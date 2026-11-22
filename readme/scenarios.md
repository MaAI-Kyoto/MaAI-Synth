# Scenarios

📄 [English](scenarios.md) | [日本語](scenarios_JP.md)

A scenario is a combination of three files:

| File | Used by | Content |
|---|---|---|
| `configs/<scenario>.yaml` | all stages | which scenario / prompt files to use, distributions, turn counts, gates, output paths; extends `configs/base.yaml` |
| `scenarios/<pack>.yaml` | Stage 1 | the system prompt that makes the LLM design situations (`system_prompt`), genre and relationship lists (`genres`, `relationships`, optional `settings`) |
| `prompts/stage2_<pack>.yaml` | Stage 2 | the system prompt and blocks for writing the dialogue (`system_prompt`, `schema_enriched`, `enriched_section`, ...) |

The prompts are written in Japanese because the dialogues are Japanese.

## Included scenarios

| Config | Stage 1 pack | Stage 2 pack | Special settings |
|---|---|---|---|
| `ft_v1` | `first_meeting.yaml` | `stage2_first_meeting.yaml` | system persona `personas/inoue_ft_v5.yaml` |
| `ft_v2` | `first_meeting_v2.yaml` | `stage2_first_meeting.yaml` | `metadata.topic_plan`, `metadata.partner_persona`, 5 system personas (`inoue_ft_v6_*`) |
| `ft_sil_v1` | `first_meeting.yaml` | `stage2_first_meeting_sil.yaml` | long user-side silences (`dialogue.silence`, Stage 3 fixed pauses) |
| `ft_sil_v2` | `first_meeting_v2.yaml` | `stage2_first_meeting_sil.yaml` | silences + persona cards |
| `ft_5min_v1` | `first_meeting_5min.yaml` | `stage2_first_meeting_5min.yaml` | 62–78 turns |
| `al_v1` | `al_v1.yaml` | `stage2_al.yaml` | `topic_plan.style: al` (episodes, listener reactions), gate `al_shape` |
| `rp_v1` | `first_meeting_v2.yaml` | `stage2_first_meeting.yaml` | `topic_plan.style: rp`: overlap scenes inserted after generation (interruption, simultaneous start, collaborative completion ...) |
| `qa_v1` | `qa_v1.yaml` | `stage2_qa.yaml` | `topic_plan.style: qa` (question kinds, answer styles), gate `qa_shape` |
| `cs_v1` | `customer_service_persona.yaml` | `stage2_customer_service_persona.yaml` | the system side is a staff member (`personas/inoue.yaml`) |
| `main_v1` | `general.yaml` | `stage2_general_gemma.yaml` | varied settings and relationships |
| `tri_ft_v1` | `tri/first_meeting.yaml` | `tri/stage2_first_meeting.yaml` | three speakers; run with `main_tri.py` / `generate_tts_tri.py` |

## Making a new scenario

1. Copy the closest config, e.g. `cp configs/ft_v1.yaml configs/my_scene.yaml`, and change the output paths
   (`data/my_scene/...` — every `*_file`, `*_dir`).
2. Copy the Stage 1 pack (`scenarios/first_meeting.yaml` → `scenarios/my_scene.yaml`) and rewrite
   `system_prompt`, `genres` and `relationships`. Keep the JSON output format in the prompt unchanged:
   Stage 2 reads those fields.
3. If the dialogue style differs, copy the Stage 2 pack as well and point `dialogue.prompt_file` to it.
4. Set `metadata.scenario_file` (and `dialogue.prompt_file`) in your config, then try a few dialogues:

   ```bash
   python main.py --config configs/my_scene.yaml --target 5
   ```

5. Look at `data/my_scene/generation.log`: dialogues that fail a quality gate are logged with the reason and
   regenerated. If a gate rejects almost everything for your style, relax it in `dialogue.gates` / `dialogue.enrich`.

Useful knobs (all in the config):

| Key | Meaning |
|---|---|
| `target_dialogues` / `--target` | number of dialogues |
| `metadata.main_speaker` | fixed system-side speaker (`name`, `gender`, `age_range`, `role`, `persona_file` or `persona_files`) |
| `metadata.style_mix` | share of polite / casual / mixed speech styles |
| `metadata.partner_roles`, `pair_mix` | who the user side is |
| `metadata.topic_plan` | how many topics, how the dialogue opens and closes, greetings, naming (`topic_plan.py`) |
| `dialogue.min_turns`, `max_turns` | dialogue length |
| `dialogue.enrich` | spoken-language targets (fillers, particles, reactions, short turns ...) |
| `dialogue.gates` | quality gates (set a gate to `false` to disable it) |
| `dialogue.proofread` | LLM proofreading passes (0 = off) |

## The system-side persona

`personas/*.yaml` describe "イノウエ (Inoue)" — personality, speech style, habits and background — and are given to
Stage 1 and/or Stage 2 (`persona.py`). To build a dataset for your own system persona, write a new file with the same
keys and set `metadata.main_speaker.persona_file` (or a list in `persona_files`, one is drawn per dialogue).
Facts written in a persona tend to appear in many dialogues; do not write anything you do not want to be said.
