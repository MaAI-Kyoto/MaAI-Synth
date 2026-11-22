# Persona pool for the user side

📄 [English](personas.md) | [日本語](personas_JP.md)

The scenarios `ft_v2`, `ft_sil_v2`, `al_v1`, `rp_v1` and `qa_v1` give the user side a **persona card**: attributes
(gender, age, prefecture, occupation, hobbies, marital status, surname), a talking style and two personal facts.
Cards are drawn from `personas/partner_pool.jsonl` (`metadata.partner_persona`, `pipeline/partner_persona.py`).
The pool is **not included** in this repository because it is derived from two datasets with their own licenses;
build it once with the commands below (about 3 minutes; the result is identical to the pool used for the samples).

| Source | License | Used for |
|---|---|---|
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) (NVIDIA) | CC BY 4.0 | attributes, surname |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) (nu-dialogue) | CC BY-SA 4.0 | personality test → talking style, two persona facts |

```bash
pip install pyarrow pandas huggingface_hub
mkdir -p third_party/data
huggingface-cli download nvidia/Nemotron-Personas-Japan --repo-type dataset \
    --local-dir third_party/data/nemotron_personas_japan
git clone https://github.com/nu-dialogue/real-persona-chat.git third_party/data/real-persona-chat
python tools/build_partner_personas.py \
    --nemotron third_party/data/nemotron_personas_japan \
    --rpc third_party/data/real-persona-chat \
    --out personas/partner_pool.jsonl
```

`--nemotron` must point to the folder with the parquet files and `--rpc` to the folder that contains
`real_persona_chat/interlocutors.json`. The pool has 2,000 cards per gender × age group (20s–70s).
If you redistribute the pool or data generated with it, follow both licenses (attribution; share-alike for RealPersonaChat).

Scenarios without persona cards (`ft_v1`, `ft_sil_v1`, `ft_5min_v1`, `cs_v1`, `main_v1`, `tri_ft_v1`) do not need the pool.
