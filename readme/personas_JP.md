# ユーザ側の人物カードの母集団

📄 [English](personas.md) | [日本語](personas_JP.md)

`ft_v2`・`ft_sil_v2`・`al_v1`・`rp_v1`・`qa_v1` は、ユーザ側に**人物カード**を付けます: 属性（性別・年齢・都道府県・職業・趣味・
婚姻・姓）、話し方、本人の事実2文。カードは `personas/partner_pool.jsonl` から引きます（`metadata.partner_persona`・`pipeline/partner_persona.py`）。
この母集団は、それぞれライセンスのある2つのデータセットから作るので、このリポジトリには**入れていません**。
下のコマンドで一度作ってください（約3分。サンプルを作ったときの母集団と同じものができます）。

| 元データ | ライセンス | 使うもの |
|---|---|---|
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan)（NVIDIA） | CC BY 4.0 | 属性・姓 |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat)（nu-dialogue） | CC BY-SA 4.0 | 性格検査 → 話し方、ペルソナ文2文 |

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

`--nemotron` は parquet のあるフォルダ、`--rpc` は `real_persona_chat/interlocutors.json` を含むフォルダを指します。
母集団は性別×年代（20〜70代）ごとに 2,000 枚です。母集団やそれで作ったデータを再配布するときは、両方のライセンス
（表示・RealPersonaChat の継承）に従ってください。

人物カードを使わないシナリオ（`ft_v1`・`ft_sil_v1`・`ft_5min_v1`・`cs_v1`・`main_v1`・`tri_ft_v1`）では母集団は要りません。
