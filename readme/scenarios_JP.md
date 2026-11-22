# シナリオ

📄 [English](scenarios.md) | [日本語](scenarios_JP.md)

シナリオは次の3つのファイルの組み合わせです。

| ファイル | 使う段 | 中身 |
|---|---|---|
| `configs/<シナリオ>.yaml` | 全段 | どのシナリオ・プロンプトのファイルを使うか、分布、ターン数、ゲート、出力先。`configs/base.yaml` を継承 |
| `scenarios/<パック>.yaml` | Stage1 | LLM に場面を設計させるシステムプロンプト（`system_prompt`）、ジャンルと関係の一覧（`genres`・`relationships`・任意で `settings`） |
| `prompts/stage2_<パック>.yaml` | Stage2 | 台本を書かせるシステムプロンプトと部品（`system_prompt`・`schema_enriched`・`enriched_section` など） |

対話が日本語なので、プロンプトも日本語で書いてあります。

## 入っているシナリオ

| config | Stage1 のパック | Stage2 のパック | 特別な設定 |
|---|---|---|---|
| `ft_v1` | `first_meeting.yaml` | `stage2_first_meeting.yaml` | システム側の人物設定 `personas/inoue_ft_v5.yaml` |
| `ft_v2` | `first_meeting_v2.yaml` | `stage2_first_meeting.yaml` | `metadata.topic_plan`・`metadata.partner_persona`・システム側の人物設定 5 種（`inoue_ft_v6_*`） |
| `ft_sil_v1` | `first_meeting.yaml` | `stage2_first_meeting_sil.yaml` | ユーザ側の長い沈黙（`dialogue.silence`・Stage3 で固定の間） |
| `ft_sil_v2` | `first_meeting_v2.yaml` | `stage2_first_meeting_sil.yaml` | 沈黙＋人物カード |
| `ft_5min_v1` | `first_meeting_5min.yaml` | `stage2_first_meeting_5min.yaml` | 62〜78 ターン |
| `al_v1` | `al_v1.yaml` | `stage2_al.yaml` | `topic_plan.style: al`（出来事・聞き手の反応）・ゲート `al_shape` |
| `rp_v1` | `first_meeting_v2.yaml` | `stage2_first_meeting.yaml` | `topic_plan.style: rp`: 生成後に重なりの場面を入れる（割り込み・同時に話し始める・言葉の引き取り など） |
| `qa_v1` | `qa_v1.yaml` | `stage2_qa.yaml` | `topic_plan.style: qa`（質問の種類・答え方）・ゲート `qa_shape` |
| `cs_v1` | `customer_service_persona.yaml` | `stage2_customer_service_persona.yaml` | システム側が担当者（`personas/inoue.yaml`） |
| `main_v1` | `general.yaml` | `stage2_general_gemma.yaml` | さまざまな場面と間柄 |
| `tri_ft_v1` | `tri/first_meeting.yaml` | `tri/stage2_first_meeting.yaml` | 3人・`main_tri.py` / `generate_tts_tri.py` で実行 |

## 新しいシナリオを作る

1. 近い config を写し（例 `cp configs/ft_v1.yaml configs/my_scene.yaml`）、出力先（`data/my_scene/...`・すべての `*_file`・`*_dir`）を変える。
2. Stage1 のパックを写し（`scenarios/first_meeting.yaml` → `scenarios/my_scene.yaml`）、`system_prompt`・`genres`・`relationships`
   を書き換える。プロンプト中の JSON の出力形式は変えないこと（Stage2 がその項目を読む）。
3. 台本の書き方が違うなら Stage2 のパックも写し、`dialogue.prompt_file` で指す。
4. config の `metadata.scenario_file`（と `dialogue.prompt_file`）を設定し、数対話で試す:

   ```bash
   python main.py --config configs/my_scene.yaml --target 5
   ```

5. `data/my_scene/generation.log` を見る。品質ゲートに落ちた対話は理由つきで記録され、作り直されます。自分の作りたい話し方だと
   ほとんど落ちるゲートがあれば、`dialogue.gates` / `dialogue.enrich` で緩めます。

おもなつまみ（すべて config）:

| キー | 意味 |
|---|---|
| `target_dialogues` / `--target` | 対話の数 |
| `metadata.main_speaker` | 固定するシステム側の話者（`name`・`gender`・`age_range`・`role`・`persona_file` か `persona_files`） |
| `metadata.style_mix` | 敬語・タメ口・混合の割合 |
| `metadata.partner_roles`・`pair_mix` | ユーザ側がどんな人か |
| `metadata.topic_plan` | 話題の数、始まり方と終わり方、挨拶、名乗り（`topic_plan.py`） |
| `dialogue.min_turns`・`max_turns` | 対話の長さ |
| `dialogue.enrich` | 話し言葉の目標値（フィラー・終助詞・反応・短いターン など） |
| `dialogue.gates` | 品質ゲート（`false` にすると切れる） |
| `dialogue.proofread` | LLM による校閲の回数（0 で無効） |

## システム側の人物設定

`personas/*.yaml` は「イノウエ」の性格・話し方・癖・背景を書いたもので、Stage1 と Stage2 に渡します（`persona.py`）。
自分のシステムの人物でデータを作るときは、同じ項目で新しいファイルを書き、`metadata.main_speaker.persona_file`
（または `persona_files` のリスト。対話ごとに1つ引く）に設定します。人物設定に書いた事実は多くの対話に出てくるので、
言わせたくないことは書かないでください。
