# パイプライン

📄 [English](pipeline.md) | [日本語](pipeline_JP.md)

この公開版に同梱したプロンプト・設定・モデルは日本語の対話用です（[対話の言語](extending_JP.md#対話の言語)）。

```
configs/<シナリオ>.yaml
   │
   ├─ Stage 1  メタデータ  pipeline/generate_metadata.py   LLM ─▶ data/<s>/metadata.jsonl
   ├─ Stage 2  台本        pipeline/generate_dialogues.py  LLM ─▶ data/<s>/dialogues.jsonl
   │                       pipeline/generate_voice_passthrough.py ─▶ data/<s>/voice_dialogues.jsonl
   └─ Stage 3  音声        generate_tts.py（TTS サーバ ＋ MaAI ＋ LLM）
                           ─▶ data/<s>/audio/maai_ref/*.flac, transcripts_maai_ref.jsonl
```

## Stage1 — メタデータ（`pipeline/generate_metadata.py`）

対話ごとに LLM へ1回要求し、場面を設計します: 場面の説明、2人の話者（名前・性別・年代・役割・性格）、関係、話し方、
感情のトーン、話題のキーワード、3〜5段階の会話の流れ。ジャンルの一覧や規則はシナリオファイル（`scenarios/*.yaml`）が与え、
config が分布（`metadata.style_mix`・`partner_roles`・`pair_mix`・`topic_plan`）を足し、システム側の話者
（`metadata.main_speaker`。`personas/` の人物設定を付けることもできる）を固定します。
v2 のシナリオではユーザ側に `personas/partner_pool.jsonl` から引いた**人物カード**を付けます
（属性は Nemotron-Personas-Japan、性格と事実2文は RealPersonaChat から。[personas_JP.md](personas_JP.md)）。

## Stage2 — 台本（`pipeline/generate_dialogues.py`）

`dialogue.mode: enriched` では、LLM に**最初から話し言葉として**台本を書かせます: フィラー、終助詞、間としての読点、
短い反応、分けたターン。目標値（100字あたりのフィラー数、短いターンの割合など）は人間の対面対話コーパスで測ったもので、
`dialogue.enrich` で指定します。生成のあとで:

- 決定的な後処理（反応の繰り返しの間引き、オウム返し、終助詞の直し、TTS 用の読みの付与: `dialogue.yomi`・`yomi_ruby`）
- LLM による校閲（任意・`dialogue.proofread`）
- **品質ゲート**（話し言葉らしさ、ターンの形、`al_shape`・`qa_shape`・沈黙などシナリオごとのゲート）。落ちた対話は
  `dialogue.retry_max_attempts` 回まで作り直し、それでも落ちたら飛ばします（`dialogues_skipped.jsonl`）。

最後に台本を Stage3 の入力の形（`voice_dialogues.jsonl`: 話者・本文・種類 `speech` / `backchannel`・おおよその開始位置・
長い間や重なりなどシナリオごとの項目）に変換します。この変換は決定的で、LLM は使いません。

## Stage3 — 音声（`pipeline/generate_tts.py`・`pipeline/retime_maai.py`・`pipeline/retime_maaibatch.py`・`pipeline/refine_maai.py`）

`--retime-refine` は、対話ごとに次の3段を行います。

### 1. TTS

声は対話ごとに割り当てます（`assign_voices`）。システム側は `tts.main_voices_file`、ユーザ側は `tts.voices_file` から、
性別と年代を合わせて選びます。発話は1つずつ、その話者の約 60 秒の参照音声で TTS サーバが合成します。既定の方式
（`tts_presets/irodori-yomi-v4.yaml`）では、長さはモーラ数から決め、発話ごとに複数テイクを作って自然さ・話速・間・末尾の雑音・
声のまま切れていないかで採点し、ASR（kotoba-whisper）での読みの検査で読み違えたテイクを引き直し、話者の検査もします。

### 2. MaAI による配置（話者交替）

[MaAI](https://github.com/MaAI-Kyoto/MaAI) は、2チャネルの音声から会話の非言語的なふるまい（話者交替・相槌・頷き）を
リアルタイムに予測する軽量なソフトウェアです。このパイプラインでは、その **Voice Activity Projection（VAP）** モデルを
一括推論の API `maai.MaaiBatch` で使います（`tts.retime.backend: maaibatch`・既定。MaAI は `./MaAI/src` から import する）。

合成した発話を先頭から順に並べていきます（`MaaiRetimer.plan`）:

- 区切りごとに、確定したタイムラインの直近 `context_len_sec`（20 秒・両チャネル）と短い無音を VAP モデルに入れます
  （同梱の config では `lang: jp`・`frame_rate: 12.5`）。
- 次の発話の開始は、予測された話者交替の確率 `p_now` のハザード関数から引き（`tts.retime.hazard`）、人間の応答遅れのモデル
  （`tts.retime.latency`。同梱の値は日本語の会話に当てはめたもの）で間と短い重なりの形を整えます。同じ話者の続きは短い間にします。
- 台本にもともとある相槌は**相槌モデル**（`mode: bc`・信号 `p_bc`）で配置します: 予定の位置の近くで、相槌の確率が最も高い時点に置きます。
- MaAI が決められないとき（はっきりしたピークがないなど）は台本の位置を使い（`fallback`）、どちらで決めたかを manifest の
  ターンごとの `method` に残します。

`MaaiBatch` はリアルタイム版の `Maai` と同じモデル（`maai-kyoto/*` の同じ重み・同じエンコーダ）を、窓まるごと GPU で
計算します。出力はリアルタイム版と float32 の丸めの範囲で一致します。

### 3. 相槌の調整（MaAI ＋ LLM）

- **候補（MaAI）** — 相槌モデルで聞き手のチャネルを対話の最初から最後まで走査し、相手の長いターンの中で `p_bc` が
  ピークになる時点を探します（`tts.refine.insert`）。
- **判断（LLM）** — LLM（`tts.refine`・既定は Gemma 4）にメタデータと、タイミングと候補を注記した対話を渡し、JSON の操作を
  返させます: 相槌の挿入、既存の相槌の言い換え・削除、フィラーを入れた発話の書き直し。決定的な検査で操作を確かめます
  （1分あたりの密度、人間のデータに近い語の配分 `quality.bc_vocab_mix`、質問の途中に相槌を入れない など）。
- **合成し直しと置き直し** — 変わったターンと新しいターンだけを合成し（相槌は声がつながるよう、その話者の直前の発話も参照に加える）、
  MaAI でもう一度全体を配置します。

最後にステレオ（ch0 = システム、ch1 = ユーザ）に合成し、間にごく小さな雑音を入れ、頭と末尾に短い無音を足して
（`tts.lead_in`・`tts.tail_pad`）、FLAC として書き出します。書き起こしと、音声と完全に一致する台本も一緒に出ます（[output_JP.md](output_JP.md)）。

## シナリオごとの処理

- **長い沈黙**（`ft_sil_*`）: Stage2 が間（`pause_before_sec`）を付け、Stage3 はそれを固定の間として残します。
- **声の重なり**（`rp_v1`）: Stage2 が重なりの場面（`overlap`・`unsaid` の項目）を入れ、Stage3 は言いかけてやめた部分を切って声を重ねます（`pipeline/rp_audio.py`）。
- **三人**（`tri_ft_v1`）: `pipeline/main_tri.py` / `pipeline/generate_tts_tri.py` が二人用のコードを使い回して3人目を足します。MaAI の二者モデルは
  「注目する話者 vs ほかの和」として使います（試験的）。
