# 出力の形式

📄 [English](output.md) | [日本語](output_JP.md)

シナリオが作るものはすべて、config で指定したディレクトリ（既定は `data/<シナリオ>/`）に出ます。

| ファイル | 段 | 中身 |
|---|---|---|
| `metadata.jsonl` | 1 | 1行1対話: `id`・`scenario`・`conversation_flow`・`speakers`・`relationship`・`speech_style`・`topic_plan` など |
| `dialogues.jsonl` | 2 | 台本: `turns`（`speaker`・`text`・`type`（`speech` / `backchannel`）・読みの注記） |
| `dialogues_skipped.jsonl` | 2 | 作り直しても品質ゲートを通らなかった id |
| `voice_dialogues.jsonl` | 2 | Stage3 の入力（同じターン＋おおよそのタイミング・`pause_before_sec`・`overlap` などシナリオごとの項目） |
| `audio/plain/turns/<id>/` | 3 | 合成した発話のキャッシュ（配置の前） |
| `audio/maai/<id>.flac` | 3 | MaAI で配置したあと・調整の前 |
| `audio/maai_ref/<id>.flac` | 3 | **最終的な音声** |
| `transcripts_maai_ref.jsonl` | 3 | **最終的な時刻つき書き起こし** |
| `audio_manifest_maai_ref.jsonl` | 3 | ターンごとの詳細: 声の id・タイミング・開始の決め方（`retime.method`）・調整の履歴 |
| `voice_dialogues_refined_maai_ref.jsonl` | 3 | 調整後の台本。ターンが音声と1対1で対応する |
| `refine_ops/` | 3 | LLM による調整の操作のキャッシュ（やり直しで LLM に問い合わせない） |
| `generation.log`・`tts.log`・`cost_log.jsonl` | 1〜3 | ログとトークン数 |

## 生成データのライセンス

`data/<シナリオ>/` の中身はすべて、使った LLM・TTS モデル・ボイスバンク（シナリオによっては人物カードのデータセット）から
作られたものです。コードの MIT License ではなく、それらのライセンス・利用規約に従います。
[THIRD_PARTY.md](../THIRD_PARTY.md) と [voices/LICENSE.md](../voices/LICENSE.md) を参照してください。

## 音声

- FLAC・48 kHz・16 bit・**2 チャネル: ch0（左）= システム側、ch1（右）= ユーザ側**。
- 各チャネルにはその話者の声だけが入ります（回り込みなし）。間にはごく小さなピンクノイズ（−80 dBFS）。
- 最初の発話の前に 0.4〜6 秒、最後の発話の後に 1〜6 秒の無音をランダムに入れます（`tts.lead_in`・`tts.tail_pad`）。

## 書き起こし（`transcripts_maai_ref.jsonl`）

1対話1つの JSON:

```json
{"id": "6630d4cd-36ff-43df-8fff-bfa421064240", "audio": "6630d4cd-36ff-43df-8fff-bfa421064240.flac",
 "sample_rate": 48000, "channels": 2, "duration_sec": 130.833,
 "channel_speakers": {"0": "イノウエ", "1": "モリ"},
 "voice_prompts": {"モリ": {"voice_id": "mori_t2", "gender": "male", ...}, "イノウエ": {"voice_id": "inoue_t4", ...}},
 "segments": [
   {"start": 0.435, "end": 4.295, "speaker": "モリ", "channel": 1, "type": "speech",
    "text": "どうも。俺、モリって言います。よろしくお願いします。"},
   {"start": 4.273, "end": 8.103, "speaker": "イノウエ", "channel": 0, "type": "speech",
    "text": "はじめまして。イノウエです。こちらこそ、よろしくお願いします。"},
   ...,
   {"start": 25.916, "end": 26.466, "speaker": "イノウエ", "channel": 0, "type": "backchannel", "text": "はい。"},
   ...]}
```

`start` / `end` は音声ファイルの中の秒です。セグメントは開始時刻の順で、重なることがあります（声の重なり・相手のターン中の相槌）。
漢字の読みの注記（`ruby`・`yomi`）が付いたセグメントもあります。
