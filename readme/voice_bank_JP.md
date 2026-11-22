# ボイスバンク

📄 [English](voice_bank.md) | [日本語](voice_bank_JP.md)

TTS は参照音声から声をクローンします。**ボイスバンク**は、1人の話者の約 30 秒の音声を入れたフォルダと、それを並べた CSV です。

```
voices/<バンク>/
├── voices.csv
└── wav/
    ├── <バンク>_t1.wav
    ├── <バンク>_t2.wav
    └── ...
```

```csv
id,gender,age_range,audio_path,text,restore,speech_db,sd_snr_db,f0_span
mori_t1,male,,voices/mori/wav/mori_t1.wav,その、そういうの作っていて、…,sidon,-23.0,42.5,10.4
```

| 列 | 必須 | 意味 |
|---|---|---|
| `id` | ○ | 一意な id。`<話者>_t<N>`（同じ話者のテイクは頭が同じ） |
| `gender` | ○ | `male` / `female`。台本の話者の性別と合わせる |
| `age_range` | | `20代`・`30代` …（Stage1 が書く年代）。空ならどの年代にも合う |
| `audio_path` | ○ | リポジトリのルートからの相対パス |
| `text` | | 音声の書き起こし（既定の Irodori の方式では使わない。ほかのエンジンで使うものがある） |
| ほかの列 | | 自由なメタデータ（`restore`・`speech_db`・`sd_snr_db`・`f0_span` は参考情報） |

## 入っているバンク

| バンク | 話者 | 音声 | 備考 |
|---|---|---|---|
| `voices/inoue/` | Inoue・男性・30代 | 27〜32 秒 × 10 | Sidon で復元・48 kHz |
| `voices/mori/` | Mori・男性 | 32〜33 秒 × 5 | 会話の収録から切り出し・Sidon で復元・48 kHz |

利用条件は [voices/LICENSE.md](../voices/LICENSE.md) にあります。

## バンクの使われ方

- `tts.main_voices_file` — **システム側**（ch0・固定の主話者）専用のバンク。
- `tts.voices_file` — **ユーザ側**（ch1）のバンク。どちらもパス1つかリストで書けます:

  ```yaml
  tts:
    main_voices_file: [voices/inoue/voices.csv]
    voices_file: [voices/mori/voices.csv, voices/alice/voices.csv, voices/bob/voices.csv]
  ```

- 対話ごとに、各話者に `gender` が合う（設定していれば `age_range` が `tts.voice_age_tolerance` 歳以内の）音声を1つ割り当てます。
  選び方は対話の id で決まります。合う性別の声が残っていないと別の声を使って警告を出すので、女性の話者が出るシナリオでは
  両方の性別の声を足してください（同梱の config はユーザ側の声が男性しかないので、`metadata.pair_mix` を男性同士だけにしています）。
- 既定の Irodori の方式は、TTS に約 60 秒の参照を渡します: 選んだ音声と、同じ話者のほかのテイク（同じフォルダの `<話者>_t*.wav`）。
  バンクは `voices/` の下の WAV のあるフォルダをすべて探して見つけます（`tts.fish_speech.irodori.voices_dirs` でも指定できる）。

## 自分の録音からバンクを作る

```bash
python tools/make_voice_bank.py --bank alice --gender female --age 20代 alice_rec1.wav alice_rec2.wav
# 任意: --sidon（音声の復元・GPU 推奨）、--asr（kotoba-whisper で書き起こし）
```

発話を検出して 25〜33 秒のテイクにまとめ、音量をそろえて（ピーク −1 dBFS・発話 −23 dBFS 以下）、`voices/alice/voices.csv` を書きます。
その CSV を `tts.voices_file`（または `tts.main_voices_file`）に足してください。

同梱のバンクを作ったときのこつ:

- **1人の声だけのきれいな音声。** ほかの声（会話の収録で相手の相槌がかぶっているなど）もクローンされます。会話の話者別トラックは、
  回り込みが小さければ（30 dB 以上低い）使えます。
- **自然な会話の音声**のほうが、朗読より生き生きした合成対話になります。参照の話し方（声の高さの幅・テンポ・フィラー）が写ります。
- 1人あたり **3〜10 テイク**。多いほど対話ごとの変化が増えます。1発話には合計約 60 秒を使います。
- **Sidon**（`--sidon`）は雑音と残響を取り除いて 48 kHz にしますが、話者の似かたは少し落ちます。
- 結果は `python manage_voices.py validate --voices voices/alice/voices.csv` で確かめられます。

**同意について。** 声のクローンに同意した人の録音だけを使い、TTS モデルのライセンスに従ってください
（Irodori-TTS のモデルカードは、なりすましと人を欺くディープフェイクを禁止しています）。
