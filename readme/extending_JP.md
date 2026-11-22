# ほかの LLM・TTS を使う

📄 [English](extending.md) | [日本語](extending_JP.md)

## LLM

テキスト生成（Stage1・Stage2・Stage2 の校閲・Stage3 の相槌の調整）はすべて OpenAI の Python SDK を通ります
（`pipeline/llm_client.py`・`pipeline/refine_maai.py`）。OpenAI の `chat.completions` API を話すサーバならどれでも使えます:
vLLM・SGLang・llama.cpp（`llama-server`）・Ollama・OpenAI API そのもの。

次のキーを変えます（`configs/base.yaml`、またはシナリオの config で上書き）:

```yaml
model: Qwen/Qwen3-32B                    # = サーバが出しているモデル名
llm:
  provider: local                        # local = OpenAI 互換サーバ、openai = OpenAI API
  model: Qwen/Qwen3-32B
  base_urls: [http://gpu1:8001/v1, http://gpu2:8001/v1]   # すべてのサーバに要求を振り分ける
  api_key_env: OPENAI_API_KEY            # キーを入れた環境変数の名前（local は何でも・無くてもよい）
  extra_body:                            # そのままサーバに渡す（「思考」を切る など）
    chat_template_kwargs: {enable_thinking: false}
tts:
  refine:
    text_model: Qwen/Qwen3-32B
    fallback_text_model: Qwen/Qwen3-32B
    base_urls: [http://gpu1:8001/v1]
```

OpenAI API なら `provider: openai` にして `base_urls` を消し、`model` / `text_model` を `gpt-5-mini` などにして
`OPENAI_API_KEY` を設定します。費用は `pricing`（100万トークンあたりの USD）で数え、`max_cost_usd` で止まります。

注意:

- `dialogue.prompt_profile` はモデルの系統ごとに調整したプロンプトの差分を選びます（同梱の config は `gemma`。`auto` は
  モデル名から推定）。新しいモデルではまず `auto` にして、`data/<シナリオ>/generation.log` のゲートの集計を見てください。
  目標から外れた対話は品質ゲートが作り直すので、指示に従いにくいモデルでは主に作り直しが増えます。
- 推論（「思考」）モデルは思考を切って使ってください。パーサは素の JSON・テキストを前提にしています。
- 同梱の設定は Gemma 4 31B で調整したものです。ほかのモデルでは `dialogue.enrich` の目標値の調整が要ることがあります。

## TTS

### TTS の方式の選び方

`tts.engine` は `tts_presets/` のファイル名です。`extends:` をすべて解いたあとで、その YAML を `tts` に deep-merge します
（`pipeline/tts_presets.py`）。つまり方式は、使い回せる TTS の設定のまとまりです。既定の `tts_presets/irodori-yomi-v4.yaml` は、
参照音声型の TTS の経路（`engine: fish_speech`）と `irodori_vllm` バックエンドを選びます（ブロック名が `fish_speech` なのは
歴史的な理由で、参照音声型のバックエンドすべての設定がここに入ります）。

既定の方式の設定を変えるときは、写して（`cp tts_presets/irodori-yomi-v4.yaml tts_presets/my-irodori.yaml`）書き換え、
`tts.engine: my-irodori` とします。作ったデータを再現できるよう、古いプリセットは書き換えないでください。

### 新しい TTS を HTTP サーバとして足す（おすすめ）

`moss_tts` バックエンド（`generate_tts.MossTtsSynth`）は汎用のクライアントで、1発話ごとに本文と話者の参照音声1本を
HTTP で送ります。自分の TTS をこの API の小さなサーバで包めば、Stage3 のほかの部分（短い発話のテイク選び・ASR による
読みの検査・話者の検査・MaAI での配置・相槌の調整）はそのまま動きます。

```
POST <base_url>/v1/audio/speech
  {"model": "...", "input": "<本文>", "ref_audio": "data:audio/wav;base64,...", "ref_text": "<参照の書き起こし or 空>",
   "seed": 123, "max_new_tokens": 300, "response_format": "wav", "mode": "clone"}
-> 200 audio/wav   （サンプルレートは何でもよい・48 kHz に変換する）
GET <base_url>/health -> 200
```

1. [`tools/tts_server_template.py`](../tools/tts_server_template.py) の `synthesize()` を実装し（標準ライブラリの HTTP サーバ。
   試験音を自分のモデルに置き換える）、起動する: `python tools/tts_server_template.py --port 8990`
2. [`tts_presets/example-http-tts.yaml`](../tts_presets/example-http-tts.yaml) を `tts_presets/<エンジン名>.yaml` に写す。
3. config に `tts: {engine: <エンジン名>, fish_speech: {base_url: http://localhost:8990}}` と書く
   （複数サーバはカンマ区切り。要求は振り分けられ、応答しないサーバは飛ばされる）。
4. `python pipeline/generate_tts.py --config ... --retime-refine --limit 1` で試し、`audio/maai_ref/` を聴く。

`moss_tts` バックエンドの便利な設定（`tts.fish_speech` の下）: `concurrency`、`moss_out_sr`（48 kHz にせずサーバのレートのまま）、
`moss_tempo`（全テイクの話速を変える）、`short_best_of`（短い発話は複数テイク）、`best_of`（全発話で複数テイクを作り
自然さ・話速・間で採点）、`regen_on_misread`（ASR が違う読みをしたテイクを引き直す）。

### 合成クラスを足す

参照音声を複数使う・文脈の音声を渡す・まとめて合成するなど、1要求1発話では足りないエンジンは、`pipeline/generate_tts.py` の
`MossTtsSynth` / `IrodoriVllmSynth` の隣にクラスを足し（`_post_speech`・テイク選び・文脈の経路の書き方の見本になる）、
`build_fish_speech_synth()` に新しい `backend` 名で登録します。`IrodoriVllmSynth` が既定の方式で使う実装で、
vLLM-Omni の Irodori パイプラインに `extra_params`（目標の長さ・声バンクの兄弟テイク `ref_wavs`・拡散の段数・CFG）を送ります。

コードにはほかのエンジンも残っています（この公開版では設定も確認もしていません）: fish-speech S2（`backend: fish` /
`vllm_omni`）、Qwen3-TTS（`engine: qwen3`）、プロセス内の Irodori-TTS（`engine: irodori`）、Sarashina2.2-TTS、
Gemini / Google Cloud TTS（プリセットの声）。

## タイミングのモデル

`tts.retime.backend` で発話の置き方を選びます。

| バックエンド | 中身 |
|---|---|
| `maaibatch`（既定） | MaAI 公式の一括推論 API `maai.MaaiBatch`（VAP・相槌モデル。`./MaAI/src` から import） |
| `maai` | MaAI のリアルタイム用 `Maai` クラスをオフラインで回す（pip の `maai`。遅い・互換のため残している） |
| `vapbatch` | 以前の自前の一括推論の再実装（この公開版には入れていない） |

VAP モデル・しきい値・ハザードと応答遅れのパラメータは `tts.retime` の下（`vap`・`bc`・`hazard`・`latency`）にあります。
