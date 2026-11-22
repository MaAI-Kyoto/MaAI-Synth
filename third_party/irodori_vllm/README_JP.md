# vLLM-Omni 上の Irodori-TTS サーバ

📄 [English](README.md) | [日本語](README_JP.md)

既定の TTS の方式（`tts_presets/irodori-yomi-v4.yaml`）は、[vLLM-Omni](https://github.com/vllm-project/vllm-omni) で立てた
[Irodori-TTS-v4.1-Small-Yomi](https://huggingface.co/takuma104/Irodori-TTS-v4.1-Small-Yomi)
（[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) v4.1 Small の漢字の読みを直したファインチューニング版。RF-DiT ＋ DACVAE・
48 kHz・日本語）に要求を送ります。vLLM-Omni は同時に来た要求を束にして処理するので、1発話ずつモデルを呼ぶより Stage3 が何倍も速くなります。

`scripts/setup_irodori_server.sh` は次のことをします（手で順に実行してもかまいません）。

| 手順 | 内容 |
|---|---|
| 1 | vLLM-Omni をコミット `a8576cc` で取ってきて [`vllm-omni-irodori.patch`](vllm-omni-irodori.patch) を当て、`pip install vllm==0.30.0` と vLLM-Omni の `pip install -e` |
| 2 | Irodori-TTS をコミット `3822056` で取ってくる（パイプラインはこのソースから `irodori_tts` を import する）。実行に要るパッケージを入れる |
| 3 | 重みを取ってきて、[`make_model_dir.py`](make_model_dir.py) でモデル dir を書く |

そのあと `scripts/serve_irodori.sh` が
`python -m vllm.entrypoints.cli.main serve <model_dir> --omni --port 8980 --served-model-name irodori --deploy-config vllm_omni/deploy/irodori_tts.yaml`
を実行します。

## パッチの中身

| ファイル | 変更 |
|---|---|
| `vllm_omni/diffusion/models/irodori/` | `IrodoriPipeline`: Irodori の `InferenceRuntime.synthesize_batch` を vLLM-Omni の拡散エンジンで回す。要求ごとの `extra_params`（`seconds`・`ref_wavs`・`max_ref_seconds`・`num_steps`・`cfg_scale_speaker`・`seed` など） |
| `vllm_omni/deploy/irodori_tts.yaml` | 拡散の1段・1束に最大 16 要求 |
| `vllm_omni/diffusion/registry.py`・`config/pipeline_registry.py` | パイプラインの登録 |
| `vllm_omni/entrypoints/openai/serving_speech.py` | 拡散の出力のサンプルレートを読む・要求のログを短く・`VLLM_OMNI_REF_AUDIO_MAX_SEC` |
| `vllm_omni/diffusion/executor/multiproc_executor.py` | 修正: 終わった束が受け取られずに残ることがあった（要求が 3600 秒後にタイムアウト） |

`helpers/irodori_fast.py` は符号化した参照音声をファイルごとにキャッシュし、60 秒の参照を発話ごとに符号化し直さないようにします（出力は同じ）。

## 注意

- **ファイルの共有。** Stage3 は声バンクのファイルをパスで送ります（`extra_params.ref_wavs`）。サーバがそのファイルを開ける必要があります
  （同じマシン、または共有ファイルシステムで同じパス）。相槌用の一時ファイルも `data/_tmp/irodori_prevref/` に書いて同じように渡します。
- **GPU メモリ。** 既定の束の大きさで約 6〜8 GB（DiT は bf16）。GPU ごとに1台立て、`tts.fish_speech.base_url` にカンマ区切りで並べます。
- **ライセンス。** Irodori-TTS-v4.1-Small-Yomi は MIT です。モデルカード（とベースモデルのモデルカード）は、なりすましと人を欺く
  ディープフェイクを禁止しています。本人の同意なく声をクローンしないこと。vLLM-Omni は Apache-2.0 で、パッチも vLLM-Omni と同じライセンスで配布します。
- **高速版。** 本番では手元で最適化した Irodori-TTS（weight norm の畳み込み・GPU 上のキャッシュ・長さで分けた束。RTX 3090 で約 1.4 倍）を
  使いました。ここには入れていません。上流の Irodori-TTS でも、束の組み方による誤差の範囲で同じ出力になります。
