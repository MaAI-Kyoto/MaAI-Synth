# セットアップ

📄 [English](setup.md) | [日本語](setup_JP.md)

パイプラインはリポジトリのルートで動かす Python スクリプト群と、2つのサーバ（Stage1〜3 の **LLM サーバ** と Stage3 の
**TTS サーバ**）からなります。サーバは同じマシンでも別のマシンでもかまいません。

| 部品 | 既定 | ハードウェア | 環境 |
|---|---|---|---|
| Stage1〜2（`main.py`） | — | CPU | `requirements.txt` |
| Stage3（`generate_tts.py`） | MaAI（MaaiBatch）・ASR・話者の検査 | GPU 1枚・約 6 GB | `requirements-tts.txt` ＋ `./MaAI` |
| LLM サーバ | Gemma 4 31B（FP8）・vLLM・`:8001` | GPU 1枚・48 GB | vLLM を入れた環境 |
| TTS サーバ | Irodori-TTS-v4.1-Small-Yomi・vLLM-Omni・`:8980` | GPU 1枚・約 8 GB | 別の venv（Python 3.12） |

動作確認: Python 3.10（パイプライン）、Python 3.12 ＋ vllm 0.30.0 ＋ vllm-omni `a8576cc`（TTS サーバ）、
torch 2.13 / CUDA 12.6、transformers 5.5.3、MaAI `95ba113`。GPU は RTX 3090 / A6000 / L40S / RTX 6000 Ada。

## 1. パイプラインの環境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA に合うものを選ぶ
pip install -r requirements-tts.txt
bash scripts/setup_maai.sh            # ./MaAI に MaAI を取ってくる（Stage3 は MaAI/src を import する）
```

Stage3 は初回に次のモデルを Hugging Face Hub から取ってきます（置き場所は `HF_HOME` で変えられる）:
`maai-kyoto/*`（MaAI の VAP・相槌モデル）、`kyutai/mimi`（MaAI のエンコーダ）、
`kotoba-tech/kotoba-whisper-v2.0`（読みの検査）、`microsoft/wavlm-base-plus-sv`（話者の検査）。

スクリプトは必ず**リポジトリのルートで**実行してください（config は `voices/...` のような相対パスを使います）。

## 2. LLM サーバ（Gemma 4）

```bash
pip install vllm          # GPU のある環境で
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_llm.sh
curl -s http://localhost:8001/v1/models      # -> "google/gemma-4-31B-it"
```

`scripts/serve_llm.sh` は `RedHatAI/gemma-4-31B-it-FP8-dynamic` を `google/gemma-4-31B-it`（config の `model:`）という名前で、
文脈長 32k で立てます。Gemma の利用規約に従ってください。
複数のサーバ（GPU ごとに1台など）は `llm.base_urls` と `tts.refine.base_urls` に並べると、要求が振り分けられます。
別のモデルや OpenAI API を使うときは [extending_JP.md](extending_JP.md) を参照。

## 3. TTS サーバ（vLLM-Omni 上の Irodori-TTS）

```bash
python3.12 -m venv .venv-tts-server && source .venv-tts-server/bin/activate
bash scripts/setup_irodori_server.sh        # vLLM-Omni＋Irodori のパッチ・Irodori-TTS・重み・モデル dir
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_irodori.sh
curl -s http://localhost:8980/health        # 準備ができると 200（初回の起動は数分かかる）
```

セットアップの中身は [third_party/irodori_vllm/README_JP.md](../third_party/irodori_vllm/README_JP.md) に書いてあります。
サーバは参照音声の WAV を**パスで**開きます。Stage3 と同じマシンで動かすか、両方から同じパスで同じファイルが見えるようにしてください。
台数を増やすときは GPU ごとに1台ずつ立て（`PORT=8981 ...`）、`tts.fish_speech.base_url: http://localhost:8980,http://localhost:8981` と書きます。

## 4. 人物カードの母集団（一部のシナリオだけ）

`ft_v2`・`ft_sil_v2`・`al_v1`・`rp_v1`・`qa_v1` は、ユーザ側に `personas/partner_pool.jsonl` から引いた人物カードを付けます。
これは公開データセット2つから作ります: [personas_JP.md](personas_JP.md)。

## 5. 実行

```bash
python main.py        --config configs/ft_v1.yaml --target 2          # Stage1＋2
python generate_tts.py --config configs/ft_v1.yaml --retime-refine --gpu 0   # Stage3
```

`generate_tts.py` のおもなオプション:

| オプション | 意味 |
|---|---|
| `--retime-refine` | Stage3 の全体（TTS → MaAI で配置 → 相槌の調整 → MaAI で配置） |
| `--retime` | TTS と MaAI での配置だけ（LLM での調整なし・出力は `audio/maai/`） |
| `--limit N` | 処理する対話を N 件までにする |
| `--gpu 0` | MaAI・ASR・話者の検査に使う GPU（`cpu` でも動くが遅い） |
| `--fish-url URL[,URL]` | `tts.fish_speech.base_url` を上書き |
| `--claim` | 同じ config で複数のワーカーが仕事を分け合う（GPU ごとに1プロセス） |
| `--merge-parts` | 並列ワーカーのあとで `*.part-*.jsonl` をまとめる |
| `--dry-run` | TTS サーバなし（音声の代わりに正弦波・配線の確認用） |

Stage1〜2 はできている対話を飛ばすので、同じコマンドをもう一度実行すれば続きから再開できます。

## 並列に回す

同じ config で `--claim` を付けたワーカーを複数起動します（それぞれが1対話ずつ取っていく）:

```bash
for g in 0 1 2 3; do python generate_tts.py --config configs/ft_v1.yaml --retime-refine --claim --gpu $g & done; wait
python generate_tts.py --config configs/ft_v1.yaml --merge-parts
```

## 困ったとき

- **`maai` の import エラー** — `bash scripts/setup_maai.sh` を実行し、`MaAI/src/maai/batch.py` があることを確かめる。
  pip で入れた `maai` とリポジトリ版を同じプロセスで混ぜないこと。
- **LLM が 404 を返す** — config の `model:` とサーバの `--served-model-name` を一致させる。
- **要求がプロキシに飛ぶ** — サーバのホストを `no_proxy` / `NO_PROXY` に足す。
- **TTS サーバに「connection refused」** — `/health` が 200 を返すまで待つ（初回はカーネルのコンパイルがある）。
- **相槌の位置がおかしい** — `NVIDIA_TF32_OVERRIDE=0` のままにする（`generate_tts.py` が自動で設定）。TF32 だと MaAI の判定が変わる。
