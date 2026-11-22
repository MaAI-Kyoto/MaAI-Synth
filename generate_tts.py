"""Stage 4: 音声対話テキスト → 音声対話データ（TTS合成）

Stage 3 で生成した音声対話テキスト（output/voice_dialogues.jsonl）を、
TTS（sbintuitions/sarashina2.2-tts）で実際の音声に合成する。

特徴:
  - 2チャネル出力。各チャネルに各話者を割り当てる（ch0=話者A, ch1=話者B）。
  - sarashina2.2-tts はゼロショット音声クローン方式のため、話者ごとに
    「参照音声(wav) + その文字起こしテキスト」のボイスバンクを用意し、
    対話内の2話者に異なる声を割り当てる。
  - 発話の配置タイミングは Stage 3 の start_offset（直前発話終了からの相対秒、
    負=オーバーラップ）を、合成後の「実際の発話長」に対して連鎖適用して再構築する。
    これにより推定 duration と実音声長のズレを吸収し、間・かぶせを実音声で再現する。

TTS本体（sarashina_tts パッケージ）が未インストールでも配線・タイムライン合成を
検証できるよう、--dry-run（正弦波プレースホルダ合成）を用意している。

使い方:
    # 本番（要 sarashina_tts インストール & GPU）
    python generate_tts.py

    # 少量テスト
    python generate_tts.py --limit 5

    # モデルなしで配線/タイムライン検証（正弦波）
    python generate_tts.py --dry-run --limit 3
"""

import os as _os
import socket as _socket

# ★numba（librosa.pyin 等）のコンパイル済みキャッシュはホストごとに分ける（2026-11-14）。既定では NFS 上の venv の
#   site-packages/librosa/__pycache__ に置かれ、AVX-512 のある機械で作られたものを
#   持たない機械が読むと pyin の中で segfault し、ワーカーが痕跡なく落ちる。
_os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{_socket.gethostname()}")
# ★MaAI の判定は TF32 で動く（相槌の挿入位置がずれる）。CUDA の初期化より前に TF32 を切る（本番は起動スクリプトで設定）。
#   Public release: set here so that every entry point gets it.
_os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import tts_reading
from rp_audio import prepare_cut_turns

from speaker_gender import infer_gender, normalize_gender
from retime_maai import MaaiRetimer, MockRetimer, RetimeConfig
from fish_tagger import FishTagger, FishTagError, DEFAULT_TAG_RETRY_MAX

logger = logging.getLogger(__name__)

# sarashina2.2-tts (FlowDecoder.sample_rate) の出力サンプルレート
DEFAULT_SAMPLE_RATE = 24000

# TTSへ渡すテキストは句読点で終わる必要がある
_SENTENCE_END = "。！？.!?"


def setup_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _deep_merge(base: dict, over: dict) -> dict:
    """over を base に再帰マージ（dict同士はマージ、それ以外は over 優先）。"""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, _raw: bool = False) -> dict:
    """設定を読み込む。トップレベル `extends: <path>` があれば、その設定をベースに
    deep-merge する（config_tri.yaml が config.yaml を継承してパスだけ上書きする等）。
    最後に TTS 方式の名前（tts.engine: pocket-ja-nc-v1 など）を tts_presets/ の中身に展開する（tts_presets.apply・2026-11-15）。"""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = cfg.pop("extends", None)
    if base_path:
        base_full = Path(path).parent / base_path
        cfg = _deep_merge(load_config(str(base_full), _raw=True), cfg)
    if _raw:
        return cfg
    import tts_presets
    return tts_presets.apply(cfg)


def _inherit_ruby_llm(config: dict) -> None:
    """tts.fish_speech.regen_on_misread.ruby に投げ先・モデルが書いてなければ、トップレベルの llm:
    （Stage1-2 と同じ LLM）を使う。設定ごとに Stage2 と同じ LLM で読みが付く。"""
    m = (((config.get("tts") or {}).get("fish_speech") or {}).get("regen_on_misread"))
    if not isinstance(m, dict) or not m.get("ruby"):
        return
    r = m["ruby"] if isinstance(m["ruby"], dict) else {"enabled": True}
    llm = config.get("llm") or {}
    if not (r.get("base_urls") or r.get("hosts")):
        if llm.get("base_urls"):
            r["base_urls"] = list(llm["base_urls"])
        elif llm.get("hosts"):
            r["hosts"] = list(llm["hosts"])
            if llm.get("ports"):
                r["ports"] = llm["ports"]
        elif llm.get("base_url"):
            r["base_urls"] = [llm["base_url"]]
    r.setdefault("model", llm.get("model"))
    r.setdefault("extra_body", llm.get("extra_body") or {})
    m["ruby"] = r


VOICE_CSV_FIELDS = ["id", "gender", "age_range", "audio_path", "text"]


class VoiceBank(list):
    """声のリスト＋バンク単位の設定。中身は今までどおり dict の list。

    age_tol / age_min_pool は assign_voices の年代の緩め方（tts.voice_age_tolerance,
    tts.voice_age_min_pool）。声の並び方ではなくバンクの性質なので、voices を受け取る
    全経路（compose / synthesize_turns / select_best_takes / refine_maai /
    gemini_batch …）に引数を増やさずに済むよう、リストに持たせている。
    素の list を渡された場合は 0.0 / 0（＝従来の厳密一致）として扱う。

    main_voice_ids は「主話者(ch0)専用に予約した声のid」（tts.main_voices_file）。
    空なら従来どおり全員が同じプールから選ぶ。非空なら assign_voices が、
    主話者はこの中からのみ、それ以外はこの外からのみ選ぶ。
    """

    age_tol: float = 0.0
    age_min_pool: int = 0
    main_voice_ids: tuple = ()

    def __init__(self, rows=(), age_tol: float = 0.0, age_min_pool: int = 0,
                 main_voice_ids=()):
        super().__init__(rows)
        self.age_tol = float(age_tol or 0.0)
        self.age_min_pool = int(age_min_pool or 0)
        self.main_voice_ids = tuple(main_voice_ids or ())


def load_voices_csv(path: str | Path) -> list[dict]:
    """ボイスバンクCSVを読み込む。空のid行はスキップ。"""
    voices = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get("id") or "").strip()
            if not vid:
                continue
            voices.append({
                "id": vid,
                "gender": (row.get("gender") or "").strip() or None,
                "age_range": (row.get("age_range") or "").strip() or None,
                "audio_path": (row.get("audio_path") or "").strip(),
                "text": (row.get("text") or "").strip(),
            })
    return voices


# プリセット音声名を使うエンジン（参照音声クローンではない）
PRESET_VOICE_ENGINES = ("gemini", "google_cloud")


def load_voices(tts_cfg: dict) -> VoiceBank:
    """有効なエンジンに応じたボイス一覧を返す。

    - gemini / google_cloud: tts.<engine>.voices（プリセット音声名）。id未指定ならvoice_nameをidに。
    - qwen3: tts.qwen3.voices（プリセット話者 speaker）。id未指定なら speaker を id に。
    - irodori / sarashina: voices_file(CSV) を優先、無ければインライン tts.voices（参照音声）。

    返すのは list 互換の VoiceBank で、tts.voice_age_tolerance /
    tts.voice_age_min_pool を持たせる（assign_voices の年代の緩め方。
    既定 0＝従来の厳密一致）。

    tts.main_voices_file があれば、そのCSVを追加で読んで主話者(ch0)専用プールにする。
    ★ここは _load_voice_rows と違って「無ければ黙ってフォールバック」しない。
      綴り間違いで素通りすると、ch0 に一般バンクの声が付いたまま1万件焼けてしまう。
    """
    rows = _load_voice_rows(tts_cfg)
    main_ids: tuple = ()
    mf = tts_cfg.get("main_voices_file")
    if isinstance(mf, (list, tuple)):   # public release: several main-speaker banks
        main_rows = []
        for one in mf:
            if not Path(one).exists():
                raise FileNotFoundError(f"tts.main_voices_file が見つかりません: {one}")
            main_rows += load_voices_csv(one)
        if not main_rows:
            raise ValueError(f"tts.main_voices_file に有効な行がありません: {mf}")
        have = {v["id"] for v in rows}
        rows = list(rows) + [v for v in main_rows if v["id"] not in have]
        main_ids = tuple(dict.fromkeys(v["id"] for v in main_rows))
        mf = None
    if mf:
        if not Path(mf).exists():
            raise FileNotFoundError(
                f"tts.main_voices_file が見つかりません: {mf}"
                "（Stage4 はリポジトリのルートから起動する前提。相対パスを確認）")
        main_rows = load_voices_csv(mf)
        if not main_rows:
            raise ValueError(f"tts.main_voices_file に有効な行がありません: {mf}")
        have = {v["id"] for v in rows}
        rows = list(rows) + [v for v in main_rows if v["id"] not in have]
        main_ids = tuple(v["id"] for v in main_rows)
        logger.info(f"主話者(ch0)専用バンク {len(main_ids)}本を併合: {mf}")
    return VoiceBank(rows,
                     age_tol=float(tts_cfg.get("voice_age_tolerance") or 0.0),
                     age_min_pool=int(tts_cfg.get("voice_age_min_pool") or 0),
                     main_voice_ids=main_ids)


def _load_voice_rows(tts_cfg: dict) -> list[dict]:
    """load_voices の中身（エンジンごとのボイス一覧の作り方）。"""
    engine = tts_cfg.get("engine", "gemini")
    if engine in PRESET_VOICE_ENGINES:
        voices = []
        for v in (tts_cfg.get(engine, {}).get("voices") or []):
            v = dict(v)
            v.setdefault("voice_name", v.get("id"))
            v.setdefault("id", v.get("voice_name"))
            voices.append(v)
        return voices
    if engine == "qwen3":
        voices = []
        for v in (tts_cfg.get("qwen3", {}).get("voices") or []):
            v = dict(v)
            v.setdefault("speaker", v.get("id"))
            v.setdefault("id", v.get("speaker"))
            voices.append(v)
        return voices
    if engine == "fish_speech":
        # reference_id 版が指定されていればそれを、無ければ voices_file(参照音声)にフォールバック
        fv = tts_cfg.get("fish_speech", {}).get("voices") or []
        if fv:
            voices = []
            for v in fv:
                v = dict(v)
                v.setdefault("id", v.get("reference_id"))
                voices.append(v)
            return voices
        # フォールバックは下の voices_file(CSV)（audio_path+text の参照音声クローン）
    vf = tts_cfg.get("voices_file")
    if isinstance(vf, (list, tuple)):   # public release: several voice banks (one CSV per bank)
        rows, seen = [], set()
        for one in vf:
            if one and Path(one).exists():
                for v in load_voices_csv(one):
                    if v["id"] not in seen:
                        seen.add(v["id"])
                        rows.append(v)
            elif one:
                raise FileNotFoundError(f"tts.voices_file: {one} が見つかりません")
        return rows
    if vf and Path(vf).exists():
        return load_voices_csv(vf)
    return tts_cfg.get("voices", [])


def build_google_cloud_synth(tts_cfg: dict) -> "GoogleCloudTTSSynth":
    """config の tts.google_cloud から Google Cloud TTS 合成器を作る。"""
    gcfg = tts_cfg.get("google_cloud", {})
    api_key = None
    if gcfg.get("api_key_env"):
        api_key = os.environ.get(gcfg["api_key_env"])
    return GoogleCloudTTSSynth(
        language_code=gcfg.get("language_code", "ja-JP"),
        audio_encoding=gcfg.get("audio_encoding", "LINEAR16"),
        sample_rate=int(gcfg.get("sample_rate", 24000)),
        concurrency=int(gcfg.get("concurrency", 8)),
        rpm=(float(gcfg["rpm"]) if gcfg.get("rpm") else None),
        prompt=gcfg.get("prompt"),
        api_key=api_key,
        model=gcfg.get("model"),
        max_retries=int(gcfg.get("max_retries", 5)),
        speaking_rate=(float(gcfg["speaking_rate"]) if gcfg.get("speaking_rate") else None),
    )


DEFAULT_GEMINI_STYLE = (
    "次のセリフを、自然な日本語の話し言葉として、感情豊かに抑揚をつけて、"
    "実際の会話のように話してください"
)


def build_gemini_synth(tts_cfg: dict) -> "GeminiSynth":
    """config の tts.gemini からGemini合成器を作る。APIキーは環境変数から。"""
    gcfg = tts_cfg.get("gemini", {})
    api_key_env = gcfg.get("api_key_env", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(f"Gemini APIキーが見つかりません。環境変数 {api_key_env} か GOOGLE_API_KEY を設定してください。")
        sys.exit(1)
    return GeminiSynth(
        model=gcfg.get("model", "gemini-2.5-flash-preview-tts"),
        style_prompt=gcfg.get("style_prompt", DEFAULT_GEMINI_STYLE),
        api_key=api_key,
        max_retries=int(gcfg.get("max_retries", 6)),
        concurrency=int(gcfg.get("concurrency", 4)),
        rate_limit_max_wait=float(gcfg.get("rate_limit_max_wait", 60.0)),
        rpm=(float(gcfg["rpm"]) if gcfg.get("rpm") else None),
    )


def build_irodori_synth(tts_cfg: dict) -> "IrodoriSynth":
    """config の tts.irodori から Irodori 合成器を作る。"""
    icfg = tts_cfg.get("irodori", {})
    dcfg = icfg.get("duration", {}) or {}
    rate_target = float((tts_cfg.get("rate") or {}).get("target_mora_per_sec", 7.3))
    return IrodoriSynth(
        duration_mode=dcfg.get("mode", "auto"),
        duration_scale=float(dcfg.get("scale", 1.0)),
        duration_min=float(dcfg.get("min_sec", 0.4)),
        duration_max=float(dcfg.get("max_sec", 30.0)),
        duration_mora_per_sec=float(dcfg.get("target_mora_per_sec", rate_target)),
        duration_trim_keep=float(dcfg.get("trim_keep", 0.94)),
        checkpoint=icfg.get("checkpoint", "Aratako/Irodori-TTS-500M-v3"),
        device=icfg.get("device"),
        precision=icfg.get("precision", "fp32"),
        codec_repo=icfg.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim"),
        caption=icfg.get("caption"),
        sampling=icfg.get("sampling"),
        checkpoint_file=icfg.get("checkpoint_file", "model.safetensors"),
    )


def build_qwen3_synth(tts_cfg: dict, device: str | None = None) -> "Qwen3Synth":
    """config の tts.qwen3 から Qwen3-TTS 合成器を作る（Apache 2.0・ローカルGPU）。"""
    qcfg = tts_cfg.get("qwen3", {})
    return Qwen3Synth(
        checkpoint=qcfg.get("checkpoint", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        device=device or qcfg.get("device", "cuda:0"),
        dtype=qcfg.get("dtype", "bfloat16"),
        attn_implementation=qcfg.get("attn_implementation", "flash_attention_2"),
        language=qcfg.get("language", "Japanese"),
        instruct=qcfg.get("instruct"),
    )


def fish_checkpoint_name(fcfg: dict) -> str:
    """tts.fish_speech.checkpoint（サーバが読んでいるはずのモデルのディレクトリ名）。既定はオリジナルの s2-pro。

    run_tts_pool.sh はこの名前と一致するモデルを読んでいる fish サーバだけを使う
    （ファインチューニング版とオリジナルを同じマシンに並べても混ざらない）。
    """
    return str(fcfg.get("checkpoint") or "s2-pro").rstrip("/").rsplit("/", 1)[-1]


def build_fish_speech_synth(tts_cfg: dict) -> "FishSpeechSynth":
    """config の tts.fish_speech から fish-speech(HTTP API) 合成器を作る。"""
    fcfg = tts_cfg.get("fish_speech", {})
    # backend: fish（既定。fish-speech の api_server）/ vllm_omni（vLLM-Omni の /v1/audio/speech）
    backend = str(fcfg.get("backend", "fish") or "fish")
    if backend not in ("fish", "vllm_omni", "moss_tts", "irodori_vllm"):
        raise ValueError(f"tts.fish_speech.backend は fish / vllm_omni / moss_tts / irodori_vllm（受領: {backend!r}）")
    extra = {}
    if backend in ("vllm_omni", "moss_tts", "irodori_vllm"):
        if fcfg.get("context_audio"):
            raise ValueError("tts.fish_speech.backend: vllm_omni は context_audio 未対応（false にする）")
        # vLLM はリクエストの model とサーバの --served-model-name が違えば弾くので、
        # 別のモデル（ファインチューニング版）のサーバに投げてしまう事故はここで止まる
        extra["model"] = str(fcfg.get("vllm_model") or fish_checkpoint_name(fcfg))
    if backend == "irodori_vllm":
        # Irodori-TTS を vLLM-Omni（IrodoriPipeline）で。長さ・参照の兄弟テイクはクライアントが決めて extra_params で渡す
        extra["irodori"] = dict(fcfg.get("irodori") or {})
    if backend in ("moss_tts", "irodori_vllm"):
        extra["moss_mode"] = str(fcfg.get("moss_mode", "clone") or "clone")
        extra["bwe"] = bool(fcfg.get("bwe", False))
        extra["commas"] = str(fcfg.get("moss_commas", "keep") or "keep")
    if backend == "moss_tts":
        extra["tempo"] = float(fcfg.get("moss_tempo", 1.0) or 1.0)
        extra["short_best_of"] = dict(fcfg.get("short_best_of") or {})
        extra["laugh_respell"] = bool(fcfg.get("moss_laugh_respell", False))
        extra["filler_respell"] = bool(fcfg.get("moss_filler_respell", False))
        if fcfg.get("pocket_temp") is not None:
            extra["pocket_temp"] = float(fcfg["pocket_temp"])
        if fcfg.get("moss_out_sr"):
            extra["out_sr"] = int(fcfg["moss_out_sr"])
        pc = dict(fcfg.get("pocket_context") or {})
        if pc:
            extra["ctx_self_turns"] = int(pc.get("self_turns", 0) or 0)
            extra["ctx_prompt_index"] = pc.get("prompt_index")
            extra["ctx_spk_guard"] = pc.get("spk_guard", {})
    synth = {"vllm_omni": VllmOmniSynth, "moss_tts": MossTtsSynth,
             "irodori_vllm": IrodoriVllmSynth}.get(backend, FishSpeechSynth)(
        base_url=fcfg.get("base_url", "http://localhost:8888"),
        endpoint=fcfg.get("endpoint", "/v1/tts"),
        fmt=fcfg.get("format", "wav"),
        reference_id=fcfg.get("reference_id"),
        gen_kwargs=fcfg.get("gen"),
        timeout=float(fcfg.get("timeout", 300)),
        use_msgpack=bool(fcfg.get("use_msgpack", True)),
        multiturn=bool(fcfg.get("multiturn", False)),
        endpoint_multiturn=fcfg.get("endpoint_multiturn", "/v1/tts_multiturn"),
        window=int(fcfg.get("window", 0) or 0),
        window_context=int(fcfg.get("window_context", 0) or 0),
        per_speaker=bool(fcfg.get("per_speaker", False)),
        context_audio=bool(fcfg.get("context_audio", False)),
        context_anchor=bool(fcfg.get("context_anchor", False)),
        context_mode=str(fcfg.get("context_mode", "history")),
        use_proxy=bool(fcfg.get("use_proxy", False)),
        concurrency=int(fcfg.get("concurrency", 1) or 1),
        use_memory_cache=str(fcfg.get("use_memory_cache", "on")),
        **extra,
    )
    # context_fixed: 文脈を「最初の窓の採用テイク」に固定する（連鎖のドリフトを止める）。
    synth.context_fixed = bool(fcfg.get("context_fixed", False))
    # サー（先頭フィラー直後の広帯域ヒス）検知→再合成の設定を持たせる（構成依存で任意）。
    synth.regen_hiss = fcfg.get("regen_on_hiss")
    # 末尾の余分な1音（入力していない音）検知→seed 替えて引き直しの設定。
    synth.regen_tail = fcfg.get("regen_on_tail")
    # 読み違い（ASR で聞き取った読みが台本と食い違う）検知→seed 替えて引き直しの設定。
    synth.regen_misread = fcfg.get("regen_on_misread")
    synth.regen_speaker = fcfg.get("regen_on_speaker_leak")
    # テイク選択（同じターンを複数回引いて中庸なものを採る）の設定。
    synth.best_of = fcfg.get("best_of")
    # 主話者(ch0)だけ生成パラメータを変える（例: main_gen: {temperature: 0.6}）。
    # 主話者の判定は main_voices_file の参照音声で行うので、それが無い config では使えない。
    main_gen = {k: v for k, v in (fcfg.get("main_gen") or {}).items() if v is not None}
    if main_gen:
        mf = tts_cfg.get("main_voices_file")
        if not mf or not Path(mf).exists():
            raise ValueError("tts.fish_speech.main_gen は tts.main_voices_file（主話者専用バンク）と一緒に使う")
        texts = frozenset(v["text"] for v in load_voices_csv(mf) if v.get("text"))
        if not texts:
            raise ValueError(f"tts.main_voices_file に書き起こし(text)のある行がありません: {mf}")
        synth.main_gen, synth.main_ref_texts = main_gen, texts
        logger.info(f"主話者(ch0)だけ生成パラメータを上書き: {main_gen}（参照 {len(texts)}本で判定）")
        if synth.multiturn and backend == "fish":
            logger.warning("main_gen は fish 本体の multiturn（全話者1リクエスト）では効かない")
    # 読み間違え対策（読み辞書・数字のかな化）。合成入力だけに効く。
    import tts_reading
    tts_reading.configure(fcfg.get("reading_normalize", False))
    if fcfg.get("reading_normalize"):
        logger.info(f"読みの正規化: 有効（{fcfg.get('reading_normalize')}）")
    return synth


GEN_KEYS = ("temperature", "top_p", "repetition_penalty", "do_sample", "max_length")


def build_gen_kwargs(tts_cfg: dict, overrides: dict | None = None) -> dict | None:
    """config の tts.gen から生成パラメータ(gen_kwargs)を組み立てる。

    話し方の単調さは主に参照音声のスタイルで決まるが、temperature を上げると
    抑揚・変化が増え（単調さが減り）、下げると安定するが平板になりやすい。
    repetition_penalty を少し上げると平坦な繰り返しを避けやすい。
    """
    g = dict(tts_cfg.get("gen") or {})
    if overrides:
        g.update({k: v for k, v in overrides.items() if v is not None})
    kw = {k: g[k] for k in GEN_KEYS if g.get(k) is not None}
    return kw or None


def normalize_tts_text(text: str) -> str:
    """TTS入力用にテキストを整える（末尾に句点を補う）。

    tts.fish_speech.reading_normalize が有効なら、読み間違えやすい語と数字をかなに開く
    （tts_reading.py。合成入力だけが変わり、manifest・字幕は原文のまま）。
    """
    text = text.strip()
    if not text:
        return "。"
    import tts_reading
    text = tts_reading.apply_if_enabled(text)
    if text[-1] not in _SENTENCE_END:
        text += "。"
    return text


def _time_stretch_numpy(wav: np.ndarray, rate: float,
                        n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """位相ボコーダによるピッチ保持タイムストレッチ（numpyのみ、依存なし）。

    rate>1 で速く（短く）、rate<1 で遅く（長く）なる。librosa 非依存の
    フォールバック実装。極短発話（1フレーム未満）はそのまま返す。
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < n_fft + hop:
        return wav  # 短すぎる（相槌の一部など）はそのまま
    win = np.hanning(n_fft).astype(np.float32)
    # 解析STFT
    n_frames = 1 + (len(wav) - n_fft) // hop
    stft = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        seg = wav[i * hop:i * hop + n_fft] * win
        stft[:, i] = np.fft.rfft(seg)
    mag = np.abs(stft)
    phase = np.angle(stft)
    # 合成フレームを rate 倍の時間刻みで補間（位相は累積）
    bins = np.arange(n_fft // 2 + 1)
    expected = 2 * np.pi * bins * hop / n_fft   # 各ビンの1ホップあたり基準位相増分
    time_steps = np.arange(0, n_frames - 1, rate)
    out_phase = phase[:, 0].copy()
    synth = np.zeros(int(len(time_steps)) * hop + n_fft, dtype=np.float32)
    wsum = np.zeros_like(synth)
    for k, t in enumerate(time_steps):
        i0 = int(np.floor(t))
        frac = t - i0
        m = (1.0 - frac) * mag[:, i0] + frac * mag[:, i0 + 1]
        # 瞬時周波数から位相前進量を求める（principal argument）
        dphi = phase[:, i0 + 1] - phase[:, i0] - expected
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        seg = np.fft.irfft((m * np.exp(1j * out_phase)).astype(np.complex64), n=n_fft)
        s = k * hop
        synth[s:s + n_fft] += seg.astype(np.float32) * win
        wsum[s:s + n_fft] += win * win
        out_phase = out_phase + expected + dphi
    nz = wsum > 1e-8
    synth[nz] /= wsum[nz]
    return synth[:int((len(wav)) / rate) + n_fft].astype(np.float32)


def time_stretch(wav: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """発話をピッチ保持で rate 倍速にする（rate>1 で速く=短く）。

    librosa があればそれを使い、無ければ numpy 実装にフォールバックする。
    rate が 1.0 付近なら無処理。
    """
    if wav is None or abs(rate - 1.0) < 1e-3 or len(wav) < 1024:
        return wav  # 等速・空・極短（相槌の一部等）はそのまま
    try:
        import librosa
        return np.asarray(
            librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32), rate=rate),
            dtype=np.float32)
    except Exception:
        return _time_stretch_numpy(wav, rate)


def time_stretch_wsola(wav: np.ndarray, rate: float, sr: int,
                       frame_ms: float = 40.0, tol_ms: float = 10.0,
                       decim: int = 4) -> np.ndarray:
    """WSOLA（波形類似オーバーラップ加算）でピッチ保持タイムストレッチする。

    rate>1 で速く（短く）、rate<1 で遅く（長く）。位相ボコーダ（time_stretch）と違い
    波形をそのまま繋ぐので、音声で位相のにじみ（にごり・残響感）が出ない。
    実測: 0.8倍に伸ばしたとき sox tempo（WSOLA の定番実装）との log-mel 距離が
    1.35dB。同じ条件で位相ボコーダは 3.44dB で、こちらのほうが素直に伸びる。

    各出力フレームは「理想の分析位置 ± tol」の範囲から、直前に採用したフレームの
    続きに最も似た区間を選んで貼る（正規化相互相関。探索は decim 間引き）。
    """
    if wav is None or abs(rate - 1.0) < 1e-3 or len(wav) < 2048:
        return wav
    w = np.asarray(wav, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    n_frame = max(256, int(sr * frame_ms / 1000.0) // 2 * 2)
    hop_s = n_frame // 2                       # 合成ホップ
    hop_a = hop_s * rate                       # 分析ホップ（実数。rate倍で進める）
    tol = max(1, int(sr * tol_ms / 1000.0))
    win = np.hanning(n_frame).astype(np.float32)
    n_corr = hop_s                             # 相関を取る長さ（重なる区間だけで足りる）

    out = np.zeros(int(len(w) / rate) + 2 * n_frame, dtype=np.float32)
    norm = np.zeros_like(out)
    pos, prev_tail, k = 0.0, None, 0
    while True:
        ideal = int(round(pos))
        if ideal + n_frame >= len(w):
            break
        if prev_tail is None:
            best = ideal
        else:
            lo = max(0, ideal - tol)
            hi = min(len(w) - n_frame, ideal + tol)
            if hi <= lo:
                best = min(ideal, len(w) - n_frame)
            else:
                seg = w[lo:hi + n_corr]
                sw = np.lib.stride_tricks.sliding_window_view(
                    seg, n_corr)[:hi - lo + 1:decim]
                num = sw @ prev_tail[:n_corr]
                den = np.sqrt((sw ** 2).sum(axis=1) + 1e-9)
                best = lo + int(np.argmax(num / den)) * decim
        frame = w[best:best + n_frame]
        if len(frame) < n_frame:
            break
        o = k * hop_s
        out[o:o + n_frame] += frame * win
        norm[o:o + n_frame] += win
        prev_tail = w[best + hop_s:best + hop_s + n_frame]
        if len(prev_tail) < n_corr:
            break
        pos += hop_a
        k += 1
    if k == 0:
        return w
    n = min(len(out), k * hop_s + n_frame)
    out, norm = out[:n], norm[:n]
    np.divide(out, norm, out=out, where=norm > 1e-6)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Synthesizer 抽象（本番 / dry-run を差し替え可能にする）
# --------------------------------------------------------------------------

class DummySynth:
    """モデルなしで使うプレースホルダ合成器。

    文字数からおおよその長さを見積もり、声ごとに異なる周波数の小音量正弦波を返す。
    タイムライン合成・2チャネル配置・再開処理の検証用。
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        sr = self.sample_rate
        # 声ごとに基音を変える（聞き分け用）
        base_freq = 180.0 + (abs(hash(voice["id"])) % 120)
        out = []
        for t in texts:
            n_chars = max(1, len(t))
            dur = max(0.3, n_chars / 7.0)  # ~7モーラ/秒の粗い見積もり
            n = int(dur * sr)
            tt = np.arange(n, dtype=np.float32) / sr
            wav = 0.15 * np.sin(2 * np.pi * base_freq * tt).astype(np.float32)
            # 自然な切れ目のため両端を軽くフェード
            fade = min(int(0.02 * sr), n // 2)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                wav[:fade] *= ramp
                wav[-fade:] *= ramp[::-1]
            out.append(wav)
        return out


class SarashinaSynth:
    """sarashina2.2-tts ラッパー。話者(声)ごとに参照特徴量をキャッシュする。"""

    def __init__(self, model_dir: str | None = None, use_vllm: bool = False,
                 device: str | None = None, watermark: bool = True,
                 gen_kwargs: dict | None = None):
        from sarashina_tts.generate.generate import SarashinaTTSGenerator
        from sarashina_tts.flow_matching.decoder import FlowDecoder

        self.generator = SarashinaTTSGenerator(
            model_dir=model_dir, use_vllm=use_vllm, device=device, watermark=watermark)
        self.sample_rate = int(FlowDecoder.sample_rate)
        self.gen_kwargs = gen_kwargs or None
        self._feat_cache: dict[str, dict] = {}

    def _features(self, voice: dict) -> dict:
        vid = voice["id"]
        if vid not in self._feat_cache:
            path = voice["audio_path"]
            if not Path(path).exists():
                raise FileNotFoundError(f"参照音声が見つかりません (voice={vid}): {path}")
            g = self.generator
            self._feat_cache[vid] = {
                "audio_prompt_tokens": g._extract_audio_prompt_tokens(audio_prompt_path=path),
                "flow_embedding": g._extract_zero_shot_embedding(audio_prompt_path=path),
                "audio_prompt_feat": g._extract_audio_prompt_feat(audio_prompt_path=path),
                "audio_prompt_text": voice["text"],
                "audio_prompt_path": path,
            }
        return self._feat_cache[vid]

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        feat = self._features(voice)
        gen_args = dict(
            flow_embedding=feat["flow_embedding"],
            audio_prompt_text=feat["audio_prompt_text"],
            audio_prompt_tokens=feat["audio_prompt_tokens"],
            audio_prompt_feat=feat["audio_prompt_feat"],
            audio_prompt_path=feat["audio_prompt_path"],
            flow_embedding_only=False,
        )
        if self.gen_kwargs:
            gen_args["gen_kwargs"] = self.gen_kwargs
        wavs = self.generator.generate(texts, **gen_args)
        # generate() は List[torch.Tensor]（mono, shape (1, samples) 等）を返す
        out = []
        for w in wavs:
            arr = w.detach().cpu().numpy().astype(np.float32).squeeze()
            out.append(np.atleast_1d(arr))
        return out


class _RateLimiter:
    """RPM(requests per minute)を超えないよう、リクエスト開始を間引く（スレッド安全）。

    各スレッドは acquire() で「次に送ってよい時刻」のスロットを予約し、そこまで待つ。
    並列(concurrency)で動いても、リクエスト開始間隔が 60/rpm 秒以上に保たれる。
    """

    def __init__(self, rpm: float | None):
        self.min_interval = (60.0 / rpm) if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_time)
            self._next_time = start + self.min_interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


class GeminiSynth:
    """Gemini TTS ラッパー（シングル話者・1発話=1API呼び出し）。

    参照音声クローンではなくプリセット音声(voice_name)を使う。会話調・抑揚は
    style_prompt（自然言語の指示）で制御する。出力は 24kHz / mono / 16bit PCM。
    """

    def __init__(self, model: str = "gemini-2.5-flash-preview-tts",
                 style_prompt: str = "", api_key: str | None = None,
                 max_retries: int = 6, retry_base: float = 2.0,
                 concurrency: int = 4, rate_limit_max_wait: float = 60.0,
                 rpm: float | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.style_prompt = (style_prompt or "").strip()
        self.sample_rate = 24000
        self.max_retries = max_retries
        self.retry_base = retry_base
        self.concurrency = max(1, int(concurrency))
        self.rate_limit_max_wait = rate_limit_max_wait
        # RPMスロットル（429を出す前に送信ペースを制御）
        self._limiter = _RateLimiter(rpm)
        # API呼び出しを並列化（I/Oバウンドなのでスレッドプール）
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)

    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        s = str(e).lower()
        return (getattr(e, "code", None) == 429 or "429" in s
                or "resource_exhausted" in s or "too many requests" in s
                or "rate limit" in s or "quota" in s)

    @staticmethod
    def _retry_delay_hint(e: Exception, default: float, cap: float) -> float:
        """エラーメッセージ内の retryDelay 等を尊重。無ければ default を cap で丸める。"""
        m = re.search(r"retry.?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s",
                      str(e), re.IGNORECASE)
        wait = float(m.group(1)) if m else default
        return min(max(wait, 1.0), cap)

    def _synth_one(self, text: str, voice_name: str) -> np.ndarray:
        types = self._types
        # 「指示: セリフ」形式。指示部分は読み上げられず、口調だけ反映される。
        contents = f"{self.style_prompt}: {text}" if self.style_prompt else text
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=contents, config=config)
        data = resp.candidates[0].content.parts[0].inline_data.data
        if isinstance(data, str):  # 念のため base64 文字列にも対応
            import base64
            data = base64.b64decode(data)
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return np.atleast_1d(pcm)

    def _synth_one_retry(self, text: str, voice_name: str) -> np.ndarray:
        last_err = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()  # RPMを超えないようペース制御（リトライも1回として数える）
            try:
                return self._synth_one(text, voice_name)
            except Exception as e:
                last_err = e
                if self._is_rate_limit(e):
                    # 429: 指定があれば従い、無ければ長めの指数バックオフ（上限まで）
                    wait = self._retry_delay_hint(
                        e, self.retry_base * (2 ** (attempt + 2)), self.rate_limit_max_wait)
                    logger.warning(f"Gemini レート制限(429)。{wait:.0f}秒待機して再試行 "
                                   f"(attempt {attempt + 1}/{self.max_retries})")
                else:
                    wait = min(30.0, self.retry_base * (2 ** attempt))
                time.sleep(wait)
        raise last_err

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        voice_name = voice.get("voice_name") or voice.get("id")
        # 発話を並列合成（順序は保持）
        futures = [self._pool.submit(self._synth_one_retry, t, voice_name) for t in texts]
        return [f.result() for f in futures]


_PERMANENT_API_ERRORS = (
    "PermissionDenied", "Unauthenticated", "NotFound", "InvalidArgument",
    "BadRequest", "Forbidden", "FailedPrecondition",
)


def _is_permanent_api_error(e: Exception) -> bool:
    """認可・不正引数・不存在など、リトライしても無駄な恒久的エラーか。"""
    if type(e).__name__ in _PERMANENT_API_ERRORS:
        return True
    code = getattr(e, "code", None)
    code = getattr(code, "value", code)
    return code in (400, 401, 403, 404)


class GoogleCloudTTSSynth:
    """Google Cloud Text-to-Speech ラッパー（Gemini APIとは別サービス）。

    プリセット音声(voice_name 例: ja-JP-Neural2-B)を使う。クォータが大きく
    Gemini プレビューTTSのような厳しいRPM制限が無いのが利点。出力は LINEAR16(WAV)。
    認証は ADC（GOOGLE_APPLICATION_CREDENTIALS のサービスアカウント or
    `gcloud auth application-default login`）。API キーでも可。
    """

    def __init__(self, language_code: str = "ja-JP", audio_encoding: str = "LINEAR16",
                 sample_rate: int = 24000, concurrency: int = 8, rpm: float | None = None,
                 prompt: str | None = None, api_key: str | None = None,
                 model: str | None = None, max_retries: int = 5, retry_base: float = 2.0,
                 speaking_rate: float | None = None):
        from google.cloud import texttospeech
        self._tts = texttospeech
        if api_key:
            from google.api_core.client_options import ClientOptions
            self.client = texttospeech.TextToSpeechClient(
                client_options=ClientOptions(api_key=api_key))
        else:
            self.client = texttospeech.TextToSpeechClient()
        self.language_code = language_code
        self.sample_rate = int(sample_rate)
        self.model = (model or "").strip() or None   # 例: gemini-2.5-flash-tts（Gemini-TTS）
        self.prompt = (prompt or "").strip() or None
        self.max_retries = max_retries
        self.retry_base = retry_base
        # 話速: AudioConfig.speaking_rate（1.0=等速、>1で速く）。ネイティブ制御で音質劣化なし。
        # Gemini-TTS/古典音声(Neural2等)とも対応。None=指定しない（等速）。
        self.speaking_rate = float(speaking_rate) if speaking_rate else None
        self._enc = getattr(texttospeech.AudioEncoding, audio_encoding)
        self._limiter = _RateLimiter(rpm)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency)))

    def _make_input(self, text: str, prompt: str | None = None):
        tts = self._tts
        p = prompt if prompt is not None else self.prompt
        if p:  # Chirp3-HD / Gemini-TTS 音声のみ対応。非対応なら無視して続行
            try:
                return tts.SynthesisInput(text=text, prompt=p)
            except (TypeError, ValueError):
                pass
        return tts.SynthesisInput(text=text)

    def _synth_one(self, text: str, voice: dict, prompt: str | None = None) -> np.ndarray:
        import io
        tts = self._tts
        vname = voice.get("voice_name") or voice.get("id")
        lang = voice.get("language_code") or self.language_code
        model = voice.get("model") or self.model
        vparams = dict(language_code=lang, name=vname)
        if model:  # Gemini-TTS等はモデル指定が必要。古典音声(Neural2等)は付けない
            vparams["model_name"] = model
        aparams = dict(audio_encoding=self._enc, sample_rate_hertz=self.sample_rate)
        if self.speaking_rate is not None:  # ネイティブ話速指定（音質劣化なし）
            aparams["speaking_rate"] = self.speaking_rate
        resp = self.client.synthesize_speech(
            input=self._make_input(text, prompt),
            voice=tts.VoiceSelectionParams(**vparams),
            audio_config=tts.AudioConfig(**aparams),
        )
        # LINEAR16 は WAV ヘッダ付きで返る → soundfile でデコード
        data, _ = sf.read(io.BytesIO(resp.audio_content), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.atleast_1d(np.asarray(data, dtype=np.float32))

    def _synth_one_retry(self, text: str, voice: dict,
                         prompt: str | None = None) -> np.ndarray:
        last_err = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            try:
                return self._synth_one(text, voice, prompt)
            except Exception as e:
                last_err = e
                if _is_permanent_api_error(e):  # 認可/不正引数/不存在は即失敗
                    raise
                time.sleep(min(30.0, self.retry_base * (2 ** attempt)))
        raise last_err

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        # prompts[i] があれば発話ごとに口調・文脈プロンプトを差し替える（短い相槌の自然化）
        futures = [self._pool.submit(self._synth_one_retry, t, voice,
                                     (prompts[i] if prompts else None))
                   for i, t in enumerate(texts)]
        return [f.result() for f in futures]


class IrodoriSynth:
    """Irodori-TTS ラッパー（ローカルGPU・ゼロショット音声クローン）。

    参照音声(ref_wav)で話者を指定する（文字起こしは不要）。出力は 48kHz mono。
    モデルは1度だけロードし、発話ごとに synthesize する。
    VoiceDesignモデル使用時は caption で口調を指示できる。
    https://github.com/Aratako/Irodori-TTS
    """

    def __init__(self, checkpoint: str = "Aratako/Irodori-TTS-500M-v3",
                 device: str | None = None, precision: str = "fp32",
                 codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
                 caption: str | None = None, sampling: dict | None = None,
                 checkpoint_file: str = "model.safetensors",
                 duration_mode: str = "auto", duration_scale: float = 1.0,
                 duration_min: float = 0.4, duration_max: float = 30.0,
                 duration_mora_per_sec: float = 7.3,
                 duration_trim_keep: float = 0.94):
        import irodori_tts.inference_runtime as _ir
        from irodori_tts.inference_runtime import (
            InferenceRuntime, RuntimeKey, SamplingRequest, default_runtime_device,
        )
        self._SamplingRequest = SamplingRequest
        dev = str(device or default_runtime_device())

        # checkpoint がローカルファイルでなければ HF リポジトリIDとみなして取得。
        # ★v4 以降は tokenizer/ がリポジトリに同梱されていて、model.safetensors だけ
        #   落とすと tokenizer が見つからない。download_hf_checkpoint（v4 で追加）は
        #   model.safetensors と tokenizer/* をまとめて取るのでそちらを優先する。
        ckpt = Path(checkpoint).expanduser()
        if ckpt.is_file():
            checkpoint_path = str(ckpt)
        elif hasattr(_ir, "download_hf_checkpoint"):
            checkpoint_path = _ir.download_hf_checkpoint(checkpoint)
        else:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=checkpoint, filename=checkpoint_file)

        self.runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=dev,
            codec_repo=codec_repo,
            model_precision=precision,
            codec_device=dev,
            codec_precision=precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        ))
        self.caption = caption or None
        self.sampling = dict(sampling or {})
        self.sample_rate = 48000  # Irodoriの出力は48kHz（初回合成で確定値に更新）
        # ★発話長の決め方（tts.irodori.duration.mode）
        #   "auto"     … モデルの duration predictor に任せる（Irodori の既定）
        #   "mora"     … モーラ数 ÷ 目標話速。既定。人間コーパスと同じ物差しで決まる
        #   "estimate" … Stage3 の推定長（turns[].duration）をそのまま seconds にする
        # v4.1-Small は参照音声を与えると duration predictor がテキスト長をほぼ無視し、
        # 何を入れても約3.9秒を返す（実測: 「はい。」96.7フレーム / 24字の発話 98.8
        # フレーム。no_ref だと 23.5 対 140.3 と正しく効く）。相槌だらけのこのコーパス
        # では短い発話が全部4秒の作り話になるので、既定は estimate 側で使う。
        self.duration_mode = str(duration_mode or "auto").lower()
        self.duration_scale = float(duration_scale)
        self.duration_min = float(duration_min)
        self.duration_max = float(duration_max)
        self.duration_mora_per_sec = float(duration_mora_per_sec)
        self.duration_trim_keep = float(duration_trim_keep)
        # synthesize_for_speaker から est_durs を渡してもらうための宣言
        self.wants_durations = self.duration_mode == "estimate"

    def _seconds_for(self, text: str, est: float | None) -> float | None:
        """このターンに指示する長さ（秒）。None ならモデルの予測に任せる。

        mode="mora" は「モーラ数 ÷ 目標話速」。目標は tts.rate.target_mora_per_sec と
        同じ 7.3（人間コーパス maai-f2f-jp-2025 の実測 7.29）。
        ★Stage3 の turns[].duration をそのまま使う（mode="estimate"）と速くなりすぎる。
          あれは「実質文字数 ÷ 6.5」で、しかも実質文字数は読点・句点・長音を
          **全部落として**数えている（generate_voice_passthrough._CORE_RE）。読点の多い
          話し言葉ほど短く見積もられ、実測で 8.3 モーラ/秒（人間 7.29 / fish 7.38）まで
          速くなった。モーラで測れば漢字の読み長も読点も正しく効く。
        """
        if self.duration_mode == "mora":
            mora = estimate_mora(text)[0]
            # 指示した窓のうち、モデルは前後に無音を作り、compose の trim_silence が
            # それを落とす。実測（713ターン）で残るのは窓の 93.6%。窓をそのまま
            # mora/7.3 にすると聞こえる話速は 7.80 になるので、削られるぶんを見込んで
            # 窓を広げる。＝ target_mora_per_sec は「聞こえる話速」を指す。
            sec = mora / max(self.duration_mora_per_sec, 1e-6)
            sec /= min(max(self.duration_trim_keep, 0.5), 1.0)
        elif self.duration_mode == "estimate":
            if est is None or est <= 0:
                est = _core_len(text) / 6.5
            sec = est
        else:
            return None
        return float(min(max(sec * self.duration_scale, self.duration_min),
                         self.duration_max))

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None,
              durations: list[float] | None = None) -> list[np.ndarray]:
        ref_wav = voice.get("audio_path")
        out = []
        for i, t in enumerate(texts):
            est = durations[i] if (durations and i < len(durations)) else None
            sec = self._seconds_for(t, est)
            kw = dict(self.sampling)
            if sec is not None and "seconds" not in kw:
                kw["seconds"] = sec
            req = self._SamplingRequest(
                text=t, ref_wav=ref_wav, caption=self.caption, **kw)
            res = self.runtime.synthesize(req)
            self.sample_rate = int(getattr(res, "sample_rate", self.sample_rate))
            audio = res.audio
            if hasattr(audio, "detach"):  # torch.Tensor
                # precision: bf16/fp16 のとき numpy() が落ちるので float32 に寄せてから渡す
                audio = audio.detach().float().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            out.append(np.atleast_1d(audio))
        return out


class Qwen3Synth:
    """Qwen3-TTS ラッパー（ローカルGPU・Apache 2.0・open-weight）。

    プリセット話者(speaker)で合成する。日本語対応（Ono_Anna=日本語女性 等、男声は
    多言語話者＋language=Japanese のクロスリンガル）。instruct でスタイルを自然言語指示。
    https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    """

    def __init__(self, checkpoint: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                 device: str = "cuda:0", dtype: str = "bfloat16",
                 attn_implementation: str = "flash_attention_2",
                 language: str = "Japanese", instruct: str | None = None):
        import torch
        from qwen_tts import Qwen3TTSModel
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "fp16": torch.float16, "bf16": torch.bfloat16,
              "float32": torch.float32, "fp32": torch.float32}.get(dtype, torch.bfloat16)
        kwargs = dict(device_map=device, dtype=dt)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = Qwen3TTSModel.from_pretrained(checkpoint, **kwargs)
        except Exception:
            kwargs.pop("attn_implementation", None)  # flash-attn 無い環境向けフォールバック
            self.model = Qwen3TTSModel.from_pretrained(checkpoint, **kwargs)
        self.language = language
        self.instruct = (instruct or "").strip() or None
        self.sample_rate = 24000  # 初回合成で確定値に更新

    def _one(self, text: str, speaker: str, instruct: str | None) -> np.ndarray:
        wavs, sr = self.model.generate_custom_voice(
            text=text, language=self.language, speaker=speaker,
            instruct=(instruct if instruct is not None else self.instruct))
        self.sample_rate = int(sr)
        w = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(w, "detach"):  # torch.Tensor
            w = w.detach().cpu().numpy()
        return np.atleast_1d(np.asarray(w, dtype=np.float32).squeeze())

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        speaker = voice.get("speaker") or voice.get("voice_name") or voice.get("id")
        out = []
        for i, t in enumerate(texts):
            instr = prompts[i] if (prompts and prompts[i]) else None
            out.append(self._one(t, speaker, instr))
        return out


class FishSpeechSynth:
    """fish-speech（OpenAudio）ラッパー。ローカルの HTTP API サーバに投げる方式。

    重い多段パイプライン（text2semantic + firefly）は別プロセス（api_server）が担い、
    本クラスは requests で /v1/tts に POST するだけ（GPU不要・依存軽量・maaiと非衝突）。
    声は reference_id（音声モデルID）or 参照音声(voices_file の audio_path+text)クローン。
    事前に GPUマシンで: python tools/api_server.py --listen 0.0.0.0:8888
    ライセンス: FISH AUDIO RESEARCH LICENSE（研究/非商用系）。
    """

    def __init__(self, base_url: str = "http://localhost:8888",
                 endpoint: str = "/v1/tts", fmt: str = "wav",
                 reference_id: str | None = None, gen_kwargs: dict | None = None,
                 timeout: float = 300.0, use_msgpack: bool = True,
                 multiturn: bool = False,
                 endpoint_multiturn: str = "/v1/tts_multiturn",
                 window: int = 0, window_context: int = 0,
                 per_speaker: bool = False,
                 context_audio: bool = False,
                 context_anchor: bool = False,
                 context_mode: str = "history",
                 use_proxy: bool = False,
                 concurrency: int = 1,
                 use_memory_cache: str = "on"):
        import requests
        self._requests = requests
        self.base = base_url.rstrip("/")
        self.url = self.base + endpoint
        self.multiturn_url = self.base + endpoint_multiturn
        self.multiturn = bool(multiturn)
        # per_speaker: 話者ごとに独立した generate_long パスを回し、そのパスでは対象話者
        # のターンだけ採用する（他話者ターンは文脈条件付けのため生成し破棄）。マルチターン
        # (全話者1ストリーム)と違い、各話者の最終音声が別ストリーム由来なので声色が混ざり
        # にくい。合成経路(synth_dialogue/窓分割)はマルチターンと共用する。
        self.per_speaker = bool(per_speaker)
        self.window = int(window or 0)                 # 1コールあたりの対象ターン数（<=0で対話全体）
        self.window_context = max(0, int(window_context or 0))  # 窓先頭に足す文脈ターン数
        # context_audio: 窓の文脈を「テキストから再生成して捨てる」のではなく、前の窓で
        # 実際に採用済みの音声そのものを context として渡す（要サーバパッチ: ServeTTSRequest.context）。
        # 参照音声(ボイスプロンプト)は毎コール先頭に貼り直されたまま、対象ターンは
        # 「本物の直前音声」に接続するよう条件付けられる（窓境界の別テイク不連続が消える）。
        self.context_audio = bool(context_audio)
        # context_anchor: context の先頭に参照音声を「フェイク直前ターン」としても注入する。
        # 採用済みターンの連鎖（context_audio）は、直近文脈への追従がsystemの参照より強い
        # ためドリフトが複利で増幅する（A/Bで確認: 後半のF0が暴れる）。参照を直近履歴にも
        # 置くことで毎コール強く引き戻し、連鎖の利得を1未満に抑える。context_audio時のみ有効。
        self.context_anchor = bool(context_anchor)
        # context_mode: context の効かせ方（context_audio時のみ意味を持つ）。
        #   "history"   = user/assistant の直前ターンとして挿入（軌道追従が強い。
        #                 連鎖でドリフトが複利増幅することをA/Bで確認済み）
        #   "reference" = system の参照音声に連結（話者見本の追加。拡散的・スタイル
        #                 レベルの条件付けで、元参照が長さで支配的なため利得が弱い）
        self.context_mode = str(context_mode or "history")
        # context_fixed: 文脈を「直前の採用テイク」ではなく「最初の窓の採用テイク（固定）」
        # にする。文脈が定数になるため連鎖＝フィードバックループが完全に消える（利得0）。
        # 連続性は直前接続ではなく「対話内の自分の声の一貫性強化」として効く。
        self.context_fixed = False
        self.fmt = fmt
        self.reference_id = reference_id or None
        self.gen_kwargs = dict(gen_kwargs or {})
        # 主話者(ch0)の参照で合成するリクエストだけに重ねる生成パラメータ（tts.fish_speech.main_gen）。
        # 主話者かどうかは参照音声の書き起こし（main_voices_file の text）で見分ける。
        # build_fish_speech_synth が設定する。空なら従来どおり全話者が gen_kwargs。
        self.main_gen: dict = {}
        self.main_ref_texts: frozenset = frozenset()
        self.temp_pinned = False         # temperature_override 中（引き直しの温度を main_gen より優先）
        self.timeout = timeout
        self.use_msgpack = use_msgpack
        # 1話者ぶんのターンを fish へ同時に何本まで投げるか（1 = 従来どおり逐次）。
        # ★fish は「使用率92%」でも実効帯域は 192〜206GB/s（RTX PRO 6000 Blackwell の
        #   公称 1.6〜1.8TB/s に対し約12%）。nvidia-smi の使用率は「カーネルが載っている
        #   時間の割合」で、計算の埋まり具合ではない。実測（1件あたり実効）:
        #     同時1 → 3.2〜4.3秒 / 同時2 → 2.22秒 / 同時4 → 2.05秒
        #   2〜4本で飽和する（各リクエストが約1.8秒間隔で完了＝部分的にしか重ならない）。
        # ★温度0.8の確率的生成なので、同時化すると音声そのものは変わる（品質の期待値は同じ）。
        #   既存出力とのbit一致は失われるので、混ぜたくなければ出力先を分けること。
        self.concurrency = max(1, int(concurrency or 1))
        # 参照音声（30秒・2.6MB）の VQ エンコード結果をサーバ側で使い回す。
        # fish の既定は "off"（schema.py:102）で、毎リクエスト参照をエンコードし直す。
        # "on" にすると sha256 をキーに **同一の prompt_tokens** を返すだけなので
        # 出力は変わらない（reference_loader.py:99-126。"off" でも書き込みはしている）。
        # 実測（node24・本番トラフィックの無いテストサーバ）:
        #   往復 4.17秒 → 3.83秒（-0.34秒/件・8〜15%）。1対話で fish 約60回なので
        #   20秒前後の短縮。fish 時間の 84〜91% は生成そのものでここは削れない。
        # ★出力不変はビット一致で確認済み（seed 固定で /v1/tts 12件・
        #   /v1/tts_multiturn 6件、全件 sha256 一致）。
        if use_memory_cache not in ("on", "off"):
            raise ValueError(
                f"use_memory_cache は 'on' か 'off'（受領: {use_memory_cache!r}）")
        self.use_memory_cache = use_memory_cache
        # 内部/LANのTTSサーバへは既定でプロキシを経由しない（http_proxy 等を無視）。
        # base_url を GPU機のIP直結にしても no_proxy 未設定で事故らないための保険。
        # requests.Session はスレッドセーフを保証しないので、スレッドごとに1本持つ。
        self._use_proxy = bool(use_proxy)
        self._local = threading.local()
        self.sample_rate = 44100  # 初回デコードで確定

    @property
    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._requests.Session()
            s.trust_env = self._use_proxy
            self._local.session = s
        return s

    def _pack(self, body: dict):
        """ormsgpack があれば msgpack、無ければ JSON でエンコード。(data, headers) を返す。"""
        if self.use_msgpack:
            try:
                import ormsgpack
                opt = getattr(ormsgpack, "OPT_SERIALIZE_PYDANTIC", 0)
                return ormsgpack.packb(body, option=opt), "application/msgpack"
            except ImportError:
                pass
        return json.dumps(body).encode("utf-8"), "application/json"

    def _gen_for(self, ref_text: str | None) -> dict:
        """1リクエストぶんの生成パラメータ。主話者の参照（ref_text）なら main_gen を重ねる。

        temperature_override（サー対策の引き直し・テイクの温度振り）の最中はその温度を優先する。
        seed だけの引き直し（末尾の1音・読み違い・取り違え）は主話者の温度のまま。
        """
        if not self.main_gen or ref_text not in self.main_ref_texts:
            return self.gen_kwargs
        over = dict(self.main_gen)
        if self.temp_pinned:
            over.pop("temperature", None)
        return {**self.gen_kwargs, **over}

    def _one(self, text: str, voice: dict) -> np.ndarray:
        import io
        body = {"text": text, "format": self.fmt, "streaming": False,
                "use_memory_cache": self.use_memory_cache}
        body.update(self._gen_for(voice.get("text")))
        ref_id = voice.get("reference_id") or self.reference_id
        if ref_id:
            body["reference_id"] = ref_id
        else:  # 参照音声クローン（voices_file の audio_path + text）
            ap = voice.get("audio_path")
            refs = []
            if ap and Path(ap).exists():
                with open(ap, "rb") as f:
                    refs.append({"audio": f.read(), "text": voice.get("text", "")})
            body["references"] = refs
        data, ctype = self._pack(body)
        resp = self._session.post(self.url, data=data,
                                  headers={"Content-Type": ctype}, timeout=self.timeout)
        resp.raise_for_status()
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sample_rate = int(sr)
        return np.atleast_1d(np.asarray(audio, dtype=np.float32))

    def synth(self, texts: list[str], voice: dict,
              prompts: list[str] | None = None) -> list[np.ndarray]:
        """1話者ぶんのターンを合成する。concurrency>1 で同時に投げる。

        `ThreadPoolExecutor.map` は入力順で結果を返すので、**出力の順序は逐次時と同じ**。
        例外も逐次時と同じく最初の失敗が呼び出し側へ上がる（呼び出し側の
        synthesize_for_speaker が個別合成へフォールバックする）。ただし同時化した分、
        例外が上がる前に後続のリクエストも送信済みになる点だけが違う。
        """
        if self.concurrency <= 1 or len(texts) <= 1:
            return [self._one(t, voice) for t in texts]
        n = min(self.concurrency, len(texts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(lambda t: self._one(t, voice), texts))

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        import io
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.atleast_1d(np.asarray(audio, dtype=np.float32)), int(sr)

    @staticmethod
    def _encode_wav(wav: np.ndarray, sr: int) -> bytes:
        """numpy波形を WAV バイト列にする（context/references 送信用）。"""
        import io
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue()

    def synth_dialogue(self, tagged_text: str, references: list[dict],
                       context: list[dict] | None = None,
                       return_codes: bool = False):
        """マルチターン: <|speaker:i|> タグ付きの対話全体を1回で合成する。

        直前ターンの生成済み音声を文脈として引き継ぎつつ（サーバ側 generate_long）、
        ターンごとに分離した波形リストを返す（入力ターンと同順・同数）。
        references は話者インデックス順の [{"audio": bytes, "text": str}, ...]。
        context は採用済み直前ターン（対話順）。各要素は {"text": タグ付きテキスト} に
        "codes"（生成時のVQトークンそのまま。無劣化・推奨）か "audio"（wavバイト列。
        サーバでVQ再エンコード。1ホップなら無害だが連鎖すると劣化が蓄積）を添える。
        サーバ側で system（参照）直後に user/assistant の本物の履歴として挿入され、
        生成は「実際に出力に載る直前音声」に接続する（要サーバパッチ。未対応サーバは
        無視するので text には対象ターンだけを渡すこと）。
        return_codes=True で (wavs, codes) を返す（codes[i] は wavs[i] のVQトークン。
        次コールの context に渡して無劣化の文脈連鎖を作る。未対応サーバでは None）。
        """
        import ormsgpack
        body = {
            "text": tagged_text,
            "format": self.fmt,
            "references": references,
            "streaming": False,      # サーバ側で内部的に streaming 収集する
            "max_speakers": 1,       # 1ターン=1バッチ=1セグメント
            "use_memory_cache": self.use_memory_cache,
        }
        if context:
            body["context"] = context
            body["context_mode"] = self.context_mode
        # /v1/tts_multiturn は全話者を1リクエストで生成するので、話者別の main_gen は
        # 参照が1本（＝1話者だけの合成）のときしか効かない。
        body.update(self._gen_for(references[0].get("text") if len(references) == 1 else None))
        data, ctype = self._pack(body)
        resp = self._session.post(self.multiturn_url, data=data,
                                  headers={"Content-Type": ctype}, timeout=self.timeout)
        resp.raise_for_status()
        obj = ormsgpack.unpackb(resp.content)
        self.sample_rate = int(obj["sample_rate"])
        out = []
        for wav_bytes in obj["segments"]:
            audio, _sr = self._decode_wav(wav_bytes)
            out.append(audio)
        if return_codes:
            codes = obj.get("codes")
            if not codes or len(codes) != len(out):
                codes = [None] * len(out)
            return out, codes
        return out


class VllmOmniSynth(FishSpeechSynth):
    """fish-speech S2-Pro を vLLM-Omni（/v1/audio/speech）で合成する（tts.fish_speech.backend: vllm_omni）。

    vLLM-Omni は複数リクエストを1回のデコードで束ねて処理する（continuous batching）。fish 本体の
    api_server は1件ずつしか処理しないので、同時本数を上げるほど差が開く（2026-10-21 の実測・同じ
    RTX 6000 Ada で、同時4本 2.7倍・同時32本 5.7倍）。concurrency は大きめ（16〜）にする。

    サーバは vllm-omni/ の dialog_synth パッチ版を VLLM_OMNI_FISH_NATIVE=1 で立て、リクエストに
    extra_params.fish_native_sampling を付ける。これで fish 本体と同じプロンプト・同じ手順のサンプリング
    （温度前の top_p/top_k・eos_bias・RAS。コードブック側も行ごとの温度・top_p・top_k）になる。

    synth_dialogue（multiturn）は fish 本体の /v1/tts_multiturn と同じ手順をクライアント側で回す:
    話者タグで分けた1ターンずつ、全話者の参照とそのコールで生成済みのターン（本文＋サーバが返した
    音声トークンそのもの）を履歴に付けて生成する。窓どうし・Pass B の対象どうしは独立なので、
    parallel_windows で並列に投げてサーバに束ねさせる。
    ★context（採用済み音声を文脈にする context_audio）は未対応。
    ★gen の repetition_penalty・chunk_length・normalize は送らない（fish 本体でも
      repetition_penalty と normalize は効いておらず、chunk_length は max_speakers=1 では効かない）。
    """

    supports_multiturn = True
    parallel_windows = True

    def __init__(self, *args, model: str = "s2-pro", **kwargs):
        super().__init__(*args, **kwargs)
        self.url = self.base + "/v1/audio/speech"
        self.model = model
        self._ref_url_cache: dict[int, str] = {}

    def _ref_data_url(self, audio: bytes) -> str:
        """参照音声を data URI にする（サーバは文字列のハッシュで DAC エンコード結果を使い回す）。"""
        import base64
        key = hash(audio)
        url = self._ref_url_cache.get(key)
        if url is None:
            url = "data:audio/wav;base64," + base64.b64encode(audio).decode()
            if len(self._ref_url_cache) > 256:
                self._ref_url_cache.clear()
            self._ref_url_cache[key] = url
        return url

    def _post_speech(self, text: str, refs: list[dict], history: list[dict] | None = None,
                     return_codes: bool = False):
        """1回の生成。refs は話者の順の [{"audio": bytes, "text": str}]。history は fish_history の形。"""
        # 生成するのは text の話者タグが指す1話者（タグ無し＝参照1本の単発合成）
        m = re.match(r"<\|speaker:(\d+)\|>", text)
        k = int(m.group(1)) if m else 0
        gk = self._gen_for(refs[k].get("text") if k < len(refs) else None)
        ep = {
            "fish_native_sampling": True,
            "temperature": float(gk.get("temperature", 0.8)),
            "top_p": float(gk.get("top_p", 0.8)),
            "top_k": int(gk.get("top_k", 30)),
            "eos_bias": float(gk.get("eos_bias", 0.0)),
        }
        body = {"model": self.model, "input": text, "voice": "default", "response_format": "wav",
                "max_new_tokens": int(gk.get("max_new_tokens", 1024)), "extra_params": ep}
        if history is None and not return_codes and len(refs) == 1:
            body["ref_audio"] = self._ref_data_url(refs[0]["audio"])     # /v1/tts と同じ1参照の形
            body["ref_text"] = refs[0].get("text", "")
        else:
            ep["fish_refs"] = [{"audio": self._ref_data_url(r["audio"]), "text": r.get("text", "")} for r in refs]
            ep["fish_history"] = list(history or [])
            ep["fish_return_codes"] = True
        if gk.get("seed") is not None:
            body["seed"] = int(gk["seed"])
        resp = self._session.post(self.url, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM-Omni {resp.status_code}: {resp.text[:300]}")
        audio, sr = self._decode_wav(resp.content)
        self.sample_rate = sr
        if not return_codes:
            return audio
        shape = resp.headers.get("X-Fish-Codes-Shape")
        b64 = resp.headers.get("X-Fish-Codes")
        if not shape or not b64:
            raise RuntimeError("vLLM-Omni が音声トークンを返さない（VLLM_OMNI_FISH_NATIVE=1 のパッチ版サーバか確認）")
        return audio, (b64, [int(x) for x in shape.split(",")])

    def _one(self, text: str, voice: dict) -> np.ndarray:
        ap = voice.get("audio_path")
        if voice.get("reference_id") or self.reference_id or not ap or not Path(ap).exists():
            raise ValueError("vLLM-Omni backend は参照音声クローン（voices_file の audio_path + text）だけ対応")
        with open(ap, "rb") as f:
            return self._post_speech(text, [{"audio": f.read(), "text": voice.get("text", "")}])

    @staticmethod
    def _split_turns(text: str) -> list[str]:
        """fish 本体の split_text_by_speaker と同じ分け方（max_speakers=1 なので1ターン＝1回の生成）。"""
        pattern = r"(<\|speaker:\d+\|>)"
        parts = re.split(pattern, text)
        turns, i = [], 0
        while i < len(parts):
            part = parts[i].strip()
            if re.match(pattern, part):
                if i + 1 < len(parts):
                    turns.append((part + parts[i + 1]).strip())
                    i += 2
                else:
                    turns.append(part)
                    i += 1
            else:
                i += 1
        return turns or [text]

    def synth_dialogue(self, tagged_text: str, references: list[dict],
                       context: list[dict] | None = None,
                       return_codes: bool = False):
        """fish 本体の /v1/tts_multiturn と同じ（ターンごとに順に生成し、生成済みターンを履歴に積む）。"""
        if context:
            raise NotImplementedError("vLLM-Omni backend は context（context_audio）未対応")
        history: list[dict] = []
        wavs, codes = [], []
        for turn in self._split_turns(tagged_text):
            wav, (b64, shape) = self._post_speech(turn, references, history, return_codes=True)
            wavs.append(wav)
            codes.append(np.frombuffer(__import__("base64").b64decode(b64), dtype="<i2").reshape(shape).astype(np.int64))
            history.append({"text": turn, "codes": b64, "shape": shape})
        return (wavs, codes) if return_codes else wavs


class TtsServerDead(RuntimeError):
    """合成サーバが全部死んでいる（MossTtsSynth.all_servers_dead）。待たずにすぐ失敗させるための例外（2026-11-19）。"""


def tts_servers_dead(synth) -> bool:
    """合成器のサーバが全部死んだか（ワーカーの対話ループが、次の対話を取る前に見る）。"""
    ev = getattr(synth, "all_servers_dead", None)
    return bool(ev is not None and ev.is_set())


class MossTtsSynth(VllmOmniSynth):
    """MOSS-TTS（既定 MOSS-TTS-Local-Transformer-v1.5・4.5B・Apache-2.0）を moss_tts_server.py 経由で合成する
    （tts.fish_speech.backend: moss_tts）。2026-11-13 試験導入（RUNBOOK 6.10）。API は vLLM-Omni の /v1/audio/speech と同じ形。

    1ターンを参照1本のボイスクローンで単発合成する。tts.fish_speech.moss_mode で条件の付け方を選ぶ:
    clone（既定・モデルカードの例。参照音声だけ）／continuation（参照の書き起こしも使う）。multiturn は使えない。サンプリングはサーバ側の既定（v1.5 のモデルカードの例: 温度1.7・
    top_p 0.8・top_k 25・repetition_penalty 1.0）で、リクエストの温度・top_p・eos_bias は効かない
    （サー対策の温度上げは seed 替えの引き直しと同じ扱いになる）。seed と max_new_tokens だけ送る。出力は 48kHz。
    max_new_tokens は本文の長さから決める（遅い話速 3字/秒＋2秒ぶん）。止まらない生成を早めに切るため。
    """

    supports_multiturn = False
    parallel_windows = False
    FRAMES_PER_SEC = 12.5              # MOSS-Audio-Tokenizer-v2

    OUT_SR = 48000                     # クライアントが返す音声のサンプルレート（固定）

    def __init__(self, *args, model: str = "moss-tts", moss_mode: str = "clone", bwe: bool = False,
                 commas: str = "keep", tempo: float = 1.0, short_best_of: dict | None = None,
                 laugh_respell: bool = False, filler_respell: bool = False, pocket_temp: float | None = None,
                 out_sr: int | None = None, ctx_self_turns: int = 0, ctx_prompt_index: str | None = None,
                 ctx_spk_guard: dict | bool | None = None, **kwargs):
        super().__init__(*args, model=model, **kwargs)
        # pocket の文脈（tts.fish_speech.pocket_context・2026-11-15 試験）。pocket サーバ（ext_tts_server --engine pocket）に
        #   「もう言い終えた音声と本文」を ctx で渡し、その続きとして今の発話だけを作らせる（pocket_batch.generate_batch_ctx）。
        #   self_turns: 話者ごとに発話を順に作り、自分の直前の採用済み発話を何個まで文脈にするか（synthesize_turns_self_context・
        #     refine で入れ直す相槌なども synthesize_targets_with_context で同じ）。各発話のテイク選択はその中で行う。
        #   prompt_index: 参照音声の抜粋（頭の 5〜10 秒）とその ASR の書き起こしの表（_moss_tune/make_ctx_excerpts.py）。
        #     あれば全部の合成でそれを先頭の文脈にする。参照音声まるごと（約 30 秒）だと文脈＋発話が学習の長さ（30 秒）を超えて崩れる。
        self.ctx_self_turns = int(ctx_self_turns or 0)
        #   spk_guard（既定で有効・false で切る）: 文脈つきのテイクの声が相手に寄っていないか WavLM-SV で見て、寄ったテイクは選ばない。
        #     文脈の続きで作ると、短い相槌（「はい。」「うん。」）がときどき本人の声から外れる（5 対話で v10g7 0 本・自分の直前 1 発話
        #     4 本・2 発話 6 本。文脈に使った本人の発話は本人の声のまま）。margin: 本人−相手の類似度がこれ未満のテイクを捨てる
        #     （既定 −0.03）。全部捨てたら文脈なしで作り直し、それでも駄目なら本人に一番近いもの。短い音は繰り返して 1 秒にして測る。
        g = ctx_spk_guard if isinstance(ctx_spk_guard, dict) else ({} if ctx_spk_guard in (None, True) else None)
        self.ctx_spk_guard = None if (g is None or g.get("enabled", True) is False) else dict(g)
        self._spk_emb: dict = {}
        self._spk_lock = threading.Lock()
        self.guard_stats = {"turns": 0, "rejected_takes": 0, "fallback": 0, "forced": 0}
        self.ctx_prompt: dict = {}
        if ctx_prompt_index:
            idx = json.loads(Path(ctx_prompt_index).read_text())
            base = Path(ctx_prompt_index).parent
            for h, e in idx.items():
                wp = Path(e["wav"]) if Path(e["wav"]).is_absolute() else base / e["wav"]
                if e.get("text") and wp.exists():
                    self.ctx_prompt[h] = (wp.read_bytes(), str(e["text"]))
            logger.info(f"pocket の文脈: 参照の抜粋 {len(self.ctx_prompt)} 声（{ctx_prompt_index}）")
        # out_sr（tts.fish_speech.moss_out_sr）: 返す音声のサンプルレート（既定 48000）。24000 にすると pocket（24kHz）の出力を
        #   再標本化も帯域拡張もせずにそのまま使い、対話全体を 24kHz で組む（2026-11-15）。パイプラインは合成器の
        #   sample_rate を対話の sr として使うので、ここを変えれば配置・refine・書き出しまで 24kHz になる。
        if out_sr:
            self.OUT_SR = int(out_sr)
        # pocket_temp（tts.fish_speech.pocket_temp）: pocket サーバ（ext_tts_server --engine pocket）の生成温度をリクエストで渡す
        #   （2026-11-15）。未指定ならサーバの値（--pocket-temp・無ければモデル既定の 0.3）。他のエンジンはこの項を見ない。
        self.pocket_temp = None if pocket_temp is None else float(pocket_temp)
        # tempo（tts.fish_speech.moss_tempo）: 返ってきた音声を受け取った直後に WSOLA で話速だけ変える（<1 で遅く。
        #   2026-11-14 pocket CC は人間の参照より約 5% 速い）。ここで掛けるのは、テイク選択・発話 wav のキャッシュ・配置・
        #   refine の全部が「伸ばした後の音声」を 1 回だけ見るようにするため（tts.rate は配置の計画の後で伸ばし、refine の
        #   融合モードでは同じ発話を 2 度伸ばす）。60 発話で 0.95/0.90 倍＋Sidon は CER・類似度・natV3・SQUIM とも変わらない
        #   （位相ボコーダは劣化）。テイク選択の前に伸ばしても後に伸ばしても同じ（_moss_tune W95_b16_sel）。
        self.tempo = float(tempo or 1.0)
        # short_best_of（tts.fish_speech.short_best_of: {takes, max_kana}）: 短い発話（相槌など・カナ max_kana 以下）を
        #   takes 本引き、人間の相槌の長さの分布（maai-f2f の語ごとの中央値・四分位）から seed で引いた目標に一番近い
        #   長さのものを採る。pocket は「ええ。」「あー。」を 1〜2.5 秒に伸ばして読むことがあり、長すぎる挿入相槌は
        #   refine で捨てられていた（5 対話で相槌 27 個・S2 45 個）。テイク選択（synth_one_seeded）の側では使わない。
        self.short_best_of = dict(short_best_of or {})
        # laugh_respell（tts.fish_speech.moss_laugh_respell）: 合成入力だけ、笑いの書き方を声になる形に置き換える（台本・字幕・
        #   採点の文はそのまま）。pocket CC は「ふふ、」を 69% で「夫婦」、「ははっ」を「母」と読み、「あはは、」も平板に読む。
        #   「ふふふっ」「はははっ」は単語に聞こえる割合 0・AudioSet 分類器（AST）の笑いの確率が元の 4〜5 倍
        #   （2026-11-14 _moss_tune/laugh_pocket2.py・3 声×2 文×6 seed）。発話の頭か句読点の後ろの笑いだけ。
        self.laugh_respell = bool(laugh_respell)
        # base_url（--fish-url）をカンマ区切りで複数渡すと、1 本ずつの合成をサーバに順番に振る（2026-11-14）。
        #   pocket のサーバは 1 本ずつしか処理しない（束ねの口が無い）ので、1 ワーカーに複数のサーバ（同じ GPU に複数プロセス可）を付ける。
        bases = [b.strip().rstrip("/") for b in str(self.base).split(",") if b.strip()]
        self._urls = [b + "/v1/audio/speech" for b in bases] or [self.url]
        self.url = self._urls[0]
        self._url_i = 0
        self._url_lock = threading.Lock()
        # ★2026-11-19 死んだサーバの遮断: 同じサーバで通信のタイムアウト・接続エラーが dead_after 回続いたら（間に成功が無い）、
        #   そのサーバは以後使わない。全部のサーバが死んだら all_servers_dead を立て、以後の合成は待たずに TtsServerDead で失敗させる
        #   （ワーカーの対話ループはこれを見て claim を手放して抜ける）。HTTP 500（OOM など）はサーバが生きているので数えない。
        #   経緯: ft_v2 002 で node25 GPU4 が落ちても pocket サーバは /health に 200 を返し、付いたワーカー 2 本が 1 ターンごとに
        #   300 秒×2 のタイムアウトを 3 時間以上待ち、作った対話は無音代替で全部捨てられていた。
        self.dead_after = int(os.environ.get("TTS_DEAD_AFTER", "2"))
        self._url_fail: dict = {}
        self._dead_urls: set = set()
        self.all_servers_dead = threading.Event()
        # filler_respell（tts.fish_speech.moss_filler_respell）: 合成入力だけ「あー/ああ→アー」「えー→エー」「ええ→エエ」（respell_filler）
        self.filler_respell = bool(filler_respell)
        # ★サンプルレートは最初から固定値にする（2026-11-13 の不具合）。パイプラインは合成器を作った直後の
        #   sample_rate（fish 系の初期値 44100）を対話全体の sr として使い、応答の実レートで更新しない。
        #   MOSS は Local 48kHz・Delay 24kHz を返すので、そのまま渡すと 1.09 倍遅く低い声／1.84 倍速の高い声として
        #   配置・照合されていた（Delay で話者の取り違え判定が全ターンに出て発覚）。応答はここで OUT_SR に揃える。
        self.sample_rate = self.OUT_SR
        # bwe: 24kHz 出力（12kHz より上が無い）の上の帯域を信号処理で補う（bandwidth_ext.py）。False なら普通の再標本化
        self.bwe = bool(bwe)
        # commas: 合成入力の読点「、」の扱い（台本・字幕・読みの検査は元の本文のまま）。MOSS は読点ごとに長めの間を
        #   取り、発話の中の無音が人間の録音の約2倍になる（2026-11-13 調整）。keep / drop / space（弱い区切りとして残す）/
        #   pause:<秒>（[pause X.Ys] の明示の間に置き換える）
        if not (commas in ("keep", "drop", "space") or re.fullmatch(r"pause:\d+(\.\d+)?", str(commas))):
            raise ValueError(f"tts.fish_speech.moss_commas は keep/drop/space/pause:<秒>（受領: {commas!r}）")
        self.commas = str(commas)
        if moss_mode not in ("clone", "continuation"):
            raise ValueError(f"tts.fish_speech.moss_mode は clone / continuation（受領: {moss_mode!r}）")
        # clone: 参照音声だけを条件にする（モデルカードの例）／continuation: 参照の書き起こし＋本文をテキストに、
        # 参照音声を先頭の音声にして続きを生成する（書き起こしを使う）
        self.moss_mode = moss_mode
        if self.multiturn:
            raise ValueError("tts.fish_speech.backend: moss_tts は multiturn 未対応（multiturn: false にする）")

    def _max_frames(self, text: str, gk: dict) -> int:
        body = re.sub(r"<\|speaker:\d+\|>|\[[^\]]*\]", "", text)
        n = len(re.sub(r"\s", "", body))
        est = int((n / 3.0 + 2.0) * self.FRAMES_PER_SEC)
        cap = int(gk.get("max_new_tokens", 500) or 500)
        return max(32, min(cap, est))

    def synth_one_seeded(self, text: str, ref: dict, seed: int) -> np.ndarray:
        """seed を引数で渡す単発合成（合成器の状態を書き換えないのでスレッドから同時に呼べる）。
        テイク選択（best_of）から呼ばれるので、短い発話の引き比べ（short_best_of）はしない（選択の側が見る）。"""
        return self._post_speech(normalize_tts_text(text), [ref], seed=seed, short=False)

    # 人間の相槌の長さ（maai-f2f-jp-2025 の vad-ver2 IPU・reference_corpus.classify で bc のもの。秒: 中央値, 25%, 75%）
    BC_DUR = {"はい": (0.29, 0.23, 0.35), "うん": (0.31, 0.24, 0.39), "うーん": (0.64, 0.51, 0.83),
              "あー": (0.67, 0.50, 0.90), "へー": (0.79, 0.57, 1.02), "あ": (0.17, 0.13, 0.23),
              "うんうん": (0.54, 0.44, 0.64), "うんうんうん": (0.76, 0.67, 0.87), "はいはい": (0.45, 0.38, 0.56),
              "はいはいはい": (0.64, 0.55, 0.72), "そうですね": (0.81, 0.65, 0.99), "えー": (0.64, 0.47, 0.85),
              "そう": (0.35, 0.28, 0.50), "え": (0.18, 0.14, 0.26), "ふーん": (0.84, 0.64, 1.18),
              "ね": (0.28, 0.22, 0.40), "ねえ": (0.44, 0.34, 0.62), "たしかに": (0.52, 0.45, 0.64),
              "確かに": (0.52, 0.45, 0.64), "なるほど": (0.59, 0.48, 0.71), "おー": (0.66, 0.48, 0.84),
              "ええ": (0.40, 0.32, 0.52), "そうなんですね": (0.99, 0.82, 1.16), "そうですよね": (0.92, 0.78, 1.14),
              "そうそう": (0.45, 0.36, 0.58)}
    _BC_ALIAS = {"へえ": "へー", "ああ": "あー", "おお": "おー", "ふうん": "ふーん", "ううん": "うーん", "えええ": "えー"}

    @classmethod
    def bc_median(cls, text: str) -> tuple[float, float]:
        """短い発話 text の人間の長さの (中央値, 対数の幅 σ)。表に無い語はモーラ数から
        （0.06 + 0.115×カナ数 + 長音 1 つにつき 0.30・幅 0.25）。"""
        import reading_check as RC
        k = re.sub(r"[。、．，！？!?…〜~\s　「」]", "", text or "")
        k = cls._BC_ALIAS.get(k, k)
        if k in cls.BC_DUR:
            med, p25, p75 = cls.BC_DUR[k]
            return med, float(np.log(p75 / p25) / 1.349)
        return 0.06 + 0.115 * len(RC.kana(k.replace("ー", ""))) + 0.30 * k.count("ー"), 0.25

    @classmethod
    def bc_target(cls, text: str, seed: int | None) -> float:
        """短い発話 text の目標の長さ（秒）。人間の分布（対数正規で近似。bc_median）から seed で 1 つ引く（毎回中央値にすると
        相槌の長さが揃いすぎるため）。"""
        med, sig = cls.bc_median(text)
        z = float(np.clip(np.random.default_rng(None if seed is None else int(seed) & 0x7FFFFFFF)
                          .standard_normal(), -1.5, 1.5))
        return float(med * np.exp(z * sig))

    def _post_speech(self, text: str, refs: list[dict], history: list[dict] | None = None,
                     return_codes: bool = False, seed: int | None = None, short: bool = True):
        if history or return_codes or len(refs) != 1:
            raise NotImplementedError("moss_tts backend は参照1本の単発合成だけ対応")
        text = re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", text)
        if self.laugh_respell:
            text = respell_laugh(text)
        if self.filler_respell:
            text = respell_filler(text)
        text = self._commas(text)
        sb = self.short_best_of
        if short and int(sb.get("takes", 1) or 1) > 1:
            import reading_check as RC
            if 0 < len(RC.kana(re.sub(r"\[[^\]]*\]", "", text))) <= int(sb.get("max_kana", 6)):
                return self._short_best(text, refs[0], seed, int(sb["takes"]))
        return self._moss_once(text, refs[0], seed)

    def _short_best(self, text: str, ref: dict, seed: int | None, takes: int) -> np.ndarray:
        """短い発話を takes 本引いて、目標の長さ（bc_target）に声の長さ（_active_and_pauses）が一番近いものを返す。"""
        base = int(seed) if seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
        seeds = [base + 7919 * k for k in range(takes)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(takes, max(1, self.concurrency))) as ex:
            ws = list(ex.map(lambda s: self._moss_once(text, ref, s), seeds))
        return self._short_pick(ws, text, base)

    def _short_pick(self, ws: list, text: str, base: int) -> np.ndarray:
        if str(self.short_best_of.get("criterion", "")) == "median":
            return self._short_median(ws)
        tgt = self.bc_target(text, base)
        w_dur = float(self.short_best_of.get("w_dur", 3.0))
        q = text.rstrip().endswith(("？", "?"))
        best, best_s = ws[0], float("inf")
        for w in ws:
            act = _active_and_pauses(w, self.OUT_SR)[0] if w is not None and len(w) else 0.0
            if act <= 0:
                continue
            # 長さ（目標との対数比）＋ short_take_score（息だけ・平ら・語尾が上がる相槌を避ける。Irodori の short_takes と同じ）
            s = w_dur * abs(float(np.log(act / tgt))) + short_take_score(w, self.OUT_SR, question=q)
            if s < best_s:
                best, best_s = w, s
        return best

    def _short_median(self, ws: list) -> np.ndarray:
        """short_best_of.criterion: median（2026-11-15・ユーザ案）: 長さ（声の区間）と音量（声の区間の RMS・dB）がどちらも
        テイクの中央に近いものを採る。距離は各量の中央値からのずれを中央絶対偏差（下限: 長さ 0.03 秒・音量 0.5 dB）で割って足す。
        人間の長さの分布や音程の形（short_take_score）は見ない。"""
        feats = []
        for k, w in enumerate(ws):
            if w is None or not len(w):
                continue
            act = _active_and_pauses(w, self.OUT_SR)[0]
            if act <= 0:
                continue
            x = np.asarray(w, dtype=np.float32).reshape(-1)
            h = max(1, int(0.01 * self.OUT_SR))
            n = len(x) // h
            e = np.sqrt((x[: n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
            db = 20 * np.log10(e)
            lv = float(10 * np.log10(np.mean(e[db > db.max() - 30] ** 2)))
            feats.append((k, act, lv))
        if not feats:
            return ws[0]
        d = np.array([f[1] for f in feats]); lv = np.array([f[2] for f in feats])
        sd = max(float(np.median(np.abs(d - np.median(d)))), 0.03)
        sl = max(float(np.median(np.abs(lv - np.median(lv)))), 0.5)
        sc = np.abs(d - np.median(d)) / sd + np.abs(lv - np.median(lv)) / sl
        return ws[feats[int(np.argmin(sc))][0]]

    def _post_once(self, text: str, ref0: dict, seed: int | None) -> np.ndarray:
        """互換用（MossTtsSynth の 1 本の合成は _moss_once）。"""
        return self._moss_once(text, ref0, seed)

    def _moss_once(self, text: str, ref0: dict, seed: int | None, ctx: list | None = None) -> np.ndarray:
        """ctx: 文脈 [(wav bytes, 書き起こし)]（自分の前の発話など）。参照の抜粋（ctx_prompt）があればその前に置く。"""
        refs = [ref0]
        gk = self._gen_for(refs[0].get("text"))
        if seed is not None:
            gk = {**gk, "seed": int(seed)}
        body = {"model": self.model, "input": text, "response_format": "wav", "mode": self.moss_mode,
                "ref_audio": self._ref_data_url(refs[0]["audio"]), "ref_text": refs[0].get("text", ""),
                "max_new_tokens": self._max_frames(text, gk)}
        if gk.get("seed") is not None:
            body["seed"] = int(gk["seed"])
        if self.pocket_temp is not None:
            body["pocket_temp"] = self.pocket_temp
        items = list(ctx or [])
        if self.ctx_prompt:
            e = self.ctx_prompt.get(hashlib.sha1(refs[0]["audio"]).hexdigest())
            if e:
                items = [e] + items
        if items:
            body["ctx"] = [{"audio": self._ref_data_url(a), "text": t} for a, t in items]
        with self._url_lock:                      # 複数サーバなら順番に振る（死んだサーバは飛ばす）
            live = [u for u in self._urls if u not in self._dead_urls]
            if not live:
                self.all_servers_dead.set()
                raise TtsServerDead(f"TTS サーバが全部応答しない（{', '.join(self._urls)}）")
            url = live[self._url_i % len(live)]
            self._url_i += 1
        import requests
        try:
            resp = self._session.post(url, json=body, timeout=self.timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            with self._url_lock:
                k = self._url_fail.get(url, 0) + 1
                self._url_fail[url] = k
                if k >= self.dead_after and url not in self._dead_urls:
                    self._dead_urls.add(url)
                    logger.error(f"TTS サーバ {url} が {k} 回続けて応答しない（{type(e).__name__}）→ 以後このサーバは使わない"
                                 f"（GPU が落ちていないか nvidia-smi で確かめる）")
                    if len(self._dead_urls) >= len(self._urls):
                        self.all_servers_dead.set()
                        logger.error("TTS サーバが全部応答しない → このワーカーは今の対話を捨てて終わる")
            raise
        with self._url_lock:
            self._url_fail[url] = 0
        if resp.status_code != 200:
            raise RuntimeError(f"MOSS-TTS {resp.status_code}: {resp.text[:300]}")
        audio, sr = self._decode_wav(resp.content)
        if abs(self.tempo - 1.0) >= 1e-3:          # 帯域拡張・再標本化の前（元のレートで伸ばす）
            audio = time_stretch_wsola(audio, self.tempo, sr)
        return self._to_out_sr(audio, sr)

    def _prep(self, text: str) -> str:
        """合成入力の書き換え（_post_speech と同じ）: 話者タグを外し、笑い・フィラーの書き方、読点。"""
        text = re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", text)
        if self.laugh_respell:
            text = respell_laugh(text)
        if self.filler_respell:
            text = respell_filler(text)
        return self._commas(text)

    def _spk_margin(self, ws: list, ref: dict, others: list) -> list:
        """各テイクの（本人の参照との類似度 − 相手の参照との類似度の最大）。WavLM-SV（speaker_check）。短い音は繰り返して 1 秒以上に。"""
        import speaker_check as SC
        ver = SC.get_verifier(str(self.ctx_spk_guard.get("model", "microsoft/wavlm-base-plus-sv")),
                              str(self.ctx_spk_guard.get("device", "cuda")))
        if ver is None:
            return [0.0] * len(ws)
        with self._spk_lock:
            embs = []
            for r in [ref] + list(others):
                h = hashlib.sha1(r["audio"]).hexdigest()
                if h not in self._spk_emb:
                    self._spk_emb[h] = ver.embed_bytes(r["audio"])
                embs.append(self._spk_emb[h])
            out = []
            for w in ws:
                if w is None or not len(w):
                    out.append(-9.0)
                    continue
                x = np.asarray(w, dtype=np.float32)
                if len(x) < self.OUT_SR:
                    x = np.tile(x, int(np.ceil(self.OUT_SR / len(x))))
                e = ver.embed(x, self.OUT_SR)
                out.append(float(e @ embs[0]) - max(float(e @ o) for o in embs[1:]))
        return out

    def _guard(self, ws: list, t: str, ref: dict, others: list, seeds: list) -> list:
        """声が相手に寄ったテイクを外す（ctx_spk_guard）。全部外れたら文脈なしで作り直す。"""
        mg = float(self.ctx_spk_guard.get("margin", -0.03))
        sc = self._spk_margin(ws, ref, others)
        ok = [w for w, m in zip(ws, sc) if m >= mg]
        self.guard_stats["turns"] += 1
        self.guard_stats["rejected_takes"] += len(ws) - len(ok)
        if ok:
            return ok
        self.guard_stats["fallback"] += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as ex:
            ws2 = list(ex.map(lambda s: self._moss_once(t, ref, s + 1), seeds))
        sc2 = self._spk_margin(ws2, ref, others)
        ok = [w for w, m in zip(ws2, sc2) if m >= mg]
        if ok:
            return ok
        self.guard_stats["forced"] += 1
        allw, alls = ws + ws2, sc + sc2
        return [allw[int(np.argmax(alls))]]

    def ctx_turn(self, text: str, ref: dict, ctx: list, seed: int | None = None, others: list | None = None) -> np.ndarray:
        """文脈 ctx（[(wav ndarray（OUT_SR）, 本文)]・自分の前の発話など）の続きとして text を作り、テイクを選んで返す（2026-11-15）。
        短い発話（カナ short_best_of.max_kana 以下）は short_best_of.takes 本から _short_pick、ほかは best_of（criterion: median_dur）の
        takes 本から声の区間の長さが中央のもの。文脈の本文も合成入力と同じように書き換える（音声はその入力で作られたため）。"""
        import reading_check as RC
        t = self._prep(normalize_tts_text(text))
        items = [(FishSpeechSynth._encode_wav(w, self.OUT_SR), self._prep(normalize_tts_text(tx)))
                 for w, tx in ctx if w is not None and len(w) and str(tx).strip()]
        sb, bo = self.short_best_of, dict(getattr(self, "best_of", None) or {})
        short = 0 < len(RC.kana(re.sub(r"\[[^\]]*\]", "", t))) <= int(sb.get("max_kana", 6))
        if short:
            takes = int(sb.get("takes", 1) or 1)
        elif bo.get("enabled", False) and str(bo.get("criterion", "")) == "median_dur":
            takes = int(bo.get("takes", 1) or 1)
        else:
            takes = 1
        base = int(seed) if seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
        seeds = [base + 7919 * k for k in range(max(1, takes))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as ex:
            ws = list(ex.map(lambda s: self._moss_once(t, ref, s, ctx=items), seeds))
        if self.ctx_spk_guard is not None and others and items:
            ws = self._guard(ws, t, ref, others, seeds)
        if len(ws) == 1:
            return ws[0]
        if short:
            return self._short_pick(ws, t, base)
        d = [(_active_and_pauses(w, self.OUT_SR)[0] if w is not None and len(w) else 0.0) for w in ws]
        ok = [k for k in range(len(ws)) if d[k] > 0]
        if not ok:
            return ws[0]
        med = float(np.median([d[k] for k in ok]))
        return ws[min(ok, key=lambda k: abs(d[k] - med))]

    def _commas(self, text: str) -> str:
        c = self.commas
        if c == "drop":
            return text.replace("、", "")
        if c == "space":
            return text.replace("、", " ")
        if c.startswith("pause:"):
            return text.replace("、", f"[pause {c.split(':', 1)[1]}s]")
        return text

    def _to_out_sr(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.OUT_SR:
            return audio
        if self.bwe and sr < self.OUT_SR:
            import bandwidth_ext
            y, _ = bandwidth_ext.extend(audio, sr, out_sr=self.OUT_SR)
            return y
        from math import gcd
        import scipy.signal as ss
        g = gcd(int(sr), self.OUT_SR)
        return ss.resample_poly(audio, self.OUT_SR // g, int(sr) // g).astype(np.float32)

    def synth_dialogue(self, tagged_text: str, references: list[dict],
                       context: list[dict] | None = None, return_codes: bool = False):
        """文脈なしのターン列を、話者タグが指す参照で1ターンずつ単発合成する。
        引き直し（_synth_one_line＝"<|speaker:0|>本文" と参照1本）がこの経路を通る。
        文脈（context）と音声トークンの返却は未対応（multiturn 本体は supports_multiturn=False で通らない）。"""
        if context or return_codes:
            raise NotImplementedError("moss_tts backend は文脈付きの合成（multiturn）未対応")
        wavs = []
        for turn in self._split_turns(tagged_text):
            m = re.match(r"\s*<\|speaker:(\d+)\|>", turn)
            k = int(m.group(1)) if m else 0
            wavs.append(self._post_speech(turn, [references[k]]))
        return wavs


def _trim_wav_bytes(b: bytes, max_sec: float) -> bytes:
    """wav のバイト列を先頭 max_sec 秒に切る（短ければそのまま返す）。"""
    import io as _io
    info = sf.info(_io.BytesIO(b))
    if info.duration <= max_sec:
        return b
    w, sr = sf.read(_io.BytesIO(b), dtype="float32", always_2d=True)
    buf = _io.BytesIO()
    sf.write(buf, w[: int(max_sec * sr)], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_LAUGH_FU = re.compile(r"(?:(?<=^)|(?<=[、。！？!?\s]))(?:う|ん)?ふ{2,}っ?(?=[、。！？!?…\s]|$)")
_LAUGH_HA = re.compile(r"(?:(?<=^)|(?<=[、。！？!?\s]))(?:あ|ア)?(?:は{2,}|ハ{2,})っ?(?=[、。！？!?…\s]|$)")


_FILLER_RESPELL = (("あー", "アー"), ("ああ", "アー"), ("えー", "エー"), ("ええ", "エエ"))
_FILLER_RE = re.compile(r"(?:(?<=^)|(?<=[、。！？!?\s]))(あー|ああ|えー|ええ)(?=[、。！？!?…\s]|$)")


def respell_filler(text: str) -> str:
    """合成入力の長音の相槌・フィラーをカタカナに（MossTtsSynth.filler_respell）。pocket CC はひらがなの「ー」を
    極端に引き延ばす（単独の「あー。」中央値 2.15 秒・人間 0.67、「ええ。」0.97・人間 0.40、「えー、そうなんですか？」全体 3.1 秒）。
    「アー。」0.62・「エエ。」0.44・「エー、そうなんですか？」1.9 秒で、ASR の聞き取りは同じ（2026-11-14 _moss_tune/bc_spell_pocket.py）。
    「へえ」「うーん」はカタカナにしても良くならないので触らない。発話の頭か句読点・空白の後ろで、後ろに句読点が来るものだけ。"""
    m = dict(_FILLER_RESPELL)
    return _FILLER_RE.sub(lambda x: m[x.group(1)], text)


def respell_laugh(text: str) -> str:
    """合成入力の笑いを声になる書き方へ（MossTtsSynth.laugh_respell）。「ふふ」「うふふ」→「ふふふっ」、
    「あはは」「はは」「アハハ」→「はははっ」。発話の頭か句読点・空白の後ろで、後ろに句読点が来るものだけ。"""
    return _LAUGH_HA.sub("はははっ", _LAUGH_FU.sub("ふふふっ", text))


# librosa.pyin（numba の gufunc）を複数スレッドから同時に呼ぶと、初回のコンパイルが競合して segfault する
# （2026-11-14 node22: 合成スレッド 16 本から short_take_score が同時に走り、ワーカーが痕跡なく落ちた）。pyin は必ずこの錠の中で。
_PYIN_LOCK = threading.Lock()


def _pyin(*args, **kwargs):
    import librosa
    with _PYIN_LOCK:
        return librosa.pyin(*args, **kwargs)


def short_take_score(w: np.ndarray, sr: int, question: bool = False) -> float:
    """短い発話（相槌）のテイクの悪さ（小さいほど良い）。
    声の出ている（F0 が取れる）フレームが 6 未満＝息だけ・ささやき → 10 以上。
    それ以外は語尾の音程の変化 slope（終わり 3 フレーム − 始め 3 フレーム・半音）で、下がる（−1 半音以下）ほど良い
    （人の相槌の「はい」は下がる。S2 も中央値 −3.8）。疑問（question）は逆に上がるほど良い。
    ＋ 有音のうち声になっていない割合。"""
    import librosa
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if len(w) < int(0.1 * sr):
        return 20.0
    h = int(0.01 * sr)
    e = np.array([np.sqrt(np.mean(w[i:i + h] ** 2)) for i in range(0, len(w) - h, h)])
    db = 20 * np.log10(e + 1e-9)
    active = int((db > db.max() - 30).sum())
    f0, vf, _ = _pyin(w, fmin=60, fmax=500, sr=sr, frame_length=int(0.04 * sr), hop_length=h)
    f = f0[vf]
    if len(f) < 6:
        return 10.0 + (6 - len(f))
    slope = 12 * np.log2(f[-3:].mean() / f[:3].mean())
    unvoiced = 1.0 - min(1.0, len(f) / max(1, active))
    return float((max(0.0, 1.0 - slope) if question else max(0.0, slope + 1.0)) + 2.0 * unvoiced)


class IrodoriVllmSynth(MossTtsSynth):
    """Irodori-TTS（v4-Large 等）を vLLM-Omni の IrodoriPipeline（/v1/audio/speech）で合成する
    （tts.fish_speech.backend: irodori_vllm。2026-11-14）。API の形は MossTtsSynth と同じで、Irodori 固有の指定は
    extra_params で送る（サーバは同じ NFS を見る前提）:
      seconds   … 長さ（前後の無音込み）。「モーラ数 ÷ mora_rate ÷ 0.94」（IrodoriSynth の mode=mora と同じ）
      ref_wavs  … 参照が声バンクのファイルなら、同じ話者の兄弟テイク（*_t<N>.wav）を足して最大 refs_sec 秒
      num_steps・cfg_scale_text・cfg_scale_speaker ほか … tts.fish_speech.irodori にあればそのまま
    irodori.laugh_emoji: split で、頭の笑いを別に作って（laugh_text「🤭🤭ふふっ。」を laugh_takes 本引き、笑いの音の
    長さが laugh_active 秒に近いもの）残りの文と laugh_gap 秒あけて繋ぐ（2026-11-14 ユーザ聴取で一番よかった方式）。
    irodori.laugh_emoji: true で、発話の頭の笑い（「ふふ。」「あはは、」等）を絵文字 🤭（モデルの笑いの指定）に
    置き換えて送る。そのまま読ませると「ふふ」が「うん」に聞こえるテイクが多い（2026-11-14 _moss_tune/laugh_test.py:
    4 seed 中 3 本で ASR が「うん」。🤭 に置き換えると文の NLL 0.57→0.16）。長さは元の文（笑い込み）で決める。
    笑いだけの発話は触らない。
    irodori.text_replace: {元: 先}…合成入力だけの文字列置換。カタカナの人名はアクセントが崩れるので漢字に
    （2026-11-14 ユーザ聴取「イノウエ」より「井上」が断然よい）。台本・字幕・採点の文はそのまま。
    irodori.short_takes: N（>1）で、モーラが short_max_mora（既定 4）以下の短い発話（相槌）は N 本引いて
    short_take_score で選ぶ（息だけ・平ら・語尾が上がる「はい」を避ける）。
    irodori.mute_end_click（既定 true）: 出力の最後に無音（0.15 秒以上）を挟んで付く 60ms 未満のクリックを無音にする。
    出力は 48kHz（OUT_SR と同じなので再標本化なし）。
    """

    _LAUGH_HEAD = re.compile(r"^(?:う?ふ(?:ふ)+|あ?は(?:は)+|え?へ(?:へ)+|くすっ?)っ?[。、！!…]*\s*")

    _PASSTHRU = ("num_steps", "cfg_scale_text", "cfg_scale_speaker", "cfg_guidance_mode", "cfg_min_t", "cfg_max_t",
                 "truncation_factor", "t_schedule_mode", "sway_coeff", "speaker_kv_scale")

    def __init__(self, *args, irodori: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.icfg = dict(irodori or {})
        self.mora_rate = float(self.icfg.get("mora_rate", 7.3) or 0)
        # 長さ＝max(min_sec, モーラ÷mora_rate÷0.94 + pad_sec)。既定（0.0・0.5）は従来どおり（2026-11-19 追加。下の _extra_params）
        self.pad_sec = float(self.icfg.get("pad_sec", 0.0) or 0.0)
        self.min_sec = float(self.icfg.get("min_sec", 0.5) or 0.5)
        self.edges = dict(self.icfg.get("edges") or {})     # {lead_keep, tail_keep, end_fade}。空なら整えない
        # irodori.bc_prev_ref（2026-11-19）: refine で入れ直す相槌の参照の先頭に、その話者の直前の採用済み発話を置く
        # （{enabled, sim_weight, takes, tmp_dir, model, device}。synthesize_targets_with_context → _targets_bc_prev_ref → bc_with_prev）
        # takes: 相槌の候補の本数（既定 short_takes）。_moss_tune/ntakes_bc_test.py（相槌 48 個×32 本）で 6→32 本:
        # 息だけ・平ら・語尾が上がる 16%→4%・人間の 1.5 倍より長い 23%→10%・声のまま切れ 3.5%→0%。ユーザ聴取で 32 本が上（v5）
        self.bc_prev = dict(self.icfg.get("bc_prev_ref") or {})
        self.bc_prev_ref = bool(self.bc_prev.get("enabled", bool(self.bc_prev)))
        self._bc_ver = None
        self._bc_ver_lock = threading.Lock()
        self._bc_emb_lock = threading.Lock()     # ★2026-11-20 _bc_embed を複数スレッドから呼ぶため
        self.refs_sec = float(self.icfg.get("refs_sec", 60))
        self._url_ok_t: dict = {}        # ★2026-11-20 サーバごとの最後の成功時刻（_irodori_note_fail のタイムアウト判定）
        self._refidx = None
        self._refidx_lock = threading.Lock()
        # ★2026-11-20 B4（irodori.laugh_reuse）: 頭の笑いを 1 ターンに 1 回だけ作って、そのターンの全テイクで使い回すための入れ物。
        #   テイクは並行に作られるので、最初に来た 1 本だけが作り、ほかはそれを待つ（_laugh_cached）。既定は使わない
        self._laugh_cache: dict = {}
        self._laugh_lock = threading.Lock()
        self._log_lock = threading.Lock()       # ★2026-11-20 _score_log の書き込み

    def _inner_short(self, text: str) -> bool:
        """★2026-11-20 B3: _post_speech が中で short_takes 本から選ぶ短い発話か（_post_speech と同じ判定）。"""
        n = int(self.icfg.get("short_takes", 1) or 1)
        return n > 1 and estimate_mora(normalize_tts_text(text))[0] <= float(self.icfg.get("short_max_mora", 4))

    def _score_log(self, path, rec: dict) -> None:
        """★2026-11-20 B1/B2 の閾値を決めるための採点の記録（best_of.score_log・bc_prev_ref.score_log にパスがあるときだけ。既定は書かない）。"""
        if not path:
            return
        rec = dict(rec, tag=getattr(self, "score_log_tag", None))
        with self._log_lock:
            with open(str(path), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _ref_key(b: bytes) -> str:
        return hashlib.md5(b).hexdigest()          # 参照は呼ぶたびに読み直されるので、中身で見分ける

    def _laugh_cached(self, text: str, refs: list[dict], sd0: int) -> np.ndarray:
        """★2026-11-20 B4（irodori.laugh_reuse: true）: ターン（合成入力の文＋参照）ごとに頭の笑いを 1 回だけ作り、以後のテイクは同じ笑いを使う。
        それまでは best_of の 8 テイクごとに笑いを laugh_takes 本ずつ引き直していた（笑いのある発話 1 本で 8×(laugh_takes＋1) 要求）。
        テイクは並行に来るので、最初の 1 本が作り、ほかは出来るまで待つ（どのテイクの seed で作るかは先着＝base の最初のテイク。
        短い発話は _post_speech が並行に投げる前に k=0 の seed で先に作る）。同じ文・同じ声のターンが対話に 2 つあれば同じ笑いになる。"""
        key = (text, self._ref_key(refs[0]["audio"]))
        with self._laugh_lock:
            ent = self._laugh_cache.get(key)
            owner = ent is None
            if owner:
                ent = [threading.Event(), None]
                self._laugh_cache[key] = ent
                while len(self._laugh_cache) > 256:          # 古いものから捨てる（対話をまたいで溜めない）
                    self._laugh_cache.pop(next(iter(self._laugh_cache)))
        if owner:
            try:
                ent[1] = self._laugh(refs, sd0)
            except Exception:
                with self._laugh_lock:
                    self._laugh_cache.pop(key, None)
                raise
            finally:
                ent[0].set()
            return ent[1]
        ent[0].wait()
        return ent[1] if ent[1] is not None else self._laugh(refs, sd0)

    def _laugh_split_text(self, text: str):
        """★2026-11-20 B4: _post_once と同じ前処理をした合成入力と、笑いを分けて作るか（laugh_emoji: split で頭に笑いがある）。"""
        text = self._commas(re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", text))
        for a, b in (self.icfg.get("text_replace") or {}).items():
            text = text.replace(str(a), str(b))
        rest = self._LAUGH_HEAD.sub("", text, count=1)
        return text, (self.icfg.get("laugh_emoji") == "split" and rest != text and bool(rest.strip()))

    def _ref_index(self):
        with self._refidx_lock:
            if self._refidx is None:
                from irodori_fast import RefIndex
                # Public release: scan every directory under voices/ that holds *.wav (voices/<bank>/wav/ etc.)
                dirs = self.icfg.get("voices_dirs") or sorted(
                    {str(p.parent) for p in (Path(__file__).resolve().parent / "voices").rglob("*.wav")})
                self._refidx = RefIndex(dirs)
            return self._refidx

    def _extra_params(self, text: str, ref_audio: bytes) -> dict:
        ep = {k: self.icfg[k] for k in self._PASSTHRU if k in self.icfg}
        if self.mora_rate > 0:
            # irodori.pad_sec（2026-11-19）: v4.1-Small は指定の長さの頭に約 0.24 秒の無音を置くので、モーラだけの長さでは
            # 短い発話・相槌の語尾が切れる（「うん。」は下限 0.5 秒で 73% が声のまま終わった）→ 固定の余裕を足し、
            # 余った前後の無音は edges で整える。_moss_tune/yomi_len_test.py
            from irodori_fast import seconds_for
            ep["seconds"] = max(self.min_sec, seconds_for(text, self.mora_rate, lo=0.0) + self.pad_sec)
            sl = float(self.icfg.get("short_len_bc", 0.0) or 0.0)
            if sl > 0 and estimate_mora(text)[0] <= float(self.icfg.get("short_max_mora", 4)):
                # irodori.short_len_bc（2026-11-19）: 相槌は人間の長さの中央値×short_len_bc＋pad_sec を下限に。「あー。」はモーラでは
                # 2 だが人間は 0.67 秒伸ばすので、下限 0.8 秒だと 67% が声のまま終わった（_moss_tune/yomi_bc_test.py）
                ep["seconds"] = max(ep["seconds"], self.bc_median(text)[0] * sl + self.pad_sec)
        if self.refs_sec > 30:
            hit = self._ref_index().find(ref_audio)
            if hit is not None:
                from irodori_fast import siblings
                ep["ref_wavs"] = [str(hit)] + siblings(hit)
                ep["max_ref_seconds"] = self.refs_sec
        return ep

    def _post_speech(self, text: str, refs: list[dict], history: list[dict] | None = None,
                     return_codes: bool = False, seed: int | None = None, short: bool = True):
        # short は MossTtsSynth.synth_one_seeded が渡す（Irodori は従来どおり short_takes を常に使う）
        if history or return_codes or len(refs) != 1:
            raise NotImplementedError("irodori_vllm backend は参照1本の単発合成だけ対応")
        n = int(self.icfg.get("short_takes", 1) or 1)
        if n > 1 and estimate_mora(normalize_tts_text(text))[0] <= float(self.icfg.get("short_max_mora", 4)):
            # 相槌などの短い発話（「はい。」）は、息だけ・平ら・上がる（「はい？」に聞こえる）テイクが3割ほど出る
            # → n 本引いて、声がはっきり出て語尾が下がるものを採る（short_take_score。2026-11-14 _moss_tune/bc_test.py）
            import random
            base = int(seed) if seed is not None else random.randrange(1 << 30)
            q = text.rstrip().endswith(("？", "?"))
            if self.icfg.get("laugh_reuse"):
                # ★2026-11-20 B4: 頭の笑いを並行に投げる前に k=0 の seed で 1 回だけ作っておく（どのテイクの seed で作るかを先着に任せない）
                lt, sp = self._laugh_split_text(text)
                if sp:
                    self._laugh_cached(lt, refs, base)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                ws = list(ex.map(lambda k: self._post_once(text, refs, base + 7919 * k), range(n)))
            return self._clean_short(ws[int(np.argmin(self._short_scores(ws, text, base, q)))])
        return self._post_once(text, refs, seed)

    def _short_scores(self, ws: list, text: str, base: int, question: bool) -> list[float]:
        """短い発話（相槌）のテイクの悪さ（小さいほど良い）。short_take_score（息だけ・平ら・語尾が上がるものを避ける）に、
        irodori.short_flaw（2026-11-19）があれば足す:
          cut  … 声のまま切れたもの（voiced_at_end）
          blob … 無音の後に余計な音が付いたもの（tail_blob）
          head … 声の途中から始まるもの（voiced_at_start。2026-11-19）
          edge_blob … 声の前後に離れた余計な音があるもの（edge_blobs。2026-11-19）
          dur  … 声の長さと人間の相槌の長さ（bc_target。pocket の short_best_of と同じ）の対数比の絶対値の重み。
                 Irodori は与えた長さいっぱいに母音を伸ばすことがある（「はーーい」）ので、自然な長さのテイクを採る"""
        sc = [short_take_score(w, self.OUT_SR, question=question) for w in ws]
        sf_ = self.icfg.get("short_flaw") or {}
        if not sf_:
            return sc
        w_dur = float(sf_.get("dur", 0.0))
        tgt = self.bc_target(text, base) if w_dur > 0 else None
        out = []
        for s, w in zip(sc, ws):
            s += float(sf_.get("cut", 5.0)) * voiced_at_end(w, self.OUT_SR)
            s += float(sf_.get("blob", 3.0)) * (tail_blob(w, self.OUT_SR) is not None)
            # 2026-11-19（irodori-yomi-v2）: 声の途中から始まる（頭の「プツッ」）・前後の余計な音（前側と、0.1 秒離れた後ろ側も）
            s += float(sf_.get("head", 0.0)) * voiced_at_start(w, self.OUT_SR)
            s += float(sf_.get("edge_blob", 0.0)) * bool(edge_blobs(w, self.OUT_SR))
            if tgt:
                act = _active_and_pauses(w, self.OUT_SR)[0] if len(w) else 0.0
                s += w_dur * abs(float(np.log(max(act, 0.05) / tgt)))
            out.append(s)
        return out

    def _bc_verifier(self):
        with self._bc_ver_lock:
            if self._bc_ver is None:
                import speaker_check as SC
                self._bc_ver = SC.get_verifier(str(self.bc_prev.get("model", "microsoft/wavlm-base-plus-sv")),
                                               str(self.bc_prev.get("device", "cuda")))
            return self._bc_ver

    def _bc_embed(self, w: np.ndarray) -> np.ndarray:
        """WavLM-SV の埋め込み。短い音は繰り返して 1 秒以上に（_spk_margin と同じ。短すぎるとモデルが落ちる）。"""
        x = np.asarray(w, np.float32).reshape(-1)
        if len(x) < self.OUT_SR:
            x = np.tile(x if len(x) else np.zeros(160, np.float32),
                        int(np.ceil(self.OUT_SR / max(len(x), 1))) + 1)[: self.OUT_SR + 1]
        ver = self._bc_verifier()
        with self._bc_emb_lock:          # ★2026-11-20 相槌を並行に作るようになったので（_targets_bc_prev_ref）、WavLM は 1 つずつ通す
            return ver.embed(x, self.OUT_SR)

    def bc_with_prev(self, text: str, ref: dict, prev_wav: np.ndarray, seed: int | None = None) -> np.ndarray:
        """相槌 text を、参照＝[その話者の直前の採用済み発話 prev_wav（OUT_SR）] ＋ 声バンク（本番と同じ 1 テイク＋兄弟テイク・
        計 refs_sec 秒）で作る（irodori.bc_prev_ref・2026-11-19）。pocket は自分の直前の発話を流し込んで続きを作る
        （pocket_context.self_turns）が、Irodori には続きを作る仕組みが無いので、声と話し方を写す参照の先頭に置く。
        短い発話は bc_prev_ref.takes（既定 short_takes）本引き、_short_scores − sim_weight×（直前の発話の末尾 3 秒との声の近さ・WavLM）で選ぶ。
        試験（_moss_tune/prevref_bc_test.py・prevref2_bc_test.py・相槌 48 個）: 声のまま切れ 2%→0%・人間の長さ比 1.17→1.10、
        近さの加点で直前とのつながり 0.674→0.697・本人らしさ 0.690→0.708。ユーザ聴取で一番（D1）。
        参照はファイルのパスで渡す（サーバは同じ NFS を見る。インラインの参照は vLLM-Omni が 1 秒未満を弾く）。
        直前の発話は tmp_dir に一時ファイルとして書き、合成後に消す。"""
        import random
        import tempfile
        tmp_dir = Path(str(self.bc_prev.get("tmp_dir") or "data/_tmp/irodori_prevref")).expanduser()
        if not tmp_dir.is_absolute():   # the Irodori server opens these files by path → always send absolute paths
            tmp_dir = Path(__file__).resolve().parent / tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        t = self._commas(re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", normalize_tts_text(text)))
        tx = t
        for a, b in (self.icfg.get("text_replace") or {}).items():
            tx = tx.replace(str(a), str(b))
        tmps = []
        try:
            fd, pp = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
            os.close(fd)
            tmps.append(pp)
            sf.write(pp, np.asarray(prev_wav, np.float32).reshape(-1), self.OUT_SR)
            ep0 = self._extra_params(t, ref["audio"])
            bank = list(ep0.get("ref_wavs") or [])
            if not bank:                       # 声バンク外の参照 → それもファイルで渡す（ref_wavs があるとインライン参照は使われない）
                fd, rp = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
                os.close(fd)
                tmps.append(rp)
                Path(rp).write_bytes(ref["audio"])
                bank = [rp]
            ep0["ref_wavs"] = [pp] + bank
            ep0["max_ref_seconds"] = self.refs_sec
            short = estimate_mora(t)[0] <= float(self.icfg.get("short_max_mora", 4))
            n = int(self.bc_prev.get("takes") or self.icfg.get("short_takes", 1) or 1) if short else 1
            base = int(seed) if seed is not None else random.randrange(1 << 30)
            stg = self.bc_prev.get("staged") or {}
            if stg and n > 1:
                # ★2026-11-20 B1（bc_prev_ref.staged）: 24 本を一度に引かず、段ごとに引いて十分よいものが出たら止める
                return self._bc_prev_staged(tx, t, ep0, ref, prev_wav, base, n, stg)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(n, 16))) as ex:
                ws = list(ex.map(lambda k: self._request(tx, dict(ep0), [ref], base + 7919 * k), range(n)))
        finally:
            for f in tmps:
                try:
                    os.unlink(f)
                except OSError:
                    pass
        if len(ws) == 1:
            return ws[0]
        sc = np.array(self._short_scores(ws, t, base, t.rstrip().endswith(("？", "?"))), dtype=float)
        w_sim = float(self.bc_prev.get("sim_weight", 3.0))
        if w_sim > 0:
            pe = self._bc_embed(np.asarray(prev_wav, np.float32).reshape(-1)[-int(3.0 * self.OUT_SR):])
            sc -= w_sim * np.array([float(self._bc_embed(w) @ pe) for w in ws])
        # 2026-11-19（irodori-yomi-v3）: 直前の自分の発話より大きく高い声（裏声）を減点。ユーザ「yomi_v6_667bd814 の 1:36 に
        # 明らかにおかしい相槌」＝男性（約 100Hz）の「あー。」が約 400Hz。本番 003 でも相槌 299 個中 7 個が直前より 11 半音以上高い。
        # 人間の相槌は直前の発話と比べて p25〜p75 で −3.4〜+1.8 半音（_moss_tune/human_bc_rel.json）。
        # 減点 = weight × clip((差 − lo) ÷ (hi − lo), 0, 1)。差＝相槌の中央値 − 直前の発話の末尾 3 秒の中央値（半音）
        pc = self.bc_prev.get("pitch") or {}
        w_p = float(pc.get("weight", 0.0))
        if w_p > 0:
            fp = median_f0(np.asarray(prev_wav, np.float32).reshape(-1)[-int(3.0 * self.OUT_SR):], self.OUT_SR)
            if fp:
                lo, hi = float(pc.get("lo", 4.0)), float(pc.get("hi", 8.0))
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                    fs = list(ex.map(lambda w: median_f0(w, self.OUT_SR), ws))
                st = np.array([12 * np.log2(f / fp) if f else 0.0 for f in fs])
                sc += w_p * np.clip((st - lo) / max(hi - lo, 1e-6), 0, 1)
        if self.bc_prev.get("score_log"):
            # ★2026-11-20 B1 の閾値を決めるための採点の記録（bc_prev_ref.score_log にパスがあるときだけ。選び方は変えない）
            out = self._clean_short(ws[int(np.argmin(sc))])
            self._score_log(self.bc_prev.get("score_log"), {"kind": "bc", "text": t, "base": int(base), "n_used": len(ws),
                                                             "n_max": int(n), "scores": [round(float(x), 4) for x in sc],
                                                             "chosen": int(np.argmin(sc)), "len": int(len(out))})
            return out
        return self._clean_short(ws[int(np.argmin(sc))])

    def _bc_prev_staged(self, tx: str, t: str, ep0: dict, ref: dict, prev_wav: np.ndarray, base: int, n: int,
                        stg: dict) -> np.ndarray:
        """★2026-11-20 B1（irodori.bc_prev_ref.staged: {first, step, thresh}）: 相槌の候補を first 本（既定 8）引いて採点し、
        いちばん良い点（bc_with_prev と同じ点＝_short_scores − sim_weight×近さ＋高すぎる声の減点。小さいほど良い）が thresh 以下なら
        そこで止め、超えていれば step 本（既定 first）ずつ足して takes（n）本まで引く。seed は従来と同じ base＋7919×k（k は 0 から順）なので、
        最初の段のテイクは従来の最初の first 本と同じ。点はテイクごとに他のテイクに依らないので、n 本まで引いたときは従来と同じものを選ぶ。
        採点の記録は bc_prev_ref.score_log（パスがあるときだけ）。"""
        first = max(1, int(stg.get("first", 8)))
        step = max(1, int(stg.get("step", first)))
        thr = float(stg.get("thresh", float("-inf")))
        q = t.rstrip().endswith(("？", "?"))
        pw = np.asarray(prev_wav, np.float32).reshape(-1)[-int(3.0 * self.OUT_SR):]
        w_sim = float(self.bc_prev.get("sim_weight", 3.0))
        pe = self._bc_embed(pw) if w_sim > 0 else None
        pc = self.bc_prev.get("pitch") or {}
        w_p = float(pc.get("weight", 0.0))
        fp = median_f0(pw, self.OUT_SR) if w_p > 0 else None
        lo_p, hi_p = float(pc.get("lo", 4.0)), float(pc.get("hi", 8.0))
        ws: list = []
        sc: list = []
        k0 = 0
        while k0 < n:
            k1 = min(n, first if k0 == 0 else k0 + step)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(k1 - k0, 16))) as ex:
                new = list(ex.map(lambda k: self._request(tx, dict(ep0), [ref], base + 7919 * k), range(k0, k1)))
            s = np.array(self._short_scores(new, t, base, q), dtype=float)
            if pe is not None:
                s -= w_sim * np.array([float(self._bc_embed(w) @ pe) for w in new])
            if w_p > 0 and fp:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                    fs = list(ex.map(lambda w: median_f0(w, self.OUT_SR), new))
                st = np.array([12 * np.log2(f / fp) if f else 0.0 for f in fs])
                s += w_p * np.clip((st - lo_p) / max(hi_p - lo_p, 1e-6), 0, 1)
            ws += new
            sc += [float(x) for x in s]
            k0 = k1
            if min(sc) <= thr:
                break
        j = int(np.argmin(sc))
        out = self._clean_short(ws[j])
        self._score_log(self.bc_prev.get("score_log"), {"kind": "bc", "text": t, "base": int(base), "n_used": len(ws),
                                                         "n_max": int(n), "scores": [round(x, 4) for x in sc],
                                                         "chosen": j, "len": int(len(out))})
        return out

    def _post_once(self, text: str, refs: list[dict], seed: int | None):
        text = self._commas(re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", text))
        ep = self._extra_params(text, refs[0]["audio"])
        for a, b in (self.icfg.get("text_replace") or {}).items():   # 合成入力だけの置換（例 イノウエ→井上。長さは置換前の文で）
            text = text.replace(str(a), str(b))
        le = self.icfg.get("laugh_emoji")
        if le:
            rest = self._LAUGH_HEAD.sub("", text, count=1)
            if rest != text and rest.strip():
                if le == "split":
                    # 笑いだけを別に作って（🤭🤭ふふっ。を何本か引いて笑いの音がしっかり出たもの）、残りの文と繋ぐ
                    # （2026-11-14 ユーザ聴取: 🤭 の置換だけ・別に作るだけでは笑いが出ないことがある。
                    #  🤭🤭ふふっ。が一番よく、🤭🤭ふふ。はたまに笑いにならず、🤭🤭🤭 は大げさ。_moss_tune/laugh_combo.py）
                    # ★2026-11-20 seed が無いとき（base の最初のテイク）は乱数にする。それまでは gen の seed も無いので 0 に落ち、
                    #   笑いの 4 本が毎回 seed 104729・209458… になって、同じ声の笑いが全部の発話・全部の対話で同じ音だった
                    sd_gen = self._gen_for(refs[0].get("text")).get("seed")
                    if seed is not None:
                        sd0 = int(seed)
                    elif sd_gen is not None:
                        sd0 = int(sd_gen)
                    else:
                        import random
                        sd0 = random.randrange(1 << 30)
                    # ★2026-11-20 B4（irodori.laugh_reuse）: ターンごとに 1 回だけ作った笑いを使い回す（既定は従来どおりテイクごとに作る）
                    lw = self._laugh_cached(text, refs, sd0) if self.icfg.get("laugh_reuse") else self._laugh(refs, sd0)
                    w = self._post_once(rest, refs, seed)
                    gap = np.zeros(int(float(self.icfg.get("laugh_gap", 0.15)) * self.OUT_SR), np.float32)
                    return np.concatenate([lw, gap, w]).astype(np.float32)
                text = "🤭" + rest
        return self._request(text, ep, refs, seed)

    @staticmethod
    def _active_sec(w: np.ndarray, sr: int, floor_db: float = -40.0) -> float:
        """10ms ごとの RMS が floor_db を超える長さ（秒）。"""
        h = int(0.01 * sr)
        if len(w) <= h:
            return 0.0
        e = np.sqrt(np.mean(w[: len(w) // h * h].reshape(-1, h) ** 2, axis=1))
        return float((20 * np.log10(e + 1e-9) > floor_db).sum() * 0.01)

    def _laugh(self, refs: list[dict], seed: int) -> np.ndarray:
        """laugh_emoji: split の笑いの部分。laugh_text（既定「🤭🤭ふふっ。」）を laugh_seconds（0.7 秒）で
        laugh_takes 本（既定 4）引き、笑いの音の長さ（有音）が laugh_active（0.35 秒）に一番近いものを採る
        （同じ文でも 8 seed で有音 0.12〜0.51 秒とばらつき、短いものは笑いに聞こえない）。"""
        lt = str(self.icfg.get("laugh_text", "🤭🤭ふふっ。"))
        ep = self._extra_params("ふふっ。", refs[0]["audio"])
        ep["seconds"] = float(self.icfg.get("laugh_seconds", 0.7))
        n = max(1, int(self.icfg.get("laugh_takes", 4)))
        seeds = [seed + 104729 * (k + 1) for k in range(n)]
        if n == 1:
            ws = [self._request(lt, dict(ep), refs, seeds[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                ws = list(ex.map(lambda s: self._request(lt, dict(ep), refs, s), seeds))
        tgt = float(self.icfg.get("laugh_active", 0.35))
        return ws[int(np.argmin([abs(self._active_sec(w, self.OUT_SR) - tgt) for w in ws]))]

    def _irodori_live_url(self) -> str:
        """★2026-11-20 生きているサーバを順番に 1 つ返す（_moss_once と同じ振り方）。全部死んでいれば TtsServerDead。"""
        with self._url_lock:
            live = [u for u in self._urls if u not in self._dead_urls]
            if not live:
                self.all_servers_dead.set()
                raise TtsServerDead(f"TTS サーバが全部応答しない（{', '.join(self._urls)}）")
            url = live[self._url_i % len(live)]
            self._url_i += 1
            return url

    def _irodori_note_ok(self, url: str) -> None:
        """★2026-11-20 成功: 続けての失敗の数を 0 に戻し、最後に成功した時刻を残す（タイムアウトの判定に使う）。"""
        with self._url_lock:
            self._url_fail[url] = 0
            self._url_ok_t[url] = time.time()

    def _irodori_note_fail(self, url: str, e: Exception, to: float) -> None:
        """★2026-11-20 通信の失敗を数え、dead_after 回続いたらそのサーバを以後使わない（全部死んだら all_servers_dead）。
        読み取りのタイムアウトは、直近 to 秒にそのサーバで成功があれば数えない（束ね待ちで止まった束だけのことがあるため）。"""
        timeout_only = (isinstance(e, self._requests.exceptions.Timeout)
                        and not isinstance(e, self._requests.exceptions.ConnectionError))
        with self._url_lock:
            last_ok = self._url_ok_t.get(url)
            if timeout_only and last_ok is not None and time.time() - last_ok < to:
                return
            k = self._url_fail.get(url, 0) + 1
            self._url_fail[url] = k
            if k >= self.dead_after and url not in self._dead_urls:
                self._dead_urls.add(url)
                logger.error(f"TTS サーバ {url} が {k} 回続けて応答しない（{type(e).__name__}）→ 以後このサーバは使わない"
                             f"（GPU が落ちていないか nvidia-smi で確かめる）")
                if len(self._dead_urls) >= len(self._urls):
                    self.all_servers_dead.set()
                    logger.error("TTS サーバが全部応答しない → このワーカーは今の対話を捨てて終わる")

    def _request(self, text: str, ep: dict, refs: list[dict], seed: int | None) -> np.ndarray:
        """1 回の /v1/audio/speech（text は合成入力そのまま・ep は extra_params）。"""
        ep = dict(ep)
        body = {"model": self.model, "input": text, "response_format": "wav",
                "ref_text": refs[0].get("text", ""), "extra_params": ep}
        if "ref_wavs" not in ep:
            # vLLM-Omni はインラインの参照音声を 30 秒までしか受けない → バンク外の参照は先頭 30 秒に切って送る
            body["ref_audio"] = self._ref_data_url(_trim_wav_bytes(refs[0]["audio"], 30.0))
        gk = self._gen_for(refs[0].get("text"))
        sd = seed if seed is not None else gk.get("seed")
        if sd is not None:
            body["seed"] = int(sd)
        # まれに vLLM-Omni の束ね待ちで要求が戻らない（2026-11-14 相槌 4 本の束が 5 分止まり、切ったら次は通った）
        # → 1 回の待ちを request_timeout（既定 180 秒）で切って、同じ seed で retries 回まで投げ直す
        tries = 1 + int(self.icfg.get("retries", 2))
        to = min(self.timeout, float(self.icfg.get("request_timeout", 180)))
        # ★2026-11-20 死んだサーバの遮断（_moss_once と同じ仕組み）を Irodori にも通す。それまでは self.url に直接投げていて
        #   all_servers_dead が立たず、サーバが落ちるとワーカーは接続エラーを即座に 3 回ずつ繰り返しながら対話を次々に取っては
        #   無音代替で捨てていた（2026-11-20 本番ログ: 応答なし 144,060 回・無音破棄 1,099 対話）。
        #   接続できない（ConnectionError＝プロセスがいない）は dead_after 回続けば即「死んだ」とする。
        #   タイムアウト（束ね待ちで戻らない）は、そのサーバで直近 request_timeout 秒の間に成功が 1 件も無いときだけ数える
        #   （同時 32 本のうち止まった束の数本だけが続けてタイムアウトしても、他の要求が通っていれば生きているので）。
        for k in range(tries):
            url = self._irodori_live_url()          # 全部死んでいれば TtsServerDead
            try:
                resp = self._session.post(url, json=body, timeout=to)
            except (self._requests.exceptions.Timeout, self._requests.exceptions.ConnectionError) as e:
                self._irodori_note_fail(url, e, to)
                if self.all_servers_dead.is_set():
                    raise TtsServerDead(f"TTS サーバが全部応答しない（{', '.join(self._urls)}）") from e
                if k == tries - 1:
                    raise
                logger.warning(f"Irodori(vLLM-Omni) 応答なし（{type(e).__name__}）→ 投げ直し {k + 1}/{tries - 1}: {text[:20]}")
                continue
            self._irodori_note_ok(url)
            break
        if resp.status_code != 200:
            raise RuntimeError(f"Irodori(vLLM-Omni) {resp.status_code}: {resp.text[:300]}")
        audio, sr = self._decode_wav(resp.content)
        audio = self._to_out_sr(audio, sr)
        if self.icfg.get("mute_end_click", True):
            # 出力の最後の数十 ms に、無音の後のクリック（プチッ。最大 −10dBFS ほど）が付くことがある
            # （2026-11-14 _moss_tune/tail_test.py: 680 テイク中 35 本。6 割がファイル末尾 40ms 以内・残りも 0.3 秒以内）。
            # 無音（0.15 秒以上）の後の 60ms 未満の音は声（1 モーラ 80ms 以上）ではない → その音だけ無音に
            tb = tail_blob(audio, self.OUT_SR)
            if tb is not None and tb[1] - tb[0] < int(0.06 * self.OUT_SR):
                audio = mute_span(audio, tb[0], tb[1], self.OUT_SR, fade=0.005)
        oc = self.icfg.get("outer_clean") or {}
        if oc:
            # irodori.outer_clean（2026-11-19・irodori-yomi-v2）: すべての発話の最初の声の前・最後の声の後ろにある短いクリックを無音に
            # （outer_clicks。mute_end_click は 0.15 秒以上離れたものだけ）
            for a, b in outer_clicks(audio, self.OUT_SR, max_click=float(oc.get("max_click", 0.06)),
                                     min_gap=float(oc.get("min_gap", 0.06)), max_db=float(oc.get("max_db", -20.0))):
                audio = mute_span(audio, a, b, self.OUT_SR, fade=0.005)
        if self.edges:
            # irodori.edges（2026-11-19）: 前後の余った無音を整え（pocket と同じ・語尾は削らない）、声のまま切れたテイクは
            # 末尾をフェードアウト（「プツッ」を消す）。整えないと頭 0.24 秒・尻 0.4 秒の無音がターン間の間を広げる
            audio = trim_edges(audio, self.OUT_SR, lead_keep=float(self.edges.get("lead_keep", 0.05)),
                               tail_keep=float(self.edges.get("tail_keep", 0.15)),
                               end_fade=float(self.edges.get("end_fade", 0.03)),
                               start_fade=float(self.edges.get("start_fade", 0.0)))
        return audio

    def _clean_short(self, w: np.ndarray) -> np.ndarray:
        """選んだ短い発話（相槌）の前後の小さな余計な音（irodori.short_clean.max_click 秒以下・min_gap 秒以上離れた塊）を無音にし、
        前後の無音を整え直す（2026-11-19・相槌の後ろ 10〜80ms のクリック）。声の塊（1 モーラ 80ms 以上）は残る長さにする。"""
        sc = self.icfg.get("short_clean") or {}
        if not sc or w is None or len(w) == 0:
            return w
        mx = float(sc.get("max_click", 0.1))
        bl = [(a, b) for a, b in edge_blobs(w, self.OUT_SR, min_gap=float(sc.get("min_gap", 0.08)), max_blob=mx)]
        if not bl:
            return w
        for a, b in bl:
            w = mute_span(w, a, b, self.OUT_SR, fade=0.005)
        if self.edges:
            w = trim_edges(w, self.OUT_SR, lead_keep=float(self.edges.get("lead_keep", 0.05)),
                           tail_keep=float(self.edges.get("tail_keep", 0.15)),
                           end_fade=float(self.edges.get("end_fade", 0.03)),
                           start_fade=float(self.edges.get("start_fade", 0.0)))
        return w


# --------------------------------------------------------------------------
# ボイス割り当て
# --------------------------------------------------------------------------

def _main_speaker_name(record: dict) -> str | None:
    """metadata.main_speaker_index が指す話者名。無い/壊れていれば None。

    主話者を「30代男性」のような属性で探さないのは、男性同士の対話では両方が
    該当してしまい一意に決まらないため。Stage1 が明示的に付けた index だけを見る。
    """
    md = record.get("metadata") or {}
    i = md.get("main_speaker_index")
    if i is None:
        return None
    try:
        return md["speakers"][int(i)].get("name")
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning(f"[{record.get('id')}] main_speaker_index={i!r} を解決できません。")
        return None


def _parse_age(age_range) -> int | None:
    """'30代' / '20代後半' / 35 などから代表年齢(int)を取り出す。"""
    if age_range is None:
        return None
    if isinstance(age_range, (int, float)):
        return int(age_range)
    digits = "".join(ch for ch in str(age_range) if ch.isdigit())
    return int(digits[:2]) if digits else None


def resolve_speaker_attrs(record: dict, speaker_names: list[str]) -> dict:
    """各話者の {gender, age} を解決する。

    性別は metadata.speakers[].gender（Stage 1 が決めた値）→ Stage 3 が付けた
    speaker_profiles → name/role からの推定 の順に見る。metadata を最優先に
    するのは、名前・役割・シナリオ本文をその性別として書いた当人の値だから。
    gender を持たない古い metadata では従来どおり speaker_profiles が使われる。
    年齢は speaker_profiles → metadata の age_range から取得。
    """
    profiles = {p["name"]: p for p in record.get("speaker_profiles", []) if p.get("name")}
    meta_by_name = {s.get("name"): s for s in record["metadata"].get("speakers", [])}
    main_name = _main_speaker_name(record)

    attrs = {}
    for name in speaker_names:
        prof = profiles.get(name, {})
        meta = meta_by_name.get(name, {})
        gender = (normalize_gender(meta.get("gender"))
                  or normalize_gender(prof.get("gender"))
                  or infer_gender(name, meta.get("role")))
        age = _parse_age(prof.get("age_range") if prof.get("age_range") is not None
                         else meta.get("age_range"))
        # main は「ch0 に固定する主話者か」。assign_voices はこれを見て
        # 主話者専用バンクと一般バンクを排他的に使い分ける。呼び出し元を
        # 一切変えずに済むよう attrs に相乗りさせている（6箇所すべてが
        # assign_voices の直前に resolve_speaker_attrs を呼んでいる）。
        attrs[name] = {"gender": gender, "age": age,
                       "main": main_name is not None and name == main_name}
    return attrs


def _voice_score(voice: dict, target_gender: str | None, target_age: int | None,
                 age_tol: float = 0.0) -> tuple:
    """声と話者属性の適合度。小さいほど良い (gender不一致ペナルティ, 年齢差)。

    age_tol > 0 なら年齢差から age_tol を引く（下限0）。差が age_tol 以内の声は
    すべて同点になり、assign_voices の tiebreak で散る。tol を超える分の順序は
    残るので、近い年代が尽きたときの候補の並びは変わらない。
    """
    gender_penalty = 0
    if target_gender and voice.get("gender") and voice["gender"] != target_gender:
        gender_penalty = 1000
    age_diff = 0.0
    va = _parse_age(voice.get("age_range"))
    if target_age is not None and va is not None:
        age_diff = max(0.0, abs(va - target_age) - max(0.0, age_tol))
    return (gender_penalty, age_diff)


def _voice_speaker_key(voice_id: str) -> str:
    """テイク違い（maai_..._t1 / _t2 等）を同一人物とみなすためのキー。"""
    return re.sub(r"_t\d+$", "", voice_id)


def assign_voices(dialogue_id: str, speaker_names: list[str], voices: list[dict],
                  attrs: dict, age_tol: float | None = None,
                  min_pool: int | None = None) -> dict:
    """各話者に、性別・年代が合う声を互いに重複しないよう割り当てる。

    手順: 話者ごとに (性別不一致ペナルティ, 年齢差) でソートし、最良群から
    dialogue_id ベースで決定的に1つ選ぶ。2話者で同じ声は使わず、
    同一人物の別テイク（_tK 接尾辞違い）も同じ対話には使わない。

    年代の緩め方（age_tol / min_pool。既定は voices が VoiceBank ならその値、
    素の list なら 0＝従来の厳密一致）:
      age_tol  … 年齢差がこの年数以内の声を同点にする幅。同点は tiebreak で散る。
      min_pool … 厳密一致（＝年齢差が最小）の候補がこの本数以上あるときは緩めない。
                 つまり **声が薄い年代だけ** 隣の年代へ広げる。

    緩めないと需要の多い年代に集中する。solo30_ok(68本) で 5000対話・話者10000を
    割り当てた実測: male 30代 は需要 22.8% に対して声が2本しかなく、その2本だけで
    全話者の22%（最多1本で 11.4%、上位5本で 33.0%、出番50回未満が26本）を担う。
      tol=10 / min_pool=4 … 完全一致 67.1% / 最多 4.1% / 上位5本 19.6% / 出番50回未満 4本
      tol=10 / min_pool=0 … 完全一致 40.8% / 最多 3.4% / 上位5本 15.6%（供給が足りている
                            年代まで散らすので、集中はわずかに改善するが年代がよくずれる）
    どちらも ±1年代に収まる割合は 99.4% で、緩める前（99.4%）と同じ。tol=20 まで
    広げると ±1年代が 73.2% に落ちるので、隣の年代までに留める tol=10 を既定に選んだ。
    """
    if age_tol is None:
        age_tol = getattr(voices, "age_tol", 0.0)
    if min_pool is None:
        min_pool = getattr(voices, "age_min_pool", 0)
    n = len(voices)
    if n < len(speaker_names):
        raise ValueError(f"声の数({n})が話者数({len(speaker_names)})より少ないため割り当てできません。")

    main_ids = set(getattr(voices, "main_voice_ids", ()) or ())

    assignment = {}
    used = set()
    for idx, name in enumerate(speaker_names):
        a = attrs.get(name, {})
        cands = [v for v in voices if _voice_speaker_key(v["id"]) not in used]
        if not cands:
            raise ValueError(
                f"話者 {name} に割り当てられる声が残っていません（全候補が使用済み話者のテイク）。")

        # 主話者(ch0)専用バンクがあるときは、プールを排他的に二分する。
        # attrs[name]["main"] は resolve_speaker_attrs が metadata.main_speaker_index
        # から立てる。主話者は record_speaker_order により常に idx==0 に来るので、
        # 先に割り当てられ、専用プールが枯れることはない。
        if main_ids:
            if a.get("main"):
                cands = [v for v in cands if v["id"] in main_ids]
            else:
                cands = [v for v in cands if v["id"] not in main_ids]
            if not cands:
                raise ValueError(
                    f"話者 {name} に割り当てられる声が残っていません "
                    f"(main={bool(a.get('main'))}, 主話者バンク{len(main_ids)}本)。")

        tol = age_tol
        if tol > 0 and min_pool > 1:
            # 厳密一致の候補が足りている年代は緩めない（この話者だけの判断）
            strict = [_voice_score(v, a.get("gender"), a.get("age")) for v in cands]
            best = min(strict)
            if sum(1 for s in strict if s == best) >= min_pool:
                tol = 0.0

        def _key(v):
            # スコア（性別不一致, 年齢差）→ 同点は dialogue×声で決定的に分散
            tiebreak = int(hashlib.md5(
                f"{dialogue_id}:{idx}:{v['id']}".encode("utf-8")).hexdigest(), 16)
            return _voice_score(v, a.get("gender"), a.get("age"), tol) + (tiebreak,)

        cands.sort(key=_key)
        chosen = cands[0]
        # 性別が合わない声が選ばれたら黙って通さない。話者の性別が None（推定にも
        # 失敗）か、その性別の声がバンクに残っていない、のどちらか。
        if a.get("gender") is None:
            logger.warning(f"話者 {name} の性別が不明のため、性別を考慮せずに声を選びました "
                           f"(voice={chosen['id']} {chosen.get('gender')})")
        elif chosen.get("gender") and chosen["gender"] != a["gender"]:
            logger.warning(f"話者 {name}({a['gender']}) に {chosen.get('gender')} の声を"
                           f"割り当てました（同性の声が残っていません: voice={chosen['id']}）")
        assignment[name] = chosen
        used.add(_voice_speaker_key(chosen["id"]))
    return assignment


# --------------------------------------------------------------------------
# 1対話の合成
# --------------------------------------------------------------------------

def _core_len(text: str) -> int:
    """末尾の句読点・空白を除いた実質の文字数。"""
    return len(text.strip().rstrip(_SENTENCE_END).strip())


# --- モーラ数の見積り（話速の計測に使う）------------------------------------
# 文字数ではなくモーラ数で測る。漢字は読みが長く（会議=4モーラ/2字）、拗音は
# 1モーラなので、文字数のままだと漢字の多い発話が不当に「速い」と判定される。
_MORA_SMALL = "ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ"      # 拗音・小書き（0モーラ）
_MORA_IGNORE = "、。！？!?,.　 \t「」『』（）()［］[]…・〜―—-\"”'’"
_MORA_KANA = re.compile(r"[ぁ-んァ-ヶーｦ-ﾟ]")   # 長音「ー」も1モーラ（ラーメン=4）
_MORA_KANJI = re.compile(r"[一-龥々〆ヵ]")
_MORA_DIGITS = re.compile(r"[0-9０-９]+(?:[.．][0-9０-９]+)?")
_MORA_LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]+")
_KANJI_MORA = 2.0            # 漢字1字あたりの平均モーラ数（会議・時間・大丈夫 いずれも2.0）
# 0〜9 の読みのモーラ数: ゼロ/いち/に/さん/よん/ご/ろく/なな/はち/きゅう
_DIGIT_MORA = (2, 2, 1, 2, 2, 1, 2, 2, 2, 2)
_UNIT_MORA = (0, 2, 2, 2)    # 一の位 / じゅう / ひゃく / せん
_BIG_UNIT_MORA = (0, 2, 2, 2)  # （万）/ まん / おく / ちょう
_ZEN_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")


def _int_mora(digits: str) -> float:
    """整数の読みのモーラ数（例: 1000→せん=2、2026→にせんにじゅうろく=8）。

    位が1のときは「いち」を言わない（100=ひゃく）ぶんを落とす。桁が大きすぎる
    ものは1桁ずつ読む扱いにする（電話番号などの列挙はそのほうが近い）。
    """
    d = digits.lstrip("0") or "0"
    if len(d) > 12:
        return sum(_DIGIT_MORA[int(c)] for c in d)
    total = 0.0
    groups = []                                   # 下位4桁ずつに分ける
    while d:
        groups.append(d[-4:])
        d = d[:-4]
    for gi, g in enumerate(groups):               # gi=0 が万未満
        sub = 0.0
        for pos, c in enumerate(reversed(g)):     # pos: 0=一, 1=十, 2=百, 3=千
            n = int(c)
            if n == 0:
                continue
            sub += _UNIT_MORA[pos]
            if not (pos > 0 and n == 1):          # 十/百/千 の「いち」は言わない
                sub += _DIGIT_MORA[n]
        if sub:
            total += sub + _BIG_UNIT_MORA[min(gi, 3)]
    return total or _DIGIT_MORA[0]


def _num_mora(token: str) -> float:
    """数（小数を含む）の読みのモーラ数。小数点は「てん」= 2モーラ。"""
    t = token.translate(_ZEN_DIGITS)
    if "." in t:
        a, _, b = t.partition(".")
        return _int_mora(a) + 2.0 + sum(_DIGIT_MORA[int(c)] for c in b)
    return _int_mora(t)


def _latin_mora(token: str) -> float:
    """ラテン文字の読みのモーラ数（概算）。

    短い大文字の並びは頭字語としてアルファベット読み（PDF=ピーディーエフ=6）、
    それ以外は英単語としてカタカナ読み（Google=グーグル=4 / Microsoft=マイクロソフト=7）
    とみなす。1文字あたり 2.5 / 0.8 モーラで近似する。
    """
    if token.isupper() and len(token) <= 5:
        return 2.5 * len(token)
    return 0.8 * len(token)


def estimate_mora(text: str) -> tuple[float, float]:
    """テキストのモーラ数の見積りと、その確からしさを返す。

    Returns: (mora, confidence)。confidence は「かな・漢字から来たモーラの割合」で、
    数字やラテン文字が多い発話ほど低くなる（読み方が一意でないため）。
    """
    if not text:
        return 0.0, 0.0
    sure = 0.0                                    # かな・漢字ぶん（読み長がほぼ確定）
    guess = 0.0                                   # 数字・ラテン文字ぶん（概算）
    rest = text
    for m in _MORA_DIGITS.finditer(rest):
        guess += _num_mora(m.group())
    rest = _MORA_DIGITS.sub("", rest)
    for m in _MORA_LATIN.finditer(rest):
        guess += _latin_mora(m.group())
    rest = _MORA_LATIN.sub("", rest)
    for ch in rest:
        if ch in _MORA_IGNORE or ch in _MORA_SMALL:
            continue
        if _MORA_KANA.match(ch):
            sure += 1.0
        elif _MORA_KANJI.match(ch):
            sure += _KANJI_MORA
        else:
            guess += 1.0
    total = sure + guess
    return total, (sure / total if total > 0 else 0.0)


def trim_silence(wav: np.ndarray, sr: int, threshold: float = 0.02,
                 thresh_abs: float = 0.005, keep_pad_sec: float = 0.03,
                 max_trim_sec: float = 0.8) -> np.ndarray:
    """発話の前後の無音をトリミングする（Gemini-TTS の前後パディング除去）。

    振幅がピークの threshold（かつ絶対下限 thresh_abs）を超える範囲を有音とみなし、
    その外側の無音を落とす。端に keep_pad 秒だけ余白を残し、片側で max_trim_sec までしか
    削らない（柔らかい語頭/語尾の切りすぎ防止）。内部のポーズは切らない（前後のみ）。
    """
    if wav is None or len(wav) == 0:
        return wav
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    env = np.abs(w)
    peak = float(env.max())
    if peak <= 0:
        return wav
    t = max(threshold * peak, thresh_abs)
    nz = np.where(env > t)[0]
    if len(nz) == 0:
        return wav
    pad = int(keep_pad_sec * sr)
    mt = int(max_trim_sec * sr)
    start = max(0, nz[0] - pad)
    start = min(start, mt)                      # 前側は max_trim まで
    end = min(len(w), nz[-1] + 1 + pad)
    end = max(end, len(w) - mt)                 # 後側は max_trim まで
    if start >= end:
        return wav
    return wav[start:end]


def speech_level_db(wav: np.ndarray, sr: int, frame_ms: float = 20.0,
                    rel: float = 0.10) -> float | None:
    """発話区間だけの RMS を dBFS で返す（無音・パディングを除いた「声の大きさ」）。

    ピーク比 rel 以上のフレームを有音とみなす。相対しきい値なので、レベルの違う
    ターン同士でも同じ基準で測れる（絶対しきい値だと小さいターンが全部無音になる）。
    """
    if wav is None or len(wav) == 0:
        return None
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    n = int(sr * frame_ms / 1000.0)
    if n <= 0 or len(w) < n:
        return None
    nfr = len(w) // n
    rms = np.sqrt((w[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    if rms.max() <= 0:
        return None
    act = rms >= max(rel * float(rms.max()), 1e-4)
    if not act.any():
        return None
    return float(20.0 * np.log10(float(np.sqrt((rms[act] ** 2).mean())) + 1e-12))


def loudness_gains(turns: list, wav_by_turn: dict[int, np.ndarray], sr: int,
                   cfg: dict, failed_turns: set[int] | None = None) -> dict[int, float]:
    """ターンごとの音量補正（線形ゲイン）を決める。配置・長さには一切触らない。

    fish-speech はターンを1コールずつ独立に生成するため、出力レベルがターンごとに
    揺れる（実測: 同一話者・同一参照で sd 2.4〜5.2dB、隣接ターン間で最大18dB の跳び）。
    参照音声を揃えても直らない（参照RMS→出力RMS の相関は r=0.38 しかなく、参照の
    幅3.6dB に対し出力の幅は11.7dB）。生成パラメータでは制御できないので、ここで均す。

    やること: 各ターンの発話RMSを target_dbfs に寄せる。ただし deadband_db 以内の
    ずれはそのまま残す（自然な強弱まで潰さないため）。補正量は max_gain_db で頭打ち。
    相槌は bc_offset_db だけ低い目標にする（相槌が小さめなのは自然なので保存する）。
    """
    failed_turns = failed_turns or set()
    target = float(cfg.get("target_dbfs", -20.0))
    dead = float(cfg.get("deadband_db", 2.0))
    ratio = float(cfg.get("ratio", 1.0))              # deadband 超過分をどれだけ戻すか
    max_gain = float(cfg.get("max_gain_db", 12.0))
    ceiling = float(cfg.get("peak_ceiling_dbfs", -1.0))
    bc_off = float(cfg.get("bc_offset_db", -3.0))
    min_sec = float(cfg.get("min_sec", 0.2))          # これ未満は測定が不安定なので触らない

    gains: dict[int, float] = {}
    for i, t in enumerate(turns):
        w = wav_by_turn.get(i)
        if w is None or len(w) < int(min_sec * sr) or i in failed_turns:
            continue
        lvl = speech_level_db(w, sr)
        if lvl is None:
            continue
        tgt = target + (bc_off if t.get("type") == "backchannel" else 0.0)
        dev = lvl - tgt
        corr = 0.0 if abs(dev) <= dead else (abs(dev) - dead) * ratio * np.sign(dev)
        g_db = float(np.clip(-corr, -max_gain, max_gain))
        if g_db == 0.0:
            continue
        g = float(10.0 ** (g_db / 20.0))
        peak = float(np.abs(w).max())
        if peak > 0:                                   # 上げすぎてクリップさせない
            g = min(g, (10.0 ** (ceiling / 20.0)) / peak)
        if abs(20.0 * np.log10(max(g, 1e-6))) >= 0.1:  # 0.1dB未満は無視
            gains[i] = g
    return gains


def speed_factors(turns: list, wav_by_turn: dict[int, np.ndarray], sr: int,
                  cfg: dict, failed_turns: set[int] | None = None) -> dict[int, float]:
    """ターンごとの話速補正（time_stretch に渡す係数。<1 で遅くする）を決める。

    fish-speech には話速の指定が無く（ServeTTSRequest に該当パラメータが無い）、
    生成パラメータでも参照音声でも決まらない。実測: 参照音声の話速 → 出力の話速は
    r=0.46・傾き0.37 しかなく、参照が2倍遅くても出力はモデル自身の速さへ引き戻される。
    そして出力は人対人より速い（実質モーラ/秒の中央値は合成 8.49、人間 maai-f2f 7.29）。
    直せるのはここ（後処理）だけなので、ターン単位で伸ばして均す。

    速さは「モーラ数 ÷ ターンの長さ（トリム後・ターン内のポーズ込み）」で測る。
    ポーズ込みで測るのが要点で、読点で間を取った発話は自然に「遅い側」に出るため
    触られない。つまり Stage2 が入れた間をこの後処理が食い潰すことがない。
    既定では遅くする方向にしか動かさない（max_faster=1.0）。

    数字・ラテン文字が多く読み長が推定しにくいターン（実測で17%）は、その対話・
    その話者で測れたターンの中央値の係数を当てる（1本だけ浮かないように）。
    """
    failed_turns = failed_turns or set()
    target = float(cfg.get("target_mora_per_sec", 7.3))
    dead = float(cfg.get("deadband", 0.3))
    ratio = float(cfg.get("ratio", 0.75))          # しきい値超過分をどれだけ戻すか
    min_f = float(cfg.get("max_slower", 0.80))     # 係数の下限（0.80 = 25%長くする）
    max_f = float(cfg.get("max_faster", 1.0))      # 係数の上限（1.0 = 速くはしない）
    min_mora = float(cfg.get("min_mora", 8.0))     # これ未満は測定が不安定
    min_sec = float(cfg.get("min_sec", 0.5))
    min_conf = float(cfg.get("min_confidence", 0.6))
    do_bc = bool(cfg.get("backchannel", False))    # 相槌は既定で触らない

    def factor_for(rate: float) -> float:
        if rate > target + dead:
            new = rate - ratio * (rate - (target + dead))
        elif max_f > 1.0 and rate < target - dead:
            new = rate + ratio * ((target - dead) - rate)
        else:
            return 1.0
        return float(np.clip(new / rate, min_f, max_f))

    factors: dict[int, float] = {}
    unsure: list[tuple[int, str]] = []              # (index, speaker) 読み長が不確かなターン
    by_speaker: dict[str, list[float]] = {}
    for i, t in enumerate(turns):
        w = wav_by_turn.get(i)
        if w is None or i in failed_turns or len(w) < int(min_sec * sr):
            continue
        if not do_bc and t.get("type") == "backchannel":
            continue
        mora, conf = estimate_mora(t.get("text", ""))
        if mora < min_mora:
            continue
        if conf < min_conf:
            unsure.append((i, t["speaker"]))
            continue
        f = factor_for(mora / (len(w) / sr))
        by_speaker.setdefault(t["speaker"], []).append(f)
        if abs(f - 1.0) >= 0.01:                    # 1%未満は掛けない
            factors[i] = f
    for i, sp in unsure:                            # 測れたターンの中央値で代用
        fs = by_speaker.get(sp)
        if not fs:
            continue
        f = float(np.median(fs))
        if abs(f - 1.0) >= 0.01:
            factors[i] = f
    return factors


def voiced_f0(wav: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 450.0,
              frame_ms: float = 40.0, hop_ms: float = 20.0,
              min_corr: float = 0.35) -> np.ndarray:
    """有声フレームの F0（Hz）を返す。自己相関（FFT）による軽量版。

    ★探索範囲は性別で切ること。範囲を広く取ると倍音の周期を掴んでオクターブ誤りを
    起こし、抑揚が実際より大きく出る（男声で17%のフレームが誤った実績がある）。
    男 60〜260Hz / 女 110〜450Hz を使う（`_F0_RANGE`）。
    """
    if wav is None or len(wav) < int(sr * frame_ms / 1000.0):
        return np.empty(0, dtype=np.float32)
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    n = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    lag_min = max(2, int(sr / fmax))
    lag_max = min(n - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return np.empty(0, dtype=np.float32)
    nfr = 1 + (len(w) - n) // hop
    frames = np.lib.stride_tricks.as_strided(
        w, shape=(nfr, n), strides=(w.strides[0] * hop, w.strides[0])).copy()
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    live = rms >= max(0.15 * float(rms.max()), 1e-4)      # 無音フレームは測らない
    if not live.any():
        return np.empty(0, dtype=np.float32)
    fr = frames[live] - frames[live].mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(fr, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:, :lag_max + 1]
    zero = ac[:, :1]
    norm = np.divide(ac, zero, out=np.zeros_like(ac), where=zero > 1e-12)
    seg = norm[:, lag_min:lag_max + 1]
    lag = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), lag]
    ok = peak >= min_corr
    if not ok.any():
        return np.empty(0, dtype=np.float32)
    return (sr / (lag[ok] + lag_min)).astype(np.float32)


_F0_RANGE = {"male": (60.0, 260.0), "female": (110.0, 450.0)}


def f0_sd_semitones(wav: np.ndarray, sr: int, gender: str | None = None) -> float | None:
    """有声F0の半音sd（＝抑揚の大きさ）。測れなければ None。"""
    fmin, fmax = _F0_RANGE.get(gender or "", (60.0, 450.0))
    f0 = voiced_f0(wav, sr, fmin=fmin, fmax=fmax)
    if len(f0) < 5:
        return None
    return float(np.std(12.0 * np.log2(f0 / 100.0)))


def tail_blob(wav: np.ndarray, sr: int, min_gap: float = 0.15, max_blob: float = 0.8,
              rel_db: float = -35.0) -> tuple[int, int] | None:
    """発話末の「無音（min_gap 秒以上）の後の短い有音区間（max_blob 秒以下）」の (始め, 終わり) サンプル。無ければ None。
    Irodori は指定の長さを埋めるとき、文の後に「ああ」「え？」や小さなクリック・息を足すことがある
    （2026-11-14 ユーザ「発話の末尾にたまに短い音声がハルシネーションのように入る」。_moss_tune/tail_test.py:
    680 テイク中 30 本が台本外の音。多くは 20〜100ms のクリック、聞こえる大きさの声が 1〜2%）。
    読点の後の最後の語（「…、あります。」）も同じ形になるので、台本の一部かどうかは呼び出し側で ASR で見る。"""
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 10:
        return None
    db = 20 * np.log10(np.sqrt((w[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    on = db > max(db.max() + rel_db, -55.0)
    idx = np.nonzero(on)[0]
    if len(idx) == 0:
        return None
    runs = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i - runs[-1][1] < 5:                  # 50ms 未満の切れ目はつなぐ
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    if len(runs) < 2:
        return None
    (_, e0), (a, b) = runs[-2], runs[-1]
    if (a - e0) * 0.01 < min_gap or (b - a) * 0.01 > max_blob:
        return None
    return a * h, min(len(w), b * h)


def mute_span(wav: np.ndarray, a: int, b: int, sr: int, fade: float = 0.01) -> np.ndarray:
    """wav[a:b] を無音にする（前後 fade 秒でなめらかに。長さは変えない）。"""
    y = np.array(wav, dtype=np.float32, copy=True)
    f = int(fade * sr)
    a0, b1 = max(0, a - f), min(len(y), b + f)
    g = np.ones(b1 - a0, np.float32)
    g[a - a0: b - a0] = 0.0
    if a - a0 > 0:
        g[: a - a0] = np.linspace(1.0, 0.0, a - a0, dtype=np.float32)
    if b1 - b > 0:
        g[b - a0:] = np.linspace(0.0, 1.0, b1 - b, dtype=np.float32)
    y[a0:b1] = (y[a0:b1].T * g).T
    return y


def voiced_at_end(wav: np.ndarray, sr: int) -> bool:
    """声のまま切れている（ファイル末尾の手前 30〜50ms が声＝ピーク−25dB 超で、その有音が 0.15 秒以上続いて末尾に届く）。
    Irodori に与えた長さが足りず語尾の途中で終わったテイク（2026-11-19 Yomi で発話末の「プツッ」・相槌の切れ）。
    末尾 20ms のフェードの後でも判定が変わらないよう、最後の 30ms は見ない。末尾のクリック（無音の後の短い音）は
    有音が 0.15 秒続かないので当たらない。"""
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 20:
        return False
    db = 20 * np.log10(np.sqrt((w[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    pk = db.max()
    if db[-5:-3].max() <= pk - 25:
        return False
    on = db[:-3] > pk - 35
    run = 0
    for v in on[::-1]:
        if not v:
            break
        run += 1
    return run >= 15


def median_f0(wav: np.ndarray, sr: int) -> float | None:
    """声の高さの中央値（Hz。pyin 60〜600Hz を 16kHz で。0.6 秒で約 0.1 秒）。有声が 3 フレーム未満なら None。
    相槌の「裏声」（男性の声の「あー。」が約 400Hz）を見つけるのに使う（2026-11-19・bc_prev_ref.pitch）。"""
    import librosa
    w = np.asarray(wav if np.ndim(wav) == 1 else np.mean(wav, axis=1), dtype=np.float32)
    if sr != 16000:
        w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    if len(w) < 1024:
        return None
    f, v, _ = librosa.pyin(w, fmin=60, fmax=600, sr=16000, frame_length=1024, hop_length=160)
    f = f[v] if v is not None else np.array([])
    return float(np.median(f)) if len(f) >= 3 else None


def voiced_at_start(wav: np.ndarray, sr: int) -> bool:
    """声の途中から始まっている（頭 20ms がすでにピーク−25dB 超）。Irodori の相槌は与えた長さを埋めるとき頭の無音を置かず、
    1 サンプル目から声のテイクが 4 割ある（2026-11-19 _moss_tune/bc_edge_check.py。長さ 0.8 秒の枠を使い切ったものは 6 割）。
    対話に置くと相槌の頭で「プツッ」と鳴る。普通の発話は 3% だけ。"""
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 3:
        return False
    db = 20 * np.log10(np.sqrt((w[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    return bool(db[:2].max() > db.max() - 25)


def edge_blobs(wav: np.ndarray, sr: int, min_gap: float = 0.1, max_blob: float = 0.8,
               rel_db: float = -35.0) -> list[tuple[int, int]]:
    """声の前後の余計な音: いちばん長い声の塊より前／後ろにあって、主の側の隣の塊から min_gap 秒以上離れた、max_blob 秒以下の
    塊の (始め, 終わり) サンプルの並び（tail_blob の前後両側版。tail_blob は後ろだけ・0.15 秒以上離れたものだけを見る）。
    2026-11-19 相槌の後ろ 10〜80ms のクリック・前の短い音（_moss_tune/bc_edge_check.py）。"""
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 10:
        return []
    db = 20 * np.log10(np.sqrt((w[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    idx = np.nonzero(db > max(db.max() + rel_db, -55.0))[0]
    if len(idx) == 0:
        return []
    runs = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i - runs[-1][1] < 5:                  # 50ms 未満の切れ目はつなぐ
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    if len(runs) < 2:
        return []
    m = max(range(len(runs)), key=lambda j: runs[j][1] - runs[j][0])
    g, mb = int(round(min_gap * 100)), int(round(max_blob * 100))
    out = []
    for j in range(m - 1, -1, -1):              # 前側: 外へ向かって、離れた短い塊を拾う（離れていない塊が来たら止める）
        a, b = runs[j]
        if runs[j + 1][0] - b >= g and b - a <= mb:
            out.append((a * h, b * h))
        else:
            break
    for j in range(m + 1, len(runs)):
        a, b = runs[j]
        if a - runs[j - 1][1] >= g and b - a <= mb:
            out.append((a * h, min(len(w), b * h)))
        else:
            break
    return out


def outer_clicks(wav: np.ndarray, sr: int, max_click: float = 0.06, min_gap: float = 0.06,
                 rel_db: float = -35.0, max_db: float = -20.0) -> list[tuple[int, int]]:
    """発話のいちばん外側（最初の声の前・最後の声の後ろ）にある、max_click 秒以下で隣の塊から min_gap 秒以上離れた短い音の
    (始め, 終わり) サンプルの並び。外から内へ向かって拾い、条件に合わない塊が来たら止める（edge_blobs はいちばん長い塊から外へ
    向かうので、句の間に間がある本文の発話では端まで届かない）。
    2026-11-19 ユーザ「発話の終わりのプツッ」: 本番 003 の本文 1380 発話のうち、声の 50〜150ms 後ろに 10〜30ms の広帯域の
    クリック（唇の音のような「プチッ」）が付くものがある。mute_end_click（tail_blob）は 0.15 秒以上離れたものしか見ない。
    1 モーラは 80ms 以上なので、60ms 以下の離れた塊はたいてい声ではない。ただし頭の無声化した「き」（北の方: 40ms・50ms 離れる・
    −24dB）や強い「い」（生け花: −8.5dB）は声なので、ピークが全体のピーク −20dB（max_db）より大きい塊と、離れが 60ms 未満の塊は残す
    （_moss_tune/outer_click_test.py: 本番 003 の 1679 発話で ASR の読みが変わったのは、この条件を足す前 4 件）。"""
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 10:
        return []
    db = 20 * np.log10(np.sqrt((w[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    idx = np.nonzero(db > max(db.max() + rel_db, -55.0))[0]
    if len(idx) == 0:
        return []
    runs = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i - runs[-1][1] < 5:                  # 50ms 未満の切れ目はつなぐ
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    if len(runs) < 2:
        return []
    g, mc = int(round(min_gap * 100)), int(round(max_click * 100))
    top = db.max() + max_db
    quiet = lambda r: db[r[0]:r[1]].max() <= top  # noqa: E731
    out, lo, hi = [], 0, len(runs) - 1
    while lo < hi and runs[lo][1] - runs[lo][0] <= mc and runs[lo + 1][0] - runs[lo][1] >= g and quiet(runs[lo]):
        out.append((runs[lo][0] * h, runs[lo][1] * h))
        lo += 1
    while hi > lo and runs[hi][1] - runs[hi][0] <= mc and runs[hi][0] - runs[hi - 1][1] >= g and quiet(runs[hi]):
        out.append((runs[hi][0] * h, min(len(w), runs[hi][1] * h)))
        hi -= 1
    return out


def trim_edges(wav: np.ndarray, sr: int, lead_keep: float = 0.05, tail_keep: float = 0.15,
               end_fade: float = 0.03, start_fade: float = 0.0) -> np.ndarray:
    """前後の無音を整える（ext_tts_server._trim_edges / Pocket._edges と同じ: 声＝10ms 窓で最大 −40dB 以上の
    始まりの lead_keep 秒前〜終わりの tail_keep 秒後を残し、切った端だけ短いフェード）。語尾は削らない。
    2026-11-19 Irodori 用に追加: 声が末尾まで続いている（切れた）ときは end_fade 秒のフェードアウトで「プツッ」を消す。"""
    w = np.asarray(wav, dtype=np.float32)
    h = int(0.01 * sr)
    n = len(w) // h
    if n < 5:
        return w
    mono = w if w.ndim == 1 else w.mean(axis=1)
    db = 20 * np.log10(np.sqrt((mono[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    act = np.nonzero(db > db.max() - 40)[0]
    if len(act) == 0:
        return w
    a = max(0, act[0] * h - int(lead_keep * sr))
    b = min(len(w), (act[-1] + 1) * h + int(tail_keep * sr))
    y = w[a:b].copy()
    # 頭: 切ったときは 10ms のフェードイン。切っていない（1 サンプル目から声）ときは start_fade 秒（2026-11-19・相槌の頭の「プツッ」）
    f = min(int((0.01 if a > 0 else start_fade) * sr), len(y))
    if f > 0:
        y[:f] = (y[:f].T * np.linspace(0.0, 1.0, f, dtype=np.float32)).T
    f = min(int((0.02 if b < len(w) else end_fade) * sr), len(y))
    if f > 0:
        y[-f:] = (y[-f:].T * np.linspace(1.0, 0.0, f, dtype=np.float32)).T
    return y


def hiss_run(wav: np.ndarray, sr: int, sil_db: float = -63.0,
             speech_db: float = -42.0, win_ms: float = 30.0,
             hop_ms: float = 10.0) -> float:
    """発話に挟まれず持続する「サーっ（広帯域ヒス）」の最長連続長を秒で返す。

    fish-speech は先頭フィラー（「うーん、」等）の直後に長い“間”を作り、そこを
    可聴のヒスで埋めることがある。ヒスは「クリーンな無音(<sil_db)」と「発話(>speech_db)」
    の中間レベル（sil_db〜speech_db）で持続するのが特徴。音素間の一瞬の遷移は短いので
    しきい値（呼び出し側で 0.6s 等）で弾ける。クリーンな無音の“間”はこの帯に入らない
    ため誤検知しない（自然な間合いは温存）。
    """
    if wav is None or len(wav) == 0:
        return 0.0
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    w = np.asarray(w, dtype=np.float32)
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if win <= 0 or hop <= 0:
        return 0.0
    nfr = 1 + max(0, (len(w) - win) // hop)
    best = cur = 0
    for k in range(nfr):
        seg = w[k * hop:k * hop + win]
        rms = float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0
        db = 20.0 * np.log10(rms + 1e-9)
        if sil_db < db < speech_db:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * hop_ms / 1000.0


def tail_burst(wav: np.ndarray, sr: int, text: str = "",
               gap: float = 0.20, burst: float = 0.50, min_burst: float = 0.12,
               max_drop: float = 14.0, max_rate: float = 13.0,
               rel_db: float = 25.0, abs_db: float = -55.0,
               merge_sec: float = 0.06) -> dict | None:
    """末尾に付いた「入力していないもう1音」を検出する。無ければ None。

    fish-speech は、入力テキストが日本語として言い終わっていない形（「〜があって。」
    のテ形、「〜なくてさ」「〜だな」の終助詞など）のとき、EOS を出す前にもう1音
    （多くは次のフィラーの出かかり）を出すことがある。実測（1440合成のA/B）:
      - 発生率はテキストで決まる（過分散 χ²=239, df=59, p=1e-22）。同じ文なら声・
        モードを変えても出やすい。声の影響も有意だが弱い（1.1〜6.1%）
      - temperature では変わらない（0.5で5.0% / 0.8で3.3% / 1.0で3.3%）ので、
        直すなら温度ではなく seed を替えて引き直す（62%が1回で消える）
    本番実測は 990ターン中 8件 = 0.81%（息・リップノイズ込みの孤立バーストは 2.4%）。

    判定は3段（返り値の stage）:
      "burst" … 最後の有音が gap 秒以上の無音で切れ、長さ burst 秒以下（＝孤立）
      "loud"  … さらに min_burst 秒以上・本体との差が max_drop dB 以内
                （0.05秒/-15dB のような息・リップノイズをここで落とす）
      "ok"    … さらに「本体（バーストを除いた長さ）だけでテキストが収まる話速」
                であること。max_rate モーラ/秒より速ければ、それは余分な1音では
                なく「長い間のあとに続いた発話の一部」（例:「どういたしまして、
                ナツさん。」の 0.46秒の間）なので除外する。未検出ターンの実測は
                中央値 8.7 / p99 11.3 / 最大 13.1 モーラ/秒
    引き直しの対象にするのは stage=="ok" のものだけ。波形を切る対処は聴感が悪く
    採らないので、この判定を外しても壊れるのは「無駄に1回引き直す」だけ。
    """
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    w = np.asarray(w, dtype=np.float32)
    win = int(sr * 20.0 / 1000)
    hop = int(sr * 10.0 / 1000)
    if win <= 0 or hop <= 0 or len(w) < win:
        return None
    nfr = 1 + (len(w) - win) // hop
    db = np.empty(nfr, dtype=np.float32)
    for k in range(nfr):
        seg = w[k * hop:k * hop + win]
        db[k] = 20.0 * np.log10(float(np.sqrt(np.mean(seg * seg))) + 1e-9)
    dt = hop / sr
    thr = max(float(db.max()) - rel_db, abs_db)
    on = db > thr
    runs: list[list[float]] = []                  # 有音区間 [開始秒, 終了秒]
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            runs.append([i * dt, (j + 1) * dt])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []                # 音素間の一瞬の落ち込みは繋ぐ
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_sec:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) < 2:
        return None
    g = merged[-1][0] - merged[-2][1]
    d = merged[-1][1] - merged[-1][0]
    if g < gap or d > burst:
        return None
    a, b = int(merged[-1][0] * sr), int(merged[-1][1] * sr)
    body_sec = merged[-1][0] - g
    body = w[:max(1, int(body_sec * sr))]

    def _db(x):
        return 20.0 * np.log10(float(np.sqrt(np.mean(x * x))) + 1e-9) if len(x) else -99.0

    lvl = _db(w[a:b]) - _db(body)
    mora, _conf = estimate_mora(text) if text else (0.0, 0.0)
    rate = mora / body_sec if body_sec > 0 else float("inf")
    stage = "burst"
    if d >= min_burst and lvl >= -max_drop:
        stage = "loud" if (text and rate > max_rate) else "ok"
    return {"gap": round(g, 3), "burst": round(d, 3), "lvl_diff": round(lvl, 1),
            "tail_start": round(merged[-1][0], 3), "speech_end": round(merged[-1][1], 3),
            "body_sec": round(body_sec, 3), "mora": round(mora, 1),
            "body_rate": round(rate, 2), "stage": stage}


@contextlib.contextmanager
def temperature_override(synth, temperature: float | None):
    """合成器の temperature を一時的に差し替える。None なら何もしない。"""
    gk = getattr(synth, "gen_kwargs", None)
    if temperature is None or not isinstance(gk, dict):
        yield
        return
    saved = dict(gk)
    pinned = getattr(synth, "temp_pinned", False)
    gk["temperature"] = float(temperature)
    synth.temp_pinned = True             # 主話者の main_gen の温度より優先させる
    try:
        yield
    finally:
        synth.gen_kwargs.clear()
        synth.gen_kwargs.update(saved)
        synth.temp_pinned = pinned


@contextlib.contextmanager
def seed_override(synth, seed: int | None):
    """合成器の seed を一時的に差し替える。None なら何もしない（＝毎回ランダム）。

    サーバの ServeTTSRequest.seed に渡る。/v1/tts と /v1/tts_multiturn の両方で
    「同じ seed なら出力バイト一致」を確認済み。温度を動かさずにテイクだけ替えたい
    とき（末尾の余分な1音の引き直し）に使う。
    """
    gk = getattr(synth, "gen_kwargs", None)
    if seed is None or not isinstance(gk, dict):
        yield
        return
    saved = dict(gk)
    gk["seed"] = int(seed)
    try:
        yield
    finally:
        synth.gen_kwargs.clear()
        synth.gen_kwargs.update(saved)


def take_temperatures(cfg: dict, n_takes: int) -> list[float | None]:
    """テイクごとの temperature。先頭は必ず None（＝config の既定値）。

    `temperatures: [0.5, 0.7, ...]` を書くと、2本目以降がその順で巡回する。
    低温は話速が遅く安定し、高温は抑揚が増える代わりに揺れる（実測: t=0.1 で
    話速 -11%）。**候補を意図的に散らしてから中庸を採る**ための設定で、
    全テイクを同じ温度で引くより当たりの幅が広がる。
    書かなければ従来どおり `temperature`（スカラ、既定 None）を全テイクに使う。
    """
    sched = cfg.get("temperatures")
    if not sched:
        t = cfg.get("temperature")
        return [None] + [t] * max(0, n_takes - 1)
    sched = [float(x) for x in sched]
    return [None] + [sched[k % len(sched)] for k in range(max(0, n_takes - 1))]


def _note_take_temp(synth, temp: float | None) -> None:
    """採用したテイクの temperature を数える（温度を振ったときの当たり率を見るため）。"""
    hist = getattr(synth, "take_temp_hist", None)
    if hist is not None:
        hist[temp] = hist.get(temp, 0) + 1


def _fmt_temp_hist(hist: dict, default_temp=None) -> str:
    """採用温度の内訳を「0.6×7, 既定×3」の形に。振っていなければ空文字。"""
    if not hist or set(hist) == {None}:
        return ""
    def lab(t):
        return "既定" if t is None else f"{t:.1f}"
    items = sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0] or 0.0))
    return "、採用温度 " + ", ".join(f"{lab(t)}×{n}" for t, n in items)


def _takes_temp_note(cfg: dict | None, n_takes: int) -> str:
    """ログ用: テイクに使う temperature の一覧。単一温度なら空文字。"""
    ts = [t for t in take_temperatures(cfg or {}, n_takes) if t is not None]
    return f"(t={'/'.join(f'{t:.1f}' for t in sorted(set(ts)))})" if ts else ""


def _synth_one_line(synth, text: str, ref: dict, temperature: float | None,
                    seed: int | None = None):
    """1ターンを単体合成する（temperature / seed を一時上書き可）。tts_multiturn 経路。"""
    with temperature_override(synth, temperature), seed_override(synth, seed):
        wavs = synth.synth_dialogue(f"<|speaker:0|>{normalize_tts_text(text)}", [ref])
    return wavs[0] if wavs else None


def regen_one_turn(synth, text: str, ref: dict, sr: int,
                   hiss_sec: float = 0.6, max_retries: int = 4,
                   temp_schedule: list | None = None):
    """サーが出たターンを、temperature を段階的に上げて再合成し直す。

    hiss_run が hiss_sec 以下になった最初のテイクを採用。全リトライで基準未達なら
    hiss_run が最小のテイクを採用（best-effort）。採用wavと診断dictを返す。
    """
    schedule = list(temp_schedule or [0.4, 0.6, 0.8, 1.0])
    best_w = None
    best_h = float("inf")
    tries = []
    for attempt in range(max(1, max_retries)):
        temp = schedule[min(attempt, len(schedule) - 1)] if schedule else None
        w = _synth_one_line(synth, text, ref, temp)
        if w is None or len(w) == 0:
            tries.append((temp, None))
            continue
        h = hiss_run(w, sr)
        tries.append((temp, round(h, 2)))
        if h < best_h:
            best_h, best_w = h, w
        if h <= hiss_sec:
            break
    return best_w, {"hiss": round(best_h, 2), "tries": tries}


def _tail_thresholds(cfg: dict) -> dict:
    """config から tail_burst のしきい値だけ取り出す（既定は tail_burst の既定に従う）。"""
    keys = ("gap", "burst", "min_burst", "max_drop", "max_rate")
    return {k: float(cfg[k]) for k in keys if cfg.get(k) is not None}


def regen_one_turn_tail(synth, text: str, ref: dict, sr: int, cfg: dict,
                        plain_text: str = "", hiss_sec: float | None = None,
                        base_burst: float = float("inf")):
    """末尾に余分な1音が出たターンを、seed だけ替えて引き直す。

    temperature は動かさない（A/Bで発生率が変わらないことを確認済み。触ると
    そのターンだけ話速・抑揚が変わってしまう）。余分な1音が無い最初のテイクを
    採用し、全リトライで駄目ならバーストが最短のテイクを採る（best-effort）。
    base_burst に元テイクのバースト長を渡すと、それより短くならないテイクは
    採らない（引き直して悪化させない）。全て駄目なら None を返す＝元のまま。
    hiss_sec を渡すと「サーが乗ったテイク」は採用しない（サー再合成の結果を
    引き直しで元に戻してしまわないため）。採用wavと診断dictを返す。
    """
    thr = _tail_thresholds(cfg)
    base = int(cfg.get("seed_base", 20260730))
    max_retries = max(1, int(cfg.get("max_retries", 2)))
    best_w, best_b = None, float(base_burst)
    tries = []
    for attempt in range(max_retries):
        seed = base + attempt
        w = _synth_one_line(synth, text, ref, None, seed=seed)
        if w is None or len(w) == 0:
            tries.append((seed, None))
            continue
        tb = tail_burst(w, sr, plain_text or text, **thr)
        bad_hiss = hiss_sec is not None and hiss_run(w, sr) > hiss_sec
        b = tb["burst"] if (tb and tb["stage"] == "ok") else 0.0
        tries.append((seed, b if not bad_hiss else f"{b}(サー)"))
        if not bad_hiss and b < best_b:
            best_b, best_w = b, w
        if b == 0.0 and not bad_hiss:
            break
    return best_w, {"burst": round(best_b, 2) if best_b < float("inf") else None,
                    "kept": best_w is None, "tries": tries}


def regenerate_tail_turns(synth, record: dict, voices: list[dict],
                          wav_by_turn: dict, sr: int, cfg: dict,
                          emotion_tags: bool = False,
                          fish_tags_map: dict | None = None,
                          prefix_tag: str | None = None,
                          hiss_sec: float | None = None) -> int:
    """合成済みターン群を走査し、末尾に余分な1音が付いたターンだけ引き直す。

    fish のクローン参照合成（_fish_multiturn_inputs が成立する構成）でのみ動作。
    非対応構成では何もせず 0 を返す。差し替えた件数を返す（wav_by_turn を破壊的更新）。
    波形は切らない（トリムは聴感が悪いので不採用）ので、引き直しても残る分は残る。
    """
    if not cfg or not cfg.get("enabled", True):
        return 0
    thr = _tail_thresholds(cfg)
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return 0
    turns, _speaker_order, references, spk_index, tts_texts, _lines = inp
    fixed = 0
    for i in sorted(wav_by_turn):
        if i >= len(turns):                      # 念のため（本文が無いと話速ガードが効かない）
            continue
        # 話速ガードは本文（タグ抜き）で測る。タグ文字はモーラ数を膨らませるため。
        plain = turns[i].get("text") or ""
        tb = tail_burst(wav_by_turn[i], sr, plain, **thr)
        if not tb or tb["stage"] != "ok":
            continue
        sp = turns[i]["speaker"]
        ref = references[spk_index[sp]]
        new_w, diag = regen_one_turn_tail(synth, tts_texts[i], ref, sr, cfg,
                                          plain_text=plain, hiss_sec=hiss_sec,
                                          base_burst=tb["burst"])
        head = (f"[{record['id']}] 末尾に余分な1音 turn{i}「{tts_texts[i][-16:]}」 "
                f"検知={tb['burst']}s(間{tb['gap']}s)")
        if new_w is None:                        # どのテイクも改善しなかった＝元のまま
            logger.info(f"{head} → 引き直し不成立、元のテイクを維持 "
                        f"tries={diag['tries']}")
            continue
        wav_by_turn[i] = new_w
        fixed += 1
        logger.info(f"{head} → 引き直し 最終={diag['burst']}s tries={diag['tries']}")
    if fixed:
        logger.info(f"[{record['id']}] 末尾の引き直し: {fixed}ターン差し替え")
    return fixed


def regenerate_speaker_leak_turns(synth, record: dict, voices: list[dict],
                                  wav_by_turn: dict, sr: int, cfg: dict,
                                  emotion_tags: bool = False,
                                  fish_tags_map: dict | None = None,
                                  prefix_tag: str | None = None) -> int:
    """声が相手に入れ替わったターンを x-vector で見つけ、本人の参照だけで seed を替えて引き直す。

    2話者まとめての生成（multiturn）で、ある発話の声が相手になることがある（2026-10-05 cmp_mt:
    20対話 667発話中2件）。判定は speaker_check.is_leak（本人との類似度 < own_max かつ
    相手のほうが margin 以上近い）。cfg: enabled / own_max（既定 0.85）/ margin（既定 0.10）/
    max_retries（既定 3）/ min_sec（これ未満の短い発話は判定が不安定なので見ない。既定 0.8）/
    model / device。どの引き直しも基準を満たさなければ、本人に最も近いテイクを採る。
    差し替えた件数を返す（wav_by_turn を破壊的更新）。依存が無ければ何もしない。
    """
    if not cfg or not cfg.get("enabled", True):
        return 0
    import speaker_check as SC
    ver = SC.get_verifier(str(cfg.get("model", "microsoft/wavlm-base-plus-sv")),
                          str(cfg.get("device", "cuda")))
    if ver is None:
        return 0
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return 0
    turns, speaker_order, references, spk_index, tts_texts, _lines = inp
    if len(references) < 2:
        return 0
    own_max = float(cfg.get("own_max", 0.85))
    margin = float(cfg.get("margin", 0.10))
    retries = int(cfg.get("max_retries", 3))
    min_n = int(float(cfg.get("min_sec", 0.8)) * sr)
    ref_emb = [ver.embed_bytes(r["audio"]) for r in references]

    def sims(w):
        e = ver.embed(w, sr)
        return [float(e @ r) for r in ref_emb]

    fixed = 0
    for i in sorted(wav_by_turn):
        w = wav_by_turn[i]
        if (i >= len(turns) or w is None or len(w) < min_n
                or turns[i].get("type", "speech") == "backchannel"):
            continue
        k = spk_index[turns[i]["speaker"]]
        s = sims(w)
        own, other = s[k], max(v for j, v in enumerate(s) if j != k)
        if not SC.is_leak(own, other, own_max, margin):
            continue
        best_w, best = None, (own, other)
        tries = [(round(own, 3), round(other, 3))]
        for n in range(retries):
            seed = int(hashlib.md5(f"{record['id']}:{i}:spk{n}".encode()).hexdigest()[:8], 16)
            w2 = _synth_one_line(synth, tts_texts[i], references[k], None, seed=seed)
            if w2 is None or len(w2) == 0:
                continue
            s2 = sims(w2)
            o2, x2 = s2[k], max(v for j, v in enumerate(s2) if j != k)
            tries.append((round(o2, 3), round(x2, 3)))
            if o2 - x2 > best[0] - best[1]:
                best_w, best = w2, (o2, x2)
            if not SC.is_leak(o2, x2, own_max, margin):
                break
        head = (f"[{record['id']}] 話者の取り違え turn{i}「{turns[i].get('text', '')[:20]}」"
                f" 本人{own:.2f}/相手{other:.2f}")
        if best_w is None:
            logger.warning(f"{head} → 引き直しで改善せず、元のテイクを維持 tries={tries}")
            continue
        wav_by_turn[i] = best_w
        fixed += 1
        logger.info(f"{head} → 差し替え 本人{best[0]:.2f}/相手{best[1]:.2f} tries={tries}")
    if fixed:
        logger.info(f"[{record['id']}] 話者の取り違えの引き直し: {fixed}ターン差し替え")
    return fixed


def regenerate_misread_turns(synth, record: dict, voices: list[dict],
                             wav_by_turn: dict, sr: int, cfg: dict,
                             emotion_tags: bool = False,
                             fish_tags_map: dict | None = None,
                             prefix_tag: str | None = None,
                             only: set | None = None) -> int:
    """合成済みターンを ASR で聞き取り、読みが台本と食い違うターンだけ seed を替えて引き直す。

    fish はたまに語を読み違える（評価と ASR 突き合わせの実例: 表書き→ひょうがき、内のし→
    からだのし、承ります→うけたもあります、お電話→ごでんわ、番号の桁抜け）。読み辞書
    （tts_reading）で既知の語は防げるが、未知の語は合成してみないと分からないので、合成後に
    聞き取って確かめる。採点は reading_check.score の sub（読みの置き換えの割合）。
    cfg: enabled / threshold（既定 0.12）/ min_chars（期待の読みがこれ未満の短い発話は見ない。
    既定 6）/ max_retries（既定 3）/ model / device。改善したテイクだけ差し替える（wav_by_turn を
    破壊的更新）。差し替えた件数を返す。依存（ASR・pyopenjtalk）が無ければ何もしない。
    only を渡すとそのターンだけ見る（refine の Pass B で合成し直したターン）。

    発話全体の読み（turn["ruby"]。yomi_ruby。2026-11-12）があれば照合はそれで行う（V3）。
    cfg.ruby（{enabled, hosts か base_urls, model, extra_body, workers, open_on_retry}）が
    有効なら、読みの無い・本文が変わった発話（refine の書き換え等）にはここで付け、ASR の文にも
    付けて比べる。引き直しでは、聞こえた読みに見当たらない語のうち確かな読み（C=B）のある語を
    ひらがなに開いて合成する（茸→たけ・獺祭 のように seed を替えても直らない読み違い）。
    """
    if not cfg or not cfg.get("enabled", True):
        return 0
    import reading_check as RC
    asr = RC.get_asr(str(cfg.get("model", "kotoba-tech/kotoba-whisper-v2.0")),
                     str(cfg.get("device", "cuda")))
    if asr is None:
        return 0
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return 0
    turns, _speaker_order, references, spk_index, tts_texts, _lines = inp
    thr = float(cfg.get("threshold", 0.12))
    min_chars = int(cfg.get("min_chars", 6))
    retries = int(cfg.get("max_retries", 3))
    # 英単語（3文字以上の英字の並び）を含む発話は見ない。期待の読みは英字を1字ずつ読む
    # （Python→ピーワイティー…）のに fish も ASR も語として読む（パイソン）ので、正しく読んでも
    # 置き換えに数えて無駄に引き直す（2026-09-26 初対面の検査で sub≥0.12 の約半数がこれ）。
    idx = [i for i in sorted(wav_by_turn)
           if i < len(turns) and wav_by_turn[i] is not None and len(wav_by_turn[i])
           and (only is None or i in only)
           and len(RC.kana(tts_texts[i])) >= min_chars
           and not re.search(r"[A-Za-zＡ-Ｚａ-ｚ]{3,}", tts_texts[i])]
    if not idx:
        return 0
    rubies, asr_ruby, open_retry = _misread_rubies(record, turns, idx, cfg)
    scores = RC.score(asr, [wav_by_turn[i] for i in idx], sr, [tts_texts[i] for i in idx],
                      hints=[turns[i].get("yomi") for i in idx],
                      rubies=[rubies.get(i) for i in idx],
                      asr_ruby=(lambda k, h: asr_ruby(idx[k], h)) if asr_ruby else None, thr=thr)
    fixed = 0
    for i, sc in zip(idx, scores):
        if sc["sub"] < thr:
            continue
        ref = references[spk_index[turns[i]["speaker"]]]
        best_w, best = None, sc
        tries = [round(sc["sub"], 3)]
        # 引き直しの入力：確かな読みのある疑わしい語をひらがなに開く（無ければ seed だけ替える）
        text_i, opened = tts_texts[i], ""
        rec = rubies.get(i)
        if open_retry and rec and rec.get("b"):
            import yomi_ruby as YR
            t2 = YR.open_words(tts_texts[i], rec, YR.suspect_words(rec, sc["got"], RC.kana))
            if t2 != tts_texts[i]:
                text_i, opened = t2, f" 開いた入力「{t2[:30]}」"
        for k in range(retries):
            seed = int(hashlib.md5(f"{record['id']}:{i}:{k}".encode()).hexdigest()[:8], 16)
            w = _synth_one_line(synth, text_i, ref, None, seed=seed)
            if w is None or len(w) == 0:
                continue
            s2 = RC.score(asr, [w], sr, [tts_texts[i]], hints=[turns[i].get("yomi")],
                          rubies=[rec], asr_ruby=(lambda _k, h: asr_ruby(i, h)) if asr_ruby else None,
                          thr=thr)[0]
            tries.append(round(s2["sub"], 3))
            if s2["sub"] < best["sub"]:
                best_w, best = w, s2
            if s2["sub"] < thr:
                break
        dw = (f"{sc['exp'][:20]}→{sc['got'][:20]}" if sc.get("v") == "ruby"
              else '; '.join(RC.diff_words(tts_texts[i], sc['got'])))
        head = (f"[{record['id']}] 読み違い turn{i}「{turns[i].get('text', '')[:24]}」 "
                f"sub={sc['sub']:.2f} {dw[:80]}{opened}")
        if best_w is None:
            logger.info(f"{head} → 引き直しで改善せず、元のテイクを維持 tries={tries}")
            continue
        wav_by_turn[i] = best_w
        fixed += 1
        logger.info(f"{head} → 差し替え sub={best['sub']:.2f} tries={tries}")
    if fixed:
        logger.info(f"[{record['id']}] 読み違いの引き直し: {fixed}ターン差し替え")
    return fixed


def _misread_rubies(record: dict, turns: list, idx: list, cfg: dict):
    """読み違いの検査に使う発話全体の読み。(ターン番号→ruby, ASR の文に読みを付ける関数 or None,
    引き直しで語を開くか)。本文に合う turn["ruby"] があれば使い、無ければ cfg.ruby の LLM で付けて
    turn に書き戻す（精製済みレコードに残り、アラインメントでも使える）。"""
    import yomi_ruby as YR
    rcfg = cfg.get("ruby") or {}
    if isinstance(rcfg, bool):
        rcfg = {"enabled": rcfg}
    on = bool(rcfg.get("enabled", True)) if rcfg else False
    chat = YR.sync_chat_from_cfg(rcfg) if on else None
    lines = [(str(t.get("speaker", "")), str(t.get("text", ""))) for t in turns]
    if chat is not None:
        need = [i for i in idx if not turns[i].get("unsaid") and YR.needs_ruby(lines[i][1])
                and not YR.valid(turns[i].get("ruby"), lines[i][1])]
        if need:
            got = YR.annotate_turns_sync(chat, lines, need, int(rcfg.get("workers", 8)))
            for i, r in got.items():
                turns[i]["ruby"] = r
            if len(got) < len(need):
                logger.info(f"[{record['id']}] 発話の読み付け: {len(got)}/{len(need)}発話"
                            "（取れなかった発話は従来の照合）")
    rubies = {}
    for i in idx:
        t = lines[i][1]
        if turns[i].get("unsaid"):          # 言いかけ（rp）は合成入力に unsaid が足されるので使わない
            continue
        if YR.valid(turns[i].get("ruby"), t):
            rubies[i] = turns[i]["ruby"]
        elif not YR.needs_ruby(t) and (chat is not None or any(
                YR.valid(turns[j].get("ruby"), lines[j][1]) for j in idx)):
            rubies[i] = {"src": t, "c": t}  # かなだけの発話（読みは本文そのもの）
    asr_ruby = None
    if chat is not None:
        def asr_ruby(i, hyp):
            r = YR.annotate_one_sync(chat, lines, i, text=hyp, with_b=False)
            return r["c"] if r else None
    return rubies, asr_ruby, bool(rcfg.get("open_on_retry", True))


def take_features(wav: np.ndarray, sr: int, mora: float,
                  gender: str | None) -> dict | None:
    """1テイクの特徴（話速・抑揚・音量・ヒス）。測れなければ None。"""
    if wav is None or len(wav) < int(0.2 * sr):
        return None
    w = trim_silence(wav, sr)
    dur = len(w) / sr
    if dur <= 0.05:
        return None
    return {"rate": mora / dur, "f0sd": f0_sd_semitones(w, sr, gender),
            "level": speech_level_db(w, sr), "hiss": hiss_run(w, sr), "dur": dur}


def score_takes(feats: list[dict | None], cfg: dict) -> list[float]:
    """テイクごとの「中庸からの外れ具合」。小さいほど採用に近い。

    基準は既定でテイク自身の中央値（＝外れテイクだけを落とす）。話速だけは
    `rate_target` に数値を書くと固定目標にできる（速い側へ寄った群ごと引き戻せる）。
    """
    ok = [f for f in feats if f]
    if not ok:
        return [float("inf")] * len(feats)

    def med(key):
        v = [f[key] for f in ok if f.get(key) is not None]
        return float(np.median(v)) if v else None

    tgt = cfg.get("rate_target", "median")
    rate_t = med("rate") if (tgt is None or tgt == "median") else float(tgt)
    f0_t, lvl_t = med("f0sd"), med("level")
    w_rate = float(cfg.get("w_rate", 1.0))
    w_f0 = float(cfg.get("w_f0sd", 0.5))
    w_lvl = float(cfg.get("w_level", 0.5))
    s_rate = float(cfg.get("rate_scale", 1.0))       # モーラ/秒
    s_f0 = float(cfg.get("f0sd_scale", 1.0))         # 半音
    s_lvl = float(cfg.get("level_scale", 3.0))       # dB
    hiss_sec = float(cfg.get("hiss_sec", 0.6))
    hiss_pen = float(cfg.get("hiss_penalty", 2.0))

    out = []
    for f in feats:
        if not f:
            out.append(float("inf"))
            continue
        s = w_rate * abs(f["rate"] - rate_t) / s_rate
        if f0_t is not None and f.get("f0sd") is not None:
            s += w_f0 * abs(f["f0sd"] - f0_t) / s_f0
        if lvl_t is not None and f.get("level") is not None:
            s += w_lvl * abs(f["level"] - lvl_t) / s_lvl
        s += hiss_pen * max(0.0, f["hiss"] - hiss_sec)
        out.append(s)
    return out


_LAUGH_HEAD_RE = re.compile(r"^\s*(?:う|ん)?(?:ふ{2,}|あ?は{2,}|ア?ハ{2,})っ?[、。！？!?…\s]")


def final_rise_st(w: np.ndarray, sr: int, gender: str | None = None) -> float:
    """発話の語尾の音程の上がり幅（半音）: 最後の 1.0 秒の有声フレーム（pyin・16kHz・10ms）のうち、最後の 5 フレームの
    平均 ÷ その前の 20 フレームの中央値。有声が 10 フレーム未満なら 0。探索範囲は性別で切る（男 60〜260・女 110〜450Hz。
    広いとオクターブ誤りで「上がった」ことになる）。疑問文の語尾の上昇の検査に使う（best_of の w_q）。"""
    import librosa
    lo, hi = (60, 260) if gender == "male" else ((110, 450) if gender == "female" else (60, 450))
    x = np.asarray(w if w.ndim == 1 else w.mean(axis=1), dtype=np.float32)
    act = np.nonzero(np.abs(x) > 0.02 * (np.abs(x).max() + 1e-9))[0]
    if len(act):
        x = x[: act[-1] + 1]
    # 最後の 1.0 秒（1.5 秒と判定一致 98.4%・pyin が 1.6 倍速い。2026-11-14 _moss_tune/_rise_check.py）
    x = librosa.resample(x[-int(1.0 * sr):], orig_sr=sr, target_sr=16000) if sr != 16000 else x[-16000:]
    if len(x) < 1600:
        return 0.0
    f0, vf, _ = _pyin(x, fmin=lo, fmax=hi, sr=16000, frame_length=640, hop_length=160)
    f = f0[vf]
    if len(f) < 10:
        return 0.0
    ref = np.median(f[-25:-5]) if len(f) >= 25 else np.median(f[:-5])
    return float(12 * np.log2(np.mean(f[-5:]) / ref))


def _active_and_pauses(w: np.ndarray, sr: int) -> tuple[float, float]:
    """20ms 窓のエネルギーで、発話区間（前後の無音を除く）の長さと、その中の 100ms 以上の無音の割合。"""
    w = np.asarray(w if w.ndim == 1 else w.mean(axis=1), dtype=np.float32)
    fr = max(1, int(sr * 0.02))
    n = len(w) // fr
    if n < 3:
        return 0.0, 0.0
    e = np.sqrt((w[: n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(e)
    v = db > np.percentile(db, 95) - 35
    idx = np.nonzero(v)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    seg = v[idx[0]: idx[-1] + 1]
    pause, run = 0, 0
    for x in seg:
        if not x:
            run += 1
        else:
            if run >= 5:
                pause += run
            run = 0
    return len(seg) * 0.02, pause / max(len(seg), 1)


_NLL_PUNCT = set("、。！？!?,.，．…「」『』（）()・ 　")


def _asr_text_nll(asr, w: np.ndarray, sr: int, text: str) -> float:
    """台本 text を ASR（Whisper 系の pipeline）に教師強制で読ませたときの、句読点を除く 1 トークンあたりの NLL。
    同じ文なら、日本語として不自然な発音・抑揚（カタコト）ほど大きくなる（_moss_tune/nat.py と同じ式）。"""
    return _asr_text_nll_batch(asr, [w], sr, text)[0]


def _asr_text_nll_batch(asr, ws: list, sr: int, text: str, batch: int = 16) -> list[float]:
    """_asr_text_nll を同じ台本の複数テイクにまとめて（台本が同じなのでデコーダ入力は共通・詰め物なし）。"""
    import torch
    import reading_check as RC
    pipe = asr.pipe
    tok, fe, m = pipe.tokenizer, pipe.feature_extractor, pipe.model
    pad = np.zeros(int(0.3 * RC.ASR_SR), dtype=np.float32)
    xs = []
    for w in ws:
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if sr != RC.ASR_SR:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=RC.ASR_SR)
        xs.append(np.concatenate([pad, w, pad]))
    prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    body = tok.encode(normalize_tts_text(text), add_special_tokens=False) + [eot]
    ids = prefix + body
    keep = [i for i, t in enumerate(body) if t == eot or not set(tok.decode([t]).strip()) <= _NLL_PUNCT]
    tgt = torch.tensor(ids[1:], device=m.device)
    out: list[float] = []
    for b in range(0, len(xs), batch):
        chunk = xs[b:b + batch]
        fk = {"device": str(m.device)} if RC.FAST_FEATURES else {}     # fast_score: 特徴量も GPU で
        feats = fe(chunk, sampling_rate=RC.ASR_SR, return_tensors="pt", **fk).input_features.to(m.device, dtype=m.dtype)
        dec = torch.tensor([ids[:-1]] * len(chunk), device=m.device)
        with torch.no_grad():
            lg = m(input_features=feats, decoder_input_ids=dec).logits.float()
        lp = torch.log_softmax(lg, -1).gather(-1, tgt[None, :, None].expand(len(chunk), -1, 1))[..., 0]
        lp = lp[:, len(prefix) - 1:][:, keep]
        out += [float(-x) for x in lp.mean(1)]
    return out


def _select_quality_takes(synth, record: dict, wav_by_turn: dict, sr: int, cfg: dict,
                          inp, n_takes: int) -> int:
    """best_of.criterion: quality（または median_dur＝長さが中央のテイク・下の med_dur）。各ターンを n_takes 本引き、score の最も小さいテイクを採る。

    score = w_cer*CER（ASR で聞き取った読み）+ |ln(カナ/秒 ÷ 目標話速)|
            + w_pause*max(0, 発話の中の無音の割合 − pause_max) + w_sim*max(0, sim_min − 本人の参照との類似度)
            + w_hiss*max(0, サーの長さ − hiss_sec)（既定 5・0.6秒。MOSS-TTS-v1.5 既定で 60発話中 2本が 0.6秒超）
    目標話速（rate_kps: ref・既定）は話者ごとに参照音声（人間の録音）から決める: 参照の書き起こしのカナ数 ÷
    参照の発話時間（100ms 以上の無音を除く）× (1 − pause_max)＝「その人の素の話速で、発話の中に1割の間」。
    他の TTS の出力は一切使わない。数値を渡せば全話者共通の目標になる。
    既定は MOSS 調整（2026-11-13・_moss_tune/select_best.py）と同じ: w_cer 5・w_pause 2・pause_max 0.10・
    w_sim 5・sim_min 0.96。合成コストは takes 倍＋ASR・x-vector。
    w_nat > 0 で「カタコトさ」+ w_nat*NLL（_asr_text_nll。同じ ASR に台本を教師強制で読ませた 1 トークンあたりの
    負の対数尤度・句読点を除く）を足す。NLL は遅く間の多いテイクほど下がるので、w_kps（話速）・w_pause も上げて
    話速と間を保つ（_moss_tune で w_nat 10・w_kps 5・w_pause 10 が、話速を変えずに独立の判定器 whisper-large-v3
    でも自然さが上がった組。2026-11-13）。
    """
    import reading_check as RC
    import speaker_check as SC
    # criterion: median_dur（2026-11-15）: 採点せず、発話区間の長さ（前後の無音を除く）がテイクの中央に一番近いものを採る。
    #   pocket CC（温度 0.3）はどのテイクも十分聞き取れる（ユーザの聴取）ので、長さの外れ（速すぎ・間延び）だけを除く。
    #   60 発話×16 本の実測で CER・自然さ（whisper-large-v3）・似方・抑揚の幅は 1 本だけのときと同じ（選んでも良くも悪くもならない）。
    #   ASR・x-vector・pyin を使わないので速い。w_q（疑問文の語尾）・w_laugh（笑い）も効かなくなる。
    med_dur = str(cfg.get("criterion", "quality")) == "median_dur"
    if med_dur:
        asr = ver = None
    else:
        asr = RC.get_asr(str(cfg.get("asr_model", "kotoba-tech/kotoba-whisper-v2.0")), str(cfg.get("device", "cuda")))
        ver = SC.get_verifier(str(cfg.get("sv_model", "microsoft/wavlm-base-plus-sv")), str(cfg.get("device", "cuda")))
    if asr is None and not med_dur:
        logger.warning("best_of.criterion=quality: ASR を用意できないのでテイク選択をしない")
        return 0
    # fast_score: 採点を GPU でまとめて回す（2026-11-14）。ASR の特徴量を GPU で一括（RC.FAST_FEATURES。pipeline は 1 本ずつ
    #   CPU で作り採点の 4 割を占めていた・書き起こしは 156 本中 135 本同一で、違いは表記ゆれか pipeline が文頭を落としたもの）、
    #   x-vector を長さの近いテイクごとに束ねる（speaker_check.embed_batch・類似度のずれ平均 0.002）。
    fast = bool(cfg.get("fast_score", False))
    if fast:
        RC.FAST_FEATURES = True
    turns, _order, references, spk_index, tts_texts, _lines = inp
    w_cer = float(cfg.get("w_cer", 5.0))
    w_pause, pause_max = float(cfg.get("w_pause", 2.0)), float(cfg.get("pause_max", 0.10))
    rate_cfg = cfg.get("rate_kps", "ref")
    rate_of: dict = {}

    def _rate_target(k: int) -> float:
        """話者 k の目標話速（カナ/秒）。"""
        if rate_cfg != "ref":
            return float(rate_cfg)
        if k not in rate_of:
            import io as _io
            rw, rsr = sf.read(_io.BytesIO(references[k]["audio"]), dtype="float32", always_2d=True)
            act, pr = _active_and_pauses(rw, rsr)
            kana = len(RC.kana(references[k].get("text", "")))
            art = kana / (act * (1 - pr)) if act > 0 and kana > 0 else 7.9   # 参照の素の話速（間を除く）
            rate_of[k] = min(10.0, max(5.0, art)) * (1 - pause_max)
        return rate_of[k]
    w_sim, sim_min = float(cfg.get("w_sim", 5.0)), float(cfg.get("sim_min", 0.96))
    w_kps = float(cfg.get("w_kps", 1.0))       # 話速の重み（median 用の w_rate とは別）
    w_nat = float(cfg.get("w_nat", 0.0))      # カタコトさ（台本を ASR に教師強制で読ませた NLL。_asr_text_nll）
    min_chars = int(cfg.get("min_chars", 2))
    w_hiss, hiss_sec = float(cfg.get("w_hiss", 5.0)), float(cfg.get("hiss_sec", 0.6))
    w_tail = float(cfg.get("w_tail", 0.0))          # 台本外の末尾の音（tail_blob）の減点。採ったテイクに残れば無音にする
    w_cut = float(cfg.get("w_cut", 0.0))            # 声のまま切れた語尾（voiced_at_end）の減点（2026-11-19 Irodori の「プツッ」）
    w_head = float(cfg.get("w_head", 0.0))          # 声の途中から始まる（voiced_at_start）の減点（2026-11-19・頭の「プツッ」）
    tail_gap, tail_short = float(cfg.get("tail_gap", 0.15)), float(cfg.get("tail_short", 0.06))
    tail_of: dict = {}
    ref_emb: dict = {}
    # w_q > 0: 疑問文（「？」で終わる）は語尾の音程が q_min 半音以上上がらないテイクを減点（2026-11-14）。pocket CC は
    #   疑問文の語尾が上がる割合が 25%（S2 53%）で平叙文のように聞こえる。16 本の中には上がるテイクがあり、60 発話の
    #   疑問 8 文で w_q 1 → 上がる割合 62%・natV3（whisper-large-v3）は変わらず（_moss_tune 2026-11-14）。
    w_q, q_min = float(cfg.get("w_q", 0.0)), float(cfg.get("q_min", 1.0))
    # w_laugh > 0: 笑いで始まる発話（ふふ・あはは…）は、頭 1.5 秒の笑いらしさ（laugh_detect・AST の笑い確率）が高いテイクを
    #   優先する（score から w_laugh×確率を引く）。同じ文でもテイクで 2 倍違う（_moss_tune/laugh_sel_test.py・2026-11-14）。
    w_laugh = float(cfg.get("w_laugh", 0.0))
    gender_of = {sp.get("name"): sp.get("gender") for sp in ((record.get("metadata") or {}).get("speakers") or [])
                 if isinstance(sp, dict)}

    def _emb_ref(k: int):
        if k not in ref_emb and ver is not None:
            import io as _io
            rw, rsr = sf.read(_io.BytesIO(references[k]["audio"]), dtype="float32", always_2d=True)
            ref_emb[k] = ver.embed(rw.mean(1), rsr)
        return ref_emb.get(k)

    def _tails(ws, i, cers) -> list:
        """各テイクの台本外の末尾の音 (始め, 終わり) か None（tail_blob。短すぎる音は即、それ以外は
        その前で切っても CER が悪くならない＝台本の一部ではないとき）。"""
        tb = [tail_blob(w, sr, min_gap=tail_gap) if w_tail > 0 else None for w in ws]
        out = [None] * len(ws)
        chk = []
        for t, x in enumerate(tb):
            if x is None:
                continue
            if (x[1] - x[0]) / sr < tail_short:
                out[t] = x
            else:
                chk.append(t)
        if chk:
            cut = [ws[t][: tb[t][0] - int(0.03 * sr)] for t in chk]
            ct = [float(r["cer"]) for r in RC.score(asr, cut, sr, [tts_texts[i]] * len(chk),
                                                     hints=[turns[i].get("yomi")] * len(chk))]
            for t, c in zip(chk, ct):
                if c <= cers[t] + 1e-9:
                    out[t] = tb[t]
        return out

    def _scores(ws, i, k) -> list[float]:
        """ターン i のテイク群 ws（どれも空でない）の score。ASR と NLL はテイクをまとめて GPU に流す。"""
        n = len(ws)
        if med_dur:
            tail_of[i] = [None] * n
            acts = [_active_and_pauses(w, sr)[0] for w in ws]
            mid = float(np.median(acts))
            return [abs(a - mid) for a in acts]
        cers = [float(r["cer"]) for r in RC.score(asr, ws, sr, [tts_texts[i]] * n, hints=[turns[i].get("yomi")] * n)]
        tails = _tails(ws, i, cers)
        tail_of[i] = tails
        nlls = (_asr_text_nll_batch(asr, ws, sr, turns[i].get("text", "") or tts_texts[i]) if w_nat > 0
                else [0.0] * n)
        e = _emb_ref(k)
        sims = None
        if fast and e is not None:
            long_k = [t for t, w in enumerate(ws) if len(w) / sr >= 0.8]
            embs = ver.embed_batch([ws[t] for t in long_k], sr) if long_k else []
            sims = {t: float(v @ e) for t, v in zip(long_k, embs)}
        txt_i = (turns[i].get("text", "") or tts_texts[i]).rstrip()
        q_on = w_q > 0 and txt_i.endswith(("？", "?"))
        laughs = [0.0] * n
        if w_laugh > 0 and _LAUGH_HEAD_RE.match(txt_i):
            import laugh_detect
            ld = laugh_detect.get(str(cfg.get("device", "cuda")))
            laughs = [ld.head(w, sr) for w in ws]
        out = []
        for w, cer, nll in zip(ws, cers, nlls):
            act, pr = _active_and_pauses(w, sr)
            kps = len(RC.kana(turns[i].get("text", ""))) / act if act > 0 else 0.1
            s = (w_cer * cer + w_kps * abs(np.log(max(kps, 0.1) / _rate_target(k)))
                 + w_pause * max(0.0, pr - pause_max) + w_nat * nll)
            s += w_hiss * max(0.0, hiss_run(w, sr) - hiss_sec)      # 先頭フィラー直後のサー（regen_on_hiss と同じ検知）
            if tails[len(out)] is not None:
                s += w_tail                                          # 台本外の末尾の音（「ああ」・クリック等）
            if w_cut > 0 and voiced_at_end(w, sr):
                s += w_cut                                           # 長さが足りず語尾の途中で終わったテイク
            if w_head > 0 and voiced_at_start(w, sr):
                s += w_head                                          # 声の途中から始まるテイク
            if e is not None and len(w) / sr >= 0.8:
                sim = sims[len(out)] if sims is not None else float(ver.embed(w, sr) @ e)
                s += w_sim * max(0.0, sim_min - sim)
            if q_on:
                s += w_q * max(0.0, q_min - final_rise_st(w, sr, gender_of.get(turns[i].get("speaker"))))
            s -= w_laugh * laughs[len(out)]
            out.append(s)
        return out

    targets = [i for i in sorted(wav_by_turn)
               if turns[i].get("type") != "backchannel" and len(turns[i].get("text", "")) >= min_chars]
    sb = getattr(synth, "short_best_of", None) or {}
    if med_dur and int(sb.get("takes", 1) or 1) > 1:
        # 短い発話（カナ max_kana 以下）は合成器の short_best_of が人間の長さの分布から引いた目標で選び済み → 中央に寄せ直さない
        mk = int(sb.get("max_kana", 6))
        targets = [i for i in targets if len(RC.kana(re.sub(r"\[[^\]]*\]", "", tts_texts[i]))) > mk]
    if cfg.get("short_single") and hasattr(synth, "_inner_short"):
        # ★2026-11-20 B3（best_of.short_single: true）: 合成器が中で short_takes 本から選ぶ短い発話（Irodori の _post_speech）は
        #   テイク選択に掛けない（最初の合成の short_takes 本からの選択だけにする）。それまでは 7 回の追加テイクがそれぞれ
        #   short_takes 本を引いて選び（1 ターン 6＋7×6＝48 要求）、しかも seed が seed_base＋7919×(j＋k)＋i で重なるので
        #   42 本の追加要求のうち違う seed は 12 本だけだった
        targets = [i for i in targets if not synth._inner_short(normalize_tts_text(tts_texts[i]))]
    jobs = [(i, j) for i in targets for j in range(1, n_takes)]
    seed_base = int(cfg.get("seed_base", 7001))

    def _gen(job):
        i, j = job
        ref = references[spk_index[turns[i]["speaker"]]]
        try:
            if hasattr(synth, "synth_one_seeded"):       # 並列に投げられる（サーバが束ねる）
                return job, synth.synth_one_seeded(tts_texts[i], ref, seed_base + 7919 * j + i)
            return job, _synth_one_line(synth, tts_texts[i], ref, None, seed=seed_base + 7919 * j + i)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{record['id']}] テイク追加合成に失敗 turn{i} take{j}: {e}")
            return job, None

    extra_takes: dict = {}
    need = {i: n_takes - 1 for i in targets}
    swapped, extra = 0, len(jobs)
    muted = 0
    _t_gen0 = time.time()
    t_score = 0.0

    def _pick(i):
        """ターン i のテイクがそろったら採点して差し替える（合成と重ねるため、そろった順に呼ぶ）。"""
        nonlocal swapped, t_score, muted
        t0 = time.time()
        k = spk_index[turns[i]["speaker"]]
        takes = [w for w in [wav_by_turn[i]] + extra_takes.get(i, []) if w is not None and len(w) > 0]
        if len(takes) >= 2:
            sc_ = _scores(takes, i, k)
            best = int(np.argmin(sc_))
            if cfg.get("score_log") and hasattr(synth, "_score_log"):
                # ★2026-11-20 B2 の閾値を決めるための採点の記録（best_of.score_log にパスがあるときだけ。選び方は変えない。
                #   追加テイクの並びはそろった順なので j の順ではない）
                synth._score_log(cfg.get("score_log"), {"kind": "speech", "id": record["id"], "turn": int(i),
                                                        "text": tts_texts[i], "n_used": len(takes), "n_max": int(n_takes),
                                                        "scores": [round(float(x), 4) for x in sc_], "chosen": best,
                                                        "len": int(len(takes[best]))})
            if takes[best] is not wav_by_turn[i]:
                wav_by_turn[i] = takes[best]
                swapped += 1
            tb = tail_of.get(i, [None] * len(takes))[best]
            if tb is not None:                       # どのテイクにも台本外の末尾の音があった → その音だけ無音に
                wav_by_turn[i] = mute_span(wav_by_turn[i], tb[0], tb[1], sr)
                muted += 1
        t_score += time.time() - t0

    def _done(i, w):
        extra_takes.setdefault(i, []).append(w)
        need[i] -= 1
        if need[i] == 0:
            _pick(i)

    conc = int(getattr(synth, "concurrency", 1) or 1) if hasattr(synth, "synth_one_seeded") else 1
    stg = cfg.get("staged") or {}
    if stg and targets:
        # ★2026-11-20 B2（best_of.staged: {first, step, thresh}）: 8 本を一度に引かず、最初の合成＋追加で first 本（既定 4）を採点し、
        #   いちばん良い点（下の _scores。小さいほど良い）が thresh 以下ならそこで止め、超えていれば step 本ずつ足して takes 本まで引く。
        #   seed は従来と同じ seed_base＋7919×j＋i（j は 1 から順）なので、最初の段は従来の j＝1..first−1 と同じテイク。
        #   点はテイクごとに他のテイクに依らないので、takes 本まで引いたときは従来と同じものを選ぶ。採点も足したテイクの分だけ。
        import concurrent.futures
        first = max(2, int(stg.get("first", 4)))
        step = max(1, int(stg.get("step", max(1, n_takes - first))))
        thr = float(stg.get("thresh", float("-inf")))
        nxt = {i: 1 for i in targets}                 # 次に引く j
        left = {i: 0 for i in targets}                # この段で未着の本数
        got: dict = {i: {} for i in targets}          # この段で届いた j → wav
        tk: dict = {i: [wav_by_turn[i]] for i in targets}     # 採点済みのテイク（j＝0 は最初の合成）
        jk: dict = {i: [0] for i in targets}
        sc: dict = {i: [] for i in targets}
        tl: dict = {i: [] for i in targets}
        n_sub = 0

        def _finish(i):
            nonlocal swapped, muted
            ok = [p for p, w in enumerate(tk[i]) if w is not None and len(w) > 0]
            if len(ok) < 2 or not sc[i]:
                return
            best = min(ok, key=lambda p: sc[i][p])
            if cfg.get("score_log") and hasattr(synth, "_score_log"):
                synth._score_log(cfg.get("score_log"), {"kind": "speech", "id": record["id"], "turn": int(i),
                                                        "text": tts_texts[i], "n_used": len(ok), "n_max": int(n_takes),
                                                        "scores": [round(float(sc[i][p]), 4) for p in ok],
                                                        "js": [int(jk[i][p]) for p in ok], "chosen": ok.index(best),
                                                        "len": int(len(tk[i][best]))})
            if tk[i][best] is not wav_by_turn[i]:
                wav_by_turn[i] = tk[i][best]
                swapped += 1
            if tl[i][best] is not None:              # どのテイクにも台本外の末尾の音があった → その音だけ無音に
                wav_by_turn[i] = mute_span(wav_by_turn[i], tl[i][best][0], tl[i][best][1], sr)
                muted += 1

        def _score_new(i):
            """届いた段のテイクを j の順に足して、足した分だけ採点する（j＝0 の最初の合成は最初の段で一緒に採点）。"""
            nonlocal t_score
            t0 = time.time()
            if not sc[i]:
                tk[i] = [wav_by_turn[i]]
                jk[i] = [0]
            new_p = []
            for j in sorted(got[i]):
                tk[i].append(got[i][j])
                jk[i].append(j)
            got[i] = {}
            start = len(sc[i])
            for p in range(start, len(tk[i])):
                new_p.append(p)
            valid = [p for p in new_p if tk[i][p] is not None and len(tk[i][p]) > 0]
            k = spk_index[turns[i]["speaker"]]
            s_new = _scores([tk[i][p] for p in valid], i, k) if valid else []
            t_new = tail_of.get(i, [None] * len(valid))
            m = {p: (s, tt) for p, s, tt in zip(valid, s_new, t_new)}
            for p in new_p:
                s, tt = m.get(p, (float("inf"), None))
                sc[i].append(float(s))
                tl[i].append(tt)
            t_score += time.time() - t0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(conc, len(targets) * (first - 1)))) as ex:
            pend: dict = {}

            def _submit(i, hi):
                nonlocal n_sub
                for j in range(nxt[i], hi):
                    pend[ex.submit(_gen, (i, j))] = (i, j)
                    n_sub += 1
                left[i] = hi - nxt[i]
                nxt[i] = hi

            for i in targets:                        # ターン順に投げる＝ターン順にそろう
                _submit(i, min(first, n_takes))
            while pend:
                done, _ = concurrent.futures.wait(list(pend), return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    i, j = pend.pop(f)
                    (_i, _j), w = f.result()
                    got[i][j] = w
                    left[i] -= 1
                    if left[i] > 0:
                        continue
                    _score_new(i)
                    best_s = min(sc[i]) if sc[i] else float("inf")
                    if best_s > thr and nxt[i] < n_takes:
                        _submit(i, min(n_takes, nxt[i] + step))
                    else:
                        _finish(i)
        logger.info(f"[{record['id']}] テイク選択（品質・段階 first={first} thresh={thr:g}）: {n_sub}テイク追加合成"
                    f"（全部引くと {len(jobs)}）→ {swapped}ターンを差し替え"
                    + (f"・末尾の余計な音を無音に {muted}ターン" if muted else "") +
                    f"（全体 {time.time() - _t_gen0:.0f}秒・うち採点 {t_score:.0f}秒）")
        return swapped
    if conc > 1:
        import concurrent.futures
        # 採点（ASR・NLL・x-vector）はこのスレッドだけで回し、その間も合成の要求はサーバに並べておく
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(conc, max(1, len(jobs)))) as ex:
            futs = [ex.submit(_gen, job) for job in jobs]   # ターン順に投げる＝ターン順にそろう
            for f in concurrent.futures.as_completed(futs):
                (i, _j), w = f.result()
                _done(i, w)
    else:
        for job in jobs:
            (i, _j), w = _gen(job)
            _done(i, w)
    if extra:
        logger.info(f"[{record['id']}] テイク選択（{'長さ中央' if med_dur else '品質'}）: {extra}テイク追加合成 → {swapped}ターンを差し替え"
                    + (f"・末尾の余計な音を無音に {muted}ターン" if muted else "") +
                    f"（全体 {time.time() - _t_gen0:.0f}秒・うち採点 {t_score:.0f}秒）")
    return swapped


def select_best_takes(synth, record: dict, voices: list[dict],
                      wav_by_turn: dict, sr: int, cfg: dict,
                      emotion_tags: bool = False,
                      fish_tags_map: dict | None = None,
                      prefix_tag: str | None = None) -> int:
    """ターンごとに複数テイクを引き、特徴が中庸なテイクを採用する。

    fish は1コールごとに話速・抑揚・音量が揺れる（同一の声・同じ文を繰り返した実測で
    話速の比の sd が 0.046、出力レベルは sd 2.4〜5.2dB）。生成パラメータでは抑えられない
    ことが分かっているので、「何本か引いて真ん中を採る」ことでばらつきだけを削る。
    音声そのものには一切手を触れない（タイムストレッチと違い信号を加工しない）。

    合成コストは takes 倍になる。短い発話・相槌は測っても意味が無いので1テイクのまま。
    fish のクローン参照合成でのみ動作し、非対応構成では何もせず 0 を返す。
    差し替えた（初回テイク以外を採った）ターン数を返す。
    """
    if not cfg or not cfg.get("enabled", False):
        return 0
    n_takes = max(1, int(cfg.get("takes", 5)))
    if n_takes < 2:
        return 0
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return 0
    if str(cfg.get("criterion", "median")) in ("quality", "median_dur"):
        return _select_quality_takes(synth, record, wav_by_turn, sr, cfg, inp, n_takes)
    turns, speaker_order, references, spk_index, tts_texts, _lines = inp
    attrs = resolve_speaker_attrs(record, speaker_order)
    assignment = assign_voices(record["id"], speaker_order, voices, attrs)
    min_mora = float(cfg.get("min_mora", 8.0))
    do_bc = bool(cfg.get("backchannel", False))
    # テイクごとの temperature。先頭（既に合成済みのテイク）は既定値のまま。
    temps = take_temperatures(cfg, n_takes)

    swapped = 0
    extra = 0
    hist: dict = {}
    for i in sorted(wav_by_turn):
        t = turns[i]
        if not do_bc and t.get("type") == "backchannel":
            continue
        mora, conf = estimate_mora(t.get("text", ""))
        if mora < min_mora or conf < float(cfg.get("min_confidence", 0.6)):
            continue
        sp = t["speaker"]
        gender = (assignment[sp].get("gender")
                  or attrs.get(sp, {}).get("gender"))
        ref = references[spk_index[sp]]
        takes = [wav_by_turn[i]]
        for k in range(1, n_takes):
            w = _synth_one_line(synth, tts_texts[i], ref, temps[k])
            extra += 1
            if w is not None and len(w) > 0:
                takes.append(w)
        if len(takes) < 2:
            continue
        feats = [take_features(w, sr, mora, gender) for w in takes]
        scores = score_takes(feats, cfg)
        best = int(np.argmin(scores))
        hist[temps[best] if best < len(temps) else None] = (
            hist.get(temps[best] if best < len(temps) else None, 0) + 1)
        if best != 0:
            wav_by_turn[i] = takes[best]
            swapped += 1
    if extra:
        ts = [t for t in temps[1:] if t is not None]
        tinfo = f"（t={'/'.join(f'{t:.1f}' for t in sorted(set(ts)))}）" if ts else ""
        logger.info(f"[{record['id']}] テイク選択: {extra}テイク追加合成{tinfo} → "
                    f"{swapped}ターンを差し替え{_fmt_temp_hist(hist)}")
    return swapped


def regenerate_hiss_turns(synth, record: dict, voices: list[dict],
                          wav_by_turn: dict, sr: int, cfg: dict,
                          emotion_tags: bool = False,
                          fish_tags_map: dict | None = None,
                          prefix_tag: str | None = None) -> int:
    """合成済みターン群を走査し、サーが乗ったターンだけ再合成して差し替える。

    fish のクローン参照合成（_fish_multiturn_inputs が成立する構成）でのみ動作。
    非対応構成では何もせず 0 を返す。差し替えた件数を返す（wav_by_turn を破壊的更新）。
    """
    if not cfg or not cfg.get("enabled", True):
        return 0
    hiss_sec = float(cfg.get("hiss_sec", 0.6))
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return 0
    turns, speaker_order, references, spk_index, tts_texts, _lines = inp
    fixed = 0
    for i in sorted(wav_by_turn):
        w = wav_by_turn[i]
        if hiss_run(w, sr) <= hiss_sec:
            continue
        sp = turns[i]["speaker"]
        ref = references[spk_index[sp]]
        new_w, diag = regen_one_turn(
            synth, tts_texts[i], ref, sr, hiss_sec=hiss_sec,
            max_retries=int(cfg.get("max_retries", 4)),
            temp_schedule=cfg.get("temp_schedule"))
        if new_w is not None:
            wav_by_turn[i] = new_w
            fixed += 1
            logger.info(f"[{record['id']}] サー検知→再合成 turn{i}「{tts_texts[i][:16]}…」 "
                        f"最終hiss={diag['hiss']}s tries={diag['tries']}")
    if fixed:
        logger.info(f"[{record['id']}] サー再合成: {fixed}ターン差し替え")
    return fixed


def dewhistle_gate(wav: np.ndarray, sr: int,
                   hop_ms: float = 10.0, win_ms: float = 25.0,
                   quiet_rel: float = 0.10, quiet_abs: float = 0.02,
                   peak_hz: float = 4000.0, flat_max: float = 0.15,
                   hf_hz: float = 3500.0, ramp_ms: float = 6.0,
                   samp_ceil: float = 0.25, lf_max: float = 0.35,
                   speech_rms: float = 0.05, guard_ms: float = 30.0,
                   min_run: int = 4, run_median_hz: float = 7000.0,
                   hi_frac_min: float = 0.6, gap_tol: int = 2) -> np.ndarray:
    """無音区間に乗る「持続する高域純音」（fish-speech 等の『キーン』）だけを消す。

    ニューラルTTSは発話の切れ目（発話内ポーズ・語尾・音節間）で ~8kHz 付近の純音を
    ハルシネーションする。振幅は小さい（RMS 0.003〜0.01）が純音なので「キーン」と残る。
    trim_silence は絶対閾値で有音判定するためこのトーンを残し、発話内ポーズは刈れない。

    難所は「キーン(8kHz純音)」と「無声摩擦音/息(サ行など5〜7kHzの高域雑音)」の分離。
    両者は周波数もスペクトル平坦度も重なるため、単一フレームでは分けられない。決め手は
    時間的性質: キーンは高い周波数をほぼ一定に保つ持続音、摩擦音は短くピーク周波数が動く。
    そこで2段構えにする:
      ① フレーム候補: 無音(rms<max(quiet_rel*peak,quiet_abs)) かつ 発話帯域(hf_hz未満)の
         エネルギー比<lf_max かつ ピーク>peak_hz かつ 高域平坦度<flat_max。
         さらに発話(rms>=speech_rms)の ±guard_ms は絶対に候補にしない（発話保護）。
         フレーム内ピーク標本>=samp_ceil の枠も保護（発話過渡音）。
      ② run採用: 候補フレームを連続run(gap_tol枠の隙間は許容)にまとめ、run長>=min_run
         かつ run内でピーク>run_median_hz のフレーム割合>=hi_frac_min のときだけ消す。
         → 「持続的で一貫して高域」＝キーンのみ採用。短い/低め/周波数が動く摩擦音は残る。
    ramp_ms の傾斜でクリックを防ぐ。trim_silence の前に適用（末尾トーンが0化され端刈りも効く）。
    設計・検証の詳細はメモリ dewhistle-gate 参照。可聴発話(|x|>0.10)の変化は 0 を実測で確認。
    """
    w = np.asarray(wav if wav.ndim == 1 else wav.mean(axis=1), dtype=np.float32)
    n = len(w)
    if n == 0:
        return wav
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = max(hop * 2, int(sr * win_ms / 1000.0))
    peak = float(np.abs(w).max())
    if peak <= 0:
        return wav
    qthr = max(quiet_rel * peak, quiet_abs)
    han = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hf_mask = freqs >= hf_hz
    lf_mask = freqs < hf_hz

    def _seg(k):
        s = k * hop
        seg = w[s:s + win]
        return seg if len(seg) == win else np.pad(seg, (0, win - len(seg)))

    nfr = 1 + max(0, (n - win) // hop)
    frms = np.array([float(np.sqrt(np.mean(_seg(k) ** 2))) for k in range(nfr)])

    # 発話(rms>=speech_rms)の ±guard_ms を保護マスクにする
    near_speech = frms >= speech_rms
    guard = max(0, int(guard_ms / hop_ms))
    if guard and near_speech.any():
        sidx = np.where(near_speech)[0]
        for d in range(1, guard + 1):
            near_speech[np.clip(sidx - d, 0, nfr - 1)] = True
            near_speech[np.clip(sidx + d, 0, nfr - 1)] = True

    # ① フレーム候補判定（+ ピーク周波数を記録）
    cand = np.zeros(nfr, dtype=bool)
    cand_hz = np.zeros(nfr, dtype=np.float32)
    for k in range(nfr):
        if near_speech[k] or frms[k] >= qthr:
            continue
        seg = _seg(k)
        if float(np.abs(seg).max()) >= samp_ceil:
            continue
        sp = np.abs(np.fft.rfft(seg * han)) ** 2
        tot = float(sp.sum()) or 1e-12
        if float(sp[lf_mask].sum()) / tot > lf_max:
            continue                                   # 低域が多い → 有声音/母音を保護
        ph = float(freqs[int(np.argmax(sp))])
        if ph <= peak_hz:
            continue
        hb = sp[hf_mask] + 1e-12
        flat = float(np.exp(np.mean(np.log(hb))) / np.mean(hb))
        if flat < flat_max:
            cand[k] = True
            cand_hz[k] = ph

    # ② 連続run単位で「持続的かつ一貫して高域」なものだけ採用
    kill = np.zeros(nfr, dtype=bool)
    k = 0
    while k < nfr:
        if not cand[k]:
            k += 1
            continue
        last = k
        j = k
        while j < nfr:
            if cand[j]:
                last = j
                j += 1
            elif j - last <= gap_tol:                  # gap_tol 枠までの隙間は同一runとみなす
                j += 1
            else:
                break
        run = [i for i in range(k, last + 1) if cand[i]]
        hi_frac = float(np.mean(cand_hz[run] > run_median_hz))
        if len(run) >= min_run and hi_frac >= hi_frac_min:
            for i in range(k, last + 1):               # 隙間も含めて連続で消す（ramp で自然に）
                kill[i] = True
        k = last + 1

    if not kill.any():
        return wav
    gain = np.ones(n, dtype=np.float32)
    for k in np.where(kill)[0]:
        s = k * hop
        gain[s:min(n, s + hop)] = 0.0
    r = max(1, int(sr * ramp_ms / 1000.0))             # クリック防止の傾斜制限
    for i in range(1, n):
        if gain[i] > gain[i - 1]:
            gain[i] = min(gain[i], gain[i - 1] + 1.0 / r)
        elif gain[i] < gain[i - 1]:
            gain[i] = max(gain[i], gain[i - 1] - 1.0 / r)
    return w * gain


def _generate_noise(n: int, color: str, rng) -> np.ndarray:
    """長さ n・RMS=1 に正規化したノイズを返す。color='pink' | 'white'。

    pink は白色雑音を FFT 上で 1/sqrt(f)（=1/f パワー）にシェイピングして生成する
    （scipy 不要、numpy のみ）。低域寄りで自然な「部屋鳴り」に近い。
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n).astype(np.float32)
    if color == "pink":
        X = np.fft.rfft(white)
        f = np.arange(X.size)
        scale = np.ones(X.size, dtype=np.float64)
        scale[1:] = 1.0 / np.sqrt(f[1:])          # f=0（DC）は据え置き
        y = np.fft.irfft(X * scale, n).astype(np.float32)
    else:
        y = white
    rms = float(np.sqrt(np.mean(y * y)))
    if rms <= 0:
        return np.zeros(n, dtype=np.float32)
    return (y / rms).astype(np.float32)


def fill_silence_with_noise(stereo: np.ndarray, placements, channel_of: dict,
                            turns: list, failed_turns: set, sr: int,
                            cfg: dict) -> None:
    """組み上がったマルチchバッファの無音区間にコンフォートノイズを敷く（in-place）。

    合成音声はノイズフロアが極めて低いため、完全無音（ゼロ）と発話の境目が目立つ。
    そこで無音区間に微小なルームトーンを敷いて境目を馴染ませる仕上げ処理。

    scope='gaps'（既定): チャネルごとに発話区間の“隙間”のみノイズを入れ、発話の
      始端/終端で fade_ms のクロスフェード（発話サンプルは一切変更しない）。
    scope='continuous': 全区間に薄いノイズを常時加算（発話下にも乗る）。
    """
    if not cfg or not cfg.get("enabled", False):
        return
    total_len, num_ch = stereo.shape
    if total_len == 0:
        return
    color = str(cfg.get("color", "pink"))
    level_dbfs = float(cfg.get("level_dbfs", -48.0))
    amp = 10.0 ** (level_dbfs / 20.0)             # RMS=1 のノイズに掛けて目標 dBFS
    scope = str(cfg.get("scope", "gaps"))
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))

    if scope == "continuous":
        for ch in range(num_ch):
            stereo[:, ch] += _generate_noise(total_len, color, rng) * amp
        return

    # scope == "gaps": チャネルごとに発話マスクを作り、その補集合を埋める
    fade_len = max(0, int(float(cfg.get("fade_ms", 30.0)) / 1000.0 * sr))
    spans_by_ch: dict[int, list] = {ch: [] for ch in range(num_ch)}
    for i, start_sample, wav, _s, _e in placements:
        if i in failed_turns or wav is None or len(wav) == 0:
            continue
        ch = channel_of[turns[i]["speaker"]]
        end_sample = min(total_len, start_sample + len(wav))
        if end_sample > start_sample:
            spans_by_ch[ch].append((max(0, start_sample), end_sample))

    for ch in range(num_ch):
        # 発話区間を union → 補集合（=無音ギャップ）を列挙
        spans = sorted(spans_by_ch[ch])
        merged: list = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        gaps = []
        cur = 0
        for s, e in merged:
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < total_len:
            gaps.append((cur, total_len))

        for g0, g1 in gaps:
            n = g1 - g0
            if n <= 0:
                continue
            seg = _generate_noise(n, color, rng) * amp
            # 発話端でノイズが 0 になるようクロスフェード（クリック・段差防止）
            f = min(fade_len, n // 2)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                seg[:f] *= ramp
                seg[-f:] *= ramp[::-1]
            stereo[g0:g1, ch] += seg          # ギャップは元々ゼロなので加算=代入


def build_context_prompt(turns: list, i: int, short_len: int = 6) -> str | None:
    """短い相槌・フィラーを文脈に合った抑揚で合成するためのプロンプトを作る。

    対象は type==backchannel または実質文字数 <= short_len の短発話。直前の「別話者の
    発話」を会話文脈として渡し、Gemini-TTS に自然な即応の相づち/短い反応として発話させる。
    対象外（長い発話）は None を返す（＝base_prompt を使う）。
    """
    t = turns[i]
    text = (t.get("text") or "").strip()
    is_short = t.get("type") == "backchannel" or _core_len(text) <= short_len
    if not is_short:
        return None
    # 直前の別話者の発話（文脈）を探す
    prev_other = None
    for j in range(i - 1, -1, -1):
        if turns[j].get("speaker") != t.get("speaker"):
            po = (turns[j].get("text") or "").strip()
            if po:
                prev_other = po
                break
    ctx = f"相手が「{prev_other[:60]}」と話している場面です。" if prev_other else ""
    return (f"これは自然な日本語の会話の一部です。{ctx}"
            "それに対する、相手の話に反応する短い相づち・返事として、"
            "会話の流れに合った自然な抑揚で、力まず、短く即座に発話してください。"
            "棒読みや不自然な強調をしないでください。")


# fish-speech S2 の感情タグ推定用（S2 は [laugh] 等の自由記述の自然言語指示を
# 本文に埋め込める。対話の文脈=テキストの語彙・記号・type・直前の相手発話から控えめに選ぶ）
_FISH_LAUGH_RE = re.compile(r"\(笑\)|（笑）|[wｗ]{2,}|あは+|ははは+|ふふ+|うふふ|えへへ|へへ+|くすっ")
_FISH_LAUGH_STRIP_RE = re.compile(r"(?:（笑）|\(笑\)|[wｗ]{2,}|笑)+\s*[。、!！?？…]*\s*$")


def fish_emotion_prefix(turns: list, i: int) -> tuple[str, str | None]:
    """発話を S2 の感情タグ付きにするため、(読ませるテキスト, タグ or None) を返す。

    タグは対話の文脈（当該発話の語彙・記号・type と直前の相手発話）から控えめに1つ選ぶ。
    読み上げてほしくない「(笑)」等のマーカーはテキストから除去する。過剰付与を避けるため
    明確な手掛かりが無ければ None（＝素の発話）。
    """
    t = turns[i]
    text = (t.get("text") or "").strip()
    prev_other = ""
    for j in range(i - 1, -1, -1):
        if turns[j].get("speaker") != t.get("speaker"):
            prev_other = (turns[j].get("text") or "").strip()
            break

    tag = None
    if _FISH_LAUGH_RE.search(text):
        tag = "[laugh]"
        text = _FISH_LAUGH_STRIP_RE.sub("", text).strip() or "ふふ"
    elif re.search(r"えー[!！]|えっ[!？っ?]|まじ[っで!！?？]|マジ|うそ[っー〜ぉ!！?？]|ウソ|ほんと[!?？]|へえ[!！]|ええっ|わっ|なんと[!！?？]", text):
        tag = "[surprised]"
    elif ("！" in text or "!" in text) and re.search(
            r"やった|すご|最高|うれし|嬉し|楽し|いいね|よかった|わー|おお|やば|わくわく", text):
        tag = "[excited]"
    elif re.search(r"ごめん|すみません|申し訳|すまない|すまん", text):
        tag = "[apologetic]"
    elif re.search(r"はぁ|はあ…|ふぅ|困った|まいった|疲れた|しんど|めんど", text):
        tag = "[sigh]"
    elif re.search(r"うーん|う〜ん|うんと|えーと|ええと|んーと|んー、", text):
        tag = "[thoughtful]"
    elif t.get("type") == "backchannel":
        # 相手が笑っている流れなら笑いの相づち、それ以外はやわらかい短い反応に
        tag = "[laugh]" if _FISH_LAUGH_RE.search(prev_other) else "[gentle]"
    return text, tag


def _synth_one_or_silence(synth, text: str, est: float, voice: dict,
                          sr: int, dialogue_id: str, prompt: str | None = None):
    """1発話を個別合成。失敗したら推定長ぶんの無音を返す。

    Returns: (wav: np.ndarray, failed: bool)
    """
    w = None
    try:
        # 発話長をモデルに指示したいエンジン（irodori）にだけ推定長を渡す
        kw = {"durations": [est]} if getattr(synth, "wants_durations", False) else {}
        r = synth.synth([text], voice, prompts=[prompt], **kw)
        if r and r[0] is not None and len(r[0]) > 0:
            w = r[0]
    except Exception as e:
        logger.warning(f"[{dialogue_id}] 個別合成失敗→無音化: '{text[:16]}' ({type(e).__name__})")
    if w is None:
        n = max(1, int(round((est if est and est > 0 else 0.3) * sr)))
        return np.zeros(n, dtype=np.float32), True
    return w, False


def synthesize_for_speaker(synth, texts: list[str], voice: dict,
                           est_durs: list[float], dialogue_id: str, sr: int,
                           short_threshold: int = 4, speed: float = 1.0,
                           prompts: list[str | None] | None = None):
    """1話者の発話列を合成する（堅牢化）。

    短すぎる相槌・フィラー（「うん」等）は sarashina2.2-tts が音声トークンを生成できず
    （"No valid semantic tokens"）、デコーダの畳み込みでクラッシュする。バッチに1つでも
    混ざると話者全体が落ちるため:

      1. 短発話(core長 <= short_threshold)はバッチから分離し、個別に合成する。
      2. 残り(長め)はまとめてバッチ合成（高速）。失敗時のみ個別にフォールバック。
      3. 個別でも合成できない発話は、Stage 3 の推定長ぶんの無音で代替（タイムライン維持）。

    prompts[i] は発話ごとの文脈プロンプト（短い相槌の自然化用。None=既定プロンプト）。

    Returns: (wavs: list[np.ndarray], failed: list[bool])
    """
    n = len(texts)
    wavs: list = [None] * n
    failed = [False] * n
    pr = prompts if prompts is not None else [None] * n

    long_idx = [j for j in range(n) if _core_len(texts[j]) > short_threshold]
    short_idx = [j for j in range(n) if j not in set(long_idx)]

    # 2. 長めの発話はバッチ（高速パス）
    if long_idx:
        batch_texts = [texts[j] for j in long_idx]
        batch_prompts = [pr[j] for j in long_idx]
        ok = False
        try:
            kw = ({"durations": [est_durs[j] for j in long_idx]}
                  if getattr(synth, "wants_durations", False) else {})
            ws = synth.synth(batch_texts, voice, prompts=batch_prompts, **kw)
            if len(ws) == len(long_idx) and all(w is not None and len(w) > 0 for w in ws):
                for j, w in zip(long_idx, ws):
                    wavs[j] = w
                ok = True
            else:
                logger.warning(f"[{dialogue_id}] バッチ結果が不完全（{len(ws)}/{len(long_idx)}）→個別合成")
        except Exception as e:
            logger.warning(f"[{dialogue_id}] バッチ合成失敗→個別合成に切替: {type(e).__name__}: {e}")
        if not ok:
            for j in long_idx:
                wavs[j], failed[j] = _synth_one_or_silence(
                    synth, texts[j], est_durs[j], voice, sr, dialogue_id, pr[j])

    # 1./3. 短い発話は個別合成（失敗時は無音）。相槌はここで文脈プロンプトが効く
    for j in short_idx:
        wavs[j], failed[j] = _synth_one_or_silence(
            synth, texts[j], est_durs[j], voice, sr, dialogue_id, pr[j])

    # 発話速度を上げる（ピッチ保持タイムストレッチ、rate>1で速く）
    if abs(speed - 1.0) >= 1e-3:
        for j in range(n):
            if wavs[j] is not None and len(wavs[j]) > 0:
                wavs[j] = time_stretch(wavs[j], speed, sr)

    return wavs, failed


def dialogue_speaker_order(turns: list) -> list[str]:
    """出現順で話者を確定（ch0 = 先に話した方）。"""
    order = []
    for t in turns:
        if t["speaker"] not in order:
            order.append(t["speaker"])
    return order


def record_speaker_order(record: dict) -> list[str]:
    """ch0 に来る話者を確定する。チャネル割り当てはこちらを使うこと。

    既定は dialogue_speaker_order（＝先に話した方が ch0）で、従来コーパスの挙動は変わらない。
    metadata.main_speaker_index を持つコーパスでは、その話者を必ず先頭（ch0）に置く。
    J-Moshi の学習では左/ch0 がモデル自身のストリームになるため、ch0 を
    「誰が先に話したか」から切り離して固定話者に固定する必要がある。

    dialogue_speaker_order 自体は別リポジトリ（dialog_synth_voice）や
    gemini_batch / audit_pipeline からも import されているので、そちらは変えない。
    """
    order = dialogue_speaker_order(record["voice_dialogue"]["turns"])
    if len(order) != 2:
        return order
    main = _main_speaker_name(record)
    if main is None:
        return order
    if main not in order:
        logger.warning(f"[{record.get('id')}] 主話者「{main}」がターンに居ません"
                       f"(order={order})。出現順のままにします。")
        return order
    return order if order[0] == main else [order[1], order[0]]


def _legacy_placements(turns: list, wav_by_turn: dict[int, np.ndarray], sr: int):
    """レガシー配置: 実発話長に対して LLM の start_offset を連鎖適用する。

    Returns: [(turn_index, start_sample, wav, start_sec, end_sec)]
    """
    placements = []
    prev_end_sec = 0.0
    for i, t in enumerate(turns):
        wav = wav_by_turn[i]
        dur = len(wav) / sr
        offset = 0.0 if i == 0 else float(t.get("start_offset", 0.0) or 0.0)
        start_sec = max(0.0, prev_end_sec + offset)
        placements.append((i, int(round(start_sec * sr)), wav, start_sec, start_sec + dur))
        prev_end_sec = start_sec + dur
    return placements


# 対話の冒頭・末尾に置く無音（audio_padding に実装。後付けスクリプトと同じ値を使う）。
from audio_padding import lead_in_sec, padding_marks, tail_pad_sec  # noqa: E402

# リタイム有効なのに失敗してレガシー配置に落ちた対話ID。
# 例外を握り潰して書き出す経路なので、集計して必ず目に見えるようにする。
RETIME_FAILURES: list[str] = []


def compose_dialogue(record: dict, voices: list[dict], sr: int,
                     wav_by_turn: dict[int, np.ndarray],
                     failed_turns: set[int] | None = None,
                     retimer=None, trim: dict | None = None,
                     noise: dict | None = None, loudness: dict | None = None,
                     rate: dict | None = None, plans: list | None = None,
                     lead_in: dict | None = None, tail_pad: dict | None = None):
    """発話ごとの波形(wav_by_turn)を、話者数ぶんのマルチチャネル・タイムラインに合成する。

    話者ごとに1チャネルを割り当てる（2人なら2ch、3人なら3ch…出現順=チャネル順）。
    ライブ合成・バッチ合成の両方から使う共通処理。wav_by_turn は各ターン index→波形。
    retimer（retime_maai.MaaiRetimer 等）を渡すと、LLM の start_offset 連鎖の
    代わりに MaAI 予測で配置タイミングを決める（2話者のみ。plan() が None を
    返すか例外時はレガシー配置にフォールバック）。
    plans に retimer.plan() の結果を渡すと、その配置をそのまま使う（同じ
    (turns, wav_by_turn) に対して呼び出し側が既に引いている場合の重複計算防け。
    融合モード --retime-refine が使う）。
    Returns: (multi: np.ndarray[shape=(N, num_speakers)], manifest_turns) または None
    """
    failed_turns = failed_turns or set()
    turns = record["voice_dialogue"]["turns"]
    speaker_order = record_speaker_order(record)
    if len(speaker_order) < 2:
        logger.warning(f"[{record['id']}] 話者が2人未満({speaker_order})。スキップ。")
        return None
    # ★2026-11-10 重なりの場面（rp_v1）: 言いかけのターンを文の途中で切る（rp_audio。切り済みなら何もしない）。
    #   トリム・話速の均し・配置より前（切ったあとの長さで配置を決める）。
    from rp_audio import apply_cuts
    apply_cuts(record, wav_by_turn, sr, failed_turns)

    # 発話の前後無音をトリミング（TTSのパディング除去。相槌が実質短くなり被りが減る）。
    # plan/配置の前に適用するので、トリミング後の長さでタイミングが決まる。
    if trim and trim.get("enabled", True):
        max_bc = float(trim.get("max_bc_sec", 0.0) or 0.0)
        dw = trim.get("dewhistle") or None             # 無音区間の高域トーン(キーン)除去
        dw_on = bool(dw and dw.get("enabled", True))
        for i, t in enumerate(turns):
            w = wav_by_turn.get(i)
            if w is None or len(w) == 0 or i in failed_turns:
                continue
            if dw_on:                                  # trim の前: 末尾トーンも0化して端刈りを効かせる
                w = dewhistle_gate(
                    w, sr,
                    quiet_rel=float(dw.get("quiet_rel", 0.10)),
                    quiet_abs=float(dw.get("quiet_abs", 0.02)),
                    peak_hz=float(dw.get("peak_hz", 4000.0)),
                    flat_max=float(dw.get("flat_max", 0.15)),
                    hf_hz=float(dw.get("hf_hz", 3500.0)),
                    ramp_ms=float(dw.get("ramp_ms", 6.0)),
                    samp_ceil=float(dw.get("samp_ceil", 0.25)),
                    lf_max=float(dw.get("lf_max", 0.35)),
                    speech_rms=float(dw.get("speech_rms", 0.05)),
                    guard_ms=float(dw.get("guard_ms", 30.0)),
                    min_run=int(dw.get("min_run", 4)),
                    run_median_hz=float(dw.get("run_median_hz", 7000.0)),
                    hi_frac_min=float(dw.get("hi_frac_min", 0.6)),
                    gap_tol=int(dw.get("gap_tol", 2)))
            w = trim_silence(
                w, sr,
                threshold=float(trim.get("threshold", 0.02)),
                thresh_abs=float(trim.get("thresh_abs", 0.005)),
                keep_pad_sec=float(trim.get("keep_pad_sec", 0.03)),
                max_trim_sec=float(trim.get("max_trim_sec", 0.8)))
            # 相槌の異常長キャップ（合成異常への保険。末尾を短いフェードで切り詰め）
            if max_bc > 0 and t.get("type") == "backchannel" and len(w) > int(max_bc * sr):
                n = int(max_bc * sr)
                fade = min(int(0.02 * sr), n)
                w = w[:n].copy()
                if fade > 0:
                    w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            wav_by_turn[i] = w

    # 話速の均し。トリムのあと・配置の前に掛ける（長さが変わるので、リタイムの
    # plan() より前でなければならない）。ターン内のポーズも同じ比率で伸びる。
    if rate and rate.get("enabled", False):
        facs = speed_factors(turns, wav_by_turn, sr, rate, failed_turns)
        for i, f in facs.items():
            w = wav_by_turn.get(i)
            if w is not None and len(w) > 0:
                wav_by_turn[i] = time_stretch_wsola(w, f, sr)
        if facs:
            fv = np.array(list(facs.values()))
            logger.info(f"[{record['id']}] 話速均し: {len(facs)}/{len(turns)}ターン補正 "
                        f"（×{fv.min():.2f}〜×{fv.max():.2f}, 平均 ×{fv.mean():.2f}）")

    num_ch = len(speaker_order)                       # 話者数ぶんのチャネル
    channel_of = {sp: i for i, sp in enumerate(speaker_order)}
    attrs = resolve_speaker_attrs(record, speaker_order)
    assignment = assign_voices(record["id"], speaker_order, voices, attrs)

    # 配置決定: retimer があれば MaAI 予測、無ければレガシー連鎖（LLM offset）
    retime_by_idx: dict[int, dict] = {}
    placements = None
    if retimer is not None:
        if plans is None:
            try:
                plans = retimer.plan(record, turns, wav_by_turn, sr, channel_of,
                                     failed_turns)
            except Exception as e:
                # ★リタイム有効なのにレガシー配置（VAPなしの固定間隔）で書き出すのは
                #   「静かな劣化」なので、WARNING ではなく ERROR にして件数を積み、
                #   終了時に集計＋非ゼロ終了させる（RUNBOOK 10章の注意点）。
                RETIME_FAILURES.append(record["id"])
                logger.error(f"[{record['id']}] リタイム失敗→レガシー配置: "
                             f"{type(e).__name__}: {e}\n"
                             + traceback.format_exc())
                plans = None
        if plans is not None:
            placements = []
            for p in plans:
                i = p["index"]
                wav = wav_by_turn[i]
                start_sec = float(p["start_sec"])
                placements.append((i, int(round(start_sec * sr)), wav,
                                   start_sec, start_sec + len(wav) / sr))
                retime_by_idx[i] = p["retime"]
    if placements is None:
        placements = _legacy_placements(turns, wav_by_turn, sr)

    # 音量の均し。配置が決まったあとに掛けるので、リタイム（VAP）の判断も
    # 各ターンの長さもこれで変わらない（＝タイミングの較正はそのまま使える）。
    gains: dict[int, float] = {}
    if loudness and loudness.get("enabled", False):
        gains = loudness_gains(turns, wav_by_turn, sr, loudness, failed_turns)
        if gains:
            gdb = np.array([20.0 * np.log10(g) for g in gains.values()])
            logger.info(f"[{record['id']}] 音量均し: {len(gains)}/{len(turns)}ターン補正 "
                        f"（{gdb.min():+.1f}〜{gdb.max():+.1f}dB, "
                        f"平均 |{np.abs(gdb).mean():.1f}|dB）")
    # 相槌だけ一律に下げる（loudness.bc_gain_db・既定 0。loudness.enabled と無関係に効く）。人間の相槌は本人の発話より
    # 約 5 dB 小さい（maai-f2f 80 人の中央値 −5.0・IQR −7.9〜−2.7。10ms 窓の 90 パーセンタイル）が、pocket CC は −1〜−2 dB
    # （2026-11-14）。一律にずらすので相槌どうしの強弱の幅は残る。配置のあとなのでタイミングは変わらない。
    bc_gain_db = float((loudness or {}).get("bc_gain_db", 0.0) or 0.0)
    if bc_gain_db:
        g_bc = float(10.0 ** (bc_gain_db / 20.0))
        for i, t in enumerate(turns):
            if t.get("type") == "backchannel" and i in wav_by_turn and i not in failed_turns:
                gains[i] = gains.get(i, 1.0) * g_bc

    # 先頭の無音（tts.lead_in）。全ターンを同じ量だけ後ろへずらす。無音区間には
    # このあとコンフォートノイズが敷かれるので、デジタル無音の手がかりは残らない。
    # 配置（plan）はターンの相対タイミングだけで決まるので、ここでずらしても
    # リタイムの判断・gap 分布には影響しない。
    lead = lead_in_sec(record["id"], lead_in)
    if lead > 0:
        off = int(round(lead * sr))
        placements = [(i, s0 + off, w, st + lead, en + lead)
                      for i, s0, w, st, en in placements]
    # 末尾の無音（tts.tail_pad）。締めの挨拶のあとに何も起きない例を残す
    # （学習データが全対話「発話で終わる」と、黙り続けることを学べない）。
    tail = tail_pad_sec(record["id"], tail_pad)
    if lead > 0 or tail > 0:
        logger.info(f"[{record['id']}] 前後の無音 冒頭{lead:.3f}秒／末尾{tail:.3f}秒")

    total_len = (max((p[1] + len(p[2]) for p in placements), default=0)
                 + int(round(tail * sr)))
    stereo = np.zeros((total_len, num_ch), dtype=np.float32)

    manifest_turns = []
    for i, start_sample, wav, start_sec, end_sec in placements:
        ch = channel_of[turns[i]["speaker"]]
        end_sample = start_sample + len(wav)
        g = gains.get(i)
        if g is not None:
            wav = wav * g
        # 同一チャネルで重なった場合は加算ミックス
        stereo[start_sample:end_sample, ch] += wav
        sp = turns[i]["speaker"]
        entry = {
            "index": i,
            "speaker": sp,
            "channel": ch,
            "voice_id": assignment[sp]["id"],
            "gender": attrs.get(sp, {}).get("gender"),
            "age": attrs.get(sp, {}).get("age"),
            "type": turns[i].get("type", "speech"),
            "continued": bool(turns[i].get("continued", False)),  # 分割発話の続きか
            "text": turns[i]["text"],
            "start_offset": turns[i].get("start_offset"),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(len(wav) / sr, 3),
            "synth_failed": i in failed_turns,  # Trueなら合成不可で無音代替
        }
        if turns[i].get("pause_before_sec") is not None:   # ユーザの長い沈黙（dialogue.silence）
            entry["pause_before_sec"] = turns[i]["pause_before_sec"]
        for k in ("overlap", "unsaid", "cut_sec"):          # 重なりの場面（rp_v1）
            if turns[i].get(k) not in (None, ""):
                entry[k] = turns[i][k]
        if g is not None:                   # 音量均しで掛けたゲイン（dB）
            entry["gain_db"] = round(float(20.0 * np.log10(g)), 2)
        if i in retime_by_idx:  # リタイム時のみ: 決定方法・元のLLM配置などを記録
            entry["retime"] = retime_by_idx[i]
        manifest_turns.append(entry)

    # クリッピング防止（オーバーラップ加算でピークが1を超えた場合のみ正規化）
    peak = float(np.max(np.abs(stereo))) if total_len else 0.0
    if peak > 1.0:
        stereo /= peak
    # 仕上げ: 無音区間にコンフォートノイズを敷いて発話との境目を馴染ませる。
    # 正規化の後に足すことで出力上のノイズ RMS を dBFS で確定させる。
    fill_silence_with_noise(stereo, placements, channel_of, turns,
                            failed_turns, sr, noise or {})
    return stereo, manifest_turns


def drop_if_synth_failed(dialogue_id: str, manifest_turns: list) -> bool:
    """無音代替のターンが1つでもあれば ERROR を出して True（＝書き出さない）を返す。

    合成に失敗した発話は推定長ぶんの無音で代替される（タイムラインは保たれるが音は無い）。
    そのまま書き出すと「成功」扱いで無音入りの音声が混ざるため、1発話でも無音代替が出た
    対話は wav / manifest / transcript を一切書かずに落とす。wav が無ければ次回実行の
    resume 対象として残るので、ログで原因（fish の停止など）を直してから再実行すればよい。
    """
    bad = [t for t in manifest_turns if t.get("synth_failed")]
    if not bad:
        return False
    idx = ", ".join(str(t.get("index")) for t in bad[:10])
    more = "…" if len(bad) > 10 else ""
    logger.error(f"[{dialogue_id}] 無音代替のため対話を破棄: "
                 f"{len(bad)}/{len(manifest_turns)}発話が合成失敗（index: {idx}{more}）"
                 "。音声・manifest・transcript とも出力しない（原因を直して再実行）")
    return True


def compute_tts_texts(record_id: str, turns: list, emotion_tags: bool,
                      fish_tags_map: dict[int, str] | None,
                      prefix_tag: str | None = None) -> list[str]:
    """各ターンの合成用テキストを作る（原文・感情タグ・LLMインラインタグを反映）。

    優先順位: fish_tags_map（LLM方式）> emotion_tags（ルール方式の接頭タグ）> 原文。
    いずれも末尾句点を補完して正規化する。マニフェストは原文を使うので、ここで
    付くタグは合成入力にのみ載る。per-turn / multiturn 両方の合成経路で共用する。

    prefix_tag（任意）: 与えると全発話の冒頭に固定タグ（例 "[excited]"）を付与する。
    感情タグ機能(emotion_tags/fish_tags_map)とは独立で、それらの前に付く。一貫して
    テンションを上げたいときの簡易スイッチ。空/None なら無効。
    """
    tag_by_turn: dict[int, str | None] = {}
    spoken_by_turn: dict[int, str] = {}
    if emotion_tags:
        for i in range(len(turns)):
            spoken_by_turn[i], tag_by_turn[i] = fish_emotion_prefix(turns, i)
        from collections import Counter
        cnt = Counter(v for v in tag_by_turn.values() if v)
        if cnt:
            logger.info(f"[{record_id}] fish感情タグ: "
                        + ", ".join(f"{k}×{v}" for k, v in cnt.most_common()))

    def _one(i: int) -> str:
        if fish_tags_map and i in fish_tags_map:
            return normalize_tts_text(fish_tags_map[i])
        if emotion_tags:
            base = normalize_tts_text(spoken_by_turn.get(i) or tts_reading.turn_text(turns[i]))
            tg = tag_by_turn.get(i)
            return f"{tg} {base}" if tg else base
        return normalize_tts_text(tts_reading.turn_text(turns[i]))

    pre = (prefix_tag or "").strip()
    if pre:
        logger.info(f"[{record_id}] fish固定接頭タグ: '{pre}' を全{len(turns)}発話に付与")
    return [f"{pre} {_one(i)}" if pre else _one(i) for i in range(len(turns))]


def synthesize_turns(synth, record: dict, voices: list[dict], sr: int,
                     prefix_tag: str | None = None,
                     short_threshold: int = 4, speed: float = 1.0,
                     context_prosody: bool = False, context_short_len: int = 6,
                     emotion_tags: bool = False,
                     fish_tags_map: dict[int, str] | None = None):
    """1対話の全ターンを発話単位で合成する。

    context_prosody=True で、短い相槌・フィラーに会話文脈プロンプトを付けて自然化する。

    fish_tags_map（LLM方式・任意）: {ターンindex: インラインタグ付きテキスト}。与えられた
    ターンはその本文で合成する（＝合成入力にのみタグが載る。マニフェスト/transcriptは原文のまま）。
    emotion_tags（ルール方式）とは排他: LLM方式が有効なときは emotion_tags=False で呼ぶ。

    Returns: (wav_by_turn: dict[int, np.ndarray], failed_turns: set[int]) または None
    """
    turns = record["voice_dialogue"]["turns"]
    speaker_order = record_speaker_order(record)
    if len(speaker_order) < 2:
        logger.warning(f"[{record['id']}] 話者が2人未満({speaker_order})。スキップ。")
        return None

    attrs = resolve_speaker_attrs(record, speaker_order)
    assignment = assign_voices(record["id"], speaker_order, voices, attrs)

    # 短発話の文脈プロンプト（対象外は None＝既定プロンプト）
    ctx_prompts = ([build_context_prompt(turns, i, context_short_len)
                    for i in range(len(turns))] if context_prosody
                   else [None] * len(turns))

    # fish-speech: 文脈から感情タグを推定して本文に付与（なるべく自然にする）。
    # マニフェストは turns[i]["text"]（原文）を使うので、タグは合成入力にのみ載る。
    tts_texts = compute_tts_texts(record["id"], turns, emotion_tags, fish_tags_map,
                                  prefix_tag=prefix_tag)

    # 話者ごとにテキストをまとめて合成（元のターン位置を保持）
    per_speaker_idx: dict[str, list[int]] = {s: [] for s in speaker_order}
    for i, t in enumerate(turns):
        per_speaker_idx[t["speaker"]].append(i)

    wav_by_turn: dict[int, np.ndarray] = {}
    failed_turns: set[int] = set()
    for speaker, indices in per_speaker_idx.items():
        if not indices:
            continue
        texts = [tts_texts[i] for i in indices]
        est_durs = [float(turns[i].get("duration", 0.0) or 0.0) for i in indices]
        wavs, failed = synthesize_for_speaker(
            synth, texts, assignment[speaker], est_durs, record["id"], sr,
            short_threshold=short_threshold, speed=speed,
            prompts=[ctx_prompts[i] for i in indices])
        for i, w, fl in zip(indices, wavs, failed):
            wav_by_turn[i] = w
            if fl:
                failed_turns.add(i)

    if failed_turns:
        logger.warning(f"[{record['id']}] {len(failed_turns)}/{len(turns)} 発話を無音で代替（合成不可）。")
    return wav_by_turn, failed_turns


def _fish_multiturn_inputs(record: dict, voices: list[dict],
                           emotion_tags: bool, fish_tags_map: dict[int, str] | None,
                           prefix_tag: str | None):
    """マルチターン系合成の共通入力を作る（multiturn / per_speaker で共用）。

    Returns: (turns, speaker_order, references, spk_index, tts_texts, lines) または None。
      - references[i] は speaker_order[i] の声（クローン参照音声 audio_path+text）。
      - spk_index: 話者名→出現順インデックス（<|speaker:i|> の i）。
      - tts_texts[i]: ターン i の合成用テキスト（話者タグ無し）。
      - lines[i]: <|speaker:idx|>（idx=グローバル話者インデックス）付きの合成行。
        multiturn 用（全話者を1ストリームに並べる）。per_speaker は自分だけの参照・タグで
        組み直すため tts_texts/spk_index/references を使う。
    全話者にクローン参照音声が必要。1人でも欠ける／話者2人未満なら None（＝呼び出し側で
    per-turn 合成にフォールバック）。
    """
    turns = record["voice_dialogue"]["turns"]
    speaker_order = record_speaker_order(record)
    if len(speaker_order) < 2:
        logger.warning(f"[{record['id']}] 話者が2人未満({speaker_order})。スキップ。")
        return None

    attrs = resolve_speaker_attrs(record, speaker_order)
    assignment = assign_voices(record["id"], speaker_order, voices, attrs)
    spk_index = {sp: i for i, sp in enumerate(speaker_order)}  # 話者→<|speaker:i|>

    # 参照音声を話者インデックス順に用意（references[i] = speaker_order[i] の声）
    references: list[dict] = []
    for sp in speaker_order:
        v = assignment[sp]
        ap = v.get("audio_path")
        if not (ap and Path(ap).exists()):
            logger.warning(f"[{record['id']}] fish文脈合成: 話者「{sp}」の参照音声(audio_path)が無い"
                           "→ per-turn 合成にフォールバック。")
            return None
        with open(ap, "rb") as f:
            references.append({"audio": f.read(), "text": v.get("text", "")})

    tts_texts = compute_tts_texts(record["id"], turns, emotion_tags, fish_tags_map,
                                  prefix_tag=prefix_tag)
    # 各ターンの <|speaker:i|>（i=グローバル話者インデックス）付き合成行
    lines = [f"<|speaker:{spk_index[turns[i]['speaker']]}|>{tts_texts[i]}"
             for i in range(len(turns))]
    return turns, speaker_order, references, spk_index, tts_texts, lines


def _adopted_context(entries, sr: int) -> list[dict]:
    """採用済みターンを synth_dialogue の context 形式に変換する。

    entries は対話順の [(タグ付きテキスト, wav, codes), ...]。codes（生成時のVQ
    トークン）があれば無劣化のままそれを使い、無ければ wav を渡す（サーバでVQ
    再エンコード）。窓をまたいで文脈を連鎖させる場合は codes が必須級:
    decode→再encode の往復劣化が連鎖で蓄積し、後半の音声が崩れるため。
    wav も codes も無いターンはスキップ（文脈が1つ減るだけで害はない）。
    """
    ctx = []
    for text, w, codes in entries:
        if codes is not None:
            ctx.append({"text": text, "codes": codes})
        elif w is not None and len(w) > 0:
            ctx.append({"text": text, "audio": FishSpeechSynth._encode_wav(w, sr)})
    return ctx


def _turn_meta(turns: list, assignment: dict, attrs: dict) -> list[dict]:
    """ターンごとの (モーラ数・確からしさ・性別・相槌か) をまとめる（テイク採点用）。"""
    out = []
    for t in turns:
        sp = t["speaker"]
        mora, conf = estimate_mora(t.get("text", ""))
        out.append({"mora": mora, "conf": conf,
                    "gender": (assignment.get(sp, {}).get("gender")
                               or attrs.get(sp, {}).get("gender")),
                    "bc": t.get("type") == "backchannel"})
    return out


def best_window_take(synth, tagged: str, references: list, context, sr: int,
                     metas: list[dict], cfg: dict):
    """同じ窓を複数回生成し、ターンごとの特徴が中庸な窓を丸ごと採る。

    ターン単位で引き直すのではなく**窓ごと**引き直すのが要点。窓の中は1ストリームで
    生成されているので、1ターンだけ差し替えるとその窓の連続性が壊れるうえ、単発合成は
    窓合成とテイクの性質そのものが違う（＝比べる土俵が違う）。

    採点は score_takes と同じ基準（話速・抑揚・音量・ヒス）を窓内の各ターンで出し、
    候補ごとに合計する。測れないターン（相槌・短い発話・数字だらけ）は採点に入れない。
    Returns: (wavs, codes) 採用した窓。
    """
    n_takes = max(1, int(cfg.get("takes", 5)))
    cands = []
    for temp in take_temperatures(cfg, n_takes):
        with temperature_override(synth, temp):
            wavs, codes = synth.synth_dialogue(tagged, references, context=context,
                                               return_codes=True)
        cands.append((wavs, codes, temp))
    good = [c for c in cands if len(c[0]) == len(metas)]
    if len(good) < 2:
        _note_take_temp(synth, cands[0][2])
        return cands[0][:2]
    min_mora = float(cfg.get("min_mora", 8.0))
    min_conf = float(cfg.get("min_confidence", 0.6))
    do_bc = bool(cfg.get("backchannel", False))
    totals = [0.0] * len(good)
    for j, meta in enumerate(metas):
        if meta["mora"] < min_mora or meta["conf"] < min_conf:
            continue
        if meta["bc"] and not do_bc:
            continue
        feats = [take_features(c[0][j], sr, meta["mora"], meta["gender"]) for c in good]
        for k, sc in enumerate(score_takes(feats, cfg)):
            totals[k] += sc if sc != float("inf") else 10.0
    pick = good[int(np.argmin(totals))]
    _note_take_temp(synth, pick[2])
    return pick[:2]


def synthesize_turns_multiturn(synth, record: dict, voices: list[dict], sr: int,
                               emotion_tags: bool = False,
                               fish_tags_map: dict[int, str] | None = None,
                               prefix_tag: str | None = None):
    """fish-speech マルチターン合成: 対話全体を1回で合成し文脈を引き継ぐ。

    <|speaker:i|>（i=話者の出現順インデックス）でタグ付けした対話全体を
    FishSpeechSynth.synth_dialogue に渡し、ターンごとに分離した波形を得る。
    戻り値は synthesize_turns と互換の (wav_by_turn, failed_turns)。

    全話者にクローン参照音声（audio_path）が必要。要件を満たさない／セグメント数が
    ターン数と一致しない／合成に失敗した場合は None を返し、呼び出し側で per-turn 合成に
    フォールバックさせる（＝既存パスへ安全に劣化）。
    """
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, speaker_order, references, _spk_index, _tts_texts, lines = inp
    n = len(turns)

    # 窓ごとのテイク選択（best_of）。窓を丸ごと引き直して中庸なものを採る。
    best_cfg = getattr(synth, "best_of", None)
    n_takes = int((best_cfg or {}).get("takes", 5)) if (
        best_cfg and best_cfg.get("enabled", False)) else 1
    metas_all = None
    if n_takes > 1:
        attrs = resolve_speaker_attrs(record, speaker_order)
        metas_all = _turn_meta(turns, assign_voices(record["id"], speaker_order,
                                                    voices, attrs), attrs)

    synth.take_temp_hist = {} if n_takes > 1 else None

    def _call(tagged, ctx, idxs):
        """1コールぶんの生成。best_of 有効なら窓ごとに引き直して中庸な窓を採る。"""
        if n_takes > 1 and metas_all is not None:
            return best_window_take(synth, tagged, references, ctx, synth.sample_rate,
                                    [metas_all[i] for i in idxs], best_cfg)
        return synth.synth_dialogue(tagged, references, context=ctx, return_codes=True)

    # 窓分割: 1コールあたり最大 W ターン（対象）＋先頭 C ターン（文脈の種・音声は捨てる）。
    # W<=0 は対話全体を1コール（旧挙動）。これで1コールの履歴長を W+C に抑え、O(N²)→O(N·W)。
    W = int(getattr(synth, "window", 0) or 0)
    C = max(0, int(getattr(synth, "window_context", 0) or 0))

    use_ctx_audio = bool(getattr(synth, "context_audio", False))

    wav_by_turn: dict[int, np.ndarray] = {}
    codes_by_turn: dict[int, object] = {}   # 採用ターンのVQトークン（無劣化の文脈連鎖用）
    try:
        if W > 0 and n > W:
            starts = list(range(0, n, W))
            # vLLM-Omni（parallel_windows）: 文脈がテキスト（context_audio でない）なら窓どうしは独立なので、
            # 窓ごとのコールを先に並列で投げてサーバに束ねさせる。結果は下の逐次ループが順に使う。
            prefetched: dict[int, tuple] = {}
            if (getattr(synth, "parallel_windows", False) and not use_ctx_audio and n_takes <= 1
                    and int(getattr(synth, "concurrency", 1) or 1) > 1 and len(starts) > 1):
                specs = []
                for s in starts:
                    e = min(s + W, n)
                    call_idx = list(range(max(0, s - C), e))
                    specs.append((s, "\n".join(lines[i] for i in call_idx), call_idx))
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(int(synth.concurrency), len(specs))) as ex:
                    futs = {s: ex.submit(_call, tagged, None, call_idx) for s, tagged, call_idx in specs}
                    prefetched = {s: f.result() for s, f in futs.items()}
            for s in starts:
                e = min(s + W, n)
                ctx_start = max(0, s - C)              # 直前 C ターンを文脈種に
                if use_ctx_audio:
                    # 文脈=前の窓で採用済みの本物のターン（VQトークン優先）。text は対象のみ。
                    call_idx = list(range(s, e))
                    context = _adopted_context(
                        [(lines[i], wav_by_turn.get(i), codes_by_turn.get(i))
                         for i in range(ctx_start, s)],
                        synth.sample_rate)
                    if getattr(synth, "context_anchor", False):
                        # 全話者の参照音声をフェイク直前ターンとしても前置き（ドリフト引き戻し）
                        context = [{"text": f"<|speaker:{k}|>{r['text']}",
                                    "audio": r["audio"]}
                                   for k, r in enumerate(references)] + context
                else:
                    # 従来: 文脈ターンもテキストで一緒に再生成し、音声は捨てる。
                    call_idx = list(range(ctx_start, e))   # 文脈 + 対象
                    context = None
                tagged = "\n".join(lines[i] for i in call_idx)
                wavs, codes = prefetched[s] if s in prefetched else _call(tagged, context, call_idx)
                if len(wavs) != len(call_idx):
                    logger.warning(f"[{record['id']}] マルチターン窓[{s}:{e}]: セグメント数"
                                   f"({len(wavs)})≠ターン数({len(call_idx)})→ per-turn にフォールバック。")
                    return None
                for local, gi in enumerate(call_idx):
                    if gi >= s:                        # 対象ターンのみ採用（文脈ターンの再生成は捨てる）
                        wav_by_turn[gi] = wavs[local]
                        if codes[local] is not None:
                            codes_by_turn[gi] = codes[local]
            logger.info(f"[{record['id']}] マルチターン合成(窓W={W}, 文脈C={C}"
                        f"{', 文脈=採用済み音声' if use_ctx_audio else ''}"
                        f"{f', 窓ごとに{n_takes}テイク{_takes_temp_note(best_cfg, n_takes)}' if n_takes > 1 else ''}): "
                        f"{n}ターンを{len(starts)}窓で生成"
                        f"{_fmt_temp_hist(synth.take_temp_hist or {})}。")
        else:
            tagged = "\n".join(lines)
            wavs, _codes = _call(tagged, None, list(range(n)))
            if len(wavs) != n:
                logger.warning(f"[{record['id']}] マルチターン: セグメント数({len(wavs)})≠ターン数"
                               f"({n})→ per-turn にフォールバック。")
                return None
            wav_by_turn = {i: wavs[i] for i in range(n)}
            logger.info(f"[{record['id']}] マルチターン合成: {n}ターンを1コールで生成。")
    except Exception as e:
        logger.warning(f"[{record['id']}] マルチターン合成失敗→ per-turn にフォールバック: "
                       f"{type(e).__name__}: {e}")
        return None

    return wav_by_turn, set()


def synthesize_turns_per_speaker(synth, record: dict, voices: list[dict], sr: int,
                                 emotion_tags: bool = False,
                                 fish_tags_map: dict[int, str] | None = None,
                                 prefix_tag: str | None = None):
    """fish-speech 話者ごと独立合成: 話者ごとに generate_long パスを分けて生成する。

    マルチターン（全話者を1ストリームで生成）と違い、話者ごとに別パスを回す。各パスは
    その話者について完全に閉じており、相手の情報は本文にも参照音声にも一切入れない:

      - 本文（text）: その話者自身のターンだけを出現順に並べる（相手の発話は含めない）。
      - 参照音声（references / system プロンプト）: その話者のクローン参照1つだけを渡す。
        fish-speech の generate_long は references を全話者ぶん連結して system プロンプト
        前置きにし、生成する全ターンをそれで条件付けする。相手の参照を渡すと相手の声色が
        毎ターンの条件に混ざるため、per_speaker では自分の参照だけにする。
      - タグ: 参照が1つなので本文・参照とも <|speaker:0|> に統一する（番号を揃えないと
        参照と生成ターンが対応しない）。

    これで各話者の音声が「自分の声・自分の発話履歴」だけで決まり、相手の声色・韻律の影響を
    受けない。window/window_context は「その話者の自ターン列」に対して適用する（直前の自
    ターンを最大 C 個だけ文脈の種として前置きし、音声は捨てる）。各 synth_dialogue コールは
    独立（サーバ側の文脈引き継ぎはコール内のみ）。

    合成コストは概ねマルチターン相当（全ターンを1回ずつ＋窓文脈の再生成ぶん）。要件・
    戻り値・フォールバック挙動は synthesize_turns_multiturn と同じ（要件未達・セグメント数
    不一致・未生成ターン有りは None を返して per-turn 合成へ安全に劣化）。
    """
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, speaker_order, references, spk_index, tts_texts, _lines = inp
    n = len(turns)
    W = int(getattr(synth, "window", 0) or 0)
    C = max(0, int(getattr(synth, "window_context", 0) or 0))

    use_ctx_audio = bool(getattr(synth, "context_audio", False))

    wav_by_turn: dict[int, np.ndarray] = {}
    codes_by_turn: dict[int, object] = {}   # 採用ターンのVQトークン（無劣化の文脈連鎖用）
    calls = 0
    try:
        for target_sp in speaker_order:
            # この話者の自ターンだけを出現順に抽出（own[p] = p番目の自ターンのグローバル index）。
            own = [i for i in range(n) if turns[i]["speaker"] == target_sp]
            m = len(own)
            if m == 0:
                continue
            # 参照は自分のぶん1つだけ。参照が1つなのでタグは <|speaker:0|> に統一。
            sp_ref = [references[spk_index[target_sp]]]
            own_lines = [f"<|speaker:0|>{tts_texts[i]}" for i in own]
            if W > 0 and m > W:
                for s in range(0, m, W):                    # s,e は自ターン列上の位置
                    e = min(s + W, m)
                    ctx_start = max(0, s - C)               # 直前の自ターンを最大 C 個、文脈種に
                    if use_ctx_audio:
                        # 文脈=採用済みの自ターン（VQトークン優先）。text は対象のみ。
                        # context_fixed: 直前ではなく最初の窓のテイク（固定・連鎖なし）。
                        call_pos = list(range(s, e))
                        if getattr(synth, "context_fixed", False):
                            ctx_range = range(0, min(s, W))
                        else:
                            ctx_range = range(ctx_start, s)
                        context = _adopted_context(
                            [(own_lines[p], wav_by_turn.get(own[p]),
                              codes_by_turn.get(own[p]))
                             for p in ctx_range],
                            synth.sample_rate)
                        if getattr(synth, "context_anchor", False):
                            # 参照音声をフェイク直前ターンとしても前置き（ドリフト引き戻し）
                            context = [{"text": f"<|speaker:0|>{sp_ref[0]['text']}",
                                        "audio": sp_ref[0]["audio"]}] + context
                    else:
                        # 従来: 文脈の自ターンもテキストで一緒に再生成し、音声は捨てる。
                        call_pos = list(range(ctx_start, e))    # 文脈(自) + 対象(自)
                        context = None
                    tagged = "\n".join(own_lines[p] for p in call_pos)
                    wavs, codes = synth.synth_dialogue(tagged, sp_ref, context=context,
                                                       return_codes=True)
                    calls += 1
                    if len(wavs) != len(call_pos):
                        logger.warning(f"[{record['id']}] 話者別合成 自ターン窓[{s}:{e}]/話者「{target_sp}」: "
                                       f"セグメント数({len(wavs)})≠ターン数({len(call_pos)})→ per-turn にフォールバック。")
                        return None
                    for local, p in enumerate(call_pos):
                        if p >= s:                          # 対象の自ターンのみ採用（文脈ターンは破棄）
                            wav_by_turn[own[p]] = wavs[local]
                            if codes[local] is not None:
                                codes_by_turn[own[p]] = codes[local]
            else:
                tagged = "\n".join(own_lines)
                wavs = synth.synth_dialogue(tagged, sp_ref)
                calls += 1
                if len(wavs) != m:
                    logger.warning(f"[{record['id']}] 話者別合成/話者「{target_sp}」: "
                                   f"セグメント数({len(wavs)})≠自ターン数({m})→ per-turn にフォールバック。")
                    return None
                for p, i in enumerate(own):
                    wav_by_turn[i] = wavs[p]
    except Exception as e:
        logger.warning(f"[{record['id']}] 話者別合成失敗→ per-turn にフォールバック: "
                       f"{type(e).__name__}: {e}")
        return None

    # 全ターンが採用されたか検証（未カバーがあれば安全側で per-turn へ）
    if len(wav_by_turn) != n:
        missing = [i for i in range(n) if i not in wav_by_turn]
        logger.warning(f"[{record['id']}] 話者別合成: 未生成ターン{len(missing)}件"
                       f"（例 {missing[:5]}）→ per-turn にフォールバック。")
        return None
    logger.info(f"[{record['id']}] 話者別合成(話者{len(speaker_order)}人・窓W={W},文脈C={C}"
                f"{', 文脈=採用済み音声' if use_ctx_audio else ''}): "
                f"{n}ターンを話者ごと独立パス（計{calls}コール）で生成。")
    return wav_by_turn, set()


def synthesize_turns_self_context(synth, record: dict, voices: list[dict], sr: int,
                                  emotion_tags: bool = False,
                                  fish_tags_map: dict[int, str] | None = None,
                                  prefix_tag: str | None = None):
    """pocket の文脈つき合成（tts.fish_speech.pocket_context.self_turns・2026-11-15）: 話者ごとに自分の発話を順に作り、
    自分の直前の採用済み発話（最大 self_turns 個・相槌も含む）の音声と本文を文脈にして次の発話を作る（MossTtsSynth.ctx_turn。
    テイク選択もその中）。相手の発話は入れない（pocket は 1 人の声の続きしか学習していない）。2 人の列は並行に回す。
    Returns: (wav_by_turn, failed_turns) / 入力が揃わなければ None（呼び出し側で従来の合成へ）。"""
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, speaker_order, references, spk_index, tts_texts, _lines = inp
    K = int(getattr(synth, "ctx_self_turns", 0) or 0)
    wav_by_turn: dict[int, np.ndarray] = {}
    failed: set[int] = set()
    # ★2026-11-19 1 ターンでも無音代替になったら、この対話は base の書き出しで捨てられる（no-silent-degraded-output）。
    #   残りのターンを作っても無駄なので、両方の列をそこで止め、残りは無音代替（failed）にして返す。
    #   死んだサーバ相手だと 1 ターン 300 秒×2 を全ターン待ち、36 ターンの対話に 3 時間かけて捨てていた（ft_v2 002）。
    abort = threading.Event()

    def _fill_silence(i):
        est = float(turns[i].get("duration", 0.0) or 0.5)
        wav_by_turn[i] = np.zeros(int(est * synth.sample_rate), dtype=np.float32)
        failed.add(i)

    def chain(sp):
        own = [i for i in range(len(turns)) if turns[i]["speaker"] == sp]
        ref = references[spk_index[sp]]
        others = [references[spk_index[o]] for o in speaker_order if o != sp]
        done: list[int] = []
        for i in own:
            if abort.is_set():
                _fill_silence(i)
                continue
            ctx = [(wav_by_turn[j], tts_texts[j]) for j in done[-K:]] if K > 0 else []
            try:
                w = synth.ctx_turn(tts_texts[i], ref, ctx, others=others)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{record['id']}] 文脈つき合成 turn{i} 失敗（{type(e).__name__}: {e}）→ 文脈なし")
                w = None
                if not isinstance(e, TtsServerDead) and not abort.is_set():
                    try:
                        w = synth.ctx_turn(tts_texts[i], ref, [])
                    except Exception:  # noqa: BLE001
                        w = None
            if w is None or not len(w):
                _fill_silence(i)
                if not abort.is_set():
                    abort.set()
                    logger.error(f"[{record['id']}] turn{i} が合成できない → この対話は捨てるので残りのターンは作らない")
                continue
            wav_by_turn[i] = w
            done.append(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(speaker_order))) as ex:
        list(ex.map(chain, speaker_order))
    gs = getattr(synth, "guard_stats", None)
    logger.info(f"[{record['id']}] pocket 文脈つき合成: 自分の直前 {K} 発話を文脈に {len(turns)} ターン"
                f"{'（参照の抜粋も文脈）' if getattr(synth, 'ctx_prompt', None) else ''}"
                + (f"｜声の確認（ワーカー累計）: {gs}" if gs and getattr(synth, 'ctx_spk_guard', None) is not None else ""))
    return wav_by_turn, failed


def _targets_self_context(synth, record: dict, voices: list[dict], target_idx, emotion_tags, fish_tags_map,
                          prefix_tag, adopted_wavs, adopted_sr):
    """refine の入れ直し（挿入相槌など）を、その話者の直前の採用済み発話（最大 self_turns 個）を文脈にして作る（2026-11-15）。"""
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, _so, references, spk_index, tts_texts, _lines = inp
    K = int(getattr(synth, "ctx_self_turns", 0) or 0)
    pool = {i: w for i, w in (adopted_wavs or {}).items() if w is not None and len(w) > 0}
    if adopted_sr and int(adopted_sr) != int(synth.sample_rate):
        return None
    out: dict[int, np.ndarray] = {}
    try:
        for ti in sorted({int(i) for i in target_idx}):
            if not (0 <= ti < len(turns)):
                continue
            sp = turns[ti]["speaker"]
            prev = [j for j in range(ti) if turns[j]["speaker"] == sp and j in pool][-K:]
            others = [references[spk_index[o]] for o in _so if o != sp]
            w = synth.ctx_turn(tts_texts[ti], references[spk_index[sp]], [(pool[j], tts_texts[j]) for j in prev],
                               others=others)
            if w is None or not len(w):
                return None
            out[ti] = w
            pool[ti] = w
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{record['id']}] refine 文脈つき合成失敗→ 文脈なし再合成: {type(e).__name__}: {e}")
        return None
    gs = getattr(synth, "guard_stats", None)
    logger.info(f"[{record['id']}] refine 文脈つき合成（pocket・自分の直前 {K} 発話）: {len(out)} ターン"
                + (f"｜声の確認（ワーカー累計）: {gs}" if gs and getattr(synth, 'ctx_spk_guard', None) is not None else ""))
    return out


def _targets_bc_prev_ref(synth, record: dict, voices: list[dict], target_idx, emotion_tags, fish_tags_map,
                         prefix_tag, adopted_wavs, adopted_sr):
    """refine の入れ直しのうち相槌を、その話者の直前の採用済み発話（相槌も含む）を参照の先頭に置いて Irodori で作る
    （irodori.bc_prev_ref・IrodoriVllmSynth.bc_with_prev・2026-11-19）。相槌でない対象・直前の発話が無い相槌は従来どおり
    （声バンクだけの単発合成）。作った相槌も後の対象の「直前の発話」に使う。失敗したら None（呼び出し側が文脈なしで作り直す）。"""
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, _so, references, spk_index, tts_texts, _lines = inp
    if adopted_sr and int(adopted_sr) != int(synth.OUT_SR):
        logger.warning(f"[{record['id']}] 相槌の直前発話参照: 採用済み音声の sr {adopted_sr} が合成器 {synth.OUT_SR} と違う → 従来の合成")
        return None
    pool = {i: w for i, w in (adopted_wavs or {}).items() if w is not None and len(w) > 0}
    out: dict[int, np.ndarray] = {}
    # ★2026-11-20 対象を 1 つずつ順に作るのをやめ、「直前の発話」の依存の段（wave）ごとにまとめて並行に作る。
    #   各対象の入力（本文・参照・直前の発話＝どのターンの音声か）は 1 つずつ作るときと同じになるように、先に決めておく:
    #   直前の発話＝自分より前の同じ話者のターンのうち、採用済み（pool）か、この関数で作る対象（自分より前なので順に作れば
    #   出来ている）の最後のもの。それが対象なら、その対象の段＋1 の段で作る（対象でなければ段 0）。相槌でない対象は直前の
    #   発話を使わないので段 0。本番（004）で相槌 8.1 個/対話に対し依存の段は平均 1.48（_scratch/irodori_speed/audit_client/sim_prev.py）。
    #   1 個ずつだと 24 本の合成と採点（pyin・WavLM）を対象の数だけ順に待ち、PassB が対話の時間の 54% を占めていた。
    #   段の中で並行に作る数は irodori.bc_prev_ref.parallel（既定 4・1 で従来どおり 1 個ずつ）。
    tgt = [ti for ti in sorted({int(i) for i in target_idx}) if 0 <= ti < len(turns)]
    tset = set(tgt)
    prev_of: dict[int, int | None] = {}
    level: dict[int, int] = {}
    for ti in tgt:
        sp = turns[ti]["speaker"]
        pv = [j for j in range(ti) if turns[j]["speaker"] == sp and (j in pool or j in tset)][-1:]
        is_bc = turns[ti].get("type", "speech") == "backchannel"
        prev_of[ti] = pv[0] if (pv and is_bc) else None
        level[ti] = (level[pv[0]] + 1) if (pv and is_bc and pv[0] in tset) else 0

    def _one(ti):
        sp = turns[ti]["speaker"]
        ref = references[spk_index[sp]]
        if prev_of[ti] is not None:
            return synth.bc_with_prev(tts_texts[ti], ref, pool[prev_of[ti]])
        return synth._post_speech(tts_texts[ti], [ref])

    par = max(1, int((getattr(synth, "bc_prev", None) or {}).get("parallel", 4) or 1))
    try:
        for lv in sorted(set(level.values())):
            wave = [ti for ti in tgt if level[ti] == lv]
            if par == 1 or len(wave) == 1:
                ws = [_one(ti) for ti in wave]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(par, len(wave))) as ex:
                    ws = list(ex.map(_one, wave))
            for ti, w in zip(wave, ws):
                if w is None or not len(w):
                    return None
                out[ti] = w
                pool[ti] = w
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{record['id']}] 相槌の直前発話参照の合成に失敗 → 文脈なしで作り直す: {type(e).__name__}: {e}")
        return None
    n_prev = sum(1 for ti in out if prev_of.get(ti) is not None)
    logger.info(f"[{record['id']}] refine 相槌（参照の先頭に自分の直前の発話）: {n_prev} / {len(out)} ターン")
    return out


def synthesize_targets_with_context(synth, record: dict, voices: list[dict], sr: int,
                                    target_idx, context_turns: int,
                                    emotion_tags: bool = False,
                                    fish_tags_map: dict[int, str] | None = None,
                                    prefix_tag: str | None = None,
                                    adopted_wavs: dict[int, np.ndarray] | None = None,
                                    adopted_sr: int | None = None):
    """fish-speech: 指定ターンだけを「直前 C ターン（両話者）」を文脈に前置きして再合成する。

    refine の Pass B 用。挿入相槌・書き換えターンなど（target_idx）は相手の発話への反応な
    ので、per_speaker（自分だけ）とは逆に、対象ターンの直前 C ターンをそのまま（相手＋
    自分）タグ付きで前置きし、generate_long にその流れを聞かせた上で対象ターンだけ採用する。
    references は全話者ぶんを渡す（文脈に相手ターンが含まれ、その声で正しく条件付けるため）。

    対象ターンごとに1コール（文脈 C ＋ 対象1）を投げ、末尾セグメント（＝対象ターン）を採る。
    C は refine では tts.fish_speech.window_context を流用する（呼び出し側が渡す）。

    adopted_wavs（任意・context_audio 時のみ使用）: {turn_index: 採用済みwav}。文脈ターンの
    うち採用済み音声があるもの（先頭連続部分）はテキスト再生成の代わりに音声そのものを
    context として渡す（＝出力に実際に載る音声への条件付け）。無いターンだけ従来どおり
    テキストで前置き再生成する。再合成した対象ターンも後続対象の文脈に使う。

    Returns: {turn_index: wav}（全対象ぶん）。fish 以外・参照音声不足・セグメント数不一致・
    対象波形が空のときは None を返し、呼び出し側で従来の（文脈なし）再合成へフォールバック。
    """
    if getattr(synth, "bc_prev_ref", False):                     # Irodori: 相槌の参照の先頭に自分の直前の発話（2026-11-19）
        return _targets_bc_prev_ref(synth, record, voices, target_idx, emotion_tags, fish_tags_map,
                                    prefix_tag, adopted_wavs, adopted_sr)
    if int(getattr(synth, "ctx_self_turns", 0) or 0) > 0:       # pocket の文脈つき（自分の直前の発話だけ）
        return _targets_self_context(synth, record, voices, target_idx, emotion_tags, fish_tags_map,
                                     prefix_tag, adopted_wavs, adopted_sr)
    if not hasattr(synth, "synth_dialogue") or not getattr(synth, "supports_multiturn", True):
        return None                      # fish（synth_dialogue 持ち）以外・multiturn 非対応は非対応
    inp = _fish_multiturn_inputs(record, voices, emotion_tags, fish_tags_map, prefix_tag)
    if inp is None:
        return None
    turns, speaker_order, references, _spk_index, _tts_texts, lines = inp
    n = len(turns)
    C = max(0, int(context_turns or 0))
    use_ctx_audio = bool(getattr(synth, "context_audio", False)) and adopted_wavs
    # 文脈に使える採用済み音声のプール {idx: (wav, sr)}。再合成結果も追記していく。
    pool: dict[int, tuple[np.ndarray, int]] = {
        i: (w, int(adopted_sr or synth.sample_rate))
        for i, w in (adopted_wavs or {}).items() if w is not None and len(w) > 0}

    out: dict[int, np.ndarray] = {}
    n_ctx_audio = 0
    # vLLM-Omni（parallel_windows）: 文脈がテキストだけなら対象どうしは独立なので、先に並列で投げる。
    prefetched: dict[int, list] = {}
    targets = [ti for ti in sorted({int(i) for i in target_idx}) if 0 <= ti < n]
    if (getattr(synth, "parallel_windows", False) and not use_ctx_audio
            and int(getattr(synth, "concurrency", 1) or 1) > 1 and len(targets) > 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(int(synth.concurrency), len(targets))) as ex:
                futs = {ti: ex.submit(synth.synth_dialogue,
                                      "\n".join(lines[i] for i in list(range(max(0, ti - C), ti)) + [ti]),
                                      references) for ti in targets}
                prefetched = {ti: fu.result() for ti, fu in futs.items()}
        except Exception as e:
            logger.warning(f"[{record['id']}] refine文脈合成失敗→ 文脈なし再合成にフォールバック: "
                           f"{type(e).__name__}: {e}")
            return None
    try:
        for ti in sorted({int(i) for i in target_idx}):
            if not (0 <= ti < n):
                continue
            ctx_start = max(0, ti - C)
            ctx_idx = list(range(ctx_start, ti))
            context = None
            if use_ctx_audio:
                # 採用済み音声がある先頭連続部分を audio 文脈に、残りはテキスト前置きのまま。
                k = 0
                while k < len(ctx_idx) and ctx_idx[k] in pool:
                    k += 1
                if k > 0:
                    context = [{"audio": FishSpeechSynth._encode_wav(*pool[j]),
                                "text": lines[j]} for j in ctx_idx[:k]]
                    n_ctx_audio += k
                    ctx_idx = ctx_idx[k:]
            call_idx = ctx_idx + [ti]                    # テキスト前置き文脈 ＋ 対象（末尾）
            tagged = "\n".join(lines[i] for i in call_idx)
            wavs = (prefetched[ti] if ti in prefetched
                    else synth.synth_dialogue(tagged, references, context=context))
            if len(wavs) != len(call_idx):
                logger.warning(f"[{record['id']}] refine文脈合成 ターン{ti}: セグメント数"
                               f"({len(wavs)})≠文脈+対象({len(call_idx)})→ 文脈なし再合成にフォールバック。")
                return None
            w = wavs[-1]                                 # 対象ターン
            if w is None or len(w) == 0:
                logger.warning(f"[{record['id']}] refine文脈合成 ターン{ti}: 対象波形が空"
                               "→ 文脈なし再合成にフォールバック。")
                return None
            out[ti] = w
            pool[ti] = (w, synth.sample_rate)            # 後続対象の文脈に使う
    except Exception as e:
        logger.warning(f"[{record['id']}] refine文脈合成失敗→ 文脈なし再合成にフォールバック: "
                       f"{type(e).__name__}: {e}")
        return None

    logger.info(f"[{record['id']}] refine文脈合成: {len(out)}ターンを直前{C}ターン"
                "（相手＋自分）文脈つきで再合成"
                + (f"（うち採用済み音声文脈 {n_ctx_audio} ターン）" if n_ctx_audio else "")
                + "。")
    return out


def save_turn_cache(cache_dir: Path, dialogue_id: str,
                    wav_by_turn: dict[int, np.ndarray],
                    failed_turns: set[int], sr: int):
    """発話単位wavをキャッシュに保存する（--recompose での再合成なし再配置用）。"""
    d = Path(cache_dir) / dialogue_id
    d.mkdir(parents=True, exist_ok=True)
    for i, w in wav_by_turn.items():
        sf.write(str(d / f"{i:04d}.wav"), w, sr)
    meta = {"sr": sr, "n_turns": len(wav_by_turn), "failed": sorted(failed_turns)}
    (d / "meta.json").write_text(json.dumps(meta))


def load_turn_cache(cache_dir: Path, dialogue_id: str, expected_n: int):
    """キャッシュから発話単位wavを読む。不整合（ターン数不一致・欠損）は None。

    Returns: (wav_by_turn, failed_turns, sr) または None
    """
    d = Path(cache_dir) / dialogue_id
    meta_p = d / "meta.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return None
    if int(meta.get("n_turns", -1)) < expected_n:  # --seconds 切り詰め時は先頭だけ使う
        return None
    wav_by_turn = {}
    for i in range(expected_n):
        p = d / f"{i:04d}.wav"
        if not p.exists():
            return None
        w, _sr = sf.read(str(p), dtype="float32")
        wav_by_turn[i] = w
    failed = {i for i in meta.get("failed", []) if i < expected_n}
    return wav_by_turn, failed, int(meta["sr"])


def synthesize_dialogue_turns(synth, record: dict, voices: list[dict], sr: int,
                              short_threshold: int = 4,
                              turn_cache_dir: Path | None = None, speed: float = 1.0,
                              context_prosody: bool = False, context_short_len: int = 6,
                              emotion_tags: bool = False,
                              fish_tags_map: dict[int, str] | None = None,
                              prefix_tag: str | None = None):
    """1対話ぶんの発話単位wavを合成する（配置はしない）。

    テイク選択・サー再合成・末尾撮り直し・発話wavキャッシュの保存まで含む。
    配置（compose_dialogue）と分けてあるのは、融合モード（--retime-refine）が
    配置前の wav_by_turn を必要とするため。

    Returns: (wav_by_turn: dict[int, np.ndarray], failed_turns: set[int]) または None
    """
    # fish-speech の文脈考慮合成（有効時）。要件を満たさない場合は None が返るので、
    # そのまま per-turn 合成（文脈なし・話者ごと個別 /v1/tts）へフォールバックする。
    #   - per_speaker: 話者ごとに独立パス。各パスは自分の発話だけで条件付け（相手発話は
    #                  文脈に入れない）。相手の声色/韻律の影響を受けない。
    #   - multiturn:   全話者を1ストリームで生成し文脈（相手発話含む）を引き継ぐ。
    # per_speaker を優先（両方 true でも per_speaker）。
    r = None
    ctx_path = False                       # 文脈つき経路（窓ごとにテイク選択済み）か
    if int(getattr(synth, "ctx_self_turns", 0) or 0) > 0:
        # pocket の文脈つき（自分の直前の発話）。テイク選択は発話ごとに済んでいる
        r = synthesize_turns_self_context(synth, record, voices, sr, emotion_tags=emotion_tags,
                                          fish_tags_map=fish_tags_map, prefix_tag=prefix_tag)
        ctx_path = r is not None
    elif getattr(synth, "per_speaker", False):
        r = synthesize_turns_per_speaker(synth, record, voices, sr,
                                         emotion_tags=emotion_tags,
                                         fish_tags_map=fish_tags_map,
                                         prefix_tag=prefix_tag)
    elif getattr(synth, "multiturn", False):
        r = synthesize_turns_multiturn(synth, record, voices, sr,
                                       emotion_tags=emotion_tags,
                                       fish_tags_map=fish_tags_map,
                                       prefix_tag=prefix_tag)
        ctx_path = r is not None
    if r is None:
        r = synthesize_turns(synth, record, voices, sr, prefix_tag=prefix_tag,
                             short_threshold=short_threshold,
                             speed=speed, context_prosody=context_prosody,
                             context_short_len=context_short_len, emotion_tags=emotion_tags,
                             fish_tags_map=fish_tags_map)
    if r is None:
        return None
    wav_by_turn, failed_turns = r
    if failed_turns:
        # ★2026-11-19 無音代替が出た対話はこの後で捨てる（書き出さない）。テイク選択・サー/声/末尾/読み違いの引き直しは
        #   failed_turns を直さないので、サーバに投げるだけ無駄（死んだサーバ相手だと 1 回 300 秒待つ）。キャッシュも書かない。
        logger.warning(f"[{record['id']}] 無音代替 {len(failed_turns)} 発話 → 引き直しは飛ばす（この対話は捨てる）")
        return wav_by_turn, failed_turns
    # テイク選択: 同じターンを複数回引き、話速・抑揚・音量が中庸なものを採る
    # （信号加工なしでばらつきだけ削る。fish のクローン参照合成でのみ作動）。
    # マルチターン経路は窓ごとに選択済みなので、ここでは引き直さない（単発合成の
    # テイクを混ぜると、その窓の連続性が壊れて窓分割の意味が無くなる）。
    best_cfg = getattr(synth, "best_of", None)
    if best_cfg and best_cfg.get("enabled", False) and not ctx_path:
        select_best_takes(synth, record, voices, wav_by_turn, sr, best_cfg,
                          emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
                          prefix_tag=prefix_tag)
    # サー（先頭フィラー直後の広帯域ヒス）が乗ったターンを検知し、temperature を上げて
    # 再合成し差し替える（発話は無傷。fish のクローン参照合成でのみ作動）。
    regen_cfg = getattr(synth, "regen_hiss", None)
    if regen_cfg and regen_cfg.get("enabled", True):
        regenerate_hiss_turns(synth, record, voices, wav_by_turn, sr, regen_cfg,
                              emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
                              prefix_tag=prefix_tag)
    # 声が相手に入れ替わったターンを x-vector で検知し、本人の参照だけで引き直す
    # （2話者まとめての生成で起きる。読み違いの検査より前に置き、差し替えたテイクも聞き取らせる）。
    spk_cfg = getattr(synth, "regen_speaker", None)
    if spk_cfg and spk_cfg.get("enabled", True):
        regenerate_speaker_leak_turns(synth, record, voices, wav_by_turn, sr, spk_cfg,
                                      emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
                                      prefix_tag=prefix_tag)
    # 末尾に「入力していない1音」が付いたターンを検知し、seed だけ替えて引き直す
    # （温度は動かさない＝A/Bで発生率が変わらない。波形も切らない）。サー再合成の
    # あとに置き、引き直したテイクがサーを持ち込まないよう hiss_sec も渡す。
    tail_cfg = getattr(synth, "regen_tail", None)
    if tail_cfg and tail_cfg.get("enabled", True):
        hs = None
        if regen_cfg and regen_cfg.get("enabled", True):
            hs = float(regen_cfg.get("hiss_sec", 0.6))
        regenerate_tail_turns(synth, record, voices, wav_by_turn, sr, tail_cfg,
                              emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
                              prefix_tag=prefix_tag, hiss_sec=hs)
    # 読み違い（「表書き→ひょうがき」等）を ASR で検知し、seed を替えて引き直す。最後に置く
    # （他の引き直しで差し替わったテイクも含めて聞き取るため）。
    misread_cfg = getattr(synth, "regen_misread", None)
    if misread_cfg and misread_cfg.get("enabled", True):
        regenerate_misread_turns(synth, record, voices, wav_by_turn, sr, misread_cfg,
                                 emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
                                 prefix_tag=prefix_tag)
    if turn_cache_dir is not None:
        # 無音代替が出た対話はこの後 drop する。無音入りのキャッシュを残すと refine や
        # --recompose がそれを拾って（合成し直さずに）無音のまま音声を作ってしまうので、
        # キャッシュも書かない（既存の正常キャッシュも上書きしない）。
        if failed_turns:
            logger.warning(f"[{record['id']}] 無音代替あり → 発話wavキャッシュは保存しない")
        else:
            save_turn_cache(turn_cache_dir, record["id"], wav_by_turn, failed_turns, sr)
    return wav_by_turn, failed_turns


def synthesize_dialogue(synth, record: dict, voices: list[dict], sr: int,
                        short_threshold: int = 4, retimer=None,
                        turn_cache_dir: Path | None = None, speed: float = 1.0,
                        context_prosody: bool = False, context_short_len: int = 6,
                        trim: dict | None = None, emotion_tags: bool = False,
                        fish_tags_map: dict[int, str] | None = None,
                        prefix_tag: str | None = None,
                        noise: dict | None = None, loudness: dict | None = None,
                        rate: dict | None = None, lead_in: dict | None = None,
                        tail_pad: dict | None = None):
    """1対話をライブ合成し、話者数ぶんのチャネルに配置する。

    Returns:
        (stereo: np.ndarray[shape=(N,2)], manifest_turns: list[dict]) または None
    """
    r = synthesize_dialogue_turns(
        synth, record, voices, sr, short_threshold=short_threshold,
        turn_cache_dir=turn_cache_dir, speed=speed,
        context_prosody=context_prosody, context_short_len=context_short_len,
        emotion_tags=emotion_tags, fish_tags_map=fish_tags_map,
        prefix_tag=prefix_tag)
    if r is None:
        return None
    wav_by_turn, failed_turns = r
    return compose_dialogue(record, voices, sr, wav_by_turn, failed_turns,
                            retimer=retimer, trim=trim, noise=noise,
                            loudness=loudness, rate=rate, lead_in=lead_in,
                            tail_pad=tail_pad)


# --------------------------------------------------------------------------
# メインループ
# --------------------------------------------------------------------------

def build_fish_tagger(config: dict, engine: str):
    """fish-speech の感情タグ付け設定を解決する（run / 融合モード共用）。

    発話に S2 の感情/デリバリ/間タグを付けて表情豊かにする（config 既定ON）。
      tag_engine="llm"  → LLMが各発話を読み、詳細なインラインタグを付与（推奨）
      tag_engine="rule" → 従来の軽量ヒューリスティック（fish_emotion_prefix）
    いずれもタグは合成入力にのみ載り、マニフェスト/transcript は原文のまま（非漏洩）。

    Returns: (fish_tagger | None, fish_rule_tags: bool, prefix_tag: str)
    """
    fcfg = (config.get("tts") or {}).get("fish_speech", {}) or {}
    fish_on = engine == "fish_speech" and bool(fcfg.get("emotion_tags", True))
    prefix_tag = str(fcfg.get("prefix_tag", "") or "").strip()  # 全発話冒頭の固定タグ
    if prefix_tag:
        logger.info(f"fish-speech: 全発話冒頭に固定タグ '{prefix_tag}' を付与します")
    tag_engine = str(fcfg.get("tag_engine", "llm")).lower()
    tag_density = str(fcfg.get("tag_density", "max")).lower()
    fish_tagger = None      # LLM方式のタガー（Noneならルール/無効）
    fish_rule_tags = False  # ルール方式（fish_emotion_prefix）を使うか
    if fish_on:
        if tag_engine == "llm":
            try:
                tag_retry_max = int(fcfg.get("tag_retry_max", DEFAULT_TAG_RETRY_MAX))
                fish_tagger = FishTagger(config, density=tag_density,
                                         retry_max=tag_retry_max)
                logger.info(f"fish-speech: LLM詳細タグ付け有効（density={tag_density}, "
                            f"リトライ上限={tag_retry_max}回・尽きたら対話をスキップ）")
            except Exception as e:
                logger.warning("fish-speech: LLMタガー構築失敗→ルールタグに降格: "
                               f"{type(e).__name__}: {e}")
                fish_rule_tags = True
        else:
            fish_rule_tags = True
            logger.info("fish-speech: ルール方式の感情タグ付け有効")
    return fish_tagger, fish_rule_tags, prefix_tag


# ★2026-11-22 出力のバリアント名からエンジン名の接頭辞を外した（ユーザ判断: TTS エンジンが
# 何種類もあり、Irodori / MOSS / pocket も engine: fish_speech の中で動くので名前が実態と合わない）。
#   旧 fish_speech            → plain     （リタイム前。発話 wav キャッシュ <audio>/plain/turns/）
#   旧 fish_speech_maai       → maai      （= retime.tag。1 段目）
#   旧 fish_speech_maai_ref   → maai_ref  （= retime.tag + "_" + refine.tag。最終成果物）
# engine（tts.engine: fish_speech）・設定の fish_speech: 節・audio_fish/ などのフォルダ名はそのまま。
# 既存データの改名は batch/rename_variant_dirs.py（--dry-run が既定・--undo で戻せる）。
PLAIN_VARIANT = "plain"
# 改名前の名前（rename_variant_dirs.py と progress_dashboard.py の読み替え用）。新しい名前 → 旧い名前
LEGACY_VARIANT_NAMES = {"plain": "fish_speech", "maai": "fish_speech_maai",
                        "maai_ref": "fish_speech_maai_ref"}


def base_variant_name(rt_tag: str) -> str:
    """リタイム済み（1 段目）のバリアント名。★2026-11-22 から retime.tag そのもの（例: maai）。"""
    return str(rt_tag)


def ref_variant_name(rt_tag: str, rf_tag: str) -> str:
    """refine 済み（最終）のバリアント名。★2026-11-22 から <retime.tag>_<refine.tag>（例: maai_ref）。"""
    return f"{base_variant_name(rt_tag)}_{rf_tag}"


def engine_output_paths(tts_cfg: dict, engine: str | None = None,
                        variant: str | None = None):
    """バリアントごとに分けた (audio_dir, manifest_file) を返す。

    variant はリタイム出力の分離用（例: "maai" / "maai_ref"）。指定時は variant 名で
    ディレクトリ/manifest を分け、既存出力と共存させる。指定が無ければリタイム前の "plain"。
    audio_dir    = <tts.audio_dir>/<name>/          （例: output/audio/plain/）
    manifest     = <stem>_<name><suffix>            （例: output/audio_manifest_plain.jsonl）
    ★2026-11-22 名前にエンジン名を入れない（旧: <engine> / <engine>_maai / <engine>_maai_ref）。
    engine 引数は呼び出し側との互換のために残すだけで、名前には使わない。
    """
    name = variant or PLAIN_VARIANT
    audio_dir = Path(tts_cfg["audio_dir"]) / name
    mp = Path(tts_cfg["manifest_file"])
    manifest_file = mp.with_name(f"{mp.stem}_{name}{mp.suffix}")
    return audio_dir, manifest_file


def transcripts_path(manifest_file: Path) -> Path:
    """manifest ファイルパスから統合書き起こし(JSONL)のパスを作る。

    例: output/audio_manifest_maai_ref.jsonl
        → output/transcripts_maai_ref.jsonl
    """
    mp = Path(manifest_file)
    name = mp.name.replace("audio_manifest", "transcripts", 1)
    if name == mp.name:  # "audio_manifest" を含まない場合のフォールバック
        name = "transcripts_" + mp.name
    return mp.with_name(name)


# ボイスプロンプトとして書き出すキー: voices.csv の列（参照音声クローン系）と
# エンジン固有の声の指定子（プリセット系）。voice の dict に有る分だけ載せる。
VOICE_PROMPT_KEYS = ("gender", "age_range", "audio_path", "text",
                     "voice_name", "speaker", "reference_id")


def collect_voice_prompts(voices: list[dict], manifest_turns: list) -> dict:
    """話者名 → 合成に使ったボイスプロンプトの情報 を作る。

    どの声で合成したかは音声からは復元できないので、manifest と書き起こしに残す。
    manifest_turns の voice_id をボイス一覧（voices_file の CSV / config のプリセット）と
    突き合わせ、参照音声のパスとプロンプト文、または音声名をそのまま載せる。
    一覧に無い id（別バンクで合成した古い manifest など）は voice_id だけを残す。
    """
    by_id = {v.get("id"): v for v in (voices or [])}
    prompts = {}
    for t in manifest_turns:
        sp, vid = t.get("speaker"), t.get("voice_id")
        if not sp or vid is None or sp in prompts:
            continue
        info = {"voice_id": vid}
        voice = by_id.get(vid) or {}
        for k in VOICE_PROMPT_KEYS:
            if voice.get(k) not in (None, ""):
                info[k] = voice[k]
        prompts[sp] = info
    return prompts


def add_segment_yomi(seg: dict, rec) -> None:
    """segment に読みを付ける: ruby（{c, b} そのまま）と yomi（[[表記, 読み, 確か], ...]）。"""
    import yomi_ruby as YR
    if rec and YR.valid(rec, seg.get("text", "")):
        seg["ruby"] = {k: rec[k] for k in ("c", "b") if rec.get(k)}
        seg["yomi"] = YR.yomi_pairs(rec)


def build_transcript(record: dict, manifest_turns: list, sr: int, channels: int,
                     duration_sec: float, channel_speakers: dict,
                     audio_name: str, voice_prompts: dict | None = None) -> dict:
    """音声対話モデル学習用の綺麗な書き起こし（発話ごとの時刻+テキスト）を作る。

    manifest_turns（index/speaker/channel/type/text/start_sec/end_sec を持つ）から
    最小限のフィールドだけ取り出し、segments を start 昇順（同時刻は channel 順）に並べる。
    voice_prompts（collect_voice_prompts の戻り）を渡すと、話者ごとに使った声の
    参照音声・プロンプト文も記録する。
    """
    segs = []
    rec_turns = (record.get("voice_dialogue") or {}).get("turns") or []
    for t in manifest_turns:
        seg = {
            "start": round(float(t["start_sec"]), 3),
            "end": round(float(t["end_sec"]), 3),
            "speaker": t["speaker"],
            "channel": int(t["channel"]),
            "type": t.get("type", "speech"),
            "text": t["text"],
        }
        # 発話全体の読み（yomi_ruby。2026-11-12）。アラインメント（jmoshi_ft/tools/align_mfa.py）が
        # 語の発音に使う。本文が読みを付けたときのままのものだけ（言いかけで切った発話などは付けない）。
        i = t.get("index")
        if isinstance(i, int) and 0 <= i < len(rec_turns):
            add_segment_yomi(seg, rec_turns[i].get("ruby"))
        segs.append(seg)
    segs.sort(key=lambda s: (s["start"], s["channel"]))
    tr = {
        "id": record["id"],
        "audio": audio_name,
        "sample_rate": int(sr),
        "channels": int(channels),
        "duration_sec": round(float(duration_sec), 3),
        "speaking_rate": record.get("speaking_rate"),
        "channel_speakers": channel_speakers,
    }
    if voice_prompts:  # 話者ごとの声（参照音声＋プロンプト文 / プリセット音声名）
        tr["voice_prompts"] = voice_prompts
    tr["segments"] = segs
    return tr


_DLG_EXT = ".wav"     # 対話単位の音声の拡張子（tts.audio_format・set_audio_format が決める・2026-11-17）


def set_audio_format(tts_cfg: dict | None) -> str:
    """対話単位の音声（audio/<variant>/<id>.*）を書く形式を config の tts.audio_format（wav / flac）から決める。
    読む側・出来ているかの判定は audio_io.has_audio でどちらの拡張子でもよい。"""
    global _DLG_EXT
    from audio_io import audio_ext
    _DLG_EXT = audio_ext(tts_cfg)
    return _DLG_EXT


def write_transcript(audio_dir: Path, transcripts_fh, record: dict,
                     manifest_turns: list, sr: int, channels: int,
                     duration_sec: float, channel_speakers: dict,
                     voice_prompts: dict | None = None):
    """対話ごとの <id>.json を音声の隣に書き、統合 JSONL に1行追記する。"""
    tr = build_transcript(record, manifest_turns, sr, channels, duration_sec,
                          channel_speakers, f"{record['id']}{_DLG_EXT}", voice_prompts)
    (Path(audio_dir) / f"{record['id']}.json").write_text(
        json.dumps(tr, ensure_ascii=False, indent=1))
    if transcripts_fh is not None:
        transcripts_fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
        transcripts_fh.flush()


def truncate_record_by_seconds(record: dict, max_seconds: float | None) -> dict:
    """テスト用: 推定タイムライン上、冒頭 max_seconds 以内に始まるターンだけを残した
    レコードのコピーを返す（プレフィックス）。合成するターン数を減らせる。
    max_seconds が None/0以下なら原本をそのまま返す。
    """
    if not max_seconds or max_seconds <= 0:
        return record
    turns = record["voice_dialogue"]["turns"]
    kept = []
    prev_end = 0.0
    for i, t in enumerate(turns):
        sa = t.get("start_abs")
        if sa is None:  # Stage3の絶対時刻が無ければ推定
            off = 0.0 if i == 0 else float(t.get("start_offset", 0.0) or 0.0)
            sa = max(0.0, prev_end + off)
        prev_end = sa + float(t.get("duration", 0.0) or 0.0)
        if sa < max_seconds:
            kept.append(t)
        else:
            break
    new = dict(record)
    new["voice_dialogue"] = dict(record["voice_dialogue"], turns=kept)
    return new


def select_gpu(gpu) -> str | None:
    """使用GPUを確定し、必要な device 文字列を返す。

    CUDA_VISIBLE_DEVICES を（torch インポート前に）設定して、TTSの全構成要素を
    指定した1枚のGPUに固定する。マスク後はそのGPUが cuda:0 になる。

      gpu = None        -> 既定（モデル側のデフォルトデバイス）
      gpu = 0 / "1"     -> そのGPU IDに固定し device="cuda:0"
      gpu = "cuda:2"    -> "2" として扱う
      gpu = "cpu"       -> CPU 実行
    """
    if gpu is None:
        return None
    g = str(gpu).strip()
    if g.lower() == "cpu":
        return "cpu"
    if g.lower().startswith("cuda:"):
        g = g.split(":", 1)[1]
    if not g.isdigit():
        raise ValueError(f"GPU 指定が不正です: {gpu!r}（例: 0, 1, cuda:1, cpu）")
    # 番号の数え方を nvidia-smi・start_servers.py と揃える（機種混在機で fish と別の GPU に載らないように）
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = g
    return "cuda:0"


def build_synth(tts_cfg: dict, engine: str, dry_run: bool,
                device: str | None = None):
    """エンジン設定から合成器を構築する（run / refine 共用）。

    Returns: (synth, sr, short_threshold)
    """
    sr = int(tts_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
    if dry_run:
        logger.info("dry-run モード: 正弦波プレースホルダで合成します（TTSモデルは使いません）。")
        synth = DummySynth(sample_rate=sr)
    elif engine == "gemini":
        synth = build_gemini_synth(tts_cfg)
        sr = synth.sample_rate
        logger.info(f"Gemini TTS を使用します（model={synth.model}, "
                    f"style='{synth.style_prompt[:30]}…'）")
    elif engine == "google_cloud":
        synth = build_google_cloud_synth(tts_cfg)
        sr = synth.sample_rate
        logger.info(f"Google Cloud TTS を使用します（lang={synth.language_code}, "
                    f"sr={sr}Hz, prompt={synth.prompt!r}）")
    elif engine == "irodori":
        logger.info("Irodori-TTS をロードします…")
        synth = build_irodori_synth(tts_cfg)
        sr = synth.sample_rate
        logger.info(f"Irodori-TTS を使用します（sr={sr}Hz, "
                    f"caption={synth.caption!r}）")
    elif engine == "qwen3":
        logger.info(f"Qwen3-TTS をロードします…（device={device or '既定'}, "
                    f"model={tts_cfg.get('qwen3', {}).get('checkpoint')}）")
        synth = build_qwen3_synth(tts_cfg, device)
        sr = synth.sample_rate
        logger.info(f"Qwen3-TTS を使用します（Apache 2.0, lang={synth.language}, "
                    f"sr={sr}Hz, instruct={synth.instruct!r}）")
    elif engine == "fish_speech":
        synth = build_fish_speech_synth(tts_cfg)
        sr = synth.sample_rate
        logger.info(f"fish-speech(HTTP) を使用します（url={synth.url}, "
                    f"format={synth.fmt}, sr={sr}Hz・初回で確定）"
                    "。要 api_server 起動（GPUマシン）")
    elif engine == "sarashina":
        logger.info(f"sarashina2.2-tts をロードします…（device={device or '既定'}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未設定')}）")
        gen_kwargs = build_gen_kwargs(tts_cfg)
        if gen_kwargs:
            logger.info(f"生成パラメータ: {gen_kwargs}")
        synth = SarashinaSynth(
            model_dir=tts_cfg.get("model_dir"),
            use_vllm=tts_cfg.get("use_vllm", False),
            device=device,
            gen_kwargs=gen_kwargs,
        )
        sr = synth.sample_rate
    else:
        logger.error(f"未知の tts.engine: {engine}"
                     "（gemini / google_cloud / qwen3 / fish_speech / irodori / sarashina）")
        sys.exit(1)
    # 短発話の分離は sarashina のクラッシュ回避用。クラウドTTS(gemini/google_cloud)は
    # 短文でも落ちないので、全発話を並列バッチに通してスループットを上げる（0=分離しない）。
    short_threshold = 0 if (dry_run or engine in ("gemini", "google_cloud")) \
        else int(tts_cfg.get("short_text_threshold", 4))
    return synth, sr, short_threshold


def apply_dialogue_speaking_rate(synth, record: dict, tts_cfg: dict):
    """対話ごとに推定された話速を合成器へ反映する（google_cloud のネイティブ speaking_rate）。

    Stage 3 が推定した record["speaking_rate"] を優先し、無ければ config の既定値。
    tts.google_cloud.speaking_rate_range（既定 [1.0, 1.2]）にクランプする。
    speaking_rate 属性を持たないエンジン（gemini/irodori/sarashina）では何もしない。
    """
    if synth is None or not hasattr(synth, "speaking_rate"):
        return
    gc = tts_cfg.get("google_cloud", {}) or {}
    rng = gc.get("speaking_rate_range") or [1.0, 1.2]
    rate = record.get("speaking_rate")
    if rate is None:
        rate = gc.get("speaking_rate")
    if rate is None:
        return
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return
    lo, hi = float(rng[0]), float(rng[1])
    synth.speaking_rate = round(min(max(rate, lo), hi), 3)


# ---------------------------------------------------------------------------
# 対話単位の動的クレーム（静的シャード分割の代替）
# ---------------------------------------------------------------------------

def worker_tag() -> str:
    """このプロセスを一意に指す短いタグ（part ファイル名・クレームの所有者記録用）。

    `w-<ホスト名>-<PID>`。ホスト名は FQDN の先頭ラベルだけを採り、英数字とハイフン
    以外を落とす。`_` を許すと `<stem>.part-w-host-123_discarded.jsonl`（退避ファイル）
    まで part とみなされてしまうため、区切りに使える文字を絞っている（_SHARD_PART_RE）。
    """
    host = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname().split(".")[0])
    return f"w-{host}-{os.getpid()}"


class DialogueClaims:
    """対話単位の排他クレーム。複数プロセス・複数マシン（NFS共有）で共用する。

    シャード分割（apply_shard）が「起動前に担当を静的に決める」のに対し、こちらは各
    プロセスが未処理の対話を先着順に取りに行く。ハッシュの偏りもマシンごとの速度差も
    末尾の遊びにならず、途中で台数を増減しても帳簿（NGLOBAL/OFFSET）が要らない。

    排他の実体は `<root>/<対話id>` ディレクトリの mkdir。mkdir は NFS でも原子的
    （O_CREAT|O_EXCL は NFS のバージョン依存があるので使わない）。**完了の判定は
    あくまで出力 wav の存在**であり、クレームは「誰かが着手した」印にすぎない。

    寿命管理: 保持中のクレームは heartbeat スレッドが定期的に mtime を更新する。
    プロセスが落ちれば更新が止まるので、ttl_sec を過ぎたクレームは他プロセスが
    引き取れる（引き取りは rename で原子的に行うので勝者は1つだけ）。これが無いと、
    落ちたプロセスの担当分が「クレームだけ残って永久に未処理」になる。

    逐次ループ1本から使う前提の薄い API。★close() は必ず通すこと（下記のように
    contextlib.closing で包む）。握ったまま落ちるとクレームの mtime が新しいまま
    残り、すぐ再実行してもその対話は ttl_sec が切れるまで飛ばされてしまう:
        claims = DialogueClaims(audio_dir / ".claims")
        with contextlib.closing(claims):    # Ctrl+C・例外でも未完了分を解放する
            for r in pending:
                if not claims.take(r["id"], audio_dir / f"{r['id']}.wav"):
                    continue                # 他プロセスが担当中 or 完了済み
                ...                         # 途中で continue しても次の take が解放する
                claims.finish(r["id"])      # 完了
    """

    def __init__(self, root: Path, ttl_sec: float = 1800.0,
                 heartbeat_sec: float = 60.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = float(ttl_sec)
        self.tag = worker_tag()
        self.taken = 0      # 実際に処理へ回した件数
        self.busy = 0       # 他プロセスが持っていて見送った件数
        self.done_elsewhere = 0  # 取得後に完了済みと分かった件数
        self.stolen = 0     # TTL切れで引き取った件数
        self._current: str | None = None
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat_loop,
                                    args=(float(heartbeat_sec),), daemon=True)
        self._hb.start()

    def _path(self, did: str) -> Path:
        return self.root / str(did)

    def _heartbeat_loop(self, interval: float):
        while not self._stop.wait(interval):
            with self._lock:
                held = list(self._held)
            for did in held:
                try:
                    os.utime(self._path(did), None)
                except OSError:
                    pass

    def _mkdir(self, did: str) -> bool:
        p = self._path(did)
        try:
            os.mkdir(p)
        except FileExistsError:
            return False
        except OSError as e:
            logger.warning(f"クレーム作成に失敗 {p}: {type(e).__name__}: {e}")
            return False
        with self._lock:
            self._held.add(did)
        try:  # 所有者の記録（デバッグ用。排他そのものは mkdir が担っている）
            (p / "owner.json").write_text(json.dumps(
                {"worker": self.tag, "started": time.time()}), encoding="utf-8")
        except OSError:
            pass
        return True

    def acquire(self, did: str) -> bool:
        """did の担当権を取れたら True。TTL切れのクレームは引き取る。"""
        did = str(did)
        if self._mkdir(did):
            return True
        p = self._path(did)
        try:
            age = time.time() - p.stat().st_mtime
        except OSError:
            return False           # 直前に解放された → 次の run で拾えばよい
        if age <= self.ttl_sec:
            return False
        # TTL切れ（保持プロセスが落ちたとみなす）→ 引き取る。
        # rename は原子的なので、同時に狙った複数プロセスのうち勝者は1つだけ。
        stale = self.root / f"{did}.stale-{self.tag}-{int(time.time())}"
        try:
            os.rename(p, stale)
        except OSError:
            return False           # 他プロセスが先に引き取った/解放した
        shutil.rmtree(stale, ignore_errors=True)
        if not self._mkdir(did):
            return False
        self.stolen += 1
        logger.warning(f"[{did}] {age / 60:.0f}分更新の無いクレームを引き取りました"
                       "（保持プロセスが落ちたとみなします）")
        return True

    def take(self, did: str, done_marker: Path | None = None) -> bool:
        """担当権を取り、まだ処理が要るなら True。直前の未 finish は自動で解放する。

        done_marker（通常は出力 wav）が既にあれば False。取得の**後**に確認するのは、
        NFS の属性キャッシュで古い「wav 無し」を掴むことがあるのと、前回 run で完了した
        対話のクレームを TTL 切れで引き取ってしまう場合があるため。
        """
        did = str(did)
        if self._current is not None and self._current != did:
            self.release(self._current)
        self._current = None
        if not self.acquire(did):
            self.busy += 1
            return False
        if done_marker is not None and Path(done_marker).exists():
            self.finish(did)       # 別プロセスが完了済み。クレームは印として残す
            self.done_elsewhere += 1
            return False
        self._current = did
        self.taken += 1
        return True

    def finish(self, did: str):
        """完了。クレームは「着手済み」の印として残し、heartbeat の対象から外す。"""
        did = str(did)
        with self._lock:
            self._held.discard(did)
        if self._current == did:
            self._current = None

    def release(self, did: str):
        """未完了のまま手放す（失敗・スキップ時）。他プロセスが即座に再挑戦できる。"""
        did = str(did)
        with self._lock:
            self._held.discard(did)
        if self._current == did:
            self._current = None
        p = self._path(did)
        try:
            (p / "owner.json").unlink(missing_ok=True)
            p.rmdir()
        except OSError:
            pass

    def close(self):
        """ループ終了時。持ったままの未完了クレームを解放し heartbeat を止める。"""
        if self._current is not None:
            self.release(self._current)
        self._stop.set()

    def log_summary(self):
        logger.info(f"クレーム: 取得{self.taken}件 / 他プロセス担当{self.busy}件 / "
                    f"取得後に完了済み{self.done_elsewhere}件"
                    + (f" / TTL切れ引き取り{self.stolen}件" if self.stolen else "")
                    + f"（worker={self.tag}）")


def parse_shard(shard: str | None) -> tuple[int, int] | None:
    """"--shard I/N" を (i, n) に解析（i は 1..N）。未指定なら None。"""
    if not shard:
        return None
    try:
        i_str, n_str = str(shard).split("/")
        i, n = int(i_str), int(n_str)
    except (ValueError, AttributeError):
        raise ValueError(f"--shard は 'I/N' 形式で指定してください（例: 1/4）。受領: {shard!r}")
    if not (n >= 1 and 1 <= i <= n):
        raise ValueError(f"--shard の値が不正です（1<=I<=N, N>=1）。受領: {shard!r}")
    return (i, n)


def apply_shard(pending: list, shard: tuple[int, int] | None) -> list:
    """pending を N 個の排他シャードに分割し、担当分（hash(id) % N == i-1）だけ返す。

    複数GPU・複数 fish サーバ（さらに複数マシン）での並列処理用。各プロセスに別の I を
    割り当てると全対話を重複なく分担できる。分割キーは **対話 id の安定ハッシュ**であり、
    resume 状態（完了済みで pending から抜けた対話）や入力の並び・プロセスに一切依存しない。
    このため、マシンごとに起動時刻や完了集合が違っても、各 id は常に同じシャードに落ち、
    重複も取りこぼしも生じない（enumerate の index だと resume で並びがずれて破綻する）。
    組み込み hash() はプロセスごとに乱数化されマシン間で不一致になるため md5 を使う。
    """
    if shard is None:
        return pending
    i, n = shard
    if n <= 1:
        return pending
    return [r for r in pending
            if int(hashlib.md5(str(r["id"]).encode()).hexdigest(), 16) % n == (i - 1)]


def shard_tag(shard: tuple[int, int] | None) -> str | None:
    """シャード指定を part ファイル名のタグ（"2of6"）にする。分割しないなら None。"""
    if shard is None or shard[1] <= 1:
        return None
    return f"{shard[0]}of{shard[1]}"


def part_output_path(path: Path, tag: str | None) -> Path:
    """並列実行時の追記系ファイルを、書き手ごとに分けたパスにする。

    複数プロセス（さらに複数マシン・NFS 跨ぎ）が同一ファイルへ追記すると原子性が
    保証されず行が壊れ得るため、書き手ごとに別ファイルへ書く。tag はシャード方式なら
    "IofN"（shard_tag）、クレーム方式なら "w-<host>-<pid>"（worker_tag）。tag=None
    （単一プロセス実行）ではパスを変えない（後方互換）。マージは merge_shard_outputs()。
    例: audio_manifest_maai.jsonl → ....part-2of6.jsonl（★2026-11-22 旧名 audio_manifest_fish_speech_maai）
                                             → ....part-w-node22-12345.jsonl
    """
    if not tag:
        return path
    return path.with_name(f"{path.stem}.part-{tag}{path.suffix}")


# part の中身は "IofN"（シャード）か "w-<host>-<pid>"（クレーム）だけを認める。
# 任意文字にすると `<stem>.part-1of2_discarded.jsonl`（audit_pipeline の退避先）まで
# part とみなしてしまい、破棄した行がマージで生き返る。だから `_` を含む形は弾く。
_SHARD_PART_RE = re.compile(
    r"^(?P<base>.+)\.part-(?:\d+of\d+|w-[A-Za-z0-9-]+)(?P<suffix>\.[^.]+)$")


def drop_empty_part(path: Path | None, tag: str | None):
    """自分の part ファイルが空なら消す。

    クレーム方式では「もう全部終わっているか」を確かめる空振りの再実行が増える。
    その都度 0 バイトの part が残るとマージ対象が無駄に増えるので片付ける。
    消すのは自分の part（tag 付き）だけで、canonical には触れない。
    """
    if not tag or path is None:
        return
    try:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def merge_shard_outputs(root: Path, extra_dirs=()) -> int:
    """root 直下の per-shard 追記ファイル(*.part-IofN.*)を canonical へマージする。

    manifest / transcripts / refined_records など part_output_path() で分割された
    追記系 JSONL を variant 非依存でまとめる。canonical 名（.part-... を除いた名前）
    ごとに全 part を読み、**対話 id で重複排除（後勝ち）** して canonical へ上書き出力。
    id を持たない行は行内容キーで順序保持。冪等（何度流しても結果不変）。
    返り値はマージした canonical ファイル数。

    ★2026-11-16 root 直下（と extra_dirs の直下）だけを見る。以前は rglob で下の階層まで
    たどっていたが、part は canonical と同じフォルダにしか書かれない（part_output_path・
    audit_pipeline.variant_files も同じフォルダしか見ない）。rglob だと 1 万対話の出力先
    （audio/・audio_raw/・audio_fish/ に wav が十数万）を NFS でたどるだけで 20 分以上かかり、
    audio_fish/_stage4/（退避した fish 版の Stage4 ファイル）の part まで統合し直していた。
    """
    groups: dict[Path, list[Path]] = {}
    dirs = list(dict.fromkeys([Path(root), *(Path(d) for d in extra_dirs)]))
    for p in sorted(q for d in dirs for q in d.glob("*.part-*.*")):
        if not p.is_file():
            continue
        m = _SHARD_PART_RE.match(p.name)
        if not m:
            continue
        canonical = p.with_name(m.group("base") + m.group("suffix"))
        groups.setdefault(canonical, []).append(p)

    for canonical, parts in groups.items():
        order: list = []
        by_key: dict = {}
        for part in sorted(parts):
            with open(part) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        rid = json.loads(line).get("id")
                    except (json.JSONDecodeError, AttributeError):
                        rid = None
                    key = ("id", rid) if rid is not None else ("line", line)
                    if key not in by_key:
                        order.append(key)
                    by_key[key] = line
        tmp = canonical.with_name(canonical.name + ".merge.tmp")
        with open(tmp, "w") as out:
            for key in order:
                out.write(by_key[key] + "\n")
        tmp.replace(canonical)
        logger.info(f"マージ: {canonical.name} ← {len(parts)}シャード / {len(order)}件")
    if not groups:
        logger.info(f"マージ対象の part ファイルが見つかりません: {root}")
    return len(groups)


def run(config: dict, dry_run: bool, limit: int | None, device: str | None = None,
        max_seconds: float | None = None, retime: bool | None = None,
        recompose: bool = False, shard: tuple[int, int] | None = None,
        claim: bool = False, claim_ttl_min: float = 30.0):
    tts_cfg = config["tts"]
    engine = tts_cfg.get("engine", "gemini")
    input_file = Path(config["voice"]["output_file"])
    speed = float(tts_cfg.get("speed", 1.0) or 1.0)  # 発話速度（>1で速く）
    cp = (tts_cfg.get("google_cloud", {}) or {}).get("context_prosody") or {}
    ctx_prosody = bool(cp.get("enabled", False)) and engine in ("google_cloud", "gemini")
    ctx_short_len = int(cp.get("short_len", 6))
    fish_tagger, fish_rule_tags, fish_prefix_tag = build_fish_tagger(config, engine)
    trim_cfg = tts_cfg.get("trim_silence") or None  # 発話前後の無音トリミング設定
    noise_cfg = tts_cfg.get("noise_fill") or None   # 無音区間のコンフォートノイズ設定
    lead_cfg = tts_cfg.get("lead_in") or None       # 対話の先頭に置く無音（lead_in_sec）
    tail_cfg = tts_cfg.get("tail_pad") or None      # 対話の末尾に置く無音（tail_pad_sec）
    loud_cfg = tts_cfg.get("loudness") or None      # ターン間の音量均し設定
    if loud_cfg and loud_cfg.get("enabled", False):
        logger.info(f"音量均し: 目標 {loud_cfg.get('target_dbfs', -20.0)}dBFS "
                    f"（許容 ±{loud_cfg.get('deadband_db', 2.0)}dB / "
                    f"最大 ±{loud_cfg.get('max_gain_db', 12.0)}dB）")
    rate_cfg = tts_cfg.get("rate") or None          # ターンごとの話速均し設定
    if rate_cfg and rate_cfg.get("enabled", False):
        logger.info(f"話速均し: 目標 {rate_cfg.get('target_mora_per_sec', 7.3)}モーラ/秒 "
                    f"（許容 +{rate_cfg.get('deadband', 0.3)} / "
                    f"戻し {rate_cfg.get('ratio', 0.75)} / "
                    f"下限 ×{rate_cfg.get('max_slower', 0.80)}）")

    # リタイム設定（CLI --retime/--no-retime > config の tts.retime.enabled）
    rcfg = RetimeConfig.from_tts_cfg(tts_cfg)
    retime_enabled = rcfg.enabled if retime is None else retime
    variant = base_variant_name(rcfg.tag) if retime_enabled else None   # ★2026-11-22 エンジン名を外した（旧 f"{engine}_{rcfg.tag}"）

    # エンジン（リタイム時はバリアント）ごとに出力先を分ける
    audio_dir, manifest_file = engine_output_paths(tts_cfg, engine, variant)
    # 並列時は追記競合回避で書き手ごとの part ファイルに分ける（マージは --merge-parts）
    part_tag = worker_tag() if claim else shard_tag(shard)
    manifest_file = part_output_path(manifest_file, part_tag)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    retimer = None
    if retime_enabled:
        if dry_run:
            retimer = MockRetimer(rcfg)
            logger.info("リタイム(dry-run): MockRetimer（クランプ済みLLMオフセット）で配線検証")
        elif rcfg.backend == "vapbatch":
            # 学習コード(vap-kth/vap-bc)の一括推論。maai パッケージは不要だが、
            # 事前に書き出した重みが無いと動かないので先に確認して落とす。
            missing = [p for p in (rcfg.vapbatch.vap_ckpt, rcfg.vapbatch.bc_ckpt)
                       if not p or not Path(p).exists()]
            if missing:
                logger.error("backend=vapbatch の重みが見つかりません: "
                             + ", ".join(repr(m) for m in missing))
                logger.error("  作り方（学習環境の python で1回だけ）:")
                logger.error("  backend=vapbatch (vapmodel/, retime_vapbatch.py) is not included in the public release; "
                             "use the default backend maaibatch")
                sys.exit(1)
            retimer = MaaiRetimer(rcfg)
            logger.info(f"リタイム有効: vapbatch（vap={rcfg.vap.mode}/th={rcfg.vap.threshold}, "
                        f"bc={rcfg.bc.mode}/th={rcfg.bc.threshold}, device={rcfg.device}, "
                        f"align={rcfg.vapbatch.mimi_align}）")
        elif rcfg.backend == "maaibatch":
            # MaAI 公式リポジトリの MaaiBatch（retime_maaibatch.py）。pip の maai は import しない
            # （先に import するとリポジトリ版と混ざるので MaaiBatchScorer が落とす）。
            from retime_maaibatch import DEFAULT_REPO_SRC, import_maai_from_repo
            try:
                import_maai_from_repo(rcfg.maaibatch.repo_src or DEFAULT_REPO_SRC)
            except Exception as e:
                logger.error(f"backend=maaibatch の MaAI を読めません: {e}")
                sys.exit(1)
            retimer = MaaiRetimer(rcfg)
            logger.info(f"リタイム有効: maaibatch（vap={rcfg.vap.mode}/th={rcfg.vap.threshold}, "
                        f"mc={rcfg.maaibatch.mc}, bc={rcfg.bc.mode}/th={rcfg.bc.threshold}, "
                        f"device={rcfg.device}）")
        else:
            try:
                import maai  # noqa: F401  # 早期に明確なエラーで中断する
            except ImportError:
                logger.error("maai パッケージが見つかりません。リタイム実行には専用venvを使ってください:")
                logger.error("  .venv-maai/bin/python generate_tts.py --retime ...")
                sys.exit(1)
            retimer = MaaiRetimer(rcfg)
            logger.info(f"リタイム有効: MaAI（vap={rcfg.vap.mode}/th={rcfg.vap.threshold}, "
                        f"bc={rcfg.bc.mode}/th={rcfg.bc.threshold}, device={rcfg.device}）")

    # 発話単位wavキャッシュはエンジン基準のパス（レガシー/リタイムで共有）。
    # dry-run の正弦波はキャッシュ汚染になるので書かない。
    cache_root = engine_output_paths(tts_cfg, engine)[0] / "turns"
    write_cache = bool(rcfg.cache_turn_wavs) and not recompose and not dry_run

    voices = load_voices(tts_cfg)
    if not voices:
        logger.error("ボイスバンクが空です。tts.voices_file(CSV) または tts.voices を設定してください。")
        logger.error("CSVは `python manage_voices.py scan --dir <wav dir>` で雛形を作れます。")
        sys.exit(1)

    if not input_file.exists():
        logger.error(f"音声対話テキストが見つかりません: {input_file}")
        logger.error("先に Stage 3（python main.py --stage voice）を実行してください。")
        sys.exit(1)

    # 合成器の準備
    engine = tts_cfg.get("engine", "gemini")
    sr = int(tts_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
    synth = None
    short_threshold = 0
    if recompose:
        logger.info(f"recompose モード: 発話wavキャッシュ({cache_root})から再配置のみ"
                    "（TTS合成なし）。")
    else:
        synth, sr, short_threshold = build_synth(tts_cfg, engine, dry_run, device)
    logger.info(f"サンプルレート: {sr} Hz / 声の数: {len(voices)} / "
                f"短発話しきい値: {short_threshold}文字")

    # 入力読み込み
    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    total = len(records)

    # 再開: 既に音声化済み(wav存在)のIDをスキップ
    from audio_io import has_audio, write_audio
    pending = [r for r in records if not has_audio(audio_dir, r["id"])]
    already = total - len(pending)
    if recompose:  # 発話wavキャッシュがある対話だけが対象（--limit を浪費しない）
        n_before = len(pending)
        pending = [r for r in pending
                   if (cache_root / r["id"] / "meta.json").exists()]
        logger.info(f"recompose: キャッシュあり {len(pending)}/{n_before}件")
    if shard is not None:  # 複数GPU/複数fishサーバでの並列処理: 排他シャードに分割
        n_before = len(pending)
        pending = apply_shard(pending, shard)
        logger.info(f"シャード {shard[0]}/{shard[1]}: 担当 {len(pending)}/{n_before}件")
    if limit is not None:
        pending = pending[:limit]
    logger.info(f"TTS合成開始: 全{total}件中、合成済み{already}件、今回処理{len(pending)}件"
                + (f" / 冒頭{max_seconds}秒まで" if max_seconds else ""))

    # クレーム方式: 未処理の対話を先着順に取りに行く（シャードの静的分割の代替）。
    # 他プロセス・他マシンと同じ pending を見て、取れたものだけ処理する。
    claims = None
    if claim:
        claims = DialogueClaims(audio_dir / ".claims", ttl_sec=claim_ttl_min * 60.0)
        logger.info(f"クレーム方式: 未処理の対話を先着順に取得 "
                    f"(worker={claims.tag}, TTL={claim_ttl_min:.0f}分, {claims.root})")

    # 書き起こし（発話ごとの時刻+テキスト）を出力するか
    write_tr = bool(tts_cfg.get("write_transcript", True))
    tr_file = transcripts_path(manifest_file) if write_tr else None

    done = 0
    failed = 0
    no_cache = 0
    tag_skipped = 0   # LLMタグ付けが全リトライ失敗してスキップした対話数
    silent_dropped = 0  # 無音代替が出て破棄した対話数（wav・manifest とも未作成）
    import contextlib
    # closing(claims) を最初に置く: with は逆順に抜けるので、manifest を閉じてから
    # クレームを解放する。★Ctrl+C や例外で抜けても必ず解放されるのが要点。握ったまま
    # 落ちるとクレームの mtime が新しいまま残り、すぐ再実行してもその対話は TTL
    # （--claim-ttl-min）が切れるまで「他プロセスが処理中」と見なされて飛ばされる。
    with (contextlib.closing(claims) if claims is not None
          else contextlib.nullcontext()), \
            open(manifest_file, "a") as mf, \
            (open(tr_file, "a") if tr_file else contextlib.nullcontext()) as trf:
        for r in pending:
            dialogue_id = r["id"]
            if tts_servers_dead(synth):
                logger.error("TTS サーバが全部応答しないので、ここで終わる（担当中の claim は解放する）")
                break
            if claims is not None and not claims.take(
                    dialogue_id, audio_dir / f"{dialogue_id}{_DLG_EXT}"):
                continue  # 他プロセスが担当中／その間に完了済み
            rr = truncate_record_by_seconds(r, max_seconds)  # テスト用: 冒頭N秒のみ
            prepare_cut_turns(rr)        # 重なりの場面（rp_v1）: 言いかけは1文全体で合成してから切る
            use_sr = sr
            try:
                if recompose:
                    n_turns = len(rr["voice_dialogue"]["turns"])
                    cached = load_turn_cache(cache_root, dialogue_id, n_turns)
                    if cached is None:
                        no_cache += 1
                        continue
                    wav_by_turn, failed_turns, use_sr = cached
                    result = compose_dialogue(rr, voices, use_sr, wav_by_turn,
                                              failed_turns, retimer=retimer,
                                              trim=trim_cfg, noise=noise_cfg,
                                              loudness=loud_cfg, rate=rate_cfg,
                                              lead_in=lead_cfg, tail_pad=tail_cfg)
                else:
                    apply_dialogue_speaking_rate(synth, rr, tts_cfg)  # 対話ごとの話速
                    # fish-speech(LLM方式): 対話単位で詳細タグを付与し、検証用サイドファイルに残す
                    fish_map = None
                    if fish_tagger is not None:
                        fish_map = fish_tagger.tag_turns(
                            rr["voice_dialogue"]["turns"], dialogue_id)
                        if fish_map:
                            (audio_dir / f"{dialogue_id}.tags.json").write_text(
                                json.dumps(
                                    {str(k): v for k, v in sorted(fish_map.items())},
                                    ensure_ascii=False, indent=1))
                    result = synthesize_dialogue(
                        synth, rr, voices, sr, short_threshold=short_threshold,
                        retimer=retimer,
                        turn_cache_dir=(cache_root if write_cache else None),
                        speed=speed, context_prosody=ctx_prosody,
                        context_short_len=ctx_short_len, trim=trim_cfg,
                        emotion_tags=fish_rule_tags, fish_tags_map=fish_map,
                        prefix_tag=fish_prefix_tag, noise=noise_cfg,
                        loudness=loud_cfg, rate=rate_cfg, lead_in=lead_cfg,
                        tail_pad=tail_cfg)
            except FishTagError as e:
                # タグ無しで合成すると base キャッシュにタグ無し音声が残り、以後この
                # 対話にタグが付かなくなる。合成せずスキップし、次回実行の resume に回す。
                logger.error(f"[{dialogue_id}] LLMタグ付け失敗→この対話はスキップ"
                             f"（次回実行で再試行）: {e}")
                tag_skipped += 1
                continue
            except Exception as e:
                logger.error(f"[{dialogue_id}] 合成失敗: {e}")
                failed += 1
                continue
            if result is None:
                failed += 1
                continue
            stereo, manifest_turns = result
            # 合成失敗→無音代替が1発話でもあれば、この対話は何も書き出さない
            # （wav が無いので次回実行で再合成される。ログで原因を確認してから回す）
            if drop_if_synth_failed(dialogue_id, manifest_turns):
                silent_dropped += 1
                continue
            if max_seconds:  # 実音声が超過した分を冒頭N秒で切る
                stereo = stereo[:int(max_seconds * use_sr)]
            out_path = audio_dir / f"{dialogue_id}{_DLG_EXT}"
            write_audio(out_path, stereo, use_sr)
            voice_prompts = collect_voice_prompts(voices, manifest_turns)
            line = {
                "id": dialogue_id,
                "audio_path": str(out_path),
                "sample_rate": use_sr,
                "channels": int(stereo.shape[1]),
                "duration_sec": round(len(stereo) / use_sr, 3),
                "channel_speakers": _channel_speakers(rr),
                "voice_prompts": voice_prompts,
                "turns": manifest_turns,
                **padding_marks(dialogue_id, lead_cfg, tail_cfg),
            }
            if retime_enabled:  # 実行時のリタイム設定を記録（A/B比較・再現用）
                line["retime_config"] = rcfg.summary()
            mf.write(json.dumps(line, ensure_ascii=False) + "\n")
            mf.flush()
            if write_tr:  # 音声対話モデル用の書き起こし（<id>.json + 統合JSONL）
                write_transcript(audio_dir, trf, rr, manifest_turns, use_sr,
                                 int(stereo.shape[1]), len(stereo) / use_sr,
                                 _channel_speakers(rr), voice_prompts)
            done += 1
            if claims is not None:
                claims.finish(dialogue_id)
            if done % 20 == 0:
                logger.info(f"進捗: {done}件合成 (失敗{failed}件)")

    drop_empty_part(manifest_file, part_tag)   # 1件も書かなかった run の空 part は残さない
    drop_empty_part(tr_file, part_tag)
    if claims is not None:  # 解放は上の closing() 済み。ここは集計ログだけ
        claims.log_summary()
    if retimer is not None:
        retimer.log_summary()
    if no_cache:
        logger.info(f"recompose: 発話wavキャッシュが無い/不整合の対話 {no_cache}件をスキップ")
    if tag_skipped:
        logger.warning(f"LLMタグ付け失敗で {tag_skipped}件をスキップしました"
                       "（音声・キャッシュとも未作成。同じコマンドの再実行で再試行されます）")
    if silent_dropped:
        logger.error(f"無音代替が出た {silent_dropped}件を破棄しました"
                     "（音声・manifest・キャッシュとも未作成。ログの「無音代替のため対話を破棄」"
                     "で対話IDを確認し、原因を直してから同じコマンドで再実行してください）")
    if RETIME_FAILURES:
        logger.error(
            f"★リタイムが {len(RETIME_FAILURES)}件で失敗し、レガシー配置"
            "（VAPなしの固定間隔）で書き出しました。VAP配置の音声ではありません。\n"
            f"  対話ID: {', '.join(RETIME_FAILURES[:10])}"
            + (f" ほか{len(RETIME_FAILURES) - 10}件" if len(RETIME_FAILURES) > 10 else "")
            + "\n  該当の wav / manifest / transcript の行を消してから --retime を流し直してください。")
    logger.info(f"TTS合成完了: 成功{done}件 / 失敗{failed}件"
                + (f" / タグ失敗スキップ{tag_skipped}件" if tag_skipped else "")
                + (f" / 無音破棄{silent_dropped}件" if silent_dropped else "")
                + (f" / ★リタイム失敗{len(RETIME_FAILURES)}件" if RETIME_FAILURES else "")
                + f" | 出力: {audio_dir}")
    return done


def _channel_speakers(record: dict) -> dict:
    order = record_speaker_order(record)
    return {str(i): name for i, name in enumerate(order)}  # 話者数ぶん（2人/3人…）


def print_paths(config: dict):
    """config が決める Stage4 の入出力パスを KEY=VALUE で出す（シェルから eval する用）。

    `extends` の解決やバリアント名の組み立て（<retime.tag>_<refine.tag>。★2026-11-22 エンジン名を外した）を
    シェル側で再現すると必ずずれるので、ここで一次情報として出す。
    """
    import shlex
    from refine_maai import RefineConfig

    tts_cfg = config["tts"]
    engine = tts_cfg.get("engine", "gemini")
    rt = RetimeConfig.from_tts_cfg(tts_cfg)
    rf = RefineConfig.from_tts_cfg(tts_cfg)
    pairs = (
        ("INPUT_FILE", config["voice"]["output_file"]),          # Stage3 の出力＝全件
        ("ENGINE_DIR", engine_output_paths(tts_cfg, engine)[0]),  # 発話wavキャッシュの親
        # ★2026-11-22 バリアント名からエンジン名を外した（plain / maai / maai_ref）
        ("BASE_DIR", engine_output_paths(tts_cfg, engine, base_variant_name(rt.tag))[0]),
        ("REF_DIR", engine_output_paths(
            tts_cfg, engine, ref_variant_name(rt.tag, rf.tag))[0]),
        # fish の実体（fish = fish-speech の api_server / vllm_omni = vLLM-Omni）。run_tts_pool.sh の検出用
        ("FISH_BACKEND", str((tts_cfg.get("fish_speech") or {}).get("backend") or "fish")),
        # fish のモデル（ディレクトリ名）。run_tts_pool.sh はこれを読んでいるサーバだけ使う
        ("FISH_CHECKPOINT", fish_checkpoint_name(tts_cfg.get("fish_speech") or {})),
    )
    for k, v in pairs:
        print(f"{k}={shlex.quote(str(v))}")


def _exit_on_sigterm(signum, _frame):
    """SIGTERM を SystemExit に変換して、巻き戻し（finally）を通す。

    既定の SIGTERM はプロセスを即座に落とすので、クレームの解放（contextlib.closing）も
    ファイルのフラッシュも走らない。クレームを握ったまま落ちると mtime が新しいまま残り、
    すぐ再実行してもその対話は TTL（--claim-ttl-min）が切れるまで飛ばされてしまう。
    停止手順が `pkill -f generate_tts`（＝SIGTERM）なので、Ctrl+C と同じ挙動に揃える。
    ※SIGKILL（kill -9）とマシンごと落ちた場合は当然ここを通らない。TTL 頼みになる。
    """
    raise SystemExit(f"シグナル {signum} を受け取りました。中断します。")


def main():
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    parser = argparse.ArgumentParser(description="Stage 4: 音声対話テキスト→TTS音声合成（2チャネル）")
    parser.add_argument("--config", default="config.yaml", help="設定ファイルパス")
    parser.add_argument("--dry-run", action="store_true", help="TTSモデルを使わず正弦波で配線検証")
    parser.add_argument("--limit", type=int, help="処理する対話数の上限（テスト用）")
    parser.add_argument("--seconds", type=float,
                        help="各対話を冒頭この秒数までに切り詰めて合成（テスト用）")
    parser.add_argument("--gpu", help="使用するGPU ID（例: 0, 1, cuda:1, cpu）。"
                                      "複数GPUサーバで合成に使うGPUを指定。config の tts.gpu を上書き")
    parser.add_argument("--retime", dest="retime", action="store_true", default=None,
                        help="MaAI でターンテイキング・相槌タイミングを決め直す"
                             "（要 maai。出力は audio/<engine>_<tag>/ に分離）")
    parser.add_argument("--no-retime", dest="retime", action="store_false",
                        help="リタイムを無効化（config の tts.retime.enabled を上書き）")
    parser.add_argument("--recompose", action="store_true",
                        help="発話wavキャッシュから再配置のみ実行（TTS合成なし。"
                             "閾値チューニング用。要 tts.retime.cache_turn_wavs での事前合成）")
    parser.add_argument("--refine", action="store_true",
                        help="Gemini リファインメント: MaAI タイミングに基づき相槌・フィラーを"
                             "再生成し密度を対話ごとに調整（retime 前提。要 GEMINI_API_KEY）")
    parser.add_argument("--retime-refine", dest="retime_refine", action="store_true",
                        help="融合モード: 1対話ごとに retime→refine を連続で実行する。"
                             "段をまたぐバリア（全機 --retime 完了を待ってから --refine）が"
                             "不要になり、MaAI のロードも1回で済む。retime 済みの対話は"
                             "発話wavキャッシュを使って1段目をスキップするので、"
                             "--retime だけ済ませた分にそのまま重ねて流せる")
    parser.add_argument("--shard", help="複数GPU/複数fishサーバでの並列処理用。'I/N' 形式で"
                                        "対話を排他分割し担当分(I)だけ処理（例: 1/4）。"
                                        "各プロセスに別の I と --gpu/--fish-url を割り当てる")
    parser.add_argument("--claim", action="store_true",
                        help="並列処理用（--shard の代替・推奨）。担当を静的に分けず、"
                             "未処理の対話を先着順に取って処理する。担当の偏りが出ず、"
                             "マシンごとの速度差が末尾の遊びにならない。全プロセスが同じ"
                             "出力ディレクトリを見ていればよく、台数の増減も自由")
    parser.add_argument("--claim-ttl-min", type=float, default=30.0,
                        help="クレームの寿命（分, 既定30）。保持プロセスが落ちて更新が"
                             "止まったクレームは、この時間を過ぎたら他プロセスが引き取る。"
                             "1対話の処理時間より十分長くする（保持中は60秒ごとに更新）")
    parser.add_argument("--fish-url", dest="fish_url",
                        help="fish-speech サーバのURLを上書き（config の tts.fish_speech.base_url）。"
                             "シャードごとに別GPUの別fishインスタンスへ振り分けるのに使う"
                             "（例: http://localhost:8889）")
    parser.add_argument("--print-paths", action="store_true",
                        help="この config の入出力パスを KEY=VALUE 形式で標準出力へ"
                             "（合成はしない）。run_tts_pool.sh の進捗表示が"
                             "監視先ディレクトリを知るために使う")
    parser.add_argument("--merge-shards", "--merge-parts", dest="merge_shards",
                        action="store_true",
                        help="並列実行で出た書き手ごとの追記ファイル(*.part-IofN.* / "
                             "*.part-w-<host>-<pid>.*: manifest/transcripts/refined)を "
                             "canonical へ統合。全マシン完了後に1回だけ実行する"
                             "（冪等なので途中経過の確認に流しても構わない）")
    args = parser.parse_args()

    config = load_config(args.config)
    set_audio_format(config.get("tts"))
    _inherit_ruby_llm(config)

    if args.print_paths:  # ログを立てる前に。標準出力は KEY=VALUE だけにする
        print_paths(config)
        return

    # 全ステージを1つの config にまとめたときに、Stage4 のログを Stage1-3 と
    # 混ぜないための逃げ道（tts.log_file。無ければ従来どおり log_file）。
    setup_logging((config.get("tts") or {}).get("log_file") or config["log_file"])

    if args.merge_shards:  # 全シャード完了後の統合（合成はしない）
        root = Path(config["tts"]["manifest_file"]).parent
        rr = ((config["tts"].get("refine") or {}).get("refined_records_file"))
        merge_shard_outputs(root, [Path(rr).parent] if rr else [])   # refine の記録が別フォルダでも拾う
        return

    if args.claim and args.shard:
        parser.error("--claim と --shard は併用できません（担当の決め方が二重になります）")
    shard = parse_shard(args.shard)
    if args.fish_url:  # シャードごとに別 fish サーバへ（torch/合成器構築の前に反映）
        config.setdefault("tts", {}).setdefault("fish_speech", {})["base_url"] = args.fish_url

    # GPU選択（CLI > config）。torch インポート前に CUDA_VISIBLE_DEVICES を設定する。
    gpu = args.gpu if args.gpu is not None else config.get("tts", {}).get("gpu")
    device = select_gpu(gpu)

    if args.refine or args.retime_refine:
        if args.retime is False:  # --no-retime と併用不可（refine は retime 前提）
            parser.error("--refine は --no-retime と併用できません（refine は retime 前提です）")
        if args.retime_refine and args.recompose:
            # --recompose は精製済みレコードからの再配置。1段目を回す余地がない。
            parser.error("--retime-refine と --recompose は併用できません"
                         "（再配置だけなら --refine --recompose を使ってください）")
        from refine_maai import run_refine
        run_refine(config, dry_run=args.dry_run, limit=args.limit, device=device,
                   max_seconds=args.seconds, recompose=args.recompose, shard=shard,
                   claim=args.claim, claim_ttl_min=args.claim_ttl_min,
                   fused=args.retime_refine)
        return

    run(config, dry_run=args.dry_run, limit=args.limit, device=device,
        max_seconds=args.seconds, retime=args.retime, recompose=args.recompose,
        shard=shard, claim=args.claim, claim_ttl_min=args.claim_ttl_min)

    if RETIME_FAILURES:
        # リタイム有効なのにレガシー配置で書き出した対話がある。成功扱いで
        # 終わらせると気づかないので非ゼロ終了する（ログに対話IDが出ている）。
        sys.exit(3)


if __name__ == "__main__":
    main()
