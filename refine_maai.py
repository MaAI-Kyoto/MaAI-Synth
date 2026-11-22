"""Gemini リファインメント（Stage 4.5）: 相槌・フィラー再生成 + 密度制御

MaAI リタイミング（retime_maai.py）の予測を注釈として Gemini（テキスト生成）に渡し、
相槌の語の言い換え・挿入・削除、フィラーを織り込んだ発話書き直しを行う。
相槌密度（多い⇔少ない）は Gemini がメタデータ（ジャンル・関係性・話し方・トーン）
から対話ごとに判断する（config の上下限内）。

対話ごとの2パス構成:
  Pass A: 発話wav（キャッシュ or 合成）→ retimer.plan() で配置・p_peak・ギャップ取得
          → 挿入候補スキャン（bcモデルを聞き手方向ごとに全編1回走査、p_bcピーク検出）
  Gemini: メタデータ + 注釈付き対話 → JSON ops（reword/delete/insert/rewrite_text/merge）
          → validate_ops（決定的ガード + 密度整合）→ apply_ops（reindex）
  Pass B: 変更/新規ターンのみ再合成 → retimer.plan() 再実行（挿入相槌は rel_pos
          ヒント窓で発話途中に配置）→ compose → 音声 + manifest + 精製済みレコード出力

出力（variant = <retime.tag>_<refine.tag>、例 maai_ref。★2026-11-22 エンジン名を外した。旧 fish_speech_maai_ref）:
  audio/<variant>/<id>.wav|.flac                精製済み音声（tts.audio_format）
  audio_manifest_<variant>.jsonl                ターン別 refine 来歴付き manifest
  voice_dialogues_refined_<variant>.jsonl       テキスト・タイミングが音声と一致する
                                                データセットレコード
  output/refine_ops/<variant>/<id>.<hash>.json  Gemini ops キャッシュ（再実行で課金なし）
  audio/<variant>/turns/                        精製後の発話wavキャッシュ（--recompose 用）

使い方:
  python generate_tts.py --refine --dry-run --limit 3     # 配線検証（ネットワーク不要）
  .venv-maai/bin/python generate_tts.py --refine --limit 2  # 実 Gemini + 実 TTS
  .venv-maai/bin/python generate_tts.py --refine --recompose  # 精製済みから再配置のみ
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import hashlib
import json
import logging
import os
import random
import re
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import tts_reading
from audio_io import find_audio, has_audio, pcm16_bytes, write_audio, write_wav_bytes_as
from llm_client import extract_json_text
from rp_audio import apply_cuts, prepare_cut_turns
from retime_maai import (MAAI_SR, MaaiRetimer, MockRetimer, RetimeConfig,
                         ScoreTimeline, _reject_unknown_keys)

logger = logging.getLogger(__name__)

_PUNCT = "、。？！?!"

# run_refine が扱う対話の話者数。この値以外の対話は数えて飛ばす（既定は2人だけ＝従来どおり）。
# ★2026-10-12 三人対話用に定数へ出した。三人用（generate_tts_tri.py）が自分のプロセスの中でだけ
#   (2, 3) にし、リタイマ・挿入候補の走査・タイムラインを三人用に差し替える。二人用の動作は変わらない。
SUPPORTED_SPEAKER_COUNTS = (2,)


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

@dataclass
class DensityParams:
    min_bc_per_min: float = 1.5
    max_bc_per_min: float = 8.0
    tolerance: float = 0.8
    # ★2026-11-07 プロンプトに相槌の総数の目安（目標密度×対話長）を書き、弱い候補（p<0.50）を意味の切れ目なら
    #   使ってよいことにする（既定 False＝従来どおり）。傾聴（al_v1）で候補を 2.5倍にしても、LLM は
    #   target_bc_per_min を 12 と決めながら挿入を「12個」（1.9分の対話で 6.2回/分）にし、
    #   「目標密度のために弱い候補を使わない」に従って強い候補だけを選んだ。
    count_hint: bool = False


@dataclass
class InsertParams:
    enabled: bool = True
    min_turn_sec: float = 4.0        # この長さ以上の speech ターン内部のみ走査
    threshold: float = 0.40          # p_bc ローカルピークの下限
    min_separation_sec: float = 3.0  # 候補同士の最小間隔
    min_dist_existing_sec: float = 1.5  # 既存相槌打点からの最小距離
    edge_margin_sec: tuple = (1.0, 0.5)  # ターン頭/末尾の除外マージン
    window_sec: float = 1.0          # Pass B 打点決定窓 ±（rel_pos 中心）
    max_per_turn: int = 2
    # 冒頭・締めの挨拶の発話には相槌を挿入しない（既定 False＝従来どおり）。
    # 初対面の「初めまして。〜です。よろしくお願いします。」や締めの「また〜」の途中に
    # 「はい」「うん」が入ると不自然（2026-10-03 output_ft_v1_gemma_inoue_002 の検証）。
    skip_greetings: bool = False
    # ★2026-10-06 相槌の位置の条件（既定は従来どおり無効）。cmp_mt 20対話の挿入相槌205個で、
    #   質問の途中（「…体調はどうですか？」の途中に「ええ」）が25個、発話の頭から8字未満
    #   （「そうですね。|あの、…」の直後に「はい」）が25個あり、ユーザー指摘「相槌の形式が不自然」。
    skip_questions: bool = False     # 位置を含む文が「？」で終わるなら打たない
    min_head_chars: int = 0          # 発話の頭からこの実質字数を話した後でないと打たない
    # ★2026-11-07 主話者（metadata.main_speaker_index。generate_tts が ch0 に置く）が聞き手になる
    #   位置だけに効く上書き {キー: 値}（空＝従来どおり）。傾聴（al_v1）では主話者の相槌が主役なのに、
    #   既定の条件（4秒以上のターン・候補3秒間隔・1ターン3個）では候補が 6回/分しか出ず、挿入は 5回/分
    #   （人間の傾聴 ERICA の聞き手 15.6回/分）。全体に緩めると主話者の短い問いにも相手の相槌が入るので、
    #   聞き手が主話者のときだけ緩める。使えるキーは MAIN_LISTENER_KEYS。
    main_listener: dict = field(default_factory=dict)

    MAIN_LISTENER_KEYS = ("min_turn_sec", "threshold", "min_separation_sec", "min_dist_existing_sec",
                          "edge_margin_sec", "max_per_turn", "skip_questions", "min_head_chars")

    def __post_init__(self):
        bad = set(self.main_listener or {}) - set(self.MAIN_LISTENER_KEYS)
        if bad:
            raise ValueError(f"tts.refine.insert.main_listener に使えないキー: {sorted(bad)}"
                             f"（使えるのは {list(self.MAIN_LISTENER_KEYS)}）")

    def for_main_listener(self) -> "InsertParams":
        """main_listener の上書きを当てた写し（上書きが空なら自分自身）。"""
        if not self.main_listener:
            return self
        kw = dict(self.main_listener)
        if isinstance(kw.get("edge_margin_sec"), list):
            kw["edge_margin_sec"] = tuple(kw["edge_margin_sec"])
        return dataclasses.replace(self, main_listener={}, **kw)


@dataclass
class LimitParams:
    max_reword_frac: float = 0.50    # 言い換えできる既存相槌割合の上限
    max_rewrite_frac: float = 0.30   # rewrite_text できる speech ターン割合の上限
    max_delete_frac: float = 0.60    # 削除できる既存相槌割合の上限


@dataclass
class QualityParams:
    min_rewrite_chars: int = 12
    min_content_retention: float = 0.75
    max_rewrite_length_ratio: float = 1.80
    duplicate_bc_window: int = 2
    auto_merge_deleted_splits: bool = True
    bc_shorten_len: int = 4          # 実質文字数がこれ超＋読点ありの相槌は短縮（衝突回避）
    # ★2026-10-28 挿入相槌の wav の長さの歯止め（ユーザ報告の「同じ発話が2回続き、2回目が
    #   相手のチャネルにある」の原因）。挿入相槌は文脈つきで合成する（相手の発話への反応
    #   なので）が、fish は短いテキストと長い文脈を渡されると、文脈の発話を**自分の声で
    #   そのまま読み上げる**ことがある。実測（010 の1万対話）: 挿入相槌 91,050件のうち
    #   243件（対話数では2.4%）が 1.6秒超で、最長は「うん。」が 10.22秒。正常な相槌は
    #   どの字数でも p99.9 が 1.07秒なので、下の式で確実に仕分けられる。
    #   許容秒数 = bc_dur_base + bc_dur_per_char × 実質文字数（0 なら検査しない）。
    #   超えたものは文脈なしで bc_dur_retries 回まで引き直し、それでも長ければ捨てる。
    bc_dur_base: float = 0.8
    bc_dur_per_char: float = 0.25
    bc_dur_retries: int = 2
    hee_thin_ratio: float = 0.5      # 「へー/へえ/へぇ」系相槌を出どころ問わずこの割合だけ他語へ置換
    insert_continuer_below_rel: float = 0.85  # rel_pos がこれ未満の途中挿入相槌は継続支持語へ強制
    # ★2026-09-26 接客（人手評価シート）向けの歯止め。既定はすべて無効（従来どおり）。
    #   実測（別の担当者ペルソナの100対話の refine）: 担当者の発話にフィラーを書き戻した 9件、客の発話の
    #   フィラー連続 44件、「……」7件、「探しているんで、すけど」1件、担当者の相槌「うんうん」1件。
    no_rewrite_speakers: list = field(default_factory=list)  # この話者の speech は rewrite しない・
                                                               # 相槌をくだけた語にしない。"@staff" は
                                                               # metadata の role が「担当者」の話者
    max_added_fillers: int = -1      # rewrite で足してよいフィラーの数（-1 で無制限）。連続も拒否
    reject_ellipsis: bool = False    # 「…」「‥」を含む new_text を拒否
    # ★2026-09-27 挿入相槌の語の配分 {語: 重み}（空＝従来どおり）。LLM は途中の挿入に
    #   「ええ・はい・うん・うんうん」の4語しか選ばない（ft で ええ 41%。人間は え系 4%弱で、
    #   「あー」15%・「へえ」6% が上位）。指定すると、プロンプトで語を示したうえで、LLM の
    #   選んだ語が配分より多い分を足りない語へ機械的に置き換える（_rebalance_bc_vocab）。
    #   有効時は「へー」系の半減（hee_thin_ratio）は行わない（へえ の割合もこの配分で決める）。
    bc_vocab_mix: dict = field(default_factory=dict)
    # ★2026-11-07 挿入相槌の語の案内の書き方（"" = 従来 / "listener" = 傾聴の聞き手向け）。傾聴（al_v1）の
    #   試走7で、LLM の選んだ挿入相槌は はい50%・ええ25%・へえ1%（後処理で替わったのは423個中5個）。
    #   従来の案内は「途中は聞いていることを示す語」「へえは1文言い終えた位置だけ」と反応の語を抑える向きで、
    #   ユーザ指摘「おー・あー・へー が少ない」。"listener" は反応の語の使いどころと bc_vocab_mix の目安の割合を
    #   示す（語を増やすのは LLM。置き換えの規則は足さない）。
    bc_vocab_style: str = ""
    # ★2026-11-08 反応の語（あー・へえ・おー）の挿入相槌を、LLM がもう一度だけ1つずつ確かめる（既定 False＝従来どおり）。
    #   試走8の視聴「あー・おー・へー はピンポイントで。迷ったら うん・はい」。大きな編集プロンプトの中の条件だけでは、
    #   Gemma は「花粉がひどくて｜へえ」「大層なもんじゃないですよ｜へえ」のような外れを2割ほど残した（同じ候補での比較）。
    #   短いプロンプトで「相槌の位置までに相手が言ったこと」と語だけを見せ、合わなければ LLM に うん・はい 系へ替えさせる
    #   （規則による置き換えではない。結果は refine_ops/<variant>/<id>.react.<hash>.json と、相槌の refine.react_orig に残る）。
    bc_react_verify: bool = False
    # ★2026-11-07 挿入相槌の声が相手に寄ったもの（文脈つき合成で起きる。試走7で「はい。」が相手の声に聞こえると
    #   ユーザ指摘。本人 0.17・相手 0.43）を x-vector で見つけ、文脈なしで引き直す（空＝従来どおり）。
    #   {enabled, own_max（既定 0.85）, margin（既定 0.10）, retries（既定 3）, min_sec（既定 0.2）, model, device}。
    #   短い音声の判定はぶれるが、引き直しても同じ語の別テイクになるだけなので、疑わしいものは引き直す。
    bc_leak_check: dict = field(default_factory=dict)
    # bc_vocab_mix の配分に合わせた確率的な置き換えを行うか（既定 True＝従来どおり）。
    # False でも、語の候補はプロンプトで示し、合わない位置の「へえ」とタメ口の話者の
    # 「はい・ええ」系の強制置き換えだけは行う。
    # ★2026-10-03 output_ft_v1_gemma_inoue_002 の検証: 5対話の相槌53個のうち34個（64%）を
    #   文脈を見ずに置き換えており、不自然な7個のうち6個がこれ（挨拶への「へえ」、聞き返しへの
    #   「あー」、「いえいえ」への「うん」等）。LLM が選んだ語はおおむね文脈に合っていた。
    bc_mix_rebalance: bool = True
    # refine で書き直した本人の発話（相槌以外）を、相手の文脈を付けずに本人の参照だけで合成し直す
    # （既定 False＝従来どおり全部を文脈つきで再合成）。★2026-10-04 文脈つき再合成（直前の相手ターン
    # ＋2人分の参照）で声が相手に寄った（cmp_new 1309f41c の turn22: 本人 0.67・相手 0.97、
    # check_speaker_leak.py）。相手への反応である挿入相槌は文脈つきのまま。
    speech_resynth_no_context: bool = False


@dataclass
class RefineConfig:
    enabled: bool = False
    tag: str = "ref"
    provider: str = "gemini"       # "gemini"(google-genai) / "openai"(ChatGPT / OpenAI互換ローカル)
    text_model: str = "gemini-3.5-flash"
    fallback_text_model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str | None = None    # provider=openai でローカルOpenAI互換サーバ(vLLM等)を使う場合
    base_urls: list | None = None  # 複数指定でリクエストごとに投げ先をランダムに選び負荷分散
                                   # （優先度: base_urls > hosts > base_url。単数なら従来どおり）
    hosts: list | None = None      # マシン名だけ書くと text_model を出しているサーバを探して使う
    ports: list | None = None      # hosts で探すポート（既定 llm_client.DEFAULT_LLM_PORTS）
    endpoint_pick: str = "least_loaded"  # 投げ先が複数のときの選び方。least_loaded＝/metrics の処理中＋待ちが
                                   # 最小のサーバ（読めなければランダム）／random＝従来のランダム（2026-11-18）
    extra_body: dict = field(default_factory=dict)  # vLLM の guided JSON / thinking無効化等
    rpm: float = 8
    max_retries: int = 3
    json_retries: int = 1
    temperature: float = 0.4
    density: DensityParams = field(default_factory=DensityParams)
    insert: InsertParams = field(default_factory=InsertParams)
    limits: LimitParams = field(default_factory=LimitParams)
    quality: QualityParams = field(default_factory=QualityParams)
    ops_dir: str = "output/refine_ops"
    refined_records_file: str = "output/voice_dialogues_refined.jsonl"
    cache_turn_wavs: bool = True

    @classmethod
    def from_tts_cfg(cls, tts_cfg: dict) -> "RefineConfig":
        rc = (tts_cfg or {}).get("refine") or {}

        def sub(cls_, key):
            d = rc.get(key) or {}
            kw = {k: v for k, v in d.items() if k in cls_.__dataclass_fields__}
            if "edge_margin_sec" in kw and isinstance(kw["edge_margin_sec"], list):
                kw["edge_margin_sec"] = tuple(kw["edge_margin_sec"])
            return cls_(**kw)

        _reject_unknown_keys("tts.refine", rc, cls,
                             {"density": DensityParams, "insert": InsertParams,
                              "limits": LimitParams, "quality": QualityParams})
        keys = set(cls.__dataclass_fields__) - {"density", "insert", "limits", "quality"}
        kwargs = {k: v for k, v in rc.items() if k in keys}
        return cls(density=sub(DensityParams, "density"),
                   insert=sub(InsertParams, "insert"),
                   limits=sub(LimitParams, "limits"),
                   quality=sub(QualityParams, "quality"), **kwargs)

    def summary(self) -> dict:
        return {
            "tag": self.tag,
            "provider": self.provider,
            "text_model": self.text_model,
            "temperature": self.temperature,
            "density": {"min": self.density.min_bc_per_min,
                        "max": self.density.max_bc_per_min,
                        "tolerance": self.density.tolerance},
            "insert": {"enabled": self.insert.enabled,
                       "threshold": self.insert.threshold,
                       "window_sec": self.insert.window_sec},
            "limits": {"max_reword_frac": self.limits.max_reword_frac,
                       "max_rewrite_frac": self.limits.max_rewrite_frac,
                       "max_delete_frac": self.limits.max_delete_frac},
            "quality": {"min_rewrite_chars": self.quality.min_rewrite_chars,
                        "min_content_retention": self.quality.min_content_retention,
                        "max_rewrite_length_ratio": self.quality.max_rewrite_length_ratio,
                        "duplicate_bc_window": self.quality.duplicate_bc_window,
                        "auto_merge_deleted_splits": self.quality.auto_merge_deleted_splits,
                        "bc_shorten_len": self.quality.bc_shorten_len,
                        "hee_thin_ratio": self.quality.hee_thin_ratio,
                        "insert_continuer_below_rel":
                            self.quality.insert_continuer_below_rel,
                        "bc_vocab_mix": self.quality.bc_vocab_mix},
        }


_CONTENT_ANCHOR_RE = re.compile(
    r"[A-Za-z]+|\d+(?:\.\d+)?|[〇零一二三四五六七八九十百千万億兆]+")
_NEGATION_RE = re.compile(
    r"ない|なく|なかった|ません|ませんでした|ではない|じゃない|ぬ(?:[。！？?!]|$)")
_QUESTION_RE = re.compile(
    r"[？?]|(?:です|ます|でしょう|でした|ました)か(?:[。！？?!]|$)|かな(?:[。！？?!]|$)")


def _text_core(text: str) -> str:
    """句読点・空白を除き、rewrite の内容保持判定に使う文字列を返す。"""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in s
                   if not unicodedata.category(ch).startswith(("P", "Z"))
                   and not ch.isspace())


def _normalize_bc_text(text: str) -> str:
    """表記上の句読点差を無視した相槌の重複判定キー。"""
    s = _text_core(text)
    return s.replace("~", "").replace("〜", "")


def _shorten_bc(text: str, max_len: int = 4) -> str | None:
    """長い相槌（2語以上の反応の連結）を最初の読点までの短い相槌に縮める。

    例「なるほど、分かりました。」→「なるほど」／「ええ、そうでしたね。」→「ええ」。
    実質文字数が max_len 以下、または読点(、,)が無い場合は None（短縮しない）。
    短縮後が空になる場合も None。
    """
    s = str(text).strip()
    core = len(_text_core(s))
    if core <= max_len:
        return None
    head = re.split(r"[、,]", s, maxsplit=1)[0]
    head = head.strip().rstrip("、,。．.！!？? 　")
    if not head or head == s.strip().rstrip("、,。．.！!？? 　"):
        return None  # 読点なし/変化なしは対象外
    return head


# 「へー」系相槌（へー/へえ/へぇ/へ〜 …）: 実質文字が「へ」＋長音・え のみ
_HEE_RE = re.compile(r"へ[ーえぇ]*$")
# 置換先: へー が担う「驚き・関心・受理」を保つ短い相槌。うん/なるほど偏重を避ける語彙。
_HEE_ALT_CASUAL = ["そうなんだ", "ほんと", "あー", "そうなの", "うわ"]
_HEE_ALT_POLITE = ["そうですか", "ほんとですか", "なるほど"]
_POLITE_RE = re.compile(r"です|ます|でしょ|ください")


def _is_hee(text: str) -> bool:
    """相槌テキストが「へー/へえ/へぇ」系か（句読点・波ダッシュは無視）。"""
    core = _normalize_bc_text(text)
    return bool(core) and _HEE_RE.fullmatch(core) is not None


def _thin_hee_family(turns: list, changed: set, quality: QualityParams,
                     window: int) -> int:
    """へー系相槌を出どころ（挿入/言い換え/据え置き）に関わらず hee_thin_ratio だけ
    他の相槌へ置換する。置換したターンは changed に入れて Pass B で再合成させる。

    自然さの低い順（挿入 slot_p / 既存 p が低い、無印は既定 0.35）に置換対象を選び、
    敬語対話には敬語寄りの置換先を使う。近傍相槌との重複は避ける。1個だけの
    へー は round(0.5*1)=0 で温存される（多用のときだけ間引く）。
    """
    ratio = getattr(quality, "hee_thin_ratio", 0.5)
    if ratio <= 0 or getattr(quality, "bc_vocab_mix", None):
        return 0  # bc_vocab_mix 有効時は へえ の割合も配分側で決める
    hee = [ni for ni, t in enumerate(turns)
           if t.get("type", "speech") == "backchannel" and _is_hee(t.get("text", ""))]
    n_replace = int(round(len(hee) * ratio))
    if n_replace <= 0:
        return 0
    speech = [t.get("text", "") for t in turns
              if t.get("type", "speech") != "backchannel"]
    polite = sum(1 for s in speech if _POLITE_RE.search(s))
    pool = (_HEE_ALT_POLITE if speech and polite >= len(speech) * 0.5
            else _HEE_ALT_CASUAL)

    def _p(ni):
        rf = turns[ni].get("refine") or {}
        return rf.get("slot_p") or rf.get("p") or 0.35

    hee.sort(key=_p)  # 自然さの低い順に置換
    n = 0
    rot = 0
    for ni in sorted(hee[:n_replace]):
        t = turns[ni]
        repl = None
        for k in range(len(pool)):
            cand = pool[(rot + k) % len(pool)]
            if not _duplicates_nearby_bc(cand, t["speaker"], ni, turns,
                                         window, exclude_index=ni):
                repl, rot = cand, (rot + k + 1) % len(pool)
                break
        if repl is None:
            repl, rot = pool[rot % len(pool)], rot + 1
        rf = dict(t.get("refine") or {})
        rf.setdefault("orig_text", t["text"])
        rf["op"] = ("hee_thin" if rf.get("op") in (None, "insert")
                    else rf["op"] + "+hee_thin")
        t["refine"] = rf
        t["text"] = repl
        changed.add(ni)
        n += 1
    return n


# 継続支持（「聞いているよ」の相づち）と見なす語。これ以外の挿入相槌は、発話の途中
# （rel_pos < 閾値）では継続支持語へ置換する。内容を理解・評価する反応（なるほど/へえ/
# そうなんだ等）が発話の途中に鳴ると、まだ喋り終えていないのに末尾へ反応しているように
# 聞こえるため。
_CONTINUER_SET = {"うん", "うんうん", "ううん", "うーん", "はい", "はいはい",
                  "ええ", "えぇ", "ん", "んー", "んん", "ふん", "ふーん",
                  "ふんふん", "あー", "えー"}
_CONTINUER_POOL_CASUAL = ["うん", "うんうん", "はい", "ええ"]
_CONTINUER_POOL_POLITE = ["はい", "ええ", "うん", "うんうん"]

# 「ふんふん」（連呼形）は TTS が笑い声として合成してしまうことがあるため、refine が
# 新しく書く相槌テキストからは排除し、同じ継続支持の意味を持つ「うんうん」へ寄せる。
# 単発の「ふん」「ふーん」は笑い声化しないので対象外。
_FUNFUN_RE = re.compile(r"(?:ふー*ん){2,}")


def _defuse_funfun(text: str) -> str:
    """相槌テキスト中の「ふんふん」系を「うんうん」へ置換する。"""
    return _FUNFUN_RE.sub("うんうん", text)


def _is_continuer(text: str, extra=frozenset()) -> bool:
    """相槌テキストが継続支持語か（句読点・波ダッシュは無視）。extra は追加で許す語。"""
    key = _normalize_bc_text(text)
    return key in _CONTINUER_SET or _mix_key(key) in extra


def _bc_dur_limit(text: str, q: QualityParams) -> float:
    """その相槌テキストとして許せる wav の長さ（秒）。0 なら検査しない。"""
    if q.bc_dur_base <= 0:
        return 0.0
    n = len(re.sub(r"[、。？！\s]", "", str(text or "")))
    return q.bc_dur_base + q.bc_dur_per_char * n


_LEAK_REF_EMB: dict = {}


def regen_leaky_bcs(synth, turns: list, wavs: dict, sr: int, targets: set,
                    assignment: dict, cfg: dict, did: str, short_threshold: int,
                    speed: float) -> int:
    """再合成した相槌のうち、声が相手に寄ったもの（speaker_check.is_leak）を文脈なしで引き直す。

    判定は本人・相手の参照音声との x-vector のコサイン類似度。疑わしいものは retries 回まで文脈なしで引き、
    本人と相手の差がいちばん大きいテイクに**必ず**差し替える。文脈なしの合成には相手の声が入らないので、
    判定がぶれていても（試走8: 声の近い組では文脈なしのテイクでも相手寄りに出た）差し替えて損はない。
    差し替えた数を返す。依存（torch / transformers）やモデルが無ければ何もしない。
    """
    import soundfile as sf
    import speaker_check as SC
    from generate_tts import normalize_tts_text, synthesize_for_speaker
    ver = SC.get_verifier(str(cfg.get("model", "microsoft/wavlm-base-plus-sv")),
                          str(cfg.get("device", "cuda")))
    if ver is None:
        return 0
    own_max = float(cfg.get("own_max", 0.85))
    margin = float(cfg.get("margin", 0.10))
    retries = int(cfg.get("retries", 3))
    min_n = int(float(cfg.get("min_sec", 0.2)) * sr)

    def ref(sp):
        path = assignment[sp]["audio_path"]
        if path not in _LEAK_REF_EMB:
            w, r = sf.read(path, dtype="float32")
            _LEAK_REF_EMB[path] = ver.embed(w, r)
        return _LEAK_REF_EMB[path]

    def sims(w, sp):
        e = ver.embed(w, sr)
        own = float(e @ ref(sp))
        other = max(float(e @ ref(o)) for o in assignment if o != sp)
        return own, other

    fixed = 0
    for ni in sorted(targets):
        t = turns[ni] if ni < len(turns) else None
        w = wavs.get(ni)
        if (t is None or t.get("type", "speech") != "backchannel" or w is None
                or len(w) < min_n or t.get("speaker") not in assignment or len(assignment) < 2):
            continue
        sp = t["speaker"]
        try:
            own, other = sims(w, sp)
        except RuntimeError:                 # 短すぎて WavLM の畳み込みに通らない
            continue
        if not SC.is_leak(own, other, own_max, margin):
            continue
        txt = normalize_tts_text(tts_reading.turn_text(t))
        best_w, best = None, float("-inf")
        tries = [(round(own, 2), round(other, 2))]
        for _ in range(max(1, retries)):
            w2s, f2 = synthesize_for_speaker(synth, [txt], assignment[sp], [0.0], did, sr,
                                             short_threshold=short_threshold, speed=speed)
            if not w2s or w2s[0] is None or f2[0] or len(w2s[0]) < min_n:
                continue
            try:
                o2, x2 = sims(w2s[0], sp)
            except RuntimeError:
                continue
            tries.append((round(o2, 2), round(x2, 2)))
            if o2 - x2 > best:
                best_w, best = w2s[0], o2 - x2
            if not SC.is_leak(o2, x2, own_max, margin):
                break
        head = f"[{did[:8]}] 挿入相槌 turn{ni}「{t.get('text', '')}」の声が相手に寄っている（本人{own:.2f}/相手{other:.2f}）"
        if best_w is None:
            logger.warning(f"{head} → 文脈なしの引き直しが合成できず、元のテイクを維持 tries={tries}")
            continue
        wavs[ni] = best_w
        fixed += 1
        logger.info(f"{head} → 文脈なしで引き直し tries={tries}")
    return fixed


def overlong_bc_indices(turns: list, wavs: dict, sr: int,
                        q: QualityParams) -> list[int]:
    """テキストに対して wav が長すぎる挿入相槌のターン番号（＝合成異常）。"""
    out = []
    for i, t in enumerate(turns):
        if t.get("type", "speech") != "backchannel":
            continue
        w = wavs.get(i)
        if w is None or sr <= 0:
            continue
        lim = _bc_dur_limit(t.get("text", ""), q)
        if lim > 0 and len(w) / sr > lim:
            out.append(i)
    return out


def drop_turns(turns: list, wavs: dict, failed: set, drop: set):
    """ターンを取り除き、wav と失敗集合の番号を詰め直す（破壊的）。"""
    if not drop:
        return
    keep = [i for i in range(len(turns)) if i not in drop]
    new_w = {k: wavs[i] for k, i in enumerate(keep) if i in wavs}
    new_f = {k for k, i in enumerate(keep) if i in failed}
    turns[:] = [turns[i] for i in keep]
    wavs.clear()
    wavs.update(new_w)
    failed.clear()
    failed.update(new_f)


def _continuer_only_inserts(turns: list, changed: set, quality: QualityParams,
                            window: int) -> int:
    """発話途中（rel_pos < insert_continuer_below_rel）に挿入された相槌のうち、継続支持語で
    ないもの（＝内容反応語）を継続支持語へ置換する。挿入相槌は MaAI の p_bc ピーク位置に
    置かれるが、LLM はそこまでの内容しか聞いていない前提の反応を書くべきで、末尾寄りでない
    位置に「なるほど」等を置くと発話末尾への反応に聞こえる。末尾寄り（rel_pos >= 閾値）の
    挿入は文が完結に近いので対象外。置換ターンは changed に入れて Pass B で再合成させる。
    """
    thr = getattr(quality, "insert_continuer_below_rel", 0.85)
    if thr <= 0:
        return 0
    targets = []
    for ni, t in enumerate(turns):
        if t.get("type", "speech") != "backchannel":
            continue
        rf = t.get("refine") or {}
        rel = rf.get("rel_pos")
        if rel is None:                      # 挿入相槌のみ（rel_pos 付き）
            continue
        if float(rel) >= thr:                # 末尾寄りは許容（文完結に近い）
            continue
        # bc_vocab_mix の語（あー・へえ・うんうんうん等）も途中で使ってよい。
        # へえ が合わない位置かどうかは _rebalance_bc_vocab が見る。
        if _is_continuer(t.get("text", ""), _mix_weights(quality).keys()):
            continue
        targets.append(ni)
    if not targets:
        return 0
    speech = [t.get("text", "") for t in turns
              if t.get("type", "speech") != "backchannel"]
    polite = sum(1 for s in speech if _POLITE_RE.search(s))
    pool = (_CONTINUER_POOL_POLITE if speech and polite >= len(speech) * 0.5
            else _CONTINUER_POOL_CASUAL)
    n = 0
    rot = 0
    for ni in targets:
        t = turns[ni]
        repl = None
        for k in range(len(pool)):
            cand = pool[(rot + k) % len(pool)]
            if not _duplicates_nearby_bc(cand, t["speaker"], ni, turns,
                                         window, exclude_index=ni):
                repl, rot = cand, (rot + k + 1) % len(pool)
                break
        if repl is None:
            repl, rot = pool[rot % len(pool)], rot + 1
        rf = dict(t.get("refine") or {})
        rf.setdefault("orig_text", t["text"])
        rf["op"] = ("insert_continuer" if rf.get("op") in (None, "insert")
                    else rf["op"] + "+insert_continuer")
        t["refine"] = rf
        t["text"] = repl
        changed.add(ni)
        n += 1
    return n


# bc_vocab_mix の語の表記ゆれ（数えるときと置き換え先を決めるときに同じ語として扱う）。
_MIX_ALIAS = {"あ": "あー", "ああ": "あー", "あぁ": "あー", "あーー": "あー",
              "へー": "へえ", "へぇ": "へえ", "へーー": "へえ",
              "おお": "おー", "おぉ": "おー", "おーー": "おー"}
# 相手の話の中身に反応する語（傾聴の案内 bc_vocab_style: listener で、うん・はい 系と分けて書く）。
_REACTIVE_BC = {"あー", "へえ", "おー"}
# タメ口の話者には使わない語（友人どうしの「はいはい」「ええ」は硬すぎる）。
_MIX_POLITE_ONLY = {"はい", "はいはい", "はいはいはい", "ええ"}
# 「へえ」は相手が新しい事実を1文言い終えた直後だけ。挨拶・礼・謝罪への へえ は変。
_HEE_NG_SENT_RE = re.compile(
    r"よろしく|お願い|はじめまして|初めまして|こんにちは|こんばんは|おはよう|"
    r"ありがとう|すみません|失礼")
_HEE_NG_END_RE = re.compile(r"(?:か|かね|よね|ね|な|かな)$")


def _mix_key(key: str) -> str:
    """正規化済みの相槌テキストを bc_vocab_mix の語にそろえる（へー→へえ 等）。"""
    return _MIX_ALIAS.get(key, key)


def _mix_weights(quality: QualityParams) -> dict:
    """bc_vocab_mix を {正規化した語: 割合（合計1）} にする。空なら {}。"""
    raw = getattr(quality, "bc_vocab_mix", None) or {}
    w = {}
    for k, v in raw.items():
        if float(v) > 0:
            key = _mix_key(_normalize_bc_text(k))
            w[key] = w.get(key, 0.0) + float(v)
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else {}


def _content_sentence_end(head: str) -> bool:
    """head が中身のある平叙文（実質8字以上・挨拶や礼でない）を言い終えたところで終わっているか。"""
    head = head.rstrip()
    if not head.endswith(("。", "！", "!")):
        return False
    sent = re.split(r"[。？！?!]", head[:-1])[-1]
    return len(_text_core(sent)) >= 8 and not _HEE_NG_SENT_RE.search(sent)


def _hee_fits(turns: list, ni: int) -> bool:
    """挿入相槌 ni の打点が「へえ」に合うか。直前の相手の発話が、打点までに平叙文を
    1つ言い終えていて（句点の直後）、その文が挨拶・礼・問いかけ・同意求めでないこと。"""
    rel = (turns[ni].get("refine") or {}).get("rel_pos")
    if rel is None:
        return False
    anchor = next((turns[j] for j in range(ni - 1, -1, -1)
                   if turns[j].get("type", "speech") != "backchannel"), None)
    if anchor is None or anchor.get("speaker") == turns[ni].get("speaker"):
        return False
    text = anchor.get("text", "")
    head = text[:snap_char_pos(text, float(rel))]
    if not head.endswith("。"):
        return False
    sent = re.split(r"[。？！?!]", head[:-1])[-1]
    core = _text_core(sent)
    return (len(core) >= 8 and not _HEE_NG_SENT_RE.search(sent)
            and not _HEE_NG_END_RE.search(core))


def _aa_fits(turns: list, ni: int) -> bool:
    """挿入相槌 ni の打点が「あー」（納得・思い当たり）に合うか。相手が文を言い終えた直後か、
    今の文で実質12字以上話した後。前置きの途中（「でも、あるときから、｜」）は合わない
    （2026-10-06 cmp_mt 20対話の「あー」21個中6個。ユーザー指摘）。"""
    rel = (turns[ni].get("refine") or {}).get("rel_pos")
    if rel is None:
        return True                     # 挿入でない（台本の）相槌は触らない
    anchor = next((turns[j] for j in range(ni - 1, -1, -1)
                   if turns[j].get("type", "speech") != "backchannel"), None)
    if anchor is None or anchor.get("speaker") == turns[ni].get("speaker"):
        return True
    text = anchor.get("text", "")
    head = text[:snap_char_pos(text, float(rel))]
    if head.rstrip().endswith(("。", "？", "！", "?", "!")):
        return True
    return len(_text_core(re.split(r"[。？！?!]", head)[-1])) >= 12


# 強制の置き換え先（タメ口の話者が使った丁寧な相槌 → 同じ意味のくだけた相槌）。
# 「へえ」「あー」（合わない位置）はここに無く、話者が丁寧なら「はい」、くだけていれば「うん」にする。
_FORCED_MAP = {"はい": "うん", "ええ": "うん", "はいはい": "うんうん", "はいはいはい": "うんうんうん"}


def _rebalance_bc_vocab(turns: list, changed: set, quality: QualityParams,
                        window: int, protected: set, seed: str) -> int:
    """挿入相槌の語を bc_vocab_mix の配分に近づける（空なら何もしない）。

    聞き手ごとに、配分より多い語を超過分の割合で確率的に選び（seed で決定的）、配分に
    足りない語へ置き換える。置き換え先は近傍の同語を避け、タメ口の話者には はい・ええ 系を
    使わず、「へえ」は _hee_fits の位置だけ。合わない位置の へえ（LLM が書いたものも）は
    必ず置き換える。配分に無い語（なるほど 等）と protected の話者の相槌は触らない。
    置き換えたターンは changed に入れて Pass B で再合成させる。
    """
    mix = _mix_weights(quality)
    if not mix:
        return 0
    rng = random.Random(seed)
    n_done = 0
    for listener in sorted({t.get("speaker") for t in turns
                            if t.get("type", "speech") == "backchannel"} - set(protected)):
        idx = [ni for ni, t in enumerate(turns)
               if t.get("type", "speech") == "backchannel"
               and t.get("speaker") == listener
               and (t.get("refine") or {}).get("rel_pos") is not None]
        if not idx:
            continue
        own = [t.get("text", "") for t in turns
               if t.get("type", "speech") != "backchannel" and t.get("speaker") == listener]
        polite = bool(own) and sum(1 for s in own if _POLITE_RE.search(s)) >= len(own) * 0.5
        wts = {k: v for k, v in mix.items() if polite or k not in _MIX_POLITE_ONLY}
        tot = sum(wts.values())
        if tot <= 0:
            continue
        wts = {k: v / tot for k, v in wts.items()}
        keys = {ni: _mix_key(_normalize_bc_text(turns[ni].get("text", ""))) for ni in idx}
        count = Counter(keys.values())
        n = len(idx)
        deficit = {w: max(0.0, p * n - count.get(w, 0)) for w, p in wts.items()}
        todo, forced = [], set()
        for ni in idx:
            w = keys[ni]
            if (w == "へえ" and not _hee_fits(turns, ni)) \
                    or (w == "あー" and not _aa_fits(turns, ni)) \
                    or (w in mix and w not in wts):  # 合わない へえ・あー／タメ口の はい・ええ 系
                todo.append(ni)
                forced.add(ni)
            elif getattr(quality, "bc_mix_rebalance", True) \
                    and w in wts and count[w] > wts[w] * n \
                    and rng.random() < (count[w] - wts[w] * n) / count[w]:
                todo.append(ni)
        for ni in todo:
            t = turns[ni]
            # ★強制の置き換え（タメ口の話者の「はい・ええ」系、合わない位置の「へえ」）は、意味の
            #   同じ語へ決め打ちで替える。配分の重みで選ぶと「はい→あー」「ええ→うんうんうん」のように
            #   意味の違う語になっていた（2026-10-04 cmp_new 49対話で55件中の多数。ユーザー指摘
            #   「相槌の形式がたまに変」）。
            if ni in forced:
                fm = _FORCED_MAP.get(keys[ni]) or ("はい" if polite else "うん")
                if fm != keys[ni] and fm in wts:
                    tail = re.search(r"[。！？!?]*$", t.get("text", "")).group()
                    rf = dict(t.get("refine") or {})
                    rf.setdefault("orig_text", t["text"])
                    rf["op"] = ("bc_mix" if rf.get("op") in (None, "insert")
                                else rf["op"] + "+bc_mix")
                    t["refine"] = rf
                    t["text"] = fm + tail
                    keys[ni] = fm
                    changed.add(ni)
                    n_done += 1
                    continue
            cands = [w for w in wts
                     if deficit[w] > 0 and w != keys[ni]
                     and (w != "へえ" or _hee_fits(turns, ni))
                     and (w != "あー" or _aa_fits(turns, ni))
                     and not _duplicates_nearby_bc(w, listener, ni, turns, window,
                                                   exclude_index=ni)]
            if cands:
                repl = rng.choices(cands, weights=[deficit[w] for w in cands])[0]
            elif ni in forced:                       # 足りない語が無くても必ず替える
                cands = [w for w in wts if w not in (keys[ni], "へえ")
                         and not _duplicates_nearby_bc(w, listener, ni, turns, window,
                                                       exclude_index=ni)]
                if not cands:
                    continue
                repl = rng.choices(cands, weights=[wts[w] for w in cands])[0]
            else:
                continue
            deficit[repl] = max(0.0, deficit[repl] - 1)
            tail = re.search(r"[。！？!?]*$", t.get("text", "")).group()
            rf = dict(t.get("refine") or {})
            rf.setdefault("orig_text", t["text"])
            rf["op"] = ("bc_mix" if rf.get("op") in (None, "insert")
                        else rf["op"] + "+bc_mix")
            t["refine"] = rf
            t["text"] = repl + tail
            keys[ni] = repl
            changed.add(ni)
            n_done += 1
    return n_done


def _rewrite_quality_error(original: str, rewritten: str,
                           quality: QualityParams) -> str | None:
    """rewrite_text が「内容保持した口語化」の範囲にあるか検査する。"""
    orig = _text_core(original)
    new = _text_core(rewritten)
    if len(orig) < quality.min_rewrite_chars:
        return f"元発話が短すぎる（{len(orig)}<{quality.min_rewrite_chars}文字）"
    if not orig or not new:
        return "内容が空"
    length_ratio = len(new) / len(orig)
    if length_ratio > quality.max_rewrite_length_ratio:
        return (f"長さが増えすぎ（{length_ratio:.2f}>"
                f"{quality.max_rewrite_length_ratio:.2f}倍）")
    matched = sum(m.size for m in difflib.SequenceMatcher(
        None, orig, new, autojunk=False).get_matching_blocks())
    retention = matched / len(orig)
    if retention < quality.min_content_retention:
        return (f"元内容の保持率が低い（{retention:.2f}<"
                f"{quality.min_content_retention:.2f}）")
    anchors = _CONTENT_ANCHOR_RE.findall(
        unicodedata.normalize("NFKC", original).lower())
    missing = [a for a in anchors if a not in new]
    if missing:
        return f"数字・英字情報が欠落（{', '.join(missing[:3])}）"
    if bool(_NEGATION_RE.search(original)) != bool(_NEGATION_RE.search(rewritten)):
        return "否定表現が変化"
    if bool(_QUESTION_RE.search(original)) != bool(_QUESTION_RE.search(rewritten)):
        return "疑問文/平叙文が変化"
    return None


_FILLER_RE = re.compile(r"(?:あの|その|えーと|えっと|ええと|えー|まあ|うーん|なんか)(?=、)")
_FILLER_CHAIN_RE = re.compile(
    r"(?:あの|その|えーと|えっと|ええと|えー|まあ|うーん|なんか)、(?:あの|その|えーと|えっと|ええと|えー|まあ|うーん|なんか)、")
_TAIL_COMMA_RE = re.compile(r"、(?:す|すけど|すが|ね|か|よ)[。？！]?$")
_CASUAL_BC_RE = re.compile(r"うん|そっか|へー|へえ|なるほど|まじ|わかる|ふーん")


def _rewrite_style_error(original: str, rewritten: str,
                         quality: QualityParams) -> str | None:
    """rewrite_text が話し言葉として壊れていないか（接客向けの歯止め。既定は無効）。"""
    if quality.reject_ellipsis and re.search(r"[…‥]", rewritten) \
            and not re.search(r"[…‥]", original):
        return "「…」を追加"
    if quality.max_added_fillers >= 0:
        added = len(_FILLER_RE.findall(rewritten)) - len(_FILLER_RE.findall(original))
        if added > quality.max_added_fillers:
            return f"フィラーを足しすぎ（+{added}>{quality.max_added_fillers}）"
        if _FILLER_CHAIN_RE.search(rewritten) and not _FILLER_CHAIN_RE.search(original):
            return "フィラーの連続"
        for sent in re.split(r"(?<=[。？！])", rewritten):
            if _TAIL_COMMA_RE.search(sent) and sent not in original:
                return "語尾の直前に読点"
    return None


def _duplicates_nearby_bc(text: str, speaker: str, position: float,
                          turns: list, window: int, exclude_index: int | None = None) -> bool:
    """同一話者の前後 window 個の相槌に同じ表現があるか。"""
    if window <= 0:
        return False
    key = _normalize_bc_text(text)
    if not key:
        return True
    before, after = [], []
    for i, t in enumerate(turns):
        if i == exclude_index or t.get("type", "speech") != "backchannel" \
                or t.get("speaker") != speaker:
            continue
        item = (i, _normalize_bc_text(t.get("text", "")))
        if i < position:
            before.append(item)
        else:
            after.append(item)
    nearby = before[-window:] + after[:window]
    return any(other == key for _, other in nearby)


# --------------------------------------------------------------------------
# Pass A: タイムライン構築 + 挿入候補スキャン
# --------------------------------------------------------------------------

def build_timeline(turns: list, plans: list, wav_by_turn: dict, sr: int,
                   channel_of: dict) -> tuple[ScoreTimeline, float]:
    """Pass A の配置結果から 16kHz スコア用タイムラインを構築する。"""
    tl = ScoreTimeline(num_ch=2, sr_engine=sr)
    end_max = 0.0
    for p in plans:
        i = p["index"]
        start = float(p["start_sec"])
        tl.place(channel_of[turns[i]["speaker"]], start, wav_by_turn[i])
        end_max = max(end_max, start + len(wav_by_turn[i]) / sr)
    return tl, end_max


def snap_char_pos(text: str, rel: float, window: int = 8) -> int:
    """rel(0..1) 相当の文字位置を、±window 文字内の最寄りの句読点直後にスナップ。"""
    n = len(text)
    pos = int(round(rel * n))
    pos = max(0, min(n, pos))
    best = pos
    best_d = window + 1
    for k in range(max(1, pos - window), min(n, pos + window + 1)):
        if text[k - 1] in _PUNCT and abs(k - pos) < best_d:
            best, best_d = k, abs(k - pos)
    return best


class PhaseTimer:
    """1対話の内訳（どの段に何秒かかったか）を測る。

    Stage4 の律速がどこかは実測しないと分からない（fish 合成 / MaAI 推論 /
    refine の LLM / 配置・書き出し）。1対話ごとに1行出して、ログから
    そのまま集計できるようにする。
    """

    def __init__(self):
        self.t: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextlib.contextmanager
    def phase(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t)

    def summary(self) -> str:
        total = time.perf_counter() - self._t0
        known = sum(self.t.values())
        parts = " ".join(f"{k}={v:.1f}s" for k, v in
                         sorted(self.t.items(), key=lambda x: -x[1]))
        return (f"内訳: 合計={total:.1f}s {parts} "
                f"その他={max(0.0, total - known):.1f}s")


# 冒頭・締めの挨拶（insert.skip_greetings）。位置と語の両方で決める（対話途中の
# 「ありがとうございます」等は挨拶扱いしない）。
_OPEN_GREET_RE = re.compile(r"初めまして|はじめまして|よろしくお願い|申します|と言います|と申し|"
                            r"お会いできて|こんにちは")
_CLOSE_GREET_RE = re.compile(r"失礼します|失礼いたします|ありがとうございました|それでは|では、|"
                             r"また|楽しかった|お話できて|お話しできて|お会いできて|お疲れ様|さようなら")
_GREET_SPAN = 4   # 冒頭・末尾それぞれ、この数の speech ターンの中だけを見る


def greeting_turns(turns: list) -> set:
    """冒頭・締めの挨拶にあたる speech ターンの index の集合。

    冒頭: 最初の _GREET_SPAN 個の speech のうち、挨拶語を含む最後のものまで全部。
    締め: 最後の _GREET_SPAN 個の speech のうち、締めの語を含む最初のものから末尾まで全部。
    """
    sp = [i for i, t in enumerate(turns) if t.get("type", "speech") != "backchannel"]
    out = set()
    head = sp[:_GREET_SPAN]
    hits = [k for k, i in enumerate(head) if _OPEN_GREET_RE.search(turns[i].get("text", ""))]
    if hits:
        out.update(head[:hits[-1] + 1])
    tail = sp[-_GREET_SPAN:]
    hits = [k for k, i in enumerate(tail) if _CLOSE_GREET_RE.search(turns[i].get("text", ""))]
    if hits:
        out.update(tail[hits[0]:])
    return out


def main_speaker_channel(record: dict, channel_of: dict) -> int | None:
    """metadata.main_speaker_index が指す話者のチャネル。無い/解決できなければ None。"""
    md = record.get("metadata") or {}
    i = md.get("main_speaker_index")
    speakers = md.get("speakers") or []
    if not isinstance(i, int) or not 0 <= i < len(speakers):
        return None
    return channel_of.get((speakers[i] or {}).get("name"))


class InsertionScanner:
    """相槌の挿入候補を p_bc のローカルピークから求める。

    聞き手方向ごとに全編1回のスコアリング（計2パス）。
    ターン別の窓切り出しはコンテキスト20秒の再供給が嵩むため使わない。

    backend=vapbatch のときは score_long_many() を使い、コンテキスト付きチャンクを
    バッチで回す（対話全長で forward 約3600回 → 十数回）。
    """

    def __init__(self, retimer: MaaiRetimer, cfg: RefineConfig):
        self.retimer = retimer
        self.cfg = cfg

    def scan(self, timeline: ScoreTimeline, turns: list, plans: list,
             wav_by_turn: dict, sr: int, channel_of: dict,
             failed_turns: set, main_ch: int | None = None) -> list[dict]:
        """main_ch: 主話者のチャネル。渡すと、聞き手が主話者になる位置だけ insert.main_listener の
        上書きを当てる（呼び出し側は main_listener があるときだけ渡す。三人用の差し替えは受けない）。"""
        base_ins = self.cfg.insert
        if not base_ins.enabled:
            return []
        main_ins = base_ins.for_main_listener() if main_ch is not None else base_ins
        speakers_by_ch = {ch: sp for sp, ch in channel_of.items()}
        dur_total = timeline._len / MAAI_SR
        if dur_total <= 0:
            return []
        x0, x1 = timeline.slice(0.0, dur_total)
        scorer = self.retimer._bc_scorer()
        # c = 聞き手（相槌側）チャネル。引数は (話し手, 相槌側) の順。
        pairs = [(x1, x0), (x0, x1)]
        if hasattr(scorer, "score_long_many"):
            # backend=vapbatch: コンテキスト付きチャンクをバッチで回し、
            # 2方向でエンコーダ結果を使い回す（対話全長で forward が 3600 回 → 十数回）
            frames_by_c = scorer.score_long_many(pairs)
        else:
            frames_by_c = [scorer.score(*xs) for xs in pairs]
        pbc, times = {}, {}
        for c in (0, 1):
            frames = frames_by_c[c]
            times[c] = np.asarray([e for e, _ in frames], dtype=float) / MAAI_SR
            pbc[c] = np.asarray(
                [self.retimer._bc_signal(fr) or 0.0 for _, fr in frames])

        # 既存相槌の打点（聞き手chごと）: 距離制約に使う
        start_by_idx = {p["index"]: float(p["start_sec"]) for p in plans}
        bc_times = {0: [], 1: []}
        for i, t in enumerate(turns):
            if t.get("type", "speech") == "backchannel" and i in start_by_idx:
                bc_times[channel_of[t["speaker"]]].append(start_by_idx[i])

        greet = greeting_turns(turns) if base_ins.skip_greetings else set()
        slots = []
        for p in plans:
            i = p["index"]
            t = turns[i]
            if t.get("type", "speech") == "backchannel" or i in failed_turns:
                continue
            if i in greet:
                continue
            # ★2026-11-10 重なりの場面（rp_v1）の言いかけ・重ねたターンには相槌を挿まない（分けると
            #   言いかけの音声を作り直すことになり、切った位置が失われる）
            if t.get("unsaid") or t.get("overlap"):
                continue
            # ★2026-11-13 rp_v1 試走13: 相手が先に話し始める同時発話（相手の発話のすぐ後に、聞き手の言いかけ "collision" が続く）
            #   では、相手の発話に挿んだ聞き手の相槌が言いかけより前に並び、言いかけが相槌のあと（相手の発話の末尾）まで
            #   押し出された（6中2）。すぐ後の speech が同時発話の言いかけなら挿まない。
            nxt = next((turns[k] for k in range(i + 1, len(turns))
                        if turns[k].get("type", "speech") != "backchannel"), None)
            if nxt is not None and nxt.get("overlap") == "collision":
                continue
            sp_ch = channel_of[t["speaker"]]
            listener_ch = 1 - sp_ch
            ins = main_ins if listener_ch == main_ch else base_ins
            d = len(wav_by_turn[i]) / sr
            if d < ins.min_turn_sec:
                continue
            # 聞き手が原本でこのターン以前に発話済み（チャネル順の保護）
            if not any(turns[j]["speaker"] != t["speaker"] for j in range(i)):
                continue
            s0 = start_by_idx[i] + ins.edge_margin_sec[0]
            s1 = start_by_idx[i] + d - ins.edge_margin_sec[1]
            if s1 <= s0:
                continue
            m = (times[listener_ch] >= s0) & (times[listener_ch] <= s1)
            seg_p, seg_t = pbc[listener_ch][m], times[listener_ch][m]
            if not len(seg_p):
                continue
            # 閾値以上のローカル最大値を p 降順で採用（間隔・既存距離・件数の制約つき）
            cand = []
            for k in range(len(seg_p)):
                if seg_p[k] < ins.threshold:
                    continue
                lo, hi = max(0, k - 1), min(len(seg_p), k + 2)
                if seg_p[k] >= seg_p[lo:hi].max():
                    cand.append((float(seg_p[k]), float(seg_t[k])))
            cand.sort(reverse=True)
            chosen = []
            for pv, tv in cand:
                if len(chosen) >= ins.max_per_turn:
                    break
                if any(abs(tv - cv) < ins.min_separation_sec for _, cv in chosen):
                    continue
                if any(abs(tv - bt) < ins.min_dist_existing_sec
                       for bt in bc_times[listener_ch]):
                    continue
                chosen.append((pv, tv))
            for pv, tv in sorted(chosen, key=lambda x: x[1]):
                rel = (tv - start_by_idx[i]) / d
                cp = snap_char_pos(t["text"], rel)
                head, tail = t["text"][:cp], t["text"][cp:]
                if ins.min_head_chars and len(_text_core(head)) < ins.min_head_chars:
                    continue
                if ins.skip_questions and re.split(r"(?<=[。？！?!])", tail)[0].rstrip().endswith(("？", "?")):
                    continue
                slots.append({
                    "slot_id": len(slots) + 1,
                    "anchor_index": i,
                    "rel_pos": round(rel, 3),
                    "t_peak": round(tv, 2),
                    "p": round(pv, 3),
                    "char_pos": snap_char_pos(t["text"], rel),
                    "listener": speakers_by_ch[listener_ch],
                })
        return slots


class MockScanner:
    """--dry-run 用: 最初の eligible speech ターン中央に slot を1つ置く（決定的）。"""

    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg

    def scan(self, timeline, turns, plans, wav_by_turn, sr, channel_of,
             failed_turns, main_ch=None) -> list[dict]:
        speakers_by_ch = {ch: sp for sp, ch in channel_of.items()}
        for p in plans:
            i = p["index"]
            t = turns[i]
            if t.get("type", "speech") == "backchannel" or i in (failed_turns or set()):
                continue
            d = len(wav_by_turn[i]) / sr
            if d < self.cfg.insert.min_turn_sec:
                continue
            if not any(turns[j]["speaker"] != t["speaker"] for j in range(i)):
                continue
            return [{"slot_id": 1, "anchor_index": i, "rel_pos": 0.5,
                     "t_peak": round(float(p["start_sec"]) + d / 2, 2), "p": 0.5,
                     "char_pos": snap_char_pos(t["text"], 0.5),
                     "listener": speakers_by_ch[1 - channel_of[t["speaker"]]]}]
        return []


# --------------------------------------------------------------------------
# Gemini プロンプト
# --------------------------------------------------------------------------

def build_prompt(record: dict, turns: list, annotations: dict, slots: list,
                 duration_sec: float, cur_bc_per_min: float,
                 cfg: RefineConfig) -> str:
    md = record.get("metadata", {})
    spk_lines = []
    for s in md.get("speakers", []):
        spk_lines.append(f"  {s.get('name')}（{s.get('role', '')}, "
                         f"{s.get('personality', '')}, {s.get('age_range', '')}）")
    slots_by_anchor: dict[int, list] = {}
    for s in slots:
        slots_by_anchor.setdefault(s["anchor_index"], []).append(s)

    n_speech = sum(1 for t in turns if t.get("type", "speech") != "backchannel")
    n_bc = len(turns) - n_speech
    eligible_rewrites = sum(
        1 for t in turns
        if t.get("type", "speech") != "backchannel"
        and len(_text_core(t.get("text", ""))) >= cfg.quality.min_rewrite_chars)
    max_rewrites = (min(eligible_rewrites,
                        max(1, int(cfg.limits.max_rewrite_frac * n_speech)))
                    if eligible_rewrites else 0)
    max_rewords = (max(1, int(cfg.limits.max_reword_frac * n_bc))
                   if n_bc else 0)
    bc_counts = Counter(
        _normalize_bc_text(t.get("text", ""))
        for t in turns if t.get("type", "speech") == "backchannel")
    bc_counts.pop("", None)
    bc_summary = ", ".join(f"{text}×{count}"
                           for text, count in bc_counts.most_common(8)) or "なし"

    lines = []
    for i, t in enumerate(turns):
        ttype = t.get("type", "speech")
        ann = annotations.get(i, {})
        if ttype == "backchannel":
            pk = ann.get("p_peak")
            tag = f"相槌 p_peak={pk:.2f}" if pk is not None else "相槌"
            if _shorten_bc(t.get("text", ""), cfg.quality.bc_shorten_len):
                tag += ", ⚠長い"  # 短縮候補（2語以上の連結）
        else:
            tag = "speech"
            if t.get("continued"):
                tag += ", continued"
        text = t["text"]
        # 挿入候補マーカーを右から埋め込む（文字位置を保つ）
        for s in sorted(slots_by_anchor.get(i, []), key=lambda x: -x["char_pos"]):
            cp = s["char_pos"]
            # ★2026-11-07 傾聴の案内（bc_vocab_style: listener）では、文を言い終えた直後の候補に「文末」と書く。
            #   試走8で LLM が「へえ」を「これが、｜」「味は、｜」のような言いかけの途中に置き、後処理（_hee_fits）で
            #   23個が「はい」に替わった。位置を LLM に見えるようにする（語を替える規則は足さない）。
            # ★2026-11-08 オフラインの比較（同じ候補で新旧の案内）: 新しい案内でも「そうなんです。｜」「そうですね。｜」
            #   「ありがとうございます。｜」の直後に「へえ」が9個（LLM は〔◆〕の後ろの文に反応していた）。中身のある文
            #   （実質8字以上の平叙文で、挨拶・礼でない）を言い終えた位置だけに「文末」と書く（_hee_fits と同じ線引き）。
            end = (cfg.quality.bc_vocab_style == "listener"
                   and _content_sentence_end(t["text"][:cp]))
            text = text[:cp] + f"〔◆{s['slot_id']} p={s['p']:.2f}{' 文末' if end else ''}〕" + text[cp:]
        lines.append(f"[{i}] {t['speaker']} ({tag}): {text}")
    dialogue_block = "\n".join(lines)

    mix = _mix_weights(cfg.quality)
    if mix and cfg.quality.bc_vocab_style == "listener":
        # ★2026-11-08 試走8の視聴「あー・おー・へー は出るようになったが、話の中身に応じてピンポイントで使う語で、
        #   タイミングを外すと印象が悪い。迷ったら汎用の うん・はい に」。試走8の案内は目安の割合（あー15・へえ12・
        #   おー8%）を示して「この割合に近づけ」と書いており、割合を満たすために中身の無い位置（「その、新｜しい薬」
        #   「行ってきたんですけ｜ど」）にも置いていた。人間の傾聴（ERICA の聞き手）は うん60・はい18・あー8.5・
        #   へー1・おー0.1%。割合の目標をやめ、既定を うん・はい にし、反応の語は条件を満たす位置だけにする。
        #   直前のきっかけの言葉を "cue" に書かせる（書けない位置では使わない。refine_ops に残るので検証にも使う）。
        conts = [w for w in sorted(mix, key=lambda k: -mix[k]) if w not in _REACTIVE_BC]
        reacts = [w for w in ("あー", "へえ", "おー") if w in mix]
        insert_vocab = (
            "  挿入相槌は相手の発話の**途中**に置かれる。基本は聞いていることを示す"
            f"{''.join(f'「{w}」' for w in conts)}で、\n"
            "  丁寧に話す聞き手でも「うん」「うんうん」を多めに使ってよい（「ええ」は控えめ）。タメ口の話者には「はい」「ええ」系を使わない。\n"
            f"  {''.join(f'「{w}」' for w in reacts)}は相手の話の中身にピンポイントで反応する語で、"
            "合わない位置に置くと話を聞いていないように聞こえる。\n"
            "  相槌は〔◆〕の位置で鳴る。〔◆〕より後ろ（右側）の言葉は、相槌の時点では相手がまだ言っていない。\n"
            "  次の3つを全部満たす〔◆〕だけに使い、少しでも迷ったら「うん」「はい」にする:\n"
            "  (a) 〔◆〕のすぐ左で、相手がその語を引き出す中身（きっかけ）を言い終えている。〔◆〕のすぐ左が\n"
            "      フィラー（あの・その・まあ・えーと・なんか）、つなぎの語（でも・それで・〜けど・〜ので）、\n"
            "      「〜が、」「〜は、」「〜を、」のような言いかけ、ターンの頭の受け答え（そうなんです。そうですね。\n"
            "      はい。ええ。ありがとうございます。）だけなら、きっかけはまだ言い終えていない。\n"
            "  (b) きっかけと語が合っている。\n"
            "      「へえ」: 聞き手が知らなかった事実（何をしている・何があった・どうなった）を、相手が1文言い終えた直後\n"
            "      （「文末」と付いた〔◆〕）。相手の感想・気持ち（〜がすごくて・〜がたまらない）や謙遜には使わない。\n"
            "      「おー」: 相手が何かをやり遂げた・うまくいった・すごいことを言い終えた直後。\n"
            "      「あー」: 相手の困りごと・気持ち・事情を聞いて「わかる」と思える中身を言い終えた直後\n"
            "      （「手が震えて、」のように話が続く形でも、困りごとや気持ちを言い終えていればよい）。\n"
            "      失敗・痛み・困りごと・悲しい話に「へえ」「おー」は使わない。ふつうの出来事・前置きに「おー」は使わない。\n"
            "  (c) 反応の語の insert には \"cue\" を付け、〔◆〕のすぐ左にある相手の言葉（きっかけ）を、〔◆〕の直前で終わる形で\n"
            "      そのまま5〜15字写す。〔◆〕より後ろの言葉は写さない。写せる言葉が無いなら、反応の語にしない。\n"
            "  反応の語は合わせて挿入の2割までにする。1つも使わない対話があってもよい。数を合わせるために使わない。\n"
            "  「なるほど」「そうなんだ」「たしかに」等の内容を評価する反応は、話がまだ完結していない途中では\n"
            "  末尾への反応に聞こえて不自然なので挿入には使わない。\n")
    elif mix:
        words = "".join(f"「{w}」" for w in sorted(mix, key=lambda k: -mix[k]))
        insert_vocab = (
            "  挿入相槌は相手の発話の**途中**に置かれるため、聞いていることを示す短い語にする。\n"
            f"  使う語: {words}。実際の会話に近い配分で使い分け、同じ語ばかりにしない。\n"
            "  「あー」は相手の話が腑に落ちた・思い当たったとき、「へえ」は直前に相手が新しい事実を\n"
            "  1文言い終えた位置だけに使う。「ええ」は控えめにし、タメ口の話者には「はい」「ええ」系を使わない。\n"
            "  「なるほど」「そうなんだ」「たしかに」等の内容を評価する反応は、話がまだ完結していない途中では\n"
            "  末尾への反応に聞こえて不自然なので挿入には使わない。\n")
    else:
        insert_vocab = (
            "  挿入相槌は相手の発話の**途中**に置かれるため、原則「うん」「うんうん」「はい」「ええ」等の\n"
            "  **継続支持（聞いているよ、という相づち）**にする。「なるほど」「へえ」「そうなんだ」「たしかに」等の\n"
            "  内容を理解・評価する反応は、話がまだ完結していない途中では末尾への反応に聞こえて不自然なので\n"
            "  挿入には使わない（そうした反応は発話が終わった後の相槌に回す）。\n")

    d = cfg.density
    if d.count_hint:
        minutes = max(duration_sec / 60.0, 1e-6)
        count_hint = (
            f"target_bc_per_min は1分あたりの回数で、総数ではありません。この対話は {minutes:.1f} 分なので、"
            f"{d.min_bc_per_min:g}回/分なら相槌は合計約{round(d.min_bc_per_min * minutes)}個、"
            f"{d.max_bc_per_min:g}回/分なら約{round(d.max_bc_per_min * minutes)}個です（既存の相槌 {n_bc} 個を含む）。"
            f"挿入候補〔◆〕は {len(slots)} 個あります。決めた密度の数だけ insert してください。\n")
        weak_rule = ("強い候補から使う。数が足りないときは、弱い候補も意味の切れ目（読点・文末の近く）にあれば使う。"
                     "\n  この対話は聞き手が相槌を多く打つ場面なので、弱いことだけを理由に外さない。")
    else:
        count_hint = ""
        weak_rule = ("弱い候補は\n  文脈上どうしても自然な場合だけ使う。目標密度を満たすためだけに弱い候補を採用しない。")
    listener_vocab = bool(mix) and cfg.quality.bc_vocab_style == "listener"
    bc_eg = "「うん」「はい」" if listener_vocab else "「うん」「へえ」"
    cue_fmt = ("\n反応の語（あー・へえ・おー）の insert だけ \"cue\" を付ける: "
               "{\"op\": \"insert\", \"slot\": 2, \"new_text\": \"（反応の語）。\", "
               "\"cue\": \"（〔◆2〕の直前の相手の言葉）\"}" if listener_vocab else "")
    return f"""あなたは日本語音声対話コーパスの品質向上を行う編集者です。以下の対話の相槌とフィラーを、メタデータと音響モデル（MaAI: ターンテイキング・相槌タイミング予測）の注釈に基づいて編集してください。

# 対話メタデータ
ジャンル: {md.get('genre', '不明')} / 関係性: {md.get('relationship', '不明')} / 話し方: {md.get('speech_style', '不明')} / トーン: {md.get('emotional_tone', '不明')}
シナリオ: {md.get('scenario', '')}
話者:
{chr(10).join(spk_lines)}

# 1. 相槌密度の決定
まずこの対話に相応しい相槌密度を決めてください:
- label: "dense" / "normal" / "sparse"
- target_bc_per_min: {d.min_bc_per_min}〜{d.max_bc_per_min} の範囲の数値
- reason: 根拠を1文（親密な関係・砕けた話し方・感情的な話題→dense、ビジネス敬語・フォーマルな場面・落ち着いた相談→sparse、など）
相槌は{bc_eg}等の短い反応なので、多めに入っても自然です。全体に相槌は多めを
基本としつつ、カジュアル・親密・感情的な場面は積極的に多く、フォーマル・重い相談・年配の落ち着いた
場面は控えめに、と**内容に応じて**判断してください。
現在の密度は {cur_bc_per_min:.1f} 回/分（対話長 {duration_sec:.0f} 秒、既存相槌 {n_bc} 個、speech {n_speech} 発話）。
現在の相槌頻度上位: {bc_summary}
{count_hint}
# 編集の優先順位
1. 元の意味・事実・話者らしさを完全に保つ
2. 文脈に合う自然な反応にする
3. 長すぎる相槌を短くする（相槌は本来短い。「⚠長い」印の相槌は短い1語に縮める）
4. 不自然な相槌の過密・不足を直す
5. 同じ相槌の機械的な反復を減らす

密度や多様性はノルマではありません。音響スコアが高くても、意味の切れ目でなければ挿入せず、
編集に確信がなければそのターンは変更しないでください。珍しい表現へ無理に言い換えるより、
自然な「はい」「うん」を残す方を優先します。

# 2. 対話本文（[index] 話者 (種別): テキスト）
〔◆n p=..〕は MaAI が「ここで聞き手の相槌が音響的に自然」と予測した挿入候補位置（p が高いほど自然）。
(相槌 p_peak=..) は既存相槌の音響的な自然さ。低いものは削除候補として優先してよい。

{dialogue_block}

# 3. 編集操作
- reword: 相槌の語を言い換える。現状「はい」「うん」に偏っているので、話し方・関係性・文脈に合う多様な語に（敬語→「ええ」「そうですか」「なるほど」「そうなんですね」等、タメ口→「うん」「へー」「まじで」「そうなんだ」「わかる」等。同じ語の連続を避ける）
- reword は最大 {max_rewords} 個。同一話者の前後{cfg.quality.duplicate_bc_window}回の相槌と、句読点を除いて同じ表現にしない。
  「なるほどですね」「ええ、なるほど」のような反応語の機械的な連結を避け、直前の内容に対する
  受理・驚き・同意・共感のどれが自然かを選ぶ。敬語レベルを変えない。
- 「⚠長い」印の相槌（例「なるほど、分かりました」「ええ、そうでしたね」）は、reword で
  短い相槌1語（例「なるほど」「ええ」「そうですね」）に**縮める**。相槌が長いと発話に
  被って不自然になるため、2語以上の反応の連結は避け、短く即応する形にする。
- delete: 相槌を削除（sparse化。p_peak が低いものを優先）
- insert: 〔◆n〕の位置に相槌を新規挿入（dense化。slot 番号で指定）。候補位置の直前までで
  意味が成立している場合だけ使い、後続を先取りする反応は書かない。同一話者の近傍相槌と同じ語を避ける。
{insert_vocab}  p>=0.60 は強い候補、0.50〜0.59 は中程度、p<0.50 は弱い候補として扱い、{weak_rule}
- rewrite_text: speech 発話の全文を書き直し、必要な箇所だけフィラーや言い直し・言い淀みを
  自然に織り込む。最大 {max_rewrites} 発話まで。元が {cfg.quality.min_rewrite_chars} 文字未満なら使用しない。
  元の内容語・数字・固有名詞・否定・時制を同じ順序で全て残し、要約・言い換え・情報追加をしない。
  長さは元の {cfg.quality.max_rewrite_length_ratio:.1f} 倍以内。「えーと」を各発話へ一律に足さない。
- merge: 相槌を delete して分割が不要になった場合、[index]（前半）と "with"（continued の後半）を1発話に結合
- 指定しないターンはそのまま維持される（keep は書かない）

制約:
- 話者名をテキストに含めない。新しい情報・話題を追加しない
- insert は〔◆〕のある位置のみ。相槌テキストは短く（句読点を除き原則6文字以内）
- 「ふんふん」は使わない（音声合成が笑い声にしてしまう）。同じ意味なら「うんうん」を使う
- 分割の間の相槌（[k] speech → [k+1] 相槌 → [k+2] continued）を delete する場合、できれば merge も指定
- target_bc_per_min と実際の増減（削除・挿入数）を整合させる

# 4. 出力形式（JSONのみ。説明文は書かない）
{{"density": {{"label": "normal", "target_bc_per_min": 4.5, "reason": "..."}},
 "ops": [
   {{"op": "reword", "index": 4, "new_text": "ええ。"}},
   {{"op": "delete", "index": 10}},
   {{"op": "insert", "slot": 1, "new_text": "うんうん。"}},
   {{"op": "rewrite_text", "index": 3, "new_text": "あの、えーと、…"}},
   {{"op": "merge", "index": 9, "with": 11}}
 ]}}{cue_fmt}"""


def _react_items(turns: list, ops: list) -> list:
    """ops の insert のうち、反応の語（あー・へえ・おー）のもの [(op, 相手の発話の相槌位置まで, 直前の聞き手の発話)]。"""
    items = []
    for o in ops:
        if o.get("op") != "insert":
            continue
        if _mix_key(_normalize_bc_text(o.get("new_text", ""))) not in _REACTIVE_BC:
            continue
        sl = o["_slot"]
        i = sl["anchor_index"]
        head = turns[i]["text"][:sl["char_pos"]].rstrip()
        prev = next((turns[j]["text"] for j in range(i - 1, -1, -1)
                     if turns[j].get("type", "speech") != "backchannel"
                     and turns[j]["speaker"] != turns[i]["speaker"]), "")
        items.append((o, head, prev))
    return items


def build_react_verify_prompt(record: dict, items: list) -> str:
    md = record.get("metadata", {})
    lines = []
    for k, (o, head, prev) in enumerate(items, 1):
        q = f"（直前の聞き手: 「{prev}」）" if prev else ""
        lines.append(f"[{k}] {q}相手: 「{head}｜」 → 聞き手の相槌: {o['new_text']}")
    return f"""あなたは日本語の音声対話コーパスの編集者です。聞き手が相手の話の途中に打つ相槌のうち、話の中身に反応する語（あー・へえ・おー）が、その瞬間に合っているかを1つずつ確かめてください。
相槌は「｜」の位置で鳴ります。相手はまだ話の途中で、「｜」より後ろは相槌の時点ではまだ言っていません。

# 合っている条件（全部満たすときだけ ok）
- 「｜」のすぐ左で、相手がその語を引き出す中身を言い終えている。すぐ左がフィラー（あの・その・まあ・えーと・なんか）、
  つなぎの語（でも・それで・〜けど・〜ので）、「〜が、」「〜は、」のような言いかけ、ターンの頭の受け答え
  （そうなんです。そうですね。はい。）だけなら合わない。「〜ので。」「〜んですけど。」のように理由や前置きで
  終わる文は、句点で切れていても、まだ話の中身を言い終えていない。
- 「へえ」: 聞き手が知らなかった事実（何をしている・何があった・どうなった）を、相手が1文言い終えた直後。
  相手の感想・気持ち・謙遜・困りごと・失敗・痛みには合わない。
- 「おー」: 相手が何かをやり遂げた・うまくいった・すごいことを言い終えた直後。
- 「あー」: 相手の困りごと・気持ち・事情を聞いて「わかる」と思える中身を言い終えた直後
  （「手が震えて、」のように話が続く形でも、困りごとや気持ちを言い終えていればよい）。
少しでも合わないところがあれば ok にせず、聞いていることを示す「うん」「うんうん」「はい」「はいはい」のうち、
聞き手の話し方（{md.get('speech_style', '不明')}）に合う語を new_text に書いてください。

# 確かめる相槌
{chr(10).join(lines)}

# 出力形式（JSONのみ。説明文は書かない）
{{"items": [{{"id": 1, "ok": true}}, {{"id": 2, "ok": false, "new_text": "うん。"}}]}}"""


_REACT_FALLBACK = ("うん", "うんうん", "はい", "はいはい")


def verify_reactive_inserts(refiner, did: str, record: dict, turns: list, ops: list) -> int:
    """反応の語の insert を LLM に確かめさせ、合わないものを LLM の選んだ うん・はい 系に替える（替えた数を返す）。
    確かめられなかったとき（呼び出し失敗・JSON 不正）は替えない。"""
    items = _react_items(turns, ops)
    if not items:
        return 0
    try:
        res = refiner.propose(f"{did}.react", build_react_verify_prompt(record, items))
    except Exception as e:                      # 確かめられないときは LLM の元の選択のまま
        logger.warning(f"[{did}] 反応の語の確認に失敗（元の語のまま）: {e}")
        return 0
    verdict = {}
    for it in (res.get("items") or []) if isinstance(res, dict) else []:
        if isinstance(it, dict) and isinstance(it.get("id"), int):
            verdict[it["id"]] = it
    n = 0
    for k, (o, _head, _prev) in enumerate(items, 1):
        v = verdict.get(k)
        if not v or v.get("ok") is not False:
            continue
        new = str(v.get("new_text") or "")
        if _normalize_bc_text(new) not in _REACT_FALLBACK:
            new = "うん。"
        o["react_orig"] = o["new_text"]
        o["new_text"] = new if new.endswith("。") else new + "。"
        n += 1
    return n


# --------------------------------------------------------------------------
# Gemini リファイナ
# --------------------------------------------------------------------------

class BaseRefiner:
    """編集 ops を得るリファイナの基底。ops キャッシュ・リトライ・JSONパースを共通化し、
    プロバイダ固有の resolve_model / _generate だけをサブクラスが実装する。"""

    provider = "base"

    def __init__(self, cfg: RefineConfig, variant: str):
        self.cfg = cfg
        self.ops_dir = Path(cfg.ops_dir) / variant
        self.ops_dir.mkdir(parents=True, exist_ok=True)
        self.model: str | None = None
        from generate_tts import _RateLimiter
        self._limiter = _RateLimiter(cfg.rpm)

    def resolve_model(self) -> str:
        raise NotImplementedError

    def _generate(self, prompt: str, model: str) -> str:
        """1回のテキスト生成。生の応答文字列（JSON想定）を返す。"""
        raise NotImplementedError

    def propose(self, dialogue_id: str, prompt: str,
                turns=None, slots=None) -> dict:
        """ops JSON を返す。パース不能は ValueError（呼び出し側がリトライ判断）。"""
        from generate_tts import GeminiSynth
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]
        cache_path = self.ops_dir / f"{dialogue_id}.{h}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                if "result" in cached:
                    logger.info(f"[{dialogue_id}] ops キャッシュ使用 ({h})")
                    return cached["result"]
            except json.JSONDecodeError:
                pass
        model = self.resolve_model()
        raw, last_err = "", None
        for attempt in range(self.cfg.max_retries + 1):
            self._limiter.acquire()
            try:
                raw = self._generate(prompt, model)
                break
            except Exception as e:
                last_err = e
                wait = GeminiSynth._retry_delay_hint(
                    e, default=2.0 * (2 ** attempt), cap=60.0)
                logger.warning(f"[{dialogue_id}] {self.provider} 呼び出し失敗"
                               f"（{attempt + 1}回目）: {type(e).__name__} → {wait:.0f}s待機")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{self.provider} ops 生成に失敗: {last_err}")
        try:
            # ローカルLLM（Qwen3等）が <think> や地の文を混ぜても JSON を頑健に抽出
            result = json.loads(extract_json_text(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.provider} 出力が JSON として不正: {e}（先頭: {raw[:80]!r}）")
        cache_path.write_text(json.dumps(
            {"prompt_hash": h, "model": model, "provider": self.provider,
             "result": result}, ensure_ascii=False, indent=1))
        return result


class GeminiRefiner(BaseRefiner):
    """Gemini（google-genai）テキスト生成で編集 ops を得る。"""

    provider = "gemini"

    def __init__(self, cfg: RefineConfig, variant: str):
        super().__init__(cfg, variant)
        self._client = None

    def client(self):
        if self._client is None:
            from google import genai
            key = (os.environ.get(self.cfg.api_key_env)
                   or os.environ.get("GOOGLE_API_KEY"))
            if not key:
                raise RuntimeError(
                    f"Gemini APIキーが見つかりません（{self.cfg.api_key_env} / GOOGLE_API_KEY）")
            self._client = genai.Client(api_key=key)
        return self._client

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        for name in (self.cfg.text_model, self.cfg.fallback_text_model):
            try:
                self.client().models.get(model=name)
                self.model = name
                break
            except Exception as e:
                logger.warning(f"Gemini モデル {name} が利用できません: {type(e).__name__}")
        if not self.model:
            raise RuntimeError("利用可能な Gemini テキストモデルがありません"
                               f"（{self.cfg.text_model} / {self.cfg.fallback_text_model}）")
        logger.info(f"Gemini テキストモデル: {self.model}")
        return self.model

    def _generate(self, prompt: str, model: str) -> str:
        from google.genai import types
        resp = self.client().models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.cfg.temperature,
                response_mime_type="application/json"))
        return resp.text or ""


class OpenAIRefiner(BaseRefiner):
    """OpenAI（ChatGPT）テキスト生成で編集 ops を得る。gpt-5.6-terra 等の新モデルは
    temperature を既定値しか受け付けないため、非対応なら自動で temperature を外す。"""

    provider = "openai"

    def __init__(self, cfg: RefineConfig, variant: str):
        super().__init__(cfg, variant)
        self._no_temp = False
        # 投げ先候補（空リスト=クラウド ChatGPT）。base_urls（複数）を優先し、
        # 無ければ base_url（単数）を1件だけの候補とする。
        # 優先順位は base_urls > hosts（マシン名から探す）> base_url。
        if cfg.base_urls:
            urls = list(cfg.base_urls)
        elif cfg.hosts:
            from llm_client import discover_llm_endpoints
            urls = discover_llm_endpoints(cfg.hosts, cfg.text_model, cfg.ports,
                                          where="tts.refine")
        else:
            urls = [cfg.base_url] if cfg.base_url else []
        self._endpoints = [u for u in urls if u]
        # base_url("" = クラウド) → 生成済みクライアント。エンドポイントごとに1つだけ作る。
        self._clients: dict = {}
        self._least_loaded = cfg.endpoint_pick == "least_loaded"
        if len(self._endpoints) > 1:
            how = "いちばん空いている所へ（/metrics）" if self._least_loaded else "ランダム分散"
            logger.info(f"OpenAI refine: {len(self._endpoints)}エンドポイントへ"
                        f"リクエストを{how} {self._endpoints}")

    def client(self):
        # 複数エンドポイントがあればリクエストのたびに1つ選ぶ。既定はいちばん空いているサーバ
        # （tts.refine.endpoint_pick: random で従来のランダム）。
        if len(self._endpoints) > 1 and self._least_loaded:
            from llm_client import pick_least_loaded
            url = pick_least_loaded(self._endpoints)
        else:
            url = random.choice(self._endpoints) if self._endpoints else ""
        c = self._clients.get(url)
        if c is None:
            from openai import OpenAI
            key = (os.environ.get(self.cfg.api_key_env)
                   or os.environ.get("OPENAI_API_KEY"))
            if url:
                # ローカルの OpenAI 互換サーバ(vLLM等)は通常 APIキー不要 → ダミー可
                c = OpenAI(api_key=key or "EMPTY", base_url=url)
            else:
                if not key:
                    raise RuntimeError(
                        "OpenAI APIキーが見つかりません"
                        f"（{self.cfg.api_key_env} / OPENAI_API_KEY）")
                c = OpenAI(api_key=key)
            self._clients[url] = c
        return c

    def _raw_call(self, model: str, prompt: str, use_temp: bool) -> str:
        kwargs = dict(model=model,
                      messages=[{"role": "user", "content": prompt}],
                      response_format={"type": "json_object"})
        if use_temp:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        r = self.client().chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        probe = 'JSONで {"ok": true} だけ返して'
        for name in (self.cfg.text_model, self.cfg.fallback_text_model):
            for use_temp in (True, False):  # temperature 非対応モデルは外して再試行
                try:
                    self._raw_call(name, probe, use_temp)
                    self.model, self._no_temp = name, (not use_temp)
                    logger.info(f"OpenAI テキストモデル: {name}"
                                + ("（temperature非対応→既定値）" if self._no_temp else ""))
                    return self.model
                except Exception as e:
                    if "temperature" in str(e).lower() and use_temp:
                        continue
                    logger.warning(f"OpenAI モデル {name} が利用できません: {type(e).__name__}")
                    break
        raise RuntimeError("利用可能な OpenAI テキストモデルがありません"
                           f"（{self.cfg.text_model} / {self.cfg.fallback_text_model}）")

    def _generate(self, prompt: str, model: str) -> str:
        return self._raw_call(model, prompt, use_temp=not self._no_temp)


def make_refiner(cfg: RefineConfig, variant: str) -> BaseRefiner:
    """provider に応じたリファイナを返す。"""
    if cfg.provider == "openai":
        return OpenAIRefiner(cfg, variant)
    return GeminiRefiner(cfg, variant)


class MockRefiner:
    """--dry-run 用: ネットワーク不要の決定的 ops。
    先頭の相槌を reword、2番目を delete、slot があれば insert、
    最初の eligible speech を rewrite（'えーと、' 前置）。"""

    model = "mock"

    def __init__(self, cfg: RefineConfig, variant: str = ""):
        self.cfg = cfg

    def resolve_model(self):
        return self.model

    def propose(self, dialogue_id: str, prompt: str,
                turns=None, slots=None) -> dict:
        turns = turns or []
        ops = []
        bc_idx = [i for i, t in enumerate(turns)
                  if t.get("type", "speech") == "backchannel"]
        if bc_idx:
            ops.append({"op": "reword", "index": bc_idx[0], "new_text": "なるほど。"})
        if len(bc_idx) >= 2:
            ops.append({"op": "delete", "index": bc_idx[1]})
        for s in (slots or [])[:1]:
            ops.append({"op": "insert", "slot": s["slot_id"], "new_text": "うんうん。"})
        for i, t in enumerate(turns):
            if t.get("type", "speech") != "backchannel" and len(t["text"]) > 20:
                ops.append({"op": "rewrite_text", "index": i,
                            "new_text": "えーと、" + t["text"]})
                break
        return {"density": {"label": "normal", "target_bc_per_min": 4.0,
                            "reason": "mock"}, "ops": ops}


# --------------------------------------------------------------------------
# ops 検証（決定的ガード）
# --------------------------------------------------------------------------

def protected_speakers(names: list, metadata: dict | None) -> set:
    """quality.no_rewrite_speakers を話者名の集合にする。

    "@staff" は metadata の role が「担当者」の話者（接客で担当者の名前が対話ごとに違う版用。
    generate_dialogues.staff_names と同じ判定）。
    """
    out = {str(n) for n in names or [] if not str(n).startswith("@")}
    if "@staff" in (names or []):
        out |= {str(sp.get("name")) for sp in ((metadata or {}).get("speakers") or [])
                if isinstance(sp, dict) and sp.get("role") == "担当者" and sp.get("name")}
    return out


# build_prompt が本文に埋め込む挿入候補マーカー（〔◆{slot_id} p={p:.2f}〕）。
_SLOT_MARK_RE = re.compile(r"〔◆[^〕]*〕")


def validate_ops(raw: dict, turns: list, slots: list, duration_sec: float,
                 annotations: dict, cfg: RefineConfig,
                 speaker_names: list[str], protected: set | None = None):
    """Gemini ops を検証・正規化する。

    protected は書き換えさせない話者名（protected_speakers。None なら
    quality.no_rewrite_speakers をそのまま名前として使う）。
    Returns: (clean_ops, density, dropped: list[str], fatal: str|None)
    fatal が非 None のときは JSON 再生成（リトライ）を推奨する。
    """
    dropped: list[str] = []
    prot = protected if protected is not None else protected_speakers(
        cfg.quality.no_rewrite_speakers, None)
    if not isinstance(raw, dict):
        return [], None, [], "出力が JSON オブジェクトではない"
    density = raw.get("density")
    if not isinstance(density, dict) \
            or not isinstance(density.get("target_bc_per_min"), (int, float)):
        return [], None, [], "density.target_bc_per_min（数値）が無い"
    if not isinstance(raw.get("ops"), list):
        return [], None, [], "ops 配列が無い"
    d = cfg.density
    target = float(density["target_bc_per_min"])
    density = {
        "label": str(density.get("label", "normal")),
        "target_bc_per_min": min(max(target, d.min_bc_per_min), d.max_bc_per_min),
        "reason": str(density.get("reason", ""))[:200],
    }

    n = len(turns)
    slot_by_id = {s["slot_id"]: s for s in slots}
    is_bc = [t.get("type", "speech") == "backchannel" for t in turns]
    n_speech = n - sum(is_bc)
    n_bc = sum(is_bc)

    def _text_ok(s, maxlen, orig=""):
        if not isinstance(s, str) or not s.strip() or "\n" in s or len(s) > maxlen:
            return False
        if "〔" in s or "◆" in s:      # 挿入候補マーカーの取り残し（_SLOT_MARK_RE で消せない崩れ形）
            return False
        # 話者名の混入を拒否（ただし元テキストに既に含まれる名前は許容）
        return not any(name and name in s and name not in orig
                       for name in speaker_names)

    seen_targets: set = set()
    deletes: list[int] = []
    inserts: list[dict] = []
    others: list[dict] = []
    merges: list[dict] = []
    for o in raw["ops"]:
        if not isinstance(o, dict):
            dropped.append(f"非オブジェクト op: {o!r:.40}")
            continue
        op = o.get("op")
        # プロンプトの挿入候補マーカー「〔◆9 p=0.67〕」を本文に写してくることがある
        # （rewrite で発生。2026-10-03 確認: 001 の 10,000 対話中 8対話、002 で 3発話が
        # そのまま合成・字幕に入っていた）。消してから検査する。
        # 濁点を分解した文字（か + U+3099）を返すこともある（2026-11-06 確認: ft_v2_001 の
        # 10,000 対話中 1対話。TTS と字幕にそのまま渡る）。合成してから検査する。
        if isinstance(o.get("new_text"), str):
            o["new_text"] = unicodedata.normalize(
                "NFC", _SLOT_MARK_RE.sub("", o["new_text"]))
        # 相槌に「ふんふん」が来たら「うんうん」へ寄せる（TTS が笑い声にするため）。
        # 近傍重複チェックより前に直しておき、置換後の語で重複判定させる。
        if op in ("insert", "reword") and isinstance(o.get("new_text"), str):
            o["new_text"] = _defuse_funfun(o["new_text"])
        if op == "insert":
            s = slot_by_id.get(o.get("slot"))
            if s is None:
                dropped.append(f"insert: 不明な slot {o.get('slot')}")
                continue
            if ("slot", o["slot"]) in seen_targets:
                dropped.append(f"insert: slot {o['slot']} が重複")
                continue
            if not _text_ok(o.get("new_text"), 20):
                dropped.append(f"insert slot {o['slot']}: new_text 不正")
                continue
            if s["listener"] in prot \
                    and _CASUAL_BC_RE.search(o["new_text"]):
                dropped.append(f"insert slot {o['slot']}: 保護話者にくだけた相槌")
                continue
            if _duplicates_nearby_bc(
                    o["new_text"], s["listener"], s["anchor_index"] + 0.5,
                    turns, cfg.quality.duplicate_bc_window):
                dropped.append(f"insert slot {o['slot']}: 近傍相槌と重複")
                continue
            seen_targets.add(("slot", o["slot"]))
            ins = {"op": "insert", "_slot": s, "new_text": o["new_text"].strip()}
            if isinstance(o.get("cue"), str) and o["cue"].strip():
                ins["cue"] = o["cue"].strip()[:40]
            inserts.append(ins)
            continue
        idx = o.get("index")
        if not isinstance(idx, int) or not (0 <= idx < n):
            dropped.append(f"{op}: index {idx} が範囲外")
            continue
        if ("idx", idx) in seen_targets and op != "merge":
            dropped.append(f"{op}: index {idx} が重複")
            continue
        if op == "delete":
            if not is_bc[idx]:
                dropped.append(f"delete: [{idx}] は相槌でない")
                continue
            seen_targets.add(("idx", idx))
            deletes.append(idx)
        elif op == "reword":
            if not is_bc[idx]:
                dropped.append(f"reword: [{idx}] は相槌でない")
                continue
            if not _text_ok(o.get("new_text"), 20, turns[idx]["text"]):
                dropped.append(f"reword [{idx}]: new_text 不正")
                continue
            if o["new_text"].strip() == turns[idx]["text"]:
                continue  # 実質 keep
            if turns[idx]["speaker"] in prot \
                    and _CASUAL_BC_RE.search(o["new_text"]):
                dropped.append(f"reword [{idx}]: 保護話者にくだけた相槌")
                continue
            if _duplicates_nearby_bc(
                    o["new_text"], turns[idx]["speaker"], idx, turns,
                    cfg.quality.duplicate_bc_window, exclude_index=idx):
                dropped.append(f"reword [{idx}]: 近傍相槌と重複")
                continue
            seen_targets.add(("idx", idx))
            others.append({"op": "reword", "index": idx,
                           "new_text": o["new_text"].strip()})
        elif op == "rewrite_text":
            if is_bc[idx]:
                dropped.append(f"rewrite_text: [{idx}] は相槌")
                continue
            if not _text_ok(o.get("new_text"),
                            max(120, 2 * len(turns[idx]["text"])),
                            turns[idx]["text"]):
                dropped.append(f"rewrite_text [{idx}]: new_text 不正")
                continue
            if o["new_text"].strip() == turns[idx]["text"]:
                continue
            if turns[idx]["speaker"] in prot:
                dropped.append(f"rewrite_text [{idx}]: 保護話者の発話")
                continue
            if turns[idx].get("unsaid") or turns[idx].get("overlap"):
                dropped.append(f"rewrite_text [{idx}]: 重なりの場面")   # rp_v1 の言いかけ・重ねたターン
                continue
            quality_error = (_rewrite_quality_error(
                turns[idx]["text"], o["new_text"], cfg.quality)
                or _rewrite_style_error(turns[idx]["text"], o["new_text"], cfg.quality))
            if quality_error:
                dropped.append(f"rewrite_text [{idx}]: {quality_error}")
                continue
            seen_targets.add(("idx", idx))
            others.append({"op": "rewrite_text", "index": idx,
                           "new_text": o["new_text"].strip()})
        elif op == "merge":
            w = o.get("with")
            if not isinstance(w, int) or not (0 <= w < n) or w <= idx:
                dropped.append(f"merge: with {w} が不正")
                continue
            if any(turns[k].get("unsaid") or turns[k].get("overlap") for k in (idx, w)):
                dropped.append(f"merge [{idx}]: 重なりの場面")   # rp_v1 の言いかけ・重ねたターンは結合しない
                continue
            merges.append({"op": "merge", "index": idx, "with": w})
        elif op == "keep":
            pass
        else:
            dropped.append(f"不明な op: {op}")

    # merge の整合: 後半が continued・同一話者・間が全て削除される相槌
    deleted_set = set(deletes)
    clean_merges = []
    merge_heads = set()
    for m in merges:
        i, w = m["index"], m["with"]
        if i in merge_heads:
            dropped.append(f"merge: [{i}] が重複")
            continue
        if is_bc[i] or is_bc[w] or not turns[w].get("continued") \
                or turns[i]["speaker"] != turns[w]["speaker"]:
            dropped.append(f"merge [{i}]+[{w}]: 構造が不適合")
            continue
        between = list(range(i + 1, w))
        if not between or not all(is_bc[j] and j in deleted_set for j in between):
            dropped.append(f"merge [{i}]+[{w}]: 間に削除されない発話がある")
            continue
        merge_heads.add(i)
        clean_merges.append(m)

    # 上限: reword / rewrite / delete
    rewords = [o for o in others if o["op"] == "reword"]
    max_reword = max(1, int(cfg.limits.max_reword_frac * max(1, n_bc)))
    if len(rewords) > max_reword:
        for o in rewords[max_reword:]:
            dropped.append(f"reword [{o['index']}]: 上限{max_reword}件超過")
        keep_ids = {id(o) for o in rewords[:max_reword]}
        others = [o for o in others
                  if o["op"] != "reword" or id(o) in keep_ids]

    rewrites = [o for o in others if o["op"] == "rewrite_text"]
    max_rw = max(1, int(cfg.limits.max_rewrite_frac * max(1, n_speech)))
    if len(rewrites) > max_rw:
        for o in rewrites[max_rw:]:
            dropped.append(f"rewrite_text [{o['index']}]: 上限{max_rw}件超過")
        keep_idx = {id(o) for o in rewrites[:max_rw]}
        others = [o for o in others if o["op"] != "rewrite_text" or id(o) in keep_idx]
    max_del = int(cfg.limits.max_delete_frac * n_bc)
    if len(deletes) > max_del:
        # p_peak が低い順に残す（自然でない相槌を優先削除）
        deletes.sort(key=lambda i: annotations.get(i, {}).get("p_peak") or 0.0)
        for i in deletes[max_del:]:
            dropped.append(f"delete [{i}]: 上限{max_del}件超過")
        deletes = deletes[:max_del]

    # 密度整合: 予測 bc/min を target±tolerance に収める
    minutes = max(duration_sec / 60.0, 1e-6)
    target = density["target_bc_per_min"]

    def projected():
        return (n_bc - len(deletes) + len(inserts)) / minutes

    # 多すぎ → 挿入を p 低い順に間引く
    inserts.sort(key=lambda o: -o["_slot"]["p"])
    while projected() > target + d.tolerance and inserts:
        o = inserts.pop()
        dropped.append(f"insert slot {o['_slot']['slot_id']}: 密度整合で間引き")
    # 少なすぎ → 削除を p_peak 高い順にキャンセル（自然な位置の相槌を復活）
    deletes.sort(key=lambda i: annotations.get(i, {}).get("p_peak") or 0.0)
    while projected() < target - d.tolerance and deletes:
        i = deletes.pop()  # p_peak 最大を復活
        dropped.append(f"delete [{i}]: 密度整合でキャンセル")
        # 対応する merge も無効化
        clean_merges = [m for m in clean_merges
                        if not (m["index"] < i < m["with"])]

    deleted_set = set(deletes)
    clean_merges = [m for m in clean_merges
                    if all(j in deleted_set for j in range(m["index"] + 1, m["with"])
                           if is_bc[j])]

    # Gemini が merge を出し忘れたり一部しか指定しなくても、削除済み相槌だけで
    # 分割されていた同一話者の continued セグメントは決定的に再結合する。
    if cfg.quality.auto_merge_deleted_splits:
        i = 0
        while i < n:
            if is_bc[i]:
                i += 1
                continue
            parts = [i]
            cursor = i
            while True:
                k = cursor + 1
                skipped_bc = False
                while k < n and is_bc[k] and k in deleted_set:
                    skipped_bc = True
                    k += 1
                if not skipped_bc or k >= n or is_bc[k] \
                        or not turns[k].get("continued") \
                        or turns[k]["speaker"] != turns[i]["speaker"]:
                    break
                parts.append(k)
                cursor = k
            if len(parts) > 1:
                exact_explicit = any(
                    len(parts) == 2 and m["index"] == parts[0]
                    and m["with"] == parts[-1] for m in clean_merges)
                if not exact_explicit:
                    clean_merges = [
                        m for m in clean_merges
                        if m["index"] not in parts and m["with"] not in parts]
                    clean_merges.append({"op": "merge", "index": parts[0],
                                         "with": parts[-1], "_parts": parts,
                                         "_auto": True})
                i = parts[-1] + 1
            else:
                i += 1

    clean = ([{"op": "delete", "index": i} for i in sorted(deletes)]
             + others + clean_merges + inserts)
    return clean, density, dropped, None


# --------------------------------------------------------------------------
# ops 適用（パッチ）
# --------------------------------------------------------------------------

def apply_ops(record: dict, turns: list, ops: list, density: dict | None):
    """検証済み ops を適用し、新しいレコードとインデックス対応を返す。

    Returns: (new_record, index_map{orig→new|None}, changed_new, inserted_new, summary)
    変更/挿入ターンには turn["refine"] に来歴が入る（manifest・データセットへ伝播）。
    """
    deleted, reword, rewrite = set(), {}, {}
    merges: dict[int, list[int]] = {}
    auto_merge_heads: set[int] = set()
    inserts_by_anchor: dict[int, list] = {}
    for o in ops:
        if o["op"] == "delete":
            deleted.add(o["index"])
        elif o["op"] == "reword":
            reword[o["index"]] = o["new_text"]
        elif o["op"] == "rewrite_text":
            rewrite[o["index"]] = o["new_text"]
        elif o["op"] == "merge":
            parts = o.get("_parts") or [o["index"], o["with"]]
            merges[o["index"]] = list(parts[1:])
            if o.get("_auto"):
                auto_merge_heads.add(o["index"])
        elif o["op"] == "insert":
            s = o["_slot"]
            inserts_by_anchor.setdefault(s["anchor_index"], []).append(
                {**s, "text": o["new_text"],
                 **{k: o[k] for k in ("cue", "react_orig") if o.get(k)}})
    merged_away = {part for parts in merges.values() for part in parts}

    new_turns: list[dict] = []
    index_map: dict[int, int | None] = {}
    changed: set[int] = set()
    inserted: set[int] = set()

    for i, t in enumerate(turns):
        if i in deleted or i in merged_away:
            index_map[i] = None
            continue
        nt = dict(t)
        prov = None
        if i in reword:
            prov = {"op": "reword", "orig_index": i, "orig_text": t["text"]}
            nt["text"] = reword[i]
        elif i in rewrite:
            prov = {"op": "rewrite", "orig_index": i, "orig_text": t["text"]}
            nt["text"] = rewrite[i]
        if i in merges:
            merged_text = nt["text"]
            merged_duration = float(t.get("duration") or 0.0)
            for partner_idx in merges[i]:
                partner = turns[partner_idx]
                head = merged_text.rstrip()
                joiner = "" if head.endswith(tuple(_PUNCT + "…")) else "、"
                merged_text = head + joiner + partner["text"]
                merged_duration += float(partner.get("duration") or 0.0)
            nt["text"] = merged_text
            nt["duration"] = round(merged_duration, 2)
            merged_with = merges[i][0] if len(merges[i]) == 1 else list(merges[i])
            prov = {"op": ("merge+" + prov["op"]) if prov else "merge",
                    "orig_index": i, "orig_text": t["text"],
                    "merged_with": merged_with}
            if i in auto_merge_heads:
                prov["auto_merge"] = True
        new_i = len(new_turns)
        index_map[i] = new_i
        if prov:
            nt["refine"] = prov
            changed.add(new_i)
        new_turns.append(nt)

        for s in sorted(inserts_by_anchor.get(i, []), key=lambda x: x["rel_pos"]):
            rel = float(s["rel_pos"])
            if i in merges:  # 結合でアンカーが伸びた分、rel を再スケール
                d_head = float(t.get("duration") or 0.0)
                d_all = float(nt.get("duration") or 0.0)
                if d_all > 0:
                    rel = rel * (d_head / d_all)
            anchor_dur = float(nt.get("duration") or 0.0)
            bc = {"speaker": s["listener"], "text": s["text"],
                  "type": "backchannel",
                  "start_offset": round((rel - 1.0) * anchor_dur, 2) if anchor_dur else -0.5,
                  "duration": 0.5,
                  "refine": {"op": "insert", "rel_pos": round(rel, 3),
                             "slot_p": s["p"],
                             **{k: s[k] for k in ("cue", "react_orig") if s.get(k)}}}
            new_i = len(new_turns)
            inserted.add(new_i)
            new_turns.append(bc)

    # merge されずに削除された「埋め込み相槌」: 続き(A2)に unmerged_split を記録
    for i in deleted:
        for j in range(i + 1, len(turns)):
            nj = index_map.get(j)
            if nj is None:
                continue
            if new_turns[nj].get("continued") \
                    and new_turns[nj]["speaker"] != turns[i]["speaker"]:
                rf = dict(new_turns[nj].get("refine") or {})
                rf["unmerged_split"] = True
                new_turns[nj]["refine"] = rf
            break

    new_record = dict(record)
    new_record["voice_dialogue"] = dict(record["voice_dialogue"], turns=new_turns)
    if density:
        new_record["refine_density"] = density
    summary = {"reword": len(reword), "delete": len(deleted),
               "insert": sum(len(v) for v in inserts_by_anchor.values()),
               "rewrite": len(rewrite), "merge": len(merges),
               "auto_merge": len(auto_merge_heads)}
    return new_record, index_map, changed, inserted, summary


def build_refined_record(new_record: dict, manifest_turns: list) -> dict:
    """最終配置（manifest）からタイミングを逆算した、データセット用レコードを作る。"""
    turns = new_record["voice_dialogue"]["turns"]
    out_turns = []
    prev_end = 0.0
    for k, (t, mt) in enumerate(zip(turns, manifest_turns)):
        nt = dict(t)
        nt["duration"] = mt["duration_sec"]
        nt["start_abs"] = mt["start_sec"]
        nt["end_abs"] = mt["end_sec"]
        nt["start_offset"] = 0.0 if k == 0 else round(mt["start_sec"] - prev_end, 2)
        prev_end = mt["end_sec"]
        out_turns.append(nt)
    rec = dict(new_record)
    rec["voice_dialogue"] = dict(new_record["voice_dialogue"], turns=out_turns)
    return rec


# --------------------------------------------------------------------------
# メインループ
# --------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_base_outputs(bmf, btrf, base_audio_dir: Path, record: dict,
                        voices: list, sr: int, wav_by_turn: dict,
                        failed_turns: set, plans: list | None, retimer,
                        rt_cfg: RetimeConfig, trim_cfg, noise_cfg,
                        loud_cfg, rate_cfg, max_seconds: float | None,
                        fish_backend: str | None = None,
                        lead_cfg: dict | None = None,
                        tail_cfg: dict | None = None) -> bool:
    """融合モード: 1段目（retime）の成果物を --retime と同じ場所・同じ形で書く。

    base wav / manifest / transcript を出す。既に base wav がある対話（＝別プロセスや
    過去の --retime が作った分）は何も書かずに True を返す。無音代替が出た対話は
    何も書かず False を返す（呼び出し側がその対話を捨てる）。
    """
    import soundfile as sf
    from generate_tts import (
        collect_voice_prompts, compose_dialogue, drop_if_synth_failed,
        write_transcript, _channel_speakers,
    )
    did = record["id"]
    out_path = base_audio_dir / f"{did}{_AUDIO_EXT}"
    result = compose_dialogue(record, voices, sr, wav_by_turn, failed_turns,
                              retimer=retimer, trim=trim_cfg, noise=noise_cfg,
                              loudness=loud_cfg, rate=rate_cfg, plans=plans,
                              lead_in=lead_cfg, tail_pad=tail_cfg)
    if result is None:
        return False
    stereo, manifest_turns = result
    from audio_padding import padding_marks
    if drop_if_synth_failed(did, manifest_turns):
        return False
    if find_audio(base_audio_dir, did) is not None:
        return True     # 既に1段目済み（キャッシュだけ欠けていた）。上書きしない（wav / flac どちらでも）
    if max_seconds:
        stereo = stereo[:int(max_seconds * sr)]
    write_audio(out_path, stereo, sr)
    voice_prompts = collect_voice_prompts(voices, manifest_turns)
    line = {
        "id": did,
        "audio_path": str(out_path),
        "sample_rate": sr,
        "channels": int(stereo.shape[1]),
        "duration_sec": round(len(stereo) / sr, 3),
        "channel_speakers": _channel_speakers(record),
        "voice_prompts": voice_prompts,
        "turns": manifest_turns,
        "retime_config": rt_cfg.summary(),
        **padding_marks(did, lead_cfg, tail_cfg),
    }
    if fish_backend:
        line["fish_backend"] = fish_backend
    bmf.write(json.dumps(line, ensure_ascii=False) + "\n")
    bmf.flush()
    if btrf is not None:
        write_transcript(base_audio_dir, btrf, record, manifest_turns, sr,
                         int(stereo.shape[1]), len(stereo) / sr,
                         _channel_speakers(record), voice_prompts)
    return True


def run_refine(config: dict, dry_run: bool, limit: int | None,
               device: str | None = None, max_seconds: float | None = None,
               recompose: bool = False, shard: tuple[int, int] | None = None,
               claim: bool = False, claim_ttl_min: float = 30.0,
               fused: bool = False) -> int:
    """refine（Stage4 2段目）を回す。

    fused=True（--retime-refine）のときは「1対話ぶんの retime → その対話の refine」を
    連続で行う。retime 済み（発話wavキャッシュがある）対話はそのキャッシュを使うので
    合成は走らず、実質 --refine と同じ経路になる。対象は「refine 済み wav が無い対話」
    なので、既に --retime だけ済ませた分は 1段目をスキップして 2段目だけが走る。
    """
    from generate_tts import (
        DialogueClaims, FishTagError,
        apply_dialogue_speaking_rate, apply_shard, assign_voices, build_context_prompt,
        build_fish_tagger, build_synth, compose_dialogue,
        drop_if_synth_failed,
        engine_output_paths, base_variant_name, ref_variant_name,
        drop_empty_part, load_turn_cache, load_voices, normalize_tts_text,
        part_output_path, record_speaker_order, resolve_speaker_attrs,
        regenerate_misread_turns,
        save_turn_cache, shard_tag, synthesize_dialogue_turns, synthesize_for_speaker,
        synthesize_targets_with_context, synthesize_turns, transcripts_path,
        truncate_record_by_seconds, worker_tag, _channel_speakers,
        tts_servers_dead,
    )

    tts_cfg = config["tts"]
    engine = tts_cfg.get("engine", "gemini")
    # どの fish で合成したか（fish = fish-speech の api_server / vllm_omni = vLLM-Omni）を manifest に残す。
    # 2026-10-22 から 003 は途中で vLLM-Omni に切り替えたので、1つのコーパスに両方が混ざる。
    fish_backend = (str((tts_cfg.get("fish_speech") or {}).get("backend") or "fish")
                    if engine == "fish_speech" else None)
    speed = float(tts_cfg.get("speed", 1.0) or 1.0)  # 発話速度（>1で速く）
    _cp = (tts_cfg.get("google_cloud", {}) or {}).get("context_prosody") or {}
    ctx_prosody = bool(_cp.get("enabled", False)) and engine in ("google_cloud", "gemini")
    ctx_short_len = int(_cp.get("short_len", 6))
    trim_cfg = tts_cfg.get("trim_silence") or None  # 発話前後の無音トリミング設定
    noise_cfg = tts_cfg.get("noise_fill") or None   # 無音区間のコンフォートノイズ設定
    lead_cfg = tts_cfg.get("lead_in") or None       # 対話の先頭に置く無音（lead_in_sec）
    tail_cfg = tts_cfg.get("tail_pad") or None      # 対話の末尾に置く無音（tail_pad_sec）
    rt_cfg = RetimeConfig.from_tts_cfg(tts_cfg)
    rf_cfg = RefineConfig.from_tts_cfg(tts_cfg)
    rt_cfg.bc.insert_window_sec = rf_cfg.insert.window_sec  # 挿入相槌のヒント窓

    variant = ref_variant_name(rt_cfg.tag, rf_cfg.tag)   # ★2026-11-22 エンジン名を外した（旧 f"{engine}_{rt_cfg.tag}_{rf_cfg.tag}"）
    if dry_run:  # モック出力（正弦波・canned ops）を本番 variant に混ぜない
        variant += "_dryrun"
    audio_dir, manifest_file = engine_output_paths(tts_cfg, engine, variant)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    global _SIDON, _AUDIO_EXT
    from generate_tts import set_audio_format
    _AUDIO_EXT = set_audio_format(tts_cfg)      # 対話単位の音声の形式（tts.audio_format: wav / flac）
    _scfg = tts_cfg.get("sidon") or {}
    _SIDON = SidonClient(_scfg) if (_scfg.get("in_worker") and not dry_run) else None
    engine_cache = engine_output_paths(tts_cfg, engine)[0] / "turns"
    # refine は感情タガーを持たない。base キャッシュが無いとここでタグ無し合成になるので、
    # LLMタグ構成のときだけ警告を出せるように判定しておく（挙動は変えない）。
    # ※融合モードでは 1段目をこのプロセスが回すのでタガーを持つ（下の fish_tagger）。
    _fcfg = tts_cfg.get("fish_speech", {}) or {}
    fish_llm_tags = (engine == "fish_speech"
                     and bool(_fcfg.get("emotion_tags", True))
                     and str(_fcfg.get("tag_engine", "llm")).lower() == "llm")

    # ---- 融合モード（--retime-refine）: 1段目の出力先と合成設定 ----
    # 1段目の成果物（base wav / manifest / transcript / タグ）は --retime で回した
    # ときと同じものを同じ場所に書く。段を分けた運用へいつでも戻せるようにするため。
    fish_tagger = fish_rule_tags = None
    fish_prefix_tag = ""
    base_audio_dir = base_manifest_file = base_tr_file = None
    loud_cfg = rate_cfg = None
    # 相槌だけ一律に下げる量（tts.loudness.bc_gain_db）は refine の合成（_ref の最終出力）にも渡す（2026-11-14）。
    # 音量の均し本体（loudness.enabled）は従来どおり 1 段目だけ（ここでは enabled を渡さない）。
    bc_loud = {"bc_gain_db": float((tts_cfg.get("loudness") or {}).get("bc_gain_db", 0.0) or 0.0)}
    if fused:
        base_variant = base_variant_name(rt_cfg.tag)   # ★2026-11-22（旧 f"{engine}_{rt_cfg.tag}"）
        if dry_run:  # モック出力（正弦波）を本番の1段目ディレクトリに混ぜない
            base_variant += "_dryrun"
        base_audio_dir, base_manifest_file = engine_output_paths(
            tts_cfg, engine, base_variant)
        base_audio_dir.mkdir(parents=True, exist_ok=True)
        loud_cfg = tts_cfg.get("loudness") or None   # ターン間の音量均し（1段目の配置時）
        rate_cfg = tts_cfg.get("rate") or None       # ターンごとの話速均し（同上）
        fish_tagger, fish_rule_tags, fish_prefix_tag = build_fish_tagger(config, engine)
    refined_cache = audio_dir / "turns"
    rp = Path(rf_cfg.refined_records_file)
    refined_records_file = rp.with_name(f"{rp.stem}_{variant}{rp.suffix}")
    # 並列時は追記競合回避で書き手ごとの part に分ける（tr_file は manifest_file から
    # 派生し自動追従）。全プロセス完了後に `generate_tts.py --merge-parts` で統合する。
    part_tag = worker_tag() if claim else shard_tag(shard)
    manifest_file = part_output_path(manifest_file, part_tag)
    refined_records_file = part_output_path(refined_records_file, part_tag)
    if fused:
        base_manifest_file = part_output_path(base_manifest_file, part_tag)
        base_manifest_file.parent.mkdir(parents=True, exist_ok=True)

    voices = load_voices(tts_cfg)
    if not voices:
        logger.error("ボイスバンクが空です。tts.voices_file / tts.voices を設定してください。")
        return 0

    if dry_run:
        retimer = MockRetimer(rt_cfg)
        refiner = MockRefiner(rf_cfg, variant)
        scanner = MockScanner(rf_cfg)
        logger.info("refine(dry-run): MockRefiner + MockScanner + MockRetimer で配線検証")
    else:
        if rt_cfg.backend == "maai":
            try:
                import maai  # noqa: F401
            except ImportError:
                logger.error("maai パッケージが見つかりません。refine は .venv-maai で実行してください:")
                logger.error("  .venv-maai/bin/python generate_tts.py --refine ...")
                return 0
        retimer = MaaiRetimer(rt_cfg)
        refiner = make_refiner(rf_cfg, variant)
        scanner = InsertionScanner(retimer, rf_cfg)
        if not recompose:
            refiner.resolve_model()  # 早期にモデル/キー問題を検出
        logger.info(f"refine 有効: provider={rf_cfg.provider}, text={rf_cfg.text_model}, "
                    f"密度範囲 {rf_cfg.density.min_bc_per_min}-"
                    f"{rf_cfg.density.max_bc_per_min}/min → 出力 {audio_dir}")
        if fused:
            logger.info("融合モード（--retime-refine）: 1対話ごとに retime→refine を"
                        f"連続実行。1段目の出力は {base_audio_dir}")

    # ---------------- recompose: 精製済みから再配置のみ ----------------
    if recompose:
        if not refined_records_file.exists():
            logger.error(f"精製済みレコードがありません: {refined_records_file}")
            return 0
        refined = {r["id"]: r for r in _load_records(refined_records_file)}
        pending = [r for r in refined.values()
                   if not has_audio(audio_dir, r["id"])
                   and (refined_cache / r["id"] / "meta.json").exists()]
        if shard is not None:
            pending = apply_shard(pending, shard)
        if limit is not None:
            pending = pending[:limit]
        logger.info(f"refine recompose: 対象 {len(pending)}件（精製済み {len(refined)}件）"
                    + (f" / シャード{shard[0]}/{shard[1]}" if shard else ""))
        claims = (DialogueClaims(audio_dir / ".claims_recompose",
                                 ttl_sec=claim_ttl_min * 60.0) if claim else None)
        done = 0
        silent_dropped = 0
        write_tr = bool(tts_cfg.get("write_transcript", True))
        tr_file = transcripts_path(manifest_file) if write_tr else None
        with (contextlib.closing(claims) if claims is not None  # 上と同じ理由
              else contextlib.nullcontext()), \
                open(manifest_file, "a") as mf, \
                (open(tr_file, "a") if tr_file else contextlib.nullcontext()) as trf:
            for rec in pending:
                if claims is not None and not claims.take(
                        rec["id"], audio_dir / f"{rec['id']}{_AUDIO_EXT}"):
                    continue  # 他プロセスが担当中／その間に完了済み
                turns = rec["voice_dialogue"]["turns"]
                cached = load_turn_cache(refined_cache, rec["id"], len(turns))
                if cached is None:
                    continue
                wavs, failed, csr = cached
                result = compose_dialogue(rec, voices, csr, wavs, failed,
                                          retimer=retimer, trim=trim_cfg,
                                          noise=noise_cfg, lead_in=lead_cfg,
                                          tail_pad=tail_cfg, loudness=bc_loud)
                if result is None:
                    continue
                stereo, manifest_turns = result
                # 無音代替（キャッシュ由来）を含む対話は書き出さない
                if drop_if_synth_failed(rec["id"], manifest_turns):
                    silent_dropped += 1
                    continue
                _merge_refine_prov(manifest_turns, turns)
                _write_outputs(mf, audio_dir, rec, stereo, manifest_turns, csr,
                               rt_cfg, rf_cfg, rec.get("refine_density"),
                               getattr(refiner, "model", None), None, None,
                               _channel_speakers, trf=trf, voices=voices,
                               fish_backend=fish_backend, lead_cfg=lead_cfg, tail_cfg=tail_cfg)
                done += 1
                if claims is not None:
                    claims.finish(rec["id"])
        drop_empty_part(manifest_file, part_tag)   # 空振りの再実行で空 part を残さない
        drop_empty_part(tr_file, part_tag)
        if claims is not None:  # 解放は上の closing() 済み。ここは集計ログだけ
            claims.log_summary()
        if hasattr(retimer, "log_summary"):
            retimer.log_summary()
        if silent_dropped:
            logger.error(f"refine recompose: 無音代替が出た {silent_dropped}件を破棄"
                         "しました（音声・manifest とも未作成）")
        logger.info(f"refine recompose 完了: {done}件"
                    + (f" / 無音破棄{silent_dropped}件" if silent_dropped else "")
                    + f" | 出力: {audio_dir}")
        return done

    # ---------------- 通常経路 ----------------
    synth, sr, short_threshold = build_synth(tts_cfg, engine, dry_run, device)

    input_file = Path(config["voice"]["output_file"])
    records = _load_records(input_file)
    total = len(records)
    pending = [r for r in records if not has_audio(audio_dir, r["id"])]
    already = total - len(pending)
    if shard is not None:  # 複数GPU/複数fishサーバでの並列処理: 排他シャードに分割
        n_before = len(pending)
        pending = apply_shard(pending, shard)
        logger.info(f"シャード {shard[0]}/{shard[1]}: 担当 {len(pending)}/{n_before}件")
    if limit is not None:
        pending = pending[:limit]
    logger.info(f"refine 開始: 全{total}件中、済{already}件、今回{len(pending)}件"
                + (f" / 冒頭{max_seconds}秒" if max_seconds else ""))
    if fused:
        # 1段目済み（発話wavキャッシュあり）＝ この実行では 2段目だけが走る対話。
        n_base = sum(1 for r in pending
                     if (engine_cache / r["id"] / "meta.json").exists())
        logger.info(f"融合モード: うち retime 済み {n_base}件は1段目をスキップ、"
                    f"残り {len(pending) - n_base}件は retime→refine を通しで実行")

    # クレーム方式: 未処理の対話を先着順に取りに行く（シャードの静的分割の代替）。
    # ★ retime 側とはクレームの置き場（audio_dir）が別なので、パスをまたいで干渉しない。
    #   ただし refine は base の発話wavキャッシュ（= retime の出力）に依存するため、
    #   全プロセスの --retime が終わってから --refine を流すこと（run_tts_pool.sh 参照）。
    claims = None
    if claim:
        claims = DialogueClaims(audio_dir / ".claims", ttl_sec=claim_ttl_min * 60.0)
        logger.info(f"クレーム方式: 未処理の対話を先着順に取得 "
                    f"(worker={claims.tag}, TTL={claim_ttl_min:.0f}分, {claims.root})")
        if fused and claim_ttl_min < 60.0:
            # 融合は1対話を retime+refine の2段ぶん抱える（実測で20分前後、長い対話は
            # それ以上）。TTL がそれを下回ると、まだ処理中の対話を他プロセスが引き取って
            # 二重に合成してしまう。
            logger.warning(f"融合モードでクレームTTLが{claim_ttl_min:.0f}分です。1対話の"
                           "処理時間（retime+refine で20分前後）を下回ると二重処理に"
                           "なります。--claim-ttl-min 90 以上を推奨します")

    done = failed_count = skipped = silent_dropped = 0
    tag_skipped = 0     # 融合モード: LLMタグ付けが全リトライ失敗してスキップした対話数
    retimed = 0         # 融合モード: この実行で1段目から回した対話数
    write_tr = bool(tts_cfg.get("write_transcript", True))
    tr_file = transcripts_path(manifest_file) if write_tr else None
    if fused and write_tr:
        base_tr_file = transcripts_path(base_manifest_file)
    # closing(claims) を最初に置く: with は逆順に抜けるので、出力を閉じてからクレームを
    # 解放する。★Ctrl+C や例外で抜けても必ず解放されるのが要点。握ったまま落ちると
    # クレームの mtime が新しいまま残り、すぐ再実行してもその対話は TTL
    # （--claim-ttl-min。融合は既定90分）が切れるまで飛ばされてしまう。
    with (contextlib.closing(claims) if claims is not None
          else contextlib.nullcontext()), \
            open(manifest_file, "a") as mf, \
            open(refined_records_file, "a") as rf_out, \
            (open(tr_file, "a") if tr_file else contextlib.nullcontext()) as trf, \
            (open(base_manifest_file, "a") if fused
             else contextlib.nullcontext()) as bmf, \
            (open(base_tr_file, "a") if base_tr_file
             else contextlib.nullcontext()) as btrf:
        for r in pending:
            did = r["id"]
            if getattr(getattr(synth, "all_servers_dead", None), "is_set", lambda: False)():
                # ★2026-11-19 TTS サーバが全部死んだ（MossTtsSynth の遮断）。残りは他のワーカー・次の回に任せる
                logger.error("TTS サーバが全部応答しないので、ここで終わる（担当中の claim は解放する）")
                break
            if claims is not None and not claims.take(
                    did, audio_dir / f"{did}{_AUDIO_EXT}"):
                continue  # 他プロセスが担当中／その間に完了済み
            try:
                rr = truncate_record_by_seconds(r, max_seconds)
                # ★2026-11-10 重なりの場面（rp_v1）: 言いかけは1文全体で合成し、配置の前に文の途中で切る
                prepare_cut_turns(rr)
                turns = rr["voice_dialogue"]["turns"]
                # ★ compose_dialogue と同じ順序関数を使うこと。ここだけ出現順のままだと、
                #   主話者が先に話さない対話で retime と refine のチャネルが左右逆になる。
                speaker_order = record_speaker_order(rr)
                if len(speaker_order) not in SUPPORTED_SPEAKER_COUNTS:
                    skipped += 1
                    continue
                channel_of = {sp: i for i, sp in enumerate(speaker_order)}
                # 対話ごとの話速（Pass A のキャッシュミス合成・Pass B の追加合成の両方に効かせる）
                apply_dialogue_speaking_rate(synth, rr, tts_cfg)

                # --- Pass A: wav 準備 + 配置 + 挿入候補 ---
                use_sr = sr
                pt = PhaseTimer()
                plans = None
                cached = load_turn_cache(engine_cache, did, len(turns))
                if cached is not None:
                    wav_by_turn, failed_turns, use_sr = cached
                    apply_cuts(rr, wav_by_turn, use_sr, failed_turns)
                elif fused:
                    # 1段目（retime）をこの対話ぶんだけ実行する。--retime で回したときと
                    # 同じ経路・同じ出力（base wav / manifest / transcript / タグ）。
                    fish_map = None
                    if fish_tagger is not None:
                        try:
                            fish_map = fish_tagger.tag_turns(turns, did)
                        except FishTagError as e:
                            # タグ無しで合成するとキャッシュにタグ無し音声が残り、以後この
                            # 対話にタグが付かなくなる。合成せず次回実行の resume に回す。
                            logger.error(f"[{did}] LLMタグ付け失敗→この対話はスキップ"
                                         f"（次回実行で再試行）: {e}")
                            tag_skipped += 1
                            continue
                        if fish_map:
                            (base_audio_dir / f"{did}.tags.json").write_text(
                                json.dumps(
                                    {str(k): v for k, v in sorted(fish_map.items())},
                                    ensure_ascii=False, indent=1))
                    with pt.phase("base合成"):
                        res = synthesize_dialogue_turns(
                        synth, rr, voices, sr, short_threshold=short_threshold,
                        turn_cache_dir=(engine_cache if rt_cfg.cache_turn_wavs
                                        and not dry_run else None),
                        speed=speed, context_prosody=ctx_prosody,
                        context_short_len=ctx_short_len,
                        emotion_tags=fish_rule_tags, fish_tags_map=fish_map,
                        prefix_tag=fish_prefix_tag)
                    if res is None:
                        failed_count += 1
                        continue
                    wav_by_turn, failed_turns = res
                    if failed_turns and tts_servers_dead(synth):
                        # ★2026-11-20 TTS サーバが全部死んで合成できなかった → 無音破棄に数えず、ここで終わる
                        #   （クレームは closing() で解放され、他のワーカー・次の回が作り直す）
                        logger.error(f"[{did}] TTS サーバが全部応答しないので、この対話は作らずに終わる（無音破棄には数えない）")
                        break
                    apply_cuts(rr, wav_by_turn, use_sr, failed_turns)   # 重なりの場面（rp_v1）
                    # 1段目の配置。ここで引いたプランは Pass A でそのまま使い回す
                    # （同じ (turns, wav_by_turn) なので MaAI を二度引く必要がない）。
                    with pt.phase("配置(vap)"):
                        plans = retimer.plan(rr, turns, wav_by_turn, use_sr,
                                             channel_of, failed_turns)
                    if not _write_base_outputs(
                            bmf, btrf, base_audio_dir, rr, voices, use_sr,
                            wav_by_turn, failed_turns, plans, retimer, rt_cfg,
                            trim_cfg, noise_cfg, loud_cfg, rate_cfg, max_seconds,
                            fish_backend=fish_backend, lead_cfg=lead_cfg,
                            tail_cfg=tail_cfg):
                        silent_dropped += 1
                        continue
                    retimed += 1
                else:
                    if fish_llm_tags:
                        # ここでの base 合成は感情タグを通らない（refine はタガーを持たない）。
                        # --retime がこの対話をタグ付け失敗でスキップした場合もここに来るので、
                        # タグ無し音声になっていることを明示しておく。
                        logger.warning(
                            f"[{did}] base の発話キャッシュが無いため refine 側で合成します"
                            "（感情タグ無し）。--retime を先に完走させてください")
                    with pt.phase("base合成"):
                        res = synthesize_turns(synth, rr, voices, sr,
                                           short_threshold=short_threshold,
                                           speed=speed, context_prosody=ctx_prosody,
                                           context_short_len=ctx_short_len)
                    if res is None:
                        failed_count += 1
                        continue
                    wav_by_turn, failed_turns = res
                    if rt_cfg.cache_turn_wavs and not dry_run and not failed_turns:
                        # 無音代替を含むキャッシュは残さない（後段が無音のまま拾う）
                        save_turn_cache(engine_cache, did, wav_by_turn,
                                        failed_turns, use_sr)
                    apply_cuts(rr, wav_by_turn, use_sr, failed_turns)   # 重なりの場面（rp_v1）
                if plans is None:
                    with pt.phase("配置(vap)"):
                        plans = retimer.plan(rr, turns, wav_by_turn, use_sr,
                                             channel_of, failed_turns)
                if plans is None:
                    skipped += 1
                    continue
                with pt.phase("候補スキャン(bc)"):
                    timeline, duration_sec = build_timeline(
                        turns, plans, wav_by_turn, use_sr, channel_of)
                    scan_kw = {}
                    if rf_cfg.insert.main_listener:
                        scan_kw["main_ch"] = main_speaker_channel(rr, channel_of)
                        if scan_kw["main_ch"] is None:
                            logger.warning(f"[{did}] metadata.main_speaker_index が無いので "
                                           "insert.main_listener は当てない")
                    slots = scanner.scan(timeline, turns, plans, wav_by_turn,
                                         use_sr, channel_of, failed_turns, **scan_kw)

                annotations = {}
                for p in plans:
                    i = p["index"]
                    rt = p["retime"]
                    if turns[i].get("type", "speech") == "backchannel":
                        annotations[i] = {"p_peak": rt.get("p_peak")}

                n_bc = sum(1 for t in turns
                           if t.get("type", "speech") == "backchannel")
                cur_density = n_bc / max(duration_sec / 60.0, 1e-6)

                # --- Gemini ops（検証込み・json_retries 回まで再生成） ---
                prompt = build_prompt(rr, turns, annotations, slots,
                                      duration_sec, cur_density, rf_cfg)
                ops, density, fail_reason = [], None, None
                cur_prompt = prompt
                for attempt in range(rf_cfg.json_retries + 1):
                    try:
                        with pt.phase("refine LLM"):
                            raw = refiner.propose(did, cur_prompt,
                                              turns=turns, slots=slots)
                    except ValueError as e:  # JSON パース不能
                        fail_reason = str(e)
                        cur_prompt = (prompt + "\n\n# 前回出力の問題\n"
                                      + f"{e}\nJSONのみを再出力してください。")
                        continue
                    ops, density, dropped, fatal = validate_ops(
                        raw, turns, slots, duration_sec, annotations,
                        rf_cfg, speaker_order,
                        protected_speakers(rf_cfg.quality.no_rewrite_speakers,
                                           rr.get("metadata")))
                    if fatal is None:
                        fail_reason = None
                        if dropped:
                            logger.info(f"[{did}] ops 検証で {len(dropped)}件を破棄: "
                                        + "; ".join(dropped[:5]))
                        break
                    fail_reason = fatal
                    cur_prompt = (prompt + "\n\n# 前回出力の問題\n"
                                  + f"{fatal}\n修正した完全なJSONを再出力してください。")
                if fail_reason is not None:
                    logger.warning(f"[{did}] Gemini ops 不成立 → 無編集で出力: {fail_reason}")
                    ops, density = [], None
                n_react = 0
                if ops and rf_cfg.quality.bc_react_verify:
                    with pt.phase("refine LLM"):
                        n_react = verify_reactive_inserts(refiner, did, rr, turns, ops)

                # --- ops 適用 + Pass B 合成 ---
                new_record, index_map, changed, inserted, op_summary = \
                    apply_ops(rr, turns, ops, density)
                new_turns = new_record["voice_dialogue"]["turns"]

                # 長い相槌（2語連結）を短縮して衝突を防ぐ（LLMの取りこぼしを機械的に保証）。
                # 短縮したターンは changed に入れて Pass B で短い音声に再合成する。
                n_short = 0
                for ni, t in enumerate(new_turns):
                    if t.get("type", "speech") != "backchannel":
                        continue
                    sh = _shorten_bc(t["text"], rf_cfg.quality.bc_shorten_len)
                    if sh and sh != t["text"]:
                        rf = dict(t.get("refine") or {})
                        rf.setdefault("orig_text", t["text"])
                        rf["op"] = ("bc_shorten" if rf.get("op") in (None, "insert")
                                    else rf["op"] + "+bc_shorten")
                        t["refine"] = rf
                        t["text"] = sh
                        changed.add(ni)
                        n_short += 1
                if n_short:
                    op_summary["bc_shorten"] = n_short

                # へー系相槌が多用され不自然になるのを機械的に抑制（半減して他語へ置換）。
                # 置換ターンは changed 経由で Pass B が新テキストで再合成する。
                n_hee = _thin_hee_family(new_turns, changed, rf_cfg.quality,
                                         rf_cfg.quality.duplicate_bc_window)
                if n_hee:
                    op_summary["hee_thin"] = n_hee

                # 発話途中に挿入された相槌は継続支持語であるべき（内容反応語は発話末尾への
                # 反応に聞こえる）。rel_pos が途中の挿入相槌を継続支持語へ機械的に置換する。
                n_cont = _continuer_only_inserts(new_turns, changed, rf_cfg.quality,
                                                 rf_cfg.quality.duplicate_bc_window)
                if n_cont:
                    op_summary["insert_continuer"] = n_cont

                # 挿入相槌の語を bc_vocab_mix の配分へ（LLM は ええ・はい・うん・うんうん に寄る）。
                n_mix = _rebalance_bc_vocab(
                    new_turns, changed, rf_cfg.quality,
                    rf_cfg.quality.duplicate_bc_window,
                    protected_speakers(rf_cfg.quality.no_rewrite_speakers,
                                       rr.get("metadata")), did)
                if n_mix:
                    op_summary["bc_mix"] = n_mix
                if n_react:
                    op_summary["react_verify"] = n_react

                new_wavs: dict[int, np.ndarray] = {}
                new_failed: set[int] = set()
                for oi, ni in index_map.items():
                    if ni is not None and ni not in changed:
                        new_wavs[ni] = wav_by_turn[oi]
                        if oi in failed_turns:
                            new_failed.add(ni)
                need = sorted(changed | inserted)
                need_t0 = time.perf_counter()
                if need:
                    attrs = resolve_speaker_attrs(rr, speaker_order)
                    assignment = assign_voices(did, speaker_order, voices, attrs)
                    # fish: 対象ターン（挿入相槌・書き換え）を、直前 window_context ターン
                    # （相手＋自分）を文脈に前置きして再合成する。挿入相槌などは相手の発話への
                    # 反応なので、per_speaker（自分だけ）とは逆に相手の発話を条件に含める。
                    # fish 以外・参照音声不足のときは None が返り、従来の文脈なし話者別合成へ劣化。
                    ctx_turns = int(getattr(synth, "window_context", 0) or 0)
                    # context_audio 時: 据え置きターンの採用済みwav（new_wavs）を渡し、
                    # 文脈をテキスト再生成ではなく実際の音声で条件付ける。
                    if rf_cfg.quality.speech_resynth_no_context:
                        ctx_need = [ni for ni in need
                                    if new_turns[ni].get("type", "speech") == "backchannel"]
                    else:
                        ctx_need = list(need)
                    plain_need = [ni for ni in need if ni not in set(ctx_need)]

                    def _plain_synth(idx_list):
                        """文脈なしの話者別合成（従来の plain_need の処理そのまま）。"""
                        per_speaker: dict[str, list[int]] = {}
                        for ni in idx_list:
                            per_speaker.setdefault(new_turns[ni]["speaker"], []).append(ni)
                        for sp, idxs in per_speaker.items():
                            texts = [normalize_tts_text(tts_reading.turn_text(new_turns[ni]))
                                     for ni in idxs]
                            ests = [float(new_turns[ni].get("duration") or 0.0)
                                    for ni in idxs]
                            prompts = ([build_context_prompt(new_turns, ni, ctx_short_len)
                                        for ni in idxs] if ctx_prosody else None)
                            wavs, fl = synthesize_for_speaker(
                                synth, texts, assignment[sp], ests, did, use_sr,
                                short_threshold=short_threshold, speed=speed,
                                prompts=prompts)
                            for ni, w, f in zip(idxs, wavs, fl):
                                new_wavs[ni] = w
                                if f:
                                    new_failed.add(ni)

                    # ★2026-11-20 書き換えなど文脈なしで作る発話（plain_need）を先に作り、相槌（ctx_need）はその後にする。
                    #   Irodori の相槌は参照の先頭に「その話者の直前の採用済み発話」を置く（bc_prev_ref）が、それまでは相槌を先に
                    #   作っていたので、直前の発話が書き換えたターンのときはそれがまだ無く、さらに前の古い発話を参照にしていた
                    #   （本番 004 で相槌の 7.3%）。先に作れば、書き換えた後の本当の直前の発話を参照にできる。
                    #   失敗して無音代替になった発話は参照に使わない（その対話はどのみち捨てる）。
                    if plain_need:
                        _plain_synth(plain_need)
                    ctx_wavs = (synthesize_targets_with_context(
                        synth, new_record, voices, use_sr, ctx_need, ctx_turns,
                        adopted_wavs={k: v for k, v in new_wavs.items() if k not in new_failed},
                        adopted_sr=use_sr) if ctx_need else {})
                    if ctx_wavs is not None:
                        for ni, w in ctx_wavs.items():
                            new_wavs[ni] = w
                    else:
                        # 文脈つきが失敗 → 相槌も文脈なしで作る（plain_need は上で作り済み。従来は need 全部を作り直していた）
                        _plain_synth(list(ctx_need))

                # 挿入相槌の wav がテキストに対して長すぎるもの（文脈の発話を自分の声で
                # 読み上げた異常）を、文脈なしで引き直す。直らなければその相槌を捨てる。
                over = overlong_bc_indices(new_turns, new_wavs, use_sr,
                                           rf_cfg.quality)
                for ni in over:
                    txt = normalize_tts_text(tts_reading.turn_text(new_turns[ni]))
                    lim = _bc_dur_limit(new_turns[ni].get("text", ""), rf_cfg.quality)
                    got = len(new_wavs[ni]) / use_sr
                    for k in range(max(1, rf_cfg.quality.bc_dur_retries)):
                        wavs2, fl2 = synthesize_for_speaker(
                            synth, [txt], assignment[new_turns[ni]["speaker"]],
                            [0.0], did, use_sr, short_threshold=short_threshold,
                            speed=speed)
                        if wavs2 and wavs2[0] is not None and not fl2[0]:
                            new_wavs[ni] = wavs2[0]
                            if len(wavs2[0]) / use_sr <= lim:
                                break
                    fixed = len(new_wavs[ni]) / use_sr
                    logger.warning(
                        f"[{did[:8]}] 挿入相槌 turn{ni}「{new_turns[ni].get('text','')}」の"
                        f"長さが異常（{got:.2f}秒 > 上限{lim:.2f}秒。文脈の発話を読み上げた"
                        f"可能性）→ 文脈なしで引き直し {fixed:.2f}秒")
                drop = {ni for ni in over
                        if len(new_wavs.get(ni, b"")) / use_sr
                        > _bc_dur_limit(new_turns[ni].get("text", ""), rf_cfg.quality)}
                if drop:
                    logger.warning(f"[{did[:8]}] 引き直しでも直らない挿入相槌 "
                                   f"{len(drop)}件を捨てる（turn {sorted(drop)}）")
                    drop_turns(new_turns, new_wavs, new_failed, drop)
                    op_summary["bc_drop_overlong"] = len(drop)
                if over:
                    op_summary["bc_overlong"] = len(over)
                if rf_cfg.quality.bc_leak_check.get("enabled") and need:
                    n_leak = regen_leaky_bcs(
                        synth, new_turns, new_wavs, use_sr, set(need), assignment,
                        rf_cfg.quality.bc_leak_check, did, short_threshold, speed)
                    if n_leak:
                        op_summary["bc_leak_regen"] = n_leak

                # 書き換え・置換で合成し直した発話（相槌は除く）の読み違いを検査する
                # （2026-11-12。それまでは Pass A の合成だけを見ていた）。
                # tts.fish_speech.regen_on_misread.check_refined: false で切れる。
                mcfg = getattr(synth, "regen_misread", None)
                if need and mcfg and mcfg.get("enabled", True) and mcfg.get("check_refined", True):
                    re_need = {ni for ni in need if ni not in new_failed and ni in new_wavs
                               and new_turns[ni].get("type", "speech") != "backchannel"}
                    if re_need:
                        n_mr = regenerate_misread_turns(synth, new_record, voices, new_wavs,
                                                        use_sr, mcfg, only=re_need)
                        if n_mr:
                            op_summary["misread_regen"] = n_mr
                pt.t["PassB再合成(fish)"] = time.perf_counter() - need_t0
                with pt.phase("配置・合成(compose)"):
                    result = compose_dialogue(new_record, voices, use_sr, new_wavs,
                                              new_failed, retimer=retimer,
                                              trim=trim_cfg, noise=noise_cfg,
                                              lead_in=lead_cfg, tail_pad=tail_cfg,
                                              loudness=bc_loud)
                if result is None:
                    failed_count += 1
                    continue
                stereo, manifest_turns = result
                # 無音代替が1発話でもあれば、この対話は何も書き出さない（音声・manifest・
                # 精製済みレコード・キャッシュとも作らない）。wav が無いので次回実行で再処理。
                if drop_if_synth_failed(did, manifest_turns):
                    if tts_servers_dead(synth):
                        # ★2026-11-20 PassB の途中で TTS サーバが全部死んだ → 無音破棄に数えず、ここで終わる
                        logger.error(f"[{did}] TTS サーバが全部応答しないので、この対話は捨てて終わる（無音破棄には数えない）")
                        break
                    silent_dropped += 1
                    continue
                if max_seconds:
                    stereo = stereo[:int(max_seconds * use_sr)]
                _merge_refine_prov(manifest_turns, new_turns)

                _write_outputs(mf, audio_dir, new_record, stereo,
                               manifest_turns, use_sr, rt_cfg, rf_cfg, density,
                               getattr(refiner, "model", None), op_summary,
                               fail_reason, _channel_speakers, trf=trf,
                               voices=voices, fish_backend=fish_backend, lead_cfg=lead_cfg, tail_cfg=tail_cfg)
                rec_out = build_refined_record(new_record, manifest_turns)
                rf_out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                rf_out.flush()
                if rf_cfg.cache_turn_wavs and not dry_run:
                    save_turn_cache(refined_cache, did, new_wavs, new_failed,
                                    use_sr)
                done += 1
                if claims is not None:
                    claims.finish(did)
                logger.info(f"[{did[:8]}] refine 完了: {op_summary}"
                            + (f" / 密度 {density['label']}"
                               f"({density['target_bc_per_min']}/min)" if density else ""))
                # 1対話の内訳。どこが律速かをログから直接集計できるようにする
                # （fish 合成 / MaAI 推論 / refine の LLM / 書き出し）。
                logger.info(f"[{did[:8]}] {pt.summary()}")
                if done % 10 == 0:
                    logger.info(f"進捗: {done}件 (失敗{failed_count}, スキップ{skipped}"
                                f"{f', 無音破棄{silent_dropped}' if silent_dropped else ''})")
            except Exception as e:
                logger.error(f"[{did}] refine 失敗: {type(e).__name__}: {e}\n"
                             + traceback.format_exc())
                failed_count += 1

    drop_empty_part(manifest_file, part_tag)   # 空振りの再実行で空 part を残さない
    drop_empty_part(tr_file, part_tag)
    drop_empty_part(refined_records_file, part_tag)
    if fused:
        drop_empty_part(base_manifest_file, part_tag)
        drop_empty_part(base_tr_file, part_tag)
    if claims is not None:  # 解放は上の closing() 済み。ここは集計ログだけ
        claims.log_summary()
    if hasattr(retimer, "log_summary"):
        retimer.log_summary()
    if tag_skipped:
        logger.warning(f"LLMタグ付け失敗で {tag_skipped}件をスキップしました"
                       "（音声・キャッシュとも未作成。同じコマンドの再実行で再試行されます）")
    if silent_dropped:
        logger.error(f"無音代替が出た {silent_dropped}件を破棄しました（音声・manifest・精製済み"
                     "レコードとも未作成。ログの「無音代替のため対話を破棄」で対話IDを確認し、"
                     "原因を直してから再実行してください）")
    logger.info(("融合" if fused else "refine") + f" 完了: 成功{done} / 失敗{failed_count}"
                f" / スキップ{skipped}"
                + (f" / タグ失敗スキップ{tag_skipped}" if tag_skipped else "")
                + (f" / 無音破棄{silent_dropped}" if silent_dropped else "")
                + (f" / うち1段目から{retimed}件" if fused else "")
                + f" | 出力: {audio_dir}")
    if fused:
        logger.info(f"1段目(retime)の出力: {base_audio_dir}")
    logger.info(f"精製済みレコード: {refined_records_file}")
    return done


class SidonClient:
    """ワーカーの中で完成した対話に Sidon を掛ける（2026-11-17・RUNBOOK 6.12）。

    config の tts.sidon に in_worker: true があるときだけ使う。Sidon のモデルは sidon_server.py（role sidon）に置き、
    ここは完成した 2ch の wav（PCM16）を送って Sidon 後の wav を受け取るだけ（48GB の GPU にはワーカーごとに
    Sidon の約 10GB を載せる余地が無いため）。処理は run_pocket_stage4.sh の後処理（sidon_dialog.process_dialog）と同じ。
      tts.sidon.urls: [http://host:port, ...]    … 直接書く。無ければ hosts × ports の /health を見て探す
      tts.sidon.hosts: [node22, ...]・ports: "9300-9307"
    どのサーバにも掛けられなかった対話は、今までどおり Sidon 前のまま書く（run_pocket_stage4.sh の後処理が拾う）。"""

    def __init__(self, scfg: dict):
        import requests
        self.cfg = scfg
        self.http = requests.Session()
        self.http.trust_env = False            # プロキシを通さない（同じ LAN のサーバ）
        self.params = {k: scfg[k] for k in ("target_dbfs", "eq_db", "eq_hz") if scfg.get(k) is not None}
        self.urls: list[str] = []
        self.last_discover = 0.0
        self.n_ok = self.n_fail = 0
        self._discover()

    def _discover(self):
        self.last_discover = time.time()
        cands = [str(u).rstrip("/") for u in (self.cfg.get("urls") or [])]
        if not cands:
            ps = str(self.cfg.get("ports") or "9300-9307")
            a, _, b = ps.partition("-")
            ports = range(int(a), int(b or a) + 1)
            cands = [f"http://{h}:{p}" for h in (self.cfg.get("hosts") or []) for p in ports]
        found = []
        for u in cands:
            try:
                r = self.http.get(f"{u}/health", timeout=3)
                if r.ok and r.json().get("model") == "sidon":
                    found.append(u)
            except Exception:  # noqa: BLE001
                pass
        self.urls = found
        logger.info("Sidon サーバ %d 台（ワーカーの中で Sidon を掛ける）: %s", len(found), " ".join(found) or "なし")

    def _order(self) -> list[str]:
        """待ちの少ない順（/health の waiting）。答えないサーバは後ろへ。"""
        w = {}
        for u in self.urls:
            try:
                w[u] = int(self.http.get(f"{u}/health", timeout=3).json().get("waiting", 0))
            except Exception:  # noqa: BLE001
                w[u] = 10 ** 6
        return sorted(self.urls, key=lambda u: (w[u], random.random()))

    def restore(self, wav_bytes: bytes, did: str) -> bytes | None:
        if not self.urls and time.time() - self.last_discover > 300:
            self._discover()
        for u in self._order():
            try:
                r = self.http.post(f"{u}/restore", params=self.params, data=wav_bytes, timeout=1800,
                                   headers={"Content-Type": "audio/wav", "X-Dialog-Id": did[:8]})
                if r.ok:
                    self.n_ok += 1
                    return r.content
                logger.warning("[%s] Sidon %s が %d: %s", did[:8], u, r.status_code, r.text[:200])
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] Sidon %s に掛けられない: %s", did[:8], u, e)
        self.n_fail += 1
        return None


_SIDON: SidonClient | None = None
_AUDIO_EXT = ".wav"      # 対話単位の音声の拡張子（run_refine が tts.audio_format から決める・2026-11-17）


def _write_final_wav(out_path: Path, stereo, sr: int, did: str):
    """完成品の音声を書く（拡張子 .wav / .flac で形式が決まる・中身はどちらも同じ PCM16 の値）。
    Sidon をワーカーの中で掛ける設定なら、Sidon 前を <出力先>/audio_raw/<variant>/<id>.flac（可逆）に残し、Sidon 後を audio/ に置く。
    済みの印（audio_raw/<variant>/<完成品の名前>.sidon・更新時刻を Sidon 後の音声と同じにする）は _moss_tune/sidon_swap.py と同じ約束
    → run_pocket_stage4.sh の後処理は、ここで掛けた対話を飛ばし、掛けられなかった対話だけに掛ける。
    書く順: Sidon 前 FLAC → Sidon 後を一時名で書く → 印 → rename（audio/<id>.* が現れた時点で全部そろっている）。"""
    import io

    import soundfile as sf
    if _SIDON is None:
        write_audio(out_path, stereo, sr)
        return
    raw = pcm16_bytes(stereo, sr)                     # 今まで wav に書いていたのと同じ PCM16
    t0 = time.time()
    res = _SIDON.restore(raw, did)
    if res is None:
        logger.warning("[%s] Sidon を掛けられなかった → Sidon 前のまま書く（後処理で掛ける）", did[:8])
        write_wav_bytes_as(out_path, raw)
        return
    raw_dir = out_path.parent.parent.parent / "audio_raw" / out_path.parent.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    pcm, _ = sf.read(io.BytesIO(raw), dtype="int16", always_2d=True)
    sf.write(str(raw_dir / f"{did}.flac"), pcm, sr, format="FLAC", subtype="PCM_16")
    (raw_dir / f"{did}.wav").unlink(missing_ok=True)  # 前の版の Sidon 前（wav）が残っていれば消す
    tmp = out_path.parent / f".{out_path.name}.sidontmp"
    write_wav_bytes_as(tmp, res, out_path.suffix.lstrip("."))
    mt = tmp.stat().st_mtime_ns
    mk = raw_dir / f"{out_path.name}.sidon"
    mk.write_text("")
    os.utime(mk, ns=(mt, mt))
    os.replace(tmp, out_path)
    logger.info("[%s] Sidon（ワーカーの中）%.1f 秒", did[:8], time.time() - t0)


def _merge_refine_prov(manifest_turns: list, turns: list):
    """ターンの refine 来歴を manifest エントリへ反映する。"""
    for entry in manifest_turns:
        t = turns[entry["index"]]
        if "refine" in t:
            entry["refine_op"] = t["refine"]


def _write_outputs(mf, audio_dir: Path, record: dict, stereo, manifest_turns,
                   sr: int, rt_cfg, rf_cfg, density, model, op_summary,
                   fail_reason, _channel_speakers, trf=None, voices=None,
                   fish_backend: str | None = None,
                   lead_cfg: dict | None = None, tail_cfg: dict | None = None):
    import soundfile as sf
    from audio_padding import padding_marks
    from generate_tts import collect_voice_prompts, write_transcript
    did = record["id"]
    out_path = audio_dir / f"{did}{_AUDIO_EXT}"
    _write_final_wav(out_path, stereo, sr, did)
    chsp = _channel_speakers(record)
    voice_prompts = collect_voice_prompts(voices, manifest_turns)
    line = {
        "id": did,
        "audio_path": str(out_path),
        "sample_rate": sr,
        "channels": int(stereo.shape[1]),
        "duration_sec": round(len(stereo) / sr, 3),
        "channel_speakers": chsp,
        "voice_prompts": voice_prompts,
        "turns": manifest_turns,
        "retime_config": rt_cfg.summary(),
        "refine_config": rf_cfg.summary(),
        "refine": {"model": model, "density": density,
                   "ops": op_summary, "failed": fail_reason},
        **padding_marks(did, lead_cfg, tail_cfg),
    }
    if fish_backend:
        line["fish_backend"] = fish_backend
    mf.write(json.dumps(line, ensure_ascii=False) + "\n")
    mf.flush()
    if trf is not None:  # 音声対話モデル用の書き起こし（<id>.json + 統合JSONL）
        write_transcript(audio_dir, trf, record, manifest_turns, sr,
                         int(stereo.shape[1]), len(stereo) / sr, chsp,
                         voice_prompts)
