"""Stage 2: メタデータに基づく対話データの生成"""

import asyncio
import difflib
import functools
import hashlib
import itertools
from collections import Counter
import json
import logging
import os
import random
import re
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from cost_tracker import CostTracker
from partner_persona import partner_card_of, stage2_card_block, strip_cards
from topic_plan import (AL_STYLE, early_self_job, naming_of, partner_knows_job, stage2_topic_block,
                        topic_plan_of, unprompted_job_statement)
from persona import META_KEY as PERSONA_KEY, load_persona, persona_files, persona_of
from al_plan import al_shape_problem
from rp_plan import (OVERLAP_KEY, UNSAID_KEY, is_rp_plan, is_rp_turn, normalize_rp, rp_indices,
                     rp_norm, rp_render, rp_scene_prompt, rp_shape_problem, rp_summary,
                     scene_candidates, scene_variant, scene_window, strip_rp_keys, validate_scene,
                     scene_opener, scene_yield,
                     rp_edit_problem, RP_PROOFREAD_NOTE, RP_NOTE_RE, rp_scene_system)
from qa_plan import qa_shape_problem
from speaker_gender import resolve_gender
from llm_client import (
    build_async_llm_client,
    build_async_llm_clients,
    endpoint_concurrency,
    emphasize_question_marks,
    extract_json_text,
    run_endpoint_worker_pool,
    sanitize_utterance,
)
# enriched モード（Stage3 を廃し Stage2 でフィラー・終助詞・読点まで書く）で使う、
# Stage3 が積み上げた「不自然形」の定義。フィラー語彙を一箇所に保つため import する。
from generate_voice_enrich import (
    _FILLER_ANY,
    _FILLER_WORDS,
    _GLUED_FILLER,
    _HEAD_FILLER,
    _HEAD_REACT,
    _HEE_HEAD,
    _MID_FILLER,
    _NA_STEM_OK,
    _NO_YONE_STEM,
    _POLITE_STEM,
    _REACT_WORDS,
    _SFP_TAIL,
    _TAIL_BREAK,
    _TAIL_COMMA,
    _TAIL_FILLER,
    fix_particle_splits,
    limit_fillers_in_turn,
)

logger = logging.getLogger(__name__)

# gpt-5.6-terra 等の新しめモデルは temperature を既定値しか受け付けない。
# 一度でも temperature 非対応エラーが出たら、以後 temperature を送らない。
_TEMP_UNSUPPORTED = False


async def _chat_create(client, **kwargs):
    """temperature 非対応モデルなら自動で temperature を外して再試行する。"""
    global _TEMP_UNSUPPORTED
    if _TEMP_UNSUPPORTED:
        kwargs.pop("temperature", None)
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:
        if "temperature" in str(e).lower() and "temperature" in kwargs:
            _TEMP_UNSUPPORTED = True
            logger.info("モデルが temperature 非対応のため、既定値で続行します。")
            kwargs.pop("temperature", None)
            return await client.chat.completions.create(**kwargs)
        raise


# ══════════════════════════════════════════════════════════════════════════
# Stage2 プロンプトパック（prompts/stage2_*.yaml）
# ══════════════════════════════════════════════════════════════════════════
# シナリオ（どんな場面の対話を作るか）に依存するのはプロンプト本文だけ。それを
# prompts/stage2_*.yaml に外出しし、config の dialogue.prompt_file で選ぶ
# （Stage1 が scenarios/*.yaml を metadata.scenario_file で選ぶのと同じやり方。
# generate_metadata.load_scenario 参照）。後処理・品質ゲート・リトライ・分量の数値
# （dialogue.enrich）はシナリオに依らないので、この先のコードは1本のまま全シナリオで
# 共有する（コピーを作ると改良が片方にしか入らない）。
DEFAULT_PROMPT_FILE = "prompts/stage2_general.yaml"
# 読みの注記の共通プロンプト（パックが yomi_prompt を持たないときに使う。全シナリオ共通）
DEFAULT_YOMI_FILE = "prompts/yomi_common.yaml"

# enriched モードの本体（enriched_section）を差し込む位置。
ENRICHED_ANCHOR = "# 会話の終わらせ方"

# system_prompt の中に**そのまま含まれていなければならない**断片。build_system_prompt
# はここを別の文面に差し替えることで、enriched モードやゲートの切り替えを効かせる。
# 本文だけ編集して断片を直し忘れると差し替えが黙って空振りし、「enriched なのに素の
# スキーマ」「ゲートを切ったのに禁止指示が残る」プロンプトがそのまま LLM に飛ぶので、
# 読み込み時に照合して落とす。
_PACK_ANCHORS = ("schema_plain", "filler_lines_loose", "no_self_name_lines",
                 "long_utt_quota_sys")


@dataclass(frozen=True)
class S2PromptPack:
    """Stage2 のプロンプト本文一式。中身は prompts/stage2_*.yaml のキーと同じ。"""
    name: str
    system_prompt: str
    schema_plain: str
    schema_enriched: str
    filler_lines_loose: str
    filler_lines_enriched: str
    enriched_section: str
    laugh_section: str
    no_self_name_lines: str
    allow_self_name_lines: str
    long_utt_quota_sys: str
    long_utt_free_sys: str
    long_utt_quota_user: str
    long_utt_free_user: str
    profile_patches: dict
    # 校閲パス（proofread_dialogue）のシステムプロンプト。無いパックでは校閲しない。
    proofread_prompt: str = ""
    # 読みの注記（annotate_readings）のシステムプロンプト。無いパックでは注記しない。
    yomi_prompt: str = ""


def build_prompt_pack(d: dict, where: str = "<dict>") -> "S2PromptPack":
    """dict からパックを組む。キーの欠落と差し替え元の不一致はここで落とす。"""
    fields = [f for f in S2PromptPack.__dataclass_fields__
              if f not in ("name", "profile_patches", "proofread_prompt", "yomi_prompt")]
    missing = [k for k in fields if not str(d.get(k) or "").strip()]
    if missing:
        raise ValueError(f"Stage2 プロンプトパック {where} にキーがありません: "
                         f"{'、'.join(missing)}")
    body = d["system_prompt"]
    for key in _PACK_ANCHORS + ("ENRICHED_ANCHOR",):
        frag = ENRICHED_ANCHOR if key == "ENRICHED_ANCHOR" else d[key]
        if frag not in body:
            raise ValueError(
                f"Stage2 プロンプトパック {where} の {key} が system_prompt の中に"
                f"見つかりません（build_system_prompt がここを差し替えます）。"
                f"本文を編集したら、この断片も同じ文面に直してください:\n{frag[:60]}...")
    patches = {}
    for prof, per_key in (d.get("profile_patches") or {}).items():
        patches[prof] = {}
        for key, pairs in (per_key or {}).items():
            try:
                patches[prof][key] = tuple((x["old"], x["new"]) for x in pairs)
            except (TypeError, KeyError) as e:
                raise ValueError(
                    f"Stage2 プロンプトパック {where} の profile_patches.{prof}.{key} は"
                    f" old / new を持つ項目の並びで書いてください: {e}") from e
    return S2PromptPack(name=str(d.get("name") or Path(where).stem),
                        profile_patches=patches,
                        proofread_prompt=str(d.get("proofread_prompt") or ""),
                        yomi_prompt=str(d.get("yomi_prompt") or _default_yomi_prompt()),
                        **{k: d[k] for k in fields})


@functools.lru_cache(maxsize=None)
def _default_yomi_prompt() -> str:
    """共通の読み注記プロンプト（prompts/yomi_common.yaml）。ファイルが無ければ空＝注記しない。"""
    p = Path(__file__).resolve().parent.parent / DEFAULT_YOMI_FILE
    if not p.exists():
        return ""
    with open(p) as f:
        return str((yaml.safe_load(f) or {}).get("yomi_prompt") or "")


@functools.lru_cache(maxsize=None)
def load_prompt_pack(path: str | None = None) -> S2PromptPack:
    """プロンプトパック(YAML)を読む。相対パスはリポジトリのルート基準（モジュールは pipeline/ にある）。"""
    p = Path(path or DEFAULT_PROMPT_FILE)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    with open(p) as f:
        return build_prompt_pack(yaml.safe_load(f) or {}, str(p))


GENERAL_S2 = load_prompt_pack()

# 既定パックの中身は、従来どおりモジュール変数としても公開する（テストや他スクリプト
# からの import 互換のため）。実行時に使うのは、引数で渡された S2PromptPack の側。
# 本文と、その文面に至った経緯のコメントは prompts/stage2_*.yaml にある。
SYSTEM_PROMPT = GENERAL_S2.system_prompt
_PLAIN_SCHEMA = GENERAL_S2.schema_plain
_ENRICHED_SCHEMA = GENERAL_S2.schema_enriched
_LOOSE_FILLER_LINES = GENERAL_S2.filler_lines_loose
_ENRICHED_FILLER_LINES = GENERAL_S2.filler_lines_enriched
ENRICHED_SECTION = GENERAL_S2.enriched_section
LAUGH_SECTION = GENERAL_S2.laugh_section
_NO_SELF_NAME_LINES = GENERAL_S2.no_self_name_lines
_ALLOW_SELF_NAME_LINES = GENERAL_S2.allow_self_name_lines
_LONG_UTT_QUOTA_SYS = GENERAL_S2.long_utt_quota_sys
_LONG_UTT_FREE_SYS = GENERAL_S2.long_utt_free_sys
_LONG_UTT_QUOTA_USER = GENERAL_S2.long_utt_quota_user
_LONG_UTT_FREE_USER = GENERAL_S2.long_utt_free_user
PROMPT_PATCHES = GENERAL_S2.profile_patches


# ══════════════════════════════════════════════════════════════════════════
# enriched モード（config の dialogue.mode: enriched）
#
# 従来は Stage2 が素の台本を書き、Stage3(generate_voice_enrich.py) が後からフィラー・
# 終助詞・冒頭リアクションを差し込んでいた。この「編集」方式は差し込み位置が機械的になり
# （実測:「うわ、マジで？、まあ、何がボトルネックか教えて。」＝疑問符の後、
# 「落とし込んで、うーん、いいかもね。」＝意味の切れ目でない位置）、量も過剰だった
# （人間コーパス maai-f2f-jp-2025 のフィラー含有ターン 15〜22% に対し Stage3 は 36.7%）。
# enriched モードは Stage2 に最初から言いよどみ込みで書かせ、Stage3 を廃止する。
# 規則（使ってよいフィラー語彙・置いてはいけない位置・終助詞の制約）は Stage3 の
# プロンプトで積み上げたものを、生成時の規則として書き直して流用している。
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class S2Params:
    """enriched モードの付与量。config の `dialogue.enrich` で上書きする。

    既定値は人間コーパス maai-f2f-jp-2025（40セッション/27,884ターン、相槌・笑いを
    除去し同一話者の連続 IPU を併合して計測）の実測値:
        フィラー含有ターン 15.3〜21.7% / 文末終助詞 27.1% / 読点 7.73個per100実質字
    """
    filler_ratio: float = 0.18       # フィラーを入れるターンの目安割合（プロンプト）
    filler_cap_ratio: float = 0.25   # フィラー付きターンの上限比率（超過分は機械間引き）
    fillers_per_turn: int = 1        # 1ターンに入れてよいフィラー数の上限
    filler_inject: bool = False      # 目安に届かないぶんを文中へ機械的に差し込むか
    # ★発話が長いほどフィラーを増やす段階制。実質 filler_len_step 文字ごとに1個増え、
    # fillers_per_turn で頭打ちになる（0 なら従来の「25字以下は1個・超えたら上限」）。
    # 長い発話をすらすら言い切らせると、音声にしたとき機械の読み上げに聞こえる、という
    # 聴感上の問題への対処（2026-07-27）。プロンプト側の分量表も同じ関数から生成する。
    filler_len_step: int = 0
    # 読点なしで続く区間がこの実質文字数を超えたら、節の切れ目で機械的に割る
    # （0 なら割らない）。割れるのは「けど/ので/から/ば/たら/ながら/連用中止の て」等の
    # 語中一致しない形が見つかったときだけなので、実測では 5% 前後のターンにしか効かない。
    comma_run_max: int = 0
    # 冒頭フィラーの目安割合。人間コーパスの内訳は 冒頭 19.6% / 文中 13.5% で、
    # 合成側は逆（冒頭 7.4% / 文中 23.3%）だった。文中の差し込み先は「モデルが打った
    # 読点」に限られて 24% で頭打ちになるので、密度を上げるぶんは冒頭で補う。
    # 0 なら冒頭には差し込まない（従来動作）。
    head_filler_ratio: float = 0.0
    # 冒頭の笑い（「あはは、」「ふふ、」）を置くターンの目安割合と上限比率。
    # 人間コーパスの実測は 2.4%（speech ターンの冒頭が笑いで始まる割合）。
    # laugh_ratio が 0 ならプロンプトに笑いの節を入れない（従来動作）。
    laugh_ratio: float = 0.0
    laugh_cap_ratio: float = 0.06
    sfp_ratio: float = 0.27          # 文末終助詞で終えるターンの目安割合
    sfp_cap_ratio: float = 0.35      # 終助詞ターンの上限比率
    # 「ね」だけの上限（発話100文字あたりの個数。0 ならゲート無し＝従来動作）。
    # 全体の上限と別に持つのは、内訳の比率（人間はね54%）は合っていて総量だけが
    # 多いのが実測だから。単位が比率でなく文字数あたりなのは、人間コーパスと同じ
    # 物差しで測るため（ターンの区切り方に左右されない）。
    ne_cap_per_100: float = 0.0
    # 上限超過ぶんのタメ口の「ね」を「な」「よ」に替えるか（既定 False＝従来動作）。
    ne_swap_casual: bool = False
    # 担当者が客の用件を復唱する言い切り（「〜をご希望でございます。」）に「ね」を足す
    # 範囲。ターン番号がこの値未満なら1文目のコピュラ言い切りすべて、それ以降は復唱の印
    # が付いたものだけを直す（add_readback_ne）。0 で無効。担当者のいないシナリオでは
    # 値に関わらず何もしない。
    readback_ne_window: int = 8
    # 客が担当者の案内を受け止める言い切り（体言止め「前日の15時。」、「〜んです。」）に
    # 「ね」を足すか（add_customer_echo_ne）。上と対。False で無効。
    customer_echo_ne: bool = True
    sfp_inject: bool = False         # 終助詞の無い平叙ターンに機械的に足すか
    reaction_ratio: float = 0.12     # 冒頭リアクションを置くターンの目安割合
    # ★文が終わっている読点を句点に替える（to_sentence_periods）。32B は文を読点で
    # つなぐ癖があり（実測 22対話841ターンで188箇所）、音声にすると一息で流れてしまう。
    # 既定 False＝従来動作（v3 以前の config の再現性を保つ）。
    period_split: bool = False
    comma_per_100: float = 7.7       # 読点の密度（実質100文字あたり）
    comma_span: int = 12             # 読点なしで続けてよい文字数（プロンプトの局所ルール）
    monotone_head_cap: int = 3       # 頭「まあ/うーん」の対話内上限
    # 下限ゲート（この値を下回る対話は作り直す）。目安割合との倍率ではなく実数で持つ。
    # 32B はフィラーだけは指示に反応しない（要求18%でも35%でも実測3〜12%で変わらない）
    # ので、フィラーの下限は「素通しでないこと」を確かめる控えめな値にし、到達不能な
    # ノルマで無限リトライしない。終助詞・読点は指示に反応するので目安の8割を要求する。
    filler_floor: float = 0.08
    sfp_floor: float = 0.20
    comma_floor: float = 6.0         # 実質100文字あたりの「間」（読点＋文中句点）の下限
    # 短い句の直後の読点（「私は、」「大学で、」）を外す（thin_short_commas）。前の句が
    # 実質この文字数以下で助詞で終わるときだけ外す。0 なら外さない（既定＝従来動作）。
    # 書き起こしとしては人間並みの読点でも、fish は読点ごとに間を入れるので音声では
    # 切れ切れに聞こえる、という聴感上の問題への対処（2026-09-27、初対面）。
    comma_thin_seg: int = 0
    # 外した結果、読点なしで続く区間がこの実質文字数を超えるなら外さない。
    comma_thin_run: int = 24
    # 冒頭リアクション率の下限。0 ならゲート無し（既定＝従来動作）。指示だけでは
    # 届かない（実測: 指示20%に対し 9.0%）ので、他の3指標と同じく測って作り直す。
    react_floor: float = 0.0
    # フィラーのターン数ノルマを「2人でほぼ半分ずつ」と添えるか（既定 False＝従来動作）。
    # ノルマは対話全体の実数なので、主話者のペルソナでフィラーを促すと相手のぶんが削られて
    # 枠ごと主話者に寄る（2026-10-02 実測: ペルソナ v2 だけで イノウエ 1.11→2.33・
    # 相手 0.97→0.47 個per100字。output_ft_v1_gemma_inoue_002/trial_v2_persona_only）。
    # ★2026-10-03 「ほぼ半分ずつ」の文言だけでは守られず（相手 0.59）、話者ごとのターン数を
    #   数字で示す形にした（比率指示より実数のほうが従う、という _enrich_norm の方針と同じ）。
    filler_split: bool = False
    # ★2026-10-03 初対面 Gemma の型（output_ft_v1_gemma_inoue_002 の20対話で計測）。既定 False＝従来動作。
    # echo_fix: 相手の語を1語くり返して助詞や句点で切った発話（「設計を。具体的に〜」「デスクワーク。」
    #   だけのターン）を「設計ですか。」の形にする（fix_echo_fragments）。
    # ne_keep_eval: 感想・評価の「ね」（「面白いですね」「素敵ですね」「ということですね」）を
    #   thin_ne / thin_sfp で落とさない。落とすと「それ、結構面白い分野です。」「料理の一部という
    #   ことです。」の硬い言い切りになる（20対話で19件。終助詞の間引きは計73件）。
    echo_fix: bool = False
    ne_keep_eval: bool = False
    # 冒頭で相手の語をくり返す「○○ですか？」の1対話あたりの上限（0＝無効）。超えた分は、後ろに文が
    # 続くものから冒頭を外す。あわせて、後ろに文が続く「○○ですか？」は「。」（受け止め）にする
    # （limit_echo_heads）。2026-10-03: プロンプトの例を「カフェですか？」にしたら 1対話平均4.0・
    # 最大9に増え、73/79 が「？」、8割は答えを待たずに感想が続く形だった。
    echo_head_cap: int = 0
    # 相手の語をくり返して受ける発話（「X。続き」「Xですか。続き」「Xですか？」）全体の1対話あたりの
    # 上限（0＝無効）。超えた分は、続きのあるものはくり返しを外し、単独の聞き返しは短い反応に替える
    # （limit_echoes）。★2026-10-04 試走100対話: 1対話平均6.6・最大12、全ターンの20%（001 は22%）。
    echo_total_cap: int = 0
    # 終助詞の間引き（thin_ne / thin_sfp）で、相手の発話を受けた最初の文（同意・感想）を削る順番の
    # 最後に回す（上限は変えない）。★2026-10-04 本番147対話: 「ね」は 0.91/100字と上限 0.5・人間 0.57 を
    # すでに超えているのに、「最近、流行っています。」「抵抗感があります。」のような受け答えの
    # 「ね」が削られて硬い言い切りが残っていた。足りないのではなく、削る文の選び方の問題。
    ne_reaction_last: bool = False
    # ── ターンの形（2026-10-28。いずれも 0／False なら従来どおり何もしない）────────
    # 人間コーパス maai-f2f-jp-2025 の**先頭33ターン**（合成と同じ尺）と比べた差から入れた。
    #   長さの帯   人間 0〜6字22%／7〜15字41%／16〜30字25%／31〜55字9%
    #              010  人間      2%／     15%／     71%／     11%
    #   疑問       人間 7.8%・010 19.1%（主話者の問い→相手の答えが主ターンの21%＝インタビュー調）
    #   受け止め語 1000ターンあたり なるほど 人間4.2/合成63.7、いいですね 6.8/59.3、
    #              素敵 0.4/13.0、そうですね 21.5/63.7、あー 47.5/11.6
    # short_turn_ratio: 7〜15実質字のターン数を実数でプロンプトに書く（比率指示は従われない）。
    short_turn_ratio: float = 0.0
    # long_turn_ratio: 31〜55実質字のターン数のノルマ（短い側だけ縛ると長めが消える）。
    long_turn_ratio: float = 0.0
    # short_turn_floor: 実質15字以下のターンの割合の下限。下回ったら作り直す（spokenness ゲート）。
    short_turn_floor: float = 0.0
    # question_ratio_cap: 「？」を含むターンの割合の上限。プロンプトに実数で書き、
    #   超えた出力は作り直す（spokenness ゲート）。
    question_ratio_cap: float = 0.0
    # ack_head_cap / ack_phrase_cap: 受け止めの定型語（なるほど・いいですね・そうですね等）で
    #   始まるターンの、1対話あたりの総数／同じ語ごとの上限（rebalance_ack_heads）。
    #   総数を超えた分は冒頭を外し、語ごとの上限を超えた分は人間に多い感動詞へ替える。
    ack_head_cap: int = 0
    ack_phrase_cap: int = 0
    # react_vocab_human: 冒頭リアクションの語彙の指示を人間コーパスの配分に寄せる
    #   （「あー」を筆頭にし、「なるほど」「いいね」を候補から外す）。
    react_vocab_human: bool = False
    # vary_closing: 締めの言い方を毎回同じにしない指示を足す。学習データの**全サンプルの
    #   末尾**なので、同じ文がそろうと目立つ（実測: 最終ターンが 20対話で12種・
    #   「ありがとうございました。失礼します。」が30%。人間コーパスは 40対話で34種）。
    vary_closing: bool = False
    # ── 2026-10-29（ft_v2 試走の評価から。0／False なら従来どおり）────────────
    # shift_mark_cap: 「（あ、）そういえば」「ところで」で始まる文の1対話あたりの上限。
    #   超えた分は前置きを外す（cap_shift_marks）。試走 1.6回/対話・人間の先頭33ターン 0.02回。
    shift_mark_cap: int = 0
    # stock_word_swap: Gemma の口癖「心地いい・心地よい」を「気持ちいい・気持ちよい」へ
    #   （swap_stock_words）。試走 29.5回/1000ターン（v1 15.7）・人間 0.2。
    stock_word_swap: bool = False
    # echo_replace_once: limit_echoes の置き換え語（「そうなんですか？」等）を1対話で各1回まで。
    echo_replace_once: bool = False
    # ── 2026-10-29 再評価（v2 最終版の評価から。既定は従来どおり何もしない）──────────
    # main_echo_cap: 主話者の単独の聞き返し（「カラオケですか？」だけのターン）の1対話の上限。
    #   超えた分は消して相手の前後の発話を1ターンにつなぐ（fold_main_echoes）。-1 で無効。
    main_echo_cap: int = -1
    # main_react_fold: 主話者の反応だけのターン（「あー。」等）を畳んで相手の発話をつなぐ1対話の上限
    #   （つないで31字以上になる箇所だけ。fold_main_reactions）。0 で無効。
    main_react_fold: int = 0
    # flag_rewrite: 機械で数えて上限を超えた発話だけを LLM に書き直させる（rewrite_flagged_turns）。
    #   calm_word_cap: 「静かな時間」語群を含むターンの1対話の上限（両話者）。
    #   main_eval_cap: 主話者の評価・称賛・一般論のまとめのターンの1対話の上限。
    #   ほかに topic_plan に締めがある対話では、末尾の時間を口実にした締めを書き直させる。
    flag_rewrite: bool = False
    calm_word_cap: int = 2
    main_eval_cap: int = 1
    #   time_close_rewrite: 末尾の時間を口実にした締めも書き直させる（★既定 False。書き直すと次の
    #   返事「こちらこそ。ありがとうございます。」が宙に浮いた。締めの指示 CLOSING_COMMON だけで 25%→5〜10%）。
    time_close_rewrite: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "S2Params | None":
        """dialogue.mode が enriched のときだけ S2Params を返す（既定は None＝従来動作）。"""
        d = (config.get("dialogue") or {})
        if d.get("mode") != "enriched":
            return None
        kw = {k: v for k, v in (d.get("enrich") or {}).items()
              if k in cls.__dataclass_fields__}
        return cls(**kw)


# 段階制のとき、これ未満の短い発話にはフィラーを入れない（前置きだけで間延びする）。
_FILLER_MIN_LEN = 16


def filler_quota(core_len: int, sp: "S2Params") -> int:
    """その発話に入れたい（＝入れてよい）フィラーの数。長い発話ほど多い。

    sp.filler_len_step が 0 なら従来どおり「実質25字以下は1個、超えたら
    fillers_per_turn 個まで」。>0 なら _FILLER_MIN_LEN 字から filler_len_step 字ごとに
    1個ずつ増やし、fillers_per_turn で頭打ちにする。プロンプトの分量表も同じ値から
    作るので、指示と後処理がずれない。
    """
    if sp.filler_len_step <= 0:
        return 1 if core_len <= 25 else sp.fillers_per_turn
    if core_len < _FILLER_MIN_LEN:
        return 0
    return min(sp.fillers_per_turn,
               1 + (core_len - _FILLER_MIN_LEN) // sp.filler_len_step)


def _len_bands(sp: "S2Params") -> list:
    """発話長の帯 (下限, 上限 or None, フィラー数, 読点の最低数) を返す。"""
    if sp.filler_len_step <= 0:                 # 従来の2段（25字以下 / 超）
        edges = [(1, 25), (26, None)]
        return [(a, b, filler_quota(a, sp), a // max(sp.comma_span, 1)) for a, b in edges]
    step = sp.filler_len_step
    edges, lo = [(1, _FILLER_MIN_LEN - 1)], _FILLER_MIN_LEN
    while len(edges) < 8:
        if filler_quota(lo, sp) >= sp.fillers_per_turn:
            edges.append((lo, None))
            break
        edges.append((lo, lo + step - 1))
        lo += step
    return [(a, b, filler_quota(a, sp), a // max(sp.comma_span, 1)) for a, b in edges]


def _len_table(sp: "S2Params") -> str:
    """プロンプトに載せる「発話の長さごとのフィラー・読点の個数」表。"""
    rows = []
    for lo, hi, nf, nc in _len_bands(sp):
        span = f"実質{hi}文字以下" if lo == 1 else (
            f"実質{lo}文字以上" if hi is None else f"実質{lo}〜{hi}文字")
        comma = "読点は無理に打たない" if nc <= 1 else f"読点{nc}個以上"
        rows.append(f"  - {span} … フィラー{nf}個。{comma}。")
    return "\n".join(rows)


# ── 品質ゲートの有効・無効 ─────────────────────────────────────────────
# ゲートはシナリオを選ぶ。既定は全部有効で、config の dialogue.gates で個別に切る。
# 例: 初対面のシナリオは冒頭で名乗るのが自然なので dialogue.gates.self_name: false。
# 切ったゲートに対応する禁止指示はプロンプト側からも外れる（build_system_prompt）。
# JSON が壊れている「不正な形式」は品質ではなく前提なので、ここには入れない。
GATE_NAMES = (
    "masked_text",          # 伏せ字（「○○さん」）を含む出力
    "truncated_utterance",  # 途中が抜けた発話（「不安がですよね。」）
    "self_name",            # 自分の発話に自分の名前（名乗り・話者取り違え）
    "missing_period",       # 文末句点の欠落（箇条書き調の退化）
    "speaker_names",        # メタデータに無い話者名（名前の文字化け・3人目）
    "padding_after_close",  # 締めの挨拶の後に続く水増し
    "vacuous_short_turn",   # 中身のない極短ターンの羅列
    "casual_leak",          # 敬語指定なのに砕けた語が混ざる（口調崩れ）
    "self_job_early",       # 主話者が序盤に聞かれずに自分の職業を言う（topic_plan があるときだけ）
    "main_closes",          # 主話者が締めの挨拶を切り出す（topic_plan に締めがあるときだけ）
    "qa_shape",             # 質問と答えの形の崩れ（topic_plan が style: qa のときだけ。qa_plan）
    "al_shape",             # 傾聴の形の崩れ（topic_plan が style: al のときだけ。al_plan）
    "rp_shape",             # 重なりの場面の不足・崩れ（topic_plan が style: rp のときだけ。rp_plan）
    "long_utterance",       # 長め発話（31文字以上）の不足
    "spokenness",           # 話し言葉らしさ（enriched の下限）
    "silence",              # ユーザ側の長い沈黙の不足（dialogue.silence があるときだけ）
)

ALL_GATES = {name: True for name in GATE_NAMES}

# self_name ゲートだけは「何回まで許すか」を持つ。名乗りのあるシナリオでゲートを
# 丸ごと切ると、32B は名乗りを繰り返す（実測: 1対話36ターン中8回、同じ「〜です」を
# 4回。プロンプトに「名乗るのは一度だけ」と書いても守らない）。回数で持てば、
# 自己紹介は通しつつ繰り返しは弾ける。0 なら従来どおり1回も許さない。
DEFAULT_SELF_NAME_MAX = 0


def load_proofread(config: dict) -> int:
    """dialogue.proofread（校閲パスの回数。既定 0＝従来動作。true は 1 回）。
    パックに proofread_prompt が無ければ何もしない。2回目は1回目で直した対話を読み直す
    （1回だけでは指摘の拾い漏れが多い: 2026-09-23 の実測で約35%が残った）。"""
    v = (config.get("dialogue") or {}).get("proofread", 0)
    return int(v) if not isinstance(v, bool) else int(v)


# 言いかけのまま切れた発話の末尾（句点なしで助詞・接続助詞・読点で終わる）
_CUT_OFF_TAIL_RE = re.compile(r"(?:まで|から|けど|けれど|ので|のに|ため|て|で|が|を|に|へ|と|の|は|も|や|、)$")


def load_yomi_ruby(config: dict) -> int:
    """dialogue.yomi_ruby（発話全体の読み。既定 false）。true なら同時依頼数 8、{conc: N} で指定。0＝無効。"""
    v = (config.get("dialogue") or {}).get("yomi_ruby", False)
    if isinstance(v, dict):
        return int(v.get("conc", 8)) if v.get("enabled", True) else 0
    return 8 if v else 0


async def annotate_ruby(client, model: str, turns: list, cost_tracker: CostTracker,
                        extra_body: dict | None = None, conc: int = 8,
                        only_missing: bool = False) -> tuple[int, int]:
    """各発話に発話全体の読み（yomi_ruby）を付け、turn["ruby"] に持たせる（本文は変えない）。

    1回1発話・対話全体を文脈に・温度0で、語単位のルビ（C）と全文ひらがな（B）を取る。
    Stage4 の読み違いの検査（reading_check の V3）と、引き直しで語をかなに開くのに使う。
    (付けた発話数, 対象の発話数) を返す。付加的な工程なので、失敗しても対話は残す。
    only_missing: 読みの無い（または本文が変わった）発話だけに付け、付いている読みは残す。
    """
    import yomi_ruby as YR
    lines = [(str(t.get("speaker", "")), str(t.get("utterance", ""))) for t in turns]
    idx = [i for i, (_, u) in enumerate(lines) if YR.needs_ruby(u)]
    if only_missing:
        idx = [i for i in idx if not YR.valid(turns[i].get("ruby"), lines[i][1])]
    else:
        for t in turns:
            t.pop("ruby", None)
    if not idx:
        return 0, 0
    try:
        got = await YR.annotate_turns_async(YR.async_chat(client, model, extra_body, cost_tracker),
                                            lines, idx, conc)
    except Exception as e:
        logger.warning(f"発話全体の読みに失敗（読みなしで採用）: {e}")
        return 0, len(idx)
    for i, r in got.items():
        turns[i]["ruby"] = r
    return len(got), len(idx)


def load_yomi(config: dict) -> bool:
    """dialogue.yomi（読みの注記。既定 false）。パックに yomi_prompt が無ければ何もしない。"""
    return bool((config.get("dialogue") or {}).get("yomi", False))


# ── ユーザ側の長い沈黙（dialogue.silence。2026-10-23 ft_sil_v1）──────────────
# ユーザ（主話者でない方）が1対話に数回、長く黙る台本を作る。沈黙が明けたターンに
# pause_before_sec（秒）を持たせ、Stage3 → Stage4 の retime までこのキーで運ぶ
# （retime_maai.MaaiRetimer.plan はこの秒数をそのまま置き、VAP では決めない）。
# 沈黙の主はいつもユーザで、並びは次の3つだけを認める（normalize_pauses）:
#   A: ユーザ → ユーザ（言いかけて途切れ、黙ったあと自分で続ける。同じ話者が2ターン続く）
#   B: 主話者 → ユーザ（主話者の問いのあと考え込んでから答える）
#   C: ユーザ → 主話者（途切れて黙ったユーザを主話者が促す。max_prompt 回まで）
# config に silence が無ければ従来どおり（pause_before_sec は消して、交互に結合する）。
PAUSE_KEY = "pause_before_sec"


@dataclass(frozen=True)
class SilenceParams:
    count_min: int = 2
    count_max: int = 3
    sec_min: float = 2.0
    sec_max: float = 8.0
    max_prompt: int = 1
    prompt_ratio: float = 0.0     # 型 C（促す）を1回入れさせる対話の割合（対話 id で決定的に選ぶ）
    prompt_sec_min: float = 0.0   # 型 C の待ちの下限（主話者が促すまで。sec の下限より長くしたいとき）
    prompt_retries: int = 2       # 型 C が出ないときの作り直しの回数（超えたら C なしで採用）
    jitter_sec: float = 0.0       # 確定した秒数を ±jitter_sec 揺らす（対話 id とターン番号で決定的）


def load_silence(config: dict) -> "SilenceParams | None":
    """dialogue.silence（既定 無効）。count / sec は [下限, 上限]。"""
    raw = (config.get("dialogue") or {}).get("silence")
    if not raw or raw.get("enabled", True) is False:
        return None
    c = raw.get("count", [2, 3])
    sec = raw.get("sec", [2.0, 8.0])
    return SilenceParams(count_min=int(c[0]), count_max=int(c[1]),
                         sec_min=float(sec[0]), sec_max=float(sec[1]),
                         max_prompt=int(raw.get("max_prompt", 1)),
                         prompt_ratio=float(raw.get("prompt_ratio", 0.0)),
                         prompt_sec_min=float(raw.get("prompt_sec_min", 0.0)),
                         prompt_retries=int(raw.get("prompt_retries", 2)),
                         jitter_sec=float(raw.get("jitter_sec", 0.0)))


def silence_roles(metadata: dict) -> tuple:
    """(主話者, ユーザ) の名前。main_speaker_index が無い・2人でないなら (None, None)。"""
    names = [str(s.get("name")) for s in (metadata.get("speakers") or [])
             if isinstance(s, dict) and s.get("name")]
    idx = metadata.get("main_speaker_index")
    if len(names) != 2 or idx is None:
        return None, None
    try:
        main = names[int(idx)]
    except (IndexError, TypeError, ValueError):
        return None, None
    return main, names[1 - names.index(main)]


def _pause_kind(turns: list, i: int, main: str, user: str) -> "str | None":
    if i == 0 or not isinstance(turns[i - 1], dict):
        return None
    prev, cur = turns[i - 1].get("speaker"), turns[i].get("speaker")
    if prev == user and cur == user:
        return "A"
    if prev == main and cur == user:
        return "B"
    if prev == user and cur == main:
        return "C"
    return None


def normalize_pauses(turns: list, main: "str | None", user: "str | None",
                     sil: SilenceParams, dropped: "list | None" = None) -> list:
    """LLM が書いた pause_before_sec を検査して残す（残した (index, 型) の一覧を返す）。
    dropped を渡すと、外したものの「turn番号:理由」を足す（ログ用）。

    数値にして sec の範囲に収める。数値でない・0以下・先頭ターン・沈黙の主がユーザに
    ならない並び（主話者 → 主話者）・C の max_prompt 超え・count の上限超え（後ろから）は外す。
    外したユーザ → ユーザの並びは、このあとの merge_consecutive_same_speaker で1ターンになる。

    ★2026-10-24 試走2回目を受けて足したもの:
    - 秒数の付け違いの直し: ユーザの言いよどみ → 主話者の促し（秒数なし）→ ユーザ（秒数あり）は、
      LLM が促しに付けるべき秒数を次のユーザに付けたもの。秒数を促しのターンへ移す
      （割り込みゲートのリトライ 19回のほとんどがこの並びだった）。
    - C は直前のユーザの発話が言いよどみ・言いさし（is_trailing）のときだけ。それ以外は主話者が
      自分で考え込んだ沈黙なので外す（「コツとかあるかな？」→ 5.2秒 → イノウエ「うーん、…」）。
    - 隣り合うターンの沈黙は前だけ残す（3回のうち2回が1か所に固まるため）。
    - C の秒数は prompt_sec_min 以上にする（促すまで待つ長さが下限の2秒にそろうため）。
    """
    if main is not None:
        for i in range(1, len(turns) - 1):
            a, b, c = turns[i - 1], turns[i], turns[i + 1]
            if (isinstance(a, dict) and isinstance(b, dict) and isinstance(c, dict)
                    and a.get("speaker") == user and b.get("speaker") == main
                    and c.get("speaker") == user and PAUSE_KEY not in b and PAUSE_KEY in c
                    and is_hesitation(a.get("utterance", ""))):
                b[PAUSE_KEY] = c.pop(PAUSE_KEY)
                if dropped is not None:
                    dropped.append(f"{i + 1}:促しへ移動")
    kept, n_prompt = [], 0
    drop = dropped.append if dropped is not None else (lambda _: None)
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or PAUSE_KEY not in t:
            continue
        raw = t.pop(PAUSE_KEY)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            drop(f"{i}:数値でない({raw!r:.12})")
            continue
        if not v > 0 or main is None:
            drop(f"{i}:{'0以下' if main else '主話者不明'}")
            continue
        kind = _pause_kind(turns, i, main, user)
        if kind is None:
            drop(f"{i}:主話者の沈黙" if i else "0:先頭")
            continue
        if kind == "C" and not is_trailing(turns[i - 1].get("utterance", "")):
            drop(f"{i}:主話者の沈黙(促しでない)")
            continue
        # B（考え込む）は主話者の問いのあとだけ。★2026-10-24 試走7回目: 誘い（「ぜひ一緒に
        # 行きましょう。」→ 5.5秒 →「うーん。そうですね。ぜひ。」）や主話者自身の「うーん。」の
        # あと（ユーザが自分の問いに自分で答える形）に付いていた。
        prev_u = str(turns[i - 1].get("utterance", ""))
        if kind == "B" and ("？" not in prev_u or is_hesitation(prev_u)):
            drop(f"{i}:問いでない主話者のあと")
            continue
        if kept and kept[-1][0] == i - 1:
            drop(f"{i}:隣の沈黙と連続")
            continue
        if len(kept) >= sil.count_max:
            drop(f"{i}:上限超え")
            continue
        if kind == "C":
            if n_prompt >= sil.max_prompt:
                drop(f"{i}:促しの上限超え")
                continue
            n_prompt += 1
        lo = max(sil.sec_min, sil.prompt_sec_min) if kind == "C" else sil.sec_min
        t[PAUSE_KEY] = round(min(max(v, lo), sil.sec_max), 1)
        kept.append((i, kind))
    return kept


def pre_pause_indices(turns: list) -> set:
    """次のターンが長い沈黙のあと（pause_before_sec を持つ）であるターンの番号。"""
    return {i for i in range(len(turns) - 1)
            if isinstance(turns[i + 1], dict) and turns[i + 1].get(PAUSE_KEY) is not None}


# 言いよどみだけの発話（「えっと。」「うーん。」）と、接続語で言いさした短い発話（「それが。」「でも。」）。
# ユーザがこう言ったのに、主話者が沈黙を置かずにすぐ話すのは「待たずに割り込む」形で、
# このコーパスの狙い（システムが待つ）と逆になる（2026-10-24 試走で 20対話中2件）。
_HESITATION_ONLY = re.compile(
    r"^(?:(?:えっと|えーっと|えーと|ええと|えー|あの|あのー|その|うーん|んー|なんだろう|"
    r"なんていうか|なんというか|まあ)[、。]?)+$")
# 「けど」「けれど」は入れない（「地味ですけど。」のような言い切りの婉曲まで拾うため）
_TRAILING_CONJ = re.compile(
    r"(?:でも|ただ|それが|だから|あとは|それで|その|そういえば)[、。]?$")
# 発話の末尾が言いよどみ・接続語で止まっている（長さは問わない。「いいですね。私は、その。」）
_TRAILING_TAIL = re.compile(
    r"(?:えっと|えーっと|えーと|ええと|えー|あの|あのー|その|うーん|んー|なんだろう|まあ|"
    r"でも|けど|けれど|ただ|それが|だから|あとは|それで|そういえば)[、。]?$")
# 接続語のあと、短い語と助詞で止まった言いさし（「でも、家で。」「実は、私も。」）。
# ★2026-10-24 試走4回目: 「でも、家で。」の直後にイノウエが「家での様子は？」と即座に引き取り、
#   割り込みゲートにかからなかった。間の語を4文字までに絞ってある（「でも、勉強が結構大変で。」
#   のような言い切りは拾わない。005 の1万対話でユーザ → 主話者の並びに誤検出 0件）。
_TRAILING_FRAG = re.compile(
    r"^(?:でも|ただ|それで|だから|あとは|あと|けど|で|それが|実は|なんか|あの|その)、"
    r"[^、。？]{0,4}(?:[^のんかし、]で|に|を|が|は|と|も)。?$")
# ユーザが主話者に向けた問い（疑問詞を含み「？」か「を。」「は。」「って。」で止まる）。
# この直後にユーザ自身の沈黙を置くと「質問して黙り、自分で答える」形になる。
_SELF_QUESTION = re.compile(
    r"(?:どんな|どういう|どう(?:し|や|で|だ|思)|どこ|どれ|どちら|どっち|どの|何|なに|いつ|誰|だれ|なぜ)")


# Gemma は短いターンで "utterance" のキーを落とすことがある（{"speaker": "ハヤシ", "えっと。"}）。
# 2026-10-24 沈黙版の試走で JSON 崩れ 2/20 の原因。この形だけキーを補う。
_MISSING_UTT_KEY = re.compile(r'(\{\s*"speaker"\s*:\s*"[^"]*"\s*,\s*)("[^"]*"\s*[,}])')


# ★2026-10-24 試走8回目: 閉じ括弧を早く打ち、沈黙の秒数をオブジェクトの外に出す
# （{"speaker": "ハヤシ", "utterance": "えっと、あの。"}, "pause_before_sec": 4.2},）。JSON 崩れ 1/20。
_STRAY_PAUSE = re.compile(r'"\s*\}\s*,\s*("pause_before_sec"\s*:\s*[0-9.]+\s*\})')


# ★2026-11-12 rp_v1 試走13: キーと値の区切りが抜けた形（{"speaker": "X", "utteranceあの、…"}）を _MISSING_UTT_KEY が
#   値だけの形として直し、発話の頭に「utterance」が残った（274中2）。先にキーと値を分ける。
_GLUED_UTT_KEY = re.compile(r'(\{\s*"speaker"\s*:\s*"[^"]*"\s*,\s*)"utterance(?=[^"\s:])')


def _fix_missing_utterance_key(text: str) -> str:
    text = _GLUED_UTT_KEY.sub(r'\1"utterance": "', text)
    text = _STRAY_PAUSE.sub(r'", \1', text)
    return _MISSING_UTT_KEY.sub(lambda m: m.group(1) + '"utterance": ' + m.group(2), text)


def is_hesitation(u: str) -> bool:
    """言いよどみだけ、または接続語で言いさした短い発話（割り込みの判定に使う。狭め）。
    最後の文だけでも見る（「いいよ。あ、そういえば。」の「あ、そういえば。」）。"""
    u = str(u).strip()
    last = [x for x in re.split(r"(?<=[。？])", u) if x.strip()][-1:] or [u]
    return any(_HESITATION_ONLY.match(x)
               or (_core_len_utt(x) <= 8 and _TRAILING_CONJ.search(x))
               or _TRAILING_FRAG.match(x)
               for x in {u, last[0].strip()})


def is_trailing(u: str) -> bool:
    """言いよどみ・言いさしで止まった発話（型 C の直前として認める。広め）。"""
    u = str(u).strip()
    return is_hesitation(u) or bool(_TRAILING_TAIL.search(u))


def hesitation_interrupts(turns: list, main: "str | None", user: "str | None") -> list:
    """ユーザの言いよどみの直後に、主話者が沈黙を置かずに話している箇所（主話者のターン番号）。"""
    if main is None:
        return []
    return [i for i in range(1, len(turns))
            if isinstance(turns[i], dict) and isinstance(turns[i - 1], dict)
            and turns[i - 1].get("speaker") == user and turns[i].get("speaker") == main
            and turns[i].get(PAUSE_KEY) is None
            and is_hesitation(turns[i - 1].get("utterance", ""))]


def _last_sentence(u: str) -> str:
    xs = [x for x in re.split(r"(?<=[。？])", str(u).strip()) if x.strip()]
    return xs[-1].strip() if xs else str(u).strip()


def is_question_to_partner(u: str) -> bool:
    """最後の文が相手への問い（疑問詞を含み「？」か「を。」「は。」「って。」で止まる）。
    「かな」を含む独り言の問い（「どうしようかな？」）は除く。"""
    x = _last_sentence(u)
    return (bool(_SELF_QUESTION.search(x)) and "かな" not in x
            and (x.endswith("？") or bool(re.search(r"(?:を|は|って)。$", x))))


def self_answers(turns: list, main: "str | None", user: "str | None") -> list:
    """型 A の沈黙のうち、直前のユーザの発話が主話者への問いになっているもの（沈黙のターン番号）。

    ★2026-10-24 試走4回目: 48件中3件が「イノウエさんは、どういうバランスが理想だと思いますか？」
    → 6.0秒 → 同じユーザ「うーん？まあ、仕事に情熱を…」のように、ユーザが自分の問いに自分で
    答えていた（答えるべき主話者の台詞をユーザに付け替えて沈黙の主をユーザにしたもの）。
    沈黙を外しても台詞の食い違いは残るので、作り直す。"""
    if main is None:
        return []
    return [i for i, t in enumerate(turns)
            if isinstance(t, dict) and t.get(PAUSE_KEY) is not None
            and _pause_kind(turns, i, main, user) == "A"
            and is_question_to_partner(turns[i - 1].get("utterance", ""))]


def wants_prompt(metadata: dict, sil: SilenceParams) -> bool:
    """この対話で型 C（主話者が促す）を1回入れさせるか（対話 id で決定的に選ぶ）。"""
    if sil.prompt_ratio <= 0 or sil.max_prompt <= 0:
        return False
    key = str(metadata.get("id", "")).encode("utf-8")
    return (zlib.crc32(key) % 1000) < sil.prompt_ratio * 1000


def drop_closing_pauses(turns: list) -> int:
    """締めの挨拶（_closing_index）以降のターンの沈黙を外す（外した数）。別れ際に考え込む理由は
    無く、回数の枠も使ってしまう（2026-10-24 試走3回目: 「それでは、失礼いたします。」→ 3.0秒 →
    「はい。失礼いたします。」など3件）。外したあとの同じ話者の連続は呼び出し側で結合する。"""
    close_at = _closing_index(turns)
    if close_at is None:
        return 0
    n = 0
    for t in turns[close_at:]:
        if isinstance(t, dict) and t.pop(PAUSE_KEY, None) is not None:
            n += 1
    return n


def jitter_pauses(turns: list, main: "str | None", user: "str | None",
                  sil: SilenceParams, key: str) -> None:
    """確定した秒数を ±jitter_sec 揺らす（範囲は normalize_pauses と同じ下限・上限に収める）。

    LLM の推定は特定の値に固まる（2026-10-24 試走3回目: 47件中14件が3.5秒、2〜3秒・7〜8秒は
    ほぼ出ない）。プロンプトの目安では直らなかったので、推定値を中心に機械的に散らす。
    乱数は対話 id とターン番号で決めるので、同じ台本からは同じ値になる。
    """
    if sil.jitter_sec <= 0:
        return
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get(PAUSE_KEY) is None:
            continue
        rng = random.Random(zlib.crc32(f"{key}:{i}".encode("utf-8")))
        lo = (max(sil.sec_min, sil.prompt_sec_min)
              if main and _pause_kind(turns, i, main, user) == "C" else sil.sec_min)
        v = float(t[PAUSE_KEY]) + rng.uniform(-sil.jitter_sec, sil.jitter_sec)
        t[PAUSE_KEY] = round(min(max(v, lo), sil.sec_max), 1)


def silence_summary(turns: list, main: "str | None", user: "str | None") -> list:
    """確定した台本の沈黙一覧（dialogue.silences に書く。検証用）。"""
    return [{"turn": i, "kind": _pause_kind(turns, i, main, user) if main else None,
             "sec": t[PAUSE_KEY]}
            for i, t in enumerate(turns) if isinstance(t, dict) and PAUSE_KEY in t]


def _silence_norm(metadata: dict, sil: SilenceParams) -> "str | None":
    """build_dialogue_prompt の「必ず交互」の一文の差し替え。名前が引けなければ None。"""
    main, user = silence_roles(metadata)
    if main is None:
        return None
    # ★回数は上限の数で頼む（下限で頼むと LLM はちょうど下限を狙い、1つ外れるだけで下限を割る。
    #   2026-10-24 試走: 「2〜3回」で20対話すべてちょうど2回、沈黙不足のリトライ 20回）。
    #   合否は count_min 以上（silence ゲート）。
    if wants_prompt(metadata, sil):
        c = (f"そのうち1回は「{user}が途切れて黙ったあと、{main}が急かさない短い一言で促し、"
             f"{user}が続ける」型にしてください。秒数は{main}の促しのターンに書き"
             f"（{max(sil.sec_min, sil.prompt_sec_min):g}秒以上。続く{user}のターンには書かない）、"
             f"促しの言葉は決まり文句にせず、{user}が言いかけた話題に触れて続きを尋ねる短い問いかけに"
             f"してください（相づち・謝罪・話題の言い直しにしない）。")
    else:
        c = f"{main}が促す型は使わないでください。"
    return (f"2人の話者は原則として交互に発話してください。ただし{user}が長く黙る箇所を、"
            f"対話全体でちょうど{sil.count_max}回、前半・中盤・後半に散らして入れてください。沈黙の型は"
            f"「{user}が言いかけて途切れ、黙ったあと{user}自身が続ける（{user}のターンが2つ続く）」と"
            f"「{main}の問いのあと、{user}が考え込んでから答える」です。{c}"
            f"沈黙が明けたターンにだけ \"{PAUSE_KEY}\" に沈黙の秒数"
            f"（{sil.sec_min:g}〜{sil.sec_max:g}の数値。軽い迷いは短く、深く迷うほど長く）を書き、"
            f"{main}は沈黙の途中で話さないでください。{user}が「えっと。」のように言いよどんだ直後に、"
            f"{main}が沈黙を置かずに話し出すことはしないでください。"
            f"沈黙以外で同じ話者を続けて2ターン置かないでください。")


def load_self_name_max(config: dict) -> int:
    return int((config.get("dialogue") or {}).get(
        "self_name_max", DEFAULT_SELF_NAME_MAX))


def unknown_speakers(turns: list, metadata: dict) -> list[str]:
    """台本の話者名のうち、メタデータの speakers に無いもの（メタデータに名前が2人分ないときは調べない）。"""
    names = {str(s["name"]) for s in (metadata.get("speakers") or [])
             if isinstance(s, dict) and s.get("name")}
    if len(names) < 2:
        return []
    return sorted({str(t.get("speaker", "")) for t in turns if isinstance(t, dict)} - names)


def load_gates(config: dict) -> dict[str, bool]:
    """config の dialogue.gates を読む。書かれていないゲートは有効のまま。"""
    raw = (config.get("dialogue") or {}).get("gates") or {}
    unknown = sorted(set(raw) - set(GATE_NAMES))
    if unknown:
        raise ValueError(f"dialogue.gates に未知のゲート名: {'、'.join(unknown)}"
                         f"（使えるのは {'、'.join(GATE_NAMES)}）")
    return {name: bool(raw.get(name, True)) for name in GATE_NAMES}


# ══════════════════════════════════════════════════════════════════════════
# プロンプトのプロファイル（使う LLM ごとの本文の差し替え）
# ══════════════════════════════════════════════════════════════════════════
# 上のプロンプトは Qwen3-Swallow-32B に合わせて調整してきたもの。Qwen は話し言葉
# らしさを**出し足りない**ので、指示は全体に「増やせ」の向きに振ってある
# （フィラー70% / 読点13個per100字 / 「ほとんど全部の末尾に終助詞」）。
#
# Gemma 4 31B は逆に出しすぎる。同じ config・同じシナリオ・同じプロンプトで
# 100対話ずつ作って測った実測（COMPARE_gemma31b_vs_qwen32b.md）:
#
#   指標              人間 maai-f2f   Qwen32B   Gemma31B(後処理なし)
#   フィラー 個per100字      1.45      1.97      3.70
#   読点 個per100字          7.73      5.29     16.23
#   終助詞 ターン率         27.1%     15.4%     39.0%
#   ターン長 median            13        19        22
#
# そのため「増やせ」の指示をそのまま当てると増幅方向に働く。プロファイルは
# **本文の差し替え**を持ち、分量の数値は config の dialogue.enrich が持つ
# （両方を Gemma 用に振ったのが config_ft_v1_gemma.yaml）。
#
# ★差し替えの old / new はプロンプトパック側（prompts/stage2_*.yaml の
#   profile_patches）にある。本文と同じファイルに置くことで、シナリオごとに本文を
#   編集したときに差し替え元も一緒に直せる。差し替え元が消えたら例外で落ちる。
# ★指示の総量には上限がある。指示を3行足したら、無関係な「自分の名前」ゲートの
#   棄却率が 9.5%→20.6% に上がった実測がある。プロファイルは行を**足さずに
#   差し替える**方針で書くこと。
PROMPT_PROFILE_NAMES = ("qwen", "gemma")

# コード側で数を決める箇所（本文の差し替えでは書けないもの）。
PROFILE_NUMS = {
    # 極短ターン（6文字以下）の比率。Gemma はターン長 median が 22（人間 13）と
    # 長いので、極短を倍にして分布を人間寄りに寄せる。
    "qwen":  {"tiny_lo": 0.08, "tiny_hi": 0.12, "tiny_pct": "1"},
    "gemma": {"tiny_lo": 0.16, "tiny_hi": 0.24, "tiny_pct": "2"},
}


def load_prompt_profile(config: dict) -> str:
    """dialogue.prompt_profile を解決する。既定 "auto" ＝ llm.model の名前から判定。

    auto にしておけば、モデルを差し替えたときにプロファイルの指定を忘れて
    「Qwen 用のプロンプトを Gemma に当てる」事故が起きない。明示指定もできる。
    """
    raw = str((config.get("dialogue") or {}).get(
        "prompt_profile", "auto")).strip().lower()
    if raw in PROMPT_PROFILE_NAMES:
        return raw
    if raw != "auto":
        raise ValueError(
            f"dialogue.prompt_profile が未知の値: {raw}"
            f"（使えるのは auto / {' / '.join(PROMPT_PROFILE_NAMES)}）")
    model = str((config.get("llm") or {}).get("model")
                or config.get("model") or "").lower()
    for name in PROMPT_PROFILE_NAMES:
        if name in model:
            return name
    return "qwen"


def _patch_prompt(text: str, profile: str, key: str,
                  pack: "S2PromptPack | None" = None) -> str:
    """プロファイルの差し替えを当てる。差し替え元が無ければ落とす（黙って素通しにしない）。"""
    pack = pack or GENERAL_S2
    for old, new in pack.profile_patches.get(profile, {}).get(key, ()):
        if old not in text:
            raise AssertionError(
                f"プロンプトのプロファイル {profile}/{key} の差し替え元が見つかりません。"
                f"パック {pack.name} の本文を編集したら、同じファイルの "
                f"profile_patches.{profile}.{key} の old も直してください:\n"
                f"{old[:80]}...")
        text = text.replace(old, new, 1)
    return text


def build_system_prompt(sp: "S2Params | None",
                        gates: dict[str, bool] | None = None,
                        self_name_max: int = DEFAULT_SELF_NAME_MAX,
                        profile: str = "qwen",
                        pack: "S2PromptPack | None" = None) -> str:
    """sp が None ならパックの system_prompt そのまま、enriched なら拡張版。

    pack を省くと既定パック（prompts/stage2_general.yaml）。シナリオごとの本文は
    config の dialogue.prompt_file で選ぶ（run_dialogue_generation が渡す）。

    gates で切ったゲートに対応する禁止指示は、プロンプトからも外す（機械で弾かない
    ものを言葉だけで禁じても、モデルが従う保証がないうえに指示が増えるほど他の
    制約が守られなくなる）。
    """
    pack = pack or GENERAL_S2
    base = _patch_prompt(pack.system_prompt, profile, "system_prompt", pack)
    if sp is None:
        return _apply_gate_lines(base, gates, self_name_max, profile, pack)
    s = base
    s = s.replace(pack.schema_plain, pack.schema_enriched)
    s = s.replace(pack.filler_lines_loose, pack.filler_lines_enriched)
    laugh = ("" if sp.laugh_ratio <= 0 else pack.laugh_section.replace(
        "__LAUGH_PCT__", f"{sp.laugh_ratio * 100:g}"))
    section = (_patch_prompt(pack.enriched_section, profile, "enriched_section", pack)
               .replace("__LAUGH_SECTION__", laugh)
               .replace("__LEN_TABLE__", _len_table(sp))
               .replace("__FILLER_PCT__", str(int(round(sp.filler_ratio * 100))))
               .replace("__FILLER_PER_TURN__", str(sp.fillers_per_turn))
               .replace("__SPAN_LONG__", str(max(4, sp.comma_span - 2)))
               .replace("__MONO_HEAD_CAP__", str(sp.monotone_head_cap))
               .replace("__REACT_PCT__", str(int(round(sp.reaction_ratio * 100))))
               .replace("__SFP_PCT__", str(int(round(sp.sfp_ratio * 100))))
               .replace("__COMMA_PER_100__", f"{sp.comma_per_100:g}")
               .replace("__COMMA_SPAN__", str(sp.comma_span)))
    return _apply_gate_lines(s.replace(ENRICHED_ANCHOR, section + ENRICHED_ANCHOR),
                             gates, self_name_max, profile, pack)


def _apply_gate_lines(prompt: str, gates: dict[str, bool] | None,
                      self_name_max: int = DEFAULT_SELF_NAME_MAX,
                      profile: str = "qwen",
                      pack: "S2PromptPack | None" = None) -> str:
    pack = pack or GENERAL_S2
    gate_off = gates is not None and not gates.get("self_name", True)
    if gate_off or self_name_max > 0:
        assert pack.no_self_name_lines in prompt
        lines = _patch_prompt(pack.allow_self_name_lines, profile,
                              "allow_self_name_lines", pack)
        if self_name_max > 1:
            lines = lines.replace("名乗るのは一度だけにし",
                                  f"名乗るのは{self_name_max}回までにし")
        prompt = prompt.replace(pack.no_self_name_lines, lines)
    if gates is not None and not gates.get("long_utterance", True):
        assert pack.long_utt_quota_sys in prompt
        prompt = prompt.replace(pack.long_utt_quota_sys, pack.long_utt_free_sys)
    return prompt


# ── enriched モードの決定的クリーンアップ ────────────────────────────────
# プロンプトで禁じても出てしまう不自然形を、元テキスト無しで直せる範囲だけ機械的に直す。

# 文末の「な」を落としてよい語幹。動詞の言い切り（「思うな」「行くな」＝禁止の意味）を
# 狙うが、平仮名だけの語幹は名詞との区別が付かない（「さかな。」の語幹「さか」）ため、
# 漢字・カタカナを含む語幹のう段止めに限定する（「思う」「確認する」「食べる」）。
_VERB_NA_STEM = re.compile(r"[一-龥ァ-ヶー].*[うくぐすずつぬふぶむるゆ]$")
_STRIP_MARKS_RE = re.compile(r"[、。？]")


def _core_len_utt(text: str) -> int:
    return len(_STRIP_MARKS_RE.sub("", text or ""))


def fix_na_tail(text: str) -> str:
    """文末の終助詞「な」が動詞の言い切り・敬語に付いていたら落とす。

    「そう思うな。」は禁止の意味になり発話の意味が反転する。判定できない語
    （平仮名だけの語幹）は触らない（「さかな。」を壊さないため）。
    """
    m = _SFP_TAIL.search(text or "")
    if not m or m.group("sfp") != "な":
        return text
    stem = text[:m.start("sfp")]
    if _POLITE_STEM.search(stem) or _VERB_NA_STEM.search(stem):
        fixed = stem.rstrip("、") + m.group("end")
        return fixed if _core_len_utt(fixed) >= 2 else text
    return text  # 「〜だな」「〜たな」「〜いな」など、正しい用法・判定不能はそのまま


# 小書き仮名で書かれたフィラー。フィラー判定の正規表現（_FILLER_ANY 等）は大書きしか
# 見ないので、揃えておかないと個数の上限・下限が数え落とされる。TTS の読みも安定する。
_SMALL_KANA_FILLER = [(re.compile(r"まぁ"), "まあ"), (re.compile(r"あのぅ|あのぉ"), "あの"),
                      (re.compile(r"そのぅ|そのぉ"), "その"), (re.compile(r"えぇと"), "ええと")]


def clean_enriched_utterance(text: str, max_fillers: int = 1,
                             trailing_off: bool = False) -> str:
    """enriched モードで生成された1発話の不自然形を機械的に除去する。

    generate_voice_enrich.clean_enriched_text の「元テキストとの照合」を除いた版
    （生成物には照合すべき元テキストが無い）。除去するのはいずれも、音声にすると
    必ず不自然になる形だけ。

    max_fillers は実質25文字を超える発話に残してよいフィラー数（短い発話は常に1個まで）。
    trailing_off は長い沈黙（dialogue.silence）の直前の言いさし（「あの、実は、その、」）。
    末尾のフィラーと言いよどみの重なりは意図したものなので、末尾フィラーの除去と
    1発話のフィラー数の上限をかけない（2026-10-24。かけると「実は。」だけが残る）。
    """
    t = fix_particle_splits(text or "")
    # 頭の読点。単体では無害に見えるが、この後 merge_consecutive_same_speaker が
    # 前のターンと連結すると「給付金？、なるほど、」という句点直後の読点になる。
    t = t.lstrip("、")
    for pat, rep in _SMALL_KANA_FILLER:
        t = pat.sub(rep, t)
    t = re.sub(r"^こう、", "", t)          # 頭の「こう、」はフィラーの誤用
    t = _HEE_HEAD.sub("", t)               # 頭の「へえ」系（禁止語）
    t = clean_laugh(t)                     # 「笑」「w」の記号笑い、伸びすぎた笑い
    if not trailing_off:
        t = _TAIL_BREAK.sub("、", t)       # 「〜から、えっと、ね。」
        t = _TAIL_FILLER.sub("", t)        # 「〜しましょう、あの。」
    t = _GLUED_FILLER.sub("", t)           # 「59センチ、んでしょうか？」
    t = _TAIL_COMMA.sub("", t)             # 「〜です、。」
    if not trailing_off:
        t = limit_fillers_in_turn(t, max_fillers)   # 長い発話は max_fillers 個まで
    t = fix_na_tail(t)
    if t and not t.endswith(("。", "？")):
        t = t.rstrip("、") + "。"
    return t or text


# 呼びかけに付く敬称。「タカシ先生」「ミキさん」の形を1語として捉えるために使う。
_VOCATIVE_HON = (r"(?:さん|くん|君|ちゃん|先生|先輩|部長|課長|係長|社長|店長|教授|"
                 r"さま|様)?")


def fix_vocative_copula(turns: list) -> int:
    """相手の名前を「です」で結んだ呼びかけ（「こんにちは、タカシ先生です。」）を直す。

    呼びかけのつもりで名乗りの形になってしまう 32B の誤り（実測）。相手の名前である以上
    名乗りではあり得ないので、「です」を落とすだけで正しい呼びかけに戻せる。
    発話の先頭か読点の直後にある形だけを対象にして、「担当はタカシさんです。」のような
    まっとうな叙述には触らない。
    """
    names = {str(t.get("speaker", "")) for t in turns if isinstance(t, dict)}
    names.discard("")
    fixed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = before = str(t.get("utterance", ""))
        for other in names - {str(t.get("speaker", ""))}:
            u = re.sub(rf"(^|、)({re.escape(other)}{_VOCATIVE_HON})です(?=[。！？])",
                       r"\1\2", u)
        if u != before:
            t["utterance"] = u
            fixed += 1
    return fixed


# 名乗りに付いた終助詞を落とすための語。自己紹介は相手がまだ知らない情報の提示
# なので、同意を求める「ね」「よね」も、教える「よ」も付かない。敬語の「です」は
# 全部落とし、タメ口の「だ」は「〜だよ」が自然に成り立つので「ね」「よね」だけ落とす。
_SELF_INTRO_DESU_SFP = r"(?:よね|かな|ね|よ|な|わ)+"
_SELF_INTRO_DA_SFP = r"(?:よね|ね)+"


def fix_self_intro_sfp(turns: list) -> int:
    """名乗りに付いた終助詞を落とす（「タカシですよね。」→「タカシです。」）。

    inject_sfp / diversify_sfp_runs は終助詞の無い平叙ターンに機械的に足すので、
    名乗りのターンにも付けてしまう（実測: 「はじめまして、タカシですよね。」）。
    付加のあとに呼ぶこと。対象は「自分の名前＋です／だ」の直後だけなので、
    相手への呼びかけ（fix_vocative_sfp の担当）とは衝突しない。
    """
    fixed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        name = str(t.get("speaker", ""))
        if not name:
            continue
        u = before = str(t.get("utterance", ""))
        n = re.escape(name)
        u = re.sub(rf"(^|、)({n}です){_SELF_INTRO_DESU_SFP}(?=[。！？]|$)", r"\1\2", u)
        u = re.sub(rf"(^|、)({n}だ){_SELF_INTRO_DA_SFP}(?=[。！？]|$)", r"\1\2", u)
        if u != before:
            t["utterance"] = u
            fixed += 1
    return fixed


_MONOTONE_HEAD_UTT = re.compile(r"^(?:まあ、|うーん、)")


def cap_monotone_heads(turns: list, cap: int = 3) -> int:
    """頭の「まあ、」「うーん、」を対話全体で cap 回までに制限する。"""
    seen = stripped = 0
    for t in turns:
        if not _MONOTONE_HEAD_UTT.match(t.get("utterance", "")):
            continue
        seen += 1
        if seen > cap:
            t["utterance"] = _MONOTONE_HEAD_UTT.sub("", t["utterance"])
            stripped += 1
    return stripped


# ── 冒頭の笑い ────────────────────────────────────────────────────────
# 人間コーパス maai-f2f-jp-2025 では、発話ターンの 2.4% が笑い声で始まる
# （アハハ/アハ 系 40%、フフ/ウフフ 系 30%、ハハ 系 15%）。同じ形を許可する。
# ★2026-09-26 句点・感嘆符・促音で切る「ふふ。」「あはは！」「ふふっ、」も拾う（接客の
#   _CS_LAUGH_HEAD と同じ原因。読点しか見ていなかったので、Gemma の初対面・汎用コーパスでは
#   冒頭の笑いの 7〜8割（各1万対話で 5,666〜9,415 発話）が上限・置き場所の検査を素通りしていた）。
_LAUGH_HEAD = re.compile(r"^(?:あは+は*|は+は+|ふふ+|うふふ+|えへへ+|へへ+)っ?[、。！]")
# 笑いだけの発話（後ろに本文が続かない形）。相槌は後工程が入れるので台本には残さない。
_LAUGH_ONLY = re.compile(r"^(?:あは+は*|は+は+|ふふ+|うふふ+|えへへ+|へへ+)[、。？]*$")
# 地の文の笑い記号（「そうだね笑」「うけるw」）。読み上げると「わら」等になるので落とす。
# 「w」はラテン文字の語尾（show・now）を削らないよう、直前がラテン文字でないときだけ。
_LAUGH_MARK = re.compile(
    r"(?:(?<=[、。？])|^)笑(?=[、。？]|$)|(?<![A-Za-zＡ-Ｚａ-ｚ])[wｗ]{1,}(?=[、。？]|$)")
# 事務的に受けるだけの発話の頭。ここに笑いが付くと笑う理由が無く不自然
# （実測:「あはは、了解です。明日までに配布しますね。」）。
_ACK_HEAD = re.compile(
    r"^(?:了解|わかりました|分かりました|わかった|分かった|承知|承りました|"
    r"かしこまりました|ありがとうございます|ありがとう|はい、)")
# ★笑ってよい理由が直前のターンにあるか（ユーザー指摘 2026-07-31:
# 「あはは、検索ワードが鍵になるのは本当です。」「ふふ、頑張ってください。」＝
# 淡々とした説明・励ましの場面で笑っている）。
#
# 「面白い」は機械では判定できないので、笑いの引き金になる形だけを白名簿にする。
# 冗談・失敗談・大げさ・意外・自虐・照れ、そして相手が既に笑いを言語化した場合。
# 取りこぼしは出るが、無い笑いは聞こえないだけで、理由のない笑いは必ず気になる。
# ＝再現率より適合率を採る（プロンプト側でも場面を指定しているので二重の網）。
# ★「笑っちゃいます」「笑えます」のような笑いの言語化は引き金に含めない。連続して
#   笑い合うのは別途禁止しているので、本物の笑いを受ける経路にはならず、理由のない
#   笑いが次の笑いを正当化する連鎖にしかならなかった（実測:「あはは、〜ちょっと
#   笑っちゃいますね。」→「ふふ、頑張ってください。」）。
# ★「！」も含めない。勢いの印であって、面白さの印ではない。
_LAUGH_TRIGGER = re.compile(
    r"冗談|大げさ|おおげさ|失敗|ミス|しくじ|やらかし|"  # 「オーバー」は
    # 「予算オーバー」に当たる誤検知だったので入れない（大げさ で足りる）

    r"すぎ(?:て|る|ちゃ|まし|です|だ)|"
    r"ちゃって|ちゃいまし|ちゃった|ちゃうん|てしまって|てしまいまし|"
    r"まさか|とは思わ|思いもし|びっくり|"
    r"苦手|下手|情けな|恥ずかし|お恥ずかし|ダメで|だめで|"
    r"全然でき|できなくて|忘れちゃ|寝坊|やっちゃ")
# 笑いを受け止める側の合図（「私も昔〜だった」＝自分の失敗談を重ねる共感）。
# 直前ターンに引き金が無くても、この形なら笑いが浮かない。
# ★過去形を伴うことを条件にする。「僕も今日中に作るから、安心してね。」のような
#   これからの話に「私も」だけで笑いが残ってしまった（実測）。
_LAUGH_EMPATHY = re.compile(
    r"^(?:わかります|わかる|分かります|分かる|そうなんです|そうなんだ|"
    r"私も|僕も|俺も|うちも|同じ)")
_PAST_IN_UTT = re.compile(r"(?:[っいきしちにみりえけせてねべ]た|ました|でした)")
# 長すぎる笑い（「あはははははは」）は TTS が不安定になるので3音節までに詰める。
_LAUGH_LONG = [(re.compile(r"^あは(?:は){2,}"), "あはは"),
               (re.compile(r"^は(?:は){2,}"), "はは"),
               (re.compile(r"^ふ(?:ふ){2,}"), "ふふふ"),
               (re.compile(r"^うふ(?:ふ){2,}"), "うふふ")]


def clean_laugh(text: str) -> str:
    """笑いの表記を音声にできる形へそろえる（記号の笑いは落とす）。"""
    t = _LAUGH_MARK.sub("", text or "")
    for pat, rep in _LAUGH_LONG:
        t = pat.sub(rep, t)
    return t.lstrip("、") or text


def cap_laugh_heads(turns: list, cap_ratio: float) -> int:
    """冒頭の笑いを対話全体の上限まで刈り、置けない位置のものを落とす。

    落とすのは (1) 1ターン目（まだ何も聞いていないので笑えない）、
    (2) 直前のターンも笑いで始まる形（笑い合いが続くと芝居がかる。プロンプトで
    禁じても実測で3組出た）、(3) 笑ってから事務的に受けるだけの発話
    （「あはは、了解です。」＝笑う理由がない）、(4) ★直前のターンに笑う理由が
    見当たらない形（「あはは、検索ワードが鍵になるのは本当です。」＝淡々とした
    説明への笑い。ユーザー指摘）、(5) 上限を超えたぶん（後ろから）。
    笑いだけのターンは剥がすと空になるので触らない。
    """
    idx = [i for i, t in enumerate(turns)
           if _LAUGH_HEAD.match(str(t.get("utterance", "")))]
    cap = max(1, int(round(cap_ratio * len(turns)))) if cap_ratio > 0 else 0
    removed = kept = 0
    prev = -2
    for i in idx:
        u = str(turns[i]["utterance"])
        if _LAUGH_ONLY.match(u):
            continue
        rest = _LAUGH_HEAD.sub("", u)
        # 笑いの対象を直前4ターンまで遡って探す。打ち解けた話は数ターン続くので
        # 直前1ターンだけ見ると取りこぼす（実測: 窓2で10件 → 窓4で13件）。
        # 遡りすぎても上限（cap_ratio）で1対話1件程度に収まるので害はない。
        before = " ".join(str(turns[j].get("utterance", ""))
                          for j in range(max(0, i - 4), i))
        no_reason = not (_LAUGH_TRIGGER.search(before)
                         or (_LAUGH_EMPATHY.match(rest)
                             and _PAST_IN_UTT.search(rest)))
        if (i == 0 or i - 1 == prev or kept >= cap
                or _ACK_HEAD.match(rest) or no_reason):
            turns[i]["utterance"] = rest
            removed += 1
        else:
            kept += 1
            prev = i
    return removed


def thin_fillers(turns: list, cap_ratio: float) -> int:
    """フィラー付きターンの比率が cap_ratio を超えたら、短いターンから剥がす。"""
    cap = int(cap_ratio * len(turns))

    def filled():
        return [i for i, t in enumerate(turns)
                if _FILLER_ANY.search(t.get("utterance", ""))]

    removed = 0
    for i in sorted(filled(), key=lambda i: _core_len_utt(turns[i]["utterance"])):
        if len(filled()) <= cap:
            break
        t = turns[i]["utterance"]
        t2 = _HEAD_FILLER.sub("", t)
        if t2 == t:
            t2 = _MID_FILLER.sub("、", t, count=1)
        if t2 != t and t2:
            turns[i]["utterance"] = t2
            removed += 1
    return removed


# ── 文中フィラーの決定的な差し込み ──────────────────────────────────────
# 32B はフィラーを発話の冒頭にしか置かない（プロンプトで位置と例を示しても、実測で
# 文中フィラーは 0%）。指示では動かないので、生成後に機械的に差し込む。
#
# Stage3 の LLM 差し込みと決定的に違うのは、位置を自分で選ばない点。「モデル自身が
# 読点を打った位置」＝モデルが節の切れ目と判断した位置だけを候補にし、その読点の直後に
# 「フィラー、」を挟む。「予算が限られてて、」→「予算が限られてて、まあ、」。
# 語の途中や助詞の直前に落ちることが構造的に起きない。

# 差し込む語。人間コーパスの文中フィラーの内訳（なんか23% / あの・あのー32% /
# ま・まあ29% / その17% / こう9%、「えっと」は1%未満）に寄せて、頻度の高い語を
# 重複させて重みにする。以前は8語を一様に回していたため、人間がほとんど使わない
# 「えっと」「ええと」が合成側のフィラーの3割を占めていた。
_FILLER_POOL = ("なんか", "あの", "まあ", "その", "こう", "なんか", "あの", "えっと",
                "まあ", "なんていうか", "その", "なんか", "あの", "うーん", "ええと")
# タメ口の発話にだけ足す語（敬語に混ぜると口調が崩れる）。文中だけで使う。
_FILLER_POOL_CASUAL = ("なんだろう", "やっぱ")
# 冒頭に置く語。人間の冒頭フィラーは「あ」「あー」「え」が半分を占めるが、これらは
# 気づき・驚きの意味を持つ反応語なので機械的には差し込まない（意味と衝突する）。
# 意味を持たない語だけを、人間の内訳（なんか・あの・まあ が中心）の順に並べる。
_HEAD_FILLER_POOL = ("なんか", "あの", "まあ", "えっと", "あの", "なんか", "うーん",
                     "その", "ええと")
# 敬語の発話（タメ口専用の語を混ぜない判定に使う）
_POLITE_UTT = re.compile(r"(?:です|ます|ません|でした|でしょう|ください|ましょう)")
# 読点の直前がここで終わっていれば、その読点は節の切れ目（＝間を置いてよい位置）。
_CLAUSE_END = re.compile(
    r"(?:って|てて|でて|て|で|けど|けれど|のに|ので|から|ため|が|し|たら|なら|れば|ば|"
    r"は|も|とか|には|では|とは|かも|など|ながら|つつ|ものの|と|たり|ても)$")
# 読点の直後がこれで始まるなら、その読点は節の切れ目ではなく句の途中に打たれている
# （「素材、を活用」）。ここに差し込むと助詞の直前へ割り込む形になる。
_LEADING_PARTICLE = re.compile(r"^(?:を|が|に|へ|と|も|の|や|か|ね|よ|で|は)")
# 挨拶・礼・詫びの直後に差し込むと、前置きが二重になって間延びする
# （実測:「こんにちは、こう、タカシさん。」＝「こんにちは」の「は」を節末と見た差し込み）。
_GREETING_CHUNK = re.compile(
    r"(?:こんにちは|こんばんは|おはようございます|おはよう|はじめまして|"
    r"お疲れ様です|お疲れさまです|お疲れ様|お疲れさま|ありがとうございます|ありがとう|"
    r"すみません|失礼します|失礼しました|よろしくお願いします|ごめんなさい|ごめん)$")
_FILLER_AT_HEAD = _HEAD_FILLER   # 「フィラー、」で始まる（差し込み先としては重複）
# 話の切り替え語の直後。「それでは」「では」の「は」を主題の助詞と読んで節末と誤判定し、
# 差し込みが起きていた（実測:「それでは、なんか、失礼します。」）。ここは前置きの位置で、
# 節の切れ目ではない。読点区間の末尾がちょうど切り替え語のときだけ弾く
# （「会社では、」のような主題の「は」は差し込み先として残す）。
_DISCOURSE_CHUNK = re.compile(
    r"^(?:それでは|では|じゃあ|じゃ|さて|ところで|とにかく|ともかく|まずは|最後に)$")
# 差し込み先の直後が挨拶・礼・締めの定型句や切り替え語なら、そこには入れない
# （定型句は一息で言うので割ると不自然。切り替え語の前に入れると前置きが二重になる）。
# 冒頭差し込みで使う _DISCOURSE_HEAD をそのまま流用する（同じ理由の同じ語彙）。


def _mid_filler_slots(text: str) -> list[int]:
    """フィラーを差し込んでよい読点の直後位置（文字インデックス）を返す。"""
    slots = []
    prev = 0
    for m in re.finditer("、", text):
        tail, head = text[m.end():], text[prev:m.start()]
        prev = m.end()
        if _core_len_utt(tail) < 5:            # 文末に近すぎると「間」にならない
            continue
        # 直前の読点からの区間が短すぎるものは節ではなく間投詞（「よし、」「そう、」）。
        # ここへ差し込むと前置きが二重になる（「よし、なんか、じゃあ〜」）。
        if _core_len_utt(head) < 4 or _GREETING_CHUNK.search(head):
            continue
        # 直前の句点以降が切り替え語だけなら、その読点は前置きの区切り（節末ではない）
        if _DISCOURSE_CHUNK.match(re.split(r"[。？]", head)[-1]):
            continue
        if not _CLAUSE_END.search(text[:m.start()]):
            continue
        if (_LEADING_PARTICLE.match(tail) or _FILLER_AT_HEAD.match(tail)
                or _DISCOURSE_HEAD.match(tail)):
            continue
        slots.append(m.end())
    return slots


# ── 読点なしで長く続く区間の分割 ────────────────────────────────────────
# 「すらすら読みすぎて不気味」の実体は、読点なしで長く続く区間（実測: 実質16字以上の
# 区間が全ターンの49%にある）。そこに「間」が無いので合成音声が一息で言い切る。
#
# 位置を推測して読点を打つと語の途中を割る（実測の失敗例:「〜のです」→「ので、す」）ので、
# 語中一致が起こり得ない形だけを白名簿にし、さらに後読み・先読みで用言か名詞かを見分ける。
# 見つからなければ何もしない（＝安全側に倒す）。実測では 5% 前後のターンにしか効かないが、
# 効くのは「いちばん読点なしで続いている」ターンなので、聴感への寄与はこの数字より大きい。
_RUN_BREAKS = [
    (re.compile(r"(?:けれども|けれど|けど)(?![もね])"), "けど"),
    (re.compile(r"(?:ですが|ますが|でしたが|ましたが|だが)"), "が"),
    (re.compile(r"ので(?![すしょ])"), "ので"),
    (re.compile(r"のに(?![おこ])"), "のに"),
    # 因果・時間の「から」。直前が用言（＝節の終わり）のときだけ。名詞に付く起点の
    # 「から」（「入口から」「10時から」「左から右へ」）を割ると意味の対を壊す。
    (re.compile(r"(?<![いま])(?<=[くぐすずつぬぶむるういたてでんない])から(?![だなぬからね])"), "から"),
    (re.compile(r"(?:ながら|つつ)(?![くけ])"), "ながら"),
    # 「1個あたり」「1kgあたり」は助数詞なので「あ」を除外する
    (re.compile(r"(?<![あ])(?<=[っいきしちにみりえけせてめ])たり(?![まし])"), "たり"),
    (re.compile(r"(?<=[っいきしちにみりえおかさなはまやわぐずづぶ])たら"), "たら"),
    (re.compile(r"(?<=[れけえせてめねべげぜ])ば(?![あいうえおかきくけこ])"), "ば"),
    (re.compile(r"(?<![らりるれろ])なら(?![なぬびべばぼわうっ])"), "なら"),
    (re.compile(r"(?<=[んいうくすつるたまえけせてねべ])し(?![まいてたなよかこちのくっ])"), "し"),
    (re.compile(r"とか(?![らりるれ])"), "とか"),
    # 連用中止の「て/で」。次が漢字・カタカナ・数字なら補助動詞（〜ている/〜ておく等）の
    # 続きではないので、節の切れ目とみなせる。
    (re.compile(r"(?:くて|して|んで|って|いて|えて|けて|せて|ねて|めて|べて|げて|ぜて|"
                r"れて|ちゃって|でて|てて)(?=[一-龥ァ-ヴ0-9０-９A-Za-zＡ-Ｚａ-ｚ])"), "て"),
]
# 割った後ろ側がこれで始まるなら、そこは節の頭ではない（助詞・活用の途中）。
_RUN_LEAD_NG = re.compile(r"^(?:を|が|に|へ|と|も|の|や|か|ね|よ|で|は|し|て|た|な|ら|り|"
                          r"る|れ|ば|ん|ど|ま)")


def _run_split_pos(run: str, min_side: int) -> int | None:
    """区間 run を割ってよい位置（文字インデックス）。無ければ None。"""
    best = None
    for pri, (rx, _tag) in enumerate(_RUN_BREAKS):
        for m in rx.finditer(run):
            p = m.end()
            if (_core_len_utt(run[:p]) < min_side
                    or _core_len_utt(run[p:]) < min_side
                    or _RUN_LEAD_NG.match(run[p:])):
                continue
            # 区間の真ん中に近いものを優先し、同じなら白名簿の上から優先する
            key = (abs(_core_len_utt(run[:p]) - _core_len_utt(run) / 2), pri)
            if best is None or key < best[0]:
                best = (key, p)
    return None if best is None else best[1]


def _break_segment(seg: str, run_max: int, min_side: int, depth: int = 3) -> tuple:
    """読点・句点で区切られた1区間を、長すぎるあいだ再帰的に割る。"""
    body = seg.rstrip("、。？")
    tail = seg[len(body):]
    if depth <= 0 or _core_len_utt(body) <= run_max:
        return seg, 0
    pos = _run_split_pos(body, min_side)
    if pos is None:
        return seg, 0
    left, n1 = _break_segment(body[:pos] + "、", run_max, min_side, depth - 1)
    right, n2 = _break_segment(body[pos:] + tail, run_max, min_side, depth - 1)
    return left + right, 1 + n1 + n2


def break_long_runs(turns: list, run_max: int, min_side: int = 6) -> int:
    """読点なしで run_max 実質字を超える区間を、節の切れ目で割る（打った読点の数を返す）。"""
    if run_max <= 0:
        return 0
    added = 0
    for t in turns:
        parts = re.split(r"(?<=[、。？])", str(t.get("utterance", "")))
        out = ""
        for seg in parts:
            s, n = _break_segment(seg, run_max, min_side)
            out += s
            added += n
        t["utterance"] = out
    return added


def _filler_rng(turns: list) -> random.Random:
    """対話ごとに固定の乱数列。同じ台本なら毎回同じ語が入る（再現性のため）。

    Python の str の hash は実行ごとに変わるので crc32 で種を作る。
    """
    key = "".join(str(t.get("utterance", "")) for t in turns[:3])
    return random.Random(zlib.crc32(key.encode("utf-8")))


def _pick_filler(pool: tuple, utterance: str, rng: random.Random,
                 avoid: str = "", casual_ok: bool = False) -> str:
    """発話の口調に合う語を1つ選ぶ（直前に使った語は避ける）。

    casual_ok はタメ口専用語（「なんだろう」「やっぱ」）を混ぜてよいか。冒頭では
    使わない（発話全体に掛かる位置なので、意味を持つ語だと前置きにならない）。
    """
    words = list(pool)
    if casual_ok and not _POLITE_UTT.search(utterance):
        words += list(_FILLER_POOL_CASUAL)
    choices = [w for w in words if w != avoid] or words
    return rng.choice(choices)


# 冒頭に既に何かが置かれている形。ここへ更にフィラーを足すと二重の前置きになる。
# 受領・了解の語（了解、わかった…）も、前に前置きを足すと間延びするので対象にする。
_SHORT_INTERJ_HEAD = re.compile(
    r"^(?:あ|あっ|え|えっ|お|おお|うわ|うん|ううん|はい|いや|ねえ|ね|ええ|よし|"
    r"了解|わかった|分かった|わかりました|分かりました|承知|オッケー|オーケー)[、。？]")


# 挨拶・礼・詫び・話の切り替え語で始まる発話。前にフィラーを足すと不自然になる
# （実測:「なんか、ありがとうございます、よろしくお願いします。」）。
_DISCOURSE_HEAD = re.compile(
    r"^(?:こんにちは|こんばんは|おはよう|はじめまして|お疲れ様|お疲れさま|"
    r"ありがとう|すみません|失礼|よろしく|ごめん|こちらこそ|本日は|"
    r"それでは|では|じゃあ|じゃ|さて|ところで)")


def _head_taken(u: str) -> bool:
    return bool(_HEAD_FILLER.match(u) or _HEAD_REACT.match(u)
                or _LAUGH_HEAD.match(u) or _SHORT_INTERJ_HEAD.match(u)
                or _DISCOURSE_HEAD.match(u))


def _inject_head_fillers(turns: list, sp: "S2Params", rng: random.Random) -> int:
    """発話の冒頭にフィラーを差し込む（sp.head_filler_ratio に届くまで）。

    人間コーパスでは冒頭フィラーのほうが文中より多い（19.6% 対 13.5%）のに対し、
    合成側は文中の差し込みしか無く冒頭 7.4% だった。文中の差し込み先は「モデルが
    打った読点」に限られて 24% で頭打ちになるので、密度を上げるぶんは冒頭で補う。

    冒頭は意味の切れ目としては常に安全だが、既にリアクション・笑い・言いかけ・
    フィラーが置かれているターンには足さない（前置きが二重になる）。
    """
    n = len(turns)
    target = int(round(sp.head_filler_ratio * n))
    have = [i for i, t in enumerate(turns)
            if _HEAD_FILLER.match(str(t.get("utterance", "")))]
    need = target - len(have)
    if need <= 0:
        return 0
    # 「まあ、」「うーん、」の頭は単調に聞こえるので、対話全体の上限を超えない
    mono_left = max(0, sp.monotone_head_cap - sum(
        1 for t in turns if _MONOTONE_HEAD_UTT.match(str(t.get("utterance", "")))))
    # 短い応答（「二つ、はい。」）の頭に前置きを足すと間延びするので、ある程度の
    # 長さがあるターンだけを対象にする。
    cands = [i for i in range(1, n)          # 1ターン目（口火）には置かない
             if not _head_taken(str(turns[i].get("utterance", "")))
             and _core_len_utt(str(turns[i].get("utterance", ""))) >= 12]
    if not cands:
        return 0
    # 長い発話から先に埋める（長いほど言いよどませたい）。前後2ターンの間隔規則が
    # 効くので、長いターンが固まっていても頭フィラーは並ばない。
    cands.sort(key=lambda i: -_core_len_utt(str(turns[i].get("utterance", ""))))
    added, last, used = 0, "", set(have)
    for i in cands:
        if added >= need:
            break
        # 前後2ターン以内に頭フィラーがあるものは飛ばす。話者は交互なので、
        # ±1 だけ見ると同じ話者（i-2）の発話の頭が軒並みフィラーで揃ってしまう。
        if used & {i - 2, i - 1, i + 1, i + 2}:
            continue
        u = str(turns[i]["utterance"])
        w = _pick_filler(_HEAD_FILLER_POOL, u, rng, avoid=last)
        if w in ("まあ", "うーん"):
            if mono_left <= 0:
                w = _pick_filler(("なんか", "あの", "その", "えっと"), u, rng, avoid=last)
            else:
                mono_left -= 1
        turns[i]["utterance"] = w + "、" + u
        used.add(i)
        last = w
        added += 1
    return added


def inject_fillers(turns: list, sp: "S2Params") -> int:
    """フィラー含有ターンが目安割合に届くまで、冒頭と文中に差し込む。

    先に冒頭（sp.head_filler_ratio ぶん）、続いて文中の節境界へ、全体が
    sp.filler_ratio に届くまで入れる。1ターンあたりの個数は sp.fillers_per_turn
    （実質25字以下のターンは1個）まで。語は口調（敬語かタメ口か）で選び分ける。
    """
    n = len(turns)
    if not n:
        return 0
    rng = _filler_rng(turns)
    added = _inject_head_fillers(turns, sp, rng) if sp.head_filler_ratio > 0 else 0

    def budget(u: str) -> int:
        return filler_quota(_core_len_utt(u), sp)

    def one_mid(i: int, last: str) -> str:
        """ターン i の節境界に1個差し込む。入れられなければ空文字を返す。"""
        u = str(turns[i].get("utterance", ""))
        if len(_FILLER_ANY.findall(u)) >= budget(u):
            return ""
        slots = _mid_filler_slots(u)
        if not slots:
            return ""
        pos = slots[len(slots) // 2]          # 発話の真ん中寄りの切れ目を選ぶ
        w = _pick_filler(_FILLER_POOL, u, rng, avoid=last, casual_ok=True)
        turns[i]["utterance"] = u[:pos] + w + "、" + u[pos:]
        return w

    filled = [i for i, t in enumerate(turns)
              if _FILLER_ANY.search(str(t.get("utterance", "")))]
    need = int(round(sp.filler_ratio * n)) - len(filled)
    mid_added, last = 0, ""

    # 第1段: フィラーの無いターンを埋めて、全体の含有率を目安まで持ち上げる。
    if need > 0:
        cands = [i for i in range(n) if i not in filled
                 and _mid_filler_slots(str(turns[i].get("utterance", "")))]
        # 連続ターンに固まらないよう、候補全体へ均等にばらす
        stride = max(1, len(cands) // max(need, 1)) if cands else 1
        for i in cands[::stride]:
            if mid_added >= need:
                break
            w = one_mid(i, last)
            if w:
                last, mid_added = w, mid_added + 1

    # 第2段: 長い発話を、長さで決まる個数（filler_quota）まで満たす。すらすら言い切る
    # 長台詞を作らないための段で、短い発話には効かない（quota が 1 で止まるため）。
    if sp.filler_len_step > 0:
        longest = sorted(range(n),
                         key=lambda i: -_core_len_utt(str(turns[i].get("utterance", ""))))
        for i in longest:
            u = str(turns[i].get("utterance", ""))
            if budget(u) <= 1:
                break                          # ここから先は短いターンだけ
            for _ in range(sp.fillers_per_turn):
                w = one_mid(i, last)
                if not w:
                    break
                last, mid_added = w, mid_added + 1
    return added + mid_added


# 間引いてよい文末終助詞。落としても文が成立し、意味も変わらない語だけ
# （「〜ですよね。」→「〜です。」、「〜だな。」→「〜だ。」）。「かな」「っけ」「よ」は
# 落とすと問いの力・語気が変わるので触らない。
_THINNABLE_SFP = {"よね", "な"}


def strip_tail_sfp(text: str) -> str:
    """文末の「よね」「な」を落とす。落とすと語が壊れる形はそのまま返す。"""
    m = _SFP_TAIL.search(text or "")
    if not m or m.group("sfp") not in _THINNABLE_SFP:
        return text
    stem = text[:m.start("sfp")]
    # 「な」は正しい用法（「〜だな」「〜たな」「〜いな」）のときだけ落とせる。
    # 語の一部（「みんな。」の「な」）を剥がすと単語が壊れる。
    if m.group("sfp") == "な" and not _NA_STEM_OK.search(stem):
        return text
    stripped = stem.rstrip("、") + m.group("end")
    return stripped if _core_len_utt(stripped) >= 2 else text


# ── 文末終助詞の機械付加 ─────────────────────────────────────────────
# 32B は終助詞の指示には反応する（要求35%で実測9〜55%）が、ばらつきが大きく
# 「ほとんどの発話に付ける」水準までは安定して届かない。読点・フィラーと同じく、
# 生成後に決定的に足す。足す位置は文末1箇所だけなので、語中を壊す余地が無い。

# 終助詞を足さない発話。定型のあいさつ・礼・詫びは、終助詞を付けると型が崩れる
# （「ありがとうございますね。」「よろしくお願いしますよ。」）。
_SFP_NO_ADD = re.compile(
    r"(?:よろしくお願いします|よろしくお願いいたします|お願いします|お願いいたします"
    r"|失礼します|失礼しました|失礼いたします|ありがとうございます|ありがとうございました"
    r"|ありがとう|すみません|すいません|申し訳ありません|申し訳ございません"
    r"|ごめんなさい|ごめん|おはようございます|おはよう|こんにちは|こんばんは"
    r"|さようなら|お疲れ様です|お疲れさまです|お疲れ様でした|お疲れさまでした"
    r"|いただきます|ごちそうさま|はじめまして|こちらこそ|承知しました|かしこまりました"
    # 受諾の定型。「了解ですね」「了解ですよね」は型が崩れる（実測「了解ですよね」）
    r"|了解です|了解|承知です|承知|オッケーです|オーケーです)$")

# ここへ足してよいのは「ね」だけ。
#   前半: 命令・依頼・勧誘・テ形（「教えてくださいね。」は自然だが
#         「教えてくださいよ。」は押しつけがましく「〜よね。」は文にならない）。
#   後半: 助詞で終わる言いさし（「まずはそこから。」「時間もないし。」）。
#         「〜からね。」「〜しね。」は自然だが「〜からよ。」は語法として壊れる。
#   末尾: 疑問の「か」（「〜ですかね」は自然だが「〜ですかよ」は壊れる）。
#   末尾: 推量の「かも」「でしょう」（「かもですね」は自然だが「かもですよね」はもたつく。
#         「でしょうよ」は押しつけがましく「でしょうよね」は実測で壊れていた）。
#   ★「よう」は意志形（「乗り越えよう」「試してみよう」）。「〜ようだな」のように
#     「だ」を挟むと壊れるので、_SFP_NEEDS_DA ではなくここで「ね」だけ許す。
_SFP_NE_ONLY = re.compile(
    r"(?:ください|下さい|なさい|ましょう|しよう|ろう|よう|て|で|ろ"
    r"|から|まで|より|ほど|くらい|ぐらい|など|とか|し|に|へ|と|も|や|を|は"
    r"|けど|けれど|けども|が|ので|のに|たり|かも|かもです|でしょう|でしょ|か)$")

# 条件形で宙ぶらりんに終わっている発話（「具体的にどうすれば。」「どうしたら。」
# 「例えば。」）。ここに終助詞を足すと「例えばよね。」のように語法が壊れるので触らない。
#   「れ」…命令形と代名詞が同じ形（「頑張れ」「それ」）。「頑張れね」は語法が壊れ、
#         「それね」は足さなくても成り立つので、どちらも触らないのが安全。
#   小書き仮名…擬態語・ナ形容詞的な語（「ぐちゃぐちゃ」）。「だ」が要る形だが、
#         語彙を列挙できないので足さない（実測「頭がぐちゃぐちゃよね」）。
_SFP_NO_ADD_TAIL = re.compile(r"(?:ば|たら|なら|れ|[ゃゅょ])$")

# 自分の能力・提案の表明。「よ」だけ許す（「調整できますね」「ご提案できますね」は
# 相手のことを言っているように聞こえ、「できますよね」は同意要求になる）。
# ★下の _POLITE_VERB_TAIL（敬語の動詞述語に「よ」を付けない）の例外。可能形は
#   「相手のためになる情報」なので「よ」が自然（「ご提案できますよ」）。
_SFP_YO_ONLY = re.compile(r"(?:できます|できる|できました|できた|可能です)$")

# ── 「よ」の用法 ─────────────────────────────────────────────────────
# 「よ」は相手が知らないこと・相手のためになることを教える語。敬語の動詞述語に付けると
# 「自分の行為の押しつけ」「自分の気持ちの突き放し」になる（ユーザー指摘 2026-07-31）:
#   「ありがとうございます。頑張りますよ。」   → 「頑張ります。」  （自分の決意）
#   「次に志望動機を具体的にまとめますよ。」   → 「まとめます。」  （手順の説明）
#   「プライベートの時間が減るのが心配ですよ。」→「心配です。」    （自分の気持ち）
#   「卒業式の準備でバタバタしていますよ。」   → 「バタバタしています。」（自分の状況）
# 一方、助言・情報（「〜すると効果的ですよ」「アプリが便利ですよ」「平均で3800万円ですよ」）、
# 自分の事情の説明（「〜やったんですよ」）、タメ口（「サポートするよ」）は自然なので触らない。
# ★「です」述語（評価・情報）と「ます」述語（自分の行為）で扱いが分かれるのが要点。
_POLITE_VERB_TAIL = re.compile(r"(?:ます|ました|ません|ませんでした)$")
# ★「ます」述語でも「ね」が自然な2種類。既存コーパスの「〜ますね」8,012件を数えて
#   多い語尾から取った（付加を全面禁止したら付加可能なターンが44%まで落ち、
#   下限ゲート45%が構造的に届かなくなったため戻す）。
#   (1) 予告 … これから自分がして、相手が結果を受け取る行動。相手に知らせる働きが
#       あるので「ね」が自然（「また連絡しますね。」「PDFで送りますね。」）。
#       「おすすめしますね」（助言）「ピックアップしましたね」（報告）は入れない。
#   (2) 帰結 … 相手と一緒に見ている数値・結果の確認（「予算内に収まりますね。」
#       「説得力が増しますね。」）。
#   除外: 「助かります」は自分の利得（ユーザー指摘「助かりましたね」）。
#   「あります」「になります」を丸ごと許すと「興味がありますね」「参考になりますね」
#   （＝自分の関心・自分の利得）まで通るので、共有の観察になる連語だけを列挙する。
_MASU_NE_OK = re.compile(
    r"(?:連絡します|共有します|確認します|報告します|提出します|返信します|回答します"
    r"|用意します|準備します|手配します|メモします|相談します|お伝えします"
    r"|送ります|お送りします|見せます|お見せします|調べます|探します"
    r"|聞いてみます|やってみます|試してみます|伝えます)$"
    r"|(?:収まります|円になります|余裕があります|可能性があります|説得力があります"
    r"|効果があります|方法があります|増します|減ります|下がります|上がります"
    r"|抑えられます|なくなります|やすくなります|楽になります)$")
# 進行中の自分の状態。「よ」も「ね」も付かない（「ね」は相手のことを指すことになる）。
# ★「〜中です」は「〜ています」と同じ進行の言い方（実測「省エネ設備を検討中ですよね」
#   ＝自分の検討状況なので同意を求める相手がいない）。第三者の状況を指す
#   「工事中ですね」は巻き込むが、付加しないだけでモデルが書くぶんは残る。
_PROGRESSIVE_TAIL = re.compile(
    r"(?:ています|てます|ております|ていません|てません|中です|中でした)$")
# 自分の気持ち。「心配ですよ」は突き放し、「心配ですね」は相手の心配を指すので、
# どちらも付けない（「心配です」「心配なんです」が自然）。
# ★「感謝」「光栄」「幸い」も自分の気持ち（実測「協力させていただき感謝ですよね」）。
_SELF_FEELING = re.compile(
    r"(?:心配|不安|残念|楽しみ|安心|嬉しい|うれしい|悲しい|つらい|辛い|寂しい|"
    r"申し訳ない|心苦しい|悔しい|感謝|光栄|幸い)(?:です|でした|だ|だった)?$")

# 自分の願望。「よね」は相手への同意要求なので付かない（「やりたいですよね」）。
_SFP_NO_YONE_WANT = re.compile(r"(?:たい|たいです|たいん?です)$")
# 自分の見込み「〜えそうです」。可能形＋様態（ユーザー指摘「書けそうですよ」）は
# 自分がやれる見込みの話なので、「よ」は押しつけ、「ね」は相手のことを指してしまう。
# ★形容詞語幹＋そう（「面白そうですね」「難しそうですね」「良さそうですね」）は
#   相手の話への評価で自然なので、可能形の語尾（え段）と「できそう」だけに絞る。
#   「そうですね」（肯定の応答）も え段ではないので当たらない。
_PROSPECT_TAIL = re.compile(r"(?:[けげせぜてでねべめれえ]|でき)そう(?:です|でした)$")

# ── 「よね」の用法 ─────────────────────────────────────────────────────
# 「よね」は相手も知っている事柄に同意を求める語。自分がこれからすること・自分だけが
# 知っている経験や気持ちには付かない（ユーザー指摘 2026-07-31）:
#   「ちょっと試してみますよね。」  → 「ちょっと試してみますね。」    （行動の宣言）
#   「SNS運用をやっていましたよね。」→ 「SNS運用をやっていました。」  （経験の報告）
# 「ね」は同じ場面でほぼ常に成立する（逆は成立しない）ので、迷う形は「ね」に寄せる。

# 「〜ます」止めで「よね」が成立する述語。状態・思考・可能・自発だけを許し、
# それ以外の「〜ます」は行動の宣言になりやすいので「よね」を付けない。
# ★相手の理解・共通の見解を確かめる語尾だけ。「あります」「なります」を許すと
#   「興味がありますよね」「参考になりますよね」（＝自分の関心・自分の利得。同意を
#   求める相手がいない）が通ってしまう実測があった。主語が自分か相手かは形からは
#   分からないので、迷う語尾は「ね」に寄せる（「ね」なら同じ場面で必ず成立する）。
# ★「います」も入れない。「行います」「使います」の語尾を丸ごと拾ってしまう
#   （「〜ています」の進行形は下の _YONE_OWN_STATE で別に弾く）。
# ★否定形は許可しない。「分かりませんよね」は自分が分からない意味にも相手が
#   分からない意味にもなり、前者では同意を求める相手がいない（実測は前者）。
_YONE_OK_MASU = re.compile(r"(?:わかります|分かります|思います)$")
# 非過去の敬語。否定「〜ません」も同じ扱い（実測「方法が分かりませんよね」）。
_MASU_TAIL = re.compile(r"(?:ます|ません)$")
# 過去の敬語。自分の行動・経験の報告になりやすい（「メモしましたよね」「やりましたよね」）。
# 「でした」「ませんでした」は共有事実の確認（「あの件、先週でしたよね」）で自然なので
# 対象外（どちらも「〜でした」で終わるので、この正規表現には当たらない）。
_PAST_POLITE_TAIL = re.compile(r"ました$")
# 自分の進行中の状態・願望。相手はまだ知らないので同意を求められない。
_YONE_OWN_STATE = re.compile(
    r"(?:たいと思います|たいです|たい|ています|てます|ております|ておりました|"
    r"ていません|てません|ておりません)$")
# これから自分がすること（て形＋補助動詞の敬語、推量）。「よね」ではなく「ね」。
_YONE_VOLITION = re.compile(
    r"(?:てみます|ておきます|てきます|ていきます|てあげます|でしょう|でしょ)$")
# タメ口の動詞の言い切り（辞書形）。「よね」が成立するのは相手の話や一般論のときだけで、
# 自分がこれからすることに付くと壊れる（実測「すぐにチェックするよね」→「〜するね」）。
# 「ある」「思う」等の状態・思考の動詞は下の allowlist で「よね」を残す。
# 「だ」で終わる形（「〜なんだ」「そうだ」）は対象外＝「〜だよね」はそのまま。
# 敬語の「〜ます」「〜です」は別扱いなので、直前が「ま」「で」の「す」は除く。
_PLAIN_VERB_TAIL = re.compile(r"(?<![まだで])[うくぐすずつぬぶむる]$")
_YONE_OK_PLAIN = re.compile(r"(?:わかる|分かる|思う)$")
# い形容詞（非過去）＋です。「よね」は相手も知っている前提が要るが、この形は
# 直前に聞いた説明への評価に使われることが多く、その場合は相手に同意を求める
# 相手がいない（ユーザー指摘 2026-08-01:「受付で受け取るんですね、分かりやすいですよね。」
# 実測でも「バランスが良いですよね」「把握しやすいですよね」＝自分の提案の自己評価）。
# 「ね」に倒せば同じ場面で必ず成立するので「よね」だけ落とす。
# 「〜くらいです」「きれいです」「〜たいです」「〜みたいです」は別の形なので除く。
_ADJ_DESU_TAIL = re.compile(r"(?<!きれ)(?<!ら)(?<!た)いです$")
# 名詞・外来語・数字で終わる形。コピュラ（だ・です）が無いと終助詞を付けられない
# （実測の壊れた出力:「オリジナルエコバッグをプレゼントよね。」）。動詞・形容詞・
# 敬語の語尾はすべて仮名で終わるので、この判定に当たるのは体言止めだけ。
_NOUN_TAIL = re.compile(r"[一-龥ァ-ヴー0-9０-９A-Za-zＡ-Ｚａ-ｚ]$")
# 過去形いっさい（答えのターンで終助詞を足さない判定に使う）
_PAST_TAIL = re.compile(r"(?:ました|でした|た)$")
# 疑問で終わるターン。次のターンは「答え」なので、相手が知らないことを伝える発話になる。
_QUESTION_TAIL = re.compile(r"[？?]$")
# 文末の「よね」。「よね？」は明示的な確認の疑問文なので触らない（「。」止めだけ直す）。
_YONE_TAIL = re.compile(r"よね(?P<end>ー*[。！]?)$")
# 「よね」の前に別の終助詞がある形（実測「予算オーバーかしらよね」）。重ね付けなので落とす。
_OTHER_SFP_TAIL = re.compile(r"(?:かしら|かな|っけ|もん|さ|わ)$")
# 文末の「よ」（「よね」は上で扱うので、後ろに「ね」が無い形だけが当たる）。
_YO_TAIL = re.compile(r"よ(?P<end>ー*[。！]?)$")

# ── 「ね」の用法 ─────────────────────────────────────────────────────
# 「ね」は相手と共有していること・相手のことに向けて使う（共感「大変でしたね」、
# 確認「水曜の2時ね」、自分の行動の予告「送りますね」、依頼の緩和「見てくださいね」）。
# 自分の行動・経験・心情の報告に付けると、相手に確認を求める形になって噛み合わない
# （ユーザー指摘 2026-07-31）:
#   「受講者の定着率が15パーセント向上しましたね。」→「〜向上しました。」
#   「先輩の話、すごく参考になりましたね。」        →「〜参考になりました。」
#   「あはは、期待していますね。」                  →「〜期待しています。」
# 「〜でした」（評価・回想の共感:「大変でしたね」「そうでしたね」）と
# 「〜んですね」（相手の話を受けて理解を示す形）は自然なので触らない。
# 言いさし（テ形・接続助詞・助詞止め）。敬語の対話でここに終助詞を付けると
# 馴れ馴れしい調子になる（実測、先生への発話で「進路について、不安が拭えなくてね。」
# 「働き方のバランスが気になるのですがね。」「法事の準備で戸惑っておりましてね。」）。
# 「〜拭えなくて。」「〜気になるのですが。」で言いさしとして成立しているので、
# 足さないのが正解。タメ口の「連絡してね」「安心してね」（依頼の緩和）は自然なので、
# 口調で切り分ける。「〜ですからね」は理由の提示として自然なので「から」は外す。
# 敬語の言いさし。対話全体が「混合」「タメ口」指定でも、その発話自体が「です／ます」
# なら敬語なので同じ扱いにする（実測「学生割引、聞いたことないんですがね。」＝
# 混合指定の対話に残った。ユーザー指摘「気になるのですがね」と同じ形）。
_POLITE_IISASHI = re.compile(
    r"(?:です|ます|ました|ません|でした)"
    r"(?:が|けど|けれど|けども|ので|のに|し)$")   # 「ですからね」は自然なので から は除く
_IISASHI_TAIL = re.compile(
    r"(?:て|で|が|けど|けれど|けども|ので|のに|し|たり|を|は|に|へ|と|も|や|とか)$")

# ★敬語の動詞述語（「〜ます」「〜ました」「〜ません」）には「ね」も足さない
#   （2026-07-31 ユーザー指摘「根本的に『ね』が多すぎるので厳選してほしい」）。
#   実測の変な例:「Googleカレンダーの活用をおすすめしますね。」
#                「インターンを2件ほどピックアップしましたね。」
#                「今日は本当に助かりましたね。」
#   「ね」を機械で足してよいのは、相手に向いている次の3つだけに絞る:
#     ・評価・確認の「です」述語（「大変ですね」「水曜の2時ですね」）
#     ・依頼・勧誘の緩和（「ご確認くださいね」「進めましょうね」… _SFP_NE_ONLY 側）
#     ・タメ口の言い切り（「そうだね」「送っておくね」）
#   自分の行動の予告（「では確認しますね」）は自然だが、機械で足すと数が出すぎるので
#   モデル自身が書いたぶんだけ残す。判定は上の _POLITE_VERB_TAIL を使い回す。
_NE_TAIL = re.compile(r"(?<!よ)(?<!もん)ね(?P<end>ー*[。！]?)$")
_NE_OWN_PAST = re.compile(r"ました$")
_NE_OWN_FEELING = re.compile(
    r"(?:期待|心配|楽しみに|感謝|反省|後悔|覚悟)(?:して|しており)(?:い)?ます$")

# 呼びかけで終わる発話。終助詞を足すと壊れる（実測「よろしくお願いします、ミキさんよね。」
# 「ぜひお願いします、タカシさんよ。」）。敬称付きは話者名を知らなくても判る。
# 裸の名前（「ありがとう、ミキ。」）は _ends_with_vocative が話者名を見て弾く。
_VOCATIVE_TAIL = re.compile(
    r"[ぁ-んァ-ヴー一-龥]{1,10}"
    r"(?:さん|くん|君|ちゃん|先生|先輩|部長|課長|係長|社長|店長|教授|さま|様)$")


def _ends_with_vocative(stem: str, names=None) -> bool:
    """語尾が相手への呼びかけ（「ミキさん」「タカシ」）か。"""
    if _VOCATIVE_TAIL.search(stem):
        return True
    return any(re.search(rf"{re.escape(str(n))}{_VOCATIVE_HON}$", stem)
               for n in (names or ()) if n)


def _speaker_names(turns: list) -> set:
    return {str(t.get("speaker", "")) for t in turns if isinstance(t, dict)} - {""}


def fix_vocative_sfp(turns: list) -> int:
    """呼びかけの後ろに付いた終助詞を落とす（落とした箇所数を返す）。

    「了解です、タカシさんよね。」「ぜひお願いします、タカシさんよ。」のように、
    呼びかけで終わる発話に終助詞が付くと呼びかけが名詞句に化けて壊れる。
    """
    names = _speaker_names(turns)
    fixed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        m = _SFP_TAIL.search(u)
        if not m or not _ends_with_vocative(u[:m.start("sfp")], names):
            continue
        new = u[:m.start("sfp")].rstrip("、") + (m.group("end") or "。")
        if _core_len_utt(new) < 2:
            continue
        t["utterance"] = new
        fixed += 1
    return fixed

# タメ口で、名詞・ナ形容詞のように「だ」を挟まないと終助詞が付かない語尾。
# 「作れそうよね」は女性語的で不自然。「作れそうだよね」にする。形式名詞も同じ
# （実測「理解してくれるはずよ」「決めるつもりよね」→「はずだよ」「つもりだよね」）。
# ★「よう」は入れない。意志形（「試してみよう」）に「だ」を挟むと
#   「試してみようだな」になる（実測）。意志形は _SFP_NE_ONLY で「ね」だけ許す。
_SFP_NEEDS_DA = re.compile(r"(?:そう|みたい|はず|つもり|わけ)$")

# 「だ」で言い切る形。書き言葉なので必ず終助詞を足す（ユーザー指摘の最優先項目）。
_SFP_DA_TAIL = re.compile(r"(?:んだ|のだ|だ)$")
# 「〜んだ」「〜のだ」「〜んです」は自分の事情の説明。ここに「ね」を足すと相手のことを
# 推し量る意味になる（「迷ってるんだね」＝相手が迷っている／「作ったことがあるんですね」
# ＝作ったのは相手）。「よ」「よね」「な」だけ許す。相手の話への相槌としての
# 「そうなんですね」は、モデル自身が既に終助詞を付けているので here には来ない。
_SFP_NDA_TAIL = re.compile(r"(?:んだ|のだ|んです|のです)$")

# 足す終助詞と重み。人間コーパスの内訳（ね51% / な17% / よね12% / よ10%）に寄せつつ、
# 合成側が自力で出しにくい「な」「よね」を厚くする（自力の実測は ね64% / よ23% /
# よね5% / な1%。「よ」は放っておいても出るので薄くする）。
_SFP_WEIGHTS = {"ね": 38, "な": 24, "よね": 22, "よ": 16}


def sfp_candidates(stem: str, answer: bool = False,
                   polite: bool = False) -> list[str]:
    """この語尾に足してよい終助詞。足せないなら空リスト。

    answer=True は直前ターンが疑問文の発話（＝答え）。相手が知らないことを伝える
    発話なので同意要求の「よね」は成立しない。過去形（自分の経験・行動の報告）には
    そもそも足さない（「やっていましたね」も相手に確認を求める形になってしまう）。
    polite=True は敬語指定の対話。テ形の言いさしに「ね」を足すと馴れ馴れしくなる。
    """
    if _SFP_NO_ADD.search(stem) or _SFP_NO_ADD_TAIL.search(stem):
        return []
    if ((polite and _IISASHI_TAIL.search(stem))
            or _POLITE_IISASHI.search(stem)):
        return []                          # 「不安が拭えなくて。」のまま置く
    if _NOUN_TAIL.search(stem) or _VOCATIVE_TAIL.search(stem):
        return []                          # 体言止め・呼びかけには付けられない
    if (_PROGRESSIVE_TAIL.search(stem) or _SELF_FEELING.search(stem)
            or _PROSPECT_TAIL.search(stem)):
        return []                          # 自分の状態・気持ち・見込みは終助詞なしが自然
    if answer and _PAST_TAIL.search(stem):
        return []                          # 「〜やっていました。」のまま置く
    if _SFP_YO_ONLY.search(stem):
        return ["よ"]                      # 自分の能力・提案の表明
    if _SFP_NE_ONLY.search(stem):
        return ["ね"]                      # 依頼・勧誘・テ形・助詞止めは「ね」だけ
    # 自分の願望。「よ」は相手への押しつけになる（実測「リストアップしたいですよ」）。
    # 「ね」「な」は同じ場面で共感として成立する（「早く休みたいですね」「行きたいな」）。
    if _SFP_NO_YONE_WANT.search(stem):
        return ["ね"] if _POLITE_STEM.search(stem) else ["ね", "な"]
    # 敬語の動詞述語（「〜ます」「〜ました」）には何も足さない。「よ」は自分の行為の
    # 押しつけになり、「ね」は数が出すぎて「ね」ばかりの対話になる（ユーザー指摘）。
    # 「です」述語（評価・情報・確認）とタメ口はそのまま許す。
    if _SFP_NDA_TAIL.search(stem):
        out = ["よ"]
    elif _POLITE_VERB_TAIL.search(stem):
        # 予告（「また連絡しますね」）と帰結（「予算内に収まりますね」）だけ「ね」を許す
        return ["ね"] if _MASU_NE_OK.search(stem) else []
    else:
        out = ["ね", "よ"]
    if not (answer                         # 答えは相手が知らないことを伝える発話
            or _NO_YONE_STEM.search(stem)  # 意志表明・依頼・願望には「よね」を付けない
            or _SFP_NO_YONE_WANT.search(stem)
            or _ADJ_DESU_TAIL.search(stem)      # い形容詞＋です＝直前の説明への評価
            or _YONE_OWN_STATE.search(stem)     # 自分の状態・願望
            or _YONE_VOLITION.search(stem)      # これから自分がすること
            or _PAST_POLITE_TAIL.search(stem)   # 自分の行動・経験の報告
            or (_MASU_TAIL.search(stem)         # 「〜ます」は状態・思考・可能だけ
                and not _YONE_OK_MASU.search(stem))
            or (_PLAIN_VERB_TAIL.search(stem)   # タメ口の言い切りも同じ
                and not _YONE_OK_PLAIN.search(stem)
                and not _SFP_NEEDS_DA.search(stem))):   # 「作れそう」は下で「だ」を挟む
        out.append("よね")
    # 「な」はタメ口の感想だけ。動詞の言い切りに付くと禁止の意味になるので
    # 「〜だ」「〜た」「〜い」止めに限り、敬語には付けない。
    if _NA_STEM_OK.search(stem) and not _POLITE_STEM.search(stem):
        out.append("な")
    # タメ口の名詞・ナ形容詞止めは「だ」を挟む（「作れそうよね」→「作れそうだよね」）
    if _SFP_NEEDS_DA.search(stem) and not _POLITE_STEM.search(stem):
        out = ["だ" + x for x in out] + ["だな"]
    return list(dict.fromkeys(out))          # 重複を落として順序は保つ


def append_sfp(text: str, sfp: str) -> str:
    """文末に終助詞を1つ足す。足せない形はそのまま返す。"""
    body = text.rstrip()
    end = ""
    if body and body[-1] in "。！?？!":
        body, end = body[:-1], body[-1]
    body = body.rstrip("、")
    if not body or sfp not in sfp_candidates(body):
        return text
    return body + sfp + (end or "。")


def _pick_sfp(cands: list[str], recent: list[str], rng) -> str | None:
    """直前に使った終助詞を避けつつ、重み付きで1つ選ぶ。"""
    pool = [c for c in cands if c.lstrip("だ") not in recent] or cands
    if not pool:
        return None
    w = [_SFP_WEIGHTS.get(c.lstrip("だ"), 1) for c in pool]
    return rng.choices(pool, weights=w, k=1)[0]


def _is_polite_style(metadata: dict) -> bool:
    """speech_style が敬語のみ（混合・タメ口を含まない）か。"""
    style = str((metadata or {}).get("speech_style") or "")
    return "敬語" in style and not ("混合" in style or "タメ口" in style
                                   or "カジュアル" in style)


def fix_ne_misuse(turns: list, polite: bool = False, skip_speaker: "str | None" = None) -> int:
    """自分の行動・経験・心情の報告に付いた敬語の「ね」を落とす（落とした箇所数）。

    polite=True（敬語指定の対話）ではテ形の言いさしの「ね」も落とす
    （「不安が拭えなくてね。」→「不安が拭えなくて。」）。

    「ね」は相手と共有していることに向ける語なので、自分の報告に付けると相手に
    確認を求める形になる（実測「定着率が向上しましたね」「参考になりましたね」
    「期待していますね」）。共感・回想の「〜でしたね」、相手の話を受ける
    「〜んですね」、自分の行動の予告「〜しますね」は自然なので触らない。

    skip_speaker の発話は直さない（傾聴 al の聞き手。★2026-11-02 試走3〜5: 聞き手の共感「それは焦りましたね」
    「安心しましたね」を自分の経験の報告とみなして落とし、「それは焦りました。」＝聞き手自身の話に変えていた。
    生の出力では聞き手のターンの43%が「ね」で終わっていたのに、最終結果に「〜ました。」の受けが約1割残った）。
    """
    fixed = 0
    for t in turns:
        if not isinstance(t, dict) or (skip_speaker and t.get("speaker") == skip_speaker):
            continue
        u = str(t.get("utterance", ""))
        m = _NE_TAIL.search(u)
        if not m:
            continue
        stem, end = u[:m.start()], m.group("end") or "。"
        if _SFP_NDA_TAIL.search(stem):
            continue                        # 「〜んですね」は相手の話を受ける形
        if not (_NE_OWN_PAST.search(stem) or _NE_OWN_FEELING.search(stem)
                or (polite and _IISASHI_TAIL.search(stem))
                or _POLITE_IISASHI.search(stem)):
            continue
        new = stem + end
        if _core_len_utt(new) < 2:
            continue
        t["utterance"] = new
        fixed += 1
    return fixed


def fix_yo_misuse(turns: list) -> int:
    """自分の行為・気持ち・状態に付いた敬語の「よ」を落とす（落とした箇所数を返す）。

    「よ」は相手が知らないこと・相手のためになることを教える語。敬語の動詞述語に
    付くと押しつけがましく響く（実測「頑張りますよ」「まとめますよ」
    「バタバタしていますよ」「心配ですよ」）。
    助言・情報の「です」述語（「効果的ですよ」「便利ですよ」）、自分の事情の説明
    （「〜やったんですよ」）、タメ口（「サポートするよ」）は自然なので触らない。
    """
    fixed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        m = _YO_TAIL.search(u)              # 「よね」は _YONE_TAIL 側の担当
        if not m:
            continue
        stem, end = u[:m.start()], m.group("end") or "。"
        if _SFP_NDA_TAIL.search(stem) or _SFP_YO_ONLY.search(stem):
            continue                        # 「〜んですよ」「ご提案できますよ」は自然
        if not (_POLITE_VERB_TAIL.search(stem) or _PROGRESSIVE_TAIL.search(stem)
                or _SELF_FEELING.search(stem)):
            continue                        # タメ口・「です」述語は触らない
        new = stem + end
        if _core_len_utt(new) < 2:
            continue
        t["utterance"] = new
        fixed += 1
    return fixed


# ── 担当者の「用件の復唱」に付ける「ね」 ─────────────────────────────────
# 接客の担当者は、客の用件を受けて要点を言い換えて返す（復唱）。ここを「です。」
# 「でございます。」で言い切ると、確認ではなく事実の断定になり、耳で聞くと事務的で
# 冷たく響く（ユーザー指摘 2026-09-19）。実測 接客コーパス（別の担当者ペルソナ） の
# 100対話では、担当者の序盤ターン400件のうち63件がこの言い切りだった:
#   「配送の一時休止をご希望でございます。承知いたしました。」
#   「サイズが合わなかったとのことです。交換をご希望でよろしいでしょうか？」
# 復唱は「相手と共有している内容を相手に向けて確かめる」発話なので「ね」が付く形が
# 正しい（「〜ご希望でございますね。」）。プロンプト側には既に「〜ですね」の復唱例が
# 入っているが従い切らないので、機械で足す（指示を増やすと無関係なゲートの棄却率が
# 上がる実測があるため、プロンプトには書かない）。
#
# 足すのはコピュラ「です／でございます」の言い切りだけにする。「〜となっております」
# 「〜がございます」のような動詞述語は情報提供であって復唱ではない。
_READBACK_COPULA = re.compile(r"(?:です|でございます)$")
# 復唱の印。これが付いていれば序盤でなくても復唱と判る（「19時頃のご到着という
# ことです。」「利用時間についてです。」）。読点をはさむ形（「〜について、です。」）も拾う。
_READBACK_MARK = re.compile(
    r"(?:ということ|とのこと|ご希望|ご要望|ご相談|ご確認|お探し|お困り|の件|について)"
    r"[、，]?(?:で)?(?:です|ございます)$")
# 復唱ではない「です／でございます」。肯定の応答・自分の評価・推薦に「ね」を足すと、
# 相手に確認を求める形になって噛み合わない（「もちろんですね」「おすすめですね」）。
_READBACK_NO_ADD = re.compile(
    r"(?:さようです|さようでございます|左様です|左様でございます"
    r"|そうです|そうでございます|その通りです|そのとおりです"
    r"|おっしゃる通りです|おっしゃるとおりです"
    r"|もちろんです|もちろんでございます|結構です|結構でございます"
    r"|大丈夫です|可能です|可能でございます|以上です|以上でございます"
    r"|よかったです|良かったです|何よりです|何よりでございます|こちらこそです"
    r"|おすすめです|おすすめでございます|お勧めです|お勧めでございます)$")
# 「はい。」「いらっしゃいませ。」のような短い受け止めは読み飛ばし、その次の文を見る
# （「いらっしゃいませ。プランの変更をご希望でございます。」）。
_READBACK_LEAD_LEN = 10
# ★序盤を過ぎた復唱の見分け（2026-09-22。ユーザー指摘「ブライダルコースです。」が
# ターン8で window の外に落ちていた）。復唱の印が無くても、客が直前に言った語をそのまま
# 繰り返す体言＋コピュラ（「ブライダルコースです。」「来週の土曜日、午後です。」
# 「090の、1234、5678でございます。」）は復唱。見分けは add_customer_echo_ne と同じ
# 「直前の発話と語（漢字・カタカナ・数字の2文字以上）が重なるか」。ただし
#   - 直前の客のターンが問い（「メールですか？」→「はい。〜のメールでございます。」）なら
#     これは答えであって復唱ではない。「お願いできますか？」のような依頼だけは例外。
#     客の短い体言止め（「ログインボタン。」）も聞き返しなので、その次は答え。
#   - 語尾は名詞・数字で終わる形（体言＋です）か「〜たのです」に限る。形容詞・動詞述語
#     （「優しいです」「よかったです」「はずです」「みたいです」）は評価・報告なので触らない。
# 実測 接客コーパス（別の担当者ペルソナ） の100対話で、序盤の規則が拾わなかった復唱の
# 言い切りは10件（住所・電話番号・個数・日時の復唱）、この見分けで全部拾い、答え・評価の
# 誤検出は0件（体言止めの聞き返しを除く前は「ログインボタン。」への答え1件を拾っていた）。
# ★語は文字種ごとに切る。「12345番のシオリ様」を「12345番」「シオリ様」と切ると、客の
#   「番号は、12345で、名前はシオリです」のどちらにも一致しない（2026-09-22 実測）。
_READBACK_ECHO_TOKEN = re.compile(r"[一-龥]{2,}|[ァ-ヴー]{2,}|[0-9０-９]{2,}")
_READBACK_ECHO_TAIL = re.compile(r"(?:[一-龥ァ-ヴー0-9０-９]|[ただ](?:の|ん))$")
_READBACK_PREV_REQUEST = re.compile(
    r"お願い(?:できますか|できませんか|してもいいですか|してもよろしいでしょうか"
    r"|したいんですけど|したいのですが|できたら)")


def _is_readback_echo(stem: str, prev_utterance: str) -> bool:
    """担当者のコピュラ文 stem（「〜です」「〜でございます」込み）が、直前の客の発話
    prev_utterance の復唱か。"""
    pu = prev_utterance.rstrip()
    if _QUESTION_TAIL.search(pu) and not _READBACK_PREV_REQUEST.search(pu):
        return False                        # 問いへの答え
    if (_NOUN_TAIL.search(pu.rstrip("。！？、"))
            and _core_len_utt(pu) <= _READBACK_LEAD_LEN):
        return False                        # 客の体言止めの聞き返し（「ログインボタン。」）への答え
    body = _READBACK_COPULA.sub("", stem)
    if not _READBACK_ECHO_TAIL.search(body):
        return False
    return any(w in pu for w in _READBACK_ECHO_TOKEN.findall(body))


def readback_sites(turns: list, metadata: dict, window: int = 8) -> list:
    """担当者が客の用件を復唱している文を (ターン番号, 文番号, 語幹, 文末記号, ねの有無)
    で列挙する。add_readback_ne（無ければ足す）と thin_ne（あれば落とさない）で共有。

    対象は metadata の role が「担当者」の話者のターンで、直前が客のターンのもの。
    序盤（ターン番号 < window）は1文目のコピュラ文すべて、それ以降は復唱の印
    （「〜ということです」「〜についてです」）が付いているもの、または客の直前の発話の
    語を繰り返す体言止め（_is_readback_echo。「ブライダルコースです。」）だけ。
    「ですか？」「ですよね。」のように別の終助詞・疑問の形になっているものは
    _READBACK_COPULA に当たらないので入らない。「です？」で終わる不完全な聞き返しは入る
    （add_readback_ne が「ですね？」にする）。

    role を見るので、担当者のいないシナリオ（general / first_meeting）では空になる。
    """
    if window <= 0:
        return []
    staff = {s.get("name") for s in (metadata.get("speakers") or [])
             if isinstance(s, dict) and s.get("role") == "担当者" and s.get("name")}
    if not staff:
        return []
    names = _speaker_names(turns)
    out = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or str(t.get("speaker", "")) not in staff:
            continue
        prev = turns[i - 1] if i > 0 else None
        if not (isinstance(prev, dict) and str(prev.get("speaker", "")) not in staff):
            continue                        # 直前が客のターンでなければ復唱ではない
        sents = [s for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s]
        if not sents:
            continue
        j = 0
        if (len(sents) > 1 and _core_len_utt(sents[0]) <= _READBACK_LEAD_LEN
                and not _READBACK_COPULA.search(sents[0].rstrip("。！？ね"))):
            j = 1                           # 「はい。」などの受け止めの次が復唱
        m = re.search(r"(?P<end>[。！？]*)$", sents[j])
        stem, end = sents[j][:m.start()], m.group("end") or "。"
        has_ne = stem.endswith("ね")
        if has_ne:
            stem = stem[:-1]
        if not _READBACK_COPULA.search(stem):
            continue
        echo = (i >= window or j > 0) and _is_readback_echo(
            stem, str(prev.get("utterance", "")))
        if j > 0 and i >= window and not echo:
            continue                        # 序盤以外の2文目は伝聞の報告のことがある
        if not _READBACK_MARK.search(stem) and (i >= window or j > 0) and not echo:
            continue                        # 序盤の1文目以外は、復唱の印か語の繰り返しが要る
        if _READBACK_NO_ADD.search(stem) or _SFP_NO_ADD.search(stem):
            continue
        if any(re.search(rf"{re.escape(str(n))}(?:でございます|です)$", stem)
               for n in names if n):
            continue                        # 名乗り（「窓口のサトウでございます。」）
        out.append((i, j, stem, end, has_ne))
    return out


def add_readback_ne(turns: list, metadata: dict, window: int = 8) -> int:
    """担当者が客の用件を復唱する言い切りに「ね」を足す（足した数を返す）。

    見分けは readback_sites。既に「ね」が付いている復唱には何もしない。
    """
    added = 0
    for i, j, stem, end, has_ne in readback_sites(turns, metadata, window):
        if has_ne:
            continue
        t = turns[i]
        sents = [s for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s]
        sents[j] = stem + "ね" + end
        t["utterance"] = "".join(sents)
        added += 1
    return added

# ── 担当者の文末の直し ───────────────────────────────────────────────
# ユーザー指摘 2026-09-22（2回目）。どちらも接客の敬語として壊れている／馴れ馴れしい形:
#   「スタンダードプランをご利用です？」 → 「か」を落とした問いは不完全（プロンプトでも
#        禁止している形）。復唱なら add_readback_ne が「ですね？」にするので、残ったものは
#        問いとして「ですか？」に補う。
#   「お気軽にご連絡くださいね。」 → 依頼の「ね」は友人向けの緩和で、担当者が客に言うと
#        馴れ馴れしい。「ご連絡ください。」で十分。
_STAFF_DESU_Q = re.compile(r"(?<!ね)(?<!か)(?<!よ)(です|でございます|ます)？")
_STAFF_KUDASAI_NE = re.compile(r"(ください|下さい|くださいませ)ね(?=[。！]|$)")


def staff_names(metadata: dict) -> set:
    """metadata の role が「担当者」の話者名。無ければ空（接客以外のシナリオ）。"""
    return {s.get("name") for s in (metadata.get("speakers") or [])
            if isinstance(s, dict) and s.get("role") == "担当者" and s.get("name")}


def sentence_ids(turns: list, speakers: set) -> set:
    """speakers の話者のターンに含まれる全文の (ターン番号, 文番号)。thin_ne の only/protect 用。"""
    out = set()
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or str(t.get("speaker", "")) not in speakers:
            continue
        n = len([s for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s])
        out.update((i, j) for j in range(n))
    return out


# ── 接客の発話の仕上げ（2026-09-23 のレビュー 628件のうち語の置き換えで直るもの）──
# 100対話を1ターンずつ読んで拾った不自然な箇所のうち、形だけで直せるものをここで直す。
# 形で決められない（文脈が要る）ものは proofread_dialogue（LLM の校閲パス）が直す。
#
# ★担当者の笑い: プロンプトで「担当者は笑い声を出さない」と禁じているのに 51対話に出た。
#   cap_laugh_heads は「ふふ、」（読点）しか見ないので「ふふ。」（句点）を素通ししていた。
_CS_LAUGH_HEAD = re.compile(r"^(?:あは+は*|は+は+|ふふ+|うふふ+|えへへ+|へへ+)っ?[、。！]")
# ★担当者のフィラー: プロンプトで「あの／その／まあ／なんか」を禁じているのに 74件。
#   「えっと」も担当者には1〜2回までとしていたが、レビューでは全て不要と判断した。
#   読点ではさまれた形（＝フィラーとして置かれたもの）だけを落とす。「その件」「あの店」の
#   指示語は後ろに読点が無いので当たらない。
_CS_STAFF_FILLER = re.compile(
    r"(?:(?<=^)|(?<=[、。？！]))(?:まあ|その|あの|えっと|ええと|うーん|あー|あ)、")
# ★担当者の受け止め・締めの語（レビューで 10〜23件ずつ）
_CS_STAFF_SUBS = [
    # 接客で客に「どういたしまして」は使わない
    (re.compile(r"どういたしまして"), "とんでもございません"),
    # 担当者が自分の安心を言う形は主客が逆（「ご納得いただけて、安心いたしました」）
    (re.compile(r"(?:私も、?)?安心いたしました(?=[。！]|$)"), "何よりでございます"),
    # 「〜円になります」「こちらが〜になります」はプロンプトで禁止している誤用
    (re.compile(r"円になります(?=[。！]|$)"), "円でございます"),
    (re.compile(r"(?<=[。？！])(こちら[がは]?、?[^。？！]{0,20}?)になります(?=[。！])"),
     r"\1でございます"),
    (re.compile(r"^(こちら[がは]?、?[^。？！]{0,20}?)になります(?=[。！])"), r"\1でございます"),
    # 自分側の確認に「ご」を付けない
    (re.compile(r"ご確認いたしました"), "確認いたしました"),
    (re.compile(r"ご確認ですが、?"), "念のため、"),
    (re.compile(r"ご確認の結果、?"), "確認いたしましたところ、"),
    # 美化語の誤り
    (re.compile(r"ご電話"), "お電話"),
    (re.compile(r"ご好み"), "お好み"),
    # 客に「ご苦労様」「お疲れ様」は目下に向ける言い方
    (re.compile(r"(?:ご苦労様|お疲れ様)でございました"), "大変でございましたね"),
    # 「入力していただいてください」（謙譲＋命令）
    (re.compile(r"していただいてください"), "してください"),
    # ★2026-09-24 作り直し後のレビューで残ったもの
    # 謙譲の不徹底（「お調べします」「お送りしますね」→「いたします」）。「お願いします」も同じ。
    (re.compile(r"([おご][一-龥ぁ-ゖ]{1,4}?)(?<!いた)します(?=[。、ねよ？]|ので|が|$)"),
     r"\1いたします"),
    (re.compile(r"ご確認(?:できました|が完了いたしました|が取れました)"), "確認いたしました"),
    # 客への「確認してください」は「ご確認ください」
    (re.compile(r"(?<![ご])確認してください"), "ご確認ください"),
    (re.compile(r"どなたでも、?(?=あること|よくある)"), "どなたにでも"),
    # 句点化が割ってしまった形を戻す（to_sentence_periods は直したが、既存データの修復用）
    (re.compile(r"こちらこそ。(?=ありがとう|よろしく)"), "こちらこそ、"),
    (re.compile(r"(?<=[^。？！\s])もちろん。"), "もちろん、"),
    (re.compile(r"(に|で)ございます。(?=[^、。？！]{1,15}(?:という|といった))"), r"\1ございます、"),
    # ★2026-09-25 評価ラン100対話のレビュー（202件）で残ったもの
    (re.compile(r"([おご][一-龥ぁ-ゖ]{1,4}?)(?<!いた)しましょう"), r"\1いたしましょう"),
    (re.compile(r"(?<![おご])(確認|案内|用意|連絡)します(?=[。、ねよ？]|ので|が|$)"),
     lambda m: ("" if m.group(1) == "確認" else "ご") + m.group(1) + "いたします"),
    (re.compile(r"ご心配(?:は)?ございません"), "ご心配には及びません"),
    (re.compile(r"(?<=い)のでございます(?=[。！]|$)"), "です"),     # 「素晴らしいのでございます」
    (re.compile(r"(お[一-龥ぁ-ゖ]{1,4}?に)なられ(た|て|ま|る)"),              # 二重敬語「お気づきになられた」
     lambda m: m.group(1) + {"た": "なった", "て": "なって", "ま": "なりま", "る": "なる"}[m.group(2)]),
    (re.compile(r"(?<![ご])不便をおかけ"), "ご不便をおかけ"),
    (re.compile(r"(?:(?<=^)|(?<=[。？！]))でも、"), "ただ、"),
]
# 文頭の軽い受け止め → 接客の言い方
_CS_STAFF_HEAD_SUBS = [
    (re.compile(r"^(?:全然)?大丈夫ですよ[。、]"), "ご安心ください。"),
    (re.compile(r"^いえいえ[。、]"), "とんでもございません。"),
    (re.compile(r"^(?:あ、)?そこなんですけど[。、]"), "申し訳ございません。"),
    (re.compile(r"^ええ[。、]"), "はい。"),
    (re.compile(r"^(?:本当に、?)?よかったです[。]"), "何よりでございます。"),
]
# 客への「なるほど」は目上に対して失礼とされる（23件）。文頭のものだけ落とす。
_CS_STAFF_NARUHODO = re.compile(r"^なるほど[。、]")
# 「そうですね」「そうですよね」: 直前が客の問いなら考える間（「そうでございますね」）、
# そうでなければ同意（「さようでございますね」）。
_CS_STAFF_SOUDESU = re.compile(r"^(?:本当に)?そうです(?:よ)?ね[。、]")
# 「よろしいでしょうか？それでは〜」: 何を確かめたのか分からない問いを投げて答えを待たずに
# 進む形（5件）。問いだけを落とす。
_CS_STAFF_LONE_Q = re.compile(r"^よろしいでしょうか？(?=.)")
# 客の側（敬語の対話）
_CS_CUST_SUBS = [
    # 客が店員に「承知いたしました」は立場が逆
    (re.compile(r"承知(?:いたしました|しました)"), "わかりました"),
    # 客が店員に「〜でございます」（担当者の言い方。2026-09-26 の確認で6件）
    (re.compile(r"でございます"), "です"),
    # 敬語の対話でのくだけた恩恵表現
    (re.compile(r"教えてくれて"), "教えていただいて"),
    (re.compile(r"対応してくれて"), "対応していただいて"),
]
# 客が聞いた内容を確かめる言い切り → 「ね」（「ここを押せばいいんです。」19件の一部）
_CS_CUST_CONFIRM = re.compile(
    r"(?:(?:ば|たら)いいんです|ということです|ってことです|ということでございます)(?=[。]|$)")
# 「か」を落とした問い（敬語の対話の客）
_CS_CUST_DESU_Q = re.compile(r"(?<![ねかよ])(です|ます|のです)？")
# 独り言の「15分か。それなら〜」は emphasize_question_marks が「15分か？」にし、店員への
# タメ口の問いに聞こえる（2026-09-25 のレビューで4件、すべて客）。体言＋「か？」で始まり
# 同じ発話が続く形だけ「〜ですか。」にする。
# ★2026-09-26 から llm_client.emphasize_question_marks が、後ろに問いが続かない独り言の
# 「15分か？それなら〜」を先に「15分か。」へ戻すので、「か。」＋続きも拾う。
_CS_BARE_KA = re.compile(
    r"^((?:あ、|えっ、)?[^、。？！]{0,10}?(?:[一-龥ァ-ヴー0-9０-９]|まで|から|だけ|ほど))か(?:？|。(?=.))")
# 同じ発話の中で同じ定型句を2回言う（「何よりでございます。〜何よりでございます。」）
_CS_DUP_STOCK = re.compile(r"^(?:それは、?)?(何よりでございます|ご安心ください|少々お待ちください(?:ませ)?)。(.*\1)")
_CS_WAIT = re.compile(r"少々お待ちください(?:ませ)?。")


# ── 人手評価（担当者役の本人の聴取、2026-09-25 受領。100対話中 Bad 37）で指摘された担当者の言い方 ──
# 形で決まるものだけをここで直す。文脈が要るもの（心の開示への冷たい返し・謝罪の欠落・外商の
# 名前を呼ばない等）はプロンプトと校閲パスで扱う。
#
# ★「〜になります」「〜形になります」（Bad 11件）: 「です」の代わりの「になります」は接客の
#   誤用として嫌われる。変化を表す正しい用法（「来月から有料になります」）と形では区別できないが、
#   接客の担当者の文では「でございます」に替えても意味が壊れないので一律に替える。
#   替えないもの: 尊敬語「お〜になります／ご〜になります」（「ご覧になります」）、気持ちの変化
#   （「不安になります」）、「ように／ことに／く」＋なります。
_CS_NARIMASU_KEEP = re.compile(r"(?:気|世話|不安|心配|好き|嫌い|楽しみ|病気|満席|満車|満室|"
                               r"綺麗|きれい|楽|便利|快適|安心|元通り|画面|表示|状態)$")
_CS_NOUN_NARIMASU = re.compile(
    r"((?<![おご])[一-龥ァ-ヴー0-9０-９A-Za-zＡ-Ｚａ-ｚ]+)になります(?=[。、！]|が|ので|けど|$)")
# 動詞の辞書形 → ます形（「〜する形になります」「店頭に並ぶ形になります」用）
_GODAN_MASU = {"う": "い", "く": "き", "ぐ": "ぎ", "す": "し", "つ": "ち", "ぬ": "に",
               "ぶ": "び", "む": "み", "る": "り"}
_ICHIDAN_PREV = set("えけげせぜてでねへべぺめれいきぎしじちぢにひびぴみり")
_GODAN_RU = ("帰る", "入る", "走る", "切る", "知る", "要る", "減る", "限る", "参る", "混じる",
             "握る", "滑る", "蹴る", "照る", "茂る", "焦る", "散る", "喋る", "しゃべる")


def _to_masu(verb: str) -> str:
    """動詞の辞書形を「〜ます」形にする（する→します、並ぶ→並びます、できる→できます）。"""
    if verb.endswith("する"):
        return verb[:-2] + "します"
    if verb.endswith("くる") and len(verb) <= 3:
        return verb[:-2] + "きます"
    if verb.endswith("る") and len(verb) >= 2 and verb[-2] in _ICHIDAN_PREV \
            and not verb.endswith(_GODAN_RU):
        return verb[:-1] + "ます"
    if verb and verb[-1] in _GODAN_MASU:
        return verb[:-1] + _GODAN_MASU[verb[-1]] + "ます"
    return verb


_CS_KATACHI = re.compile(
    r"([一-龥ぁ-ゖァ-ヴー]{1,12}?)(?:という|といった)?形(?:に|と)なります(?=[。、！]|が|ので|$)")
# 店側の行為として「〜いたします」に替えてよい動作（「ご案内する形になります」→「ご案内いたします」）
_CS_STAFF_ACT = re.compile(r"(?:案内|手配|発送|送付|交換|返金|対応|連絡|確認|修理|用意|準備|"
                           r"お届け|お送り|お預かり|お調べ|ご案内|ご用意|ご連絡|ご返金)する$")


# 客にしてもらう動作（「マイページから入力する形になります」→「入力していただきます」）
_CS_CUST_ACT = re.compile(r"(?:入力|提示|登録|選択|申請|来店|持参|記入|署名|操作|ログイン|"
                          r"返送|返品|予約|申し込み|お申し込み|お支払い|支払い)する$")


def _fix_katachi(m: "re.Match") -> str:
    head = m.group(1)
    # 形容詞＋「形」（「モダンな形になります」＝見た目の形）は「形でございます」
    if re.search(r"(?:な|の|い)$", head) and not head.endswith("ない"):
        return head + "形でございます"
    # 打ち消し（「発生しない形になります」→「発生しません」「かからない」→「かかりません」）
    if head.endswith("ない") and len(head) >= 3:
        stem = head[:-2]
        a2i = dict(zip("かさたなまらわがばぱ", "きしちにみりいぎびぴ"))
        if stem.endswith("し"):
            return stem + "ません"
        if stem[-1] in a2i:
            return stem[:-1] + a2i[stem[-1]] + "ません"
        return stem + "ません"
    # 過去（「適用されていた形になります」）
    if head.endswith("ていた"):
        return head[:-2] + "おりました"
    if head.endswith("た"):
        return head + "のでございます"
    # 動詞で終わる（「ご案内する」「店頭に並ぶ」「入力していただく」）
    vm = re.search(r"([一-龥ぁ-ゖァ-ヴー]*?(?:する|る|う|く|ぐ|す|つ|ぬ|ぶ|む))$", head)
    if vm and not re.search(r"[一-龥ァ-ヴー]$", head):
        v = vm.group(1)
        pre = head[:vm.start(1)]
        cm = re.search(r"([おご]?)(?:[一-龥ァ-ヴー]+|申し込み|お支払い|支払い)する$", v)
        if re.search(r"[おご](?:確認|入力|記入|提示|来店|持参|署名|返送|返品|予約|申し込み|支払い)する$", v):
            return pre + v[:-2] + "いただきます"          # 「ご確認いただきます」「お支払いいただきます」
        ctx = m.string[max(0, m.start(1) - 12):m.start(1)] + pre
        staff_side = re.search(r"(?:こちら|当店|弊社|私ども|担当者|スタッフ)(?:で|から|にて)、?$", ctx)
        if _CS_CUST_ACT.search(v) and not (cm and cm.group(1)) and not staff_side:
            return pre + v[:-2] + "していただきます"
        if _CS_STAFF_ACT.search(v) or (staff_side and v.endswith("する")):
            return pre + v[:-2] + "いたします"
        return pre + _to_masu(v)
    return head + "でございます"                     # 体言（「毎月500円という形になります」）


_CS_STAFF_MANNER_SUBS = [
    # 笑い声の代わりに笑うしぐさを語る（「ふふっと笑ってしまい、」「ふふっと笑いながら」。
    # 心の開示に温かく返す指示を足した後に2件）。句ごと落とす
    (re.compile(r"(?:(?<=^)|(?<=[。？！]))(?:ふふっと|くすっと|思わず)?笑(?:ってしまい|いながら|ってしまいました|っちゃい)[、。]?"), ""),
    (re.compile(r"(?:(?<=^)|(?<=[。？！]))ふふっと笑[^、。？！]{0,6}[、。]"), ""),
    # ★2026-09-26 最終確認（100対話・130件）で残った形
    (re.compile(r"お疲れ様(?:でございます|です)(?=[。！]|$)"), "ご足労いただき、ありがとうございます"),
    (re.compile(r"でございましたでしょうか"), "でしょうか"),
    (re.compile(r"ございますかと存じます"), "ございます"),               # 「可能性がございますかと存じます」
    (re.compile(r"([一-龥]{2,})がいたしかねます"), r"\1ができかねます"),
    (re.compile(r"のご件"), "の件"),
    (re.compile(r"お足を(?:使って|お運び)いただき"), "ご足労いただき"),
    (re.compile(r"((?:高|安|多|少な|近|遠|早|遅|広|狭|新し|古|大き|小さ|難し|易し|詳し|珍し|強|弱|長|短|重|軽)い)"
                r"でございます"), r"\1かと存じます"),
    (re.compile(r"(?<=[。？！])なるほど[、。]"), ""),
    (re.compile(r"(?:それは、?|本当に、?)?(?:良|よ)かったです(?=[。！]|$)"), "何よりでございます"),
    (re.compile(r"(を完了|をお済ませ)させていただければ"), r"\1していただければ"),
    (re.compile(r"(開い|使っ|持っ|入れ|見)ておりますか？"), r"\1ていらっしゃいますか？"),
    # 指示語の「あそこ」は接客では「あちら」
    (re.compile(r"あそこ"), "あちら"),
    # 誤用の「よろしかった」
    (re.compile(r"でよろしかったでしょうか"), "でよろしいでしょうか"),
    (re.compile(r"^(?:それは、?)?(?:良|よ|宜|よろし)かったです(?:ね)?[。]"), "何よりでございます。"),
    (re.compile(r"(?<=[。？！])(?:それは、?)?(?:良|よ|よろし)かったです(?:ね)?[。]"), "何よりでございます。"),
    # 耳で聞いて伝わりにくい業界用語（評価で「同梱」「焼成（小生に聞こえる）」）
    (re.compile(r"同梱"), "同封"),
    (re.compile(r"焼成スケジュール"), "焼き上がりの予定"),
    (re.compile(r"焼成"), "焼き上げ"),
]
# 丁寧な対話の担当者だけ（子ども相手の「混合」では、やさしい「〜ですよ」はそのまま）
_CS_STAFF_POLITE_SUBS = [
    (re.compile(r"(?:全然)?大丈夫ですよ(?:ね)?"), "ご安心ください"),
    (re.compile(r"(です|ます)よ(?=[。！]|$)"), r"\1"),
    (re.compile(r"(です|ます)よね(?=[。？！]|$)"), r"\1ね"),
    (re.compile(r"ありますね(?=[。]|$)"), "ございます"),
]
# 担当者の上がり調子の確認（評価で最多の「〜す？」）: 「〜ですね？」「〜でございますね？」は
# 下がり調子の「〜ですね。」にする（復唱は念押しの確認で、問いとして上げると客を問い詰める
# 響きになる）。「〜んですね？」は「〜のですね。」。
_CS_STAFF_NE_Q = re.compile(r"(です|でございます|ます|ません|でした|ました)ね？")
_CS_STAFF_NDESU_NE_Q = re.compile(r"んですね[？。]")
# 担当者のむき出しの問い（「短いネジ？」「固定されている？」「マイページ？」）
_CS_STAFF_BARE_Q = re.compile(r"(?:(?<=^)|(?<=[。？！、]))([^。？！、]{1,20}?)？")


def _fix_bare_q(m: "re.Match") -> str:
    body = m.group(1)
    if re.search(r"(?:か|でしょう|ます|です|ません|ました|でした|ございます|ね|っけ)$", body):
        return m.group(0)
    if re.search(r"(?:る|た|ない|だ)$", body):
        return body + "でしょうか？"
    return body + "でございますね。"


def polish_cs_staff_manner(text: str, prev_cust: str, polite: bool) -> str:
    """人手評価で Bad の理由になった担当者の言い方を、形で決まる範囲で直す。"""
    new = _CS_KATACHI.sub(_fix_katachi, text)

    def _narimasu(m: "re.Match") -> str:
        if _CS_NARIMASU_KEEP.search(m.group(1)):
            return m.group(0)
        return m.group(1) + "でございます"
    new = _CS_NOUN_NARIMASU.sub(_narimasu, new)
    for rx, rep in _CS_STAFF_MANNER_SUBS:
        new = rx.sub(rep, new)
    if polite:
        for rx, rep in _CS_STAFF_POLITE_SUBS:
            new = rx.sub(rep, new)
    new = _CS_STAFF_NE_Q.sub(r"\1ね。", new)
    new = _CS_STAFF_NDESU_NE_Q.sub("のですね。", new)
    new = _CS_STAFF_BARE_Q.sub(_fix_bare_q, new)
    # 客の言葉をそのまま上がり調子で聞き返す（「ガタガタという音ですか？」）→ 下げて受け止める
    m = re.match(r"^([^。？！、]{2,16}?)(?:ですか|でしょうか)？", new)
    if m and any(w in prev_cust for w in re.findall(r"[ァ-ヴー一-龥]{2,}", m.group(1))):
        new = m.group(1) + "でございますね。" + new[m.end():]
    new = re.sub(r"拝見します", "拝見いたします", new)
    # 客のお詫び・お礼への「大丈夫でございます」は「とんでもございません」、それ以外は「ご安心ください」
    if re.match(r"^大丈夫でございます[。]", new):
        rep = ("とんでもございません。" if re.search(r"ありがと|すみません|申し訳|お手数", prev_cust)
               else "ご安心ください。")
        new = re.sub(r"^大丈夫でございます[。]", rep, new)
    # 文頭の「いいえ」: 客のお礼・お詫びへの返しなら「とんでもございません」、それ以外は「いえ、」
    if re.match(r"^いいえ[。、]", new):
        if re.search(r"ありがと|すみません|申し訳|恐縮|お手数", prev_cust):
            new = re.sub(r"^いいえ[。、]", "とんでもございません。", new)
        else:
            new = re.sub(r"^いいえ[。、]", "いえ、", new)
    return new


def polish_cs_speech(turns: list, metadata: dict) -> int:
    """接客の担当者・客の発話を、形だけで決まる範囲で直す（直したターン数を返す）。

    role が「担当者」の話者がいない対話では何もしない。客側の直しは敬語指定の対話だけ
    （混合＝子ども相手の対話では客のタメ口がそのまま正しい）。
    """
    staff = staff_names(metadata)
    if not staff:
        return 0
    polite = _is_polite_style(metadata)
    fixed = 0
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        new = u
        if str(t.get("speaker", "")) in staff:
            new = _CS_LAUGH_HEAD.sub("", new)
            new = _CS_STAFF_FILLER.sub("", new)
            for rx, rep in _CS_STAFF_HEAD_SUBS:
                new = rx.sub(rep, new)
            for rx, rep in _CS_STAFF_SUBS:
                new = rx.sub(rep, new)
            if _CS_STAFF_NARUHODO.match(new) and _core_len_utt(new) > 6:
                new = _CS_STAFF_NARUHODO.sub("", new)
            if polite and _CS_STAFF_SOUDESU.match(new):
                prev = str(turns[i - 1].get("utterance", "")) if i > 0 else ""
                rep = ("そうでございますね。" if prev.rstrip().endswith("？")
                       else "さようでございますね。")
                new = _CS_STAFF_SOUDESU.sub(rep, new)
            new = _CS_STAFF_LONE_Q.sub("", new)
            prev_cust = str(turns[i - 1].get("utterance", "")) if i > 0 else ""
            new = polish_cs_staff_manner(new, prev_cust, polite)
            # 同じ定型句が2つ並んだら1つに（「ご安心ください。〜ご安心ください。」）
            new = _CS_DUP_STOCK.sub(r"\2", new)
            # 直前の自分のターンでも「少々お待ちください」と言っていたら、2回目は落とす
            # （他に中身がある発話だけ。それだけの発話は校閲に任せる）
            prev_own = str(turns[i - 2].get("utterance", "")) if i >= 2 else ""
            if _CS_WAIT.search(prev_own) and _CS_WAIT.search(new):
                rest = _CS_WAIT.sub("", new)
                if _core_len_utt(rest) >= 4:
                    new = rest
        elif polite:
            for rx, rep in _CS_CUST_SUBS:
                new = rx.sub(rep, new)
            new = _CS_CUST_CONFIRM.sub(lambda m: m.group(0) + "ね", new)
            new = _CS_CUST_DESU_Q.sub(r"\1か？", new)
        if polite:
            # 後ろに続きがあれば独り言の受け止め（「ですか。」）、発話の末尾なら問い（「ですか？」）
            new = _CS_BARE_KA.sub(lambda m: m.group(1) + ("ですか。" if m.end() < len(new) else "ですか？"), new)
        new = re.sub(r"お構いません", "構いません", new)
        new = re.sub(r"お間に合い", "間に合い", new)
        new = new.lstrip("、")
        if new and new != u:
            t["utterance"] = new
            fixed += 1
    return fixed


# ── 客の意味のないオウム返し（2026-09-23 のレビューで 33件 / 26対話）────────────
# 極短ターンのノルマを満たすために、客が担当者の言葉をそのまま聞き返す退化。短い聞き返し
# 自体は181件あり、うち157件（料金・日時・初めて聞く名称への聞き返し）は自然だった。
# 不自然だったのは次の2つで、形で見分けられる:
#   (1) 客が自分で先に言った語を聞き返す（「ゴールド会員？」「3時間ですね。」）
#   (2) 担当者が**自分の作業**を予告・報告した文の語を聞き返す
#       （「ただいま、履歴をお調べいたしますね。」→「履歴。」、
#         「お客様情報を照会いたしますね。」→「照会。」、「お調べします？」）
# ★落とさずに、客が実際に返すはずの短い返事に**置き換える**。落として担当者のターンを
#   つなぐと「ただいま、お調べいたしますね。お待たせいたしました。」が一息になり、
#   調べている間の「待ち」が消える（最初の実装で 27件中 12件がこうなった）。
#   返事は直前の担当者の発話で決める: 作業の予告 → 「お願いします。」、確認・問い →
#   「はい、そうです。」、それ以外 → 「はい。」。後ろの担当者ターン頭の「はい。」
#   （聞き返しへの返事）は宙に浮くので剥がす。
_ECHO_TURN_MAX = 12
# (2) の「作業」は、調べる・確かめる動作に限る。「白身魚に変更させていただきますね」→
#   「白身魚ですか？」や「明日の午前中にお届けできる手配をさせていただきます」→「明日ですか？」
#   は案内された内容への驚き・確認で自然（動詞の形だけで判定したら 31件を誤って落とした）。
_ECHO_STAFF_ACTION = re.compile(
    r"(?:確認|お調べ|調べ|照会|照合|検索)(?:を)?"
    r"(?:いたし|し|させていただき)(?:ます|ますね|ております|ました)[。！]?$")
_ECHO_REPLY_HEAD = re.compile(r"^(?:はい|さようでございます|左様でございます|ええ)[。、]")
_ECHO_STOCK = re.compile(r"ありがと|お願い|本当|大丈夫|了解|承知|分か|わか|失礼|すみません|助か")
_ECHO_PREV_CONFIRM = re.compile(r"(?:ね|か)[？。]?$|？$")
# 「お願いします」と返せるのは、担当者が客のために何かをすると予告したときだけ
# （「〜差額分だけを請求いたします。」のような説明に「お願いします」は噛み合わない）。
_ECHO_PREV_ANNOUNCE = re.compile(
    r"(?:確認|お調べ|調べ|照会|照合|検索|ご案内|案内|手続き|お手続き|手配|ご用意|用意|"
    r"お送り|送信|再送|登録|変更|お取り置き|お預かり)"
    r"(?:を)?(?:いたし|させていただき|し)ます(?:ね)?[。]?$")
_ECHO_PREV_EXPLAIN = re.compile(r"(?:ます|です|ございます|おります)[。]?$")


def fix_echo_turns(turns: list, metadata: dict) -> int:
    """客の意味のないオウム返しを、短い返事に置き換える（置き換えた数を返す）。"""
    staff = staff_names(metadata)
    if not staff or not _is_polite_style(metadata):
        return 0                            # 子ども相手（混合）の聞き返しは語の意味を尋ねている
    core = lambda x: re.sub(r"[、。？！\s]", "", str(x))
    fixed = 0
    for i in range(1, len(turns) - 1):
        t, prev, nxt = turns[i], turns[i - 1], turns[i + 1]
        if (not isinstance(t, dict) or str(t.get("speaker", "")) in staff
                or str(prev.get("speaker", "")) not in staff):
            continue
        u = str(t.get("utterance", ""))
        if _core_len_utt(u) > _ECHO_TURN_MAX:
            continue
        stem = re.sub(r"[。？！]+$", "", u)
        stem = re.sub(r"^(?:あ、|えっ、|えっ。)", "", stem)
        stem = re.sub(r"[、]?(?:ですか|ですね|でしょうか)$", "", stem)
        sc = core(stem)
        if len(sc) < 2:
            continue
        pu = str(prev.get("utterance", "")).strip()
        src = [x for x in re.split(r"(?<=[。？！])", pu) if sc in core(x)]
        own = "".join(core(x.get("utterance", "")) for x in turns[:i - 1]
                      if isinstance(x, dict) and x.get("speaker") == t.get("speaker"))
        # 自分で先に言った語（「配送先を変えたい」→ 数ターン後の「あ、配送先？」）は、直前の
        # 担当者の発話にその語が無くても聞き返しとして不自然（2026-09-24 のレビュー）。
        # 対象は内容語（漢字・カタカナ・数字を含む）の聞き返しだけ。「ありがとうございます。」
        # 「本当ですか？」「はい。」のような定型の返事は、前に言っていても聞き返しではない
        # （最初の実装で 31件中 12件をこれで誤って置き換えた）。形は「〜？」か体言止め。
        bare = u.rstrip()
        own_echo = (len(sc) >= 2 and sc in own
                    and re.search(r"[一-龥ァ-ヴ0-9０-９]", sc)
                    and not _ECHO_STOCK.search(sc)
                    and (bare.endswith("？") or _NOUN_TAIL.search(bare.rstrip("。"))))
        if not (own_echo or (src and any(_ECHO_STAFF_ACTION.search(x.strip()) for x in src))):
            continue
        last = [x for x in re.split(r"(?<=[。？！])", pu) if x][-1]
        if _ECHO_PREV_ANNOUNCE.search(last):
            t["utterance"] = "お願いします。"
        elif _ECHO_PREV_CONFIRM.search(last):
            t["utterance"] = "はい、そうです。"
        elif _ECHO_PREV_EXPLAIN.search(last):
            t["utterance"] = "そうなんですね。"    # 説明を受けた返事
        else:
            t["utterance"] = "はい。"
        if isinstance(nxt, dict) and str(nxt.get("speaker", "")) in staff:
            rest = str(nxt.get("utterance", ""))
            while _ECHO_REPLY_HEAD.match(rest):     # 「はい。左様でございます。」は両方剥がす
                rest = _ECHO_REPLY_HEAD.sub("", rest, count=1)
            if rest:
                nxt["utterance"] = rest
        fixed += 1
    return fixed


# ── 校閲パス（2026-09-23 のレビュー 628件のうち文脈の判断が要るもの）──────────
# 問いに答えていない・受け止めがずれている・敬語の向きの誤り・語の誤用・数字の食い違い
# などは形では見分けられない（レビューで約400件）。生成し終えた対話を同じ LLM にもう一度
# 渡し、問題のある発話だけを直させる。返させるのは「直した発話」だけにして、話者・ターン数・
# 問題の無い発話には触れさせない。直した発話は機械で検証し、通らないものは捨てる。
_PROOF_MAX_CORE = 60        # 直した発話の実質字数の上限（プロンプトの 55 字＋余裕）
_PROOF_MAX_EDIT_RATIO = 0.5  # 対話のターン数に対する修正数の上限（超えたら全体を信用しない）


_PAUSE_NOTE_RE = re.compile(r"[（(][0-9.]+秒の沈黙のあと[）)]")


def _proofread_user_message(turns: list, metadata: dict) -> str:
    staff = staff_names(metadata)
    who = []
    for sp in metadata.get("speakers") or []:
        if isinstance(sp, dict) and sp.get("name"):
            who.append(f"{sp['name']}（{sp.get('role', '')}、{sp.get('age_range', '')}）")
    lines = [f"場面: {metadata.get('setting', '')}",
             f"話者: {'／'.join(who)}",
             f"口調: {metadata.get('speech_style', '')}",
             "", "対話:"]
    for i, t in enumerate(turns):
        # 担当者のいない対話（初対面など）は話者名で示す（全員が「客」になってしまうため）。
        role = (("担当者" if str(t.get("speaker", "")) in staff else "客") if staff
                else str(t.get("speaker", "")))
        pause = (f"（{t[PAUSE_KEY]:g}秒の沈黙のあと）" if PAUSE_KEY in t else "")
        lines.append(f"{i} {role}: {pause}{rp_render(t) if is_rp_turn(t) else t.get('utterance', '')}")
    return "\n".join(lines)


# 校閲パスの「直しすぎ」。100対話に当てた最初の実行で、指摘していない 197ターンが書き換わり、
# 次の形が繰り返し出た（プロンプトで禁じたうえで、ここでも機械で弾く）:
#   担当者の復唱を消す     「ＱＲコードですね。」→「はい、承知いたしました。」（約10件）
#   客の聞き返しを潰す     「サクラパーキング？」「延長保証。」→「はい、お願いします。」（約15件）
#   客の問いを付加疑問に   「いつまで飲めるんですか？」→「いつまで飲めるんですね？」（6件）
#   敬語の新しい誤り       「良かったですね」→「良かったでございますね」
#   子ども相手の担当者を硬くする  「〜と思います」→「〜かと存じます」
# 守る復唱は「名詞＋です／でございます＋ね」の形だけ（「ＱＲコードですね。」「3年前ですね？」）。
# 「さようでございますね」「残念でいらっしゃいますね」「間に合わせますね」は直してよい。
_PROOF_SHORT_NE = re.compile(
    r"(?<!さよう)(?<!左様)(?<!そう)[一-龥ァ-ヴー0-9０-９A-Za-zＡ-Ｚａ-ｚ](?:です|でございます)ね[。？]$")
_PROOF_ACK_ONLY = re.compile(r"^(?:はい[、。]?)?(?:お願いします|承知いたしました|かしこまりました)。$")
_PROOF_ADJ_GOZAIMASU = re.compile(r"(?:かった|しい|たい|ない)でございます")


# 担当者だけが使う言い回し（客の発話に入ったら役割の取り違え）
_STAFF_ONLY_RE = re.compile(r"拝見|さようでございます|左様でございます|承知いたしました|かしこまりました|"
                            r"お調べいたします|ご案内いたします|お客様|承ります|ご用件")


def _proofread_regression(turns: list, i: int, old: str, new: str,
                          staff: set, polite: bool) -> str:
    """校閲の修正案が、既知の直しすぎに当たるなら理由を返す（当たらなければ空文字）。"""
    is_staff = str(turns[i].get("speaker", "")) in staff
    # ★2026-09-26: 番号を取り違えて、隣の相手の発話を写す（客の発話が担当者の台詞になる）。
    #   100対話中2件（f9a5523c・ef8a1369）。修正前の隣の相手の発話と長く重なる修正は捨てる。
    for j in (i - 1, i + 1):
        if 0 <= j < len(turns) and (str(turns[j].get("speaker", "")) in staff) != is_staff:
            other = str(turns[j].get("utterance", ""))
            m = difflib.SequenceMatcher(None, new, other, autojunk=False).find_longest_match(
                0, len(new), 0, len(other))
            if m.size >= 12 and m.size > len(re.sub(r"[、。？！\s]", "", old)) * 0.5 \
                    and new[m.a:m.a + m.size] not in old:
                return f"隣の相手の発話を写している（{new[m.a:m.a + m.size][:16]}）"
    if not is_staff and _STAFF_ONLY_RE.search(new) and not _STAFF_ONLY_RE.search(old):
        return f"客の発話に担当者の言い回しを入れている（{_STAFF_ONLY_RE.search(new).group(0)}）"
    sents = lambda x: [s for s in re.split(r"(?<=[。？！])", x) if s]
    if is_staff:
        prev = str(turns[i - 1].get("utterance", "")) if i > 0 else ""
        asked = prev.rstrip().endswith("？")
        for s in sents(old):
            # 守るのは、客の言葉を繰り返した短い復唱だけ。客の問いに復唱だけで返している
            # （＝答えていない）なら直してよいし、客の言葉と重ならない「〜15分ほどですね。」
            # は情報に付いた「ね」で、復唱ではない。
            if (_PROOF_SHORT_NE.search(s) and _core_len_utt(s) <= 20 and s not in new
                    and not asked
                    and any(w in prev for w in _READBACK_ECHO_TOKEN.findall(
                        _READBACK_COPULA.sub("", re.sub(r"ね[。？]$", "", s))))):
                return f"担当者の復唱を消している（{s}）"
        if (_PROOF_ACK_ONLY.match(new) and _core_len_utt(old) <= 20
                and re.search(r"(?:ね[。？]|か？)$", old.rstrip())):
            return "担当者の復唱・確認を受け止めだけに置き換えている"
        if not polite and re.search(r"存じます|でございます", new) and not re.search(
                r"存じます|でございます", old):
            return "子ども相手の担当者の言い方を硬くしている"
    else:
        if old.rstrip().endswith("？") and not old.rstrip().endswith("ね？") \
                and new.rstrip().endswith("ね？"):
            return "客の問いを付加疑問に変えている"
        # 客の自分の気持ち・報告に「ね」を足す（「本当に、助かります。」→「助かりますね。」）。
        # 「ね」を足してよいのは聞いた内容を確かめる「〜んです／〜ということです」だけ。
        for s_old in sents(old):
            if (re.search(r"(?:ます|ました|ません)。$", s_old)
                    and s_old[:-1] + "ね。" in new):
                return f"客の自分の気持ち・報告に「ね」を足している（{s_old}）"
        if _PROOF_ACK_ONLY.match(new):
            prev = str(turns[i - 1].get("utterance", "")).strip() if i > 0 else ""
            last = ([s for s in re.split(r"(?<=[。？！])", prev) if s] or [""])[-1]
            if not _ECHO_PREV_ANNOUNCE.search(last):
                return "客の反応を、頼む場面でない所で「お願いします」にしている"
    if _PROOF_ADJ_GOZAIMASU.search(new) and not _PROOF_ADJ_GOZAIMASU.search(old):
        return "形容詞＋「でございます」（敬語の誤り）を作っている"
    return ""


# ★2026-11-09 傾聴（al）試走10: 校閲が聞き役の応答を40対話で123か所直し、46か所を「〜んですね」の言い直しにしていた。
#   問いを言い直しに替えたものもある（「そのあとどうやって正しい切符を、手に入れたんですか？」→「諦めて降りられたんですね」。
#   次の話し手の発話はその問いの答えなので宙に浮く）。校閲は al_shape ゲートの後なので、くり返しの上限も素通りする。
#   校閲のプロンプトで止めたうえで、この2つの直しすぎだけは修正案を捨てる（発話は元のまま残る）。
_AL_ECHO_TAIL = re.compile(r"(?<!そう)(?:ん|な)ですね[。！]*$")


def _al_listener_regression(turns: list, i: int, old: str, new: str, metadata: dict) -> str:
    """傾聴の聞き役の発話への修正案が、問いを消す・言い直しの受けにする直しすぎなら理由を返す。"""
    plan = topic_plan_of(metadata) or {}
    if plan.get("style") != AL_STYLE:
        return ""
    mi = metadata.get("main_speaker_index")
    if mi is None or str(turns[i].get("speaker", "")) != _main_name(metadata, mi):
        return ""
    if "？" in old and "？" not in new:
        return "聞き役の問いを問いでなくしている"
    if _AL_ECHO_TAIL.search(new.rstrip()) and not re.search(r"ね[。！？]*$", old.rstrip()):
        return "聞き役の応答を言い直しの受け（〜んですね）にしている"
    # 版c（40対話）: 校閲が「それは大変でしたね」を聞き役の問いの頭や受けに足した（校閲前0件→後4件。うれしい話・おかしい話にも）
    if "大変" in new and "大変" not in old:
        return "聞き役の応答に「大変」を足している"
    return ""


def _proofread_validate(turns: list, edits: list, metadata: dict) -> tuple[dict, list]:
    """LLM の修正案を検証する。(採用する {i: 新しい発話}, 捨てた理由のリスト) を返す。"""
    staff = staff_names(metadata)
    polite = _is_polite_style(metadata)
    ok, rejected = {}, []
    for e in edits if isinstance(edits, list) else []:
        try:
            i = int(e.get("i"))
            text = str(e.get("text", ""))
        except (AttributeError, TypeError, ValueError):
            rejected.append(f"形式不正: {e!r:.60}")
            continue
        if not (0 <= i < len(turns)):
            rejected.append(f"{i}: 範囲外")
            continue
        text = emphasize_question_marks(sanitize_utterance(text))
        if text and not text.endswith(("。", "？")):
            text = text.rstrip("、") + "。"
        old = str(turns[i].get("utterance", ""))
        n_new, n_old = _core_len_utt(text), _core_len_utt(old)
        rp_why = rp_edit_problem(turns[i], old, text)   # 重なりの場面（rp_v1）の形を壊す直しは捨てる
        if rp_why:
            rejected.append(f"{i}: {rp_why}")
        elif n_new < 2:
            rejected.append(f"{i}: 空")
        elif text == old:
            continue
        # ★2026-11-13 rp_v1 試走15: 校閲が反応のあとの続きを、2つ前の同じ話者の発話と同じ文に書き換えていた
        #   （「あー、なんか、そういうのって、知れば知るほど…」が2回）→ 近くの同じ話者の発話の写しは捨てる
        elif any(j != i and 0 <= j < len(turns)
                 and str(turns[j].get("speaker", "")) == str(turns[i].get("speaker", ""))
                 and str(turns[j].get("utterance", "")).strip() == text.strip()
                 for j in range(i - 4, i + 5)):
            rejected.append(f"{i}: 近くの発話の写し")
        elif n_new > max(_PROOF_MAX_CORE, n_old + 10) or (
                n_new < n_old * 0.3 and n_old > _ECHO_TURN_MAX):   # 短い聞き返しは短い返事になってよい
            rejected.append(f"{i}: 長さ {n_old}→{n_new}")
        elif str(turns[i].get("speaker", "")) in staff and (
                _CS_LAUGH_HEAD.match(text) or _CS_STAFF_FILLER.search(text)):
            rejected.append(f"{i}: 担当者に笑い・フィラー")
        elif _PLACEHOLDER_RE.search(text):
            rejected.append(f"{i}: 伏せ字")
        elif (why := _proofread_regression(turns, i, old, text, staff, polite)
              or _al_listener_regression(turns, i, old, text, metadata)):
            rejected.append(f"{i}: {why}")
        else:
            ok[i] = text
    if len(ok) > max(3, _PROOF_MAX_EDIT_RATIO * len(turns)):
        rejected.append(f"修正が多すぎる（{len(ok)}/{len(turns)}）ので全て捨てる")
        ok = {}
    return ok, rejected


async def proofread_dialogue(client, model: str, turns: list, metadata: dict,
                             pack: "S2PromptPack", cost_tracker: CostTracker,
                             extra_body: dict | None = None,
                             metadata_index=None) -> tuple[int, list]:
    """対話を校閲し、問題のある発話だけを直す（turns を直接書き換える）。

    (直した数, 捨てた修正の理由) を返す。応答が壊れていたら何もしない（元のまま）。
    直したあとは polish_cs_speech / fix_staff_tails をもう一度かける（LLM が戻した
    言い回しにも同じ規則を当てるため）。
    """
    if not pack.proofread_prompt:
        return 0, []
    try:
        # 重なりの場面のある対話（rp_v1）は、場面の印の読み方と敬語の向きの注意を足す（2026-11-11）
        system = pack.proofread_prompt + (RP_PROOFREAD_NOTE if is_rp_plan(topic_plan_of(metadata)) else "")
        response = await _chat_create(
            client, model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": _proofread_user_message(turns, metadata)}],
            temperature=0.2, max_completion_tokens=8192,
            extra_body=extra_body or {})
        usage = response.usage
        cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)
        data = json.loads(extract_json_text(response.choices[0].message.content or ""),
                          strict=False)
    except Exception as e:                  # 校閲は付加的な工程。失敗しても対話は残す
        logger.warning(f"[{metadata_index}] 校閲パス失敗（元のまま採用）: {e}")
        return 0, [f"失敗: {e}"]
    data = data or {}
    if isinstance(data.get("checks"), list):   # 全発話の判定から、直したものだけ取り出す
        edits = [c for c in data["checks"] if isinstance(c, dict)
                 and not c.get("ok") and c.get("text")]
    else:
        edits = data.get("edits")
    for e in edits or []:
        # 沈黙の注記（_proofread_user_message の「（N秒の沈黙のあと）」）を写してきたら外す
        if isinstance(e, dict) and isinstance(e.get("text"), str):
            e["text"] = RP_NOTE_RE.sub("", _PAUSE_NOTE_RE.sub("", e["text"])).strip()
    ok, rejected = _proofread_validate(turns, edits, metadata)
    for i, text in ok.items():
        turns[i]["utterance"] = text
    if ok:
        polish_cs_speech(turns, metadata)
        fix_staff_tails(turns, metadata)
    return len(ok), rejected


# ── 重なりの場面を1つずつ入れる（topic_plan.style: rp。2026-11-10 rp_v1）──────────────
# 本生成（ふつうの雑談）のあと、場面ごとに LLM に「turn lo〜hi の1〜4ターンを、手順どおりの新しいターンに
# 置き換える」書き換えを頼み、rp_plan.validate_scene で形を検査して当てる。だめなら場面ごとに作り直す。
RP_SCENE_TRIES = 3


async def insert_rp_scenes(client, model: str, turns: list, metadata: dict, persona: str,
                           cost_tracker: CostTracker, extra_body: dict | None = None,
                           metadata_index=None, system_prompt: str = "") -> tuple[list, list, list]:
    """(入れた場面, 入れられなかった場面, 不採用の理由) を返す。turns はその場で書き換える。

    system_prompt は本生成の system プロンプト（「話し言葉」の段だけを切り出して場面の呼び出しにも渡す。rp_scene_system）。
    """
    sys_msg = rp_scene_system(system_prompt)
    if not sys_msg:
        logger.warning(f"[{metadata_index}] 本生成のプロンプトに「話し言葉」の段が無い（場面の依頼は system なしで送る）")
    plan = topic_plan_of(metadata) or {}
    kinds = list(plan.get("overlaps") or [])
    mi = metadata.get("main_speaker_index")
    main = _main_name(metadata, mi) if mi is not None else ""
    partner = _partner_name(metadata, mi) if mi is not None else ""
    done, failed, why_all = [], [], []
    clean = lambda u: emphasize_question_marks(sanitize_utterance(u))
    for j, kind in enumerate(kinds):
        lo, hi = scene_window(len(turns), j, len(kinds))
        cands = scene_candidates(turns, kind, lo, hi, main, partner, seed=str(metadata_index))
        if not cands:                           # 範囲に置き場所が無ければ本体全体から
            cands = scene_candidates(turns, kind, *scene_window(len(turns), 0, 1), main, partner,
                                     seed=str(metadata_index))
        ok = False
        for att in range(min(RP_SCENE_TRIES, len(cands)) if cands else 0):
            a, b = cands[att]
            try:
                response = await _chat_create(
                    client, model=model,
                    messages=([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": rp_scene_prompt(
                        turns, metadata, kind, a, b, main, partner, persona,
                        variant=scene_variant(kind, str(metadata_index)),
                        opener=scene_opener(kind, str(metadata_index), scene_variant(kind, str(metadata_index))),
                        yield_style=scene_yield(kind, str(metadata_index)),
                        style_norm=_style_norm(metadata))}],
                    temperature=0.8, max_completion_tokens=3000, extra_body=extra_body or {})
                usage = response.usage
                cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)
                data = json.loads(_fix_missing_utterance_key(
                    extract_json_text(response.choices[0].message.content or "")), strict=False)
            except Exception as e:
                why_all.append(f"{kind}#{att + 1}: 失敗 {type(e).__name__}")
                continue
            new, why = validate_scene(turns, data, kind, main, partner, a, b, clean=clean,
                                      variant=scene_variant(kind, str(metadata_index)))
            # ★2026-11-13 rp_v1 試走14: 敬語の対話の場面に「まじで」が 269中7（本生成のターンは 0。本生成には casual_leak の
            #   ゲートがある）→ 場面にも同じ直し（頭のタメ口リアクションを丁寧形に）と検査をかける。言いかけは文末の判定が
            #   効かないので外す。
            if new is not None and _is_polite_style(metadata):
                whole = [t for t in new if not t.get(UNSAID_KEY)]
                fix_casual_reactions(whole)
                leak = next((str(t.get("utterance", "")) for t in whole
                             if _CASUAL_LEAK_RE.search(str(t.get("utterance", "")))), None)
                if leak:
                    new, why = None, f"敬語指定に砕けた語（{leak}）"
            if new is None:
                why_all.append(f"{kind}#{att + 1}: {why}")
                continue
            turns[a:b + 1] = new
            done.append(kind)
            ok = True
            break
        if not ok:
            failed.append(kind if cands else f"{kind}(置き場所なし)")
    return done, failed, why_all


# ── 読みの注記（2026-09-26、ユーザー指摘「上司の方」を fish が「ほう」と読む）──────────
# 「方（かた／ほう）」のように文脈で読みが分かれる語は、辞書（pyopenjtalk）でも規則でも決められない
# （pyopenjtalk は「上司の方」「担当の方」も含めて全部「ほう」。規則では「佐藤様という方」「相手の方」
# 「忙しい方」を取り違える）。意味の分かる LLM に、対象の語を含む発話だけ見せて読みを答えさせる。
# 発話の本文は変えず、ターンの "yomi" に [{"phrase", "surface", "reading"}] を持たせる。
# fish に渡す直前（tts_reading.apply_yomi）と読みの検査（reading_check）だけがこれを使う。
_YOMI_TARGETS = re.compile(r"方|何|一日|明日|今日|他|上手|下手|一時|最中|市場|人気|辛い|空い|開い|十分|大分")
_YOMI_KANA = re.compile(r"^[ぁ-ゖー]+$")
_YOMI_NO_COMPOUND = {"方", "他"}


def _yomi_user_message(turns: list, metadata: dict, idx: list) -> str:
    staff = staff_names(metadata)
    scene = metadata.get("setting") or metadata.get("scenario") or ""
    lines = [f"場面: {scene}", "", "対話（前後の流れを見るため全文。読みを答えるのは * の付いた発話だけ）:"]
    for i, t in enumerate(turns):
        sp = str(t.get("speaker", ""))
        # 接客は役割（担当者／客）で、ほかのシナリオは話者名で示す
        role = ("担当者" if sp in staff else "客") if staff else sp
        lines.append(f"{'*' if i in idx else ' '}{i} {role}: {t.get('utterance', '')}")
    return "\n".join(lines)


def validate_yomi(turns: list, items) -> tuple[dict, list]:
    """LLM の注記を検証して {ターン番号: [注記]} にする（捨てた理由も返す）。"""
    out: dict = {}
    rej = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        i, ph, sf, rd = it.get("i"), str(it.get("phrase") or ""), str(it.get("surface") or ""), \
            str(it.get("reading") or "")
        if not isinstance(i, int) or not (0 <= i < len(turns)):
            rej.append(f"番号不正 {i}")
            continue
        u = str(turns[i].get("utterance", ""))
        if not sf or sf not in ph or ph not in u:
            rej.append(f"{i}: 語句が本文に無い（{ph}/{sf}）")
            continue
        if not _YOMI_KANA.match(rd) or len(rd) > 3 * len(sf) + 2:
            rej.append(f"{i}: 読みがかなでない（{rd}）")
            continue
        okuri = re.search(r"[ぁ-ゖ]+$", sf)             # 送りがな（「開い」の「い」）は読みの末尾にもある
        if okuri and not rd.endswith(okuri.group(0)):
            rej.append(f"{i}: 送りがなが読みと合わない（{sf}→{rd}）")
            continue
        # 「方」「他」が前後の漢字と熟語になっている（方向・方法・両方・夕方・他社）ものは、
        # 1字だけ開くと「反対ほう向」「夕かた」になる（2026-09-26 試走: 方向 3件）。「何」は
        # 「何度→なん度」「何人→なん人」のように開いても読みが崩れないので対象外。
        k = u.find(ph) + ph.find(sf)                   # 語の本文中の位置（前後の字は本文で見る）
        if sf in _YOMI_NO_COMPOUND and (re.match(r"[一-龥々]", u[k + len(sf):k + len(sf) + 1])
                                        or re.search(r"[一-龥々]$", u[:k])):
            rej.append(f"{i}: 熟語の一部（{ph}）")
            continue
        out.setdefault(i, []).append({"phrase": ph, "surface": sf, "reading": rd})
    return out, rej


async def annotate_readings(client, model: str, turns: list, metadata: dict,
                            pack: "S2PromptPack", cost_tracker: CostTracker,
                            extra_body: dict | None = None,
                            metadata_index=None) -> tuple[int, list]:
    """文脈で読みが分かれる語（方・何・一日・明日…）の読みを LLM に付けさせる。

    (注記した発話数, 捨てた理由) を返す。turns の各ターンに "yomi" を付ける（本文は変えない）。
    """
    if not pack.yomi_prompt:
        return 0, []
    idx = [i for i, t in enumerate(turns) if _YOMI_TARGETS.search(str(t.get("utterance", "")))]
    for t in turns:
        t.pop("yomi", None)
    if not idx:
        return 0, []
    try:
        response = await _chat_create(
            client, model=model,
            messages=[{"role": "system", "content": pack.yomi_prompt},
                      {"role": "user", "content": _yomi_user_message(turns, metadata, set(idx))}],
            temperature=0.0, max_completion_tokens=4096, extra_body=extra_body or {})
        usage = response.usage
        cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)
        data = json.loads(extract_json_text(response.choices[0].message.content or ""),
                          strict=False) or {}
    except Exception as e:                  # 付加的な工程。失敗しても対話は残す
        logger.warning(f"[{metadata_index}] 読みの注記に失敗（注記なしで採用）: {e}")
        return 0, [f"失敗: {e}"]
    got, rej = validate_yomi(turns, data.get("yomi"))
    for i, items in got.items():
        turns[i]["yomi"] = items
    return len(got), rej


def fix_staff_tails(turns: list, metadata: dict) -> int:
    """担当者の「〜です？」を「〜ですか？」に、「〜くださいね。」を「〜ください。」に直す。"""
    staff = staff_names(metadata)
    if not staff:
        return 0
    fixed = 0
    for t in turns:
        if not isinstance(t, dict) or str(t.get("speaker", "")) not in staff:
            continue
        u = str(t.get("utterance", ""))
        new = _STAFF_DESU_Q.sub(r"\1か？", u)
        new = _STAFF_KUDASAI_NE.sub(r"\1", new)
        if new != u:
            t["utterance"] = new
            fixed += 1
    return fixed


# ── 客が案内を受け止めるときの「ね」 ────────────────────────────────────
# 担当者の案内を客が言い直す「受け止め」も、言い切ると確認にならない
# （ユーザー指摘 2026-09-19。担当者側の add_readback_ne と対になる問題）。形は2つ:
#   体言止め   担「明日、10日の午前中には到着する予定でございます。」
#              客「10日の午前中。」          → 「10日の午前中ですね。」
#   「〜んです」担「12歳以下のお子様向けの、キッズプレートがございます。」
#              客「キッズプレートがあるんです。」→ 「〜あるんですね。」
# 後者は特に、聞いた話を受けているのに自分の事情を説明する形になっていて壊れている
# （「〜んですね」が相手の話を受ける形。generate_dialogues の _SFP_NDA_TAIL 参照）。
#
# 受け止めかどうかは**直前の担当者の発話と語が重なるか**で判る。重ならない発話は
# 客が自分の用件・事情を新しく話しているので触らない。
_ECHO_NDESU = re.compile(r"(?:んです|のです)$")
# 自分の事情・希望の説明。「〜んです」でも「ね」を付けてはいけない
# （「部屋を広いのに変えたいんですね」＝自分の希望を相手に確認する形になる）。
_ECHO_SELF = re.compile(r"(?:たいんです|たいのです)$|実は|私|僕|俺|自分|うち")
# 受け止めそのものの語。ここに「ね」を足すと二重になる。
_ECHO_NO_ADD = re.compile(r"(?:なるほど|そうですか|わかりました|分かりました|承知)$")
# 直前の担当者の発話が問い・確認なら、客のターンは「答え」であって受け止めではない
# （「〜でよろしいでしょうか？」→「1回分です。」、「〜ということですね。」→「そうなんです。」）。
_ECHO_PREV_ASK = re.compile(
    r"(?:[？?]|でしょうか。?|ますか。?|ですか。?|ください。?"
    r"|(?:です|ます|ました|でした|ません)ね。?)$")
# 語の重なりを見る単位（漢字・カタカナ・数字の2文字以上）。平仮名は助詞が拾えてしまう。
_ECHO_TOKEN = re.compile(r"[一-龥ァ-ヴー0-9０-９]{2,}")
# 客の口調の判定に使う（体言止めに足すのが「ですね」か「ね」かを決める）。
_ECHO_POLITE = re.compile(
    r"(?:です|ます|ました|ません|でした|ください|ましょう|でしょう)(?:か|ね|よ|よね)?$")
# 受け止めは短い。これを超える文は、語が重なっていても自分の話をしている。
_ECHO_NOUN_MAXLEN = 14
# 口調の判定に要る文の数。接客の客はほぼ敬語（実測 100対話中99）なので、
# これに満たないときは「ですね」側に倒す。
_ECHO_POLITE_MIN = 3
_ECHO_NDESU_MAXLEN = 24


def _echo_polite_speaker(turns: list, name: str, metadata: dict) -> bool:
    """この話者に「ですね」を足してよいか（敬語か）。

    対話全体が敬語指定ならそれに従う。混合・タメ口の対話だけ、その話者自身の
    文末を数えて決める（接客では客が子供のことがある）。
    """
    if _is_polite_style(metadata or {}):
        return True
    tails = [re.sub(r"[。！？]+$", "", s)
             for t in turns if isinstance(t, dict) and t.get("speaker") == name
             for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s]
    if len(tails) < _ECHO_POLITE_MIN:       # 判断材料が足りないときは敬語側に倒す
        return True
    return sum(1 for s in tails if _ECHO_POLITE.search(s)) / len(tails) >= 0.3


def add_customer_echo_ne(turns: list, metadata: dict, enabled: bool = True) -> int:
    """客が担当者の案内を受け止める言い切りに「ね」を足す（足した数を返す）。

    1ターンにつき最初に見つかった1文だけ直す。role が「担当者」の話者がいない
    シナリオでは何もしない（add_readback_ne と同じ）。
    """
    if not enabled:
        return 0
    staff = {s.get("name") for s in (metadata.get("speakers") or [])
             if isinstance(s, dict) and s.get("role") == "担当者" and s.get("name")}
    if not staff:
        return 0
    added = 0
    for i, t in enumerate(turns):
        if i == 0 or not isinstance(t, dict) or str(t.get("speaker", "")) in staff:
            continue
        prev = turns[i - 1]
        if not isinstance(prev, dict) or str(prev.get("speaker", "")) not in staff:
            continue
        pu = str(prev.get("utterance", ""))
        if _ECHO_PREV_ASK.search(pu.rstrip()):
            continue                        # 直前が問い・確認なら、これは「答え」
        sents = [s for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s]
        for j, s in enumerate(sents):
            m = re.search(r"(?P<end>[。！？]*)$", s)
            stem, end = s[:m.start()], m.group("end") or "。"
            if end in ("？", "！") or _ECHO_NO_ADD.search(stem):
                continue
            if not any(w in pu for w in _ECHO_TOKEN.findall(stem)):
                continue                    # 直前の発話と語が重ならなければ受け止めでない
            if (_ECHO_NDESU.search(stem) and not _ECHO_SELF.search(stem)
                    and _core_len_utt(s) <= _ECHO_NDESU_MAXLEN):
                sents[j] = stem + "ね" + end
            elif (_NOUN_TAIL.search(stem)
                  and _core_len_utt(s) <= _ECHO_NOUN_MAXLEN):
                sents[j] = stem + ("ですね" if _echo_polite_speaker(
                    turns, str(t.get("speaker", "")), metadata) else "ね") + end
            else:
                continue
            t["utterance"] = "".join(sents)
            added += 1
            break                           # 1ターンに1箇所だけ
    return added

def _is_answer(turns: list, i: int) -> bool:
    """直前が相手の疑問文なら、このターンは「答え」（相手が知らないことを伝える発話）。"""
    if i <= 0:
        return False
    prev = turns[i - 1]
    if not isinstance(prev, dict) or prev.get("speaker") == turns[i].get("speaker"):
        return False
    return bool(_QUESTION_TAIL.search(str(prev.get("utterance", "")).rstrip()))


def fix_yone_misuse(turns: list, skip_speaker: "str | None" = None) -> int:
    """誤用の「よね」を直す（直した箇所数を返す）。

    付加規則（sfp_candidates）を直しても、モデル自身が書いた「よね」には同じ誤用が
    出る（実測: prod_v4 の16対話で「よね」96箇所のうち27箇所が「〜ます/ました+よね」）。
    形と直前ターンだけで判定できるものに限って、次のどちらかにする。
      ・「ね」に替える … これから自分がすること（「試してみますよね」→「試してみますね」）
      ・落とす         … 自分の経験・状態の報告（「やっていましたよね」→「やっていました」）
    共有体験の確認（「あの件、先週でしたよね」「あのとき忙しかったよね」）は触らない。
    skip_speaker の発話は直さない（傾聴 al の聞き手の共感「それは焦りましたよね」。fix_ne_misuse の注記）。
    """
    fixed = 0
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or (skip_speaker and t.get("speaker") == skip_speaker):
            continue
        u = str(t.get("utterance", ""))
        m = _YONE_TAIL.search(u)
        if not m:
            continue
        stem, end = u[:m.start()], m.group("end") or "。"
        # 「〜んだよね」から「よね」を落とすと「〜んだ。」＝書き言葉の言い切りになる
        # （この工程がいちばん潰したい形）。自分の事情の説明なので「よ」に替える。
        drop = stem + ("よ" if _SFP_DA_TAIL.search(stem) else "") + end
        if _SFP_NO_ADD.search(stem) or _SFP_NO_ADD_TAIL.search(stem):
            new = drop                             # 定型のあいさつ・受諾（「了解ですよね」）
        elif _OTHER_SFP_TAIL.search(stem):
            new = drop                             # 既に終助詞がある（「かしらよね」）
        elif _SFP_NE_ONLY.search(stem):
            new = stem + "ね" + end                # 依頼・勧誘・推量は「ね」だけ
        elif _SFP_NEEDS_DA.search(stem):
            new = stem + "だよね" + end             # 「作れそうよね」→「作れそうだよね」
        elif (_PAST_POLITE_TAIL.search(stem) or _YONE_OWN_STATE.search(stem)
                or _NOUN_TAIL.search(stem)):
            new = drop                             # 自分の経験・状態・体言止め
        elif (_YONE_VOLITION.search(stem)
                or _ADJ_DESU_TAIL.search(stem)   # い形容詞＋です＝直前の説明への評価
                or (_MASU_TAIL.search(stem) and not _YONE_OK_MASU.search(stem))
                or (_PLAIN_VERB_TAIL.search(stem)
                    and not _YONE_OK_PLAIN.search(stem))):
            new = stem + "ね" + end                # これから自分がすること → 「ね」
        elif _is_answer(turns, i):
            new = drop                             # 答えに同意要求は成立しない
        else:
            continue
        if _core_len_utt(new) < 2:
            continue
        t["utterance"] = new
        fixed += 1
    return fixed


def inject_sfp(turns: list, sp: "S2Params", polite: bool = False) -> int:
    """終助詞の無い平叙ターンに終助詞を足す。

    まず「だ」で言い切っているターンを全部埋め（書き言葉に聞こえる最大の原因）、
    そのあと残りを散らして sfp_ratio まで埋める。疑問文・あいさつ・極短ターンは
    触らない。同じ終助詞が続かないよう、直前2ターンで使った語は避ける。
    """
    rng = random.Random(f"sfp:{len(turns)}:{turns[0].get('utterance', '')[:16]}")
    target = int(round(sp.sfp_ratio * len(turns)))
    names = _speaker_names(turns)           # 裸の名前の呼びかけ（「ありがとう、ミキ。」）用

    def has_sfp(i):
        return bool(_SFP_TAIL.search(turns[i].get("utterance", "")))

    def eligible(i):
        u = turns[i].get("utterance", "")
        if has_sfp(i) or u.rstrip().endswith(("？", "?")):
            return False
        stem = u.rstrip().rstrip("。")
        if _ends_with_vocative(stem, names):
            return False
        return _core_len_utt(u) >= 4 and bool(
            sfp_candidates(stem, _is_answer(turns, i), polite))

    added = 0

    def tail_of(i) -> str:
        if 0 <= i < len(turns):
            m = _SFP_TAIL.search(turns[i].get("utterance", ""))
            if m:
                return m.group("sfp")
        return ""

    def fill(i) -> bool:
        nonlocal added
        u = turns[i]["utterance"]
        cands = sfp_candidates(u.rstrip().rstrip("。"), _is_answer(turns, i), polite)
        # 直前2ターンの語尾（モデルが自分で書いたものも含む）と同じ語は避ける
        recent = [t.lstrip("だ") for t in (tail_of(i - 2), tail_of(i - 1)) if t]
        pick = _pick_sfp(cands, recent, rng)
        if not pick:
            return False
        new = append_sfp(u, pick)
        if new == u:
            return False
        turns[i]["utterance"] = new
        added += 1
        return True

    # 第1段: 「だ」止めは全部埋める
    for i in range(len(turns)):
        if eligible(i) and _SFP_DA_TAIL.search(turns[i]["utterance"].rstrip().rstrip("。")):
            fill(i)
    # 第2段: 目安に届くまで、残りを散らして埋める
    rest = [i for i in range(len(turns)) if eligible(i)]
    have = sum(1 for i in range(len(turns)) if has_sfp(i))
    if have < target and rest:
        need = target - have
        stride = max(1, len(rest) // max(1, need))
        for i in rest[::stride] + rest:
            if have >= target:
                break
            if eligible(i) and fill(i):
                have += 1
    return added


def diversify_sfp_runs(turns: list, max_run: int = 2,
                       polite: bool = False, skip_speaker: "str | None" = None) -> int:
    """同じ終助詞が続く区間を崩す（替えた／落とした箇所数を返す）。

    プロンプトには「同じ終助詞を3ターン以上続けない」と書いてあるが守られない
    （実測: 「ね」で終わるターンが8連続する対話があり、全部同じ調子に聞こえる）。
    3つめ以降を替える／落とすが、語気を変えないことを優先する:
      ・敬語 … 「よね」に替えるか落とす。「大事ですね」（共感）を「大事ですよ」（主張）に
               すると意味が変わるので、敬語では「よ」に替えない。
      ・タメ口 … 「よ」「な」に替える（「送るね」→「送るよ」）。落として言い切りに
               するより自然で、終助詞率も保てる。
      ・「〜だね」… 落とすと「〜だ。」＝書き言葉になるので、必ず替える側に回す。
      ・「〜ください」… 落とすと依頼がきつくなるので触らない。
    """
    rng = random.Random(f"run:{len(turns)}:{turns[0].get('utterance', '')[:16]}")
    changed, run_sfp, run_len = 0, None, 0
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            run_sfp, run_len = None, 0
            continue
        u = str(t.get("utterance", ""))
        m = _SFP_TAIL.search(u)
        if not m:
            run_sfp, run_len = None, 0
            continue
        sfp = m.group("sfp")
        if sfp != run_sfp:
            run_sfp, run_len = sfp, 1
            continue
        run_len += 1
        if run_len <= max_run:
            continue
        if skip_speaker is not None and t.get("speaker") == skip_speaker:
            continue                    # 傾聴の聞き手の終助詞は替えない（連続の勘定には入れる）
        stem, end = u[:m.start("sfp")], m.group("end") or "。"
        alts = [c for c in sfp_candidates(stem, _is_answer(turns, i), polite)
                if c != sfp]
        if _POLITE_STEM.search(stem) and not _SFP_DA_TAIL.search(stem):
            alts = [c for c in alts if c.lstrip("だ") == "よね"]   # 共感は共感のまま
        if alts:
            pick = _pick_sfp(alts, [sfp], rng) or alts[0]
            t["utterance"] = stem.rstrip("、") + pick + end
            run_sfp, run_len = _SFP_TAIL.search(t["utterance"]).group("sfp"), 1
            changed += 1
            continue
        # 替えられない語尾。落として良いものだけ落とす（「〜だ。」と「〜ください。」は残す）
        if _SFP_DA_TAIL.search(stem) or re.search(r"(?:ください|下さい|なさい)$", stem):
            continue
        new = stem.rstrip("、") + end
        if _core_len_utt(new) < 2:
            continue
        t["utterance"] = new
        run_sfp, run_len = None, 0
        changed += 1
    return changed


# 「ね」の間引き。落としても語は壊れないが共感・確認の機能が消えるので、一律ではなく
# 「多すぎる」印象の主因から落とす。
# ★2026-08-01 の実測（ユーザー指摘「ねが多すぎます」）。区切り単位に影響されない
#   「発話100文字あたりの個数」で人間コーパス(maai 54セッション)と比べると:
#     ね 人間 0.57 / 合成 1.34（2.4倍）、よ 0.11/0.57、よね 0.13/0.25、な 0.19/0.15
#   内訳の比率（人間のね 54%）は合っており、総量だけが2倍だった。原因は目標値を
#   人間の**ターン**単位の70%に合わせていたこと（人間のターンは IPU が何個も連なる
#   長いもので、こちらの1〜2文のターンとは長さが5〜10倍違う）。
#   なお受理20対話の終助詞406ターンのうち211が付加工程の産物で、モデル自身が書いた
#   分（195ターン＝全ターンの25%）はほぼ人間の密度だった。間引きが一度も効かなかったのは
#   間引ける語が「よね」「な」だけで「ね」が対象外だったため。
_NE_KEEP_SHORT = 10      # 実質10文字以下は「ね」が発話の機能そのもの（「そうですね。」）
# 「ね」を落として良いのは丁寧形の言い切りだけ。20対話への試走で分かった誤り:
#   ・タメ口の終止形は「ね」を外すと書き言葉になる（「共有するね。」→「共有する。」）。
#     これは inject_sfp が埋めている穴そのものなので、外したぶんを作り直してしまう。
#   ・言いさしは「ね」が言いさしの印（「決まらなくてね。」→「決まらなくて。」は素っ気ない）。
#   ・依頼の「ください」は「ね」が緩和（「ご相談くださいね。」→命令に聞こえる）。
_NE_THIN_OK = re.compile(
    r"(?<!ください)(?<!下さい)"
    r"(?:です|ます|ません|ました|ませんでした|でした|ましょう|でしょう)$")

# タメ口の「ね」は落とせない（「共有するね。」→「共有する。」＝書き言葉）ので、
# 替える。上限を超えたぶんだけ「な」または「よ」にする。2026-08-06 の実測で、
# 口調別の「ね」密度は 敬語 0.68 / 混合 0.56 に対し **タメ口 1.19**（人間 0.57）と
# タメ口だけが突出しており、原因は thin_ne が丁寧形しか落とせないことだった。
# 替え先は「な」を優先する（同じ実測で な は 0.11 と人間の 0.19 を下回っていて余裕がある。
# 「よ」は 0.25 で人間 0.11 の2倍なので後回し）。
# 気づき・納得の「ね」（「〜んですね」「そうだったんですね」「そうですよね」の「よね」側）。
# 落とすと相手から聞いた内容を自分の主張として言い切る形になる（thin_ne / thin_sfp の keep_realize）。
_NE_REALIZE = re.compile(r"(?:[んの]で(?:す|した)|(?:^|、)(?:やっぱり、?|やはり、?)?そうです)$")
_NE_SWAP_SKIP = re.compile(r"(?:て|で)$")   # テ形の依頼「連絡してね」→「〜してよ」は催促になる


# 感想・評価の「ね」（ne_keep_eval）。形容詞（〜いです）・様態（〜そうです）・評価の形容動詞・
# 「〜ということです」。「〜ます」（行動の予告）や名詞の説明は対象外（落としても硬くならない）。
_NE_EVAL = re.compile(r"(?:い|そう|ということ|素敵|大変|残念|贅沢|最高|便利|魅力的|大事|大切|"
                      r"不思議|意外|上手|立派|新鮮|素晴らしい|綺麗|きれい|楽しみ|好き|嫌い|苦手|"
                      # 形容詞＋短い名詞（「面白い分野です」「すごい情熱です」「微笑ましいお話です」）
                      r"い(?:お|ご)?[一-龥]{1,3}|"
                      # 相手の仕事・話への評価（「最先端のお仕事です」「いいアイデアです」）
                      r"お仕事|お話|お考え|アイデア|心掛け|心がけ|経験|体験|発想|出会い)です$")


def _casual_ne_swap(stem: str, avoid: set) -> str | None:
    """タメ口の文末「ね」の替え先を返す（替えられないなら None）。

    「な」は「〜だ」「〜た」「〜い」で終わる形にだけ付く（動詞の言い切りに付けると
    「行くな」＝禁止で意味が反転する）。動詞の言い切りは「よ」にする。
    avoid は前後の文の終助詞。同じ語が並ばないようにする。
    """
    if _POLITE_STEM.search(stem) or _NE_SWAP_SKIP.search(stem):
        return None
    if _SFP_NEEDS_DA.search(stem):          # 「そう」「みたい」「はず」等は「だ」が要る
        return None
    cands = []
    if _NA_STEM_OK.search(stem):
        cands.append("な")
    if _PLAIN_VERB_TAIL.search(stem):
        cands.append("よ")
    for c in cands:
        if c not in avoid:
            return c
    return None


def _reaction_turns(turns: list) -> set:
    """相手の発話の直後のターン（その最初の文が相手への受け答えになる）の index の集合。"""
    out = set()
    for i in range(1, len(turns)):
        if (isinstance(turns[i], dict) and isinstance(turns[i - 1], dict)
                and turns[i].get("speaker") != turns[i - 1].get("speaker")):
            out.add(i)
    return out


def thin_ne(turns: list, cap_per_100: float, swap_casual: bool = False,
            protect: "set | frozenset" = frozenset(),
            only: "set | frozenset | None" = None,
            keep_realize: bool = False, keep_eval: bool = False,
            reaction_last: bool = False) -> int:
    """文末の「ね」を、発話100文字あたり cap_per_100 個まで減らす（減らした数を返す）。

    swap_casual を立てると、落とせないタメ口の「ね」を「な」「よ」に**替えて**減らす
    （落とすと「共有する。」＝書き言葉になるため）。既定は False＝従来動作。
    protect は落としてはいけない文の (ターン番号, 文番号) の集合。接客の担当者の復唱
    （readback_sites）に使う。上限の勘定には入れたまま、落とす候補から外すだけ
    （★2026-09-22: 実測で raw 出力の復唱の「ね」39件のうち15件をここで落としていて、
    add_readback_ne が序盤ぶんしか戻せずに「ブライダルコースです。」の言い切りが残った。
    落として戻すのではなく、最初から落とさない）。
    only を渡すと、その (ターン番号, 文番号) の文**だけ**を勘定にも候補にも入れる
    （接客では客の文だけを渡す。担当者の「ね」は復唱・受け止め・予告の機能語で、
    密度の上限で落とすと「ご利用です？」「お調べいたします。」「いらっしゃるんです。」の
    ような壊れた・冷たい形になる。同日のユーザー指摘4件はすべてここが原因だった。
    担当者側の付けすぎはプロンプトと fix_ne_misuse で抑える）。
    keep_realize を立てると、気づき・納得の「〜んですね」「〜のですね」（_NE_REALIZE）を
    落とさない（落とすと「あるんです。」「合算されていたんです。」のように、相手から聞いた
    ばかりの内容を自分の主張として言い切る形になる。2026-09-25 の接客レビューで客の終助詞の
    指摘5件はすべてここか thin_sfp が原因。2026-09-26 に全シナリオで立てた。汎用・初対面の
    各1万対話でも「お子さんがいらっしゃるんです。」型が140〜380件あった）。

    単位はターン比ではなく文字数あたりにする。ターンの区切り方（1ターンに何文を
    書くか）に左右されず、人間コーパスと同じ物差しで測れるため。ターン末だけでなく
    **発話の中の文末も対象**（「〜ですね。でも〜ですね。」の前半も数える）。

    落とす順は「隣の文も『ね』で終わっている箇所」→「長い文」。連続が多すぎる印象の
    主因で、長い情報提供の文では「ね」の機能が薄いため。
    """
    if cap_per_100 <= 0:
        return 0
    # (ターン番号, 文番号, 文) に分解。文の区切りは句点・感嘆符・疑問符。
    sents: list[list[str]] = [
        [s for s in re.split(r"(?<=[。！？])", str(t.get("utterance", ""))) if s]
        for t in turns
    ]
    nchar = sum(_core_len_utt(s) for i, ss in enumerate(sents) for j, s in enumerate(ss)
                if only is None or (i, j) in only)
    cap = round(cap_per_100 * nchar / 100)   # 切り捨てだと短い対話で厳しすぎる

    def ne_end(s: str) -> "re.Match | None":
        m = _SFP_TAIL.search(s)
        return m if m and m.group("sfp") == "ね" else None

    flat = [(i, j) for i, ss in enumerate(sents) for j in range(len(ss))
            if only is None or (i, j) in only]
    idx = [(i, j) for i, j in flat if ne_end(sents[i][j])]
    hit, pos = set(idx), {p: k for k, p in enumerate(flat)}
    react = _reaction_turns(turns) if reaction_last else set()
    order = sorted(idx, key=lambda p: (
        (p[1] == 0 and p[0] in react),          # 受け答えの最初の文は最後に回す
        -sum(1 for k in (pos[p] - 1, pos[p] + 1)
             if 0 <= k < len(flat) and flat[k] in hit),
        -_core_len_utt(sents[p[0]][p[1]])))
    def sfp_at(k: int) -> str:
        """flat 上の位置 k の文末終助詞（無ければ空文字）。"""
        if not (0 <= k < len(flat)):
            return ""
        m = _SFP_TAIL.search(sents[flat[k][0]][flat[k][1]])
        return m.group("sfp") if m else ""

    removed = 0
    for i, j in order:
        if len(idx) - removed <= cap:
            break
        if (i, j) in protect:
            continue
        s = sents[i][j]
        m = ne_end(s)
        stem = s[:m.start("sfp")].rstrip("、")
        if _core_len_utt(s) <= _NE_KEEP_SHORT:
            continue
        if keep_realize and _NE_REALIZE.search(stem):
            continue
        if keep_eval and _NE_EVAL.search(stem):
            continue
        if _NE_THIN_OK.search(stem):                    # 丁寧形は落とす
            sents[i][j] = stem + m.group("end")
            removed += 1
        elif swap_casual:                               # タメ口は替える
            k = pos[(i, j)]
            w = _casual_ne_swap(stem, {sfp_at(k - 1), sfp_at(k + 1)})
            if w:
                sents[i][j] = stem + w + m.group("end")
                removed += 1
    if removed:
        for i, ss in enumerate(sents):
            turns[i]["utterance"] = "".join(ss)
    return removed


def thin_sfp(turns: list, cap_ratio: float, keep_realize: bool = False,
             keep_eval: bool = False, reaction_last: bool = False,
             skip_speaker: "str | None" = None) -> int:
    """終助詞で終わるターンの比率が cap_ratio を超えたら「よね」「な」を落とす。

    keep_realize を立てると、気づき・同意の「〜んですよね」「そうですよね」は落とさない
    （「やっぱり、そうです。」になる。thin_ne の keep_realize と同じ理由）。
    skip_speaker の発話は数えも落としもしない（傾聴の聞き手。共感・受けの終助詞は応答の機能）。
    """
    cap = int(cap_ratio * len(turns))

    def tailed():
        return [i for i, t in enumerate(turns)
                if _SFP_TAIL.search(t.get("utterance", ""))
                and (skip_speaker is None or t.get("speaker") != skip_speaker)]

    removed = 0
    react = _reaction_turns(turns) if reaction_last else set()
    # 受け答えのターンで、文末の「よね」「な」が最初の文にある（＝1文だけの受け答え）ものは最後に回す
    def _key(i):
        u = turns[i]["utterance"]
        one = len([x for x in re.split(r"(?<=[。！？])", u) if x]) == 1
        return (i in react and one, _core_len_utt(u))
    for i in sorted(tailed(), key=_key):
        if len(tailed()) <= cap:
            break
        t = turns[i]["utterance"]
        t2 = strip_tail_sfp(t)
        if t2 != t and (keep_realize or keep_eval):
            last = [x for x in re.split(r"(?<=[。！？])", t2) if x][-1]
            stem = re.sub(r"[。！？]+$", "", last)
            if (keep_realize and _NE_REALIZE.search(stem)) or (keep_eval and _NE_EVAL.search(stem)):
                continue
        if t2 != t:
            turns[i]["utterance"] = t2
            removed += 1
    return removed


# ── 文の切れ目を読点から句点へ ──────────────────────────────────────────
# 32B は文が終わっているのに読点でつなぐ（実測: prod_v3 の22対話841ターンで、
# 「です/ます+、」が78箇所、冒頭リアクション+「、」が63箇所）。音声にすると
# 「来週月曜に再査定に行きます、結果は火曜までに出せますよね。」が一息で流れてしまう。
# 句点なら fish が明確に間を置くので、ここを替えるのが「すらすら読みすぎ」への直接の対処。
#
# 位置を推測して読点を「足す」のは割に合わない（既存の break_long_runs の注記のとおり）が、
# 既にモデルが打った読点を「替える」だけなら、文字を増やさないので語を壊しようがない。
# それでも誤ると意味が変わる（呼びかけを切ると別人への発話に聞こえる）ので、
# 前後の両方に条件を課し、確信が持てるものだけ替える。

# 前提: 読点の直前が「文が終わった形」で終わっている。常体は終助詞付きの形だけを見る
# （「読んだ、」の連用形・名詞止めと区別できないため）。
_SENT_END_FORM = re.compile(
    r"(?:ませんでした|ましょう|ません|ました|ます|でした|です|ください|ございます|"
    r"んだよ|んだね|だよね|だよ|だね|だな|かな|"
    r"どういたしまして|こちらこそ|もちろん|了解|承知しました|失礼しました|"
    r"お疲れ様でした|お疲れさまでした)"
    r"(?:よね|ね|よ|な|わ|さ)?$")
# 発話の頭に置かれた冒頭リアクション語。「なるほど、でも〜」→「なるほど。でも〜」。
_REACT_PERIOD_HEAD = re.compile(
    r"(?:なるほど|そうそう|そうなんだ|そっか|たしかに|いいね|そう|あー)$")
# 読点の直後がこれで始まるなら、そこは新しい文の頭とみなしてよい（許可リスト）。
# ★助詞の先頭1文字で弾く方式（_LEADING_PARTICLE）は使わない。「でも」「よろしく」を
#   巻き添えにして、いちばん替えたい「なるほど、でも〜」が残ってしまう。
_NEW_SENT_HEAD = re.compile(
    rf"^(?:{_FILLER_WORDS}|"
    r"でも|ですが|ただし|ただ|あと|それと|それに|そして|また|なので|だから|"
    r"じゃあ|じゃ|では|さて|ところで|ちなみに|まずは|まず|次に|とりあえず|"
    r"もし|ぜひ|実は|一応|とにかく|やはり|よろしく|ありがとう|お願い|"
    r"これ|それ|あれ|この|その|あの|ここ|そこ|こちら|そちら|私|僕|俺|うち|"
    r"[一-龥ァ-ヴ0-9０-９A-Za-zＡ-Ｚａ-ｚ])")
# 替えたあとの後半がこれ未満の実質字数なら替えない（「そう、それ！」の断片化を防ぐ）
_PERIOD_MIN_TAIL = 5
# ★それだけで受け答えになる語（_SENT_END_FORM の後半）。これは文頭に単独で置かれた
# ときだけ文の終わりとみなす（2026-09-23 のレビューで38件の誤分割）:
#   「配送サイクルの変更につきましては、もちろん、可能でございます。」
#     → 「〜もちろん。可能でございます。」 ＝ 文中の副詞を文末扱いしていた
#   「こちらこそ、ありがとうございます。」→「こちらこそ。ありがとうございます。」
#     ＝ 一続きの定型句を割っていた（後ろがお礼・挨拶なら割らない）
_STANDALONE_REPLY = re.compile(
    r"(?:どういたしまして|こちらこそ|もちろん|了解|承知しました|失礼しました|"
    r"お疲れ様でした|お疲れさまでした)$")
_REPLY_CONTINUES = re.compile(r"^(?:ありがとう|よろしく|お願い|失礼|お世話)")
# 「〜にございます、マイページというボタンを」の「ございます」は連体修飾（後ろの名詞に
# かかる）なので文の終わりではない。「〇〇という」「〇〇の」が続く形で見分ける。
_RELATIVE_TAIL = re.compile(r"^[^、。？！]{1,15}(?:という|といった)")


# ── 1語のオウム返しの切れ端（echo_fix）────────────────────────────────
# Gemma は相手の語を「設計を。」「デスクワーク。」と句点で言い切って受ける（初対面20対話で全発話の
# 13% が「X。」で始まり、うち助詞で切れる形7件・それだけのターン9件）。助詞で切れた形と、
# それだけで1ターンになっている形は「Xですか。」（タメ口の話者は「Xか。」）にする。
# 「スタジアム。いいですね。」のように感想が続く形は触らない（数はプロンプトで抑える）。
# ★単独のものは「Xですか？」（相手の説明を待つ問い）、後ろに文が続くものは「Xですか。」。
_ECHO_FRAG = re.compile(r"^(?P<w>[^、。？！?!\s]{1,12}?)(?P<p>[をにがでとはも])?。(?P<rest>.*)$")
_ECHO_NOUN_END = re.compile(r"[一-龥ァ-ヴーＡ-Ｚａ-ｚA-Za-z0-9０-９]$")
_ECHO_SKIP = re.compile(r"^(?:はい|いえ|いや|そう|なるほど|たしかに|確かに|本当|ほんと|ぜひ|ありがとう|"
                        r"こちらこそ|あ|え|うん|あー|へえ|へー|ええ|いいですね|分かります|わかります|"
                        r"こんにちは|はじめまして|初めまして|さすが|お疲れ様)")


def fix_echo_fragments(turns: list) -> int:
    """1語のオウム返しの切れ端を「Xですか。」に直す（直した数を返す）。"""
    fixed = 0
    for i in range(1, len(turns)):
        t = turns[i]
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        m = _ECHO_FRAG.match(u)
        if not m:
            continue
        w, p, rest = m.group("w"), m.group("p"), m.group("rest")
        if p is None and rest.strip():
            continue                       # 「スタジアム。いいですね。」は残す
        prev = (str(turns[i - 1].get("utterance", ""))
                if isinstance(turns[i - 1], dict)
                and turns[i - 1].get("speaker") != t.get("speaker") else "")   # 沈黙を挟む自分の続きは相手ではない
        if (_core_len_utt(w) < 2 or w not in prev or _ECHO_SKIP.match(w)
                or not _ECHO_NOUN_END.search(w)):
            continue
        own = [str(x.get("utterance", "")) for x in turns
               if isinstance(x, dict) and x.get("speaker") == t.get("speaker")]
        polite = sum(1 for x in own if _POLITE_RE_S2.search(x)) >= len(own) * 0.5
        # 単独の聞き返しは相手の説明を待つ問い（「？」）、後ろに文が続くなら受け止め（「。」）
        end = "？" if not rest.strip() else "。"
        # 「も」は落とさない（「何枚も。」→「何枚ですか。」では意味が変わる。2026-10-24 聴取の指摘）。
        # 格助詞（を・に・が・で・と・は）は「スタジアムに。」→「スタジアムですか。」のように落としてよい。
        if p == "も":
            w += "も"
        t["utterance"] = w + ("ですか" if polite else "か") + end + rest
        fixed += 1
    return fixed


_POLITE_RE_S2 = re.compile(r"(?:です|ます|でした|ました|ません)(?:か|ね|よ)?[。？！]")


# ── 冒頭のオウム返しの数と「？／。」（echo_head_cap）──────────────────────
# 相手の語を受けた冒頭の「○○ですか？」。後ろに自分の感想・質問が続くなら、答えを待たない受け止め
# なので「。」にする（「機械設計ですか。それはすごい経験ですね。」）。「え、」「えっ」で始まる驚きと、
# それだけで終わる聞き返し（相手の説明を待つ）は「？」のまま。
_ECHO_HEAD = re.compile(r"^(?P<pre>(?:え|えっ|あ|へえ|へー)、)?(?P<w>[^、。？！?!]{1,14}?)"
                        r"(?P<q>ですか|か)(?P<p>[？?。！])(?P<rest>.*)$")
_ECHO_HEAD_WH = re.compile(r"何|どこ|どう|いつ|誰|どれ|どんな|本当|ほんと|そう|大丈夫|いかが")


def _echo_head(turns: list, i: int) -> "re.Match | None":
    """turns[i] が相手の語を受けた冒頭の「○○ですか」なら、その match。"""
    if i == 0 or not isinstance(turns[i], dict) or not isinstance(turns[i - 1], dict):
        return None
    if turns[i - 1].get("speaker") == turns[i].get("speaker"):
        return None                         # 沈黙を挟む自分の続き（dialogue.silence）
    m = _ECHO_HEAD.match(str(turns[i].get("utterance", "")))
    if not m:
        return None
    w = m.group("w")
    prev = str(turns[i - 1].get("utterance", ""))
    if _ECHO_HEAD_WH.search(w) or not any(w[j:j + 2] in prev for j in range(len(w) - 1)):
        return None
    return m


def limit_echo_heads(turns: list, cap: int) -> int:
    """冒頭のオウム返しの「？／。」を整え、cap を超えた分の冒頭を外す（変えた数を返す）。"""
    if cap <= 0:
        return 0
    changed = 0
    heads = []
    for i in range(len(turns)):
        m = _echo_head(turns, i)
        if not m:
            continue
        rest = m.group("rest").strip()
        surprise = m.group("pre") in ("え、", "えっ、")
        if rest and m.group("p") in "？?" and not surprise:
            u = turns[i]["utterance"]
            turns[i]["utterance"] = u[:m.start("p")] + "。" + u[m.end("p"):]
            changed += 1
        heads.append((i, rest))
    excess = len(heads) - cap
    # 後ろに文が続くものから、後ろの方を優先して冒頭を外す（単独の聞き返しは外すと空になる）
    for i, rest in reversed(heads):
        if excess <= 0:
            break
        if _core_len_utt(rest) < 6:
            continue
        m = _echo_head(turns, i)
        pre = m.group("pre") if m and m.group("pre") in ("あ、",) else ""
        turns[i]["utterance"] = pre + rest
        excess -= 1
        changed += 1
    return changed


# ── 受け止めの定型語の上限（ack_head_cap / ack_phrase_cap）─────────────
# Gemma は相手の話を「なるほど。」「いいですね。」「そうですね。」で受けてから中身に入る。
# 1つずつは自然だが、1対話平均7.3回（全ターンの21%）出ると型として聞こえ、学習した
# モデルもそれを真似る。人間コーパス（maai-f2f-jp-2025）の同じ形は全ターンの7.0%で、
# しかもその6割は「あー」（1000ターンあたり 人間47.5 対 合成11.6）。
# 総数の上限を超えた分は冒頭を外して中身だけ残し（＝短いターンが増える副産物もある）、
# 同じ語の繰り返しは人間に多い感動詞へ替える。語の置き換えは id で決まるので再現する。
# ★一族には感動詞（あー・ああ・すごい・えっ）も入れる。「なるほど」を禁じると
#   そのぶん「あー、」に寄るだけだった（1回目の作り直しの実測: 11.6→140.0 per 1000
#   ターン。人間 47.5）。同じ語の上限は一族のどの語にもかける。
_ACK_HEAD = re.compile(
    r"^(?P<w>なるほど|たしかに|確かに|いいですね|いいね|素敵ですね|素敵です"
    r"|わかります|分かります|そうですね|そうなんですね|そうなんですか"
    r"|それはいいですね|それはすごいですね|それは素敵ですね"
    r"|あー|ああ|すごい|えっ|へえ|へー)"
    r"(?P<p>[、。])(?P<rest>.*)$")
# 置き換え先。人間の実測（あー 47.5／すごい 23.2／えっ 3.7 per 1000ターン）に合わせ、
# 敬語・タメ口のどちらでも使える語だけにする。その対話でまだ使っていない語を選ぶ。
_ACK_SWAP = ("あー", "すごい", "ああ", "えっ")


def rebalance_ack_heads(turns: list, cap: int, phrase_cap: int,
                        seed: str = "") -> int:
    """受け止めの定型語で始まるターンを cap まで減らし、同じ語の重複を感動詞へ替える。

    cap を超えた分は、後ろに文が続くものから（後ろのターンを優先して）冒頭の語を外す。
    冒頭を外せない（その語だけのターン）ものと、同じ語が phrase_cap を超えたものは
    _ACK_SWAP の語に替える。変えたターン数を返す。
    """
    if cap <= 0 and phrase_cap <= 0:
        return 0
    sites = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        m = _ACK_HEAD.match(str(t.get("utterance", "")))
        if m:
            sites.append((i, m))
    changed = 0
    dropped = set()
    if cap > 0:
        excess = len(sites) - cap
        for i, m in reversed(sites):
            if excess <= 0:
                break
            rest = m.group("rest").strip()
            if _core_len_utt(rest) < 6:
                continue                       # 外すと空になる・短すぎるものは残す
            turns[i]["utterance"] = rest
            dropped.add(i)
            excess -= 1
            changed += 1
    if phrase_cap > 0:
        used = {}
        k = 0
        for i, m in sites:
            if i in dropped:
                continue
            w = m.group("w")
            used[w] = used.get(w, 0) + 1
            if used[w] <= phrase_cap:
                continue
            # その対話でいちばん使っていない語を選ぶ（同数なら id で決める）。
            # 1語に寄せると、その語が今度は人間の3倍出る。
            off = hash_index(f"{seed}:{i}")
            swap = min(_ACK_SWAP,
                       key=lambda w, o=off: (used.get(w, 0),
                                             (_ACK_SWAP.index(w) + o) % len(_ACK_SWAP)))
            used[swap] = used.get(swap, 0) + 1
            k += 1
            rest = m.group("rest").strip()
            # 中身が続くときは人間と同じ「あー、」の形（読点）にする
            turns[i]["utterance"] = (f"{swap}、{rest}" if rest else f"{swap}。")
            changed += 1
    return changed


# ── 話題転換の前置きの上限（shift_mark_cap）／口癖の言い換え（stock_word_swap）──
# 文頭の「（あ、）そういえば、」「ところで、」。文の途中（「そういえばそうですね」）は数えない。
_SHIFT_MARK = re.compile(r"^(?:あ、|あっ、)?(?:そういえば|ところで)(?:、|(?=[^、。？]))")


def cap_shift_marks(turns: list, cap: int) -> int:
    """文頭の「そういえば」「ところで」を1対話 cap 個までにし、超えた分の前置きを外す。

    前から数えて cap 個は残す。外すと中身が実質6字未満になる文は残す。外した数を返す。
    """
    if cap <= 0:
        return 0
    seen = changed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        sents = re.split(r"(?<=[。？])", str(t.get("utterance", "")))
        out = []
        for s in sents:
            m = _SHIFT_MARK.match(s)
            if m:
                seen += 1
                rest = s[m.end():]
                if seen > cap and _core_len_utt(rest) >= 6:
                    s = rest
                    changed += 1
            out.append(s)
        t["utterance"] = "".join(out)
    return changed


# 「心地いい・心地よい」系 → 「気持ちいい・気持ちよい」系。「居心地」は対象外。
_STOCK_SWAPS = ((re.compile(r"(?<!居)心地(?:いい|良い|よい)"), "気持ちいい"),
                (re.compile(r"(?<!居)心地(?:よ|良)"), "気持ちよ"))


def swap_stock_words(turns: list) -> int:
    """Gemma の口癖の語を人間の話し言葉に多い語へ替える。替えたターン数を返す。"""
    changed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = old = str(t.get("utterance", ""))
        for rx, rep in _STOCK_SWAPS:
            u = rx.sub(rep, u)
        if u != old:
            t["utterance"] = u
            changed += 1
    return changed


def hash_index(key: str) -> int:
    """文字列から再現する非負整数（語の置き換えの選択に使う）。"""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def turn_shape_ratios(turns: list) -> tuple[float, float]:
    """(実質15字以下のターンの割合, 「？」を含むターンの割合)。

    人間コーパスの先頭33ターンは 15字以下が 62.8%・疑問が 7.8%。
    """
    n = max(len(turns), 1)
    short = sum(1 for t in turns
                if _core_len_utt(str(t.get("utterance", ""))) <= 15)
    q = sum(1 for t in turns if "？" in str(t.get("utterance", "")))
    return short / n, q / n


# ── オウム返し全体の上限（echo_total_cap）──────────────────────────────
# Gemma は相手の語をくり返して受ける（「4時半。それはすごい早起きです。」「ドキュメンタリーですか？」）。
# 1つずつは自然でも、同じ人が「朝型。」「4時半。」「ヨガ。」「読書。」と続けると単調になる。
_ECHO_LEAD = re.compile(r"^(?P<w>[^、。？！?!]{1,12})。(?P<rest>.+)$")
_ECHO_REPLACE_POLITE = ("そうなんですか？", "へえ、そうなんですね。", "どんな感じなんですか？")
_ECHO_REPLACE_CASUAL = ("そうなんだ？", "へえ、そうなんだ。", "どんな感じなの？")


def _echo_sites(turns: list) -> list:
    """相手の語をくり返して受ける発話の (index, 種類, 続き)。種類は "lead"（X。続き）/
    "head"（Xですか。続き）/ "alone"（Xですか？ だけ）。"""
    out = []
    for i in range(1, len(turns)):
        if not isinstance(turns[i], dict) or not isinstance(turns[i - 1], dict):
            continue
        m = _echo_head(turns, i)
        if m:
            rest = m.group("rest").strip()
            out.append((i, "head" if rest else "alone", rest))
            continue
        if turns[i - 1].get("speaker") == turns[i].get("speaker"):
            continue                        # 沈黙を挟む自分の続き（dialogue.silence）
        u = str(turns[i].get("utterance", ""))
        prev = str(turns[i - 1].get("utterance", ""))
        m = _ECHO_LEAD.match(u)
        if not m:
            continue
        w = m.group("w")
        if (_ECHO_SKIP.match(w) or _ECHO_HEAD_WH.search(w)
                or not any(w[j:j + 2] in prev for j in range(len(w) - 1))):
            continue
        out.append((i, "lead", m.group("rest").strip()))
    return out


def limit_echoes(turns: list, cap: int, replace_once: bool = False) -> int:
    """オウム返し全体を cap 回までにする（変えた数を返す）。最初の cap 回は残す。

    replace_once: 単独の聞き返しを置き換える語を、1対話でそれぞれ1回までにする。使い切ったら
    聞き返しはそのまま残す（★2026-10-29 ft_v2 試走20対話で「そうなんですか？」11回・
    「どんな感じなんですか？」2回。同じ人が1対話に2〜3回言い、前後とかみ合わないものもあった）。
    """
    if cap <= 0:
        return 0
    sites = _echo_sites(turns)
    # 残すのは最初の cap 回。ただし同じ人の続けての発話（間に相手の1ターンだけ）が両方オウム返しに
    # なるときは2回目以降も外す（「ダンサー。」「練習。」「音楽。」と続けて受ける単調さ。2026-10-04）。
    kept, excess = set(), []
    for site in sites:
        i = site[0]
        spk = turns[i].get("speaker")
        prev_own = next((j for j in range(i - 1, -1, -1)
                         if isinstance(turns[j], dict) and turns[j].get("speaker") == spk), None)
        if len(kept) < cap and prev_own not in kept:
            kept.add(i)
        else:
            excess.append(site)
    changed = 0
    used: dict = {}
    for i, kind, rest in excess:
        t = turns[i]
        if kind in ("lead", "head"):
            if _core_len_utt(rest) < 4:
                continue
            t["utterance"] = rest
        else:
            own = [str(x.get("utterance", "")) for x in turns
                   if isinstance(x, dict) and x.get("speaker") == t.get("speaker")]
            polite = sum(1 for x in own if _POLITE_RE_S2.search(x)) >= len(own) * 0.5
            pool = _ECHO_REPLACE_POLITE if polite else _ECHO_REPLACE_CASUAL
            # 次の発話が「はい」「ええ」「うん」で始まる（＝もとの「○○ですか？」への答え）なら、
            # 何かを尋ねる「どんな感じなんですか？」は選ばない（「はい。忙しい朝は〜」とかみ合わない）。
            # ★2026-10-24: 次が自分の続き（沈黙を挟む。dialogue.silence）や問い返しのときも選ばない
            #   （「どんな感じなんですか？」→ 3.2秒 → 同じ人「今度、絶対に行ってみます。」と、問いが
            #   宙に浮く。005 でも次が「？」で終わる問いが112件中5件あった）。
            nt = turns[i + 1] if i + 1 < len(turns) and isinstance(turns[i + 1], dict) else {}
            nxt = str(nt.get("utterance", ""))
            if (re.match(r"(?:はい|ええ|うん|そう)", nxt) or nxt.rstrip().endswith("？")
                    or nt.get("speaker") == t.get("speaker")):
                pool = tuple(x for x in pool if "どんな" not in x)
            if replace_once:
                # 尋ねる形（「どんな感じなんですか？」）は前後とかみ合わないことが多いので使わない
                said = {str(x.get("utterance", "")) for x in turns if isinstance(x, dict)}
                pool = tuple(x for x in pool if x not in said and "どんな" not in x)
                if not pool:
                    continue
            # 候補の使い始めを発話ごとに散らす（毎回「そうなんですか？」から使うと、それが
            # 新しい決まり文句になる。試走100対話で置き換え27件中23件が同じ語だった）。
            k = used.get(t.get("speaker"), 0) + zlib.crc32(str(t.get("utterance", "")).encode())
            t["utterance"] = pool[k % len(pool)]
            used[t.get("speaker")] = used.get(t.get("speaker"), 0) + 1
        changed += 1
    return changed


# ── 主話者のおうむ返しの聞き返しを畳む（main_echo_cap）──────────────────
# ★2026-10-29 ft_v2 再評価: 主話者の問いの3/4が相手の語をくり返すだけの「カラオケですか？」で、
#   全ターンの 7.2%（人間 0%・v1 3.5%）。相手の話はそのたびに短く刻まれ、31字以上のターンが
#   6%（人間 29%）しかないインタビュー調の往復になる。学習するのは主話者（ch0）なので、
#   主話者の単独の聞き返しを cap 回まで残し、残りは消して前後の相手の発話を1ターンにつなぐ
#   （相槌は Stage4 が入れる）。次の発話の頭の答え（「はい。」「ええ、」）は外す。
_FOLD_ANSWER_HEAD = re.compile(r"^(?:はい|ええ|うん|そう(?:なん)?です|そうなんだ|そう)[、。！]\s*")
_FOLD_MAX_CORE = 80     # つないだ発話の実質字数の上限（人間の先頭33ターンで 56字以上は 12%）


def _is_main_bare_echo(turns: list, i: int, main: str) -> bool:
    t = turns[i]
    if not isinstance(t, dict) or t.get("speaker") != main:
        return False
    u = str(t.get("utterance", "")).strip()
    if u in _ECHO_REPLACE_POLITE or u in _ECHO_REPLACE_CASUAL:
        return True
    m = _echo_head(turns, i)
    return bool(m) and not m.group("rest").strip() and m.group("p") in "？?"


def fold_main_echoes(turns: list, main: str, cap: int,
                     max_core: int = _FOLD_MAX_CORE) -> tuple[list, int]:
    """主話者の単独の聞き返しを cap 回までにし、超えた分を消して相手の前後の発話をつなぐ。

    (新しい turns, 畳んだ数) を返す。沈黙を挟む箇所（pause_before_sec）には触れない。
    """
    if cap < 0 or not main:
        return turns, 0
    sites = [i for i in range(1, len(turns) - 1) if _is_main_bare_echo(turns, i, main)]
    out = list(turns)
    n = 0
    for i in reversed(sites[cap:]):
        a, e, b = out[i - 1], out[i], out[i + 1]
        if not (isinstance(a, dict) and isinstance(b, dict)) or main in (a.get("speaker"), b.get("speaker")):
            continue
        if a.get("speaker") != b.get("speaker") or PAUSE_KEY in e or PAUSE_KEY in b:
            continue
        if is_rp_turn(a) or is_rp_turn(e) or is_rp_turn(b):
            continue                             # 重なりの場面（rp_v1）は畳まない
        if str(a.get("utterance", "")).rstrip().endswith("？"):
            continue                             # 相手の問いへの聞き返しは畳まない
        rest = _FOLD_ANSWER_HEAD.sub("", str(b.get("utterance", "")).strip())
        joined = str(a.get("utterance", "")).rstrip()
        if rest:
            joined = joined + rest
        if _core_len_utt(joined) > max_core:
            continue
        merged = dict(a)
        merged["utterance"] = joined
        out[i - 1:i + 2] = [merged]
        n += 1
    return out, n


# 主話者の反応だけのターン（「あー。」「そうなんですね。」）も、相手の話の途中にあるものは畳める。
# ★2026-10-29 再評価: 聞き返しを畳んでも 31字以上のターンは 4.8%→4.7%（人間 29%）のまま。相手の話が
#   「話す→主話者の短い反応→続き」と刻まれている。つないで31字以上になる箇所だけを main_react_fold
#   回まで畳む（反応は Stage4 の相槌が受け持つ）。相手の問いへの返事（直前が「？」）は畳まない。
_REACT_ONLY = re.compile(
    r"^(?:え[ー〜]?、)?(?:あー|ああ|へえ|へー|ほう|なるほど|そうなんですね|そうなんだ|そうですね|"
    r"そうですか|いいですね|いいなあ|確かに|たしかに|わかります|分かります|ふふ|あはは|"
    r"すごい(?:ですね)?|本当(?:ですか|ですね)?|ええ|はい|うん)"
    r"(?:、(?:あー|はい|ええ|なるほど|確かに|たしかに))?[。！]?$")


def fold_main_reactions(turns: list, main: str, cap: int, min_core: int = 31,
                        max_core: int = _FOLD_MAX_CORE) -> tuple[list, int]:
    """主話者の反応だけのターンを、つないだ相手の発話が min_core 字以上になる箇所だけ cap 回まで畳む。"""
    if cap <= 0 or not main:
        return turns, 0
    out = list(turns)
    n = 0
    i = len(out) - 2
    while i >= 1 and n < cap:
        a, e, b = out[i - 1], out[i], out[i + 1]
        if (isinstance(a, dict) and isinstance(e, dict) and isinstance(b, dict)
                and e.get("speaker") == main and a.get("speaker") == b.get("speaker") != main
                and _REACT_ONLY.match(str(e.get("utterance", "")).strip())
                and PAUSE_KEY not in e and PAUSE_KEY not in b
                and not (is_rp_turn(a) or is_rp_turn(e) or is_rp_turn(b))
                and not str(a.get("utterance", "")).rstrip().endswith("？")):
            rest = _FOLD_ANSWER_HEAD.sub("", str(b.get("utterance", "")).strip())
            joined = str(a.get("utterance", "")).rstrip() + rest
            if rest and min_core <= _core_len_utt(joined) <= max_core:
                merged = dict(a)
                merged["utterance"] = joined
                out[i - 1:i + 2] = [merged]
                n += 1
                i -= 2
                continue
        i -= 1
    return out, n


# ── 名乗り方（topic_plan.naming）──────────────────────────────────────
# ★2026-10-29 再評価: 最初の2ターンで両者が名乗り合う形が 17/20。指示（topic_plan.NAMING_TEXT）は
#   破られる前提で、名乗りの文を機械で外す。相手がその名前で呼びかけている対話では外さない
#   （名乗っていない名前を知っていることになる）。外すと中身が残らない発話（「ハナキです。」だけ）も外さない。
_NAMING_LOOK = 4         # 名乗りを探す先頭のターン数


def _self_intro_re(name: str) -> "re.Pattern":
    return re.compile(rf"(?:(?:私|わたし|僕|俺|あたし)は)?{re.escape(name)}"
                      rf"(?:です|と申します|といいます|っていいます|だよ|よ|だ)(?:[。！、]|$)")


def apply_naming(turns: list, metadata: dict, label: "str | None") -> int:
    """名乗り方のラベルに合わせて、先頭の名乗りの文を外す（外した数を返す）。"""
    if label not in ("片方だけ", "なし"):
        return 0
    names = [s.get("name") for s in metadata.get("speakers") or []
             if isinstance(s, dict) and s.get("name")]
    found = []                                   # (ターン番号, 名前, match)
    for i, t in enumerate(turns[:_NAMING_LOOK]):
        if not isinstance(t, dict):
            continue
        name = t.get("speaker")
        if name not in names or any(f[1] == name for f in found):
            continue
        m = _self_intro_re(name).search(str(t.get("utterance", "")))
        if m:
            found.append((i, name, m))
    drop = found if label == "なし" else found[1:]
    n = 0
    for i, name, m in drop:
        others = "".join(str(t.get("utterance", "")) for j, t in enumerate(turns)
                         if isinstance(t, dict) and j != i)
        if name in others:
            continue                             # 呼びかけ・言及があるなら名乗りは要る
        u = str(turns[i]["utterance"])
        rest = (u[:m.start()] + u[m.end():]).strip().lstrip("、")
        if _core_len_utt(rest) < 2:
            continue
        turns[i]["utterance"] = rest
        n += 1
    return n


# ── 指摘した発話だけを LLM に書き直させる（flag_rewrite）──────────────────
# ★2026-10-29 ft_v2 再評価で残った型。どれも1文だけのターンなので、文を消す機械の直しは効かない:
#   calm  「静かな時間を楽しむ」語群（静か・落ち着く・没頭・心地・気持ちいい・時間を忘れ・贅沢・癒やし）。
#         両話者で1対話 4.5回（人間ほぼ0）。stock_word_swap で「心地」を「気持ちいい」に替えたら、
#         心地 29.5→0・気持ちいい 1.5→24.5/1000ターンと癖が移っただけだった。
#   eval  主話者が相手の話を評価・称賛・一般論でまとめる（「大切なお仕事ですね」「〜こそ醍醐味」）。
#         15.5/1000ターン（人間 1.1）。ペルソナ v6 の指示では減らなかった。
#   time  時間を口実にした締め（「そろそろ時間が」）。15〜20%（人間 4%）。
# 機械で数えて上限を超えた発話だけを示し、書き直しを機械で検査してから採る（校閲パスと同じ扱い）。
_CALM_RE = re.compile(r"静か|落ち着|没頭|心地|気持ち(?:い|よ)|時間を忘れ|贅沢|癒(?:や|し)")
_EVAL_RE = re.compile(r"(?:大切|大事|贅沢|醍醐味|素敵|素晴らし|貴重|魅力的|やりがい|豊か)"
                      r"(?:な|だ|です|でした|ですね|ですよね|に)|一番(?:です|ですね|ですよ|だ)|"
                      r"ものです|からこそ|証拠|心身ともに|姿勢ですね|伝わってきます")
_TIME_CLOSE_RE = re.compile(r"そろそろ|時間が(?:来|き|な)|こんな時間|時間(?:です|ね|だ)|明日も早い")
_TIME_CLOSE_WINDOW = 6   # 末尾のこのターン数だけを見る
# 「心地」は語群の上限に関係なく毎回直す（Gemma の口癖。人間 0.2/1000ターン・合成 24〜30。
#   語群の上限 2 だと、残る2回がほぼ「心地」になった: 26.7/1000ターン）
_CALM_ALWAYS_RE = re.compile(r"(?<!居)心地")
_REWRITE_RULES = {
    "calm": (_CALM_RE, "「静か」「落ち着く」「没頭」「心地よい」「気持ちいい」「時間を忘れる」「贅沢」"
                       "「癒やし」の語を使わず、いま話している具体的な中身（物・場所・やり方・出来事）に"
                       "触れる言い方にする"),
    "eval": (_EVAL_RE, "相手や相手の話を評価・称賛したり、一般論や教訓にまとめたり、相手の言ったことを"
                       "言い換えて確かめたりせず、相手の話のどこが気になったかを具体的に言うひと言か、"
                       "自分の似た経験のひと言にする"),
    "time": (_TIME_CLOSE_RE, "時間や次の予定を理由にせず、それまでの話の中身を受けたひと言で切り上げる形にする"),
}
_REWRITE_SYSTEM = (
    "あなたは会話台本の編集者です。指定した番号の発話だけを、それぞれの指示に従って書き直してください。\n"
    "- 前後の発話とかみ合い、次の発話がそのまま自然に続くこと。\n"
    "- 話者の口調（敬語かタメ口か）とフィラーの有無を保ち、長さは元と同じくらいか短くする。\n"
    "- 問いかけでなかった発話を問いかけにしない。新しい人名・固有名詞を足さない。\n"
    "- 指定していない発話は出力しない。\n"
    '出力は JSON のみ: {"edits": [{"i": 番号, "text": "書き直した発話"}]}')


def flag_turns(turns: list, metadata: dict, main: str, sp: "S2Params",
               skip: "set | None" = None) -> dict:
    """書き直す発話 {ターン番号: [理由キー, ...]}。skip（沈黙の直前など）は指摘しない。"""
    skip = skip or set()
    flags: dict = {}
    calm = [i for i, t in enumerate(turns)
            if isinstance(t, dict) and _CALM_RE.search(str(t.get("utterance", "")))]
    over = set(calm[sp.calm_word_cap:] if sp.calm_word_cap >= 0 else [])
    for i in calm:
        if i in over or _CALM_ALWAYS_RE.search(str(turns[i].get("utterance", ""))):
            flags.setdefault(i, []).append("calm")
    if main and sp.main_eval_cap >= 0:
        ev = [i for i, t in enumerate(turns) if isinstance(t, dict) and t.get("speaker") == main
              and _EVAL_RE.search(str(t.get("utterance", "")))]
        for i in ev[sp.main_eval_cap:]:
            flags.setdefault(i, []).append("eval")
    plan = topic_plan_of(metadata) or {}
    if plan.get("closing") and sp.time_close_rewrite:
        for i in range(max(0, len(turns) - _TIME_CLOSE_WINDOW), len(turns)):
            if isinstance(turns[i], dict) and _TIME_CLOSE_RE.search(str(turns[i].get("utterance", ""))):
                flags.setdefault(i, []).append("time")
    return {i: r for i, r in flags.items() if i not in skip}


def _rewrite_user_message(turns: list, metadata: dict, flags: dict) -> str:
    lines = [f"口調: {metadata.get('speech_style', '')}", "", "対話:"]
    for i, t in enumerate(turns):
        pause = (f"（{t[PAUSE_KEY]:g}秒の沈黙のあと）" if PAUSE_KEY in t else "")
        lines.append(f"{i} {t.get('speaker', '')}: {pause}"
                     f"{rp_render(t) if is_rp_turn(t) else t.get('utterance', '')}")
    lines += ["", "書き直す発話と指示:"]
    for i, keys in sorted(flags.items()):
        lines.append(f"{i}: " + "。また、".join(_REWRITE_RULES[k][1] for k in keys) + "。")
    return "\n".join(lines)


def validate_rewrites(turns: list, flags: dict, edits) -> tuple[dict, list]:
    """書き直し案のうち採れるもの {i: text} と、捨てた理由。"""
    ok, rejected = {}, []
    for e in edits or []:
        if not isinstance(e, dict):
            continue
        try:
            i = int(e.get("i"))
        except (TypeError, ValueError):
            continue
        text = _PAUSE_NOTE_RE.sub("", str(e.get("text") or "")).strip()
        if i not in flags or not text:
            continue
        old = str(turns[i].get("utterance", ""))
        why = ""
        if text == old:
            why = "変化なし"
        elif any(_REWRITE_RULES[k][0].search(text) for k in ("calm", "eval")) or (
                "time" in flags[i] and _TIME_CLOSE_RE.search(text)):
            why = "指摘の語が残る"
        elif "？" in text and "？" not in old:
            why = "問いかけになった"
        elif _core_len_utt(text) > max(_core_len_utt(old) + 8, 20):
            why = "長くなった"
        elif _POLITE_RE_S2.search(old) and not _POLITE_RE_S2.search(text) and _core_len_utt(text) > 8:
            why = "敬語が落ちた"
        if why:
            rejected.append(f"turn{i} {why}「{text}」")
        else:
            ok[i] = text
    return ok, rejected


async def rewrite_flagged_turns(client, model: str, turns: list, metadata: dict, flags: dict,
                                cost_tracker: CostTracker, extra_body: dict | None = None,
                                metadata_index=None) -> tuple[int, list]:
    """flags の発話だけを書き直す（turns を直接書き換える）。(直した数, 捨てた理由) を返す。"""
    if not flags:
        return 0, []
    try:
        response = await _chat_create(
            client, model=model,
            messages=[{"role": "system", "content": _REWRITE_SYSTEM},
                      {"role": "user", "content": _rewrite_user_message(turns, metadata, flags)}],
            temperature=0.4, max_completion_tokens=4096,
            extra_body=extra_body or {})
        usage = response.usage
        cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)
        data = json.loads(extract_json_text(response.choices[0].message.content or ""),
                          strict=False)
    except Exception as e:                  # 付加的な工程。失敗しても対話は残す
        logger.warning(f"[{metadata_index}] 指摘箇所の書き直し失敗（元のまま採用）: {e}")
        return 0, [f"失敗: {e}"]
    ok, rejected = validate_rewrites(turns, flags, (data or {}).get("edits"))
    for i, text in ok.items():
        logger.info(f"[{metadata_index}] 書き直し turn{i}（{'・'.join(flags[i])}）"
                    f"「{turns[i].get('utterance', '')}」→「{text}」")
        turns[i]["utterance"] = text
    return len(ok), rejected


def to_sentence_periods(turns: list, names=None) -> int:
    """文が終わっている読点を句点に替える（替えた箇所数を返す）。

    names に話者名を渡すと「〜です、ミキさん。」の呼びかけを除外する（実測11箇所）。
    """
    name_rx = None
    alt = "|".join(re.escape(n) for n in
                   sorted({str(n) for n in (names or []) if n}, key=len, reverse=True))
    if alt:
        name_rx = re.compile(rf"^(?:{alt}){_VOCATIVE_HON}")
    changed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        if not u:
            continue
        out, pos = "", 0
        for m in re.finditer("、", u):
            if m.start() < pos:                       # 直前の置換に含まれた読点
                continue
            head, tail = u[:m.start()], u[m.end():]
            if _core_len_utt(tail) < _PERIOD_MIN_TAIL:
                continue
            if not (_SENT_END_FORM.search(head)
                    or _REACT_PERIOD_HEAD.fullmatch(head)):
                continue
            if _STANDALONE_REPLY.search(head):
                sent_head = re.split(r"[。？！]", head)[-1]
                if (not _STANDALONE_REPLY.fullmatch(sent_head)
                        or _REPLY_CONTINUES.match(tail)):
                    continue                          # 文中の「もちろん」／「こちらこそ、ありがとう」
            if _RELATIVE_TAIL.match(tail):
                continue                              # 連体修飾の「ございます、〇〇という」
            if name_rx is not None and name_rx.match(tail):
                continue                              # 呼びかけ
            if not _NEW_SENT_HEAD.match(tail):
                continue
            out += u[pos:m.start()] + "。"
            pos = m.end()
            changed += 1
        if pos:
            t["utterance"] = out + u[pos:]
    return changed


# thin_short_commas が外してよい「短い句」。末尾が助詞のものだけ（主題・格・並立）。
# 接続詞（でも、だから、）・フィラー・リアクション・呼びかけは間として残す。
_THIN_HEAD_END = re.compile(r"(?:は|が|も|で|に|を|と|へ|から|まで|って|には|では|とは|でも|という|っていう)$")
# 「完璧にする、という、感じ」の「、という」も外す（前の句の長さに関わらず）。
_THIN_NEXT_QUOTE = re.compile(r"^(?:という|っていう)")
_THIN_KEEP_HEAD = re.compile(
    rf"^(?:{_FILLER_WORDS}|{_REACT_WORDS}|でも|だから|それで|それでは|では|じゃあ|"
    r"ただ|あと|なので|はい|いえ|ええ|うん|ううん|あ|え|へえ|ほう|実際|正直|例えば|たとえば|"
    # 笑い声。「あはは」は末尾が助詞の「は」に見えるので、入れないと「あははそれは」になる
    # （2026-10-03 output_ft_v1_gemma_inoue_002「戦場。あははそれは相当な忙しさそうです。」）。
    r"あはは|あははは|はは|ははは|ふふ|ふふふ|えへへ|へへ)$")
_THIN_NEXT_FILLER = re.compile(rf"^(?:{_FILLER_WORDS})、")


def thin_short_commas(turns: list, max_seg: int, max_run: int = 24,
                      names=None) -> int:
    """短い句の直後の読点を外す（外した個数を返す）。0 なら何もしない。

    外すのは、直前の句（直前の句読点から読点まで）が実質 max_seg 文字以下で助詞で
    終わるもの（と「、という」の読点）のうち、フィラー・リアクション・接続詞・呼びかけでなく、読点の直後がフィラーで
    なく（「私は、まあ、」の形は残す）、外しても読点なしの区間が max_run 文字を
    超えないものだけ。1つの句で連続して外さないよう、外した後の区間長で判定する。
    """
    if max_seg <= 0:
        return 0
    alt = "|".join(re.escape(n) for n in
                   sorted({str(n) for n in (names or []) if n}, key=len, reverse=True))
    voc = re.compile(rf"(?:{alt}){_VOCATIVE_HON}$") if alt else None
    removed = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        if "、" not in u:
            continue
        pieces = re.split(r"([、。？！?!])", u)
        out, run = "", 0               # run: 現在の読点なし区間の実質長
        i = 0
        while i < len(pieces):
            seg = pieces[i]
            mark = pieces[i + 1] if i + 1 < len(pieces) else ""
            nxt = pieces[i + 2] if i + 2 < len(pieces) else ""
            core = _core_len_utt(seg)
            if mark == "、":
                head = seg.strip()
                nxt_run = _core_len_utt(nxt)
                drop = (0 < core
                        and ((core <= max_seg and _THIN_HEAD_END.search(head))
                             or _THIN_NEXT_QUOTE.match(nxt))
                        and not _THIN_KEEP_HEAD.fullmatch(head)
                        and not (voc and voc.search(head))
                        and not _THIN_NEXT_FILLER.match(nxt + (pieces[i + 3] if i + 3 < len(pieces) else ""))
                        and run + core + nxt_run <= max_run)
                if drop:
                    out += seg
                    run += core
                    removed += 1
                else:
                    out += seg + mark
                    run = 0
            else:
                out += seg + mark
                run = 0
            i += 2
        if out != u:
            t["utterance"] = out
    return removed


# ★計測用。_FILLER_ANY / _HEAD_REACT は区切りが「、」限定なので、to_sentence_periods
# のあとでは「なるほど。でも〜」を数え落とす（実測でリアクション率が 9.0%→0.6% に化けた）。
# 計測だけは句点でも数える形にする（差し込み・間引き側は句点化の前に走るので元のままでよい）。
_FILLER_ANY_MARK = re.compile(rf"(?:^|(?<=、)|(?<=。))(?:{_FILLER_WORDS})[、。]")
_HEAD_REACT_MARK = re.compile(rf"^(?:{_REACT_WORDS})[、。]")


def enriched_ratios(turns: list) -> tuple[float, float, float, float]:
    """(フィラー含有ターン率, 文末終助詞ターン率, 「間」per実質100文字, 冒頭リアクション率)。

    3つめは読点だけでなく**文中の句点・疑問符も数える**。to_sentence_periods は
    「です、」→「です。」と印を置き換えるだけで間の数を変えないのに、読点だけ数えると
    密度が落ちて comma_floor を誤って割る（実測 6.43→5.75、総数では 6.79 のまま）。
    どちらも音声では間になるので、合算したほうが指標として素直でもある。
    """
    n = max(len(turns), 1)
    nf = sum(1 for t in turns if _FILLER_ANY_MARK.search(t.get("utterance", "")))
    ns = sum(1 for t in turns if _SFP_TAIL.search(t.get("utterance", "")))
    nr = sum(1 for t in turns if _HEAD_REACT_MARK.match(t.get("utterance", "")))
    nchar = max(sum(_core_len_utt(t.get("utterance", "")) for t in turns), 1)
    # 文末の1個は「間」ではなくターンの終わりなので、ターンごとに1個だけ差し引く
    npause = sum(u.count("、") + max(0, u.count("。") + u.count("？") - 1)
                 for u in (str(t.get("utterance", "")) for t in turns))
    return nf / n, ns / n, npause / nchar * 100, nr / n


# 伏せ字・プレースホルダに使われがちな記号（sanitize で消えて穴あき文になる）
_PLACEHOLDER_RE = re.compile(
    r"[○◯〇●△▲□■×☆★◎]"
    # ★伏せ字を読み上げた形（「えみー、まるまる、あっと、じーめいる」）。記号を禁じたら
    #   モデルが読みで書くようになった（2026-09-24 のレビューで2対話）。
    r"|(?:^|(?<=[、「\s\"]))(?:まるまる|ばつばつ|マルマル)(?=[、。」？\s\"])")

# 締めの挨拶。これが出たら会話は終わっているはずなので、後ろに続くターンは
# ターン数ノルマを埋めるための水増しとみなす。
_CLOSING_RE = re.compile(
    r"(よろしくお願い|ありがとうございました|失礼します|失礼いたします|"
    r"お疲れ様でした|お疲れさまでした|それでは、また|では、また|またね)")

# 締めの挨拶と判定する上限の実質文字数。「では、金曜の午後に資料室へご案内
# いたしますので、よろしくお願いします。」のように長い文に埋め込まれた
# 「よろしくお願いします」は締めではなく本題の依頼なので、締めとみなさない。
_CLOSING_MAX_LEN = 25

# 内容を持たない極短の反応。文脈と無関係に貼れる間投詞だけのターンで、
# 「6文字以下の極短を◯ターン以上」というノルマを機械的に消化した退化出力の印
# （実測: 敬語のビジネス対話に「え？」「まじ？」「うそ。」「で？」が4連発）。
_VACUOUS_SHORT_RE = re.compile(
    r"^(え|えっ|あ|あっ|お|で|は|ん|うそ|ウソ|嘘|まじ|マジ|ほんと|ホント|"
    r"へえ|へー|ふーん|うーん|そう|うん|はい|ええ|ねえ)[、。？]?$")

# 助詞・条件形の直後に「です」が来る形。文の途中が抜けた出力（実測:「進路に不安がですよね。」
# 「具体的なキーワードはですか？」「具体的にどうすればですよね。」）。語が欠けているので
# 後処理では直せない＝作り直させる。
#   ・「が」は「さすがです」「いかがですか」を除く。「ば」は条件形だけ
#     （「言葉です」を巻き込まないため）。
#   ・「は」は「〜ならではです」を除く（「実家ならではですよね。」は正しい日本語。
#     Gemma 4 が使うため露出した。上の「が」の さすが/いかが と同じ形の除外）。
#   ・「に」「と」「も」は名詞の末尾（兄・蟹・こと・とも）と区別できないので入れない。
_TRUNCATED_COPULA_RE = re.compile(
    r"(?:(?<!さす)(?<!いか)が|(?<!ならで)は|を|へ|(?:れ|え|け|せ|て|め|べ|げ)ば)"
    r"(?:です|でした)")

# 動詞の終止形に「です」が直接つく破格（実測:「最後に一言添えるですよね。」
# 「バランス良く見えるですよ。」）。列挙の締めで出やすい。名詞化（「〜のです」
# 「〜んです」）や「はず・つもり・わけ・そう・よう・みたい＋です」は自然なので、
# 直前が「ま・だ・で・の・ん」の場合を除く。
#   ・「く」「ぐ」「つ」は入れない: 「よろしくです」「すぐですか」「3つですか」
#     「いつですか」を誤検知する（形容詞連用形・副詞・助数詞）。
#   ・既存192,876ターンで検出52件・誤検知0件（全て「〜するです」型の破格）。
#   ・「あるある」は名詞なので除く。「あはは。それは、あるあるですね。」は正しい
#     日本語だが、上の除外文字（ま・だ・で・の・ん）に「あ」が入っていないため
#     破格として検出され、作り直しが空回りしていた（Gemma 4 は「あるあるですね」を
#     多用するため露出。実測: 100対話のリトライ42回のうち10回がこれ）。
#     上の「誤検知0件」は Qwen の語彙で測った値で、モデルを変えると破れる。
_VERB_DESU_RE = re.compile(r"(?<![まだでのん])(?<!あるあ)[すぬぶむる](?:です|でした)")

# 敬語指定の対話に混ざってはいけない砕けた語（プロンプト例示の写しで実測）。
# ★文末のタメ口の断定形も加える（ユーザー指摘 2026-08-01:「まあ、そうなんだ。大変ですね」）。
#   文末に限るのは、引用・従属節・連体修飾の「だ」「てる」を巻き込まないため
#   （「重要だと言われました」「行くと思います」「担当する資料」は文末ではない）。
#   敬語指定 1,671ターンで検出20件・誤検知0件。うち15件は頭のリアクション語なので、
#   下の fix_casual_reactions が先に丁寧形へ直す（作り直させるのは残りだけ）。
_CASUAL_LEAK_RE = re.compile(
    r"(まじで|マジで|うそでしょ|ウソでしょ|嘘でしょ|じゃん|っしょ|てかさ|"
    r"だぜ|やべえ|やべー|うける|ウケる"
    r"|(?<!ま)だ(?:ね|な|よ|よね|ろう|ろ|っけ)?(?=[。！？]|$)"
    r"|(?:てる|てた|ちゃう|ちゃった)(?=[。！？]|$))")

# 敬語指定の対話に出てしまう、頭のタメ口リアクション語 → 丁寧形へ置換する対応表。
# プロンプトが使ってよい語として「そうなんだ」「そっか」「いいね」を口調に関係なく
# 挙げていたのが原因（プロンプト側も直したが、モデルは指示を破る前提で機械でも直す）。
# 作り直させるより置換のほうが安い（意味は変わらず、試行を消費しない）。
_CASUAL_REACT = {
    "そうなんだね": "そうなんですね", "そうなんだよ": "そうなんです",
    "そうなんだ": "そうなんですね", "そうだね": "そうですね",
    "そうだな": "そうですね", "そっか": "なるほど", "いいね": "いいですね",
}
# 前にフィラーや別のリアクション語が1つ付く形も拾う
# （実測「まあ、そうなんだ。」「なるほど、そうなんだね。」「そう、そうなんだ。」）。
_CASUAL_REACT_RE = re.compile(
    r"^(?P<lead>[^、。！？]{1,6}、)?(?P<w>"
    + "|".join(sorted(_CASUAL_REACT, key=len, reverse=True)) + r")(?=[、。！？])")


# 文末に付く相槌の反響（実測「…人手不足の緩和に繋がりますね、そうだよね。」）。
# 内容を持たないので落とす。口調崩れゲートは検知するが、最終試行では通してしまう
# （作り直せないときは受け入れる設計）ので、機械でも直せるようにしておく。
# 188,025ターンで2件。「うん、そうそう。」のような短い相槌は敬語の対話には出ないうえ、
# この関数自体が敬語指定のときしか呼ばれないので巻き込まない。
_CASUAL_TAIL_ECHO = re.compile(r"、(?:そうだよね|そうだね|そうだな|そうだ)(?=[。！]?$)")

# 文末のタメ口の断定 → 丁寧形（実測 2026-08-01:「実際に利用者数が過去5年で30パーセントも
# 増えているんだよ。」が敬語指定の対話に出た）。口調崩れゲートは検知するが最終試行では
# 通してしまうので、置換で直せる形は置換する（作り直しの試行を使わない）。
# 敬語指定 146,116ターンから掘った文末の「だ」系は54件で、すべて
# 「〜んだ(＋終助詞)」「体言・ナ形容詞＋だ(＋終助詞)」「だ止め」の3型に収まる。
# 「まだ」「ただ」だけは語の一部なので除く（他に文末が「〜だ」になる語は実測で無かった）。
_CASUAL_TAIL_DA = re.compile(
    r"(?<![また])だ(?P<sfp>よね|ね|な|よ|っけ|ろう|ろ)?(?=[。！？]|$)")
_DA_SFP_POLITE = {
    None: "です", "よ": "ですよ", "ね": "ですね", "な": "ですね",
    "よね": "ですよね", "っけ": "でしたっけ", "ろう": "でしょう", "ろ": "でしょう",
}
# 同じく文末のタメ口の活用形。活用は機械で一意に決まる。
_CASUAL_TAIL_TE = re.compile(r"(?:てる|てた|ちゃう|ちゃった)(?=[。！？]|$)")
_TE_POLITE = {"てる": "ています", "てた": "ていました",
              "ちゃう": "ちゃいます", "ちゃった": "ちゃいました"}
# 括弧なしの引用（「〜と伝える」「〜と聞く」「〜はどうでしょう」）。引用の中身は
# タメ口のままが正しいので直さない。146,116ターンの試走で 53件中3件がこれだった。
_QUOTATIVE_AFTER = re.compile(r"[。！？]?(?:と|って|は)[^。！？]")


def _to_polite_tail(u: str) -> str:
    """文末のタメ口の断定・活用を丁寧形に直す（引用の中は触らない）。"""
    def sub(m, table, key=None):
        if _QUOTATIVE_AFTER.match(m.string, m.end()):
            return m.group(0)
        return table[m.group(key) if key else m.group(0)]

    u = _CASUAL_TAIL_DA.sub(lambda m: sub(m, _DA_SFP_POLITE, "sfp"), u)
    return _CASUAL_TAIL_TE.sub(lambda m: sub(m, _TE_POLITE), u)


def fix_casual_reactions(turns: list) -> int:
    """敬語の対話のタメ口を直す（直した数を返す）。

    冒頭のリアクション語は丁寧形に置き換え、文末に付いた反響は落とし、
    文末のタメ口の断定・活用は丁寧形に直す。
    """
    n = 0
    for t in turns:
        u = str(t.get("utterance", ""))
        m = _CASUAL_REACT_RE.match(u)
        if m:
            u = ((m.group("lead") or "")
                 + _CASUAL_REACT[m.group("w")] + u[m.end():])
            n += 1
        stripped = _CASUAL_TAIL_ECHO.sub("", u)
        if stripped != u and _core_len_utt(stripped) >= 6:
            u, n = stripped, n + 1
        fixed = _to_polite_tail(u)
        if fixed != u:
            u, n = fixed, n + 1
        t["utterance"] = u
    return n


# --- 一人称の性別合わせ -----------------------------------------------------
# 話者の性別と食い違う一人称は、そのまま合成すると「声の性別が合っていない」と
# 聞こえる（実測: 女性のサラが「PDFは俺が作業するから」「俺もデザインに集中できる」
# 「フォントサイズは僕が調整するね」。7,942ターンで3件）。
# metadata に gender を書くようにしたので Stage2 のプロンプトからも見えるが、
# 指示は破られる前提で機械でも直す。
#
# ひらがなの「おれ」「ぼく」は他の語に埋まる（お礼・折れる・倒れる・簿区…）ので
# 助詞・文末が続くときだけ拾う。「僕」「俺」は漢字なので語中誤爆がない
# （「公僕」は話し言葉に出ない）。
_PARTICLE_AFTER = r"(?=[はがもをにでとのら、。！？]|$)"
_MALE_FIRST_PERSON = re.compile(r"俺|僕|(?:おれ|ぼく|わし)" + _PARTICLE_AFTER)
_FEMALE_FIRST_PERSON = re.compile(r"あたし")
_QUOTED = re.compile(r"[「『][^」』]*[」』]")


def _outside_quotes(u: str, span: tuple) -> bool:
    """その位置が引用（「」『』）の外か。引用の中は相手の言葉なので触らない。"""
    return not any(q.start() < span[0] and span[1] <= q.end()
                   for q in _QUOTED.finditer(u))


def fix_gendered_first_person(turns: list, gender_by_name: dict) -> list:
    """話者の性別と逆の一人称を「私」に直す（直した発話の一覧を返す）。

    「私」は男女どちらでも使えるので、置換先はこれ1つで足りる。タメ口でも
    「私」は不自然にならない（「あたし」に寄せると別の性格付けになる）。
    """
    fixed = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        g = gender_by_name.get(t.get("speaker"))
        if g not in ("male", "female"):
            continue
        pat = _MALE_FIRST_PERSON if g == "female" else _FEMALE_FIRST_PERSON
        u = before = str(t.get("utterance", ""))
        for m in reversed(list(pat.finditer(u))):
            if _outside_quotes(u, m.span()):
                u = u[:m.start()] + "私" + u[m.end():]
        if u != before:
            t["utterance"] = u
            fixed.append(f"{t.get('speaker')}({g}): {before}")
    return fixed


def _is_vacuous_short(utterance: str) -> bool:
    """「え？」「まじ？」のような、直前の発話と無関係に置ける空の極短ターンか。"""
    return len(re.sub(r"[、。？]", "", utterance)) <= 4 and bool(
        _VACUOUS_SHORT_RE.match(utterance))


def _closing_index(turns: list) -> int | None:
    """対話後半で最初に締めの挨拶が現れるターン番号（無ければ None）。

    前半の「よろしくお願いします」は本題の依頼であることが多いので見ない。
    """
    for i in range(len(turns) // 2, len(turns)):
        t = turns[i]
        if not isinstance(t, dict):
            continue
        u = str(t.get("utterance", ""))
        if _CLOSING_RE.search(u) and len(re.sub(r"[、。？]", "", u)) <= _CLOSING_MAX_LEN:
            return i
    return None


# 締めを切り出したとみなす語（main_closes ゲート）。_CLOSING_RE と違い「よろしくお願いします」は
# 入れない（「今度教えますね」への返事に多い）。相手の「楽しかったです」「そろそろ」は締めの切り出し。
_CLOSE_START_RE = re.compile(r"失礼します|失礼いたします|ありがとうございました|お話しできて|"
                             r"楽しかったです|そろそろ|(?:それ|)では、?また|じゃあ、?また|お疲れ様|"
                             r"ゆっくり、?お話し")
_CLOSE_START_WINDOW = 6


def main_close_start(turns: list, main: str) -> int | None:
    """末尾 _CLOSE_START_WINDOW ターンで最初に締めを切り出した発話が主話者なら、その番号。"""
    if not main:
        return None
    for i in range(max(0, len(turns) - _CLOSE_START_WINDOW), len(turns)):
        t = turns[i]
        if isinstance(t, dict) and _CLOSE_START_RE.search(str(t.get("utterance", ""))):
            return i if t.get("speaker") == main else None
    return None


def drop_padding_after_close(turns: list) -> tuple[list, int]:
    """締めの挨拶より後ろに置かれた「中身のない極短ターン」を落とす。

    会話が着地した後に極短ノルマを消化する退化出力（実測: 締めの2ターン後から
    「え？」「まじ？」「うそ。」「で？」が4連発し、その後何事もなく本題が再開する）
    への機械的な後始末。落とすのは情報を持たないターンだけなので内容は減らない。
    落として生じた同一話者の連続は、呼び出し側が merge で交互構造に戻す。
    """
    close_at = _closing_index(turns)
    if close_at is None:
        return turns, 0
    kept = [
        t for i, t in enumerate(turns)
        if i <= close_at
        or not (isinstance(t, dict) and _is_vacuous_short(str(t.get("utterance", ""))))
    ]
    return kept, len(turns) - len(kept)

_SENT_SPLIT_RE = re.compile(r"(?<=[。？])")


def split_multi_sentence_turns(turns: list) -> list:
    """2文以上入ったターンを、同一話者の連続ターンに機械分割する。

    「1ターン1文」はプロンプト指示だけでは徹底されない（特に敬語・ビジネス対話で
    「挨拶文+用件文」の2文ターンが残る）ため、文境界「。」「？」で確定的に分割する。
    同一話者の連続ターンは Stage 3 enrich が continued: true を機械付与するので、
    retime 後も自然な「継ぎ足し」になる。
    3文字未満の断片（「ね。」「はい。」等の言いさし）は前の文に残す。
    """
    out = []
    for t in turns:
        u = t.get("utterance") if isinstance(t, dict) else None
        if not isinstance(u, str):
            out.append(t)
            continue
        parts = [p for p in _SENT_SPLIT_RE.split(u) if p]
        merged = []
        for p in parts:
            if merged and len(re.sub(r"[、。？ー]", "", p)) < 3:
                merged[-1] += p
            else:
                merged.append(p)
        for p in merged:
            nt = dict(t)
            nt["utterance"] = p
            out.append(nt)
    return out


def merge_consecutive_same_speaker(turns: list) -> list:
    """同一話者の連続ターンを1ターンに統合し、話者が必ず交互になるよう機械的に保証する。

    Stage 2 の方針は「2人が交互に発話する台本」（同一話者の連続を作らない）。
    プロンプト指示だけでは徹底されないため、モデルが同じ話者を続けて出した場合は
    発話を結合して交互構造を確定させる。結合時は前ターンの文末が「。」「？」で
    終わるよう補い、文の癒着を防ぐ。以降の Stage 3(enrich) は続き(continued)を
    作らず1ターン=1発話として扱えるようになる。
    """
    out = []
    for t in turns:
        if not isinstance(t, dict):
            out.append(t)
            continue
        if (out and isinstance(out[-1], dict)
                and out[-1].get("speaker") == t.get("speaker")
                and PAUSE_KEY not in t        # 長い沈黙を挟む続き（dialogue.silence）はつながない
                and not is_rp_turn(t) and not is_rp_turn(out[-1])):   # 重なりの場面（rp_v1）もつながない
            prev = str(out[-1].get("utterance", "")).rstrip()
            cur = str(t.get("utterance", "")).lstrip()
            if prev and not prev.endswith(("。", "？")):
                prev = prev.rstrip("、") + "。"
            out[-1] = {**out[-1], "utterance": prev + cur}
        else:
            out.append(dict(t))
    return out


def build_dialogue_prompt(metadata: dict, sp: "S2Params | None" = None,
                          gates: dict[str, bool] | None = None,
                          profile: str = "qwen",
                          pack: "S2PromptPack | None" = None,
                          persona: str = "",
                          silence: "SilenceParams | None" = None) -> str:
    # 交互発話・1ターン1発話の設計では、Stage3(enrich)がターンを増やさず、相槌は
    # 音声段(MaAI)で足すので、metadata の target_turns がほぼ最終ターン数になる。
    # 旧来の 1.5〜2.0倍は、短文分割・相槌挿入で膨らむ前提の名残で、過大なノルマは
    # flow 着地後の「締めの挨拶の繰り返し」による引き延ばし（終盤の似た発話の連続）を
    # 招く。そのため metadata の値をほぼそのまま使い、わずかな幅だけ持たせる。
    # モデルは metadata 内の数値に強くアンカーするため、書き換えた範囲を
    # プロンプト末尾（最も従いやすい位置）にも数値ノルマとして明示する。
    pack = pack or GENERAL_S2
    meta = dict(metadata)
    # ch0 固定用の下流専用マーカー。台本生成には関係ないので出さない
    # （enrich の数値は現行プロンプトに対して調整済みなので、プロンプトを変えない）。
    main_idx = meta.pop("main_speaker_index", None)
    # 対話ごとに引いた主話者のペルソナ（persona_files）。ファイル名は台本に関係ないので出さず、
    # 中身を下の【名前の人物設定】段落で渡す。無ければ config 全体のペルソナ（引数 persona）。
    meta.pop(PERSONA_KEY, None)
    persona = persona_of(metadata, persona)
    # 相手側の人物カード（metadata.partner_persona）は JSON に生で出さず、下の段落で渡す。
    partner = partner_card_of(metadata)
    if partner:
        meta["speakers"] = strip_cards(meta["speakers"])
    # 話題の指定（metadata.topic_plan）も JSON に生で出さず、下の段落で渡す。
    plan = topic_plan_of(metadata)
    meta.pop("topic_plan", None)
    listener = bool(plan) and plan.get("style") == AL_STYLE     # 傾聴: 主話者が聞き役
    alt = "2人の話者は必ず交互に発話し、同じ話者を続けて2ターン置かないでください。"
    if silence is not None:
        alt = _silence_norm(metadata, silence) or alt
    if is_rp_plan(plan):
        alt = alt + rp_norm(plan)
    base = int(meta.get("target_turns") or 18)
    lo, hi = base, int(base * 1.15)
    meta["target_turns"] = f"{lo}〜{hi}"
    # 極短ターンは「4ターン以上」の下限ノルマだと、着地後に例示文字列をそのまま
    # 並べて消化される（実測）。比率で示し、上限も与えて詰め込みを止める。
    nums = PROFILE_NUMS.get(profile, PROFILE_NUMS["qwen"])
    # ★2026-11-07 傾聴（al）では極短の反応を Gemma 向けの2割にしない（聞き手が相手の話の途中に口を挟む
    #   原因になる。途中の相槌は Stage4 の挿入相槌が受け持つ）。
    if listener:
        nums = PROFILE_NUMS["qwen"]
    q_lo, q_hi = (max(2, round(lo * nums["tiny_lo"])),
                  max(3, round(hi * nums["tiny_hi"])))
    return (
        "以下のメタデータに基づいて対話を生成してください:\n\n"
        + json.dumps(meta, ensure_ascii=False, indent=2)
        + _persona_block(meta, main_idx, persona)
        + (stage2_card_block(*partner) if partner else "")
        + (stage2_topic_block(plan, _main_name(meta, main_idx),
                              _partner_name(meta, main_idx)) if plan else "")
        # ★2026-11-07 傾聴（al）では、話し手はひとつの話をひと区切りまで1ターンで話す（1ターン1文だと話し手の話が
        #   細切れになり、聞き手が区切りごとに「〜ですね」と口を挟む。試走7の視聴「発話の途中で割り込みすぎ」）。
        # ★2026-11-08 聞き役も「受け＋ひと言」（人間の傾聴の「熊に会われたんですね、怖くなかったですか」）を書けるよう
        #   1〜2文に。長さは【傾聴の流れ】で15字（続けるときも20字）までと書いている。
        + ("\n\n聞き役の発話は1ターン1〜2文の短いもの、話し手の発話は1ターン1〜3文にし、"
           if listener else f"\n\n発話は1ターン1文だけにし、")
        + f"ターン数は{lo}〜{hi}ターンにしてください。"
        + ("55文字を超える発話を作らないでください。" if listener else
           "55文字を超える発話や、2文以上入ったターンを作らないでください。")
        + alt
        + "発話の長さは極短〜長めをランダムに混ぜてばらつかせてください: "
        + f"6文字以下の極短の反応を全体の{nums['tiny_pct']}割程度"
          f"（{q_lo}〜{q_hi}ターン）、"
        + _short_turn_norm(lo, hi, sp, **({"listener": True} if listener else {}))
        + (pack.long_utt_quota_user
           if gates is None or gates.get("long_utterance", True)
           else pack.long_utt_free_user)
        # ★2026-11-02 傾聴（al）では短い反応のほとんどを聞き手が受け持つので、「直前の語を使った聞き返し」が
        #   そのまま言い直しの多さになった（試走1: 主話者のターンの40%。人間の傾聴 10%）。
        # ★2026-11-09 傾聴の【傾聴の流れ】で「〜んですね」を2回までにしたのに揃える。
        + ("極短の反応は、直前の発話への驚き・感心・共感のひと言にし（相手の語のくり返しは対話全体で2回まで）、"
           if listener else
           "極短の反応は、直前の発話に出てきた語を使った聞き返し・驚き・確認にし、")
        + "話題と無関係な間投詞だけのターンにしないでください。"
        + "極短の反応は前半〜中盤に置き、2ターン続けないでください。"
        + "締めの挨拶を言ったら、そのやり取りで対話を終えてください"
        + "（締めの後に話題・確認・極短の反応を続けない）。"
        + ("締めの言い方は対話ごとに変えてください。"
           "「ありがとうございました。失礼します。」のような決まった組み合わせだけに"
           "せず、その対話で話したことに触れて終える、相手の名前を呼んで終える、"
           "短く「それでは。」で終える、など別の終わり方も使ってください。"
           if sp is not None and sp.vary_closing
           and not (plan and plan.get("closing")) else "")
        + _question_norm(lo, hi, sp)
        + _style_norm(metadata)
        # listener は傾聴のときだけ渡す（三者対話の generate_dialogues_tri は _enrich_norm を自前の版に差し替える）
        + _enrich_norm(lo, hi, sp, **({"listener": True} if listener else {}))
        + "\nJSONのみを出力してください。"
    )


def _partner_name(meta: dict, main_idx) -> str:
    """主話者でない方（2人の対話の相手）の名前。分からなければ空文字。"""
    try:
        names = [str(s["name"]) for i, s in enumerate(meta["speakers"]) if i != int(main_idx)]
    except (KeyError, IndexError, TypeError, ValueError):
        return ""
    return names[0] if len(names) == 1 else ""


def _main_name(meta: dict, main_idx) -> str:
    """主話者（ch0 固定話者）の名前。分からなければ空文字。"""
    try:
        return str(meta["speakers"][int(main_idx)]["name"])
    except (KeyError, IndexError, TypeError, ValueError):
        return ""


def _persona_block(meta: dict, main_idx, persona: str) -> str:
    """主話者のペルソナ（config: metadata.main_speaker.persona_file）の段落。

    シナリオのパックとは別に、メタデータの直後に置く（末尾の数値ノルマより前。
    ノルマは末尾が最も従いやすいので位置を譲らない）。main_speaker_index の無い
    メタデータ（固定話者でないコーパス）やペルソナが空のときは何も足さない。
    """
    if not persona or main_idx is None:
        return ""
    try:
        name = meta["speakers"][int(main_idx)]["name"]
    except (KeyError, IndexError, TypeError, ValueError):
        return ""
    return (f"\n\n【{name}の人物設定】{name}はどの対話でも同じ人物で、次のような人です。"
            f"{name}のセリフの言葉選び・口調・反応のしかたは、この人物設定に合わせてください"
            f"（メタデータの personality と食い違う点は、この人物設定を優先します）。"
            f"人物設定の内容を{name}自身に説明させたり、台本に無理に盛り込んだりはしないでください。\n"
            f"{persona}")


def _half_range(lo: int, hi: int) -> str:
    """対話全体のターン数 lo〜hi を2人で分けたときの1人ぶん（「4〜5」「5」）。"""
    a, b = lo // 2, -(-hi // 2)
    return f"{a}" if a >= b else f"{a}〜{b}"


def _short_turn_norm(lo: int, hi: int, sp: "S2Params | None", listener: bool = False) -> str:
    """7〜15文字の短い発話のターン数ノルマ（sp.short_turn_ratio が 0 なら何も足さない）。

    人間コーパスの先頭33ターンはこの帯が 40.8% なのに、合成は 20% で、代わりに
    16〜30字が 64%（人間 25%）に固まる。比率の指示は本文にすでにあって従われないので、
    プロンプト末尾に実数で書く（極短・長めのノルマと同じ手口）。
    """
    if sp is None or sp.short_turn_ratio <= 0:
        return ""
    n = (lo + hi) // 2
    q = max(2, round(n * sp.short_turn_ratio))
    # 短い側だけ実数で縛ると長め発話が消える（1回目の作り直しの実測: 31〜55字が
    # 11.1%→4.2%。人間は 11.6%）。長い側も同じ文で実数にする。
    lq = max(1, round(n * sp.long_turn_ratio)) if sp.long_turn_ratio > 0 else 0
    long_part = (f"読点で区切りながらひと息に話す31〜55文字の長めの発話を{lq}ターン、"
                 if lq else "")
    # ★2026-10-29 括弧内の「相手の話への具体的な受け止め」を外した。ft_v2 試走で短いターンが
    #   「素材の味がそのまま、伝わります。」「向き不向きは、誰にでもあることですから。」のような
    #   感想・一般論で埋まった（一般論・教訓のまとめ 23〜25回/1000ターン・人間 1.1）。
    # ★2026-11-07 傾聴（al）では「言いさし」を外す（話し手が言いかけで切ると聞き手が口を挟む）。
    # ★2026-11-08 傾聴は「聞き役の焦点の受け・掘り下げの問い」→ 受け・驚き/共感・問い（試走8で受けと問いの交互になった）。
    kinds = ("聞き役の短い受け・驚きや共感のひと言・問い、話し手の問いへの短い答え" if listener else
             "相手の問いへの短い答え・言いさし・体言止め・短い自己開示")
    # ★2026-11-09 傾聴では「なるほど」で埋めない、を外す。人間の傾聴の聞き手は「そうなんですね・なるほど」の短い受けが
    #   37%でいちばん多く（ERICA）、【傾聴の流れ】でも短い受けをいちばん多く使うと書いている。
    stock = "" if listener else "「なるほど」のような受け止めの決まり文句で埋めない"
    return (f"7〜15文字の短い発話を{q}ターン以上"
            f"（{kinds}。{stock}）、"
            + long_part)


def _question_norm(lo: int, hi: int, sp: "S2Params | None") -> str:
    """「？」で終わる問いかけの上限（sp.question_ratio_cap が 0 なら何も足さない）。

    010（1万対話）は全ターンの19.1%が問いかけで、主話者の問い→相手の答えが主話者の
    ターンの21%を占める（人間コーパスの同じ尺では 7.8% と 5%）。質問で話を広げる
    インタビュー調になっていて、学習したモデルも同じ型で話す。
    """
    if sp is None or sp.question_ratio_cap <= 0:
        return ""
    n = (lo + hi) // 2
    q = max(2, round(n * sp.question_ratio_cap))
    return (f"\n\n問いかけ（「？」で終わる発話）はこの対話全体で{q}ターンまでにしてください。"
            f"相手の話は質問で広げるのではなく、自分の似た経験や言いさしで受けて"
            f"ください（人間の初対面の雑談でも、問いかけは全体の1割以下です）。"
            f"同じ人が問いかけを2回続けない、相手の答えのあとすぐ次の問いを重ねない、"
            f"話題を変えるときも質問ではなく自分の話から入る、の3つを守ってください。")


def _enrich_norm(lo: int, hi: int, sp: "S2Params | None", listener: bool = False) -> str:
    """enriched モードの個数ノルマ。比率指示だけでは従わないので実数で示す。

    Stage2 の極短ターン・長め発話ノルマと同じ手口（プロンプト末尾が最も従いやすい）。
    """
    if sp is None:
        return ""
    n = (lo + hi) // 2
    fq = max(2, round(n * sp.filler_ratio))
    fcap = max(fq + 1, round(n * sp.filler_cap_ratio))
    rq = max(1, round(n * sp.reaction_ratio))
    # ★2026-11-02 傾聴（al）では「相手の話に出た具体的な語を使う、自分の経験をそのまま言う」が、聞き手の
    #   言い直しと自分の話を促す。
    own_words = ("（相手の出来事や気持ちへの具体的なひと言にする）。" if listener else
                 "（相手の話に出た具体的な語を使う、自分の経験をそのまま言う、など）。")
    sq = max(2, round(n * sp.sfp_ratio))
    # sfp_cap_ratio は機械間引きの上限。プロンプトには載せない（上限を数で示すと
    # そこまで埋めようとするため。2026-08-01 に「〜scap ターン」の表示を外した）。
    lq = max(1, round(n * sp.laugh_ratio)) if sp.laugh_ratio > 0 else 0
    # ★2026-11-09 傾聴（al）試走10: 聞き役が、話し手の焦った・困った話に「あはは、それは焦りましたね」と笑って返す
    #   （判定で1対話0.1〜0.3件）。「失敗談」は笑ってよい場面に読まれるので、聞き役には話し手が自分で笑い話にしたときだけにする。
    # 版e（100対話）: 「自分で笑い話にしている失敗談」でも、話し手が笑っていない迷子・設定ミス・魚を逃した話に
    #   「あはは、それは焦りますね」が11回 → 話し手が先に笑ったとき（発話に笑い声があるとき）だけにする。
    scene = ("話し手が先に笑ったとき（話し手の発話に笑い声があるとき）だけ、そのすぐ後の聞き役の返しの冒頭に"
             if listener else "相手が冗談・失敗談・大げさなことを言った場面の返しの冒頭に、")
    laugh = ("" if not lq else
             f"\n- {scene}"
             f"笑い声（「あはは、」「はは、」「ふふ、」「ふふふ、」）を置くターンを"
             f"{lq}ターン。数ではなく場面で決めてください。この対話でいちばん打ち解けている"
             f"場面（雑談・打ち明け話・失敗談・軽い自虐）を選び、その受け手の発話の冒頭に"
             f"置いてください。全編が深刻な相談・トラブル・クレームの対話のときだけ0回で"
             f"構いません。1ターン目には置かないでください。"
             f"「（笑）」「w」のような記号では書かないでください。"
             f"深刻な相談・トラブル・謝罪の場面では使わないでください。"
             + ("相手が焦った・困った・不安だった・痛かった話には、笑い声を付けないでください。" if listener else ""))
    return (
        f"\n言いよどみの分量（この対話全体での実数。守れていない出力は不合格です）:"
        f"\n- フィラー（なんか、あの、まあ、その、こう、えっと等。末尾に「ー」を付けない）を"
        f"入れるターンは{fq}〜{fcap}ターン。"
        + (f"このうち{_half_range(fq, fcap)}ターンずつを2人の話者それぞれに入れてください"
           f"（どちらか一方に偏らせない。相手役の話者のフィラーも減らさない）。"
           if sp.filler_split else "") +
        f"1ターンに入れる個数は発話の長さで決めてください（"
        + "、".join(
            f"実質{hi}文字以下は{nf}個" if lo == 1 else
            (f"実質{lo}文字以上は{nf}個" if hi is None else f"実質{lo}〜{hi}文字は{nf}個")
            for lo, hi, nf, _nc in _len_bands(sp)) +
        f"）。長い発話ほど、途中で言葉に詰まらせてください。"
        f"「えっと」「ええと」に偏らせず、「なんか」「あの」「まあ」「その」を多めに"
        f"使ってください。"
        f"残りのターンにはフィラーを入れないでください。"
        f"文末の「。」「？」の直前と助詞の直前には絶対に置かないでください。"
        f"入れたうちの半分くらいは、発話の頭ではなく文の途中（「、フィラー、」の形）にしてください。"
        + (f"\n- 発話の冒頭に短いリアクション（あー、ああ、そう、そっか、すごい、えっ、"
           f"そうなんだ。「へえ」「へー」は使用禁止）を置くターンは{rq}ターン程度。"
           f"どれか1語に偏らせず散らしてください。"
           f"\n- 次の決まり文句は、発話のどこに出てきても**それぞれ1対話で2回まで**です"
           f"（数えて守ってください）: 「なるほど」「いいですね」「そうですね」"
           f"「そうなんですね」「たしかに」「素敵ですね」「わかります」「私も」。"
           f"3回目が必要になったら、決まり文句ではなく中身のある言葉で受けてください"
           f"{own_words}"
           f"反論・訂正・質問の発話の頭には付けないでください。"
           if sp.react_vocab_human else
           f"\n- 発話の冒頭に短いリアクション（そう、そうなんだ、なるほど、たしかに、そっか、"
           f"あー、いいね。「へえ」「へー」は使用禁止）を置くターンは{rq}ターン程度。"
           f"反論・訂正・質問の発話の頭には付けないでください。")
        + laugh +
        f"\n- 読点「、」を打たずに{sp.comma_span}文字以上続けないでください"
        f"（長い発話は{max(4, sp.comma_span - 2)}文字前後ごとに「、」で区切る。"
        f"実質100文字あたり{sp.comma_per_100:g}個が目安）。"
        f"読点の数も発話の長さで決めます（"
        + "、".join(f"実質{lo}文字以上は{nc}個以上"
                   for lo, _hi, _nf, nc in _len_bands(sp) if nc >= 2) +
        f"）。長い発話ほど細かく切ってください。"
        f"例:「来週から展示会が始まって資料が追いつかなくて焦ってるんだよね。」ではなく"
        f"「来週から展示会が始まって、資料が追いつかなくて、焦ってるんだよね。」と書く。"
        # 終助詞は指示に反応する（要求35%で実測9〜55%）。以前はここに「ほとんどの発話の
        # 末尾に終助詞を付けてください」と書いていたが、2026-08-01 の実測で**モデルが
        # 自分で書く量（全ターンの25〜47%）だけで既に人間の密度**であることが分かった
        # （人間は発話100字あたり ね0.57/よ0.11。合成は付加前で ね0.8〜1.4）。
        # 増やす指示は不要なので落とした。上限側の禁止（「だ」止め・「よね」「な」の
        # 用法）だけを残す。足りないぶんの穴埋めは inject_sfp が「だ」止めに限って行う。
        f"\n- 文末を終助詞で終えるターンは{sq}ターン程度で、多すぎないようにしてください。"
        f"★「〜だ。」「〜んだ。」で言い切らないでください（書き言葉です）。"
        f"「〜だよ。」「〜だね。」「〜だな。」「〜んだよ。」のどれかにしてください。"
        f"「な」を優先し（「ね」「よ」は意識せずとも出るので数に頼らない）、"
        f"★「よね」は相手も知っている事柄に同意を求めるときだけに使ってください。"
        f"これから自分がすること・自分だけが知っている経験や気持ち、"
        f"相手の質問への答えには付けないでください"
        f"（×「ちょっと試してみますよね。」→○「ちょっと試してみますね。」、"
        f"×「インターンでSNS運用をやっていましたよね。」"
        f"→○「インターンでSNS運用をやっていました。」）。"
        f"「な」はタメ口で「〜だ」「〜た」「〜い」で終わる発話にだけ付けてください。"
        f"動詞の言い切り（「行く」「思う」「見る」等）と敬語には「な」を付けないでください"
        f"（「行くな」「思うな」は禁止の意味になり、発話の意味が反転します）。"
    )


def _style_norm(metadata: dict) -> str:
    """speech_style の厳守指示。短文化ノルマに気を取られてスタイルが崩れるのを防ぐ。"""
    style = str(metadata.get("speech_style") or "")
    if not style:
        return ""
    norm = f"\nspeech_style「{style}」は全ターンで厳守してください。"
    if "混合" in style:
        norm += "（関係性に合わせて、一方の話者は敬語、もう一方はタメ口のように非対称にする）"
    elif "敬語" in style:
        norm += "（くだけた話し言葉の敬語にしつつ、「です・ます」は全発話で維持する）"
    return norm


async def generate_single_dialogue(
    client: AsyncOpenAI,
    model: str,
    metadata: dict,
    metadata_index: int,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    retry_max: int = 3,
    retry_delay: float = 1.0,
    request_interval: float = 0.0,
    extra_body: dict | None = None,
    sp: "S2Params | None" = None,
    gates: dict[str, bool] | None = None,
    self_name_max: int = DEFAULT_SELF_NAME_MAX,
    profile: str = "qwen",
    pack: "S2PromptPack | None" = None,
    proofread: int = 0,
    yomi: bool = False,
    persona: str = "",
    silence: "SilenceParams | None" = None,
    yomi_ruby: int = 0,
) -> dict | None:
    gates = gates if gates is not None else ALL_GATES
    sil_main, sil_user = silence_roles(metadata) if silence is not None else (None, None)
    pack = pack or GENERAL_S2
    system_prompt = build_system_prompt(sp, gates, self_name_max, profile, pack)
    async with semaphore:
        if request_interval > 0:
            await asyncio.sleep(request_interval)
        # 最終試行の失敗種別を記録する。品質ゲート由来(content)なら恒久スキップに、
        # 例外(接続/レート制限/JSON崩れ等)なら一時失敗として扱い区別する。
        last_was_gate = False
        last_reason = ""
        # enriched モードで下限に届かなかった候補のうち、いちばん惜しかったもの。
        # 最終試行は下限を諦めて採用する作りなので、これが無いと「たまたま最後に引いた
        # 悪い候補」が通ってしまう（実測: 6回中いちばん悪い回が採用された）。
        best: tuple[float, list] | None = None
        for attempt in range(retry_max):
            if cost_tracker.exceeded():
                return None
            try:
                response = await _chat_create(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": build_dialogue_prompt(metadata, sp, gates,
                                                          profile, pack, persona,
                                                          silence)},
                    ],
                    temperature=0.9,
                    max_completion_tokens=12288,   # 50-100+ターンの長い対話JSONが途中で切れないよう拡大（4096だとlengthで打切→JSON破損）
                    extra_body=extra_body or {},
                )
                usage = response.usage
                cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)

                text = extract_json_text(response.choices[0].message.content or "")

                # strict=False: 文字列中の生の制御文字（改行等）を許容する。
                # 30Bは発話内に生改行を入れることがあり、既定だと
                # "Invalid control character" でリトライを浪費する。
                # 改行・空白は後段の sanitize_utterance が除去する。
                dialogue = json.loads(_fix_missing_utterance_key(text), strict=False)
                if "turns" in dialogue and isinstance(dialogue["turns"], list):
                    # 伏せ字（○○さん等）は sanitize で消えて「監督はさんで」のような
                    # 穴あき文になるため、含まれていたら出力ごと作り直す
                    if gates["masked_text"] and _PLACEHOLDER_RE.search(text):
                        logger.warning(f"[{metadata_index}] 伏せ字記号を含む出力。リトライ。")
                        last_was_gate, last_reason = True, "伏せ字"
                        continue
                    # 助詞・条件形の直後に「です」が来る出力（「進路に不安がですよね。」）は
                    # 文の途中が抜けている。語が欠けていて後処理では直せないので作り直す。
                    # 動詞終止形＋「です」の破格（「最後に一言添えるですよね。」）も同じ扱い
                    # （正しくは「〜添えます」「〜添えるのが良いですね」等で、
                    # どう補うかは文脈依存＝機械では選べない）。
                    # ★2026-09-26: 発話の末尾が切れた出力（「それでは、準備が整うまで」）も同じ扱い。
                    #   句点で終わらず、助詞・読点で終わる発話は言いかけのまま切れている。
                    # ★長い沈黙（dialogue.silence）の直前の言いさし（「あの、実は、その、」）は
                    #   意図した途切れなので数えない（末尾の読点は下の句点補完で「。」になる）。
                    #   2026-10-24 試走: このゲートのリトライ 26回がすべてこれだった。
                    pre_pause = (pre_pause_indices(dialogue["turns"])
                                 if silence is not None else set())
                    # ★2026-11-10 重なりの場面（topic_plan.style: rp）。言いかけ（"unsaid" 付き）は意図した途切れ。
                    rp_on = is_rp_plan(topic_plan_of(metadata))
                    rp_cut = ({k for k, t in enumerate(dialogue["turns"])
                               if isinstance(t, dict) and t.get(UNSAID_KEY)} if rp_on else set())
                    trunc = [str(t.get("utterance", ""))
                             for k, t in enumerate(dialogue["turns"]) if isinstance(t, dict)
                             and k not in pre_pause and k not in rp_cut
                             and (_TRUNCATED_COPULA_RE.search(str(t.get("utterance", "")))
                                  or _VERB_DESU_RE.search(str(t.get("utterance", "")))
                                  or _CUT_OFF_TAIL_RE.search(str(t.get("utterance", "")).strip()))]
                    if gates["truncated_utterance"] and trunc and attempt < retry_max - 1:
                        logger.warning(
                            f"[{metadata_index}] 途中が抜けた発話が{len(trunc)}件"
                            f"（例:「{trunc[0]}」）。リトライ。")
                        last_was_gate, last_reason = True, "途中が抜けた発話"
                        continue
                    # 発話から「？」以外の記号と空白を除去（記号のみで空になったターンは捨てる）
                    cleaned = []
                    for k, t in enumerate(dialogue["turns"]):
                        if isinstance(t, dict) and "utterance" in t:
                            t["utterance"] = emphasize_question_marks(
                                sanitize_utterance(t["utterance"]))
                            if rp_on and t.get(UNSAID_KEY):     # 言いかけの残り（rp_v1）
                                t[UNSAID_KEY] = emphasize_question_marks(
                                    sanitize_utterance(str(t[UNSAID_KEY])))
                            if sp is not None:
                                # enriched モードは言いよどみ込みで生成させるので、
                                # 音声にすると必ず壊れる形（文末直前のフィラー、助詞への
                                # 割り込み、動詞＋「な」）をここで機械的に落とす。
                                # 残してよいフィラー数も発話の長さで決める（長い発話ほど
                                # 多く残す）。段階制でないときは従来どおり一律の上限。
                                cap = (max(1, filler_quota(
                                    _core_len_utt(t["utterance"]), sp))
                                    if sp.filler_len_step > 0 else sp.fillers_per_turn)
                                t["utterance"] = clean_enriched_utterance(
                                    t["utterance"], cap,
                                    trailing_off=k in pre_pause or k in rp_cut)
                            if not t["utterance"]:
                                continue
                        cleaned.append(t)
                    if sp is not None:
                        n_voc = fix_vocative_copula(cleaned)
                        if n_voc:
                            logger.info(f"[{metadata_index}] 相手への呼びかけの"
                                        f"「〜です」を{n_voc}ターン修正。")
                    # 自分の発話に自分の名前が出る出力（「ねえ、ユウだよ。」の名乗りや、
                    # 「ユウくん、どう思う？」を本人に言わせる話者取り違え）は作り直す
                    # self_name_max を上げたぶんは名乗りとして許し、超えたら作り直す。
                    per_speaker = Counter(
                        str(t["speaker"]) for t in cleaned
                        if isinstance(t, dict) and t.get("speaker")
                        and str(t["speaker"]) in str(t.get("utterance", ""))
                    ) if gates["self_name"] else Counter()
                    over = {k: v for k, v in per_speaker.items() if v > self_name_max}
                    if over:
                        detail = "、".join(f"{k}が{v}回" for k, v in sorted(over.items()))
                        logger.warning(
                            f"[{metadata_index}] 自分の名前を含む発話が上限"
                            f"({self_name_max}回)を超えています（{detail}）。"
                            f"{'名乗りの繰り返し' if self_name_max else '話者取り違え'}"
                            "とみなしてリトライ。")
                        last_was_gate, last_reason = True, "自分の名前"
                        continue
                    # 文末の「。」を打たない箇条書き調の退化出力（実測: 対話全体で
                    # 句点が消え「キャスティングも豪華だ」のような電報文が並ぶ）は作り直す
                    n_miss = sum(
                        1 for t in cleaned
                        if isinstance(t, dict) and not (rp_on and t.get(UNSAID_KEY))
                        and not str(t.get("utterance", "")).endswith(("。", "？")))
                    if gates["missing_period"] and cleaned and n_miss > 0.25 * len(cleaned):
                        logger.warning(
                            f"[{metadata_index}] 文末句点の欠落が{n_miss}/{len(cleaned)}ターン。"
                            "箇条書き調の退化出力とみなしてリトライ。")
                        last_was_gate, last_reason = True, "文末句点欠落"
                        continue
                    # ★2026-11-08 メタデータに無い話者名（al_v1 試走9で「イノウ\u17c1」＝最後の1文字がクメール文字に化けた）は、
                    #   どのゲートにも掛からず3人目の話者として Stage4 に流れる。作り直す。
                    odd = unknown_speakers(cleaned, metadata)
                    if gates["speaker_names"] and odd:
                        logger.warning(f"[{metadata_index}] メタデータに無い話者名 {odd}。リトライ。")
                        last_was_gate, last_reason = True, "話者名"
                        continue
                    # 残った少数の句点欠落は機械補完する
                    for t in cleaned:
                        if isinstance(t, dict) and not (rp_on and t.get(UNSAID_KEY)):
                            u = str(t.get("utterance", ""))
                            if u and not u.endswith(("。", "？")):
                                t["utterance"] = u.rstrip("、") + "。"
                    # 長い沈黙（dialogue.silence）を検査する。無効なら消して従来どおり結合させる。
                    if silence is not None:
                        sil_drop: list = []
                        normalize_pauses(cleaned, sil_main, sil_user, silence, sil_drop)
                        if sil_drop:
                            logger.info(f"[{metadata_index}] 沈黙の指定を外した: "
                                        + "、".join(sil_drop))
                    else:
                        for t in cleaned:
                            if isinstance(t, dict):
                                t.pop(PAUSE_KEY, None)
                    # 重なりの場面（style: rp）の "unsaid" / "overlap" を検査して整える。rp でなければ外す。
                    if rp_on:
                        rp_drop: list = []
                        normalize_rp(cleaned, rp_drop)
                        if rp_drop:
                            logger.info(f"[{metadata_index}] 重なりの指定を直した: " + "、".join(rp_drop))
                    else:
                        strip_rp_keys(cleaned)
                    # 同一話者の連続を機械的に結合し、話者が必ず交互になるようにする
                    # （沈黙を挟む続きだけは残す）。
                    split_turns = merge_consecutive_same_speaker(cleaned)
                    # 締めの後に置かれた空の極短ターン（ノルマ消化の水増し）を落とし、
                    # 落として生じた同一話者の連続を結合し直す。
                    split_turns, n_pad = drop_padding_after_close(split_turns)
                    if n_pad:
                        split_turns = merge_consecutive_same_speaker(split_turns)
                        logger.info(f"[{metadata_index}] 締めの後の空の極短ターンを"
                                    f"{n_pad}件除去しました。")
                    if silence is not None and drop_closing_pauses(split_turns):
                        split_turns = merge_consecutive_same_speaker(split_turns)
                        logger.info(f"[{metadata_index}] 締めの挨拶以降の沈黙を外しました。")
                    # ユーザの長い沈黙が下限に届かない台本は作り直す。「確実に出る」ことが
                    # このコーパスの目的なので、最終試行でも妥協しない（届かなければスキップ）。
                    if silence is not None and gates["silence"]:
                        n_sil = sum(1 for t in split_turns
                                    if isinstance(t, dict) and PAUSE_KEY in t)
                        if n_sil < silence.count_min:
                            logger.warning(
                                f"[{metadata_index}] ユーザの長い沈黙が{n_sil}回"
                                f"（下限{silence.count_min}回）。リトライ。")
                            last_was_gate, last_reason = True, "沈黙不足"
                            continue
                        # ユーザの言いよどみに主話者が待たずに割り込む形は、狙いと逆の振る舞いを
                        # 教えるので作り直す（最終試行でも妥協しない）。
                        cut = hesitation_interrupts(split_turns, sil_main, sil_user)
                        if cut:
                            logger.warning(
                                f"[{metadata_index}] ユーザの言いよどみの直後に主話者が割り込む"
                                f"箇所が{len(cut)}件（例: turn{cut[0]}「"
                                f"{split_turns[cut[0] - 1].get('utterance', '')}」→「"
                                f"{split_turns[cut[0]].get('utterance', '')}」）。リトライ。")
                            last_was_gate, last_reason = True, "言いよどみへの割り込み"
                            continue
                        # ユーザが主話者に質問して黙り、自分で答える形（最終試行でも妥協しない）
                        selfq = self_answers(split_turns, sil_main, sil_user)
                        if selfq:
                            logger.warning(
                                f"[{metadata_index}] ユーザが自分の問いに沈黙のあと自分で答える"
                                f"箇所が{len(selfq)}件（例: turn{selfq[0]}「"
                                f"{split_turns[selfq[0] - 1].get('utterance', '')}」→「"
                                f"{split_turns[selfq[0]].get('utterance', '')}」）。リトライ。")
                            last_was_gate, last_reason = True, "自問自答"
                            continue
                        # 促す型を頼んだ対話で出なかったら作り直す（prompt_retries 回まで。超えたら
                        # 諦めて採用。試走3回目では2対話だけで14回空回りした）
                        if (wants_prompt(metadata, silence)
                                and attempt < min(retry_max - 1, silence.prompt_retries)
                                and not any(_pause_kind(split_turns, i, sil_main, sil_user) == "C"
                                            for i, t in enumerate(split_turns)
                                            if isinstance(t, dict) and PAUSE_KEY in t)):
                            logger.warning(f"[{metadata_index}] 促す型の沈黙がありません。リトライ。")
                            last_was_gate, last_reason = True, "促しなし"
                            continue
                    # 締めの挨拶の後に会話が続く水増し出力は作り直す（実測: 締めた後に
                    # 7〜14ターン、別の話題や締め直しが続く）。
                    close_at = _closing_index(split_turns)
                    if close_at is not None:
                        n_after = len(split_turns) - 1 - close_at
                        if (gates["padding_after_close"] and n_after >= 5
                                and attempt < retry_max - 1):
                            logger.warning(
                                f"[{metadata_index}] 締めの挨拶(turn {close_at})の後に"
                                f"{n_after}ターン続いています。水増しとみなしてリトライ。")
                            last_was_gate, last_reason = True, "締め後の水増し"
                            continue
                    # 中身のない極短ターンの羅列（例示文字列の貼り付け）は作り直す
                    vac = [i for i, t in enumerate(split_turns)
                           if isinstance(t, dict) and not is_rp_turn(t)
                           and _is_vacuous_short(str(t.get("utterance", "")))]
                    vac_consec = any(b - a == 1 for a, b in zip(vac, vac[1:]))
                    vac_over = len(vac) > max(2, 0.06 * len(split_turns))
                    if (gates["vacuous_short_turn"] and (vac_consec or vac_over)
                            and attempt < retry_max - 1):
                        logger.warning(
                            f"[{metadata_index}] 中身のない極短ターンが{len(vac)}件"
                            f"{'（連続あり）' if vac_consec else ''}。"
                            "極短ノルマの機械的消化とみなしてリトライ。")
                        last_was_gate, last_reason = True, "空の極短ターン"
                        continue
                    # 敬語指定なのにタメ口の驚き・同意表現が混ざる出力は作り直す
                    # （プロンプトの例示を口調に関係なく写す退化。実測: 敬語対話に
                    # 「まじで、」「うそでしょ？」が出る）。
                    if _is_polite_style(metadata):
                        # 頭のリアクション語（「そうなんだ。」）は置換で直せるので、
                        # 作り直させる前に直す。残った砕けた語だけをゲートにかける。
                        fix_casual_reactions(split_turns)
                        leaks = [str(t.get("utterance", "")) for t in split_turns
                                 if isinstance(t, dict)
                                 and _CASUAL_LEAK_RE.search(str(t.get("utterance", "")))]
                        if gates["casual_leak"] and leaks and attempt < retry_max - 1:
                            logger.warning(
                                f"[{metadata_index}] 敬語指定に砕けた語が{len(leaks)}件"
                                f"（例:「{leaks[0]}」）。口調崩れとみなしてリトライ。")
                            last_was_gate, last_reason = True, "口調崩れ"
                            continue
                    # 主話者が序盤に聞かれずに自分の研究・職業の話をする出力は作り直す
                    # （topic_plan のある対話だけ。v2 の目的そのもの。topic_plan.early_self_job）。
                    if topic_plan_of(metadata) and gates["self_job_early"]:
                        mi = metadata.get("main_speaker_index")
                        main = (_main_name(metadata, mi) if mi is not None else "")
                        is_al = (topic_plan_of(metadata) or {}).get("style") == AL_STYLE
                        j = early_self_job(split_turns, main, skip_echo=is_al)
                        if j is None:
                            j = unprompted_job_statement(split_turns, main)
                        if j is not None and attempt < retry_max - 1:
                            logger.warning(
                                f"[{metadata_index}] 主話者が聞かれずに序盤(turn{j})で"
                                f"職業の話「{split_turns[j].get('utterance', '')}」。リトライ。")
                            last_was_gate, last_reason = True, "主話者の職業の自己言及"
                            continue
                        # 聞いていないのに相手が主話者の職業を知っている（role の漏れ）
                        k = partner_knows_job(split_turns, main, need_name=is_al)
                        if k is not None and attempt < retry_max - 1:
                            logger.warning(
                                f"[{metadata_index}] 相手が聞く前に主話者の職業を知っている(turn{k})"
                                f"「{split_turns[k].get('utterance', '')}」。リトライ。")
                            last_was_gate, last_reason = True, "主話者の職業の漏れ"
                            continue
                    # 主話者が締めの挨拶を切り出す（topic_plan の締めは相手から。★2026-10-29 再評価:
                    #   相手が話している途中で主話者が「ありがとうございます。失礼します。」と畳んだ）
                    if (topic_plan_of(metadata) or {}).get("closing") and gates["main_closes"]:
                        mi = metadata.get("main_speaker_index")
                        main = (_main_name(metadata, mi) if mi is not None else "")
                        ci = main_close_start(split_turns, main)
                        if (ci is not None
                                and attempt < retry_max - 1):
                            logger.warning(
                                f"[{metadata_index}] 主話者が締めを切り出す(turn{ci})"
                                f"「{split_turns[ci].get('utterance', '')}」。リトライ。")
                            last_was_gate, last_reason = True, "主話者の締め"
                            continue
                    # 質問と答えの対話（topic_plan.style: qa）: 相手の質問の数・主話者の問いかけの数・
                    # 主話者の発話量を数える。主話者が答えるのが中心、を指示だけに頼らない（qa_plan）。
                    if gates["qa_shape"]:
                        mi = metadata.get("main_speaker_index")
                        main = (_main_name(metadata, mi) if mi is not None else "")
                        why = qa_shape_problem(split_turns, main, topic_plan_of(metadata) or {})
                        if why and attempt < retry_max - 1:
                            logger.warning(f"[{metadata_index}] 質問と答えの形が崩れている（{why}）。リトライ。")
                            last_was_gate, last_reason = True, "質問と答えの形"
                            continue
                    # 傾聴の対話（topic_plan.style: al）: 主話者の発話量・掘り下げの問いの数・相手からの質問の数・
                    # 主話者の長い発話の数を数える。主話者が聞き役、を指示だけに頼らない（al_plan）。
                    if gates["al_shape"]:
                        mi = metadata.get("main_speaker_index")
                        main = (_main_name(metadata, mi) if mi is not None else "")
                        why = al_shape_problem(split_turns, main, topic_plan_of(metadata) or {})
                        if why and attempt < retry_max - 1:
                            logger.warning(f"[{metadata_index}] 傾聴の形が崩れている（{why}）。リトライ。")
                            last_was_gate, last_reason = True, "傾聴の形"
                            continue
                    # 重なりの場面（topic_plan.style: rp）: 指定した場面が台本の形で揃っているか（rp_plan）。
                    # 最終試行でも、重なりが1つも無い台本は採らない（このコーパスの目的そのもの）。
                    # （場面は本生成のあとに1つずつ入れる。下の insert_rp_scenes）
                    # 発話長のばらつき検証: 長め（実質31文字以上）の発話が乏しい出力は
                    # 単調（全部短文）とみなして作り直す。プロンプトの分布指示だけでは
                    # 徹底されないため機械検証する。極短側は従来から安定して出るので
                    # 長め側のみ検証する（最終試行では諦めてそのまま採用）。
                    core_lens = [
                        len(re.sub(r"[、。？]", "", str(t.get("utterance", ""))))
                        for t in split_turns if isinstance(t, dict)
                    ]
                    n_long = sum(1 for L in core_lens if L >= 31)
                    need_long = 4 if len(core_lens) >= 40 else 3
                    if (gates["long_utterance"] and len(core_lens) >= 20
                            and n_long < need_long and attempt < retry_max - 1):
                        logger.warning(
                            f"[{metadata_index}] 長め発話(31文字以上)が{n_long}ターンのみ。"
                            "長さのばらつき不足とみなしてリトライ。")
                        last_was_gate, last_reason = True, "長め発話不足"
                        continue
                    # 話者の性別と逆の一人称（女性が「俺」）は、声の性別が外れて
                    # 聞こえる直接の原因になるので必ず直す。性別は metadata を正とし、
                    # 無ければモデルが返した speakers、それも無ければ名前・役割から。
                    gmap = {}
                    said = {s.get("name"): s.get("gender")
                            for s in (dialogue.get("speakers") or [])
                            if isinstance(s, dict)}
                    for s in metadata.get("speakers") or []:
                        if isinstance(s, dict) and s.get("name"):
                            gmap[s["name"]] = resolve_gender(
                                s.get("gender") or said.get(s["name"]),
                                s.get("name"), s.get("role"))
                    wrong_fp = fix_gendered_first_person(split_turns, gmap)
                    if wrong_fp:
                        logger.info(f"[{metadata_index}] 性別と逆の一人称を"
                                    f"{len(wrong_fp)}件「私」に直しました"
                                    f"（例:「{wrong_fp[0]}」）")
                    # enriched モード: 付与量を上限まで機械間引きし、下限に満たなければ
                    # 作り直す（比率指示だけでは 32B が従わない実績があるため、Stage2 の
                    # 他のノルマと同じく「測って落とす/やり直す」で担保する）。
                    if sp is not None:
                        # 長い沈黙の直前の言いさし（「あの、実は、その。」）は、以下の間引き・読点外しの
                        # 対象から外す（かけると「実はその。」になる）。終わったら書き戻す。
                        trailing = {i: split_turns[i]["utterance"]
                                    for i in pre_pause_indices(split_turns)}
                        # 重なりの場面（rp_v1）の言いかけ・重ねたターンも同じく外して書き戻す
                        trailing.update({i: split_turns[i]["utterance"]
                                         for i in rp_indices(split_turns)})
                        n_laugh = cap_laugh_heads(split_turns, sp.laugh_cap_ratio)
                        n_mono = cap_monotone_heads(split_turns, sp.monotone_head_cap)
                        n_thin = thin_fillers(split_turns, sp.filler_cap_ratio)
                        # 誤用の終助詞は足す前に直す（直したターンは終助詞が無くなる
                        # ものもあるので、そのあとの付加で新しい規則で埋め直させる）
                        # 敬語指定の対話ではテ形の言いさしに「ね」を足さない
                        polite = _is_polite_style(metadata)
                        # 傾聴（al）の聞き手の「〜ましたね」「〜ましたよね」は相手の出来事への共感なので直さない
                        listener = (_main_name(metadata, metadata.get("main_speaker_index"))
                                    if (topic_plan_of(metadata) or {}).get("style") == AL_STYLE
                                    and metadata.get("main_speaker_index") is not None else None)
                        n_yone = (fix_yone_misuse(split_turns, skip_speaker=listener)
                                  + fix_yo_misuse(split_turns)
                                  + fix_ne_misuse(split_turns, polite, skip_speaker=listener)
                                  + fix_vocative_sfp(split_turns))
                        # 終助詞は足してから上限で間引く（「だ」止めを先に潰す）
                        n_add = (inject_sfp(split_turns, sp, polite)
                                 if sp.sfp_inject else 0)
                        # 気づきの「〜んですよね」「そうですよね」は残す（接客で始め、2026-09-26 に
                        # 全シナリオへ。汎用・初対面でも「お子さんがいらっしゃるんです。」のように
                        # 相手のことを言い切る形が各コーパス140〜380件あった）
                        n_sfp = thin_sfp(split_turns, sp.sfp_cap_ratio, keep_realize=True,
                                         keep_eval=sp.ne_keep_eval,
                                         reaction_last=sp.ne_reaction_last,
                                         skip_speaker=listener)
                        # 最後に、同じ終助詞が3ターン以上続く区間を崩す
                        n_run = diversify_sfp_runs(split_turns, polite=polite,
                                                   skip_speaker=listener)
                        # 名乗りに付いた終助詞は、付加・間引きの後に落とす
                        # （付加前に落としても inject_sfp が付け直す）
                        n_yone += fix_self_intro_sfp(split_turns)
                        # 読点なしで長く続く区間を先に割る（フィラーの差し込み先にもなる）
                        n_brk = break_long_runs(split_turns, sp.comma_run_max)
                        n_inj = (inject_fillers(split_turns, sp)
                                 if sp.filler_inject else 0)
                        # 句点化は最後。先にやると inject_fillers の差し込み先
                        # （モデルが打った読点）が減ってしまう。
                        n_per = (to_sentence_periods(
                            split_turns, {t.get("speaker") for t in split_turns
                                          if isinstance(t, dict)})
                            if sp.period_split else 0)
                        # 短い句の直後の読点を外す（句点化の後。先に外すと文の終わりの
                        # 読点まで消して句点化の手がかりが無くなる）。
                        n_per_thin = thin_short_commas(
                            split_turns, sp.comma_thin_seg, sp.comma_thin_run,
                            {t.get("speaker") for t in split_turns if isinstance(t, dict)})
                        # ★「ね」の上限は句点化の**後**にかける（2026-08-06 の実測）。
                        # _SFP_TAIL は「。」「？」か行末でしか終助詞を拾わないので、
                        # 読点で続く「ね」（「〜送りますね、その後で確認します。」）は
                        # 上限の計算から漏れる。句点化がそれを文末に変えるため、
                        # 前に置くと上限を通り抜けた「ね」が残る（敬語 0.68 対 0.54）。
                        # ★担当者の復唱の「ね」（「ブライダルコースですね。」）は thin_ne に
                        # 落とさせない。モデルは復唱をほぼ「ね」で書いていて、言い切りの
                        # 主因は thin_ne だった（2026-09-22 実測: raw の復唱39件中15件を
                        # 落としていた）。
                        protect = {(i, j) for i, j, *_ in readback_sites(
                            split_turns, metadata, sp.readback_ne_window)}
                        # ★接客では担当者の文を thin_ne の対象から外す（客の文だけ間引く）。
                        #   担当者の「ね」は復唱・受け止め・予告の機能語（2026-09-22）。
                        staff = staff_names(metadata)
                        # ★2026-11-07 傾聴（al）の聞き手の文も同じ理由で外す。聞き手の共感の「〜でしたね」が
                        #   上限（1.5個/100字）で落ちて「勇気がいる決断でした。」「それはかなり焦りました。」の
                        #   言い切り（聞き手自身の経験の報告）になっていた（試走7の視聴で不自然と指摘）。
                        exempt = staff or ({listener} if listener else set())
                        cust_only = (sentence_ids(
                            split_turns, {str(t.get("speaker", "")) for t in split_turns
                                          if isinstance(t, dict)} - exempt)
                            if exempt else None)
                        n_sfp += thin_ne(split_turns, sp.ne_cap_per_100,
                                         sp.ne_swap_casual, protect, cust_only,
                                         keep_realize=True, keep_eval=sp.ne_keep_eval,
                                         reaction_last=sp.ne_reaction_last)
                        # 復唱の「ね」は thin_ne の**後**に足す。thin_ne は丁寧形の
                        # 「ね」を落とすので、先に足すと上限超過ぶんとして削られる。
                        # 数は1対話あたり0.7個（実測）なので密度への影響は無視できる。
                        n_rb = add_readback_ne(split_turns, metadata,
                                               sp.readback_ne_window)
                        n_rb += add_customer_echo_ne(split_turns, metadata,
                                                     sp.customer_echo_ne)
                        # 担当者の「です？」「くださいね」は復唱の「ね」を足した後に直す
                        n_yone += fix_staff_tails(split_turns, metadata)
                        # 接客の発話の仕上げ（担当者の笑い・フィラー・定型句のずれ、客の
                        # 立場が逆の敬語）と、客の意味のないオウム返しの置き換え（2026-09-23）
                        # オウム返しを先に見る（polish が客の「です？」を「ですか？」に直すと
                        # 担当者の発話との一致が取れなくなる）
                        n_echo = fix_echo_turns(split_turns, metadata)
                        if sp.echo_fix:
                            n_echo += fix_echo_fragments(split_turns)
                        n_echo += limit_echo_heads(split_turns, sp.echo_head_cap)
                        n_echo += limit_echoes(split_turns, sp.echo_total_cap, sp.echo_replace_once)
                        # 受け止めの定型語（なるほど・いいですね…）の偏りを均す。
                        # オウム返しの間引きの後に置く（limit_echoes が「へえ、そうなんですね。」
                        # を作るので、その分も数に入れる）。
                        n_ack = rebalance_ack_heads(
                            split_turns, sp.ack_head_cap, sp.ack_phrase_cap,
                            seed=str(metadata_index))
                        n_ack += (cap_shift_marks(split_turns, sp.shift_mark_cap)
                                  + (swap_stock_words(split_turns)
                                     if sp.stock_word_swap else 0))
                        n_pol = polish_cs_speech(split_turns, metadata)
                        for i, u in trailing.items():
                            split_turns[i]["utterance"] = u
                        fr, sr, cm, rr = enriched_ratios(split_turns)
                        # ターンの形（短い発話の割合の下限・問いかけの割合の上限）。
                        # 人間コーパスの先頭33ターン: 15字以下 62.8%・疑問 7.8%。
                        shq, quq = turn_shape_ratios(split_turns)
                        short = [f"フィラー{fr*100:.0f}%<{sp.filler_floor*100:.0f}%"
                                 if fr < sp.filler_floor else "",
                                 f"終助詞{sr*100:.0f}%<{sp.sfp_floor*100:.0f}%"
                                 if sr < sp.sfp_floor else "",
                                 f"間{cm:.1f}<{sp.comma_floor:.1f}"
                                 if cm < sp.comma_floor else "",
                                 f"リアクション{rr*100:.0f}%<{sp.react_floor*100:.0f}%"
                                 if rr < sp.react_floor else "",
                                 f"短い発話{shq*100:.0f}%<{sp.short_turn_floor*100:.0f}%"
                                 if (sp.short_turn_floor > 0
                                     and shq < sp.short_turn_floor) else "",
                                 f"問いかけ{quq*100:.0f}%>{sp.question_ratio_cap*100:.0f}%"
                                 if (sp.question_ratio_cap > 0
                                     and quq > sp.question_ratio_cap) else ""]
                        short = [s for s in short if s]
                        # 下限からの不足量（0なら合格）。候補どうしの優劣付けに使う。
                        score = (max(0.0, sp.filler_floor - fr) / max(sp.filler_floor, 1e-9)
                                 + max(0.0, sp.sfp_floor - sr) / max(sp.sfp_floor, 1e-9)
                                 + max(0.0, sp.comma_floor - cm) / max(sp.comma_floor, 1e-9)
                                 + max(0.0, sp.react_floor - rr) / max(sp.react_floor, 1e-9)
                                 + max(0.0, sp.short_turn_floor - shq)
                                 / max(sp.short_turn_floor, 1e-9)
                                 + max(0.0, quq - sp.question_ratio_cap)
                                 / max(sp.question_ratio_cap, 1e-9))
                        if gates["spokenness"] and short and attempt < retry_max - 1:
                            if best is None or score < best[0]:
                                best = (score, [dict(t) for t in split_turns])
                            logger.warning(f"[{metadata_index}] 話し言葉らしさ不足"
                                           f"（{'、'.join(short)}）。リトライ。")
                            last_was_gate, last_reason = True, "話し言葉らしさ不足"
                            continue
                        if best is not None and best[0] < score:
                            split_turns = best[1]
                            fr, sr, cm, rr = enriched_ratios(split_turns)
                            logger.info(f"[{metadata_index}] 最終試行より良かった"
                                        "途中の候補を採用。")
                        logger.info(
                            f"[{metadata_index}] enriched: フィラー{fr*100:.0f}% "
                            f"終助詞{sr*100:.0f}% 間{cm:.1f}/100字 "
                            f"リアクション{rr*100:.0f}%"
                            f"（付加 終助詞{n_add}/受け止めの「ね」{n_rb}"
                            f"、終助詞直し{n_yone}/連続{n_run}"
                            f"、間引き フィラー{n_thin}/終助詞{n_sfp}"
                            f"/単調な頭{n_mono}/笑い{n_laugh}、差し込み{n_inj}"
                            f"/読点{n_brk}、句点化{n_per}、読点外し{n_per_thin}"
                            f"、仕上げ{n_pol}/オウム返し{n_echo}/受け止め{n_ack}"
                            f"、短い発話{shq*100:.0f}%/問いかけ{quq*100:.0f}%"
                            f"、全{len(split_turns)}ターン）")
                    # 主話者の単独の聞き返しを畳む（ゲートの後。畳むと短いターンが減るので、
                    # 話し言葉らしさの下限はその前の形で測る）
                    if sp is not None and (sp.main_echo_cap >= 0 or sp.main_react_fold > 0):
                        mi = metadata.get("main_speaker_index")
                        main = (_main_name(metadata, mi) if mi is not None else "")
                        split_turns, n_fold = fold_main_echoes(split_turns, main, sp.main_echo_cap)
                        split_turns, n_react = fold_main_reactions(split_turns, main, sp.main_react_fold)
                        if n_fold or n_react:
                            logger.info(f"[{metadata_index}] 主話者の聞き返しを{n_fold}件・反応を"
                                        f"{n_react}件畳んだ（全{len(split_turns)}ターン）")
                    # ★2026-11-10 重なりの場面（style: rp）を1つずつ入れる（本生成はふつうの雑談）。指定の場面の
                    #   1つまでの欠けは許す（場面ごとに RP_SCENE_TRIES 回試す）。2つ以上欠けたら作り直す
                    #   （最終試行は1つでも入っていれば採る）。入れたターンは以降の校閲・書き直しで触らない。
                    if rp_on:
                        rp_done, rp_failed, rp_why = await insert_rp_scenes(
                            client, model, split_turns, metadata, persona_of(metadata, persona),
                            cost_tracker, extra_body, metadata_index, system_prompt=system_prompt)
                        logger.info(f"[{metadata_index}] 重なりの場面を入れた: {'・'.join(rp_done) or 'なし'}"
                                    + (f"（入らず: {'・'.join(rp_failed)}）" if rp_failed else "")
                                    + (f" 不採用: {'／'.join(rp_why[:6])}" if rp_why else ""))
                        need = max(1, len(rp_done) + len(rp_failed) - 1)
                        if gates["rp_shape"] and (
                                (len(rp_done) < need and attempt < retry_max - 1) or not rp_done):
                            logger.warning(f"[{metadata_index}] 重なりの場面が{len(rp_done)}つしか入らない。リトライ。")
                            last_was_gate, last_reason = True, "重なりの場面"
                            continue
                        dialogue["overlaps_failed"] = rp_failed
                    # 沈黙の直前の発話は、校閲とその後のオウム返しの間引きでも書き換えさせない。
                    # ★2026-10-24 試走6回目: 校閲が言いさしを「目標は、何かあるんですか？」という問いに
                    #   直し（→ 3.2秒 → 同じ人が答える自問自答）、間引きが「そうなんですか？」に置き換えた
                    #   （→ 3.6秒 → 同じ人「なるほど。」）。沈黙ゲートは校閲の前なので見逃す。
                    pre_pause = ({i: split_turns[i]["utterance"] for i in pre_pause_indices(split_turns)}
                                 if silence is not None else {})
                    # 重なりの場面（rp_v1）のターンは、校閲（形を壊さない直しだけ通す）のあとの文で取っておき、
                    # 校閲のあとの規則の間引き（limit_echoes・rebalance_ack_heads 等）が触らないよう書き戻す（下の rp_after）。
                    # 校閲パス: 採用が決まった候補にだけかける（作り直しの候補にかけると
                    # 1回の生成ごとに LLM 呼び出しが倍になる）。失敗しても対話は残す。
                    for k in range(proofread if pack.proofread_prompt else 0):
                        n_pf, rej = await proofread_dialogue(
                            client, model, split_turns, metadata, pack, cost_tracker,
                            extra_body, metadata_index)
                        logger.info(f"[{metadata_index}] 校閲{k + 1}回目: {n_pf}発話を修正"
                                    + (f"（不採用{len(rej)}: {'／'.join(rej[:3])}）" if rej else ""))
                        if n_pf == 0:
                            break                   # 直すものが無くなったら打ち切る
                    rp_after = {i: split_turns[i]["utterance"] for i in rp_indices(split_turns)}
                    # 校閲は「〜ですか。」を「？」に戻したり、外した冒頭のオウム返しを書き戻したり
                    # する（2026-10-03 実測: 20対話で、文が続く「〜ですか？」17件が校閲後に残った）。
                    # 接客の polish_cs_speech と同じく、校閲の後にもう一度かける。
                    if sp is not None and proofread and pack.proofread_prompt:
                        if sp.echo_fix:
                            fix_echo_fragments(split_turns)
                        limit_echo_heads(split_turns, sp.echo_head_cap)
                        limit_echoes(split_turns, sp.echo_total_cap, sp.echo_replace_once)
                        rebalance_ack_heads(split_turns, sp.ack_head_cap,
                                            sp.ack_phrase_cap,
                                            seed=str(metadata_index))
                        cap_shift_marks(split_turns, sp.shift_mark_cap)
                        if sp.stock_word_swap:
                            swap_stock_words(split_turns)
                    for i, u in pre_pause.items():
                        split_turns[i]["utterance"] = u
                    for i, u in rp_after.items():
                        split_turns[i]["utterance"] = u
                    mi = metadata.get("main_speaker_index")
                    main = (_main_name(metadata, mi) if mi is not None else "")
                    if sp is not None and sp.main_echo_cap >= 0 and proofread and pack.proofread_prompt:
                        # 校閲が聞き返しを書き戻すことがある。畳むとターン番号がずれるので、
                        # 沈黙の直前の発話の番号は畳んだ後に取り直す（反応の畳み込みは1回だけ）
                        split_turns, _ = fold_main_echoes(split_turns, main, sp.main_echo_cap)
                    # 名乗り方（topic_plan.naming）。指示は破られる前提で、余分な名乗りを外す
                    n_name = apply_naming(split_turns, metadata, naming_of(topic_plan_of(metadata)))
                    if n_name:
                        logger.info(f"[{metadata_index}] 名乗りを{n_name}件外した"
                                    f"（{naming_of(topic_plan_of(metadata))}）")
                    # 機械で数えて上限を超えた発話だけを LLM に書き直させる
                    if sp is not None and sp.flag_rewrite:
                        skip = (pre_pause_indices(split_turns) if silence is not None else set())
                        skip = skip | rp_indices(split_turns)      # 重なりの場面（rp_v1）は書き直さない
                        flags = flag_turns(split_turns, metadata, main, sp, skip)
                        if flags:
                            n_rw, rej_rw = await rewrite_flagged_turns(
                                client, model, split_turns, metadata, flags, cost_tracker,
                                extra_body, metadata_index)
                            kinds = Counter(k for ks in flags.values() for k in ks)
                            logger.info(f"[{metadata_index}] 指摘箇所の書き直し: {n_rw}/{len(flags)}発話"
                                        f"（{'・'.join(f'{k}{v}' for k, v in kinds.items())}）"
                                        + (f"（不採用{len(rej_rw)}: {'／'.join(rej_rw[:3])}）"
                                           if rej_rw else ""))
                    # 読みの注記（「方」のかた／ほう等）。本文が確定した後にかける。
                    if yomi and pack.yomi_prompt:
                        n_y, rej_y = await annotate_readings(
                            client, model, split_turns, metadata, pack, cost_tracker,
                            extra_body, metadata_index)
                        logger.info(f"[{metadata_index}] 読みの注記: {n_y}発話"
                                    + (f"（不採用{len(rej_y)}: {'／'.join(rej_y[:3])}）" if rej_y else ""))
                    # 発話全体の読み（語単位のルビ。yomi_ruby。2026-11-12）。本文が確定した後にかける。
                    if yomi_ruby:
                        n_r, n_need = await annotate_ruby(client, model, split_turns, cost_tracker,
                                                          extra_body, yomi_ruby)
                        logger.info(f"[{metadata_index}] 発話全体の読み: {n_r}/{n_need}発話")
                    dialogue["turns"] = split_turns
                    if rp_on:
                        dialogue["overlaps"] = rp_summary(split_turns, main)
                        logger.info(f"[{metadata_index}] 重なりの場面: " + "、".join(
                            f"turn{x['turn']} {x['label']}" for x in dialogue["overlaps"]))
                    if silence is not None:
                        jitter_pauses(split_turns, sil_main, sil_user, silence, str(metadata_index))
                        dialogue["silences"] = silence_summary(split_turns, sil_main, sil_user)
                        logger.info(f"[{metadata_index}] 長い沈黙: " + "、".join(
                            f"turn{x['turn']} {x['kind']} {x['sec']:g}秒"
                            for x in dialogue["silences"]))
                    return {
                        "id": metadata_index,
                        "metadata": metadata,
                        "dialogue": dialogue,
                    }
                else:
                    logger.warning(f"[{metadata_index}] 不正な形式。リトライ。")
                    last_was_gate, last_reason = True, "不正な形式"
            except (json.JSONDecodeError, Exception) as e:
                err_str = str(e)
                if isinstance(e, json.JSONDecodeError) and e.doc:
                    # 崩れた箇所の前後を残す（2026-10-24: 沈黙版で JSON 崩れが 3/20 と多く、原因を追うため）
                    logger.info(f"[{metadata_index}] JSON 崩れの箇所: "
                                f"{e.doc[max(0, e.pos - 120):e.pos + 60]!r}")
                # 例外は接続/レート制限/JSON崩れ等の一時失敗。恒久スキップにはしない。
                last_was_gate, last_reason = False, "error"
                if "401" in err_str or "429" in err_str:
                    delay = max(10, retry_delay * (3 ** attempt))
                    logger.warning(f"[{metadata_index}] 対話生成失敗 (attempt {attempt+1}, 認証/レート制限エラー、{delay:.0f}秒待機): {e}")
                else:
                    # ローカル推論はレート制限がないため、JSON崩れ等の一時エラーで
                    # 指数バックオフを長時間積み上げない（上限30秒）。
                    delay = min(30.0, retry_delay * (2 ** attempt))
                    logger.warning(f"[{metadata_index}] 対話生成失敗 (attempt {attempt+1}, {delay:.0f}秒待機): {e}")
                if attempt < retry_max - 1:
                    await asyncio.sleep(delay)
        logger.error(f"[{metadata_index}] 全リトライ失敗 (最終理由: {last_reason or '不明'})")
        # 品質ゲートを retry_max 回すべて通過できなかった item は、内容起因で
        # 再実行しても直りにくいため恒久スキップ用のマーカーを返す（呼び出し側が
        # skip リストに追記し、以後の pending から除外する）。接続・レート制限・
        # JSON崩れ等の一時失敗は None を返し、再実行で再試行する。
        if last_was_gate:
            return {"__skip__": True, "id": metadata_index, "reason": last_reason}
        return None


async def run_dialogue_generation(config: dict, cost_tracker: CostTracker) -> int:
    # 複数エンドポイント（同一モデルを別GPUに複数インスタンス）に対応。
    # llm.base_urls を指定するとラウンドロビンで負荷分散する（単数 base_url なら1台）。
    clients = build_async_llm_clients(config)
    num_endpoints = len(clients)
    gates = load_gates(config)
    self_name_max = load_self_name_max(config)
    profile = load_prompt_profile(config)
    logger.info(f"プロンプトのプロファイル: {profile}"
                f"（config の dialogue.prompt_profile。既定 auto＝モデル名から判定）")
    pack = load_prompt_pack((config.get("dialogue") or {}).get("prompt_file"))
    proofread = load_proofread(config)
    yomi = load_yomi(config)
    yomi_ruby = load_yomi_ruby(config)
    if yomi_ruby:
        logger.info(f"発話全体の読みを有効化（config の dialogue.yomi_ruby。同時 {yomi_ruby}）")
    persona = load_persona((config.get("metadata") or {}).get("main_speaker"), stage2=True)
    silence = load_silence(config)
    if silence is not None:
        logger.info(f"ユーザの長い沈黙: 1対話{silence.count_min}〜{silence.count_max}回・"
                    f"{silence.sec_min:g}〜{silence.sec_max:g}秒・促し{silence.max_prompt}回まで"
                    "（config の dialogue.silence）")
    if persona:
        logger.info("主話者のペルソナを Stage2 に追加"
                    f"（config の metadata.main_speaker.persona_file）:\n{persona}")
    multi_persona = persona_files((config.get("metadata") or {}).get("main_speaker"))
    if multi_persona:
        logger.info("主話者のペルソナは対話ごと（メタデータの "
                    f"{PERSONA_KEY}。config の metadata.main_speaker.persona_files）: "
                    + "、".join(multi_persona))
    if yomi and pack.yomi_prompt:
        logger.info("読みの注記を有効化（config の dialogue.yomi）")
    if proofread:
        logger.info(f"校閲パスを有効化（{proofread}回。config の dialogue.proofread）"
                    + ("" if pack.proofread_prompt else "。ただしパックに proofread_prompt が無いので実行しない"))
    logger.info(f"Stage2 プロンプトパック: {pack.name}"
                f"（config の dialogue.prompt_file。既定 {DEFAULT_PROMPT_FILE}）")
    off = [k for k, v in gates.items() if not v]
    if off:
        logger.info(f"品質ゲートを無効化: {'、'.join(off)}（config の dialogue.gates）")
    if self_name_max:
        logger.info(f"名乗りを1話者あたり{self_name_max}回まで許可"
                    f"（config の dialogue.self_name_max）")
    sp = S2Params.from_config(config)
    if sp is not None:
        logger.info(f"enriched モード: filler={sp.filler_ratio}(cap {sp.filler_cap_ratio}) "
                    f"sfp={sp.sfp_ratio}(cap {sp.sfp_cap_ratio}) "
                    f"reaction={sp.reaction_ratio}(floor {sp.react_floor}) "
                    f"comma/100={sp.comma_per_100} period_split={sp.period_split}")
    metadata_file = Path(config["metadata"]["output_file"])
    output_file = Path(config["dialogue"]["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # 対話生成は品質ゲート(伏せ字/名前/句点)が厳しく通過に試行を要するため、
    # metadata 段(既定 retry_max_attempts)と分けて dialogue.retry_max_attempts で
    # 上書きできる（未指定なら全体値にフォールバック）。
    retry_max = config.get("dialogue", {}).get(
        "retry_max_attempts", config["retry_max_attempts"])
    retry_delay = config["retry_base_delay"]
    # max_concurrent_requests は「1エンドポイントあたりの並行度」。
    # 全体の同時リクエスト数はエンドポイント数倍にする（各インスタンスを遊ばせない）。
    # llm.ctx_per_request があれば台ごとにサーバの /props から決める（endpoint_concurrency）。
    per_endpoint = endpoint_concurrency(config, clients)
    total_concurrency = sum(per_endpoint)
    semaphore = asyncio.Semaphore(total_concurrency)
    if num_endpoints > 1:
        logger.info(f"エンドポイント {num_endpoints}台でラウンドロビン分散 "
                    f"(各{min(per_endpoint)}〜{max(per_endpoint)}並行 / 合計{total_concurrency}並行): "
                    f"{[str(c[0].base_url) for c in clients]}")

    # メタデータ読み込み（idがないエントリにはUUIDを付与してファイルを更新）
    metadata_list = []
    needs_rewrite = False
    with open(metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                meta = json.loads(line)
                if "id" not in meta:
                    meta["id"] = str(uuid.uuid4())
                    needs_rewrite = True
                metadata_list.append(meta)

    if needs_rewrite:
        logger.info("idのないメタデータにUUIDを付与しました。ファイルを更新します。")
        with open(metadata_file, "w") as f:
            for meta in metadata_list:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    total = len(metadata_list)

    # 既に生成済みのIDをチェック（途中再開対応）
    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    # 品質ゲートを retry_max 回通過できず恒久スキップに記録済みのID。
    # 内容起因で再実行しても直りにくいため、pending から除外して焼き直さない。
    skip_file = output_file.parent / f"{output_file.stem}_skipped.jsonl"
    skipped_ids = set()
    if skip_file.exists():
        with open(skip_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    skipped_ids.add(json.loads(line)["id"])

    # 未生成かつ未スキップのメタデータのみ抽出
    pending = [
        m for m in metadata_list
        if m.get("id") not in existing_ids and m.get("id") not in skipped_ids
    ]
    logger.info(
        f"対話生成開始: 全{total}件中、生成済み{len(existing_ids)}件、"
        f"スキップ済み{len(skipped_ids)}件、未生成{len(pending)}件")
    if multi_persona:
        # persona_files の config なのに、ペルソナを引く前に作ったメタデータ（キーが無い）は
        # 人物設定なしで台本になる。黙って混ざらないよう数を出す。
        n_none = sum(1 for m in pending if not m.get(PERSONA_KEY))
        if n_none:
            logger.warning(f"メタデータ {n_none}件に {PERSONA_KEY} がありません"
                           "（persona_files を入れる前の Stage1）。その対話は主話者の人物設定なしで作ります。")

    # --limit: 試行用に先頭 N 件だけ回す（metadata の順序は固定なので、同じ N なら
    # 別バリアント間・既存の本番出力と同じシナリオが揃う）。
    limit = config.get("limit")
    if limit and len(pending) > limit:
        pending = pending[:limit]
        logger.info(f"--limit {limit}: 先頭{len(pending)}件のみ生成します。")

    if not pending:
        logger.info("未生成のメタデータはありません。")
        return len(existing_ids)

    generated = len(existing_ids)
    done = 0  # このrunで完了処理したタスク数（進捗ログ間引き用）

    # グループバリア（total_concurrency 件を gather で待つ）を廃し、ワーカーが空き次第
    # 次の対話を引く pull 型に変更。担当クライアントは固定 modulo 割当ではなくワーカー
    # 束縛＋共有キューで自然に均等分散され、遅い対話が他エンドポイントを待たせない。
    with open(output_file, "a") as f, open(skip_file, "a") as sf:
        async def process(client, model, extra_body, meta, idx):
            return await generate_single_dialogue(
                client, model, meta, meta["id"],
                cost_tracker, semaphore, retry_max, retry_delay,
                request_interval=config.get("request_interval", 0.0),
                extra_body=extra_body, sp=sp, gates=gates,
                self_name_max=self_name_max, profile=profile, pack=pack,
                proofread=proofread, yomi=yomi, persona=persona,
                silence=silence, yomi_ruby=yomi_ruby,
            )

        async def on_result(result, meta, idx):
            nonlocal generated, done
            if isinstance(result, dict) and result.get("__skip__"):
                # 品質ゲートを retry_max 回通過できなかった item。以後の再実行で
                # 焼き直さないよう skip リストに記録する（生成数は数件減る）。
                sf.write(json.dumps(
                    {"id": meta["id"], "reason": result.get("reason", "")},
                    ensure_ascii=False) + "\n")
                sf.flush()
                logger.warning(
                    f"[{meta['id']}] 品質ゲート{retry_max}回失敗。恒久スキップに記録"
                    f"（理由: {result.get('reason', '')}）。")
            elif result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                generated += 1
            done += 1
            # 進捗は1対話完了ごとに出す。コストログのファイル保存だけ
            # total_concurrency 件ごと（従来のグループ相当の頻度）に間引く。
            s = cost_tracker.summary()
            logger.info(f"進捗: {generated}/{total}件 (このrun {done}/{len(pending)}) | 入力={s['total_input_tokens']:,}tok 出力={s['total_output_tokens']:,}tok | コスト: ${s['total_cost_usd']}")
            if done % total_concurrency == 0:
                cost_tracker.save_log()

        await run_endpoint_worker_pool(
            clients, pending, per_endpoint, process, on_result,
            should_stop=lambda: cost_tracker.exceeded(),
        )
        if cost_tracker.exceeded():
            logger.warning(f"コスト上限 (${cost_tracker.max_cost_usd}) に到達。停止しました。")
        cost_tracker.save_log()

    logger.info(f"対話生成完了: {generated}件")
    return generated
