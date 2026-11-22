"""Stage 2（三人対話）: 対話生成

二人用の generate_dialogues.py を**そのまま使い**、二人を前提にした部分だけを
モジュール属性として差し替えてから gd.run_dialogue_generation を呼ぶ（apply_patches）。
後処理・品質ゲート・校閲・読み注記・ペルソナの段落は二人用と共有する。
差し替えは三人用のプロセス（main_tri.py）の中でしか起きない。

差し替えるもの:
  - build_dialogue_prompt: 末尾ノルマの「2人の話者は必ず交互に発話し、…」の一文を
    三人の発話順の指示に置き換える（元の関数の出力を置換。文言が見つからなければ例外）。
  - _enrich_norm: filler_split の「2人の話者それぞれ」を「3人の話者それぞれ」に。
  - _style_norm: 「混合」の「一方は敬語、もう一方はタメ口」を三人向けに。
  - generate_single_dialogue: 包んで発話順ゲートと宛先の検査を足す（下記）。
  - merge_consecutive_same_speaker: 同じ話者の連続を併合するとき、宛先（addressee）は
    後ろの部分のものを採る（ターンの終わりが次の話者を呼び込むため。「そうなんですね。
    ゴロウさんはどうですか？」の宛先はゴロウ）。
  - _proofread_user_message: 校閲に渡す各行に宛先を「話者→宛先」で見せる
    （校閲が呼びかけの名前と宛先の食い違いを直せるように）。

発話の宛先（addressee。2026-10-12）:
  各ターンに、ほかの話者1人の名前か "both" を書かせる（prompts/tri の「# 最重要: 三人の会話の
  順番と、発話の宛先」）。宛先は次の話者の指定ではない（宛先でない三人目が先に口を挟む流れも
  書かせる）。生成後に turn_order.normalize_addressee でそろえ、読めない宛先が多い
  （tri.addressee.max_invalid を超える）候補は発話順と同じく作り直す。残った読めない宛先は
  推定で埋め、そのターンに addressee_src: inferred を付ける。対話ごとの内訳は
  dialogue.addressee_stats（turn_order.addressee_stats）。

発話順ゲート（turn_order.py）:
  元の generate_single_dialogue（内部のリトライ・後処理・校閲・読みまで済む）の結果を
  turn_order で判定し、閾値を外れたら呼び直す（tri.turn_order.retry 回まで）。
  外れ続けたら、指標が最も良い候補を採る。ただし「1人がほとんど話さない」
  （最少話者の占有率が min_share 未満）候補しか無いときは恒久スキップにする。
  閾値の既定は人間の鼎談（reference/teidan_tri.json）の40ターン窓の値から決めた
  （turn_order.TurnOrderLimits）。
"""

from __future__ import annotations

import dataclasses
import logging
import re

import generate_dialogues as gd
from turn_order import (BOTH, TurnOrderLimits, addressee_stats, normalize_addressee, penalty,
                        turn_order_stats, violations)

logger = logging.getLogger(__name__)

NUM_SPEAKERS = 3

# 二人用 build_dialogue_prompt の末尾ノルマにある交互発話の一文（置き換え元）。
TWO_PARTY_ALTERNATION = "2人の話者は必ず交互に発話し、同じ話者を続けて2ターン置かないでください。"
TRI_ORDER_NORM = (
    "3人の話者全員を会話に参加させ、同じ話者を続けて2ターン置かないでください。"
    "次に話す人は決まった順番で回さず、直前の発話が誰に向いているかで決めてください"
    "（問いかけられた人・名前を呼ばれた人が答える。二人のやりとりが2〜4往復続き、"
    "三人目は言いたいことがあるときに入る。三人が順番に一言ずつ話す並びを続けない）。"
    "各ターンには、その発話を誰に向けて言ったか（addressee: ほかの1人の名前か both）を"
    "書いてください。宛先の人が次に話すとは限りません（宛先でない人が先に口を挟む流れも入れる）。"
)

# 二人用 _enrich_norm の filler_split の文（置き換え元。フィラーのターン数を2人で分ける）。
_SPLIT_TWO = ("このうち{half}ターンずつを2人の話者それぞれに入れてください"
              "（どちらか一方に偏らせない。相手役の話者のフィラーも減らさない）。")
_SPLIT_THREE = ("このうち{third}ターンずつを3人の話者それぞれに入れてください"
                "（だれか1人に偏らせない。主に話す人以外のフィラーも減らさない）。")

_STYLE_MIXED_TWO = "（関係性に合わせて、一方の話者は敬語、もう一方はタメ口のように非対称にする）"
_STYLE_MIXED_THREE = ("（関係性に合わせて、3人のうち一部の人がくだけた話し方をする。"
                      "以前からの知り合いどうしはタメ口、初対面の相手には敬語、のように"
                      "相手によって口調を変えてもよい。誰がどの口調かは対話の途中で変えない）")


def _third_range(lo: int, hi: int) -> str:
    a, b = lo // 3, -(-hi // 3)
    return f"{a}" if a >= b else f"{a}〜{b}"


def _filler_bounds(lo: int, hi: int, sp) -> tuple[int, int]:
    """二人用 _enrich_norm と同じ式で、フィラーを入れるターン数の下限・上限。"""
    n = (lo + hi) // 2
    fq = max(2, round(n * sp.filler_ratio))
    fcap = max(fq + 1, round(n * sp.filler_cap_ratio))
    return fq, fcap


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{where}: 置き換え元の文言が {text.count(old)} 回見つかりました"
                           f"（1回のはず。二人用 generate_dialogues.py の文言が変わった？）: {old!r}")
    return text.replace(old, new)


# 元の関数は apply_patches で控える（二重に差し替えても元を指し続けるように）。
_ORIG: dict = {}


def build_dialogue_prompt(metadata, sp=None, gates=None, profile="qwen", pack=None, persona=""):
    out = _ORIG["build_dialogue_prompt"](metadata, sp, gates, profile, pack, persona)
    return _replace_once(out, TWO_PARTY_ALTERNATION, TRI_ORDER_NORM, "build_dialogue_prompt")


def _enrich_norm(lo, hi, sp):
    out = _ORIG["_enrich_norm"](lo, hi, sp)
    if sp is None or not sp.filler_split:
        return out
    fq, fcap = _filler_bounds(lo, hi, sp)
    old = _SPLIT_TWO.format(half=gd._half_range(fq, fcap))
    return _replace_once(out, old, _SPLIT_THREE.format(third=_third_range(fq, fcap)), "_enrich_norm")


def _style_norm(metadata):
    out = _ORIG["_style_norm"](metadata)
    if _STYLE_MIXED_TWO in out:
        out = out.replace(_STYLE_MIXED_TWO, _STYLE_MIXED_THREE)
    return out


def merge_consecutive_same_speaker(turns: list) -> list:
    """二人用の併合をそのまま使い、併合したターンの宛先を後ろの部分のものにする。"""
    out = _ORIG["merge_consecutive_same_speaker"](turns)
    # 二人用と同じ規則（dict どうしで speaker が同じなら併合）で、併合後の各ターンの最後の宛先を求める
    last: list = []
    for t in turns:
        if not isinstance(t, dict):
            last.append(None)
            continue
        a = t.get("addressee")
        if last and isinstance(last[-1], list) and last[-1][0] == t.get("speaker"):
            if a not in (None, ""):
                last[-1][1] = a
        else:
            last.append([t.get("speaker"), a])
    if len(last) != len(out):          # 二人用の併合規則が変わった（テストで検出する）
        raise RuntimeError("merge_consecutive_same_speaker の併合結果と宛先の対応が取れません")
    for t, g in zip(out, last):
        if isinstance(t, dict) and g is not None and g[1] not in (None, ""):
            t["addressee"] = g[1]
    return out


def _speaker_names(metadata: dict) -> list[str]:
    return [str(s.get("name")) for s in (metadata.get("speakers") or [])
            if isinstance(s, dict) and s.get("name")]


def _proofread_user_message(turns: list, metadata: dict) -> str:
    """二人用の校閲メッセージの各行「i 話者: 発話」を「i 話者→宛先: 発話」にする。"""
    msg = _ORIG["_proofread_user_message"](turns, metadata)
    names = _speaker_names(metadata)
    lines = msg.split("\n")
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        sp = str(t.get("speaker", ""))
        a = normalize_addressee(t.get("addressee"), sp, names)
        if a is None:
            continue
        head = f"{i} {sp}: "
        for k, line in enumerate(lines):
            if line.startswith(head):
                lines[k] = f"{i} {sp}→{a}: " + line[len(head):]
                break
    return "\n".join(lines)


def _infer_addressee(turns: list, i: int, names: list[str]) -> str:
    """読めない宛先の推定: 発話中で呼びかけた名前 → 直前の話者 → both。"""
    t = turns[i]
    sp = t.get("speaker")
    text = str(t.get("utterance", ""))
    called = [n for n in names if n != sp
              and re.search(re.escape(n) + r"(さん|くん|君|ちゃん)?[、。？]", text)]
    if len(called) == 1:
        return called[0]
    if i > 0 and turns[i - 1].get("speaker") not in (None, sp):
        return turns[i - 1]["speaker"]
    return BOTH


def apply_addressees(turns: list, names: list[str]) -> int:
    """宛先をそろえ、読めないものは推定で埋める（addressee_src: inferred）。読めなかった数を返す。"""
    bad = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        a = normalize_addressee(t.get("addressee"), str(t.get("speaker", "")), names)
        if a is None:
            bad.append(i)
        else:
            t["addressee"] = a
            t.pop("addressee_src", None)
    for i in bad:
        turns[i]["addressee"] = _infer_addressee(turns, i, names)
        turns[i]["addressee_src"] = "inferred"
    return len(bad)


# 発話順ゲートの設定（run_dialogue_generation が config から入れる）
_GATE = {"limits": TurnOrderLimits(), "retry": 2, "max_invalid_addressee": 0.1,
         "stats": {"pass": 0, "retry": 0, "best": 0, "skip": 0, "addressee_inferred": 0}}


async def generate_single_dialogue(client, model, metadata, metadata_index, *args, **kwargs):
    """二人用 generate_single_dialogue を包み、発話順ゲートと宛先の検査を足す。"""
    lim: TurnOrderLimits = _GATE["limits"]
    names = [str(s.get("name")) for s in (metadata.get("speakers") or []) if isinstance(s, dict)]
    best = None   # (penalty, result, stats)
    for k in range(_GATE["retry"] + 1):
        res = await _ORIG["generate_single_dialogue"](client, model, metadata, metadata_index,
                                                      *args, **kwargs)
        if res is None or res.get("__skip__"):
            return res          # 一時失敗・内容起因の恒久スキップは二人用の扱いのまま
        turns = res["dialogue"]["turns"]
        n_bad_addr = apply_addressees(turns, names)
        st = turn_order_stats(turns, speakers=names)
        bad = violations(st, lim)
        if n_bad_addr > _GATE["max_invalid_addressee"] * max(len(turns), 1):
            bad.append(f"宛先が読めない {n_bad_addr}/{len(turns)}")
        res["dialogue"]["turn_order"] = {
            "aba_ratio": round(st.aba_ratio, 3), "max_cycle_run": st.max_cycle_run,
            "min_share": round(st.min_share, 3), "attempt": k + 1}
        res["dialogue"]["addressee_stats"] = {**addressee_stats(turns, names),
                                              "inferred": n_bad_addr}
        if not bad:
            _GATE["stats"]["pass"] += 1
            _GATE["stats"]["addressee_inferred"] += n_bad_addr
            ad = res["dialogue"]["addressee_stats"]
            logger.info(f"[{metadata_index}] 発話順: A→B→A {st.aba_ratio:.2f} / "
                        f"順に回る連続 {st.max_cycle_run} / 最少話者 {st.min_share:.2f} / "
                        f"宛先 both {ad['both_ratio']:.0%}・次が宛先の人 "
                        + (f"{ad['next_is_addressee']:.0%}" if ad["next_is_addressee"] is not None else "-")
                        + (f"・推定{n_bad_addr}" if n_bad_addr else "")
                        + (f"（{k + 1}回目で合格）" if k else ""))
            return res
        p = penalty(st, lim) + n_bad_addr / max(len(turns), 1)
        if best is None or p < best[0]:
            best = (p, res, st)
        if k < _GATE["retry"]:
            _GATE["stats"]["retry"] += 1
            logger.warning(f"[{metadata_index}] 発話順が不自然（{'、'.join(bad)}）。作り直し。")
    _, res, st = best
    if st.min_share < lim.min_share:
        _GATE["stats"]["skip"] += 1
        logger.warning(f"[{metadata_index}] 発話順: 1人がほとんど話さない候補しか無い"
                       f"（最少話者 {st.min_share:.2f}）。恒久スキップ。")
        return {"__skip__": True, "id": metadata_index, "reason": "発話順（1人がほぼ話さない）"}
    _GATE["stats"]["best"] += 1
    _GATE["stats"]["addressee_inferred"] += res["dialogue"]["addressee_stats"]["inferred"]
    logger.warning(f"[{metadata_index}] 発話順: 閾値を満たす候補が無いので最良の候補を採用"
                   f"（{'、'.join(violations(st, lim))}）")
    return res


_PATCH_NAMES = ("build_dialogue_prompt", "_enrich_norm", "_style_norm", "generate_single_dialogue",
                "merge_consecutive_same_speaker", "_proofread_user_message")


def apply_patches() -> None:
    """二人用モジュール generate_dialogues の該当関数を三人用に差し替える（何度呼んでもよい）。"""
    if _ORIG:
        return
    for name in _PATCH_NAMES:
        if not callable(getattr(gd, name, None)):
            raise RuntimeError(f"generate_dialogues.{name} が見つかりません（二人用が変わった？）")
        _ORIG[name] = getattr(gd, name)
    gd.build_dialogue_prompt = build_dialogue_prompt
    gd._enrich_norm = _enrich_norm
    gd._style_norm = _style_norm
    gd.generate_single_dialogue = generate_single_dialogue
    gd.merge_consecutive_same_speaker = merge_consecutive_same_speaker
    gd._proofread_user_message = _proofread_user_message


def remove_patches() -> None:
    """差し替えを元に戻す（テスト用。同じプロセスで二人用を動かすときに使う）。"""
    for name, f in _ORIG.items():
        setattr(gd, name, f)
    _ORIG.clear()


async def run_dialogue_generation(config: dict, cost_tracker) -> int:
    md = config.get("metadata") or {}
    if md.get("num_speakers") != NUM_SPEAKERS:
        raise ValueError(f"三人対話の config は metadata.num_speakers: {NUM_SPEAKERS} にしてください"
                         f"（受領: {md.get('num_speakers')}）")
    to = (config.get("tri") or {}).get("turn_order") or {}
    _GATE["limits"] = TurnOrderLimits.from_config(to)
    _GATE["retry"] = int(to.get("retry", 2))
    ad = (config.get("tri") or {}).get("addressee") or {}
    unknown = sorted(set(ad) - {"max_invalid"})
    if unknown:
        raise ValueError(f"tri.addressee に未知のキー: {unknown}（使えるのは ['max_invalid']）")
    _GATE["max_invalid_addressee"] = float(ad.get("max_invalid", 0.1))
    apply_patches()
    logger.info(f"三人対話の Stage2: 発話順ゲート {dataclasses.asdict(_GATE['limits'])}"
                f"・作り直し{_GATE['retry']}回まで（config の tri.turn_order）")
    n = await gd.run_dialogue_generation(config, cost_tracker)
    s = _GATE["stats"]
    logger.info(f"発話順ゲート: 合格{s['pass']} / 作り直し{s['retry']}回 / "
                f"最良候補で採用{s['best']} / 恒久スキップ{s['skip']} / "
                f"推定で埋めた宛先{s['addressee_inferred']}ターン")
    return n
