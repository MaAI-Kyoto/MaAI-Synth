"""三人対話のリタイム・相槌の候補走査を、二人用の二者モデル（VAP / 相槌予測）で行う。

二人用（retime_maai.MaaiRetimer / refine_maai.InsertionScanner）は、モデルに渡す2本の
チャネルを ScoreTimeline.slice() から取り、「相手」を 1 - ch で決めている。三人の対話では
判断のたびに、3本のチャネルを「注目する話者」と「それ以外の全員の和」の2本に組み替えて
同じモデル・同じ判断のコード（_decide_vap / _decide_hold / _decide_bc・ハザード・潜時）に渡す。

  - 話者交替（次の話者 N が入る時刻）: (N, ほか2人の和) で p_now[N]
  - 同じ話者の続き（hold）:           (本人, ほか2人の和)
  - 台本上の相槌ターン（聞き手 L）:    (L 以外の全員の和, L) で p_bc
  - 挿入相槌の候補走査:               聞き手ごとに (その聞き手以外の全員の和, その聞き手) で p_bc。
                                      2人の聞き手はそれぞれ独立に候補を持つ
人間の鼎談（非公開のコーパス）で、話し手の2秒以上の発話に入る相槌は、話し手が応答している
相手 49%・三人目 51% とほぼ半々で、両方の聞き手から入る発話が 29%、聞き手どうしの相槌の 23% は
0.3秒以内に重なる。聞き手ごとに独立に打たせるのはこの実測に合わせたもの。

★2本への並べ方: 注目する話者のチャネルが 0 なら (ch0, ほか)、それ以外なら (ほか, 注目) にする
  （_slot）。こうすると2話者の対話では常に (ch0, ch1) がそのまま渡り、二人用と完全に同じ入力・
  同じ判断になる（tests/test_tri.py が二人用の plan() / scan() と一致することを確かめている）。
★「ほか2人の和」に2人の声が同時に入るのは、モデルが学習していない入力。台本の三人対話では
  ほか2人が同時に話す場面はまれなので、影響は小さいと見ている（試走の聴き取りで確かめる）。

plan() と scan() の本体は二人用の写しで、チャネルの扱いだけを一般化してある。
二人用の plan() / scan() を変えたら、ここも合わせること（2話者での一致テストが落ちる）。
"""

from __future__ import annotations

import logging
import re

import numpy as np

from refine_maai import InsertionScanner, _text_core, greeting_turns, snap_char_pos
from retime_maai import MAAI_SR, MaaiRetimer, ScoreTimeline, _core_len

logger = logging.getLogger(__name__)


def _slot(focus: int) -> int:
    """注目する話者を2本のうちどちらに置くか（ch0 なら 0、それ以外は 1）。"""
    return 0 if focus == 0 else 1


class TriScoreTimeline(ScoreTimeline):
    """チャネル数ぶんを持つスコア用タイムライン。view(focus) で2本に組み替えて渡す。"""

    def slice_all(self, t0: float, t1: float) -> np.ndarray:
        """[t0, t1) 秒の全チャネル (samples, num_ch)。確定済み末尾より先はゼロ。"""
        s0 = max(0, int(round(t0 * MAAI_SR)))
        s1 = max(s0, int(round(t1 * MAAI_SR)))
        out = np.zeros((s1 - s0, self.num_ch), dtype=np.float32)
        avail = min(self._len, s1)
        if avail > s0:
            out[:avail - s0] = self._buf[s0:avail]
        return out

    def pair(self, out: np.ndarray, focus: int) -> tuple[np.ndarray, np.ndarray]:
        """全チャネルから (注目, ほかの和) を _slot の順で返す。ほかが1本ならその列そのもの。"""
        others = [c for c in range(self.num_ch) if c != focus]
        rest = out[:, others[0]] if len(others) == 1 else out[:, others].sum(axis=1)
        f = out[:, focus]
        return (f, rest) if _slot(focus) == 0 else (rest, f)

    def view(self, focus: int) -> "_FocusView":
        return _FocusView(self, focus)


class _FocusView:
    """二人用の判断関数に渡す、2本に組み替えたタイムライン（slice だけを持つ）。"""

    def __init__(self, tl: TriScoreTimeline, focus: int):
        self.tl = tl
        self.focus = focus

    @property
    def _len(self) -> int:
        return self.tl._len

    def slice(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        return self.tl.pair(self.tl.slice_all(t0, t1), self.focus)


def _proj_channel_of(channel_of: dict, focus_speaker: str) -> dict:
    """_decide_bc に渡す2本ぶんの channel_of（注目する話者が _slot、ほかの全員がもう一方）。"""
    s = _slot(channel_of[focus_speaker])
    return {name: (s if name == focus_speaker else 1 - s) for name in channel_of}


class TriVapRetimer(MaaiRetimer):
    """二人用の MaaiRetimer を、話者2人以上の対話で使えるようにしたもの。"""

    def plan(self, record: dict, turns: list, wav_by_turn: dict, sr: int,
             channel_of: dict, failed_turns: set | None = None) -> list[dict] | None:
        """全ターンの配置を決める（二人用 MaaiRetimer.plan の写し。チャネルの扱いだけ一般化）。"""
        n_ch = len(set(channel_of.values()))
        if n_ch < 2:
            return None
        failed_turns = failed_turns or set()
        cfg = self.cfg
        self._seed_rng(record.get("id", ""))
        frame_dur = 1.0 / cfg.frame_rate
        timeline = TriScoreTimeline(num_ch=n_ch, sr_engine=sr)

        plans: list[dict] = []
        placed: dict[int, tuple[float, float]] = {}
        frontier = 0.0
        frontier_idx = -1
        last_end_on_channel = {c: 0.0 for c in range(n_ch)}
        last_longbc_end = {c: 0.0 for c in range(n_ch)}
        last_speech_end_by_speaker: dict[str, float] = {}
        prev_speech_speaker: str | None = None
        prev_placed_end = 0.0
        legacy_prev_end = 0.0

        for i, t in enumerate(turns):
            sp = t["speaker"]
            ch = channel_of[sp]
            wav = wav_by_turn[i]
            dur = len(wav) / sr
            ttype = t.get("type", "speech")
            if ttype not in ("speech", "backchannel"):
                ttype = "speech"
            llm_off = 0.0 if i == 0 else float(t.get("start_offset", 0.0) or 0.0)
            llm_start = max(0.0, legacy_prev_end + llm_off)

            p_peak = None
            anchor_sec = prev_placed_end
            vap_detail: dict = {}
            if i == 0:
                start, method = 0.0, "first"
            elif i in failed_turns:
                start, method = self._fallback_start(prev_placed_end, llm_off), "llm-synthfailed"
            elif ttype == "backchannel":
                start, method, p_peak, anchor_sec = self._decide_bc(
                    i, t, turns, placed, timeline.view(ch), failed_turns,
                    prev_placed_end, llm_off, _proj_channel_of(channel_of, sp),
                    frame_dur, dur)
            elif t.get("continued") and sp in last_speech_end_by_speaker:
                anchor_sec = last_speech_end_by_speaker[sp]
                start, method = anchor_sec + cfg.continued_gap, "continued"
            elif sp == prev_speech_speaker:
                hold_anchor = last_speech_end_by_speaker.get(sp, prev_placed_end)
                start, method, p_peak, anchor_sec = self._decide_hold(
                    _slot(ch), timeline.view(ch), hold_anchor, i - 1, failed_turns,
                    prev_placed_end, llm_off, frame_dur)
            else:
                self._vap_detail = {}
                start, method, p_peak, anchor_sec = self._decide_vap(
                    _slot(ch), timeline.view(ch), frontier, frontier_idx, failed_turns,
                    prev_placed_end, llm_off, frame_dur)
                vap_detail = self._vap_detail

            # ほかの話者が再生中の長い相槌を、実質 speech が踏み潰さない（二人用は相手ch1本）
            bc_waited = False
            if (cfg.repair.enabled and ttype == "speech"
                    and _core_len(t.get("text", "")) > cfg.repair.min_speech_len):
                lim = max(last_longbc_end[c] for c in range(n_ch) if c != ch) \
                    - cfg.repair.max_speech_overlap
                if start < lim - 1e-6:
                    start = lim
                    bc_waited = True

            clamped = False
            if start < last_end_on_channel[ch] - 1e-6:
                start = last_end_on_channel[ch]
                clamped = True
            start = max(0.0, start)

            end = start + dur
            plans.append({
                "index": i,
                "start_sec": round(start, 3),
                "retime": {
                    "method": method,
                    "anchor_sec": round(anchor_sec, 3),
                    "llm_start_sec": round(llm_start, 3),
                    "decided_offset": round(start - anchor_sec, 3),
                    "p_peak": round(p_peak, 3) if p_peak is not None else None,
                    "clamped": clamped,
                    **vap_detail,
                },
            })
            if bc_waited:
                plans[-1]["retime"]["bc_waited"] = True
                self.method_counts["bc-waited"] = \
                    self.method_counts.get("bc-waited", 0) + 1
            self.method_counts[method] = self.method_counts.get(method, 0) + 1

            timeline.place(ch, start, wav)
            placed[i] = (start, end)
            if end > frontier:
                frontier, frontier_idx = end, i
            last_end_on_channel[ch] = max(last_end_on_channel[ch], end)
            if (ttype == "backchannel"
                    and _core_len(t.get("text", "")) > cfg.repair.min_speech_len):
                last_longbc_end[ch] = max(last_longbc_end[ch], end)
            if ttype == "speech":
                last_speech_end_by_speaker[sp] = end
                prev_speech_speaker = sp
            prev_placed_end = end
            legacy_prev_end = llm_start + dur

        self._repair_collisions(plans, turns, wav_by_turn, sr, channel_of)
        return plans

    def _repair_collisions(self, plans: list, turns: list, wav_by_turn: dict,
                           sr: int, channel_of: dict):
        """二人用の写し。「相手」をほかの全チャネルの最も遅い末尾にしただけ。"""
        rp = self.cfg.repair
        if not rp.enabled or not plans:
            return
        starts = {p["index"]: float(p["start_sec"]) for p in plans}
        durs = {p["index"]: len(wav_by_turn[p["index"]]) / sr for p in plans}
        chans = sorted(set(channel_of.values()))

        order = sorted((p["index"] for p in plans), key=lambda i: starts[i])
        last_sub_end = {c: -1e9 for c in chans}
        repaired: dict[int, dict] = {}
        for pos, i in enumerate(order):
            t = turns[i]
            ch = channel_of[t["speaker"]]
            if _core_len(t.get("text", "")) <= rp.min_speech_len:
                continue
            s = starts[i]
            if t.get("type", "speech") == "speech":
                other_end = max(last_sub_end[c] for c in chans if c != ch)
                allowed = other_end - rp.max_speech_overlap
                if s < allowed - 1e-6:
                    delta = allowed - s
                    repaired[i] = {"shift_sec": round(delta, 3),
                                   "overlap_before": round(other_end - s, 3)}
                    for j in order[pos:]:
                        starts[j] += delta
                    s = starts[i]
            last_sub_end[ch] = max(last_sub_end[ch], s + durs[i])

        if repaired:
            logger.info(f"衝突修復: {len(repaired)}件の speech 大被りを後ろにずらしました。")
        for p in plans:
            i = p["index"]
            p["start_sec"] = round(starts[i], 3)
            if i in repaired:
                p["retime"]["repaired"] = repaired[i]
                self.method_counts["repaired"] = self.method_counts.get("repaired", 0) + 1


def tri_build_timeline(turns: list, plans: list, wav_by_turn: dict, sr: int,
                       channel_of: dict) -> tuple[TriScoreTimeline, float]:
    """refine_maai.build_timeline の写し（チャネル数ぶんのタイムラインを作る）。"""
    tl = TriScoreTimeline(num_ch=len(set(channel_of.values())), sr_engine=sr)
    end_max = 0.0
    for p in plans:
        i = p["index"]
        start = float(p["start_sec"])
        tl.place(channel_of[turns[i]["speaker"]], start, wav_by_turn[i])
        end_max = max(end_max, start + len(wav_by_turn[i]) / sr)
    return tl, end_max


class TriInsertionScanner(InsertionScanner):
    """挿入相槌の候補を聞き手ごとに求める（二人用 InsertionScanner.scan の写し）。

    走査は聞き手の数だけ（2話者=2方向、3話者=3方向）。各方向は
    (その聞き手以外の全員の和, その聞き手)。候補の間隔・既存相槌からの距離・1ターンの
    件数の制約は、聞き手ごとに掛ける（聞き手どうしの同時の相槌は許す）。
    """

    def scan(self, timeline: TriScoreTimeline, turns: list, plans: list,
             wav_by_turn: dict, sr: int, channel_of: dict,
             failed_turns: set) -> list[dict]:
        ins = self.cfg.insert
        if not ins.enabled:
            return []
        speakers_by_ch = {ch: sp for sp, ch in channel_of.items()}
        chans = sorted(speakers_by_ch)
        dur_total = timeline._len / MAAI_SR
        if dur_total <= 0:
            return []
        out = timeline.slice_all(0.0, dur_total)
        scorer = self.retimer._bc_scorer()
        # c = 聞き手（相槌側）チャネル。引数は (話し手, 相槌側) の順（二人用と同じ規約）。
        # ★列は1回だけ作って使い回す。vapbatch の score_long_many は配列の id で
        #   「同じ配列は1回しかエンコードしない」ので、方向ごとに切り出すとエンコードが増える
        #   （2話者なら二人用と同じ2回、3話者なら各チャネル3本＋「ほかの和」3本の6回）。
        cols = {c: out[:, c] for c in chans}
        pairs = []
        for c in chans:
            others = [k for k in chans if k != c]
            rest = cols[others[0]] if len(others) == 1 else out[:, others].sum(axis=1)
            pairs.append((rest, cols[c]))
        if hasattr(scorer, "score_long_many"):
            frames_by_c = scorer.score_long_many(pairs)
        else:
            frames_by_c = [scorer.score(*xs) for xs in pairs]
        pbc, times = {}, {}
        for k, c in enumerate(chans):
            frames = frames_by_c[k]
            times[c] = np.asarray([e for e, _ in frames], dtype=float) / MAAI_SR
            pbc[c] = np.asarray(
                [self.retimer._bc_signal(fr) or 0.0 for _, fr in frames])

        start_by_idx = {p["index"]: float(p["start_sec"]) for p in plans}
        bc_times = {c: [] for c in chans}
        for i, t in enumerate(turns):
            if t.get("type", "speech") == "backchannel" and i in start_by_idx:
                bc_times[channel_of[t["speaker"]]].append(start_by_idx[i])

        greet = greeting_turns(turns) if ins.skip_greetings else set()
        slots = []
        for p in plans:
            i = p["index"]
            t = turns[i]
            if t.get("type", "speech") == "backchannel" or i in failed_turns:
                continue
            if i in greet:
                continue
            d = len(wav_by_turn[i]) / sr
            if d < ins.min_turn_sec:
                continue
            sp_ch = channel_of[t["speaker"]]
            for listener_ch in chans:
                if listener_ch == sp_ch:
                    continue
                listener = speakers_by_ch[listener_ch]
                # 聞き手が原本でこのターン以前に発話済み（チャネル順の保護。2話者なら
                # 二人用の「話し手以外が発話済み」と同じ。3話者では、まだ話していない人の
                # 相槌が先に入って出現順＝チャネル順が変わるのを防ぐ）
                if not any(turns[j]["speaker"] == listener for j in range(i)):
                    continue
                s0 = start_by_idx[i] + ins.edge_margin_sec[0]
                s1 = start_by_idx[i] + d - ins.edge_margin_sec[1]
                if s1 <= s0:
                    continue
                m = (times[listener_ch] >= s0) & (times[listener_ch] <= s1)
                seg_p, seg_t = pbc[listener_ch][m], times[listener_ch][m]
                if not len(seg_p):
                    continue
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
                        "listener": listener,
                    })
        return slots


class TriMockScanner:
    """--dry-run 用（二人用 MockScanner の写し）。聞き手は話し手以外で最初に話した人。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def scan(self, timeline, turns, plans, wav_by_turn, sr, channel_of,
             failed_turns) -> list[dict]:
        for p in plans:
            i = p["index"]
            t = turns[i]
            if t.get("type", "speech") == "backchannel" or i in (failed_turns or set()):
                continue
            d = len(wav_by_turn[i]) / sr
            if d < self.cfg.insert.min_turn_sec:
                continue
            others = [turns[j]["speaker"] for j in range(i) if turns[j]["speaker"] != t["speaker"]]
            if not others:
                continue
            return [{"slot_id": 1, "anchor_index": i, "rel_pos": 0.5,
                     "t_peak": round(float(p["start_sec"]) + d / 2, 2), "p": 0.5,
                     "char_pos": snap_char_pos(t["text"], 0.5),
                     "listener": others[0]}]
        return []
