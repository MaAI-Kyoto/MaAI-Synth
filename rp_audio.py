"""重なりの場面（rp_v1。rp_plan.py）の Stage4 側: 言いかけの発話を文の途中で切る。

言いかけのターン（voice_dialogue.turns の "unsaid" 付き）は、言いかけた部分だけを合成すると文末の下がり調子
（言い切りの抑揚）になる。そこで「言いかけ＋残り」の1文を合成し、言いかけの終わりに当たる位置で音声を切る。

  prepare_cut_turns(record)  合成の前に呼ぶ。text を「言いかけ＋残り」にし、言いかけを cut_text に退避する
                             （合成・テイク選択・読み違いの検査は1文全体で行われる）。
  apply_cuts(record, wavs)   合成の後・配置の前に呼ぶ。言いかけの終わりの位置で波形を切り、text を言いかけに戻す
                             （manifest・書き起こしは言いかけだけになる）。何度呼んでも1回だけ効く。

切る位置: 言いかけと全文のモーラ数の比を、発話の「声のある時間」（ターン内の長いポーズを除いた時間）の上に
取り、その近くのいちばん音の小さい点（言いかけが読点で終わるなら、近くのポーズの頭）に寄せる。
試走3（48件）を ASR で確かめた: 台本の言いかけとの CER 中央値 0.033・残りが聞こえる 0件・最後の1語が欠ける 5件。モーラ数は
pyopenjtalk（Stage4 の venv にある）の読み、無ければ generate_tts.estimate_mora の見積り。
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

CUT_TEXT_KEY = "cut_text"
_SMALL = set("ャュョァィゥェォヮゃゅょぁぃぅぇぉゎ")
_KANA = re.compile(r"[ァ-ヶーぁ-ん]")

try:                                   # Stage4 の venv（.venv-maai-cu12 / cu129）にある
    import pyopenjtalk as _pjt
except Exception:                      # pragma: no cover - 無い環境は見積りで代える
    _pjt = None


def mora_count(text: str) -> float:
    """発話のモーラ数（読み）。"""
    text = str(text or "")
    if not text.strip():
        return 0.0
    if _pjt is not None:
        try:
            kana = _pjt.g2p(text, kana=True)
            return float(sum(1 for c in kana if _KANA.match(c) and c not in _SMALL))
        except Exception:
            pass
    from generate_tts import estimate_mora
    return float(estimate_mora(text)[0])


def prepare_cut_turns(record: dict) -> int:
    """合成の前: 言いかけのターンの text を「言いかけ＋残り」にする（言いかけは cut_text に退避）。"""
    n = 0
    for t in (record.get("voice_dialogue") or {}).get("turns") or []:
        if not t.get("unsaid") or CUT_TEXT_KEY in t or "cut_sec" in t:
            continue
        spoken = str(t.get("text") or "")
        t[CUT_TEXT_KEY] = spoken
        t["text"] = spoken.rstrip() + str(t["unsaid"]).strip()
        n += 1
    return n


def _frames(w: np.ndarray, sr: int, hop_s: float = 0.010, win_s: float = 0.025):
    hop, win = max(1, int(hop_s * sr)), max(1, int(win_s * sr))
    n = max(1, 1 + (len(w) - win) // hop) if len(w) >= win else 1
    rms = np.empty(n, dtype=np.float64)
    for k in range(n):
        seg = w[k * hop:k * hop + win]
        rms[k] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0
    return rms, hop


def cut_sample(wav: np.ndarray, sr: int, spoken: str, full: str,
               min_head_sec: float = 0.25, min_tail_sec: float = 0.12) -> tuple[int, dict]:
    """言いかけの終わりに当たるサンプル位置と、その求め方の記録を返す。"""
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    rms, hop = _frames(w, sr)
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    voiced = db > (db.max() - 35.0)
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return len(w), {"why": "無音"}
    on, off = int(idx[0]), int(idx[-1]) + 1
    # ターン内の長いポーズ（120ms 以上の無声の連続）を「声のある時間」から除く
    keep = voiced[on:off].copy()
    run = 0
    for k in range(len(keep)):
        if not keep[k]:
            run += 1
        else:
            if 0 < run < 12:
                keep[k - run:k] = True        # 短い無声（子音・つなぎ）は声のある時間に数える
            run = 0
    cum = np.cumsum(keep)
    total = float(cum[-1]) if len(cum) else 0.0
    m_sp, m_all = mora_count(spoken), mora_count(full)
    ratio = (m_sp / m_all) if m_all > 0 else (len(spoken) / max(1, len(full)))
    ratio = float(min(max(ratio, 0.0), 1.0))
    target = ratio * total
    k_est = on + int(np.searchsorted(cum, target))
    info = {"ratio": round(ratio, 3), "mora": [m_sp, m_all]}
    # 寄せる: 言いかけが読点で終わるなら近くのポーズ（60ms 以上の無声）の頭、それ以外は ±80ms で最も小さい点
    k_cut = k_est
    if str(spoken).rstrip().endswith(("、", ",")):
        lo, hi = max(on, k_est - 35), min(off, k_est + 35)
        best = None
        k = lo
        while k < hi:
            if not voiced[k]:
                j = k
                while j < off and not voiced[j]:
                    j += 1
                if j - k >= 6 and (best is None or abs(k - k_est) < abs(best - k_est)):
                    best = k
                k = j
            else:
                k += 1
        if best is not None:
            k_cut, info["snap"] = best + 2, "pause"
    if "snap" not in info:
        # ★2026-11-10 試走3の ASR（48件）: 残りが聞こえる切り遅れ 0件・最後の1語が欠ける切り早め 5件
        #   （書き起こしにある語が音に無い）→ 探す範囲を遅い側に寄せる（−40ms〜+120ms）
        lo, hi = max(on, k_est - 4), min(off, k_est + 13)
        if hi > lo:
            k_cut = lo + int(np.argmin(rms[lo:hi]))
            info["snap"] = "dip"
    n = k_cut * hop
    n = max(n, int((on * hop) + min_head_sec * sr))          # 少なくとも少しは声を残す
    n = min(n, max(int((on * hop) + min_head_sec * sr), int(off * hop - min_tail_sec * sr)))
    n = int(min(max(n, 1), len(w)))
    info["est_sec"] = round(k_est * hop / sr, 3)
    return n, info


# ── ASR で切り位置を直す（2026-11-10 試走7）─────────────────────────────
# 試走7の ASR（55件）: 末尾が欠ける 6件（うち5件は言いかけに「えっと、あの」などのフィラー。フィラーは1モーラあたりが
# 長いので、モーラの比で置くと早く切る）。切った音を ASR にかけ、台本の全文（カナ）のどこまで読まれたかを編集距離で
# 合わせて、言いかけとの差のカナ数ぶん動かす（2回まで）。ASR は読み違いの検査と同じもの（reading_check.get_asr）。
_FILLER_KANA = re.compile(r"エーット|エット|エート|エー|アノー|アノ|マア|ナンカ|ウーン|ンー")


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def heard_prefix_len(hyp_kana: str, full_kana: str, around: int, span: int = 20) -> int:
    """ASR の読み（カナ）が、台本の全文（カナ）の先頭何文字ぶんに当たるか（around の前後 span で探す）。"""
    best, best_key = around, None
    for L in range(max(0, around - span), min(len(full_kana), around + span) + 1):
        key = (_lev(hyp_kana, full_kana[:L]), abs(L - around))
        if best_key is None or key < best_key:
            best, best_key = L, key
    return best


def _asr_kana(text: str) -> str:
    from reading_check import kana
    return _FILLER_KANA.sub("", kana(text))


def refine_cut_with_asr(wav: np.ndarray, sr: int, n0: int, spoken: str, full: str, asr,
                        max_iter: int = 6, max_shift_kana: int = 10) -> tuple[int, dict]:
    """ASR で確かめて切り位置を直す。(新しい位置, 記録)。ASR が無い・読みが取れないときは n0 のまま。

    試した位置で「早すぎ／遅すぎ」の範囲を狭め、次の推定が範囲の外に出たら中点を取る（1回目の修正が行き過ぎて
    次で戻る振動を防ぐ。試走7の比べ合いで d=[10, -10] が出た）。試した中でずれの最も小さい位置を採る（同じなら先に試した側
    ＝最初はモーラ比の推定。ずれが上限に張り付いたまま遅い側を採ると、切らずに全文を残すことがあった）。
    """
    try:
        ks, kf = _asr_kana(spoken), _asr_kana(full)
    except Exception:
        return n0, {}
    if not ks or not kf or asr is None:
        return n0, {}
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    rms, hop = _frames(w, sr)
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    voiced = db > (db.max() - 35.0)
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return n0, {}
    on, off = int(idx[0]), int(idx[-1]) + 1
    sec_per_kana = (off - on) * hop / sr / max(1, len(kf))
    lo_b, hi_b = on * hop + int(0.25 * sr), len(w)
    n, log, tried = n0, [], []
    for _ in range(max_iter):
        try:
            hyp = asr([w[:n]], sr)[0]
            kh = _FILLER_KANA.sub("", __import__("reading_check").kana(hyp))
        except Exception:
            break
        L = heard_prefix_len(kh, kf, len(ks))
        d = max(-max_shift_kana, min(max_shift_kana, len(ks) - L))
        log.append({"hyp": hyp, "d": d})
        tried.append((abs(d), len(tried), n))
        if abs(d) <= 1:
            break
        if d > 0:
            lo_b = max(lo_b, n + 1)
        else:
            hi_b = min(hi_b, n - 1)
        if lo_b >= hi_b:
            break
        target = n + d * sec_per_kana * sr
        if not lo_b < target < hi_b:
            target = (lo_b + hi_b) / 2
        k = int(target / hop)
        lo, hi = max(on, k - 4), min(off, k + 7)                 # 少し遅い側に寄せて最小音量点へ
        if hi > lo:
            k = lo + int(np.argmin(rms[lo:hi]))
        n_new = int(min(max(k * hop, lo_b), hi_b))
        if n_new == n:
            break
        n = n_new
    best = min(tried)[2] if tried else n0
    return best, {"asr_fix": log} if log else {}


def _default_asr():
    """Stage4 のワーカーで読み違いの検査に使っている ASR（同じものを使い回す）。無ければ None。"""
    try:
        import torch
        from reading_check import get_asr
        return get_asr("kotoba-tech/kotoba-whisper-v2.0", "cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


def snap_to_quiet(wav: np.ndarray, sr: int, n: int, back_sec: float = 0.04, fwd_sec: float = 0.06) -> int:
    """切る位置を前後の最も音の小さい点（10ms 刻み）に寄せる。

    ★2026-11-11 試走8: 切り口の直前30msが声の中央値から -6dB より大きい（声の途中で切っている）のが 62件中31件。
    人の割り込まれた側は語の切れ目で止まりやすく、声の途中でブツッと切れると編集の跡に聞こえる。
    """
    w = wav if wav.ndim == 1 else wav.mean(axis=1)
    hop, win = int(0.01 * sr), int(0.02 * sr)
    lo, hi = max(win, n - int(back_sec * sr)), min(len(w), n + int(fwd_sec * sr))
    if hi - lo < hop:
        return n
    best, best_e = n, None
    for c in range(lo, hi + 1, hop):
        e = float(np.mean(w[c - win:c].astype(np.float64) ** 2))
        if best_e is None or e < best_e:
            best, best_e = c, e
    return int(best)


def apply_cuts(record: dict, wav_by_turn: dict, sr: int, failed_turns=None,
               fade_sec: float = 0.06, asr="auto") -> int:
    """合成の後・配置の前: 言いかけのターンの波形を言いかけの終わりで切り、text を言いかけに戻す。

    切ったターンには cut_sec（切ったあとの長さ）と cut（求め方の記録）を残す。合成に失敗したターンは
    切らずに text だけ戻す。切った数を返す。
    """
    failed_turns = failed_turns or set()
    n_cut = 0
    turns = (record.get("voice_dialogue") or {}).get("turns") or []
    if asr == "auto":
        asr = _default_asr() if any(CUT_TEXT_KEY in t for t in turns) else None
    for i, t in enumerate(turns):
        if CUT_TEXT_KEY not in t:
            continue
        spoken = str(t.pop(CUT_TEXT_KEY))
        full = str(t.get("text") or "")
        t["text"] = spoken.rstrip().rstrip("、,")
        w = wav_by_turn.get(i)
        if w is None or len(w) == 0 or i in failed_turns:
            continue
        n, info = cut_sample(w, sr, spoken, full)
        if asr is not None:
            n2, fix = refine_cut_with_asr(w, sr, n, spoken, full, asr)
            if fix:
                info.update(fix, est_cut_sec=round(n / sr, 3))
            n = n2
        n_snap = snap_to_quiet(w, sr, n)
        if n_snap != n:
            info["snap_quiet_ms"] = int(round((n_snap - n) / sr * 1000))
            n = n_snap
        out = np.array(w[:n], dtype=np.float32, copy=True)
        f = min(int(fade_sec * sr), len(out))
        if f > 0:
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, f)))
            if out.ndim == 1:
                out[-f:] *= ramp.astype(np.float32)
            else:
                out[-f:] *= ramp[:, None].astype(np.float32)
        wav_by_turn[i] = out
        t["cut_sec"] = round(n / sr, 3)
        t["cut"] = {**info, "full_sec": round(len(w) / sr, 3)}
        n_cut += 1
    if n_cut:
        logger.info(f"[{record.get('id', '')}] 言いかけ {n_cut}ターンを文の途中で切った（rp_v1）")
    return n_cut
