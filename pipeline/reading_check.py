"""合成音声の読みを ASR で確かめる共通部品（check_tts_readings.py と generate_tts の引き直しで使う）。

ASR（既定 kotoba-whisper-v2.0）で聞き取った文と台本を、どちらも pyopenjtalk でカナの読みに
直して突き合わせる。ASR は聞こえた音を別の漢字に当てることがあるので、表記ではなく読みで比べる。
採点は「置き換え（別の音に聞こえた）」の割合 sub。ASR の取りこぼし（削除）は数えない
（短い頭の語を落とす癖があり、読み間違いとは関係ないため）。

依存: torch / transformers（ASR）、pyopenjtalk（pyopenjtalk-plus）。無ければ get_asr が None を返す。
実測（評価版100対話・3,363発話）: 真の読み間違い（表書き・内のし・一足・承ります・お電話）は
sub 0.1〜0.3、ASR の揺れ（すみません→すいません等）は 0.03〜0.07。
"""
from __future__ import annotations

import difflib
import functools
import logging
import re
import unicodedata

import numpy as np

logger = logging.getLogger(__name__)

ASR_SR = 16000
_PUNCT = re.compile(r"[、。，．,.？！?!「」『』（）()\s・…ー―〜~’']")
# 合成入力に付くタグ（fish の感情タグ・話者タグ）
_TAGS = re.compile(r"<\|[^|]*\|>|\[[^\]]*\]|\([^)]*\)|（[^）]*）")


def _g2p():
    import pyopenjtalk
    return pyopenjtalk


def kana(text: str) -> str:
    """文をカタカナの読みにする（記号・長音・タグは落とす。数字は tts_reading の規則で開く）。"""
    import tts_reading
    text = _TAGS.sub("", text or "")
    text = tts_reading.normalize_numbers(unicodedata.normalize("NFKC", text))
    k = _g2p().g2p(text, kana=True)
    # pyopenjtalk は同じ「シー」を文脈で「スィ」とも「シ」とも読むので寄せる
    k = k.replace("スィ", "シ").replace("ズィ", "ジ")
    return _PUNCT.sub("", k)


# ASR は英字・数字を半角で書く（「ブイシー、ゼロ、ナナ、ゼロ」→「VC070」）。台本側は読みを
# かなに開いてあるので、ASR 側も英字は文字名に、数字は「数として」と「1桁ずつ」の両方で開き、
# 台本に近い方で比べる（そうしないと型番・番号のターンが毎回誤検知になる）。
_LETTER = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   ["エー", "ビー", "シー", "ディー", "イー", "エフ", "ジー", "エイチ", "アイ", "ジェー",
                    "ケー", "エル", "エム", "エヌ", "オー", "ピー", "キュー", "アール", "エス", "ティー",
                    "ユー", "ブイ", "ダブリュー", "エックス", "ワイ", "ゼット"]))


_TSU = {"1": "ひとつ", "2": "ふたつ", "3": "みっつ", "4": "よっつ", "5": "いつつ",
        "6": "むっつ", "7": "ななつ", "8": "やっつ", "9": "ここのつ"}


def _hyp_variants(hyp: str) -> list[str]:
    import tts_reading
    h = unicodedata.normalize("NFKC", hyp or "")
    # ASR の表記の癖（「すいません」「2つ」）を台本側の書き方に寄せる
    h = h.replace("すいません", "すみません").replace("スイマセン", "すみません")
    h = re.sub(r"(?<![0-9])([1-9])つ", lambda m: _TSU[m.group(1)], h)
    h = re.sub(r"[A-Za-z]", lambda m: _LETTER[m.group(0).upper()], h)
    out = [h]
    if re.search(r"[0-9]", h):
        out.append(re.sub(r"[0-9]+", lambda m: tts_reading.read_digits(m.group(0)), h))
    return out


def word_readings(text: str) -> list[tuple[str, str]]:
    """文を (表層, 読み) の語の列にする。読み間違いの語を特定するのに使う。"""
    out = []
    for w in _g2p().run_frontend(unicodedata.normalize("NFKC", _TAGS.sub("", text or ""))):
        surf = w.get("string", "")
        pron = _PUNCT.sub("", (w.get("pron", "") or w.get("read", "")))
        if surf and not _PUNCT.fullmatch(surf):
            out.append((surf, pron))
    return out


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes()
               if op != "equal") / len(ref)


def sub_rate(ref: str, hyp: str) -> float:
    """置き換え（別の音に聞こえた）だけの割合。"""
    if not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(i2 - i1 for op, i1, i2, j1, j2 in sm.get_opcodes() if op == "replace") / len(ref)


def diff_words(text: str, got: str) -> list[str]:
    """台本の語ごとの読みと聞こえた読みを突き合わせ、置き換わった語を「語:期待→聞こえた」で返す。"""
    words = word_readings(text)
    exp = "".join(r for _, r in words)
    spans, pos = [], 0
    for surf, r in words:
        spans.append((surf, pos, pos + len(r)))
        pos += len(r)
    sm = difflib.SequenceMatcher(None, exp, got, autojunk=False)
    bad = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue
        hit = [s for s in spans if s[1] < i2 and s[2] > i1]
        if not hit:
            continue
        a, b = hit[0][1], hit[-1][2]
        ja, jb = max(0, j1 - (i1 - a)), max(0, j2 + (b - i2))
        bad.append(f"{''.join(s[0] for s in hit)}:{exp[a:b]}→{got[ja:jb]}")
    return bad


# True で ASR が特徴量を GPU でまとめて作る（ASR._fast）。generate_tts の best_of.fast_score が立てる。
FAST_FEATURES = False


class ASR:
    def __init__(self, model: str = "kotoba-tech/kotoba-whisper-v2.0", device: str = "cuda",
                 batch_size: int = 16):
        import torch
        from transformers import pipeline
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model,
                             dtype=dtype, device=device)
        self.batch_size = batch_size
        self.kw = {"language": "ja", "task": "transcribe"}

    def __call__(self, wavs: list[np.ndarray], sr: int) -> list[str]:
        xs = []
        for w in wavs:
            w = np.asarray(w, dtype=np.float32)
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != ASR_SR:
                import librosa
                w = librosa.resample(w, orig_sr=sr, target_sr=ASR_SR)
            # 端ぎりぎりだと頭の語を落とすので前後に無音を足す
            pad = np.zeros(int(0.3 * ASR_SR), dtype=np.float32)
            xs.append({"raw": np.concatenate([pad, w, pad]), "sampling_rate": ASR_SR})
        if not xs:
            return []
        if FAST_FEATURES and all(len(x["raw"]) <= 30 * ASR_SR for x in xs):
            return self._fast(xs)
        # ★2026-11-22 30 秒を超える音声が 2 本以上あると、pipeline のまとめ処理が長さの違う特徴量（128×3000 超）を
        #   詰められずに RuntimeError で落ちていた（011 の bf76cd73: 30 秒超のターンの best-of 採点で毎回失敗→対話ごと落ちた）。
        #   30 秒以下はこれまでどおりまとめて、30 秒を超えるものだけ 1 本ずつ通す（結果の順番は元のまま）。
        lim = 30 * ASR_SR
        if len(xs) > 1 and any(len(x["raw"]) > lim for x in xs):
            out: list[str | None] = [None] * len(xs)
            short = [k for k, x in enumerate(xs) if len(x["raw"]) <= lim]
            if short:
                rs = self.pipe([xs[k] for k in short], batch_size=self.batch_size, generate_kwargs=self.kw)
                for k, r in zip(short, rs):
                    out[k] = r["text"].strip()
            for k, x in enumerate(xs):
                if out[k] is None:
                    # 30 秒超は Whisper の長文モード（時刻トークンが要る）でしか通らない
                    out[k] = self.pipe(x, generate_kwargs=self.kw, return_timestamps=True)["text"].strip()
            return out
        res = self.pipe(xs, batch_size=self.batch_size, generate_kwargs=self.kw)
        return [r["text"].strip() for r in res]

    def _fast(self, xs: list[dict]) -> list[str]:
        """pipeline を通さず、特徴量（log-mel）を GPU でまとめて作って generate する（2026-11-14）。pipeline は
        特徴量を 1 本ずつ CPU で作り（30 秒に詰めた STFT）、テイク選択の採点の 4 割がここだった。30 秒以下の音声だけ。"""
        import torch
        pipe = self.pipe
        m, fe, tok = pipe.model, pipe.feature_extractor, pipe.tokenizer
        out: list[str] = []
        for b in range(0, len(xs), self.batch_size):
            ch = [x["raw"] for x in xs[b:b + self.batch_size]]
            f = fe(ch, sampling_rate=ASR_SR, return_tensors="pt", return_attention_mask=True, device=str(m.device))
            with torch.no_grad():
                ids = m.generate(input_features=f.input_features.to(m.device, dtype=m.dtype),
                                 attention_mask=f.attention_mask.to(m.device), **self.kw)
            out += [t.strip() for t in tok.batch_decode(ids, skip_special_tokens=True)]
        return out


@functools.lru_cache(maxsize=2)
def get_asr(model: str = "kotoba-tech/kotoba-whisper-v2.0", device: str = "cuda"):
    """プロセスで1回だけ読み込む。依存が無ければ None（呼び出し側は検査を飛ばす）。"""
    try:
        _g2p()
        return ASR(model, device)
    except Exception as e:                     # 依存不足・モデル未取得
        logger.warning(f"読みの検査を無効化（ASR を用意できない）: {e}")
        return None


def score(asr: "ASR", wavs: list[np.ndarray], sr: int, texts: list[str],
          hints: list | None = None, rubies: list | None = None,
          asr_ruby=None, thr: float = 0.12) -> list[dict]:
    """各テイクの {sub, cer, asr, exp, got}。texts は合成に使った入力（読みを開いた後）。

    hints[k] はテイク k の読みの注記（turn["yomi"]）。ASR は「かた」と聞こえても「方」と
    書くので、注記のある語は ASR 側も注記の読みに置き換えた候補を足して比べる。

    rubies[k] はテイク k の発話全体の読み（turn["ruby"]。yomi_ruby。今の本文に合うものだけ）。
    あれば期待の読みを pyopenjtalk ではなくその読みで作る（V3。2026-11-12）。ASR 側にも台本の
    「語→読み」を当てた候補を足す。それでも sub >= thr なら、asr_ruby(k, ASR の文) が返す
    ASR の文のルビ（人名を ASR が別の漢字で書いたとき）でも比べる。
    """
    hyps = asr(wavs, sr)
    out = []
    for k, (t, h) in enumerate(zip(texts, hyps)):
        rec = rubies[k] if rubies and k < len(rubies) else None
        if rec:
            out.append(_score_ruby(t, h, rec, (lambda hh, k=k: asr_ruby(k, hh)) if asr_ruby else None,
                                   thr))
            continue
        exp = kana(t)
        variants = _hyp_variants(h)
        for it in (hints[k] if hints and k < len(hints) else None) or []:
            sf, rd = it.get("surface", ""), it.get("reading", "")
            if sf and rd:
                variants += [v.replace(sf, rd) for v in variants]
        got = min((kana(v) for v in variants), key=lambda g: sub_rate(exp, g))
        out.append({"sub": sub_rate(exp, got), "cer": cer(exp, got), "asr": h,
                    "exp": exp, "got": got})
    return out


def _score_ruby(text: str, hyp: str, rec: dict, asr_ruby, thr: float) -> dict:
    """発話全体の読み（yomi_ruby）で照合する（V3）。"""
    import yomi_ruby as YR
    base = [YR.interj(v) for v in _hyp_variants(hyp)]
    best = None
    for txt, subs in YR.expected_texts(rec):
        exp = kana(txt)
        variants = list(base)
        sub_v = []
        for v in variants:                 # 台本の「語→読み」を全部当てた形（組み合わせは作らない）
            for sf, rd in sorted(subs, key=lambda p: -len(p[0])):
                v = v.replace(sf, YR.kata(rd))
            sub_v.append(v)
        for v in variants + sub_v:
            g = kana(v)
            s = sub_rate(exp, g)
            if best is None or s < best[0]:
                best = (s, exp, g)
    if best[0] >= thr and asr_ruby is not None:
        c = None
        try:
            c = asr_ruby(hyp)
        except Exception as e:
            logger.debug(f"ASR の文の読み付けに失敗: {e}")
        if c:
            g = kana(YR.plain_reading(c))
            for txt, _subs in YR.expected_texts(rec):
                exp = kana(txt)
                s = sub_rate(exp, g)
                if s < best[0]:
                    best = (s, exp, g)
    s, exp, got = best
    return {"sub": s, "cer": cer(exp, got), "asr": hyp, "exp": exp, "got": got, "v": "ruby"}
