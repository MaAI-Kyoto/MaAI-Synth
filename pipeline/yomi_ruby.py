"""発話全体の読み（語単位のルビ）を LLM に付けさせ、合成の読み違いの検査と引き直しに使う。

2026-11-12 追加。評価（約1万発話）で決めた形:
  - 1回1発話、対話全体を文脈に、温度0。まとめて付けさせると取り違える（河川敷→かせき）。
  - 主は C＝語単位のルビ「凝り性{こりしょう}」。漢字のかたまりごと（凝{こ}り性{せい}）より誤りが少ない。
  - 確かめに B＝発話全体のひらがな。B は脱落・書き損じが多い（体→から）が、C と誤る場所が違う。
    C と B の読みが一致した語だけを「確かな読み」として合成の入力に開く。
  - 照合（V3）: 期待の読み＝C の読み（B と食い違う語は B の読みの候補も許す）。ASR 側は台本の
    「語→読み」を当てた候補を足し、まだ引っかかるときは ASR の文にも C で読みを付けて比べる
    （人名を ASR が別の漢字で書く：ヒラノ→平野→ヘヤ）。V0（いまの照合）比で、引っかかる発話が
    320→52（引き直しで直らなかった 306 → 40）、無作為の対照 0.19%→0.08%。

ターンには {"src": 付けたときの本文, "c": C のルビ文, "b": B のひらがな} を "ruby" として持たせる。
本文が後で変わった（refine の書き換え・Stage3 の変換）ら src が合わないので使わず、付け直す。
"""
from __future__ import annotations

import asyncio
import difflib
import itertools
import json
import logging
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# 読みを付ける対象の字（漢字・数字・英字）
R = "一-龯㐀-䶵々〆ヵヶ0-9０-９A-Za-zＡ-Ｚａ-ｚ"
_HAS_R = re.compile(f"[{R}]")
_PUN = r"[、。，．,.？！?!「」『』（）()\s・…ー―〜~]"

PROMPT_C = """あなたは日本語の音声合成のための読み付け係です。渡された対話で * の付いた発話について、
漢字・数字・英字を含む語のあとに、その語全体の読みを {ひらがな} で付けてください。
- 語は送り仮名・語中のかなを含めた一続きの単位（凝り性{こりしょう}、使って{つかって}、
  血の気{ちのけ}、お土産{おみやげ}、3歳{さんさい}、何時間{なんじかん}）。かなだけの語には付けない。
- 読みはこの文脈で話し手が実際に読むとおり。人名・地名は、その対話で最も自然な読み。
- {} を取り除くと本文と完全に一致すること（本文の文字を変えない・足さない・消さない）。
# 出力（JSON のみ）
{"ruby": "..."}"""

PROMPT_B = """あなたは日本語の音声合成のための読み付け係です。渡された対話で * の付いた発話を、
この文脈で話し手が実際に読むとおりに、全部ひらがなで書いてください。
- 漢字・数字・英字・カタカナもすべてひらがなにする。句読点（、。？！）は本文のまま残す。
- 助詞の「は」「へ」「を」は表記どおり書く。長音は表記どおり（とうきょう、せんせい）。本文にある「ー」は残す。
- 人名・地名は、その対話で最も自然な読み。
# 出力（JSON のみ）
{"yomi": "ひらがなの読み"}"""

_RETRY_MSG = ("* の発話の本文と文字が一致していません。本文の文字を一字も変えず、"
              "前置き（番号・話者名）も付けずに、もう一度出力してください。")


def needs_ruby(text: str) -> bool:
    return bool(text and _HAS_R.search(text))


def hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def _nz(s: str) -> str:
    return re.sub(_PUN, "", hira(s))


# ── 出力の検査 ─────────────────────────────────────────────────────────────
def clean_c(text: str, y: str) -> str:
    """C の出力の軽い崩れを直す：話者名の前置き（「0 イノウエ: 」）、本文にない空白。"""
    y = re.sub(r"^\s*\d+\s*[^:：{}\s]+(?:\{[^{}]*\})?\s*[:：]\s*", "", y or "")
    if " " not in text and "　" not in text:
        y = re.sub(r"[ 　]", "", y)
    # 読みをカタカナで書いたもの（AI{エーアイ}）はひらがなに直す
    return re.sub(r"\{([^{}]*)\}", lambda m: "{" + hira(m.group(1)) + "}", y)


def ok_c(text: str, y: str) -> bool:
    """{} を外すと本文に一致し、読みがひらがなであること。

    どこからが語かは segs() が読みと突き合わせて決める（決まらない読みは捨てる）ので、ここでは
    見ない。付け漏れの語は許す（その語は pyopenjtalk の読みで補う）。
    2026-11-12: 以前は「語の中のかなが読みに同じ順で出る」ことも求めていたが、前に付け漏れの語や
    カタカナ語があると（私も機械{きかい}、マツモトと申します{もうします}）正しい出力まで捨てていた。"""
    if not y or re.sub(r"\{[^{}]*\}", "", y) != text:
        return False
    return all(rd and not re.search(r"[^ぁ-ゖー]", rd) for rd in re.findall(r"\{([^{}]*)\}", y))


def ok_b(text: str, y: str) -> bool:
    return bool(y) and not re.search(rf"[{R}ァ-ヺ]", y)


# ── ルビ文の分解 ───────────────────────────────────────────────────────────
def segs(c: str) -> list[tuple[str, str | None]]:
    """C のルビ文を [(表記, 読み or None)] に分ける。

    {読み} の直前のどこから語が始まるかは書かれていないので、読みの付いた区間のうち、
    漢字・数字・英字・カタカナ（または「お・ご」＋漢字）で始まり、区間のかなが読みにも
    同じ順で出る最も長いものを語とする（「私は東京{とうきょう}」→「東京」）。
    語が見つからない（かなだけの語に付いたルビ）ときは読みを捨てる。
    """
    out, pos = [], 0
    for m in re.finditer(r"\{([^{}]*)\}", c):
        seg, rd = c[pos:m.start()], m.group(1)
        p = len(seg)
        for q in range(len(seg)):
            w = seg[q:]
            if re.search(r"[、。？！?!\s]", w):
                continue
            if re.match(rf"[{R}ァ-ヺ]|[おご][{R}]", w):
                it = iter(rd)
                if all(ch in it for ch in hira(re.sub(rf"[{R}]", "", w))):
                    p = q
                    break
        # 読みが語の前のかなまで含んでいる（よろしくお願いします{よろしくおねがいします}）ときは、
        # その分を読みから外す（外さないと「よろしく」が二重になる）
        # ただし読みが語の頭のかなと偶然同じ（と鳥栖{とす}）こともあるので、外した方が語の
        # 辞書の読み（pyopenjtalk）に近いときだけ外す。
        pre = re.search(r"[ぁ-ゖー]+$", seg[:p])
        if pre and p < len(seg):
            for n in range(len(pre.group(0)), 0, -1):
                if rd.startswith(pre.group(0)[-n:]) and len(rd) > n:
                    if _closer(seg[p:], rd[n:], rd):
                        rd = rd[n:]
                    break
        if seg[:p]:
            out.append((seg[:p], None))
        if seg[p:]:
            out.append((seg[p:], rd))
        pos = m.end()
    if c[pos:]:
        out.append((c[pos:], None))
    return out


def _closer(surface: str, a: str, b: str) -> bool:
    """語の辞書の読み（pyopenjtalk）に a の方が b より近いか。pyopenjtalk が無ければ False。"""
    try:
        import pyopenjtalk
        d = hira(pyopenjtalk.g2p(surface, kana=True))
    except Exception:
        return False

    def sim(x):
        return difflib.SequenceMatcher(None, d, x, autojunk=False).ratio()
    return sim(a) > sim(b)


def b_readings(sg: list, b: str, all_segs: bool = False) -> list[str | None]:
    """C の各語に対応する B の読み（記号を除いたひらがな）。B の文と C の読みの列を突き合わせて切り出す。
    all_segs=True なら読みの無い区間（付け漏れの語を含む）にも B の対応部分を返す。"""
    cs, spans = "", []
    for sf, rd in sg:
        s = len(cs)
        cs += _nz(rd if rd is not None else sf)
        spans.append((s, len(cs)))
    bs = _nz(b or "")
    ops = difflib.SequenceMatcher(None, cs, bs, autojunk=False).get_opcodes()

    def mp(i, end):
        for tag, i1, i2, j1, j2 in ops:
            if (i1 < i <= i2) if end else (i1 <= i < i2):
                return j1 + (i - i1) if tag == "equal" else (j2 if end else j1)
        return len(bs) if end else 0

    return [None if (rd is None and not all_segs) or s == e else bs[mp(s, False):mp(e, True)]
            for (sf, rd), (s, e) in zip(sg, spans)]


def valid(rec, text: str) -> bool:
    """ターンの "ruby" が今の本文に対して使えるか（本文が変わっていたら使わない）。"""
    return bool(isinstance(rec, dict) and rec.get("c") and rec.get("src") == text
                and re.sub(r"\{[^{}]*\}", "", rec["c"]) == text)


def words(rec) -> list[dict]:
    """[{surface, reading(C), b(B の読み or None), sure(C と B が一致)}]（読みの付いた語だけ）。"""
    sg = segs(rec["c"])
    rb = b_readings(sg, rec["b"]) if rec.get("b") else [None] * len(sg)
    out = []
    for (sf, rd), b in zip(sg, rb):
        if rd is None:
            continue
        out.append({"surface": sf, "reading": rd, "b": b, "sure": bool(b) and b == _nz(rd)})
    return out


def yomi_pairs(rec) -> list[list]:
    """アラインメント用の読み。[[表記, 読み(ひらがな) or None, C と B が一致したか], ...]（表記をつなぐと本文）。
    読みのない区間（かな・付け漏れの語・句読点）は None。<id>.json の segments[].yomi に書く。"""
    sg = segs(rec["c"])
    rb = b_readings(sg, rec["b"]) if rec.get("b") else [None] * len(sg)
    return [[sf, rd, bool(rd and b and b == _nz(rd))] for (sf, rd), b in zip(sg, rb)]


def interj(t: str) -> str:
    """笑い・感動詞は pyopenjtalk が助詞と読む（はは→ワワ、へえ→エエ）ので、照合の両側でカタカナにそろえる。"""
    t = re.sub(r"(^|[、。？！\s])はは(?=[、。？！\s]|$)", r"\1ハハ", t)
    return re.sub(r"へえ|へー", "ヘー", t)


def _build(sg, rds, hira_idx=frozenset()) -> str:
    # 語の読みはカタカナで入れる（ひらがなだと「3歳はなれた」の「は」を pyopenjtalk が助詞と読む）。
    # hira_idx の区間（B で補った、助詞を含みうる区間）はひらがなのまま入れて助詞として読ませる。
    return interj("".join(sf if r is None else (r if k in hira_idx else kata(r))
                          for k, ((sf, _), r) in enumerate(zip(sg, rds))))


def expected_texts(rec) -> list[tuple[str, list]]:
    """照合の期待の読みの候補 [(pyopenjtalk に渡す文, [(表記, 読み)])]。
    1つ目は C の読み、2つ目は B と食い違う語を B の読みにしたもの（食い違いがあるときだけ）。"""
    sg = segs(rec["c"])
    rc = [rd for _, rd in sg]
    out = [(_build(sg, rc), [(sf, rd) for sf, rd in sg if rd])]
    if rec.get("b"):
        rb = b_readings(sg, rec["b"])
        mix = [rd if rd is None or not b or b == _nz(rd) else b for (_, rd), b in zip(sg, rb)]
        if mix != rc:
            out.append((_build(sg, mix), [(sf, m) for (sf, _), m in zip(sg, mix) if m]))
        # C が読みを付けなかった漢字・数字・英字（付け漏れ）を B の対応部分で補った候補。補わないと
        # pyopenjtalk が読むが、カタカナに開いた語に挟まると読みを誤る（アタタカイ人→ジン）。
        ra = b_readings(sg, rec["b"], all_segs=True)
        fill = [rd if rd is not None else (b if b and _HAS_R.search(sf) else None)
                for (sf, rd), b in zip(sg, ra)]
        if fill != rc:
            hi = frozenset(k for k, ((_, rd), f) in enumerate(zip(sg, fill)) if rd is None and f)
            out.append((_build(sg, fill, hi), [(sf, f) for (sf, _), f in zip(sg, fill) if f]))
    return out


def plain_reading(c: str) -> str:
    """ルビ文の語を読み（カタカナ）に置き換えた文（ASR の文に付けたルビを照合に使う）。"""
    sg = segs(c)
    return _build(sg, [rd for _, rd in sg])


def open_words(tts_text: str, rec, suspects: list[str]) -> str:
    """合成入力の、疑わしい語（suspects の表記）のうち確かな読み（C=B）があるものをひらがなに開く。"""
    # カタカナ・かなだけの語（イノウエ）は読みが決まっているので開かない
    sure = {w["surface"]: w["reading"] for w in words(rec) if w["sure"] and _HAS_R.search(w["surface"])}
    for sf in sorted(set(suspects), key=len, reverse=True):
        if sf in sure and sf in tts_text:
            tts_text = tts_text.replace(sf, sure[sf], 1)
    return tts_text


def suspect_words(rec, got_kana: str, norm=None) -> list[str]:
    """聞こえた読み（カタカナ）に、語の読みが見当たらない語の表記。
    norm は語の読みを got_kana と同じ形にする関数（reading_check.kana。長音を落とす等）。"""
    norm = norm or kata
    return [w["surface"] for w in words(rec) if norm(kata(w["reading"])) not in got_kana]


# ── LLM への依頼 ───────────────────────────────────────────────────────────
def user_message(lines: list[tuple[str, str]], j: int, text: str | None = None) -> str:
    """lines = [(話者, 本文)]。j 番目に * を付ける（text を渡すとその発話の本文を差し替える：ASR の文）。"""
    rows = []
    for k, (sp, t) in enumerate(lines):
        if k == j and text is not None:
            t = text
        rows.append(f"{'*' if k == j else ' '}{k} {sp}: {t}")
    return "# 対話\n" + "\n".join(rows)


def _parse(content: str, key: str) -> str:
    s = content or ""
    m = re.search(r"\{.*\}", s, re.S)
    try:
        return str(json.loads(m.group(0) if m else s).get(key, ""))
    except Exception:
        return ""


def _jobs(text):
    return (("c", PROMPT_C, "ruby", ok_c, clean_c), ("b", PROMPT_B, "yomi", ok_b, lambda t, y: y))


def _ask_sync(chat, sysmsg, user, key, text, ok, fix):
    msgs = [{"role": "user", "content": user}]
    for _ in range(2):
        try:
            raw = _parse(chat([{"role": "system", "content": sysmsg}] + msgs), key)
        except Exception as e:
            logger.debug(f"読みの依頼に失敗: {e}")
            continue
        y = fix(text, raw)
        if ok(text, y):
            return y
        msgs = msgs + [{"role": "assistant", "content": json.dumps({key: raw}, ensure_ascii=False)},
                       {"role": "user", "content": _RETRY_MSG}]
    return None


async def _ask_async(chat, sysmsg, user, key, text, ok, fix):
    msgs = [{"role": "user", "content": user}]
    for _ in range(2):
        try:
            raw = _parse(await chat([{"role": "system", "content": sysmsg}] + msgs), key)
        except Exception as e:
            logger.debug(f"読みの依頼に失敗: {e}")
            continue
        y = fix(text, raw)
        if ok(text, y):
            return y
        msgs = msgs + [{"role": "assistant", "content": json.dumps({key: raw}, ensure_ascii=False)},
                       {"role": "user", "content": _RETRY_MSG}]
    return None


def annotate_one_sync(chat, lines, j, text=None, with_b=True) -> dict | None:
    """1発話に C（と B）を付ける。C が取れなければ None。"""
    t = lines[j][1] if text is None else text
    user = user_message(lines, j, text)
    out = {"src": t}
    for name, P, key, ok, fix in _jobs(t):
        if name == "b" and not with_b:
            continue
        out[name] = _ask_sync(chat, P, user, key, t, ok, fix)
        if name == "c" and out["c"] is None:
            return None
    return out


async def annotate_turns_async(chat, lines, idx, conc: int = 8, with_b=True) -> dict:
    """lines の idx の発話に読みを付ける（非同期）。{番号: ruby}（取れなかった発話は入らない）。"""
    sem = asyncio.Semaphore(conc)

    async def one(j):
        t = lines[j][1]
        user = user_message(lines, j)
        out = {"src": t}
        async with sem:
            for name, P, key, ok, fix in _jobs(t):
                if name == "b" and not with_b:
                    continue
                out[name] = await _ask_async(chat, P, user, key, t, ok, fix)
                if name == "c" and out["c"] is None:
                    return j, None
        return j, out

    res = await asyncio.gather(*[one(j) for j in idx])
    return {j: r for j, r in res if r}


def async_chat(client, model: str, extra_body: dict | None = None, cost_tracker=None):
    """openai.AsyncOpenAI 互換クライアント → chat(messages) コルーチン（Stage2・後処理用）。"""
    async def chat(messages):
        r = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=2000,
            response_format={"type": "json_object"}, extra_body=extra_body or {})
        if cost_tracker is not None and getattr(r, "usage", None):
            cost_tracker.add(r.usage.prompt_tokens, r.usage.completion_tokens)
        return r.choices[0].message.content or ""
    return chat


class SyncChat:
    """OpenAI 互換サーバ（vLLM）への同期の呼び出し（Stage4 用）。投げ先は順番に回す。"""

    def __init__(self, base_urls: list[str], model: str, extra_body: dict | None = None,
                 timeout: float = 300):
        self.urls = [u.rstrip("/") for u in base_urls if u]
        if not self.urls:
            raise ValueError("読み付けの LLM の投げ先がありません")
        self.model, self.extra_body, self.timeout = model, dict(extra_body or {}), timeout
        self._cyc = itertools.cycle(self.urls)
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, messages) -> str:
        with self._lock:
            url = next(self._cyc)
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 2000,
                "response_format": {"type": "json_object"}, **self.extra_body}
        req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(self._opener.open(req, timeout=self.timeout))
        return d["choices"][0]["message"]["content"] or ""


def annotate_turns_sync(chat, lines, idx, workers: int = 8, with_b=True) -> dict:
    """lines の idx の発話に読みを付ける（スレッド並列）。{番号: ruby}。"""
    if not idx:
        return {}
    with ThreadPoolExecutor(max(1, min(workers, len(idx)))) as ex:
        res = list(ex.map(lambda j: (j, annotate_one_sync(chat, lines, j, with_b=with_b)), idx))
    return {j: r for j, r in res if r}


_SYNC_CACHE: dict = {}


def sync_chat_from_cfg(cfg: dict | None):
    """regen_on_misread.ruby の設定（base_urls か hosts、model、extra_body）から SyncChat を作る。
    同じ設定の2回目以降は使い回す。作れなければ None。"""
    if not cfg:
        return None
    key = json.dumps(cfg, sort_keys=True, default=str)
    if key in _SYNC_CACHE:
        return _SYNC_CACHE[key]
    chat = None
    try:
        model = str(cfg.get("model") or "google/gemma-4-31B-it")
        urls = list(cfg.get("base_urls") or [])
        if not urls and cfg.get("hosts"):
            from llm_client import discover_llm_endpoints
            urls = discover_llm_endpoints(cfg["hosts"], model, cfg.get("ports"),
                                          where="regen_on_misread.ruby")
        chat = SyncChat(urls, model, cfg.get("extra_body"))
    except Exception as e:
        logger.warning(f"読み付けの LLM を用意できない（ルビなしの照合で続ける）: {e}")
    _SYNC_CACHE[key] = chat
    return chat
