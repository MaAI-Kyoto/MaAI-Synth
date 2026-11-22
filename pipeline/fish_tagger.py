"""Fish Audio S2 Pro 向けの詳細タグ注釈（LLM方式）。

`engine: fish_speech` のとき、各発話に S2 の自由記述インラインタグ `[...]` を LLM で付与し、
合成入力だけを表情豊かにする（マニフェスト/transcript は原文のまま＝タグ非漏洩）。

- テキスト生成基盤（`llm_client.build_async_llm_client` / `extract_json_text`）を再利用するので、
  クラウド ChatGPT でも OpenAI 互換ローカルサーバ（vLLM の OSS-LLM 等）でも `config` だけで切替可能。
- 1対話=1 LLM呼び出しで全ターンをまとめて注釈する（文脈整合・呼び出し数最小）。
- LLM は **原文の文字を変えず `[tag]` を挿入するだけ**。返却後に「タグ除去→原文一致」で検証し、
  逸脱・欠損した発話はタグ破棄（呼び出し側が原文で合成）。
- LLM 呼び出しが全リトライ落ちた対話は `FishTagError` を送出して**その対話をスキップ**する
  （原文で合成するとタグ無しの音声が base キャッシュに焼き付き、以後タグが付かなくなるため）。
  スキップした対話は出力が残らないので、次回実行の resume で自動的に再試行される。

pipeline は同期なので `tag_turns()` は `asyncio.run` で包む。クロスループ問題を避けるため
AsyncOpenAI クライアントは呼び出しごとに生成・クローズする（構築自体は軽量・ネットワーク無し）。
"""

import asyncio
import json
import logging
import re

from llm_client import build_async_llm_client, extract_json_text

logger = logging.getLogger(__name__)

# S2 の代表的な感情/デリバリ/間・息づかいタグ（プロンプトの手掛かり用）。
# S2 は固定集合に限定されず自由記述に対応するので、これは「例示」であって網羅ではない。
FISH_TAG_LIBRARY = (
    "[pause] [short pause] [emphasis] [inhale] [exhale] [sigh] [breath] "
    "[laughing] [laughing tone] [chuckle] [chuckling] [giggle] [tsk] "
    "[excited] [excited tone] [delight] [surprised] [shocked] [angry] [sad] "
    "[whisper] [low voice] [low volume] [soft voice] [gentle tone] "
    "[volume up] [volume down] [loud] [shouting] [screaming] [pitch up] "
    "[clearing throat] [panting] [interrupting] [apologetic tone] [thoughtful tone]"
)

_TAG_RE = re.compile(r"\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")

# リトライ既定値。LLM 呼び出しが落ちるのはサーバ側の一時障害（500/接続断/レート）が
# 主なので回数は多め、待ちは指数バックオフだが上限で頭打ちにする（10回で 512秒待ちに
# ならないように）。
DEFAULT_TAG_RETRY_MAX = 10
MAX_RETRY_DELAY = 30.0        # 通常エラーの待機上限（秒）
MAX_RETRY_DELAY_RATE = 60.0   # 401/429（認証・レート）の待機上限（秒）
# 「呼び出しは成功したがタグ採用率が低い」ときの撮り直し回数。サーバ障害ではなくモデルの
# 出来の問題で、moderate では採用0件も正当なため、通信エラー用の retry_max とは分ける
# （ここまで粘って駄目なら best を採用して先に進む＝スキップしない）。
LOW_RATE_RETRY_MAX = 4


class FishTagError(RuntimeError):
    """LLM タグ付けが全リトライ失敗した（＝この対話は合成せずスキップすべき）。"""

# 一部のクラウドモデル（例: gpt-5.6-terra）は temperature=既定値しか受け付けない。
# 一度でも非対応エラーが出たら以後 temperature を送らない（generate_dialogues と同じ流儀）。
_TEMP_UNSUPPORTED = False


async def _chat_create(client, **kwargs):
    """temperature 非対応モデルなら自動で temperature を外して再試行する。"""
    global _TEMP_UNSUPPORTED
    if _TEMP_UNSUPPORTED:
        kwargs.pop("temperature", None)
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        if "temperature" in str(e).lower() and "temperature" in kwargs:
            _TEMP_UNSUPPORTED = True
            logger.info("fish タグ付け: モデルが temperature 非対応のため既定値で続行します。")
            kwargs.pop("temperature", None)
            return await client.chat.completions.create(**kwargs)
        raise


def strip_tags(s: str) -> str:
    """`[tag]` を全て取り除く（検証・保険用）。"""
    return _TAG_RE.sub("", s or "")


# タグを置いてよい位置: 発話の先頭、または句読点・疑問符の直後のみ。
# 語中挿入（例:「熱[exhale]出てる」）は fish が単語の途中に息継ぎ/間を入れて
# 発話を壊すため、機械的に除去する（プロンプトの位置指示は破られる前提）。
_TAG_BOUNDARY = ("、", "。", "？")
# 発話末尾（最後の文字より後ろ）のタグ。[sigh][exhale][breath] 等が末尾に付くと
# 発話後に息音だけが合成されるため、全て除去する（ユーザー要望）。
_TRAILING_TAG_RE = re.compile(r"(?:\[[^\]]*\]\s*)+$")


def limit_tag_positions(tagged: str, max_head: int = 2, max_total: int = 5) -> str:
    """発話頭・句読点直後以外のタグを除去し、個数を頭 max_head / 全体 max_total に制限する。"""
    parts = []
    last = 0
    for m in _TAG_RE.finditer(tagged):
        parts.append(("text", tagged[last:m.start()]))
        parts.append(("tag", m.group(0)))
        last = m.end()
    parts.append(("text", tagged[last:]))

    out = []
    plain = ""  # ここまでに出力した素テキスト（タグはカウントしない）
    n_head = n_total = 0
    for kind, seg in parts:
        if kind == "text":
            out.append(seg)
            plain += seg
        elif not plain:  # 発話頭
            if n_head < max_head and n_total < max_total:
                out.append(seg)
                n_head += 1
                n_total += 1
        elif plain[-1] in _TAG_BOUNDARY and n_total < max_total:
            out.append(seg)
            n_total += 1
    return _TRAILING_TAG_RE.sub("", "".join(out)).strip()


def _compact(s: str) -> str:
    """比較用に空白を除いた素テキスト（日本語は空白が無いので空白除去だけで十分頑健）。"""
    return _WS_RE.sub("", strip_tags(s or ""))


def build_system_prompt(density: str = "max") -> str:
    """二者日本語会話に S2 タグを注釈させるシステムプロンプトを組む。"""
    if density == "moderate":
        density_rule = (
            "タグ付けは **選択的** に行う。感情や話し方が動く発話（驚き・笑い・ためらい・"
            "強調・声をひそめる・しみじみ等）にだけ 1〜2個 付け、平坦な事務的発話や"
            "ごく短い発話はタグなしのまま返してよい（タグを付けるのは全発話の6〜7割が目安）。"
            "「全発話に先頭タグを1個ずつ」のような機械的な付け方は禁止。"
        )
    else:  # "max"
        density_rule = (
            "各発話に積極的に（1発話2〜5個目安で）タグを付ける。ただし全発話を同じ型"
            "（先頭トーン+全読点にポーズ）にする機械的な付け方は禁止。"
            "付けどころのない平坦な発話は1個や0個でもよい。\n"
            "  ・先頭のトーンタグは発話の内容・感情に合わせて毎回選び直す\n"
            "  ・読点『、』直後の [short pause] / [pause] は、間が効果的な箇所だけに絞る"
            "（すべての読点に置かない）\n"
            "  ・強調したい文節が句読点の直後から始まる場合、その頭に [emphasis]\n"
            "  ・ため息・息継ぎ・笑い（[sigh] / [inhale] / [exhale] / [chuckle]）は"
            "内容に合う箇所だけに置く"
        )
    return (
        "あなたは Fish Audio S2 Pro（音声合成）向けに、日本語の二者会話の各発話へ "
        "感情・話し方・間（ま）を表す **インラインタグ** を付与するアノテータです。\n"
        "タグは半角角括弧 `[...]` で、テキスト内の任意の位置に埋め込めます。固定集合に縛られず、"
        "英語の自由記述（例: [warm broadcast tone], [whisper in small voice], [pitch up]）も使えます。\n"
        f"よく使うタグの例: {FISH_TAG_LIBRARY}\n\n"
        "【厳守事項】\n"
        "1. 原文の日本語（かな・漢字・数字・記号）は **一字一句そのまま**。語順も変えない。"
        "許されるのは `[tag]` を **挿入すること だけ**。単語の言い換え・追加・削除・要約は禁止。\n"
        "2. タグを置いてよい位置は【発話の先頭】と【文中の「、」「。」「？」の直後】だけ。"
        "単語や文節の途中に挿入してはならない"
        "（悪例:「熱[exhale]出てる」/ 良例:「[exhale]熱出てるんだよね。」）。"
        "発話の末尾（最後の文字より後ろ）にもタグを置かない"
        "（悪例:「ぐずぐずしてさ。[sigh]」— 末尾の [sigh][exhale][breath] は禁止）。\n"
        f"3. {density_rule}\n"
        "4. 【多様性】機械的な繰り返しを避け、タグの種類を対話全体で幅広く散らす。\n"
        "   ・同じタグを連続する発話の先頭に置かない（例: [soft voice] 始まりが3発話続くのは禁止）\n"
        "   ・対話全体を2〜3種類のタグの使い回しで済ませない。会話の感情の流れ"
        "（例: 平静→驚き→食いつき→納得→笑い）に沿って、場面ごとに違うタグを選ぶ\n"
        "   ・ライブラリ例に無い自由記述タグ（[hesitant tone], [warm tone], [teasing tone], "
        "[relieved], [curious tone] 等）も積極的に使ってよい\n"
        "5. 発話内容と会話の流れに合わせる。直前の相手発話や、その発話が相槌(type=backchannel)か"
        "本発話(type=speech)かも考慮する（相槌は短く柔らかい反応、驚きには[surprised]など）。\n"
        "6. 出力は次の JSON のみ（前置き・解説・コードフェンス不要）:\n"
        '   {"turns": [{"index": <元のindex>, "text": "<タグ付きテキスト>"}, ...]}\n'
        "   すべての発話について index を1件ずつ返すこと。"
    )


def build_user_prompt(payload: list[dict]) -> str:
    """注釈対象のターン列（index/speaker/type/text）を JSON で提示する。"""
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    return (
        "次の会話の各発話に、上記ルールに従って S2 のインラインタグを付与してください。\n"
        "原文の文字は変えず、タグを挿入するだけです。\n\n"
        f"{body}\n\n"
        '出力は {"turns":[{"index":..,"text":".."}, ...]} の JSON のみ。'
    )


def _client_config(config: dict) -> dict:
    """`tts.fish_speech.tag_llm` があれば LLM 設定として差し込んだ config を返す。

    無ければ config 直下の `llm:`（テキスト生成と同じサーバ）をそのまま流用する。
    """
    tag_llm = ((config.get("tts") or {}).get("fish_speech") or {}).get("tag_llm")
    if not tag_llm:
        return config
    merged = dict(config)
    merged["llm"] = tag_llm
    if tag_llm.get("model"):
        merged["model"] = tag_llm["model"]
    return merged


class FishTagger:
    """1対話ぶんの発話に S2 タグを LLM で付与する。

    構築時に LLM 設定を検証・解決する（誤設定・APIキー無しはここで例外→呼び出し側で降格）。
    実際のクライアントは呼び出しごとに生成・クローズする（asyncio.run のループを跨がないため）。
    """

    def __init__(self, config: dict, density: str = "max",
                 retry_max: int = DEFAULT_TAG_RETRY_MAX, retry_delay: float = 1.0,
                 temperature: float = 0.7):
        self._cfg = _client_config(config)
        # 早期検証: モデル/extra_body を解決しつつ、誤設定なら例外を出す（run() が降格判断）。
        _client, self.model, self.extra_body = build_async_llm_client(self._cfg)
        self.density = density
        self.retry_max = max(1, int(retry_max))
        self.retry_delay = float(retry_delay)
        self.temperature = float(temperature)
        self.system_prompt = build_system_prompt(density)

    def _validate(self, items, turns: list, dialogue_id: str) -> dict[int, str]:
        """LLM返却を検証し、{index: タグ付きテキスト} を返す（原文一致した発話のみ）。"""
        out: dict[int, str] = {}
        if not isinstance(items, list):
            return out
        n_tags = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(turns)):
                continue
            tagged = it.get("text")
            if not isinstance(tagged, str) or not tagged.strip():
                continue
            orig = (turns[idx].get("text") or "")
            # タグを除いた素テキストが原文と（空白無視で）一致するもののみ採用（言い換え防止）
            if _compact(tagged) != _compact(orig):
                logger.debug(f"[{dialogue_id}] タグ検証NG idx={idx}: "
                             f"'{strip_tags(tagged)[:24]}' != '{orig[:24]}'")
                continue
            if not _TAG_RE.search(tagged):
                continue  # タグが1つも無ければ原文と同じなので採用しない
            # 位置ガード: 発話頭と句読点直後以外のタグ（語中挿入）を除去。
            # 極短発話（実質6文字以下）は頭の1個だけ許す。
            core_len = len(re.sub(r"[、。？]", "", _compact(orig)))
            if core_len <= 6:
                head_cap, total_cap = 1, 1
            elif self.density == "moderate":
                head_cap, total_cap = 1, 2
            else:
                head_cap, total_cap = 2, 5
            tagged = limit_tag_positions(tagged.strip(), head_cap, total_cap)
            if not _TAG_RE.search(tagged):
                continue  # ガードで全滅したら原文合成と同じなので採用しない
            out[idx] = tagged
            n_tags += len(_TAG_RE.findall(tagged))
        if out:
            logger.info(f"[{dialogue_id}] fish LLMタグ: {len(out)}/{len(turns)}発話 / "
                        f"計{n_tags}タグ付与")
        else:
            logger.info(f"[{dialogue_id}] fish LLMタグ: 採用0件→原文で合成")
        return out

    async def _tag_async(self, turns: list, dialogue_id: str) -> dict[int, str]:
        payload = [{"index": i,
                    "speaker": t.get("speaker", ""),
                    "type": t.get("type", "speech"),
                    "text": (t.get("text") or "")}
                   for i, t in enumerate(turns)]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_prompt(payload)},
        ]
        total_chars = sum(len(p["text"]) for p in payload)
        # GPT-OSS系は思考(reasoning)にも completion トークンを消費するため、タグ付き
        # JSON本文の見積もり(×6)だけだと finish_reason=length で content が空になる
        # （実測: 2048トークン全てが思考で尽きて全対話 JSONDecodeError → rule降格）。
        # 上限は :8003 の n_ctx=32768 からプロンプト分を引いた安全圏。
        # 85ターン級の長対話は 16384 だと思考+本文で length になる（実測）。
        max_tokens = min(24576, max(12288, total_chars * 8 + 4096))

        client, _model, _extra = build_async_llm_client(self._cfg)
        best: dict[int, str] = {}  # 採用率リトライで最良の結果を保持
        answered = False           # 一度でも LLM が応答し JSON を解釈できたか
        low_rate = 0               # 応答はしたが採用率が低かった回数
        last_err = ""
        try:
            for attempt in range(self.retry_max):
                try:
                    resp = await _chat_create(
                        client,
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=max_tokens,
                        extra_body=self.extra_body or {},
                    )
                    choice = resp.choices[0]
                    if choice.finish_reason == "length":
                        raise ValueError(
                            "finish_reason=length（思考+本文が max_tokens を超過。"
                            f"completion上限={max_tokens}）")
                    text = extract_json_text(choice.message.content or "")
                    data = json.loads(text)
                    items = data.get("turns") if isinstance(data, dict) else data
                    answered = True   # ここまで来たら「LLMは答えた」＝スキップ対象外
                    res = self._validate(items, turns, dialogue_id)
                    if len(res) > len(best):
                        best = res
                    # 呼び出しが成功しても、言い換え等の検証NGで採用率が低すぎる
                    # ことがある（実測 3/50）。しきい値未満ならリトライで撮り直す。
                    # moderate は「平坦な発話はタグなし」が正当なので下限を緩める。
                    min_rate = 0.5 if self.density == "max" else 0.35
                    if len(res) >= min_rate * len(turns):
                        return res
                    low_rate += 1
                    if low_rate >= LOW_RATE_RETRY_MAX:
                        logger.info(f"[{dialogue_id}] タグ採用率が {low_rate} 回連続で"
                                    f"低いまま（{len(best)}/{len(turns)}発話）→ この結果で続行")
                        return best
                    raise ValueError(f"タグ採用率が低すぎます ({len(res)}/{len(turns)}発話)")
                except Exception as e:  # noqa: BLE001  (json/接続/レート全て拾って降格)
                    err = str(e)
                    last_err = f"{type(e).__name__}: {e}"
                    # 指数バックオフ。回数が多いので上限で頭打ちにする
                    # （上限が無いと10回目で 512秒待ちになる）。
                    if "401" in err or "429" in err:
                        delay = min(MAX_RETRY_DELAY_RATE,
                                    max(10.0, self.retry_delay * (3 ** attempt)))
                    else:
                        delay = min(MAX_RETRY_DELAY, self.retry_delay * (2 ** attempt))
                    last = attempt >= self.retry_max - 1
                    logger.warning(
                        f"[{dialogue_id}] fish LLMタグ付け失敗 "
                        f"(試行 {attempt + 1}/{self.retry_max}"
                        + ("" if last else f", {delay:.0f}秒待機") + f"): {last_err}")
                    if not last:
                        await asyncio.sleep(delay)
            if best:
                return best   # 採用率は低いが有効なタグが取れている → それを使う
            if not answered:
                # 全試行でLLMに届かなかった/壊れた応答しか来なかった。原文で合成すると
                # タグ無し音声が base キャッシュに残るので、対話ごとスキップさせる。
                raise FishTagError(
                    f"LLMタグ付けが {self.retry_max} 回とも失敗しました: {last_err}")
            return best   # LLMは答えたが採用0件（moderate では正当）→ 原文で合成
        finally:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

    def tag_turns(self, turns: list, dialogue_id: str) -> dict[int, str]:
        """{index: タグ付きテキスト} を返す（同期）。

        LLM が答えたうえでタグ採用0件なら空dict（＝原文で合成）。
        LLM 呼び出しが全リトライ失敗したときは `FishTagError` を送出する
        （呼び出し側はその対話をスキップする＝タグ無し音声をキャッシュに残さない）。
        """
        if not turns:
            return {}
        try:
            return asyncio.run(self._tag_async(turns, dialogue_id))
        except FishTagError:
            raise
        except Exception as e:  # noqa: BLE001 (想定外もスキップ扱いに揃える)
            raise FishTagError(f"{type(e).__name__}: {e}") from e
