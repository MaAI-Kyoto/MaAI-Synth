"""LLM クライアント構築の共通ヘルパ（Stage 1-3 のテキスト生成）。

テキスト生成は全て OpenAI Python SDK の chat.completions 経由なので、`base_url` を
差し替えるだけで任意の **OpenAI 互換サーバ**（vLLM / SGLang / Ollama など）に話せる。
これを利用し、ライセンス/コストがクリアなオープンソースLLM（例: vLLM で立てた
tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2）を config だけで切り替えられるようにする。

config の `llm:` ブロック（省略時は従来どおり ChatGPT クラウド）:

    llm:
      provider: "openai"        # "openai"=クラウド / "local"=OpenAI互換ローカルサーバ
      model: null               # 省略時は top-level `model` を使用
      base_url: null            # local時 "http://host:8000/v1"
      hosts: [gpu-node1, gpu-node2]   # マシン名だけ書くと、model を出しているサーバを探して使う
                                # （優先順位 base_urls > hosts > base_url。resolve_llm_hosts）
      api_key_env: "OPENAI_API_KEY"   # local はダミー可（未設定なら "EMPTY"）
      extra_body:               # サーバ固有オプション（vLLM の guided JSON / thinking無効化等）
        chat_template_kwargs: {enable_thinking: false}
      ctx_per_request: 16384    # 書くと1台あたりの同時数をサーバの /props から決める
                                # （endpoint_concurrency。無ければ max_concurrent_requests）
"""

import asyncio
import os
import random
import re
import unicodedata

from openai import AsyncOpenAI

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# `hosts:` で探すポート。batch/remote_servers.py の LLM 役割は 8001 から GPU 1枚ごとに
# 1つずつ振るので、1台8枚まで見れば足りる。`ports:` で上書きできる。
DEFAULT_LLM_PORTS = list(range(8001, 8011))
_DISCOVERY_CACHE: dict = {}


def discover_llm_endpoints(hosts, model: str, ports=None, timeout: float = 3.0,
                           where: str = "llm") -> list:
    """マシン名のリストから、`model` を出している OpenAI 互換サーバの URL を探す。

    各ホストの各ポートに GET /v1/models を投げ、返ってきたモデル ID に `model` が
    含まれるものだけを `http://<IP>:<port>/v1` の形で返す（IP にするのは no_proxy の
    ホスト名漏れで社内プロキシへ飛ぶ事故を避けるため）。1つも無ければ、見つかった
    別モデルのサーバを添えて止める。同じ引数の2回目以降はキャッシュを返す
    （リクエストごとにクライアントを作り直す呼び出し側があるため）。
    """
    import concurrent.futures as cf
    import json
    import logging
    import socket
    import urllib.request

    if isinstance(hosts, str):
        hosts = [h for h in hosts.split(",") if h.strip()]
    hosts = [str(h).strip() for h in hosts or []]
    ports = [int(p) for p in (ports or DEFAULT_LLM_PORTS)]
    key = (tuple(hosts), model, tuple(ports))
    if key in _DISCOVERY_CACHE:
        return list(_DISCOVERY_CACHE[key])

    log = logging.getLogger(__name__)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ips = {}
    for h in hosts:
        try:
            ip = socket.gethostbyname(h)
            # 自機は /etc/hosts で 127.0.1.1 に解決されることがあり、no_proxy に無いので
            # httpx がプロキシへ送って 503 になる。ループバックは 127.0.0.1 に揃える
            ips[h] = "127.0.0.1" if ip.startswith("127.") else ip
        except OSError as e:
            log.warning(f"[{where}.hosts] {h} の名前解決に失敗したので飛ばす: {e}")

    def probe(job):
        host, ip, port = job
        try:
            with opener.open(f"http://{ip}:{port}/v1/models", timeout=timeout) as r:
                data = json.load(r)
            return host, ip, port, [m.get("id") for m in data.get("data") or []]
        except Exception:
            return None

    jobs = [(h, ip, p) for h, ip in ips.items() for p in ports]
    with cf.ThreadPoolExecutor(max_workers=min(64, len(jobs) or 1)) as ex:
        found = [r for r in ex.map(probe, jobs) if r]

    urls = [f"http://{ip}:{port}/v1" for _, ip, port, ids in found if model in ids]
    if not urls:
        others = [f"{h}:{p}={ids}" for h, _, p, ids in found]
        raise RuntimeError(
            f"[{where}.hosts] {hosts} に {model} を出しているサーバがありません"
            f"（ポート {ports[0]}-{ports[-1]} を確認）。"
            f"見つかったサーバ: {others or 'なし'}")
    log.info(f"[{where}.hosts] {model} のサーバを {len(urls)}台見つけた: "
             + ", ".join(f"{h}:{p}" for h, _, p, ids in found if model in ids))
    # OpenAI クライアント(httpx)は環境の http_proxy に従うので、見つけた IP を no_proxy に足す
    import os
    hit = {ip for _, ip, port, ids in found if model in ids}
    for var in ("no_proxy", "NO_PROXY"):
        cur = [x for x in os.environ.get(var, "").split(",") if x]
        os.environ[var] = ",".join(cur + sorted(hit - set(cur)))
    _DISCOVERY_CACHE[key] = urls
    return list(urls)


# 投げ先の混み具合（endpoint_load / pick_least_loaded）に数える /metrics の項目。処理中＋待ち。
_LOAD_METRICS = {
    "vllm:num_requests_running", "vllm:num_requests_waiting",       # vLLM
    "llamacpp:requests_processing", "llamacpp:requests_deferred",   # llama.cpp（--metrics 付きのとき）
}
_LOAD_POOL = None


def endpoint_load(url: str, timeout: float = 0.5):
    """OpenAI 互換サーバ（http://<ip>:<port>/v1）の /metrics を読み、処理中＋待ちの依頼数を返す。
    読めない・項目が無いときは None。vLLM の /metrics は約 65KB・同じ LAN から約 10ms（2026-11-18 実測）。"""
    import urllib.request
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/metrics", timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    n, seen = 0.0, False
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in _LOAD_METRICS:
            try:
                n += float(line.rsplit(" ", 1)[1])
                seen = True
            except (IndexError, ValueError):
                pass
    return n if seen else None


def pick_least_loaded(urls, timeout: float = 0.5, rng=random):
    """いちばん空いている投げ先（endpoint_load が最小）を返す。同点はランダム。
    読めない投げ先はその回だけ候補から外し、どれも読めなければ従来どおりランダム。
    他のワーカーの依頼も数に入るので、速いサーバほど多く受け持つ（2026-11-18・refine の LLM で
    ランダム等分だと A6000 の node11 に待ちが溜まり、Ada の node23 が空いていた）。"""
    global _LOAD_POOL
    urls = [u for u in urls if u]
    if len(urls) <= 1:
        return urls[0] if urls else ""
    if _LOAD_POOL is None:
        import concurrent.futures as cf
        _LOAD_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-load")
    loads = list(_LOAD_POOL.map(lambda u: endpoint_load(u, timeout), urls))
    known = [(x, u) for x, u in zip(loads, urls) if x is not None]
    if not known:
        return rng.choice(urls)
    m = min(x for x, _ in known)
    return rng.choice([u for x, u in known if x == m])


def resolve_llm_hosts(config: dict) -> dict:
    """`llm.hosts`（マシン名だけ）があれば、生きているサーバを探して `llm.base_urls` に展開する。

    優先順位は base_urls（明示したURL）> hosts > base_url。子 config で base_urls を書けば
    親の hosts より勝つ。親の base_urls を捨てて hosts で探させたいときは `base_urls: null`。
    config をその場で書き換えて返す。
    """
    llm = config.get("llm") or {}
    hosts = llm.get("hosts")
    if not hosts or llm.get("base_urls"):
        llm.pop("hosts", None)
        llm.pop("ports", None)
        return config
    model = llm.get("model") or config["model"]
    llm["base_urls"] = discover_llm_endpoints(hosts, model, llm.get("ports"), where="llm")
    llm.pop("base_url", None)
    llm.pop("hosts", None)
    llm.pop("ports", None)
    config["llm"] = llm
    return config


def build_async_llm_client(config: dict):
    """(client, model, extra_body) を返す。

    - `llm` 未指定 or provider=openai かつ base_url 無し → 従来のクラウド ChatGPT
      （後方互換: `AsyncOpenAI(api_key=os.environ[api_key_env])`）。
    - base_url 指定 or provider=local → OpenAI 互換ローカルサーバ（api_key はダミー可）。
    - base_url 無し・`base_urls`（複数）あり → この1クライアントの投げ先を **ランダムに1つ**
      選ぶ。呼び出しのたびに本関数でクライアントを作り直す利用側（fish_tagger 等）では、
      これがそのまま「リクエストごとにランダムな投げ先へ分散」になる。
      （固定割当のラウンドロビンで束ねたいときは build_async_llm_clients を使う。）
    """
    resolve_llm_hosts(config)
    llm = config.get("llm") or {}
    provider = str(llm.get("provider") or "openai").lower()
    base_url = llm.get("base_url")
    if not base_url:
        base_urls = llm.get("base_urls")
        if base_urls:
            if not isinstance(base_urls, list):
                raise ValueError("llm.base_urls はURLのリストで指定してください")
            base_url = random.choice(base_urls)
    api_key_env = llm.get("api_key_env", DEFAULT_API_KEY_ENV)
    model = llm.get("model") or config["model"]
    extra_body = llm.get("extra_body") or {}

    # タイムアウト/内部リトライを config で上書き可能にする。
    #   - OpenAI SDK 既定は timeout=600秒・max_retries=2。ローカルの遅い大規模モデル
    #     (例 32B) で長大出力(voice工程の 150+ターン)を生成すると、1リクエストが
    #     600秒を超えて SDK が内部で 3回(初回+2)タイムアウト → 実質1800秒浪費してから
    #     "Request timed out" を返す。既定のままだと 32B voice がまともに通らない。
    #   - 対策: local では timeout を長め(config で 1800 等)に、max_retries=0 にして
    #     SDK の3倍リトライを止め、リトライは呼び出し側(retry_max_attempts)に一本化する。
    client_kwargs = {}
    timeout = llm.get("timeout")
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    max_retries = llm.get("max_retries")
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries

    if base_url or provider == "local":
        if not base_url:
            raise ValueError("llm.provider=local には llm.base_url が必要です")
        # ローカルの OpenAI 互換サーバは通常 APIキー不要 → ダミーで可
        api_key = os.environ.get(api_key_env) or "EMPTY"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
    else:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"LLM APIキーが見つかりません（環境変数 {api_key_env}）")
        client = AsyncOpenAI(api_key=api_key, **client_kwargs)

    return client, model, extra_body


def build_async_llm_clients(config: dict):
    """複数エンドポイントに対応した (client, model, extra_body) のリストを返す。

    同一モデルを別GPUに複数インスタンス立てて負荷分散（ラウンドロビン）するための
    ヘルパ。呼び出し側はタスク index を len(clients) で割った剰余でクライアントを
    選ぶことで、リクエストを各インスタンスに均等配分できる。

    設定:
        llm:
          provider: "local"
          base_urls:                         # ← 複数指定でラウンドロビン
            - "http://localhost:8001/v1"
            - "http://localhost:8010/v1"
            - "http://localhost:8011/v1"
          # base_url (単数) しか無ければ従来どおり1エンドポイント。
          # model / api_key_env / extra_body / timeout / max_retries は全URLで共有。

    - `llm.base_urls` が非空リストなら、その各URLへ1クライアントずつ構築（他設定は共有）。
    - 無ければ `build_async_llm_client(config)` を1要素だけ返す（完全な後方互換）。

    ※ こちらは各URLへクライアントを常設し index で固定割当（ラウンドロビン）する用途。
      1リクエストごとに投げ先をランダムに選びたいだけなら build_async_llm_client 側が
      `base_urls` からランダムに1つ選ぶので、そのまま毎回作り直せばよい。
    """
    resolve_llm_hosts(config)
    llm = config.get("llm") or {}
    base_urls = llm.get("base_urls")
    if not base_urls:
        return [build_async_llm_client(config)]

    if not isinstance(base_urls, list):
        raise ValueError("llm.base_urls はURLのリストで指定してください")

    out = []
    for url in base_urls:
        # 各URL用に llm ブロックを複製し base_url だけ差し替えて既存ロジックを再利用。
        sub = dict(config)
        sub_llm = dict(llm)
        sub_llm["base_url"] = url
        sub_llm.pop("base_urls", None)
        sub_llm.setdefault("provider", "local")
        sub["llm"] = sub_llm
        out.append(build_async_llm_client(sub))
    return out


def endpoint_concurrency(config: dict, clients) -> list:
    """clients（build_async_llm_clients の戻り値）の各台に送る同時リクエスト数のリスト。

    `llm.ctx_per_request` が無ければ従来どおり全台 `max_concurrent_requests`。
    あれば各台の GET /props を読み、`min(total_slots, n_ctx // ctx_per_request)`（最低1）にする。
    llama-server を --kv-unified で立てると KV は全スロットで n_ctx を共有するので、
    同時数を「KV に入る本数」で抑えないと溢れて "Context size has been exceeded" の 500 になり、
    その時点で処理中の全リクエストが巻き添えで落ちる（2026-10-20 に -c 32768 へ3本で実際に発生）。
    unified でないサーバでは /props の n_ctx が1スロットぶんなので少なめに出る（安全側）。
    vLLM の台（/props が無く、/v1/models の owned_by が "vllm"）は `llm.vllm_concurrency`（既定 16）。
    vLLM は KV に入りきらない分を落とさず待たせるだけなので、KV に入る本数より多めに送って
    サーバ側の待ち行列を切らさない方が速い（2026-10-22 A6000・KV 48,884 トークンで16本 → 走行5〜7本・2.3倍）。
    どちらとも分からない台は `max_concurrent_requests` に戻す。
    """
    import concurrent.futures as cf
    import json
    import logging
    import urllib.request

    fallback = int(config["max_concurrent_requests"])
    llm = config.get("llm") or {}
    budget = llm.get("ctx_per_request")
    if not budget:
        return [fallback] * len(clients)
    budget = int(budget)
    vllm_n = int(llm.get("vllm_concurrency") or 16)
    log = logging.getLogger(__name__)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def probe(client):
        url = str(client[0].base_url).rstrip("/")
        url = url[:-3] if url.endswith("/v1") else url
        try:
            with opener.open(f"{url}/props", timeout=5) as r:
                d = json.load(r)
            slots = int(d["total_slots"])
            n_ctx = int(d["default_generation_settings"]["n_ctx"])
            return url, max(1, min(slots, n_ctx // budget)), f"{slots}slots/{n_ctx}"
        except Exception as e:
            err = e
        try:
            with opener.open(f"{url}/v1/models", timeout=5) as r:
                owners = {m.get("owned_by") for m in json.load(r).get("data") or []}
            if "vllm" in owners:
                return url, vllm_n, "vLLM"
        except Exception:
            pass
        log.warning(f"[llm] {url} は llama.cpp（/props）とも vLLM とも分からないので同時数 {fallback} にする: {err}")
        return url, fallback, "fallback"

    with cf.ThreadPoolExecutor(max_workers=min(32, len(clients) or 1)) as ex:
        found = list(ex.map(probe, clients))
    log.info(f"[llm] 1台あたりの同時リクエスト数（llama.cpp は min(スロット数, n_ctx // {budget})・"
             f"vLLM は {vllm_n}）: "
             + ", ".join(f"{u.split('//')[-1]}={n}({s})" for u, n, s in found)
             + f" → 合計 {sum(n for _, n, _ in found)}")
    return [n for _, n, _ in found]


async def run_endpoint_worker_pool(
    clients,
    items,
    per_endpoint,
    process,
    on_result,
    *,
    should_stop=None,
):
    """複数エンドポイントに常駐ワーカーを張り、共有キューから連続的にタスクを処理する。

    従来の「グループバリア」方式（total_concurrency 件ぶんのタスクを asyncio.gather で
    まとめて待ってから次の波を出す）は、対話長のばらつきでグループ内の最遅タスクが
    バリアになり、先に終わったエンドポイント（GPU）が次の波まで待機・遊休になっていた。
    本ヘルパはワーカーが空き次第すぐ次のタスクを引く pull 型にしてバリアを無くす。

    引数:
      clients:      [(client, model, extra_body), ...]（各エンドポイント1組）
      items:        処理対象の列（各要素＝1タスク）
      per_endpoint: 1エンドポイントあたりの常駐ワーカー数（＝その台への同時リクエスト数）。
                    clients と同じ長さのリストなら台ごとの数（endpoint_concurrency の戻り値）
      process:      async (client, model, extra_body, item, index) -> result
      on_result:    async (result, item, index) -> None
                    完了順に、内部ロックで直列化して呼ばれる（ファイル追記・カウンタ更新用）。
      should_stop:  () -> bool。True で新規タスクの取得を止める（実行中タスクは完走させる）。

    総同時リクエスト数 = per_endpoint × len(clients)（リストなら合計）。ワーカーは固定エンドポイントに
    束縛されるが、共有キューを空いた順に引くため、速い台へ自然に多く仕事が回る
    （index % num_endpoints の固定割当より均等に分散する）。
    """
    queue: asyncio.Queue = asyncio.Queue()
    for idx, item in enumerate(items):
        queue.put_nowait((idx, item))

    write_lock = asyncio.Lock()
    stopped = False

    async def worker(client, model, extra_body):
        nonlocal stopped
        while True:
            if stopped or (should_stop is not None and should_stop()):
                stopped = True
                return
            try:
                idx, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await process(client, model, extra_body, item, idx)
            async with write_lock:
                await on_result(result, item, idx)

    if isinstance(per_endpoint, int):
        per_endpoint = [per_endpoint] * len(clients)
    if len(per_endpoint) != len(clients):
        raise ValueError(f"per_endpoint の長さ {len(per_endpoint)} が clients {len(clients)} と違う")
    workers = [
        asyncio.create_task(worker(client, model, extra_body))
        for (client, model, extra_body), n in zip(clients, per_endpoint)
        for _ in range(max(1, n))
    ]
    await asyncio.gather(*workers)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def extract_json_text(text: str) -> str:
    """モデル応答から JSON 本文を頑健に取り出す。

    1. `<think>...</think>`（Qwen3 等の思考出力）を除去
    2. ```` ``` ```` / ```` ```json ```` フェンスがあれば中身を採用
    3. 先頭の `{`/`[` から末尾の `}`/`]` までを切り出す（前後の地の文を除去）

    従来のフェンス除去（text.startswith("```")）を包含しつつ、思考トークン混入や
    前置きテキストにも耐える。JSON オブジェクト（{...}）・配列（[...]）の両方に対応。
    """
    if not text:
        return text
    s = _THINK_RE.sub("", text).strip()

    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    ends = [i for i in (s.rfind("}"), s.rfind("]")) if i != -1]
    if starts and ends:
        start, end = min(starts), max(ends)
        if end >= start:
            s = s[start:end + 1]
    return s.strip()


# 発話テキスト（dialogue の utterance / voice の text）のTTS向け正規化。
# 方針（config main の生成要件）:
#   - 記号は「？」以外使わない。「…」「！」「〜」「・」やカッコ書き等の記号・約物は全て除去。
#   - 残すのは句読点「、」「。」と疑問符「？」のみ（半角「?」は全角「？」に統一）。
#   - 発話中の余計な空白（半角/全角スペース・改行・タブ等）は全て除去。
# 判定は Unicode カテゴリで行う: P*(約物)・S*(記号)・Z*(空白)・C*(制御) を、
# 下記の許可文字を除いて落とす。かな/漢字/英数字/長音符「ー」(Lm) は残る。
_KEEP_PUNCT = {"、", "。", "？"}

# フィラー末尾の延ばし棒（sanitize_utterance で除去）
_FILLER_ELONG_RE = re.compile(r"(えっと|ええと|あの|その)ー+(?=[、。？]|$)")
_FILLER_ELONG_N_RE = re.compile(r"(^|、)(ん)ー+(?=[、。？]|$)")


def sanitize_utterance(text):
    """発話テキストから「？」以外の記号と空白を除去して返す（非文字列はそのまま返す）。"""
    if not isinstance(text, str):
        return text
    # 疑問符は半角・全角とも全角「？」に統一（後段のカテゴリ判定で残す）
    text = text.replace("?", "？")
    # 「！」は削除すると前後の文が癒着する（「まじで！特に〜」→「まじで特に〜」）ので
    # 文境界を保って「。」に置換する。「！？」は疑問を優先。
    text = text.replace("！？", "？").replace("!?", "？")
    text = text.replace("!", "。").replace("！", "。")
    # 「%」は削除すると数値の意味が壊れる（「30%の消費者」→「30の消費者」）
    text = text.replace("%", "パーセント").replace("％", "パーセント")
    # 小数点も削除すると数値が壊れる（「38.5度」→「385度」）ので読みの「点」に置換。
    # 全角数字（３８．５）もカテゴリ判定で数字だけ残って癒着するので対象に含める。
    text = re.sub(r"([0-9０-９])[\.．]([0-9０-９])", r"\1点\2", text)
    # フィラー末尾の延ばし棒は除去する（「えっとー」→「えっと」「んー」→「ん」。
    # TTSで間延びするため）。「うーん」の語中・「へー」「あー」等の反応語は触らない。
    # 「ん」は「うん」「うーん」を巻き込まないよう語頭・読点直後のみ対象。
    text = _FILLER_ELONG_RE.sub(r"\1", text)
    text = _FILLER_ELONG_N_RE.sub(r"\1\2", text)
    out = []
    for ch in text:
        if ch in _KEEP_PUNCT:
            out.append(ch)
            continue
        if unicodedata.category(ch)[0] in ("P", "S", "Z", "C"):
            continue  # 記号・約物・空白・制御は除去
        out.append(ch)
    # 連続する「？」「。」は1つにまとめる（「？。」も「？」に）
    result = re.sub("？{2,}", "？", "".join(out))
    result = re.sub("。{2,}", "。", result)
    # 連続する読点を1つにまとめ、文末記号と読点が隣り合う形も潰す。
    # Gemma 4 は「間を長く取る」つもりで読点を重ねて書く（実測: 後処理を全て切った
    # 100対話3,362ターンで「、、」519件・「。、」153件・「？、」18件。Qwen は3種とも0件）。
    # TTS では無音が伸びて不自然な間になり、後段の文分割・読点密度の計測もずれる。
    # 語が消えるわけではないので、作り直させずここで機械的に直す。
    result = re.sub("、{2,}", "、", result)
    result = re.sub("([。？])、+", r"\1", result)   # 「。、」「？、」→「。」「？」
    result = re.sub("、+([。？])", r"\1", result)   # 「、。」「、？」→「。」「？」
    return result.replace("？。", "？")


# 「〜か」「〜っけ」等で終わる問いかけの文末は「？」で終わるべきだが、モデルはしばしば
# 「よろしいでしょうか。」「どんな項目があるんですか。」のように「。」で終えてしまう。
# TTS/後段のため、明確な疑問文末の句点「。」を疑問符「？」に補正する（プロンプトの
# 指示の取りこぼしを機械的に救済）。文中・文末いずれの疑問文にも効くよう全体を置換する。
#   - 「〜か。」「〜か」(文末) → 「〜か？」
#   - 「〜っけ。」「〜っけ」(文末) → 「〜っけ？」
# 「か」で終わる文は日本語の会話ではほぼ必ず問いかけ（ですか/ますか/でしょうか/ないか 等）。
_Q_KA_MID_RE = re.compile(r"か。")
_Q_KA_END_RE = re.compile(r"か$")
_Q_KKE_MID_RE = re.compile(r"っけ。")
_Q_KKE_END_RE = re.compile(r"っけ$")
# ★例外: 相手の話を受け止める「さようでございますか。」「そうでしたか。」は、形は「か」で
# 終わるが問いではなく相槌（接客の担当者が客の事情を聞いて返す語）。「？」にすると TTS が
# 語尾を上げて聞き返しになる（ユーザー指摘 2026-09-22。実測 接客コーパス（別の担当者ペルソナ）
# の100対話で担当者の「さようでございますか？」22件・「左様〜」3件・「そうでしたか？」5件、
# すべて受け止めで、問いは0件）。上の置換で「？」にしないだけでなく、モデル自身が「？」で
# 書いたものも「。」に戻す。「そうですか」「そうなんですか」は客の驚きの問い返し
# （「そうなんですか？」）と区別が付かないので含めない。
_ACK_KA_RE = re.compile(
    r"((?:さよう|左様)(?:でございます|です|でございました|でした)"
    r"|そう(?:でございます|でございました|でした))か(?![ねなよ])[？。、]?")
# ★タメ口の納得「そっか」「そうか」「そういうことか」も同じ（2026-09-26 実測: 汎用・初対面の
# Gemma 版4コーパス各1万対話で「そっか？」「そうか？」が計1,756件、「そっか。」は5件。
# 読むと全件が相手の話の受け止めで、問い返しは0件）。語頭の境界を要求する
# （「話そうか」「どうしようか」は意向の問いなので対象外）。読点はそのまま残す。
_REALIZE_KA_RE = re.compile(
    r"(?:^|(?<=[、。？！\s]))"
    r"(そっか|そうか|そうだったか|そうだったのか|そういうことか|そういうことだったのか)"
    r"(?:？|(?=。)|$)")
# 独り言の「経済学部か。僕は〜」も「か」の置換で「経済学部か？僕は〜」になり、語尾を上げた
# 問い返しに聞こえる（同じ実測で文頭の体言＋「か？」が約9,000件。例「15日か？いいタイミングだね。」）。
# 体言（漢字・カタカナ・数字、または「まで／から／だけ／ほど」止め）＋「か？」の後に同じ発話が
# 続き、その残りに問いが無いときだけ「か。」にする（残りに「？」があれば問い返しの可能性を残す）。
# 疑問詞（何・誰…）は対象外（「何か？」）。接客の客は polish_cs_speech が「〜ですか。」にする。
_MUSING_KA_RE = re.compile(
    r"(?:^|(?<=[。！？]))((?:あ、|ああ、|あー、|えっ、|へえ、|へー、|ほう、|なるほど、)?"
    r"(?![^、。？！]*(?:何|誰|どれ|どこ|いつ|どう|なぜ|いくら))"
    r"[^、。？！]{0,10}?(?:[一-龥ァ-ヴー0-9０-９]|まで|から|だけ|ほど))か？(?=[^？]+$)")


def emphasize_question_marks(text):
    """疑問文末（「〜か」「〜っけ」で終わる文）の「。」/無句点を「？」に補正する。

    受け止めの「さようでございますか」「そうでしたか」「そっか」「そうか」は逆に「。」で閉じ、
    独り言の「経済学部か。僕は〜」（体言＋か、後ろに問いが続かない）も「。」のままにする。
    """
    if not isinstance(text, str):
        return text
    text = _Q_KA_MID_RE.sub("か？", text)
    text = _Q_KA_END_RE.sub("か？", text)
    text = _Q_KKE_MID_RE.sub("っけ？", text)
    text = _Q_KKE_END_RE.sub("っけ？", text)
    text = _ACK_KA_RE.sub(r"\1か。", text)
    text = _REALIZE_KA_RE.sub(r"\1。", text)
    return _MUSING_KA_RE.sub(r"\1か。", text)
