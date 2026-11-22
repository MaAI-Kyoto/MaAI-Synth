import collections
import random
import unittest

from llm_client import build_async_llm_client


def _cfg(llm):
    """fish_tagger._client_config が作る形（tag_llm を llm ブロックに差し込む）。"""
    return {"model": "m", "llm": llm}


class BaseUrlsRandomTests(unittest.TestCase):
    """base_urls（複数）を指定すると、1クライアントの投げ先がランダムに選ばれること。

    fish_tagger は対話ごとに build_async_llm_client を呼び直すので、これが
    そのまま「リクエストごとに投げ先をランダム分散」になる。
    """

    def test_base_urls_spread_randomly(self):
        urls = ["http://a:1/v1", "http://b:2/v1", "http://c:3/v1"]
        random.seed(0)
        picks = collections.Counter()
        for _ in range(3000):
            client, _model, _extra = build_async_llm_client(_cfg(
                {"provider": "local", "base_urls": urls, "api_key_env": "NOPE"}))
            picks[str(client.base_url).rstrip("/")] += 1
        self.assertEqual(set(picks), set(urls))
        self.assertTrue(all(c > 700 for c in picks.values()))

    def test_single_base_url_unchanged(self):
        client, _model, _extra = build_async_llm_client(_cfg(
            {"provider": "local", "base_url": "http://solo:9/v1", "api_key_env": "NOPE"}))
        self.assertEqual(str(client.base_url).rstrip("/"), "http://solo:9/v1")

    def test_base_url_takes_precedence_over_base_urls(self):
        client, _model, _extra = build_async_llm_client(_cfg(
            {"provider": "local", "base_url": "http://solo:9/v1",
             "base_urls": ["http://a:1/v1", "http://b:2/v1"], "api_key_env": "NOPE"}))
        self.assertEqual(str(client.base_url).rstrip("/"), "http://solo:9/v1")

    def test_non_list_base_urls_rejected(self):
        with self.assertRaises(ValueError):
            build_async_llm_client(_cfg(
                {"provider": "local", "base_urls": "http://x/v1"}))


if __name__ == "__main__":
    unittest.main()


class AckQuestionMarkTests(unittest.TestCase):
    """受け止めの「さようでございますか」は問いではないので「。」で閉じる
    （llm_client.emphasize_question_marks。ユーザー指摘 2026-09-22）。"""

    def test_acknowledgements_end_with_a_period(self):
        from llm_client import emphasize_question_marks as f
        self.assertEqual(f("さようでございますか？まずは、お日にちを教えていただけますか？"),
                         "さようでございますか。まずは、お日にちを教えていただけますか？")
        self.assertEqual(f("さようでございますか。"), "さようでございますか。")
        self.assertEqual(f("さようでございますか"), "さようでございますか。")
        self.assertEqual(f("左様でございますか？"), "左様でございますか。")
        self.assertEqual(f("そうでしたか？それは大変でしたね。"), "そうでしたか。それは大変でしたね。")
        self.assertEqual(f("はい。そうでございますか、では"), "はい。そうでございますか。では")

    def test_real_questions_and_other_particles_are_untouched(self):
        from llm_client import emphasize_question_marks as f
        self.assertEqual(f("そうなんですか？"), "そうなんですか？")
        self.assertEqual(f("そうですか。"), "そうですか？")      # 従来どおり（客の問い返しと区別できない）
        self.assertEqual(f("よろしいでしょうか。"), "よろしいでしょうか？")
        self.assertEqual(f("さようでございますかね。"), "さようでございますかね。")


class EndpointConcurrencyTests(unittest.TestCase):
    """llama.cpp（/props）と vLLM（/v1/models の owned_by）を台ごとに見分けて同時数を決める。"""

    @staticmethod
    def _serve(routes):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = routes.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def _clients(self, servers):
        from types import SimpleNamespace
        return [(SimpleNamespace(base_url=f"http://127.0.0.1:{s.server_port}/v1/"), "m", {})
                for s in servers]

    def test_llama_vllm_and_unknown(self):
        from llm_client import endpoint_concurrency
        llama = self._serve({"/props": {"total_slots": 4,
                                        "default_generation_settings": {"n_ctx": 65536}}})
        vllm = self._serve({"/v1/models": {"data": [{"id": "google/gemma-4-31B-it",
                                                     "owned_by": "vllm"}]}})
        other = self._serve({"/v1/models": {"data": [{"id": "x", "owned_by": "someone"}]}})
        try:
            cfg = {"max_concurrent_requests": 2,
                   "llm": {"ctx_per_request": 16384, "vllm_concurrency": 12}}
            with self.assertLogs("llm_client", level="INFO"):
                got = endpoint_concurrency(cfg, self._clients([llama, vllm, other]))
            self.assertEqual(got, [4, 12, 2])
            cfg["llm"].pop("vllm_concurrency")      # 既定は 16
            with self.assertLogs("llm_client", level="INFO"):
                self.assertEqual(endpoint_concurrency(cfg, self._clients([vllm])), [16])
        finally:
            for s in (llama, vllm, other):
                s.shutdown()

    def test_without_ctx_per_request_keeps_fixed_value(self):
        from llm_client import endpoint_concurrency
        cfg = {"max_concurrent_requests": 3, "llm": {"ctx_per_request": None}}
        self.assertEqual(endpoint_concurrency(cfg, [None, None]), [3, 3])


# ---- 投げ先をいちばん空いているサーバに（2026-11-18） ----
def _metrics_server(body: str):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = body.encode()
            self.send_response(200 if self.path == "/metrics" else 404)
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def test_endpoint_load_reads_vllm_and_llamacpp_metrics():
    import llm_client as L
    vllm = ('# HELP vllm:num_requests_running x\n'
            'vllm:num_requests_running{engine="0",model_name="m"} 8.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="m"} 4.0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",reason="deferred"} 4.0\n'
            'vllm:kv_cache_usage_perc{engine="0"} 0.8\n')
    s1, u1 = _metrics_server(vllm)
    s2, u2 = _metrics_server("llamacpp:requests_processing 2\nllamacpp:requests_deferred 1\n")
    s3, u3 = _metrics_server("other_metric 5\n")
    try:
        assert L.endpoint_load(u1) == 12.0          # by_reason は二重に数えない
        assert L.endpoint_load(u2) == 3.0
        assert L.endpoint_load(u3) is None          # 項目が無い
        assert L.endpoint_load("http://127.0.0.1:9/v1", timeout=0.2) is None   # 繋がらない
    finally:
        for s in (s1, s2, s3):
            s.shutdown()


def test_pick_least_loaded(monkeypatch):
    import random
    import llm_client as L
    loads = {"a": 9.0, "b": 2.0, "c": None, "d": 2.0}
    monkeypatch.setattr(L, "endpoint_load", lambda u, timeout=0.5: loads[u])
    picks = {L.pick_least_loaded(["a", "b", "c", "d"], rng=random.Random(i)) for i in range(50)}
    assert picks == {"b", "d"}                      # 最小の同点はランダム・読めない c は外す
    assert L.pick_least_loaded(["only"]) == "only" and L.pick_least_loaded([]) == ""
    loads.update(a=None, b=None, d=None)
    picks = {L.pick_least_loaded(["a", "b", "c"], rng=random.Random(i)) for i in range(50)}
    assert picks == {"a", "b", "c"}                 # どれも読めなければ従来どおりランダム
