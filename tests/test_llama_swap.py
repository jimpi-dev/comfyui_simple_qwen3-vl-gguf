#!/usr/bin/env python3
"""Tests for llama-swap model list / load / log tailing."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen3vl_run
from llama_swap_client import LlamaSwapClient, parse_model_list, snapshot, tail_log_text


class MockLlamaSwap:
    def __init__(self, support_load_api: bool = True):
        self.requests: List[Dict[str, Any]] = []
        self.loaded = "Qwen3-VL-8B"
        self.support_load_api = support_load_api
        self._httpd = None
        self._thread = None
        parent = self
        log_lines = [f"line-{i} model={parent.loaded}" for i in range(1, 301)]
        parent.log_text = "\n".join(log_lines) + "\n"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _send(self, code, body, content_type="application/json"):
                if isinstance(body, (bytes, bytearray)):
                    data = body
                elif isinstance(body, str):
                    data = body.encode("utf-8")
                else:
                    data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parent.requests.append({"method": "GET", "path": self.path})
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    self._send(200, "OK", "text/plain")
                    return
                if path in ("/v1/models", "/models"):
                    self._send(200, {
                        "data": [
                            {"id": "Qwen3-VL-8B", "object": "model"},
                            {"id": "Gemma4-E4B", "object": "model"},
                        ]
                    })
                    return
                if path == "/running":
                    self._send(200, {"running": [{"model": parent.loaded}]})
                    return
                if path == "/logs":
                    self._send(200, parent.log_text, "text/plain")
                    return
                if path.startswith("/upstream/"):
                    model = unquote(path[len("/upstream/"):])
                    parent.loaded = model
                    self._send(200, "<html>upstream</html>", "text/html")
                    return
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    body = {}
                parent.requests.append({"method": "POST", "path": self.path, "body": body})
                path = self.path.split("?", 1)[0]
                if path.startswith("/api/models/load/"):
                    if not parent.support_load_api:
                        self._send(404, {"error": {"message": "not found"}})
                        return
                    model = unquote(path[len("/api/models/load/"):])
                    parent.loaded = model
                    self._send(200, {"success": True, "model": model})
                    return
                if path.startswith("/v1/chat/completions"):
                    self._send(200, {
                        "choices": [{"message": {"content": f"swap-ok {parent.loaded}"}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                    })
                    return
                self._send(404, {"error": {"message": "not found"}})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


class TestLlamaSwapHelpers(unittest.TestCase):
    def test_parse_model_list_openai(self):
        names = parse_model_list({"data": [{"id": "b"}, {"id": "a"}]})
        self.assertEqual(names, ["a", "b"])

    def test_parse_running(self):
        names = parse_model_list({"running": [{"model": "Qwen3-VL-8B"}]})
        self.assertEqual(names, ["Qwen3-VL-8B"])

    def test_tail_log_text(self):
        text = "\n".join(f"L{i}" for i in range(1, 11))
        self.assertEqual(tail_log_text(text, 3), "L8\nL9\nL10")
        self.assertEqual(tail_log_text(text, 0), "")
        self.assertEqual(tail_log_text(text, 200), text)


class TestLlamaSwapClient(unittest.TestCase):
    def test_list_load_logs_and_inference(self):
        srv = MockLlamaSwap()
        self.addCleanup(srv.close)
        client = LlamaSwapClient(srv.url, timeout=5)
        models = client.list_models()
        self.assertEqual(models, ["Gemma4-E4B", "Qwen3-VL-8B"])
        self.assertEqual(client.running_models(), ["Qwen3-VL-8B"])

        client.load_model("Gemma4-E4B")
        self.assertEqual(srv.loaded, "Gemma4-E4B")
        load_posts = [r for r in srv.requests if r["method"] == "POST" and "/api/models/load/" in r["path"]]
        self.assertTrue(load_posts)

        log = client.get_logs(max_lines=5)
        self.assertEqual(log.count("\n"), 4)
        self.assertTrue(log.endswith("line-300 model=Qwen3-VL-8B") or "line-300" in log.splitlines()[-1])

        snap = snapshot(srv.url, timeout=5)
        self.assertEqual(snap["models"], ["Gemma4-E4B", "Qwen3-VL-8B"])
        self.assertEqual(snap["current"], "Gemma4-E4B")

        result, _ = qwen3vl_run.run_inference_direct({
            "use_llama_swap": True,
            "llama_swap_url": srv.url,
            "llama_swap_model": "Gemma4-E4B",
            "llama_swap_log_lines": 12,
            "user_prompt": "hi",
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertIn("swap-ok", result["output"])
        self.assertIn("Gemma4-E4B", result["output"])
        self.assertTrue(result.get("llama_swap_log"))
        self.assertLessEqual(len(result["llama_swap_log"].splitlines()), 12)
        chat = [r for r in srv.requests if r["method"] == "POST" and r["path"].startswith("/v1/chat/completions")]
        self.assertTrue(chat)
        self.assertEqual(chat[-1]["body"]["model"], "Gemma4-E4B")

    def test_load_falls_back_to_upstream(self):
        srv = MockLlamaSwap(support_load_api=False)
        self.addCleanup(srv.close)
        client = LlamaSwapClient(srv.url, timeout=5)
        client.load_model("Gemma4-E4B")
        self.assertEqual(srv.loaded, "Gemma4-E4B")
        upstream = [r for r in srv.requests if r["method"] == "GET" and r["path"].startswith("/upstream/")]
        self.assertTrue(upstream)

    def test_health_plain_ok(self):
        srv = MockLlamaSwap()
        self.addCleanup(srv.close)
        from llama_server_client import LlamaServerClient
        health = LlamaServerClient(srv.url, timeout=5).health()
        self.assertEqual(health.get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
