#!/usr/bin/env python3
"""Unit tests for the llama-server HTTP backend.

These tests do not require a real GGUF or llama-server binary.
A local mock HTTP server implements the llama.cpp API surface used by the node.
"""
from __future__ import annotations

import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llama_server_client as lsc
import qwen3vl_run


class MockLlamaServer:
    def __init__(self, scenario: str = "ok"):
        self.scenario = scenario
        self.requests: List[Dict[str, Any]] = []
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _read_json(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {"_raw": raw.decode("utf-8", errors="replace")}

            def _send(self, code: int, body: Any, content_type: str = "application/json"):
                data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parent.requests.append({"method": "GET", "path": self.path, "body": None})
                if parent.scenario == "offline":
                    self._send(500, {"error": {"message": "offline", "code": 500}})
                    return
                if parent.scenario == "loading" and self.path.startswith("/health"):
                    self._send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
                    return
                if self.path.startswith("/health") or self.path.startswith("/v1/health"):
                    self._send(200, {"status": "ok"})
                    return
                if self.path.startswith("/v1/models") or self.path.startswith("/models"):
                    self._send(200, {"data": [{"id": "Qwen3-VL-8B-Instruct", "object": "model"}]})
                    return
                self._send(404, {"error": {"message": "not found", "code": 404}})

            def do_POST(self):
                body = self._read_json()
                parent.requests.append({"method": "POST", "path": self.path, "body": body})
                if parent.scenario == "http500":
                    self._send(500, {"error": {"code": 500, "message": "internal server boom", "type": "server_error"}})
                    return
                if parent.scenario == "malformed":
                    self._send(200, b"not-json{{{", content_type="application/json")
                    return
                if parent.scenario == "empty_choices":
                    self._send(200, {"id": "chatcmpl-x", "choices": []})
                    return
                if self.path.startswith("/tokenize"):
                    text = str(body.get("content") or "")
                    self._send(200, {"tokens": [ord(c) % 100 + 1 for c in text[:8]] or [1]})
                    return
                if self.path.startswith("/embedding"):
                    self._send(200, {"embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})
                    return
                if self.path.startswith("/v1/embeddings"):
                    self._send(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
                    return
                if self.path.startswith("/completion"):
                    prompt = body.get("prompt")
                    n_media = 0
                    if isinstance(prompt, dict):
                        n_media = len(prompt.get("multimodal_data") or [])
                    self._send(200, {
                        "content": f"raw-ok media={n_media}",
                        "tokens_predicted": 4,
                        "tokens_evaluated": 12,
                    })
                    return
                if self.path.startswith("/v1/chat/completions"):
                    messages = body.get("messages") or []
                    n_images = 0
                    n_audio = 0
                    n_video = 0
                    user_text = ""
                    for msg in messages:
                        content = msg.get("content")
                        if isinstance(content, str):
                            user_text += content
                        elif isinstance(content, list):
                            for part in content:
                                ptype = part.get("type")
                                if ptype == "image_url":
                                    n_images += 1
                                elif ptype == "input_audio":
                                    n_audio += 1
                                elif ptype == "input_video":
                                    n_video += 1
                                elif ptype == "text":
                                    user_text += str(part.get("text") or "")
                    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
                    answer = f"ok images={n_images} audio={n_audio} video={n_video} text={user_text[:80]}"
                    message = {"role": "assistant", "content": answer}
                    if thinking:
                        message["reasoning_content"] = "step by step"
                    self._send(200, {
                        "id": "chatcmpl-test",
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    })
                    return
                self._send(404, {"error": {"message": f"unknown {self.path}", "code": 404}})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def last_post(self, path_prefix: str) -> Optional[dict]:
        for item in reversed(self.requests):
            if item["method"] == "POST" and item["path"].startswith(path_prefix):
                return item
        return None

    def close(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


def _red_image():
    img = Image.new("RGB", (32, 24), color=(200, 30, 30))
    return img


class TestLlamaServerClient(unittest.TestCase):
    def test_health_and_models(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        client = lsc.LlamaServerClient(srv.url, timeout=5)
        health = client.health()
        self.assertEqual(health.get("status"), "ok")
        models = client.list_models()
        self.assertTrue(models)
        self.assertEqual(models[0]["id"], "Qwen3-VL-8B-Instruct")

    def test_server_offline(self):
        client = lsc.LlamaServerClient("http://127.0.0.1:1", timeout=1)
        with self.assertRaises(lsc.LlamaServerError) as ctx:
            client.health()
        self.assertIn("not reachable", str(ctx.exception).lower())

    def test_wrong_port(self):
        client = lsc.LlamaServerClient("http://127.0.0.1:9", timeout=1)
        with self.assertRaises(lsc.LlamaServerError):
            client.chat_completions({"model": "x", "messages": [{"role": "user", "content": "hi"}]})

    def test_loading_503(self):
        srv = MockLlamaServer("loading")
        self.addCleanup(srv.close)
        client = lsc.LlamaServerClient(srv.url, timeout=5)
        with self.assertRaises(lsc.LlamaServerError) as ctx:
            client.health()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("loading", str(ctx.exception).lower())

    def test_http_500(self):
        srv = MockLlamaServer("http500")
        self.addCleanup(srv.close)
        client = lsc.LlamaServerClient(srv.url, timeout=5)
        with self.assertRaises(lsc.LlamaServerError) as ctx:
            client.chat_completions({"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("boom", str(ctx.exception))

    def test_malformed_json(self):
        srv = MockLlamaServer("malformed")
        self.addCleanup(srv.close)
        client = lsc.LlamaServerClient(srv.url, timeout=5)
        with self.assertRaises(lsc.LlamaServerError) as ctx:
            client.chat_completions({"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("malformed JSON", str(ctx.exception))

    def test_parse_reasoning_and_empty_content(self):
        content, reasoning, usage = lsc.parse_chat_message({
            "choices": [{"message": {"content": "", "reasoning_content": "think hard"}}],
            "usage": {"completion_tokens": 3},
        })
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "think hard")
        self.assertEqual(usage["completion_tokens"], 3)

        with self.assertRaises(lsc.LlamaServerError):
            lsc.parse_chat_message({"choices": []})


class TestQwen3VLRunHttp(unittest.TestCase):
    def test_text_prompt(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, data = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Hello from text mode",
            "system_prompt": "You are a test bot.",
            "debug": False,
            "request_timeout": 5,
            "max_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertIn("Hello from text mode", result["output"])
        post = srv.last_post("/v1/chat/completions")
        self.assertIsNotNone(post)
        body = post["body"]
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_single_image(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Describe this image.",
            "images": [_red_image()],
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertIn("images=1", result["output"])
        post = srv.last_post("/v1/chat/completions")
        content = post["body"]["messages"][-1]["content"]
        image_parts = [p for p in content if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        url = image_parts[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(url), 40)

    def test_multiple_images(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Compare these images.",
            "images": [_red_image(), _red_image(), _red_image()],
            "add_image_id": "\\n[Image {num}]:",
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertIn("images=3", result["output"])
        post = srv.last_post("/v1/chat/completions")
        content = post["body"]["messages"][-1]["content"]
        self.assertEqual(sum(1 for p in content if p.get("type") == "image_url"), 3)
        labels = [p["text"] for p in content if p.get("type") == "text" and "Image" in p.get("text", "")]
        self.assertEqual(len(labels), 3)

    def test_thinking_on_off(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        on, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Think then answer.",
            "enable_thinking": True,
            "remove_thinking": False,
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(on["status"], "success", on)
        self.assertIn("<think>", on["output"])
        self.assertIn("step by step", on["output"])
        body = srv.last_post("/v1/chat/completions")["body"]
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["reasoning_format"], "deepseek")

        off, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Think then answer.",
            "enable_thinking": True,
            "remove_thinking": True,
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(off["status"], "success", off)
        self.assertNotIn("<think>", off["output"])
        self.assertNotIn("step by step", off["output"])

    def test_force_reasoning(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "hi",
            "enable_thinking": False,
            "force_reasoning": True,
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        body = srv.last_post("/v1/chat/completions")["body"]
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])

    def test_repeated_image_requests(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        outputs = []
        for i in range(3):
            result, _ = qwen3vl_run.run_inference_direct({
                "server_url": srv.url,
                "user_prompt": f"Describe image {i}.",
                "images": [_red_image()],
                "debug": False,
                "request_timeout": 5,
            })
            self.assertEqual(result["status"], "success", result)
            outputs.append(result["output"])
        posts = [r for r in srv.requests if r["method"] == "POST" and r["path"].startswith("/v1/chat/completions")]
        self.assertEqual(len(posts), 3)
        for post in posts:
            images = [p for p in post["body"]["messages"][-1]["content"] if p.get("type") == "image_url"]
            self.assertEqual(len(images), 1)

    def test_sampling_and_stop(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "Write JSON",
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 8,
            "max_tokens": 64,
            "repeat_penalty": 1.2,
            "stop": '["</s>", "END"]',
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        body = srv.last_post("/v1/chat/completions")["body"]
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["top_p"], 0.5)
        self.assertEqual(body["top_k"], 8)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["repeat_penalty"], 1.2)
        self.assertEqual(body["stop"], ["</s>", "END"])

    def test_words_to_ban_uses_tokenize(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "hello",
            "words_to_ban": "cat,dog",
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        tokenize_calls = [r for r in srv.requests if r["path"].startswith("/tokenize")]
        self.assertGreaterEqual(len(tokenize_calls), 2)
        body = srv.last_post("/v1/chat/completions")["body"]
        self.assertTrue(body.get("logit_bias"))

    def test_raw_mode_multimodal_uses_completion(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "caption",
            "system_prompt": "sys",
            "raw_mode": True,
            "prompt_template": "{system}{images}{user}",
            "images": [_red_image()],
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertIn("raw-ok", result["output"])
        post = srv.last_post("/completion")
        self.assertIsNotNone(post)
        prompt = post["body"]["prompt"]
        self.assertIsInstance(prompt, dict)
        self.assertIn("<__media__>", prompt["prompt_string"])
        self.assertEqual(len(prompt["multimodal_data"]), 1)

    def test_embeddings(self):
        srv = MockLlamaServer("ok")
        self.addCleanup(srv.close)
        result, data = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "embed me",
            "extract_embedding": True,
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["data_type"], 1)
        self.assertIsNotNone(data)
        self.assertEqual(data.shape[-1], 3)

    def test_error_offline_message(self):
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": "http://127.0.0.1:1",
            "user_prompt": "hi",
            "debug": False,
            "request_timeout": 1,
        })
        self.assertEqual(result["status"], "error")
        self.assertIn("not reachable", result["message"].lower())

    def test_error_http_500(self):
        srv = MockLlamaServer("http500")
        self.addCleanup(srv.close)
        result, _ = qwen3vl_run.run_inference_direct({
            "server_url": srv.url,
            "user_prompt": "hi",
            "debug": False,
            "request_timeout": 5,
        })
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["message"])

    def test_image_content_helper_base64(self):
        part = qwen3vl_run._build_image_content(_red_image(), quality=80)
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_no_llama_cpp_import(self):
        source = Path(qwen3vl_run.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import llama_cpp", source)
        self.assertNotIn("from llama_cpp", source)
        client_src = Path(lsc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("llama_cpp", client_src)


if __name__ == "__main__":
    unittest.main()
