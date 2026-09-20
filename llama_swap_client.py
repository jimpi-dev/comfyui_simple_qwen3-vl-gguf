# llama_swap_client.py
"""llama-swap helpers: list models, load/swap, tail logs.

https://github.com/mostlygeek/llama-swap
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from llama_server_client import LlamaServerClient, LlamaServerError, normalize_base_url

NONE_MODEL = "(none)"


def _id_from_item(item: Any) -> Optional[str]:
    if isinstance(item, str):
        name = item.strip()
        return name or None
    if not isinstance(item, dict):
        return None
    for key in ("id", "name", "model", "model_id", "modelId"):
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def parse_model_list(body: Any) -> List[str]:
    items: List[Any] = []
    if isinstance(body, dict):
        for key in ("data", "models", "running"):
            if isinstance(body.get(key), list):
                items = body[key]
                break
        else:
            if "id" in body or "name" in body:
                items = [body]
    elif isinstance(body, list):
        items = body
    names = []
    seen = set()
    for item in items:
        name = _id_from_item(item)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    names.sort(key=str.lower)
    return names


def tail_log_text(text: str, max_lines: int) -> str:
    if not text:
        return ""
    try:
        n = int(max_lines)
    except (TypeError, ValueError):
        n = 200
    if n <= 0:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return "\n".join(lines[-n:])


def is_placeholder_model(name: Optional[str]) -> bool:
    if not name:
        return True
    val = str(name).strip().lower()
    return val in ("", NONE_MODEL, "none", "(loading...)", "(error)")


class LlamaSwapClient:
    def __init__(self, server_url: str, api_key: Optional[str] = None, timeout: float = 30):
        self.http = LlamaServerClient(server_url, api_key=api_key, timeout=timeout)

    def list_models(self) -> List[str]:
        last_error = None
        for path in ("/v1/models", "/models", "/api/models"):
            try:
                body = self.http.request("GET", path, timeout=min(8.0, self.http.timeout))
                names = parse_model_list(body)
                if names:
                    return names
            except LlamaServerError as e:
                last_error = e
                continue
        if last_error:
            raise last_error
        return []

    def running_models(self) -> List[str]:
        for path in ("/running", "/api/ps"):
            try:
                body = self.http.request("GET", path, timeout=min(8.0, self.http.timeout))
                names = parse_model_list(body)
                if names:
                    return names
            except LlamaServerError:
                continue
        return []

    def load_model(self, model_id: str) -> None:
        if is_placeholder_model(model_id):
            return
        encoded = quote(model_id, safe="/")
        errors = []
        try:
            self.http.request("POST", f"/api/models/load/{encoded}", payload={}, timeout=self.http.timeout)
            return
        except LlamaServerError as e:
            errors.append(e)
            if e.status_code not in (404, 405, None):
                # 404/405 = older llama-swap without this route
                if e.status_code and e.status_code < 500 and e.status_code not in (404, 405):
                    raise
        try:
            self.http.request(
                "GET",
                f"/upstream/{encoded}",
                as_text=True,
                timeout=self.http.timeout,
            )
            return
        except LlamaServerError as e:
            errors.append(e)
        raise errors[-1]

    def get_logs(self, max_lines: int = 200, model_id: Optional[str] = None) -> str:
        paths = ["/logs"]
        if model_id and not is_placeholder_model(model_id):
            encoded = quote(model_id, safe="/")
            paths.insert(0, f"/logs/stream/{encoded}")
        text = ""
        last_error = None
        for path in paths:
            # /logs is buffered plain text. /logs/stream keeps the socket open.
            if path.startswith("/logs/stream"):
                continue
            try:
                text = self.http.request(
                    "GET",
                    path,
                    as_text=True,
                    accept="text/plain",
                    timeout=min(15.0, self.http.timeout),
                )
                if text:
                    break
            except LlamaServerError as e:
                last_error = e
                continue
        if not text and last_error:
            return f"[llama-swap logs unavailable: {last_error}]"
        return tail_log_text(str(text or ""), max_lines)


def snapshot(server_url: str, api_key: str = "", timeout: float = 15) -> Dict[str, Any]:
    url = normalize_base_url(server_url)
    client = LlamaSwapClient(url, api_key=api_key, timeout=timeout)
    models = client.list_models()
    running = client.running_models()
    return {
        "url": url,
        "models": models,
        "running": running,
        "current": running[0] if running else (models[0] if models else NONE_MODEL),
    }


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    def _bad(msg, status=400):
        return web.json_response({"error": msg}, status=status)

    @PromptServer.instance.routes.get("/simpleqwenvl/llama_swap/models")
    async def llama_swap_models(request):
        url = (request.query.get("url") or "").strip()
        api_key = (request.query.get("api_key") or "").strip()
        if not url:
            return _bad("url is required")
        try:
            data = snapshot(url, api_key=api_key)
            return web.json_response(data)
        except LlamaServerError as e:
            return web.json_response({"error": str(e), "models": [], "running": []}, status=502)
        except Exception as e:
            return web.json_response({"error": str(e), "models": [], "running": []}, status=500)

    @PromptServer.instance.routes.post("/simpleqwenvl/llama_swap/load")
    async def llama_swap_load(request):
        try:
            data = await request.json()
        except Exception:
            return _bad("JSON body required")
        url = str(data.get("url") or "").strip()
        model = str(data.get("model") or "").strip()
        api_key = str(data.get("api_key") or "").strip()
        timeout = float(data.get("timeout") or 300)
        if not url or is_placeholder_model(model):
            return _bad("url and model are required")
        try:
            client = LlamaSwapClient(url, api_key=api_key, timeout=timeout)
            client.load_model(model)
            running = client.running_models()
            return web.json_response({"success": True, "model": model, "running": running})
        except LlamaServerError as e:
            return web.json_response({"error": str(e)}, status=502)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.get("/simpleqwenvl/llama_swap/logs")
    async def llama_swap_logs(request):
        url = (request.query.get("url") or "").strip()
        model = (request.query.get("model") or "").strip()
        api_key = (request.query.get("api_key") or "").strip()
        try:
            lines = int(request.query.get("lines") or 200)
        except (TypeError, ValueError):
            lines = 200
        if not url:
            return _bad("url is required")
        try:
            client = LlamaSwapClient(url, api_key=api_key, timeout=15)
            text = client.get_logs(max_lines=lines, model_id=model)
            return web.json_response({"log": text, "lines": lines})
        except LlamaServerError as e:
            return web.json_response({"error": str(e), "log": ""}, status=502)
        except Exception as e:
            return web.json_response({"error": str(e), "log": ""}, status=500)


_register_routes()
