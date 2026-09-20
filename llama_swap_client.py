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


LOADED_STATES = {"ready", "running", "loaded", "idle", "ok"}
LOADING_STATES = {"starting", "loading"}
UNLOADED_STATES = {"stopped", "unloaded", "stopping", "failed"}


def is_placeholder_model(name: Optional[str]) -> bool:
    if not name:
        return True
    val = str(name).strip().lower()
    return val in ("", NONE_MODEL, "none", "(loading...)", "(error)")


def _item_state(item: Any) -> str:
    if not isinstance(item, dict):
        return "ready"
    st = item.get("state")
    if isinstance(st, dict):
        st = st.get("value")
    status = item.get("status")
    if st is None and isinstance(status, dict):
        st = status.get("value")
    elif st is None:
        st = status
    return str(st or "ready").strip().lower()


def parse_running(body: Any) -> tuple[List[str], bool]:
    """Return (loaded model ids, loading). Understands llama-swap /running shapes."""
    items: List[Any] = []
    if isinstance(body, dict):
        if isinstance(body.get("running"), list):
            items = body["running"]
        elif isinstance(body.get("models"), list):
            items = body["models"]
        elif "model" in body or "id" in body or "name" in body:
            items = [body]
    elif isinstance(body, list):
        items = body

    names: List[str] = []
    loading = False
    seen = set()
    for item in items:
        state = _item_state(item)
        if state in LOADING_STATES:
            loading = True
        if state in UNLOADED_STATES:
            continue
        name = _id_from_item(item)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names, loading


def _router_loaded(body: Any) -> tuple[List[str], bool]:
    """llama-server router GET /models with status.value."""
    items: List[Any] = []
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        items = body["data"]
    elif isinstance(body, list):
        items = body
    names: List[str] = []
    loading = False
    had_status = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if "status" in item or "state" in item:
            had_status = True
        state = _item_state(item)
        if state in LOADING_STATES:
            loading = True
        if state in UNLOADED_STATES:
            continue
        if state in LOADED_STATES:
            name = _id_from_item(item)
            if name and name not in names:
                names.append(name)
    if not had_status:
        return [], False
    return names, loading


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
        names, _loading, _has = self.running_status()
        return names or []

    def running_status(self) -> tuple[Optional[List[str]], bool, bool]:
        """(loaded ids or None, loading, has /running-like endpoint)."""
        for path in ("/running", "/api/ps"):
            try:
                body = self.http.request("GET", path, timeout=min(8.0, self.http.timeout))
                names, loading = parse_running(body)
                return names, loading, True
            except LlamaServerError as e:
                if e.status_code in (404, 405):
                    continue
                raise
        return None, False, False

    def unload_model(self, model_id: Optional[str] = None) -> None:
        specific = None if is_placeholder_model(model_id) else (str(model_id).strip() if model_id else None)
        attempts: List[tuple[str, str, Optional[dict]]] = []
        if specific:
            encoded = quote(specific, safe="/")
            attempts.extend(
                [
                    ("POST", f"/api/models/unload/{encoded}", {}),
                    ("POST", "/api/models/unload", {"model": specific}),
                    ("POST", "/models/unload", {"model": specific}),
                ]
            )
        else:
            attempts.extend(
                [
                    ("POST", "/api/models/unload", {}),
                    ("POST", "/models/unload", {}),
                ]
            )
        errors: List[LlamaServerError] = []
        for method, path, payload in attempts:
            try:
                self.http.request(method, path, payload=payload, timeout=self.http.timeout)
                return
            except LlamaServerError as e:
                errors.append(e)
                msg = str(e).lower()
                if e.status_code in (400, 404) and ("not running" in msg or "not loaded" in msg):
                    return
                if e.status_code in (404, 405, None):
                    continue
                continue
        hint = (
            "Could not unload the GGUF remotely. "
            "llama-swap: POST /api/models/unload. "
            "llama-server router: POST /models/unload {\"model\": id}. "
            "A single-model llama-server started with -m cannot unload without stopping the process."
        )
        last = f" Last error: {errors[-1]}" if errors else ""
        raise LlamaServerError(hint + last, status_code=errors[-1].status_code if errors else None)

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


def probe_status(server_url: str, api_key: str = "", timeout: float = 8) -> Dict[str, Any]:
    """Whether llama-swap / llama-server currently has a GGUF in VRAM."""
    url = normalize_base_url(server_url)
    result: Dict[str, Any] = {
        "url": url,
        "reachable": False,
        "loaded": False,
        "loading": False,
        "models": [],
        "running": [],
        "source": "unknown",
        "message": "",
    }
    client = LlamaSwapClient(url, api_key=api_key, timeout=timeout)
    try:
        health = client.http.health()
    except LlamaServerError as e:
        if e.status_code == 503:
            result["reachable"] = True
            result["loading"] = True
            result["source"] = "llama-server"
            result["message"] = "loading"
            return result
        result["message"] = str(e)
        return result

    result["reachable"] = True
    plain_health = bool(health.get("plain")) if isinstance(health, dict) else False

    try:
        running, loading, has_running = client.running_status()
        if has_running:
            result["source"] = "llama-swap"
            result["running"] = running or []
            result["models"] = running or []
            result["loading"] = loading
            result["loaded"] = bool(running) and not loading
            if loading and not running:
                result["message"] = "loading"
            elif running:
                result["message"] = "loaded " + ", ".join(running)
            else:
                result["message"] = "idle"
            return result
    except LlamaServerError:
        pass

    try:
        body = client.http.request("GET", "/models", timeout=min(8.0, timeout))
        names, loading = _router_loaded(body)
        if names or loading:
            result["source"] = "llama-server-router"
            result["running"] = names
            result["models"] = names
            result["loading"] = loading
            result["loaded"] = bool(names)
            result["message"] = "loaded " + ", ".join(names) if names else ("loading" if loading else "idle")
            return result
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            if any(isinstance(it, dict) and ("status" in it or "state" in it) for it in body["data"]):
                result["source"] = "llama-server-router"
                result["message"] = "idle"
                return result
    except LlamaServerError:
        pass

    # llama-swap /health is plain "OK" even with nothing loaded.
    if plain_health:
        result["source"] = "llama-swap"
        result["message"] = "idle"
        return result

    result["source"] = "llama-server"
    result["loaded"] = True
    try:
        listed = client.list_models()
        result["models"] = listed
        result["running"] = listed
        result["message"] = "loaded " + ", ".join(listed) if listed else "loaded"
    except LlamaServerError:
        result["message"] = "loaded"
    return result


def unload_models(
    server_url: str,
    model_id: Optional[str] = None,
    api_key: str = "",
    timeout: float = 60,
) -> Dict[str, Any]:
    url = normalize_base_url(server_url)
    client = LlamaSwapClient(url, api_key=api_key, timeout=timeout)
    client.unload_model(model_id)
    status = probe_status(url, api_key=api_key, timeout=min(8.0, timeout))
    status["success"] = True
    status["unloaded"] = "(all)" if is_placeholder_model(model_id) else str(model_id).strip()
    return status


def snapshot(server_url: str, api_key: str = "", timeout: float = 15) -> Dict[str, Any]:
    url = normalize_base_url(server_url)
    client = LlamaSwapClient(url, api_key=api_key, timeout=timeout)
    try:
        models = client.list_models()
    except LlamaServerError:
        models = []
    status = probe_status(url, api_key=api_key, timeout=min(8.0, timeout))
    running = status.get("running") or []
    return {
        **status,
        "url": url,
        "models": models or status.get("models") or [],
        "running": running,
        "current": running[0] if running else (models[0] if models else NONE_MODEL),
        "loaded": bool(status.get("loaded")),
        "loading": bool(status.get("loading")),
        "reachable": bool(status.get("reachable")),
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

    @PromptServer.instance.routes.get("/simpleqwenvl/llama_swap/status")
    async def llama_swap_status(request):
        url = (request.query.get("url") or "").strip()
        api_key = (request.query.get("api_key") or "").strip()
        if not url:
            return _bad("url is required")
        try:
            data = probe_status(url, api_key=api_key)
            return web.json_response(data)
        except LlamaServerError as e:
            return web.json_response({"error": str(e), "reachable": False, "loaded": False}, status=502)
        except Exception as e:
            return web.json_response({"error": str(e), "reachable": False, "loaded": False}, status=500)

    @PromptServer.instance.routes.post("/simpleqwenvl/llama_swap/unload")
    async def llama_swap_unload(request):
        try:
            data = await request.json()
        except Exception:
            return _bad("JSON body required")
        url = str(data.get("url") or "").strip()
        model = str(data.get("model") or "").strip()
        api_key = str(data.get("api_key") or "").strip()
        timeout = float(data.get("timeout") or 60)
        if not url:
            return _bad("url is required")
        try:
            result = unload_models(url, model_id=model, api_key=api_key, timeout=timeout)
            return web.json_response(result)
        except LlamaServerError as e:
            return web.json_response({"error": str(e), "success": False}, status=502)
        except Exception as e:
            return web.json_response({"error": str(e), "success": False}, status=500)

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
