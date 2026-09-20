# llama_server_client.py
"""HTTP client for official ggml-org llama.cpp llama-server.

Talks to the OpenAI-compatible API. No llama.cpp Python bindings.
"""
from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 300
DEFAULT_API_KEY = "no-key"
HEALTH_TIMEOUT = 8
MTMD_MEDIA_MARKER = "<__media__>"


class LlamaServerError(Exception):
    """User-facing llama-server HTTP/API error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Any = None,
        url: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.url = url


def normalize_base_url(server_url: Optional[str]) -> str:
    url = (server_url or DEFAULT_SERVER_URL).strip() or DEFAULT_SERVER_URL
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def _join(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _friendly_connection_error(exc: BaseException, base_url: str) -> str:
    host = urlparse(base_url).netloc or base_url
    return (
        f"llama-server is not reachable at {base_url} ({host}). "
        "Start llama-server first, then retry. Example:\n"
        "  llama-server -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf "
        "--mmproj mmproj-Qwen3-VL-8B-Instruct-F16.gguf -ngl 99 --port 8080\n"
        f"Original error: {exc}"
    )


def _extract_api_error(body: Any, status_code: Optional[int] = None) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or str(err)
            code = err.get("code", status_code)
            err_type = err.get("type")
            parts = [str(msg)]
            if code is not None:
                parts.append(f"(HTTP {code})")
            if err_type:
                parts.append(f"[{err_type}]")
            return " ".join(parts)
        if isinstance(err, str) and err.strip():
            return err
        if body.get("message"):
            return str(body["message"])
    if isinstance(body, str) and body.strip():
        preview = body.strip()
        if len(preview) > 800:
            preview = preview[:800] + "..."
        return preview
    if status_code:
        return f"HTTP {status_code} from llama-server"
    return "Unknown llama-server error"


def _decode_json_body(raw: bytes) -> Any:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LlamaServerError(
            f"llama-server returned malformed JSON: {e}. Body preview: {text[:400]}"
        ) from e


class LlamaServerClient:
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        debug: bool = False,
    ):
        self.base_url = normalize_base_url(server_url)
        key = (api_key or "").strip() or DEFAULT_API_KEY
        self.api_key = key
        self.timeout = DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self.debug = debug

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
        accept: Optional[str] = None,
        stream: bool = False,
        as_text: bool = False,
    ):
        url = _join(self.base_url, path)
        if as_text and not accept:
            accept = "text/plain"
        headers = self._headers({"Accept": accept} if accept else None)
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        wait = self.timeout if timeout is None else timeout
        try:
            resp = urllib.request.urlopen(req, timeout=wait)
        except urllib.error.HTTPError as e:
            raw = e.read() if e.fp is not None else b""
            try:
                body = json.loads(raw.decode("utf-8", errors="replace") or "null")
            except json.JSONDecodeError:
                body = raw.decode("utf-8", errors="replace")
            message = _extract_api_error(body, e.code)
            if e.code == 503:
                message = (
                    "llama-server is still loading the model (HTTP 503). "
                    "Wait until GET /health returns 200, then retry. "
                    f"{message}"
                )
            raise LlamaServerError(message, status_code=e.code, body=body, url=url) from e
        except socket.timeout as e:
            raise LlamaServerError(
                f"llama-server request timed out after {wait:.0f}s ({url}). "
                "Increase request_timeout or check that the model is not stuck.",
                url=url,
            ) from e
        except TimeoutError as e:
            raise LlamaServerError(
                f"llama-server request timed out after {wait:.0f}s ({url}). "
                "Increase request_timeout or check that the model is not stuck.",
                url=url,
            ) from e
        except urllib.error.URLError as e:
            reason = e.reason if e.reason is not None else e
            raise LlamaServerError(_friendly_connection_error(reason, self.base_url), url=url) from e
        except ConnectionError as e:
            raise LlamaServerError(_friendly_connection_error(e, self.base_url), url=url) from e

        if stream:
            return resp
        try:
            raw = resp.read()
        finally:
            resp.close()
        if as_text:
            return raw.decode("utf-8", errors="replace")
        return _decode_json_body(raw)

    def health(self) -> Dict[str, Any]:
        try:
            body = self.request(
                "GET",
                "/health",
                timeout=min(HEALTH_TIMEOUT, self.timeout),
                as_text=True,
            )
        except LlamaServerError as e:
            if e.status_code == 503:
                raise
            if e.status_code == 404:
                body = self.request(
                    "GET",
                    "/v1/health",
                    timeout=min(HEALTH_TIMEOUT, self.timeout),
                    as_text=True,
                )
            else:
                raise
        if isinstance(body, dict):
            return body
        text = str(body or "").strip()
        if not text or text.upper() in ("OK", "OK."):
            return {"status": "ok", "plain": True}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"status": "ok", "raw": text}

    def list_models(self) -> List[dict]:
        body = self.request("GET", "/v1/models", timeout=min(HEALTH_TIMEOUT, self.timeout))
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            return body["data"]
        return []

    def ensure_ready(self) -> None:
        try:
            self.health()
        except LlamaServerError:
            raise

    def tokenize(self, content: str, add_special: bool = False, parse_special: bool = True) -> List[int]:
        body = self.request(
            "POST",
            "/tokenize",
            {
                "content": content,
                "add_special": add_special,
                "parse_special": parse_special,
            },
            timeout=min(60.0, self.timeout),
        )
        tokens = body.get("tokens") if isinstance(body, dict) else None
        if not isinstance(tokens, list):
            raise LlamaServerError("llama-server /tokenize did not return a tokens array")
        ids: List[int] = []
        for item in tokens:
            if isinstance(item, int):
                ids.append(item)
            elif isinstance(item, dict) and "id" in item:
                ids.append(int(item["id"]))
        return ids

    def chat_completions(self, payload: dict, stream: bool = False) -> Dict[str, Any]:
        body = dict(payload)
        body["stream"] = bool(stream)
        if stream:
            return self._chat_completions_stream(body)
        result = self.request("POST", "/v1/chat/completions", body)
        if not isinstance(result, dict):
            raise LlamaServerError("llama-server /v1/chat/completions returned a non-JSON object")
        return result

    def completions(self, payload: dict, stream: bool = False) -> Dict[str, Any]:
        """Native llama.cpp /completion endpoint (used for raw_mode)."""
        body = dict(payload)
        body["stream"] = bool(stream)
        if stream:
            return self._completion_stream(body)
        result = self.request("POST", "/completion", body)
        if not isinstance(result, dict):
            raise LlamaServerError("llama-server /completion returned a non-JSON object")
        return result

    def embeddings(self, content: Any, embd_normalize: int = -1) -> Any:
        payload = {"content": content, "embd_normalize": embd_normalize}
        result = self.request("POST", "/embedding", payload)
        if isinstance(result, dict):
            return result
        raise LlamaServerError("llama-server /embedding returned a non-JSON object")

    def openai_embeddings(self, text: str, model: str = "local") -> Any:
        return self.request(
            "POST",
            "/v1/embeddings",
            {"input": text, "model": model, "encoding_format": "float"},
        )

    def _iter_sse(self, resp) -> Iterator[dict]:
        buffer = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                if buffer.strip():
                    line = buffer.decode("utf-8", errors="replace").strip()
                    parsed = _parse_sse_data_line(line)
                    if parsed is not None:
                        yield parsed
                break
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip()
                parsed = _parse_sse_data_line(line)
                if parsed is not None:
                    yield parsed

    def _chat_completions_stream(self, payload: dict) -> Dict[str, Any]:
        resp = self.request(
            "POST",
            "/v1/chat/completions",
            payload,
            accept="text/event-stream",
            stream=True,
        )
        collected_content: List[str] = []
        collected_reasoning: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        tick = 0
        check_every = 8
        max_tokens = int(payload.get("max_tokens") or 0)
        try:
            has_comfy, throw_interrupt = _comfy_interrupt()
            for event in self._iter_sse(resp):
                tick += 1
                if has_comfy and tick >= check_every:
                    tick = 0
                    throw_interrupt()
                if "usage" in event and event["usage"]:
                    prompt_tokens = event["usage"].get("prompt_tokens", prompt_tokens)
                    completion_tokens = event["usage"].get("completion_tokens", completion_tokens)
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if content:
                    collected_content.append(content)
                    completion_tokens += 1
                    _maybe_progress(completion_tokens, max_tokens)
                if reasoning:
                    collected_reasoning.append(reasoning)
        finally:
            try:
                resp.close()
            except Exception:
                pass
            if max_tokens > 0 or collected_content:
                sys.stderr.write("\n")
                sys.stderr.flush()

        content = "".join(collected_content)
        reasoning = "".join(collected_reasoning)
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _completion_stream(self, payload: dict) -> Dict[str, Any]:
        resp = self.request(
            "POST",
            "/completion",
            payload,
            accept="text/event-stream",
            stream=True,
        )
        collected: List[str] = []
        tokens_predicted = 0
        tokens_evaluated = 0
        tick = 0
        try:
            has_comfy, throw_interrupt = _comfy_interrupt()
            for event in self._iter_sse(resp):
                tick += 1
                if has_comfy and tick >= 8:
                    tick = 0
                    throw_interrupt()
                piece = event.get("content")
                if piece:
                    collected.append(piece)
                tokens_predicted = event.get("tokens_predicted", tokens_predicted)
                tokens_evaluated = event.get("tokens_evaluated", tokens_evaluated)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        return {
            "content": "".join(collected),
            "tokens_predicted": tokens_predicted,
            "tokens_evaluated": tokens_evaluated,
        }


def _parse_sse_data_line(line: str) -> Optional[dict]:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _comfy_interrupt() -> Tuple[bool, Any]:
    try:
        import comfy.model_management

        def _throw():
            comfy.model_management.throw_exception_if_processing_interrupted()

        return True, _throw
    except Exception:
        return False, lambda: None


def _maybe_progress(completion_tokens: int, max_tokens: int, bar_width: int = 30) -> None:
    if max_tokens > 0:
        progress = min(completion_tokens / max_tokens, 1.0)
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stderr.write(f"\r[{bar}] {completion_tokens}/{max_tokens} tokens")
    else:
        sys.stderr.write(f"\rGenerating: {completion_tokens} tokens...")
    sys.stderr.flush()


def parse_chat_message(result: dict) -> Tuple[str, str, dict]:
    """Return (content, reasoning_content, usage)."""
    if not isinstance(result, dict):
        raise LlamaServerError("Invalid chat completion response (not an object)")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaServerError(
            "llama-server returned no choices. "
            f"Response keys: {list(result.keys())}"
        )
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        # Some servers return a list of content parts
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            content = "".join(parts)
        else:
            content = str(content)
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )
    if reasoning is None:
        reasoning = ""
    usage = result.get("usage") or {}
    return content, str(reasoning), usage if isinstance(usage, dict) else {}


def parse_completion_text(result: dict) -> Tuple[str, dict]:
    if not isinstance(result, dict):
        raise LlamaServerError("Invalid /completion response (not an object)")
    if "content" in result and isinstance(result.get("content"), str):
        text = result["content"]
    elif result.get("choices"):
        choice = result["choices"][0]
        text = choice.get("text") or (choice.get("message") or {}).get("content") or ""
    else:
        raise LlamaServerError("llama-server /completion response has no content")
    usage = {
        "prompt_tokens": result.get("tokens_evaluated", 0),
        "completion_tokens": result.get("tokens_predicted", 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return text, usage


def wait_briefly(seconds: float = 0.0) -> None:
    if seconds > 0:
        time.sleep(seconds)
