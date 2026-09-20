# qwen3vl_run.py
"""Qwen3-VL (and other GGUF VL models) inference via llama-server HTTP API.

ComfyUI never loads GGUF files. llama.cpp llama-server holds the model and mmproj.
"""
import sys
import io
import json
import os
import base64
import time
import gc
import numpy as np
import tempfile
import traceback
import re
from PIL import Image
from typing import List, Optional, Any, Dict, Tuple

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from debug_print import _debug_print, _debug_info
from llama_server_client import (
    DEFAULT_SERVER_URL,
    LlamaServerClient,
    LlamaServerError,
    MTMD_MEDIA_MARKER,
    parse_chat_message,
    parse_completion_text,
)
from llama_swap_client import LlamaSwapClient, is_placeholder_model

# Kept for UnloadQwenModel / cache_mode compatibility. The LLM itself lives in llama-server.
_model_caches = {
    "keep_vram": {"llm": None, "hash": None},
    "save1":      {"llm": None, "hash": None},
    "save2":      {"llm": None, "hash": None},
    "save3":      {"llm": None, "hash": None},
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _norm_str(value, none_is_empty=True):
    """
    Нормализует строковое значение из конфига.
    - None, "" -> ""
    - " None " -> "" (если none_is_empty=True)
    - " Qwen3 " -> "qwen3"
    """
    if not value:
        return ""
    val = str(value).lower().strip()
    if none_is_empty and val == "none":
        return ""
    return val


def _norm_3state_bool(value):
    """
    Нормализует значение в True или False.
    Возвращает None для всего остального (например, "auto", None, пустая строка, опечатки).
    """
    if value is True or str(value).strip().lower() in ("true", "1"):
        return True
    if value is False or str(value).strip().lower() in ("false", "0"):
        return False
    return None


def _norm_default(value, default):
    """
    Если значение равно значению по умолчанию (после обработки JSON), рассматривать его как None.
    Полезно для числовых значений по умолчанию, таких как n_keep=-1, embedding_scale=1.0.
    """
    if value is None:
        return None
    try:
        if value == default:
            return None
    except Exception:
        pass
    return value


def _parse_strings_list(value):
    """
    Parse stop sequences from widget string.
    Accepts:
      - JSON list with double quotes: '["a","b"]'
      - JSON-like with single quotes:  "['a','b']"
      - Comma-separated:               'a,b' or '"a","b"' or "'a','b'"
    Returns list of clean strings (empty list if no value).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().strip('\'"') for x in value if str(x).strip()]

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip().strip('\'"') for x in parsed if str(x).strip()]
            return [str(parsed).strip().strip('\'"')]
        except Exception:
            pass

        try:
            fixed = s.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
            if not s:
                return []

    return [x.strip().strip('\'"') for x in s.split(",") if x.strip().strip('\'"')]


def _parse_float_list(value):
    """
    Parse tensor_split-like list.
    Accepts: '[0.7,0.3]' or '0.7,0.3'
    """
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except (ValueError, TypeError):
            return []

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
            if not s:
                return []

    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def build_prompt(template: str, system: str, user: str):
    result = template.replace("{system}", system).replace("{user}", user)
    result = result.replace('\\n', '\n')

    if "{images}" in result:
        parts = result.split("{images}", 1)
        return parts[0], parts[1]
    else:
        return result, ""


def _file_to_data_url(path: str, mime: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        print(f"build_media: file not found: {path}", file=sys.stderr)
        return None
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _pil_to_data_url(image_item: Image.Image, quality: int = 95) -> str:
    buffer = io.BytesIO()
    image_item.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def _data_url_to_raw_b64(data_url: str) -> str:
    if ";base64," in data_url:
        return data_url.split(";base64,", 1)[1]
    return data_url


def _build_image_content(image_item, quality=95):
    """OpenAI-compatible image part for llama-server /v1/chat/completions."""
    if isinstance(image_item, Image.Image):
        file_url = _pil_to_data_url(image_item, quality=quality)
        return {"type": "image_url", "image_url": {"url": file_url}}

    if isinstance(image_item, str):
        path = Path(image_item)
        if path.exists():
            suffix = path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
            }.get(suffix, "image/jpeg")
            if suffix in (".png", ".webp", ".gif", ".bmp"):
                file_url = _file_to_data_url(image_item, mime)
            else:
                with Image.open(path) as img:
                    file_url = _pil_to_data_url(img, quality=quality)
            if not file_url:
                return None
            return {"type": "image_url", "image_url": {"url": file_url}}
        print(f"build_image: Image file not found: {image_item}", file=sys.stderr)
        return None

    print(f"build_image: Unsupported type: {type(image_item)}", file=sys.stderr)
    return None


def _build_audio_content(audio_item):
    if isinstance(audio_item, bytes):
        b64_data = base64.b64encode(audio_item).decode("utf-8")
        return {
            "type": "input_audio",
            "input_audio": {"data": b64_data, "format": "wav"},
        }

    if isinstance(audio_item, str):
        path = Path(audio_item)
        if path.exists():
            with open(path, "rb") as f:
                wav_bytes = f.read()
            b64_data = base64.b64encode(wav_bytes).decode("utf-8")
            fmt = path.suffix.lower().lstrip(".") or "wav"
            return {
                "type": "input_audio",
                "input_audio": {"data": b64_data, "format": fmt},
            }
        print(f"build_audio: Audio file not found: {audio_item}", file=sys.stderr)
        return None

    print(f"build_audio: Unsupported type: {type(audio_item)}", file=sys.stderr)
    return None


def _build_native_video_content(video_input):
    """llama-server input_video part (ffmpeg on the server)."""
    if isinstance(video_input, bytes):
        b64_data = base64.b64encode(video_input).decode("utf-8")
        return {"type": "input_video", "input_video": {"data": b64_data}}
    if isinstance(video_input, str) and Path(video_input).exists():
        raw = Path(video_input).read_bytes()
        b64_data = base64.b64encode(raw).decode("utf-8")
        return {"type": "input_video", "input_video": {"data": b64_data}}
    return None


def _build_video_content(video_input, config):
    """Default: subsample frames as JPEG image_url parts (preserves max_frames / trim)."""
    native = _norm_3state_bool(config.get("native_video"))
    if native:
        native_part = _build_native_video_content(video_input)
        if native_part is not None:
            return [native_part]
        print("[WARNING] native_video requested but input is not a file/bytes; falling back to frames.", file=sys.stderr)

    max_frames = config.get("max_frames", 24)
    quality = config.get("frame_quality", 75)
    trim_start = config.get("trim_start", 0.0)
    trim_duration = config.get("trim_duration", 0.0)
    frame_id = _norm_default(config.get("add_frame_id", ""), "")

    video_content_items = []
    frames_to_process = []

    if isinstance(video_input, str):
        import cv2

        video_path = video_input
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        start_frame = int(trim_start * fps)
        if trim_duration > 0:
            end_frame = int((trim_start + trim_duration) * fps)
        else:
            end_frame = total_frames

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        effective_total = end_frame - start_frame

        if effective_total > max_frames:
            indices = set(np.linspace(0, effective_total - 1, max_frames, dtype=int).tolist())
        else:
            indices = set(range(effective_total))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_idx = start_frame
        selected_count = 0

        while cap.isOpened() and current_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_idx - start_frame) in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_to_process.append(frame_rgb)
                selected_count += 1
                if selected_count >= max_frames:
                    break
            current_idx += 1
        cap.release()

    elif isinstance(video_input, np.ndarray):
        np_frames = video_input
        if len(np_frames.shape) != 4:
            print(f"[ERROR] Invalid numpy array shape for video: {np_frames.shape}", file=sys.stderr)
            return []

        total_frames = np_frames.shape[0]

        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames_to_process = [np_frames[i] for i in indices]
        else:
            frames_to_process = [np_frames[i] for i in range(total_frames)]

    else:
        print(f"[ERROR] Unsupported video_input type in _build_video_content: {type(video_input)}", file=sys.stderr)
        return []

    num = 0
    for frame_rgb in frames_to_process:
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        img_bytes = buf.getvalue()
        b64_data = base64.b64encode(img_bytes).decode("utf-8")

        if frame_id:
            video_content_items.append({"type": "text", "text": frame_id.replace("{num}", str(num))})

        video_content_items.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
        })
        num += 1

    return video_content_items


def _debug_calc_speed(result, exec_time):
    if exec_time == 0:
        return 0, 0
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", 0) or 0
    speed = completion_tokens / exec_time if exec_time else 0
    return completion_tokens, speed


def _resolve_model_name(config: dict) -> str:
    if config.get("use_llama_swap"):
        swap_model = (config.get("llama_swap_model") or "").strip()
        if not is_placeholder_model(swap_model):
            return swap_model
    model = (config.get("model") or "").strip()
    if model:
        return model
    model_path = (config.get("model_path") or "").strip()
    if model_path:
        return Path(model_path).name
    return "local"


def _apply_llama_swap_url(config: dict) -> None:
    if not config.get("use_llama_swap"):
        return
    swap_url = (config.get("llama_swap_url") or config.get("server_url") or "").strip()
    if swap_url:
        config["server_url"] = swap_url


def _resolve_timeout(config: dict) -> float:
    timeout = config.get("request_timeout", config.get("timeout", 300))
    try:
        return float(timeout)
    except (TypeError, ValueError):
        return 300.0


def _thinking_enabled(config: dict) -> bool:
    enable = _norm_3state_bool(config.get("enable_thinking"))
    force = _norm_3state_bool(config.get("force_reasoning"))
    return bool(enable) or bool(force)


def _apply_thinking_fields(payload: dict, config: dict) -> None:
    """Map node thinking flags to current llama-server chat API fields.

    llama-server documents:
      chat_template_kwargs: {"enable_thinking": false}
      reasoning_effort: "none" disables thinking
      reasoning_format: "none" | "deepseek" | ...
    """
    thinking = _thinking_enabled(config)
    kwargs = dict(config.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = thinking
    payload["chat_template_kwargs"] = kwargs
    if thinking:
        payload["reasoning_format"] = config.get("reasoning_format") or "deepseek"
        effort = config.get("reasoning_effort")
        if effort and str(effort).strip().lower() not in ("", "default"):
            payload["reasoning_effort"] = effort
    else:
        payload["reasoning_effort"] = "none"
        payload["reasoning_format"] = config.get("reasoning_format") or "none"


def _sampling_payload(config: dict) -> dict:
    payload = {
        "max_tokens": config.get("max_tokens", config.get("output_max_tokens", 2048)),
        "temperature": config.get("temperature", 0.7),
        "seed": config.get("seed", 42),
        "repeat_penalty": config.get("repeat_penalty", 1.1),
        "frequency_penalty": config.get("frequency_penalty", 0.0),
        "presence_penalty": config.get("presence_penalty", config.get("present_penalty", 0.0)),
        "top_p": config.get("top_p", 0.92),
        "min_p": config.get("min_p", 0.05),
        "top_k": config.get("top_k", 0),
    }
    custom_stop = _parse_strings_list(config.get("stop"))
    if custom_stop:
        payload["stop"] = custom_stop
    return payload


def _merge_extra_completion(payload: dict, config: dict) -> None:
    for key, value in config.items():
        if key.startswith("extra_completion_"):
            payload[key[len("extra_completion_"):]] = value
    if config.get("response_format"):
        payload["response_format"] = config["response_format"]
    if config.get("json_schema"):
        payload["response_format"] = {
            "type": "json_object",
            "schema": config["json_schema"],
        }
    if config.get("grammar"):
        payload["grammar"] = config["grammar"]


def _logit_bias_from_banned_words(client: LlamaServerClient, words_to_ban, debug=False) -> Optional[dict]:
    words = _parse_strings_list(words_to_ban)
    if not words:
        return None
    banned_ids = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        variants = {word, " " + word}
        for variant in variants:
            try:
                tokens = client.tokenize(variant, add_special=False, parse_special=False)
                banned_ids.extend(tokens)
            except Exception as e:
                print(f"[LogitBias Warning] Failed to tokenize '{variant}': {e}", file=sys.stderr)
    unique_ids = sorted(set(int(t) for t in banned_ids))
    if not unique_ids:
        return None
    if debug:
        _debug_info(debug, "logit_bias", text=f"{len(unique_ids)} banned token ids")
    return {str(token_id): -100.0 for token_id in unique_ids}


def _build_multimodal_content(config, images, audios, videos, user_prompt: str) -> List[dict]:
    content = []
    image_quality = config.get("image_quality", 95)
    user_prompt_after_content = config.get("user_prompt_after_content", True)
    image_id = _norm_default(config.get("add_image_id", ""), "")
    audio_id = _norm_default(config.get("add_audio_id", ""), "")

    if not user_prompt_after_content:
        content.append({"type": "text", "text": user_prompt})

    num = 0
    for img_item in images:
        img_content = _build_image_content(img_item, quality=image_quality)
        if img_content is not None:
            if image_id:
                content.append({"type": "text", "text": image_id.replace("{num}", str(num))})
            content.append(img_content)
            num += 1

    num = 0
    for aud_item in audios:
        aud_content = _build_audio_content(aud_item)
        if aud_content is not None:
            if audio_id:
                content.append({"type": "text", "text": audio_id.replace("{num}", str(num))})
            content.append(aud_content)
            num += 1

    for path in videos:
        frames_items = _build_video_content(path, config)
        if frames_items:
            content.extend(frames_items)

    if user_prompt_after_content:
        content.append({"type": "text", "text": user_prompt})
    return content


def _build_chat_messages(config, images, audios, videos) -> List[dict]:
    system_prompt = config.get("system_prompt", "").strip()
    user_prompt = config.get("user_prompt", "").strip()
    has_media = bool(images or audios or videos)

    if has_media:
        content = _build_multimodal_content(config, images, audios, videos, user_prompt)
        user_message = {"role": "user", "content": content}
    else:
        user_message = {"role": "user", "content": user_prompt}

    if system_prompt:
        return [{"role": "system", "content": system_prompt}, user_message]
    return [user_message]


def _extract_media_b64(content_parts: List[dict]) -> List[str]:
    raw = []
    for part in content_parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            raw.append(_data_url_to_raw_b64(url))
        elif part.get("type") == "input_audio":
            data = (part.get("input_audio") or {}).get("data") or ""
            raw.append(data)
        elif part.get("type") == "input_video":
            data = (part.get("input_video") or {}).get("data") or ""
            raw.append(data)
    return raw


def _raw_mode_prompt_and_media(config, images, audios, videos) -> Tuple[str, List[str]]:
    template_str = config.get("prompt_template", "")
    if not template_str:
        raise ValueError("raw_mode is enabled but prompt_template is empty. Please provide a valid prompt_template")

    system_prompt = config.get("system_prompt", "").strip()
    user_prompt = config.get("user_prompt", "").strip()
    text_before, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)

    image_quality = config.get("image_quality", 95)
    media_b64 = []
    marker_blob = ""

    for img_item in images:
        img_content = _build_image_content(img_item, quality=image_quality)
        if img_content is not None:
            url = img_content["image_url"]["url"]
            media_b64.append(_data_url_to_raw_b64(url))
            marker_blob += MTMD_MEDIA_MARKER

    for path in videos:
        frames_items = _build_video_content(path, config)
        if not frames_items:
            continue
        for part in frames_items:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                media_b64.append(_data_url_to_raw_b64(url))
                marker_blob += MTMD_MEDIA_MARKER
            elif part.get("type") == "input_video":
                media_b64.append(part["input_video"]["data"])
                marker_blob += MTMD_MEDIA_MARKER

    prompt = text_before + marker_blob + text_after
    return prompt, media_b64


def _postprocess_output(output: str, reasoning: str, config: dict) -> str:
    if reasoning and reasoning.strip():
        if "<think>" not in (output or "") and "</think>" not in (output or ""):
            output = f"<think>\n{reasoning.strip()}\n</think>\n{output or ''}"

    if config.get("raw_output", False):
        return output or ""

    if config.get("remove_thinking", False):
        cut_prefix = config.get("answer_delimiter")
        if cut_prefix and cut_prefix in output:
            output = output.split(cut_prefix)[-1]

        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        if "</think>" in output:
            output = output.split("</think>")[-1]

        output = re.sub(r"<\|channel>.*?<channel\|>", "", output, flags=re.DOTALL)
        if "<channel|>" in output:
            output = output.split("<channel|>")[-1]

        output = re.sub(r"\n\s*\n+", "\n\n", output)

    return (output or "").strip()


def _make_client(config: dict) -> LlamaServerClient:
    return LlamaServerClient(
        server_url=config.get("server_url") or DEFAULT_SERVER_URL,
        api_key=config.get("api_key") or config.get("api_token") or "",
        timeout=_resolve_timeout(config),
        debug=config.get("debug", True),
    )


def _safe_swap_logs(config: dict, model_name: str = "") -> str:
    if not config.get("use_llama_swap"):
        return ""
    try:
        lines = int(config.get("llama_swap_log_lines", 200) or 200)
    except (TypeError, ValueError):
        lines = 200
    try:
        url = config.get("llama_swap_url") or config.get("server_url") or DEFAULT_SERVER_URL
        swap = LlamaSwapClient(url, api_key=config.get("api_key") or "", timeout=15)
        return swap.get_logs(max_lines=lines, model_id=model_name)
    except Exception as e:
        return f"[llama-swap logs unavailable: {e}]"


def _with_swap_log(result: dict, config: dict, model_name: str = "") -> dict:
    if isinstance(result, dict) and "llama_swap_log" not in result:
        result["llama_swap_log"] = _safe_swap_logs(config, model_name)
    return result


def _warn_ignored_local_loader_settings(config: dict, debug: bool) -> None:
    ignored = []
    if config.get("n_gpu_layers") not in (None, -1):
        ignored.append("n_gpu_layers")
    if config.get("n_ctx") not in (None, 8192):
        ignored.append("n_ctx")
    if config.get("speculative_enabled"):
        ignored.append("speculative_enabled")
    if config.get("chat_handler") and _norm_str(config.get("chat_handler")):
        ignored.append("chat_handler")
    if not ignored:
        return
    _debug_info(
        debug,
        "llama-server",
        text=(
            "The following settings are llama-server CLI / GGUF-template concerns "
            f"and are not applied per HTTP request: {', '.join(ignored)}. "
            "Start llama-server with -m / --mmproj / -ngl / -c as needed."
        ),
        file=sys.stderr,
    )


def _inference(config):
    """HTTP inference against llama-server. Never imports llama_cpp."""
    try:
        debug = config.get("debug", True)
        streaming_mode = config.get("streaming_mode", False)
        extract_embedding = config.get("extract_embedding", False)
        extract_tts = config.get("extract_tts", False)
        raw_mode = config.get("raw_mode", False)

        system_prompt = config.get("system_prompt", "").strip()
        user_prompt = config.get("user_prompt", "").strip()
        if not (config.get("server_url") or "").strip() and not (config.get("model_path") or "").strip():
            # Keep a useful error if neither a server nor a legacy model_path hint exists.
            # model_path is optional now; server_url has a default.
            pass

        images = config.get("images") or config.get("images_path") or []
        if not isinstance(images, list):
            images = [images] if images else []
        num_images = len(images)

        audios = config.get("audios") or config.get("audios_path") or []
        if not isinstance(audios, list):
            audios = [audios] if audios else []
        num_audios = len(audios)

        videos = config.get("videos") or config.get("videos_path") or []
        if not isinstance(videos, list):
            videos = [videos] if videos else []
        num_videos = len(videos)

        num_content = num_images + num_audios + num_videos
        content_text = ""
        if num_content:
            content_text = f"(with {num_images}/{num_audios}/{num_videos} image/audio/video)"

        _apply_llama_swap_url(config)
        use_llama_swap = bool(config.get("use_llama_swap"))

        t_client = time.perf_counter()
        client = _make_client(config)
        _debug_print(debug, f"llama-server {client.base_url}", t_client, file=sys.stderr)
        _warn_ignored_local_loader_settings(config, debug)

        try:
            client.ensure_ready()
        except LlamaServerError as e:
            return _with_swap_log({"status": "error", "message": str(e), "traceback": traceback.format_exc()}, config, ""), None

        model_name = _resolve_model_name(config)

        if use_llama_swap and not is_placeholder_model(config.get("llama_swap_model")):
            t_swap = time.perf_counter()
            try:
                swap = LlamaSwapClient(
                    client.base_url,
                    api_key=config.get("api_key") or "",
                    timeout=_resolve_timeout(config),
                )
                swap.load_model(model_name)
                _debug_print(debug, f"llama-swap load {model_name}", t_swap, file=sys.stderr)
            except Exception as e:
                return _with_swap_log({
                    "status": "error",
                    "message": f"llama-swap could not load model '{model_name}': {e}",
                    "traceback": traceback.format_exc(),
                }, config, model_name), None

        output = ""
        output_data = None
        data_type = 0

        if extract_tts:
            return _with_swap_log({
                "status": "error",
                "message": "extract_tts is not supported with the llama-server HTTP backend.",
            }, config, model_name), None

        if extract_embedding:
            t_emb = time.perf_counter()
            try:
                template_str = config.get("prompt_template", "")
                if template_str:
                    text_before, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)
                    prompt = text_before + text_after
                else:
                    prompt = user_prompt

                if config.get("tokenizer_path"):
                    print(
                        "[WARNING] tokenizer_path is ignored: embeddings use the tokenizer inside llama-server.",
                        file=sys.stderr,
                    )

                try:
                    response = client.embeddings(prompt, embd_normalize=-1)
                except LlamaServerError as e:
                    print(f"[WARNING] /embedding failed ({e}); trying /v1/embeddings", file=sys.stderr)
                    response = client.openai_embeddings(prompt, model=model_name)

                emb = None
                if isinstance(response, dict):
                    if "embedding" in response:
                        emb = response["embedding"]
                    elif response.get("data"):
                        emb = response["data"][0].get("embedding")

                if emb is None:
                    raise LlamaServerError("Embedding response did not contain an embedding vector")

                if isinstance(emb, list):
                    emb_np = np.array(emb, dtype=np.float32)
                else:
                    emb_np = np.array([emb], dtype=np.float32)

                scale = _norm_default(config.get("embedding_scale"), 1.0)
                if scale is not None:
                    emb_np = (emb_np * scale).astype(np.float32)

                output_data = emb_np
                data_type = 1
            except Exception as e:
                print(f"[WARNING] Embedding extraction failed: {e}", file=sys.stderr)
            _debug_print(debug, "get embedding", t_emb, file=sys.stderr)
            return _with_swap_log({"status": "success", "output": output, "data_type": data_type}, config, model_name), output_data

        sampling = _sampling_payload(config)
        logit_bias = _logit_bias_from_banned_words(client, config.get("words_to_ban"), debug=debug)
        if logit_bias:
            sampling["logit_bias"] = logit_bias
        _merge_extra_completion(sampling, config)

        t3 = time.perf_counter()
        t_inference0 = time.perf_counter()
        reasoning = ""
        usage = {}

        if raw_mode:
            prompt, media_b64 = _raw_mode_prompt_and_media(config, images, audios, videos)
            completion_body = {
                "prompt": prompt if not media_b64 else {
                    "prompt_string": prompt,
                    "multimodal_data": media_b64,
                },
                "n_predict": sampling.get("max_tokens", 2048),
                "temperature": sampling.get("temperature"),
                "top_p": sampling.get("top_p"),
                "top_k": sampling.get("top_k"),
                "min_p": sampling.get("min_p"),
                "repeat_penalty": sampling.get("repeat_penalty"),
                "presence_penalty": sampling.get("presence_penalty"),
                "frequency_penalty": sampling.get("frequency_penalty"),
                "seed": sampling.get("seed"),
            }
            if sampling.get("stop"):
                completion_body["stop"] = sampling["stop"]
            if sampling.get("logit_bias"):
                completion_body["logit_bias"] = sampling["logit_bias"]
            if sampling.get("grammar"):
                completion_body["grammar"] = sampling["grammar"]
            if sampling.get("json_schema") or (isinstance(sampling.get("response_format"), dict) and sampling["response_format"].get("schema")):
                schema = sampling.get("json_schema") or sampling["response_format"].get("schema")
                completion_body["json_schema"] = schema

            _debug_print(debug, f"create raw prompt {content_text}", t3, file=sys.stderr)
            t_inference0 = time.perf_counter()
            result = client.completions(completion_body, stream=streaming_mode)
            output, usage = parse_completion_text(result)
            result = {"choices": [{"message": {"content": output}}], "usage": usage}
        else:
            messages = _build_chat_messages(config, images, audios, videos)
            payload = {
                "model": model_name,
                "messages": messages,
                **sampling,
            }
            _apply_thinking_fields(payload, config)
            _debug_print(debug, f"create message {content_text}", t3, file=sys.stderr)
            t_inference0 = time.perf_counter()
            result = client.chat_completions(payload, stream=streaming_mode)
            output, reasoning, usage = parse_chat_message(result)
            result = {
                "choices": [{"message": {"content": output, "reasoning_content": reasoning}}],
                "usage": usage,
            }

        t_inference1 = time.perf_counter()
        if debug:
            completion_tokens, speed = _debug_calc_speed(result, t_inference1 - t_inference0)
            _debug_print(
                debug,
                "inference (llama-server)",
                t_inference0,
                text=f"{speed:.2f} tok/sec {completion_tokens} tokens",
                file=sys.stderr,
            )

        output = _postprocess_output(output, reasoning, config)

        if config.get("debug_output", False):
            print(f"[DEBUG] LLM output: {output}", file=sys.stderr)

        return _with_swap_log({"status": "success", "output": output, "data_type": data_type}, config, model_name), output_data

    except LlamaServerError as e:
        return _with_swap_log({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }, config, config.get("llama_swap_model") or ""), None
    except Exception as e:
        return _with_swap_log({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }, config, config.get("llama_swap_model") or ""), None


def run_inference_direct(config):
    """Функция для прямого вызова. Возвращает словарь с результатом."""
    return _inference(config)


def unload_llama_model(gccollect, debug=False, target="all", server_url=None, model_id=None):
    """Clear local caches and remotely unload the GGUF (llama-swap / router)."""
    global _model_caches
    targets = list(_model_caches.keys()) if target == "all" else ([target] if target in _model_caches else [])
    for key in targets:
        _model_caches[key]["llm"] = None
        _model_caches[key]["hash"] = None

    url = (server_url or "").strip()
    if url:
        from llama_swap_client import unload_models
        result = unload_models(url, model_id=model_id)
        _debug_info(debug, "unload_llama_model", text=str(result), file=sys.stderr)
        if gccollect:
            t_start = time.perf_counter()
            gc.collect()
            _debug_print(debug, "gc.collect", t_start, file=sys.stderr)
        return result

    _debug_info(
        debug,
        "unload_llama_model",
        text=(
            "No server_url: only local caches were cleared. "
            "Pass the llama-swap / llama-server URL to free GGUF VRAM."
        ),
        file=sys.stderr,
    )
    if gccollect:
        t_start = time.perf_counter()
        gc.collect()
        _debug_print(debug, "gc.collect", t_start, file=sys.stderr)
    return {"success": False, "message": "no server_url"}


original_stdout_fd = None


def save_dup():
    global original_stdout_fd
    try:
        original_stdout_fd = os.dup(1)
    except OSError:
        pass


def swap_dup():
    global original_stdout_fd
    if original_stdout_fd is not None:
        try:
            os.dup2(2, 1)
        except OSError as e:
            print(f"Warning: Failed to redirect stdout: {e}", file=sys.stderr)


def restore_dup():
    global original_stdout_fd
    if original_stdout_fd is not None:
        try:
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
        except OSError:
            pass
        original_stdout_fd = None


def main():
    try:
        if len(sys.argv) != 2:
            print(json.dumps({"status": "error", "message": "sys.argv != 2"}, ensure_ascii=True), flush=True)
            os._exit(1)
        config_path = sys.argv[1]

        if not Path(config_path).exists():
            print(json.dumps({"status": "error", "message": "Config file not found"}, ensure_ascii=True), flush=True)
            os._exit(1)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Failed to load config: {e}"}, ensure_ascii=True), flush=True)
            os._exit(1)

        save_dup()
        swap_dup()

        result, data_type = _inference(config)

        if data_type is not None:
            import pickle

            t_save_data = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(data_type, f)
                data_path = f.name
            result["data_file"] = data_path
            debug = config.get("debug", True)
            _debug_print(debug, "save data", t_save_data, file=sys.stderr)

        restore_dup()

        print(json.dumps(result, ensure_ascii=True), flush=True)

        if result["status"] == "error":
            os._exit(1)

        os._exit(0)

    except Exception as e:
        restore_dup()
        print(json.dumps({"status": "error", "message": f"Critical error in main: {e}"}, ensure_ascii=True), flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
