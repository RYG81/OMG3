"""Shared, synchronous Ollama HTTP client for every ComfyUI-OMG node.

The module intentionally has no ComfyUI imports. It can therefore be unit-tested
independently and used by both the core and creative node families.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Generator
from typing import Any, TypeAlias
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_log = logging.getLogger(__name__)

Think: TypeAlias = bool | str | None
FormatType: TypeAlias = str | dict[str, Any] | None

__all__ = [
    "Think",
    "FormatType",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaHTTPError",
    "OllamaResponseError",
    "normalize_base_url",
    "check_ollama_alive",
    "get_installed_models",
    "get_model_names",
    "get_running_models",
    "get_model_info",
    "generate_stream",
    "generate",
    "chat_stream",
    "chat",
    "embed",
    "pull_model_stream",
    "delete_model",
    "copy_model",
    "close_session",
    "clear_session_cache",
]

_session_cache: dict[str, requests.Session] = {}
_session_lock = threading.Lock()


class OllamaError(RuntimeError):
    """Base error raised by the ComfyUI-OMG Ollama client."""


class OllamaConnectionError(OllamaError):
    """The configured Ollama endpoint could not be reached."""


class OllamaTimeoutError(OllamaConnectionError):
    """An Ollama request exceeded its timeout."""


class OllamaHTTPError(OllamaError):
    """Ollama returned a non-success HTTP response."""


class OllamaResponseError(OllamaError):
    """Ollama returned malformed or unexpected response data."""


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize an Ollama HTTP(S) base URL.

    Credentials are rejected because embedding them in a workflow leaks them into
    workflow JSON and logs. Remote Ollama authentication should be handled by a
    trusted reverse proxy or a future secret-backed profile implementation.
    """

    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Ollama URL cannot be empty")

    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama URL must use http:// or https://")
    if not parsed.hostname:
        raise ValueError("Ollama URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("Do not embed credentials in the Ollama URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not include a query string or fragment")

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + "/" + path.lstrip("/")


def _make_session(base_url: str) -> requests.Session:
    normalized = normalize_base_url(base_url)
    with _session_lock:
        cached = _session_cache.get(normalized)
        if cached is not None:
            return cached

        session = requests.Session()
        # Retry only idempotent discovery requests. Retrying generation, model pull,
        # copy, or delete can replay expensive/side-effecting POST/DELETE operations.
        retry = Retry(
            total=3,
            connect=3,
            read=0,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _session_cache[normalized] = session
        _log.debug("Created Ollama session for %s", normalized)
        return session


def close_session(base_url: str) -> None:
    normalized = normalize_base_url(base_url)
    with _session_lock:
        session = _session_cache.pop(normalized, None)
    if session is not None:
        session.close()


def clear_session_cache() -> None:
    with _session_lock:
        sessions = list(_session_cache.values())
        _session_cache.clear()
    for session in sessions:
        session.close()


def _extract_ollama_error(response: requests.Response) -> str:
    try:
        data = response.json()
    except (ValueError, requests.JSONDecodeError):
        data = None
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    text = (response.text or "").strip()
    return text[:1000] if text else f"HTTP {response.status_code}"


def _translate_request_error(exc: requests.RequestException, operation: str, base_url: str) -> OllamaError:
    endpoint = normalize_base_url(base_url)
    if isinstance(exc, requests.Timeout):
        return OllamaTimeoutError(f"{operation} timed out while contacting Ollama at {endpoint}")
    if isinstance(exc, requests.ConnectionError):
        return OllamaConnectionError(
            f"Cannot connect to Ollama at {endpoint} during {operation}. Is the server running?"
        )
    return OllamaConnectionError(f"Ollama request failed during {operation} at {endpoint}: {exc}")


def _raise_http(response: requests.Response, operation: str) -> None:
    if response.ok:
        return
    message = _extract_ollama_error(response)
    raise OllamaHTTPError(f"Ollama {operation} failed ({response.status_code}): {message}")


def _response_json(response: requests.Response, operation: str) -> dict[str, Any]:
    try:
        data = response.json()
    except (ValueError, requests.JSONDecodeError) as exc:
        raise OllamaResponseError(f"Ollama returned invalid JSON during {operation}") from exc
    if not isinstance(data, dict):
        raise OllamaResponseError(f"Ollama returned a non-object JSON response during {operation}")
    return data


def check_ollama_alive(base_url: str, timeout: float = 5.0) -> bool:
    try:
        response = _make_session(base_url).get(_url(base_url, "/"), timeout=timeout)
        return response.ok
    except (requests.RequestException, ValueError):
        return False


def get_installed_models(base_url: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    operation = "model discovery"
    try:
        response = _make_session(base_url).get(_url(base_url, "/api/tags"), timeout=timeout)
        _raise_http(response, operation)
        models = _response_json(response, operation).get("models", [])
    except requests.RequestException as exc:
        raise _translate_request_error(exc, operation, base_url) from exc
    if not isinstance(models, list):
        raise OllamaResponseError("Ollama /api/tags response has an invalid 'models' field")
    return [model for model in models if isinstance(model, dict)]


def get_model_names(base_url: str, timeout: float = 10.0) -> list[str]:
    names = [str(model["name"]) for model in get_installed_models(base_url, timeout) if model.get("name")]
    return sorted(set(names))


def get_running_models(base_url: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    operation = "running-model discovery"
    try:
        response = _make_session(base_url).get(_url(base_url, "/api/ps"), timeout=timeout)
        _raise_http(response, operation)
        models = _response_json(response, operation).get("models", [])
    except requests.RequestException as exc:
        raise _translate_request_error(exc, operation, base_url) from exc
    if not isinstance(models, list):
        raise OllamaResponseError("Ollama /api/ps response has an invalid 'models' field")
    return [model for model in models if isinstance(model, dict)]


def get_model_info(base_url: str, model: str, timeout: float = 10.0) -> dict[str, Any]:
    operation = f"model info for {model!r}"
    try:
        response = _make_session(base_url).post(
            _url(base_url, "/api/show"), json={"model": model}, timeout=timeout
        )
        _raise_http(response, operation)
        return _response_json(response, operation)
    except requests.RequestException as exc:
        raise _translate_request_error(exc, operation, base_url) from exc


def _stream_post(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    operation: str,
) -> Generator[dict[str, Any], None, None]:
    try:
        with _make_session(base_url).post(
            _url(base_url, endpoint), json=payload, stream=True, timeout=timeout
        ) as response:
            _raise_http(response, operation)
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise OllamaResponseError(
                        f"Ollama returned malformed streaming JSON during {operation}"
                    ) from exc
                if not isinstance(event, dict):
                    raise OllamaResponseError(
                        f"Ollama returned a non-object stream event during {operation}"
                    )
                if event.get("error"):
                    raise OllamaResponseError(f"Ollama {operation} failed: {event['error']}")
                yield event
    except requests.RequestException as exc:
        raise _translate_request_error(exc, operation, base_url) from exc


def _build_options(
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    num_predict: int,
    num_ctx: int,
    seed: int,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }
    if seed >= 0:
        options["seed"] = seed
    return options


def _normalize_format(response_format: FormatType, legacy_format: FormatType) -> FormatType:
    resolved = response_format if response_format not in ("", None) else legacy_format
    if resolved in ("", None):
        return None
    if isinstance(resolved, dict):
        return resolved
    if isinstance(resolved, str) and resolved.strip().lower() == "json":
        return "json"
    raise ValueError("Ollama response format must be 'json', a JSON Schema object, or empty")


def generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_predict: int = 512,
    num_ctx: int = 8192,
    seed: int = -1,
    response_format: FormatType = "",
    format: FormatType = "",
    think: Think = False,
    filter_thinking: bool = True,
    keep_alive: str = "5m",
    timeout: float = 120.0,
    images: list[str] | None = None,
) -> Generator[str, None, None]:
    resolved_format = _normalize_format(response_format, format)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": _build_options(
            temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed
        ),
    }
    if system:
        payload["system"] = system
    if resolved_format is not None:
        payload["format"] = resolved_format
    if think is not None:
        payload["think"] = think
    if images:
        payload["images"] = images

    saw_content = False
    thinking_fallback: list[str] = []
    # For filtering thinking tags <think>...</think> from output stream
    in_thinking_block = False
    buffer = ""

    for event in _stream_post(base_url, "/api/generate", payload, timeout, "generation"):
        token = event.get("response", "")
        if token:
            saw_content = True
            if filter_thinking:
                # Handle filtering of <think> tags that may appear in stream tokens
                # Accumulate tokens and filter blocks
                buffer += str(token)
                # Check for think tags in buffer
                # If we are inside thinking block, look for closing tag
                # This handles streaming where <think> and </think> may be split across tokens
                if "<think>" in buffer.lower() or "<thinking>" in buffer.lower():
                    # If buffer contains opening tag, enter thinking block mode
                    # Remove everything from opening tag to closing tag if both present
                    # Use regex to remove complete blocks
                    import re
                    # Remove complete <think>...</think> blocks
                    buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'<thinking>.*?</thinking>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'<thought>.*?</thought>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    # If buffer still contains opening tag without closing, hold it (don't yield yet)
                    if re.search(r'<think[^>]*>|<thinking[^>]*>|<thought[^>]*>', buffer, re.IGNORECASE):
                        # Still inside thinking block, don't yield, wait for closing
                        in_thinking_block = True
                        # If buffer is getting large without closing, yield what we have before opening tag
                        # Find opening tag position
                        m = re.search(r'<think[^>]*>|<thinking[^>]*>|<thought[^>]*>', buffer, re.IGNORECASE)
                        if m:
                            before = buffer[:m.start()]
                            if before:
                                yield before
                            buffer = buffer[m.start():]
                        continue
                    else:
                        in_thinking_block = False
                        # Buffer now has no thinking tags, yield it
                        if buffer:
                            yield buffer
                            buffer = ""
                        continue
                else:
                    # No think tags, yield token directly
                    if in_thinking_block:
                        # Check if closing tag appears
                        import re
                        if re.search(r'</think>|</thinking>|</thought>', buffer, re.IGNORECASE):
                            # Closing found, remove thinking block
                            buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                            buffer = re.sub(r'<thinking>.*?</thinking>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                            buffer = re.sub(r'<thought>.*?</thought>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                            in_thinking_block = False
                            if buffer:
                                yield buffer
                                buffer = ""
                            continue
                        else:
                            # Still inside thinking block, don't yield
                            continue
                    else:
                        yield str(token)
                        buffer = ""
            else:
                yield str(token)
        elif not saw_content and event.get("thinking"):
            thinking_fallback.append(str(event["thinking"]))
        if event.get("done"):
            if not saw_content and thinking_fallback:
                # If filter_thinking is True, don't yield thinking fallback as it's thinking content
                if not filter_thinking:
                    yield "".join(thinking_fallback)
                else:
                    # Filter thinking tags from fallback as well
                    import re
                    thinking_text = "".join(thinking_fallback)
                    thinking_text = re.sub(r'<think>.*?</think>', '', thinking_text, flags=re.DOTALL | re.IGNORECASE)
                    thinking_text = re.sub(r'<thinking>.*?</thinking>', '', thinking_text, flags=re.DOTALL | re.IGNORECASE)
                    if thinking_text.strip():
                        yield thinking_text
            # Yield any remaining buffer that is not thinking
            if filter_thinking and buffer:
                import re
                buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                buffer = re.sub(r'<thinking>.*?</thinking>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                if buffer and not re.search(r'<think[^>]*>|<thinking[^>]*>', buffer, re.IGNORECASE):
                    yield buffer
            return
    raise OllamaResponseError("Ollama generation stream ended before a completion event")


def generate(base_url: str, model: str, prompt: str, filter_thinking: bool = True, **kwargs: Any) -> str:
    # Pass filter_thinking to generate_stream
    return "".join(generate_stream(base_url, model, prompt, filter_thinking=filter_thinking, **kwargs))


def chat_stream(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    system: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_predict: int = 512,
    num_ctx: int = 8192,
    seed: int = -1,
    response_format: FormatType = "",
    format: FormatType = "",
    think: Think = False,
    filter_thinking: bool = True,
    keep_alive: str = "5m",
    timeout: float = 180.0,
) -> Generator[str, None, None]:
    resolved_format = _normalize_format(response_format, format)
    full_messages: list[dict[str, Any]] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "stream": True,
        "keep_alive": keep_alive,
        "options": _build_options(
            temperature, top_p, top_k, repeat_penalty, num_predict, num_ctx, seed
        ),
    }
    if resolved_format is not None:
        payload["format"] = resolved_format
    if think is not None:
        payload["think"] = think

    saw_content = False
    thinking_fallback: list[str] = []
    in_thinking_block = False
    buffer = ""
    for event in _stream_post(base_url, "/api/chat", payload, timeout, "chat"):
        message = event.get("message", {})
        if not isinstance(message, dict):
            raise OllamaResponseError("Ollama chat stream returned an invalid message")
        token = message.get("content", "")
        if token:
            saw_content = True
            if filter_thinking:
                import re
                buffer += str(token)
                buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                buffer = re.sub(r'<thinking>.*?</thinking>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                buffer = re.sub(r'<thought>.*?</thought>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                if re.search(r'<think[^>]*>|<thinking[^>]*>|<thought[^>]*>', buffer, re.IGNORECASE):
                    in_thinking_block = True
                    m = re.search(r'<think[^>]*>|<thinking[^>]*>|<thought[^>]*>', buffer, re.IGNORECASE)
                    if m:
                        before = buffer[:m.start()]
                        if before:
                            yield before
                        buffer = buffer[m.start():]
                    continue
                else:
                    in_thinking_block = False
                    if buffer:
                        yield buffer
                        buffer = ""
                    continue
            else:
                yield str(token)
        elif not saw_content and message.get("thinking"):
            thinking_fallback.append(str(message["thinking"]))
        if event.get("done"):
            if not saw_content and thinking_fallback:
                if not filter_thinking:
                    yield "".join(thinking_fallback)
                else:
                    import re
                    thinking_text = "".join(thinking_fallback)
                    thinking_text = re.sub(r'<think>.*?</think>', '', thinking_text, flags=re.DOTALL | re.IGNORECASE)
                    thinking_text = re.sub(r'<thinking>.*?</thinking>', '', thinking_text, flags=re.DOTALL | re.IGNORECASE)
                    if thinking_text.strip():
                        yield thinking_text
            if filter_thinking and buffer:
                import re
                buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                buffer = re.sub(r'<thinking>.*?</thinking>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                if buffer and not re.search(r'<think[^>]*>|<thinking[^>]*>', buffer, re.IGNORECASE):
                    yield buffer
            return
    raise OllamaResponseError("Ollama chat stream ended before a completion event")


def chat(base_url: str, model: str, messages: list[dict[str, Any]], filter_thinking: bool = True, **kwargs: Any) -> str:
    return "".join(chat_stream(base_url, model, messages, filter_thinking=filter_thinking, **kwargs))


def embed(
    base_url: str,
    model: str,
    texts: list[str],
    timeout: float = 60.0,
) -> list[list[float]]:
    operation = "embedding"
    try:
        response = _make_session(base_url).post(
            _url(base_url, "/api/embed"),
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        _raise_http(response, operation)
        embeddings = _response_json(response, operation).get("embeddings", [])
    except requests.RequestException as exc:
        raise _translate_request_error(exc, operation, base_url) from exc
    if not isinstance(embeddings, list) or not all(isinstance(item, list) for item in embeddings):
        raise OllamaResponseError("Ollama /api/embed returned an invalid embeddings field")
    return embeddings


def pull_model_stream(
    base_url: str,
    model: str,
    timeout: float = 600.0,
) -> Generator[dict[str, Any], None, None]:
    payload = {"model": model, "stream": True}
    yield from _stream_post(base_url, "/api/pull", payload, timeout, "model pull")


def delete_model(base_url: str, model: str, timeout: float = 30.0) -> bool:
    operation = f"delete model {model!r}"
    try:
        response = _make_session(base_url).delete(
            _url(base_url, "/api/delete"), json={"model": model}, timeout=timeout
        )
        _raise_http(response, operation)
        return True
    except (requests.RequestException, OllamaError, ValueError) as exc:
        _log.warning("%s failed: %s", operation, exc)
        return False


def copy_model(
    base_url: str,
    source: str,
    destination: str,
    timeout: float = 30.0,
) -> bool:
    operation = f"copy model {source!r} to {destination!r}"
    try:
        response = _make_session(base_url).post(
            _url(base_url, "/api/copy"),
            json={"source": source, "destination": destination},
            timeout=timeout,
        )
        _raise_http(response, operation)
        return True
    except (requests.RequestException, OllamaError, ValueError) as exc:
        _log.warning("%s failed: %s", operation, exc)
        return False
