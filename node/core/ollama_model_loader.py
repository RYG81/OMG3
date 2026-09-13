"""Ollama model connection/profile node.

Model discovery is deliberately not performed from INPUT_TYPES: ComfyUI calls
schema discovery while loading the UI, and network work there can block startup
or incorrectly tie every workflow to localhost. Users enter an exact model name;
execution validates it against the selected local or remote Ollama server.
"""

from __future__ import annotations

import logging
from typing import Any

from ...utils.capabilities import build_capability_report
from ...utils.system_prompt import custom_system_prompt_hash, normalize_custom_system_prompt
from ...utils.text_utils import filter_thinking
from .ollama_client import (
    OllamaError,
    check_ollama_alive,
    get_model_info,
    get_model_names,
    get_running_models,
    normalize_base_url,
)

_log = logging.getLogger(__name__)
_DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaModelLoader:
    """Validate an Ollama server/model and emit a reusable model profile."""

    CATEGORY = "ComfyUI-OMG/Core"
    FUNCTION = "load_model"
    RETURN_TYPES = ("OLLAMA_MODEL", "STRING")
    RETURN_NAMES = ("ollama_model", "model_name")
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Main Ollama entry node: validate a local/remote model, configure shared sampling/context, "
        "and optionally append global custom system instructions to every connected generative node."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Keep this method pure and offline-safe. A frontend refreshable model
        # selector will be added later without making /object_info do network I/O.
        return {
            "required": {
                "ollama_url": (
                    "STRING",
                    {
                        "default": _DEFAULT_URL,
                        "multiline": False,
                        "tooltip": "Ollama server URL, for example http://127.0.0.1:11434",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "llama3.2:latest",
                        "multiline": False,
                        "tooltip": "Exact installed model name, including tag; run `ollama list` to check",
                    },
                ),
                "keep_alive": (
                    ["5m", "10m", "30m", "1h", "-1", "0"],
                    {
                        "default": "5m",
                        "tooltip": "Model residency: duration, -1 keeps loaded, 0 unloads after use",
                    },
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 40, "min": 1, "max": 200, "step": 1},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Random seed; -1 lets Ollama choose",
                    },
                ),
                "num_ctx": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 512,
                        "max": 131072,
                        "step": 512,
                        "tooltip": "Requested context window; larger values use more memory",
                    },
                ),
                "custom_system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Global custom system instructions appended to the built-in system "
                            "prompt of every connected generative Ollama node."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        relevant = {
            "ollama_url": kwargs.get("ollama_url", _DEFAULT_URL),
            "model": kwargs.get("model", ""),
            "keep_alive": kwargs.get("keep_alive", "5m"),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "seed": kwargs.get("seed", -1),
            "num_ctx": kwargs.get("num_ctx", 8192),
            "custom_system_prompt_sha256": custom_system_prompt_hash(
                str(kwargs.get("custom_system_prompt", "") or "")
            ),
        }
        return tuple(sorted(relevant.items()))

    def load_model(
        self,
        ollama_url: str,
        model: str,
        keep_alive: str = "5m",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: int = -1,
        num_ctx: int = 8192,
        custom_system_prompt: str = "",
    ):
        base_url = normalize_base_url(ollama_url)
        custom_system_prompt = normalize_custom_system_prompt(custom_system_prompt)
        model = str(model or "").strip()
        if not model:
            raise ValueError("Ollama model name cannot be empty. Run `ollama list` and enter a model.")
        if not check_ollama_alive(base_url):
            raise ConnectionError(
                f"Cannot reach Ollama at {base_url}. Start `ollama serve` or check the server URL."
            )

        try:
            installed = get_model_names(base_url)
            if model not in installed:
                available = ", ".join(installed[:8]) or "none"
                raise ValueError(
                    f"Model {model!r} is not installed on {base_url}. Available models: {available}"
                )

            info = get_model_info(base_url, model)
            running = get_running_models(base_url)
        except OllamaError as exc:
            raise RuntimeError(str(exc)) from exc

        is_loaded = any(item.get("name") == model or item.get("model") == model for item in running)
        details = info.get("details", {}) if isinstance(info.get("details"), dict) else {}
        metadata = {
            "family": details.get("family", "unknown"),
            "param_size": details.get("parameter_size", "unknown"),
            "quant": details.get("quantization_level", "unknown"),
            "is_loaded": is_loaded,
        }
        capabilities = build_capability_report(model, info, is_loaded)

        profile: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "keep_alive": keep_alive,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
            "num_ctx": num_ctx,
            "_custom_system_prompt": custom_system_prompt,
            "_info": metadata,
            "_capabilities": capabilities,
        }
        _log.info(
            "Ollama model ready: %s at %s (family=%s, ctx=%d, loaded=%s)",
            model,
            base_url,
            metadata["family"],
            num_ctx,
            is_loaded,
        )
        return (profile, model)
