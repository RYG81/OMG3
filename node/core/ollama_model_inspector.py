"""Inspect live Ollama model capabilities and metadata."""

from __future__ import annotations

import json

from ...utils.capabilities import build_capability_report
from .ollama_client import OllamaError, get_model_info, get_running_models


class OllamaModelInspector:
    """Read capability, context, quantization, family, and loaded-state metadata."""

    CATEGORY = "ComfyUI-OMG/Core"
    FUNCTION = "inspect_model"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "capability_json",
        "summary",
        "supports_vision",
        "supports_tools",
        "supports_thinking",
        "supports_embeddings",
        "context_length",
        "is_loaded",
    )
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Inspect live Ollama metadata and expose conservative capability flags for routing and validation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ollama_model": ("OLLAMA_MODEL",)},
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "include_raw_model_info": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include Ollama's full model_info object in the JSON output",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Loaded state and server metadata are external state and should refresh.
        return float("nan")

    def inspect_model(self, ollama_model: dict, include_raw_model_info: bool = False, think_mode: str = "off", filter_thinking: bool = True):
        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        try:
            info = get_model_info(base_url, model)
            running = get_running_models(base_url)
        except OllamaError as exc:
            raise RuntimeError(f"Could not inspect Ollama model: {exc}") from exc

        is_loaded = any(item.get("name") == model or item.get("model") == model for item in running)
        report = build_capability_report(model, info, is_loaded)
        if include_raw_model_info:
            report["raw_model_info"] = info.get("model_info", {})

        capability_labels = [
            label
            for enabled, label in (
                (report["supports_vision"], "vision"),
                (report["supports_tools"], "tools"),
                (report["supports_thinking"], "thinking"),
                (report["supports_embeddings"], "embeddings"),
            )
            if enabled
        ]
        capabilities = ", ".join(capability_labels) or "text/completion"
        context = report["context_length"] or ollama_model.get("num_ctx", 0)
        summary = (
            f"{model} | {report['family']} | {report['parameter_size']} | "
            f"{report['quantization_level']} | {capabilities} | "
            f"context={context or 'unknown'} | loaded={'yes' if is_loaded else 'no'}"
        )
        return (
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            summary,
            report["supports_vision"],
            report["supports_tools"],
            report["supports_thinking"],
            report["supports_embeddings"],
            int(report["context_length"]),
            is_loaded,
        )
