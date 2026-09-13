"""
OllamaImageEdit converts analyzer output and user changes into an edit-model prompt.
"""
from __future__ import annotations

import json
import logging

from ...ollama_client import generate
from ...tasks.creative_prompt_tools import IMAGE_EDIT_FIELDS, IMAGE_EDIT_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.image_edit import (
    IMAGE_EDIT_MODELS,
    build_system_prompt,
    build_user_prompt,
)

_log = logging.getLogger(__name__)


def _as_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


class OllamaImageEdit:
    """Builds model-specific image-edit prompts from structured image analysis."""
    DESCRIPTION = 'Builds model-specific image-edit prompts from structured image analysis.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "build_edit_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "edit_prompt",
        "preservation_notes",
        "change_summary",
        "full_result",
    )
    OUTPUT_TOOLTIPS = (
        "Model-ready image editing prompt",
        "Source-image details that should remain unchanged",
        "Summary of requested changes",
        "Complete structured result as JSON",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analyzed_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Connect full_analysis from Ollama Image Analyzer",
                }),
                "ollama_model": ("OLLAMA_MODEL",),
                "requested_changes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe only what should change in the source image",
                }),
                "image_edit_model": (IMAGE_EDIT_MODELS, {
                    "default": "FLUX 2",
                    "tooltip": "Target image-edit model used to structure the final prompt",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        analyzed_data: str,
        ollama_model: dict,
        requested_changes: str,
        image_edit_model: str = "FLUX 2",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        cfg = ollama_model or {}
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        cache_key = {
            "analyzed_data": analyzed_data,
            "requested_changes": requested_changes,
            "image_edit_model": image_edit_model,
            "cache_policy": cache_policy,
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"))

    def build_edit_prompt(
        self,
        analyzed_data: str,
        ollama_model: dict,
        requested_changes: str,
        image_edit_model: str = "FLUX 2",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        analysis = analyzed_data.strip()
        changes = requested_changes.strip()
        if not analysis:
            raise ValueError("analyzed_data is empty. Connect full_analysis from the analyzer.")
        if not changes:
            raise ValueError("requested_changes is empty. Describe what should change.")

        cfg = ollama_model
        _log.info(
            "[OllamaNodes] Building image edit prompt with %s for %s",
            cfg["model"],
            image_edit_model,
        )

        execution = run_structured_task(
            task_id="image_edit_prompt",
            template_version="1",
            model_profile=cfg,
            prompt=build_user_prompt(analysis, changes, image_edit_model),
            system=build_system_prompt(image_edit_model),
            schema=IMAGE_EDIT_SCHEMA,
            num_predict=2048,
            temperature=min(float(cfg.get("temperature", 0.3)), 0.5),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": f"{analysis}\n{changes}",
                "primary_field": "edit_prompt",
                "target_model": image_edit_model,
            },
            generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Image Edit schema validation failed: %s", execution.report)
            raw = execution.raw_output
            return (raw.strip(), "", changes, raw)

        result = {
            "edit_prompt": _as_string(parsed.get(IMAGE_EDIT_FIELDS[0])),
            "preservation_notes": _as_string(parsed.get(IMAGE_EDIT_FIELDS[1])),
            "change_summary": _as_string(parsed.get(IMAGE_EDIT_FIELDS[2])) or changes,
        }
        return (
            result["edit_prompt"],
            result["preservation_notes"],
            result["change_summary"],
            json.dumps(result, indent=2, ensure_ascii=False),
        )
