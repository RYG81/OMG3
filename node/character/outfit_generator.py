"""
OllamaOutfitGenerator — Outfit Generator
Generates detailed outfit/costume descriptions.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.outfit_generator import build_system_prompt, build_user_prompt
from ...tasks.creative_prompt_tools import OUTFIT_FIELDS, OUTFIT_SCHEMA
from ...tasks.engine import run_structured_task
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

OUTFIT_TYPES = ["casual", "formal", "fantasy", "sci-fi", "historical", "uniform", "costume", "streetwear", "haute_couture"]
SEASONS = ["spring", "summer", "fall", "winter", "any"]


class OllamaOutfitGenerator:
    """Generates detailed outfit descriptions for characters."""
    DESCRIPTION = 'Generates detailed outfit descriptions for characters.'

    CATEGORY = "ComfyUI-OMG/Character"
    FUNCTION = "generate_outfit"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "outfit_prompt", "top_description", "bottom_description",
        "accessories", "footwear", "outfit_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Complete outfit description prompt",
        "Upper body clothing",
        "Lower body clothing",
        "Accessories and jewelry",
        "Shoes/boots description",
        "Comma-separated clothing tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Who is wearing the outfit",
                }),
                "outfit_type": (OUTFIT_TYPES, {
                    "default": "casual",
                    "tooltip": "Type of outfit to generate",
                }),
                "season": (SEASONS, {
                    "default": "any",
                    "tooltip": "Season/weather the outfit is for",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "color_scheme": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Preferred colors (optional)",
                }),
                "reference_image": ("IMAGE", {"tooltip": "Optional outfit reference"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_outfit(self, ollama_model: dict, character_description: str,
                       outfit_type: str = "casual", season: str = "any",
                       color_scheme: str = "", reference_image=None,
                       cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        has_reference = reference_image is not None
        system = build_system_prompt(character_description, outfit_type, season, color_scheme, has_reference)
        user_prompt = build_user_prompt(has_reference)

        _log.info("[OllamaNodes] Generating outfit (type=%s, season=%s)", outfit_type, season)

        images = []
        if has_reference:
            require_declared_capability(cfg, "vision")
            images.append(tensor_to_base64(reference_image))

        execution = run_structured_task(
            task_id="outfit_generator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=OUTFIT_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=cache_policy,
            images=images or None,
            quality_context={
                "source_text": character_description,
                "primary_field": "outfit_prompt",
            },
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Outfit Generator schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in OUTFIT_FIELDS)
