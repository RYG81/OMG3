"""
OllamaColorPalette — Color Palette Extractor
Extracts detailed color palettes from images.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.color_palette import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.vision_prompt_tools import COLOR_PALETTE_FIELDS, COLOR_PALETTE_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

PALETTE_TYPES = ["dominant", "harmonious", "contrast", "mood"]


class OllamaColorPalette:
    """Extracts color palettes from images with hex codes and descriptions."""
    DESCRIPTION = 'Extracts color palettes from images with hex codes and descriptions.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "extract_palette"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "palette_hex", "palette_names", "palette_description",
        "primary_color", "accent_colors", "mood_description", "color_prompt_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Hex codes comma-separated",
        "Descriptive color names",
        "Natural description of palette",
        "Most dominant color",
        "Accent/highlight colors",
        "Mood the palette evokes",
        "Color tags for prompts",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to extract palette from"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 3, "max": 12, "step": 1,
                    "tooltip": "Number of colors to extract",
                }),
                "palette_type": (PALETTE_TYPES, {
                    "default": "dominant",
                    "tooltip": "Type of palette to extract",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def extract_palette(self, image, ollama_model: dict,
                       num_colors: int = 5, palette_type: str = "dominant",
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
        require_declared_capability(cfg, "vision")
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(num_colors, palette_type)

        _log.info("[OllamaNodes] Extracting color palette (%d colors, type=%s)", num_colors, palette_type)

        execution = run_structured_task(
            task_id="color_palette",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=COLOR_PALETTE_SCHEMA,
            num_predict=1500,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Color Palette schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 6
        return tuple(str(parsed.get(field, "")) for field in COLOR_PALETTE_FIELDS)
