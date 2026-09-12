"""
OllamaRegionalPrompts — Regional Prompt Generator
Divides image into regions and generates prompts for each.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.regional_prompts import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.regional_prompts import build_regional_schema, normalize_regional_outputs
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

GRID_OPTIONS = ["2x2", "3x3", "2x3", "3x2", "1x3", "3x1"]


class OllamaRegionalPrompts:
    """Generates region-specific prompts for compositional control."""
    DESCRIPTION = 'Generates region-specific prompts for compositional control.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "generate_regions"
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "top_left", "top_center", "top_right",
        "middle_left", "center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
        "full_composition", "regional_weights", "overlap_notes",
    )
    OUTPUT_TOOLTIPS = tuple([
        "Top-left region", "Top-center region", "Top-right region",
        "Middle-left region", "Center region", "Middle-right region",
        "Bottom-left region", "Bottom-center region", "Bottom-right region",
        "Full composition description", "Suggested attention weights", "Elements spanning regions",
    ])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Reference image to analyze"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "grid_size": (GRID_OPTIONS, {
                    "default": "3x3",
                    "tooltip": "How to divide the image",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "focus_areas": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Specific areas to focus on",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_regions(self, image, ollama_model: dict, grid_size: str = "3x3",
                        focus_areas: str = "", cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
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
        system = build_system_prompt(grid_size, focus_areas)

        _log.info("[OllamaNodes] Generating regional prompts (grid=%s)", grid_size)

        execution = run_structured_task(
            task_id="regional_prompts",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=build_regional_schema(grid_size),
            num_predict=3000,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Regional Prompt schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 11
        regions = normalize_regional_outputs(parsed, grid_size)
        return tuple(regions) + (
            str(parsed.get("full_composition", "")),
            str(parsed.get("regional_weights", "")),
            str(parsed.get("overlap_notes", "")),
        )
