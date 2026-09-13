"""
OllamaImageComparator — Image Comparator
Compares two images and describes differences/similarities.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.image_comparator import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.vision_prompt_tools import COMPARATOR_FIELDS, IMAGE_COMPARATOR_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

FOCUS_OPTIONS = ["all", "composition", "style", "subject", "colors", "quality"]


class OllamaImageComparator:
    """Compares two images and identifies similarities/differences."""
    DESCRIPTION = 'Compares two images and identifies similarities/differences.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "compare"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "comparison_summary", "similarities", "differences",
        "image_a_unique", "image_b_unique", "quality_comparison", "recommendation",
    )
    OUTPUT_TOOLTIPS = (
        "Overall comparison summary",
        "What's the same",
        "What's different",
        "Elements only in Image A",
        "Elements only in Image B",
        "Which is better and why",
        "Suggestions for improvement",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image"}),
                "image_b": ("IMAGE", {"tooltip": "Second image"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "comparison_focus": (FOCUS_OPTIONS, {
                    "default": "all",
                    "tooltip": "What aspect to focus the comparison on",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def compare(self, image_a, image_b, ollama_model: dict,
                comparison_focus: str = "all", cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
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
        img_a_b64 = tensor_to_base64(image_a)
        img_b_b64 = tensor_to_base64(image_b)
        system = build_system_prompt(comparison_focus)

        _log.info("[OllamaNodes] Comparing images (focus=%s)", comparison_focus)

        execution = run_structured_task(
            task_id="image_comparator",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=IMAGE_COMPARATOR_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_a_b64, img_b_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Image Comparator schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in COMPARATOR_FIELDS)
