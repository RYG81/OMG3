"""
OllamaImageMerger — Two Pictures to One
Analyzes two images + user instruction and generates a merge prompt.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.image_merge import BLEND_STYLES, build_system_prompt, build_user_prompt
from ...tasks.creative_prompt_tools import IMAGE_MERGER_FIELDS, IMAGE_MERGER_SCHEMA
from ...tasks.engine import run_structured_task
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)


class OllamaImageMerger:
    """Takes two images and a text instruction, generates a prompt to create a merged composite."""
    DESCRIPTION = 'Takes two images and a text instruction, generates a prompt to create a merged composite.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "merge"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "merged_prompt", "image_1_analysis", "image_2_analysis",
        "negative_prompt", "composition_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Ready-to-use prompt for the merged/composite image",
        "Analysis of Image 1 (source)",
        "Analysis of Image 2 (target/base)",
        "Suggested negative prompt",
        "Composition and spatial arrangement notes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {"tooltip": "Source image — take elements FROM this"}),
                "image_2": ("IMAGE", {"tooltip": "Target/base image — merge elements INTO this"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What to take from Image 1 and how to place it in Image 2",
                }),
                "blend_style": (list(BLEND_STYLES.keys()), {
                    "default": "seamless",
                    "tooltip": "How the images should be blended together",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def merge(self, image_1, image_2, ollama_model: dict, instruction: str,
              blend_style: str = "seamless", cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
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
        img1_b64 = tensor_to_base64(image_1)
        img2_b64 = tensor_to_base64(image_2)

        system = build_system_prompt(blend_style)
        user_prompt = build_user_prompt(instruction)

        _log.info("[OllamaNodes] Merging images with %s (style=%s)", cfg["model"], blend_style)

        execution = run_structured_task(
            task_id="image_merger",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=IMAGE_MERGER_SCHEMA,
            num_predict=4096,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img1_b64, img2_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Image Merger schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in IMAGE_MERGER_FIELDS)
