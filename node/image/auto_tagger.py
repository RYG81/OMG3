"""
OllamaAutoTagger — Auto Tagger
Generates comprehensive tags from images.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.auto_tagger import build_system_prompt, USER_PROMPT
from ...tasks.auto_tagger import AUTO_TAG_FIELDS, AUTO_TAG_SCHEMA
from ...tasks.engine import run_structured_task
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

FORMAT_OPTIONS = ["booru", "natural", "weighted", "all"]


class OllamaAutoTagger:
    """Generates comprehensive tags from images in various formats."""
    DESCRIPTION = 'Generates comprehensive tags from images in various formats.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "tag"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "booru_tags", "natural_tags", "weighted_tags",
        "character_tags", "style_tags", "quality_tags", "meta_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Danbooru-style tags",
        "Natural language tags",
        "Tags with weights",
        "Character-specific tags",
        "Style/aesthetic tags",
        "Quality/technical tags",
        "Meta information tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to tag"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "tag_format": (FORMAT_OPTIONS, {
                    "default": "all",
                    "tooltip": "Tag format to generate",
                }),
                "max_tags": ("INT", {
                    "default": 50,
                    "min": 10, "max": 100, "step": 5,
                    "tooltip": "Maximum number of tags",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "include_nsfw_tags": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include content rating tags",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def tag(self, image, ollama_model: dict, tag_format: str = "all",
           max_tags: int = 50, include_nsfw_tags: bool = False,
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
        system = build_system_prompt(tag_format, max_tags, include_nsfw_tags)

        _log.info("[OllamaNodes] Auto-tagging (format=%s, max=%d)", tag_format, max_tags)

        execution = run_structured_task(
            task_id="auto_tagger",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=AUTO_TAG_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.4)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Auto Tagger schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 6
        return tuple(str(parsed.get(field, "")) for field in AUTO_TAG_FIELDS)
