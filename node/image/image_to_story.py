"""
OllamaImageToStory — Generate narrative content from images.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.creative_prompt_tools import IMAGE_STORY_FIELDS, IMAGE_STORY_SCHEMA
from ...tasks.engine import run_structured_task
from ...utils.capabilities import require_declared_capability
from ...prompts.image_to_story import (
    build_system_prompt, USER_PROMPT,
    STORY_TYPE, TONE, PERSPECTIVE, LENGTH,
)
from ...utils.image_utils import tensor_to_base64

_log = logging.getLogger(__name__)


class OllamaImageToStory:
    """Generate stories and narratives from images."""
    DESCRIPTION = 'Generate stories and narratives from images.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "generate_story"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "story", "title", "setting_description", "character_profiles",
        "mood_analysis", "dialogue_sample", "before_scene",
        "after_scene", "themes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "story_type": (STORY_TYPE, {"default": "narrative description"}),
                "tone": (TONE, {"default": "neutral"}),
                "perspective": (PERSPECTIVE, {"default": "third person omniscient"}),
                "length": (LENGTH, {"default": "medium - 2-3 paragraphs"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_story(self, image, ollama_model: dict, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
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
        
        system = build_system_prompt(
            story_type=kwargs.get("story_type", "narrative description"),
            tone=kwargs.get("tone", "neutral"),
            perspective=kwargs.get("perspective", "third person omniscient"),
            length=kwargs.get("length", "medium - 2-3 paragraphs"),
        )
        
        _log.info("[OllamaNodes] Generating story from image")
        
        execution = run_structured_task(
            task_id="image_to_story",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=IMAGE_STORY_SCHEMA,
            num_predict=3000,
            temperature=float(cfg.get("temperature", 0.8)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 8
        return tuple(str(parsed.get(field, "")) for field in IMAGE_STORY_FIELDS)
