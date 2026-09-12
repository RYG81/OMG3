"""
OllamaStoryboardGenerator — Storyboard Generator
Creates a sequence of prompts for visual storytelling.
"""
from __future__ import annotations
import json
import logging
from ...utils.text_utils import filter_thinking

from ...ollama_client import generate
from ...prompts.storyboard import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.storyboard import PANEL_FIELDS, build_storyboard_schema

_log = logging.getLogger(__name__)

STYLE_OPTIONS = ["comic", "manga", "cinematic", "animation", "storyboard"]
RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4"]


class OllamaStoryboardGenerator:
    """Creates sequences of prompts for comics/storyboards."""
    DESCRIPTION = 'Creates sequences of prompts for comics/storyboards.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "generate_storyboard"
    RETURN_TYPES = ("STRING",) * 16
    RETURN_NAMES = (
        "panel_1", "panel_2", "panel_3", "panel_4", "panel_5", "panel_6",
        "panel_7", "panel_8", "panel_9", "panel_10", "panel_11", "panel_12",
        "all_panels", "character_reference", "story_summary", "camera_directions",
    )
    OUTPUT_TOOLTIPS = tuple([f"Panel {i} prompt" for i in range(1, 13)] + [
        "All panels as JSON array",
        "Character descriptions for consistency",
        "Narrative summary",
        "Camera movement notes",
    ])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "story_concept": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The story or scene sequence to visualize",
                }),
                "num_panels": ("INT", {
                    "default": 6,
                    "min": 2, "max": 12, "step": 1,
                    "tooltip": "Number of panels to generate",
                }),
                "style": (STYLE_OPTIONS, {
                    "default": "cinematic",
                    "tooltip": "Visual style for the panels",
                }),
                "maintain_characters": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep character consistency across panels",
                }),
                "aspect_ratio": (RATIO_OPTIONS, {
                    "default": "16:9",
                    "tooltip": "Aspect ratio for panels",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def generate_storyboard(self, ollama_model: dict, story_concept: str,
                           num_panels: int = 6, style: str = "cinematic",
                           maintain_characters: bool = True, aspect_ratio: str = "16:9",
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
        system = build_system_prompt(story_concept, num_panels, style, maintain_characters, aspect_ratio)
        user_prompt = build_user_prompt()

        _log.info("[OllamaNodes] Generating storyboard (%d panels, style=%s)", num_panels, style)

        execution = run_structured_task(
            task_id="storyboard_generator",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=build_storyboard_schema(num_panels),
            num_predict=6000,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=cache_policy,
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Storyboard schema validation failed: %s", execution.report)
            return (execution.raw_output,) + ("",) * 15

        panels = [str(parsed.get(field, "")) for field in PANEL_FIELDS]
        all_panels_value = parsed.get("all_panels")
        if isinstance(all_panels_value, list):
            all_panels = json.dumps(all_panels_value, ensure_ascii=False)
        elif all_panels_value:
            all_panels = str(all_panels_value)
        else:
            all_panels = json.dumps([panel for panel in panels if panel], ensure_ascii=False)
        return tuple(panels) + (
            all_panels,
            str(parsed.get("character_reference", "")),
            str(parsed.get("story_summary", "")),
            str(parsed.get("camera_directions", "")),
        )
