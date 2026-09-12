"""
Video shot director node.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.engine import run_structured_task
from ...tasks.video_shot_director import VIDEO_SHOT_FIELDS, VIDEO_SHOT_SCHEMA
from ...prompts.video_shot_director import (
    SEQUENCE_STYLE,
    SHOT_COUNTS,
    VIDEO_MODELS,
    build_system_prompt,
    build_user_prompt,
)

_log = logging.getLogger(__name__)


class OllamaVideoShotDirector:
    """Expands a scene/video concept into a shot-by-shot prompt sequence."""
    DESCRIPTION = 'Expands a scene/video concept into a shot-by-shot prompt sequence.'

    CATEGORY = "ComfyUI-OMG/Video"
    FUNCTION = "direct_video_shots"
    RETURN_TYPES = ("STRING",) * 6
    RETURN_NAMES = (
        "sequence_prompt", "shot_list", "per_shot_prompts",
        "continuity_notes", "negative_prompt", "model_notes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scene_prompt": ("STRING", {"default": "", "multiline": True}),
                "target_model": (VIDEO_MODELS, {"default": "Wan 2.2"}),
                "shot_count": (SHOT_COUNTS, {"default": "5 shots"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "sequence_style": (SEQUENCE_STYLE, {"default": "single scene coverage"}),
                "total_duration": ("STRING", {"default": "10 seconds"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def direct_video_shots(
        self, ollama_model: dict, scene_prompt: str,
        target_model: str = "Wan 2.2", shot_count: str = "5 shots",
        sequence_style: str = "single scene coverage",
        total_duration: str = "10 seconds", custom_system_prompt: str = "",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        system = build_system_prompt(
            target_model, shot_count, sequence_style, total_duration, custom_system_prompt,
        )
        user_prompt = build_user_prompt(scene_prompt)

        _log.info("[OllamaNodes] Directing video shot sequence")

        execution = run_structured_task(
            task_id="video_shot_director",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=VIDEO_SHOT_SCHEMA,
            num_predict=3500,
            temperature=float(cfg.get("temperature", 0.6)),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": scene_prompt,
                "primary_field": "sequence_prompt",
                "target_model": target_model,
            },
            generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 5
        return tuple(str(parsed.get(field, "")) for field in VIDEO_SHOT_FIELDS)
