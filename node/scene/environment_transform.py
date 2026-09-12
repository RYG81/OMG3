"""
OllamaEnvironmentTransform — Environment Transformer
Transforms scene weather, time, and season.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.environment_transform import build_system_prompt, USER_PROMPT
from ...tasks.engine import run_structured_task
from ...tasks.vision_prompt_tools import ENVIRONMENT_FIELDS, ENVIRONMENT_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)

WEATHER_OPTIONS = ["sunny", "cloudy", "rainy", "stormy", "snowy", "foggy", "windy"]
TIME_OPTIONS = ["dawn", "morning", "noon", "afternoon", "sunset", "dusk", "night", "midnight"]
SEASON_OPTIONS = ["spring", "summer", "autumn", "winter", "keep_original"]


class OllamaEnvironmentTransform:
    """Transforms scene environment — weather, time, and season."""
    DESCRIPTION = 'Transforms scene environment — weather, time, and season.'

    CATEGORY = "ComfyUI-OMG/Scene"
    FUNCTION = "transform"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "transformed_prompt", "original_analysis", "lighting_description",
        "atmosphere_description", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt for the transformed scene",
        "Analysis of original environment",
        "New lighting setup",
        "New atmosphere/mood",
        "What to avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original scene image"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "target_weather": (WEATHER_OPTIONS, {
                    "default": "sunny",
                    "tooltip": "Target weather condition",
                }),
                "target_time": (TIME_OPTIONS, {
                    "default": "noon",
                    "tooltip": "Target time of day",
                }),
                "target_season": (SEASON_OPTIONS, {
                    "default": "keep_original",
                    "tooltip": "Target season",
                }),
                "transformation_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.3, "max": 1.0, "step": 0.1,
                    "tooltip": "How dramatically to transform",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def transform(self, image, ollama_model: dict, target_weather: str = "sunny",
                 target_time: str = "noon", target_season: str = "keep_original",
                 transformation_strength: float = 0.7, cache_policy: str = "use", think_mode: str = "off", filter_thinking: bool = True):
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
        system = build_system_prompt(target_weather, target_time, target_season, transformation_strength)

        _log.info("[OllamaNodes] Environment transform (weather=%s, time=%s, season=%s)",
                  target_weather, target_time, target_season)

        execution = run_structured_task(
            task_id="environment_transform",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=ENVIRONMENT_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Environment Transform schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in ENVIRONMENT_FIELDS)
