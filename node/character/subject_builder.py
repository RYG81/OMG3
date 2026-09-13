"""
OllamaSubjectBuilder — Build detailed character/subject descriptions.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...tasks.design_character_tools import SUBJECT_FIELDS, SUBJECT_SCHEMA
from ...tasks.engine import run_structured_task
from ...prompts.subject_builder import (
    build_system_prompt, build_user_prompt,
    SPECIES, GENDER_PRESENTATION, BUILD, SKIN, DISTINGUISHING_FEATURES, PERSONALITY_VIBE,
)
from ...prompts.prompt_builder import (
    AGE_APPEARANCE, HAIR_STYLE, HAIR_COLOR, FACIAL_FEATURES,
)

_log = logging.getLogger(__name__)


class OllamaSubjectBuilder:
    """Build detailed, consistent character descriptions."""
    DESCRIPTION = 'Build detailed, consistent character descriptions.'

    CATEGORY = "ComfyUI-OMG/Character"
    FUNCTION = "build_subject"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "full_description", "appearance_tags", "physical_summary",
        "face_description", "body_description", "clothing_suggestion",
        "consistency_anchors", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Complete character description",
        "Appearance tags (booru-style)",
        "Physical summary",
        "Face and expression",
        "Body type and posture",
        "Suggested default clothing",
        "Key elements for consistency",
        "What to avoid",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "character_name": ("STRING", {"default": "", "tooltip": "Optional character name"}),
                
                "species": (SPECIES, {"default": "human", "tooltip": "Species/type"}),
                "species_custom": ("STRING", {"default": "", "tooltip": "Custom species"}),
                
                "gender": (GENDER_PRESENTATION, {"default": "custom", "tooltip": "Gender presentation"}),
                "gender_custom": ("STRING", {"default": "", "tooltip": "Custom gender"}),
                
                "age": (AGE_APPEARANCE, {"default": "young adult (20s)", "tooltip": "Age appearance"}),
                "age_custom": ("STRING", {"default": "", "tooltip": "Custom age"}),
                
                "build": (BUILD, {"default": "custom", "tooltip": "Body build"}),
                "build_custom": ("STRING", {"default": "", "tooltip": "Custom build"}),
                
                "skin": (SKIN, {"default": "custom", "tooltip": "Skin/surface"}),
                "skin_custom": ("STRING", {"default": "", "tooltip": "Custom skin"}),
                
                "hair_style": (HAIR_STYLE, {"default": "custom", "tooltip": "Hair style"}),
                "hair_style_custom": ("STRING", {"default": "", "tooltip": "Custom hair style"}),
                
                "hair_color": (HAIR_COLOR, {"default": "custom", "tooltip": "Hair color"}),
                "hair_color_custom": ("STRING", {"default": "", "tooltip": "Custom hair color"}),
                
                "eyes": ("STRING", {"default": "", "tooltip": "Eye description"}),
                
                "facial": (FACIAL_FEATURES, {"default": "custom", "tooltip": "Facial features"}),
                "facial_custom": ("STRING", {"default": "", "tooltip": "Custom facial features"}),
                
                "distinguishing": (DISTINGUISHING_FEATURES, {"default": "custom", "tooltip": "Distinguishing features"}),
                "distinguishing_custom": ("STRING", {"default": "", "tooltip": "Custom features"}),
                
                "personality": (PERSONALITY_VIBE, {"default": "custom", "tooltip": "Personality vibe"}),
                "personality_custom": ("STRING", {"default": "", "tooltip": "Custom personality"}),
                
                "additional_details": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional character details",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def build_subject(self, ollama_model: dict, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        cfg = ollama_model
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        
        def get_val(dropdown_key, custom_key):
            dropdown_val = kwargs.get(dropdown_key, "custom")
            custom_val = kwargs.get(custom_key, "")
            if dropdown_val == "custom" and custom_val.strip():
                return custom_val.strip()
            elif dropdown_val != "custom":
                return dropdown_val
            return "not specified"
        
        system = build_system_prompt(
            species=get_val("species", "species_custom"),
            gender=get_val("gender", "gender_custom"),
            age=get_val("age", "age_custom"),
            build=get_val("build", "build_custom"),
            skin=get_val("skin", "skin_custom"),
            hair_style=get_val("hair_style", "hair_style_custom"),
            hair_color=get_val("hair_color", "hair_color_custom"),
            eyes=kwargs.get("eyes", "not specified"),
            facial=get_val("facial", "facial_custom"),
            distinguishing=get_val("distinguishing", "distinguishing_custom"),
            personality=get_val("personality", "personality_custom"),
            custom_details=kwargs.get("additional_details", ""),
        )
        
        user_prompt = build_user_prompt(kwargs.get("character_name", ""))

        _log.info("[OllamaNodes] Building subject description")

        execution = run_structured_task(
            task_id="subject_builder",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=SUBJECT_SCHEMA,
            num_predict=2500,
            temperature=float(cfg.get("temperature", 0.7)),
            repair_once=True,
            cache_policy=kwargs.get("cache_policy", "use"),
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            return (execution.raw_output,) + ("",) * 7
        return tuple(str(parsed.get(field, "")) for field in SUBJECT_FIELDS)
