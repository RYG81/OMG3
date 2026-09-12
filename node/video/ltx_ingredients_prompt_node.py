"""Ollama prompt generator for LTX-2.3 22B IC-LoRA Ingredients."""
from __future__ import annotations

import logging

from ...ollama_client import generate
from ...tasks.engine import run_structured_task
from ...tasks.video_generators import LTX_INGREDIENT_SCHEMA

_log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert text-to-video prompt engineer for LTX-2.3 22B IC-LoRA Ingredients workflows.

The user will provide visual ingredients for a reference sheet and a generated-video intent. Your job is to write the exact two-part prompt format expected by the Ingredients IC-LoRA:

Reference sheet: ...

Generated video: ...

The reference sheet section must describe a clean static sheet of reusable visual ingredients. It should include characters, props, and location details as separate readable panels on a black background with no labels or text. It must make identities, wardrobe, important props, and environment details easy for the model to preserve.

The generated video section must describe one continuous shot using those reference-sheet ingredients. It must be specific about subject action, camera behavior, shot style, lighting, temporal continuity, and which referenced elements must appear.

Rules:
- Preserve character identity, wardrobe, prop appearance, and location layout.
- Avoid asking for scene cuts unless the user explicitly requests them.
- Avoid vague phrases like "make it cinematic" unless paired with concrete camera, lighting, and motion details.
- Keep the prompt direct and model-ready.
- Do not include markdown, explanations, or extra keys.
- Output ONLY valid JSON in this exact schema:
{
  "prompt": "Reference sheet: ...\n\nGenerated video: ...",
  "negative_prompt": "...",
  "recommended_settings": "..."
}
"""

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "identity drift, costume drift, missing referenced prop, wrong location"
)

DEFAULT_SETTINGS = (
    "Recommended for LTX-2.3-22B IC-LoRA Ingredients: LoRA strength 1.4, "
    "30 steps, guidance 4.0, 768x448, 121 frames, 24 fps. Use the reference "
    "sheet as a static video/control input with at least 121 frames."
)


def _clean(value):
    return " ".join(str(value or "").split())


def _sentence(label, value):
    value = _clean(value)
    return f"{label}: {value}" if value else ""


def _fallback_prompt(characters, props, location, generated_video, shot_style, camera_motion, lighting):
    reference_parts = [
        _sentence("Characters", characters),
        _sentence("Props", props),
        _sentence("Location", location),
        (
            "Reference sheet layout: black background, no text, one clean "
            "panel per distinct visual element; make important elements large "
            "and uncluttered."
        ),
    ]
    video_parts = [
        _clean(generated_video),
        _sentence("Shot style", shot_style),
        _sentence("Camera motion", camera_motion),
        _sentence("Lighting", lighting),
    ]
    return (
        "Reference sheet: "
        + " ".join(part for part in reference_parts if part)
        + "\n\nGenerated video: "
        + " ".join(part for part in video_parts if part)
    )


def _build_user_prompt(characters, props, location, generated_video, shot_style, camera_motion, lighting, negative_prompt):
    return f"""Create an LTX-2.3 IC-LoRA Ingredients prompt from these inputs.

Characters:
{characters}

Props:
{props}

Location:
{location}

Generated video intent:
{generated_video}

Shot style:
{shot_style}

Camera motion:
{camera_motion}

Lighting:
{lighting}

Requested negative prompt additions:
{negative_prompt}

Return valid JSON only."""


class LTXIngredientsPrompt:
    DESCRIPTION = 'L T X Ingredients Prompt'
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "characters": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "One clean panel per character: face close-up, "
                            "full-body view, outfit, key identifying features."
                        ),
                    },
                ),
                "props": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "One clean product-style panel per important prop.",
                    },
                ),
                "location": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "One clean location panel showing the environment, "
                            "layout, mood, and key background details."
                        ),
                    },
                ),
                "generated_video": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Describe the exact action and shot. Mention which "
                            "reference-sheet characters, props, and location must appear."
                        ),
                    },
                ),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "shot_style": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "cinematic medium shot, natural motion",
                    },
                ),
                "camera_motion": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "subtle handheld camera, smooth tracking",
                    },
                ),
                "lighting": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "soft realistic lighting",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_NEGATIVE_PROMPT,
                    },
                ),
                "custom_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "recommended_settings")
    FUNCTION = "generate_prompt"
    CATEGORY = "ComfyUI-OMG/Video"

    def generate_prompt(
        self,
        ollama_model,
        characters,
        props,
        location,
        generated_video,
        shot_style="",
        camera_motion="",
        lighting="",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        custom_system_prompt="",
        cache_policy="use",
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
        system = SYSTEM_PROMPT
        custom_system_prompt = str(custom_system_prompt or "").strip()
        if custom_system_prompt:
            system = f"{system}\n\nAdditional user system guidance:\n{custom_system_prompt}"

        user_prompt = _build_user_prompt(
            characters,
            props,
            location,
            generated_video,
            shot_style,
            camera_motion,
            lighting,
            negative_prompt,
        )

        _log.info(
            "[OllamaNodes] Generating LTX Ingredients prompt with %s",
            cfg["model"],
        )

        execution = run_structured_task(
            task_id="ltx_ingredients_prompt",
            template_version="1",
            model_profile=cfg,
            prompt=user_prompt,
            system=system,
            schema=LTX_INGREDIENT_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.5)),
            repair_once=True,
            cache_policy=cache_policy,
            quality_context={
                "source_text": generated_video,
                "primary_field": "prompt",
                "target_model": "LTX Video",
            },
            generate_fn=generate,
        )
        parsed = execution.structured.value
        fallback = _fallback_prompt(
            characters, props, location, generated_video, shot_style, camera_motion, lighting
        )
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("LTX Ingredients schema validation failed: %s", execution.report)
            return (fallback, _clean(negative_prompt), DEFAULT_SETTINGS)
        return (
            str(parsed.get("prompt", "") or fallback),
            _clean(parsed.get("negative_prompt", "") or negative_prompt),
            str(parsed.get("recommended_settings", "") or DEFAULT_SETTINGS),
        )


NODE_CLASS_MAPPINGS = {
    "LTXIngredientsPrompt": LTXIngredientsPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXIngredientsPrompt": "LTX Ingredients Prompt",
}
