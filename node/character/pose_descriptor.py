"""
OllamaPoseDescriptor — Pose Descriptor
Extracts detailed pose descriptions from images.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.pose_descriptor import build_system_prompt, USER_PROMPT
from ...tasks.creative_prompt_tools import POSE_FIELDS, POSE_SCHEMA
from ...tasks.engine import run_structured_task
from ...utils.capabilities import require_declared_capability
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

_log = logging.getLogger(__name__)


class OllamaPoseDescriptor:
    """Extracts detailed pose/body position descriptions from images."""
    DESCRIPTION = 'Extracts detailed pose/body position descriptions from images.'

    CATEGORY = "ComfyUI-OMG/Character"
    FUNCTION = "describe_pose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "pose_description", "body_position", "limb_positions",
        "hand_description", "face_expression", "pose_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Natural language pose description",
        "Core body positioning",
        "Detailed limb placement",
        "Hand poses and gestures",
        "Facial expression details",
        "Comma-separated pose tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image containing the pose"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["simple", "detailed", "anatomical"], {
                    "default": "detailed",
                    "tooltip": "How detailed the pose description should be",
                }),
                "include_hands": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extra detail for hand positions",
                }),
                "include_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extra detail for facial expression",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    def describe_pose(self, image, ollama_model: dict, detail_level: str = "detailed",
                     include_hands: bool = True, include_face: bool = True,
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
        system = build_system_prompt(detail_level, include_hands, include_face)

        _log.info("[OllamaNodes] Describing pose (detail=%s)", detail_level)

        execution = run_structured_task(
            task_id="pose_descriptor",
            template_version="1",
            model_profile=cfg,
            prompt=USER_PROMPT,
            system=system,
            schema=POSE_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.3)),
            repair_once=True,
            cache_policy=cache_policy,
            images=[img_b64],
            think=think, filter_thinking=filter_thinking, generate_fn=generate,
        )
        parsed = execution.structured.value
        if not execution.structured.valid or not isinstance(parsed, dict):
            _log.warning("Pose Descriptor schema validation failed: %s", execution.report)
            return (execution.raw_output, "", "", "", "", "")
        return tuple(str(parsed.get(field, "")) for field in POSE_FIELDS)
