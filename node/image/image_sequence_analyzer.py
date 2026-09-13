"""
OllamaImageSequenceAnalyzer - Analyze up to 4 sequential images for video prompts.
"""
from __future__ import annotations

import json
import logging

from ...ollama_client import generate
from ...prompts.image_analysis import build_system_prompt, build_user_prompt
from ...tasks.engine import run_structured_task
from ...tasks.image_analysis import IMAGE_ANALYSIS_SCHEMA
from ...tasks.image_consistency import build_consistency_report, compact_consistency_guidance
from ...tasks.image_evidence import build_evidence_report
from ...tasks.image_sequence import VIDEO_SEQUENCE_SCHEMA
from ...utils.capabilities import require_declared_capability
from ...utils.text_utils import filter_thinking
from ...utils.image_utils import image_tensor_hash, tensor_to_base64
from ...utils.progress import ProgressReporter

_log = logging.getLogger(__name__)


VIDEO_SYSTEM_PROMPT = """You are an expert image-to-video prompt director.

You will receive structured analyses for 1 to 4 images. Treat them as ordered keyframes in a single continuous video sequence: image 1 leads to image 2, then image 3, then image 4.

Create one model-ready positive video prompt that preserves subject identity, wardrobe, setting, lighting direction, style, and camera continuity while describing the temporal progression between the keyframes.

Respond in this EXACT JSON format:
{
    "video_prompt": "One direct positive prompt only. No labels, markdown, negative prompt, explanation, or notes.",
    "video_notes": "Short optional explanation of keyframe progression, camera motion, continuity locks, and timing decisions."
}

Rules:
- The video_prompt value must be usable directly in a video generation prompt box.
- Mention the keyframe progression in order inside the video_prompt.
- Prefer one continuous shot unless the analyses clearly require a transition.
- Include camera movement, subject movement, environment movement, lighting continuity, and temporal pacing.
- Keep all explanation out of video_prompt and place it only in video_notes.
- Output ONLY valid JSON.
"""


def _compact_analysis(parsed) -> str:
    if isinstance(parsed, dict):
        parts = []
        for key in (
            "subject", "body_details", "clothing", "pose", "location", "lighting", "camera",
            "color_palette", "art_style", "mood", "technical",
            "reconstruction_prompt",
        ):
            value = parsed.get(key, "")
            if value:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)
    return str(parsed or "")


class OllamaImageSequenceAnalyzer:
    """Analyzes up to four images one by one and creates a sequential video prompt."""
    DESCRIPTION = 'Analyzes up to four images one by one and creates a sequential video prompt.'

    CATEGORY = "ComfyUI-OMG/Image"
    FUNCTION = "analyze_sequence"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "full_sequence_analysis",
        "image_1_analysis",
        "image_2_analysis",
        "image_3_analysis",
        "image_4_analysis",
        "video_prompt",
        "video_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Combined JSON analysis for every provided image",
        "Full JSON analysis for image 1",
        "Full JSON analysis for image 2",
        "Full JSON analysis for image 3",
        "Full JSON analysis for image 4",
        "Clean positive video prompt using the images as ordered keyframes",
        "Explanation, shot-plan, and continuity notes for the video prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["basic", "detailed", "exhaustive"], {
                    "default": "detailed",
                    "tooltip": "How detailed each image analysis should be",
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "custom_categories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional categories to analyze for every image",
                }),
                "video_direction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional guidance for motion, camera, duration, model, or style",
                }),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image_1, ollama_model: dict, detail_level: str = "detailed",
                   image_2=None, image_3=None, image_4=None,
                   custom_categories: str = "", video_direction: str = "",
                   cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
        **_,
    ):
        cfg = ollama_model or {}
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        image_hashes = [
            image_tensor_hash(img) if img is not None else ""
            for img in (image_1, image_2, image_3, image_4)
        ]
        cache_key = {
            "images": image_hashes,
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
            "keep_alive": cfg.get("keep_alive", "5m"),
            "detail_level": detail_level,
            "custom_categories": custom_categories,
            "video_direction": video_direction,
            "cache_policy": cache_policy,
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"))

    def analyze_sequence(
        self,
        image_1,
        ollama_model: dict,
        detail_level: str = "detailed",
        image_2=None,
        image_3=None,
        image_4=None,
        custom_categories: str = "",
        video_direction: str = "",
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        cfg = ollama_model
        require_declared_capability(cfg, "vision")
        images = [img for img in (image_1, image_2, image_3, image_4) if img is not None]
        if not images:
            empty_analysis = json.dumps({"images": [], "error": "No images provided."}, indent=2)
            return (empty_analysis, "", "", "", "", "", "")

        system = build_system_prompt(detail_level, custom_categories)
        user_prompt = build_user_prompt(detail_level)
        analyses = []
        progress = ProgressReporter(len(images) + 1)

        for index, image in enumerate(images, start=1):
            progress.check_interrupted()
            _log.info(
                "[OllamaNodes] Analyzing sequence image %d/%d with %s",
                index, len(images), cfg["model"],
            )
            image_b64 = tensor_to_base64(image)
            execution = run_structured_task(
                task_id="image_sequence_frame",
                template_version="1",
                model_profile=cfg,
                prompt=f"Image {index} of {len(images)}.\n{user_prompt}",
                system=system,
                schema=IMAGE_ANALYSIS_SCHEMA,
                num_predict=4096,
                temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
                repair_once=True,
                cache_policy=cache_policy,
                images=[image_b64],
                generate_fn=generate,
            )
            parsed_value = execution.structured.value
            if execution.structured.valid and isinstance(parsed_value, dict):
                parsed = dict(parsed_value)
                parsed["_quality"] = build_evidence_report(parsed)
            else:
                parsed = {"raw_output": execution.raw_output}

            analyses.append({
                "frame": index,
                "analysis": parsed,
            })
            progress.update()

        consistency = build_consistency_report(
            [(f"frame_{item['frame']}", item["analysis"]) for item in analyses]
        )
        sequence_payload = {
            "frame_count": len(analyses),
            "frames": analyses,
            "video_direction": video_direction.strip(),
            "_consistency": consistency,
        }
        full_analysis = json.dumps(sequence_payload, indent=2)
        frame_outputs = [
            json.dumps(item["analysis"], indent=2) for item in analyses
        ]
        while len(frame_outputs) < 4:
            frame_outputs.append("")

        compact_frames = "\n\n".join(
            f"Keyframe {item['frame']}:\n{_compact_analysis(item['analysis'])}"
            for item in analyses
        )
        direction = video_direction.strip() or "Create a coherent cinematic image-to-video prompt."
        consistency_guidance = compact_consistency_guidance(consistency)
        synthesis_parts = [
            f"User video direction:\n{direction}",
            f"Ordered keyframe analyses:\n{compact_frames}",
        ]
        if consistency_guidance:
            synthesis_parts.append(consistency_guidance)
        synthesis_parts.append("Return JSON with video_prompt and video_notes.")
        synthesis_prompt = "\n\n".join(synthesis_parts)
        video_execution = run_structured_task(
            task_id="image_sequence_video_prompt",
            template_version="1",
            model_profile=cfg,
            prompt=synthesis_prompt,
            system=VIDEO_SYSTEM_PROMPT,
            schema=VIDEO_SEQUENCE_SCHEMA,
            num_predict=2048,
            temperature=float(cfg.get("temperature", 0.6)),
            repair_once=True,
            cache_policy=cache_policy,
            generate_fn=generate,
        )
        parsed_video = video_execution.structured.value
        if video_execution.structured.valid and isinstance(parsed_video, dict):
            video_prompt = str(parsed_video.get("video_prompt", "") or "").strip()
            video_notes = str(parsed_video.get("video_notes", "") or "").strip()
        else:
            video_prompt = video_execution.raw_output.strip()
            video_notes = ""
        progress.update()
        return (full_analysis, *frame_outputs[:4], video_prompt, video_notes)
