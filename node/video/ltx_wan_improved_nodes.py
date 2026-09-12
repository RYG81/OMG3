"""
Improved LTX and Wan nodes based on PromptSama studies
Original video_prompt_generator was good but missed authority assignment, screen geography, timed end states.
This version injects PromptSama learnings: reference use, identity locks, screen geography, 0-4/4-9/9-15 timing.

Compatible with Comfy-OMG structure.
"""

from __future__ import annotations
import logging

from ...ollama_client import generate
from ...utils.text_utils import extract_json_block, filter_thinking
from ...prompts.minimax_h3_prompt import build_system_prompt_improved

_log = logging.getLogger(__name__)

VIDEO_STYLE = [
    "auto-infer", "cinematic realism", "documentary realism", "anime",
    "stylized 3D", "concept art motion", "music video", "commercial product",
    "fashion film", "dark fantasy", "sci-fi", "dreamlike surreal", "custom"
]

SHOT_TYPE = [
    "auto-infer", "establishing shot", "wide shot", "medium shot", "close-up",
    "extreme close-up", "two-shot", "tracking shot", "point-of-view",
    "over-the-shoulder", "insert shot", "macro shot", "custom"
]

CAMERA_MOTION = [
    "auto-infer", "locked-off tripod", "slow push-in", "slow pull-back",
    "left-to-right dolly", "right-to-left dolly", "orbit around subject",
    "handheld subtle sway", "crane up", "crane down", "drone glide",
    "tilt up", "tilt down", "pan left", "pan right", "rack focus",
    "push in at 00:06.500", "tracking", "custom"
]

SUBJECT_MOTION = [
    "auto-infer", "almost still", "gentle natural movement", "walking",
    "running", "turning toward camera", "turning away", "reaching", "gesturing",
    "hair/clothing moving in wind", "object rotating", "vehicle moving", "crowd motion", "explosive action", "custom"
]

MOTION_INTENSITY = ["auto-infer", "very subtle", "subtle", "moderate", "dynamic", "fast", "chaotic but readable", "custom"]
PACING = ["auto-infer", "slow contemplative", "steady cinematic", "rising tension", "quick energetic", "single continuous action", "loop-friendly", "custom"]
LIGHTING = ["auto-infer", "natural daylight", "golden hour", "blue hour", "moonlight", "soft studio", "dramatic low-key", "neon practicals", "firelight", "volumetric rays", "high contrast", "soft overcast", "monsoon overcast + neon reflections", "custom"]
TRANSITION_STYLE = ["auto-infer", "no transition", "smooth continuous motion", "start still then move", "end on held frame", "loopable ending", "match first and last frame", "custom"]


class _BaseVideoPromptImproved:
    CATEGORY = "ComfyUI-OMG/Video/Improved"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "shot_plan", "motion_notes", "model_settings")
    MODEL_PROFILE = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {"multiline": True, "default": "", "tooltip": "Few words allowed - e.g. 'chai rain Mumbai' - auto-expands to full brief with geography locks"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Optional: Ollama model - if missing uses CLIP fallback + fal presets"}),
                "clip": ("CLIP", {"tooltip": "Optional: CLIP from ComfyUI (downloaded model) - alternative path without Ollama, uses CLIP tokenizer + prompt templates"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional reference image 1 - identity/scene - analyzed via vision if Ollama vision model"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional reference image 2"}),
                "duration": (["auto-infer", "2 seconds", "5 seconds", "10 seconds", "15 seconds", "30 seconds"], {"default": "auto-infer"}),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "video_style": (VIDEO_STYLE, {"default": "auto-infer"}),
                "style_custom": ("STRING", {"default": ""}),
                "shot_type": (SHOT_TYPE, {"default": "auto-infer"}),
                "shot_custom": ("STRING", {"default": ""}),
                "camera_motion": (CAMERA_MOTION, {"default": "auto-infer"}),
                "camera_custom": ("STRING", {"default": ""}),
                "subject_motion": (SUBJECT_MOTION, {"default": "auto-infer"}),
                "subject_custom": ("STRING", {"default": ""}),
                "motion_intensity": (MOTION_INTENSITY, {"default": "auto-infer"}),
                "motion_custom": ("STRING", {"default": ""}),
                "pacing": (PACING, {"default": "auto-infer"}),
                "pacing_custom": ("STRING", {"default": ""}),
                "lighting": (LIGHTING, {"default": "auto-infer"}),
                "lighting_custom": ("STRING", {"default": ""}),
                "transition_style": (TRANSITION_STYLE, {"default": "auto-infer"}),
                "transition_custom": ("STRING", {"default": ""}),
                "reference_use": ("STRING", {"multiline": True, "default": "", "tooltip": "NEW: Authority assignment e.g. Image1 identity only, Video1 motion only - auto if empty"}),
                "screen_geography": ("STRING", {"multiline": True, "default": "", "tooltip": "NEW: Who stays left/right, foreground/mid/background - auto if empty, prevents drift"}),
                "negatives_focus": ("STRING", {"multiline": True, "default": "", "tooltip": "NEW: Focus on drift/swaps/broken eyeline vs generic bad quality"}),
                "first_frame": ("STRING", {"multiline": True, "default": ""}),
                "last_frame": ("STRING", {"multiline": True, "default": ""}),
                "additional_details": ("STRING", {"multiline": True, "default": "", "tooltip": "Continuity, wardrobe, props, voice, etc"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "High priority guidance"}),
                "use_clip_fallback": ("BOOLEAN", {"default": False, "tooltip": "Force CLIP offline template mode without Ollama"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller models - lean prompts, 8192 ctx, retries, lower temp"}),
                "fal_preset": (["auto-infer", "type_text_to_video", "type_first_last_frame", "type_identity_lock", "type_multi_asset_remix", "type_motion_transfer", "type_camera_transfer", "type_voice_clone", "type_object_replace", "type_green_screen_replace", "type_multi_edit", "type_product_showcase", "type_vertical_drama", "type_game_ui", "type_fashion_film", "vintage_binocular_brand_film", "epic_space_opera_teaser", "desert_fashion_campaign", "retro_anime_crime_title_sequence", "snowy_bamboo_wuxia_mystery", "interactive_game_equipment_ui", "claymation_lava_canyon_leap", "capybara_motion_recreation"], {"default": "auto-infer", "tooltip": "Optional: generic type presets (type_*) reusable, or fal 44 specific. Generic types are not topic-specific."}),
            }
        }

    def _get(self, kwargs, dropdown_key, custom_key, fallback="auto-infer"):
        dv = kwargs.get(dropdown_key, "auto-infer")
        cv = str(kwargs.get(custom_key, "") or "").strip()
        if dv == "auto-infer" and cv:
            return cv
        if dv not in ["auto-infer", "custom"] and dv:
            return dv
        if dv == "custom" and cv:
            return cv
        return fallback

    def generate(self, concept: str, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        use_clip_fallback = kwargs.get("use_clip_fallback", False)

        # Reference images
        from ...utils.image_utils import tensor_to_base64
        images_b64 = []
        num_images = 0
        for key in ["image_ref_1", "image_ref_2"]:
            img = kwargs.get(key)
            if img is not None:
                num_images += 1
                try:
                    images_b64.append(tensor_to_base64(img))
                except:
                    pass

        fal_preset = kwargs.get("fal_preset", "auto-infer")

        # CLIP fallback if no Ollama
        if ollama_model is None or use_clip_fallback:
            # Try to load preset if selected
            if fal_preset != "auto-infer":
                import os, json
                presets_path = os.path.join(os.path.dirname(__file__), "..", "..", "presets", "fal_minimax_h3_44_presets.json")
                try:
                    with open(presets_path, "r") as f:
                        data = json.load(f)
                        presets = {p["id"]: p for p in data.get("presets", [])}
                        p = presets.get(fal_preset)
                        if p:
                            base = p.get("prompt_template", "{concept}")
                            if "{concept}" in base:
                                positive = base.replace("{concept}", concept)
                            else:
                                positive = base + f"\n\n[USER CONCEPT]\n{concept}"
                            return (
                                positive,
                                p.get("negatives", ""),
                                p.get("shot_plan", ""),
                                f"Fal preset {fal_preset} with {num_images} image refs, CLIP {type(clip).__name__ if clip else 'none'}",
                                f"Preset {p.get('id','')} {p.get('category', p.get('type','generic'))} - CLIP fallback Offline"
                            )
                except Exception as e:
                    pass

            # Generic CLIP template for LTX/Wan
            style = self._get(kwargs, "video_style", "style_custom", "auto-infer")
            prompt = f"""[CLIP Fallback - No Ollama - {self.MODEL_PROFILE}]

[REFERENCE USE]
Image1 defines identity if provided ({num_images} images), Video reference defines motion if any. Ignore background unless needed.

[IDENTITY LOCKS]
Keep exactly one main subject, preserve face, clothing, distinctive props. No swaps.

[SCENE]
{concept} - Expand few words into cinematic moment with location, action, emotional turn.
Style: {style}
Duration: {kwargs.get('duration','auto')}
Aspect: {kwargs.get('aspect_ratio','16:9')}

[SCREEN GEOGRAPHY]
Subject frame center, stable left/right, foreground/mid/background legible.

[SHOT LIST]
0-4s Establish with reference if provided, 4-9s Escalate closer reaction, 9-15s Resolve stable tableau.

[LIGHT]
{kwargs.get('lighting','auto')}

[CAMERA]
{kwargs.get('camera_motion','slow dolly in')} - axis stable

[NEGATIVES]
No face drift, No wardrobe changes, No extra people, No broken eyeline, No unmotivated cuts

[CLIP INFO]
Using CLIP model {type(clip).__name__ if clip else 'none'} from ComfyUI downloaded models, {num_images} image references.
"""
            return (
                prompt,
                "No face drift, No wardrobe changes, No extra people, No broken eyeline",
                "0-4s Establish, 4-9s Escalate, 9-15s Resolve with end states",
                f"Reference {num_images} images locked, geography stable",
                f"{self.MODEL_PROFILE} CLIP Fallback Offline - uses fal presets templates, no Ollama needed"
            )

        cfg = ollama_model

        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        small_model_mode = kwargs.get("small_model_mode", False)

        system = build_system_prompt_improved(
            model_profile=self.MODEL_PROFILE,
            duration=self._get(kwargs, "duration", "", "auto-infer"),
            aspect_ratio=self._get(kwargs, "aspect_ratio", "", "16:9"),
            video_style=self._get(kwargs, "video_style", "style_custom", "auto-infer"),
            shot_type=self._get(kwargs, "shot_type", "shot_custom", "auto-infer"),
            camera_motion=self._get(kwargs, "camera_motion", "camera_custom", "auto-infer"),
            subject_motion=self._get(kwargs, "subject_motion", "subject_custom", "auto-infer"),
            motion_intensity=self._get(kwargs, "motion_intensity", "motion_custom", "auto-infer"),
            pacing=self._get(kwargs, "pacing", "pacing_custom", "auto-infer"),
            lighting=self._get(kwargs, "lighting", "lighting_custom", "auto-infer"),
            transition_style=self._get(kwargs, "transition_style", "transition_custom", "auto-infer"),
            first_frame=kwargs.get("first_frame", ""),
            last_frame=kwargs.get("last_frame", ""),
            reference_use=kwargs.get("reference_use", ""),
            screen_geography=kwargs.get("screen_geography", ""),
            negatives_focus=kwargs.get("negatives_focus", ""),
            additional_details=kwargs.get("additional_details", ""),
            custom_system_prompt=kwargs.get("custom_system_prompt", "")
        )

        user_prompt = f"""Create improved {self.MODEL_PROFILE} video prompt for concept (few words allowed, expand fully):

{concept}

Output valid JSON only."""

        # Small model handling: lean system, lower ctx, retries
        if small_model_mode and len(system) > 2000:
            system = system[:1500] + "\n\n[Lean 9B mode] Keep output short under 1000 chars, valid JSON only, no markdown.\n" + system[-500:]

        _log.info(f"[OMG Improved] Generating {self.MODEL_PROFILE} prompt for: {concept[:60]} small_model={small_model_mode}")

        # Retry logic for 9B
        raw = None
        for attempt in range(3 if small_model_mode else 1):
            try:
                raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.4 if small_model_mode else cfg.get("temperature", 0.65, think=think, filter_thinking=filter_thinking),
                    num_ctx=8192 if small_model_mode else cfg.get("num_ctx", 12288),
                    num_predict=1500 if small_model_mode else 3000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                    images=images_b64 if images_b64 else None
                )
                # Try parse early to check if valid
                test_parsed = extract_json_block(raw)
                if test_parsed is not None or not small_model_mode:
                    break
                else:
                    _log.warning(f"[SmallModel Retry {attempt+1}/3] JSON parse failed, retrying")
                    # Simplify system further on retry
                    system = f"You are {self.MODEL_PROFILE} prompt engineer for small 9B. Concept: {concept}. Output JSON with positive_prompt (full brief), negative_prompt, shot_plan. Keep under 1000 chars. JSON only."
            except TypeError:
                raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.4 if small_model_mode else cfg.get("temperature", 0.65, think=think, filter_thinking=filter_thinking),
                    num_ctx=8192 if small_model_mode else cfg.get("num_ctx", 12288),
                    num_predict=1500 if small_model_mode else 3000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json"
                )
                break
            except Exception as e:
                _log.warning(f"[Retry {attempt+1}] failed: {e}")
                if attempt == 2:
                    raise

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OMG Improved] Failed to parse JSON for %s", self.MODEL_PROFILE)
            return (raw, "", "", "", "")

        return (
            parsed.get("positive_prompt", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("shot_plan", ""),
            parsed.get("motion_notes", ""),
            parsed.get("model_settings", "")
        )


class OllamaLTXVVideoPromptImproved(_BaseVideoPromptImproved):
    MODEL_PROFILE = "LTXV 2.3"

class OllamaWanVideoPromptImproved(_BaseVideoPromptImproved):
    MODEL_PROFILE = "Wan 2.2"


NODE_CLASS_MAPPINGS = {
    "OllamaLTXVVideoPromptImproved": OllamaLTXVVideoPromptImproved,
    "OllamaWanVideoPromptImproved": OllamaWanVideoPromptImproved,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaLTXVVideoPromptImproved": "LTX-V Prompt - Improved",
    "OllamaWanVideoPromptImproved": "Wan Prompt - Improved",
}
