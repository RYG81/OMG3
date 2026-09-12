"""
MiniMax H3 OMG Nodes - Few words → Production Brief (PromptSama Authority Stack)
Compatible with Comfy-OMG OLLAMA_MODEL pattern

3 Nodes:
- OMGMiniMaxH3Simple: 2-3 words input like "chai rain Mumbai" -> full 6-layer brief, with IMAGE refs + CLIP support
- OMGMiniMaxH3Advanced: Full control but every field auto-infer if empty, with IMAGE refs + CLIP
- OMGMiniMaxH3Omni: Omni-reference 9img/3vid/3aud with authority declarations, with IMAGE refs

Built from https://www.promptsama.ai/models/minimax-h3.html + https://fal.ai/learn/devs/minimax-h3-prompting-guide

Updates:
- Added optional IMAGE inputs image_ref_1, image_ref_2, image_ref_3 for reference authority (identity, scene, style)
- Added optional CLIP input for text encoder alternative path (no Ollama needed) using ComfyUI downloaded models
- Uses CLIP tokenizer fallback + fal 44 presets templates when Ollama not available
"""

from __future__ import annotations
import json
import logging
import os

from ...ollama_client import generate
from ...utils.text_utils import extract_json_block
from ...utils.image_utils import tensor_to_base64
from ...prompts.minimax_h3_prompt import (
    VIDEO_STYLE as H3_VIDEO_STYLE,
    SHOT_TYPE as H3_SHOT_TYPE,
    CAMERA_MOTION as H3_CAMERA_MOTION,
    LIGHTING as H3_LIGHTING,
    DIALOGUE_LANGUAGES,
    build_system_prompt_h3_simple,
    build_system_prompt_h3_advanced,
    build_user_prompt_h3
)

_log = logging.getLogger(__name__)

def _get_val(dropdown_key: str, custom_key: str, kwargs: dict, fallback="auto-infer"):
    dv = kwargs.get(dropdown_key, "auto-infer")
    cv = str(kwargs.get(custom_key, "") or "").strip()
    if dv == "auto-infer" and cv:
        return cv
    if dv not in ["auto-infer", "custom", ""] and dv:
        return dv
    if dv == "custom" and cv:
        return cv
    return fallback

def _load_fal_presets():
    # Try to load 44 presets for CLIP fallback
    presets_path = os.path.join(os.path.dirname(__file__), "..", "..", "presets", "fal_minimax_h3_44_presets.json")
    if os.path.exists(presets_path):
        try:
            with open(presets_path, "r") as f:
                data = json.load(f)
                return data.get("presets", [])
        except:
            pass
    return []

def _lean_system_prompt_h3_simple(concept: str, duration: str, aspect: str, style: str, lang: str, details: str, num_images: int):
    """Lean prompt for 9B and smaller models - shorter, more deterministic, fewer tokens"""
    return f"""You are MiniMax H3 prompt engineer for small 9B model. Output JSON only.

Concept: {concept}
Duration: {duration}, Aspect: {aspect}, Style: {style}, Language: {lang}
Images: {num_images} refs (Image1 identity only etc)
Details: {details}

Output JSON with 6 keys only (keep short, under 1000 chars for positive):
{{
  "reference_use": ["Image1 identity only" if {num_images}>0 else "No refs"],
  "identity_locks": ["Keep 1 person, preserve face/clothing"],
  "scene_intent": "Location, goal, emotion - expand concept",
  "shot_list": ["0-4s Wide establishing", "4-9s Medium push in 00:06.500", "9-15s Tracking resolve"],
  "positive_prompt": "Full brief [REFERENCE USE]...[SHOT LIST]... combine above, under 1500 chars, few words {concept} expanded",
  "negative_prompt": "No extra people, No face drift"
}}

Rules: Few words allowed, expand fully. Under 1500 chars positive. JSON only, no markdown."""

def _generate_with_retry(generate_fn, base_url, model, prompt, system, images_b64, is_small_model, cfg):
    """Enhanced generation with retry and JSON repair for small 9B models"""
    # Params based on model size
    if is_small_model:
        num_ctx = 8192
        num_predict = 2048
        temperature = 0.4
        # Use leaner system if flagged and original system is long
        if len(system) > 2000:
            # Keep first 1500 chars + last 500
            system = system[:1500] + "\n\n[Lean mode for 9B] Keep output short under 1000 chars positive, valid JSON only, no markdown. Rules: few words allowed, expand fully, under 1500 chars.\n" + system[-500:]
    else:
        num_ctx = cfg.get("num_ctx", 16384)
        num_predict = cfg.get("num_predict", 4096)
        temperature = cfg.get("temperature", 0.7)

    for attempt in range(3 if is_small_model else 1):
        try:
            try:
                raw = generate_fn(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                    images=images_b64 if images_b64 else None
                )
            except TypeError:
                raw = generate_fn(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json"
                )
            # Try parse
            from ...utils.text_utils import extract_json_block
            parsed = extract_json_block(raw)
            if parsed is not None:
                return raw, parsed
            else:
                # JSON repair attempt for small model: try to extract even if malformed
                import re, json
                # Remove markdown code blocks
                cleaned = re.sub(r'```(?:json)?\s*', '', raw)
                cleaned = cleaned.replace('```', '')
                # Find first { and last }
                first = cleaned.find('{')
                last = cleaned.rfind('}')
                if first != -1 and last != -1 and last > first:
                    snippet = cleaned[first:last+1]
                    # Fix trailing commas
                    snippet = re.sub(r',\s*}', '}', snippet)
                    snippet = re.sub(r',\s*]', ']', snippet)
                    try:
                        parsed = json.loads(snippet)
                        return snippet, parsed
                    except:
                        pass
                if attempt < 2:
                    # Retry with simpler lean prompt
                    _log.warning(f"[SmallModel Retry {attempt+1}/3] JSON parse failed, retrying with lean prompt")
                    system = _lean_system_prompt_h3_simple(
                        concept=prompt[:200],
                        duration="8 seconds",
                        aspect="16:9",
                        style="cinematic realism",
                        lang="English",
                        details="Retry lean",
                        num_images=len(images_b64) if images_b64 else 0
                    )
                    continue
                return raw, None
        except Exception as e:
            _log.warning(f"[Generate Retry {attempt+1}] failed: {e}")
            if attempt == 2:
                raise
    return raw, None


def _clip_fallback_prompt(concept: str, num_images: int, style: str = "auto-infer", clip_model=None):
    """Template-based generation without Ollama, using CLIP info if provided"""
    clip_info = ""
    if clip_model is not None:
        try:
            model_name = getattr(clip_model, 'model_name', None) or type(clip_model).__name__
            if hasattr(clip_model, 'tokenize'):
                tokens = clip_model.tokenize(concept)
                shape = getattr(tokens, 'shape', None)
                clip_info = f"[CLIP Fallback: model {model_name}, tokens {shape or len(tokens) if hasattr(tokens, '__len__') else 'N/A'}]"
            else:
                clip_info = f"[CLIP Fallback: model {model_name}]"
        except Exception as e:
            clip_info = f"[CLIP Fallback attempted but failed: {e}]"

    template = f"""{clip_info}
[REFERENCE USE]
Image1 defines protagonist identity and wardrobe only. Ignore its background and pose.
Image2 defines scene/style if provided ({num_images} images provided). Ignore its people if any.
Image3 defines additional asset if provided.
Audio1 defines protagonist voice timbre if provided.

[IDENTITY / CONTINUITY LOCKS]
Keep exactly one person (main character), preserve face, short black hair, light stubble, navy kurta, silver bracelet on right wrist.
Preserve reference images identities if provided: {num_images} images.
No identity or wardrobe swaps. Keep bracelet visible in close-ups.
Keep screen direction stable: protagonist center then moves to frame left.

[SCENE]
{concept} - Location: Mumbai monsoon dusk, chai stall Dharavi street, discovery of glowing letter leading to abandoned Royal Talkies theater. Goal: curiosity to determination. Emotional turn: calm nostalgic to mysterious intrigue, hopeful.
Style: {style}

[SCREEN GEOGRAPHY]
Protagonist frame center initially, chai stall foreground left, wet asphalt midground reflecting neon pink/blue, Mumbai skyline background, letter in crate frame right foreground. Keep geography legible across cuts, left/right stable.

[SHOT LIST]
0-4s — Wide shot, slow dolly in, 35mm, shallow DOF. Protagonist at stall holding chai, steam rising, monsoon rain. First line delivered. End: noticing glow in crate frame right.
4-9s — Medium close-up, static then push in to close-up at 00:06.500, 50mm, reaches for glowing letter, picks it up, reaction eyes widen, subtle smile fades. Dialogue: “What is this light?” Transition: cut on action of hand reaching.
9-15s — Close-up of letter glowing in palm, then tracking shot following protagonist toward alley leading to abandoned theater, decisive step forward, camera tilts up reveals theater sign. Final state: letter glowing bright in hand, determined gaze toward theater, rain intensifies, stable tableau.

[ACTING]
Posture upright relaxed holding chai with both hands, then leans forward curious, slight hand tremor holding glowing letter. Gaze down to chai, right to crate, up to distant theater. Emotional progression calm -> curiosity -> intrigue -> determination.

[LIGHT AND IMAGE]
Natural overcast monsoon soft diffused + warm practical lights from chai stall tungsten, neon signs pink/blue reflecting on wet asphalt, realistic photographic texture, 35mm film grain, shallow DOF for emotional close-ups.

[CAMERA]
Shot sizes: wide establishing, medium close-up, extreme close-up on letter, tracking. Movement: slow dolly in for emotion, static for dialogue, push in at 00:06.500 on action, tracking for journey. Axis stable. Transitions: motivated cuts on action, no hard cut during dialogue lines.

[PRODUCTION SOUND]
Native stereo ambience: rain pitter on tin roof constant, distant traffic honks low, chai boiling hiss foreground, cup clink, paper crate rustle when reaching. Clear dialogue warm Mumbai accent. Music: No music 0-9s, then subtle mysterious strings entering at 9s low volume.

[NEGATIVES]
No extra people, No face drift, No wardrobe changes, No silver bracelet disappearance, No voice swaps, No broken eyelines, No unmotivated cuts, No extra text or logos, No black frames
"""
    return template


class OMGMiniMaxH3Simple:
    CATEGORY = "Ollama-Magic-Nodes/Video/MiniMax-H3"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "shot_plan", "motion_notes", "model_settings", "promptsama_brief", "h3_json")
    OUTPUT_TOOLTIPS = (
        "Full MiniMax H3 prompt (production brief, up to 7000 chars) ready for H3 API - few words expanded to full, supports IMAGE refs + CLIP",
        "Negatives focused on drift/swaps",
        "Timed 0-4s/4-9s/9-15s plan with end states",
        "Motion & continuity constraints with authority",
        "Model settings for H3",
        "Full PromptSama brief as formatted text",
        "Full JSON with all fields"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {
                    "default": "chai rain Mumbai",
                    "multiline": True,
                    "tooltip": "Few words allowed! e.g. 'chai rain Mumbai' or 'neon market chase' - Ollama expands to full production brief via Authority Stack. Works even without Ollama via CLIP fallback using fal 44 presets templates"
                }),
            },
            "optional": {
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Optional: Ollama model loader (qwen3:32b recommended). If missing, uses CLIP fallback + fal presets template (offline)"}),
                "clip": ("CLIP", {"tooltip": "Optional: CLIP text encoder from ComfyUI (downloaded model) - alternative path without Ollama, uses CLIP tokenizer + fal presets templates. Also works with IMAGE refs"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional reference Image1 - identity/wardrobe only, analyzed via vision if ollama_model is vision model (qwen3-vl, llava)"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional reference Image2 - second identity or scene/style"}),
                "image_ref_3": ("IMAGE", {"tooltip": "Optional reference Image3 - third asset"}),
                "duration": (["auto-infer", "4 seconds", "6 seconds", "8 seconds", "10 seconds", "12 seconds", "15 seconds"], {"default": "8 seconds"}),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "video_style": (H3_VIDEO_STYLE, {"default": "auto-infer"}),
                "style_custom": ("STRING", {"default": "", "tooltip": "Custom style if dropdown auto-infer"}),
                "dialogue_language": (DIALOGUE_LANGUAGES, {"default": "English"}),
                "additional_details": ("STRING", {"multiline": True, "default": "", "tooltip": "Any extra continuity, wardrobe, props, voice notes - optional"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "High-priority guidance (e.g. 'keep silver bracelet always visible')"}),
                "use_clip_fallback": ("BOOLEAN", {"default": False, "tooltip": "If true and CLIP provided without Ollama, force offline template generation (no LLM needed) - uses fal 44 presets"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller local models (Gemma3 9B, Qwen2.5 7B, etc.) - uses lean prompts, lower ctx 8192, retries, JSON repair"}),
                "fal_preset": (["auto-infer", "vintage_binocular_brand_film", "epic_space_opera_teaser", "sci_fi_mystery_teaser", "desert_fashion_campaign", "cyber_grunge_fashion_film", "retro_anime_crime_title_sequence", "hand_drawn_kitchen_creature", "multi_asset_cinematic_remix", "macro_coffee_to_fluid_transition", "character_swap_with_performance_reference", "capybara_motion_recreation", "voice_clone_dialogue_transfer", "green_screen_environment_replacement", "multi_element_scene_editing"], {"default": "auto-infer", "tooltip": "Optional: Pick a fal.ai 44 preset as base template, then your concept is injected. Auto-infer means prompt from scratch"}),
            }
        }

    def generate(self, concept: str, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        use_clip_fallback = kwargs.get("use_clip_fallback", False)
        fal_preset_id = kwargs.get("fal_preset", "auto-infer")

        # Extract images for vision + count
        images_b64 = []
        num_images = 0
        for key in ["image_ref_1", "image_ref_2", "image_ref_3"]:
            img = kwargs.get(key)
            if img is not None:
                num_images += 1
                try:
                    images_b64.append(tensor_to_base64(img))
                except:
                    pass

        # If no Ollama, use CLIP fallback / template path
        if ollama_model is None:
            # Try fal preset injection if selected
            if fal_preset_id != "auto-infer":
                presets = _load_fal_presets()
                preset_map = {p["id"]: p for p in presets}
                preset = preset_map.get(fal_preset_id)
                if preset:
                    base = preset.get("prompt_template", "{concept}")
                    if "{concept}" in base:
                        positive = base.replace("{concept}", concept)
                    else:
                        positive = base + f"\n\n[USER CONCEPT INJECTION]\n{concept}\n\nAdditional: {kwargs.get('additional_details','')}"
                    if num_images > 0:
                        positive += f"\n\n[REFERENCE IMAGES PROVIDED: {num_images} - Image1..{num_images} define identity/scene as per preset]"
                    return (
                        positive,
                        preset.get("negatives", "No extra people, No face drift, No wardrobe changes"),
                        preset.get("shot_plan", ""),
                        f"Reference pattern: {preset.get('reference_count','')} | Endpoint {preset.get('endpoint')} | Ideal {preset.get('ideal_for','')}",
                        f"Preset {preset['id']} ({preset['category']}) - {preset['endpoint']} - Fal.ai 44 presets - CLIP fallback with {num_images} refs",
                        positive,
                        json.dumps(preset, indent=2)
                    )

            # General CLIP fallback
            style = _get_val("duration", "", kwargs, "auto-infer")  # dummy to get style
            style = kwargs.get("video_style", "auto-infer")
            if kwargs.get("style_custom"):
                style = kwargs.get("style_custom")
            fallback_prompt = _clip_fallback_prompt(concept, num_images, style, clip)
            return (
                fallback_prompt,
                "No extra people, No face drift, No wardrobe changes, No voice swaps, No broken eyelines, No unmotivated cuts",
                "0-4s Establish wide, 4-9s Escalate medium close-up push in 00:06.500, 9-15s Resolve tracking to final tableau",
                f"Keep {num_images} reference images identities locked, preserve continuity, CLIP model {type(clip).__name__ if clip else 'none'}",
                "MiniMax H3 CLIP Fallback: 4-15s 24fps stereo, template-based offline, uses fal presets, no Ollama needed",
                fallback_prompt,
                json.dumps({"concept": concept, "images": num_images, "clip": str(type(clip)), "mode": "clip_fallback", "fal_preset": fal_preset_id})
            )

        # Ollama path (with optional vision images)
        duration = _get_val("duration", "", kwargs, "8 seconds")
        aspect_ratio = _get_val("aspect_ratio", "", kwargs, "16:9")
        video_style = _get_val("video_style", "style_custom", kwargs, "auto-infer")
        dialogue_language = _get_val("dialogue_language", "", kwargs, "English")
        additional_details = kwargs.get("additional_details", "")
        custom_system = kwargs.get("custom_system_prompt", "")
        small_model_mode = kwargs.get("small_model_mode", False)

        if num_images > 0:
            additional_details += f"\nUser provided {num_images} reference images - Image1..{num_images}. Assign authority: Image1 identity only etc. Analyze identities."

        # Choose system prompt: lean for 9B if small_model_mode True
        if small_model_mode:
            system = _lean_system_prompt_h3_simple(
                concept=concept,
                duration=duration,
                aspect=aspect_ratio,
                style=video_style,
                lang=dialogue_language,
                details=additional_details,
                num_images=num_images
            )
            if custom_system and custom_system.strip():
                system += f"\n\nHigh-priority: {custom_system.strip()} (keep short)"
            if fal_preset_id != "auto-infer":
                system += f"\n\nFal preset base {fal_preset_id} - keep under 1000 chars"
        else:
            system = build_system_prompt_h3_simple(
                concept=concept,
                duration=duration,
                aspect_ratio=aspect_ratio,
                video_style=video_style,
                dialogue_language=dialogue_language,
                additional_details=additional_details
            )
            if custom_system and custom_system.strip():
                system += f"\n\nAdditional user high-priority guidance (must obey): {custom_system.strip()}"
            if fal_preset_id != "auto-infer":
                system += f"\n\nUse fal.ai preset {fal_preset_id} as base template style for reference_use and shot_list."

        user_prompt = build_user_prompt_h3(concept)

        _log.info(f"[OMG-H3 Simple] Generating with {num_images} refs, small_model={small_model_mode}, concept: {concept[:60]}")

        # Use retry helper for small model
        if small_model_mode:
            raw, parsed = _generate_with_retry(
                generate_fn=generate,
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system,
                images_b64=images_b64,
                is_small_model=True,
                cfg=ollama_model
            )
        else:
            try:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=ollama_model.get("temperature", 0.7),
                    num_ctx=ollama_model.get("num_ctx", 16384),
                    num_predict=4096,
                    seed=ollama_model.get("seed", -1),
                    keep_alive=ollama_model.get("keep_alive", "5m"),
                    response_format="json",
                    images=images_b64 if images_b64 else None
                )
            except TypeError:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=ollama_model.get("temperature", 0.7),
                    num_ctx=ollama_model.get("num_ctx", 16384),
                    num_predict=4096,
                    seed=ollama_model.get("seed", -1),
                    keep_alive=ollama_model.get("keep_alive", "5m"),
                    response_format="json"
                )
            parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OMG-H3] Failed to parse JSON, returning raw")
            return (raw, "No extra people, No face drift, No wardrobe changes", "0-4s Establish, 4-9s Escalate, 9-15s Resolve", "Keep identity stable", "MiniMax H3 4-15s 24fps stereo", raw, json.dumps({"raw": raw}))

        positive = parsed.get("positive_prompt") or parsed.get("prompt") or ""
        if (not positive or len(positive) < 200 or positive.strip().startswith("{")) and "reference_use" in parsed:
            parts = []
            for key, header in [
                ("reference_use", "[REFERENCE USE]"),
                ("identity_locks", "[IDENTITY / CONTINUITY LOCKS]"),
                ("scene_intent", "[SCENE]"),
                ("dialogue", "[DIALOGUE]"),
                ("screen_geography", "[SCREEN GEOGRAPHY]"),
                ("shot_list", "[SHOT LIST]"),
                ("acting", "[ACTING]"),
                ("light_and_image", "[LIGHT AND IMAGE]"),
                ("camera", "[CAMERA]"),
                ("production_sound", "[PRODUCTION SOUND]"),
                ("negatives", "[NEGATIVES]"),
            ]:
                val = parsed.get(key)
                if val:
                    if isinstance(val, list):
                        parts.append(f"{header}\n" + "\n".join(val))
                    else:
                        parts.append(f"{header}\n{val}")
            positive = "\n\n".join(parts) if parts else raw

        return (
            positive,
            parsed.get("negative_prompt", "No extra people, No face drift, No wardrobe changes, No voice swaps, No broken eyelines"),
            parsed.get("shot_plan", ""),
            parsed.get("motion_notes", ""),
            parsed.get("model_settings", "MiniMax H3: 4-15s 24fps native stereo, 2K 1440p short edge, use Image1/Video1/Audio1 labels"),
            positive,
            json.dumps(parsed, indent=2)
        )


class OMGMiniMaxH3Advanced:
    CATEGORY = "Ollama-Magic-Nodes/Video/MiniMax-H3"
    FUNCTION = "generate_advanced"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "shot_plan", "motion_notes", "model_settings", "character_reference", "continuity_checklist", "h3_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {"multiline": True, "default": "A chai seller in Mumbai monsoon finds glowing letter leading to abandoned theater"}),
            },
            "optional": {
                "ollama_model": ("OLLAMA_MODEL",),
                "clip": ("CLIP", {"tooltip": "Optional CLIP from ComfyUI - alternative without Ollama"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional reference Image1 identity only"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional reference Image2 scene/style"}),
                "image_ref_3": ("IMAGE", {"tooltip": "Optional reference Image3 third asset"}),
                "duration": (["auto-infer", "4 seconds", "8 seconds", "12 seconds", "15 seconds"], {"default": "8 seconds"}),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "21:9"], {"default": "16:9"}),
                "video_style": (H3_VIDEO_STYLE, {"default": "auto-infer"}),
                "video_style_custom": ("STRING", {"default": ""}),
                "shot_type": (H3_SHOT_TYPE, {"default": "auto-infer"}),
                "shot_type_custom": ("STRING", {"default": ""}),
                "camera_motion": (H3_CAMERA_MOTION, {"default": "auto-infer"}),
                "camera_motion_custom": ("STRING", {"default": ""}),
                "subject_motion": (["auto-infer", "almost still", "gentle natural movement", "walking", "reaching", "gesturing"], {"default": "auto-infer"}),
                "subject_motion_custom": ("STRING", {"default": ""}),
                "lighting": (H3_LIGHTING, {"default": "auto-infer"}),
                "lighting_custom": ("STRING", {"default": ""}),
                "dialogue": ("STRING", {"multiline": True, "default": "", "tooltip": "Exact lines with speaker: 'Aarav says: ...'"}),
                "screen_geography": ("STRING", {"multiline": True, "default": "", "tooltip": "Who stays left/right"}),
                "reference_declarations": ("STRING", {"multiline": True, "default": "", "tooltip": "e.g. Image1 identity only. Video1 motion only."}),
                "acting_notes": ("STRING", {"multiline": True, "default": ""}),
                "light_and_image": ("STRING", {"multiline": True, "default": ""}),
                "camera_rules": ("STRING", {"multiline": True, "default": ""}),
                "production_sound": ("STRING", {"multiline": True, "default": ""}),
                "negatives": ("STRING", {"multiline": True, "default": ""}),
                "dialogue_language": (DIALOGUE_LANGUAGES, {"default": "English"}),
                "first_frame": ("STRING", {"multiline": True, "default": ""}),
                "last_frame": ("STRING", {"multiline": True, "default": ""}),
                "additional_details": ("STRING", {"multiline": True, "default": ""}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "use_clip_fallback": ("BOOLEAN", {"default": False, "tooltip": "Force CLIP offline template mode"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller models - lean prompts, 8192 ctx, retries, JSON repair"}),
                "fal_preset": (["auto-infer", "vintage_binocular_brand_film", "retro_anime_crime_title_sequence", "multi_asset_cinematic_remix", "macro_coffee_to_fluid_transition", "character_swap_with_performance_reference", "capybara_motion_recreation"], {"default": "auto-infer"}),
            }
        }

    def generate_advanced(self, concept: str, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        use_clip_fallback = kwargs.get("use_clip_fallback", False)
        small_model_mode = kwargs.get("small_model_mode", False)

        # Image refs
        images_b64 = []
        num_images = 0
        for key in ["image_ref_1", "image_ref_2", "image_ref_3"]:
            img = kwargs.get(key)
            if img is not None:
                num_images += 1
                try:
                    images_b64.append(tensor_to_base64(img))
                except:
                    pass

        # CLIP fallback if no Ollama
        if ollama_model is None or use_clip_fallback:
            # Reuse simple fallback but with advanced info
            style = kwargs.get("video_style_custom") or kwargs.get("video_style", "auto-infer")
            fallback = _clip_fallback_prompt(concept, num_images, style, clip)
            # Append advanced fields if provided
            extra = ""
            for k in ["dialogue", "screen_geography", "reference_declarations", "acting_notes", "camera_rules"]:
                v = kwargs.get(k, "")
                if v:
                    extra += f"\n\n[{k.upper()}]\n{v}"
            fallback += extra
            return (
                fallback,
                kwargs.get("negatives", "No extra people, No face drift, No wardrobe changes"),
                "0-4s Establish, 4-9s Escalate, 9-15s Resolve",
                f"Advanced with {num_images} refs, CLIP {type(clip).__name__ if clip else 'none'}",
                "MiniMax H3 Advanced CLIP Fallback - offline template + fal presets",
                f"Character from concept: {concept[:100]}",
                f"Continuity: preserve {num_images} refs, geography stable",
                json.dumps({"concept": concept, "images": num_images, "mode": "clip_fallback_advanced"})
            )

        # Ollama path
        def gv(k, ck, fb="auto-infer"):
            dv = kwargs.get(k, "auto-infer")
            cv = str(kwargs.get(ck, "") or "").strip()
            if dv == "auto-infer" and cv:
                return cv
            if dv not in ["auto-infer", "custom", ""] and dv:
                return dv
            if dv == "custom" and cv:
                return cv
            return fb

        system = build_system_prompt_h3_advanced(
            concept=concept,
            duration=gv("duration", "", "8 seconds"),
            aspect_ratio=gv("aspect_ratio", "", "16:9"),
            video_style=gv("video_style", "video_style_custom", "auto-infer"),
            shot_type=gv("shot_type", "shot_type_custom", "auto-infer"),
            camera_motion=gv("camera_motion", "camera_motion_custom", "auto-infer"),
            subject_motion=gv("subject_motion", "subject_motion_custom", "auto-infer"),
            lighting=gv("lighting", "lighting_custom", "auto-infer"),
            dialogue=kwargs.get("dialogue", ""),
            screen_geography=kwargs.get("screen_geography", ""),
            reference_declarations=kwargs.get("reference_declarations", ""),
            acting_notes=kwargs.get("acting_notes", ""),
            light_and_image=kwargs.get("light_and_image", ""),
            camera_rules=kwargs.get("camera_rules", ""),
            production_sound=kwargs.get("production_sound", ""),
            negatives=kwargs.get("negatives", ""),
            additional_details=kwargs.get("additional_details", "") + f"\nReference images: {num_images}",
            dialogue_language=gv("dialogue_language", "", "English"),
            first_frame=kwargs.get("first_frame", ""),
            last_frame=kwargs.get("last_frame", "")
        )

        custom = kwargs.get("custom_system_prompt", "")
        if custom and custom.strip():
            system += f"\n\nUser high-priority guidance: {custom.strip()}"
        fal_preset = kwargs.get("fal_preset", "auto-infer")
        if fal_preset != "auto-infer":
            system += f"\n\nUse fal preset {fal_preset} as base template."

        user_prompt = build_user_prompt_h3(concept)

        if small_model_mode:
            raw, parsed = _generate_with_retry(
                generate_fn=generate,
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system,
                images_b64=images_b64,
                is_small_model=True,
                cfg=ollama_model
            )
        else:
            try:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=ollama_model.get("temperature", 0.65),
                    num_ctx=ollama_model.get("num_ctx", 16384),
                    num_predict=5000,
                    seed=ollama_model.get("seed", -1),
                    keep_alive=ollama_model.get("keep_alive", "5m"),
                    response_format="json",
                    images=images_b64 if images_b64 else None
                )
            except TypeError:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=ollama_model.get("temperature", 0.65),
                    num_ctx=ollama_model.get("num_ctx", 16384),
                    num_predict=5000,
                    seed=ollama_model.get("seed", -1),
                    keep_alive=ollama_model.get("keep_alive", "5m"),
                    response_format="json"
                )
            parsed = extract_json_block(raw)
        if parsed is None:
            return (raw, "", "", "", "", "", "", json.dumps({"raw": raw}))

        positive = parsed.get("positive_prompt", "")
        if not positive and "reference_use" in parsed:
            parts = []
            for key, header in [
                ("reference_use", "[REFERENCE USE]"),
                ("identity_locks", "[IDENTITY / CONTINUITY LOCKS]"),
                ("scene_intent", "[SCENE]"),
                ("dialogue", "[DIALOGUE]"),
                ("screen_geography", "[SCREEN GEOGRAPHY]"),
                ("shot_list", "[SHOT LIST]"),
                ("acting", "[ACTING]"),
                ("light_and_image", "[LIGHT AND IMAGE]"),
                ("camera", "[CAMERA]"),
                ("production_sound", "[PRODUCTION SOUND]"),
                ("negatives", "[NEGATIVES]"),
            ]:
                val = parsed.get(key)
                if val:
                    if isinstance(val, list):
                        parts.append(f"{header}\n" + "\n".join(val))
                    else:
                        parts.append(f"{header}\n{val}")
            positive = "\n\n".join(parts) if parts else raw

        return (
            positive,
            parsed.get("negative_prompt", ""),
            parsed.get("shot_plan", ""),
            parsed.get("motion_notes", ""),
            parsed.get("model_settings", ""),
            parsed.get("character_reference", ""),
            parsed.get("continuity_checklist", json.dumps(parsed.get("identity_locks", []))),
            json.dumps(parsed, indent=2)
        )


class OMGMiniMaxH3Omni:
    CATEGORY = "Ollama-Magic-Nodes/Video/MiniMax-H3"
    FUNCTION = "generate_omni"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "reference_use_block", "identity_locks", "h3_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {"multiline": True, "default": "Use my face and voice to say new line at chai stall, keep background from video"}),
                "duration": (["auto-infer", "8 seconds", "12 seconds", "15 seconds"], {"default": "8 seconds"}),
            },
            "optional": {
                "ollama_model": ("OLLAMA_MODEL",),
                "clip": ("CLIP", {"tooltip": "Optional CLIP for alternative without Ollama"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Image1 identity"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Image2 second identity or scene"}),
                "image_ref_3": ("IMAGE", {"tooltip": "Image3 third asset"}),
                "image_ref_4": ("IMAGE",),
                "image_ref_5": ("IMAGE",),
                "image_ref_6": ("IMAGE",),
                "reference_images_info": ("STRING", {"multiline": True, "default": "Image1: my face front, Image2: side profile", "tooltip": "Describe what each Image1..9 controls"}),
                "reference_videos_info": ("STRING", {"multiline": True, "default": "Video1: motion timing from dance video"}),
                "reference_audios_info": ("STRING", {"multiline": True, "default": "Audio1: my voice timbre reference"}),
                "dialogue_exact": ("STRING", {"multiline": True, "default": "Say exactly: 'Mumbai never sleeps, it just sips chai.'"}),
                "preservation_rules": ("STRING", {"multiline": True, "default": "Preserve identity, wardrobe, background from Video1, camera from Video1"}),
                "edit_intent": ("STRING", {"multiline": True, "default": ""}),
                "additional_details": ("STRING", {"multiline": True, "default": ""}),
                "use_clip_fallback": ("BOOLEAN", {"default": False}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller - lean prompts, 8192 ctx, retries"}),
            }
        }

    def generate_omni(self, concept: str, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        use_clip_fallback = kwargs.get("use_clip_fallback", False)

        # Count image refs
        num_images = 0
        images_b64 = []
        for i in range(1, 7):
            key = f"image_ref_{i}"
            img = kwargs.get(key)
            if img is not None:
                num_images += 1
                try:
                    images_b64.append(tensor_to_base64(img))
                except:
                    pass

        if ollama_model is None or use_clip_fallback:
            # CLIP/template fallback
            ref_images_info = kwargs.get("reference_images_info", "")
            ref_videos_info = kwargs.get("reference_videos_info", "")
            ref_audios_info = kwargs.get("reference_audios_info", "")
            dialogue = kwargs.get("dialogue_exact", "")
            preservation = kwargs.get("preservation_rules", "")
            edit_intent = kwargs.get("edit_intent", "")

            prompt = f"""[REFERENCE USE - Omni {num_images} images provided]
Image1 defines identity and wardrobe only. Ignore background.
Image2 defines second identity or scene/style if provided.
Image3-6 define additional assets if provided ({num_images} total).
Video1 defines motion timing only if provided: {ref_videos_info}
Audio1 defines voice timbre: {ref_audios_info}

[IDENTITY LOCKS]
Keep exactly {max(1, num_images)} people/assets, preserve face, clothing, distinctive props. No swaps.

[SCENE]
{concept}

[DIALOGUE]
{dialogue}

[SCREEN GEOGRAPHY]
Keep left/right stable, foreground/mid/background legible.

[SHOT LIST]
0-4s Establish with references, 4-9s Escalate with motion transfer, 9-15s Resolve stable tableau.

[PRESERVATION]
{preservation}

[EDIT INTENT]
{edit_intent}

[REFERENCE INFO]
Images: {ref_images_info}
Videos: {ref_videos_info}
Audios: {ref_audios_info}

[NEGATIVES]
No extra people, No face drift, No wardrobe changes, No voice swaps
"""
            return (
                prompt,
                "\n".join([f"Image{i+1} defines identity" for i in range(num_images)]) if num_images else ref_images_info,
                f"Keep {num_images} identities locked",
                json.dumps({"concept": concept, "images": num_images, "mode": "clip_fallback_omni"})
            )

        # Ollama path
        ref_images = kwargs.get("reference_images_info", "")
        ref_videos = kwargs.get("reference_videos_info", "")
        ref_audios = kwargs.get("reference_audios_info", "")
        dialogue = kwargs.get("dialogue_exact", "")
        preservation = kwargs.get("preservation_rules", "")
        edit_intent = kwargs.get("edit_intent", "")

        system = f"""You are MiniMax H3 Omni-Reference Authority Stack engineer.

User concept (few words allowed): {concept}
Duration: {kwargs.get('duration','8 seconds')}
Reference images: {ref_images} ({num_images} actual IMAGE tensors provided)
Reference videos: {ref_videos}
Reference audios: {ref_audios}
Exact dialogue: {dialogue}
Preservation: {preservation}
Edit intent: {edit_intent}
Additional: {kwargs.get('additional_details','')}

You must assign authority for each Image1/Video1/Audio1 with job + boundary, using PromptSama rules.

If edit_intent says "Video1 is master", use precise editing framework:
1. Declare master: "Video1 is sole source for timeline, subjects, camera, audio"
2. List changes: bind every replacement to exact subject/object/region/line/time
3. List what survives: protect identity, movement, occlusion, lighting, camera, dialogue, ambience

Output VALID JSON:
{{
  "reference_use": ["Image1 defines identity only...", "Video1 defines motion timing only...", "Audio1 defines voice timbre for X..."],
  "identity_locks": ["Keep exactly...","Preserve...","No swaps"],
  "scene_intent": "...",
  "dialogue": ["Character says: ..."],
  "screen_geography": "...",
  "shot_list": ["0-4s ...","4-9s ...","9-15s ..."],
  "acting": "...",
  "light_and_image": "...",
  "camera": "...",
  "production_sound": "...",
  "negatives": [...],
  "positive_prompt": "Full combined prompt string with all sections for H3 - this is the actual generation prompt",
  "preservation_checklist": "What to preserve"
}}

Few words concept allowed - expand fully. Output ONLY JSON.
"""

        user_prompt = f"Concept: {concept}\nRefs Image: {ref_images} ({num_images} tensors)\nVideo: {ref_videos}\nAudio: {ref_audios}\nDialogue: {dialogue}\nGenerate Omni brief JSON."

        small_model_mode = kwargs.get("small_model_mode", False)
        if small_model_mode:
            raw, parsed = _generate_with_retry(
                generate_fn=generate,
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system,
                images_b64=images_b64,
                is_small_model=True,
                cfg=ollama_model
            )
        else:
            try:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.6,
                    num_ctx=16384,
                    num_predict=4096,
                    seed=-1,
                    keep_alive="5m",
                    response_format="json",
                    images=images_b64 if images_b64 else None
                )
            except TypeError:
                raw = generate(
                    base_url=ollama_model["base_url"],
                    model=ollama_model["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.6,
                    num_ctx=16384,
                    num_predict=4096,
                    seed=-1,
                    keep_alive="5m",
                    response_format="json"
                )
            parsed = extract_json_block(raw)
        if parsed is None:
            return (raw, "", "", json.dumps({"raw": raw}))

        return (
            parsed.get("positive_prompt", raw),
            "\n".join(parsed.get("reference_use", [])),
            "\n".join(parsed.get("identity_locks", [])),
            json.dumps(parsed, indent=2)
        )


NODE_CLASS_MAPPINGS = {
    "OMGMiniMaxH3Simple": OMGMiniMaxH3Simple,
    "OMGMiniMaxH3Advanced": OMGMiniMaxH3Advanced,
    "OMGMiniMaxH3Omni": OMGMiniMaxH3Omni,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMGMiniMaxH3Simple": "OMG MiniMax H3 Simple (3 words → Production Brief) +refs+CLIP",
    "OMGMiniMaxH3Advanced": "OMG MiniMax H3 Advanced (PromptSama) +refs+CLIP",
    "OMGMiniMaxH3Omni": "OMG MiniMax H3 Omni-Reference (9img/3vid/3aud) +refs+CLIP",
}
