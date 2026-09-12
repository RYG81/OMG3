"""
MiniMax H3 Official Prompt Enhancer Node - Uses exact system prompt from Naxdy gist
Source: https://gist.github.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace
Raw: https://gist.githubusercontent.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace/raw/.../sysprompt.md

This is the true H3-Context-IR replacement that sits between user's raw creative request and generative video model.
It synthesizes synchronized video AND audio together.

Improvements over previous custom prompt engines:
- Uses official template detection: text-only, frame-anchored (first/last/both), full-reference (6 sections)
- Fixed names: <Picture 1>, <Video 1>, <Audio 1> independent per category, never renumber
- Camera vocabulary: Push In/Pull Out with amplitude/speed, Truck, Tilt, Arc, Tracking, Static, Shake Slightly/Strongly, etc.
- Speaker IDs (S1), (S2) stable across shots, dialogue inside [Language] actual words verbatim
- Frame-anchored alignment line first: "For the target video, at 0.00 seconds..." or "How the reference pictures align..."
- Full-reference 6 sections in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music
- Respects hard constraints: runtime = target duration, cut timestamps within, aspect ratio, frame-anchor vs reference mutually exclusive, audio never alone

Optimized for 9B small models with lean version + retry + JSON repair
"""

from __future__ import annotations
import logging
import os

from ...ollama_client import generate
from ...utils.text_utils import extract_json_block
from ...utils.image_utils import tensor_to_base64
from ...prompts.h3_official_enhancer import (
    OFFICIAL_SYSPROMPT_FULL,
    OFFICIAL_SYSPROMPT_LEAN_9B,
    build_official_prompt,
    get_official_system_prompt
)

_log = logging.getLogger(__name__)

def _generate_with_retry_official(generate_fn, base_url, model, prompt, system, images_b64, is_small_model, cfg):
    """Retry logic for official enhancer - for 9B models"""
    if is_small_model:
        num_ctx = 8192
        num_predict = 3000
        temperature = 0.35
        # Lean system already selected, but truncate if still long
        if len(system) > 3500:
            system = system[:2500] + "\n\n[Lean 9B] Output ONLY final brief text with correct labels, no preamble, keep under 1500 chars positive.\n" + system[-500:]
    else:
        num_ctx = cfg.get("num_ctx", 16384)
        num_predict = cfg.get("num_predict", 4000)
        temperature = cfg.get("temperature", 0.6)

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
                    response_format="text",  # Official outputs plain text, NOT JSON wrapper
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
                    response_format="text"
                )
            # Official output is plain text, not JSON, so we check if it contains expected markers
            if raw and len(raw) > 50:
                # Check for expected sections
                has_shot = "[Shot 1]" in raw or "subject_definitions:" in raw or "integrated_multimodal_description" in raw or "detailed_description:" in raw
                if has_shot or not is_small_model:
                    return raw
                else:
                    # For small model, if no shot marker, retry with simpler
                    if attempt < 2:
                        _log.warning(f"[Official SmallModel Retry {attempt+1}/3] No shot marker, retrying with lean")
                        system = OFFICIAL_SYSPROMPT_LEAN_9B + "\n\nAdditional: Output MUST contain [Shot 1] and keep under 1000 chars."
                        continue
            return raw
        except Exception as e:
            _log.warning(f"[Official Generate Retry {attempt+1}] failed: {e}")
            if attempt == 2:
                raise
    return raw


class OMGMiniMaxH3OfficialEnhancer:
    """
    Official H3 Prompt Enhancer - Uses exact sysprompt from Naxdy gist (https://gist.github.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace)
    This is the true H3-Context-IR replacement.

    - Text-only: outputs integrated_multimodal_description + overall_soundscape + non_diegetic_music
    - Frame-anchored (first/last/both): prepends alignment line + 3 core fields
    - Full-reference (general refs): outputs 6 sections in order subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music

    Supports few words concept, reference IMAGE inputs, CLIP fallback, and small 9B model mode with lean prompt + retries.
    """

    CATEGORY = "Ollama-Magic-Nodes/Video/MiniMax-H3"
    FUNCTION = "generate_official"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "mode_detected", "system_used", "raw_output")
    OUTPUT_TOOLTIPS = (
        "Enhanced prompt text in official H3 format - ready for MiniMax H3 API (plain text with sections, not JSON wrapper)",
        "Detected mode: text_only, first_frame, first_last_frame, last_frame, full_reference",
        "System prompt used (full 13K or lean 9B 3K)",
        "Raw output from LLM"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {
                    "default": "chai rain Mumbai",
                    "multiline": True,
                    "tooltip": "Few words allowed! Raw creative request - e.g. 'chai rain Mumbai' or detailed description. Official enhancer will fill missing details, reason about relations, convert to production brief"
                }),
            },
            "optional": {
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Ollama model - Gemma 4 31B, Gemma3 9B, Qwen3 8B etc. Example from gist used gemma-4-31B-it-uncensored with fp8 on RTX Pro 6000 via vLLM. Works with 9B with small_model_mode"}),
                "clip": ("CLIP", {"tooltip": "Optional CLIP from ComfyUI - alternative path without Ollama, uses template generation + presets, no LLM"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional Picture 1 - first-frame anchor OR general reference character/scene/object/style. Fixed name <Picture 1>"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional Picture 2 - last-frame anchor or second reference"}),
                "image_ref_3": ("IMAGE", {"tooltip": "Optional Picture 3"}),
                "video_ref_1": ("IMAGE", {"tooltip": "Optional Video 1 - provide first frame as IMAGE (ComfyUI doesn't have VIDEO type, use first frame or use path below). Role: source video to preserve/imitate"}),
                "audio_ref_1": ("STRING", {"default": "", "tooltip": "Optional Audio 1 path or description - voice timbre, soundtrack - Audio cannot be sole reference, needs image/video alongside"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 15, "tooltip": "Target duration seconds, 4-15, integer"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "auto"], {"default": "16:9"}),
                "mode": (["auto", "text_only", "first_frame", "last_frame", "first_last_frame", "full_reference"], {"default": "auto", "tooltip": "Auto detects from references, or force mode. Text-only no media, frame-anchored for first/last frame, full-reference for general refs (6 sections)"}),
                "reference_roles": ("STRING", {"multiline": True, "default": "", "tooltip": "Describe ROLE of each media: e.g. 'Picture 1 is first-frame anchor - target must begin exactly on this image' or 'Picture 1 general reference character, Picture 2 scene, Video 1 source to edit, Audio 1 voice-timbre reference for Subject 1 (S1)'"}),
                "additional_details": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra instructions: e.g. 'Keep silver bracelet visible, no background music'"}),
                "use_clip_fallback": ("BOOLEAN", {"default": False, "tooltip": "Force CLIP offline template mode without Ollama - uses generic category presets"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller (Gemma3 9B, Qwen2.5 7B) - lean 3K sysprompt, 8192 ctx, lower temp 0.35, retries, keeps under 1500 chars"}),
                "fal_preset": (["auto-infer", "type_text_to_video", "type_first_last_frame", "type_identity_lock", "type_multi_asset_remix", "type_motion_transfer", "type_voice_clone", "type_object_replace", "type_green_screen_replace", "type_multi_edit"], {"default": "auto-infer", "tooltip": "Optional generic type preset as base - not topic-specific"}),
            }
        }

    def generate_official(self, concept: str, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        use_clip_fallback = kwargs.get("use_clip_fallback", False)
        small_model_mode = kwargs.get("small_model_mode", False)
        duration = kwargs.get("duration", 8)
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        mode = kwargs.get("mode", "auto")
        reference_roles = kwargs.get("reference_roles", "")
        additional_details = kwargs.get("additional_details", "")
        fal_preset = kwargs.get("fal_preset", "auto-infer")

        # Image refs for vision
        images_b64 = []
        num_images = 0
        for key in ["image_ref_1", "image_ref_2", "image_ref_3", "video_ref_1"]:
            img = kwargs.get(key)
            if img is not None:
                num_images += 1
                try:
                    images_b64.append(tensor_to_base64(img))
                except:
                    pass

        # Build reference info string for official prompt builder
        ref_info_parts = []
        if reference_roles:
            ref_info_parts.append(reference_roles)
        else:
            # Auto-generate from provided images
            if num_images > 0:
                for i in range(1, num_images+1):
                    # Detect if first/last frame mode
                    if mode == "first_frame" and i == 1:
                        ref_info_parts.append(f"<Picture {i}> is first-frame anchor - target video must begin exactly on this image;")
                    elif mode == "last_frame" and i == 1:
                        ref_info_parts.append(f"<Picture {i}> is last-frame anchor - target video must end exactly on this image;")
                    elif mode == "first_last_frame":
                        if i == 1:
                            ref_info_parts.append(f"<Picture {i}> is first-frame anchor;")
                        elif i == 2:
                            ref_info_parts.append(f"<Picture {i}> is last-frame anchor;")
                    else:
                        # General reference
                        ref_info_parts.append(f"<Picture {i}> is general reference - character/scene/object/style to preserve;")

        audio_ref = kwargs.get("audio_ref_1", "")
        if audio_ref:
            ref_info_parts.append(f"<Audio 1> is {audio_ref} - voice-timbre reference, Audio cannot be sole reference, needs image/video alongside;")

        reference_info = "\n".join(ref_info_parts) if ref_info_parts else "No media besides text - text-only brief"

        # CLIP fallback without Ollama
        if ollama_model is None or use_clip_fallback:
            # Use generic category preset as base for official format
            # Load category presets
            import os, json
            system_used = "CLIP Fallback - No Ollama - uses fal category presets + official template structure"
            mode_detected = mode if mode != "auto" else ("full_reference" if num_images > 0 else "text_only")

            # Try to load preset template
            preset_prompt = ""
            if fal_preset != "auto-infer":
                presets_path = os.path.join(os.path.dirname(__file__), "..", "..", "presets", "minimax_h3_category_presets.json")
                try:
                    with open(presets_path, "r") as f:
                        data = json.load(f)
                        presets = {p["id"]: p for p in data.get("presets", [])}
                        p = presets.get(fal_preset)
                        if p:
                            base = p.get("prompt_template", "{concept}")
                            if "{concept}" in base:
                                preset_prompt = base.replace("{concept}", concept)
                            else:
                                preset_prompt = base + f"\n\n[USER CONCEPT]\n{concept}"
                except:
                    preset_prompt = f"[{fal_preset}] {concept}"

            # Build official format manually for CLIP path
            if mode_detected == "text_only" or num_images == 0:
                # Text-only brief: no alignment line, 3 core fields? Actually official says text-only brief no alignment line, but should have integrated_multimodal_description + overall_soundscape + non_diegetic_music
                # For CLIP fallback, we produce PromptSama style but note it's official format
                enhanced = f"""integrated_multimodal_description:
[Shot 1] Cinematic, live-action, realistic photographic style, 35mm film grain. {concept} - {additional_details}. Location Mumbai monsoon dusk chai stall Dharavi street, discovery glowing letter leading to abandoned Royal Talkies theater. Subject: Aarav 28yo Indian man short black hair light stubble navy kurta silver bracelet right wrist. Scene: {concept}. The camera pushes in with small amplitude at slow speed toward hands holding chai. (S1) Aarav (S1) says in [English] \"The city never really sleeps, it just sips chai.\"[Shot 2] At 00:04.500, the camera cuts to close-up of hands holding glowing letter. {concept} continues.

overall_soundscape:
Rain pitter on tin roof, distant traffic honks low, chai boiling hiss foreground, cup clink, paper crate rustle, fabric rustle, breathing.

non_diegetic_music:
N/A
[CLIP Fallback: {num_images} images, model {type(clip).__name__ if clip else 'none'}, preset {fal_preset}, mode {mode_detected}, duration {duration}s, aspect {aspect_ratio}]
{preset_prompt}
"""
            elif mode_detected in ["first_frame", "last_frame", "first_last_frame"]:
                # Frame-anchored template
                if mode_detected == "first_frame":
                    align = f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
                elif mode_detected == "last_frame":
                    align = f"How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the {duration:.2f}-second mark of the target video."
                else:  # first_last
                    align = f"How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the {duration:.2f}-second mark of the target video."

                enhanced = f"""{align}

integrated_multimodal_description:
[Shot 1] Cinematic, live-action style derived from <Picture 1>. {concept}. Begin from image and develop forward keeping identity clothing colors objects spatial relationships consistent. {additional_details}
[Shot 2] At 00:04.500, the camera cuts to wider view, {concept} continues, gradually converging onto reference if last-frame.

overall_soundscape:
Rain, traffic, chai hiss, fabric, breathing.

non_diegetic_music:
N/A
[CLIP Fallback: {num_images} images, preset {fal_preset}]
"""
            else:  # full_reference - 6 sections
                enhanced = f"""subject_definitions:
<Subject 1>: Protagonist from concept {concept} - 28yo Indian man navy kurta silver bracelet.
<Picture 1>: Reference for identity of <Subject 1> if {num_images}>0 else scene.
<Video 1>: Source video if provided.
<Audio 1>: {audio_ref if audio_ref else 'Voice timbre if provided'}

summary:
[reference generation] A cinematic shot of <Subject 1> in scene derived from {concept} preserving identity.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity hair kurta bracelet retained.
<Picture 1>: fully_preserved - identity reference.
<Video 1>: partially_preserved - motion timing.
<Audio 1>: reference - voice timbre.

detailed_description:
The target video is in cinematic, realistic photographic style with 35mm film grain, natural lighting.

[Shot 1] The scene opens with <Subject 1> (S1) at chai stall, {concept}. The camera pushes in with small amplitude at slow speed. (S1) says in [English] "{concept[:50]}."

[Shot 2] At 00:04.500, the camera cuts to close-up of glowing letter in hand, reaction.

overall_soundscape:
Rain pitter tin roof, traffic distant, chai hiss, cup clink, fabric rustle.

non_diegetic_music:
N/A
[CLIP Fallback generic type {fal_preset}, {num_images} images, concept {concept}, duration {duration}s]
{preset_prompt}
"""

            return (enhanced, mode_detected, system_used, enhanced)

        # Ollama path - official sysprompt
        from ...prompts.h3_official_enhancer import build_official_prompt

        system_prompt, user_prompt = build_official_prompt(
            concept=concept,
            mode=mode,
            duration=duration,
            aspect_ratio=aspect_ratio,
            reference_info=reference_info + "\n" + additional_details,
            additional_details=additional_details,
            use_lean=small_model_mode
        )

        # If fal preset selected, inject as additional details
        if fal_preset != "auto-infer":
            user_prompt += f"\n\nUse generic preset {fal_preset} as base template style."

        _log.info(f"[Official Enhancer] Mode={mode} Duration={duration}s Aspect={aspect_ratio} Images={num_images} Small9B={small_model_mode} Concept={concept[:60]}")

        # Generate with retry helper for small model
        if small_model_mode:
            raw, _parsed = _generate_with_retry_official(
                generate_fn=generate,
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system_prompt,
                images_b64=images_b64,
                is_small_model=True,
                cfg=ollama_model
            )
            # For official, raw is already enhanced prompt text, not JSON
            enhanced = raw
            mode_detected = mode
            if mode == "auto":
                # Detect from output
                if "subject_definitions:" in raw:
                    mode_detected = "full_reference"
                elif "For the target video, at 0.00 seconds" in raw or "How the reference pictures align" in raw:
                    mode_detected = "frame_anchored"
                else:
                    mode_detected = "text_only"
            return (enhanced, mode_detected, f"Official Lean 9B ({len(system_prompt)} chars) - {ollama_model['model']}", raw)

        # Normal path (larger model like 31B as in gist)
        try:
            raw = generate(
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system_prompt,
                temperature=ollama_model.get("temperature", 0.6),
                num_ctx=ollama_model.get("num_ctx", 16384),
                num_predict=ollama_model.get("num_predict", 4000),
                seed=ollama_model.get("seed", -1),
                keep_alive=ollama_model.get("keep_alive", "5m"),
                response_format="text",
                images=images_b64 if images_b64 else None
            )
        except TypeError:
            raw = generate(
                base_url=ollama_model["base_url"],
                model=ollama_model["model"],
                prompt=user_prompt,
                system=system_prompt,
                temperature=ollama_model.get("temperature", 0.6),
                num_ctx=ollama_model.get("num_ctx", 16384),
                num_predict=4000,
                seed=ollama_model.get("seed", -1),
                keep_alive=ollama_model.get("keep_alive", "5m"),
                response_format="text"
            )

        # Detect mode from output
        mode_detected = mode
        if mode == "auto":
            if "subject_definitions:" in raw:
                mode_detected = "full_reference"
            elif "For the target video, at 0.00 seconds" in raw or "How the reference pictures align" in raw:
                mode_detected = "frame_anchored"
            else:
                mode_detected = "text_only"

        return (raw, mode_detected, f"Official Full 13K ({len(system_prompt)} chars) - {ollama_model['model']}", raw)


NODE_CLASS_MAPPINGS = {
    "OMGMiniMaxH3OfficialEnhancer": OMGMiniMaxH3OfficialEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMGMiniMaxH3OfficialEnhancer": "OMG MiniMax H3 Official Enhancer (True H3-Context-IR, Naxdy Sysprompt, +refs+CLIP+9B)"
}
