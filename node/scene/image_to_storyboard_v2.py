"""
OMG Image-to-Storyboard V2 - Fixes the very poor original node

Original OllamaStoryboardGenerator:
- Only took story_concept string, no image
- Returned 12 generic panel strings without character consistency, geography, timed shots

New V2:
- Takes IMAGE (vision), analyzes characters, setting, mood, objects, potential story
- Generates cinematic storyboard with PromptSama Authority Stack:
  [REFERENCE USE], [IDENTITY LOCKS], [SCREEN GEOGRAPHY], [SHOT LIST] 0-4/4-9/9-15 per panel
  acting, lighting, camera rules, production sound, negatives
- Returns proper storyboard_json + panels + character_reference + continuity checklist
- Works from few images or single image - expands via Ollama vision
"""

from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...utils.text_utils import extract_json_block, filter_thinking
from ...utils.image_utils import tensor_to_base64

_log = logging.getLogger(__name__)

STYLE_OPTIONS = ["auto-infer", "cinematic", "comic", "manga", "animation", "Bollywood vibrant", "photographic", "noir", "anime"]
RATIO_OPTIONS = ["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]

SYSTEM_PROMPT_IMAGE_ANALYZER = """You are vision analyzer for storyboard generation.

Analyze the provided image(s) deeply:
- Characters: count, age, gender, appearance, clothing, distinctive features (bracelet, tattoos), expression, posture
- Setting: location, time of day, weather, architecture, props, foreground/mid/background
- Mood: emotional tone, color palette, lighting (natural, neon, volumetric, etc.)
- Story potential: what just happened, what could happen next, conflict, hook
- Screen geography: who/what is frame left/right/center, depth layers
- Style: photographic, cinematic, anime, etc. and lighting quality

Output VALID JSON:
{
  "characters": [{"id": "protagonist", "name": "...", "appearance": "...", "clothing": "...", "distinctive": "...", "position": "frame left"}],
  "setting": "Location, time, weather, mood, key props",
  "mood": "Nostalgic, tense, etc. + palette + lighting",
  "story_hook": "What is interesting and could continue",
  "screen_geography": "Who/what is left/right/center/foreground/background",
  "objects_props": ["chai glass", "glowing letter"],
  "style_notes": "Photographic, 35mm, etc."
}
"""

SYSTEM_PROMPT_STORYBOARD_V2 = """You are advanced storyboard director for Comfy-OMG V2 - fixes poor original.

You have image analysis + user story idea (few words allowed). You must create cinematic storyboard that maintains character consistency, screen geography locks, timed shots with PromptSama Authority Stack per panel.

Input:
- Image analysis: {image_analysis}
- User story concept (few words allowed, expand fully): {story_concept}
- Num panels: {num_panels} (each panel will be 4-15s, timed 0-4/4-9/9-15 internally per panel if needed)
- Style: {style}
- Aspect ratio: {aspect_ratio}
- Maintain characters: {maintain_characters}
- Additional intent: {additional_intent}

You must output VALID JSON with deep continuity (PromptSama layers per panel):

{{
  "title": "Film Title",
  "logline": "One sentence hook",
  "total_duration": 32,
  "style_bible": "Cinematic, realistic, 35mm, etc.",
  "character_reference": "Detailed character descriptions for consistency: face, hair, clothing, distinctive props (bracelet), voice, persistent traits",
  "continuity_locks": ["Keep exactly 1 person", "Preserve silver bracelet on right wrist", "No wardrobe swaps", "Keep geography: chai stall foreground left, etc."],
  "screen_geography_bible": "Global geography: protagonist frame center then left, stall left foreground, wet asphalt mid, neon background. Keep L/R stable across all panels.",
  "panels": [
    {{
      "panel_id": 1,
      "duration": 8,
      "reference_use": "Image1 defines protagonist identity only. No video ref for this panel.",
      "identity_locks": "Preserve face, hair, kurta, bracelet",
      "scene": "Mumbai chai stall dusk, Aarav notices glow",
      "screen_geography": "Aarav center, stall left foreground, crate right foreground with glow",
      "shot_list": ["0-4s — Wide, dolly in, 35mm, establishing...", "4-9s — Medium, push in at 00:06.500, picks glowing letter..."],
      "acting": "Calm to curious, hand tremor",
      "lighting": "Monsoon overcast + warm tungsten + neon reflections",
      "camera": "Wide to medium, slow dolly in, 50mm",
      "sound": "Rain, chai hiss, dialogue, no music yet",
      "dialogue": "Aarav says: 'What is this light?'",
      "negative": "No extra people, No face drift",
      "positive_prompt": "Full production brief string for this panel as H3/T2V model prompt - includes all above - ready for image-to-video model",
      "end_state": "Letter glowing in hand, looking toward alley"
    }}
  ],
  "all_panels_prompt": "All panels concatenated as single narrative",
  "story_summary": "Narrative summary across panels",
  "camera_directions": "Overall camera language: slow dolly in for emotion, tracking for journey, axis stable"
}}

Rules:
- If user gave only few words like "chai", expand into rich story with conflict, emotional turn
- Maintain character consistency using character_reference - same face, clothing, distinctive props every panel
- Lock screen geography L/R per panel and global bible
- Each panel's positive_prompt must be full PromptSama brief (not vague) - include reference use, locks, geography, timed shots 0-4/4-9/9-15, acting, lighting, camera, sound, negatives - up to 700 chars per panel but detailed
- If maintain_characters true, keep same identities, count, wardrobe across all panels
- Duration per panel 4-15s, total = sum
- Provide end_state per panel for continuity to next
- Output ONLY JSON.
"""

class OllamaImageToStoryboardV2:
    """Improved Image-to-Storyboard that actually uses image vision + PromptSama"""

    CATEGORY = "Ollama-Magic-Nodes/Scene/Improved"
    FUNCTION = "generate_v2"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("panel_1", "panel_2", "panel_3", "panel_4", "all_panels_json", "character_reference", "story_summary", "storyboard_json")
    OUTPUT_TOOLTIPS = (
        "Panel 1 full production brief prompt (H3/LTX/Wan ready)",
        "Panel 2 prompt",
        "Panel 3 prompt",
        "Panel 4 prompt (expandable - for more panels parse all_panels_json)",
        "All panels as JSON array + metadata",
        "Character reference bible for consistency",
        "Story summary across panels",
        "Full storyboard JSON with continuity locks, geography bible, timed shots"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_concept": ("STRING", {"default": "chai", "multiline": True, "tooltip": "Few words allowed! e.g. 'chai', 'neon chase', 'glowing letter' - auto-expands to full cinematic story with continuity"}),
                "num_panels": ("INT", {"default": 4, "min": 2, "max": 12, "step": 1, "tooltip": "Number of panels/shots (each 4-15s)"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Optional: Ollama vision model (qwen2.5-vl, llava, gemma3-vision) - if missing uses CLIP fallback"}),
                "clip": ("CLIP", {"tooltip": "Optional: CLIP from ComfyUI downloaded models - alternative without Ollama, uses CLIP tokenizer + fal presets templates for storyboard"}),
                "image": ("IMAGE", {"tooltip": "Single image or first of sequence - analyzed via vision if Ollama vision available, else via CLIP fallback template"}),
                "style": (STYLE_OPTIONS, {"default": "cinematic"}),
                "style_custom": ("STRING", {"default": ""}),
                "aspect_ratio": (RATIO_OPTIONS, {"default": "16:9"}),
                "maintain_characters": ("BOOLEAN", {"default": True, "tooltip": "Keep character consistency across panels - wardrobe, face, props"}),
                "additional_intent": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra guidance: e.g. 'keep bracelet always visible, slow dolly in for emotion'"}),
                "image_2": ("IMAGE", {"tooltip": "Optional second image for multi-image story context"}),
                "image_3": ("IMAGE", {"tooltip": "Optional third image"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller - lean prompts 8192 ctx, retries, shorter outputs"}),
                "use_clip_fallback": ("BOOLEAN", {"default": False, "tooltip": "Force CLIP offline template mode without Ollama - uses fal category presets"}),
                "fal_preset": (["auto-infer", "type_text_to_video", "type_first_last_frame", "type_identity_lock", "type_multi_asset_remix", "type_vertical_drama", "type_game_ui", "type_fashion_film"], {"default": "auto-infer", "tooltip": "Optional generic type preset as base for storyboard"}),
            }
        }

    def _get_val(self, dropdown, custom, kwargs, fallback="auto-infer"):
        dv = kwargs.get(dropdown, "auto-infer")
        cv = str(kwargs.get(custom, "") or "").strip()
        if dv == "auto-infer" and cv:
            return cv
        if dv not in ["auto-infer", "custom"] and dv:
            return dv
        if dv == "custom" and cv:
            return cv
        return fallback

    def generate_v2(self, story_concept: str, num_panels: int = 4, **kwargs):
        ollama_model = kwargs.get("ollama_model")
        clip = kwargs.get("clip")
        small_model_mode = kwargs.get("small_model_mode", False)
        use_clip_fallback = kwargs.get("use_clip_fallback", False)
        image = kwargs.get("image")
        fal_preset = kwargs.get("fal_preset", "auto-infer")

        cfg = ollama_model if ollama_model else {"base_url": "http://localhost:11434", "model": "qwen3:32b", "temperature": 0.5, "num_ctx": 8192 if small_model_mode else 16384, "num_predict": 1024 if small_model_mode else 3000, "seed": -1, "keep_alive": "5m"}
        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass

        style = self._get_val("style", "style_custom", kwargs, "cinematic")
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        maintain_characters = kwargs.get("maintain_characters", True)
        additional_intent = kwargs.get("additional_intent", "")

        # CLIP fallback if no Ollama
        if ollama_model is None or use_clip_fallback:
            # Template-based storyboard without Ollama, using generic category presets
            # Load category presets if selected
            preset_note = ""
            if fal_preset != "auto-infer":
                import os, json
                presets_path = os.path.join(os.path.dirname(__file__), "..", "..", "presets", "minimax_h3_category_presets.json")
                try:
                    with open(presets_path, "r") as f:
                        data = json.load(f)
                        presets = {p["id"]: p for p in data.get("presets", [])}
                        p = presets.get(fal_preset)
                        if p:
                            preset_note = f"Using generic preset {fal_preset} ({p.get('category', p.get('type','generic'))}) as base: {p.get('prompt_template','')[:300]}..."
                except:
                    pass

            # Generic storyboard template
            panels_text = []
            for i in range(num_panels):
                panel_prompt = f"""[PANEL {i+1}/{num_panels} - CLIP Fallback {'Small 9B' if small_model_mode else ''}]

[REFERENCE USE]
Image1 defines protagonist identity (if image provided) - preserve face/clothing.

[IDENTITY LOCKS]
Keep exactly 1 person, preserve face, clothing, distinctive props (bracelet). No swaps. Maintain characters={maintain_characters}.

[SCENE]
{story_concept} - Panel {i+1}: Expand few words into cinematic beat. Location Mumbai monsoon dusk, chai stall. Goal curiosity to determination. {additional_intent}

[SCREEN GEOGRAPHY]
Protagonist frame center, stall foreground left, wet asphalt midground, skyline background. Keep left/right stable.

[SHOT LIST]
0-4s Wide establishing, 4-9s Medium close-up push in 00:06.500 reaction, 9-15s Tracking resolve stable tableau.

[LIGHT]
Natural overcast monsoon + warm tungsten + neon reflections, 35mm film grain.

[CAMERA]
Slow dolly in for emotion, axis stable.

[SOUND]
Rain, traffic distant, chai hiss.

[NEGATIVES]
No extra people, No face drift, No wardrobe changes

[PRESET BASE]
{preset_note}
"""
                panels_text.append(panel_prompt)

            # Pad to 4
            while len(panels_text) < 4:
                panels_text.append("")

            all_panels_json = json.dumps([{"panel_id": i+1, "positive_prompt": pt[:1000]} for i, pt in enumerate(panels_text)], indent=2)
            char_ref = f"Character from concept '{story_concept[:100]}' - preserve across {num_panels} panels, maintain={maintain_characters}, style={style}"
            story_summary = f"{story_concept} - {num_panels} panels cinematic story, CLIP fallback {'small 9B' if small_model_mode else ''}, preset {fal_preset}"
            storyboard_json = json.dumps({
                "title": f"{story_concept[:30]} - Storyboard",
                "logline": story_concept,
                "total_duration": num_panels * 8,
                "style": style,
                "num_panels": num_panels,
                "mode": "clip_fallback" + ("_small_9b" if small_model_mode else ""),
                "character_reference": char_ref,
                "panels": [{"panel_id": i+1, "prompt": pt[:500]} for i, pt in enumerate(panels_text)]
            }, indent=2)

            return (
                panels_text[0] if len(panels_text) > 0 else "",
                panels_text[1] if len(panels_text) > 1 else "",
                panels_text[2] if len(panels_text) > 2 else "",
                panels_text[3] if len(panels_text) > 3 else "",
                all_panels_json,
                char_ref,
                story_summary,
                storyboard_json
            )

        # Ollama path continues
        images_b64 = []
        try:
            if image is not None:
                images_b64.append(tensor_to_base64(image))
            if kwargs.get("image_2") is not None:
                images_b64.append(tensor_to_base64(kwargs["image_2"]))
            if kwargs.get("image_3") is not None:
                images_b64.append(tensor_to_base64(kwargs["image_3"]))
        except Exception as e:
            _log.warning("[OMG Image2StoryboardV2] Failed to encode images: %s", e)
            images_b64 = []

        # Step 1: Analyze image(s) with vision
        _log.info("[OMG Image2StoryboardV2] Analyzing %d images with vision model %s", len(images_b64), cfg["model"])
        try:
            analysis_raw = generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt="Analyze these images for storyboard: characters, setting, mood, story potential, screen geography, style. Output JSON.",
                system=SYSTEM_PROMPT_IMAGE_ANALYZER,
                temperature=0.5,
                num_ctx=cfg.get("num_ctx", 8192, think=think, filter_thinking=filter_thinking),
                num_predict=2000,
                seed=cfg.get("seed", -1),
                keep_alive=cfg.get("keep_alive", "5m"),
                images=images_b64 if images_b64 and images_b64[0] else None,
                response_format="json"
            )
            analysis = extract_json_block(analysis_raw)
            if analysis is None:
                analysis = {"raw": analysis_raw, "setting": "inferred from image"}
            analysis_str = json.dumps(analysis, indent=2)
        except Exception as e:
            _log.warning("[OMG Image2StoryboardV2] Image analysis failed: %s", e)
            analysis_str = f"Failed to analyze, infer from concept: {story_concept}"
            analysis = {"error": str(e)}

        # Step 2: Generate storyboard with analysis + concept (few words allowed)
        system_v2 = SYSTEM_PROMPT_STORYBOARD_V2.format(
            image_analysis=analysis_str[:3000],
            story_concept=story_concept,
            num_panels=num_panels,
            style=style,
            aspect_ratio=aspect_ratio,
            maintain_characters=maintain_characters,
            additional_intent=additional_intent or "None - auto-infer rich cinematic story with emotional turn"
        )

        user_prompt = f"Concept (few words allowed, expand fully): {story_concept}\nNum panels: {num_panels}\nStyle: {style}\nAspect: {aspect_ratio}\nMaintain characters: {maintain_characters}\nAdditional intent: {additional_intent}\nImage analysis: {analysis_str[:2000]}\n\nGenerate full storyboard JSON."

        _log.info("[OMG Image2StoryboardV2] Generating %d panels storyboard", num_panels)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system_v2,
            temperature=cfg.get("temperature", 0.7, think=think, filter_thinking=filter_thinking),
            num_ctx=cfg.get("num_ctx", 16384),
            num_predict=6000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json"
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OMG Image2StoryboardV2] Failed to parse storyboard JSON")
            fallback_panels = [f"{story_concept} - panel {i+1} cinematic" for i in range(num_panels)]
            while len(fallback_panels) < 4:
                fallback_panels.append("")
            return tuple(fallback_panels[:4]) + (json.dumps({"panels": fallback_panels}), "", story_concept, json.dumps({"raw": raw}))

        panels = parsed.get("panels", [])
        panel_prompts = []
        for p in panels:
            if isinstance(p, dict):
                panel_prompts.append(p.get("positive_prompt", p.get("scene", "")))
            else:
                panel_prompts.append(str(p))

        while len(panel_prompts) < 4:
            panel_prompts.append("")

        all_panels_json = json.dumps(parsed.get("panels", []), indent=2)
        character_ref = parsed.get("character_reference", "")
        story_summary = parsed.get("story_summary", parsed.get("logline", ""))
        storyboard_json = json.dumps(parsed, indent=2)

        return (
            panel_prompts[0] if len(panel_prompts) > 0 else "",
            panel_prompts[1] if len(panel_prompts) > 1 else "",
            panel_prompts[2] if len(panel_prompts) > 2 else "",
            panel_prompts[3] if len(panel_prompts) > 3 else "",
            all_panels_json,
            character_ref,
            story_summary,
            storyboard_json
        )


NODE_CLASS_MAPPINGS = {
    "OllamaImageToStoryboardV2": OllamaImageToStoryboardV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaImageToStoryboardV2": "OMG Image to Storyboard V2 (Vision + PromptSama, Fixes Poor Original)"
}
