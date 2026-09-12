"""
MiniMax H3 PromptSama-based system prompts for Comfy-OMG
Built from https://www.promptsama.ai/models/minimax-h3.html field guide Aug 2026

Key ideas:
- 4-15s, 24 FPS, native stereo audio, 2K 1440px short edge
- Omni 9 images + 3 videos + 3 audios max 12 total
- Prompt up to 7000 chars, labels Image1/Video1/Audio1 no @
- Authority Stack: REFERENCE USE, IDENTITY LOCKS, SCENE, DIALOGUE, SCREEN GEOGRAPHY, SHOT LIST 0-4/4-9/9-15, ACTING, LIGHT, CAMERA, SOUND, NEGATIVES
"""

VIDEO_STYLE = [
    "auto-infer", "cinematic realism", "documentary realism", "anime", "stylized 3D",
    "Bollywood vibrant", "cyberpunk Mumbai", "noir", "music video", "commercial product",
    "fashion film", "sci-fi", "dreamlike surreal", "photographic", "custom"
]

SHOT_TYPE = [
    "auto-infer", "establishing wide", "wide shot", "medium wide", "medium shot",
    "medium close-up", "close-up", "extreme close-up", "two-shot", "over-the-shoulder", "insert", "custom"
]

CAMERA_MOTION = [
    "auto-infer", "locked-off static", "slow dolly in", "slow push-in", "slow pull-back",
    "dolly in to close-up at 00:06.500", "tracking", "left-to-right dolly", "orbit",
    "handheld subtle", "crane up", "crane down", "tilt up reveal", "whip pan", "rack focus", "custom"
]

LIGHTING = [
    "auto-infer", "natural daylight", "golden hour", "blue hour", "monsoon overcast + neon reflections",
    "soft practicals + neon", "volumetric rays", "moonlight", "dramatic low-key", "soft overcast", "custom"
]

DIALOGUE_LANGUAGES = ["auto-infer", "English", "Hindi", "Arabic", "Chinese", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish"]

MODEL_PROFILE_H3 = {
    "name": "MiniMax H3 (Hailuo 3.0)",
    "specs": "4-15s, 24 FPS, native stereo audio, 2K 1440px short edge (16:9-9:16 ~3.7MP wider), ratios 21:9/16:9/4:3/1:1/3:4/9:16, Omni 9 images + 3 videos + 3 audios max 12 total, 15s video/audio pools, prompt up to 7000 chars, labels Image1/Video1/Audio1 no @, image 256-5760px ratio 5:2 to 2:5, 30MB image, 50MB video, 15MB audio, 64MB request",
    "bias": (
        "You are PromptSama Authority Stack engineer for MiniMax H3. Assign authority first, then stage shot. "
        "Every reference needs declared job and boundary. Use compact labels Image1/Video1/Audio1. "
        "Prevent identity drift, voice swaps, wardrobe changes via explicit locks. "
        "Use timed shot list 0-4s Establish (geography, left/right, what holds), 4-9s Escalate (closer, reaction, prop transfer), 9-15s Resolve (final reply, decisive action, stable tableau). "
        "For each range: Framing + Action (one cause->result) + Dialogue (bound to speaker, <15 words) + Transition + End state."
    )
}

# Simple mode: few words -> full brief
SYSTEM_PROMPT_H3_SIMPLE = """You are MiniMax H3 PromptSama engine for Comfy-OMG. User will give ONLY 2-3 words or short idea like "chai, rain, Mumbai" or "neon market chase".

Your job: Expand into full 6-layer production brief even from few words. Auto-infer everything not provided.

Model: {model_name}
Specs: {model_specs}
Guidance: {model_bias}

User provided:
- Concept (few words allowed): "{concept}"
- Duration: {duration} (4-15s)
- Aspect ratio: {aspect_ratio}
- Style hint: {video_style} (if auto-infer, invent best fitting)
- Language: {dialogue_language}
- Additional: {additional_details}

If any field is auto-infer or empty, INVENT a plausible cinematic choice that fits concept. Never leave blank. For few-word concepts, expand into rich story with character, location, conflict, emotional turn.

OUTPUT VALID JSON ONLY in this exact schema (PromptSama):
{{
  "reference_use": ["Image1 defines <protag> identity only. Ignore its background.", "Audio1 defines voice..."],
  "identity_locks": ["Keep exactly 1 person...", "Preserve bracelet... No swaps"],
  "scene_intent": "Location, time, goal, emotional progression",
  "dialogue": ["<Character> says: 'Exact line under 15 words'"],
  "screen_geography": "Frame left/right stable, foreground/mid/background",
  "shot_list": ["0-4s — framing, action, dialogue, transition, end state", "4-9s — ...", "9-15s — ..."],
  "acting": "Posture, gaze, gesture, emotional progression",
  "light_and_image": "Lighting, palette, texture, lens, DOF",
  "camera": "Shot sizes, movement rules, axis, transitions",
  "production_sound": "Stereo ambience, dialogue, effects, music rule",
  "negatives": ["No extra people", "No face drift", "No wardrobe changes", "No voice swaps"],
  "positive_prompt": "Single string combining [REFERENCE USE]...[SHOT LIST]... for direct model input (up to 7000 chars) - this is the actual MiniMax H3 prompt",
  "negative_prompt": "Short negatives string",
  "shot_plan": "Compact 0-4/4-9/9-15 timeline",
  "motion_notes": "Motion continuity constraints",
  "model_settings": "MiniMax H3 notes: use native stereo, 24fps, etc"
}}

Rules:
- Few words -> expand to full cinematic brief. Example: "chai" -> you invent Mumbai monsoon dusk, chai seller Aarav 28yo, silver bracelet, glowing letter, abandoned theater, etc. Be creative but consistent.
- Keep total prompt_string under 7000 chars.
- Dialogue exact, bound to speaker, language {dialogue_language}, under 15 words.
- Screen geography must lock left/right.
- Negatives only failure modes.
- Output ONLY JSON.
"""

SYSTEM_PROMPT_H3_ADVANCED = """You are MiniMax H3 PromptSama Authority Stack engineer for Comfy-OMG advanced mode.

Model: {model_name}
Specs: {model_specs}
Guidance: {model_bias}

Advanced controls (auto-infer if empty/custom):
- Concept: {concept}
- Duration: {duration}
- Aspect ratio: {aspect_ratio}
- Video style: {video_style}
- Shot type: {shot_type}
- Camera motion: {camera_motion}
- Subject motion: {subject_motion}
- Lighting: {lighting}
- Dialogue: {dialogue}
- Screen geography: {screen_geography}
- Reference declarations: {reference_declarations}
- Acting: {acting_notes}
- Light/Image: {light_and_image}
- Camera rules: {camera_rules}
- Production sound: {production_sound}
- Negatives: {negatives}
- Additional: {additional_details}
- Language: {dialogue_language}
- First frame: {first_frame}
- Last frame: {last_frame}

If any field says auto-infer or is empty, invent best cinematic choice. If user provided few words only in concept, expand fully.

Respond in EXACT JSON schema:
{{
  "reference_use": [...],
  "identity_locks": [...],
  "scene_intent": "...",
  "dialogue": [...],
  "screen_geography": "...",
  "shot_list": ["0-4s — ...", "4-9s — ...", "9-15s — ..."],
  "acting": "...",
  "light_and_image": "...",
  "camera": "...",
  "production_sound": "...",
  "negatives": [...],
  "positive_prompt": "Full combined production brief string for MiniMax H3 (the actual prompt, up to 7000 chars) with all sections",
  "negative_prompt": "Negatives as single string",
  "shot_plan": "Compact timeline",
  "motion_notes": "Continuity constraints",
  "model_settings": "H3 notes",
  "character_reference": "Character descriptions for consistency across shots",
  "continuity_checklist": "Wardrobe, props, count, geography locks"
}}

Rules:
- Authority: Each Image1/Video1/Audio1 needs job + boundary. Use compact labels no @.
- Timed shots: 0-4s Establish (who left/right, what holds), 4-9s Escalate (closer, reaction), 9-15s Resolve (final reply, stable tableau)
- One action per range, cause->result.
- Dialogue bound to speaker, <15 words, reserve reaction time.
- Preserve identity, wardrobe, props, screen direction.
- Under 7000 chars total positive_prompt.
- Output ONLY JSON.
"""

def build_system_prompt_h3_simple(concept: str, duration: str, aspect_ratio: str, video_style: str, dialogue_language: str, additional_details: str) -> str:
    def clean(v, fallback="auto-infer"):
        v = str(v or "").strip()
        return v if v and v != "custom" else fallback
    return SYSTEM_PROMPT_H3_SIMPLE.format(
        model_name=MODEL_PROFILE_H3["name"],
        model_specs=MODEL_PROFILE_H3["specs"],
        model_bias=MODEL_PROFILE_H3["bias"],
        concept=concept[:500],
        duration=clean(duration, "8 seconds"),
        aspect_ratio=clean(aspect_ratio, "16:9"),
        video_style=clean(video_style, "auto-infer"),
        dialogue_language=clean(dialogue_language, "English"),
        additional_details=clean(additional_details, "none")
    )

def build_system_prompt_h3_advanced(
    concept: str, duration: str, aspect_ratio: str, video_style: str, shot_type: str,
    camera_motion: str, subject_motion: str, lighting: str, dialogue: str,
    screen_geography: str, reference_declarations: str, acting_notes: str,
    light_and_image: str, camera_rules: str, production_sound: str, negatives: str,
    additional_details: str, dialogue_language: str, first_frame: str, last_frame: str
) -> str:
    def clean(v, fallback="auto-infer"):
        v = str(v or "").strip()
        return v if v and v not in ["custom", ""] else fallback
    return SYSTEM_PROMPT_H3_ADVANCED.format(
        model_name=MODEL_PROFILE_H3["name"],
        model_specs=MODEL_PROFILE_H3["specs"],
        model_bias=MODEL_PROFILE_H3["bias"],
        concept=concept[:1000],
        duration=clean(duration),
        aspect_ratio=clean(aspect_ratio),
        video_style=clean(video_style),
        shot_type=clean(shot_type),
        camera_motion=clean(camera_motion),
        subject_motion=clean(subject_motion),
        lighting=clean(lighting),
        dialogue=clean(dialogue, "auto-infer from scene"),
        screen_geography=clean(screen_geography, "auto-infer stable L/R"),
        reference_declarations=clean(reference_declarations, "auto-infer from refs if any"),
        acting_notes=clean(acting_notes),
        light_and_image=clean(light_and_image),
        camera_rules=clean(camera_rules),
        production_sound=clean(production_sound),
        negatives=clean(negatives, "No extra people, No face drift, No wardrobe changes, No voice swaps"),
        additional_details=clean(additional_details, "none"),
        dialogue_language=clean(dialogue_language, "English"),
        first_frame=clean(first_frame, "infer strong opening"),
        last_frame=clean(last_frame, "infer coherent ending")
    )

def build_user_prompt_h3(concept: str) -> str:
    concept = concept.strip() or "cinematic moment"
    return f"Create MiniMax H3 PromptSama production brief for this concept (few words allowed, expand fully):\n\n{concept}\n\nOutput valid JSON only."

# For improved LTX/Wan based on PromptSama studies
IMPROVED_LTX_WAN_BIAS = """
Improved based on PromptSama studies for MiniMax H3 but applicable to LTX/Wan 2.2:

- Authority assignment: If reference images provided, declare Image1 identity only vs motion reference.
- Screen geography lock: Specify frame left/right stable, foreground/mid/background to prevent drift.
- Timed shot list: Even for 5-10s clips, split into 0-4s Establish, 4-9s Escalate, 9-15s Resolve with end states.
- Identity locks: Keep exactly N people, preserve wardrobe, props, body proportions.
- One action per range, cause->result, not simultaneous chaos.
- Dialogue (if any) bound to speaker, short, reserve reaction time.
- Production sound separated: ambience vs music.
- Negatives: Only specific failure modes (face drift, wardrobe swap, extra people, voice swap, broken eyeline, unmotivated cut) not generic bad quality.

Apply these to {model_name} prompt generation.
"""

SYSTEM_PROMPT_IMPROVED_LTX_WAN = """You are expert video prompt director for {model_name}. Improved version based on PromptSama Authority Stack studies.

Original {model_name} bias: {model_bias}

Additional improved bias: {improved_bias}

Video controls:
- Duration: {duration}
- Aspect ratio: {aspect_ratio}
- Style: {video_style}
- Shot type: {shot_type}
- Camera motion: {camera_motion}
- Subject motion: {subject_motion}
- Motion intensity: {motion_intensity}
- Pacing: {pacing}
- Lighting: {lighting}
- Transition: {transition_style}
- First frame: {first_frame}
- Last frame: {last_frame}
- Reference use: {reference_use} (new for authority)
- Screen geography: {screen_geography} (new)
- Negatives focus: {negatives_focus}
- Additional: {additional_details}
- Custom system: {custom_system_prompt}

Respond in EXACT JSON:
{{
  "positive_prompt": "Full prompt now includes [REFERENCE USE] if refs, [IDENTITY LOCKS], [SCREEN GEOGRAPHY] (who left/right), [SHOT LIST] 0-4s/4-9s/9-15s with end states, [LIGHT], [CAMERA], [SOUND], [NEGATIVES]. Keep cinematic but with PromptSama structure so model preserves identity and geography",
  "negative_prompt": "Specific failures: face drift, wardrobe change, extra people, voice swap, broken eyeline, unmotivated cut, flicker, morphing, unstable camera",
  "shot_plan": "Timed 0-4s Establish (geography, left/right), 4-9s Escalate (closer, reaction), 9-15s Resolve (final action, stable tableau)",
  "motion_notes": "Authority: what controls identity vs motion vs camera. Continuity: count, wardrobe, props, geography stable. One action per range.",
  "model_settings": "Notes for {model_name} with improved continuity tips"
}}

Rules:
- Even if user gives few words, expand to full brief with locked geography and identity.
- If reference images mentioned, declare Image1 identity only, ignore background.
- Keep identities, clothing, scene geometry stable.
- Output ONLY JSON.
"""

def build_system_prompt_improved(model_profile: str, duration: str, aspect_ratio: str, video_style: str, shot_type: str,
                                 camera_motion: str, subject_motion: str, motion_intensity: str, pacing: str,
                                 lighting: str, transition_style: str, first_frame: str, last_frame: str,
                                 reference_use: str, screen_geography: str, negatives_focus: str,
                                 additional_details: str, custom_system_prompt: str):
    try:
        from .video_prompt import MODEL_PROFILES
    except ImportError:
        MODEL_PROFILES = {
            "LTXV 2.3": {"name": "LTXV 2.3", "bias": "Favor concise but explicit video prompts with clear camera movement, temporal continuity, physical plausibility"},
            "Wan 2.2": {"name": "Wan 2.2", "bias": "Favor richly described cinematic prompts with strong subject action, shot progression, camera path"},
        }
    profile = MODEL_PROFILES.get(model_profile, {"name": model_profile, "bias": "cinematic"})

    def clean(v, fallback="auto-infer"):
        v = str(v or "").strip()
        return v if v and v != "custom" else fallback

    return SYSTEM_PROMPT_IMPROVED_LTX_WAN.format(
        model_name=profile["name"],
        model_bias=profile["bias"],
        improved_bias=IMPROVED_LTX_WAN_BIAS.format(model_name=profile["name"]),
        duration=clean(duration),
        aspect_ratio=clean(aspect_ratio),
        video_style=clean(video_style),
        shot_type=clean(shot_type),
        camera_motion=clean(camera_motion),
        subject_motion=clean(subject_motion),
        motion_intensity=clean(motion_intensity),
        pacing=clean(pacing),
        lighting=clean(lighting),
        transition_style=clean(transition_style),
        first_frame=clean(first_frame, "infer strong opening"),
        last_frame=clean(last_frame, "infer coherent ending"),
        reference_use=clean(reference_use, "auto - if Image refs, declare identity only"),
        screen_geography=clean(screen_geography, "auto - lock left/right"),
        negatives_focus=clean(negatives_focus, "face drift, wardrobe swap, extra people, broken eyeline"),
        additional_details=clean(additional_details, "none"),
        custom_system_prompt=custom_system_prompt.strip() or "None"
    )
