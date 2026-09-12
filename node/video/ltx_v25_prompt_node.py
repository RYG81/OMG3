"""
LTX 2.5 Prompt Generation - Full support for ALL LTX 2.5 prompting types
Per https://docs.ltx.io/open-source-model/usage-guides/prompting-guide
+ https://docs.ltx.io/open-source-model/feature-guides/audio/dub-it-beta
+ https://docs.ltx.io/open-source-model/feature-guides/audio/text-to-audio

Covers:
- Single-shot (1 continuous take) - 4-8 sentences, single flowing paragraph, present tense
- Multi-shot 2-4 shots - ONE chronological paragraph, explicit cuts named in natural language, re-establish shot, identity consistent, audio continuity at every cut
- Screenplay-style with scene headers, character cues, quoted dialogue
- Dialogue: in quotes, language/accent, volume (whisper/mutter/shout/scream), physical emotion cues not abstract labels
- Music: ambient, score, instrumentation, tempo, continues/drops across cuts
- Audio: ambient sound, music, speech, singing, audio continuity stated at every cut
- On-screen text: short prominent, LTX 2.5 improved accuracy but not guaranteed
- Dub-It (IC-LoRA Beta): speech replacement "[Speaker] is speaking [Language/Accent], saying: \"[Dialogue]\"" - validated EN/FR/ES/DE/RU, single speaker, match audio length
- Character definition: age, hairstyle, clothing, distinguishing features, physical cues for emotion
- Camera: follows, tracks, pans across, circles around, tilts, pushes in/pulls back, overhead, handheld, etc.
- Lighting, textures, color palette, atmosphere, film characteristics, scale, pacing, VFX
"""

from __future__ import annotations
import logging
import json

from ...ollama_client import generate
from ...utils.text_utils import extract_json_block, filter_thinking

_log = logging.getLogger(__name__)

# ── Dropdowns ──────────────────────────────────────────────────
VIDEO_STYLE = [
    "auto-infer", "cinematic realism", "documentary realism", "anime",
    "stylized 3D", "concept art motion", "music video", "commercial product",
    "fashion film", "dark fantasy", "sci-fi", "dreamlike surreal", "period drama",
    "film noir", "thriller", "documentary", "animation - stop-motion / 2D / 3D / claymation / hand-drawn",
    "stylized - comic book / cyberpunk / 8-bit pixel / surreal / minimalist / painterly / illustrated",
    "custom"
]

SHOT_TYPE_V25 = [
    "auto-infer",
    "wide establishing shot", "wide shot", "medium wide", "medium shot",
    "medium close-up", "close-up", "extreme close-up", "low-angle shot",
    "high-angle shot", "overhead view", "over-the-shoulder", "point-of-view",
    "two-shot", "insert shot", "macro shot", "custom"
]

CAMERA_MOTION_V25 = [
    "auto-infer",
    "static frame", "locked-off tripod", "slow push-in", "slow pull-back",
    "pan left", "pan right", "tilt up", "tilt down",
    "tracking shot", "dolly left-to-right", "dolly right-to-left",
    "orbit around subject", "handheld subtle sway", "handheld movement",
    "crane up", "crane down", "follows", "circles around", "pushes in / pulls back",
    "overhead view", "over-the-shoulder", "wide establishing shot",
    "film grain", "lens flares", "custom"
]

TRANSITION_V25 = [
    "auto-infer",
    "hard cut", "match cut", "dissolve", "fade-in / fade-out",
    "no transition - single continuous take",
    "custom"
]

LIGHTING_V25 = [
    "auto-infer", "natural daylight", "warm early sun", "golden hour", "blue hour",
    "moonlight", "neon glow", "flickering candles", "dramatic shadows",
    "soft studio", "volumetric rays", "high contrast", "muted", "vibrant",
    "monochromatic", "fog", "rain", "dust", "smoke", "particles",
    "monsoon overcast + neon reflections", "custom"
]

PROMPT_MODE_V25 = [
    "auto-infer (choose best for concept)",
    "single-shot - one continuous take (4-8 sentences, single paragraph, present tense)",
    "multi-shot 2 shots (prefer 2-4, each clear job: establish → detail)",
    "multi-shot 3 shots (wide → medium → close-up, explicit cuts)",
    "multi-shot 4 shots (establish → detail → reaction → close-up)",
    "screenplay-style with scene headers, character cues, quoted dialogue",
    "dub-it speech replacement (IC-LoRA Beta) - single speaker lip-synced",
    "text-to-audio (audio-only, no video)",
]

AUDIO_CONTINUITY_V25 = [
    "auto-infer",
    "music continues across cuts, ambience continues",
    "music drops to low drone at cut, ambience changes",
    "dialogue continues across cut",
    "dialogue drops at cut, only wind/ambience remains",
    "piano score continues across cut, traffic muffled",
    "synth score continues across cut, traffic muffled",
    "music drops to low drone, only wind remains",
    "custom"
]

DIALOGUE_MODE_V25 = [
    "auto-infer",
    "no dialogue",
    "single speaker with quoted dialogue",
    "multi speaker with quoted dialogue and character cues",
    "dub-it speech replacement - [Speaker] is speaking [Language/Accent], saying: \"Dialogue\"",
    "singing",
    "custom"
]

DIALOGUE_LANGUAGE_V25 = [
    "auto-infer", "English", "French", "Spanish", "German", "Russian",
    "Hindi", "Arabic", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "custom"
]

DIALOGUE_VOLUME_V25 = [
    "auto-infer", "whisper", "mutter", "shout", "scream",
    "energetic announcer", "resonant voice with gravitas", "distorted radio-style",
    "robotic monotone", "childlike curiosity", "custom"
]

MUSIC_STYLE_V25 = [
    "auto-infer", "no music", "soft synth music", "piano score", "low drone",
    "electronic pulse", "orchestral", "ambient", "coffeeshop noise",
    "wind and rain", "forest ambience with birds", "custom"
]

ON_SCREEN_TEXT_MODE_V25 = [
    "auto-infer", "no on-screen text",
    "short prominent text (LTX 2.5 improved accuracy but not guaranteed)",
    "title / label / logo - verify throughout clip, add critical titles in post",
    "custom"
]

# ── System Prompt ──────────────────────────────────────────────
LTX25_SYSTEM_PROMPT = """You are an expert LTX 2.5 video prompt engineer following official guides:
- Prompting Guide: https://docs.ltx.io/open-source-model/usage-guides/prompting-guide
- Dub-It IC-LoRA Beta: https://docs.ltx.io/open-source-model/feature-guides/audio/dub-it-beta
- Text-to-Audio: https://docs.ltx.io/open-source-model/feature-guides/audio/text-to-audio

Model: LTX-2.5 (ltx-2.5-fast, ltx-2.5-pro) supports text-to-video, image-to-video, audio-to-video, text-to-audio, portrait and landscape.

=== ALL PROMPTING TYPES LTX 2.5 SUPPORTS ===

1. SINGLE-SHOT (one continuous take):
- Write as SINGLE FLOWING PARAGRAPH, present tense, 4-8 descriptive sentences
- Match detail level to shot scale (close-ups need more detail)
- Describe camera movement relative to subject
- Keep scene focused: few clear characters/actions better than crowded frame
- Keep lighting consistent: one coherent light logic, mixed sources confuse result
- Start simple and layer: core shot then add detail

2. MULTI-SHOT (2-4 shots, LTX-2.5 NEW):
- Write full scene as ONE CHRONOLOGICAL PARAGRAPH (or short sequence), NOT shot list, NOT numbered beats, NOT screenplay sluglines unless cut described in prose
- How differs:
  Camera: One continuous take vs New framing after each cut
  Transitions: Camera moves only vs Name the edit: hard cut, match cut, dissolve
  Continuity: Same space/subjects throughout vs Re-identify subjects when they reappear; say what carries across cut
  Audio: One continuous soundscape vs At every cut say whether music/dialogue/ambience continues or changes
- What to include at EVERY cut (MANDATORY):
  1. Name transition in natural language: "A hard cut transitions to...", "The view cuts to a close-up of...", "A match cut connects...", "The image dissolves into..."
  2. Re-establish new shot: shot scale, camera angle, who/what in frame, lighting if changed
  3. Keep identity consistent: reuse same visual identifiers ("the woman in the yellow raincoat, earlier at the table, now...")
  4. State audio continuity: "the piano score continues across the cut" or "the dialogue drops; only wind remains" or "the synth score continues, traffic muffled" or "music drops to low drone"
- Tips: Prefer 2-4 shots; each shot clear job (establish → detail → reaction, or wide → medium → close-up); chronological Initially..., A moment later..., Simultaneously...; same rules: present tense, physical emotion cues, quoted dialogue, concrete camera language; Avoid conflicting geography or unexplained costume changes unless time/place jump explicitly stated
- Example: "A wide shot frames a rainy city intersection at dusk, neon signs reflecting on wet asphalt. A young woman in a yellow raincoat walks toward camera, gripping a folded newspaper, while cars hiss past behind her. Soft synth music and distant traffic fill the air. A hard cut transitions to a medium close-up of her face under the hood, raindrops catching the neon as she looks off-screen left; the synth score continues across the cut, traffic muffled. She whispers, "He's late." Another hard cut jumps to a low-angle shot of a man's scuffed boots stepping into a puddle at the curb; the music drops to a low drone. He lifts his head into frame — short dark hair, soaked jacket — and smiles toward her off-screen as a bus rumbles past."
- When to stay single-shot: unbroken camera motion, intimate performance, dialogue lip-synced in one framing. For image-to-video from first frame, prefer single continuous take unless intentionally describing cut away

3. SCREENPLAY-STYLE (dialogue, multiple beats, precise timing):
- Use scene headers, character cues, quoted dialogue as sample prompts do
- Keep fundamentals: present tense, physical emotion cues, dialogue in quotation marks
- Example structure from official docs:
  EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST
  The shot opens on...
  Reporter (live): "Thank you, Sylvia..."
  He gestures...
  The camera pans right, slowly revealing...
- Length can be longer if every sentence adds concrete visual/audio detail

4. DIALOGUE & AUDIO (applies to all types):
- Key Elements:
  Establish Shot: cinematography terms, shot scale
  Set Scene: lighting, color palette, surface textures, atmosphere
  Describe Action: core action natural sequence beginning to end
  Define Character(s): age, hairstyle, clothing, distinguishing features. Emotion via PHYSICAL CUES not abstract labels. Not "she is sad" but "her shoulders slump, gaze drops, fingers tighten around newspaper"
  Identify Camera Movement(s): how and when camera moves. Describing how subjects appear AFTER movement helps model complete motion. Use: follows, tracks, pans across, circles around, tilts upward, pushes in / pulls back, overhead view, handheld, over-the-shoulder, wide establishing, static frame
  Describe Audio: ambient sound, music, speech, singing. Place spoken dialogue in QUOTATION MARKS, specify language and accent if needed
- Dialogue volume/style: Whisper, Mutter, Shout, Scream, Energetic announcer, Resonant voice with gravitas, Distorted radio-style, Robotic monotone, Childlike curiosity
- Music: Describe instrumentation, tempo, pulse, rhythm, accent hits, rises, drops, bursts, final resolution, fade
- Ambient: Coffeeshop noise, Wind and rain, Forest ambience with birds
- At every cut for multi-shot: say whether music/dialogue/ambience continues or changes (mandatory)
- Present tense, physical emotion cues, quoted dialogue preserved verbatim

5. ON-SCREEN TEXT (LTX-2.5 improved):
- LTX-2.5 improves short-text accuracy and preserves fine details better than earlier, but exact spelling and consistency across frames not guaranteed
- Keep text short and prominent, verify throughout clip, add critical titles, labels, logos in post
- Place visible text in description, short and prominent: e.g. On-screen text "GET READY TO" appears...

6. DUB-IT (Speech Replacement) IC-LoRA Beta (LTX-2.3 validated, LTX-2.5 in development):
- Prompt Template: [Speaker] is speaking [Language/Accent], saying: "[Dialogue]"
- Example: A woman speaking in Russian saying: "Сегодня отличный день, чтобы протестировать рабочие процессы ComfyUI для дубляжа с использованием LTX."
- Languages validated: English, French, Spanish, German, Russian (LTX-2.3) - LTX-2.5 support in dev
- Requirements: Provide full dialogue text (model does NOT translate), use native script (Cyrillic for Russian, Chinese characters for Mandarin), single speaker (beta does not distinguish multiple speakers)
- Best Practices: Match audio length ~ same timing and syllable length as original dialogue. Slightly longer better than too short. Too long = might skip words. Too short = slow unnatural.
- Technical: Preserves full video except lip region, generates new lip movements synced to prompt, matches original speaker tone, attempts to match delivery/emotion
- Two-stage pipeline: Stage1 low res with source video frames + reference audio as conditioning, Upsample video latent while audio frozen, Stage2 high res
- For ComfyUI: LTXICLoRALoaderModelOnly, LTXAddVideoICLoRAGuide, LTXVSetAudioRefTokens, LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVLatentUpsampler

7. TEXT-TO-AUDIO (Audio-only, no video):
- Workflow: LTXVAudioOnlyModel disables video stream and cross-modal attention, only audio pathway runs
- Use case: generate audio from text description without video. If need audio synchronized to video, use standard T2V which generates jointly
- Files: ltx-2.5-22b-distilled-transformer-bf16.safetensors, gemma4-12b-with-proj-ltx-2.5-bf16.safetensors, gemma4_e2b_it_bf16.safetensors, ltx-2.5-audio-vae-bf16.safetensors
- Steps: Load model → LTXVAudioOnlyModel disables video stream → Encode prompt with Gemma 4 → Prepare empty audio latent for target duration (duration = num-frames ÷ frame-rate, frames follow 1 + multiple of 8) → Sample distilled 8-step schedule at CFG 1 → Decode via audio VAE → PreviewAudio
- Tips: Keep cfg near 1 (distilled bakes guidance), audio length via frame count
- Example prompt: A woman saying: "Oh, what a lovely day we are having!"

8. CHARACTER DEFINITION (for all types):
- Include age, hairstyle, clothing, distinguishing features
- Emotion via physical cues: shoulders slump, gaze drops, fingers tighten, eyes darting, tongue snaps out, lowers head in shame, folding hands, eyes closed too tightly
- Example from docs: The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. "We are one with the pond." All the frogs answer softly: "Ommm..." He smiles faintly. Senior frog vs guilty frog twitching, eyes darting, tongue snaps, lowers head in shame
- Reuse same visual identifiers for recurring subjects in multi-shot: "the woman in the yellow raincoat, earlier at the table, now..."

9. CAMERA, LIGHTING, STYLE TERMS:
- Categories: Animation (Stop-motion, 2D/3D, Claymation, Hand-drawn), Stylized (Comic book, Cyberpunk, 8-bit pixel, Surreal, Minimalist, Painterly, Illustrated), Cinematic (Period drama, Film noir, Fantasy, Epic space opera, Thriller, Modern romance, Experimental, Arthouse, Documentary)
- Lighting: Flickering candles, Neon glow, Natural sunlight, Dramatic shadows, etc.
- Textures: Rough stone, Smooth metal, Worn fabric, Glossy surfaces
- Color Palette: Vibrant, Muted, Monochromatic, High contrast
- Atmosphere: Fog, Rain, Dust, Smoke, Particles
- Camera: Follows, Tracks, Pans across, Circles around, Tilts upward, Pushes in / pulls back, Overhead view, Handheld, Over-the-shoulder, Wide establishing, Static frame
- Film: Film grain, Lens flares, Pixelated edges, Jittery stop-motion
- Scale: Expansive, Epic, Intimate, Claustrophobic
- Pacing: Slow motion, Time-lapse, Rapid cuts, Lingering shot, Continuous shot, Freeze-frame, Fade-in/out, Seamless transition, Sudden stop
- VFX: Particle systems, Motion blur, Depth of field

10. KEEP IN MIND:
- On-screen text: improved in 2.5 but not guaranteed, keep short prominent
- Complex physics: highly chaotic motion can introduce artifacts; simpler plausible motion more reliable
- Keep scene focused: few clear characters/actions better than crowded frame
- Keep lighting consistent: one coherent logic per shot
- Start simple and layer: core shot then add detail

You must produce prompt that follows official guides exactly for LTX-2.5 covering ALL types when relevant.

User gives concept and preferences including prompt_mode, dialogue, music, on-screen text, character details.

Output VALID JSON ONLY:
{
  "positive_prompt": "For single-shot: single flowing paragraph 4-8 sentences present tense with all key elements. For multi-shot: ONE chronological paragraph with explicit cuts named in natural language like 'A hard cut transitions to...' with re-established shot, identity consistent with same identifiers, audio continuity at every cut. For screenplay-style: scene headers + character cues + quoted dialogue. For dub-it: '[Speaker] is speaking [Language/Accent], saying: \"Dialogue\"' with full dialogue. For text-to-audio: description of audio scene. Must include establish shot, set scene, describe action, define characters with physical cues, camera movement, audio with quoted dialogue, present tense, concrete language. Every sentence adds concrete visual/audio detail.",
  "negative_prompt": "Short negatives: avoid generic bad quality, focus on specific failures like face drift, wardrobe change, extra people, flicker, morphing, unstable camera, conflicting geography, unexplained costume change, abstract emotion labels without physical cues, crowded frame, mixed light sources, chaotic motion",
  "shot_plan": "Brief breakdown: For single-shot '1 shot continuous take with X' For multi-shot 'Shot1: wide establish at dusk with neon, rain ... -> hard cut -> Shot2: medium close-up re-established with same woman yellow raincoat, audio continues muffled -> hard cut -> Shot3: low-angle boots, music drops' For screenplay 'EXT. ... Reporter (live): ...' For dub-it 'Single speaker, language, full dialogue, length matched' For text-to-audio 'Audio-only, duration via frames, cfg near 1'",
  "audio_notes": "How audio handled: ambient (coffeeshop noise/wind and rain/forest ambience), music (soft synth, piano score, low drone, electronic pulse, orchestral), speech (quoted dialogue with language/accent/volume whisper/mutter/shout, physical cues), singing, audio continuity at every cut (continues/muffled/drops to drone/resolves), for dub-it: match timing syllable length, single speaker, native script",
  "model_settings": "LTX 2.5 notes: present tense, physical emotion cues, quoted dialogue, concrete camera, on-screen text short prominent, complex physics simple plausible motion. For first-frame image-to-video prefer single continuous take unless multi-shot explicitly requested. For text-to-audio: audio-only mode LTXVAudioOnlyModel disables video stream, duration = frames/frame-rate, cfg near 1, distilled 8-step. For Dub-It: template [Speaker] is speaking [Language/Accent], saying: Dialogue, validated EN/FR/ES/DE/RU, single speaker, full dialogue, native script, match audio length."
}

Rules:
- Every sentence concrete visual/audio detail, not filler
- Prefer 2-4 shots for multi-shot, not more
- For image-to-video from first frame, prefer single continuous take unless user explicitly wants multi-shot cut away
- Positive prompt ONE chronological paragraph for multi-shot (not shot list numbered, not sluglines unless cut described in prose) - or screenplay style if requested with headers and character cues
- Name transitions explicitly in natural language at each cut for multi-shot
- Re-establish new shot after each cut: shot scale, camera angle, who/what in frame, lighting if changed
- Reuse same visual identifiers for recurring subjects
- State audio continuity at every cut or continuous soundscape for single-shot
- Present tense, physical emotion cues, dialogue in quotes preserved verbatim
- For dub-it: use exact template [Speaker] is speaking [Language/Accent], saying: "Dialogue", full dialogue, native script, single speaker
- For on-screen text: keep short prominent, LTX 2.5 improved but not guaranteed
- Output ONLY JSON, no markdown
"""


def _build_system_prompt_ltx25(
    video_style: str, shot_type: str, camera_motion: str, lighting: str,
    prompt_mode: str, transition_style: str, audio_continuity: str,
    dialogue_mode: str, dialogue_language: str, dialogue_volume: str,
    music_style: str, on_screen_text_mode: str, custom_system_prompt: str,
    duration: str, aspect_ratio: str
) -> str:
    base = LTX25_SYSTEM_PROMPT
    extras = []

    def add(label, value):
        if value and value != "auto-infer" and value != "auto-infer (choose best for concept)" and value != "auto-infer (choose best)":
            extras.append(f"{label}: {value}")

    add("Video style", video_style)
    add("Shot type", shot_type)
    add("Camera motion", camera_motion)
    add("Lighting", lighting)
    add("Prompt mode", prompt_mode)
    add("Transition style", transition_style)
    add("Audio continuity", audio_continuity)
    add("Dialogue mode", dialogue_mode)
    add("Dialogue language", dialogue_language)
    add("Dialogue volume/style", dialogue_volume)
    add("Music style", music_style)
    add("On-screen text mode", on_screen_text_mode)
    add("Duration", duration)
    add("Aspect ratio", aspect_ratio)

    if custom_system_prompt and custom_system_prompt.strip():
        extras.append(f"Additional user guidance (high priority): {custom_system_prompt.strip()}")

    if extras:
        return base + "\n\nUser preferences for this generation:\n" + "\n".join(f"- {e}" for e in extras) + "\n\nFollow user preferences while respecting official LTX 2.5 guide rules for ALL types (single, multi-shot, screenplay, dub-it, text-to-audio, dialogue, music, on-screen text)."

    return base


class LTX25Base:
    CATEGORY = "ComfyUI-OMG/Video/LTX"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "shot_plan", "audio_notes", "model_settings")
    MODEL_PROFILE = "LTX 2.5"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {"multiline": True, "default": "", "tooltip": "Few words allowed - e.g. 'rainy city intersection woman yellow raincoat' or 'news reporter small town oil geyser' or 'frog yoga studio' - expands to official LTX 2.5 prompt covering single-shot, multi-shot, screenplay, dialogue, music"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Optional: Ollama model for LLM generation - if missing uses CLIP template fallback with official guide for ALL types"}),
                "clip": ("CLIP", {"tooltip": "Optional: CLIP - alternative without Ollama, uses template following official guide for ALL types"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional first-frame reference - per guide for image-to-video from first frame, prefer single continuous take unless multi-shot explicitly requested"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional second reference"}),
                "duration": (["auto-infer", "4 seconds", "6 seconds", "8 seconds", "10 seconds", "12 seconds", "15 seconds"], {"default": "auto-infer", "tooltip": "Duration affects shot count - longer allows more cuts but guide says prefer 2-4 shots. For text-to-audio: duration = num-frames / frame-rate, frames follow 1 + multiple of 8 pattern"}),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "prompt_mode": (PROMPT_MODE_V25, {"default": "auto-infer (choose best for concept)", "tooltip": "LTX 2.5 ALL types: single-shot, multi-shot 2/3/4, screenplay-style with dialogue headers, dub-it speech replacement [Speaker] is speaking [Lang], saying: \"Dialogue\", text-to-audio audio-only"}),
                "video_style": (VIDEO_STYLE, {"default": "auto-infer"}),
                "style_custom": ("STRING", {"default": ""}),
                "shot_type": (SHOT_TYPE_V25, {"default": "auto-infer", "tooltip": "Shot scale - for multi-shot, first shot establishes, then detail, then reaction. For single-shot, match detail to scale"}),
                "shot_custom": ("STRING", {"default": ""}),
                "camera_motion": (CAMERA_MOTION_V25, {"default": "auto-infer", "tooltip": "How camera moves - present tense, describe subjects after movement helps complete motion"}),
                "camera_custom": ("STRING", {"default": ""}),
                "lighting": (LIGHTING_V25, {"default": "auto-infer", "tooltip": "Lighting, color palette, textures, atmosphere - keep consistent per shot, one coherent logic"}),
                "lighting_custom": ("STRING", {"default": ""}),
                "transition_style": (TRANSITION_V25, {"default": "auto-infer", "tooltip": "Multi-shot only: Name edit in natural language - hard cut, match cut, dissolve. Single-shot: no transition"}),
                "transition_custom": ("STRING", {"default": ""}),
                "audio_continuity": (AUDIO_CONTINUITY_V25, {"default": "auto-infer", "tooltip": "State at every cut whether music/dialogue/ambience continues or changes - mandatory for multi-shot per guide"}),
                "audio_custom": ("STRING", {"default": "", "tooltip": "Custom audio: ambient (coffeeshop noise, wind and rain), music (soft synth, piano score, low drone), speech in quotes with language/accent/volume"}),
                "dialogue_mode": (DIALOGUE_MODE_V25, {"default": "auto-infer", "tooltip": "Dialogue handling: no dialogue, single speaker quoted, multi speaker with character cues, dub-it template [Speaker] is speaking [Lang], saying: \"Dialogue\", singing"}),
                "dialogue_language": (DIALOGUE_LANGUAGE_V25, {"default": "auto-infer", "tooltip": "Dialogue language - for dub-it validated EN/FR/ES/DE/RU, must use native script (Cyrillic for Russian etc), single speaker"}),
                "dialogue_volume": (DIALOGUE_VOLUME_V25, {"default": "auto-infer", "tooltip": "Volume/style: whisper, mutter, shout, scream, energetic announcer, resonant gravitas, distorted radio, robotic monotone, childlike curiosity"}),
                "dialogue": ("STRING", {"multiline": True, "default": "", "tooltip": "Exact dialogue lines with quotes preserved verbatim - e.g. She whispers, \"He's late.\" or Reporter (live): \"Thank you, Sylvia...\" For dub-it: full dialogue text, model does NOT translate, match audio length syllable length"}),
                "music_style": (MUSIC_STYLE_V25, {"default": "auto-infer", "tooltip": "Music: ambient setting, instrumentation, tempo, pulse, rhythm, accent hits, rises, drops, bursts, final resolution, fade - state continuity at every cut"}),
                "music_custom": ("STRING", {"default": "", "tooltip": "Custom music description - e.g. Soft synth music and distant traffic, piano score, low drone, electronic pulse"}),
                "ambient_sound": ("STRING", {"multiline": True, "default": "", "tooltip": "Ambient sound: coffeeshop noise, wind and rain, forest ambience with birds, chatter, distant drilling, incense smoke drifting, etc."}),
                "on_screen_text_mode": (ON_SCREEN_TEXT_MODE_V25, {"default": "auto-infer", "tooltip": "On-screen text handling - LTX 2.5 improves short-text accuracy but not guaranteed, keep short prominent, verify throughout, add critical titles in post"}),
                "on_screen_text": ("STRING", {"multiline": True, "default": "", "tooltip": "On-screen text short prominent - e.g. GET READY TO - LTX 2.5 improved but not guaranteed, keep short"}),
                "character_details": ("STRING", {"multiline": True, "default": "", "tooltip": "Character age, hairstyle, clothing, distinguishing features, physical emotion cues NOT abstract labels - e.g. shoulders slump, gaze drops, fingers tighten, not 'she is sad'. Reuse same visual identifiers for recurring subjects in multi-shot"}),
                "additional_details": ("STRING", {"multiline": True, "default": "", "tooltip": "Additional scene details, continuity locks, geography, props, etc."}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "High priority additional guidance for ALL types"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "For 9B or smaller - lean prompt, 8192 ctx, retries"}),
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

        system = _build_system_prompt_ltx25(
            video_style=self._get(kwargs, "video_style", "style_custom", "auto-infer"),
            shot_type=self._get(kwargs, "shot_type", "shot_custom", "auto-infer"),
            camera_motion=self._get(kwargs, "camera_motion", "camera_custom", "auto-infer"),
            lighting=self._get(kwargs, "lighting", "lighting_custom", "auto-infer"),
            prompt_mode=kwargs.get("prompt_mode", "auto-infer (choose best for concept)"),
            transition_style=self._get(kwargs, "transition_style", "transition_custom", "auto-infer"),
            audio_continuity=self._get(kwargs, "audio_continuity", "audio_custom", "auto-infer"),
            dialogue_mode=kwargs.get("dialogue_mode", "auto-infer"),
            dialogue_language=kwargs.get("dialogue_language", "auto-infer"),
            dialogue_volume=kwargs.get("dialogue_volume", "auto-infer"),
            music_style=kwargs.get("music_style", "auto-infer"),
            on_screen_text_mode=kwargs.get("on_screen_text_mode", "auto-infer"),
            custom_system_prompt=kwargs.get("custom_system_prompt", ""),
            duration=kwargs.get("duration", "auto-infer"),
            aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
        )

        # Build comprehensive user prompt covering ALL types
        dialogue = kwargs.get("dialogue", "")
        on_screen_text = kwargs.get("on_screen_text", "")
        character_details = kwargs.get("character_details", "")
        additional_details = kwargs.get("additional_details", "")
        ambient_sound = kwargs.get("ambient_sound", "")
        music_custom = kwargs.get("music_custom", "")
        audio_continuity = self._get(kwargs, "audio_continuity", "audio_custom", "auto-infer")

        user_prompt_parts = [
            f"Concept (few words allowed, expand to official LTX 2.5 prompt covering ALL types - single, multi-shot 2-4, screenplay with dialogue, dub-it, text-to-audio, dialogue, music, on-screen text): {concept}",
            f"Aspect ratio: {kwargs.get('aspect_ratio','16:9')}",
            f"Duration: {kwargs.get('duration','auto-infer')} (for text-to-audio: duration = num-frames / frame-rate, frames follow 1 + multiple of 8)",
            f"Prompt mode: {kwargs.get('prompt_mode','auto-infer (choose best for concept)')}",
        ]
        if num_images > 0:
            user_prompt_parts.append(f"Reference images: {num_images} provided - per official guide for image-to-video from first frame, prefer single continuous take unless multi-shot explicitly requested and described as cut away from opening image. Preserve first frame reference identity and lighting.")
        
        # Dialogue ALL types
        dm = kwargs.get("dialogue_mode", "auto-infer")
        if dm != "auto-infer" and dm != "no dialogue":
            user_prompt_parts.append(f"Dialogue mode: {dm}")
        if kwargs.get("dialogue_language", "auto-infer") != "auto-infer":
            user_prompt_parts.append(f"Dialogue language: {kwargs.get('dialogue_language')} - for dub-it must use native script (Cyrillic for Russian etc), single speaker, validated EN/FR/ES/DE/RU for LTX-2.3, LTX-2.5 in dev")
        if kwargs.get("dialogue_volume", "auto-infer") != "auto-infer":
            user_prompt_parts.append(f"Dialogue volume/style: {kwargs.get('dialogue_volume')} - e.g. Whisper, Mutter, Shout, Scream, Energetic announcer, Resonant gravitas, Distorted radio, Robotic monotone, Childlike curiosity")
        if dialogue.strip():
            user_prompt_parts.append(f"Exact dialogue to preserve verbatim in quotes (present tense, physical cues, for dub-it full dialogue text model does NOT translate, match audio length syllable length): {dialogue}")
        
        # Music ALL types
        if kwargs.get("music_style", "auto-infer") != "auto-infer":
            user_prompt_parts.append(f"Music style: {kwargs.get('music_style')}")
        if music_custom.strip():
            user_prompt_parts.append(f"Music description (instrumentation, tempo, pulse, rhythm, accent hits, rises, drops, bursts, final resolution, fade, continuity at every cut): {music_custom}")
        if ambient_sound.strip():
            user_prompt_parts.append(f"Ambient sound (coffeeshop noise, wind and rain, forest ambience, chatter, distant drilling): {ambient_sound}")
        if audio_continuity != "auto-infer":
            user_prompt_parts.append(f"Audio continuity requirement (mandatory for multi-shot, state at every cut whether music/dialogue/ambience continues or changes): {audio_continuity}")

        # Character definition
        if character_details.strip():
            user_prompt_parts.append(f"Character details (age, hairstyle, clothing, distinguishing features, physical emotion cues NOT abstract labels like 'she is sad' but 'shoulders slump, gaze drops, fingers tighten', reuse same visual identifiers for recurring subjects in multi-shot): {character_details}")

        # On-screen text
        if kwargs.get("on_screen_text_mode", "auto-infer") != "auto-infer":
            user_prompt_parts.append(f"On-screen text mode: {kwargs.get('on_screen_text_mode')}")
        if on_screen_text.strip():
            user_prompt_parts.append(f"On-screen text short prominent (LTX 2.5 improves short-text accuracy but not guaranteed, keep short prominent, verify throughout, add critical titles in post): {on_screen_text}")

        if additional_details.strip():
            user_prompt_parts.append(f"Additional scene details, continuity locks, geography, props: {additional_details}")

        user_prompt = "\n\n".join(user_prompt_parts) + "\n\nOutput valid JSON only in the exact schema from system prompt covering ALL LTX 2.5 types."

        small_model_mode = kwargs.get("small_model_mode", False)
        if small_model_mode and len(system) > 2500:
            system = system[:1800] + "\n\n[Lean 9B mode] Keep output under 1200 chars, valid JSON only, single chronological paragraph with explicit cuts if multi-shot, present tense, physical cues, quoted dialogue, all types covered: single, multi, screenplay, dub-it, text-to-audio, music, on-screen text.\n" + system[-600:]

        _log.info(f"[OMG LTX 2.5 ALL TYPES] Generating {self.MODEL_PROFILE} prompt for: {concept[:80]} mode={kwargs.get('prompt_mode','auto')} dialogue_mode={kwargs.get('dialogue_mode','auto')}")

        if ollama_model is None:
            return self._clip_fallback(concept, kwargs, num_images, clip)

        cfg = ollama_model

        # Think mode handling - available on all Ollama nodes
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        raw = None
        for attempt in range(3 if small_model_mode else 1):
            try:
                raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.4 if small_model_mode else cfg.get("temperature", 0.7, think=think, filter_thinking=filter_thinking),
                    num_ctx=8192 if small_model_mode else cfg.get("num_ctx", 12288),
                    num_predict=1500 if small_model_mode else 3000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                    images=images_b64 if images_b64 else None,
                )
                parsed = extract_json_block(raw)
                if parsed is not None or not small_model_mode:
                    break
                else:
                    _log.warning(f"[LTX 2.5 ALL TYPES Retry {attempt+1}/3] JSON parse failed, retrying lean")
                    system = f"You are LTX 2.5 prompt engineer ALL TYPES. Concept: {concept}. Mode: {kwargs.get('prompt_mode','auto')}. Dialogue: {dialogue[:100]}. Music: {music_custom[:100]}. Output JSON with positive_prompt (single chronological paragraph with explicit cuts for multi-shot, present tense, physical cues, quoted dialogue, dialogue language/accent/volume, music/ambient, on-screen text short, dub-it template if needed), negative_prompt, shot_plan, audio_notes, model_settings. JSON only."
            except TypeError:
                raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=user_prompt,
                    system=system,
                    temperature=0.4 if small_model_mode else cfg.get("temperature", 0.7, think=think, filter_thinking=filter_thinking),
                    num_ctx=8192 if small_model_mode else cfg.get("num_ctx", 12288),
                    num_predict=1500 if small_model_mode else 3000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                )
                break
            except Exception as e:
                _log.warning(f"[LTX 2.5 ALL TYPES Retry {attempt+1}] failed: {e}")
                if attempt == 2:
                    raise

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning(f"[OMG LTX 2.5 ALL TYPES] Failed to parse JSON, returning raw")
            return (raw, "", "", "", "")

        return (
            parsed.get("positive_prompt", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("shot_plan", ""),
            parsed.get("audio_notes", ""),
            parsed.get("model_settings", ""),
        )

    def _clip_fallback(self, concept: str, kwargs, num_images: int, clip):
        """Template fallback covering ALL LTX 2.5 types without LLM - single, multi, screenplay, dub-it, text-to-audio, dialogue, music, on-screen text"""
        prompt_mode = kwargs.get("prompt_mode", "auto-infer (choose best for concept)")
        transition = self._get(kwargs, "transition_style", "transition_custom", "hard cut")
        audio_cont = self._get(kwargs, "audio_continuity", "audio_custom", "music continues across cuts")
        duration = kwargs.get("duration", "8 seconds")
        aspect = kwargs.get("aspect_ratio", "16:9")
        style = self._get(kwargs, "video_style", "style_custom", "cinematic realism")
        shot_type = self._get(kwargs, "shot_type", "shot_custom", "wide shot")
        camera_motion = self._get(kwargs, "camera_motion", "camera_custom", "slow push-in")
        lighting = self._get(kwargs, "lighting", "lighting_custom", "neon glow")
        dialogue = kwargs.get("dialogue", "")
        dialogue_mode = kwargs.get("dialogue_mode", "auto-infer")
        dialogue_language = kwargs.get("dialogue_language", "auto-infer")
        dialogue_volume = kwargs.get("dialogue_volume", "auto-infer")
        music_style = kwargs.get("music_style", "auto-infer")
        music_custom = kwargs.get("music_custom", "")
        ambient_sound = kwargs.get("ambient_sound", "")
        on_screen_text_mode = kwargs.get("on_screen_text_mode", "auto-infer")
        on_screen_text = kwargs.get("on_screen_text", "")
        character_details = kwargs.get("character_details", "")
        additional_details = kwargs.get("additional_details", "")

        is_multi = "multi-shot" in prompt_mode.lower()
        num_shots = 1
        if "2 shots" in prompt_mode:
            num_shots = 2
        elif "3 shots" in prompt_mode:
            num_shots = 3
        elif "4 shots" in prompt_mode:
            num_shots = 4
        elif is_multi:
            num_shots = 3

        is_screenplay = "screenplay" in prompt_mode.lower()
        is_dub_it = "dub-it" in prompt_mode.lower() or (dialogue_mode and "dub-it" in dialogue_mode.lower())
        is_text_to_audio = "text-to-audio" in prompt_mode.lower()

        if num_images > 0 and "multi-shot" not in prompt_mode.lower() and "auto-infer" in prompt_mode.lower() and not is_screenplay and not is_dub_it and not is_text_to_audio:
            is_multi = False
            num_shots = 1

        # Build positive prompt covering ALL types
        if is_text_to_audio:
            # Text-to-Audio: audio-only, no video, duration via frames, cfg near 1
            base = (
                f"Audio-only generation: {concept}. {style} audio treatment with {lighting if lighting != 'auto-infer' else 'clear acoustic space'}. "
                f"{ambient_sound[:200] if ambient_sound else 'Clean acoustic environment'}. "
                f"{music_custom[:200] if music_custom else (music_style if music_style != 'auto-infer' else 'No music, only ambient and speech')}. "
            )
            if dialogue.strip():
                if is_dub_it:
                    # Dub-It template
                    lang = dialogue_language if dialogue_language != "auto-infer" else "English"
                    base += f'{dialogue.strip()} - Dub-It template: [Speaker] is speaking {lang}, saying: "{dialogue.strip()}" - full dialogue, native script, single speaker, match audio length syllable length. '
                else:
                    base += f'Dialogue: {dialogue.strip()} preserved verbatim in quotes. Language: {dialogue_language if dialogue_language != "auto-infer" else "auto"}. Volume: {dialogue_volume if dialogue_volume != "auto-infer" else "auto"}. '
            base += f"Duration {duration}, audio-only mode LTXVAudioOnlyModel disables video stream, duration = frames/frame-rate, frames follow 1 + multiple of 8, cfg near 1 distilled 8-step, decoded via audio VAE."
            positive = base
            shot_plan = f"Audio-only: duration {duration} via frames, cfg near 1, 8-step distilled, audio VAE decode, no video"
            audio_notes = f"Audio-only generation: {ambient_sound[:100] if ambient_sound else 'ambient'} + {music_custom[:100] if music_custom else music_style} + dialogue {dialogue[:100] if dialogue else 'none'} in {dialogue_language} with {dialogue_volume} volume. Duration {duration} = frames/frame-rate."

        elif is_dub_it:
            # Dub-It speech replacement
            lang = dialogue_language if dialogue_language != "auto-infer" else "English"
            vol = dialogue_volume if dialogue_volume != "auto-infer" else "natural"
            base = (
                f"A woman speaking in {lang} saying: \"{dialogue.strip() if dialogue.strip() else concept}\". "
                f"{character_details[:200] if character_details else ''} {additional_details[:200] if additional_details else ''} "
                f"Speaker preserves full video except lip region, new lip movements synced to prompt dialogue, matches original speaker tone, attempts to match delivery {vol}. "
                f"Full dialogue text: \"{dialogue.strip() if dialogue.strip() else concept}\" - model does NOT translate, use native script (Cyrillic for Russian etc), single speaker, validated EN/FR/ES/DE/RU for LTX-2.3, LTX-2.5 in development. "
                f"Match audio length ~ same timing syllable length as original: slightly longer better than too short. Too long = might skip words, too short = slow unnatural. "
                f"Two-stage pipeline: Stage1 low res with source video frames and reference audio as conditioning, IC-LoRA guides lip-sync, upsample video latent while audio frozen, Stage2 high res sharper detail."
            )
            positive = base
            shot_plan = f"Dub-It IC-LoRA Beta: single speaker, language {lang}, full dialogue, native script, match audio length, two-stage low res → upsample → high res, mask-free robust to occlusions, crop to face region to save compute"
            audio_notes = f"Dub-It: {lang} speech replacement, volume {vol}, full dialogue text, single speaker, match timing syllable length, preserves original tone and delivery, lip-synced new dialogue"

        elif is_screenplay:
            # Screenplay-style with scene headers, character cues, quoted dialogue
            base = (
                f"EXT. {character_details[:100] if character_details else 'SMALL TOWN STREET'} – MORNING – {style.upper() if style != 'auto-infer' else 'LIVE ACTION'}\n"
                f"The shot opens on {concept}. Light is {lighting}, {style} visual treatment. {additional_details[:200] if additional_details else ''}\n"
                f"{character_details[:200] if character_details else 'A young woman in yellow raincoat, early 20s, short dark hair'}\n"
            )
            if dialogue.strip():
                # If dialogue contains character cues like Reporter (live):, keep it, else create
                if ":" in dialogue and "\n" in dialogue:
                    base += f"\n{dialogue.strip()}\n"
                else:
                    base += f"\nCharacter (live):\n\"{dialogue.strip()}\"\n"
            else:
                base += f"\nThe camera {camera_motion} relative to subject, slowly revealing {ambient_sound[:100] if ambient_sound else 'environment'}.\n"

            base += f"\nCamera {camera_motion} slowly, {shot_type}. {music_custom[:150] if music_custom else 'Soft music and ambient fill air.'} "
            if on_screen_text.strip():
                base += f'On-screen text "{on_screen_text.strip()}" appears short and prominent, LTX 2.5 improved accuracy but verify. '
            base += f"Present tense, physical emotion cues: {character_details[:150] if character_details else 'shoulders slump, gaze drops, fingers tighten'}."
            positive = base
            shot_plan = f"Screenplay-style: scene header EXT. ... MORNING – {style}, character cues with quoted dialogue, camera pans/dolly, present tense, physical cues, multiple beats, precise timing"
            audio_notes = f"Screenplay audio: {ambient_sound[:100] if ambient_sound else 'chatter, distant drilling'} + {music_custom[:100] if music_custom else 'soft music'} + dialogue in quotes with character cues {dialogue[:100] if dialogue else ''} preserved verbatim, language {dialogue_language}, volume {dialogue_volume}"

        elif not is_multi or num_shots == 1:
            # Single-shot: 4-8 sentences, single flowing paragraph, present tense
            base = (
                f"A {shot_type} frames {concept}. {style} visual treatment with {lighting}, establishing {character_details[:200] if character_details else additional_details[:200] if additional_details else 'a coherent environment'}. "
                f"The scene {camera_motion} relative to subject, present tense action unfolds naturally. "
            )
            if num_images > 0:
                base += f"The camera maintains the opening composition from the first frame reference, preserving identity and lighting, preferring a single continuous take as per official guide for image-to-video. "

            # Dialogue with language/accent/volume
            if dialogue.strip():
                lang_note = f" in {dialogue_language}" if dialogue_language != "auto-infer" else ""
                vol_note = f" with {dialogue_volume} delivery" if dialogue_volume != "auto-infer" else ""
                base += f'{dialogue.strip()} Dialogue placed in quotation marks verbatim{lang_note}{vol_note}. '

            # Music
            if music_custom.strip():
                base += f"{music_custom.strip()} "
            elif music_style != "auto-infer":
                base += f"{music_style} fills the air. "

            # Ambient
            if ambient_sound.strip():
                base += f"{ambient_sound.strip()} "

            if on_screen_text.strip():
                base += f'On-screen text "{on_screen_text.strip()}" appears short and prominent, LTX 2.5 improved accuracy but verify throughout clip. '

            base += (
                f"Soft ambient sound of {audio_cont if audio_cont != 'auto-infer' else 'environment and subtle music'} fills the air. "
                f"The lighting remains consistent with one coherent logic, {lighting}. "
                f"Concrete camera language and physical emotion cues anchor the performance: {character_details[:150] if character_details else 'shoulders slump, gaze drops, physical cues not abstract labels'}."
            )
            positive = base
            shot_plan = "1 shot continuous take: establish → action → resolve with consistent lighting, present tense, physical cues, quoted dialogue"
            audio_notes = f"Single continuous soundscape: {music_custom[:100] if music_custom else music_style} + {ambient_sound[:100] if ambient_sound else 'ambient'} + dialogue {dialogue[:100] if dialogue else 'none'} in {dialogue_language} with {dialogue_volume} volume, continuous"

        else:
            # Multi-shot: 2-4 shots joined by explicit cuts inside ONE chronological paragraph
            if transition == "auto-infer":
                transition_phrase = "A hard cut transitions to"
            elif "match cut" in transition:
                transition_phrase = "A match cut connects to"
            elif "dissolve" in transition:
                transition_phrase = "The image dissolves into"
            else:
                transition_phrase = "A hard cut transitions to"

            shot1 = (
                f"A {shot_type} frames {concept} at {duration}, {style} with {lighting}. "
                f"{character_details[:200] if character_details else additional_details[:200] if additional_details else ''} "
                f"{ambient_sound[:150] if ambient_sound else ''} "
            )
            if num_images > 0:
                shot1 = f"A wide shot frames the scene from the first frame reference at {duration}, {style} with {lighting}. The opening composition from the reference is preserved initially. {concept}. "

            audio_shot1 = f"Soft {music_custom[:100] if music_custom else (music_style if music_style != 'auto-infer' else 'ambient sound and music')} fills the air."

            shot2_scale = "medium close-up" if "close-up" not in shot_type else "medium shot"
            # Include dialogue with language/volume
            dialogue_part = ""
            if dialogue.strip():
                lang_note = f" in {dialogue_language}" if dialogue_language != "auto-infer" else ""
                vol_note = f" with {dialogue_volume} delivery" if dialogue_volume != "auto-infer" else ""
                dialogue_part = f'{dialogue.strip()} Dialogue in quotes verbatim{lang_note}{vol_note}. '
                if is_dub_it:
                    lang = dialogue_language if dialogue_language != "auto-infer" else "English"
                    dialogue_part = f'A woman speaking in {lang} saying: "{dialogue.strip()}" '

            shot2 = (
                f"{transition_phrase} a {shot2_scale} of the same subject, earlier identified, now in new framing under the hood of {lighting}, raindrops or light catching details as they look off-screen left; "
                f"the score continues across the cut, ambience muffled, re-establishing camera angle and lighting. "
                f"{dialogue_part}"
            )

            shot3 = ""
            if num_shots >= 3:
                transition2 = "Another hard cut jumps to" if num_shots == 3 else "A hard cut transitions to"
                shot3 = (
                    f"{transition2} a low-angle shot of the same subject with consistent visual identifiers — {character_details[:150] if character_details else additional_details[:150] if additional_details else 'preserved clothing and features'} — "
                    f"now in final beat, physical emotion cues visible: shoulders, gaze, fingers tighten, not abstract labels. The music drops to a low drone, only wind remains, stating audio continuity explicitly."
                )
                if on_screen_text.strip():
                    shot3 += f' On-screen text "{on_screen_text.strip()}" appears short and prominent, LTX 2.5 improved.'

            shot4 = ""
            if num_shots >= 4:
                shot4 = (
                    f"A final hard cut transitions to a close-up reaction, re-establishing the same woman in the yellow raincoat, earlier at the intersection, now smiling faintly; "
                    f"the piano score that continued across previous cuts now resolves, traffic ambience fades, keeping identity consistent with reused visual identifiers."
                )

            parts = [shot1.strip(), audio_shot1] + ([shot2] if shot2 else []) + ([shot3] if shot3 else []) + ([shot4] if shot4 else [])
            positive = " ".join(p for p in parts if p.strip())

            shot_plan = f"{num_shots} shots: Shot1 wide establish at {duration} with {lighting} → {transition} → Shot2 {shot2_scale} re-established with identity consistent, audio: {audio_cont} continues/muffled → " + (f"{transition} → Shot3 low-angle final beat with physical cues, music drops to drone → " if num_shots>=3 else "") + (f"hard cut → Shot4 close-up reaction, audio resolves" if num_shots>=4 else "") + "Each shot clear job: establish → detail → reaction"

            audio_notes = f"At every cut, state audio continuity per official guide: Shot1 {audio_cont if audio_cont != 'auto-infer' else 'music and ambience continuous'}; after first cut: score continues across cut, traffic muffled; after second cut: music drops to low drone; final: score resolves, wind remains. Dialogue in quotes preserved verbatim in {dialogue_language} with {dialogue_volume} volume. Music: {music_custom[:100] if music_custom else music_style}. Ambient: {ambient_sound[:100] if ambient_sound else 'auto'}. Dialogue: {dialogue[:100] if dialogue else 'none'}."

        # Negative prompt - specific failures per guide tips
        negative = (
            "No face drift, No wardrobe changes, No extra people, No flicker, No morphing, "
            "No unstable camera, No conflicting geography, No unexplained costume change between cuts unless time/place jump explicitly stated, "
            "No abstract emotion labels without physical cues, No crowded frame, No mixed light sources, No chaotic motion"
        )

        # Model settings
        extra_notes = []
        if is_dub_it:
            extra_notes.append(f"Dub-It IC-LoRA Beta: template [Speaker] is speaking [Language/Accent], saying: \"Dialogue\", validated EN/FR/ES/DE/RU, single speaker, full dialogue, native script, match audio length. Two-stage low res → upsample → high res.")
        if is_text_to_audio:
            extra_notes.append(f"Text-to-Audio: audio-only mode LTXVAudioOnlyModel disables video stream, duration = frames/frame-rate frames follow 1 + multiple of 8, cfg near 1 distilled 8-step, audio VAE decode.")
        if on_screen_text.strip():
            extra_notes.append(f"On-screen text short prominent: {on_screen_text[:50]} - LTX 2.5 improved accuracy but not guaranteed, verify throughout, add critical titles in post.")
        if character_details.strip():
            extra_notes.append(f"Character definition with physical emotion cues: {character_details[:80]}")

        model_settings = (
            f"LTX 2.5 official guide ALL TYPES: present tense, physical emotion cues, quoted dialogue in {kwargs.get('dialogue_language','auto')} with {kwargs.get('dialogue_volume','auto')} volume, concrete camera language, "
            f"{'single continuous take preferred for first-frame image-to-video' if num_images>0 else '2-4 shots preferred for multi-shot'}, "
            f"each shot clear job (establish → detail → reaction, wide → medium → close-up), chronological, identity consistent with same visual identifiers, audio continuity stated at every cut, "
            f"music: {music_custom[:50] if music_custom else kwargs.get('music_style','auto')}, ambient: {ambient_sound[:50] if ambient_sound else 'auto'}, dialogue: {dialogue[:50] if dialogue else 'none'} in {dialogue_language}, on-screen text short prominent if any: {on_screen_text[:50]}. "
            f"{' '.join(extra_notes)} "
            f"Duration {duration}, aspect {aspect}, style {style}, lighting {lighting}, camera {camera_motion}"
        )

        return (
            positive,
            negative,
            shot_plan,
            audio_notes,
            model_settings,
        )


class OllamaLTX25Prompt(LTX25Base):
    MODEL_PROFILE = "LTX 2.5"


NODE_CLASS_MAPPINGS = {
    "OllamaLTX25Prompt": OllamaLTX25Prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaLTX25Prompt": "LTX 2.5 Prompt - Multi-Shot Official",
}
