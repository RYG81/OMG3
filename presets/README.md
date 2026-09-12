# Fal.ai 44 Presets + PromptSama + CLIP Support

This folder contains:

- `fal_minimax_h3_44_presets.json` - 44 video examples from https://fal.ai/learn/devs/minimax-h3-prompting-guide curated into Authority Stack format. Categories: Brand Films, Visual Concepts, AI-Native Storytelling, Product, Game UI, Animation, Multi-Asset, Character Motion, Voice Clone, Precise Editing.

Each preset includes:
- id, name, category, endpoint (text-to-video / first-and-last-frame / reference-to-video)
- reference_count, reference_use (authority declarations)
- prompt_template (full production brief with timed shot list 0-4s/4-9s/9-15s, screen geography, acting, light, camera, sound, negatives)
- shot_plan, negatives, ideal_for

Use via `OMG MiniMax H3 Fal.ai 44 Presets` node in ComfyUI.

## Reference Inputs Added

Every Ollama video node now has optional IMAGE inputs:
- `image_ref_1`, `image_ref_2`, `image_ref_3` (up to 6 for Omni)
- These are ComfyUI IMAGE tensors (from LoadImage etc.)
- If Ollama model is vision capable (qwen3-vl, llava, gemma3-vision), images are converted to base64 and passed to Ollama generate via `images` param for vision-aware prompting
- Images are also counted for authority: "User provided 3 reference images for identity/scene"

## CLIP / Text Encoder Model Support

Every node now has optional `clip` input (CLIP from ComfyUI) and `use_clip_fallback` boolean.

**Two paths:**

1. **Ollama path (default):** If `ollama_model` connected, uses Ollama LLM (qwen3:32b etc.) to expand few words into full PromptSama brief with vision.

2. **CLIP fallback path (no Ollama needed):** If `ollama_model` missing and `clip` connected OR `use_clip_fallback=True`, uses offline template generation:
   - Loads `fal_minimax_h3_44_presets.json`
   - If `fal_preset` selected (e.g. `vintage_binocular_brand_film`), injects your few-words concept into preset template
   - Else uses generic cinematic template with 6-layer Authority Stack
   - Uses CLIP tokenizer if available to validate concept tokens: `clip.tokenize(concept)` -> shape
   - Returns positive_prompt as full production brief string ready for MiniMax H3 / LTX / Wan API
   - No Ollama server needed — works with ComfyUI downloaded CLIP models (clip_l, t5xxl, etc.)

This answers: "can we use the text encode model downloaded inside comfyui for same task"

Yes — connect any CLIP loader (e.g. `CLIPLoader` or `DualCLIPLoader`) to the `clip` input, enable `use_clip_fallback`, and node will work offline using fal presets templates + your concept, no Ollama.

**Example without Ollama:**
```
LoadImage (your face) -> image_ref_1
CLIPLoader (clip_l) -> clip
Concept: "chai rain Mumbai" (3 words)
Fal Preset: "vintage_binocular_brand_film" or "auto-infer"
Use CLIP Fallback: True
-> OMG MiniMax H3 Simple -> positive_prompt (3248 chars with [REFERENCE USE] etc.)
-> Goes to Fal.ai MiniMax H3 API or ComfyUI MiniMax H3 wrapper
```

## Nodes that now support refs + CLIP

- `OMG MiniMax H3 Simple` — now: concept (required) + optional ollama_model, clip, image_ref_1/2/3, fal_preset, use_clip_fallback
- `OMG MiniMax H3 Advanced` — same + full advanced fields
- `OMG MiniMax H3 Omni` — image_ref_1..6 + clip + fal 44 presets
- `OMG MiniMax H3 Fal.ai 44 Presets` — preset dropdown + concept + ollama_model/clip + image_ref_1/2/3
- `OMG LTX-V Improved` — concept + ollama_model/clip + image_ref_1/2 + fal_preset + use_clip_fallback
- `OMG Wan 2.2 Improved` — same
- `OMG Image to Storyboard V2` — already had IMAGE, now also clip optional for alternative path

All original nodes still work if only ollama_model connected (backward compatible).
