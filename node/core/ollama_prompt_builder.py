"""
ollama_prompt_builder.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Prompt Builder - Split into Basic + Batch

Two nodes per user request:
- Ollama Prompt Builder (simple, no batch) - few inputs, not tall, worth doing
- Ollama Prompt Builder Batch (with batch) - ultra-long generations 60-200+ prompts

Plus OllamaSinglePromptBuilder unchanged

All Ollama nodes now have:
- think_mode (off/on/low/medium/high) - Ollama thinking
- filter_thinking (BOOLEAN) - filter <think>...</think> tags from output for qwen3, deepseek-r1 etc.
- Proper prompt generation using Ollama with good system prompts, not just templates
- Tooltips on every input, IS_CHANGED hashing, ProgressBar+interrupt for batch, no import side effects
"""

from __future__ import annotations

import json
import re
import sys
import logging
import hashlib
from typing import Dict, Any, List

from ...prompts.prompt_enhance import PROMPTFORGE_2026U_SYSTEM_PROMPT
from ...utils.system_prompt import compose_system_prompt
from ...utils.text_utils import extract_json_block, filter_thinking

_log = logging.getLogger(__name__)

_SYSTEM_PROMPTS = {
    "enhance_sd": """You are an expert Stable Diffusion prompt engineer.
Transform the user's rough concept into a detailed, effective image generation prompt.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines for POSITIVE:
- Add art style, lighting, camera angle, mood, quality boosters
- Use parentheses for emphasis: (masterpiece:1.2)
- Include artist names if appropriate

Guidelines for NEGATIVE:
- Always include: blurry, lowres, bad anatomy, watermark, text, error
""",

    "enhance_flux": """You are an expert prompt engineer for FLUX.1 image models.
FLUX works best with natural language sentences, not comma-separated tags.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: <2-4 natural language sentences describing the scene>
NEGATIVE: 

Make the description vivid, specific, and cinematic.
""",

    "2026U": PROMPTFORGE_2026U_SYSTEM_PROMPT,

    "summarize": """You are a text summarisation expert.
Condense the user's text into a concise, clear single-line summary that captures the core subject matter.

OUTPUT FORMAT – respond with EXACTLY two lines:
POSITIVE: 
NEGATIVE: irrelevant, off-topic
""",

    "translate": """You are a professional translator and prompt cleaner.
Translate the user's text to English, fix grammar, and optimise it for use as an image generation prompt.

OUTPUT FORMAT – respond with EXACTLY two lines:
POSITIVE: 
NEGATIVE: 
""",

    "custom": "",

    "qwen_image": """You are an expert prompt writer for Qwen-Image style models.
Create a visually rich, direct image prompt in natural language with strong scene specificity.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines:
- Include subject, composition, camera framing, lighting, color palette, environment, mood
- Prefer concrete visual details over abstract adjectives
- If user intent is photographic, include lens/framing cues
- Keep prompt coherent and production-ready
""",

    "chatgpt_image_2": """You are an expert prompt writer for ChatGPT image generation style workflows.
Convert the user's idea into a clear instruction-style image prompt that emphasizes intent and constraints.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines:
- Structure prompt as: subject + context + style + composition + lighting + quality targets
- Mention specific materials/textures where relevant
- Include "no text or watermark" unless user explicitly asks for text
- Keep it concise but specific
""",

    "gemini_image": """You are an expert prompt writer for Gemini image generation style prompting.
Produce a polished, natural-language visual brief suitable for multimodal image generation.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: <2-5 sentence visual brief with cinematic clarity>
NEGATIVE: 

Guidelines:
- Emphasize narrative clarity, visual hierarchy, and realistic lighting behavior
- Describe foreground/midground/background when useful
- Add rendering intent (photo, illustration, 3D, watercolor, etc.) based on user request
- Avoid contradictory style instructions
""",
}

_MARKDOWN_BLOCK = re.compile(r"```(?:\w+)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)
_POS_RE = re.compile(r"^\s*POSITIVE\s*:\s*", re.IGNORECASE)
_NEG_RE = re.compile(r"^\s*NEGATIVE\s*:\s*", re.IGNORECASE)


def _parse_positive_negative(raw: str) -> tuple[str, str]:
    clean = _MARKDOWN_BLOCK.sub(r"\1", raw).strip()
    positive = negative = ""
    found_pos = False
    for line in clean.splitlines():
        if m := _POS_RE.match(line):
            positive = line[m.end():].strip()
            found_pos = True
        elif m := _NEG_RE.match(line):
            negative = line[m.end():].strip()
    if not found_pos:
        paragraphs = [p.strip() for p in clean.split("\n\n") if len(p.strip()) > 10]
        positive = paragraphs[0] if paragraphs else clean
        _log.warning("POSITIVE: label missing. Using first substantial paragraph as fallback.")
    return positive.strip(), negative.strip()


def _append_smart(base: str, extra: str, sep: str = ", ") -> str:
    if not extra.strip():
        return base
    base = base.rstrip(" ,.;:")
    extra = extra.lstrip(" ,.;:")
    return f"{base}{sep}{extra}" if base else extra


def _repair_json(text: str) -> tuple[dict | list | None, str]:
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        first_err = str(e)
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
    candidate = text.rstrip()
    fixes = []
    if in_str:
        candidate += '"'
        fixes.append("closed an open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped trailing comma")
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma(s) before closing bracket")
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"
    if stack:
        fixes.append("added " + "".join("}" if o == "{" else "]" for o in reversed(stack)))
    if not fixes:
        return None, first_err
    try:
        return json.loads(candidate), ", ".join(fixes)
    except json.JSONDecodeError as e:
        return None, str(e)


def _ensure_valid_json_only(text: str) -> tuple[str, bool, str]:
    original = text.strip()
    inner = original
    m = _MARKDOWN_BLOCK.search(original)
    if m:
        inner = m.group(1).strip()
    try:
        data = json.loads(inner)
        cleaned = json.dumps(data, indent=2, ensure_ascii=False)
        return cleaned, True, "valid on first parse"
    except Exception:
        pass
    data, note = _repair_json(inner)
    if data is not None:
        try:
            cleaned = json.dumps(data, indent=2, ensure_ascii=False)
            return cleaned, True, f"repaired: {note}"
        except:
            pass
    try:
        data = extract_json_block(inner)
        if data is not None:
            cleaned = json.dumps(data, indent=2, ensure_ascii=False)
            return cleaned, True, "repaired via extract_json_block"
    except:
        pass
    try:
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            matches = list(re.finditer(pattern, inner))
            if matches:
                largest = max(matches, key=lambda m: len(m.group(0)))
                candidate = largest.group(0)
                data, note = _repair_json(candidate)
                if data is not None:
                    cleaned = json.dumps(data, indent=2, ensure_ascii=False)
                    return cleaned, True, f"repaired via regex {pattern}: {note}"
    except:
        pass
    return original, False, "failed to repair, returning original"


def _generate_chunk(
    base_url: str,
    model: str,
    system: str,
    user_input: str,
    temperature: float,
    ollama_model: Dict[str, Any],
    num_predict: int,
    num_ctx: int,
    keep_alive: str,
    stream_to_console: bool,
    think,
    chunk_label: str = "",
) -> tuple[str, str, str]:
    # Lazy import inside function per skill performance rules
    from .ollama_client import generate_stream, OllamaConnectionError

    if chunk_label:
        _log.info("▶ Chunk %s | temp=%.2f | ctx=%d | think=%s", chunk_label, temperature, num_ctx, str(think))

    tokens: list[str] = []
    try:
        for token in generate_stream(
            base_url=base_url,
            model=model,
            prompt=user_input,
            system=system,
            temperature=temperature,
            top_p=ollama_model.get("top_p", 0.9),
            top_k=ollama_model.get("top_k", 40),
            repeat_penalty=ollama_model.get("repeat_penalty", 1.1),
            num_predict=num_predict,
            num_ctx=num_ctx,
            seed=ollama_model.get("seed", -1),
            think=think,
            keep_alive=keep_alive,
        ):
            tokens.append(token)
            if stream_to_console:
                sys.stdout.write(token)
                sys.stdout.flush()
    except OllamaConnectionError as exc:
        raise RuntimeError(f"Ollama chunk generation failed: {exc}") from exc

    raw_output = "".join(tokens)
    if stream_to_console and chunk_label:
        sys.stdout.write(f"\n[Chunk {chunk_label}] ✅\n")
        sys.stdout.flush()

    if not raw_output.strip():
        raise RuntimeError("Model returned empty output for chunk.")

    pos, neg = _parse_positive_negative(raw_output)
    return pos, neg, raw_output


def _base_required_inputs():
    return {
        "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Ollama model - qwen3, gemma3, llava etc. For prompt generation"}),
        "concept": (
            "STRING",
            {
                "multiline": True,
                "default": "a futuristic city at night with neon lights",
                "tooltip": "Rough idea / concept / text to process - few words allowed, will be expanded to detailed prompt",
            },
        ),
        "mode": (
            list(_SYSTEM_PROMPTS.keys()),
            {"default": "enhance_sd", "tooltip": "Prompt style preset - enhance_sd, enhance_flux, qwen_image, etc. When custom, uses custom_system_prompt block"},
        ),
    }


def _base_optional_inputs():
    return {
        "custom_system_prompt": (
            "STRING",
            {"multiline": True, "default": "", "tooltip": "Additional instructions appended to every preset; used alone when mode=custom - when you pick custom in dropdown, this block is used"},
        ),
        "style_modifier": (
            "STRING",
            {"multiline": False, "default": "", "tooltip": "Extra style hint (e.g. 'cyberpunk', 'watercolor')"},
        ),
        "num_predict": (
            "INT",
            {"default": 2048, "min": 64, "max": 8192, "step": 64, "tooltip": "Max tokens per generation - for batch mode keep ≤4096"},
        ),
        "num_ctx": (
            "INT",
            {"default": 8192, "min": 2048, "max": 32768, "step": 1024, "tooltip": "Ollama context window. Must be >= input + num_predict."},
        ),
        "temperature_override": (
            "FLOAT",
            {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Override model temperature"},
        ),
        "use_model_temperature": (
            "BOOLEAN",
            {"default": False, "tooltip": "If True, ignore temperature_override"},
        ),
        "stream_to_console": (
            "BOOLEAN", {"default": True, "tooltip": "Print tokens to console as they arrive - live preview"},
        ),
        "think_mode": (
            ["off", "on", "low", "medium", "high"],
            {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels. Available on all Ollama nodes to filter thinking."},
        ),
        "filter_thinking": (
            "BOOLEAN",
            {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only. Available on all Ollama nodes."},
        ),
        "append_to_positive": (
            "STRING", {"multiline": False, "default": "", "tooltip": "Text appended to positive output"},
        ),
        "append_to_negative": (
            "STRING", {"multiline": False, "default": "", "tooltip": "Text appended to negative output"},
        ),
    }


def _batch_optional_inputs():
    return {
        "batch_mode": (
            "BOOLEAN",
            {"default": False, "tooltip": "Enable batch generation for long outputs or multiple variants - for ultra-long 4K-6K H3 prompts, location DB with 5 zones x 36 pieces etc"},
        ),
        "batch_strategy": (
            ["unique_varied", "continuation", "chaptered", "json_array"],
            {"default": "continuation", "tooltip": "unique_varied: each prompt unique varied. continuation: parts of one long continuous response, each continues previous (for long story / 4K-6K H3 prompt). chaptered: each part is a chapter. json_array: each chunk is one JSON object (location), final combined as JSON array inside single code block - for location database generator with 5 zones x 36 pieces."},
        ),
        "batch_count": (
            "INT",
            {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of parts/prompts to generate in batch mode"},
        ),
        "batch_separator": (
            "STRING",
            {"multiline": False, "default": "\n\n", "tooltip": "Separator between batch items in final combined output"},
        ),
    }


# ── Basic Node (NO BATCH) - Worth doing: few inputs, not tall, sane defaults ──

def _apply_thinking_filter(text: str, filter_enabled: bool) -> str:
    if not filter_enabled or not text:
        return text
    try:
        from ...utils.text_utils import filter_thinking
        return filter_thinking(text, True)
    except Exception:
        import re as _re
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r'<thinking>.*?</thinking>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        return text.strip()


class OllamaPromptBuilder:
    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "build_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "raw_output")
    OUTPUT_NODE = True
    DESCRIPTION = "Ollama Prompt Builder (Simple, no batch) - Expand a short concept into full SD/SDXL prompt with proper Ollama generation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _base_required_inputs(),
            "optional": _base_optional_inputs(),
        }

    @classmethod
    def IS_CHANGED(cls, ollama_model: Dict[str, Any], concept: str, mode: str = "enhance_sd", custom_system_prompt: str = "", style_modifier: str = "", num_predict: int = 2048, num_ctx: int = 8192, temperature_override: float = 0.6, use_model_temperature: bool = False, stream_to_console: bool = True, think_mode: str = "off", filter_thinking: bool = True, append_to_positive: str = "", append_to_negative: str = "", **kwargs):
        # Proper caching - hash actual dependencies per skill §3
        try:
            cfg = ollama_model or {}
            key = {
                "model": cfg.get("model", ""),
                "concept": concept[:500],
                "mode": mode,
                "custom_system_prompt": custom_system_prompt[:300],
                "style_modifier": style_modifier,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "temperature_override": temperature_override,
                "think_mode": think_mode,
                "filter_thinking": filter_thinking,
                "append_pos": append_to_positive[:100],
                "append_neg": append_to_negative[:100],
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def build_prompt(
        self,
        ollama_model: Dict[str, Any],
        concept: str,
        mode: str = "enhance_sd",
        custom_system_prompt: str = "",
        style_modifier: str = "",
        num_predict: int = 2048,
        num_ctx: int = 8192,
        temperature_override: float = 0.6,
        use_model_temperature: bool = False,
        stream_to_console: bool = True,
        think_mode: str = "off",
        filter_thinking: bool = True,
        append_to_positive: str = "",
        append_to_negative: str = "",
    ) -> tuple[str, str, str]:
        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")
        temperature = ollama_model.get("temperature", 0.7) if use_model_temperature else temperature_override
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add think and filter_thinking to cfg for run_structured_task to pick up
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass

        system = compose_system_prompt(_SYSTEM_PROMPTS[mode], ollama_model, custom_system_prompt)
        user_input = concept.strip()
        if style_modifier.strip():
            user_input += f"\n\nStyle: {style_modifier.strip()}"

        estimated_input_tokens = (len(system) + len(user_input)) // 4
        if estimated_input_tokens + num_predict > num_ctx:
            _log.warning("⚠️ Context window may truncate: input(~%d) + num_predict(%d) > num_ctx(%d)", estimated_input_tokens, num_predict, num_ctx)

        _log.info("▶ Generating with %s | mode=%s | temp=%.2f | ctx=%d | think=%s | filter_thinking=%s", model, mode, temperature, num_ctx, think_mode, filter_thinking)

        pos, neg, raw = _generate_chunk(
            base_url=base_url, model=model, system=system, user_input=user_input,
            temperature=temperature, ollama_model=ollama_model, num_predict=num_predict,
            num_ctx=num_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, think=think,
        )

        # Filter thinking tags from output if enabled - available on all Ollama nodes
        if filter_thinking:
            pos = filter_thinking(pos, True)
            neg = filter_thinking(neg, True)
            raw = filter_thinking(raw, True)

        # JSON cleanup if needed
        raw_stripped = raw.strip()
        is_json_like = (
            "photo_zones" in raw_stripped or "prompt_pieces" in raw_stripped
            or raw_stripped.startswith("```json") or raw_stripped.startswith("```")
            or raw_stripped.startswith("{") or raw_stripped.startswith("[")
        )
        if is_json_like:
            cleaned, is_valid, note = _ensure_valid_json_only(raw)
            if is_valid:
                if raw_stripped.startswith("```"):
                    pos = f"```json\n{cleaned}\n```"
                else:
                    pos = f"```json\n{cleaned}\n```"
                _log.info(f"✓ JSON cleanup: valid after {note}")

        pos = _append_smart(pos, append_to_positive)
        neg = _append_smart(neg, append_to_negative)
        return pos, neg, raw


# ── Batch Node (WITH BATCH) - Advanced variant per skill §5 ──
class OllamaPromptBuilderBatch:
    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "build_prompt_batch"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "raw_output")
    OUTPUT_NODE = True
    DESCRIPTION = "Ollama Prompt Builder Batch - Ultra-long generations 60-200+ prompts, location DB 5 zones x 36 pieces, long story continuation with proper Ollama generation and thinking filter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _base_required_inputs(),
            "optional": {**_base_optional_inputs(), **_batch_optional_inputs()},
        }

    @classmethod
    def IS_CHANGED(cls, ollama_model: Dict[str, Any], concept: str, mode: str = "enhance_sd", custom_system_prompt: str = "", style_modifier: str = "", num_predict: int = 2048, num_ctx: int = 8192, temperature_override: float = 0.6, use_model_temperature: bool = False, stream_to_console: bool = True, think_mode: str = "off", filter_thinking: bool = True, append_to_positive: str = "", append_to_negative: str = "", batch_mode: bool = False, batch_strategy: str = "continuation", batch_count: int = 10, batch_separator: str = "\n\n", **kwargs):
        try:
            cfg = ollama_model or {}
            key = {
                "model": cfg.get("model", ""),
                "concept": concept[:500],
                "mode": mode,
                "custom_system_prompt": custom_system_prompt[:300],
                "style_modifier": style_modifier,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "temperature_override": temperature_override,
                "think_mode": think_mode,
                "filter_thinking": filter_thinking,
                "batch_mode": batch_mode,
                "batch_strategy": batch_strategy,
                "batch_count": batch_count,
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def build_prompt_batch(
        self,
        ollama_model: Dict[str, Any],
        concept: str,
        mode: str = "enhance_sd",
        custom_system_prompt: str = "",
        style_modifier: str = "",
        num_predict: int = 2048,
        num_ctx: int = 8192,
        temperature_override: float = 0.6,
        use_model_temperature: bool = False,
        stream_to_console: bool = True,
        think_mode: str = "off",
        filter_thinking: bool = True,
        append_to_positive: str = "",
        append_to_negative: str = "",
        batch_mode: bool = False,
        batch_strategy: str = "continuation",
        batch_count: int = 10,
        batch_separator: str = "\n\n",
    ) -> tuple[str, str, str]:
        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")
        temperature = ollama_model.get("temperature", 0.7) if use_model_temperature else temperature_override
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

        system = compose_system_prompt(_SYSTEM_PROMPTS[mode], ollama_model, custom_system_prompt)
        user_input = concept.strip()
        if style_modifier.strip():
            user_input += f"\n\nStyle: {style_modifier.strip()}"

        # ── Batch Mode: Generate in chunks ──────────────────────────
        if batch_mode and batch_count > 1:
            _log.info("🔄 Batch mode: strategy=%s generating %d parts | think=%s filter_thinking=%s", batch_strategy, batch_count, think_mode, filter_thinking)
            all_positives: List[str] = []
            all_negatives: List[str] = []
            all_raw: List[str] = []
            chunk_predict = min(num_predict, 4096)
            chunk_ctx = max(num_ctx, 8192)
            accumulated_text = ""
            all_json_objects = []

            # ProgressBar per skill §3 for >2s loop work
            try:
                from comfy.utils import ProgressBar
                pbar = ProgressBar(batch_count)
            except ImportError:
                pbar = None

            for i in range(batch_count):
                # Honor interrupts per skill §3
                try:
                    from comfy.model_management import throw_exception_if_processing_interrupted
                    throw_exception_if_processing_interrupted()
                except ImportError:
                    pass

                chunk_label = f"{i+1}/{batch_count}"
                if batch_strategy == "unique_varied":
                    batched_input = f"{user_input}\n\nGenerate prompt #{i+1} of {batch_count}. Make each prompt unique and varied."
                elif batch_strategy == "chaptered":
                    batched_input = f"{user_input}\n\nGenerate Chapter {i+1} of {batch_count}. Title: Chapter {i+1}. Each chapter should be distinct but part of same overall story/series."
                elif batch_strategy == "json_array":
                    batched_input = f"{user_input}\n\nGenerate item #{i+1} of {batch_count} as valid JSON inside single markdown code block. Output ONLY valid JSON (object) inside single ```json code block for THIS ONE item only, not the full array yet. Follow the system prompt schema exactly. Keep it concise but complete per schema. Do not output array yet, just one JSON object."
                else:
                    if i == 0:
                        batched_input = f"{user_input}\n\nGenerate PART 1 of {batch_count} of a single long continuous response. This is the beginning. Start the long response, set up scene/characters, do not conclude."
                    else:
                        prev_for_context = accumulated_text[-6000:] if len(accumulated_text) > 6000 else accumulated_text
                        batched_input = f"{user_input}\n\nPrevious parts so far (for context, do NOT repeat verbatim, continue seamlessly):\n---\n{prev_for_context}\n---\n\nGenerate PART {i+1} of {batch_count} that CONTINUES seamlessly from where previous part ended. Do not repeat, do not summarize previous parts, continue where it left off. If this is final part ({i+1}=={batch_count}), conclude and provide strong ending."

                try:
                    pos, neg, raw = _generate_chunk(
                        base_url=base_url, model=model, system=system, user_input=batched_input,
                        temperature=temperature, ollama_model=ollama_model, num_predict=chunk_predict,
                        num_ctx=chunk_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, think=think, chunk_label=chunk_label,
                    )
                    # Filter thinking if enabled
                    if filter_thinking:
                        pos = filter_thinking(pos, True)
                        neg = filter_thinking(neg, True)
                        raw = filter_thinking(raw, True)

                    if batch_strategy == "json_array":
                        json_obj = None
                        try:
                            m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw, re.IGNORECASE)
                            if m:
                                json_obj = json.loads(m.group(1).strip())
                            else:
                                json_obj = json.loads(raw.strip())
                        except:
                            try:
                                json_obj = extract_json_block(raw)
                            except:
                                json_obj = None
                        if isinstance(json_obj, dict):
                            all_json_objects.append(json_obj)
                            name = json_obj.get("name") or json_obj.get("id") or f"item {i+1}"
                            all_positives.append(f"{name}")
                            all_raw.append(raw)
                        elif isinstance(json_obj, list):
                            for item in json_obj:
                                if isinstance(item, dict):
                                    all_json_objects.append(item)
                            all_positives.append(f"Chunk {i+1}: array with {len(json_obj)} items")
                            all_raw.append(raw)
                        else:
                            _log.warning(f"Chunk {chunk_label}: failed to extract valid JSON")
                            all_raw.append(raw)
                            continue
                    else:
                        if batch_strategy == "continuation" and all_positives and pos:
                            last_prev = all_positives[-1][-200:] if all_positives[-1] else ""
                            if last_prev and pos[:100] in last_prev:
                                for k in range(100, 10, -10):
                                    if pos[:k] in all_positives[-1][-300:]:
                                        pos = pos[k:].lstrip(" ,.;:\n")
                                        break
                        all_positives.append(pos)
                        if neg:
                            all_negatives.append(neg)
                        all_raw.append(raw)
                        if batch_strategy in ["continuation", "chaptered"]:
                            accumulated_text += (batch_separator if accumulated_text else "") + pos
                            if len(accumulated_text) > 12000:
                                accumulated_text = accumulated_text[:1000] + "\n\n[...middle truncated...]\n\n" + accumulated_text[-7000:]

                except Exception as e:
                    _log.error("❌ Chunk %s failed: %s", chunk_label, e)
                    continue
                finally:
                    if pbar:
                        pbar.update(1)

            if batch_strategy == "continuation":
                final_positive = batch_separator.join(all_positives)
                unique_negs = list(dict.fromkeys(n for n in all_negatives if n))
                final_negative = unique_negs[0] if unique_negs else ""
                final_raw = f"\n\n--- PART BOUNDARY ---\n\n".join(all_raw)
            elif batch_strategy == "chaptered":
                chaptered = []
                for idx, p in enumerate(all_positives):
                    if not p.lower().startswith(f"chapter {idx+1}"):
                        chaptered.append(f"Chapter {idx+1}:\n{p}")
                    else:
                        chaptered.append(p)
                final_positive = batch_separator.join(chaptered)
                unique_negs = list(dict.fromkeys(n for n in all_negatives if n))
                final_negative = ", ".join(unique_negs) if unique_negs else ""
                final_raw = "\n\n---\n\n".join(all_raw)
            elif batch_strategy == "json_array":
                if all_json_objects:
                    final_json_array = json.dumps(all_json_objects, indent=2)
                    cleaned, is_valid, note = _ensure_valid_json_only(final_json_array)
                    if is_valid:
                        final_json_array = cleaned
                    final_positive = f"```json\n{final_json_array}\n```"
                    final_negative = ""
                    final_raw = f"Generated {len(all_json_objects)} objects as valid JSON array:\n\n" + "\n\n---\n\n".join(all_raw)
                else:
                    fallback_text = batch_separator.join(all_positives) if all_positives else "[]"
                    cleaned, is_valid, note = _ensure_valid_json_only(fallback_text)
                    final_positive = f"```json\n{cleaned}\n```" if is_valid else batch_separator.join(all_positives) if all_positives else "```json\n[]\n```"
                    final_negative = ""
                    final_raw = "\n\n---\n\n".join(all_raw)
            else:
                final_positive = batch_separator.join(all_positives)
                unique_negs = list(dict.fromkeys(n for n in all_negatives if n))
                final_negative = ", ".join(unique_negs) if unique_negs else ""
                final_raw = "\n\n---\n\n".join(all_raw)

            # Filter thinking for final combined as well
            if filter_thinking:
                final_positive = filter_thinking(final_positive, True)
                final_negative = filter_thinking(final_negative, True)
                final_raw = filter_thinking(final_raw, True)

            final_positive = _append_smart(final_positive, append_to_positive)
            final_negative = _append_smart(final_negative, append_to_negative)
            # Filter thinking tags if enabled - available on all Ollama nodes
            try:
                if filter_thinking:
                    if 'positive_prompt' in locals() and isinstance(positive_prompt, str):
                        positive_prompt = _apply_thinking_filter(positive_prompt, True)
                    if 'negative_prompt' in locals() and isinstance(negative_prompt, str):
                        negative_prompt = _apply_thinking_filter(negative_prompt, True)
                    if 'enhanced_prompt' in locals() and isinstance(enhanced_prompt, str):
                        enhanced_prompt = _apply_thinking_filter(enhanced_prompt, True)
                    if 'generated_text' in locals() and isinstance(generated_text, str):
                        generated_text = _apply_thinking_filter(generated_text, True)
                    if 'generated' in locals() and isinstance(generated, str):
                        generated = _apply_thinking_filter(generated, True)
                    if 'result' in locals() and isinstance(result, str):
                        result = _apply_thinking_filter(result, True)
                    if 'video_prompt' in locals() and isinstance(video_prompt, str):
                        video_prompt = _apply_thinking_filter(video_prompt, True)
                    if 'final_prompt' in locals() and isinstance(final_prompt, str):
                        final_prompt = _apply_thinking_filter(final_prompt, True)
            except Exception:
                pass

            return final_positive, final_negative, final_raw

        # Single-pass mode
        estimated_input_tokens = (len(system) + len(user_input)) // 4
        if estimated_input_tokens + num_predict > num_ctx:
            _log.warning("⚠️ Context window may truncate: input(~%d) + num_predict(%d) > num_ctx(%d)", estimated_input_tokens, num_predict, num_ctx)

        _log.info("▶ Generating with %s | mode=%s | temp=%.2f | ctx=%d | think=%s | filter_thinking=%s", model, mode, temperature, num_ctx, think_mode, filter_thinking)
        pos, neg, raw = _generate_chunk(
            base_url=base_url, model=model, system=system, user_input=user_input,
            temperature=temperature, ollama_model=ollama_model, num_predict=num_predict,
            num_ctx=num_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, think=think,
        )

        if filter_thinking:
            pos = filter_thinking(pos, True)
            neg = filter_thinking(neg, True)
            raw = filter_thinking(raw, True)

        raw_stripped = raw.strip()
        is_json_like = (
            "photo_zones" in raw_stripped or "prompt_pieces" in raw_stripped
            or raw_stripped.startswith("```json") or raw_stripped.startswith("```")
            or raw_stripped.startswith("{") or raw_stripped.startswith("[")
        )
        if is_json_like:
            cleaned, is_valid, note = _ensure_valid_json_only(raw)
            if is_valid:
                if raw_stripped.startswith("```"):
                    pos = f"```json\n{cleaned}\n```"
                else:
                    pos = f"```json\n{cleaned}\n```"
                _log.info(f"✓ JSON cleanup: valid after {note}")

        pos = _append_smart(pos, append_to_positive)
        neg = _append_smart(neg, append_to_negative)
        # Filter thinking tags if enabled - available on all Ollama nodes
        try:
            if filter_thinking:
                if 'positive_prompt' in locals() and isinstance(positive_prompt, str):
                    positive_prompt = _apply_thinking_filter(positive_prompt, True)
                if 'negative_prompt' in locals() and isinstance(negative_prompt, str):
                    negative_prompt = _apply_thinking_filter(negative_prompt, True)
                if 'enhanced_prompt' in locals() and isinstance(enhanced_prompt, str):
                    enhanced_prompt = _apply_thinking_filter(enhanced_prompt, True)
                if 'generated_text' in locals() and isinstance(generated_text, str):
                    generated_text = _apply_thinking_filter(generated_text, True)
                if 'generated' in locals() and isinstance(generated, str):
                    generated = _apply_thinking_filter(generated, True)
                if 'result' in locals() and isinstance(result, str):
                    result = _apply_thinking_filter(result, True)
                if 'video_prompt' in locals() and isinstance(video_prompt, str):
                    video_prompt = _apply_thinking_filter(video_prompt, True)
                if 'final_prompt' in locals() and isinstance(final_prompt, str):
                    final_prompt = _apply_thinking_filter(final_prompt, True)
        except Exception:
            pass

        return pos, neg, raw


# ── Single Prompt Builder (unchanged but with thinking filter added) ──
_SINGLE_PROMPT_SYSTEM = """You are an expert AI image prompt engineer.
Create ONE production-ready image generation prompt from the user's input.

Respond with ONLY valid JSON. Do not include markdown, explanations, or extra text.

Exact JSON object:
{
  "positive_prompt": "single complete prompt ready for an image model",
  "negative_prompt": "concise negative prompt or empty string",
  "prompt_tags": "comma-separated tag version of the positive prompt",
  "summary": "short human-readable summary"
}

Rules:
- Generate exactly one prompt, not a list or photoset.
- Keep the subject and user intent intact.
- Add useful camera, lighting, composition, style, material, and quality details.
- Do not invent identity-sensitive details unless requested.
- If a locked prefix is provided by the user, copy it verbatim at the start of positive_prompt.
- If a negative base is provided by the user, include it in negative_prompt.
"""


def _parse_single_prompt_output(raw: str) -> tuple[str, str, str, str]:
    parsed = extract_json_block(raw)
    if isinstance(parsed, dict):
        positive = str(parsed.get("positive_prompt") or parsed.get("positive") or parsed.get("prompt") or "").strip()
        negative = str(parsed.get("negative_prompt") or parsed.get("negative") or "").strip()
        tags = str(parsed.get("prompt_tags") or parsed.get("tags") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        if positive:
            return positive, negative, tags, summary
    positive, negative = _parse_positive_negative(raw)
    return positive.strip(), negative.strip(), "", "Parsed from non-JSON model output"


class OllamaSinglePromptBuilder:
    CATEGORY = "ComfyUI-OMG/Prompt"
    FUNCTION = "build_single_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "prompt_tags", "summary", "raw_output")
    OUTPUT_NODE = True
    DESCRIPTION = 'Ollama Single Prompt Builder with thinking filter'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Ollama model"}),
                "concept": ("STRING", {"multiline": True, "default": "a cinematic portrait photo in soft window light", "tooltip": "Single idea to turn into one image-generation prompt"}),
            },
            "optional": {
                "model_style": (["photoreal", "flux", "sdxl", "qwen_image", "chatgpt_image", "gemini_image", "anime", "cinematic", "custom"], {"default": "photoreal", "tooltip": "Target style/model family"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Extra system guidance; strongest when model_style=custom - when you pick custom in dropdown, this block is used"}),
                "style_modifier": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional style, lens, artist, or mood hint"}),
                "locked_prefix": ("STRING", {"multiline": True, "default": "", "tooltip": "Text copied verbatim at the start of the positive prompt"}),
                "negative_base": ("STRING", {"multiline": True, "default": "", "tooltip": "Required negative text to include"}),
                "num_predict": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64, "tooltip": "Max tokens"}),
                "num_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 1024, "tooltip": "Context window"}),
                "temperature_override": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01}),
                "use_model_temperature": ("BOOLEAN", {"default": False}),
                "stream_to_console": ("BOOLEAN", {"default": False}),
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - available on all Ollama nodes"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, ollama_model: Dict[str, Any], concept: str, model_style: str = "photoreal", custom_system_prompt: str = "", style_modifier: str = "", locked_prefix: str = "", negative_base: str = "", num_predict: int = 1024, num_ctx: int = 8192, temperature_override: float = 0.45, use_model_temperature: bool = False, stream_to_console: bool = False, think_mode: str = "off", filter_thinking: bool = True, **kwargs):
        try:
            cfg = ollama_model or {}
            key = {
                "model": cfg.get("model", ""),
                "concept": concept[:500],
                "model_style": model_style,
                "custom_system_prompt": custom_system_prompt[:300],
                "style_modifier": style_modifier,
                "locked_prefix": locked_prefix[:100],
                "negative_base": negative_base[:100],
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "think_mode": think_mode,
                "filter_thinking": filter_thinking,
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def build_single_prompt(
        self,
        ollama_model: Dict[str, Any],
        concept: str,
        model_style: str = "photoreal",
        custom_system_prompt: str = "",
        style_modifier: str = "",
        locked_prefix: str = "",
        negative_base: str = "",
        num_predict: int = 1024,
        num_ctx: int = 8192,
        temperature_override: float = 0.45,
        use_model_temperature: bool = False,
        stream_to_console: bool = False,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ) -> tuple[str, str, str, str, str]:
        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")
        temperature = ollama_model.get("temperature", 0.7) if use_model_temperature else temperature_override
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

        from ...utils.system_prompt import compose_system_prompt
        style_note = "" if model_style == "custom" else f"Target style/model family: {model_style}."
        defined_system = "\n\n".join(part for part in [_SINGLE_PROMPT_SYSTEM, style_note] if part)
        system = compose_system_prompt(defined_system, ollama_model, custom_system_prompt)

        user_parts = [f"Concept:\n{concept.strip()}"]
        if style_modifier.strip():
            user_parts.append(f"Style modifier:\n{style_modifier.strip()}")
        if locked_prefix.strip():
            user_parts.append(f"Locked prefix, copy verbatim at the start of positive_prompt:\n{locked_prefix.strip()}")
        if negative_base.strip():
            user_parts.append(f"Negative base, include in negative_prompt:\n{negative_base.strip()}")
        user_parts.append("Return exactly one JSON object. No markdown. No list.")
        user_input = "\n\n".join(user_parts)

        tokens: list[str] = []
        try:
            from .ollama_client import generate_stream, OllamaConnectionError
            for token in generate_stream(
                base_url=base_url, model=model, prompt=user_input, system=system,
                temperature=temperature, top_p=ollama_model.get("top_p", 0.9),
                top_k=ollama_model.get("top_k", 40), repeat_penalty=ollama_model.get("repeat_penalty", 1.1),
                num_predict=num_predict, num_ctx=num_ctx, seed=ollama_model.get("seed", -1),
                response_format="json", think=think, keep_alive=keep_alive,
            ):
                tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()
        except Exception as exc:
            raise RuntimeError(f"Ollama single prompt generation failed: {exc}") from exc

        raw = "".join(tokens).strip()
        if not raw:
            raise RuntimeError("Model returned empty output.")

        if filter_thinking:
            raw = filter_thinking(raw, True)

        positive, negative, tags, summary = _parse_single_prompt_output(raw)

        if filter_thinking:
            positive = filter_thinking(positive, True)
            negative = filter_thinking(negative, True)
            tags = filter_thinking(tags, True)
            summary = filter_thinking(summary, True)

        if locked_prefix.strip() and positive and not positive.startswith(locked_prefix.strip()):
            positive = _append_smart(locked_prefix.strip(), positive)
        if negative_base.strip():
            negative = _append_smart(negative_base.strip(), negative)
        # Filter thinking tags if enabled - available on all Ollama nodes
        try:
            if filter_thinking:
                if 'positive_prompt' in locals() and isinstance(positive_prompt, str):
                    positive_prompt = _apply_thinking_filter(positive_prompt, True)
                if 'negative_prompt' in locals() and isinstance(negative_prompt, str):
                    negative_prompt = _apply_thinking_filter(negative_prompt, True)
                if 'enhanced_prompt' in locals() and isinstance(enhanced_prompt, str):
                    enhanced_prompt = _apply_thinking_filter(enhanced_prompt, True)
                if 'generated_text' in locals() and isinstance(generated_text, str):
                    generated_text = _apply_thinking_filter(generated_text, True)
                if 'generated' in locals() and isinstance(generated, str):
                    generated = _apply_thinking_filter(generated, True)
                if 'result' in locals() and isinstance(result, str):
                    result = _apply_thinking_filter(result, True)
                if 'video_prompt' in locals() and isinstance(video_prompt, str):
                    video_prompt = _apply_thinking_filter(video_prompt, True)
                if 'final_prompt' in locals() and isinstance(final_prompt, str):
                    final_prompt = _apply_thinking_filter(final_prompt, True)
        except Exception:
            pass

        return positive, negative, tags, summary, raw
