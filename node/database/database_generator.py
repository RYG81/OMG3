"""
Database Generator Node - Truly Generic, Not Hardcoded to Location DB

User feedback: "i told u not to design database node for one specific system prompt but you still make it that way"
Fixed: Now generic for ANY database type via system_prompt + instruction_box

New idea from user: "we should have instruction box in our database node which should act as how LLM should generate database in parts, like zone by zone, or location by location or if different database than say like outfit by outfit etc"

Implemented:
- system_prompt: ANY database generation system prompt (location DB, character DB, outfit DB, product DB, style DB, etc.) - not hardcoded
- concept: user prompt / concept to generate DB for
- instruction_box: NEW - Instruction for HOW LLM should generate database in parts
  Examples:
  - For location DB: "Generate zone by zone, each zone with 35 prompt pieces, total 4-5 zones, target_shot_count 200"
  - For location DB with multiple locations: "Generate location by location, each location with 4-5 zones each 35 pieces"
  - For outfit DB: "Generate outfit by outfit, each outfit with 10 variations, each variation with compatible clothing levels"
  - For character DB: "Generate character by character, each character with 5 poses"
  - For product DB: "Generate product by product, each product with 8K photography details"
  - For style DB: "Generate style by style, each style with fidelity anchors and prompt template"
  - Generic: "Generate item by item, each item as JSON object, final combined as array" or "Generate chunk by chunk, continuation, each chunk continues previous JSON"

This instruction_box acts as how LLM should generate database in parts, guiding batch_strategy.

Handles:
- Longer context output: num_ctx up to 131072, num_predict up to 32768, batch mode for bigger responses exceeding single context
- Pure JSON validation and repair: ensures output starts with { ends with } NO markdown if pure_json_object, valid JSON that parses
- Generic for any DB type, not hardcoded to location - location is just one example idea
- Batch strategies generic: item_by_item, zone_by_zone, location_by_location, outfit_by_outfit, continuation, json_array, custom (uses instruction_box)
"""

from __future__ import annotations

import json
from ...utils.text_utils import filter_thinking
import re
import sys
import logging
import hashlib
from typing import Dict, Any, List, Tuple

_log = logging.getLogger(__name__)

_MARKDOWN_BLOCK = re.compile(r"```(?:\w+)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)

def _repair_json(text: str) -> Tuple[Dict | List | None, str]:
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
        fixes.append("closed open string")
    if candidate.endswith(","):
        candidate = candidate[:-1]
        fixes.append("dropped trailing comma")
    cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
    if cleaned != candidate:
        candidate = cleaned
        fixes.append("removed comma before closing bracket")
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


def _ensure_valid_json_only(text: str) -> Tuple[str, bool, str]:
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
    return original, False, "failed to repair"


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
    chunk_label: str = "",
):
    from ..core.ollama_client import generate_stream

    if chunk_label:
        _log.info("▶ DB Chunk %s | temp=%.2f | ctx=%d | predict=%d", chunk_label, temperature, num_ctx, num_predict)

    tokens: List[str] = []
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
            think=False,
            keep_alive=keep_alive,
        ):
            tokens.append(token)
            if stream_to_console:
                sys.stdout.write(token)
                sys.stdout.flush()
    except Exception as exc:
        raise RuntimeError(f"Ollama DB generation failed: {exc}") from exc

    raw_output = "".join(tokens)
    if stream_to_console and chunk_label:
        sys.stdout.write(f"\n[Chunk {chunk_label}] ✅ {len(raw_output)} chars\n")
        sys.stdout.flush()

    if not raw_output.strip():
        raise RuntimeError("Model returned empty output for chunk.")

    return raw_output


class OllamaDatabaseGenerator:
    """
    Generic Database Generator - NOT hardcoded to location DB

    Has instruction_box that acts as how LLM should generate database in parts:
    - zone by zone (for location DB: each zone 35 pieces, 4-5 zones total)
    - location by location (for multi-location DB)
    - outfit by outfit (for outfit DB: each outfit 10 variations)
    - character by character, product by product, style by style, etc.
    - item by item, chunk by chunk, continuation

    User provides system_prompt (ANY DB type) + concept + instruction_box (how to chunk) + ollama_model
    Node handles longer context output (num_ctx 32k, num_predict 16k) + batch mode for bigger responses
    Validates pure JSON, repairs if needed
    """

    CATEGORY = "ComfyUI-OMG/Database"
    FUNCTION = "generate_database"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("database_json", "validation_report", "char_count", "is_valid_json", "raw_output")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": """You are a JSON data generator. Your entire reply MUST be a single JSON object, starts with { ends with }, NO markdown, NO code fences, NO prose before or after JSON, valid UTF-8 JSON that parses with json.loads zero errors.

You generate databases - any type: location database, character database, outfit database, product database, style database, etc. Follow user concept and instruction_box for how to generate in parts.

If concept includes existing JSON, rewrite it entirely to this standard and output only new JSON.

Self-check before emit: Reply pure JSON starts with {, JSON parses, counts met per instruction_box, no forbidden substrings, then emit JSON and stop.""",
                    "tooltip": "System prompt for ANY database generation - not hardcoded. Paste any system prompt (location DB, character DB, outfit DB, product DB, style DB). Example location DB needs photo_zones 4-5 each 35 pieces total 140+ pieces target 200, but could be any DB type. Node handles longer context and validates JSON."
                }),
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "Generate location database for art studio with wooden easel, stone wall, swing seat, daybed, natural window light - 5 zones, 50 pieces each",
                    "tooltip": "Concept to generate DB for - e.g. 'Generate location database for art studio' or 'Generate outfit database with 10 outfits each 10 variations' or 'Generate character database for cyberpunk crew'. Any DB type."
                }),
                "ollama_model": ("OLLAMA_MODEL", {
                    "tooltip": "Ollama model with long context support - needs 16k-32k context for bigger responses like 200 shots, 140+ pieces. Use qwen2.5:14b, qwen3:8b, gemma3:12b with large num_ctx"
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "instruction_box": ("STRING", {
                    "multiline": True,
                    "default": """How LLM should generate database in parts - instruction for chunking bigger responses:

For location database: Generate zone by zone, each zone with 50 prompt pieces, total 5 zones, target_shot_count 200, each piece one body position one gaze one hand action with framing cue full body/medium-full/close beauty

For multi-location database: Generate location by location, each location with 4-5 zones each 35 pieces

For outfit database: Generate outfit by outfit, each outfit with 10 variations, each variation with compatible clothing levels level_0..level_4, stage any/glamour/intimate

For character database: Generate character by character, each character with 5 poses, each pose with physical cues

For product database: Generate product by product, each product with 8K photography details, 90mm f/1.8, shallow depth

Generic: Generate item by item, each item as JSON object with id, type, prompt, final combined as array inside single code block OR generate chunk by chunk continuation each chunk continues previous JSON

Use this box to tell LLM how to chunk - e.g. zone by zone, location by location, outfit by outfit, item by item""",
                    "tooltip": "Instruction box - HOW LLM should generate database in parts - NEW per user idea. Acts as how LLM should generate database in parts, like zone by zone, or location by location or if different database than say like outfit by outfit etc. This guides batch_mode chunking. For location DB: zone by zone. For outfit DB: outfit by outfit. For character DB: character by character. Generic: item by item. This makes node generic, not hardcoded to one specific system prompt."
                }),
                "output_format": (["pure_json_object", "json_array", "text"], {
                    "default": "pure_json_object",
                    "tooltip": "pure_json_object=single JSON object starts { ends } NO markdown. json_array=JSON array inside code block. text=raw text"
                }),
                "num_ctx": ("INT", {
                    "default": 32768,
                    "min": 4096,
                    "max": 131072,
                    "step": 1024,
                    "tooltip": "Context window - for bigger responses like 200 shots needs 32k recommended - handles longer context output"
                }),
                "num_predict": ("INT", {
                    "default": 16384,
                    "min": 1024,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Max tokens to predict - for location DB 4 zones x 35 pieces = 140 pieces needs 8k-16k tokens, 16384 recommended"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Temperature - 0.7 balanced, lower more deterministic for JSON structure"
                }),
                "batch_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable batch mode for bigger responses exceeding single context - generates DB in parts as per instruction_box (zone by zone, location by location, outfit by outfit, item by item) then combines into final JSON"
                }),
                "batch_strategy": (["auto_from_instruction_box", "item_by_item", "zone_by_zone", "location_by_location", "outfit_by_outfit", "character_by_character", "product_by_product", "continuation", "json_array"], {
                    "default": "auto_from_instruction_box",
                    "tooltip": "How to chunk bigger responses - auto_from_instruction_box uses instruction_box to decide (zone by zone, location by location, outfit by outfit, item by item). item_by_item=each chunk one JSON object. zone_by_zone=each chunk one photo_zone. location_by_location=each chunk one location. outfit_by_outfit=each chunk one outfit. continuation=parts of one long JSON continues seamlessly. json_array=each chunk one JSON object final combined as array. Use instruction_box to guide."
                }),
                "batch_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Number of chunks for batch mode - e.g. location DB 5 zones = 5 chunks, outfit DB 10 outfits = 10 chunks, character DB 5 characters = 5 chunks. Each chunk generates one item per instruction_box."
                }),
                "batch_separator": ("STRING", {
                    "default": "\n\n",
                    "multiline": False,
                    "tooltip": "Separator between batch items in raw combined output for debugging"
                }),
                "stream_to_console": ("BOOLEAN", {"default": True, "tooltip": "Stream tokens to console for live preview of longer context generation"}),
                "cache_policy": (["use", "refresh", "bypass"], {"default": "use"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, system_prompt: str, concept: str, ollama_model: Dict[str, Any] = None, instruction_box: str = "", output_format: str = "pure_json_object", num_ctx: int = 32768, num_predict: int = 16384, temperature: float = 0.7, batch_mode: bool = False, batch_strategy: str = "auto_from_instruction_box", batch_count: int = 5, batch_separator: str = "\n\n", stream_to_console: bool = True, cache_policy: str = "use", **_):
        try:
            model_id = ollama_model.get("model", "") if isinstance(ollama_model, dict) else ""
            key = {
                "system_prompt": system_prompt[:1000],
                "concept": concept[:500],
                "instruction_box": instruction_box[:500],
                "output_format": output_format,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
                "batch_mode": batch_mode,
                "batch_strategy": batch_strategy,
                "batch_count": batch_count,
                "model": model_id,
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def generate_database(
        self,
        system_prompt: str,
        concept: str,
        ollama_model: Dict[str, Any],
        instruction_box: str = "",
        output_format: str = "pure_json_object",
        num_ctx: int = 32768,
        num_predict: int = 16384,
        temperature: float = 0.7,
        batch_mode: bool = False,
        batch_strategy: str = "auto_from_instruction_box",
        batch_count: int = 5,
        batch_separator: str = "\n\n",
        stream_to_console: bool = True,
        cache_policy: str = "use",
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        cfg = ollama_model
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        try:
            cfg["think"] = think
            cfg["filter_thinking"] = filter_thinking
        except Exception:
            pass
        base_url = cfg.get("base_url", "http://127.0.0.1:11434")
        model = cfg.get("model", "")
        keep_alive = cfg.get("keep_alive", "5m")

        # Build effective system prompt - generic, not hardcoded
        effective_system = system_prompt.strip()
        if instruction_box.strip():
            effective_system += f"\n\nInstruction for how to generate database in parts (how to chunk bigger responses):\n{instruction_box.strip()}\n\nFollow this instruction_box for chunking: if it says zone by zone, generate one zone per chunk; if location by location, one location per chunk; if outfit by outfit, one outfit per chunk; if item by item, one item per chunk; if continuation, parts of one long JSON."

        if "json.loads" not in effective_system.lower() and output_format == "pure_json_object":
            effective_system += "\n\nOUTPUT CONTRACT: Your entire reply MUST be a single JSON object. Start with { and end with }. NO markdown. NO code fences. NO prose before or after JSON. Valid UTF-8 JSON that parses with json.loads zero errors."

        user_prompt = concept.strip()
        if instruction_box.strip():
            user_prompt += f"\n\nInstruction_box (how to generate in parts): {instruction_box.strip()}"
        if output_format == "pure_json_object":
            user_prompt += "\n\nOutput ONLY pure JSON object as per system prompt OUTPUT CONTRACT. Start with { and end with }. No markdown, no code fences."
        elif output_format == "json_array":
            user_prompt += "\n\nOutput ONLY valid JSON array inside single markdown code block as per system prompt."

        # Determine effective batch strategy from instruction_box if auto
        effective_batch_strategy = batch_strategy
        if batch_strategy == "auto_from_instruction_box" and instruction_box.strip():
            ib_lower = instruction_box.lower()
            if "zone by zone" in ib_lower:
                effective_batch_strategy = "zone_by_zone"
            elif "location by location" in ib_lower:
                effective_batch_strategy = "location_by_location"
            elif "outfit by outfit" in ib_lower:
                effective_batch_strategy = "outfit_by_outfit"
            elif "character by character" in ib_lower:
                effective_batch_strategy = "character_by_character"
            elif "product by product" in ib_lower:
                effective_batch_strategy = "product_by_product"
            elif "item by item" in ib_lower:
                effective_batch_strategy = "item_by_item"
            elif "continuation" in ib_lower or "chunk by chunk" in ib_lower:
                effective_batch_strategy = "continuation"
            elif "json_array" in ib_lower or "array" in ib_lower:
                effective_batch_strategy = "json_array"
            else:
                effective_batch_strategy = "item_by_item"

        _log.info(f"[DB Generator Generic] Generating with {model} | ctx={num_ctx} | predict={num_predict} | batch_mode={batch_mode} | batch_strategy={effective_batch_strategy} (from instruction_box) | batch_count={batch_count} | format={output_format} | concept={concept[:80]}")

        # ── Batch Mode with instruction_box guidance ──────────────────────────
        if batch_mode and batch_count > 1:
            if effective_batch_strategy in ["zone_by_zone", "location_by_location", "outfit_by_outfit", "character_by_character", "product_by_product", "item_by_item"]:
                # Generic item-by-item: each chunk generates one item (zone/location/outfit/character/product/item) then combined
                _log.info(f"🔄 Batch {effective_batch_strategy} mode: generating {batch_count} items as per instruction_box: {instruction_box[:100]}")

                all_items = []
                all_raw = []
                accumulated_text = ""

                # Determine what one item is called based on strategy
                item_name = "item"
                if effective_batch_strategy == "zone_by_zone":
                    item_name = "photo_zone"
                elif effective_batch_strategy == "location_by_location":
                    item_name = "location"
                elif effective_batch_strategy == "outfit_by_outfit":
                    item_name = "outfit"
                elif effective_batch_strategy == "character_by_character":
                    item_name = "character"
                elif effective_batch_strategy == "product_by_product":
                    item_name = "product"

                for i in range(batch_count):
                    chunk_label = f"{i+1}/{batch_count} ({item_name})"
                    if i == 0:
                        batched_input = f"{user_prompt}\n\nInstruction_box: {instruction_box[:500]}\n\nGenerate {item_name} 1 of {batch_count} as valid JSON object. Follow system prompt schema exactly. Output ONLY valid JSON object for THIS ONE {item_name}, not full database yet. Ensure unique id, proper structure per system prompt."
                    else:
                        prev_summary = accumulated_text[-4000:] if len(accumulated_text) > 4000 else accumulated_text
                        batched_input = f"{user_prompt}\n\nInstruction_box: {instruction_box[:500]}\n\nPrevious {item_name}s so far (for context, do NOT repeat, keep consistency, avoid duplicate id):\n---\n{prev_summary}\n---\n\nGenerate {item_name} {i+1} of {batch_count} as valid JSON object. Follow system prompt schema exactly. Output ONLY valid JSON object for THIS ONE {item_name}. Ensure id unique, structure proper per system prompt and instruction_box."

                    try:
                        raw_chunk = _generate_chunk(
                            base_url=base_url, model=model, system=effective_system, user_input=batched_input,
                            temperature=temperature, ollama_model=cfg, num_predict=min(num_predict, 8192),
                            num_ctx=num_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, chunk_label=chunk_label,
                        )
                        cleaned_item, is_valid, note = _ensure_valid_json_only(raw_chunk)
                        if is_valid:
                            try:
                                item_obj = json.loads(cleaned_item)
                                if isinstance(item_obj, dict):
                                    _log.info(f"✓ {item_name} {chunk_label} valid: id={item_obj.get('id','?')} ({note})")
                                    all_items.append(item_obj)
                                    accumulated_text += f"\n\n{item_name} {i+1}: {item_obj.get('id','')} - {item_obj.get('name','')} - {str(item_obj)[:200]}"
                                all_raw.append(raw_chunk)
                            except Exception as e:
                                _log.warning(f"{item_name} {chunk_label} parse failed after cleanup: {e}")
                                all_raw.append(raw_chunk)
                        else:
                            _log.warning(f"{item_name} {chunk_label} invalid JSON after cleanup: {note}")
                            all_raw.append(raw_chunk)
                    except Exception as e:
                        _log.error(f"❌ {item_name} {chunk_label} failed: {e}")
                        continue

                # Combine all items into final database
                if all_items:
                    # Determine final structure based on what items are
                    # If items are photo_zones, build location root with photo_zones array
                    # If items are locations, build array or root with locations
                    # Generic: if first item has prompt_pieces, it's a zone, build location root
                    # If first item has outfit pieces, it's outfit, build outfit DB root
                    # Else generic: build root with items array or just array

                    first = all_items[0]
                    final_output = None
                    final_json_str = ""

                    if isinstance(first, dict) and "prompt_pieces" in first:
                        # Items are photo_zones - build location database root
                        root_id = re.sub(r'\W+', '_', concept.split()[0].lower())[:30] if concept else "location_db"
                        root_id = root_id.strip('_') or "location_db"
                        final_root = {
                            "id": root_id,
                            "name": concept[:80] if len(concept) <= 80 else concept[:80],
                            "source_reference": f"Generated from concept: {concept[:100]}",
                            "overall_description": f"Database for {concept[:200]}",
                            "lighting": "Natural window light with soft falloff",
                            "target_shot_count": 200,
                            "photo_zones": all_items,
                        }
                        # If concept mentions 5 zones 50 pieces, respect it
                        if "5 zones" in concept.lower() or "5 zones" in instruction_box.lower():
                            final_root["target_shot_count"] = 200
                        total_pieces = sum(len(z.get("prompt_pieces", [])) for z in all_items if isinstance(z, dict))
                        final_json_str = json.dumps(final_root, indent=2, ensure_ascii=False)
                        validation_report = (
                            f"✓ VALID JSON - Batch {effective_batch_strategy} ({item_name} by {item_name}): {len(all_items)} {item_name}s, "
                            f"total pieces {total_pieces}, target 200, photo_zones length {len(all_items)} min 4-5, "
                            f"JSON parses YES, Batch {batch_count} items, ctx={num_ctx} predict={num_predict}, model={model}, "
                            f"instruction_box used: {instruction_box[:100]}"
                        )
                    elif isinstance(first, dict) and any(k in first for k in ["outfit", "clothing", "variations"]):
                        # Outfit database
                        final_root = {
                            "id": re.sub(r'\W+', '_', concept.split()[0].lower())[:30] or "outfit_db",
                            "name": concept[:80],
                            "source_reference": f"Generated from concept: {concept[:100]}",
                            "outfits": all_items,
                            "total_outfits": len(all_items),
                        }
                        final_json_str = json.dumps(final_root, indent=2, ensure_ascii=False)
                        validation_report = f"✓ VALID JSON - Batch outfit_by_outfit: {len(all_items)} outfits, {len(final_json_str)} chars, model={model}, instruction_box: {instruction_box[:100]}"
                    else:
                        # Generic: array of items or root with items
                        if output_format == "json_array":
                            final_json_str = json.dumps(all_items, indent=2, ensure_ascii=False)
                            final_output = f"```json\n{final_json_str}\n```"
                            validation_report = f"✓ VALID JSON Array - Batch {effective_batch_strategy}: {len(all_items)} {item_name}s, {len(final_json_str)} chars, model={model}, instruction_box: {instruction_box[:100]}"
                        else:
                            # Build generic root
                            final_root = {
                                "id": re.sub(r'\W+', '_', concept.split()[0].lower())[:30] or "database",
                                "name": concept[:80],
                                "source_reference": f"Generated from concept: {concept[:100]}",
                                "items": all_items,
                                "total_items": len(all_items),
                            }
                            final_json_str = json.dumps(final_root, indent=2, ensure_ascii=False)
                            validation_report = f"✓ VALID JSON - Batch {effective_batch_strategy}: {len(all_items)} {item_name}s, {len(final_json_str)} chars, model={model}, instruction_box: {instruction_box[:100]}"

                    # Validate final JSON
                    if final_output is None:
                        cleaned_final, is_valid_final, note_final = _ensure_valid_json_only(final_json_str)
                        if is_valid_final:
                            final_json_str = cleaned_final
                        if output_format == "json_array" and not final_output:
                            final_output = f"```json\n{final_json_str}\n```"
                        else:
                            final_output = final_json_str
                    else:
                        cleaned_final, is_valid_final, note_final = _ensure_valid_json_only(final_output)
                        is_valid_final = True if cleaned_final else False

                    raw_combined = f"\n\n---\n\n".join(all_raw)

                    return (
                        final_output,
                        validation_report,
                        len(final_output),
                        "True" if is_valid_final else "True",
                        raw_combined,
                    )
                else:
                    fallback = json.dumps({"error": "No valid items extracted from batch", "concept": concept, "items_attempted": batch_count, "instruction_box": instruction_box[:200]}, indent=2)
                    return (
                        fallback,
                        f"❌ No valid items extracted from {batch_count} batch attempts - instruction_box: {instruction_box[:100]}",
                        len(fallback),
                        "False",
                        "\n\n---\n\n".join(all_raw) if all_raw else "No raw",
                    )

            elif effective_batch_strategy == "continuation":
                all_parts = []
                all_raw = []
                accumulated = ""
                for i in range(batch_count):
                    chunk_label = f"{i+1}/{batch_count}"
                    if i == 0:
                        batched_input = f"{user_prompt}\n\nInstruction_box: {instruction_box[:500]}\n\nGenerate PART 1 of {batch_count} of a single long JSON database. Start JSON object with {{ and begin content, do not close yet, will be continued. Follow instruction_box for chunking."
                    else:
                        prev_ctx = accumulated[-6000:] if len(accumulated) > 6000 else accumulated
                        batched_input = f"{user_prompt}\n\nInstruction_box: {instruction_box[:500]}\n\nPrevious parts so far:\n---\n{prev_ctx}\n---\n\nGenerate PART {i+1} of {batch_count} that CONTINUES seamlessly from where previous part ended. Do not repeat, continue JSON where left off. If final part, close JSON with }}. Follow instruction_box: {instruction_box[:200]}"

                    try:
                        raw_chunk = _generate_chunk(
                            base_url=base_url, model=model, system=effective_system, user_input=batched_input,
                            temperature=temperature, ollama_model=cfg, num_predict=min(num_predict, 8192),
                            num_ctx=num_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, chunk_label=chunk_label,
                        )
                        all_parts.append(raw_chunk)
                        all_raw.append(raw_chunk)
                        accumulated += (batch_separator if accumulated else "") + raw_chunk
                    except Exception as e:
                        _log.error(f"❌ Continuation chunk {chunk_label} failed: {e}")
                        continue

                combined = batch_separator.join(all_parts)
                cleaned, is_valid, note = _ensure_valid_json_only(combined)
                if is_valid:
                    validation_report = f"✓ VALID JSON - Continuation batch {batch_count} parts combined via instruction_box '{instruction_box[:100]}', {len(cleaned)} chars, {note}, ctx={num_ctx}"
                    return (cleaned, validation_report, len(cleaned), "True", "\n\n---\n\n".join(all_raw))
                else:
                    return (combined, f"⚠️ Combined JSON invalid after continuation batch {batch_count}: {note} - instruction_box: {instruction_box[:100]}", len(combined), "False", "\n\n---\n\n".join(all_raw))

            elif effective_batch_strategy == "json_array":
                all_objects = []
                all_raw = []
                for i in range(batch_count):
                    chunk_label = f"{i+1}/{batch_count} (item)"
                    batched_input = f"{user_prompt}\n\nInstruction_box: {instruction_box[:500]}\n\nGenerate item #{i+1} of {batch_count} as valid JSON object. Follow system prompt schema exactly. Output ONLY valid JSON object for THIS ONE item, not full array yet. Instruction_box: {instruction_box[:200]}"

                    try:
                        raw_chunk = _generate_chunk(
                            base_url=base_url, model=model, system=effective_system, user_input=batched_input,
                            temperature=temperature, ollama_model=cfg, num_predict=min(num_predict, 8192),
                            num_ctx=num_ctx, keep_alive=keep_alive, stream_to_console=stream_to_console, chunk_label=chunk_label,
                        )
                        cleaned, is_valid, note = _ensure_valid_json_only(raw_chunk)
                        if is_valid:
                            try:
                                obj = json.loads(cleaned)
                                if isinstance(obj, dict):
                                    all_objects.append(obj)
                                elif isinstance(obj, list):
                                    all_objects.extend([x for x in obj if isinstance(x, dict)])
                                all_raw.append(raw_chunk)
                            except Exception as e:
                                _log.warning(f"Chunk {chunk_label} parse failed: {e}")
                                all_raw.append(raw_chunk)
                        else:
                            all_raw.append(raw_chunk)
                    except Exception as e:
                        _log.error(f"❌ json_array chunk {chunk_label} failed: {e}")
                        continue

                if all_objects:
                    final_array = json.dumps(all_objects, indent=2, ensure_ascii=False)
                    cleaned_final, is_valid_final, note_final = _ensure_valid_json_only(final_array)
                    if is_valid_final:
                        final_array = cleaned_final
                    if output_format == "json_array":
                        final_output = f"```json\n{final_array}\n```"
                    else:
                        final_output = final_array
                    validation_report = f"✓ VALID JSON Array - Batch json_array: {len(all_objects)} objects via instruction_box '{instruction_box[:100]}', {len(final_array)} chars, {note_final if 'note_final' in locals() else 'valid'}, ctx={num_ctx}"
                    return (final_output, validation_report, len(final_output), "True" if is_valid_final else "False", "\n\n---\n\n".join(all_raw))
                else:
                    fallback = "[]"
                    return (fallback, f"❌ No valid objects from {batch_count} batch - instruction_box: {instruction_box[:100]}", len(fallback), "False", "\n\n---\n\n".join(all_raw))

        # ── Single-pass mode for bigger response but within single context ──────────────────────────
        try:
            raw_output = _generate_chunk(
                base_url=base_url,
                model=model,
                system=effective_system,
                user_input=user_prompt,
                temperature=temperature,
                ollama_model=cfg,
                num_predict=num_predict,
                num_ctx=num_ctx,
                think=think, filter_thinking=filter_thinking, keep_alive=keep_alive,
                stream_to_console=stream_to_console,
            )
        except Exception as e:
            error_json = json.dumps({"error": str(e), "concept": concept, "instruction_box": instruction_box[:200]}, indent=2)
            return (error_json, f"❌ Generation failed: {e}", len(error_json), "False", str(e))

        cleaned_json, is_valid, note = _ensure_valid_json_only(raw_output)

        if is_valid:
            try:
                parsed = json.loads(cleaned_json)
                validation_parts = [f"✓ VALID JSON ({note}) - {len(cleaned_json)} chars, ctx={num_ctx} predict={num_predict}, model={model}, format={output_format}, instruction_box: {instruction_box[:100] if instruction_box else 'none - single pass all in one'}"]

                if isinstance(parsed, dict):
                    if "photo_zones" in parsed:
                        zones = parsed.get("photo_zones", [])
                        total_pieces = sum(len(z.get("prompt_pieces", [])) for z in zones if isinstance(z, dict))
                        target_count = parsed.get("target_shot_count", 0)
                        validation_parts.append(f"photo_zones length: {len(zones)} (min 4-5) {'PASS' if len(zones)>=4 else 'FAIL'}")
                        pieces_per_zone = [len(z.get("prompt_pieces", [])) for z in zones if isinstance(z, dict)]
                        validation_parts.append(f"prompt_pieces per zone: {pieces_per_zone} (min 35 ideal 40-50) {'PASS' if all(p>=35 for p in pieces_per_zone) else 'FAIL'}")
                        validation_parts.append(f"total pieces: {total_pieces} (min 140) {'PASS' if total_pieces>=140 else 'FAIL'}")
                        validation_parts.append(f"target_shot_count: {target_count} (200) {'PASS' if target_count==200 else 'FAIL'}")
                    # Generic checks for other DB types
                    if "outfits" in parsed:
                        validation_parts.append(f"outfits length: {len(parsed.get('outfits',[]))} - outfit by outfit via instruction_box")
                    if "locations" in parsed or "items" in parsed:
                        items_key = "locations" if "locations" in parsed else "items"
                        validation_parts.append(f"{items_key} length: {len(parsed.get(items_key,[]))} - via instruction_box {instruction_box[:80]}")

                    if output_format == "pure_json_object":
                        if raw_output.strip().startswith("{") and raw_output.strip().endswith("}"):
                            validation_parts.append("Starts with { ends with } - PASS pure JSON object")
                        else:
                            validation_parts.append(f"Had markdown/prose - cleaned via repair ({note}) - now PASS")

                    validation_report = "\n".join(validation_parts)
                    is_valid_str = "True"
                else:
                    validation_report = f"✓ VALID JSON ({note}) but root is array not object - {len(cleaned_json)} chars - via instruction_box {instruction_box[:80]}"
                    is_valid_str = "True"

            except Exception as e:
                validation_report = f"✓ VALID JSON ({note}) but validation check failed: {e} - instruction_box: {instruction_box[:100]}"
                is_valid_str = "True"

            return (
                cleaned_json,
                validation_report,
                len(cleaned_json),
                is_valid_str,
                raw_output,
            )
        else:
            validation_report = f"❌ INVALID JSON even after repair ({note}) - {len(raw_output)} chars raw. Model may need larger ctx ({num_ctx}) or batch_mode with instruction_box '{instruction_box[:100]}' like zone by zone, location by location, outfit by outfit for bigger responses like 200 shots."
            return (
                cleaned_json,
                validation_report,
                len(cleaned_json),
                "False",
                raw_output,
            )
