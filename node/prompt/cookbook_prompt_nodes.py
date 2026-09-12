"""
Cookbook Style Node - One node that handles 66 different styles with different input requirements, makes user life easy.

Problem: Each style.json in AI-Visual-Prompt-Cookbook has different environment_variables
Example Mono Noir: SUBJECT, SUBJECT_ACTION, PRODUCT_OR_PROP, LOCATION, BACKGROUND_ELEMENTS, MAIN_TEXT, SECONDARY_TEXT, ACCENT_SYMBOL, WARDROBE_STYLE, STYLE_FIDELITY_ANCHORS, SOURCE_CONTENT_TO_AVOID, ASPECT_RATIO (12 vars)
Example Market Brush Produce: might have SUBJECT, PRODUCE_TYPE, BACKGROUND_COLOR, TEXT_STYLE etc (different set)

Just adding a preset loader won't help - user still needs to figure out what variables to fill.

Solution: Design node accordingly - user gives ONE concept (few words), node auto-fills ALL required variables for selected style using technology (Ollama + style_fidelity_anchors + design_rules + examples).

Worth doing per skill:
- Fewer smarter nodes (one node handles 66 styles) vs 66 separate nodes
- Sane defaults that produce good result first run - concept "a tired architect" -> full prompt with all vars filled
- Tooltips on every input, short display name not wide
- Autogrow not needed here, but dynamic variable handling via Ollama is the tech
- Performance: cache styles, lazy import ollama_client, IS_CHANGED hashing
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import logging

_log = logging.getLogger(__name__)

STYLES_DIR = Path(__file__).parent.parent.parent / "presets" / "cookbook_styles"

_CACHE_STYLES_LIST = None
_CACHE_STYLES_DICT = None


def _load_all_styles():
    """Load all 66 style.json files, cache for performance"""
    global _CACHE_STYLES_LIST, _CACHE_STYLES_DICT
    if _CACHE_STYLES_LIST is not None and _CACHE_STYLES_DICT is not None:
        return _CACHE_STYLES_LIST, _CACHE_STYLES_DICT

    styles_list = []
    styles_dict = {}

    if not STYLES_DIR.exists():
        _log.warning(f"Cookbook styles dir not found: {STYLES_DIR}")
        _CACHE_STYLES_LIST = []
        _CACHE_STYLES_DICT = {}
        return [], {}

    for json_file in sorted(STYLES_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                slug = data.get("style_slug") or json_file.stem
                name = data.get("style_name") or slug
                # Build summary for dropdown
                summary = data.get("style_summary", "")[:120]
                display = f"{name} ({slug})"

                styles_list.append({
                    "slug": slug,
                    "name": name,
                    "display": display,
                    "summary": summary,
                    "file": json_file.name,
                    "data": data,
                })
                styles_dict[slug] = data
                styles_dict[display] = data
                styles_dict[name] = data
        except Exception as e:
            _log.warning(f"Failed to load style {json_file}: {e}")

    _CACHE_STYLES_LIST = styles_list
    _CACHE_STYLES_DICT = styles_dict
    _log.info(f"[Cookbook] Loaded {len(styles_list)} styles from {STYLES_DIR}")
    return styles_list, styles_dict


def _get_style_options():
    styles_list, _ = _load_all_styles()
    # Return display names for combo
    options = ["auto-infer (pick best for concept)"]
    for item in styles_list:
        options.append(item["display"])
    return options


def _find_style_by_selection(selection: str):
    _, styles_dict = _load_all_styles()
    # Direct hit
    if selection in styles_dict:
        return styles_dict[selection]
    # Try slug match
    for key, data in styles_dict.items():
        if isinstance(data, dict) and data.get("style_slug") == selection:
            return data
        if selection.lower() in key.lower():
            if isinstance(data, dict) and "style_name" in data:
                return data
    return None


def _extract_variables_needed(style_data: Dict[str, Any]) -> Dict[str, str]:
    """Extract environment_variables dict - different for each style"""
    return style_data.get("environment_variables", {})


class CookbookStyleAutoFiller:
    """
    ONE NODE that handles 66 different styles with different input requirements.

    User gives:
    - Style selection (66 options + auto-infer)
    - Concept (few words, e.g. "a tired architect" or "biker")

    Node does (technology makes life easy):
    - Loads selected style.json
    - Reads its environment_variables (different per style)
    - Auto-fills ALL required variables from single concept using Ollama + style_fidelity_anchors + design_rules + examples
    - If Ollama not available, uses concept-aware fallback template
    - Applies filled variables to prompt_template
    - Returns final prompt ready to send to image model

    Worth doing: User doesn't need to know what SUBJECT, MAIN_TEXT, etc. mean for each style. Just concept -> full prompt.
    """

    CATEGORY = "ComfyUI-OMG/Prompt/Cookbook"
    FUNCTION = "generate_cookbook_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("final_prompt", "negative_prompt", "variables_used_json", "style_info", "char_count")

    @classmethod
    def INPUT_TYPES(cls):
        style_options = _get_style_options()

        return {
            "required": {
                "style_selection": (style_options, {
                    "default": "auto-infer (pick best for concept)",
                    "tooltip": "Select from 66 cookbook styles - each has different input requirements (e.g. Mono Noir needs SUBJECT, MAIN_TEXT, etc. Market Brush needs PRODUCE_TYPE). Node auto-fills all from single concept, so you don't need to learn each style's vars."
                }),
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "a tired architect with silver hair studying blueprint late night",
                    "tooltip": "Few words allowed - main idea that will auto-fill ALL required variables for selected style. E.g. 'biker' or 'a tired architect with silver hair' - technology fills SUBJECT, ACTION, LOCATION, TEXT, etc. from this one concept."
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "ollama_model": ("OLLAMA_MODEL", {
                    "tooltip": "Optional Ollama model to intelligently auto-fill all variables from concept + style fidelity anchors + design rules. If missing, uses smart template fallback concept-aware."
                }),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "3:4 vertical", "4:5"], {
                    "default": "auto-infer",
                    "tooltip": "Aspect ratio - will auto-fill ASPECT_RATIO variable if style needs it. 16:9 landscape poster with subject right, headline left. 9:16 vertical poster with headline stacked upper-left."
                }),
                "custom_variables_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Optional override - JSON mapping of variable name to value to manually control any variable. e.g. {\"SUBJECT\": \"a tired architect\", \"MAIN_TEXT\": \"focus\\noutlasts\\nnoise.\"} - overrides auto-filled values. For advanced users who want manual control after auto-fill."
                }),
                "use_examples_as_few_shot": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use style's examples as few-shot inspiration for Ollama filling - improves auto-fill quality by showing how variables were filled before"
                }),
                "small_model_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For 9B or smaller models - lean prompt, 8192 ctx, lower temp for more deterministic variable filling"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, style_selection: str, concept: str, ollama_model: Dict[str, Any] = None, aspect_ratio: str = "auto-infer", custom_variables_json: str = "{}", use_examples_as_few_shot: bool = True, small_model_mode: bool = False,
        think_mode: str = "off",
        filter_thinking: bool = True,
        **kwargs,
    ):
        # Hash concept + style + custom vars + model for caching - per skill performance rules
        import hashlib, json
        try:
            model_id = ollama_model.get("model", "") if isinstance(ollama_model, dict) else ""
            key = {
                "style": style_selection,
                "concept": concept[:300],
                "aspect_ratio": aspect_ratio,
                "custom_vars": custom_variables_json[:500],
                "use_examples": use_examples_as_few_shot,
                "model": model_id,
                "small_mode": small_model_mode,
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def generate_cookbook_prompt(
        self,
        style_selection: str,
        concept: str,
        ollama_model: Dict[str, Any] = None,
        aspect_ratio: str = "auto-infer",
        custom_variables_json: str = "{}",
        use_examples_as_few_shot: bool = True,
        small_model_mode: bool = False,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        # Load styles
        styles_list, styles_dict = _load_all_styles()

        if not styles_list:
            return (
                f"Failed to load cookbook styles from {STYLES_DIR} - ensure 66 json files exist. Concept was: {concept}",
                "No styles loaded",
                "{}",
                "No styles found",
                0
            )

        # Select style
        if style_selection.startswith("auto-infer"):
            # Auto-pick best style for concept - simple heuristic
            concept_lower = concept.lower()
            # Heuristics for auto-infer
            if any(k in concept_lower for k in ["architect", "portrait", "person", "man", "woman", "tired"]):
                # Prefer portrait poster styles
                selected_style_data = None
                for item in styles_list:
                    if "portrait" in item["slug"] or "mono-noir" in item["slug"]:
                        selected_style_data = item["data"]
                        style_selection = item["display"]
                        break
                if not selected_style_data:
                    selected_style_data = styles_list[0]["data"]
            elif any(k in concept_lower for k in ["biker", "motorcycle", "fashion", "model"]):
                # Prefer fashion/poster
                selected_style_data = None
                for item in styles_list:
                    if "fashion" in item["slug"] or "supermodel" in item["slug"] or "streetwear" in item["slug"]:
                        selected_style_data = item["data"]
                        style_selection = item["display"]
                        break
                if not selected_style_data:
                    selected_style_data = styles_list[0]["data"]
            else:
                # Default to first style
                selected_style_data = styles_list[0]["data"]
                style_selection = styles_list[0]["display"]
        else:
            selected_style_data = _find_style_by_selection(style_selection)
            if not selected_style_data:
                # Fallback to first
                selected_style_data = styles_list[0]["data"]
                style_selection = styles_list[0]["display"]

        # Extract what variables this specific style needs
        env_vars = _extract_variables_needed(selected_style_data)
        style_fidelity_anchors = selected_style_data.get("style_fidelity_anchors", [])
        source_content_to_avoid = selected_style_data.get("source_content_to_avoid", [])
        design_rules = selected_style_data.get("design_rules", [])
        do_rules = selected_style_data.get("do", [])
        avoid_rules = selected_style_data.get("avoid", [])
        prompt_template = selected_style_data.get("prompt_template", "")
        negative_prompt_template = selected_style_data.get("negative_prompt", "")
        examples = selected_style_data.get("examples", [])
        style_summary = selected_style_data.get("style_summary", "")
        style_name = selected_style_data.get("style_name", style_selection)
        style_slug = selected_style_data.get("style_slug", "")

        # Parse custom overrides
        custom_overrides = {}
        try:
            if custom_variables_json.strip():
                custom_overrides = json.loads(custom_variables_json)
                if not isinstance(custom_overrides, dict):
                    custom_overrides = {}
        except Exception as e:
            _log.warning(f"Failed to parse custom_variables_json: {e}")
            custom_overrides = {}

        # Auto-fill variables from concept
        filled_vars = {}

        # Always include ASPECT_RATIO if required and provided
        if "ASPECT_RATIO" in env_vars:
            if aspect_ratio != "auto-infer":
                filled_vars["ASPECT_RATIO"] = aspect_ratio
            else:
                # Default based on style - check if style has aspect behavior
                filled_vars["ASPECT_RATIO"] = "16:9"  # default

        # For STYLE_FIDELITY_ANCHORS and SOURCE_CONTENT_TO_AVOID, use style's own anchors, not user concept
        if "STYLE_FIDELITY_ANCHORS" in env_vars:
            # Join anchors into string
            if isinstance(style_fidelity_anchors, list):
                filled_vars["STYLE_FIDELITY_ANCHORS"] = ", ".join(style_fidelity_anchors[:5])
            else:
                filled_vars["STYLE_FIDELITY_ANCHORS"] = str(style_fidelity_anchors)[:500]

        if "SOURCE_CONTENT_TO_AVOID" in env_vars:
            if isinstance(source_content_to_avoid, list):
                filled_vars["SOURCE_CONTENT_TO_AVOID"] = ", ".join(source_content_to_avoid[:3])
            else:
                filled_vars["SOURCE_CONTENT_TO_AVOID"] = str(source_content_to_avoid)[:500]

        # Remaining variables need to be filled from concept via Ollama or fallback
        remaining_vars = {k: v for k, v in env_vars.items() if k not in filled_vars and k not in custom_overrides}

        # Apply custom overrides first (advanced user manual control)
        for k, v in custom_overrides.items():
            filled_vars[k] = v

        # Auto-fill remaining via Ollama if available
        if remaining_vars and ollama_model:
            try:
                from ...ollama_client import generate as ollama_generate
                from ...utils.text_utils import extract_json_block

                # Build system prompt for variable filling
                variables_desc = "\n".join([f"- {var_name}: {var_desc}" for var_name, var_desc in remaining_vars.items()])

                fidelity_text = "\n".join([f"- {anchor}" for anchor in style_fidelity_anchors[:8]]) if style_fidelity_anchors else "No specific anchors"
                design_rules_text = "\n".join([f"- {rule}" for rule in design_rules[:8]]) if design_rules else "No specific design rules"

                examples_text = ""
                if use_examples_as_few_shot and examples:
                    # Take 2 examples as few-shot
                    few_shot_parts = []
                    for ex in examples[:2]:
                        case_name = ex.get("case_name", "example")
                        values = ex.get("values", {})
                        values_str = "\n".join([f"  {k} = {v}" for k, v in list(values.items())[:8]])
                        few_shot_parts.append(f"Example case '{case_name}':\n{values_str}")
                    examples_text = "\n\n".join(few_shot_parts)

                system_prompt = f"""You are a variable filler for cookbook style '{style_name}' ({style_slug}).

Style summary: {style_summary}

Style fidelity anchors (must preserve this style's look):
{fidelity_text}

Design rules:
{design_rules_text}

Do:
{chr(10).join([f"- {d}" for d in do_rules[:5]]) if do_rules else "- Follow style summary"}

Avoid:
{chr(10).join([f"- {a}" for a in avoid_rules[:5]]) if avoid_rules else "- Avoid breaking style fidelity"}

Variables to fill (different for each style - this style needs {len(remaining_vars)} variables):
{variables_desc}

{f"Examples of how variables were filled before (few-shot inspiration):\n{examples_text}" if examples_text else "No examples - fill creatively"}

Concept to use as source for filling: {concept}

Aspect ratio: {aspect_ratio}

Rules for filling:
- Use concept as main inspiration but adapt to style's required variables
- For SUBJECT: main person/character who anchors poster, from concept
- For SUBJECT_ACTION: quiet pose, intense gaze, paused task, reflective moment, or restrained movement relevant to concept
- For PRODUCT_OR_PROP: object/tool/instrument relevant to concept, or no prop
- For LOCATION: dim studio, late-night workspace, rehearsal room, tunnel, library corner, workshop, or minimal interior relevant to concept
- For BACKGROUND_ELEMENTS: soft charcoal gradient, blurred wall texture, deep negative space, etc. relevant to concept
- For MAIN_TEXT: three-line lowercase headline with short words and final period - CREATE original short headline from concept, not copy example, e.g. for architect concept "focus / outlasts / noise."
- For SECONDARY_TEXT: tiny optional supporting microcopy, date, or section label - e.g. "studio log 02:14"
- For ACCENT_SYMBOL: minimal dot, short rule, small plus, or no visible accent
- For WARDROBE_STYLE: dark plain clothing, structured jacket, workwear, athletic layer, or simple editorial styling relevant to concept
- For other variables: fill appropriately from concept, keep style fidelity, avoid source content to avoid
- Keep values concise: 1-10 words for most, except MAIN_TEXT which is 3 short lines
- Return ONLY valid JSON mapping variable name to value, e.g. {{"SUBJECT": "a tired architect with silver hair", "MAIN_TEXT": "focus\\noutlasts\\nnoise."}}
- No markdown, no explanation, only JSON
"""

                user_prompt = f"""Concept: {concept}

Fill these {len(remaining_vars)} variables for style '{style_name}':

{variables_desc}

Aspect ratio: {aspect_ratio}

Return JSON mapping variable name to value, no markdown, only JSON.
"""

                cfg = ollama_model

                think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

                try:

                    cfg["think"] = think

                    cfg["filter_thinking"] = filter_thinking

                except Exception:

                    pass
                num_ctx = 8192 if small_model_mode else cfg.get("num_ctx", 12288)
                raw = ollama_generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.4 if small_model_mode else cfg.get("temperature", 0.7),
                    num_ctx=num_ctx,
                    num_predict=1500 if small_model_mode else 2000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                )

                parsed = extract_json_block(raw)
                if isinstance(parsed, dict) and parsed:
                    # Merge auto-filled from LLM
                    for k, v in parsed.items():
                        # Only take if k is in remaining_vars (or case-insensitive match)
                        # Allow case-insensitive matching
                        found_key = None
                        for req_key in remaining_vars.keys():
                            if req_key.lower() == k.lower() or req_key == k:
                                found_key = req_key
                                break
                        if found_key:
                            filled_vars[found_key] = str(v)
                        elif k in env_vars:
                            filled_vars[k] = str(v)
                    _log.info(f"[Cookbook] Ollama filled {len(parsed)} variables for style {style_slug}")

            except Exception as e:
                _log.warning(f"[Cookbook] Ollama auto-fill failed for style {style_slug}: {e}, falling back to template")

        # Fallback for any remaining unfilled variables (no Ollama or Ollama failed) - concept-aware template filling
        for var_name, var_desc in remaining_vars.items():
            if var_name in filled_vars:
                continue

            var_lower = var_name.lower()
            concept_lower = concept.lower()

            # Smart fallback based on variable name
            if "subject" in var_lower:
                filled_vars[var_name] = concept[:100] if len(concept) <= 100 else concept[:100]
            elif "action" in var_lower:
                if "architect" in concept_lower:
                    filled_vars[var_name] = "studying a folded blueprint in a late-night pause"
                elif "biker" in concept_lower or "motorcycle" in concept_lower:
                    filled_vars[var_name] = "riding through busy city road, leaning into turn"
                elif "dance" in concept_lower or "party" in concept_lower:
                    filled_vars[var_name] = "dancing with dynamic posture, wide stance"
                else:
                    filled_vars[var_name] = "quiet pose, intense gaze, reflective moment"
            elif "product" in var_lower or "prop" in var_lower:
                if "architect" in concept_lower:
                    filled_vars[var_name] = "a rolled plan tube and a pencil held low"
                elif "biker" in concept_lower:
                    filled_vars[var_name] = "helmet and leather jacket"
                else:
                    filled_vars[var_name] = "minimal prop relevant to concept or no prop"
            elif "location" in var_lower:
                if "architect" in concept_lower:
                    filled_vars[var_name] = "a dim concrete studio after midnight"
                elif "biker" in concept_lower:
                    filled_vars[var_name] = "busy New York city road with dense traffic"
                else:
                    filled_vars[var_name] = "dim studio, late-night workspace, minimal interior"
            elif "background" in var_lower:
                filled_vars[var_name] = "soft charcoal wall gradient, blurred wall texture, deep negative space, faint window falloff"
            elif "main_text" in var_lower or var_name == "MAIN_TEXT":
                # Create 3-line lowercase headline from concept
                words = concept.split()[:6]
                if len(words) >= 3:
                    # Create 3 short lines with period at end
                    line1 = words[0].lower()
                    line2 = words[1].lower() if len(words) > 1 else "outlasts"
                    line3 = words[2].lower() + "." if len(words) > 2 else "noise."
                    filled_vars[var_name] = f"{line1}\n{line2}\n{line3}"
                else:
                    filled_vars[var_name] = "focus\noutlasts\nnoise."
            elif "secondary" in var_lower:
                filled_vars[var_name] = "studio log 02:14"
            elif "accent" in var_lower or "symbol" in var_lower:
                filled_vars[var_name] = "a tiny white plus"
            elif "wardrobe" in var_lower:
                if "architect" in concept_lower:
                    filled_vars[var_name] = "dark work jacket over a plain black shirt"
                elif "biker" in concept_lower:
                    filled_vars[var_name] = "black leather jacket, helmet, gloves"
                else:
                    filled_vars[var_name] = "dark plain clothing, structured jacket, workwear"
            elif "color" in var_lower:
                filled_vars[var_name] = "matte black and white"
            elif "brand" in var_lower:
                filled_vars[var_name] = concept.split()[0].title() if concept else "Brand"
            elif "object" in var_lower:
                filled_vars[var_name] = concept[:50]
            elif "landmark" in var_lower:
                filled_vars[var_name] = concept[:50]
            else:
                # Generic fallback - use var description as hint and concept as value
                filled_vars[var_name] = f"{concept[:80]} - {var_desc[:60]}"

        # Now apply filled variables to prompt_template
        final_prompt = prompt_template
        # Replace {VAR} and {VAR} style placeholders
        for var_name, var_value in filled_vars.items():
            # Replace {VAR} and {{VAR}} and { VAR } etc.
            # Template uses {VAR} like {SUBJECT}, {MAIN_TEXT} etc.
            final_prompt = final_prompt.replace(f"{{{var_name}}}", var_value)
            # Also handle lowercase? But spec says uppercase
            # Also handle with spaces?

        # Also ensure ASPECT_RATIO replacement if not already done
        if "{ASPECT_RATIO}" in final_prompt and "ASPECT_RATIO" not in filled_vars:
            final_prompt = final_prompt.replace("{ASPECT_RATIO}", aspect_ratio if aspect_ratio != "auto-infer" else "16:9")

        # Build style info
        style_info_dict = {
            "style_name": style_name,
            "style_slug": style_slug,
            "style_version": selected_style_data.get("style_version", ""),
            "style_summary": style_summary,
            "variables_needed": list(env_vars.keys()),
            "variables_needed_count": len(env_vars),
            "variables_filled": list(filled_vars.keys()),
            "variables_filled_count": len(filled_vars),
            "style_fidelity_anchors": style_fidelity_anchors[:5],
            "design_rules": design_rules[:5],
            "source_content_to_avoid": source_content_to_avoid[:3],
            "examples_count": len(examples),
            "concept_used": concept,
            "aspect_ratio_used": aspect_ratio,
            "custom_overrides": custom_overrides,
        }

        style_info = (
            f"Style: {style_name} ({style_slug}) v{style_info_dict['style_version']}\n"
            f"Summary: {style_summary}\n"
            f"Variables needed ({len(env_vars)}): {', '.join(list(env_vars.keys()))}\n"
            f"Variables filled ({len(filled_vars)}): {', '.join(list(filled_vars.keys()))}\n"
            f"Fidelity anchors: {', '.join([a[:60] for a in style_fidelity_anchors[:3]])}...\n"
            f"Design rules: {', '.join([r[:60] for r in design_rules[:2]])}...\n"
            f"Concept: {concept}\n"
            f"Aspect: {aspect_ratio} | Custom overrides: {len(custom_overrides)} | Examples used as few-shot: {len(examples) if use_examples_as_few_shot else 0}"
        )

        variables_used_json = json.dumps(filled_vars, indent=2, ensure_ascii=False)

        # Handle negative prompt as well - apply variables to it too
        final_negative = negative_prompt_template
        if final_negative:
            for var_name, var_value in filled_vars.items():
                final_negative = final_negative.replace(f"{{{var_name}}}", var_value)

        # If final prompt still has unfilled {VAR} placeholders, warn and leave them
        remaining_placeholders = re.findall(r'\{[A-Z_]+\}', final_prompt)
        if remaining_placeholders:
            _log.warning(f"[Cookbook] {len(remaining_placeholders)} placeholders remain unfilled for style {style_slug}: {remaining_placeholders[:5]}")

        return (
            final_prompt,
            final_negative,
            variables_used_json,
            style_info,
            len(final_prompt),
        )


NODE_CLASS_MAPPINGS = {
    "CookbookStyleAutoFiller": CookbookStyleAutoFiller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CookbookStyleAutoFiller": "Cookbook - Auto (66 Styles)",
}
