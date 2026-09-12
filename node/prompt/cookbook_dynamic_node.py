"""
Cookbook Dynamic Inputs Node - One node with JS widget where input section handled by all system prompts from github repo

Design per user: node has dropdown for various system prompt and can be picked by user, as user changes item in dropdown we change inputs.

Worth doing: User life easy, use technology as much as possible - not 66 nodes, one dynamic node that changes inputs per style.

Implementation:
- Python: defines style_selection dropdown (66 styles) + hidden dynamic_vars_json that stores JSON from JS widget
- JS: beforeRegisterNodeDef for this node type, creates DOM widget container, fetches style details via /comfy-omg/cookbook/styles and /comfy-omg/cookbook/style/{slug}
- When style_selection changes, JS fetches that style's environment_variables and dynamically creates inputs (text, textarea) for each variable with description as tooltip
- User fills those dynamic inputs, JS serializes to hidden dynamic_vars_json widget
- Python execute reads dynamic_vars_json + fills prompt_template

This is C4 JS widgets (in-node UI) + C5 chrome per skill: addDOMWidget with getValue/setValue/serialize, min_size guard, idempotency
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any
import logging

_log = logging.getLogger(__name__)

STYLES_DIR = Path(__file__).parent.parent.parent / "presets" / "cookbook_styles"

_CACHE_STYLES_LIST = None
_CACHE_STYLES_DICT = None


def _load_all_styles():
    global _CACHE_STYLES_LIST, _CACHE_STYLES_DICT
    if _CACHE_STYLES_LIST is not None and _CACHE_STYLES_DICT is not None:
        return _CACHE_STYLES_LIST, _CACHE_STYLES_DICT
    styles_list = []
    styles_dict = {}
    if not STYLES_DIR.exists():
        _CACHE_STYLES_LIST = []
        _CACHE_STYLES_DICT = {}
        return [], {}
    for jf in sorted(STYLES_DIR.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                slug = data.get("style_slug") or jf.stem
                name = data.get("style_name") or slug
                display = f"{name} ({slug})"
                styles_list.append({"slug": slug, "name": name, "display": display, "data": data})
                styles_dict[slug] = data
                styles_dict[display] = data
                styles_dict[name] = data
        except Exception as e:
            continue
    _CACHE_STYLES_LIST = styles_list
    _CACHE_STYLES_DICT = styles_dict
    return styles_list, styles_dict


def _get_style_options():
    styles_list, _ = _load_all_styles()
    options = ["auto-infer (pick best for concept)"]
    for item in styles_list:
        options.append(item["display"])
    return options


def _find_style(selection: str):
    _, styles_dict = _load_all_styles()
    if selection in styles_dict:
        return styles_dict[selection]
    for key, data in styles_dict.items():
        if isinstance(data, dict) and data.get("style_slug") == selection:
            return data
        if selection.lower() in key.lower():
            if isinstance(data, dict) and "style_name" in data:
                return data
    return None


class CookbookDynamicInputs:
    """ONE NODE with JS widget - dropdown for various system prompts from github repo, as user changes item we change inputs - makes user life easy"""

    CATEGORY = "ComfyUI-OMG/Prompt/Cookbook"
    FUNCTION = "generate_dynamic"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("final_prompt", "negative_prompt", "variables_json", "char_count")
    DESCRIPTION = "Dynamic inputs node - dropdown for 66 cookbook system prompts, as you change dropdown, JS widget changes inputs to show exactly what that style needs (SUBJECT, MAIN_TEXT etc). Technology auto-fills from concept but you can edit each field. One node handles 66 different input requirements, not 66 nodes. Worth doing."

    @classmethod
    def INPUT_TYPES(cls):
        style_options = _get_style_options()
        return {
            "required": {
                "style_selection": (style_options, {
                    "default": "auto-infer (pick best for concept)",
                    "tooltip": "Pick system prompt / style from 66 cookbook styles - each has different input requirements. As you change this dropdown, JS widget below changes inputs to show exactly what this style needs. Technology makes life easy."
                }),
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "a tired architect with silver hair studying blueprint late night",
                    "tooltip": "Few words - main concept that auto-fills all dynamic inputs for selected style. You can then edit each dynamic field manually in widget below."
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "ollama_model": ("OLLAMA_MODEL", {
                    "tooltip": "Optional Ollama to intelligently auto-fill dynamic inputs from concept + style fidelity anchors"
                }),
                "aspect_ratio": (["auto-infer", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "3:4 vertical", "4:5"], {
                    "default": "auto-infer",
                    "tooltip": "Aspect ratio - auto-fills ASPECT_RATIO if style needs it"
                }),
                "dynamic_vars_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Hidden JSON from JS widget - contains all dynamic inputs for selected style. This is auto-managed by JS widget below - you can also paste custom JSON here for advanced override. Technology as much as possible."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, style_selection: str, concept: str, ollama_model: Dict[str, Any] = None, aspect_ratio: str = "auto-infer", dynamic_vars_json: str = "{}",
        think_mode: str = "off",
        filter_thinking: bool = True,
        **kwargs,
    ):
        import hashlib, json
        try:
            model_id = ollama_model.get("model", "") if isinstance(ollama_model, dict) else ""
            key = {
                "style": style_selection,
                "concept": concept[:500],
                "aspect_ratio": aspect_ratio,
                "dynamic_vars": dynamic_vars_json[:1000],
                "model": model_id,
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return concept

    def generate_dynamic(
        self,
        style_selection: str,
        concept: str,
        ollama_model: Dict[str, Any] = None,
        aspect_ratio: str = "auto-infer",
        dynamic_vars_json: str = "{}",
        unique_id: str = None,
        think_mode: str = "off",
        filter_thinking: bool = True,
        **kwargs
    ):
        styles_list, styles_dict = _load_all_styles()

        if not styles_list:
            return (f"No styles found in {STYLES_DIR}", "", "{}", 0)

        # Find selected style
        if style_selection.startswith("auto-infer"):
            # Auto-pick based on concept
            c_lower = concept.lower()
            selected_data = None
            for item in styles_list:
                if any(k in c_lower for k in ["architect", "portrait"]) and "portrait" in item["slug"]:
                    selected_data = item["data"]
                    break
            if not selected_data:
                selected_data = styles_list[0]["data"]
        else:
            selected_data = _find_style(style_selection)
            if not selected_data:
                selected_data = styles_list[0]["data"]

        env_vars = selected_data.get("environment_variables", {})
        prompt_template = selected_data.get("prompt_template", "")
        negative_template = selected_data.get("negative_prompt", "")
        style_fidelity_anchors = selected_data.get("style_fidelity_anchors", [])
        source_avoid = selected_data.get("source_content_to_avoid", [])

        # Parse dynamic_vars_json from JS widget (contains user-filled dynamic inputs)
        filled_vars = {}
        try:
            if dynamic_vars_json.strip():
                parsed = json.loads(dynamic_vars_json)
                if isinstance(parsed, dict):
                    filled_vars = parsed
        except Exception as e:
            _log.warning(f"Failed to parse dynamic_vars_json: {e}")

        # If dynamic_vars_json empty, try to auto-fill from concept via Ollama or fallback
        if not filled_vars:
            # Try Ollama auto-fill if model provided
            if ollama_model:
                try:
                    from ...ollama_client import generate as ollama_generate
                    from ...utils.text_utils import extract_json_block

                    vars_desc = "\n".join([f"- {k}: {v}" for k, v in env_vars.items() if k not in ["ASPECT_RATIO", "STYLE_FIDELITY_ANCHORS", "SOURCE_CONTENT_TO_AVOID"]])

                    system_prompt = f"""You are variable filler for style '{selected_data.get('style_name','')}'.

Style summary: {selected_data.get('style_summary','')}
Fidelity anchors: {', '.join(style_fidelity_anchors[:5])}
Variables to fill:
{vars_desc}

Concept: {concept}
Aspect: {aspect_ratio}

Return ONLY JSON mapping variable name to value, no markdown."""

                    user_prompt = f"Concept: {concept}\n\nFill variables:\n{vars_desc}\n\nReturn JSON only."

                    cfg = ollama_model

                    think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

                    try:

                        cfg["think"] = think

                        cfg["filter_thinking"] = filter_thinking

                    except Exception:

                        pass
                    raw = ollama_generate(
                        base_url=cfg["base_url"],
                        model=cfg["model"],
                        prompt=user_prompt,
                        system=system_prompt,
                        temperature=0.7,
                        num_ctx=cfg.get("num_ctx", 8192),
                        num_predict=1500,
                        seed=cfg.get("seed", -1),
                        keep_alive=cfg.get("keep_alive", "5m"),
                        response_format="json",
                    )
                    parsed = extract_json_block(raw)
                    if isinstance(parsed, dict):
                        filled_vars.update(parsed)
                except Exception as e:
                    _log.warning(f"Ollama auto-fill failed: {e}")

            # Fallback: fill from concept heuristics for any remaining
            for var_name in env_vars.keys():
                if var_name in filled_vars:
                    continue
                if var_name == "ASPECT_RATIO":
                    filled_vars[var_name] = aspect_ratio if aspect_ratio != "auto-infer" else "16:9"
                elif var_name == "STYLE_FIDELITY_ANCHORS":
                    filled_vars[var_name] = ", ".join(style_fidelity_anchors[:5]) if isinstance(style_fidelity_anchors, list) else str(style_fidelity_anchors)[:500]
                elif var_name == "SOURCE_CONTENT_TO_AVOID":
                    filled_vars[var_name] = ", ".join(source_avoid[:3]) if isinstance(source_avoid, list) else str(source_avoid)[:500]
                elif "SUBJECT" in var_name:
                    filled_vars[var_name] = concept[:100]
                elif "MAIN_TEXT" in var_name:
                    words = concept.split()[:6]
                    if len(words) >= 3:
                        filled_vars[var_name] = f"{words[0].lower()}\n{words[1].lower() if len(words)>1 else 'outlasts'}\n{words[2].lower()+'.' if len(words)>2 else 'noise.'}"
                    else:
                        filled_vars[var_name] = "focus\noutlasts\nnoise."
                elif "SECONDARY" in var_name:
                    filled_vars[var_name] = "studio log 02:14"
                elif "ACCENT" in var_name:
                    filled_vars[var_name] = "a tiny white plus"
                elif "WARDROBE" in var_name:
                    filled_vars[var_name] = "dark work jacket over plain black shirt"
                elif "LOCATION" in var_name:
                    filled_vars[var_name] = "dim concrete studio after midnight"
                elif "BACKGROUND" in var_name:
                    filled_vars[var_name] = "soft charcoal wall gradient, blurred wall, deep negative space"
                elif "PRODUCT" in var_name or "PROP" in var_name:
                    filled_vars[var_name] = "a rolled plan tube and pencil held low"
                else:
                    filled_vars[var_name] = concept[:80]

        # Apply filled vars to template
        final_prompt = prompt_template
        for k, v in filled_vars.items():
            final_prompt = final_prompt.replace(f"{{{k}}}", str(v))

        # Handle ASPECT_RATIO if not in filled_vars but in template
        if "{ASPECT_RATIO}" in final_prompt:
            final_prompt = final_prompt.replace("{ASPECT_RATIO}", aspect_ratio if aspect_ratio != "auto-infer" else "16:9")

        final_negative = negative_template
        if final_negative:
            for k, v in filled_vars.items():
                final_negative = final_negative.replace(f"{{{k}}}", str(v))

        # Save last enhanced prompt for copy button
        try:
            # For JS copy button via localStorage, we can't set from Python, but we can log
            pass
        except:
            pass

        variables_json = json.dumps(filled_vars, indent=2, ensure_ascii=False)

        return (final_prompt, final_negative, variables_json, len(final_prompt))


NODE_CLASS_MAPPINGS = {
    "CookbookDynamicInputs": CookbookDynamicInputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CookbookDynamicInputs": "Cookbook - Dynamic (66 Styles)",
}
