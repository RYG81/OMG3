"""
OMG Advanced Grouping Nodes - More advanced than rgthree group/context

RGThree has Context and Reroute that bundle model/clip/vae/positive/negative.
This is more advanced: bundles full MiniMax H3 workflow + generic * inputs into single wire.

- OMGFlowContextCreate/Update/Get/Inspect: Like WorldBible/CharacterBible but for MiniMax Flow - bundles concept, positive, negative, shot_plan, motion_notes, bible_json, storyboard_json, character_reference, continuity, h3_json, preset_json, enhanced_prompt, image refs, clip, ollama_model, fal_preset, small_model_mode
- OMGGroupBundle/Unbundle/Merge/Switch: Generic grouping of any types (STRING, IMAGE, CLIP, OLLAMA_MODEL, etc.) into single OMG_GROUP_CONTEXT that can be passed with one wire and unpacked later - more advanced than rgthree's reroute because it preserves type info and allows nested groups and merging

Works with few words concept, reference IMAGEs, CLIP fallback, 9B small model mode, 71 fal presets.
"""

from __future__ import annotations
import json
import logging
from ...utils.text_utils import filter_thinking
from typing import Dict, Any, List

_log = logging.getLogger(__name__)

# Custom types for grouping - these are just strings for ComfyUI type system, actual data is dict or any
# OMG_FLOW_CONTEXT contains full MiniMax flow data
# OMG_GROUP_CONTEXT contains generic bundled data

class OMGFlowContextCreate:
    """Create a flow context that bundles full MiniMax H3 workflow - more advanced than rgthree context"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "create_context"
    RETURN_TYPES = ("OMG_FLOW_CONTEXT", "STRING")
    RETURN_NAMES = ("flow_context", "context_json")
    OUTPUT_TOOLTIPS = (
        "Flow context containing all MiniMax H3 workflow data - pass with single wire instead of many",
        "JSON representation for debugging or saving",
    )
    DESCRIPTION = "Bundle full MiniMax H3 workflow into single context - concept, prompts, bible, storyboard, refs, clip, ollama_model, presets, small_model_mode. More advanced than rgthree group which only bundles model/clip/vae/pos/neg."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {"default": "chai rain Mumbai", "multiline": True, "tooltip": "Few words allowed! Main concept - e.g. 'chai rain Mumbai' or detailed description"}),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Removes thinking blocks to get proper prompt only."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Positive prompt from MiniMax nodes - production brief"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "shot_plan": ("STRING", {"multiline": True, "default": ""}),
                "motion_notes": ("STRING", {"multiline": True, "default": ""}),
                "model_settings": ("STRING", {"multiline": True, "default": ""}),
                "bible_json": ("STRING", {"multiline": True, "default": "", "tooltip": "Character bible JSON from CharacterBibleCreate or our earlier bible nodes"}),
                "storyboard_json": ("STRING", {"multiline": True, "default": "", "tooltip": "Storyboard JSON from ShotList or Storyboard V2"}),
                "character_reference": ("STRING", {"multiline": True, "default": ""}),
                "continuity_checklist": ("STRING", {"multiline": True, "default": ""}),
                "h3_json": ("STRING", {"multiline": True, "default": "", "tooltip": "Full H3 JSON with all fields"}),
                "preset_json": ("STRING", {"multiline": True, "default": ""}),
                "enhanced_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Enhanced prompt from Official Enhancer"}),
                "world_bible": ("OMG_WORLD_BIBLE", {"tooltip": "Optional world bible from WorldBibleCreate"}),
                "character_bible": ("OMG_CHARACTER_BIBLE", {"tooltip": "Optional character bible"}),
                "shot_list": ("OMG_SHOT_LIST", {"tooltip": "Optional shot list"}),
                "image_ref_1": ("IMAGE", {"tooltip": "Optional reference image 1"}),
                "image_ref_2": ("IMAGE", {"tooltip": "Optional reference image 2"}),
                "image_ref_3": ("IMAGE", {"tooltip": "Optional reference image 3"}),
                "clip": ("CLIP", {"tooltip": "CLIP from ComfyUI - for CLIP fallback path"}),
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Ollama model - for LLM expansion"}),
                "fal_preset_id": ("STRING", {"default": "auto-infer", "tooltip": "Fal preset id, e.g. type_identity_lock or vintage_binocular..."}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable for 9B or smaller models"}),
                "use_clip_fallback": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 15}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "auto"], {"default": "16:9"}),
            }
        }

    def create_context(self, concept: str, **kwargs):
        # Build context dict with all provided values, only include non-empty
        context: Dict[str, Any] = {
            "concept": concept,
            "type": "OMG_FLOW_CONTEXT",
            "version": "1.0",
            "created_from": "OMGFlowContextCreate",
        }
        # Add all string fields if provided
        for key in ["positive_prompt", "negative_prompt", "shot_plan", "motion_notes", "model_settings",
                    "bible_json", "storyboard_json", "character_reference", "continuity_checklist",
                    "h3_json", "preset_json", "enhanced_prompt", "fal_preset_id"]:
            val = kwargs.get(key)
            if val not in (None, "", []):
                context[key] = val

        # Add primitives
        for key in ["small_model_mode", "use_clip_fallback", "duration", "aspect_ratio"]:
            if key in kwargs:
                context[key] = kwargs[key]

        # Add complex objects as references (we store type info, actual objects kept separately in context for extraction)
        # For IMAGE, CLIP, OLLAMA_MODEL, WORLD_BIBLE, CHARACTER_BIBLE, SHOT_LIST - we store them in a separate hidden dict that ComfyUI will pass via context
        # Since RETURN_TYPES is OMG_FLOW_CONTEXT which is actually a dict, we can store the objects inside dict
        # ComfyUI will handle IMAGE, CLIP etc. as part of dict values if we return them as part of context dict
        # For simplicity, store them in context dict under same keys
        for key in ["world_bible", "character_bible", "shot_list", "image_ref_1", "image_ref_2", "image_ref_3", "clip", "ollama_model"]:
            val = kwargs.get(key)
            if val is not None:
                context[key] = val  # Keep actual object, not just string

        # Build JSON for debugging (exclude IMAGE, CLIP, OLLAMA_MODEL actual objects, replace with type info)
        json_safe = {}
        for k, v in context.items():
            if k in ["image_ref_1", "image_ref_2", "image_ref_3", "clip", "ollama_model", "world_bible", "character_bible", "shot_list"]:
                json_safe[k] = f"<{type(v).__name__} object>" if v is not None else None
            else:
                # Try to make json serializable
                try:
                    json.dumps(v)
                    json_safe[k] = v
                except:
                    json_safe[k] = str(v)[:500]

        context_json = json.dumps(json_safe, indent=2)

        return (context, context_json)


class OMGFlowContextUpdate:
    """Update existing flow context with new values - like CharacterBibleUpdate but for full flow"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "update_context"
    RETURN_TYPES = ("OMG_FLOW_CONTEXT", "STRING")
    RETURN_NAMES = ("flow_context", "context_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_context": ("OMG_FLOW_CONTEXT",),
            },
            "optional": {
                "concept": ("STRING", {"multiline": True, "default": ""}),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "shot_plan": ("STRING", {"multiline": True, "default": ""}),
                "motion_notes": ("STRING", {"multiline": True, "default": ""}),
                "bible_json": ("STRING", {"multiline": True, "default": ""}),
                "storyboard_json": ("STRING", {"multiline": True, "default": ""}),
                "character_reference": ("STRING", {"multiline": True, "default": ""}),
                "h3_json": ("STRING", {"multiline": True, "default": ""}),
                "preset_json": ("STRING", {"multiline": True, "default": ""}),
                "enhanced_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_ref_1": ("IMAGE",),
                "image_ref_2": ("IMAGE",),
                "image_ref_3": ("IMAGE",),
                "clip": ("CLIP",),
                "ollama_model": ("OLLAMA_MODEL",),
                "fal_preset_id": ("STRING", {"default": ""}),
                "small_model_mode": ("BOOLEAN", {"default": False}),
                "use_clip_fallback": ("BOOLEAN", {"default": False}),
            }
        }

    def update_context(self, flow_context: Dict, **kwargs):
        # Copy existing context
        new_context = dict(flow_context) if isinstance(flow_context, dict) else {"concept": str(flow_context)}

        # Update with non-empty new values
        for key, val in kwargs.items():
            if val not in (None, "", []):
                # For booleans, allow False to overwrite? Only update if provided and not empty string
                # For small_model_mode and use_clip_fallback, we should update even if False if explicitly passed
                if key in ["small_model_mode", "use_clip_fallback"] or val not in (None, ""):
                    new_context[key] = val

        # Build json safe
        json_safe = {}
        for k, v in new_context.items():
            if k in ["image_ref_1", "image_ref_2", "image_ref_3", "clip", "ollama_model", "world_bible", "character_bible", "shot_list"]:
                json_safe[k] = f"<{type(v).__name__}>"
            else:
                try:
                    json.dumps(v)
                    json_safe[k] = v
                except:
                    json_safe[k] = str(v)[:500]

        return (new_context, json.dumps(json_safe, indent=2))


class OMGFlowContextGet:
    """Extract specific field from flow context - more advanced than rgthree which only extracts fixed fields"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "get_from_context"
    RETURN_TYPES = ("STRING", "STRING", "OMG_FLOW_CONTEXT")
    RETURN_NAMES = ("value", "value_as_string", "flow_context_passthrough")
    OUTPUT_TOOLTIPS = (
        "Value of requested key as original type (STRING, IMAGE, CLIP, etc. - use forceInput to get correct type)",
        "Value as string for debugging",
        "Passthrough of original context",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_context": ("OMG_FLOW_CONTEXT",),
                "key": ("STRING", {"default": "positive_prompt", "tooltip": "Key to extract: concept, positive_prompt, negative_prompt, shot_plan, bible_json, storyboard_json, character_reference, h3_json, preset_json, enhanced_prompt, image_ref_1, clip, ollama_model, fal_preset_id, small_model_mode, etc."}),
            },
            "optional": {
                "default_value": ("STRING", {"multiline": True, "default": "", "tooltip": "Default if key not found"}),
            }
        }

    def get_from_context(self, flow_context: Dict, key: str, default_value: str = ""):
        if not isinstance(flow_context, dict):
            return (default_value, default_value, flow_context)

        val = flow_context.get(key, default_value)

        # Return as string version for debugging, and original value for actual use (with forced input, ComfyUI will handle type)
        # Since RETURN_TYPES first is STRING, we need to return string, but if user needs IMAGE/CLIP, they can use forceInput on second input? Actually we return STRING, but we can also return original object via third? Simpler: return string, string, passthrough
        # For generic extraction of IMAGE/CLIP, user should use OMGGroupUnbundle or specific extractor with * type
        # Here we return stringified version for simplicity, and also passthrough context

        if isinstance(val, str):
            str_val = val
        else:
            try:
                str_val = json.dumps(val, indent=2) if isinstance(val, (dict, list)) else str(val)
            except:
                str_val = str(val)

        return (str_val, str_val, flow_context)


class OMGFlowContextToPrompt:
    """Convert flow context to final MiniMax H3 prompt - like CharacterBibleToPrompt but for full flow"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "context_to_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "full_context_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_context": ("OMG_FLOW_CONTEXT",),
            },
            "optional": {
                "use_enhanced": ("BOOLEAN", {"default": True, "tooltip": "If true, use enhanced_prompt from Official Enhancer if available, else positive_prompt"}),
                "include_bible": ("BOOLEAN", {"default": True, "tooltip": "Include character bible and continuity checklist in prompt"}),
                "include_storyboard": ("BOOLEAN", {"default": False, "tooltip": "Include storyboard_json shot list in prompt (makes it longer)"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "If true, truncate to under 1500 chars for 9B models"}),
            }
        }

    def context_to_prompt(self, flow_context: Dict, use_enhanced: bool = True, include_bible: bool = True, include_storyboard: bool = False, small_model_mode: bool = False):
        if not isinstance(flow_context, dict):
            return (str(flow_context), "", json.dumps({"error": "not a dict"}))

        # Choose main prompt
        positive = ""
        if use_enhanced and flow_context.get("enhanced_prompt"):
            positive = flow_context["enhanced_prompt"]
        elif flow_context.get("positive_prompt"):
            positive = flow_context["positive_prompt"]
        else:
            # Build from concept + other fields
            concept = flow_context.get("concept", "")
            positive = f"[SCENE]\n{concept}\n\n"

        if include_bible:
            bible = flow_context.get("bible_json", "") or flow_context.get("character_reference", "")
            if bible:
                positive += f"\n\n[CHARACTER BIBLE]\n{bible[:1000] if small_model_mode else bible[:3000]}"

            continuity = flow_context.get("continuity_checklist", "")
            if continuity:
                positive += f"\n\n[CONTINUITY LOCKS]\n{continuity[:500] if small_model_mode else continuity[:1000]}"

        if include_storyboard:
            sb = flow_context.get("storyboard_json", "") or flow_context.get("shot_plan", "")
            if sb:
                positive += f"\n\n[STORYBOARD]\n{sb[:1000] if small_model_mode else sb[:3000]}"

        negative = flow_context.get("negative_prompt", "No extra people, No face drift, No wardrobe changes, No voice swaps, No broken eyelines")

        # Truncate for small model
        if small_model_mode and len(positive) > 1500:
            positive = positive[:1500] + "\n\n[Truncated for 9B small model]"

        # Build json
        json_safe = {}
        for k, v in flow_context.items():
            if k in ["image_ref_1", "image_ref_2", "image_ref_3", "clip", "ollama_model"]:
                json_safe[k] = f"<{type(v).__name__}>"
            else:
                try:
                    json.dumps(v)
                    json_safe[k] = v if not isinstance(v, str) or len(v) < 2000 else v[:2000] + "...[truncated]"
                except:
                    json_safe[k] = str(v)[:500]

        return (positive, negative, json.dumps(json_safe, indent=2))


class OMGGroupBundle:
    """Bundle up to 10 generic inputs of any type into single OMG_GROUP_CONTEXT - more advanced than rgthree reroute which only handles 1 type"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "bundle"
    RETURN_TYPES = ("OMG_GROUP_CONTEXT", "STRING")
    RETURN_NAMES = ("group_context", "group_json")
    DESCRIPTION = "Bundle multiple values of any type (STRING, IMAGE, CLIP, OLLAMA_MODEL, etc.) into single group context with one wire - more advanced than rgthree which needs separate reroutes per type. Supports nested groups and preserves type info."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_name": ("STRING", {"default": "my_group", "tooltip": "Name for this group for debugging"}),
            },
            "optional": {
                "input_1": ("*", {"tooltip": "Any type - STRING, IMAGE, CLIP, OLLAMA_MODEL, etc."}),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
                "input_6": ("*",),
                "input_7": ("*",),
                "input_8": ("*",),
                "key_1": ("STRING", {"default": "concept", "tooltip": "Key name for input_1 in group"}),
                "key_2": ("STRING", {"default": ""}),
                "key_3": ("STRING", {"default": ""}),
                "key_4": ("STRING", {"default": ""}),
                "key_5": ("STRING", {"default": ""}),
                "key_6": ("STRING", {"default": ""}),
                "key_7": ("STRING", {"default": ""}),
                "key_8": ("STRING", {"default": ""}),
            }
        }

    def bundle(self, group_name: str, **kwargs):
        group_dict: Dict[str, Any] = {
            "group_name": group_name,
            "type": "OMG_GROUP_CONTEXT",
            "keys": [],
        }

        for i in range(1, 9):
            inp_key = f"input_{i}"
            key_name_key = f"key_{i}"
            val = kwargs.get(inp_key)
            key_name = kwargs.get(key_name_key, "")

            if val is None:
                continue
            if isinstance(val, str) and val == "":
                # Allow empty string? Skip if key also empty and value empty?
                # But if key provided and value empty string, we still skip to avoid clutter
                if not key_name:
                    continue

            # Determine key name
            if not key_name:
                # Auto-generate key if not provided
                if isinstance(val, str) and len(val) < 50:
                    key_name = f"input_{i}_{val[:20]}"
                else:
                    key_name = f"input_{i}"

            group_dict[key_name] = val
            group_dict["keys"].append(key_name)

        # Build json safe for display
        json_safe = {
            "group_name": group_name,
            "keys": group_dict.get("keys", []),
        }
        for k in group_dict.get("keys", []):
            v = group_dict.get(k)
            if k in ["group_name", "keys", "type"]:
                continue
            if v is None:
                json_safe[k] = None
            else:
                # For IMAGE, CLIP etc, just show type
                if k.startswith("image_") or "IMAGE" in str(type(v)):
                    json_safe[k] = f"<{type(v).__name__} IMAGE>"
                elif "CLIP" in str(type(v)) or "clip" in k.lower():
                    json_safe[k] = f"<{type(v).__name__} CLIP>"
                elif "OLLAMA" in str(type(v)) or "ollama" in k.lower():
                    json_safe[k] = f"<{type(v).__name__} OLLAMA_MODEL>"
                else:
                    try:
                        s = str(v)
                        json_safe[k] = s[:500] + ("...[truncated]" if len(s) > 500 else "")
                    except:
                        json_safe[k] = f"<{type(v).__name__}>"

        return (group_dict, json.dumps(json_safe, indent=2))


class OMGGroupUnbundle:
    """Unbundle OMG_GROUP_CONTEXT back to individual values - supports any type"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "unbundle"
    RETURN_TYPES = ("*", "*", "*", "*", "*", "*", "*", "*")
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5", "output_6", "output_7", "output_8")
    DESCRIPTION = "Unbundle group context back to up to 8 individual values. More advanced than rgthree reroute - preserves original types."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_context": ("OMG_GROUP_CONTEXT",),
            },
            "optional": {
                "key_1": ("STRING", {"default": "", "tooltip": "Key to extract for output_1, leave empty to use first key in group"}),
                "key_2": ("STRING", {"default": ""}),
                "key_3": ("STRING", {"default": ""}),
                "key_4": ("STRING", {"default": ""}),
                "key_5": ("STRING", {"default": ""}),
                "key_6": ("STRING", {"default": ""}),
                "key_7": ("STRING", {"default": ""}),
                "key_8": ("STRING", {"default": ""}),
            }
        }

    def unbundle(self, group_context: Dict, **kwargs):
        if not isinstance(group_context, dict):
            return tuple([None]*8)

        keys_in_group = group_context.get("keys", []) or [k for k in group_context.keys() if k not in ["group_name", "type", "keys"]]

        outputs = []
        for i in range(1, 9):
            key_to_extract = kwargs.get(f"key_{i}", "")
            if not key_to_extract:
                # Use i-th key in group if available
                if i-1 < len(keys_in_group):
                    key_to_extract = keys_in_group[i-1]
                else:
                    key_to_extract = ""

            if key_to_extract and key_to_extract in group_context:
                outputs.append(group_context[key_to_extract])
            else:
                outputs.append(None)

        return tuple(outputs)


class OMGGroupMerge:
    """Merge two group contexts or flow contexts into one - more advanced than rgthree context switch"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "merge"
    RETURN_TYPES = ("OMG_GROUP_CONTEXT", "OMG_FLOW_CONTEXT", "STRING")
    RETURN_NAMES = ("merged_group", "merged_flow", "merge_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_a": ("OMG_GROUP_CONTEXT",),
                "group_b": ("OMG_GROUP_CONTEXT",),
            },
            "optional": {
                "merge_strategy": (["b_overwrites_a", "a_overwrites_b", "keep_both_with_prefix"], {"default": "b_overwrites_a", "tooltip": "If same key exists in both groups, which wins"}),
                "prefix_a": ("STRING", {"default": "a_"}),
                "prefix_b": ("STRING", {"default": "b_"}),
            }
        }

    def merge(self, group_a: Dict, group_b: Dict, merge_strategy: str = "b_overwrites_a", prefix_a: str = "a_", prefix_b: str = "b_"):
        if not isinstance(group_a, dict):
            group_a = {}
        if not isinstance(group_b, dict):
            group_b = {}

        merged = {}

        # Copy keys list handling
        keys_a = group_a.get("keys", []) or [k for k in group_a.keys() if k not in ["group_name", "type", "keys"]]
        keys_b = group_b.get("keys", []) or [k for k in group_b.keys() if k not in ["group_name", "type", "keys"]]

        if merge_strategy == "b_overwrites_a":
            merged = dict(group_a)
            merged.update(group_b)
            merged_keys = list(dict.fromkeys(keys_a + keys_b))  # preserve order, dedup
            # Ensure b overwrites in keys order too
            # Rebuild keys to have b's keys last
            merged["keys"] = merged_keys
            merged["group_name"] = f"{group_a.get('group_name','a')}+{group_b.get('group_name','b')}"
            merged["type"] = "OMG_GROUP_CONTEXT"

        elif merge_strategy == "a_overwrites_b":
            merged = dict(group_b)
            merged.update(group_a)
            merged_keys = list(dict.fromkeys(keys_b + keys_a))
            merged["keys"] = merged_keys
            merged["group_name"] = f"{group_b.get('group_name','b')}+{group_a.get('group_name','a')}"
            merged["type"] = "OMG_GROUP_CONTEXT"

        else:  # keep_both_with_prefix
            merged = {"type": "OMG_GROUP_CONTEXT", "keys": []}
            for k in keys_a:
                new_k = f"{prefix_a}{k}"
                merged[new_k] = group_a.get(k)
                merged["keys"].append(new_k)
            for k in keys_b:
                new_k = f"{prefix_b}{k}"
                merged[new_k] = group_b.get(k)
                merged["keys"].append(new_k)
            merged["group_name"] = f"{group_a.get('group_name','a')}+{group_b.get('group_name','b')}_prefixed"

        # For flow context, try to merge as flow context if both are flow contexts
        is_flow_a = group_a.get("type") == "OMG_FLOW_CONTEXT"
        is_flow_b = group_b.get("type") == "OMG_FLOW_CONTEXT"
        merged_flow = merged if (is_flow_a or is_flow_b) else {"concept": "merged", "type": "OMG_FLOW_CONTEXT"}

        if is_flow_a or is_flow_b:
            # For flow, also merge concept etc.
            # Simple merge: b overwrites a
            merged_flow = dict(group_a)
            merged_flow.update(group_b)
            merged_flow["type"] = "OMG_FLOW_CONTEXT"

        # JSON safe
        json_safe = {"group_name": merged.get("group_name"), "keys": merged.get("keys", [])[:20]}
        return (merged, merged_flow, json.dumps(json_safe, indent=2))


class OMGGroupSwitch:
    """Switch between two group contexts based on boolean or int - more advanced than rgthree context switch which only handles model/clip/vae"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "switch"
    RETURN_TYPES = ("OMG_GROUP_CONTEXT", "OMG_FLOW_CONTEXT", "STRING")
    RETURN_NAMES = ("selected_group", "selected_flow", "selection_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_a": ("OMG_GROUP_CONTEXT",),
                "group_b": ("OMG_GROUP_CONTEXT",),
                "select_b": ("BOOLEAN", {"default": False, "tooltip": "False = select group_a, True = select group_b"}),
            },
            "optional": {
                "select_index": ("INT", {"default": 0, "min": 0, "max": 1, "tooltip": "Alternative int selector: 0 = a, 1 = b"}),
            }
        }

    def switch(self, group_a: Dict, group_b: Dict, select_b: bool = False, select_index: int = 0):
        # Use select_index if provided and select_b is False? Actually boolean takes precedence if True
        # If select_b False and select_index 1, select b
        use_b = select_b or (select_index == 1)

        selected = group_b if use_b else group_a
        if not isinstance(selected, dict):
            selected = {}

        # Determine if it's flow context
        is_flow = selected.get("type") == "OMG_FLOW_CONTEXT"
        flow_out = selected if is_flow else {"concept": selected.get("concept", "from_group"), "type": "OMG_FLOW_CONTEXT"}

        info = f"Selected {'B' if use_b else 'A'}: {selected.get('group_name', selected.get('concept', 'unnamed')[:50])} (type {selected.get('type')})"

        return (selected, flow_out, info)


# ===== NEW: Enable/Disable Group Flow Control - for image -> video sequential workflows =====

class OMGGroupGate:
    CATEGORY = "ComfyUI-OMG/Utility/Flow"
    """Gate that enables/disables a group based on boolean - image generated now generate video"""

    CATEGORY = "ComfyUI-OMG/Utility/Grouping"
    FUNCTION = "gate"
    RETURN_TYPES = ("OMG_GROUP_CONTEXT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("group_context", "was_enabled", "status")
    DESCRIPTION = "Enable/disable a group in flow. If enabled=True, passes group through. If False, outputs empty group and was_enabled=False. Use to manage image and video generation groups together - e.g. image group completes -> enables video group."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_context": ("OMG_GROUP_CONTEXT", {"tooltip": "Group to gate - e.g. video generation group"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "True = group enabled and passes through, False = group blocked"}),
            },
            "optional": {
                "group_name": ("STRING", {"default": "", "tooltip": "Optional override group name for debugging"}),
                "bypass_on_disabled": ("BOOLEAN", {"default": False, "tooltip": "If True and disabled, still output group but with status disabled. If False, output empty dict when disabled."}),
            }
        }

    def gate(self, group_context: Dict, enabled: bool = True, group_name: str = "", bypass_on_disabled: bool = False):
        if not isinstance(group_context, dict):
            group_context = {"type": "OMG_GROUP_CONTEXT", "group_name": group_name or "unnamed", "keys": []}

        if enabled:
            # Pass through, optionally update group_name
            if group_name:
                group_context = dict(group_context)
                group_context["group_name"] = group_name
            status = f"Enabled: {group_context.get('group_name','group')} passed through ({len(group_context.get('keys',[]))} keys)"
            return (group_context, True, status)
        else:
            # Disabled
            if bypass_on_disabled:
                status = f"Disabled but bypass_on: {group_context.get('group_name','group')} still passed but marked disabled"
                # Add disabled flag
                gc = dict(group_context)
                gc["_disabled"] = True
                return (gc, False, status)
            else:
                empty = {"type": "OMG_GROUP_CONTEXT", "group_name": (group_name or group_context.get("group_name","group")) + " [DISABLED]", "keys": [], "_disabled": True}
                status = f"Disabled: {empty['group_name']} blocked, outputs empty group"
                return (empty, False, status)


class OMGFlowTrigger:
    """Trigger that converts any signal (IMAGE, VIDEO path, STRING) into BOOLEAN enabled for next group"""

    CATEGORY = "ComfyUI-OMG/Utility/Flow"
    FUNCTION = "trigger"
    RETURN_TYPES = ("BOOLEAN", "STRING", "*")
    RETURN_NAMES = ("is_ready", "status", "passthrough")
    DESCRIPTION = "Converts image generated / video generated / any signal into boolean trigger to enable next group. E.g. image generated now generate video - image output -> this trigger -> enables video generation group via GroupGate."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_mode": (["any_input_exists", "image_valid", "string_not_empty", "always_true", "always_false"], {"default": "any_input_exists"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "If image is valid (generated), trigger True - use to trigger video group after image gen"}),
                "video_path": ("STRING", {"default": "", "tooltip": "Video path string - if not empty, trigger True"}),
                "string_signal": ("STRING", {"default": "", "multiline": True, "tooltip": "Any string signal - if not empty, trigger True"}),
                "any_value": ("*", {"tooltip": "Any value - if exists and not None, trigger True in any_input_exists mode"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Invert result"}),
            }
        }

    def trigger(self, trigger_mode: str = "any_input_exists", **kwargs):
        image = kwargs.get("image")
        video_path = kwargs.get("video_path", "")
        string_signal = kwargs.get("string_signal", "")
        any_value = kwargs.get("any_value")
        invert = kwargs.get("invert", False)

        is_ready = False
        reason = ""

        if trigger_mode == "always_true":
            is_ready = True
            reason = "always_true mode"
        elif trigger_mode == "always_false":
            is_ready = False
            reason = "always_false mode"
        elif trigger_mode == "image_valid":
            # Check if IMAGE tensor is valid
            if image is not None:
                try:
                    # IMAGE should be torch tensor with shape [B,H,W,C] and not all zeros
                    import torch
                    if isinstance(image, torch.Tensor):
                        is_ready = image.numel() > 0 and image.shape[0] > 0
                        reason = f"IMAGE valid: shape {list(image.shape)}"
                    else:
                        is_ready = True
                        reason = f"IMAGE object exists: {type(image).__name__}"
                except Exception as e:
                    is_ready = image is not None
                    reason = f"IMAGE exists check: {e}"
            else:
                is_ready = False
                reason = "No IMAGE provided"
        elif trigger_mode == "string_not_empty":
            # Check string signals
            if video_path and str(video_path).strip():
                is_ready = True
                reason = f"video_path not empty: {str(video_path)[:50]}"
            elif string_signal and str(string_signal).strip():
                is_ready = True
                reason = f"string_signal not empty: {str(string_signal)[:50]}"
            else:
                is_ready = False
                reason = "Both video_path and string_signal empty"
        else:  # any_input_exists
            if image is not None:
                is_ready = True
                reason = "any_input_exists: image exists"
            elif video_path and str(video_path).strip():
                is_ready = True
                reason = f"any_input_exists: video_path exists"
            elif string_signal and str(string_signal).strip():
                is_ready = True
                reason = f"any_input_exists: string_signal exists"
            elif any_value is not None:
                # Check if any_value is not empty
                if isinstance(any_value, str):
                    is_ready = len(any_value.strip()) > 0
                    reason = f"any_input_exists: any_value string length {len(any_value)}"
                elif isinstance(any_value, (list, dict)):
                    is_ready = len(any_value) > 0
                    reason = f"any_input_exists: any_value len {len(any_value)}"
                else:
                    is_ready = True
                    reason = f"any_input_exists: any_value exists type {type(any_value).__name__}"
            else:
                is_ready = False
                reason = "any_input_exists: no inputs provided"

        if invert:
            is_ready = not is_ready
            reason += " [inverted]"

        # Passthrough the triggering value (prefer image, then video_path, then string_signal, then any_value)
        passthrough = None
        if image is not None:
            passthrough = image
        elif video_path:
            passthrough = video_path
        elif string_signal:
            passthrough = string_signal
        elif any_value is not None:
            passthrough = any_value

        status = f"Trigger {trigger_mode}: is_ready={is_ready} - {reason}"
        return (is_ready, status, passthrough)


class OMGGroupSequencer:
    """Manages two groups: image generation group and video generation group together - enables video after image done"""

    CATEGORY = "ComfyUI-OMG/Utility/Flow"
    FUNCTION = "sequence_groups"
    RETURN_TYPES = ("OMG_GROUP_CONTEXT", "OMG_GROUP_CONTEXT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image_group_out", "video_group_out", "video_should_run", "status")
    DESCRIPTION = "Manage image and video generation groups together. Image group always enabled first. Video group enabled after image_generated signal True. Use case: image generated now generate video - two groups managed together."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_group": ("OMG_GROUP_CONTEXT", {"tooltip": "Image generation group - always enabled first"}),
                "video_group": ("OMG_GROUP_CONTEXT", {"tooltip": "Video generation group - enabled after image_generated trigger"}),
                "image_generated": ("BOOLEAN", {"default": False, "tooltip": "True when image generation completed - e.g. from FlowTrigger with IMAGE valid"}),
            },
            "optional": {
                "auto_enable_video": ("BOOLEAN", {"default": True, "tooltip": "If True, video group auto-enabled when image_generated=True. If False, video group stays disabled until manually enabled"}),
                "disable_image_after_video": ("BOOLEAN", {"default": False, "tooltip": "If True, disable image group after video should run (to save VRAM)"}),
            }
        }

    def sequence_groups(self, image_group: Dict, video_group: Dict, image_generated: bool = False, auto_enable_video: bool = True, disable_image_after_video: bool = False):
        if not isinstance(image_group, dict):
            image_group = {"type": "OMG_GROUP_CONTEXT", "group_name": "image_group", "keys": []}
        if not isinstance(video_group, dict):
            video_group = {"type": "OMG_GROUP_CONTEXT", "group_name": "video_group", "keys": []}

        # Image group always out (enabled first)
        image_out = dict(image_group)
        if disable_image_after_video and image_generated and auto_enable_video:
            # Disable image group to save VRAM after video should run
            image_out["_disabled"] = True
            image_out["group_name"] = image_out.get("group_name", "image_group") + " [DISABLED after video trigger]"

        # Video group logic
        video_should_run = False
        if auto_enable_video and image_generated:
            video_should_run = True
            video_out = dict(video_group)
            video_out["_enabled_by"] = "image_generated trigger"
            status = f"Image generated=True -> Video group ENABLED: {video_out.get('group_name','video_group')} should run now. Image group {'DISABLED' if disable_image_after_video else 'still enabled'}."
        elif not auto_enable_video:
            video_should_run = False
            video_out = {"type": "OMG_GROUP_CONTEXT", "group_name": video_group.get("group_name","video_group") + " [WAITING for manual enable]", "keys": [], "_disabled": True, "_waiting": True}
            status = f"Auto_enable_video=False -> Video group stays DISABLED waiting manual enable. Image_generated={image_generated}"
        else:
            # image_generated False, video not yet
            video_should_run = False
            video_out = {"type": "OMG_GROUP_CONTEXT", "group_name": video_group.get("group_name","video_group") + " [DISABLED waiting image]", "keys": [], "_disabled": True, "_waiting_for_image": True}
            status = f"Image_generated=False -> Video group DISABLED waiting for image. Will enable when image_generated=True."

        return (image_out, video_out, video_should_run, status)


class OMGFlowBarrier:
    """Barrier that waits for multiple groups to be ready before enabling next group - for complex flows"""

    CATEGORY = "ComfyUI-OMG/Utility/Flow"
    FUNCTION = "barrier"
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("all_ready", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition_1": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "condition_2": ("BOOLEAN", {"default": True, "tooltip": "If not connected, treated as True (ignored)"}),
                "condition_3": ("BOOLEAN", {"default": True}),
                "condition_4": ("BOOLEAN", {"default": True}),
                "mode": (["all_true", "any_true", "all_false", "count_true"], {"default": "all_true", "tooltip": "all_true=AND, any_true=OR, all_false=NOR, count_true=at least N true"}),
                "required_count": ("INT", {"default": 2, "min": 1, "max": 4, "tooltip": "For count_true mode, how many must be true"}),
            }
        }

    def barrier(self, condition_1: bool = False, condition_2: bool = True, condition_3: bool = True, condition_4: bool = True, mode: str = "all_true", required_count: int = 2):
        conditions = [condition_1, condition_2, condition_3, condition_4]
        # Filter out those that were not connected? In ComfyUI, unconnected BOOLEAN defaults to True per our INPUT_TYPES, so we need to treat True as ignored? Actually we set default True for optional conditions so they don't block
        # For all_true mode, we want all conditions to be true, but optional ones default True so they don't block unless explicitly False

        if mode == "all_true":
            all_ready = all(conditions)
            status = f"Barrier all_true: {conditions} -> all_ready={all_ready}"
        elif mode == "any_true":
            all_ready = any(conditions)
            status = f"Barrier any_true: {conditions} -> all_ready={all_ready}"
        elif mode == "all_false":
            all_ready = not any(conditions)
            status = f"Barrier all_false: {conditions} -> all_ready={all_ready} (none true)"
        else:  # count_true
            true_count = sum(1 for c in conditions if c)
            all_ready = true_count >= required_count
            status = f"Barrier count_true: {true_count}/{len(conditions)} true, required {required_count} -> all_ready={all_ready}, conditions {conditions}"

        return (all_ready, status)


NODE_CLASS_MAPPINGS = {
    "OMGFlowContextCreate": OMGFlowContextCreate,
    "OMGFlowContextUpdate": OMGFlowContextUpdate,
    "OMGFlowContextGet": OMGFlowContextGet,
    "OMGFlowContextToPrompt": OMGFlowContextToPrompt,
    "OMGGroupBundle": OMGGroupBundle,
    "OMGGroupUnbundle": OMGGroupUnbundle,
    "OMGGroupMerge": OMGGroupMerge,
    "OMGGroupSwitch": OMGGroupSwitch,
    "OMGGroupGate": OMGGroupGate,
    "OMGFlowTrigger": OMGFlowTrigger,
    "OMGGroupSequencer": OMGGroupSequencer,
    "OMGFlowBarrier": OMGFlowBarrier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMGFlowContextCreate": "OMG Flow Context Create (Bundle MiniMax Flow - Advanced Grouping)",
    "OMGFlowContextUpdate": "OMG Flow Context Update",
    "OMGFlowContextGet": "OMG Flow Context Get Field",
    "OMGFlowContextToPrompt": "OMG Flow Context To Prompt (Final H3 Prompt)",
    "OMGGroupBundle": "OMG Group Bundle (Any Type - Advanced Reroute)",
    "OMGGroupUnbundle": "OMG Group Unbundle (Any Type)",
    "OMGGroupMerge": "OMG Group Merge (2 Groups -> 1)",
    "OMGGroupSwitch": "OMG Group Switch (A/B Select)",
    "OMGGroupGate": "OMG Group Gate (Enable/Disable Group in Flow)",
    "OMGFlowTrigger": "OMG Flow Trigger (Image/Video/String -> Boolean for Next Group)",
    "OMGGroupSequencer": "OMG Group Sequencer (Image Group -> Video Group Manager)",
    "OMGFlowBarrier": "OMG Flow Barrier (Wait for Multiple Groups)",
}
