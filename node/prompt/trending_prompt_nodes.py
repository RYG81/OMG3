"""
Trending Prompt Nodes - Utilizing nanobanana-trending-prompts (1,446 prompts, CC BY 4.0)

Source: https://github.com/jau123/nanobanana-trending-prompts
Categories: Photography (533), Illustration & 3D (370), Product & Brand (239), Food & Drink (156), Poster Design (146), UI & Graphic (52)
Models: nanobanana (1148), gptimage (298)
Top placeholders: [BRAND NAME] 299, [COLOR] 50, [OBJECT] 37, [LANDMARK] 13, etc.

How we utilize in ComfyUI-OMG pack (worth doing, not just preset loader):

1. OllamaTrendingPromptLoader - Searchable preset library with category/model/likes filter, placeholder extraction, not just loading all
2. OllamaTrendingPromptFiller - Auto-fills placeholders like [OBJECT], [BRAND NAME] via user inputs or Ollama, concept-aware (biker vs party)
3. OllamaPromptBuilderWithTrending - Uses Nanobanana system prompt rules (6 rules) + few-shot trending examples to enhance user's short concept into high-quality structured prompt for image AND video (H3/LTX)

Rules from system-prompt-en.md:
- Rule 1: Replace feeling words with professional terms (Wong Kar-wai, Saul Leiter, Kodak Vision3 500T, Swiss International Style)
- Rule 2: Replace adjectives with quantified parameters (90mm f/1.8, 45-degree overhead, shallow depth of field, Dutch angle, volumetric light)
- Rule 3: Add negative constraints (No text, no low-key dark lighting, no high-saturation neon, product must not be distorted)
- Rule 4: Sensory stacking (visual + tactile + olfactory + motion + temperature)
- Rule 5: Group and cluster (Visual Rules, Lighting & Style, Overall Feel, Constraints)
- Rule 6: Format adaptation (simple scenes natural paragraphs, complex scenes structured groupings)
- Plus scene adaptation guide: Product, Portrait, Food, Cinematic, Japanese Style, Design Poster

Worth doing for video: Product photography rules (Hasselblad, 90mm f/1.8, shallow depth, Apple product aesthetics) improve H3/LTX prompts, not just image.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

_log = logging.getLogger(__name__)

# Cache for performance
_TRENDING_CACHE = None
_TRENDING_PATH = Path(__file__).parent.parent.parent / "presets" / "nanobanana_trending_1446.json"

SYSTEM_PROMPT_EN_PATH = Path(__file__).parent.parent.parent / "presets" / "nanobanana_system_prompt_en.md"
SYSTEM_PROMPT_ZH_PATH = Path(__file__).parent.parent.parent / "presets" / "nanobanana_system_prompt_zh.md"

# Load system prompt content for use in nodes
def _load_system_prompt(lang="en"):
    path = SYSTEM_PROMPT_EN_PATH if lang == "en" else SYSTEM_PROMPT_ZH_PATH
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except:
            pass
    # Fallback embedded rules
    return """
You are a professional AI image prompt optimization expert. Transform simple inputs into high-quality structured prompts.

Core Rules:
1. Replace feeling words with professional terms: Wong Kar-wai, Saul Leiter, Kodak Vision3 500T, Swiss International Style, Bauhaus
2. Replace adjectives with quantified parameters: 90mm f/1.8, 45-degree overhead, shallow depth of field, Dutch angle, volumetric light, 16mm wide-angle
3. Add negative constraints: No text, No low-key dark lighting, No high-saturation neon, Product must not be distorted, Do not obscure face
4. Sensory stacking: Visual + Tactile (tangible texture) + Olfactory (aroma) + Motion (trembles, steam wisps) + Temperature (steamy warmth, moist)
5. Group and cluster: Visual Rules, Lighting & Style, Overall Feel, Constraints
6. Format adaptation: Simple scenes natural paragraphs, complex scenes structured groupings
"""

NANOBANANA_SYSTEM_PROMPT = _load_system_prompt("en")


def _load_trending_prompts():
    global _TRENDING_CACHE
    if _TRENDING_CACHE is not None:
        return _TRENDING_CACHE
    try:
        if _TRENDING_PATH.exists():
            with open(_TRENDING_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _TRENDING_CACHE = data
                _log.info(f"[Trending Prompts] Loaded {len(data)} prompts from { _TRENDING_PATH }")
                return data
    except Exception as e:
        _log.warning(f"Failed to load trending prompts: {e}")
    _TRENDING_CACHE = []
    return []


def _extract_placeholders(prompt: str) -> List[str]:
    """Extract placeholders like [OBJECT], [BRAND NAME], [LANDMARK], 【...】"""
    pattern = re.compile(r'\[[^\]]+\]|【[^】]+】')
    found = pattern.findall(prompt)
    # Deduplicate but preserve order
    seen = set()
    result = []
    for ph in found:
        if ph not in seen:
            seen.add(ph)
            result.append(ph)
    return result


def _filter_prompts(
    prompts: List[Dict],
    category: str = "all",
    model: str = "all",
    min_likes: int = 0,
    search_query: str = "",
    max_results: int = 50,
) -> List[Dict]:
    filtered = prompts

    if category != "all":
        filtered = [p for p in filtered if category in p.get("categories", [])]

    if model != "all":
        filtered = [p for p in filtered if p.get("model", "") == model]

    if min_likes > 0:
        filtered = [p for p in filtered if p.get("likes", 0) >= min_likes]

    if search_query.strip():
        q = search_query.lower()
        filtered = [
            p for p in filtered
            if q in p.get("prompt", "").lower()
            or any(q in str(cat).lower() for cat in p.get("categories", []))
            or q in p.get("id", "").lower()
        ]

    # Sort by likes/views score
    filtered.sort(key=lambda x: (x.get("likes", 0), x.get("views", 0)), reverse=True)

    return filtered[:max_results]


class OllamaTrendingPromptLoader:
    """Trending Prompt Loader - Searchable preset library for 1,446 nanobanana trending prompts"""

    CATEGORY = "ComfyUI-OMG/Prompt/Trending"
    FUNCTION = "load_trending"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_prompt", "placeholders_needed", "metadata_json", "all_filtered_prompts", "total_found", "top_categories")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_query": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Search query - e.g. 'biker', 'product photography', 'isometric infographic', 'fashion campaign' - searches prompt content, categories, id"
                }),
                "category": (["all", "Photography", "Illustration & 3D", "Product & Brand", "Food & Drink", "Poster Design", "UI & Graphic"], {
                    "default": "all",
                    "tooltip": "Filter by category - Photography 533, Illustration & 3D 370, Product & Brand 239, Food & Drink 156, Poster Design 146, UI & Graphic 52"
                }),
                "model": (["all", "nanobanana", "gptimage"], {
                    "default": "all",
                    "tooltip": "Filter by model - nanobanana 1148, gptimage 298"
                }),
                "min_likes": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Minimum likes threshold - e.g. 1000 for viral only, 0 for all"
                }),
                "rank_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1446,
                    "step": 1,
                    "tooltip": "Pick Nth result from filtered sorted list (1 = top liked)"
                }),
                "max_results": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Max results to consider for filtering - performance tweak"
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "custom_trending_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional custom path to prompts.json - leave empty uses built-in presets/nanobanana_trending_1446.json"
                }),
            }
        }

    def load_trending(
        self,
        search_query: str = "",
        category: str = "all",
        model: str = "all",
        min_likes: int = 0,
        rank_index: int = 1,
        max_results: int = 50,
        custom_trending_path: str = "",
    ):
        # Load prompts
        if custom_trending_path.strip():
            try:
                custom_path = Path(custom_trending_path)
                if custom_path.exists():
                    with open(custom_path, "r", encoding="utf-8") as f:
                        all_prompts = json.load(f)
                else:
                    all_prompts = _load_trending_prompts()
            except Exception as e:
                _log.warning(f"Failed to load custom path {custom_trending_path}: {e}, using built-in")
                all_prompts = _load_trending_prompts()
        else:
            all_prompts = _load_trending_prompts()

        if not all_prompts:
            return ("", "", "{}", "", 0, "No prompts loaded")

        filtered = _filter_prompts(all_prompts, category, model, min_likes, search_query, max_results)

        total_found = len(filtered)
        if total_found == 0:
            return ("No matching prompts found", "", "{}", "", 0, f"Category {category}")

        # Clamp rank_index
        idx = max(0, min(rank_index - 1, total_found - 1))
        selected = filtered[idx]

        selected_prompt = selected.get("prompt", "")
        placeholders = _extract_placeholders(selected_prompt)
        placeholders_str = ", ".join(placeholders) if placeholders else "No placeholders - ready to use"

        metadata = {
            "rank": selected.get("rank"),
            "id": selected.get("id"),
            "author": selected.get("author"),
            "author_name": selected.get("author_name"),
            "likes": selected.get("likes"),
            "views": selected.get("views"),
            "model": selected.get("model"),
            "categories": selected.get("categories"),
            "placeholders": placeholders,
            "likes": selected.get("likes"),
            "source_url": selected.get("source_url"),
        }

        # All filtered prompts as simple list for preview
        all_filtered_text = "\n\n---\n\n".join(
            [f"Rank {p.get('rank')} | Likes {p.get('likes')} | {p.get('categories')} | ID {p.get('id')}\n{p.get('prompt','')[:400]}..." for p in filtered[:10]]
        )

        # Top categories for this filter
        from collections import Counter
        cat_counter = Counter()
        for p in filtered:
            for c in p.get("categories", []):
                cat_counter[c] += 1
        top_cats = ", ".join([f"{k}:{v}" for k, v in cat_counter.most_common(6)])

        return (
            selected_prompt,
            placeholders_str,
            json.dumps(metadata, indent=2, ensure_ascii=False),
            all_filtered_text,
            total_found,
            top_cats,
        )


class OllamaTrendingPromptFiller:
    """Trending Prompt Filler - Auto-fills placeholders like [OBJECT], [BRAND NAME] via user inputs or Ollama concept-aware"""

    CATEGORY = "ComfyUI-OMG/Prompt/Trending"
    FUNCTION = "fill_placeholders"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("filled_prompt", "remaining_placeholders", "char_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trending_prompt_template": ("STRING", {
                    "multiline": True,
                    "default": "Create a technical infographic of [OBJECT] with a 45-degree isometric 3D perspective showing the device slightly tilted to reveal depth and dimension.",
                    "tooltip": "Prompt template with placeholders like [OBJECT], [BRAND NAME], [LANDMARK], [COLOR], etc. - from Trending Prompt Loader"
                }),
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "a fast pace biker driving through busy new york city road",
                    "tooltip": "Your concept to fill placeholders - e.g. 'motorcycle', 'Nike', 'Taj Mahal' - concept-aware filling biker vs product vs landmark"
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "ollama_model": ("OLLAMA_MODEL", {
                    "tooltip": "Optional Ollama model to auto-fill placeholders concept-aware - if missing uses direct template replacement"
                }),
                "brand_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Fill [BRAND NAME], [brand name], [BRAND]"}),
                "object_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Fill [OBJECT], [PRODUCT], [product], [FOOD], [DISH]"}),
                "color": ("STRING", {"default": "", "multiline": False, "tooltip": "Fill [COLOR], [HERO COLOR], [BACKGROUND COLOR]"}),
                "landmark": ("STRING", {"default": "", "multiline": False, "tooltip": "Fill [LANDMARK], [LANDMARK NAME]"}),
                "custom_placeholder_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Custom placeholder mapping JSON - e.g. {\"[OBJECT]\": \"motorcycle\", \"[BRAND NAME]\": \"Nike\"} - overrides auto-fill"
                }),
                "use_ollama_for_unfilled": ("BOOLEAN", {"default": False, "tooltip": "Use Ollama to intelligently fill remaining unfilled placeholders concept-aware"}),
            }
        }

    def fill_placeholders(
        self,
        trending_prompt_template: str,
        concept: str = "",
        ollama_model: Dict[str, Any] = None,
        brand_name: str = "",
        object_name: str = "",
        color: str = "",
        landmark: str = "",
        custom_placeholder_json: str = "{}",
        use_ollama_for_unfilled: bool = False,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        template = trending_prompt_template

        # Build placeholder map from explicit inputs + concept-aware auto-fill
        placeholder_map = {}

        # Custom JSON overrides first
        try:
            if custom_placeholder_json.strip():
                custom_map = json.loads(custom_placeholder_json)
                if isinstance(custom_map, dict):
                    placeholder_map.update(custom_map)
        except Exception as e:
            _log.warning(f"Failed to parse custom placeholder JSON: {e}")

        # Explicit inputs
        if brand_name.strip():
            for key in ["[BRAND NAME]", "[brand name]", "[BRAND]", "[brand]", "[BRAND NAME]: The name of the brand."]:
                placeholder_map[key] = brand_name.strip()
        if object_name.strip():
            for key in ["[OBJECT]", "[PRODUCT]", "[product]", "[FOOD]", "[DISH]", "[PRODUCT TYPE]", "[PRODUCT NAME]", "[product name]", "[TYPE]", "[ITEM]"]:
                placeholder_map[key] = object_name.strip()
        if color.strip():
            for key in ["[COLOR]", "[HERO COLOR]", "[BACKGROUND COLOR]", "[COLOR SCHEME]"]:
                placeholder_map[key] = color.strip()
        if landmark.strip():
            for key in ["[LANDMARK]", "[LANDMARK NAME]"]:
                placeholder_map[key] = landmark.strip()

        # Concept-aware auto-fill for common placeholders if not already filled
        concept_lower = concept.lower() if concept else ""
        # If concept contains biker/motorcycle, fill OBJECT with motorcycle etc.
        auto_fills = {}

        if concept.strip():
            # For [OBJECT] - if concept is biker, fill with motorcycle, etc.
            if "[OBJECT]" not in placeholder_map:
                if any(k in concept_lower for k in ["biker", "motorcycle", "bike"]):
                    auto_fills["[OBJECT]"] = "motorcycle"
                elif any(k in concept_lower for k in ["car", "vehicle", "truck"]):
                    auto_fills["[OBJECT]"] = concept.strip()[:50]
                else:
                    # Use concept as object if no explicit object
                    if len(concept.strip()) < 60:
                        auto_fills["[OBJECT]"] = concept.strip()

            # For [BRAND NAME] - if concept has brand-like, else leave
            # For [LANDMARK] - if concept mentions landmark-like places, use concept

            # For [COLOR] - detect colors in concept
            colors = ["red", "blue", "black", "white", "green", "yellow", "orange", "purple", "pink", "gold", "silver", "matte black"]
            for col in colors:
                if col in concept_lower and "[COLOR]" not in placeholder_map:
                    auto_fills["[COLOR]"] = col
                    break

        # Merge auto_fills if not already in placeholder_map
        for k, v in auto_fills.items():
            if k not in placeholder_map:
                placeholder_map[k] = v

        # If use_ollama_for_unfilled and ollama_model provided, use LLM to fill remaining placeholders
        remaining_placeholders = _extract_placeholders(template)
        unfilled = [ph for ph in remaining_placeholders if ph not in placeholder_map]

        if use_ollama_for_unfilled and unfilled and ollama_model:
            try:
                from ...ollama_client import generate as ollama_generate

                # Build prompt for LLM to fill placeholders
                placeholders_list = "\n".join([f"- {ph}" for ph in unfilled])
                system_prompt = f"""You are a placeholder filler for trending image prompts. Given a concept and list of placeholders from a prompt template, fill each placeholder with appropriate value derived from concept.

Concept: {concept}

Placeholders to fill:
{placeholders_list}

Rules:
- For [OBJECT], [PRODUCT], etc. - use main subject from concept
- For [BRAND NAME], [BRAND] - if concept has brand, use it, else generate plausible brand or use concept keyword
- For [COLOR], [HERO COLOR] - detect color from concept or choose appropriate
- For [LANDMARK] - if concept mentions landmark, use it
- Return ONLY valid JSON mapping placeholder to value, e.g. {{"[OBJECT]": "motorcycle", "[BRAND NAME]": "Nike"}}
- Keep values short, 1-5 words, no brackets
"""

                user_prompt = f"Concept: {concept}\n\nPlaceholders: {', '.join(unfilled)}\n\nFill each placeholder concept-aware and return JSON mapping."

                cfg = ollama_model

                # Think mode handling - available on all Ollama nodes

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
                    num_ctx=cfg.get("num_ctx", 8192, think=think, filter_thinking=filter_thinking),
                    num_predict=1000,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                )
                try:
                    from ...utils.text_utils import extract_json_block, filter_thinking
                    parsed = extract_json_block(raw)
                    if isinstance(parsed, dict):
                        placeholder_map.update(parsed)
                        _log.info(f"Ollama filled placeholders: {parsed}")
                except Exception as e:
                    _log.warning(f"Failed to parse Ollama placeholder fill JSON: {e}")
            except Exception as e:
                _log.warning(f"Ollama placeholder fill failed: {e}")

        # Now do replacement
        filled = template
        for ph, val in placeholder_map.items():
            # Replace all occurrences, case-sensitive for bracketed
            filled = filled.replace(ph, val)

        # Also try case-insensitive for some common ones like [Brand Name] vs [BRAND NAME]
        # Use regex with ignore case for remaining
        remaining_after = _extract_placeholders(filled)

        return (
            filled,
            ", ".join(remaining_after) if remaining_after else "All placeholders filled - ready to use",
            len(filled),
        )


class OllamaPromptBuilderWithTrending:
    """Prompt Builder with Trending Inspiration - Uses Nanobanana 6 rules + few-shot trending examples to enhance concept"""

    CATEGORY = "ComfyUI-OMG/Prompt/Trending"
    FUNCTION = "build_with_trending"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "trending_inspiration_used", "negative_prompt", "char_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept": ("STRING", {
                    "multiline": True,
                    "default": "a fast pace biker driving through busy new york city road",
                    "tooltip": "Few words allowed - e.g. 'biker', 'product photo of coffee can', 'freshers party' - will be expanded using Nanobanana 6 rules + trending inspiration"
                }),
            },
            "optional": {
                "think_mode": (["off", "on", "low", "medium", "high"], {"default": "off", "tooltip": "Ollama thinking mode - off/on/low/medium/high"}),
                "filter_thinking": ("BOOLEAN", {"default": True, "tooltip": "Filter thinking tags <think>...</think> from output"}),
                "ollama_model": ("OLLAMA_MODEL", {"tooltip": "Ollama model for enhancement - uses Nanobanana system prompt rules"}),
                "trending_category": (["all", "Photography", "Illustration & 3D", "Product & Brand", "Food & Drink", "Poster Design", "UI & Graphic"], {"default": "all", "tooltip": "Which trending category to use as inspiration - Product & Brand for product shots, Photography for portraits, etc."}),
                "trending_search": ("STRING", {"default": "", "multiline": False, "tooltip": "Search trending prompts for inspiration - e.g. 'isometric infographic', 'fashion campaign', 'product photography'"}),
                "num_trending_examples": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "tooltip": "Number of trending prompts to use as few-shot inspiration (1-10)"}),
                "min_likes": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 50, "tooltip": "Minimum likes for trending inspiration - higher = more viral proven"}),
                "use_nanobanana_rules": ("BOOLEAN", {"default": True, "tooltip": "Use Nanobanana 6 rules: Replace feeling words with professional terms (Wong Kar-wai, Kodak Vision3 500T), Replace adjectives with quantified params (90mm f/1.8, shallow depth), Add negative constraints, Sensory stacking, Group and cluster, Format adaptation"}),
                "target_for": (["image", "video H3 4000-6000", "video LTX 2.5 single/multi-shot", "product photography", "poster design", "food photography"], {"default": "image", "tooltip": "Target use - image, video H3 prompt (4000-6000 chars with alignment line + 3 fields), video LTX 2.5 (single/multi-shot with explicit cuts), product, poster, food"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Additional system guidance on top of Nanobanana rules"}),
                "small_model_mode": ("BOOLEAN", {"default": False, "tooltip": "For 9B or smaller models - lean prompt, 8192 ctx, retries"}),
            }
        }

    def build_with_trending(
        self,
        concept: str,
        ollama_model: Dict[str, Any] = None,
        trending_category: str = "all",
        trending_search: str = "",
        num_trending_examples: int = 3,
        min_likes: int = 100,
        use_nanobanana_rules: bool = True,
        target_for: str = "image",
        custom_system_prompt: str = "",
        small_model_mode: bool = False,
        think_mode: str = "off",
        filter_thinking: bool = True,
    ):
        all_prompts = _load_trending_prompts()
        if not all_prompts:
            return (f"Failed to load trending prompts, using concept directly: {concept}", "", "No trending loaded", str(len(concept)))

        # Filter trending for inspiration
        filtered = _filter_prompts(all_prompts, trending_category, "all", min_likes, trending_search, max_results=100)

        # If no results for search, try without search
        if not filtered and trending_search.strip():
            filtered = _filter_prompts(all_prompts, trending_category, "all", min_likes, "", max_results=100)

        # Pick top N by likes
        examples = filtered[:num_trending_examples]

        # Build few-shot examples text
        few_shot_text = ""
        if examples:
            few_shot_parts = []
            for idx, ex in enumerate(examples, 1):
                # Truncate prompt for few-shot
                prompt_text = ex.get("prompt", "")[:800]
                few_shot_parts.append(f"Trending Example {idx} (Likes {ex.get('likes')}, Categories {ex.get('categories')}, Model {ex.get('model')}):\n{prompt_text}")
            few_shot_text = "\n\n---\n\n".join(few_shot_parts)

        # Build system prompt with Nanobanana rules
        system_base = NANOBANANA_SYSTEM_PROMPT if use_nanobanana_rules else "You are a professional prompt optimization expert."

        # Add target-specific guidance
        target_guidance = ""
        if "video H3" in target_for.lower():
            target_guidance = """
Target: Video H3 prompt 4000-6000 chars with official structure:
- Alignment line exact verbatim when applicable (I2VA: For the target video, at 0.00 seconds... / FL2VA: How the reference pictures align... 0.00-second mark... [S.SS]-second mark)
- One blank line
- Exactly 3 fields: integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:03.500, camera cuts to... , overall_soundscape: ..., non_diegetic_music: ...
- Duration-aware cuts, concept-aware (biker=engine roar not bar murmur), preserve dialogue verbatim
- Also apply Nanobanana rules: quantified params 90mm f/1.8, professional terms Wong Kar-wai, sensory stacking
"""
        elif "LTX" in target_for:
            target_guidance = """
Target: LTX 2.5 video prompt per official guide https://docs.ltx.io/open-source-model/usage-guides/prompting-guide
- Single-shot: single flowing paragraph 4-8 sentences present tense
- Multi-shot 2-4: ONE chronological paragraph with explicit cuts named natural language like 'A hard cut transitions to...' + re-established shot + identity consistent + audio continuity at every cut
- All types: dialogue in quotes with language/accent/volume, music ambient, on-screen text short prominent, character physical cues not abstract labels, present tense
- Dub-It template: [Speaker] is speaking [Language/Accent], saying: "Dialogue" for speech replacement
- Text-to-audio: audio-only mode
"""
        elif "product" in target_for.lower():
            target_guidance = """
Target: Product & Brand photography - use high-end commercial product photography rules:
- Hasselblad, Apple product aesthetics, Studio lighting high dynamic range
- 90mm, f/1.8, shallow depth of field, extreme clarity micro-detail
- No product distortion, no text watermarks, no low-key dark lighting
"""
        elif "poster" in target_for.lower():
            target_guidance = """
Target: Poster Design - use Swiss International Style, Bauhaus, grid system, minimal color palette, clear information hierarchy
"""
        elif "food" in target_for.lower():
            target_guidance = """
Target: Food & Drink photography - High-end culinary magazine style, 45-degree overhead, soft side light, texture tangible
"""

        full_system = system_base
        if target_guidance.strip():
            full_system += "\n\n" + target_guidance
        if custom_system_prompt.strip():
            full_system += "\n\nAdditional user guidance (high priority): " + custom_system_prompt.strip()

        # Build user prompt
        user_prompt = f"""Concept to enhance (few words allowed, expand to high-quality structured prompt):
{concept}

Target for: {target_for}

Use Nanobanana 6 rules:
1. Replace feeling words with professional terms (Wong Kar-wai, Saul Leiter, Kodak Vision3 500T, Swiss International Style, Bauhaus)
2. Replace adjectives with quantified parameters (90mm f/1.8, 45-degree overhead angle, shallow depth of field, Dutch angle, volumetric light, 16mm wide-angle)
3. Add negative constraints (No text, No low-key dark lighting, No high-saturation neon, Product must not be distorted, Do not obscure face)
4. Sensory stacking (Visual + Tactile tangible texture + Olfactory aroma + Motion trembles/steam + Temperature steamy warmth)
5. Group and cluster (Visual Rules, Lighting & Style, Overall Feel, Constraints)
6. Format adaptation (Simple scenes natural paragraphs, Complex scenes structured groupings)

{f"Trending inspiration - use these {len(examples)} viral prompts as few-shot examples of high-quality structure (do not copy verbatim, use their structure/style):\n\n{few_shot_text}" if few_shot_text else "No trending examples - use rules only"}

Generate enhanced prompt immediately. Output ONLY enhanced prompt text, no markdown, no explanation, no "Rewritten prompt:" prefix.
"""

        # If no Ollama model, fallback to template using trending examples directly
        if ollama_model is None:
            # Simple template: take top trending prompt structure and fill with concept
            if examples:
                top_example = examples[0]
                template = top_example.get("prompt", "")
                placeholders = _extract_placeholders(template)
                # Simple fill placeholders with concept
                filled = template
                for ph in placeholders[:3]:  # Fill first 3 placeholders
                    if "[OBJECT]" in ph or "[PRODUCT]" in ph or "[FOOD]" in ph or "[DISH]" in ph:
                        filled = filled.replace(ph, concept[:60])
                    elif "[BRAND" in ph:
                        filled = filled.replace(ph, concept.split()[0].title() if concept else "Brand")
                    elif "[COLOR]" in ph:
                        filled = filled.replace(ph, "matte black")
                    elif "[LANDMARK]" in ph:
                        filled = filled.replace(ph, concept[:60])
                # Add Nanobanana rules summary
                enhanced = f"{filled}\n\n---\n\nEnhanced from concept '{concept}' using Nanobanana rules + trending rank {top_example.get('rank')} likes {top_example.get('likes')} model {top_example.get('model')} categories {top_example.get('categories')}"
                negative = "No text, No low-key dark lighting, No high-saturation neon, Product must not be distorted, Do not obscure face, No lowres, No blurry"
                inspiration = f"Used trending example Rank {top_example.get('rank')} Likes {top_example.get('likes')} ID {top_example.get('id')} - {top_example.get('categories')} - Placeholders {placeholders}"
                return (enhanced, inspiration, negative, str(len(enhanced)))
            else:
                # No examples, just enhance concept with rules
                enhanced = f"{concept} - {target_for} - Enhanced with Nanobanana rules: 90mm f/1.8, shallow depth of field, volumetric light, Wong Kar-wai aesthetics, Kodak Vision3 500T, Swiss International Style, tactile texture tangible, aroma, motion trembles, steamy warmth. Visual Rules: {concept}. Lighting & Style: cinematic studio lighting controlled highlights shadows. Overall Feel: high-end. Constraints: No text, No distortion."
                return (enhanced, "No trending examples found, used rules only", "No text, No low-key dark lighting, No high-saturation neon, No distortion", str(len(enhanced)))

        # With Ollama model - generate via LLM
        cfg = ollama_model
        small_ctx = 8192 if small_model_mode else cfg.get("num_ctx", 12288)
        small_predict = 1500 if small_model_mode else 3000
        small_temp = 0.4 if small_model_mode else cfg.get("temperature", 0.7)

        try:
            from ...ollama_client import generate as ollama_generate
            raw = ollama_generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt=user_prompt,
                system=full_system,
                temperature=small_temp,
                num_ctx=small_ctx,
                num_predict=small_predict,
                seed=cfg.get("seed", -1, think=think, filter_thinking=filter_thinking),
                keep_alive=cfg.get("keep_alive", "5m"),
                response_format="text",
            )
            # Clean up - remove markdown, "Rewritten prompt:" prefix etc per system prompt rule
            cleaned = raw.strip()
            # Remove common prefixes
            for prefix in ["Rewritten prompt:", "Enhanced prompt:", "Output:", "Result:"]:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
            # Remove markdown code blocks
            cleaned = re.sub(r"^```(?:\w+)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            cleaned = cleaned.strip()

            # Use first 3 examples as inspiration reference
            inspiration_used = f"Used {len(examples)} trending examples: " + ", ".join([f"Rank {e.get('rank')} Likes {e.get('likes')} ({', '.join(e.get('categories',[])[:2])})" for e in examples[:3]])

            negative = "No text, No low-key dark lighting, No high-saturation neon colors or artificial plastic textures, Product must not be distorted, warped, or redesigned, Do not obscure the face, No lowres, blurry, bad anatomy, watermark"

            return (cleaned, inspiration_used, negative, str(len(cleaned)))

        except Exception as e:
            _log.warning(f"Trending enhancer LLM failed: {e}, falling back to template")
            # Fallback same as no Ollama case
            if examples:
                top_example = examples[0]
                template = top_example.get("prompt", "")
                filled = template
                # Simple fill [OBJECT] etc with concept
                for ph in _extract_placeholders(template)[:3]:
                    filled = filled.replace(ph, concept[:60])
                enhanced = f"{filled}\n\nEnhanced from concept '{concept}' using trending rank {top_example.get('rank')} likes {top_example.get('likes')}"
                inspiration = f"Fallback used trending Rank {top_example.get('rank')} due to LLM error {e}"
                return (enhanced, inspiration, "No text, No distortion", str(len(enhanced)))
            else:
                enhanced = f"{concept} - Enhanced with Nanobanana rules fallback due to error {e}"
                return (enhanced, f"LLM failed: {e}", "No text", str(len(enhanced)))


NODE_CLASS_MAPPINGS = {
    "OllamaTrendingPromptLoader": OllamaTrendingPromptLoader,
    "OllamaTrendingPromptFiller": OllamaTrendingPromptFiller,
    "OllamaPromptBuilderWithTrending": OllamaPromptBuilderWithTrending,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaTrendingPromptLoader": "Trending Prompt Loader - 1446 Nanobanana",
    "OllamaTrendingPromptFiller": "Trending Prompt Filler - Auto Fill Placeholders",
    "OllamaPromptBuilderWithTrending": "Prompt Builder - With Trending Inspiration (Nanobanana Rules)",
}
