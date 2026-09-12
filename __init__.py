"""ComfyUI-OMG custom-node registration - Robust per-module try/except so one broken module doesn't kill pack (Fill-Nodes pattern, skill C3)"""

# A repository checkout may live in a hyphenated directory (``ComfyUI-OMG``).
# ComfyUI loads it with a package-aware import, while generic tools such as
# pytest can import this file as top-level ``__init__``. Establish a synthetic
# package only for that tooling case so relative imports remain valid.
# ruff: noqa: E402
if not __package__:
    from pathlib import Path
    __path__ = [str(Path(__file__).resolve().parent)]
    if __spec__ is not None:
        __spec__.submodule_search_locations = __path__
        __package__ = __spec__.parent
    else:
        __package__ = __name__

import logging
import traceback

_log = logging.getLogger(__name__)

from .api import register_routes
from .stability import stability_counts
from .utils.system_prompt import install_custom_system_prompt_inputs
from .version import __version__

# ── Aggregators with per-module try/except (skill C3: Fill-Nodes 204 nodes pattern) ──
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _safe_import(module_path, class_names, display_names=None):
    """Import classes from module with try/except, return dicts"""
    try:
        # Dynamic import via relative import
        parts = module_path.split('.')
        # module_path like ".nodes.core.ollama_model_loader"
        mod = __import__(module_path, fromlist=class_names)
        result = {}
        disp = {}
        for cls_name in class_names:
            try:
                cls = getattr(mod, cls_name)
                result[cls_name] = cls
                if display_names and cls_name in display_names:
                    disp[cls_name] = display_names[cls_name]
            except Exception as e:
                _log.warning(f"[OMG] Failed to get class {cls_name} from {module_path}: {e}")
        return result, disp
    except Exception as e:
        _log.warning(f"[OMG] Failed to import module {module_path}: {e}\n{traceback.format_exc()}")
        return {}, {}

def _merge(mappings, display):
    NODE_CLASS_MAPPINGS.update(mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display)

# Core - try each individually so one failure doesn't kill pack
try:
    from .nodes.core.ollama_model_loader import OllamaModelLoader
    from .nodes.core.ollama_text_generate import OllamaTextGenerate
    from .nodes.core.ollama_chat import OllamaChat
    from .nodes.core.ollama_vision import OllamaVision
    from .nodes.core.ollama_embeddings import OllamaEmbeddings
    from .nodes.core.ollama_prompt_builder import OllamaPromptBuilder, OllamaPromptBuilderBatch, OllamaSinglePromptBuilder
    from .nodes.core.ollama_json_extractor import OllamaJSONExtractor
    from .nodes.core.ollama_model_manager import OllamaModelManager
    from .nodes.core.ollama_model_inspector import OllamaModelInspector
    from .nodes.core.ollama_cache_control import OllamaTaskCacheControl
    from .nodes.core.ollama_structured_generate import OllamaStructuredGenerate
    _merge({
        "OllamaModelLoader": OllamaModelLoader,
        "OllamaTextGenerate": OllamaTextGenerate,
        "OllamaChat": OllamaChat,
        "OllamaVision": OllamaVision,
        "OllamaEmbeddings": OllamaEmbeddings,
        "OllamaPromptBuilder": OllamaPromptBuilder,
        "OllamaPromptBuilderBatch": OllamaPromptBuilderBatch,
        "OllamaSinglePromptBuilder": OllamaSinglePromptBuilder,
        "OllamaJSONExtractor": OllamaJSONExtractor,
        "OllamaModelManager": OllamaModelManager,
        "OllamaModelInspector": OllamaModelInspector,
        "OllamaTaskCacheControl": OllamaTaskCacheControl,
        "OllamaStructuredGenerate": OllamaStructuredGenerate,
    }, {
        "OllamaModelLoader": "Ollama Model Loader",
        "OllamaTextGenerate": "Ollama Text Generate",
        "OllamaChat": "Ollama Chat",
        "OllamaVision": "Ollama Vision (Multi-modal)",
        "OllamaEmbeddings": "Ollama Embeddings",
        "OllamaPromptBuilder": "Ollama Prompt Builder",
        "OllamaPromptBuilderBatch": "Ollama Prompt Builder - Batch",
        "OllamaSinglePromptBuilder": "Ollama Single Prompt Builder",
        "OllamaJSONExtractor": "Ollama JSON Extractor",
        "OllamaModelManager": "Ollama Model Manager",
        "OllamaModelInspector": "Ollama Model Inspector",
        "OllamaTaskCacheControl": "Ollama Task Cache Control",
        "OllamaStructuredGenerate": "Ollama Structured Generate",
    })
except Exception as e:
    _log.warning(f"[OMG] Core nodes failed: {e}\n{traceback.format_exc()}")

# Prompt enhancer
try:
    from .nodes.prompt.prompt_enhancer import OllamaPromptEnhancer
    _merge({"OllamaPromptEnhancer": OllamaPromptEnhancer}, {"OllamaPromptEnhancer": "Ollama Prompt Enhancer"})
except Exception as e:
    _log.warning(f"[OMG] Prompt enhancer failed: {e}")

# Scene
try:
    from .nodes.scene.scene_director import OllamaSceneDirector
    from .nodes.scene.world_bible import (
        OllamaWorldBibleCreate,
        OllamaWorldBibleUpdate,
        OllamaWorldBibleToPrompt,
        OllamaWorldBibleInspect,
    )
    _merge({
        "OllamaSceneDirector": OllamaSceneDirector,
        "OllamaWorldBibleCreate": OllamaWorldBibleCreate,
        "OllamaWorldBibleUpdate": OllamaWorldBibleUpdate,
        "OllamaWorldBibleToPrompt": OllamaWorldBibleToPrompt,
        "OllamaWorldBibleInspect": OllamaWorldBibleInspect,
    }, {
        "OllamaSceneDirector": "Ollama Scene Director",
        "OllamaWorldBibleCreate": "World Bible Create",
        "OllamaWorldBibleUpdate": "World Bible Update",
        "OllamaWorldBibleToPrompt": "World Bible to Prompt",
        "OllamaWorldBibleInspect": "World Bible Inspect",
    })
except Exception as e:
    _log.warning(f"[OMG] Scene nodes failed: {e}")

# Image
try:
    from .nodes.image.style_transfer import OllamaStyleTransfer
    from .nodes.image.image_analyzer import OllamaImageAnalyzer
    from .nodes.image.analysis_quality import OllamaAnalysisQualityInspector
    from .nodes.image.multi_image_consistency import OllamaMultiImageConsistencyInspector
    from .nodes.image.image_edit import OllamaImageEdit
    from .nodes.image.image_sequence_analyzer import OllamaImageSequenceAnalyzer
    from .nodes.image.photoset_folder_analyzer import OllamaPhotosetFolderAnalyzer
    from .nodes.image.analyzer_prompt_saver import OllamaAnalyzerPromptSaver
    from .nodes.image.web_image_tool import OllamaWebImageTool
    from .nodes.image.image_merger import OllamaImageMerger
    _merge({
        "OllamaStyleTransfer": OllamaStyleTransfer,
        "OllamaImageAnalyzer": OllamaImageAnalyzer,
        "OllamaAnalysisQualityInspector": OllamaAnalysisQualityInspector,
        "OllamaMultiImageConsistencyInspector": OllamaMultiImageConsistencyInspector,
        "OllamaImageEdit": OllamaImageEdit,
        "OllamaImageSequenceAnalyzer": OllamaImageSequenceAnalyzer,
        "OllamaPhotosetFolderAnalyzer": OllamaPhotosetFolderAnalyzer,
        "OllamaAnalyzerPromptSaver": OllamaAnalyzerPromptSaver,
        "OllamaWebImageTool": OllamaWebImageTool,
        "OllamaImageMerger": OllamaImageMerger,
    }, {
        "OllamaStyleTransfer": "Ollama Style Transfer",
        "OllamaImageAnalyzer": "Ollama Image Analyzer",
        "OllamaAnalysisQualityInspector": "Image Analysis Quality Inspector",
        "OllamaMultiImageConsistencyInspector": "Multi-Image Consistency Inspector",
        "OllamaImageEdit": "Ollama Image Edit Prompt",
        "OllamaImageSequenceAnalyzer": "Ollama Image Sequence Analyzer",
        "OllamaPhotosetFolderAnalyzer": "Ollama Photoset Folder Analyzer",
        "OllamaAnalyzerPromptSaver": "Ollama Analyzer Prompt Saver",
        "OllamaWebImageTool": "Ollama Web Image Tool",
        "OllamaImageMerger": "Ollama Image Merger",
    })
except Exception as e:
    _log.warning(f"[OMG] Image nodes failed: {e}\n{traceback.format_exc()}")

# Character
try:
    from .nodes.character.character_sheet import OllamaCharacterSheet
    from .nodes.character.character_bible import (
        OllamaCharacterBibleCreate,
        OllamaCharacterBibleUpdate,
        OllamaCharacterBibleToPrompt,
        OllamaCharacterBibleInspect,
    )
    _merge({
        "OllamaCharacterSheet": OllamaCharacterSheet,
        "OllamaCharacterBibleCreate": OllamaCharacterBibleCreate,
        "OllamaCharacterBibleUpdate": OllamaCharacterBibleUpdate,
        "OllamaCharacterBibleToPrompt": OllamaCharacterBibleToPrompt,
        "OllamaCharacterBibleInspect": OllamaCharacterBibleInspect,
    }, {
        "OllamaCharacterSheet": "Ollama Character Sheet",
        "OllamaCharacterBibleCreate": "Character Bible Create",
        "OllamaCharacterBibleUpdate": "Character Bible Update",
        "OllamaCharacterBibleToPrompt": "Character Bible to Prompt",
        "OllamaCharacterBibleInspect": "Character Bible Inspect",
    })
except Exception as e:
    _log.warning(f"[OMG] Character nodes failed: {e}")

# Extended prompt nodes
try:
    from .nodes.prompt.negative_prompt import OllamaNegativePrompt
    from .nodes.prompt.prompt_variations import OllamaPromptVariations
    from .nodes.prompt.trending_prompt_nodes import (
        OllamaTrendingPromptLoader,
        OllamaTrendingPromptFiller,
        OllamaPromptBuilderWithTrending,
    )
    from .nodes.prompt.cookbook_prompt_nodes import CookbookStyleAutoFiller
    from .nodes.prompt.cookbook_dynamic_node import CookbookDynamicInputs
    from .nodes.image.inpaint_prompt import OllamaInpaintPrompt
    from .nodes.character.pose_descriptor import OllamaPoseDescriptor
    from .nodes.character.outfit_generator import OllamaOutfitGenerator
    from .nodes.scene.environment_transform import OllamaEnvironmentTransform
    from .nodes.image.image_comparator import OllamaImageComparator
    from .nodes.scene.regional_prompts import OllamaRegionalPrompts
    from .nodes.prompt.prompt_critic import OllamaPromptCritic
    from .nodes.image.auto_tagger import OllamaAutoTagger
    from .nodes.image.color_palette import OllamaColorPalette
    from .nodes.image.style_identifier import OllamaStyleIdentifier
    from .nodes.scene.aspect_optimizer import OllamaAspectOptimizer
    _merge({
        "OllamaNegativePrompt": OllamaNegativePrompt,
        "OllamaPromptVariations": OllamaPromptVariations,
        "OllamaTrendingPromptLoader": OllamaTrendingPromptLoader,
        "OllamaTrendingPromptFiller": OllamaTrendingPromptFiller,
        "OllamaPromptBuilderWithTrending": OllamaPromptBuilderWithTrending,
        "CookbookStyleAutoFiller": CookbookStyleAutoFiller,
        "CookbookDynamicInputs": CookbookDynamicInputs,
        "OllamaInpaintPrompt": OllamaInpaintPrompt,
        "OllamaPoseDescriptor": OllamaPoseDescriptor,
        "OllamaOutfitGenerator": OllamaOutfitGenerator,
        "OllamaEnvironmentTransform": OllamaEnvironmentTransform,
        "OllamaImageComparator": OllamaImageComparator,
        "OllamaRegionalPrompts": OllamaRegionalPrompts,
        "OllamaPromptCritic": OllamaPromptCritic,
        "OllamaAutoTagger": OllamaAutoTagger,
        "OllamaColorPalette": OllamaColorPalette,
        "OllamaStyleIdentifier": OllamaStyleIdentifier,
        "OllamaAspectOptimizer": OllamaAspectOptimizer,
    }, {
        "OllamaNegativePrompt": "Negative Prompt Generator",
        "OllamaPromptVariations": "Batch Prompt Variations",
        "OllamaTrendingPromptLoader": "Trending Prompt Loader - 1446 Nanobanana",
        "OllamaTrendingPromptFiller": "Trending Prompt Filler - Auto Fill Placeholders",
        "OllamaPromptBuilderWithTrending": "Prompt Builder - With Trending Inspiration (Nanobanana Rules)",
        "CookbookStyleAutoFiller": "Cookbook - Auto (66 Styles)",
        "CookbookDynamicInputs": "Cookbook - Dynamic (66 Styles)",
        "OllamaInpaintPrompt": "Inpainting Prompt Generator",
        "OllamaPoseDescriptor": "Pose Descriptor",
        "OllamaOutfitGenerator": "Outfit Generator",
        "OllamaEnvironmentTransform": "Environment Transformer",
        "OllamaImageComparator": "Image Comparator",
        "OllamaRegionalPrompts": "Regional Prompt Generator",
        "OllamaPromptCritic": "Prompt Critic",
        "OllamaAutoTagger": "Auto Tagger",
        "OllamaColorPalette": "Color Palette Extractor",
        "OllamaStyleIdentifier": "Art Style Identifier",
        "OllamaAspectOptimizer": "Aspect Ratio Optimizer",
    })
except Exception as e:
    _log.warning(f"[OMG] Extended prompt nodes failed: {e}\n{traceback.format_exc()}")

# Builder nodes
try:
    from .nodes.scene.advanced_scene_director import OllamaAdvancedSceneDirector
    from .nodes.character.subject_builder import OllamaSubjectBuilder
    from .nodes.design.lighting_designer import OllamaLightingDesigner
    from .nodes.design.creative_brief import OllamaCreativeBriefCompiler, OllamaCreativeBriefToPrompt
    from .nodes.prompt.prompt_combiner import OllamaPromptCombiner
    from .nodes.prompt.prompt_translator import OllamaPromptTranslator
    from .nodes.prompt.wildcard_generator import OllamaWildcardGenerator
    _merge({
        "OllamaAdvancedSceneDirector": OllamaAdvancedSceneDirector,
        "OllamaSubjectBuilder": OllamaSubjectBuilder,
        "OllamaLightingDesigner": OllamaLightingDesigner,
        "OllamaCreativeBriefCompiler": OllamaCreativeBriefCompiler,
        "OllamaCreativeBriefToPrompt": OllamaCreativeBriefToPrompt,
        "OllamaPromptCombiner": OllamaPromptCombiner,
        "OllamaPromptTranslator": OllamaPromptTranslator,
        "OllamaWildcardGenerator": OllamaWildcardGenerator,
    }, {
        "OllamaAdvancedSceneDirector": "Advanced Scene Director",
        "OllamaSubjectBuilder": "Subject Builder",
        "OllamaLightingDesigner": "Lighting Designer",
        "OllamaCreativeBriefCompiler": "Creative Brief Compiler",
        "OllamaCreativeBriefToPrompt": "Creative Brief to Prompt",
        "OllamaPromptCombiner": "Prompt Combiner",
        "OllamaPromptTranslator": "Prompt Translator",
        "OllamaWildcardGenerator": "Wildcard Generator",
    })
except Exception as e:
    _log.warning(f"[OMG] Builder nodes failed: {e}")

# Character continuity
try:
    from .nodes.character.character_consistency_tools import (
        OllamaCharacterAnchorExtractor,
        OllamaExpressionSheetGenerator,
        OllamaOutfitSheetGenerator,
        OllamaPoseSheetGenerator,
        OllamaPromptContinuityChecker,
    )
    _merge({
        "OllamaCharacterAnchorExtractor": OllamaCharacterAnchorExtractor,
        "OllamaOutfitSheetGenerator": OllamaOutfitSheetGenerator,
        "OllamaPoseSheetGenerator": OllamaPoseSheetGenerator,
        "OllamaExpressionSheetGenerator": OllamaExpressionSheetGenerator,
        "OllamaPromptContinuityChecker": OllamaPromptContinuityChecker,
    }, {
        "OllamaCharacterAnchorExtractor": "Character Consistency Anchor Extractor",
        "OllamaOutfitSheetGenerator": "Outfit Sheet Generator",
        "OllamaPoseSheetGenerator": "Pose Sheet Generator",
        "OllamaExpressionSheetGenerator": "Expression Sheet Generator",
        "OllamaPromptContinuityChecker": "Prompt Continuity Checker",
    })
except Exception as e:
    _log.warning(f"[OMG] Character continuity failed: {e}")

# Video shot director + shot list
try:
    from .nodes.video.video_shot_director import OllamaVideoShotDirector
    from .nodes.video.shot_list import (
        OllamaShotListCreate,
        OllamaShotListUpdate,
        OllamaShotListGetShot,
        OllamaShotListInspect,
    )
    _merge({
        "OllamaVideoShotDirector": OllamaVideoShotDirector,
        "OllamaShotListCreate": OllamaShotListCreate,
        "OllamaShotListUpdate": OllamaShotListUpdate,
        "OllamaShotListGetShot": OllamaShotListGetShot,
        "OllamaShotListInspect": OllamaShotListInspect,
    }, {
        "OllamaVideoShotDirector": "Video Shot Director",
        "OllamaShotListCreate": "Shot List Create",
        "OllamaShotListUpdate": "Shot List Update",
        "OllamaShotListGetShot": "Shot List Get Shot",
        "OllamaShotListInspect": "Shot List Inspect",
    })
except Exception as e:
    _log.warning(f"[OMG] Video shot director failed: {e}")

# Creator/designer/helper
try:
    from .nodes.scene.action_choreographer import OllamaActionChoreographer
    from .nodes.scene.background_generator import OllamaBackgroundGenerator
    from .nodes.utility.controlnet_helper import OllamaControlNetHelper
    from .nodes.character.creature_creator import OllamaCreatureCreator
    from .nodes.prompt.detail_injector import OllamaDetailInjector
    from .nodes.scene.emotion_director import OllamaEmotionDirector
    from .nodes.character.hand_pose_helper import OllamaHandPoseHelper
    from .nodes.utility.lora_suggester import OllamaLoraSuggester
    from .nodes.design.texture_material import OllamaTextureMaterial
    _merge({
        "OllamaActionChoreographer": OllamaActionChoreographer,
        "OllamaBackgroundGenerator": OllamaBackgroundGenerator,
        "OllamaControlNetHelper": OllamaControlNetHelper,
        "OllamaCreatureCreator": OllamaCreatureCreator,
        "OllamaDetailInjector": OllamaDetailInjector,
        "OllamaEmotionDirector": OllamaEmotionDirector,
        "OllamaHandPoseHelper": OllamaHandPoseHelper,
        "OllamaLoraSuggester": OllamaLoraSuggester,
        "OllamaTextureMaterial": OllamaTextureMaterial,
    }, {
        "OllamaActionChoreographer": "Action Choreographer",
        "OllamaBackgroundGenerator": "Background Generator",
        "OllamaControlNetHelper": "ControlNet Helper",
        "OllamaCreatureCreator": "Creature Creator",
        "OllamaDetailInjector": "Detail Injector",
        "OllamaEmotionDirector": "Emotion Director",
        "OllamaHandPoseHelper": "Hand Pose Helper",
        "OllamaLoraSuggester": "LoRA Suggester",
        "OllamaTextureMaterial": "Texture & Material Designer",
    })
except Exception as e:
    _log.warning(f"[OMG] Creator nodes failed: {e}")

# RAG
try:
    from .nodes.rag.rag_nodes import (
        OllamaDocumentsFromText,
        OllamaReferenceFolderLoader,
        OllamaRAGIndexBuild,
        OllamaRAGSearch,
        OllamaRAGIndexSave,
        OllamaRAGIndexLoad,
    )
    _merge({
        "OllamaDocumentsFromText": OllamaDocumentsFromText,
        "OllamaReferenceFolderLoader": OllamaReferenceFolderLoader,
        "OllamaRAGIndexBuild": OllamaRAGIndexBuild,
        "OllamaRAGSearch": OllamaRAGSearch,
        "OllamaRAGIndexSave": OllamaRAGIndexSave,
        "OllamaRAGIndexLoad": OllamaRAGIndexLoad,
    }, {
        "OllamaDocumentsFromText": "RAG Documents from Text",
        "OllamaReferenceFolderLoader": "RAG Reference Folder Loader",
        "OllamaRAGIndexBuild": "RAG Index Build",
        "OllamaRAGSearch": "RAG Semantic Search",
        "OllamaRAGIndexSave": "RAG Index Save",
        "OllamaRAGIndexLoad": "RAG Index Load",
    })
except Exception as e:
    _log.warning(f"[OMG] RAG nodes failed: {e}")

# Evaluation
try:
    from .nodes.evaluation.evaluation_nodes import (
        OllamaPromptEvaluator,
        OllamaImageEvaluator,
        OllamaEvaluationCompare,
        OllamaEvaluationInspect,
    )
    _merge({
        "OllamaPromptEvaluator": OllamaPromptEvaluator,
        "OllamaImageEvaluator": OllamaImageEvaluator,
        "OllamaEvaluationCompare": OllamaEvaluationCompare,
        "OllamaEvaluationInspect": OllamaEvaluationInspect,
    }, {
        "OllamaPromptEvaluator": "Prompt Evaluator",
        "OllamaImageEvaluator": "Image Evaluator",
        "OllamaEvaluationCompare": "Evaluation Compare",
        "OllamaEvaluationInspect": "Evaluation Inspect",
    })
except Exception as e:
    _log.warning(f"[OMG] Evaluation nodes failed: {e}")

# Database
try:
    from .nodes.database.database_generator import OllamaDatabaseGenerator
    _merge({"OllamaDatabaseGenerator": OllamaDatabaseGenerator}, {"OllamaDatabaseGenerator": "Database Generator"})
except Exception as e:
    _log.warning(f"[OMG] Database generator failed: {e}\n{traceback.format_exc()}")

# H3 nodes - solid unified structure, Autogrow refs, system_prompt restored
try:
    from .nodes.video.h3_prompt_generator_ollama_simple import H3PromptGeneratorOllamaSimple
    from .nodes.video.h3_prompt_generator_clip_simple import H3PromptGeneratorClipSimple
    from .nodes.video.ltx_wan_improved_nodes import (
        OllamaLTXVVideoPromptImproved,
        OllamaWanVideoPromptImproved,
    )
    from .nodes.video.ltx_v25_prompt_node import OllamaLTX25Prompt
    from .nodes.scene.image_to_storyboard_v2 import OllamaImageToStoryboardV2
    _merge({
        "H3PromptGeneratorOllamaSimple": H3PromptGeneratorOllamaSimple,
        "H3PromptGeneratorClipSimple": H3PromptGeneratorClipSimple,
        "OllamaLTXVVideoPromptImproved": OllamaLTXVVideoPromptImproved,
        "OllamaWanVideoPromptImproved": OllamaWanVideoPromptImproved,
        "OllamaLTX25Prompt": OllamaLTX25Prompt,
        "OllamaImageToStoryboardV2": OllamaImageToStoryboardV2,
    }, {
        "H3PromptGeneratorOllamaSimple": "H3 Prompt - Ollama",
        "H3PromptGeneratorClipSimple": "H3 Prompt - CLIP",
        "OllamaLTXVVideoPromptImproved": "LTX-V Prompt - Improved",
        "OllamaWanVideoPromptImproved": "Wan Prompt - Improved",
        "OllamaLTX25Prompt": "LTX 2.5 Prompt - All Types Official",
        "OllamaImageToStoryboardV2": "Image to Storyboard V2",
    })
except Exception as e:
    _log.warning(f"[OMG] H3/LTX V2 nodes failed: {e}\n{traceback.format_exc()}")

# ── Non-Ollama utility nodes - DISABLED for now per user request to focus more towards Ollama use case ──
# These nodes do NOT use Ollama (IMAGE, MASK, TEXT, MATH, FLOW, UTILITY, GROUPING, ANIMATION, COLOR, VIDEO, STORYBOARD, COMPOSITION, VFX, LOADERS)
# They are separated in Utility and disabled for now - will be re-enabled later as separate pack or optional
# Per skill: Fewer, smarter nodes beats twenty micro-nodes - pack should be more towards Ollama use case
# Set ENABLE_UTILITY_NODES = True to re-enable them

ENABLE_UTILITY_NODES = False  # Disabled for now - focus on Ollama

if ENABLE_UTILITY_NODES:
    # Loaders and integrated utility/media/composition nodes - per-group isolation with proper importlib (fixes Empty module name warnings)
    try:
        from .nodes.loaders.loading_nodes import (
            NODE_CLASS_MAPPINGS as LOADER_NODES,
            NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY,
        )
        NODE_CLASS_MAPPINGS.update(LOADER_NODES)
        NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY)
    except Exception as e:
        _log.warning(f"[OMG] Loader nodes failed: {e}")

    # Each of these groups already has per-module try/except inside their __init__.py if implemented, but we wrap again with proper relative import
    import importlib
    for group_module in [
        ".nodes.image.image_nodes",
        ".nodes.mask.mask_nodes",
        ".nodes.text.text_nodes",
        ".nodes.math.math_nodes",
        ".nodes.flow.flow_nodes",
        ".nodes.utility.utility_nodes",
        ".nodes.utility.grouping_nodes",
        ".nodes.animation.animation_nodes",
        ".nodes.color.color_nodes",
        ".nodes.video.video_nodes",
        ".nodes.storyboard.storyboard_nodes",
        ".nodes.composition.composition_nodes",
        ".nodes.vfx.vfx_nodes",
    ]:
        try:
            mod = importlib.import_module(group_module, package=__package__)
            NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        except Exception as e:
            _log.warning(f"[OMG] Group {group_module} failed: {e}")
else:
    _log.info("[OMG] Non-Ollama utility nodes DISABLED - focusing on Ollama use case per user request. Enable via ENABLE_UTILITY_NODES=True")

SYSTEM_PROMPT_NODE_IDS = install_custom_system_prompt_inputs(NODE_CLASS_MAPPINGS)

WEB_DIRECTORY = "./web/js"

register_routes(__version__, len(NODE_CLASS_MAPPINGS), stability_counts(NODE_CLASS_MAPPINGS))

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SYSTEM_PROMPT_NODE_IDS",
    "WEB_DIRECTORY",
    "__version__",
]
