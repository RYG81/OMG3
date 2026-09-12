# ComfyUI-OMG Node Catalog

ComfyUI-OMG `0.1.0` currently registers **204 nodes**.

The original 170 node IDs remain compatibility-sensitive; workflows store IDs even when display names change.

## ComfyUI-OMG/Animate

| Display name | Node ID | Purpose |
|---|---|---|
| 🎬 Curve Editor | `OllamaNodesCurveEditor` | Evaluate animation curves defined by JSON keyframes. |
| 🎬 Interpolate | `OllamaNodesInterpolate` | Interpolate between two values with 17 easing functions. |
| 🎬 Scheduler | `OllamaNodesScheduler` | Schedule parameter values over steps: linear, ease in/out, constant, or step function. |
| 🎬 Wave Generator | `OllamaNodesWaveGenerator` | Generate periodic wave values: sine, cosine, triangle, sawtooth, square, bounce. |

## ComfyUI-OMG/Character

| Display name | Node ID | Purpose |
|---|---|---|
| Character Consistency Anchor Extractor | `OllamaCharacterAnchorExtractor` | Extracts compact reusable identity anchors from character text. |
| Creature Creator | `OllamaCreatureCreator` | Design unique fantasy, sci-fi, or hybrid creatures. |
| Expression Sheet Generator | `OllamaExpressionSheetGenerator` | Creates same-character expression sheet prompts. |
| Hand Pose Helper | `OllamaHandPoseHelper` | Generate detailed, AI-optimized hand pose descriptions. |
| Ollama Character Sheet | `OllamaCharacterSheet` | Creates multi-view character sheet prompts from a text description. |
| Outfit Generator | `OllamaOutfitGenerator` | Generates detailed outfit descriptions for characters. |
| Outfit Sheet Generator | `OllamaOutfitSheetGenerator` | Creates same-character outfit sheet prompts. |
| Pose Descriptor | `OllamaPoseDescriptor` | Extracts detailed pose/body position descriptions from images. |
| Pose Sheet Generator | `OllamaPoseSheetGenerator` | Creates same-character pose sheet prompts. |
| Subject Builder | `OllamaSubjectBuilder` | Build detailed, consistent character descriptions. |

## ComfyUI-OMG/Character/Bible

| Display name | Node ID | Purpose |
|---|---|---|
| Character Bible Create | `OllamaCharacterBibleCreate` | Create a versioned Character Bible with identity, face, body, wardrobe, style, immutable anchors, flexible traits, master prompt, and negative prompt. |
| Character Bible Inspect | `OllamaCharacterBibleInspect` | Inspect a typed Character Bible through focused string outputs. |
| Character Bible to Prompt | `OllamaCharacterBibleToPrompt` | Compile a validated Character Bible into General, Flux, SDXL, Qwen, Wan, or LTX prompts. |
| Character Bible Update | `OllamaCharacterBibleUpdate` | Update wardrobe or revise a Character Bible while enforcing character_id and optional identity-protection rules. |

## ComfyUI-OMG/Color

| Display name | Node ID | Purpose |
|---|---|---|
| 🎨 Color Harmonies | `OllamaNodesColorHarmonies` | Generate color harmonies: complementary, analogous, triadic, tetradic, split-complementary, monochromatic. |
| 🎨 Color Picker | `OllamaNodesColorPicker` | Parse a HEX color and output RGB integer and float channels. |
| 🎨 Color → Image | `OllamaNodesColorToImage` | Create a solid color image from a hex color code. |
| 🎨 Gradient Image | `OllamaNodesGradientImage` | Create gradient images: linear, radial, angular, or diamond with optional midpoint color. |

## ComfyUI-OMG/Composition

| Display name | Node ID | Purpose |
|---|---|---|
| 🖌 Animated Text | `OllamaNodesAnimatedText` | Add animated text effects to video frames. Animation Types: - typewriter: Characters appear one by one - fade_in/fade_out: Opacity animation - slide_in_*: Slide from edge - scale_in: Grow from center - bounce_in: Elastic bounce effect Connect to OllamaNodesCounter or time-based values for smooth animation. |
| 🖌 Compose Layers | `OllamaNodesComposeLayers` | Compose up to 6 layers with blend modes and opacity. Blend Modes: - normal: Standard alpha compositing - multiply: Darken - screen: Lighten - overlay: Contrast - add: Additive (good for glow effects) |
| 🖌 Image Overlay | `OllamaNodesImageOverlay` | Overlay a transparent image (PNG) onto background. Useful for: - Watermarks and logos - Character sprites/overlays - UI elements - Visual effects - Decorative elements Tip: Use RGBA PNG images for transparency support. |
| 🖌 Text Badge | `OllamaNodesTextBadge` | Add a text badge/label with colored background. Great for: - Episode numbers - Scene titles - Section labels - Corner watermarks with background |
| 🖌 Text Overlay | `OllamaNodesTextOverlay` | Add styled text overlay to images. Position Modes: - pixel: Exact pixel coordinates - percentage: % of image dimensions (0-100) - center: Centered on canvas - Named positions: top_left, middle_center, bottom_right, etc. Features: - Shadow with configurable offset - Outline/stroke effect - Opacity control - Custom font support |
| 🖌 Video Progress Bar | `OllamaNodesVideoProgress` | Add video progress bar overlay to frames. Features: - Progress bar at top or bottom - Percentage display - Timecode display (current/total) - Customizable colors |
| 🖌 Vignette Effect | `OllamaNodesVignette` | Add vignette (darkened edges) effect. Common in cinematic/storyboard work: - Draws focus to center - Adds dramatic mood - Simulates lens effects |

## ComfyUI-OMG/Core

| Display name | Node ID | Purpose |
|---|---|---|
| Ollama Chat | `OllamaChat` | Ollama Chat |
| Ollama Embeddings | `OllamaEmbeddings` | Ollama Embeddings |
| Ollama Model Inspector | `OllamaModelInspector` | Inspect live Ollama metadata and expose conservative capability flags for routing and validation. |
| Ollama Model Loader | `OllamaModelLoader` | Main Ollama entry node: validate a local/remote model, configure shared sampling/context, and optionally append global custom system instructions to every connected generative node. |
| Ollama Model Manager | `OllamaModelManager` | Ollama Model Manager |
| Ollama Structured Generate | `OllamaStructuredGenerate` | Generate schema-constrained JSON with Ollama, validate it locally, and optionally make one correction attempt when the first response is invalid. |
| Ollama Task Cache Control | `OllamaTaskCacheControl` | Inspect or clear the private, bounded, process-memory cache used by ComfyUI-OMG tasks. Cached prompts and responses are never written to disk. |
| Ollama Text Generate | `OllamaTextGenerate` | Ollama Text Generate |

## ComfyUI-OMG/Design

| Display name | Node ID | Purpose |
|---|---|---|
| Creative Brief Compiler | `OllamaCreativeBriefCompiler` | Compile an idea into a validated, reusable creative brief and a target-model prompt pair. |
| Creative Brief to Prompt | `OllamaCreativeBriefToPrompt` | Deterministically adapt an OMG Creative Brief to General, Flux, SDXL, Qwen, Wan, or LTX prompting. |
| Lighting Designer | `OllamaLightingDesigner` | Design detailed, production-quality lighting setups. |
| Texture & Material Designer | `OllamaTextureMaterial` | Design detailed materials and textures. |

## ComfyUI-OMG/Evaluation

| Display name | Node ID | Purpose |
|---|---|---|
| Evaluation Compare | `OllamaEvaluationCompare` | Compare two OMG Evaluations using their locally normalized scores. |
| Evaluation Inspect | `OllamaEvaluationInspect` | Inspect a typed OMG Evaluation through scores, criteria, issues, and recommendation. |
| Image Evaluator | `OllamaImageEvaluator` | Evaluate a candidate image against source prompt, optional reference image, and typed creative/character/world/shot requirements. |
| Prompt Evaluator | `OllamaPromptEvaluator` | Evaluate a generation prompt with weighted criteria and optional Creative Brief, Character Bible, World Bible, and Shot List requirements. |

## ComfyUI-OMG/Flow

| Display name | Node ID | Purpose |
|---|---|---|
| 🔄 Any Switch | `OllamaNodesAnySwitch` | Universal switch: pass through any type based on boolean condition. |
| 🔄 Float → Int | `OllamaNodesFloatToInt` | Convert FLOAT to INT with rounding mode (round, floor, ceil, truncate). |
| 🔄 If/Else Image | `OllamaNodesIfElseImage` | Output one of two images based on a boolean condition. |
| 🔄 If/Else Text | `OllamaNodesIfElse` | Output if_true when condition is True, otherwise output if_false. |
| 🔄 Int → Float | `OllamaNodesIntToFloat` | Convert an INT value to FLOAT. |
| 🔄 Pack Values | `OllamaNodesPack` | Pack up to 4 key-value pairs into a JSON bundle string. |
| 🔄 Type Convert | `OllamaNodesTypeConvert` | Convert a value between INT, FLOAT, STRING, and BOOLEAN types. Outputs all types simultaneously. |
| 🔄 Unpack Values | `OllamaNodesUnpack` | Unpack values from a JSON bundle by key names. |

## ComfyUI-OMG/Image

| Display name | Node ID | Purpose |
|---|---|---|
| Art Style Identifier | `OllamaStyleIdentifier` | Identifies artistic styles, movements, and techniques. |
| Auto Tagger | `OllamaAutoTagger` | Generates comprehensive tags from images in various formats. |
| Color Palette Extractor | `OllamaColorPalette` | Extracts color palettes from images with hex codes and descriptions. |
| Image Analysis Quality Inspector | `OllamaAnalysisQualityInspector` | Inspect deterministic field confidence, visibility limitations, hedging, person-count, pose/support, clothing contradiction, and reconstruction-format diagnostics. |
| Image Comparator | `OllamaImageComparator` | Compares two images and identifies similarities/differences. |
| Image to Story | `OllamaImageToStory` | Generate stories and narratives from images. |
| Inpainting Prompt Generator | `OllamaInpaintPrompt` | Generates inpainting prompts based on image context. |
| Multi-Image Consistency Inspector | `OllamaMultiImageConsistencyInspector` | Compare structured analyses for identity, clothing, environment, camera, lighting, palette, and style drift using bounded local heuristics with no extra Ollama call. |
| Ollama Analyzer Prompt Saver | `OllamaAnalyzerPromptSaver` | Save analyzer-generated prompts or full analysis text to disk. |
| Ollama Image Analyzer | `OllamaImageAnalyzer` | Extracts detailed composition information from an image using Ollama vision models. |
| Ollama Image Edit Prompt | `OllamaImageEdit` | Builds model-specific image-edit prompts from structured image analysis. |
| Ollama Image Merger | `OllamaImageMerger` | Takes two images and a text instruction, generates a prompt to create a merged composite. |
| Ollama Image Sequence Analyzer | `OllamaImageSequenceAnalyzer` | Analyzes up to four images one by one and creates a sequential video prompt. |
| Ollama Photoset Folder Analyzer | `OllamaPhotosetFolderAnalyzer` | Analyze a folder of same-photoset images and create a consistent prompt. |
| Ollama Style Transfer | `OllamaStyleTransfer` | Analyzes an image's style and generates prompts to apply it to a new subject. |
| Ollama Vision (Multi-modal) | `OllamaVision` | Ollama Vision |
| Ollama Web Image Tool | `OllamaWebImageTool` | Search web images, save them to ComfyUI temp, output an image batch, and optionally analyze them. |
| 🎨 Channel Merge | `OllamaNodesImageChannelMerge` | Merge separate R, G, B channel images into a single RGB image. |
| 🎨 Channel Split | `OllamaNodesImageChannelSplit` | Split an image into its Red, Green, and Blue channels (each output as grayscale). |
| 🎨 Color Adjust | `OllamaNodesImageColorAdjust` | Adjust image brightness, contrast, saturation, hue, and gamma. |
| 🎨 Color Palette | `OllamaNodesColorPalette` | Extract the dominant color palette from an image. Outputs a visual palette strip and hex color codes. |
| 🎨 Grayscale | `OllamaNodesImageToGrayscale` | Convert to grayscale using various methods (luminance, average, per-channel, or custom weights). |
| 🎨 Image Batch | `OllamaNodesImageBatch` | Batch operations: join images into batch, split first/last, get by index, count batch size. |
| 🎨 Image Blend | `OllamaNodesImageBlend` | Blend two images using various blend modes (multiply, screen, overlay, etc.). |
| 🎨 Image Blur | `OllamaNodesImageBlur` | Apply Gaussian, box, or median blur to images. |
| 🎨 Image Compare | `OllamaNodesImageCompare` | Compare two images using side-by-side, overlay, difference, or split view. |
| 🎨 Image Crop | `OllamaNodesImageCrop` | Crop an image to specified coordinates and dimensions. |
| 🎨 Image Flip/Rotate | `OllamaNodesImageFlip` | Flip or rotate images (horizontal, vertical, 90°, 180°). |
| 🎨 Image Grid | `OllamaNodesImageGrid` | Arrange batch images into a grid/contact sheet with configurable columns and spacing. |
| 🎨 Image Noise | `OllamaNodesImageNoise` | Add Gaussian, uniform, or salt & pepper noise to images. |
| 🎨 Image Resize | `OllamaNodesImageResize` | Resize images with multiple fitting modes: stretch, fit (contain/cover), pad, or center crop. |
| 🎨 Image Sharpen | `OllamaNodesImageSharpen` | Sharpen images using unsharp mask with adjustable strength, radius, and threshold. |

## ComfyUI-OMG/Loaders

| Display name | Node ID | Purpose |
|---|---|---|
| Folder File Catalog | `ComfyUIOMGFolderCatalog` | Create a lightweight local manifest for image, video, audio, or text/data files. No media is decoded and the path list can feed the Image List Loader. |
| Image Folder Batch Loader | `ComfyUIOMGImageFolderLoader` | Load a bounded, naturally sorted image batch from a ComfyUI input folder with alpha masks, resizing, filtering, stride, metadata, and memory/pixel safety limits. |
| Image From Folder | `ComfyUIOMGImageFolderSelect` | Load one clamped, 1-based image from a naturally sorted ComfyUI input folder without decoding the rest of the folder. |
| Multiple Image Paths Loader | `ComfyUIOMGImageListLoader` | Load multiple ordered images from newline-separated paths or a JSON string array, normalizing them into one bounded ComfyUI image/mask batch. |
| Text & Data File Loader | `ComfyUIOMGDataFileLoader` | Load bounded TXT, Markdown, prompt, JSON/JSONL, CSV/TSV, YAML, XML/HTML, subtitle, log, INI, or CFG files; JSON and tabular formats receive optional local normalization. |

## ComfyUI-OMG/Mask

| Display name | Node ID | Purpose |
|---|---|---|
| 🎭 Mask Blur | `OllamaNodesMaskBlur` | Blur/feather mask edges with Gaussian blur. |
| 🎭 Mask Combine | `OllamaNodesMaskCombine` | Combine two masks: add, subtract, multiply, intersect, union, XOR, or difference. |
| 🎭 Mask Create | `OllamaNodesMaskCreate` | Create a mask from geometric shapes (rectangle, ellipse, triangle, diamond) with feathering. |
| 🎭 Mask Expand | `OllamaNodesMaskExpand` | Expand (positive) or contract (negative) mask boundaries by pixel amount. |
| 🎭 Mask Gradient | `OllamaNodesMaskGradient` | Create linear or radial gradient masks. |
| 🎭 Mask Invert | `OllamaNodesMaskInvert` | Invert a mask (swap black and white). |
| 🎭 Mask Preview | `OllamaNodesMaskPreview` | Preview a mask, optionally overlaid on an image with configurable color and opacity. |
| 🎭 Mask Threshold | `OllamaNodesMaskThreshold` | Apply a threshold to a mask, creating a binary (black/white) result. |

## ComfyUI-OMG/Math

| Display name | Node ID | Purpose |
|---|---|---|
| 🔢 Compare | `OllamaNodesCompare` | Compare two values with configurable tolerance for float equality. |
| 🔢 Counter | `OllamaNodesCounter` | Auto-incrementing counter. Increments each execution, can reset. |
| 🔢 Logic Gate | `OllamaNodesLogicGate` | Boolean logic gates: AND, OR, NOT, XOR, NAND, NOR, XNOR. |
| 🔢 Math Advanced | `OllamaNodesMathAdvanced` | Advanced math: trigonometry, logarithms, rounding, and more. |
| 🔢 Math Expression | `OllamaNodesMathExpression` | Evaluate math expressions with variables x, y, z. Supports sin, cos, sqrt, pi, lerp, clamp, etc. |
| 🔢 Math Map Range | `OllamaNodesMathMap` | Map/remap a value from one range to another, with optional clamping. |
| 🔢 Math Operation | `OllamaNodesMathOp` | Basic arithmetic: add, subtract, multiply, divide, modulo, power, min, max. |
| 🔢 Random Number | `OllamaNodesMathRandom` | Generate random numbers (float or int) within a specified range with seed control. |

## ComfyUI-OMG/Prompt

| Display name | Node ID | Purpose |
|---|---|---|
| Batch Prompt Variations | `OllamaPromptVariations` | Generates N variations of a prompt while maintaining the core concept. |
| Detail Injector | `OllamaDetailInjector` | Inject specific types of details into existing prompts. |
| Negative Prompt Generator | `OllamaNegativePrompt` | Generates optimized negative prompts based on the positive prompt. |
| Ollama Prompt Builder | `OllamaPromptBuilder` | Ollama Prompt Builder |
| Ollama Prompt Enhancer | `OllamaPromptEnhancer` | Enhances simple prompts into detailed, optimized prompts for AI image generation. |
| Ollama Single Prompt Builder | `OllamaSinglePromptBuilder` | Ollama Single Prompt Builder |
| Prompt Combiner | `OllamaPromptCombiner` | Intelligently combine multiple prompts into one cohesive prompt. |
| Prompt Critic | `OllamaPromptCritic` | Analyzes prompts and provides detailed feedback. |
| Prompt Translator | `OllamaPromptTranslator` | Translate prompts between different AI model formats. |
| Wildcard Generator | `OllamaWildcardGenerator` | Generate prompts with {option1\|option2\|option3} wildcard syntax. |

## ComfyUI-OMG/RAG

| Display name | Node ID | Purpose |
|---|---|---|
| RAG Documents from Text | `OllamaDocumentsFromText` | Create a typed OMG Documents collection from text without filesystem access. |
| RAG Index Build | `OllamaRAGIndexBuild` | Build a local semantic index with Ollama embeddings, deterministic chunks, normalized vectors, and a bounded process-memory embedding cache. |
| RAG Index Load | `OllamaRAGIndexLoad` | Load a validated local RAG index from ComfyUI input or output. |
| RAG Index Save | `OllamaRAGIndexSave` | Save a validated RAG index, including local text and vectors, under ComfyUI output. |
| RAG Reference Folder Loader | `OllamaReferenceFolderLoader` | Load text, Markdown, JSON, CSV, or YAML references from ComfyUI input with file-count and byte safety limits. |
| RAG Semantic Search | `OllamaRAGSearch` | Retrieve top local reference chunks with cosine similarity and source citations. |

## ComfyUI-OMG/Scene

| Display name | Node ID | Purpose |
|---|---|---|
| Action Choreographer | `OllamaActionChoreographer` | Choreograph dynamic action sequences and poses. |
| Advanced Scene Director | `OllamaAdvancedSceneDirector` | Advanced scene director with comprehensive dropdown options for every aspect. |
| Aspect Ratio Optimizer | `OllamaAspectOptimizer` | Optimizes prompts for specific aspect ratios. |
| Background Generator | `OllamaBackgroundGenerator` | Generate detailed background/environment descriptions. |
| Emotion Director | `OllamaEmotionDirector` | Direct and intensify emotional content in prompts. |
| Environment Transformer | `OllamaEnvironmentTransform` | Transforms scene environment — weather, time, and season. |
| Ollama Scene Director | `OllamaSceneDirector` | Directs scenes with subject, pose, blocking, camera, lighting, and style control. |
| Regional Prompt Generator | `OllamaRegionalPrompts` | Generates region-specific prompts for compositional control. |
| Storyboard Generator | `OllamaStoryboardGenerator` | Creates sequences of prompts for comics/storyboards. |

## ComfyUI-OMG/Scene/Bible

| Display name | Node ID | Purpose |
|---|---|---|
| World Bible Create | `OllamaWorldBibleCreate` | Create a versioned World Bible with geography, culture, technology, architecture, locations, immutable rules, visual style, and environment prompts. |
| World Bible Inspect | `OllamaWorldBibleInspect` | Inspect a typed World Bible through focused string outputs. |
| World Bible to Prompt | `OllamaWorldBibleToPrompt` | Compile a World Bible into General, Flux, SDXL, Qwen, Wan, or LTX prompts. |
| World Bible Update | `OllamaWorldBibleUpdate` | Expand locations or revise a World Bible while protecting canonical world rules. |

## ComfyUI-OMG/Storyboard

| Display name | Node ID | Purpose |
|---|---|---|
| 🎬 Character Sheet | `OllamaNodesCharacterSheet` | Define a character for consistent representation across storyboard panels. Provides: - JSON data for storage - Image generation prompt - Human-readable summary |
| 🎬 Image Prompt Gen | `OllamaNodesImagePromptGenerator` | Generate image generation prompts from scene descriptions. Creates consistent prompts optimized for: - Stable Diffusion / SDXL / Flux - DALL-E / Midjourney style - ComfyUI workflows with IPAdapter |
| 🎬 Scene to Prompt | `OllamaNodesSceneToPrompt` | Convert scene description to optimized image generation prompt. Combines all scene elements into a single coherent prompt for consistent visual style across storyboard panels. |
| 🎬 Shot Definition | `OllamaNodesShotDefinition` | Define a single shot with camera specifications. Shot types define framing: - WS/LS: Wide/Long shot for establishing - MS/MCU: Medium shots for dialogue - CU/ECU: Close-ups for emotion |
| 🎬 Shot List Gen | `OllamaNodesShotListGenerator` | Generate a shot list from scene description. Scene types determine shot patterns: - Dialogue: Shot-reverse-shot, over-shoulder - Action: Wide establishing, dynamic tracking - Establishing: Wide to medium progression - Emotional: Close-ups, tight framing |
| 🎬 Story Scene | `OllamaNodesStoryScene` | Define a storyboard scene with full metadata. This node structures scene data that can be connected to: - OllamaNodesShotListGenerator: To break scene into shots - OllamaNodesImagePromptGenerator: To create image generation prompts - OllamaNodesStoryboardExport: To export the final storyboard |
| 🎬 Storyboard Export | `OllamaNodesStoryboardExport` | Export storyboard to different formats. Formats: - Markdown: Human-readable with formatting - JSON: Structured data for further processing - CSV: Spreadsheet-compatible - Plain text: Simple text output |
| 🎬 Storyboard Page | `OllamaNodesStoryboardPage` | Create a storyboard page combining scene info with panel descriptions. Layout options: - 2x2: 4 panels (standard) - 3x2: 6 panels (detailed) - 4x3: 12 panels (comprehensive) - Strip: Horizontal or vertical scrolling |

## ComfyUI-OMG/Text

| Display name | Node ID | Purpose |
|---|---|---|
| 📝 Prompt Combine | `OllamaNodesPromptCombine` | Combine prompts with optional quality presets for positive and negative. |
| 📝 Prompt Styler | `OllamaNodesPromptStyler` | Apply art style templates to your prompt with configurable prefix and suffix. |
| 📝 Prompt Weight | `OllamaNodesPromptWeight` | Apply weights to prompt tokens using a text weight specification. |
| 📝 Text Case | `OllamaNodesTextCase` | Change text case: uppercase, lowercase, title case, sentence case, etc. |
| 📝 Text Concat | `OllamaNodesTextConcat` | Concatenate up to 4 text strings with a configurable separator. |
| 📝 Text Length | `OllamaNodesTextLength` | Get character count, word count, and line count of a text string. |
| 📝 Text List | `OllamaNodesTextList` | Text list operations: get by index, random pick, count, sort, reverse, unique, shuffle. |
| 📝 Text Regex | `OllamaNodesTextRegex` | Regex operations: find all/first matches, replace, split, or test pattern. |
| 📝 Text Replace | `OllamaNodesTextReplace` | Find and replace text with optional regex and case sensitivity support. |
| 📝 Text Switch | `OllamaNodesTextSwitch` | Output one of two text values based on a boolean condition. |
| 📝 Text Template | `OllamaNodesTextTemplate` | Apply template variables: use {variable_name} in template, define up to 4 variables. |

## ComfyUI-OMG/Utility

| Display name | Node ID | Purpose |
|---|---|---|
| ControlNet Helper | `OllamaControlNetHelper` | Optimize prompts for ControlNet workflows. |
| LoRA Suggester | `OllamaLoraSuggester` | Analyze a prompt and suggest helpful LoRA types. |
| Ollama JSON Extractor | `OllamaJSONExtractor` | Ollama J S O N Extractor |
| Prompt Continuity Checker | `OllamaPromptContinuityChecker` | Compares prompts and fixes identity/style drift. |
| 🛠 Debug Inspector | `OllamaNodesDebug` | Debug inspector — display the value and type of any connected input. |
| 🛠 Image Info | `OllamaNodesImageInfo` | Get image dimensions: width, height, batch size, channels, and formatted info text. |
| 🛠 Load Text | `OllamaNodesLoadText` | Load text from a file. Returns default_text if file not found. |
| 🛠 Note | `OllamaNodesNote` | Rich text note/annotation node. Use for documenting workflow sections. |
| 🛠 Resolution Preset | `OllamaNodesResolution` | Quick resolution presets for SD1.5, SDXL, SD3, Flux, or custom. Supports scaling and dimension swap. |
| 🛠 Save Text | `OllamaNodesSaveText` | Save text content to a file in the ComfyUI output directory. |
| 🛠 Seed Generator | `OllamaNodesSeed` | Seed generator with modes: fixed, random, increment, decrement. |
| 🛠 Timer | `OllamaNodesTimer` | Measure execution time. Start, stop, or lap a named timer. |

## ComfyUI-OMG/VFX/Light

| Display name | Node ID | Purpose |
|---|---|---|
| ✨ Effect Lens Flare | `OllamaNodesEffectLensFlare` | Add lens flare effect to image. |

## ComfyUI-OMG/VFX/Particles

| Display name | Node ID | Purpose |
|---|---|---|
| ✨ Particle Confetti | `OllamaNodesParticleConfetti` | Generate colorful confetti particle effect. |
| ✨ Particle Dust/Bokeh | `OllamaNodesParticleDust` | Generate floating dust/bokeh particle effect for atmosphere. |
| ✨ Particle Fire | `OllamaNodesParticleFire` | Generate fire/flame particle effect with color gradient. |
| ✨ Particle Rain | `OllamaNodesParticleRain` | Generate rain particle effect. Parameters: - density: Number of rain drops - speed: How fast drops fall - wind: Horizontal wind effect - angle: Rain angle from vertical |
| ✨ Particle Snow | `OllamaNodesParticleSnow` | Generate snow particle effect with wobble motion. |
| ✨ Particle Sparks | `OllamaNodesParticleSparks` | Generate sparks/embers particle effect with trails. |

## ComfyUI-OMG/VFX/Screen

| Display name | Node ID | Purpose |
|---|---|---|
| ✨ Effect Chromatic | `OllamaNodesEffectChromatic` | Apply chromatic aberration (color fringing) effect. |
| ✨ Effect Film Noise | `OllamaNodesEffectNoise` | Apply film grain/noise effect. |
| ✨ Effect Glitch | `OllamaNodesEffectGlitch` | Apply glitch/distortion effect to images/video frames. |
| ✨ Effect Scanlines | `OllamaNodesEffectScanlines` | Apply CRT/TV scanline effect. |
| ✨ Effect Speed Lines | `OllamaNodesEffectSpeedLines` | Generate anime-style speed lines for action scenes. |
| ✨ Effect VHS Retro | `OllamaNodesEffectVHS` | Apply VHS/retro tape effect with tracking and color bleed. |

## ComfyUI-OMG/VFX/Shapes

| Display name | Node ID | Purpose |
|---|---|---|
| ✨ Shape Grid | `OllamaNodesShapeGrid` | Generate animated grid pattern (sci-fi/cyberpunk style). |
| ✨ Shape Pulse | `OllamaNodesShapePulse` | Generate pulsing shape animation (circle, ring, square, diamond). |
| ✨ Shape Rings | `OllamaNodesShapeRings` | Generate expanding ring animation (radar/sonar style). |
| ✨ Shape Wave | `OllamaNodesShapeWave` | Generate animated wave pattern. |

## ComfyUI-OMG/Video

| Display name | Node ID | Purpose |
|---|---|---|
| LTX Ingredients Prompt | `LTXIngredientsPrompt` | L T X Ingredients Prompt |
| LTXV 2.3 Video Prompt Generator | `OllamaLTXVVideoPrompt` | _ Base Video Prompt Generator |
| Video Shot Director | `OllamaVideoShotDirector` | Expands a scene/video concept into a shot-by-shot prompt sequence. |
| Wan 2.2 Video Prompt Generator | `OllamaWanVideoPrompt` | _ Base Video Prompt Generator |
| 🎬 Batch Join | `OllamaNodesVideoBatchJoin` | Join multiple frame batches into one sequence. Transitions: - none: Direct concatenation - crossfade: Blend between segments - fade_black: Fade out to black, then in |
| 🎬 Batch Split | `OllamaNodesVideoBatchSplit` | Split frame batch into parts. Modes: - equal_parts: Divide evenly into N parts - by_count: Each part has fixed frame count - by_indices: Split at specified frame indices |
| 🎬 Frame Extract | `OllamaNodesVideoFrameExtract` | Extract specific frames from video. Modes: - by_index: Specify frame numbers (e.g., "0, 10, 20" or "0-30") - by_time: Specify timestamps in seconds - uniform_sample: Extract N evenly-spaced frames - first_last: Get only first and last frames |
| 🎬 Video Fade | `OllamaNodesVideoFade` | Apply fade in/out effects. Fades to/from specified color (default black). |
| 🎬 Video Info | `OllamaNodesVideoInfo` | Get video metadata without loading frames into memory. |
| 🎬 Video Loader | `OllamaNodesVideoLoader` | Load video with performance optimization. Parameters: - video_path: Path to video file (mp4, avi, mov, webm, mkv) - start_frame: First frame to load - max_frames: Maximum frames to load (memory safety) - frame_skip: Skip N frames between each loaded frame (e.g., 1 = every other frame) - scale_factor: Resize frames (0.5 = half size, faster loading) - force_fps: Override video FPS (0 = use original) |
| 🎬 Video Loop | `OllamaNodesVideoLoop` | Loop/repeat video frames. Modes: - repeat: A, A, A... (simple repeat) - ping_pong: A, reverse(A), A, reverse(A)... - reverse_loop: A, reverse(A)... |
| 🎬 Video Overlay | `OllamaNodesVideoOverlay` | Overlay video/image on background. - x_position/y_position: -1 to 1 (0 = center) - scale: Size of overlay relative to background - opacity: Transparency of overlay - mask: Optional mask for overlay (white = visible) |
| 🎬 Video Reverse | `OllamaNodesVideoReverse` | Reverse the order of frames in a batch. |
| 🎬 Video Saver | `OllamaNodesVideoSaver` | Save image batch as video file. Codecs: - mp4v: Standard MP4 (most compatible) - avc1: H.264 (better compression) - XVID: AVI format - MJPG: Motion JPEG (large files, fast encoding) |
| 🎬 Video Speed | `OllamaNodesVideoSpeed` | Change video speed. - speed_factor > 1: Faster (fewer frames) - speed_factor < 1: Slower (more frames via interpolation) Interpolation (for slowdown): - nearest: Duplicate frames - linear: Blend between frames |
| 🎬 Video Transition | `OllamaNodesVideoTransition` | Create transitions between clips. Types: crossfade, wipe (4 directions), fade to color, dissolve, zoom |
| 🎬 Video Trim | `OllamaNodesVideoTrim` | Trim video to specific range. - by_frames: Use frame indices (end_frame=-1 means last frame) - by_percentage: Use percentage of total length |

## ComfyUI-OMG/Video/Shot List

| Display name | Node ID | Purpose |
|---|---|---|
| Shot List Create | `OllamaShotListCreate` | Create a validated shot list with duration, first/video/last-frame prompts, motion, camera, lighting, transitions, and linked continuity assets. |
| Shot List Get Shot | `OllamaShotListGetShot` | Extract one indexed shot with frame prompts, motion, camera, lighting, and locks. |
| Shot List Inspect | `OllamaShotListInspect` | Inspect a typed Shot List through sequence metadata and compact timeline JSON. |
| Shot List Update | `OllamaShotListUpdate` | Revise or append shots while protecting shot-list ID, target model, aspect ratio, Character Bible links, and World Bible link. |

