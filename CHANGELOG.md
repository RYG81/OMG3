# Changelog

All notable changes to ComfyUI-OMG will be documented here. The project follows Semantic Versioning.

## [Unreleased]

### Added

- Phase 0 repository, packaging, security, and test foundation.
- Comfy Registry metadata for publisher `RYG81`.
- AGPL-3.0 license.
- GitHub Actions for linting, contract tests, packaging, and manual Registry publishing.
- Offline node-contract, client, path, network, and expression tests.
- ComfyUI progress/cancellation integration for long folder, web, model-pull, and video operations.
- Ollama Structured Generate node with Draft 2020-12 JSON Schema validation and one repair attempt.
- Ollama Model Inspector with live vision, tools, thinking, embedding, context, and loaded-state outputs.
- Shared structured-output parsing, validation, and model-capability utilities.
- Schema-driven task engine with request fingerprints, provenance, validation, repair, and cache policies.
- Multimodal task fingerprints/provenance use image SHA-256 hashes without storing image data in provenance.
- Private bounded process-memory task cache and Cache Control node; no prompt data is written to disk.
- Typed `OMG_CREATIVE_BRIEF` compiler and deterministic target-family adapter node.
- Versioned `OMG_CHARACTER_BIBLE` with Create, protected Update, deterministic To Prompt, and Inspect nodes.
- Deterministic Character Bible IDs and adapters for General, Flux, SDXL, Qwen Image, Wan Video, and LTX Video.
- Protected-update validation that rejects wardrobe-scope violations and identity drift before exposing an updated bible.
- Versioned `OMG_WORLD_BIBLE` with Create, protected Update, deterministic To Prompt, and Inspect nodes.
- Deterministic World Bible IDs, structured key locations, and target adapters for General, Flux, SDXL, Qwen Image, Wan Video, and LTX Video.
- Protected-update validation that rejects location-expansion scope violations and global world-rule drift.
- Versioned `OMG_SHOT_LIST` with Create, protected Update, Get Shot, and Inspect nodes.
- Shot Lists can link up to three Character Bibles, one World Bible, and one Creative Brief using exact typed IDs.
- Shot-domain validation for unique IDs, sequential ordering, requested count, duration totals, linked assets, and append-only preservation.
- Typed `OMG_DOCUMENTS` and `OMG_RAG_INDEX` assets with manual-text and bounded folder loaders.
- Local Ollama embedding index builder with deterministic chunk IDs, overlap, normalized vectors, batching, and bounded embedding cache.
- Semantic RAG search with cosine scores, source citations, bounded context assembly, and exact embedding-model checks.
- Safe RAG index save/load under ComfyUI directories with atomic writes, vector validation, and a 256 MB load cap.
- Typed `OMG_EVALUATION` assets with Prompt Evaluator, Image Evaluator, deterministic Compare, and Inspect nodes.
- Seven-criterion weighted evaluation with locally recomputed scores/verdicts and critical-issue failure enforcement.
- Bounded typed-asset context support for Creative Brief, Character Bible, World Bible, and Shot List evaluation.
- Local ComfyUI-OMG dashboard with Ollama model/running-state table, latency, package status, and memory-cache controls.
- Loopback-only dashboard model probes by default with `COMFY_OMG_ALLOWED_OLLAMA_HOSTS` remote allowlist.
- Ollama Model Loader refresh button, context-menu action, and frontend-only detected-model selector.
- Four browsable starter workflows with matching 960×540 local thumbnails.
- Four native ComfyUI subgraph blueprints for prompt evaluation, Character Bible prompts, local RAG search, and Shot List prompt extraction.
- Core Beta / Media / Labs stability classification and dashboard tier counts without changing persistent IDs/categories.
- Thirty embedded Markdown help pages for core, typed assets, RAG, evaluation, and image-quality nodes.
- Character Bible, World Bible, linked Shot List, and Prompt Evaluation starter workflows with thumbnails.
- On-demand dashboard and Model Loader capability inspection for text, vision, tools, thinking, embeddings, context, and loaded state.
- Frontend accessibility improvements: semantic table metadata, ARIA labels/live regions, focus-visible rings, high contrast, and reduced motion.
- Domain-level output-quality validation layered after JSON Schema validation.
- Automatic low-temperature repair for schema-valid but unusably short, placeholder-filled, or format-invalid outputs.
- Conservative local prompt sanitization for outer whitespace, full-value fences, known output labels, and exact duplicate sentences; formatting-only defects no longer require a second Ollama call.
- Content-free sanitization provenance preserves the original raw response and records only changed paths/actions.
- Versioned quality and sanitization rules included in task fingerprints and provenance, plus task-specific wildcard, LTX, palette, Bible, variation, and storyboard checks.
- Empty-response and requested-JSON validation for free-form Text Generate, Chat, and Vision nodes.
- Embedding response-count, finite-value, zero-vector, and dimension-consistency validation.
- Source-intent preservation checks for enhanced, translated, edited, scene, creature, outfit, style-transfer, and video prompts.
- Target-family checks for Flux/Qwen natural language, SDXL parameter compatibility, and Wan/LTX motion specificity.
- Cross-field consistency checks for Creative Brief constraints, image-analysis reconstruction evidence, and Character/World Bible master prompts.
- Shot List pre-exposure quality checks for frame/video prompt depth, target motion, typed links, exact model/aspect/FPS/duration, ordering, IDs, and append semantics.
- Golden-output regression fixtures for Prompt Enhancer, Image Analyzer, Character Bible, World Bible, Shot List, and Evaluation.
- Deterministic image-analysis field confidence, visibility, hedging, contradiction, pose/support, person-count, and reconstruction-format diagnostics.
- Image Analysis Quality Inspector for direct Image Analyzer JSON and aggregate Sequence/Photoset/Web Image analysis payloads.
- Embedded `_quality` diagnostics in structured image-analysis outputs without changing legacy sockets.
- Bounded deterministic multi-image consistency reports for identity, clothing, environment, camera, lighting, palette, and style drift.
- Multi-Image Consistency Inspector with stable/variable/conflicting classifications, field drift scores, and explicit claim conflicts.
- Embedded `_consistency` reports and compact conflict guardrails in Sequence/Photoset synthesis without image decoding, embeddings, network work, or extra Ollama calls.
- Image From Folder loader for clamped, 1-based low-memory selection that decodes only the chosen naturally sorted item.
- Image Folder Batch Loader with natural sorting, recursive/filter/start/stride controls, alpha masks, normalization modes, metadata, and hard source/output memory limits.
- Multiple Image Paths Loader for ordered newline or JSON-array paths, compatible with Folder File Catalog output.
- Metadata-only Folder File Catalog covering supported image, video, audio, and text/data families without media decoding.
- Text & Data File Loader for text, Markdown, prompts, YAML, XML/HTML, subtitles, logs, INI/CFG, plus bounded JSON/JSONL/CSV/TSV normalization.
- Global custom system-prompt textbox on the main Ollama Model Loader plus node-specific textboxes on all 65 generative `OLLAMA_MODEL` consumers.
- Deterministic built-in → global → node system-prompt composition with 16,000-character custom-layer limits and content-free hashes.

### Changed

- Rebranded the user-facing node pack and every ComfyUI category root from `Ollama-Magic-Nodes` / Comfy-OMG to **ComfyUI-OMG** while preserving every serialized node ID and the existing `comfy-omg` package/repository slug.
- Kept the original main `OllamaModelLoader` registered under `ComfyUI-OMG/Core`; the new local file loaders do not replace or alias it.
- Consolidated two Ollama clients into one implementation and shared session/error system.
- Consolidated the model loader, added `num_ctx`, remote-server validation, and offline-safe schema discovery.
- Fixed Prompt Builder fingerprints by restoring ComfyUI's complete default input fingerprinting.
- Migrated the default Prompt Enhancer preset to schema validation/cache/provenance internally while preserving its output contract and free-form `2026U` behavior.
- Migrated Scene Director, Character Sheet, Image Analyzer, Video Shot Director, and Prompt Critic to the shared task engine without changing their original output sockets.
- Migrated Prompt Variations, Negative Prompt, Advanced Scene Director, all five Character Consistency tools, and Photoset Folder Analyzer without changing their original output sockets.
- Migrated Image Sequence Analyzer, Auto Tagger, Style Identifier, Regional Prompts, and Storyboard Generator without changing their original output sockets.
- Added dynamic Prompt Variations and Storyboard schemas so only the requested variation/panel count is required while all legacy outputs remain available.
- Corrected 1x3 and 3x1 Regional Prompt results by mapping left/right/top/middle/bottom into the corresponding legacy 3x3-compatible sockets.
- Migrated Image Comparator, Color Palette, Inpaint Prompt, Style Transfer, and Environment Transform without changing their documented output order or types.
- Made Inpaint Prompt mask-aware by sending the optional mask as a lossless second image with explicit white-fill/black-preserve semantics.
- Migrated Image Merger, Image Edit, Image to Story, Pose Descriptor, and Outfit Generator without changing their documented output meanings or order.
- Preserved source/base ordering for Image Merger and made Outfit Generator require vision only when a reference image is connected.
- Migrated Subject Builder, Creature Creator, Hand Pose Helper, Lighting Designer, and Texture & Material Designer without changing their documented output meanings or order.
- Kept prompt-requested supplemental fields such as personality expression, environmental context, difficulty rating, time-of-day match, and lighting tips internal rather than shifting legacy sockets.
- Migrated Action Choreographer, Background Generator, Emotion Director, Detail Injector, and ControlNet Helper without changing their documented output meanings or order.
- Kept supplemental timing, mood, composition, placement, and ControlNet-strength fields internal rather than shifting legacy sockets.
- Migrated Prompt Combiner, Prompt Translator, Wildcard Generator, Aspect Optimizer, and LoRA Suggester while preserving shortcuts, socket order, and list-to-string normalization.
- Migrated LTX Ingredients plus LTXV/Wan video prompt generators while preserving model-specific fallbacks and output contracts.
- Migrated Web Image Tool analysis to the multimodal task engine without altering search, download, metadata, or frame-analysis outputs.
- Completed migration of all repeated creative `generate → JSON extraction → socket mapping` wrappers; specialized free-form core parsers remain intentional.
- Added lossless mask-to-PNG conversion and clamped image tensor conversion to prevent out-of-range uint8 wrapping.
- Added explicit vision-capability rejection when Ollama declares that a selected vision node model is text-only.
- Added descriptions for all registered nodes.
- Marked every custom `OMG_…` datatype socket with `forceInput=True` for modern ComfyUI validation.
- Reduced mandatory dependencies to the minimal runtime set.
- Moved video and agent-assisted web search dependencies to optional extras.

### Security

- Sandboxed text, image-loader, catalog, and video paths under ComfyUI directories by default.
- Added explicit opt-in for loader/photoset paths outside ComfyUI input.
- Added symlink containment plus scan, file-count, byte, source-megapixel, batch-pixel, and parsed-row limits to new loaders.
- Enabled HTTPS certificate verification by default and added redirect, SSRF, size, and pixel limits for web images.
- Replaced expression `eval` with an AST-whitelist evaluator.
- Removed the subprocess-based dependency installer.

## [0.1.0] - TBD

Initial public beta. This version will be tagged only after Phase 0 acceptance checks pass.
