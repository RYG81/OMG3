# Phase 1 Status — Reliable Core and Typed Creative Workflows

Phase 1 builds reusable infrastructure under the existing node pack while retaining all original 170 IDs.

## Completed

- [x] Canonical Ollama client, error hierarchy, URL validation, schema formats, and strict streaming.
- [x] Offline-safe main `OllamaModelLoader` retained under `ComfyUI-OMG/Core` with normalized capabilities in every `OLLAMA_MODEL` profile.
- [x] Added a global custom-system textbox to Model Loader and a node-specific textbox to all 65 generative model consumers.
- [x] Compose built-in, global, and node-specific instructions additively with bounded lengths and content-free fingerprints.
- [x] Model Inspector for vision, tools, thinking, embeddings, context, quantization, and loaded state.
- [x] Draft 2020-12 structured generation with local validation and one repair attempt.
- [x] Shared schema-driven task engine.
- [x] SHA-256 request fingerprints and content-free provenance records.
- [x] Cache policies: `use`, `refresh`, and `bypass`.
- [x] Bounded 128-entry / 64 MB process-memory LRU cache.
- [x] Cache Control node with stats and clear actions.
- [x] No disk persistence of cached prompts or responses.
- [x] Typed `OMG_CREATIVE_BRIEF` envelope.
- [x] Creative Brief Compiler with a strict schema.
- [x] Deterministic Creative Brief adapters for General, Flux, SDXL, Qwen Image, Wan Video, and LTX Video.
- [x] Source package includes `tasks/` infrastructure.
- [x] Prompt Enhancer migrated to the task engine for its default structured preset while preserving four legacy outputs and free-form `2026U` behavior.
- [x] Scene Director migrated while preserving all six outputs and its complete control surface.
- [x] Character Sheet migrated while preserving all twelve outputs.
- [x] Image Analyzer migrated with multimodal image hashes in cache fingerprints/provenance while preserving all twelve outputs and scene-graph generation.
- [x] Video Shot Director migrated while preserving all six outputs.
- [x] Prompt Critic migrated while preserving all eight outputs.
- [x] Prompt Variations migrated with a dynamic schema matching the requested 2–10 variations while preserving eleven outputs.
- [x] Negative Prompt migrated while preserving four outputs and accepting the existing internal reasoning field.
- [x] Advanced Scene Director migrated while preserving its full control surface and ten outputs.
- [x] Five Character Consistency tools migrated through one shared runner while preserving every existing output.
- [x] Photoset Folder Analyzer migrated across per-image vision analysis, incremental continuity updates, and final synthesis.
- [x] Image Sequence Analyzer migrated across per-frame vision analysis and final video-prompt synthesis.
- [x] Auto Tagger and Style Identifier migrated with image-aware cache fingerprints.
- [x] Regional Prompts migrated with dynamic grid schemas and corrected 1x3/3x1 mapping into legacy sockets.
- [x] Storyboard Generator migrated with a dynamic panel-count schema and normalized JSON-array output.
- [x] Image Comparator migrated with ordered Image A/Image B inputs and seven unchanged semantic outputs.
- [x] Color Palette migrated while keeping all seven palette outputs as strings in their documented order.
- [x] Inpaint Prompt migrated with actual mask awareness: the optional mask is now sent as a lossless second PNG image while preserving three outputs.
- [x] Style Transfer and Environment Transform migrated while preserving their original five-output contracts.
- [x] Image Merger migrated with ordered source/base image semantics and five unchanged outputs.
- [x] Image Edit migrated while preserving its three semantic outputs plus derived full-result JSON.
- [x] Image to Story migrated while preserving all nine narrative outputs.
- [x] Pose Descriptor migrated while preserving its six pose-specific outputs and allowing internal camera/difficulty fields.
- [x] Outfit Generator migrated with text-only execution when no reference is supplied and conditional vision execution when one is supplied.
- [x] Subject Builder migrated while preserving its eight character-description outputs and internal personality field.
- [x] Creature Creator migrated while preserving ten outputs and internal environmental context.
- [x] Hand Pose Helper migrated while preserving eleven hand-specific outputs and internal difficulty rating.
- [x] Lighting Designer migrated while preserving eight production-lighting outputs and internal time-of-day match.
- [x] Texture & Material Designer migrated while preserving eleven material outputs and internal lighting advice.
- [x] Action Choreographer migrated while preserving eleven action-specific outputs and internal timing notes.
- [x] Background Generator migrated while preserving nine environment outputs and internal mood elements.
- [x] Emotion Director migrated while preserving nine emotion outputs and internal composition advice.
- [x] Detail Injector migrated while preserving five prompt-detail outputs and internal placement notes.
- [x] ControlNet Helper migrated while preserving nine workflow outputs and internal strength notes.
- [x] Prompt Combiner and Prompt Translator migrated while preserving their no-op/single-input shortcuts.
- [x] Wildcard Generator migrated with a strict three-sample array mapped to the original sample sockets.
- [x] Aspect Optimizer migrated while preserving four outputs and internal spatial-planning fields.
- [x] LoRA Suggester migrated with list-or-string normalization for LoRA category sockets.
- [x] LTX Ingredients retained its deterministic model-specific fallback while moving valid responses to the task engine.
- [x] LTXV and Wan video prompt generators migrated through their shared base class.
- [x] Web Image Tool analysis migrated to the multimodal task engine without altering search/download outputs.
- [x] All repeated creative `generate → extract_json_block → socket mapping` wrappers have now been migrated; only specialized core parsers retain direct parsing intentionally.
- [x] Added versioned `OMG_CHARACTER_BIBLE` envelopes with deterministic character IDs.
- [x] Added Character Bible Create, protected Update, deterministic To Prompt, and Inspect nodes.
- [x] Wardrobe-only updates reject changes outside wardrobe/flexible/master/negative fields.
- [x] Identity-preserving revisions reject changes to core identity, species, face, eyes, body proportions, distinguishing features, or immutable anchors.
- [x] Character Bible adapters support General, Flux, SDXL, Qwen Image, Wan Video, and LTX Video without another LLM call.
- [x] Added versioned `OMG_WORLD_BIBLE` envelopes with deterministic world IDs and structured key locations.
- [x] Added World Bible Create, protected Update, deterministic To Prompt, and Inspect nodes.
- [x] Location-expansion updates reject changes outside locations/flexible/master/negative fields.
- [x] Rule-preserving revisions reject drift in premise, genre, era/technology, geography, architecture, culture, visual style, immutable rules, and forbidden drift.
- [x] World Bible adapters support location-specific General, Flux, SDXL, Qwen Image, Wan Video, and LTX Video prompts without another LLM call.
- [x] Added versioned `OMG_SHOT_LIST` envelopes with deterministic IDs, exact linked Character/World Bible IDs, and structured shots.
- [x] Added Shot List Create, protected Update, Get Shot, and Inspect nodes.
- [x] Shot List domain validation enforces unique IDs, sequential order, exact requested count, and duration totals.
- [x] Append mode preserves every existing shot exactly; ID-preserving revisions reject shot-ID drift.
- [x] Each shot exposes first-frame, continuous-video, last-frame, negative, camera, lighting, and continuity-lock outputs.
- [x] Added typed `OMG_DOCUMENTS` collections from manual text or bounded local reference folders.
- [x] Added typed `OMG_RAG_INDEX` construction with deterministic chunks, normalized Ollama embeddings, and bounded embedding cache.
- [x] Added semantic search with cosine scores, bounded context assembly, source citations, and model/dimension checks.
- [x] Added safe RAG index save/load under ComfyUI input/output with validation and a 256 MB load limit.
- [x] Extended Cache Control to inspect and clear both task-response and embedding caches.
- [x] Added typed `OMG_EVALUATION` assets with prompt and image evaluator nodes.
- [x] Evaluation schemas cover seven weighted criteria, strengths, severity-ranked issues, missing requirements, recommendations, improved prompts, and confidence.
- [x] Overall scores and verdicts are recomputed locally; critical issues force failure regardless of the model-reported verdict.
- [x] Added deterministic Evaluation Compare and Inspect nodes using normalized scores.
- [x] Evaluation can consume Creative Brief, Character Bible, World Bible, and Shot List requirements through bounded compact context.
- [x] Added domain-level quality validation after JSON Schema validation and before outputs are accepted.
- [x] Safe local sanitization removes only full-value fences, known output labels, outer whitespace, and exact duplicate sentences while preserving raw responses.
- [x] Formatting-only defects can pass without a second Ollama call; substantive quality failures still trigger one low-temperature repair with actionable errors.
- [x] Quality/sanitization rule versions participate in request fingerprints and provenance to prevent stale cache acceptance.
- [x] Added prompt length/format/placeholder checks plus wildcard, LTX, palette, Character Bible, and World Bible rules.
- [x] Added duplicate/length checks for Prompt Variations and Storyboard panels.
- [x] Hardened free-form Text, Chat, and Vision nodes against empty output and invalid requested JSON.
- [x] Hardened Embeddings against missing vectors, dimension mismatch, non-finite values, zero vectors, and model-count mismatch.
- [x] Added source-intent overlap checks for enhancement, translation, editing, scene, creature, outfit, style-transfer, and video prompt tasks.
- [x] Added target-model rules for Flux/Qwen natural language, SDXL syntax, and Wan/LTX motion specificity.
- [x] Added cross-field consistency checks for Creative Brief negatives/targets/aspect, image reconstruction evidence, and Bible master-prompt anchor coverage.
- [x] Added Shot List quality repair for prompt depth, target motion, asset links, exact settings, IDs, order, durations, and append semantics.
- [x] Added six golden-output fixtures covering Prompt Enhancer, Image Analyzer, Character Bible, World Bible, Shot List, and Evaluation.
- [x] Added deterministic image-analysis evidence scores for every structured field and scene elements.
- [x] Added hallucination/cross-field checks for hedging, visibility/detail conflicts, prompt labels/Markdown, clothing-state contradictions, sitting/support mismatch, and person-count drift.
- [x] Embedded quality diagnostics in Image Analyzer, Image Sequence, Photoset, and Web Image analysis JSON.
- [x] Added Image Analysis Quality Inspector for direct and aggregate analysis payloads.
- [x] Added bounded multi-image identity, clothing, environment, camera, lighting, palette, and style drift diagnostics.
- [x] Embedded `_consistency` reports in Image Sequence and Photoset JSON without changing legacy sockets.
- [x] Routed compact repeated-evidence/conflict guardrails into existing synthesis calls without adding model calls.
- [x] Added a local-only Multi-Image Consistency Inspector with capped input, text, token, conflict-detail, and guidance work.
- [x] Added low-memory single-image folder selection plus folder and explicit-path image batches with natural sorting, alpha masks, normalization, and hard batch-memory bounds.
- [x] Added a metadata-only multi-family Folder File Catalog and bounded Text & Data File Loader without new dependencies.
- [x] Kept loader paths sandboxed by default with explicit external opt-in, symlink containment, and scan/file/byte/pixel/row limits.

## Current automated result

- Registered nodes: 204
- Original IDs preserved: 170/170
- Display mappings: 204
- Tests: 137 passing
- Ruff: passing
- High-severity Bandit findings: 0
- Mandatory dependency vulnerabilities: 0
- Source distribution build: passing

## Structured-wrapper migration status

The repeated creative-node wrapper pattern has been migrated. `OllamaJSONExtractor` and Prompt Builder retain specialized parsing intentionally because their public purpose includes arbitrary/free-form parsing rather than a fixed task schema.

The planned typed workflow assets are also complete:

1. `OMG_CHARACTER_BIBLE`
2. `OMG_WORLD_BIBLE`
3. `OMG_SHOT_LIST`
4. Local `OMG_DOCUMENTS` / `OMG_RAG_INDEX`
5. `OMG_EVALUATION`

## Phase 1 conclusion

The reliable core, schema-task engine, typed creative assets, local retrieval, and evaluation layer are implemented. Further work should move to Phase 2 UX rather than add more backend wrappers.

## Phase 2 priorities

1. ComfyUI-OMG dashboard and live model refresh
2. Example workflows with thumbnails
3. Reusable subgraphs for briefs, continuity, shots, RAG, and evaluation
4. Core / Media / Labs stability presentation
5. Embedded node documentation and onboarding
