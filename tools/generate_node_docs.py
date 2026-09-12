"""Generate embedded Markdown help pages for major ComfyUI-OMG nodes."""
from __future__ import annotations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'web'/'js'/'docs'
OUT.mkdir(parents=True,exist_ok=True)

DOCS={
'OllamaModelLoader':('Ollama Model Loader','The main Ollama entry node. It validates an installed model and emits the shared `OLLAMA_MODEL` profile without changing the legacy `OllamaModelLoader` ID.','Find it under `ComfyUI-OMG/Core`, enter the exact model tag from `ollama list`, optionally add global custom system instructions, then connect `ollama_model` to AI nodes.','Schema discovery is offline-safe. Built-in system prompts remain active; global instructions are appended and can be followed by a per-node custom system prompt. Remote servers receive prompts and images.'),
'OllamaModelInspector':('Ollama Model Inspector','Reads live model metadata and conservative capability flags.','Connect an `OLLAMA_MODEL` profile to inspect vision, tool, thinking, embedding, context, quantization, and loaded state.','Capability inference is conservative; declared server capabilities take precedence.'),
'OllamaStructuredGenerate':('Ollama Structured Generate','Generates JSON constrained by a Draft 2020-12 JSON Schema.','Supply a prompt and schema. Use `validated_json` only when `is_valid` is true; inspect the validation report and provenance when debugging.','One optional repair attempt is supported. `use`, `refresh`, and `bypass` control the private memory cache.'),
'OllamaTaskCacheControl':('Ollama Task Cache Control','Inspects or clears task-response and embedding memory caches.','Use `stats` for counts/size or `clear` to evict all process-memory entries.','Caches are never persisted by this node. Saved RAG indexes are separate explicit files.'),
'OllamaCreativeBriefCompiler':('Creative Brief Compiler','Turns a concept into a validated, typed `OMG_CREATIVE_BRIEF`.','Connect Model Loader, describe the concept/constraints, select target family, then pass `creative_brief` to adapters or evaluators.','The output includes provenance and validity. Invalid data is not exposed as a usable typed asset.'),
'OllamaCreativeBriefToPrompt':('Creative Brief to Prompt','Deterministically adapts a Creative Brief without another LLM call.','Connect a valid brief and choose General, Flux, SDXL, Qwen Image, Wan Video, or LTX Video.','This node does not mutate the source brief.'),
'OllamaCharacterBibleCreate':('Character Bible Create','Creates a canonical `OMG_CHARACTER_BIBLE` with identity, wardrobe, immutable anchors, master prompt, and provenance.','Use one stable concept/name and production-lock strictness. Reuse the resulting Bible across scenes and Shot Lists.','The character ID is deterministic from name + concept.'),
'OllamaCharacterBibleUpdate':('Character Bible Update','Applies wardrobe-only or controlled revisions with identity safeguards.','Use wardrobe mode for costume changes. Use controlled revision for broader changes and keep identity preservation enabled unless intentionally redesigning.','Rejected changes return the original valid Bible and an explanatory report.'),
'OllamaCharacterBibleToPrompt':('Character Bible to Prompt','Compiles scene, pose, expression, and optional outfit override into target-family prompts.','Use scene-specific overrides without editing the canonical Bible.','Video adapters add frame-to-frame identity locks.'),
'OllamaCharacterBibleInspect':('Character Bible Inspect','Exposes the major Bible sections as strings/JSON.','Use it to view or route identity, face, body, wardrobe, anchors, negatives, and master prompt.','Inspection does not call Ollama.'),
'OllamaWorldBibleCreate':('World Bible Create','Creates a canonical `OMG_WORLD_BIBLE` with geography, culture, architecture, locations, visual rules, and environment prompt.','Define the world concept and desired location count, then reuse the typed Bible across scene and shot planning.','The world ID is deterministic from title + concept.'),
'OllamaWorldBibleUpdate':('World Bible Update','Expands locations or applies controlled revisions with protected global rules.','Use location expansion to add locations without changing global identity.','Rejected drift returns the original valid Bible.'),
'OllamaWorldBibleToPrompt':('World Bible to Prompt','Compiles global rules and an optional named location into a target-family environment prompt.','Enter a key-location name exactly to include its specific details.','Video targets add geography, architecture, palette, and lighting continuity.'),
'OllamaWorldBibleInspect':('World Bible Inspect','Exposes title, premise, technology, geography, architecture, locations, rules, and master prompt.','Use it for debugging or routing Bible sections.','Inspection does not call Ollama.'),
'OllamaShotListCreate':('Shot List Create','Creates a typed `OMG_SHOT_LIST` with exact duration, linked Bibles, camera/motion plans, and first/video/last-frame prompts.','Optionally connect Creative Brief, up to three Character Bibles, and a World Bible.','Domain validation requires unique IDs, sequential order, exact count, and duration sum.'),
'OllamaShotListUpdate':('Shot List Update','Revises, appends, or resequences a Shot List while protecting asset links and core sequence settings.','Append mode preserves every existing shot exactly. Enable shot-ID preservation for normal revisions.','Rejected changes return the original valid Shot List.'),
'OllamaShotListGetShot':('Shot List Get Shot','Extracts one indexed shot for downstream image/video generation.','Use `shot_index` to retrieve frame prompts, video prompt, negative, camera, lighting, and locks.','The index is clamped to the available range.'),
'OllamaShotListInspect':('Shot List Inspect','Exposes sequence metadata and compact timeline JSON.','Use it to audit total duration, links, target model, and shot ordering.','Inspection does not call Ollama.'),
'OllamaDocumentsFromText':('RAG Documents from Text','Creates a typed `OMG_DOCUMENTS` collection from manual text.','Provide a stable source name for meaningful citations.','No filesystem or Ollama call occurs.'),
'OllamaReferenceFolderLoader':('RAG Reference Folder Loader','Loads bounded text/Markdown/JSON/CSV/YAML references from ComfyUI input.','Place files under input/references or explicitly allow an external folder.','Per-file, total-byte, count, UTF-8, symlink, and path-sandbox protections apply.'),
'OllamaRAGIndexBuild':('RAG Index Build','Chunks documents and embeds them with an Ollama embedding model.','Use an embedding model such as `nomic-embed-text`; tune chunk size/overlap for the source material.','Vectors are normalized and dimension-checked. The summary omits vectors.'),
'OllamaRAGSearch':('RAG Semantic Search','Retrieves top local chunks with cosine similarity and citations.','Use the same embedding model used to build the index. Connect `context` to a prompt/agent node.','Query/index model mismatch is rejected.'),
'OllamaRAGIndexSave':('RAG Index Save','Atomically saves an index under ComfyUI output.','Use a `.json` filename and enable overwrite only deliberately.','Saved files contain source text and embedding vectors.'),
'OllamaRAGIndexLoad':('RAG Index Load','Loads and validates an index from ComfyUI input or output.','Select the correct location and relative filename.','A 256 MB cap and vector validation apply.'),
'OllamaPromptEvaluator':('Prompt Evaluator','Evaluates a prompt against intent, model family, and optional typed assets.','Use normalized score/verdict and severity-ranked issues; feed `improved_prompt` into the next iteration.','The final weighted score and verdict are recomputed locally.'),
'OllamaImageEvaluator':('Image Evaluator','Evaluates a candidate image against prompt, optional reference, and typed assets.','Image 1 is the candidate; Image 2 is the optional reference.','Critical issues force a fail verdict regardless of the model-reported score.'),
'OllamaEvaluationCompare':('Evaluation Compare','Compares two typed evaluations using normalized scores.','Label candidates and tune tie tolerance.','No LLM call occurs.'),
'OllamaEvaluationInspect':('Evaluation Inspect','Exposes evaluation type, score, verdict, criteria, issues, recommendation, and confidence.','Use for routing or debugging evaluation results.','Inspection does not call Ollama.'),
'OllamaAnalysisQualityInspector':('Image Analysis Quality Inspector','Computes deterministic field confidence and hallucination/cross-field warnings from Image Analyzer, Image Sequence, Photoset, or Web Image analysis JSON.','Connect `full_analysis` or aggregate analysis JSON and inspect confidence, grade, low-confidence fields, warnings, and reconstruction prompt.','Scores are transparent heuristics, not calibrated probabilities; no LLM call occurs.'),
'OllamaMultiImageConsistencyInspector':('Multi-Image Consistency Inspector','Compares identity, clothing, environment, camera, lighting, palette, and style across structured image analyses.','Connect aggregate JSON from Image Sequence or Photoset Folder Analyzer to inspect stable, variable, and conflicting fields plus explicit drift scores.','The bounded local comparison decodes no images and makes no embedding, network, or LLM calls.'),
'ComfyUIOMGImageFolderSelect':('Image From Folder','Loads one indexed image and alpha mask from a naturally sorted folder without decoding the remaining images.','Choose a folder, a 1-based index, target dimensions, and resize mode; the index clamps to the available range.','Only the selected image is decoded. Folder scans and source file/pixel sizes remain bounded, and external paths require explicit opt-in.'),
'ComfyUIOMGImageFolderLoader':('Image Folder Batch Loader','Loads a naturally sorted folder of images into one normalized ComfyUI IMAGE/MASK batch.','Choose a folder under ComfyUI input, target dimensions, resize mode, optional filter/stride, and a bounded image count.','Loading is local-only. Source bytes, source pixels, scan entries, output pixels, and batch size are capped; external paths require explicit opt-in.'),
'ComfyUIOMGImageListLoader':('Multiple Image Paths Loader','Loads an ordered newline list or JSON array of image paths into one normalized batch.','Connect `file_paths` from Folder File Catalog or paste paths in the required order, then select target dimensions and resize mode.','The same local path sandbox, alpha-mask semantics, file/pixel checks, and total batch-memory cap apply.'),
'ComfyUIOMGFolderCatalog':('Folder File Catalog','Builds a lightweight metadata catalog for image, video, audio, or text/data files without decoding media.','Select a file family and connect an image-only catalog path list to Multiple Image Paths Loader, or use the JSON manifest for routing.','Only metadata is read. Scans, output entries, total bytes, symlinks, and external paths are bounded or validated.'),
'ComfyUIOMGDataFileLoader':('Text & Data File Loader','Loads bounded text, prompt, subtitle, and structured-data files from ComfyUI input.','Use raw text directly, or consume normalized JSON for JSON, JSONL, CSV, and TSV files.','No network or LLM call occurs. Extensions, byte size, encoding, and parsed row counts are restricted.'),
}

for node_id,(title,purpose,usage,notes) in DOCS.items():
 text=f'''# {title}

## Purpose

{purpose}

## Typical workflow

{usage}

## Reliability and privacy

{notes}

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
'''
 (OUT/f'{node_id}.md').write_text(text,encoding='utf-8')
print(f'Generated {len(DOCS)} embedded node help pages')
