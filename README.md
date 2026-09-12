# ComfyUI-OMG

**Local-first Ollama creative intelligence and multimodal workflow nodes for ComfyUI.**

ComfyUI-OMG connects ComfyUI to a local or remote [Ollama](https://ollama.com) server and provides nodes for text generation, vision analysis, prompt engineering, character continuity, scene direction, storyboards, image/video prompt planning, and supporting media utilities.

> **Project status:** `0.1.0-alpha`. The node IDs are being kept compatible, but internal architecture and documentation are actively being stabilized.

## Highlights

- Local text generation, chat, vision and embeddings through Ollama
- Live model capability/context/loaded-state inspection
- Draft 2020-12 JSON Schema-constrained generation with local validation and repair
- Typed Creative Briefs with deterministic Flux, SDXL, Qwen, Wan, and LTX adapters
- Versioned Character Bibles with protected identity updates and scene-specific prompt adapters
- Versioned World Bibles with protected global rules, key locations, and environment adapters
- Typed Shot Lists linked to Creative Briefs, Character Bibles, and World Bibles
- Local document loading, Ollama embedding indexes, semantic retrieval, citations, and safe persistence
- Typed prompt/image evaluations with locally recomputed weighted scores and deterministic A/B comparison
- Private bounded memory cache and per-task provenance hashes/timing
- Domain-level output quality checks with conservative local cleanup before low-temperature repair
- Formatting-only prompt defects are fixed locally without a second Ollama call
- Deterministic image-analysis evidence scores and hallucination/cross-field diagnostics
- Lightweight multi-image identity, wardrobe, scene, camera, lighting, palette, and style drift checks with no extra model call
- Lightweight single-image folder selection plus bounded folder/image-list batches with alpha masks, natural sorting, and memory caps
- Metadata-only image/video/audio/text-data folder catalogs plus JSON/JSONL/CSV/TSV normalization
- Prompt enhancement and target-model prompt building
- Structured image, sequence and photoset analysis
- Character sheets, continuity checks, pose/outfit/expression planning
- Scene direction, lighting, environment and storyboard tools
- LTX and Wan video prompt planning
- Image, mask, text, math, flow, composition and VFX helpers
- 204 currently registered nodes organized under `ComfyUI-OMG`

## Requirements

- A working ComfyUI installation
- Python 3.10 or newer in the ComfyUI environment
- [Ollama](https://ollama.com/download) running locally or on a trusted remote server
- At least one Ollama model

Example Ollama setup:

```bash
ollama serve
ollama pull llama3.2:latest
```

For image-analysis nodes, install a vision-capable Ollama model. For the Embeddings node, use an embedding model.

## Installation

### ComfyUI Manager / Registry

Registry publication is being prepared for the first beta. Once published, search for **ComfyUI-OMG** in ComfyUI Manager or run:

```bash
comfy node install comfy-omg
```

### Manual installation

From the ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/RYG81/Comfy-OMG.git
cd Comfy-OMG
python -m pip install -r requirements.txt
```

Use the Python executable belonging to ComfyUI. On Windows Portable, this is normally `python_embeded/python.exe` from the portable directory.

Restart ComfyUI after installation.

## Optional dependencies

The base installation intentionally stays small. Some node groups need optional packages:

```bash
# Video loading/saving with OpenCV
python -m pip install "opencv-python>=4.8.0"

# Optional smolagents-backed web image provider
python -m pip install "ddgs>=9.0.0" "smolagents[toolkit]>=1.26.0"
```

The built-in DuckDuckGo/Wikimedia fallback in the Web Image Tool does not require `smolagents`.

## Five-minute quick start

1. Start Ollama: `ollama serve`.
2. Add **Ollama Model Loader** in ComfyUI.
3. Enter the server URL, normally `http://127.0.0.1:11434`.
4. Enter an installed model name, such as `llama3.2:latest`.
5. Connect the `OLLAMA_MODEL` output to **Ollama Text Generate** or **Ollama Prompt Builder**.
6. Enter a prompt and queue the workflow.

The original main loader is still registered with its compatibility-sensitive ID **`OllamaModelLoader`** and display name **Ollama Model Loader**. Find it under **`ComfyUI-OMG/Core`**. It was not replaced by the new local file-loader nodes.

If Ollama is remote, remember that prompts and attached images are sent to that server.

## System prompt layering

Every generative node that consumes `OLLAMA_MODEL` now exposes a multiline **custom_system_prompt** input. System instructions are assembled in this order:

1. the node's built-in/task-specific system prompt;
2. the optional global custom system prompt from **Ollama Model Loader**;
3. the optional node-specific custom system prompt.

The built-in contract is retained rather than replaced. This is especially important for structured JSON fields and legacy socket semantics. Global and per-node additions are capped at 16,000 characters each, participate in execution/cache fingerprints, and are represented only by hashes in provenance. Embeddings, semantic RAG embedding/search, and Model Inspector do not show this field because those Ollama endpoints do not accept a generative system prompt. For stateful Chat, reset/start a conversation when changing system instructions so the new system message becomes the conversation head.

## Local folder and file loading

The `ComfyUI-OMG/Loaders` category includes five local-only nodes:

- **Image From Folder** — loads only one clamped, 1-based item from a naturally sorted folder for low-memory iteration;
- **Image Folder Batch Loader** — naturally sorted folder loading with recursive/filter/start/stride controls, alpha masks, three resize modes, and a bounded output batch;
- **Multiple Image Paths Loader** — ordered newline or JSON-array paths, including direct connection from Folder File Catalog;
- **Folder File Catalog** — metadata-only catalogs for supported image, video, audio, and text/data extensions without decoding media;
- **Text & Data File Loader** — raw loading for text, Markdown, prompts, YAML, XML/HTML, subtitles, logs, INI/CFG, plus normalized JSON for JSON, JSONL, CSV, and TSV.

Paths stay under ComfyUI input unless external access is explicitly enabled. Image file bytes, source megapixels, folder scan entries, output pixels, file bytes, and parsed rows are bounded. These nodes make no Ollama or network calls and add no dependencies. See [Local Loaders](docs/LOCAL_LOADERS.md) for exact formats, mask semantics, and limits.

## Dashboard and live model refresh

ComfyUI-OMG ships a local frontend extension with:

- a bottom-panel dashboard showing version, stability tiers, Ollama latency, installed/running models, family, parameter size, quantization, disk size, and on-demand capability/context inspection;
- task-response and embedding-cache statistics plus a clear action;
- a **Refresh Ollama models** button and context-menu action on Ollama Model Loader;
- a detected-model combo that copies the selected model into the existing model field without changing the workflow socket contract.

For security, browser dashboard probes are limited to loopback Ollama by default. To permit a trusted remote Ollama dashboard host, start ComfyUI with an exact allowlist:

```bash
COMFY_OMG_ALLOWED_OLLAMA_HOSTS=ollama.lan,other-trusted-host.example
```

Using `*` allows all dashboard hosts and is not recommended. Workflow nodes can still use an explicitly configured remote Ollama URL regardless of this dashboard-only restriction.

## Workflow templates

Open **Workflow → Browse Templates** and select ComfyUI-OMG. Included starter workflows:

1. Ollama Text Quickstart
2. Structured Generate
3. Creative Brief to Prompt
4. Local RAG Search
5. Character Bible to Prompt
6. World Bible to Prompt
7. Shot List from linked Character/World Bibles
8. Prompt Evaluation

Each template ships with a local 16:9 thumbnail and uses only ComfyUI-OMG nodes.

## Reusable subgraph blueprints

Modern ComfyUI installations scan ComfyUI-OMG's `subgraphs/` directory and expose these reusable components in the subgraph browser:

1. **OMG Prompt Evaluation** — model + prompt → evaluation, score, verdict, improved prompt
2. **OMG Character Bible Prompt** — Character Bible + scene → positive/negative prompt and locks
3. **OMG Local RAG Search** — embedding model + index + query → context and citations
4. **OMG Shot Prompt Extractor** — Shot List → first/video/last-frame prompts, negative, and locks

The files use native ComfyUI `definitions.subgraphs` format with virtual input/output nodes and promoted widgets.

## Embedded node documentation

The ComfyUI node-docs panel automatically discovers 35 Markdown help pages under `web/js/docs/`. These cover Model Loader, structured generation, cache control, Creative/Character/World/Shot typed assets, RAG, evaluation, and image-quality nodes. Pages include purpose, typical workflow, reliability, caching, and privacy guidance.

## Node families

| Family | Purpose |
|---|---|
| Core | Model configuration, text, chat, vision, embeddings and model management |
| Loaders | Bounded image-folder/image-list batches, metadata catalogs, and text/data files |
| Prompt | Enhancement, criticism, translation, combination, variation and wildcards |
| Character | Character sheets, anchors, outfits, poses, expressions and continuity |
| Scene / Design | Scene direction, action, emotion, lighting, backgrounds and materials |
| Image | Analysis, comparison, edit prompts, style, palettes, photosets and web references |
| Storyboard / Video | Story planning, shots, LTX/Wan prompts and frame-batch utilities |
| Media helpers | Image, mask, color, text, composition, animation and VFX operations |
| Utility / Flow / Math | Routing, conversion, inspection, files, seeds and calculations |

The dashboard groups the catalog into **Core Beta**, **Media**, and **Labs** without changing persistent IDs or categories. See [Stability Tiers](docs/STABILITY_TIERS.md) and the [Output Quality Pipeline](docs/OUTPUT_QUALITY.md).

See the complete [204-node catalog](docs/NODES.md), [Phase 0 status](docs/PHASE_0_STATUS.md), [Phase 1 status](docs/PHASE_1_STATUS.md), and [Phase 2 status](docs/PHASE_2_STATUS.md).

## Troubleshooting

### Ollama cannot be reached

- Confirm `ollama serve` is running.
- Open `http://127.0.0.1:11434` in a browser or run `curl http://127.0.0.1:11434/api/tags`.
- Check the URL entered in Ollama Model Loader.
- If ComfyUI runs in Docker, `127.0.0.1` points at the container; use an accessible host address.

### Model not found

Run `ollama list`, then enter the exact model name, including its tag.

### Video node says OpenCV is missing

Install the optional video dependency in ComfyUI's Python environment, then restart ComfyUI.

### A web provider is unavailable

Use `auto`, `duckduckgo`, or `wikimedia`, or install the optional web-search dependencies shown above.

## Privacy and security

- Local Ollama requests stay on the configured Ollama server; a remote server receives prompts and images.
- Web-search nodes send the query to the selected public provider and download third-party content.
- Keep SSL verification enabled unless diagnosing a trusted local environment.
- Do not place API keys or credentials inside workflow widgets or URLs.
- File, image-folder, and video paths stay inside ComfyUI input/output directories by default; loaders and the photoset analyzer require explicit opt-in for external absolute paths.

Please report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

Copyright © 2026 RYG81 and contributors.

ComfyUI-OMG is licensed under the [GNU Affero General Public License v3.0](LICENSE). Distributed modifications must remain available under the same license, and network-hosted modified versions must offer their corresponding source as required by AGPL-3.0.
