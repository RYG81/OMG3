# Phase 2 Status — UX, Dashboard, Templates, and Onboarding

## Completed

- [x] Restored a real `WEB_DIRECTORY` backed by shipped JavaScript.
- [x] Added a ComfyUI-OMG bottom-panel dashboard using local DOM/CSS only.
- [x] Dashboard status shows version and registered node count.
- [x] Loopback-safe Ollama dashboard probe with explicit remote-host allowlist.
- [x] Installed/running model table with family, parameters, quantization, size, and latency.
- [x] Task-response and embedding-cache stats plus clear action.
- [x] Ollama Model Loader refresh button and context-menu action.
- [x] Frontend-only detected-model combo copies into the existing serialized model field.
- [x] Added dashboard API tests, allowlist tests, route-registration tests, and frontend-asset checks.
- [x] Added four browsable workflow templates with matching 960×540 local thumbnails.
- [x] Template contract tests verify registered node IDs, link endpoints/types, widgets, and thumbnails.
- [x] Added four native ComfyUI `definitions.subgraphs` blueprints with virtual I/O and promoted widgets.
- [x] Blueprint contract tests verify UUIDs, internal registered nodes, virtual links, socket types, and proxy widgets.
- [x] Added Core Beta / Media / Labs classification without changing node IDs or categories.
- [x] Dashboard displays live tier counts; full tier policy is documented.
- [x] Added 35 embedded Markdown help pages discovered by ComfyUI's node-docs panel.
- [x] Added Character Bible, World Bible, linked Shot List, and Prompt Evaluation templates with thumbnails.
- [x] Added `forceInput=True` to every typed `OMG_…` input and a contract test enforcing it.
- [x] Added on-demand model capability/context inspection to dashboard rows and Model Loader.
- [x] Added semantic table captions/headings, ARIA labels/live regions, keyboard focus rings, high-contrast support, and reduced-motion handling.

## Security model

Dashboard Ollama probes permit only `localhost` and loopback IPs by default. Trusted remote dashboard hosts must be listed exactly in:

```text
COMFY_OMG_ALLOWED_OLLAMA_HOSTS=ollama.lan,host.example
```

This restriction is dashboard-only. Workflow nodes continue to use the explicitly configured Model Loader URL. The dashboard does not expose model pull, copy, delete, filesystem, or arbitrary web-fetch routes.

## Included templates

1. Ollama Text Quickstart
2. Structured Generate
3. Creative Brief to Prompt
4. Local RAG Search
5. Character Bible to Prompt
6. World Bible to Prompt
7. Shot List from linked Character/World Bibles
8. Prompt Evaluation

## Included subgraph blueprints

1. OMG Prompt Evaluation
2. OMG Character Bible Prompt
3. OMG Local RAG Search
4. OMG Shot Prompt Extractor

## Stability presentation

- **Core Beta** — Ollama, structured creative, typed assets, RAG, and evaluation nodes
- **Media** — generic image, mask, text, video, color, composition, storyboard, and utility helpers
- **Labs** — VFX, animation, math, and flow experiments

See `docs/STABILITY_TIERS.md` for guarantees and limitations.

## Current automated result

- Registered nodes: 204
- Original IDs preserved: 170/170
- Display mappings: 204
- Tests: 137 passing
- Python/Ruff: passing
- Frontend JavaScript syntax: passing when Node.js is available
- High-severity Bandit findings: 0
- Mandatory dependency vulnerabilities: 0
- Source distribution build: passing

## Remaining Phase 2 priorities

1. Optional localized node documentation
2. Additional end-to-end media workflow templates
3. Optional dashboard model filtering/sorting
4. Real ComfyUI frontend smoke testing when manual validation is resumed
