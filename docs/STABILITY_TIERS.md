# ComfyUI-OMG Stability Tiers

ComfyUI-OMG retains every original node ID for workflow compatibility. Stability tiers communicate maintenance priority without renaming or deleting nodes.

## Core Beta

Includes Ollama connectivity, generation, structured tasks, creative AI nodes, typed assets, RAG, evaluation, and continuity workflows.

Expectations:

- covered by node-contract tests;
- schema validation where a fixed output contract exists;
- errors normalized through the canonical client;
- compatibility changes require migration notes;
- highest maintenance priority.

“Beta” remains intentional until real ComfyUI/Ollama and cross-platform smoke testing is completed.

## Media

Includes generic `OllamaNodes…` image, mask, text, color, video, composition, storyboard, and utility helpers.

Expectations:

- IDs and socket contracts retained;
- import/schema contracts tested;
- selected safety and correctness fixes applied;
- runtime/media benchmarks and broader tensor tests remain future work.

## Labs

Includes generic `OllamaNodes…` VFX, animation, math, and flow experiments.

Expectations:

- available for experimentation and backward compatibility;
- syntax and node contracts tested;
- effects may be CPU-heavy or evolve more quickly;
- not promoted as production-stable.

## Classification rule

Non-`OllamaNodes…` IDs are Core Beta. Generic `OllamaNodes…` IDs are Media except VFX, Animate, Math, and Flow categories, which are Labs.

This presentation changes no persistent node ID, category string, socket, or workflow behavior.
