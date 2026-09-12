# Ollama Model Loader

## Purpose

The main Ollama entry node. It validates an installed model and emits the shared `OLLAMA_MODEL` profile without changing the legacy `OllamaModelLoader` ID.

## Typical workflow

Find it under `ComfyUI-OMG/Core`, enter the exact model tag from `ollama list`, optionally add global custom system instructions, then connect `ollama_model` to AI nodes.

## Reliability and privacy

Schema discovery is offline-safe. Built-in system prompts remain active; global instructions are appended and can be followed by a per-node custom system prompt. Remote servers receive prompts and images.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
