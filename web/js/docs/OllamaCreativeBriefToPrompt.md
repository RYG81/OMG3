# Creative Brief to Prompt

## Purpose

Deterministically adapts a Creative Brief without another LLM call.

## Typical workflow

Connect a valid brief and choose General, Flux, SDXL, Qwen Image, Wan Video, or LTX Video.

## Reliability and privacy

This node does not mutate the source brief.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
