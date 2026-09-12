# Shot List Get Shot

## Purpose

Extracts one indexed shot for downstream image/video generation.

## Typical workflow

Use `shot_index` to retrieve frame prompts, video prompt, negative, camera, lighting, and locks.

## Reliability and privacy

The index is clamped to the available range.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
