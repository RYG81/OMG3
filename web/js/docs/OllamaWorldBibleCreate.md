# World Bible Create

## Purpose

Creates a canonical `OMG_WORLD_BIBLE` with geography, culture, architecture, locations, visual rules, and environment prompt.

## Typical workflow

Define the world concept and desired location count, then reuse the typed Bible across scene and shot planning.

## Reliability and privacy

The world ID is deterministic from title + concept.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
