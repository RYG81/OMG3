# Character Bible Create

## Purpose

Creates a canonical `OMG_CHARACTER_BIBLE` with identity, wardrobe, immutable anchors, master prompt, and provenance.

## Typical workflow

Use one stable concept/name and production-lock strictness. Reuse the resulting Bible across scenes and Shot Lists.

## Reliability and privacy

The character ID is deterministic from name + concept.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
