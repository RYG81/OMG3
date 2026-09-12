# World Bible Inspect

## Purpose

Exposes title, premise, technology, geography, architecture, locations, rules, and master prompt.

## Typical workflow

Use it for debugging or routing Bible sections.

## Reliability and privacy

Inspection does not call Ollama.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
