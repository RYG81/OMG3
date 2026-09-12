# Shot List Create

## Purpose

Creates a typed `OMG_SHOT_LIST` with exact duration, linked Bibles, camera/motion plans, and first/video/last-frame prompts.

## Typical workflow

Optionally connect Creative Brief, up to three Character Bibles, and a World Bible.

## Reliability and privacy

Domain validation requires unique IDs, sequential order, exact count, and duration sum.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
