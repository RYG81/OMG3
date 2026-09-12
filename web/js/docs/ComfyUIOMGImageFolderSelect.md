# Image From Folder

## Purpose

Loads one indexed image and alpha mask from a naturally sorted folder without decoding the remaining images.

## Typical workflow

Choose a folder, a 1-based index, target dimensions, and resize mode; the index clamps to the available range.

## Reliability and privacy

Only the selected image is decoded. Folder scans and source file/pixel sizes remain bounded, and external paths require explicit opt-in.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
