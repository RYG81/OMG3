# Image Folder Batch Loader

## Purpose

Loads a naturally sorted folder of images into one normalized ComfyUI IMAGE/MASK batch.

## Typical workflow

Choose a folder under ComfyUI input, target dimensions, resize mode, optional filter/stride, and a bounded image count.

## Reliability and privacy

Loading is local-only. Source bytes, source pixels, scan entries, output pixels, and batch size are capped; external paths require explicit opt-in.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
