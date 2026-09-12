# Multi-Image Consistency Inspector

## Purpose

Compares identity, clothing, environment, camera, lighting, palette, and style across structured image analyses.

## Typical workflow

Connect aggregate JSON from Image Sequence or Photoset Folder Analyzer to inspect stable, variable, and conflicting fields plus explicit drift scores.

## Reliability and privacy

The bounded local comparison decodes no images and makes no embedding, network, or LLM calls.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
