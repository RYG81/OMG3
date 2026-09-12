# Folder File Catalog

## Purpose

Builds a lightweight metadata catalog for image, video, audio, or text/data files without decoding media.

## Typical workflow

Select a file family and connect an image-only catalog path list to Multiple Image Paths Loader, or use the JSON manifest for routing.

## Reliability and privacy

Only metadata is read. Scans, output entries, total bytes, symlinks, and external paths are bounded or validated.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
