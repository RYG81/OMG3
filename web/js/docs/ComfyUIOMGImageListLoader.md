# Multiple Image Paths Loader

## Purpose

Loads an ordered newline list or JSON array of image paths into one normalized batch.

## Typical workflow

Connect `file_paths` from Folder File Catalog or paste paths in the required order, then select target dimensions and resize mode.

## Reliability and privacy

The same local path sandbox, alpha-mask semantics, file/pixel checks, and total batch-memory cap apply.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
