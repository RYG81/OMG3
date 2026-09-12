# Shot List Update

## Purpose

Revises, appends, or resequences a Shot List while protecting asset links and core sequence settings.

## Typical workflow

Append mode preserves every existing shot exactly. Enable shot-ID preservation for normal revisions.

## Reliability and privacy

Rejected changes return the original valid Shot List.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
