# Creative Brief Compiler

## Purpose

Turns a concept into a validated, typed `OMG_CREATIVE_BRIEF`.

## Typical workflow

Connect Model Loader, describe the concept/constraints, select target family, then pass `creative_brief` to adapters or evaluators.

## Reliability and privacy

The output includes provenance and validity. Invalid data is not exposed as a usable typed asset.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
