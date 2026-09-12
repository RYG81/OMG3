# Image Analysis Quality Inspector

## Purpose

Computes deterministic field confidence and hallucination/cross-field warnings from Image Analyzer, Image Sequence, Photoset, or Web Image analysis JSON.

## Typical workflow

Connect `full_analysis` or aggregate analysis JSON and inspect confidence, grade, low-confidence fields, warnings, and reconstruction prompt.

## Reliability and privacy

Scores are transparent heuristics, not calibrated probabilities; no LLM call occurs.

## Common guidance

- Connect typed sockets only to matching `OMG_…` types.
- Check validity outputs before using newly generated typed assets.
- Use `cache_policy=refresh` when intentionally regenerating an LLM result.
- A remote Ollama server receives prompts and any attached images.

See the repository README, node catalog, workflow templates, and Phase status documents for broader examples.
