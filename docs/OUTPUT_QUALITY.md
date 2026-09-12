# ComfyUI-OMG Output Quality Pipeline

ComfyUI-OMG does not treat syntactically valid JSON as sufficient. Fixed-contract creative nodes pass through five lightweight layers before their outputs are exposed.

## 1. Ollama schema constraint

The task-specific Draft 2020-12 JSON Schema is sent directly to Ollama as the response format. It defines exact fields, types, arrays, required values, bounds, and additional-property behavior.

## 2. Local JSON Schema validation

The returned value is parsed and validated locally with `jsonschema`. Invalid or missing fields, wrong types, invalid enums, bad bounds, and unexpected fields produce actionable validation errors.

## 3. Conservative local sanitization

Schema-valid direct prompt fields receive only meaning-preserving surface cleanup:

- trim outer whitespace;
- unwrap a code fence only when it surrounds the entire direct output value;
- remove a known output label such as `Positive Prompt:` only from the start of a direct output;
- remove exact duplicate sentences without paraphrasing the remaining text.

Internal code fences, unknown labels, non-prompt fields, and semantically distinct clauses are left untouched. The generic Structured Generate node is excluded because its user-defined schema may require exact string preservation. The original raw model response stays available in the task trace; provenance records only changed field paths and action names, never field content. A formatting-only result can therefore pass without a second Ollama repair request, reducing latency and compute. Sanitization rules are versioned in task fingerprints.

## 4. Domain-level quality checks

Schema-valid values are checked for usability. Current rules include:

- minimum useful length for directly usable prompts and production descriptions;
- no Markdown code fences in direct prompt outputs;
- no output labels such as `Positive Prompt:` inside direct prompt fields;
- no unresolved `[insert subject]`-style placeholders;
- no serialized JSON masquerading as a prompt string;
- required wildcard `{option1|option2}` syntax;
- required LTX Ingredients `Reference sheet:` and `Generated video:` sections;
- valid `#RRGGBB` colors in palette output;
- unique Character Bible immutable anchors;
- unique World Bible key-location names;
- useful, distinct Prompt Variations and Storyboard panels;
- non-empty free-form Text, Chat, and Vision responses;
- locally parsed and normalized JSON when core nodes request JSON mode;
- finite, non-zero, dimension-consistent embeddings with exact response count;
- concrete concept overlap between source intent and enhanced/translated/edited outputs;
- natural-language structure for Flux/Qwen targets instead of excessive tag soup;
- explicit camera or subject motion for Wan/LTX/video targets;
- must-avoid coverage plus target-model/aspect consistency in Creative Briefs;
- reconstruction-prompt overlap with analyzed subject/location/lighting/camera evidence;
- master-prompt coverage of Character Bible anchors and World Bible rules;
- Shot List prompt depth, motion, links, identifiers, ordering, duration, and append/revision semantics.

Task quality rules have their own version and are included in cache fingerprints and provenance. Updating a rule therefore cannot silently reuse an older cached response.

## 5. One repair attempt

If schema or quality checks fail, the model receives:

- the original request;
- validation/quality errors;
- its previous response;
- the same JSON Schema constraint;
- a low repair temperature.

Only a repaired response that passes both schema and quality checks is exposed as valid. An unresolved response follows the node's documented fallback behavior.

## Additional deterministic safeguards

Some typed assets receive further local checks:

- Character Bible protected identity fields;
- World Bible protected global rules;
- Shot List unique IDs, sequential ordering, exact count, duration sums, links, and append semantics;
- RAG vector dimensions, finite values, normalized embeddings, and exact embedding model;
- Evaluation scores and verdicts recomputed from fixed local weights;
- Regional Prompt 1x3/3x1 mapping normalized into legacy sockets;
- Storyboard arrays normalized to string outputs;
- LoRA list outputs normalized to strings;
- Inpaint masks sent losslessly with explicit fill/preserve semantics.

## Provenance

Task provenance records:

- task and template version;
- model and endpoint;
- request, prompt, system, schema, and image hashes;
- generation options;
- quality-rule version;
- cache policy/hit;
- attempts;
- validity and final validation errors;
- execution duration.

Raw prompts and images are not copied into provenance.

## Layered custom system prompts

Generative `OLLAMA_MODEL` consumers expose a multiline `custom_system_prompt` input. The final system message preserves this order:

1. built-in task/schema instructions;
2. Model Loader global custom instructions;
3. node-specific custom instructions.

Custom text is additive rather than a replacement for the node's output contract. Each custom layer is limited to 16,000 characters. The fully composed system hash participates in cache fingerprints/provenance, but private instruction text is not copied into provenance. Non-generative model consumers—Embeddings, RAG embedding/search, and Model Inspector—are intentionally excluded because their API operations have no generative system-message field.

## Image-analysis evidence and hallucination diagnostics

Image Analyzer, Image Sequence Analyzer, Photoset Folder Analyzer, and Web Image Tool analyses now include `_quality` diagnostics with:

- per-field confidence scores;
- aggregate confidence and high/medium/low grade;
- low-confidence fields;
- explicit visibility limitations;
- hedged or alternate language;
- visibility/detail conflicts;
- field labels or Markdown inside reconstruction prompts;
- contradictory clothing state;
- sitting poses without visible support;
- subject count versus scene-element count drift.

Critical warnings and very low evidence quality participate in repair before output is accepted. The separate Image Analysis Quality Inspector can process direct or aggregate analysis JSON. Scores are deterministic heuristics, not calibrated probabilities.

## Multi-image consistency and drift

Image Sequence and Photoset Folder outputs also include `_consistency` diagnostics across:

- subject identity;
- clothing and coverage;
- environment;
- camera and lens language;
- lighting setup;
- color palette;
- visual medium/style.

Each field is classified as `stable`, `variable`, `conflicting`, or `insufficient_evidence`, with an explicit 0–1 drift score, repeated terms, bounded conflict details, and an overall consistency grade. Explicit contradictions such as person-count, hair-color, clothing-coverage, indoor/outdoor, setting, time-of-day, and photographic-versus-painted medium drift are routed into the existing sequence/photoset synthesis prompt as guardrails.

This check is intentionally lightweight: it uses bounded local text/claim comparison, decodes no images, loads no embedding model, performs no network request, and adds no Ollama call. Inputs are capped at 128 analyses, text and token work are bounded, conflict detail output is capped, and synthesis guidance is limited to 1,600 characters. The separate Multi-Image Consistency Inspector exposes the report without changing legacy node sockets.

## Golden-output regression fixtures

Six representative outputs are stored under `tests/fixtures/golden_outputs/`:

- Prompt Enhancer
- Image Analyzer
- Character Bible
- World Bible
- Shot List
- Evaluation

CI validates these fixtures against current schemas, quality rules, typed-asset domain checks, target-model rules, and locally normalized evaluation scores. This catches unintended output-contract or quality regressions without requiring a live model.
