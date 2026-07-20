# Lesson Generation Overlap and Grounding Proposal

**Status:** Draft proposal
**Related:** [ADR-002: Lesson Blueprint Architecture](adr/ADR-002-lesson-blueprint-architecture.md)

## Goal

Catch repeated teaching and unsupported claims after lesson generation, without weakening ADR-002's parallel chapter blueprint.

## Current Architecture Fit

- Qdrant already supports session-scoped semantic retrieval, but it currently indexes ingested source documents, not generated lessons.
- Agno is the correct layer for verifier agents and already powers `verifier_content`.
- `avoid_concepts` currently means concepts covered by other chapters in the same lesson. It should stay scoped to intra-lesson overlap.
- The existing intervention agent plans repairs after feedback; duplicate-content detection should be a verifier/gate, not an intervention task.
- ADR-002 already owns generation-time chapter boundaries through the lesson blueprint; this proposal adds quality gates after generation.
- New agents must follow the repo pattern: Pydantic schema first, `build_agno_agent()`, `get_agent()`, promptfoo evals, and `docs/AGENTS_REFERENCE.md` updates.

## Proposed Changes

### 1. Index Generated Lessons in Qdrant

After `node_merge_lesson_content`, write generated lesson chunks to Qdrant with metadata:

- `source_type=generated_lesson`
- `session_id`
- `module_idx`
- `lesson_idx`
- `chapter_index`
- `block_title`
- `blooms_level`
- `gagne_event`

This enables semantic retrieval against completed generated content.

Implementation note: retrieval needs metadata filters beyond `session_id`, especially `source_type=generated_lesson`, so generated lessons do not mix with uploaded source documents unless explicitly requested.

### 2. Add Duplicate Content Verifier

After `_merged_lesson_out`, retrieve similar prior generated chunks from Qdrant and pass the strongest matches to an Agno verifier.

Verifier classifications:

- `allowed_reference`: brief bridge/callback.
- `repeated_teaching`: same explanation retaught.
- `acceptable_spiral`: same concept at deeper Bloom level.
- `blocking_overlap`: chapter should be regenerated.

Use dynamic top-k:

- `top_k=3` for weak matches.
- `top_k=5` for medium matches.
- `top_k=10` for high-confidence overlap.

Map `blocking_overlap` to the existing `CHAPTER_LEVEL` reroll path.

### 3. Add Grounding Verifier

Add a separate hallucination/grounding verifier that checks generated content against:

- `rag_context`
- `document_maps`
- retrieved source chunks from Qdrant
- optional trace data for audit

Suggested output:

```python
grounding_score: float
unsupported_claims: list[str]
source_coverage: list[str]
hard_fail: bool
```

Hard fail only for unsupported factual claims, invented tools/APIs/statistics, or contradictions with uploaded source material.

## Engineering Requirements

- Define new contracts in `src/common/schemas/` before implementation.
- Update `docs/schemas/` for any new or changed schema.
- Build agents only through `build_agno_agent()` and retrieve them only through `get_agent()`.
- Keep LLM prompts and agent instructions in the Agno agent layer, not inside LangGraph node logic.
- LangGraph nodes should orchestrate retrieval, call registered agents, and return new state dicts.
- Use `structlog` for duplicate/grounding verifier telemetry, including `error=str(e)` and `exc_type=type(e).__name__` in exception paths.
- Add mocked unit tests for new nodes/agents; do not hit live LLMs in unit tests.
- Add promptfoo evals for any new agent.
- If accepted for implementation, create an ADR in `docs/adr/` because this changes lesson-generation architecture.
- If new agents are added, update `docs/AGENTS_REFERENCE.md`.

## Non-Goals

- Do not repurpose `avoid_concepts` for cross-lesson behavior.
- Do not use the intervention agent for automated duplicate-content detection.

## MVP Sequence

1. Define schemas and schema docs for duplicate-content and grounding verifier outputs.
2. Index generated lessons into Qdrant after merge.
3. Add metadata-filtered retrieval for generated lessons.
4. Add duplicate-content verifier using Qdrant hits.
5. Add grounding verifier as a separate quality gate.

## Expected Outcome

ADR-002's parallel generation path stays intact, while post-generation checks catch real overlap and unsupported claims before content is accepted.
