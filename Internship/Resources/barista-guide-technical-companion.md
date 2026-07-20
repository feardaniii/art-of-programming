# Technical Companion to the Barista Field Guide

This is not a rewrite proposal for `the-art-of-barista-at-home-field-guide.md`. Use the field guide for the story and onboarding arc. Use this companion when you need the deeper implementation view: what the guide compresses, which tech is involved, and which code paths are worth asking an AI CLI agent to inspect.

Source repo checked: `/home/mintmainog/workspace/vs_code_workspace/skillbrain-ai-content-engine`

## How To Read This

The field guide is intentionally narrative. This document stays close to the current repo and avoids repeating the coffee-course examples unless a repo detail changes the meaning.

Most exact paths are collapsed into search anchors. The useful workflow is: read the concept here, then search the repo for the named node, class, or schema.

## Tech Stack At A Glance

- LangGraph owns the course-generation state machine through `CourseGenState`.
- Agno agents do most LLM work; `agent_registry` caches agents by agent name, user, session, and web-search setting.
- Pydantic schemas validate important agent outputs and state handoffs, though some intermediate staging fields are plain dicts.
- Ingestion parses and normalizes files, chunks them semantically, writes embeddings to Qdrant, and builds `DocumentMap` objects.
- Qdrant stores dense OpenAI embeddings and sparse BM25 vectors. The default embedding model is `text-embedding-3-small`.
- `DocumentMapBuilder` uses Gemini Flash (`gemini-2.0-flash`) for ingestion-time section summaries and global document synthesis.
- Retrieval is hybrid: dense semantic search plus sparse BM25 keyword search, fused with RRF, then reranked.
- Presentation methods are represented by `PresentationMethod`, `PresentationMethodInput`, `PresentationMethodOutput`, method-specific schemas, and optional `MediaPrompt` payloads.

## Current Graph Shape

The current main graph starts with ingestion, conditionally assembles RAG, then moves through pedagogy, structure, lesson generation, presentation generation, and final content verification.

```text
ingest_documents
  -> assemble_rag_context        # only when document_maps exist
  -> content_summarizer
  -> learning_objectives
  -> blooms_taxonomy
  -> verifier_pedagogical_design
  -> layer_architect_real_events
  -> layer_architect_practice
  -> layer_architect_theory
  -> layer_architect_bonus
  -> merge_layer_structure
  -> verifier_layer_architects
  -> architecture_selector
  -> module_generator
  -> curriculum_designer
  -> verifier_structure
  -> prepare_next_lesson
       -> lesson_generator
       -> exercise_generator
       -> example_generator
       -> merge_lesson_content
       -> presentation_generator
       -> prepare_next_lesson
  -> verifier_content
  -> END
```

The default local HITL interrupts are currently after:

- `verifier_pedagogical_design`
- `verifier_layer_architects`
- `verifier_structure`
- `merge_lesson_content`

That means the guide's "each phase pauses" wording is a simplification. The code pauses at major verified milestones and at each merged lesson, not after every named phase.

Search anchors: `build_course_gen_graph`, `PIPELINE_HITL_INTERRUPT_NODES`, `CourseGenState`.

## Phase 1: Ingestion, Indexing, And RAG

The guide's two-layer framing is right: `DocumentMap` plus `RAGContext`. The important correction is that ingestion already creates summaries.

The actual ingestion shape is:

```text
source files / web docs
  -> parsed document
  -> ChunkBatch
  -> ParentChunk[]
  -> DocumentChunk[]
  -> Qdrant write
  -> SectionSummary[]
  -> global synthesis
  -> DocumentMap
```

`DocumentMap` is not just an AI-served table of contents. It is a compact source-intelligence artifact: section summaries, section topics, prerequisites, learning outcomes, style, coverage summary, global key concepts, and a flat topic index. "High-resolution table of contents" is a useful metaphor, but "source map" is more precise.

`SectionSummary` is the per-section summary. `DocumentMap` is the full ingestion-time summary artifact that later agents can browse.

`RAGContext` is assembled after ingestion. It turns the course topic into five targeted queries, searches the Qdrant collection, dedupes chunks, and keeps the top retrieved chunks for prompt injection.

The retrieval path is hybrid:

- Dense search: OpenAI embeddings capture semantic similarity.
- Sparse search: BM25 captures exact words, names, acronyms, and rare terms.
- RRF: Reciprocal Rank Fusion combines the dense and sparse result lists by rank.
- Reranking: a reranker sorts the candidate chunks again before they become `RAGContext`.

RRF's rough idea:

```text
final_score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
```

A chunk gets credit if it ranks well in either result list, and extra credit if it ranks well in both.

Search anchors: `node_ingest_documents`, `IngestionPipeline`, `DocumentMapBuilder`, `SectionSummary`, `DocumentMap`, `RAGContextAssembler`, `HybridRetrievalService`, `QdrantVectorStore`, `Fusion.RRF`.

## Phase 2: Content Analysis

The guide says the content summarizer reads the uploads. For onboarding, that is fine. Technically, it reads a prompt assembled from course input, document maps, retrieved chunks, course context, and optional human feedback.

The node writes a simple state field:

```text
content_summary: str
```

The agent/verifier path uses `ContentSummaryOutput`, but downstream state keeps the final summary string.

Search anchors: `node_content_summarizer`, `_build_content_summary_prompt`, `build_rag_prompt_block`, `ContentSummaryOutput`.

## Phase 3: Pedagogical Design

The guide's "two agents collaborate" framing matches the current graph, with a verifier after them:

```text
learning_objectives
  -> blooms_taxonomy
  -> verifier_pedagogical_design
```

The Bloom's output is not only a label beside each objective. The state keeps a Bloom-level mapping that later phases can use as cognitive scaffolding:

```text
learning_objectives: list[str]
blooms_objectives: dict[str, list[str]]
```

Search anchors: `node_learning_objectives`, `node_blooms_taxonomy`, `node_verifier_pedagogical_design`, `PedagogicalDesignOutput`.

## Phase 4: 7-Layer Structure

The guide explains the 7-layer model as the intellectual core. The current implementation generates the course-specific parts of that model through four architect nodes, then merges and verifies them.

```text
layer_architect_real_events   # layers 1 and 6
layer_architect_practice      # layer 2
layer_architect_theory        # layer 3
layer_architect_bonus         # layer 4
merge_layer_structure
verifier_layer_architects
```

The merged structure currently carries layers 1, 2, 3, 4, and 6. In `Layer7Structure`, layers 5 and 7 are fixed platform behavior rather than generated per course.

So the guide's version is good as a mental model, but the implementation is narrower:

- Real events and emotional design are generated together.
- Practice, theory, and bonuses are generated separately.
- Assessment-like work appears later through exercises, quizzes, lesson merging, and verification.
- There is naming drift between `LayerStructureOutput`, `Layer7Structure`, and the guide's "meta-learning" wording.

Search anchors: `node_layer_architect_real_events`, `node_merge_layer_structure`, `LayerStructureOutput`, `Layer7Structure`, `layer_5_social`, `layer_7_organization`.

## Phase 5: Curriculum Structure

The guide is accurate that this phase chooses the learning architecture, creates modules, and writes the `DetailedCourse` skeleton with no chapter content yet.

The current flow is:

```text
architecture_selector
  -> module_generator
  -> curriculum_designer
  -> verifier_structure
```

This phase writes more than the curriculum:

```text
selected_architecture
modules
curriculum
retrieval_schedule
named_tools
```

`retrieval_schedule` is a non-LLM spaced-retrieval map computed after the curriculum exists. The exercise generator uses it later to pull earlier lessons back into practice at planned intervals.

The guide's "contract-first" line should be read as "validated at the boundaries." Agent outputs are validated with Pydantic, then parts may be stored as dicts in staging keys before being converted into downstream schemas.

Search anchors: `node_architecture_selector`, `node_module_generator`, `node_curriculum_designer`, `node_verifier_structure`, `_build_retrieval_schedule`, `DetailedCourse`, `StructureOutput`, `named_tools`.

## Phase 6: Lesson Loop And Presentation Generation

The guide's loop diagram is close, but it omits the current `presentation_generator` node and makes `verifier_content` look more per-lesson than it is in the graph edges.

Current loop:

```text
prepare_next_lesson
  -> lesson_generator
  -> exercise_generator
  -> example_generator
  -> merge_lesson_content
  -> presentation_generator
  -> prepare_next_lesson

all lessons done:
  -> verifier_content
```

`prepare_next_lesson` selects a presentation method before generation. It uses `presentation_selector`, tracks the last five methods in `recent_presentation_methods`, and marks module-ending lessons as milestones.

`lesson_generator` is more than one flat agent call. It builds RAG and layer blocks, gets previous-lesson continuity, asks a coordinator for the lesson arc, then generates chapters, usually in parallel from that blueprint. Chapter Bloom's level and Gagne event guidance are computed from position.

`exercise_generator` and `example_generator` run after lesson content exists. Exercises can use the `retrieval_schedule`, so retrieval practice is tied to the curriculum structure, not just nearby text.

`merge_lesson_content` mutates the `DetailedCourse`, stores `generated_lessons_context`, and pushes the completed lesson into the section store. This is why completed sections can be surfaced before the whole course is done.

`presentation_generator` then creates or routes presentation-specific output. Text methods are built inline, Mermaid/visual/interactive methods go through the unified presentation agent, audio/video methods produce media prompts, and some legacy methods still use dedicated agents.

Search anchors: `node_prepare_next_lesson`, `presentation_selector`, `recent_presentation_methods`, `node_lesson_generator`, `lesson_coordinator`, `_compute_chapter_blooms_level`, `_compute_gagne_event`, `node_merge_lesson_content`, `node_presentation_generator`, `node_verifier_content`.

## Presentation Methods In The Repo

The field guide is right that presentation methods are where interns' tasks usually land, but the current repo splits the work across several surfaces.

A method may need:

- an enum entry in `PresentationMethod`
- a method-specific schema in `presentation.py`
- prompt or generator support
- registry documentation
- tests
- frontend/rendering support for visual or interactive output
- optional `MediaPrompt` generation for image, audio, or video providers

The common output contract is `PresentationMethodOutput`. Method-specific data lives in `generation_schema`.

There is also a count mismatch to keep in mind: the guide mentions 277 PDF cards, while repo docs say "120+" methods, and the current enum represents the implemented/registered subset. For onboarding, exact counts should be treated as product/library context unless the source of truth is clarified.

Search anchors: `PresentationMethod`, `PresentationMethodOutput`, `generation_schema`, `MediaPrompt`, `TEXT_METHODS`, `MERMAID_METHODS`, `_REGISTRY.md`, `_INTEGRATION_GUIDE.md`.

## Cross-Cutting Pieces The Guide Only Gestures At

`CourseContext` grows as nodes finish. It gives downstream agents compact awareness of prior decisions without making every prompt manually pull every state field.

`agent_registry` separates pipeline, verifier, and chat/HITL agents. Pipeline agents are cached per user/session and verifier agents stay stateless.

HITL feedback can become an `intervention_request`. The intervention planner creates an `InterventionPlanOutput`, chooses affected pipeline nodes, and the intervention graph can dispatch repairs with LangGraph `Send`.

`SectionResultStore` is an in-memory per-session store. `merge_lesson_content` pushes completed lessons there so a backend can poll or receive lesson-level results without waiting for full course completion.

Search anchors: `CourseContext`, `update_course_context`, `agent_registry`, `_AGENT_CATEGORIES`, `node_intervention_hub`, `InterventionPlanOutput`, `SectionResultStore`.

## Watch Items

- Read "DocumentMap does not summarize yet" as "it does not create the Phase 2 course summary yet." Ingestion does create section and document summaries.
- The top docstring in `graph.py` is older than the actual graph edges. Trust node registration and edges over the introductory diagram.
- The guide's HITL wording is broader than `PIPELINE_HITL_INTERRUPT_NODES`.
- The guide's 7-layer story is broader than the generated `Layer7Structure` fields.
- Presentation-method counts should be normalized before the guide is treated as a source of truth.
- `CourseContext` layer highlights are worth checking before relying on them; some field names appear to lag the current `Layer7Structure` schema.

