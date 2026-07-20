# Quick PR Guide for Presentation Method Tasks

This is the short version. Use it when you are working on a presentation method and want to avoid the most common PR review problems.

You do **not** need to understand the whole repo. Focus on your method, compare with similar methods, use the AI checklist, and ask for help early.

---

## 1. Start from the right place

Before coding:

- Work on a feature branch based on `development`.
- Read the task description carefully.
- Check `docs/presentation-methods/_REGISTRY.md` for your method.
- Find one similar existing method and use it as your reference.
  - For interactive methods, check examples like `QUIZ`, `EMPTY_CELLS`, `CHECKLIST_TASK`, or recent PRs.
- Avoid touching unrelated files.

A good mindset:

> “I am adding one method, not fixing the whole repo.”

---

## 2. Know what kind of method you are building

Try to understand the method route before writing code.

Most methods are one of these:

- **Simple inline method** — mostly rewrites lesson text.
- **Dedicated generator** — needs schema, prompt, agent, registry, tests, and docs.
- **Generic presentation agent method** — can reuse the existing presentation agent.

If your method needs a structured `generation_schema`, stable IDs, validation rules, answer data, categories, scoring, or special frontend payloads, it probably needs a dedicated generator.

If unsure, ask on Discord before building too much.

---

## 3. Keep schema, prompt, and output aligned

This is one of the biggest review issues.

Make sure:

- The schema defines the real output.
- The prompt asks for the same output.
- The agent response model expects the same output.
- The docs show examples that match the real output.

For dedicated generators, the prompt usually needs to ask for the full wrapper, not only the inner `generation_schema`.

Example wrapper fields often include:

```text
method
title
sections
generation_schema
is_valid
issues
```

If these do not match, the agent may fail even if the code “looks right.”

---

## 4. Expected files for a dedicated generator

If you add a dedicated `presentation_generator_<method>` agent, expect to touch files like:

```text
src/common/schemas/presentation.py
src/agno_agents/agents/presentation_generators.py
src/agno_agents/agents/prompts/presentation_generator.yaml
src/agno_agents/agent_registry.py
src/agno_agents/models.py
src/graphs/course_generation/nodes/agent_nodes/presentation_generator.py
tests/evals/baseline/model_config.json
tests/evals/presentation_generator_<method>.yaml
docs/AGENTS_REFERENCE.md
docs/presentation-methods/_REGISTRY.md
docs/presentation-methods/<method>.md
tests/...
```

You may not need all of them for every task, but if you register a new agent, check all registration/config/docs/test surfaces.

Common mistake:

> “I added the agent in one place, but forgot model config, evals, docs, or node routing.”

---

## 5. Be clear about selector behavior

Every method should say whether it is:

- **explicit-only** — it runs only when selected directly
- **auto-selectable** — the selector may choose it automatically

For new or complex interactive methods, explicit-only is often safer unless the task says otherwise.

Document this in the method doc and PR description.

---

## 6. Use the AI PR Checklist before review

There is a stricter AI checklist guide. Use it before requesting review.

You can use it in three simple ways:

### Chat / web AI

Paste the checklist and ask:

```text
Review my PR plan using this checklist.
Only list likely blockers and missing files.
Do not rewrite anything yet.
```

### Codex / VS Code AI

Ask it to inspect your branch:

```text
Compare my current branch against origin/development using the PR checklist.
Do not edit files.
List missing registrations, schema/prompt mismatches, test gaps, and unrelated files.
```

### GitHub CLI after opening a PR

If your PR exists:

```text
Use gh to inspect PR #XX comments, inline review comments, changed files, and failed checks.
Summarize required fixes before editing anything.
```

Important:

> Ask AI for a plan or diff first. Do not let it freely edit the repo. If it touches unrelated files, stop and review.

---

## 7. Test the important path

You do not need to run the whole repo every time.

At minimum, try to prove:

- the schema validates
- the factory builds
- the registry can find the agent
- the node routes correctly
- conversion to `PresentationMethodOutput` works
- eval/config files exist if required

Useful examples:

```bash
uv run pytest tests/agno_agents/test_presentation_generator_<method>_factory.py -v
uv run pytest tests/agno_agents/nodes/test_node_presentation_generator_<method>.py -v
```

If tests fail, read the first real error and ask for help with the exact output.

---

## 8. Before opening or updating the PR

Quick final check:

```bash
git status --short --branch
git diff --name-status origin/development...HEAD
git diff --stat origin/development...HEAD
```

Look for:

- unrelated files
- generated files
- logs/traces/cache files
- local notes
- `.env` or secrets
- broad refactors you did not intend

Your PR description should mention:

- what method you added/changed
- what route it uses
- what docs/tests were added
- whether it is explicit-only or auto-selectable
- what tests you ran
- anything intentionally left out

---

## 9. If CI or review fails

Do not panic.

Classify the issue:

- **Your code** — fix it.
- **Existing repo baseline** — mention it, but do not fix unrelated files unless asked.
- **Tooling/API issue** — ask for help before wasting hours.

When asking for help, include:

```text
task:
branch/PR:
failing check:
error message:
files changed:
what I already tried:
```

This makes it much easier for others to help.

---

## Final reminder

You are not expected to know the entire repo.

Make small changes.  
Compare with working examples.  
Use the AI checklist.  
Ask for help early.  
Keep the PR focused.

A good PR is not perfect on the first try.  
A good PR is clear, scoped, and easy to review.
