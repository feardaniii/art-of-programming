# Ghid Rapid pentru PR-uri — Task-uri cu Presentation Methods

Acesta este un ghid scurt pentru colegii care lucrează la un **presentation method** și vor să evite cele mai comune probleme la PR review.

Nu trebuie să înțelegi tot repo-ul. Concentrează-te pe metoda ta, compară cu metode similare, folosește AI Checklist-ul și cere ajutor devreme.

---

## 1. Începe din locul corect

Înainte să scrii cod:

- Lucrează pe un feature branch pornit din `development`.
- Citește cu atenție descrierea task-ului.
- Verifică metoda în `docs/presentation-methods/_REGISTRY.md`.
- Caută o metodă similară deja existentă și folosește-o ca model.
  - Pentru metode interactive, uită-te la exemple precum `QUIZ`, `EMPTY_CELLS`, `CHECKLIST_TASK` sau PR-uri recente.
- Evită să modifici fișiere care nu au legătură cu task-ul tău.

Mentalitate utilă:

> „Adaug o singură metodă, nu repar tot repo-ul.”

---

## 2. Înțelege ce tip de metodă construiești

Înainte de implementare, încearcă să înțelegi ruta metodei.

De obicei, o metodă intră într-una dintre aceste categorii:

- **Simple inline method** — rescrie sau structurează direct conținutul lecției.
- **Dedicated generator** — are nevoie de schema proprie, prompt, agent, registry, teste și docs.
- **Generic presentation agent method** — poate folosi deja sistemul generic de presentation agent.

Dacă metoda are nevoie de `generation_schema`, ID-uri stabile, reguli de validare, answer data, categorii, scoring sau payload special pentru frontend, probabil are nevoie de un dedicated generator.

Dacă nu ești sigur, întreabă pe Discord înainte să construiești prea mult.

---

## 3. Aliniază schema, prompt-ul și output-ul

Aici apar multe probleme la review.

Verifică următoarele:

- Schema definește structura reală a output-ului.
- Prompt-ul cere aceeași structură.
- Agent response model așteaptă aceeași structură.
- Docs-urile au exemple care respectă structura reală.

Pentru dedicated generators, prompt-ul trebuie de obicei să ceară wrapper-ul complet, nu doar `generation_schema`.

Exemple de câmpuri care apar frecvent în wrapper:

```text
method
title
sections
generation_schema
is_valid
issues
```

Dacă aceste părți nu se potrivesc, agentul poate eșua chiar dacă implementarea pare corectă la prima vedere.

---

## 4. Fișiere la care te poți aștepta pentru un dedicated generator

Dacă adaugi un agent de tip `presentation_generator_<method>`, e posibil să modifici fișiere precum:

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

Nu toate task-urile vor atinge toate fișierele, dar dacă înregistrezi un agent nou, verifică toate zonele de registration/config/docs/tests.

Greșeală comună:

> „Am adăugat agentul într-un loc, dar am uitat model config, evals, docs sau node routing.”

---

## 5. Clarifică selector behavior

Fiecare metodă ar trebui să spună clar dacă este:

- **explicit-only** — rulează doar când este aleasă direct
- **auto-selectable** — selector-ul o poate alege automat

Pentru metode noi sau interactive complexe, `explicit-only` este de multe ori alegerea mai sigură, dacă task-ul nu cere altceva.

Menționează acest lucru în method doc și în PR description.

---

## 6. Folosește AI PR Checklist înainte de review

Există și un AI Checklist mai strict. Este recomandat să îl folosești înainte să ceri review.

Îl poți folosi în câteva moduri simple:

### Chat / web AI

Poți lipi checklist-ul și cere:

```text
Review my PR plan using this checklist.
Only list likely blockers and missing files.
Do not rewrite anything yet.
```

### Codex / VS Code AI

Poți cere AI-ului să inspecteze branch-ul local:

```text
Compare my current branch against origin/development using the PR checklist.
Do not edit files.
List missing registrations, schema/prompt mismatches, test gaps, and unrelated files.
```

### GitHub CLI după ce PR-ul există

Dacă PR-ul este deja deschis:

```text
Use gh to inspect PR #XX comments, inline review comments, changed files, and failed checks.
Summarize required fixes before editing anything.
```

Important:

> Cere mai întâi un plan sau un diff. Nu lăsa AI-ul să modifice liber repo-ul. Dacă atinge fișiere fără legătură cu task-ul, oprește-te și verifică.

---

## 7. Testează path-ul important

Nu trebuie să rulezi tot repo-ul de fiecare dată.

Încearcă măcar să demonstrezi că:

- schema validează corect
- factory-ul se construiește
- registry-ul poate găsi agentul
- node-ul face routing corect
- conversia către `PresentationMethodOutput` funcționează
- există eval/config files, dacă sunt necesare

Exemple utile:

```bash
uv run pytest tests/agno_agents/test_presentation_generator_<method>_factory.py -v
uv run pytest tests/agno_agents/nodes/test_node_presentation_generator_<method>.py -v
```

Dacă testele pică, citește primul error real și cere ajutor cu output-ul exact.

---

## 8. Înainte să deschizi sau să actualizezi PR-ul

Fă o verificare rapidă:

```bash
git status --short --branch
git diff --name-status origin/development...HEAD
git diff --stat origin/development...HEAD
```

Caută:

- fișiere fără legătură cu task-ul
- generated files
- logs / traces / cache files
- local notes
- `.env` sau secrets
- refactorizări mari pe care nu le-ai intenționat

PR description ar trebui să menționeze:

- ce metodă ai adăugat/modificat
- ce route folosește
- ce docs/tests ai adăugat
- dacă este `explicit-only` sau `auto-selectable`
- ce teste ai rulat
- ce ai lăsat intenționat pentru mai târziu, dacă este cazul

---

## 9. Dacă CI sau review-ul pică

Nu intra în panică.

Încearcă să clasifici problema:

- **Ține de codul tău** — repar-o.
- **Este repo baseline debt** — menționează, dar nu modifica fișiere fără legătură decât dacă ți se cere.
- **Este tooling/API issue** — cere ajutor înainte să pierzi ore întregi.

Când ceri ajutor, include:

```text
task:
branch/PR:
failing check:
error message:
files changed:
what I already tried:
```

Așa va fi mult mai ușor pentru ceilalți să te ajute rapid.

---

## Reminder final

Nu trebuie să știi tot repo-ul.

Fă modificări mici.  
Compară cu metode deja funcționale.  
Folosește AI Checklist-ul.  
Cere ajutor devreme.  
Ține PR-ul cât mai focusat.

Un PR bun nu trebuie să fie perfect din prima.  
Un PR bun este clar, limitat ca scope și ușor de review-uit.
