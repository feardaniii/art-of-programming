// tools.jsx — Route Helper, PR Description Generator, Help Template, Pitfalls

const ROUTES = [
  {
    key: "A",
    name: "Inline text method",
    desc: "Mostly rewrites lesson text. No extra LLM generation, no method-specific schema, no answer keys, no learner state, no media, no scoring, no frontend interaction.",
  },
  {
    key: "B",
    name: "Dedicated typed generator",
    desc: "Needs structured payload + strong contract tests. Document/template output, interactive exercise with stable item IDs, answer/validation metadata, task/checklist/card/grid schema, method-specific prompt guardrails.",
  },
  {
    key: "C",
    name: "Unified presentation agent",
    desc: "The existing generic presentation agent already covers the family — Mermaid variants, audio/video script methods, generic visual methods where sections + MediaPrompt is enough.",
  },
];

const ROUTE_QUESTIONS = [
  {
    id: "rewrite",
    q: "Can the method be represented by existing lesson content, just rewrapped?",
    opts: ["Yes — text only", "No — needs more"],
  },
  {
    id: "schema",
    q: "Does it need a structured generation_schema with stable IDs, validators, answers, or scoring?",
    opts: ["Yes", "No"],
  },
  {
    id: "family",
    q: "Is it a Mermaid / audio / video / generic visual method?",
    opts: ["Yes", "No"],
  },
];

function decideRoute(answers) {
  if (answers.rewrite === "Yes — text only" && answers.schema === "No") return "A";
  if (answers.schema === "Yes") return "B";
  if (answers.family === "Yes") return "C";
  return null;
}

function RouteHelper() {
  const [answers, setAnswers] = React.useState({});
  const route = decideRoute(answers);
  return (
    <div>
      <p>Three quick questions decide A / B / C.</p>
      {ROUTE_QUESTIONS.map((q) => (
        <div key={q.id} className="question-block">
          <div className="q">{q.q}</div>
          <div className="opt-row">
            {q.opts.map((o) => (
              <button
                key={o}
                className={"opt" + (answers[q.id] === o ? " selected" : "")}
                onClick={() => setAnswers({ ...answers, [q.id]: o })}
              >
                {o}
              </button>
            ))}
          </div>
        </div>
      ))}
      <div style={{ marginTop: 22, paddingTop: 18, borderTop: "1px solid var(--line)" }}>
        <div style={{ fontSize: 12, fontFamily: "Geist Mono", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
          Suggestion
        </div>
        {ROUTES.map((r) => (
          <div key={r.key} className={"route-card" + (route === r.key ? " match" : "")}>
            <h3><span className="route-tag">Route {r.key}</span> {r.name}</h3>
            <p>{r.desc}</p>
          </div>
        ))}
        {!route && (
          <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 14 }}>
            Answer all three questions to see a recommendation. When in doubt for high-complexity interactive methods, prefer Route B — more work, clearer contract, safer PR.
          </p>
        )}
      </div>
    </div>
  );
}

function PRTemplate() {
  const [f, setF] = React.useState({
    method: "",
    type: "interactive",
    complexity: "medium",
    selector: "explicit-only",
    schema: "",
    agent: "",
    prompt: "",
    route: "B",
    risks: "",
    tests: { schema: false, factory: false, node: false, ruff: false, promptfoo: false, live: false, pipeline: false },
  });
  const set = (k, v) => setF({ ...f, [k]: v });
  const setTest = (k, v) => setF({ ...f, tests: { ...f.tests, [k]: v } });
  const slug = f.method.trim().toLowerCase().replace(/\s+/g, "_") || "<method_slug>";
  const checkbox = (b) => (b ? "[x]" : "[ ]");

  const md = `## Summary

Adds \`${f.method.toUpperCase() || "<METHOD>"}\` as a \`${f.type}/${f.complexity}\` presentation method.

Includes:
- schema: \`${f.schema || `Presentation${slug}Schema`}\`
- agent: \`presentation_generator_${slug}\`
- prompt block: \`${f.prompt || `${slug}_generator`}\`
- node routing: Route ${f.route}
- docs: \`docs/presentation-methods/${slug}.md\`
- eval: \`tests/evals/presentation_generator_${slug}.yaml\`

## Design Notes

- Auto-selection: ${f.selector}
- Route: ${ROUTES.find((r) => r.key === f.route)?.name || ""}

## Validation

- ${checkbox(f.tests.schema)} Targeted schema pytest
- ${checkbox(f.tests.factory)} Targeted factory pytest
- ${checkbox(f.tests.node)} Targeted node-routing pytest
- ${checkbox(f.tests.ruff)} Ruff on changed Python files
- ${checkbox(f.tests.promptfoo)} Promptfoo canary
- ${checkbox(f.tests.live)} Live generation scenarios
- ${checkbox(f.tests.pipeline)} Full pipeline run

## Known Risks / Follow-ups

${f.risks || "- (none)"}
`;

  return (
    <div>
      <p>Fill in what applies — the markdown updates as you type. Copy-paste into GitHub.</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div>
          <label>Method name</label>
          <input type="text" value={f.method} onChange={(e) => set("method", e.target.value)} placeholder="e.g. flashcards" />
        </div>
        <div>
          <label>Route</label>
          <select value={f.route} onChange={(e) => set("route", e.target.value)}>
            <option value="A">A — Inline text</option>
            <option value="B">B — Dedicated generator</option>
            <option value="C">C — Unified agent</option>
          </select>
        </div>
        <div>
          <label>Type</label>
          <select value={f.type} onChange={(e) => set("type", e.target.value)}>
            <option>text</option><option>interactive</option><option>visual</option><option>audio</option><option>video</option>
          </select>
        </div>
        <div>
          <label>Complexity</label>
          <select value={f.complexity} onChange={(e) => set("complexity", e.target.value)}>
            <option>low</option><option>medium</option><option>high</option>
          </select>
        </div>
        <div>
          <label>Selector</label>
          <select value={f.selector} onChange={(e) => set("selector", e.target.value)}>
            <option>explicit-only</option><option>auto-selectable</option>
          </select>
        </div>
      </div>
      <label>Tests run</label>
      <div className="opt-row" style={{ marginTop: 4 }}>
        {Object.entries({ schema: "Schema", factory: "Factory", node: "Node", ruff: "Ruff", promptfoo: "Promptfoo", live: "Live", pipeline: "Pipeline" }).map(([k, v]) => (
          <button key={k} className={"opt" + (f.tests[k] ? " selected" : "")} onClick={() => setTest(k, !f.tests[k])}>{v}</button>
        ))}
      </div>
      <label>Known risks / follow-ups (one per line, with leading "- ")</label>
      <textarea value={f.risks} onChange={(e) => set("risks", e.target.value)} placeholder="- frontend submission path unresolved&#10;- answer-key exposure pending review" />
      <label>Generated PR description</label>
      <CodeBlock code={md} label="copy into github" />
    </div>
  );
}

function HelpTemplate() {
  const [f, setF] = React.useState({ task: "", branch: "", check: "", error: "", files: "", tried: "" });
  const set = (k, v) => setF({ ...f, [k]: v });
  const md = `task: ${f.task || "<task name / link>"}
branch/PR: ${f.branch || "<branch or PR #>"}
failing check: ${f.check || "<which CI step / which review comment>"}
error message:
${f.error || "<paste the first real error>"}
files changed:
${f.files || "<short list>"}
what I already tried:
${f.tried || "<bullets>"}`;

  return (
    <div>
      <p>Format your ask so a teammate can help in &lt; 2 minutes. Fill any fields you have — the others stay as placeholders.</p>
      <label>Task</label>
      <input type="text" value={f.task} onChange={(e) => set("task", e.target.value)} placeholder="Add FLASHCARDS presentation method" />
      <label>Branch / PR</label>
      <input type="text" value={f.branch} onChange={(e) => set("branch", e.target.value)} placeholder="feature/presentation-flashcards-generator or PR #142" />
      <label>Failing check</label>
      <input type="text" value={f.check} onChange={(e) => set("check", e.target.value)} placeholder="pytest tests/agno_agents/... factory test" />
      <label>Error message</label>
      <textarea value={f.error} onChange={(e) => set("error", e.target.value)} placeholder="Paste the FIRST real error, not the last line" />
      <label>Files changed</label>
      <textarea value={f.files} onChange={(e) => set("files", e.target.value)} style={{ minHeight: 60 }} placeholder="schemas/presentation.py, agent_registry.py, ..." />
      <label>What I already tried</label>
      <textarea value={f.tried} onChange={(e) => set("tried", e.target.value)} style={{ minHeight: 60 }} placeholder="- ran ruff, clean&#10;- regenerated lockfile" />
      <label>Message to send</label>
      <CodeBlock code={md} label="copy & paste" />
    </div>
  );
}

const PITFALLS = [
  "Adding a schema but not registering the agent in agent_registry.py",
  "Adding a factory but not adding the model tier in models.py",
  "Updating docs but forgetting node routing",
  "Letting the selector auto-pick a method before the frontend supports it",
  "Treating an interactive method as 'done' when answer submission, scoring, or retries are unresolved",
  "Exposing expected answers without a PR caveat",
  "Promptfoo passes a canary but no pytest covers schema/node contracts",
  "Promptfoo passes while the Agno response model fails under OpenAI strict structured outputs",
  "Running live tests but not recording which scenario was tested",
  "Claiming full-pipeline validation when only direct agent generation was run",
  "Ignoring non-English language behavior",
  "Forgetting source-material prompt-injection guardrails",
  "Using raw dicts where a Pydantic schema should define the contract",
  "Modifying broad architecture to support one method when a local route is enough",
];

function Pitfalls() {
  return (
    <div>
      <p>Scan before requesting review. If any of these sound like you — fix it first.</p>
      <div className="pitfalls">
        {PITFALLS.map((p, i) => (<div key={i} className="pitfall">{p}</div>))}
      </div>
    </div>
  );
}

function CodeBlock({ code, label }) {
  const [copied, setCopied] = React.useState(false);
  const onCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  };
  return (
    <div className="code-block">
      {label && <div className="label">{label}</div>}
      <pre>{code}</pre>
      <button className={"copy" + (copied ? " copied" : "")} onClick={onCopy}>
        {copied ? "✓ copied" : "copy"}
      </button>
    </div>
  );
}

window.RouteHelper = RouteHelper;
window.PRTemplate = PRTemplate;
window.HelpTemplate = HelpTemplate;
window.Pitfalls = Pitfalls;
window.CodeBlock = CodeBlock;
