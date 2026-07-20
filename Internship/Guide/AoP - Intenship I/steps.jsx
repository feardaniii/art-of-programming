// steps.jsx — content for the 9-step PR guide

const STEPS = [
  {
    id: "start",
    title: "Start from the right place",
    sub: "Set up the branch, find your reference, scope down.",
    intro: "Before you write any code, get oriented. Pick the right base branch, find one similar method to model on, and resist touching anything outside your scope.",
    quote: "I am adding one method, not fixing the whole repo.",
    items: [
      "Created a feature branch based on `development`",
      "Read the task description carefully",
      "Checked `docs/presentation-methods/_REGISTRY.md` for my method",
      "Found one similar existing method to use as reference",
      "Confirmed I'm not touching unrelated files",
    ],
    deeper: {
      title: "Reference methods worth modelling on",
      body: (
        <div>
          <p>For interactive methods, look at how these examples are wired end-to-end:</p>
          <ul>
            <li><code>QUIZ</code> — answer-bearing, scored, validation</li>
            <li><code>EMPTY_CELLS</code> — stable item IDs, learner state</li>
            <li><code>CHECKLIST_TASK</code> — simpler interactive shape</li>
          </ul>
          <p>Recent merged PRs are usually the cleanest reference for current conventions — newer than the docs.</p>
        </div>
      ),
    },
    code: [
      { label: "branch", lang: "bash", code: "git checkout development\ngit pull\ngit checkout -b feature/presentation-<method-slug>-generator" },
    ],
  },
  {
    id: "route",
    title: "Know what kind of method you're building",
    sub: "Inline, dedicated generator, or unified agent?",
    intro: "Most methods are one of three shapes. Pick the route before you start writing — it determines which files you'll touch.",
    items: [
      "Identified my method's route: A (inline) / B (dedicated) / C (unified)",
      "If I'm unsure, I'll ask on Discord before building too much",
    ],
    deeper: {
      title: "Use the Route Helper",
      body: (
        <div>
          <p>Open the <strong>Route Helper</strong> tool at the top of the page — it walks through the questions that decide A vs B vs C.</p>
          <p>Quick rule: if your method needs a structured <code>generation_schema</code>, stable IDs, validation rules, answer data, categories, scoring, or special frontend payloads — it probably needs a <strong>dedicated generator (Route B)</strong>.</p>
        </div>
      ),
    },
  },
  {
    id: "align",
    title: "Keep schema, prompt, and output aligned",
    sub: "The biggest review issue, by far.",
    intro: "Your schema, your prompt, your agent's response model, and your docs all describe the same thing. If any one of them drifts, the agent fails — even when the code 'looks right'.",
    items: [
      "Schema defines the real output shape",
      "Prompt asks for the same output",
      "Agent response model expects the same output",
      "Docs show examples that match the real output",
      "For dedicated generators, the prompt asks for the full wrapper — not just `generation_schema`",
    ],
    deeper: {
      title: "Wrapper fields the prompt needs to ask for",
      body: (
        <div>
          <p>For dedicated generators, the agent returns a wrapper object — not just the inner schema. Make sure your prompt example shows the full wrapper:</p>
        </div>
      ),
    },
    code: [
      { label: "wrapper fields", lang: "text", code: "method\ntitle\nsections\ngeneration_schema\nis_valid\nissues" },
    ],
  },
  {
    id: "files",
    title: "Expected files for a dedicated generator",
    sub: "Scan this list before opening the PR.",
    intro: "If you registered a new agent, audit every registration / config / docs / test surface. The most common 'I forgot one' mistake lives here.",
    items: [
      "`src/common/schemas/presentation.py` — schema added",
      "`src/agno_agents/agents/presentation_generators.py` — wrapper + factory",
      "`src/agno_agents/agents/prompts/presentation_generator.yaml` — prompt block",
      "`src/agno_agents/agent_registry.py` — agent registered",
      "`src/agno_agents/models.py` — model tier mapped",
      "`src/graphs/course_generation/nodes/agent_nodes/presentation_generator.py` — node routes",
      "`tests/evals/baseline/model_config.json` — eval baseline entry",
      "`tests/evals/presentation_generator_<method>.yaml` — promptfoo canary",
      "`docs/AGENTS_REFERENCE.md` — agent listed",
      "`docs/presentation-methods/_REGISTRY.md` — registry updated",
      "`docs/presentation-methods/<method>.md` — method spec written",
      "Targeted pytest files for schema, factory, and node added",
    ],
    deeper: {
      title: "The classic forget-one mistake",
      body: (
        <p style={{margin: 0}}>"I added the agent in one place, but forgot model config, evals, docs, or node routing." — every reviewer, eventually. Use the checklist above as a literal pass before requesting review.</p>
      ),
    },
  },
  {
    id: "selector",
    title: "Be clear about selector behavior",
    sub: "Explicit-only or auto-selectable?",
    intro: "Every method should declare whether the selector can pick it automatically, or only when explicitly requested. For new or complex interactive methods, explicit-only is usually safer.",
    items: [
      "Decided: explicit-only / auto-selectable",
      "Documented the choice in the method doc",
      "Mentioned it in the PR description",
    ],
  },
  {
    id: "ai-checklist",
    title: "Use the AI PR Checklist before review",
    sub: "Plan first, edit later.",
    intro: "Before requesting human review, run your branch through an AI checklist. Three modes — pick whichever fits where you're working.",
    items: [
      "Ran an AI review of my PR plan against the checklist",
      "Asked for a list of likely blockers — not freeform edits",
      "If AI suggested touching unrelated files, I stopped and reviewed",
    ],
    quote: "Ask AI for a plan or diff first. Do not let it freely edit the repo.",
    code: [
      { label: "chat / web AI", lang: "text", code: "Review my PR plan using this checklist.\nOnly list likely blockers and missing files.\nDo not rewrite anything yet." },
      { label: "codex / vs code AI", lang: "text", code: "Compare my current branch against origin/development using the PR checklist.\nDo not edit files.\nList missing registrations, schema/prompt mismatches, test gaps, and unrelated files." },
      { label: "github cli (after PR is open)", lang: "text", code: "Use gh to inspect PR #XX comments, inline review comments, changed files, and failed checks.\nSummarize required fixes before editing anything." },
    ],
  },
  {
    id: "tests",
    title: "Test the important path",
    sub: "Prove the contract — not the whole repo.",
    intro: "You don't need to run everything. Prove the few things reviewers will check first.",
    items: [
      "The schema validates",
      "The factory builds",
      "The registry can find the agent",
      "The node routes correctly",
      "Conversion to `PresentationMethodOutput` works",
      "Eval / config files exist where required",
    ],
    code: [
      { label: "factory tests", lang: "bash", code: "uv run pytest tests/agno_agents/test_presentation_generator_<method>_factory.py -v" },
      { label: "node tests", lang: "bash", code: "uv run pytest tests/agno_agents/nodes/test_node_presentation_generator_<method>.py -v" },
    ],
    deeper: {
      title: "When tests fail",
      body: (
        <p style={{margin: 0}}>Read the <strong>first real error</strong> — not the last line of the stack trace. If you're stuck, ask for help with the exact output. Use the <strong>"Help me ask for help"</strong> tool to format your message.</p>
      ),
    },
  },
  {
    id: "pre-pr",
    title: "Before opening or updating the PR",
    sub: "One last sweep for stowaways.",
    intro: "Inspect what you're actually pushing. The goal is to catch unrelated files, generated files, logs, secrets, and any broad refactors that snuck in.",
    items: [
      "Ran `git status` and `git diff` to inspect changes",
      "No unrelated files",
      "No generated files / logs / traces / cache files",
      "No local notes, `.env`, or secrets",
      "No broad refactors I didn't intend",
      "PR description includes: method, route, docs/tests added, explicit-only vs auto-selectable, what tests I ran, anything intentionally left out",
    ],
    code: [
      { label: "inspect", lang: "bash", code: "git status --short --branch\ngit diff --name-status origin/development...HEAD\ngit diff --stat origin/development...HEAD" },
    ],
    deeper: {
      title: "Generate your PR description",
      body: (
        <p style={{margin: 0}}>Open the <strong>PR Description</strong> tool to fill in a template based on what you did. Copy-paste straight into GitHub.</p>
      ),
    },
  },
  {
    id: "if-fails",
    title: "If CI or review fails",
    sub: "Don't panic. Classify, then act.",
    intro: "Not every red check is something you broke. Sort the failure into a bucket before fixing anything.",
    items: [
      "Classified the issue: my code / repo baseline / tooling",
      "If it's my code → fixed it",
      "If it's existing repo baseline → mentioned it, didn't unilaterally fix",
      "If it's a tooling/API issue → asked for help instead of grinding for hours",
    ],
    deeper: {
      title: "When asking for help",
      body: (
        <p style={{margin: 0}}>Use the <strong>"Help me ask for help"</strong> tool — it formats the message in the way that makes it fastest for someone to actually help you.</p>
      ),
    },
  },
];

window.STEPS = STEPS;
