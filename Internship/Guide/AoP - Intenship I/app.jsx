// app.jsx — main app shell

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "dark": false,
  "density": "regular",
  "accent": "#3a6dd1",
  "showDeeper": true
}/*EDITMODE-END*/;

// Each accent: hex (used as the swatch in TweakColor) → resolved oklch tokens.
const ACCENTS = {
  "#3a6dd1": { bg: "oklch(0.55 0.14 252)", soft: "oklch(0.55 0.14 252 / 0.10)", ink: "oklch(0.40 0.14 252)", inkDark: "oklch(0.85 0.10 252)" },
  "#3e8c5e": { bg: "oklch(0.55 0.13 150)", soft: "oklch(0.55 0.13 150 / 0.10)", ink: "oklch(0.40 0.13 150)", inkDark: "oklch(0.85 0.10 150)" },
  "#c46a3a": { bg: "oklch(0.58 0.15 40)",  soft: "oklch(0.58 0.15 40 / 0.10)",  ink: "oklch(0.42 0.15 40)",  inkDark: "oklch(0.85 0.10 40)" },
  "#8a5cd4": { bg: "oklch(0.55 0.16 295)", soft: "oklch(0.55 0.16 295 / 0.10)", ink: "oklch(0.42 0.16 295)", inkDark: "oklch(0.85 0.10 295)" },
};

const TOTAL_ITEMS = STEPS.reduce((s, st) => s + st.items.length, 0);

function storageKey(prName) {
  return "prc::" + (prName || "_default");
}

function loadState(prName) {
  try {
    const raw = localStorage.getItem(storageKey(prName));
    if (!raw) return {};
    return JSON.parse(raw);
  } catch { return {}; }
}
function saveState(prName, state) {
  try { localStorage.setItem(storageKey(prName), JSON.stringify(state)); } catch {}
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [prName, setPrName] = React.useState(() => localStorage.getItem("prc::current") || "");
  const [checks, setChecks] = React.useState(() => loadState(prName).checks || {});
  const [openSteps, setOpenSteps] = React.useState(() => loadState(prName).open || { [STEPS[0].id]: true });
  const [modal, setModal] = React.useState(null);

  // apply tweaks to document
  React.useEffect(() => {
    document.documentElement.dataset.theme = t.dark ? "dark" : "light";
    document.documentElement.dataset.density = t.density;
    const a = ACCENTS[t.accent] || ACCENTS["#3a6dd1"];
    const root = document.documentElement.style;
    root.setProperty("--accent", a.bg);
    root.setProperty("--accent-soft", a.soft);
    root.setProperty("--accent-ink", t.dark ? a.inkDark : a.ink);
  }, [t.dark, t.density, t.accent]);

  // persist on change
  React.useEffect(() => {
    saveState(prName, { checks, open: openSteps });
    if (prName) localStorage.setItem("prc::current", prName);
  }, [checks, openSteps, prName]);

  // when prName changes, reload its state
  const onChangePR = (name) => {
    setPrName(name);
    const s = loadState(name);
    setChecks(s.checks || {});
    setOpenSteps(s.open || { [STEPS[0].id]: true });
  };

  const totalChecked = Object.values(checks).filter(Boolean).length;
  const pct = Math.round((totalChecked / TOTAL_ITEMS) * 100);

  const stepCount = (sid) => {
    const step = STEPS.find((s) => s.id === sid);
    let done = 0;
    step.items.forEach((_, i) => { if (checks[`${sid}:${i}`]) done++; });
    return { done, total: step.items.length };
  };

  const toggleStep = (sid) => setOpenSteps((s) => ({ ...s, [sid]: !s[sid] }));
  const toggleCheck = (sid, i) => setChecks((c) => ({ ...c, [`${sid}:${i}`]: !c[`${sid}:${i}`] }));

  const reset = () => {
    if (confirm("Reset all progress for this PR?")) {
      setChecks({});
    }
  };

  const exportProgress = () => {
    const lines = [`# PR Companion progress`, ``, `PR: ${prName || "(unnamed)"}`, `Done: ${totalChecked} / ${TOTAL_ITEMS} (${pct}%)`, ``];
    STEPS.forEach((s, idx) => {
      const c = stepCount(s.id);
      lines.push(`## ${idx + 1}. ${s.title}  (${c.done}/${c.total})`);
      s.items.forEach((item, i) => {
        const done = checks[`${s.id}:${i}`];
        lines.push(`- [${done ? "x" : " "}] ${item.replace(/`/g, "")}`);
      });
      lines.push("");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pr-progress-${(prName || "unnamed").replace(/\W+/g, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // progress ring
  const C = 2 * Math.PI * 11;
  const dash = (totalChecked / TOTAL_ITEMS) * C;

  return (
    <React.Fragment>
      {/* topbar */}
      <div className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="glyph">PR</div>
            <div>
              <div>PR Companion</div>
              <small>Presentation methods, demystified</small>
            </div>
          </div>
          <label className="pr-input" title="Progress is saved per PR name in your browser">
            <span className="lbl">PR</span>
            <input
              type="text"
              value={prName}
              onChange={(e) => onChangePR(e.target.value)}
              placeholder="branch or PR name (e.g. flashcards)"
            />
          </label>
          <div className="progress-cluster">
            <svg className="progress-ring" viewBox="0 0 28 28">
              <circle className="bg" cx="14" cy="14" r="11" />
              <circle
                className="fg"
                cx="14" cy="14" r="11"
                strokeDasharray={`${dash} ${C}`}
                transform="rotate(-90 14 14)"
              />
            </svg>
            <div className="progress-text">{totalChecked}/{TOTAL_ITEMS}</div>
          </div>
        </div>
      </div>

      <div className="shell">
        {/* hero */}
        <div className="hero">
          <div className="eyebrow">Internship · Presentation Method PRs</div>
          <h1>A friendly checklist for shipping your first PR.</h1>
          <p className="lede">
            Nine steps, lifted from the Quick PR Guide. Tick boxes as you go — your progress saves automatically, keyed to your PR name. You don't need to know the whole repo. Focus on your method, follow the path, and ask for help early.
          </p>
          <div className="hero-meta">
            <span><b>9 steps</b></span>
            <span><b>~30 min</b> first read</span>
            <span><b>Save progress per PR</b></span>
          </div>

          {/* tool strip */}
          <div className="toolstrip">
            <button className="tool-btn" onClick={() => setModal("route")}>
              <div className="t-title"><span className="t-glyph">A?</span> Route Helper</div>
              <div className="t-desc">A, B, or C? Three questions to find out.</div>
            </button>
            <button className="tool-btn" onClick={() => setModal("pr")}>
              <div className="t-title"><span className="t-glyph">PR</span> PR Description</div>
              <div className="t-desc">Generate a clean, complete PR write-up.</div>
            </button>
            <button className="tool-btn" onClick={() => setModal("help")}>
              <div className="t-title"><span className="t-glyph">SOS</span> Ask for Help</div>
              <div className="t-desc">Format an effective ask in 30 seconds.</div>
            </button>
            <button className="tool-btn" onClick={() => setModal("pitfalls")}>
              <div className="t-title"><span className="t-glyph">!</span> Common Pitfalls</div>
              <div className="t-desc">Scan before you request review.</div>
            </button>
          </div>
        </div>

        {/* steps */}
        <div className="steps">
          {STEPS.map((step, idx) => {
            const c = stepCount(step.id);
            const isOpen = !!openSteps[step.id];
            const complete = c.done === c.total && c.total > 0;
            return (
              <div key={step.id} className={"step" + (complete ? " complete" : "")} data-open={isOpen}>
                <div className="step-head" onClick={() => toggleStep(step.id)}>
                  <div className="step-num">{complete ? "✓" : (idx + 1).toString().padStart(2, "0")}</div>
                  <div>
                    <div className="step-title">{step.title}</div>
                    <div className="step-sub">{step.sub}</div>
                  </div>
                  <div className="step-meta">
                    <div className={"pill" + (complete ? " full" : "")}>{c.done}/{c.total}</div>
                    <div className="chev">▾</div>
                  </div>
                </div>
                {isOpen && (
                  <div className="step-body">
                    <div className="step-intro">{step.intro}</div>
                    {step.quote && <div className="step-quote">"{step.quote}"</div>}

                    <div className="checklist">
                      {step.items.map((item, i) => (
                        <label key={i} className="check">
                          <input
                            type="checkbox"
                            checked={!!checks[`${step.id}:${i}`]}
                            onChange={() => toggleCheck(step.id, i)}
                          />
                          <span className="check-text" dangerouslySetInnerHTML={{ __html: formatItem(item) }} />
                        </label>
                      ))}
                    </div>

                    {step.code && step.code.map((c, i) => (
                      <CodeBlock key={i} code={c.code} label={c.label} />
                    ))}

                    {t.showDeeper && step.deeper && (
                      <details className="deeper">
                        <summary>
                          {step.deeper.title}
                          <span className="deeper-tag">go deeper</span>
                        </summary>
                        <div>{step.deeper.body}</div>
                      </details>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* reminder */}
        <div className="reminder">
          <h3>Final reminder</h3>
          <ul>
            <li>You are not expected to know the entire repo.</li>
            <li>Make small changes.</li>
            <li>Compare with working examples.</li>
            <li>Use the AI checklist.</li>
            <li>Ask for help early.</li>
            <li>Keep the PR focused.</li>
          </ul>
          <p>A good PR is not perfect on the first try. A good PR is clear, scoped, and easy to review.</p>
        </div>

        <div className="footer-actions">
          <button className="btn primary" onClick={exportProgress}>Export progress (.md)</button>
          <button className="btn danger" onClick={reset}>Reset this PR</button>
        </div>
      </div>

      {/* modal */}
      {modal && (
        <div className="modal-bg" onClick={(e) => { if (e.target === e.currentTarget) setModal(null); }}>
          <div className="modal">
            <div className="modal-head">
              <h2>{
                modal === "route" ? "Route Helper" :
                modal === "pr" ? "PR Description Generator" :
                modal === "help" ? "Ask for Help — Template" :
                "Common Pitfalls"
              }</h2>
              <button className="close" onClick={() => setModal(null)}>×</button>
            </div>
            <div className="modal-body">
              {modal === "route" && <RouteHelper />}
              {modal === "pr" && <PRTemplate />}
              {modal === "help" && <HelpTemplate />}
              {modal === "pitfalls" && <Pitfalls />}
            </div>
          </div>
        </div>
      )}

      {/* tweaks */}
      <TweaksPanel>
        <TweakSection label="Appearance" />
        <TweakToggle label="Dark mode" value={t.dark} onChange={(v) => setTweak("dark", v)} />
        <TweakRadio label="Density" value={t.density} options={["compact", "regular", "comfy"]} onChange={(v) => setTweak("density", v)} />
        <TweakColor
          label="Accent"
          value={t.accent}
          options={Object.keys(ACCENTS)}
          onChange={(v) => setTweak("accent", v)}
        />
        <TweakSection label="Content" />
        <TweakToggle label="Show 'go deeper' panels" value={t.showDeeper} onChange={(v) => setTweak("showDeeper", v)} />
      </TweaksPanel>
    </React.Fragment>
  );
}

// inline-code formatter: wrap backticked spans in <code>
function formatItem(s) {
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/`([^`]+)`/g, '<code style="font-size: 0.86em; padding: 1px 5px; border-radius: 4px; background: var(--panel-2); border: 1px solid var(--line); color: var(--ink);">$1</code>');
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
