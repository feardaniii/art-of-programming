# MCP (Model Context Protocol) - Claude Code Production Setup Guide

> Tested on macOS (Darwin). Last updated: February 2025.
> Based on real production setup across multiple projects.

---

## TL;DR

```bash
# Global (all projects)
claude mcp add --scope user --transport http <name> "<url>"
claude mcp add --scope user --transport stdio <name> -- npx -y <package>

# Per-project — private to you (stored in ~/.claude.json under project path)
claude mcp add --transport http <name> "<url>"

# Per-project — shared with team (stored in <project>/.mcp.json, git-trackable)
claude mcp add --scope project --transport http <name> "<url>"
```

Always quote URLs containing `?` or `&` — zsh interprets them as glob patterns.

---

## 1. The Three Scopes Explained

### User Scope (Global)
- **Flag**: `--scope user`
- **Stored in**: `~/.claude.json` → top-level `mcpServers` object
- **Available**: In every project you open with Claude Code
- **Use for**: Ref, Context7, Playwright — tools you always want

### Local Scope (Per-Project, Private)
- **Flag**: none (default)
- **Stored in**: `~/.claude.json` → `projects.<full-path>.mcpServers`
- **Available**: Only when Claude Code is opened from that specific directory
- **Use for**: Supabase (each project has different `project_ref`)

**Important**: When you run `claude mcp add` without `--scope`, it writes to `~/.claude.json` but *scoped to the current project path*. This is what the output means:
```
Added HTTP MCP server supabase with URL: https://...
File modified: /Users/neo/.claude.json [project: /Users/neo/Neo/gymcam/front-end]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        Stored globally, but scoped to this path
```

### Project Scope (Per-Project, Shared via Git)
- **Flag**: `--scope project`
- **Stored in**: `<project-root>/.mcp.json` (file in your repo)
- **Available**: Anyone who clones the repo (each person must authenticate separately)
- **Use for**: Team-shared Supabase, shared tooling configs

**First-time approval**: When a project has `.mcp.json`, Claude Code prompts you to approve those servers. This approval is tracked in `settings.local.json` via `enabledMcpjsonServers`.

---

## 2. File Paths on macOS (Complete Map)

### Single source of truth for MCP config
```
~/.claude.json
├── mcpServers: { ... }                    ← USER scope (global servers)
└── projects:
    ├── "/full/path/to/project-a":
    │   ├── mcpServers: { ... }            ← LOCAL scope (project-a only)
    │   └── enabledMcpjsonServers: [...]   ← Which .mcp.json servers are approved
    └── "/full/path/to/project-b":
        └── mcpServers: { ... }            ← LOCAL scope (project-b only)
```

### Claude Code directory
```
~/.claude/
├── CLAUDE.md                              ← Global instructions (loaded every session)
├── settings.json                          ← Global settings (theme, etc.)
└── projects/                              ← Auto-memory files per project
```

### Per-project files (in your repo)
```
<project-root>/
├── .mcp.json                              ← PROJECT scope servers (git-tracked)
└── .claude/
    ├── CLAUDE.md                          ← Project instructions (git-tracked)
    └── settings.local.json                ← Permissions, enabledMcpjsonServers
```

### Authentication storage
```
macOS Keychain                             ← OAuth tokens (Supabase, GitHub, etc.)
                                              Automatic — no keys in config files
```

### Cloud-linked (automatic)
```
claude.ai Account                          ← Zapier and other claude.ai MCPs
                                              Configured at claude.ai, always available
```

### Enterprise (admin-deployed, requires sudo)
```
/Library/Application Support/ClaudeCode/managed-mcp.json
```

### NOT used (common mistake — delete if present)
```
~/.claude/mcp.json                         ← IGNORED by Claude Code. Does nothing.
```

---

## 3. Setup: Global MCP Servers

Run these ONCE from any directory. Available in every project forever.

### Ref (Documentation search)
```bash
claude mcp add --transport http --scope user Ref \
  "https://api.ref.tools/mcp?apiKey=YOUR_API_KEY"
```
Get API key at https://ref.tools

### Context7 (Library documentation)
```bash
claude mcp add --transport stdio --scope user context7 \
  -- npx -y @upstash/context7-mcp@latest
```

### Playwright (Browser automation/testing)
```bash
claude mcp add --transport stdio --scope user playwright \
  -- npx -y @playwright/mcp@latest
```

### Verify
```bash
claude mcp list
# Should show all 3 as ✓ Connected
```

### What this creates in ~/.claude.json
```json
{
  "mcpServers": {
    "Ref": {
      "type": "http",
      "url": "https://api.ref.tools/mcp?apiKey=ref-xxxxx"
    },
    "context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "env": {}
    },
    "playwright": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"],
      "env": {}
    }
  }
}
```

---

## 4. Setup: Per-Project Supabase MCP

Supabase must be per-project — each project has its own `project_ref` and separate OAuth session.

### Step 1: Find your Supabase project ref
- https://supabase.com/dashboard → Select project → Settings → General → **Reference ID**
- Also in URL: `https://supabase.com/dashboard/project/THIS_IS_YOUR_REF`

### Step 2: Choose your approach

#### Option A: Local scope (private to you — recommended for solo work)
```bash
cd /path/to/your/project
claude mcp add --transport http supabase \
  "https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF"
```

**Where it's stored**: `~/.claude.json` under `projects."/path/to/your/project".mcpServers`

**Real example** — after running this for gymcam:
```json
// Inside ~/.claude.json
{
  "projects": {
    "/Users/neo/Neo/gymcam/front-end": {
      "mcpServers": {
        "supabase": {
          "type": "http",
          "url": "https://mcp.supabase.com/mcp?project_ref=ncftcrhvxoipfidymjpc"
        }
      }
    }
  }
}
```

#### Option B: Project scope (shared with team via git)
```bash
cd /path/to/your/project
claude mcp add --transport http --scope project supabase \
  "https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF"
```

**Where it's stored**: `<project-root>/.mcp.json`

**Real example** — acasa-terapie uses this approach:
```json
// File: /Users/neo/Neo/acasa-terapie/acasa-terapie-conexiune/.mcp.json
{
  "mcpServers": {
    "supabase": {
      "type": "http",
      "url": "https://mcp.supabase.com/mcp?project_ref=rxcbvobdkjxlialvodik"
    }
  }
}
```

**First-time approval flow**: When Claude Code sees a `.mcp.json`, it asks you to approve/enable those servers. Your choice is saved in `.claude/settings.local.json`:
```json
{
  "enableAllProjectMcpServers": true,
  "enabledMcpjsonServers": ["supabase"]
}
```

### Step 3: Authenticate (required for both options)
```
claude                  # Start Claude Code in the project
/mcp                    # Type this inside Claude Code
→ Select supabase
→ Authenticate
→ Browser opens → Log into Supabase → Done
```

No API key stored anywhere. OAuth token lives in **macOS Keychain** automatically.

---

## 5. Local vs Project Scope: When to Use Which

| Scenario | Use | Why |
|----------|-----|-----|
| Solo developer, one machine | **Local** (default) | Simpler, no file in repo |
| Team project, shared config | **Project** (`--scope project`) | `.mcp.json` in git, everyone gets it |
| Both (current acasa-terapie setup) | **Project** for supabase | Team can clone and just `/mcp` authenticate |
| Sensitive server with API key | **Local** (default) | Never committed to git |

**Can you use both?** Yes. If the same server name exists in both local and project scope, **local wins** (local overrides project, project overrides user).

---

## 6. Real-World: Complete New Project Setup

```bash
# ──────────────────────────────────────────────
# ONE-TIME GLOBAL SETUP (skip if already done)
# ──────────────────────────────────────────────
claude mcp add --transport http --scope user Ref \
  "https://api.ref.tools/mcp?apiKey=YOUR_REF_API_KEY"

claude mcp add --transport stdio --scope user context7 \
  -- npx -y @upstash/context7-mcp@latest

claude mcp add --transport stdio --scope user playwright \
  -- npx -y @playwright/mcp@latest

# ──────────────────────────────────────────────
# PER-PROJECT SETUP (repeat for each project)
# ──────────────────────────────────────────────
cd /path/to/new/project

# Add Supabase (local scope — just for you)
claude mcp add --transport http supabase \
  "https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF"

# OR: Add Supabase (project scope — shared via .mcp.json)
claude mcp add --transport http --scope project supabase \
  "https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF"

# Start Claude Code and authenticate
claude
# Inside: /mcp → supabase → Authenticate → Browser login
```

**Result** — `/mcp` shows:
```
Ref              ✓ Connected    (user scope — global)
context7         ✓ Connected    (user scope — global)
playwright       ✓ Connected    (user scope — global)
supabase         ✓ Connected    (local/project scope — this project only)
claude.ai Zapier ✓ Connected    (claude.ai account — automatic)
```

---

## 7. Replicating on a New Machine

### Step 1: Install Claude Code
```bash
# https://code.claude.com — follow official install for your platform
```

### Step 2: Global MCP servers (3 commands)
```bash
claude mcp add --transport http --scope user Ref \
  "https://api.ref.tools/mcp?apiKey=YOUR_REF_API_KEY"

claude mcp add --transport stdio --scope user context7 \
  -- npx -y @upstash/context7-mcp@latest

claude mcp add --transport stdio --scope user playwright \
  -- npx -y @playwright/mcp@latest
```

### Step 3: Copy global instructions
```bash
mkdir -p ~/.claude
# Copy CLAUDE.md from old machine, USB, cloud backup, etc.
cp /path/to/backup/CLAUDE.md ~/.claude/CLAUDE.md
```

### Step 4: Per-project Supabase (for each project)
```bash
cd /path/to/project

# If the project has .mcp.json (project scope) — just authenticate:
claude
# /mcp → supabase → Authenticate

# If the project uses local scope — add it first:
claude mcp add --transport http supabase \
  "https://mcp.supabase.com/mcp?project_ref=PROJECT_REF"
claude
# /mcp → supabase → Authenticate
```

### Step 5: Verify everything
```bash
claude mcp list
# Should show all global + local servers as ✓ Connected
```

---

## 8. Managing MCP Servers

```bash
# ── List & Inspect ──
claude mcp list                          # All servers with status
claude mcp get supabase                  # Details for one server

# ── Remove ──
claude mcp remove Ref                    # Remove from default (local) scope
claude mcp remove Ref -s user            # Remove from user (global) scope
claude mcp remove supabase -s project    # Remove from project scope (.mcp.json)

# ── Inside Claude Code ──
/mcp                                     # Check status, authenticate, manage

# ── Reset ──
claude mcp reset-project-choices         # Reset .mcp.json approval dialogs
```

---

## 9. Common Gotchas

| Problem | Cause | Fix |
|---------|-------|-----|
| `zsh: no matches found` | Unquoted URL with `?` or `&` | Quote the URL: `"https://...?param=value"` |
| Edited `~/.claude/mcp.json` but nothing loads | Wrong file — Claude Code ignores it | Use `claude mcp add` (writes to `~/.claude.json`) |
| MCP not showing after `claude mcp add` | Servers load at startup | Restart Claude Code |
| Supabase stops working mid-session | OAuth token expired | `/mcp` → supabase → Authenticate |
| `.mcp.json` servers not loading | Need first-time approval | Claude Code prompts on first session, or set `enableAllProjectMcpServers: true` in `settings.local.json` |
| Same server name in multiple scopes | Scope precedence applies | Local > Project > User (local always wins) |
| Server shows in one project but not another | It's local-scoped to that project path | Add with `--scope user` for global, or add separately per project |
| `enabledMcpjsonServers` is empty but servers work | `enableAllProjectMcpServers: true` overrides | Both approaches work — either enable-all or list specific servers |

---

## 10. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  ~/.claude.json  (SINGLE SOURCE OF TRUTH)                       │
│                                                                 │
│  mcpServers: {                         ← USER SCOPE (GLOBAL)    │
│    "Ref":        { http, ref.tools }                            │
│    "context7":   { stdio, npx ... }                             │
│    "playwright": { stdio, npx ... }                             │
│  }                                                              │
│                                                                 │
│  projects: {                           ← LOCAL SCOPE (PER-PATH) │
│    "/Users/neo/.../acasa-terapie-conexiune": {                  │
│      mcpServers: {                                              │
│        "playwright": { stdio, npx ... }  ← old local duplicate  │
│      }                                                          │
│    },                                                           │
│    "/Users/neo/.../gymcam/front-end": {                         │
│      mcpServers: {                                              │
│        "supabase": { http, project_ref=ncftcr... }              │
│      }                                                          │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  <project-root>/.mcp.json              ← PROJECT SCOPE          │
│                                          (git-tracked, shared)  │
│  Example: acasa-terapie-conexiune/.mcp.json                     │
│  {                                                              │
│    "mcpServers": {                                              │
│      "supabase": { http, project_ref=rxcbvo... }                │
│    }                                                            │
│  }                                                              │
│                                                                 │
│  Requires approval on first use.                                │
│  Each teammate authenticates via /mcp independently.            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  <project-root>/.claude/settings.local.json                     │
│                                                                 │
│  Controls which .mcp.json servers are enabled:                  │
│  {                                                              │
│    "enableAllProjectMcpServers": true,       ← approve all      │
│    "enabledMcpjsonServers": ["supabase"]     ← or list specific │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  macOS Keychain                        ← AUTH TOKENS             │
│                                                                 │
│  OAuth tokens for Supabase, GitHub, etc.                        │
│  No API keys in config files for OAuth servers.                 │
│  Managed automatically by Claude Code.                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  claude.ai Account                     ← CLOUD-LINKED MCPs      │
│                                                                 │
│  Zapier and other claude.ai integrations.                       │
│  Configured at claude.ai website, always available.             │
└─────────────────────────────────────────────────────────────────┘


SCOPE PRECEDENCE (when same server name exists in multiple):

  LOCAL (project path in ~/.claude.json)
    ↓ overrides
  PROJECT (.mcp.json in repo)
    ↓ overrides
  USER (top-level in ~/.claude.json)
```

---

## 11. Quick Reference Card

| Action | Command |
|--------|---------|
| **Add global HTTP** | `claude mcp add --scope user --transport http <name> "<url>"` |
| **Add global stdio** | `claude mcp add --scope user --transport stdio <name> -- npx -y <pkg>` |
| **Add per-project (private)** | `claude mcp add --transport http <name> "<url>"` |
| **Add per-project (shared)** | `claude mcp add --scope project --transport http <name> "<url>"` |
| **List servers** | `claude mcp list` |
| **Server details** | `claude mcp get <name>` |
| **Remove (local)** | `claude mcp remove <name>` |
| **Remove (global)** | `claude mcp remove <name> -s user` |
| **Remove (project)** | `claude mcp remove <name> -s project` |
| **Status (in Claude)** | `/mcp` |
| **Authenticate** | `/mcp` → Select → Authenticate |
| **Reset approvals** | `claude mcp reset-project-choices` |
| **Import from Desktop** | `claude mcp add-from-claude-desktop` |
| **Set timeout** | `MCP_TIMEOUT=10000 claude` |
| **Set max output** | `MAX_MCP_OUTPUT_TOKENS=50000 claude` |
