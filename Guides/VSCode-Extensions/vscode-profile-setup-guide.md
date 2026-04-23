# VS Code Profile Setup Guide

Use this guide to create a clean VS Code profile for development without losing your old setup.

## What Is a VS Code Profile?

A VS Code profile is a separate editor setup. It can have its own:

- Settings
- Extensions
- Keyboard shortcuts
- Snippets
- Tasks
- UI layout

This means you can keep your old setup and create a new recommended setup for a project or course.

## 1. Save Your Old Setup

1. Open VS Code.
2. Open the Command Palette:
   - Windows/Linux: `Ctrl+Shift+P`
   - macOS: `Cmd+Shift+P`
3. Run `Profiles: Create Profile`.
4. Choose `Create from Current Profile`.
5. Name the copied profile something clear, for example:
   - `Old Setup`
   - `Personal Setup`
   - `Before Course`
6. Create it.

Optional backup:

1. Open the Profiles menu from the gear icon in the lower-left corner.
2. Choose the profile you just created.
3. Select `Export Profile`.
4. Save it as a `.code-profile` file.

## 2. Create a New Profile

1. Open the Command Palette.
2. Run `Profiles: Create Profile`.
3. Choose `Create Empty Profile` for a clean start.
4. Name it, for example:
   - `Frontend Development`
   - `Course Setup`
   - `Project Setup`
5. Select the new profile for the current window.

You can switch profiles later with:

- Command Palette: `Profiles: Switch Profile`
- Gear icon: `Profiles`

## 3. Install Recommended Extensions

Install these from the Extensions panel with `Ctrl+Shift+X` / `Cmd+Shift+X`.

Required:

| Extension | ID | Why |
| --- | --- | --- |
| Prettier - Code formatter | `esbenp.prettier-vscode` | Formats code consistently |
| ESLint | `dbaeumer.vscode-eslint` | Finds and fixes JavaScript/TypeScript problems |

Recommended:

| Extension | ID | Why |
| --- | --- | --- |
| Python | `ms-python.python` | Official Microsoft support for Python editing, debugging, environments, testing, and language tooling |
| GitLens | `eamodio.gitlens` | Better Git history, blame, and file insights |
| GitHub Pull Requests | `GitHub.vscode-pull-request-github` | Work with PRs and issues in VS Code |
| Path Intellisense | `christian-kohler.path-intellisense` | Autocomplete file paths in imports |
| Auto Rename Tag | `formulahendry.auto-rename-tag` | Rename matching HTML/JSX tags automatically |
| Material Icon Theme | `PKief.material-icon-theme` | Clearer file icons |
| Thunder Client | `rangav.vscode-thunder-client` | Test REST APIs inside VS Code |
| Live Share | `ms-vsliveshare.vsliveshare` | Pair programming and shared debugging |

Usually skip `Rainbow Brackets`: VS Code already has built-in bracket pair colorization.

## 4. Add Recommended Settings

Open settings as JSON:

1. Open the Command Palette.
2. Run `Preferences: Open User Settings (JSON)`.
3. Paste or merge this configuration into the new profile.

```json
{
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "explicit"
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],
  "git.autofetch": true,
  "explorer.confirmDelete": true,
  "explorer.confirmDragAndDrop": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.eol": "\n",
  "workbench.iconTheme": "material-icon-theme",

  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[javascriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[jsonc]": {
    "editor.defaultFormatter": "vscode.json-language-features"
  },
  "[html]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[css]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[scss]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[markdown]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.wordWrap": "on"
  }
}
```

Why these settings matter:

- `editor.formatOnSave`: formats the file every time you save.
- `editor.defaultFormatter`: makes Prettier the formatter.
- `source.fixAll.eslint`: fixes ESLint problems when possible.
- `source.organizeImports`: cleans unused or unordered imports when supported.
- Language blocks like `[typescriptreact]`: make the formatter explicit per file type.

## 5. Check the Project Config Files

Most real projects define their own rules. Look for these files in the project root:

- `.prettierrc`
- `.prettierrc.json`
- `prettier.config.js`
- `.eslintrc`
- `.eslintrc.json`
- `eslint.config.js`

If they exist, do not copy random formatting rules from the internet. Let the project config decide the final style.

## 6. Test the Setup

1. Open a JavaScript, TypeScript, HTML, CSS, or Markdown file.
2. Make spacing intentionally messy.
3. Save the file.
4. Confirm that Prettier formats it.
5. Add a simple ESLint problem, then save.
6. Confirm that ESLint fixes it or shows a clear warning.

Useful commands:

- `Format Document`
- `ESLint: Fix all auto-fixable Problems`
- `Developer: Reload Window`

## 7. Common Problems

### Format on save does nothing

Check:

- Prettier extension is installed in the active profile.
- The active file type has Prettier selected as formatter.
- The project has dependencies installed with `npm install`, `pnpm install`, or `yarn install`.

### ESLint does nothing

Check:

- ESLint extension is installed in the active profile.
- The project has ESLint installed.
- The project has an ESLint config file.
- VS Code status bar does not show ESLint errors.

### The wrong profile is active

Check the profile name:

- In the VS Code title bar
- On the gear icon in the lower-left corner
- In the Profiles editor from the gear icon

## 8. Optional: Share the Profile

After the setup is ready:

1. Open the Profiles menu.
2. Select the new profile.
3. Choose `Export Profile`.
4. Save it as a `.code-profile` file or export it as a GitHub gist.

Others can import it with `Profiles: Import Profile`.

## References

- VS Code Profiles: <https://code.visualstudio.com/docs/configure/profiles>
- VS Code Formatting: <https://code.visualstudio.com/docs/editing/codebasics#_formatting>
- VS Code Code Actions on Save: <https://code.visualstudio.com/docs/editing/refactoring#_code-actions-on-save>
- Python extension: <https://marketplace.visualstudio.com/items?itemName=ms-python.python>
- Prettier extension: <https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode>
- ESLint extension: <https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint>
- Live Share extension: <https://marketplace.visualstudio.com/items?itemName=ms-vsliveshare.vsliveshare>
