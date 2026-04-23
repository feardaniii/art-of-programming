# VS Code Profile Setup Guides

Beginner-friendly VS Code profile setup guides plus a ready-to-import `ZentR-Pro` profile.

## Files

- [ZentR-Pro.code-profile](./ZentR-Pro.code-profile) - ready VS Code profile export with recommended settings and extensions.
- [profile-import-quick-steps.md](./profile-import-quick-steps.md) - shortest import instructions.
- [vscode-profile-setup-guide.md](./vscode-profile-setup-guide.md) - English beginner guide.
- [vscode-profile-setup-guide-ro.md](./vscode-profile-setup-guide-ro.md) - Romanian guide with diacritics.
- [zentr-pro-checklist.md](./zentr-pro-checklist.md) - checklist for the fresh `ZentR-Pro` profile.

## Quick Import

1. Download [ZentR-Pro.code-profile](./ZentR-Pro.code-profile).
2. Open VS Code.
3. Press `Ctrl+Shift+P` / `Cmd+Shift+P`.
4. Run `Profiles: Import Profile`.
5. Select `ZentR-Pro.code-profile`.
6. Keep `Settings` and `Extensions` checked.
7. Click `Create Profile`.
8. Switch to `ZentR-Pro` when VS Code asks.

## Included Extensions

- GitHub Copilot Chat
- Python
- Prettier
- ESLint
- GitLens
- GitHub Pull Requests
- Path Intellisense
- Auto Rename Tag
- Material Icon Theme
- Thunder Client
- Live Share

## Included Settings

- Format on save
- Prettier as default formatter
- ESLint fix on save
- Organize imports on save
- Final newline and trailing whitespace cleanup
- Per-language formatter settings for JS, TS, React, JSON, HTML, CSS, SCSS, and Markdown

## Marketplace Links

- Python: <https://marketplace.visualstudio.com/items?itemName=ms-python.python>
- Live Share: <https://marketplace.visualstudio.com/items?itemName=ms-vsliveshare.vsliveshare>

## Verify

After import, open a JS/TS/HTML/CSS/Markdown file, make formatting messy, and save. Prettier should format the file. In a project with ESLint, run `ESLint: Fix all auto-fixable Problems` or save the file to confirm lint fixes.
