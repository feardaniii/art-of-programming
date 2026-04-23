# Import `ZentR-Pro` in VS Code

## Fast Steps

1. Download `ZentR-Pro.code-profile`.
2. Open VS Code.
3. Open Command Palette:
   - Windows/Linux: `Ctrl+Shift+P`
   - macOS: `Cmd+Shift+P`
4. Run `Profiles: Import Profile`.
5. Select `ZentR-Pro.code-profile`.
6. Keep these checked:
   - `Settings`
   - `Extensions`
7. Click `Create Profile`.
8. Switch to `ZentR-Pro`.

## Confirm It Worked

1. Open Command Palette.
2. Run `Profiles: Show Contents`.
3. Confirm the active profile is `ZentR-Pro`.
4. Run `Extensions: Show Installed Extensions`.
5. Confirm `Python`, `Prettier`, `ESLint`, and `Live Share` are installed.

## Test Formatting

1. Open a `.js`, `.jsx`, `.ts`, `.tsx`, `.html`, `.css`, or `.md` file.
2. Break the spacing intentionally.
3. Save the file.
4. Confirm VS Code formats the file automatically.

If formatting does not run, open Command Palette and run `Format Document With...`, then choose `Prettier - Code formatter`.
