# Ghid de configurare profil VS Code

Folosește acest ghid pentru a crea un profil VS Code curat pentru development, fără să pierzi configurația veche.

## Ce este un profil VS Code?

Un profil VS Code este o configurație separată a editorului. Poate avea propriile:

- Setări
- Extensii
- Scurtături de tastatură
- Snippeturi
- Taskuri
- Layout al interfeței

Asta înseamnă că poți păstra setupul vechi și poți crea un setup nou, recomandat pentru un proiect sau curs.

## 1. Salvează setupul vechi

1. Deschide VS Code.
2. Deschide Command Palette:
   - Windows/Linux: `Ctrl+Shift+P`
   - macOS: `Cmd+Shift+P`
3. Rulează `Profiles: Create Profile`.
4. Alege `Create from Current Profile`.
5. Denumește copia profilului clar, de exemplu:
   - `Old Setup`
   - `Personal Setup`
   - `Before Course`
6. Creează profilul.

Backup opțional:

1. Deschide meniul Profiles din iconița de rotiță din colțul stânga-jos.
2. Alege profilul pe care tocmai l-ai creat.
3. Selectează `Export Profile`.
4. Salvează-l ca fișier `.code-profile`.

## 2. Creează un profil nou

1. Deschide Command Palette.
2. Rulează `Profiles: Create Profile`.
3. Alege `Create Empty Profile` pentru un început curat.
4. Denumește profilul, de exemplu:
   - `Frontend Development`
   - `Course Setup`
   - `Project Setup`
5. Selectează profilul nou pentru fereastra curentă.

Poți schimba profilul oricând folosind:

- Command Palette: `Profiles: Switch Profile`
- Iconița de rotiță: `Profiles`

## 3. Instalează extensiile recomandate

Instalează extensiile din panoul Extensions cu `Ctrl+Shift+X` / `Cmd+Shift+X`.

Obligatorii:

| Extensie | ID | De ce |
| --- | --- | --- |
| Prettier - Code formatter | `esbenp.prettier-vscode` | Formatează codul consecvent |
| ESLint | `dbaeumer.vscode-eslint` | Găsește și repară probleme în JavaScript/TypeScript |

Recomandate:

| Extensie | ID | De ce |
| --- | --- | --- |
| Python | `ms-python.python` | Suport oficial Microsoft pentru editare, debugging, medii, testare și tooling Python |
| GitLens | `eamodio.gitlens` | Istoric Git, blame și informații utile pe fișiere |
| GitHub Pull Requests | `GitHub.vscode-pull-request-github` | Lucru cu PR-uri și issues direct în VS Code |
| Path Intellisense | `christian-kohler.path-intellisense` | Autocomplete pentru căi de fișiere în importuri |
| Auto Rename Tag | `formulahendry.auto-rename-tag` | Redenumește automat tagurile pereche HTML/JSX |
| Material Icon Theme | `PKief.material-icon-theme` | Iconițe mai clare pentru fișiere |
| Thunder Client | `rangav.vscode-thunder-client` | Testare REST API direct în VS Code |
| Live Share | `ms-vsliveshare.vsliveshare` | Pair programming și debugging partajat |

De obicei poți sări peste `Rainbow Brackets`: VS Code are deja colorare integrată pentru perechi de paranteze.

## 4. Adaugă setările recomandate

Deschide setările în format JSON:

1. Deschide Command Palette.
2. Rulează `Preferences: Open User Settings (JSON)`.
3. Lipește sau combină această configurație în profilul nou.

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

De ce contează aceste setări:

- `editor.formatOnSave`: formatează fișierul la fiecare salvare.
- `editor.defaultFormatter`: setează Prettier ca formatter implicit.
- `source.fixAll.eslint`: repară automat problemele ESLint când este posibil.
- `source.organizeImports`: curăță importurile nefolosite sau neordonate, când limbajul suportă asta.
- Blocurile pe limbaj, precum `[typescriptreact]`: fac formatterul explicit pentru fiecare tip de fișier.

## 5. Verifică fișierele de configurare ale proiectului

Majoritatea proiectelor reale definesc propriile reguli. Caută aceste fișiere în rootul proiectului:

- `.prettierrc`
- `.prettierrc.json`
- `prettier.config.js`
- `.eslintrc`
- `.eslintrc.json`
- `eslint.config.js`

Dacă există, nu copia reguli de formatare la întâmplare de pe internet. Lasă configurația proiectului să decidă stilul final.

## 6. Testează setupul

1. Deschide un fișier JavaScript, TypeScript, HTML, CSS sau Markdown.
2. Strică intenționat spațierea.
3. Salvează fișierul.
4. Confirmă că Prettier îl formatează.
5. Adaugă o problemă simplă detectabilă de ESLint, apoi salvează.
6. Confirmă că ESLint o repară sau afișează un warning clar.

Comenzi utile:

- `Format Document`
- `ESLint: Fix all auto-fixable Problems`
- `Developer: Reload Window`

## 7. Probleme frecvente

### Format on save nu face nimic

Verifică:

- Extensia Prettier este instalată în profilul activ.
- Tipul de fișier activ are Prettier selectat ca formatter.
- Dependențele proiectului sunt instalate cu `npm install`, `pnpm install` sau `yarn install`.

### ESLint nu face nimic

Verifică:

- Extensia ESLint este instalată în profilul activ.
- Proiectul are ESLint instalat.
- Proiectul are fișier de configurare ESLint.
- Bara de status VS Code nu afișează erori ESLint.

### Este activ profilul greșit

Verifică numele profilului:

- În bara de titlu VS Code
- Pe iconița de rotiță din colțul stânga-jos
- În editorul Profiles deschis din iconița de rotiță

## 8. Opțional: partajează profilul

După ce setupul este gata:

1. Deschide meniul Profiles.
2. Selectează profilul nou.
3. Alege `Export Profile`.
4. Salvează-l ca fișier `.code-profile` sau exportă-l ca GitHub gist.

Alții îl pot importa cu `Profiles: Import Profile`.

## Referințe

- VS Code Profiles: <https://code.visualstudio.com/docs/configure/profiles>
- VS Code Formatting: <https://code.visualstudio.com/docs/editing/codebasics#_formatting>
- VS Code Code Actions on Save: <https://code.visualstudio.com/docs/editing/refactoring#_code-actions-on-save>
- Extensia Python: <https://marketplace.visualstudio.com/items?itemName=ms-python.python>
- Extensia Prettier: <https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode>
- Extensia ESLint: <https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint>
- Extensia Live Share: <https://marketplace.visualstudio.com/items?itemName=ms-vsliveshare.vsliveshare>
