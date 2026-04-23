# Checklist pentru profilul VS Code `ZentR-Pro`

Verificare locală:

- Profil găsit: `ZentR-Pro`
- Folder profil: `~/.config/Code/User/profiles/143cce9c`
- Extensii instalate în profil acum: `github.copilot-chat`
- Lipsesc setările profilului: nu există `settings.json` în folderul profilului
- Profilul este fresh și trebuie configurat aproape de la zero

## 1. Instalează extensiile obligatorii

- [ ] `esbenp.prettier-vscode` - Prettier, formatter pentru cod
- [ ] `dbaeumer.vscode-eslint` - ESLint, linting și auto-fix pentru JS/TS

Comandă rapidă:

```bash
code --profile "ZentR-Pro" --install-extension esbenp.prettier-vscode
code --profile "ZentR-Pro" --install-extension dbaeumer.vscode-eslint
```

## 2. Instalează extensiile recomandate

- [ ] `eamodio.gitlens` - GitLens
- [ ] `GitHub.vscode-pull-request-github` - GitHub Pull Requests and Issues
- [ ] `ms-python.python` - Python, suport oficial Microsoft pentru Python
- [ ] `christian-kohler.path-intellisense` - Path Intellisense
- [ ] `formulahendry.auto-rename-tag` - Auto Rename Tag
- [ ] `PKief.material-icon-theme` - Material Icon Theme
- [ ] `rangav.vscode-thunder-client` - Thunder Client
- [ ] `ms-vsliveshare.vsliveshare` - Live Share

Comandă rapidă:

```bash
code --profile "ZentR-Pro" --install-extension eamodio.gitlens
code --profile "ZentR-Pro" --install-extension GitHub.vscode-pull-request-github
code --profile "ZentR-Pro" --install-extension ms-python.python
code --profile "ZentR-Pro" --install-extension christian-kohler.path-intellisense
code --profile "ZentR-Pro" --install-extension formulahendry.auto-rename-tag
code --profile "ZentR-Pro" --install-extension PKief.material-icon-theme
code --profile "ZentR-Pro" --install-extension rangav.vscode-thunder-client
code --profile "ZentR-Pro" --install-extension ms-vsliveshare.vsliveshare
```

## 3. Opțional, în funcție de proiect

- [ ] `ms-python.vscode-pylance` - completări și type checking pentru Python
- [ ] `ms-toolsai.jupyter` - doar dacă folosești notebookuri
- [ ] `ms-vscode-remote.remote-ssh` - doar dacă lucrezi pe servere remote prin SSH
- [ ] `ms-azuretools.vscode-containers` - doar dacă lucrezi cu Docker/containers
- [ ] `ritwickdey.liveserver` - util pentru HTML/CSS simplu, mai puțin necesar pentru React/Vite/Next

## 4. Adaugă setările recomandate

- [ ] Deschide profilul `ZentR-Pro`.
- [ ] Rulează `Preferences: Open User Settings (JSON)`.
- [ ] Adaugă setările din [ghidul în română](./vscode-profile-setup-guide-ro.md#4-adaugă-setările-recomandate).
- [ ] Salvează fișierul.
- [ ] Rulează `Developer: Reload Window`.

## 5. Verifică după instalare

Rulează:

```bash
code --profile "ZentR-Pro" --list-extensions
```

Trebuie să vezi cel puțin:

- [ ] `github.copilot-chat`
- [ ] `ms-python.python`
- [ ] `esbenp.prettier-vscode`
- [ ] `dbaeumer.vscode-eslint`

Pentru setupul recomandat complet, trebuie să vezi și:

- [ ] `eamodio.gitlens`
- [ ] `GitHub.vscode-pull-request-github`
- [ ] `christian-kohler.path-intellisense`
- [ ] `formulahendry.auto-rename-tag`
- [ ] `PKief.material-icon-theme`
- [ ] `rangav.vscode-thunder-client`
- [ ] `ms-vsliveshare.vsliveshare`

## 6. Test final

- [ ] Deschide un proiect cu `.prettierrc` sau `prettier.config.js`.
- [ ] Deschide un fișier `.js`, `.jsx`, `.ts`, `.tsx`, `.html`, `.css` sau `.md`.
- [ ] Strică intenționat formatarea.
- [ ] Salvează fișierul și confirmă că Prettier îl formatează.
- [ ] Rulează `ESLint: Fix all auto-fixable Problems` într-un proiect cu ESLint.
- [ ] Confirmă că profilul activ afișat în VS Code este `ZentR-Pro`.
