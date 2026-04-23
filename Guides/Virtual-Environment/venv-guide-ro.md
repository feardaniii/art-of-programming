# Medii virtuale Python: ghid rapid pentru predare

Scop: fiecare membru ar trebui să poată crea un venv pentru un proiect, să știe unde se află, să îl șteargă când se strică, să îl refacă din `requirements.txt` și să ruleze un script de test foarte simplu.

## 0. Ce este un venv?

Un mediu virtual este o instalare Python locală proiectului. Pachetele instalate în el rămân separate de Python-ul sistemului și de celelalte proiecte.

Regulă practică:

- Pune codul în folderul proiectului.
- Pune venv-ul în proiect ca `.venv/`.
- Commit la `requirements.txt`.
- Nu face commit la `.venv/`.

Adaugă asta în `.gitignore`:

```gitignore
.venv/
```

## 1. Creează un proiect demo mic

### Windows PowerShell

```powershell
mkdir venv-demo
cd venv-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Dacă PowerShell blochează activarea, rulează o singură dată:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS / Linux

```bash
mkdir venv-demo
cd venv-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Acum ar trebui să vezi `(.venv)` în promptul terminalului.

## 2. Demonstrează că folosești venv-ul

Rulează:

```bash
python --version
python -m pip --version
```

Găsește executabilul Python exact:

### Windows PowerShell

```powershell
Get-Command python
```

Calea așteptată arată cam așa:

```text
...\venv-demo\.venv\Scripts\python.exe
```

### macOS / Linux

```bash
which python
```

Calea așteptată arată cam așa:

```text
/.../venv-demo/.venv/bin/python
```

Important: dacă path-ul nu conține `.venv`, nu folosești venv-ul.

## 3. Instalează un pachet și salvează requirements

Instalează un pachet simplu:

```bash
python -m pip install requests
```

Salvează lista de pachete din mediu:

```bash
python -m pip freeze > requirements.txt
```

Creează `test_venv.py`:

```python
import sys
import requests

print("Python used:", sys.executable)
print("requests version:", requests.__version__)
print("venv test: OK")
```

Rulează scriptul:

```bash
python test_venv.py
```

Rezultat așteptat: afișează un path Python din `.venv`, o versiune de `requests` și `venv test: OK`.

## 4. Găsește unde este folderul venv

Din interiorul proiectului:

### Windows PowerShell

```powershell
pwd
dir -Force
```

Deschide folderul în File Explorer:

```powershell
explorer .
```

Venv-ul este:

```text
venv-demo\.venv\
```

### macOS

```bash
pwd
ls -la
open .
```

Venv-ul este:

```text
venv-demo/.venv/
```

### Linux

```bash
pwd
ls -la
```

Dacă ai o comandă pentru file browser:

```bash
xdg-open .
```

Venv-ul este:

```text
venv-demo/.venv/
```

## 5. Simulează că venv-ul este corupt: șterge-l

Mai întâi ieși din venv:

```bash
deactivate
```

Șterge `.venv`.

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force .venv
```

### macOS / Linux

```bash
rm -rf .venv
```

Verifică faptul că proiectul încă are fișierele importante:

```text
venv-demo/
  requirements.txt
  test_venv.py
```

Venv-ul este de unică folosință. Fișierele sursă și `requirements.txt` nu sunt.

## 6. Refă venv-ul din `requirements.txt`

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test_venv.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test_venv.py
```

Dacă scriptul afișează `venv test: OK`, refacerea a funcționat.

## Particularități pe sisteme de operare

Windows:

- Folosește `py -m venv .venv` dacă `python` nu este găsit.
- Calea de activare în PowerShell este `.venv\Scripts\Activate.ps1`.
- Dacă activarea este blocată, folosește `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

macOS:

- Folosește `python3`, nu de obicei `python`, când creezi venv-ul.
- Dacă Python lipsește sau este prea vechi, instalează-l de pe Python.org sau prin Homebrew.
- Calea de activare este `.venv/bin/activate`.

Linux:

- Folosește `python3 -m venv .venv`.
- Pe Debian/Ubuntu, dacă venv-ul nu se poate crea, instalează pachetul de sistem:

```bash
sudo apt install python3-venv
```

- Distribuțiile Linux moderne pot bloca `pip install` global; acesta este un motiv să folosești venv-uri, nu un motiv să forțezi instalări globale.

## Tool-uri opționale, explicație pe scurt

- `pyenv`: instalează și schimbă versiuni de Python.
- `pipx`: instalează aplicații CLI Python global, fiecare în propriul mediu izolat.
- `poetry`: gestionează dependențe, packaging, lock files și virtualenvs.
- `uv`: package/project manager Python modern și rapid; poate înlocui multe workflow-uri cu `pip`, `venv` și `pip-tools`.

Pentru începători, începeți cu `venv`, care este inclus în Python. Adăugați tool-urile de mai sus doar când echipa are o nevoie reală.

## Greșeli comune

- Instalezi cu `pip install ...` înainte să activezi venv-ul.
- Folosești `pip` dintr-un Python și `python` din altul. Obicei mai sigur: rulează mereu `python -m pip ...`.
- Faci commit la `.venv/`.
- Ștergi `requirements.txt` în loc de `.venv/`.
- Refolosești un singur venv pentru mai multe proiecte.

## Checklist pentru exercițiu

- [ ] Creează `venv-demo`.
- [ ] Creează și activează `.venv`.
- [ ] Arată că `python` indică în `.venv`.
- [ ] Instalează `requests`.
- [ ] Creează `requirements.txt`.
- [ ] Rulează `test_venv.py`.
- [ ] Localizează `.venv` în folderul proiectului.
- [ ] Șterge `.venv`.
- [ ] Refă `.venv` din `requirements.txt`.
- [ ] Rulează din nou `test_venv.py`.

## Linkuri de referință folosite

Documentație oficială folosită:

- Python `venv` docs: https://docs.python.org/3/library/venv.html
- Python Packaging User Guide: instalare pachete cu `pip` și `venv`: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
- Python Packaging note despre externally managed environments: https://packaging.python.org/en/latest/specifications/externally-managed-environments/

Documentație folosită pentru tool-uri opționale:

- `pyenv`: https://github.com/pyenv/pyenv
- `pipx`: https://pipx.pypa.io/stable/
- `poetry`: https://python-poetry.org/docs/
- `uv`: https://docs.astral.sh/uv/

Video-uri YouTube folosite:

- Corey Schafer, Windows: Python Tutorial: VENV (Windows): https://www.youtube.com/watch?v=APOPm01BVrk
- Corey Schafer, Mac/Linux: Python Tutorial: VENV (Mac & Linux): https://www.youtube.com/watch?v=N5vscPTWKOk
