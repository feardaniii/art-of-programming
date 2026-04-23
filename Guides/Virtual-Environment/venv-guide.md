# Python virtual environments: quick teaching guide

Goal: every member should be able to create a project venv, know where it lives, delete it when it breaks, rebuild it from `requirements.txt`, and run a tiny test script.

## 0. What is a venv?

A virtual environment is a project-local Python setup. Packages installed inside it stay separate from your system Python and from other projects.

Rule of thumb:

- Put code in the project folder.
- Put the venv inside the project as `.venv/`.
- Commit `requirements.txt`.
- Do **not** commit `.venv/`.

Add this to `.gitignore`:

```gitignore
.venv/
```

## 1. Create a tiny demo project

### Windows PowerShell

```powershell
mkdir venv-demo
cd venv-demo
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run once:

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

You should now see `(.venv)` in the terminal prompt.

## 2. Prove you are using the venv

Run:

```bash
python --version
python -m pip --version
```

Find the exact Python executable:

### Windows PowerShell

```powershell
Get-Command python
```

Expected path looks like:

```text
...\venv-demo\.venv\Scripts\python.exe
```

### macOS / Linux

```bash
which python
```

Expected path looks like:

```text
/.../venv-demo/.venv/bin/python
```

Important: if the path does not include `.venv`, you are not using the venv.

## 3. Install a package and save requirements

Install one simple package:

```bash
python -m pip install requests
```

Save the environment package list:

```bash
python -m pip freeze > requirements.txt
```

Create `test_venv.py`:

```python
import sys
import requests

print("Python used:", sys.executable)
print("requests version:", requests.__version__)
print("venv test: OK")
```

Run it:

```bash
python test_venv.py
```

Expected result: it prints a Python path inside `.venv`, a `requests` version, and `venv test: OK`.

## 4. Find where the venv folder is

From inside the project:

### Windows PowerShell

```powershell
pwd
dir -Force
```

Open the folder in File Explorer:

```powershell
explorer .
```

The venv is:

```text
venv-demo\.venv\
```

### macOS

```bash
pwd
ls -la
open .
```

The venv is:

```text
venv-demo/.venv/
```

### Linux

```bash
pwd
ls -la
```

If you have a file browser command available:

```bash
xdg-open .
```

The venv is:

```text
venv-demo/.venv/
```

## 5. Act like the venv is corrupted: delete it

First leave the venv:

```bash
deactivate
```

Delete `.venv`.

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force .venv
```

### macOS / Linux

```bash
rm -rf .venv
```

Check that your project still has the important files:

```text
venv-demo/
  requirements.txt
  test_venv.py
```

The venv is disposable. Your source files and `requirements.txt` are not.

## 6. Rebuild from `requirements.txt`

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

If the script prints `venv test: OK`, the rebuild worked.

## OS quirks to mention while teaching

Windows:

- Use `py -m venv .venv` if `python` is not found.
- Activation path is `.venv\Scripts\Activate.ps1` in PowerShell.
- If activation is blocked, use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

macOS:

- Use `python3`, not usually `python`, when creating the venv.
- If Python is missing or old, install from Python.org or Homebrew.
- Activation path is `.venv/bin/activate`.

Linux:

- Use `python3 -m venv .venv`.
- On Debian/Ubuntu, if venv creation fails, install the OS package:

```bash
sudo apt install python3-venv
```

- Modern Linux distros may block global `pip install`; that is a reason to use venvs, not a reason to force global installs.

## Optional tools, one-line explanation

- `pyenv`: installs and switches between Python versions.
- `pipx`: installs Python command-line apps globally, each in its own isolated env.
- `poetry`: manages dependencies, packaging, lock files, and virtualenvs.
- `uv`: fast modern Python package/project manager; can replace many `pip`, `venv`, and `pip-tools` workflows.

For beginners, start with built-in `venv` first. Add these tools only when the team has a real need.

## Common mistakes

- Installing with `pip install ...` before activating the venv.
- Using `pip` from one Python and `python` from another. Safer habit: always run `python -m pip ...`.
- Committing `.venv/` to git.
- Deleting `requirements.txt` instead of `.venv/`.
- Reusing one venv for many projects.

## Teach-back checklist

- [ ] Create `venv-demo`.
- [ ] Create and activate `.venv`.
- [ ] Show that `python` points inside `.venv`.
- [ ] Install `requests`.
- [ ] Create `requirements.txt`.
- [ ] Run `test_venv.py`.
- [ ] Locate `.venv` in the project folder.
- [ ] Delete `.venv`.
- [ ] Rebuild `.venv` from `requirements.txt`.
- [ ] Run `test_venv.py` again.

## Reference links used

Official docs used:

- Python `venv` docs: https://docs.python.org/3/library/venv.html
- Python Packaging User Guide: installing packages with `pip` and `venv`: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
- Python Packaging note on externally managed environments: https://packaging.python.org/en/latest/specifications/externally-managed-environments/

Optional tool docs used:

- `pyenv`: https://github.com/pyenv/pyenv
- `pipx`: https://pipx.pypa.io/stable/
- `poetry`: https://python-poetry.org/docs/
- `uv`: https://docs.astral.sh/uv/

YouTube videos used:

- Corey Schafer, Windows: Python Tutorial: VENV (Windows): https://www.youtube.com/watch?v=APOPm01BVrk
- Corey Schafer, Mac/Linux: Python Tutorial: VENV (Mac & Linux): https://www.youtube.com/watch?v=N5vscPTWKOk
