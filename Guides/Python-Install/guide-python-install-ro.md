# Python, VS Code și consolă: ghid rapid de predare

## 1. Instalează și verifică Python

Scop: Python trebuie să funcționeze din terminal/consolă, nu doar dintr-o pictogramă de aplicație.

1. Deschide un terminal:
   - Windows: deschide **PowerShell** sau **Command Prompt**.
   - macOS: deschide **Terminal**.
   - Linux: deschide **Terminal**.

2. Verifică dacă Python este deja instalat:

   **Windows**
   ```powershell
   py --version
   python --version
   ```

   **macOS/Linux**
   ```bash
   python3 --version
   ```

3. Rezultat bun: vezi ceva de forma `Python 3.x.x`.

4. Dacă Python nu este găsit:
   - Windows/macOS: instalează-l de la https://www.python.org/downloads/
   - Windows: în timpul instalării, bifează **Add python.exe to PATH** dacă apare.
   - Linux: instalează Python 3 folosind managerul de software/pachete al distribuției tale.

5. Redeschide terminalul și verifică din nou versiunea.

## 2. Instalează VS Code și extensia Python

1. Instalează VS Code de la https://code.visualstudio.com/download
2. Deschide VS Code.
3. Mergi la **Extensions**.
4. Caută **Python**.
5. Instalează extensia **Python** de la **Microsoft**.
6. Important: deschide un **folder/proiect**, nu doar un singur fișier `.py`.
   - Folosește **File > Open Folder...**
   - Un folder de proiect este locul unde stau împreună fișierele scriptului.

## 3. Baze pentru consolă/terminal

Terminalul este o casetă text în care scrii comenzi. Scripturile Python rulează relativ la folderul în care terminalul se află în acel moment.

1. Deschide terminalul în VS Code:
   - **View > Terminal**
   - Scurtătură: apasă `Ctrl` + tasta backtick pe Windows/Linux sau `Control` + tasta backtick pe macOS

2. Afișează folderul curent:

   **Windows/macOS/Linux**
   ```bash
   pwd
   ```

3. Listează fișierele din folderul curent:

   **Windows**
   ```powershell
   dir
   ```

   **macOS/Linux**
   ```bash
   ls
   ```

4. Intră într-un folder:

   ```bash
   cd folder-name
   ```

5. Urca un folder:

   ```bash
   cd ..
   ```

6. Regula importantă: dacă terminalul este în folderul greșit, Python poate să nu găsească scriptul.

## 4. Sarcină de bază pentru proiect

Scop: creează și rulează un script Python real.

1. Creează un folder numit:

   ```text
   python-console-demo
   ```

2. Deschide acel folder în VS Code:
   - **File > Open Folder...**
   - Selectează `python-console-demo`

3. În VS Code, creează un fișier nou numit:

   ```text
   hello.py
   ```

4. Pune acest cod în `hello.py`:

   ```python
   import sys
   from pathlib import Path

   print("Hello from Python")
   print("Python executable:", sys.executable)
   print("Script location:", Path(__file__).resolve())
   print("Current folder:", Path.cwd())
   ```

5. Salvează fișierul.

## 5. Rulează scriptul în VS Code

1. Asigură-te că folderul `python-console-demo` este deschis în VS Code.
2. Deschide `hello.py`.
3. Apasă butonul **Run Python File** din dreapta sus a editorului.
4. Rezultat așteptat: output-ul apare în terminal și include:

   ```text
   Hello from Python
   ```

5. Verifică și că output-ul arată:
   - ce executabil Python este folosit
   - unde se află `hello.py`
   - care este folderul curent

## 6. Rulează scriptul din consolă

Înainte de rulare, confirmă că terminalul este în folderul corect.

**Windows**
```powershell
cd path\to\python-console-demo
dir
py hello.py
```

**macOS/Linux**
```bash
cd path/to/python-console-demo
ls
python3 hello.py
```

Checklist înainte de rulare:
- `dir` sau `ls` trebuie să arate `hello.py`.
- Dacă `hello.py` nu apare, ești în folderul greșit.
- Folosește `pwd` ca să vezi unde se află terminalul acum.

Rezultat așteptat:

```text
Hello from Python
```

## 7. Greșeli frecvente

- Python a fost instalat pe Windows, dar nu a fost adăugat în `PATH`.
- Rulezi `py hello.py` sau `python3 hello.py` din folderul greșit.
- Deschizi doar `hello.py` în VS Code în loc să deschizi tot folderul `python-console-demo`.
- Numele fișierului este accidental `hello.py.txt`.
- Folosești `python` când sistemul așteaptă `python3`.
- Ai instalat VS Code, dar nu ai instalat extensia Python de la Microsoft.

## 8. Checklist de predare înapoi

- [ ] Python este instalat și versiunea a fost verificată.
- [ ] VS Code este instalat.
- [ ] Extensia Python este instalată în VS Code.
- [ ] Folderul `python-console-demo` a fost creat.
- [ ] Folderul este deschis în VS Code.
- [ ] `hello.py` a fost creat.
- [ ] Scriptul rulează din VS Code.
- [ ] Terminalul este deschis în VS Code.
- [ ] Membrul poate arăta folderul curent.
- [ ] Membrul poate lista fișierele și găsi `hello.py`.
- [ ] Scriptul rulează din consolă.
- [ ] Membrul poate explica unde se află fișierul scriptului.

## Linkuri de referință folosite

Tutoriale Corey Schafer:
- Canalul YouTube Corey Schafer: https://www.youtube.com/@coreyms
- Playlist Corey Schafer pentru începători Python: https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU
- Python Tutorial for Beginners 1: Install and Setup for Mac and Windows: https://www.youtube.com/watch?v=YYXdXT2l-Gg
- Visual Studio Code pe Windows pentru Python: https://www.youtube.com/watch?v=-nh9rCzPJ20
- Visual Studio Code pe macOS pentru Python: https://www.youtube.com/watch?v=06I63_p-2A4

Documentație oficială:
- Python downloads: https://www.python.org/downloads/
- Python pe Windows: https://docs.python.org/3/using/windows.html
- Python pe macOS: https://docs.python.org/3/using/mac.html
- Python pe Unix/Linux: https://docs.python.org/3/using/unix.html
- VS Code download: https://code.visualstudio.com/download
- Python în VS Code: https://code.visualstudio.com/docs/languages/python
- VS Code Python quick start: https://code.visualstudio.com/docs/python/python-quick-start
- VS Code integrated terminal: https://code.visualstudio.com/docs/terminal/getting-started
- Extensia Microsoft Python pentru VS Code: https://marketplace.visualstudio.com/items?itemName=ms-python.python

Alte tutoriale YouTube utile:
- Setting up VS Code for Python Beginners: https://youtu.be/7FltByLPnrg
- Getting Started with Python in Visual Studio Code: https://www.youtube.com/watch?v=E9U-EBG8jVk
- Python Tutorial for Beginners with mini-projects: https://www.youtube.com/watch?v=qwAFL1597eM
