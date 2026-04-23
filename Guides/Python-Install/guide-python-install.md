# Python, VS Code, and Console: quick teaching guide

## 1. Install and check Python

Goal: Python must work from the terminal/console, not only from an app icon.

1. Open a terminal:
   - Windows: open **PowerShell** or **Command Prompt**.
   - macOS: open **Terminal**.
   - Linux: open **Terminal**.

2. Check whether Python is already installed:

   **Windows**
   ```powershell
   py --version
   python --version
   ```

   **macOS/Linux**
   ```bash
   python3 --version
   ```

3. Good result: you see something like `Python 3.x.x`.

4. If Python is not found:
   - Windows/macOS: install it from https://www.python.org/downloads/
   - Windows: during install, enable **Add python.exe to PATH** if shown.
   - Linux: install Python 3 using your distro's software manager/package manager.

5. Re-open the terminal and check the version again.

## 2. Install VS Code and the Python extension

1. Install VS Code from https://code.visualstudio.com/download
2. Open VS Code.
3. Go to **Extensions**.
4. Search for **Python**.
5. Install the **Python** extension by **Microsoft**.
6. Important: open a **folder/project**, not just a single `.py` file.
   - Use **File > Open Folder...**
   - A project folder is where your script files live together.

## 3. Console/terminal basics

The terminal is a text box where you type commands. Python scripts run relative to the folder the terminal is currently "inside".

1. Open the terminal inside VS Code:
   - **View > Terminal**
   - Shortcut: press `Ctrl` + the backtick key on Windows/Linux, or `Control` + the backtick key on macOS

2. Show the current folder:

   **Windows/macOS/Linux**
   ```bash
   pwd
   ```

3. List files in the current folder:

   **Windows**
   ```powershell
   dir
   ```

   **macOS/Linux**
   ```bash
   ls
   ```

4. Move into a folder:

   ```bash
   cd folder-name
   ```

5. Move up one folder:

   ```bash
   cd ..
   ```

6. Key rule: if your terminal is in the wrong folder, Python may not find your script.

## 4. Basic project task

Goal: create and run one real Python script.

1. Create a folder named:

   ```text
   python-console-demo
   ```

2. Open that folder in VS Code:
   - **File > Open Folder...**
   - Select `python-console-demo`

3. In VS Code, create a new file named:

   ```text
   hello.py
   ```

4. Put this code inside `hello.py`:

   ```python
   import sys
   from pathlib import Path

   print("Hello from Python")
   print("Python executable:", sys.executable)
   print("Script location:", Path(__file__).resolve())
   print("Current folder:", Path.cwd())
   ```

5. Save the file.

## 5. Run the script in VS Code

1. Make sure the `python-console-demo` folder is open in VS Code.
2. Open `hello.py`.
3. Click the **Run Python File** play button in the top-right of the editor.
4. Expected result: output appears in the terminal and includes:

   ```text
   Hello from Python
   ```

5. Also check that the output shows:
   - which Python executable is being used
   - where `hello.py` is located
   - what the current folder is

## 6. Run the script from the console

Before running, confirm the terminal is in the correct folder.

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

Checklist before running:
- `dir` or `ls` should show `hello.py`.
- If `hello.py` does not appear, you are in the wrong folder.
- Use `pwd` to see where the terminal currently is.

Expected result:

```text
Hello from Python
```

## 7. Common mistakes

- Python was installed on Windows, but not added to `PATH`.
- Running `py hello.py` or `python3 hello.py` from the wrong folder.
- Opening only `hello.py` in VS Code instead of opening the whole `python-console-demo` folder.
- Naming the file `hello.py.txt` by accident.
- Using `python` when the system expects `python3`.
- Installing VS Code but not installing the Microsoft Python extension.

## 8. Teach-back checklist

- [ ] Python installed and version checked.
- [ ] VS Code installed.
- [ ] Python extension installed in VS Code.
- [ ] `python-console-demo` folder created.
- [ ] Folder opened in VS Code.
- [ ] `hello.py` created.
- [ ] Script runs from VS Code.
- [ ] Terminal opened inside VS Code.
- [ ] Member can show current folder.
- [ ] Member can list files and find `hello.py`.
- [ ] Script runs from console.
- [ ] Member can explain where the script file is located.

## Reference links used

Corey Schafer tutorials:
- Corey Schafer YouTube channel: https://www.youtube.com/@coreyms
- Corey Schafer Python beginner playlist: https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU
- Python Tutorial for Beginners 1: Install and Setup for Mac and Windows: https://www.youtube.com/watch?v=YYXdXT2l-Gg
- Visual Studio Code on Windows for Python: https://www.youtube.com/watch?v=-nh9rCzPJ20
- Visual Studio Code on macOS for Python: https://www.youtube.com/watch?v=06I63_p-2A4

Official docs:
- Python downloads: https://www.python.org/downloads/
- Python on Windows: https://docs.python.org/3/using/windows.html
- Python on macOS: https://docs.python.org/3/using/mac.html
- Python on Unix/Linux: https://docs.python.org/3/using/unix.html
- VS Code download: https://code.visualstudio.com/download
- Python in VS Code: https://code.visualstudio.com/docs/languages/python
- VS Code Python quick start: https://code.visualstudio.com/docs/python/python-quick-start
- VS Code integrated terminal: https://code.visualstudio.com/docs/terminal/getting-started
- Microsoft Python extension for VS Code: https://marketplace.visualstudio.com/items?itemName=ms-python.python

Additional useful YouTube tutorials:
- Setting up VS Code for Python Beginners: https://youtu.be/7FltByLPnrg
- Getting Started with Python in Visual Studio Code: https://www.youtube.com/watch?v=E9U-EBG8jVk
- Python Tutorial for Beginners with mini-projects: https://www.youtube.com/watch?v=qwAFL1597eM
