# Tema 29: ML | Face Recognition

Sistem de recunoaștere facială și extragere landmark-uri folosind Python și OpenCV.

## 📋 Cerințe Implementate

### 1. Sistem de Recunoaștere Facială
Sistem care pornește doar dacă recunoaște fața utilizatorului autorizat.

**Caracteristici:**
- Detectare automată a fețelor în imagini
- Comparare cu imagine de referință
- Feedback vizual (culori: verde = autorizat, roșu = respins)
- Scor de similitudine pentru fiecare comparație
- Status final: sistem deblocat/blocat

### 2. Salvare Landmark-uri în CSV
Extragere și salvare coordonatelor punctelor cheie faciale.

**Caracteristici:**
- Detectare automată a landmark-urilor faciale (5 puncte: ochi, nas, gură)
- Procesare batch pentru multiple imagini
- Export în format CSV pentru analiză ulterioară
- Vizualizare în timp real a punctelor detectate

---

## 📁 Structura Proiectului

```
tema29/
│
├── pasul1/                              # Imagini pentru testare
│   ├── 7005d7a0a6f254d93f8323b96024af5d.png
│   ├── 7f355d6f347b05e9e595078672f5e452.png
│   ├── fae249759d796e7e07e03eeca0c0d9b0.png
│   ├── jdsfndfo8u4028ednj2hjdh2308shfgj2.png
│   └── guy-funny.png
│
├── pasul2/                              # Script-uri Python
│   ├── face_recognition_system.py       # Pasul 1: Recunoaștere facială
│   ├── face_landmarks_csv.py            # Pasul 2: Extragere landmarks
│   └── face_landmarks.csv               # Output CSV generat
│
└── README.md                            # Acest fișier

```

---

## 🛠️ Tehnologii Utilizate

- **Python 3.x**
- **OpenCV (cv2)** - Computer vision și procesare imagini
- **NumPy** - Operații matematice și array-uri
- **dlib** (opțional) - Pentru landmark-uri precise (68 puncte)
- **CSV** - Stocare date

---

## 📦 Instalare

### 1. Clonează/Descarcă proiectul

```bash
cd tema29
```

### 2. Creează mediu virtual (recomandat)

```bash
python -m venv venv
```

### 3. Activează mediul virtual

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalează dependențele

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
```

**Opțional (pentru 68 landmark-uri precise):**
```bash
pip install dlib
```

---

## 🚀 Utilizare

### Pasul 1: Sistem de Recunoaștere Facială

Acest script compară imagini cu o imagine de referință pentru a decide dacă utilizatorul este autorizat.

**Rulare:**
```bash
cd pasul2
python face_recognition_system.py
```

**Cum funcționează:**
1. Prima imagine (alfabetic) din `pasul1/` = imagine de referință
2. Restul imaginilor = imagini de test
3. Sistemul compară fiecare imagine test cu referința
4. Afișează rezultate vizuale (3 secunde per imagine)
5. Generează raport final cu scoruri de similitudine

**Parametri ajustabili în cod:**
- `similarity_threshold` (linia ~139): Pragul de acceptare (default: 40%)
  - Valori mai mari = mai strict
  - Valori mai mici = mai permisiv

**Output:**
```
✅ POTRIVIRE  imagine.png  (65.2% similitudine) → ACEEAȘI PERSOANĂ
❌ DIFERIT    stranger.png (28.5% similitudine) → PERSOANĂ DIFERITĂ
```

---

### Pasul 2: Extragere Landmark-uri în CSV

Acest script detectează și salvează coordonatele punctelor cheie faciale.

**Rulare:**
```bash
cd pasul2
python face_landmarks_csv.py
```

**Cum funcționează:**
1. Citește toate imaginile din `pasul1/`
2. Detectează fețele în fiecare imagine
3. Extrage landmark-uri (puncte cheie: ochi, nas, gură)
4. Salvează coordonatele în `face_landmarks.csv`
5. Afișează vizualizare pentru fiecare imagine (2 secunde)

**Landmark-uri detectate (mod simplu):**
- Ochi stâng (x, y)
- Ochi drept (x, y)
- Vârf nas (x, y)
- Colț gură stânga (x, y)
- Colț gură dreapta (x, y)

**Output CSV:**
```csv
imagine,numar_landmarks,face_x,face_y,face_width,face_height,ochi_stang_x,ochi_stang_y,...
imagine1.png,5,100,120,200,250,145,180,245,180,195,210,160,245,230,245
```

**Pentru 68 landmark-uri precise (opțional):**
1. Descarcă: [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Extrage fișierul
3. Plasează `shape_predictor_68_face_landmarks.dat` în folderul `pasul2/`
4. Scriptul va detecta automat și va folosi cei 68 de puncte

---

## 📊 Interpretarea Rezultatelor

### Scor de Similitudine (Pasul 1)
- **60-100%**: Foarte probabil aceeași persoană
- **40-60%**: Posibil aceeași persoană (depinde de lumină/unghi)
- **0-40%**: Probabil persoane diferite

### Landmark-uri (Pasul 2)
Coordonatele sunt în pixeli, relativ la imaginea originală:
- `x`: poziția orizontală (0 = stânga)
- `y`: poziția verticală (0 = sus)

---

## ⚙️ Configurare și Ajustări

### Face Recognition System

**Ajustare prag de similitudine:**
```python
# Linia ~139 în face_recognition_system.py
similarity_threshold = 40  # Schimbă această valoare
```

**Ajustare timp de afișare:**
```python
# Linia ~197
cv2.waitKey(3000)  # 3000 = 3 secunde
```

### Landmarks CSV

**Schimbă folderul sursă:**
```python
# Linia ~72
images_folder = os.path.join('..', 'pasul1')  # Modifică path-ul
```

**Ajustare timp de afișare:**
```python
# Linia ~176
cv2.waitKey(2000)  # 2000 = 2 secunde
```

---

## 🐛 Troubleshooting

### Problema: "Nu s-a detectat nicio față"
**Soluții:**
- Asigură-te că fața este clară și vizibilă în imagine
- Verifică că imaginea nu este prea mică sau prea întunecată
- Încearcă cu o altă imagine

### Problema: "Folderul pasul1/ nu a fost găsit"
**Soluții:**
- Verifică că rulezi scriptul din folderul `pasul2/`
- Asigură-te că folderul `pasul1/` există la nivelul părinte
- Verifică că path-ul în cod este corect

### Problema: Similitudine prea mică pentru aceeași persoană
**Soluții:**
- Scade `similarity_threshold` la 35% sau mai puțin
- Verifică că imaginile au lumină și unghi similar
- Încearcă cu imagini de calitate mai bună

### Problema: "cmake not installed" la instalarea dlib
**Soluții:**
- Instalează CMake: https://cmake.org/download/
- Sau folosește versiunea simplă (5 landmark-uri în loc de 68)
- Scriptul funcționează și fără dlib!

---

## 📝 Note Tehnice

### Algoritmi Utilizați

**Detectare față:**
- Haar Cascade Classifier (OpenCV)
- Metoda clasică, rapidă, funcționează bine pentru fețe frontale

**Comparare fețe:**
- Histogram Correlation
- Template Matching
- Mean Squared Error
- Scor combinat ponderat

**Landmark-uri:**
- Mod simplu: Estimare bazată pe proporții faciale
- Mod avansat: dlib shape predictor (68 puncte)

### Limitări

- Precizia depinde de calitate/lumină/unghi imagine
- OpenCV basic nu e la fel de robust ca modelele deep learning
- Pentru producție, se recomandă face_recognition (deep learning)