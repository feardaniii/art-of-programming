# 👁️ OPENCV VIZIUNE COMPUTERIZATĂ - SISTEM DE SUPRAVEGHERE & SIGURANȚĂ

---

## 📚 Introducere

Bine ai venit la **Viziune Computerizată** - unde codul tău învață să "vadă" lumea!

În acest capitol, vei descoperi că imaginile sunt doar **array-uri NumPy deghizate**. Fiecare pixel este un număr, iar manipulând aceste numere, poți:
- 🎨 **Detecta culori** (găsește toate obiectele albastre)
- 🔍 **Urmări mișcarea** (observă schimbările între cadre)
- 🚨 **Recunoaște pattern-uri** (detectează fum, foc, fețe)
- 📹 **Construi sisteme în timp real** (supraveghere, alerte de siguranță)

La final, vei construi un **sistem real de supraveghere și siguranță contra incendiilor** folosind doar matematică de pixeli și OpenCV!

---

## 🎯 Ce este OpenCV?

**OpenCV** (Open Source Computer Vision Library) este standardul industriei pentru viziune computerizată:

```
Cameră/Imagine → Array NumPy → Filtre OpenCV → Insight-uri/Acțiuni
      ↓              ↓              ↓              ↓
   Pixeli      [valori 0-255]  Algoritmi      Decizii
```

**Concepte cheie pe care le vei stăpâni:**
- **Spații de culoare**: RGB (ce vedem noi), HSV (mai ușor de filtrat), Grayscale (doar forme)
- **Thresholding**: Convertirea imaginilor în binar (alb/negru) pe bază de condiții
- **Contururi**: Găsirea marginilor obiectelor
- **Diferența de cadre**: Detectarea a ceea ce s-a schimbat între două imagini
- **Operații morfologice**: Curățarea zgomotului, umplerea golurilor

---

## 🚀 Setup & Instalare

Asigură-te că ai OpenCV instalat:
```bash
pip install opencv-python matplotlib numpy
```

**Test rapid:**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

print(f"Versiune OpenCV: {cv2.__version__}")

# Creează un pătrat roșu de 100x100
red_square = np.zeros((100, 100, 3), dtype=np.uint8)
red_square[:, :] = [255, 0, 0]  # RGB: Roșu

plt.imshow(red_square)
plt.title("Prima ta creație OpenCV!")
plt.axis('off')
plt.show()
```

**💡 Amintește-ți:** Imaginile sunt doar array-uri 3D: `(înălțime, lățime, canale)`

---

# PARTEA 1: EXERCIȚII DE ÎNCĂLZIRE

---

## 📝 EXERCIȚIUL 1: Detector de Obiecte Albastre

**Obiectiv:** Detectează toate obiectele albastre dintr-o imagine folosind spațiul de culoare HSV

### 📖 Concepte pe care le vei învăța:

**`Spații de culoare`**: Moduri diferite de a reprezenta culorile
- **RGB**: Roșu-Verde-Albastru (0-255 fiecare) - cum văd camerele
- **HSV**: Hue-Saturation-Value - mai ușor pentru detectarea culorilor
  - **Hue**: Culoarea propriu-zisă (0-180 în OpenCV)
  - **Saturation**: Intensitatea culorii (0-255)
  - **Value**: Luminozitatea (0-255)

**`cv2.inRange()`**: Creează o mască unde pixelii în interval = alb (255), restul = negru (0)

**`cv2.findContours()`**: Găsește marginile regiunilor albe în imagini binare

**`cv2.contourArea()`**: Calculează aria unui contur în pixeli

### 🎯 Misiunea Ta:

1. Încarcă o imagine (fă o poză cu telefonul tău cu obiecte colorate!)
2. Convertește din BGR în spațiul de culoare HSV
3. Creează o mască care izolează DOAR pixelii albaștri
4. Găsește contururile (marginile) obiectelor albastre
5. Filtrează zgomotul mic (păstrează doar obiecte > 500 pixeli)
6. Desenează dreptunghiuri în jurul obiectelor albastre detectate
7. Afișează: imaginea originală, masca, și detecția finală

### 💻 Cod de Pornire:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Încarcă imaginea (înlocuiește cu a ta!)
image = cv2.imread('your_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"📸 Imagine încărcată: {image_rgb.shape}")

# TODO 1: Convertește în HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# TODO 2: Definește intervalul de culoare albastră în HSV
# Albastru în HSV: Hue ~100-130, saturație mare, valoare decentă
lower_blue = np.array([100, 50, 50])    # Ajustează acestea!
upper_blue = np.array([130, 255, 255])  # Ajustează acestea!

# TODO 3: Creează masca
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# TODO 4: Găsește contururile
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# TODO 5: Filtrează și desenează
result = image_rgb.copy()
detected_count = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 500:  # Filtrează zgomotul mic
        # Obține dreptunghiul delimitat
        x, y, w, h = cv2.boundingRect(contour)

        # Desenează pe rezultat
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result, f"Albastru #{detected_count+1}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detected_count += 1

        print(f"   Obiect #{detected_count}: {int(area)} pixeli la ({x}, {y})")

print(f"\n✅ Detectate {detected_count} obiecte albastre!")

# TODO 6: Vizualizează rezultatele
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title('Imagine Originală')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mască Albastră (Alb = Pixeli albaștri)')
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title(f'Detecție: {detected_count} obiecte albastre')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 💡 Indicii:

- **Găsirea intervalului HSV corect**: Folosește trial and error! Intervale comune:
  - Albastru: `[100, 50, 50]` până la `[130, 255, 255]`
  - Verde: `[40, 50, 50]` până la `[80, 255, 255]`
  - Roșu: `[0, 50, 50]` până la `[10, 255, 255]` SAU `[170, 50, 50]` până la `[180, 255, 255]`
- **Pro tip**: Creează un trackbar pentru a ajusta intervalele în timp real (avansat!)

### ❓ Întrebări de Reflecție:

**1. De ce folosim HSV în loc de RGB pentru detectarea culorilor?**

_Răspunsul tău:_ _______________________________________________

**2. Ce se întâmplă dacă setezi `area > 100` în loc de `area > 500`?**

_Răspunsul tău:_ _______________________________________________

**3. Provocare: Modifică codul pentru a detecta obiecte ROȘII. Ce se schimbă?**

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 2: Îmbunătățirea Textului Vechi cu Threshold Adaptiv

**Obiectiv:** Fă textul vechi și decolorat lizibil din nou folosind thresholding adaptiv

### 📖 Concepte pe care le vei învăța:

**`Thresholding`**: Convertirea imaginilor în tonuri de gri în alb/negru pur
- **Threshold global**: O singură valoare pentru întreaga imagine
- **Threshold adaptiv**: Valori diferite pentru regiuni diferite (mai bun pentru iluminare neuniformă!)

**`cv2.adaptiveThreshold()`**: Thresholding inteligent care gestionează umbrele și variațiile de lumină

**De ce contează:** Scanarea documentelor vechi, citirea bonurilor, preprocesarea OCR

### 🎯 Misiunea Ta:

1. Găsește o imagine cu text (pagină veche de carte, notă scrisă de mână, bon fiscal)
2. Convertește în tonuri de gri
3. Aplică threshold obișnuit (global) - observă limitările
4. Aplică threshold adaptiv - vezi magia!
5. Compară rezultatele side-by-side

### 💻 Cod de Pornire:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Încarcă imagine cu text (fă o poză la o pagină de carte!)
image = cv2.imread('old_text.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"📄 Imagine încărcată: {gray.shape}")

# TODO 1: Aplică threshold global
# Valoare threshold: 127 (mijlocul dintre 0-255)
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# TODO 2: Aplică threshold adaptiv (metoda Gaussian)
adaptive_thresh = cv2.adaptiveThreshold(
    gray,
    255,                              # Valoare maximă
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Metodă
    cv2.THRESH_BINARY,                # Tip
    11,                               # Dimensiune bloc (trebuie să fie impar!)
    2                                 # Constantă C (scăzută din medie)
)

# TODO 3: Încearcă și metoda Mean
adaptive_mean = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,       # Metodă diferită
    cv2.THRESH_BINARY,
    11,
    2
)

# Vizualizează toate versiunile
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Original Tonuri de Gri')
axes[0, 0].axis('off')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('Threshold Global (127)')
axes[0, 1].axis('off')

axes[1, 0].imshow(adaptive_thresh, cmap='gray')
axes[1, 0].set_title('Threshold Adaptiv (Gaussian)')
axes[1, 0].axis('off')

axes[1, 1].imshow(adaptive_mean, cmap='gray')
axes[1, 1].set_title('Threshold Adaptiv (Mean)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('text_enhancement_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Comparație salvată în: text_enhancement_comparison.png")
plt.show()

# Calculează scor de lizibilitate (euristică simplă)
def calculate_readability(binary_image):
    """Scor mai mare = separare text/fundal mai bună"""
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)
    ratio = min(white_pixels, black_pixels) / max(white_pixels, black_pixels)
    return ratio * 100

print("\n📊 Scoruri de Lizibilitate:")
print(f"   Threshold global: {calculate_readability(global_thresh):.1f}/100")
print(f"   Adaptiv (Gaussian): {calculate_readability(adaptive_thresh):.1f}/100")
print(f"   Adaptiv (Mean): {calculate_readability(adaptive_mean):.1f}/100")
```

### 💡 Indicii:

- **Dimensiune bloc**: Mai mare = mai neted (încearcă 11, 15, 21, 31)
- **Constantă C**: Valori pozitive fac imaginea mai luminoasă, negative = mai întunecată
- **Testează cu**: Bonuri vechi, poze la tablă albă, notițe scanate

### 🎓 Provocare Avansată:

Aplică **operații morfologice** pentru a curăța rezultatul:
```python
# Elimină zgomotul mic
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
```

### ❓ Întrebări de Reflecție:

**1. De ce eșuează threshold-ul global pe imagini cu iluminare neuniformă?**

_Răspunsul tău:_ _______________________________________________

**2. Când ai folosi threshold adaptiv GAUSSIAN vs MEAN?**

_Răspunsul tău:_ _______________________________________________

**3. Aplicație din lumea reală: Cum ar putea ajuta acest lucru la OCR (recunoașterea textului)?**

_Răspunsul tău:_ _______________________________________________

---

## 📝 EXERCIȚIUL 3: Analiza Ariei Contururilor - Găsește Obiecte Mari

**Obiectiv:** Extrage și analizează obiecte mai mari decât o dimensiune specifică

### 📖 Concepte pe care le vei învăța:

**`Contururi`**: Marginile formelor în imagini binare
**`Operații morfologice`**: Operații matematice pe imagini binare
- **Eroziune**: Micșorează obiectele (elimină zgomotul)
- **Dilatare**: Mărește obiectele (umple golurile)
- **Opening**: Eroziune → Dilatare (elimină zgomotul mic)
- **Closing**: Dilatare → Eroziune (umple găurile mici)

### 🎯 Misiunea Ta:

1. Încarcă o imagine cu mai multe obiecte
2. Convertește în binar (threshold)
3. Găsește TOATE contururile
4. Filtrează: păstrează doar contururi > 500 pixeli
5. Desenează fiecare contur cu o culoare diferită
6. Afișează aria, perimetrul și centroidul fiecărui obiect

### 💻 Cod de Pornire:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Încarcă imaginea
image = cv2.imread('objects.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"📸 Imagine încărcată: {image_rgb.shape}")

# TODO 1: Convertește în binar
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# TODO 2: Curăță cu operații morfologice
kernel = np.ones((5, 5), np.uint8)
binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

# TODO 3: Găsește contururile
contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"🔍 Găsite {len(contours)} contururi totale")

# TODO 4: Filtrează și analizează
result = image_rgb.copy()
large_objects = []

# Generează culori aleatorii pentru fiecare obiect
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(contours))]

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    if area > 500:  # Prag minim de arie
        # Calculează proprietăți
        perimeter = cv2.arcLength(contour, True)

        # Calculează centroidul (centrul de masă)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Obține dreptunghiul delimitat
        x, y, w, h = cv2.boundingRect(contour)

        # Stochează informații
        large_objects.append({
            'id': len(large_objects) + 1,
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'bbox': (x, y, w, h)
        })

        # Desenează conturul
        cv2.drawContours(result, [contour], -1, colors[i], 3)

        # Desenează centroidul
        cv2.circle(result, (cx, cy), 7, (255, 0, 0), -1)

        # Desenează caseta delimitatoare
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Adaugă etichetă
        cv2.putText(result, f"#{len(large_objects)}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# Afișează analiza
print(f"\n✅ Găsite {len(large_objects)} obiecte mari (> 500 pixeli)\n")
print("📊 Analiza Obiectelor:")
print("-" * 70)
for obj in sorted(large_objects, key=lambda x: x['area'], reverse=True):
    print(f"   Obiect #{obj['id']}:")
    print(f"      Arie: {int(obj['area'])} pixeli")
    print(f"      Perimetru: {int(obj['perimeter'])} pixeli")
    print(f"      Centroid: {obj['centroid']}")
    print(f"      Casetă delimitatoare: {obj['bbox']}")
    print()

# Vizualizează
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image_rgb)
axes[0].set_title('Imagine Originală')
axes[0].axis('off')

axes[1].imshow(binary_clean, cmap='gray')
axes[1].set_title('Binar (După Morfologie)')
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title(f'Detectate: {len(large_objects)} obiecte')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 💡 Indicii:

- **Experimentează cu threshold**: Încearcă adaptiv în loc de global
- **Dimensiunea kernel contează**: Kernel mai mare = netezire mai agresivă
- **RETR_EXTERNAL**: Doar contururile exterioare (folosește RETR_TREE pentru cele imbricate)

### ❓ Întrebări de Reflecție:

**1. Care este diferența între arie și perimetru? Când este utilă fiecare?**

_Răspunsul tău:_ _______________________________________________

**2. De ce avem nevoie de operații morfologice înainte de a găsi contururi?**

_Răspunsul tău:_ _______________________________________________

**3. Provocare: Cum ai identifica cercuri vs dreptunghiuri folosind contururi?**

_Răspunsul tău:_ _______________________________________________

---

# PARTEA 2: PROIECT FINAL - SISTEM DE SUPRAVEGHERE & SIGURANȚĂ CONTRA INCENDIILOR

---

## 🚨 PROIECT: Sistem de Supraveghere în Timp Real & Detecție Fum

**Obiectiv:** Construiește un sistem complet de securitate care detectează:
1. **Intruși**: Detecția mișcării prin diferența de cadre
2. **Pericole de incendiu**: Detecția fumului prin analiza culorii și pattern-urilor

Aceasta este o **aplicație din lumea reală** care combină tot ce ai învățat!

---

## 📋 Cerințe Sistem

Sistemul tău trebuie să:

### 🎯 Funcționalități de Bază:

1. **Stream Webcam**: Capturează video live de la cameră
2. **Detecție Mișcare**:
   - Compară cadrele consecutive
   - Calculează diferențele de pixeli
   - Declanșează alarma când schimbările depășesc pragul
   - Desenează casete delimitatoare în jurul obiectelor în mișcare
3. **Detecție Fum**:
   - Analizează culorile cadrului (fum = gradient gri/alb)
   - Urmărește pattern-ul de răspândire (fumul se ridică și se extinde)
   - Detectează schimbări graduale de culoare pe mai multe cadre
4. **Sistem de Alarmă**:
   - Alerte vizuale (bordură roșie intermitentă)
   - Avertismente în consolă cu timestamp-uri
   - Salvează capturi de ecran ale evenimentelor

### 🏆 Funcționalități Avansate (Opțional):

- Ignoră mișcări mici (animale de companie, umbre)
- Monitorizare pe zone (sensibilitate diferită pe zonă)
- Logging evenimente în fișier
- Alerte email/SMS (folosind `smtplib` sau Twilio)

---

## 🧠 Abordare Tehnică

### Partea A: Algoritmul de Detecție Mișcare

```
Pas 1: Capturează cadru_t și cadru_t-1
Pas 2: Convertește ambele în tonuri de gri
Pas 3: Calculează |cadru_t - cadru_t-1| = hartă_diferențe
Pas 4: Aplică threshold → hartă binară
Pas 5: Găsește contururi în harta binară
Pas 6: Filtrează contururi după arie → schimbări mari = mișcare!
Pas 7: Dacă total_pixeli_schimbați > PRAG_MIȘCARE → ALERTĂ!
```

**Insight cheie:** Corpurile sunt obiecte mari, deci concentrează-te pe contururi > 1000 pixeli

### Partea B: Algoritmul de Detecție Fum

```
Pas 1: Convertește cadrul în HSV
Pas 2: Definește intervalul de culoare fum (gri/alb: saturație scăzută, valoare mare)
Pas 3: Creează mască fum
Pas 4: Urmărește aria de fum pe ultimele N cadre
Pas 5: Dacă aria_fum CREȘTE în timp → ALERTĂ FOC!
Pas 6: Bonus: Verifică dacă fumul "se ridică" (coordonatele y scad)
```

**Insight cheie:** Fumul are:
- Saturație scăzută (nu este colorat)
- Valoare mare (luminos/deschis)
- Pattern de extindere (crește pe cadre)
- Mișcare ascendentă (se deplasează în sus)

---

## 💻 Cod de Pornire Proiect

### Partea 1: Doar Detecție Mișcare

```python
import cv2
import numpy as np
from datetime import datetime

# Configurare
MOTION_THRESHOLD = 2000  # Pixeli schimbați pentru a declanșa alarma
MIN_CONTOUR_AREA = 1000  # Dimensiune minimă obiect (filtrează zgomotul)

# Inițializează webcam
cap = cv2.VideoCapture(0)
ret, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

print("🎥 Sistem de Supraveghere Pornește...")
print(f"   Prag mișcare: {MOTION_THRESHOLD} pixeli")
print(f"   Dimensiune min obiect: {MIN_CONTOUR_AREA} pixeli")
print("\nApasă 'q' pentru a ieși\n")

alarm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TODO 1: Pregătește cadrul curent
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

    # TODO 2: Calculează diferența de cadre
    frame_diff = cv2.absdiff(previous_gray, current_gray)

    # TODO 3: Threshold pe diferență
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # TODO 4: Dilatează pentru a umple golurile
    thresh = cv2.dilate(thresh, None, iterations=2)

    # TODO 5: Găsește contururile
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # TODO 6: Analizează mișcarea
    motion_detected = False
    total_motion_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > MIN_CONTOUR_AREA:
            motion_detected = True
            total_motion_area += area

            # Desenează caseta delimitatoare
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # TODO 7: Declanșează alarma
    if total_motion_area > MOTION_THRESHOLD:
        if not alarm_active:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🚨 MIȘCARE DETECTATĂ! [{timestamp}] - {int(total_motion_area)} pixeli schimbați")
            alarm_active = True

            # Salvează captură
            cv2.imwrite(f'motion_{timestamp.replace(":", "-")}.jpg', frame)

        # Alarmă vizuală
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        cv2.putText(frame, "!!! ALERTA MISCARE !!!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        alarm_active = False

    # TODO 8: Afișează informații
    cv2.putText(frame, f"Miscare: {int(total_motion_area)} px", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Arată feed-urile
    cv2.imshow('Sistem Supraveghere', frame)
    cv2.imshow('Mască Mișcare', thresh)

    # Actualizează cadrul anterior
    previous_gray = current_gray

    # Ieșire la 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Sistem de Supraveghere Oprit")
```

---

### Partea 2: Sistem Complet cu Detecție Fum

**SARCINA TA:** Extinde codul de mai sus pentru a include detecția fumului!

```python
# Adaugă după detecția mișcării (în interiorul buclei while)

# TODO: IMPLEMENTARE DETECȚIE FUM

# 1. Convertește cadrul în HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 2. Definește intervalul de culoare pentru fum
# Caracteristicile fumului în HSV:
#   - Hue: Oricare (0-180)
#   - Saturation: SCĂZUTĂ (0-50) - nu foarte colorat
#   - Value: MARE (200-255) - luminos/deschis
lower_smoke = np.array([0, 0, 200])      # TODO: Ajustează aceste valori!
upper_smoke = np.array([180, 50, 255])   # TODO: Ajustează aceste valori!

# 3. Creează mască fum
smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

# 4. Curăță zgomotul
kernel = np.ones((5, 5), np.uint8)
smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

# 5. Calculează aria de fum
smoke_pixels = np.sum(smoke_mask > 0)

# TODO: Urmărește fumul în timp (folosește o listă pentru a stoca ultimele 10 cadre)
# smoke_history = []  # Inițializează înainte de buclă
# smoke_history.append(smoke_pixels)
# if len(smoke_history) > 10:
#     smoke_history.pop(0)

# 6. Verifică pattern-ul de fum în extindere
SMOKE_THRESHOLD = 5000  # Ajustează pe baza testării
smoke_detected = smoke_pixels > SMOKE_THRESHOLD

# TODO: Avansat - verifică dacă fumul CREȘTE
# if len(smoke_history) >= 5:
#     recent_avg = np.mean(smoke_history[-5:])
#     old_avg = np.mean(smoke_history[:5])
#     if recent_avg > old_avg * 1.5:  # Creștere de 50%
#         smoke_detected = True

# 7. Alarmă de incendiu!
if smoke_detected:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔥 FUM DETECTAT! [{timestamp}] - {int(smoke_pixels)} pixeli de fum")

    # Alarmă vizuală
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 255), 15)
    cv2.putText(frame, "!!! ALERTA FOC !!!", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 4)

    # Salvează captură
    cv2.imwrite(f'smoke_{timestamp.replace(":", "-")}.jpg', frame)

# 8. Afișează informații fum
cv2.putText(frame, f"Fum: {int(smoke_pixels)} px", (10, frame.shape[0] - 40),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Arată masca de fum
cv2.imshow('Detecție Fum', smoke_mask)
```

---

## 🧪 Testarea Sistemului Tău

### Testarea Detecției Mișcare:
1. Pornește programul
2. Stai nemișcat 3 secunde
3. Fluturați mâna în fața camerei
4. Mergi peste cadru
5. **Așteptat**: Casete verzi în jurul tău, bordură de alertă roșie

### Testarea Detecției Fum:
Deoarece fumul real este periculos, simulează-l:
1. **Test hârtie albă**: Mișcă încet o hârtie albă/gri în sus
2. **Test lanternă**: Luminează prin țesătură albă
3. **Abur (sigur)**: Fierbe apă și capturează aburul care se ridică
4. **Test imagine**: Afișează imagine cu fum pe ecranul telefonului, mișcă-l

**💡 Pro tip:** Ajustează pragurile pe baza iluminării mediului tău!

---

## 📊 Livrabile

Submission-ul tău final ar trebui să includă:

```
opencv_surveillance_project/
├── surveillance_system.py          # Cod principal
├── test_results/
│   ├── motion_2024-12-30_14-23-10.jpg
│   ├── smoke_2024-12-30_14-25-33.jpg
│   └── screenshots.png (toate cele 4 ferestre)
├── README.md                       # Documentație proiect
└── reflection_answers.md
```

### README.md ar trebui să conțină:
- **Cum să rulezi** sistemul
- **Valorile pragurilor** pe care le-ai folosit (și de ce)
- **Metodologia de testare** (cum ai testat mișcarea/fumul)
- **Provocările întâmpinate** și soluții
- **Link video demo** (opțional dar impresionant!)

---

## ❓ Întrebări Finale de Reflecție

### 1. Explică cum diferența de cadre detectează mișcarea. De ce folosim diferența absolută?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________

### 2. Ce face detecția fumului mai dificilă decât detecția mișcării?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________

### 3. Cum ai reduce alarmele false (de ex., ignoră umbrele, schimbările de lumină)?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________

### 4. Sistemul tău prinde un intrus la ora 3 dimineața. Ce funcționalități suplimentare l-ar face mai util?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________

### 5. Implementare în lumea reală: Care sunt considerațiile etice ale sistemelor de supraveghere?

_Răspunsul tău:_
_______________________________________________________________
_______________________________________________________________

---

## 🎓 Rubrica de Evaluare

| Criteriu | Puncte | Descriere |
|-----------|--------|-------------|
| **Detecție Mișcare** | 30 | Detectează și evidențiază corect mișcarea |
| **Detecție Fum** | 30 | Identifică pattern-uri asemănătoare fumului |
| **Calitate Cod** | 15 | Curat, comentat, organizat |
| **Testare** | 10 | Proces de testare documentat cu capturi de ecran |
| **Alarme** | 10 | Alerte vizuale + log-uri în consolă funcționale |
| **Reflecție** | 5 | Răspunsuri gândite la întrebări |
| **Funcționalități Bonus** | +10 | Logging evenimente, zone multiple, alerte email, etc. |

**Total: 100 puncte** (+ 10 bonus)

---

## 💡 Resurse Utile

- **Documentație OpenCV**: https://docs.opencv.org/4.x/
- **Convertor spațiu culoare**: https://colorizer.org/
- **Selector culori HSV**: https://alloyui.com/examples/color-picker/hsv
- **Tutorial diferență cadre**: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
- **Operații morfologice**: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

---

## 🔥 Provocări Extra (Pentru Cei Curajoși!)

### Provocare 1: Integrare Detecție Fețe
Adaugă recunoașterea feței pentru a identifica persoane autorizate vs neautorizate:
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Dacă fața detectată dar nu în lista autorizată → ALERTĂ INTRUS
```

### Provocare 2: Monitorizare Bazată pe Zone
Definește regiuni specifice (de ex., ușă, fereastră) cu sensibilități diferite:
```python
zones = {
    'usa': {'bbox': (0, 0, 200, 480), 'threshold': 1000},
    'fereastra': {'bbox': (400, 0, 640, 300), 'threshold': 2000}
}
```

### Provocare 3: Detecție Fum Bazată pe ML
Antrenează o rețea neuronală mică pentru a clasifica fum vs non-fum:
```python
# Folosește keras/tensorflow cu dataset fum/fără-fum
# Mult mai precis decât detecția bazată pe culoare!
```

---

## ✅ Lista de Verificare Finală

Înainte de submission, verifică:

- [ ] Detecția mișcării funcționează și declanșează alarme
- [ ] Detecția fumului identifică pattern-uri asemănătoare fumului
- [ ] Ambele sisteme pot rula simultan
- [ ] Capturile sunt salvate corect
- [ ] Log-urile în consolă arată timestamp-uri
- [ ] Codul este bine comentat
- [ ] Testat cu scenarii multiple
- [ ] Valorile pragurilor sunt documentate
- [ ] Întrebările de reflecție au răspuns
- [ ] README.md este complet

---

**🎉 Felicitări! 🎉**

Ai construit un **sistem real de supraveghere** folosind doar pixeli și matematică. Aceasta este fundația:
- 🏠 Securitate pentru casa inteligentă
- 🏭 Monitorizare siguranță industrială
- 🚗 Percepție vehicul autonom
- 📱 Funcționalități cameră mobilă
- 🤖 Sisteme de viziune roboți

**Amintește-ți:** Viziunea computerizată este doar **array-uri NumPy + algoritmi inteligenți**. Acum ai instrumentele pentru a face computerele să vadă!

---

*"Camera este un instrument care îi învață pe oameni cum să vadă fără cameră."* - Dorothea Lange

**Acum du-te și construiește ceva care vede lumea diferit!** 👁️✨
