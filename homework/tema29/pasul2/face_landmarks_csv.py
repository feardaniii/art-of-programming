import cv2
import numpy as np
import os
import csv
import time

print("=== EXTRAGERE LANDMARK-URI FACIALE ===\n")

# Încarcă clasificatorul pentru detectarea fețelor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Încarcă detectorul de landmark-uri (facial landmarks)
# OpenCV nu are built-in 68 puncte, dar putem folosi dlib SAU face detection de bază
# Pentru simplitate, vom folosi dlib dacă e disponibil, altfel folosim detectare simplă

try:
    import dlib
    USE_DLIB = True
    print("✓ dlib disponibil - folosim 68 landmark-uri\n")
    
    # Încarcă predictor-ul dlib (acest fișier trebuie descărcat)
    # Vom încerca să-l găsim local sau vom folosi o alternativă
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        print("⚠️  Fișierul shape_predictor_68_face_landmarks.dat nu a fost găsit")
        print("   Descarcă-l de la: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("   Extrage și pune-l în folder-ul curent\n")
        print("   Continuăm cu detectare simplă (5 puncte)...\n")
        USE_DLIB = False
    else:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
except ImportError:
    USE_DLIB = False
    print("⚠️  dlib nu este instalat")
    print("   Continuăm cu detectare simplă (5 puncte: ochi, nas)...\n")

# PASUL 1: Găsește toate imaginile
print("🔍 Căutare imagini în folder pasul1/...")
time.sleep(0.3)

# Folderul cu imaginile (caută în folderul părinte)
images_folder = os.path.join('..', 'pasul1')

# Verifică dacă folderul există
if not os.path.exists(images_folder):
    print(f"❌ EROARE: Folderul '{images_folder}/' nu a fost găsit!")
    print(f"   Asigură-te că rulezi scriptul din folderul tema29/")
    exit()

all_images = []
extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

for file in os.listdir(images_folder):
    if any(file.endswith(ext) for ext in extensions):
        all_images.append(file)

all_images.sort()

if len(all_images) == 0:
    print("❌ Nu s-au găsit imagini în folder!")
    exit()

print(f"✓ S-au găsit {len(all_images)} imagini\n")
time.sleep(0.3)

# PASUL 2: Procesează fiecare imagine și extrage landmark-uri
print("="*70)
print("📊 EXTRAGERE LANDMARK-URI")
print("="*70)
time.sleep(0.5)

all_landmarks_data = []

for idx, image_name in enumerate(all_images):
    print(f"\n📸 [{idx+1}/{len(all_images)}] Procesare: {image_name}")
    print("-" * 70)
    time.sleep(0.2)
    
    # Construiește path-ul complet către imagine
    image_path = os.path.join(images_folder, image_name)
    
    # Încarcă imaginea
    img = cv2.imread(image_path)
    
    if img is None:
        print("   ❌ Nu s-a putut încărca imaginea")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if USE_DLIB:
        # Folosește dlib pentru 68 landmark-uri
        print("   🔍 Detectare față (dlib)...", end=" ")
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            print("❌ Nicio față detectată")
            continue
        
        print(f"✓ Găsite {len(faces)} față/fețe")
        
        # Procesează prima față
        face = faces[0]
        
        print("   🎯 Extragere 68 landmark-uri...", end=" ")
        time.sleep(0.3)
        
        landmarks = predictor(gray, face)
        
        print("✓")
        
        # Extrage coordonatele
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            
            # Desenează punctele pe imagine
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        # Salvează în listă
        row_data = {
            'imagine': image_name,
            'numar_landmarks': 68
        }
        
        for i, (x, y) in enumerate(landmarks_points):
            row_data[f'landmark_{i+1}_x'] = x
            row_data[f'landmark_{i+1}_y'] = y
        
        all_landmarks_data.append(row_data)
        
        print(f"   ✓ Extrase 68 puncte cheie")
        
    else:
        # Folosește OpenCV pentru detectare simplă (ochi + nas)
        print("   🔍 Detectare față (OpenCV)...", end=" ")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print("❌ Nicio față detectată")
            continue
        
        print("✓")
        
        (x, y, w, h) = faces[0]
        
        # Desenează dreptunghi în jurul feței
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Estimăm pozițiile aproximative ale trăsăturilor faciale
        # (metoda simplificată - nu e precisă ca dlib)
        
        print("   🎯 Estimare puncte cheie simple...", end=" ")
        time.sleep(0.3)
        
        # Puncte estimate (bazate pe proporții tipice ale feței)
        landmarks_points = []
        
        # Ochi stâng (aproximativ)
        left_eye_x = x + int(w * 0.3)
        left_eye_y = y + int(h * 0.4)
        landmarks_points.append(('ochi_stang', left_eye_x, left_eye_y))
        cv2.circle(img, (left_eye_x, left_eye_y), 3, (0, 255, 0), -1)
        
        # Ochi drept
        right_eye_x = x + int(w * 0.7)
        right_eye_y = y + int(h * 0.4)
        landmarks_points.append(('ochi_drept', right_eye_x, right_eye_y))
        cv2.circle(img, (right_eye_x, right_eye_y), 3, (0, 255, 0), -1)
        
        # Vârf nas
        nose_x = x + int(w * 0.5)
        nose_y = y + int(h * 0.6)
        landmarks_points.append(('nas', nose_x, nose_y))
        cv2.circle(img, (nose_x, nose_y), 3, (0, 255, 0), -1)
        
        # Colțuri gură
        mouth_left_x = x + int(w * 0.35)
        mouth_left_y = y + int(h * 0.75)
        landmarks_points.append(('gura_stanga', mouth_left_x, mouth_left_y))
        cv2.circle(img, (mouth_left_x, mouth_left_y), 3, (0, 255, 0), -1)
        
        mouth_right_x = x + int(w * 0.65)
        mouth_right_y = y + int(h * 0.75)
        landmarks_points.append(('gura_dreapta', mouth_right_x, mouth_right_y))
        cv2.circle(img, (mouth_right_x, mouth_right_y), 3, (0, 255, 0), -1)
        
        print("✓")
        
        # Salvează în listă
        row_data = {
            'imagine': image_name,
            'numar_landmarks': len(landmarks_points),
            'face_x': x,
            'face_y': y,
            'face_width': w,
            'face_height': h
        }
        
        for name, lx, ly in landmarks_points:
            row_data[f'{name}_x'] = lx
            row_data[f'{name}_y'] = ly
        
        all_landmarks_data.append(row_data)
        
        print(f"   ✓ Extrase {len(landmarks_points)} puncte cheie")
    
    # Afișează imaginea cu landmark-uri
    print(f"   👁️  Afișare rezultat (2 secunde)...")
    cv2.imshow(f'Landmarks: {image_name}', img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    time.sleep(0.3)

# PASUL 3: Salvează în CSV
print("\n" + "="*70)
print("💾 SALVARE ÎN CSV")
print("="*70)
time.sleep(0.5)

if len(all_landmarks_data) == 0:
    print("\n❌ Nu s-au găsit date de salvat!")
    exit()

csv_filename = "face_landmarks.csv"

print(f"\n📝 Scriere date în '{csv_filename}'...")
time.sleep(0.3)

# Determină toate coloanele (pentru a avea un header complet)
all_keys = set()
for row in all_landmarks_data:
    all_keys.update(row.keys())

all_keys = sorted(all_keys)

# Scrie CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_keys)
    
    writer.writeheader()
    writer.writerows(all_landmarks_data)

print(f"✓ Date salvate cu succes!")
time.sleep(0.3)

# PASUL 4: Rezumat
print("\n" + "="*70)
print("📊 REZUMAT")
print("="*70)

print(f"\n✅ Procesate: {len(all_landmarks_data)} imagini")
print(f"💾 Fișier CSV: {csv_filename}")
print(f"📋 Total coloane: {len(all_keys)}")

print(f"\n📸 Detalii imagini procesate:")
for data in all_landmarks_data:
    img_name = data['imagine']
    num_landmarks = data['numar_landmarks']
    print(f"   • {img_name:40} → {num_landmarks} landmark-uri")

print(f"\n💡 Poți deschide '{csv_filename}' în Excel sau orice editor CSV")
print("\n" + "="*70)
print("Proces finalizat!")
print("="*70)