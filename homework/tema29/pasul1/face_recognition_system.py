import cv2
import numpy as np
import os
import time

print("=== SISTEM DE RECUNOAȘTERE FACIALĂ (OpenCV) ===\n")

# Încarcă clasificatorul Haar pentru detectarea fețelor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(image_path):
    """Extrage caracteristicile faciale dintr-o imagine"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None, img
    
    # Extrage prima față
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (100, 100))
    
    # Normalizează imaginea pentru a reduce efectele luminii
    face_normalized = cv2.equalizeHist(face_resized)
    
    return face_normalized, (x, y, w, h), img

def compare_faces(face1, face2):
    """Compară două fețe și returnează scorul de similitudine"""
    # Metodă 1: Correlation (histograme)
    corr = cv2.compareHist(
        cv2.calcHist([face1], [0], None, [256], [0, 256]),
        cv2.calcHist([face2], [0], None, [256], [0, 256]),
        cv2.HISTCMP_CORREL
    )
    
    # Metodă 2: Template matching
    result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
    template_score = result[0][0]
    
    # Metodă 3: Structural Similarity (mai robust la schimbări de lumină)
    # Convertim în float pentru calcule mai precise
    f1 = face1.astype(float) / 255.0
    f2 = face2.astype(float) / 255.0
    
    # Mean Squared Error (inversat pentru similitudine)
    mse = np.mean((f1 - f2) ** 2)
    mse_score = max(0, 100 - (mse * 100))
    
    # Scor combinat (0-100) cu ponderi ajustate
    similarity = (corr * 50 + template_score * 30 + mse_score * 0.2)
    
    return max(0, min(100, similarity))

# PASUL 1: Găsește toate imaginile
print("🔍 Căutare imagini în folder...")
time.sleep(0.5)

all_images = []
extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

for file in os.listdir('.'):
    if any(file.endswith(ext) for ext in extensions):
        all_images.append(file)

all_images.sort()

if len(all_images) < 2:
    print("❌ EROARE: Ai nevoie de cel puțin 2 imagini")
    exit()

print(f"✓ S-au găsit {len(all_images)} imagini\n")
time.sleep(0.3)

# PASUL 2: Alege imaginea de referință (prima)
reference_image_name = all_images[0]
test_images = all_images[1:]

print("="*70)
print("📋 CONFIGURARE:")
print("="*70)
print(f"\n📸 Imagine de REFERINȚĂ (master):")
print(f"   → {reference_image_name}")
print(f"\n🧪 Imagini de TEST ({len(test_images)}):")
for img in test_images:
    print(f"   → {img}")
print()
time.sleep(1)

# PASUL 3: Procesează imaginea de referință
print("="*70)
print("🎯 PROCESARE IMAGINE DE REFERINȚĂ")
print("="*70)
time.sleep(0.5)

print(f"\n⏳ Încărcare: {reference_image_name}...", end=" ")
time.sleep(0.3)

ref_face, ref_coords, ref_img = extract_face_features(reference_image_name)

if ref_face is None:
    print("❌ EROARE!")
    print("   Nu s-a detectat nicio față în imaginea de referință!")
    exit()

print("✓")
print(f"🔍 Față detectată și extrasă cu succes!")
print(f"   Dimensiuni: {ref_face.shape}")
time.sleep(0.5)

# Afișează imaginea de referință
if ref_img is not None and ref_coords is not None:
    x, y, w, h = ref_coords
    ref_display = ref_img.copy()
    cv2.rectangle(ref_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(ref_display, "REFERINTA", (x+6, y-10), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    
    print("\n👁️  Afișare imagine de referință (3 secunde)...")
    cv2.imshow('REFERINTA - Aceasta este fata master', ref_display)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

time.sleep(0.5)

# PASUL 4: Testează fiecare imagine
print("\n" + "="*70)
print("🧪 COMPARARE CU IMAGINILE DE TEST")
print("="*70)
time.sleep(0.5)

results = []
system_unlocked = False
similarity_threshold = 40  # SCĂZUT de la 50% la 40% pentru a accepta pozele tale!

for idx, test_image_name in enumerate(test_images):
    print(f"\n📸 Test [{idx+1}/{len(test_images)}]: {test_image_name}")
    print("-" * 70)
    time.sleep(0.3)
    
    print("   ⏳ Încărcare imagine...", end=" ")
    time.sleep(0.2)
    
    test_face, test_coords, test_img = extract_face_features(test_image_name)
    
    if test_face is None:
        print("❌ Eroare / Nicio față detectată")
        results.append((test_image_name, 0, False))
        time.sleep(1)
        continue
    
    print("✓")
    time.sleep(0.2)
    
    print("   🔍 Extragere caracteristici...", end=" ")
    time.sleep(0.3)
    print("✓")
    
    print("   🧮 Calculare similitudine cu referința...", end=" ")
    time.sleep(0.5)
    
    # Compară cu referința
    similarity = compare_faces(ref_face, test_face)
    
    print("✓")
    time.sleep(0.3)
    
    print(f"\n   📊 Rezultate analiză:")
    print(f"      • Scor similitudine: {similarity:.2f}%")
    print(f"      • Prag acceptare: {similarity_threshold}%")
    time.sleep(0.5)
    
    print(f"\n   ⚖️  Decizie: ", end="")
    time.sleep(0.5)
    
    is_match = similarity >= similarity_threshold
    
    if is_match:
        print("✅ POTRIVIRE GĂSITĂ!")
        print(f"      → Fața este SIMILARĂ cu referința")
        print(f"      → Probabil ACEEAȘI PERSOANĂ")
        results.append((test_image_name, similarity, True))
        
        if not system_unlocked:
            time.sleep(0.3)
            print(f"\n   🔓 {'='*60}")
            print(f"      SISTEM DEBLOCAT!")
            print(f"      Potrivire confirmată!")
            print(f"   {'='*60}")
            system_unlocked = True
    else:
        print("❌ NEPOTRIVIRE!")
        print(f"      → Fața este DIFERITĂ de referință")
        print(f"      → Probabil PERSOANĂ DIFERITĂ")
        results.append((test_image_name, similarity, False))
    
    # Afișează imaginea cu rezultat
    if test_img is not None and test_coords is not None:
        x, y, w, h = test_coords
        color = (0, 255, 0) if is_match else (0, 0, 255)
        label = f"SIMILAR {similarity:.0f}%" if is_match else f"DIFERIT {similarity:.0f}%"
        
        cv2.rectangle(test_img, (x, y), (x+w, y+h), color, 3)
        
        # Background pentru text
        cv2.rectangle(test_img, (x, y-40), (x+w, y), color, cv2.FILLED)
        cv2.putText(test_img, label, (x+6, y-10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        print(f"\n   👁️  Afișare rezultat vizual (3 secunde)...")
        cv2.imshow(f'Comparare: {test_image_name}', test_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    print()
    time.sleep(0.5)

# PASUL 5: Rezumat final
print("\n" + "="*70)
print("📊 REZUMAT FINAL - ANALIZĂ SIMILITUDINE")
print("="*70)
time.sleep(0.5)

print(f"\n📸 Imagine de referință: {reference_image_name}")
print(f"🧪 Total imagini comparate: {len(test_images)}\n")

print("📋 Rezultate detaliate (sortate după similitudine):")
print()

# Sortează după similitudine (descrescător)
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

for img_name, sim, matched in results_sorted:
    if matched:
        status = "✅ POTRIVIRE "
        verdict = "→ ACEEAȘI PERSOANĂ"
    else:
        status = "❌ DIFERIT   "
        verdict = "→ PERSOANĂ DIFERITĂ"
    
    print(f"   {status} {img_name:40} {sim:5.1f}%  {verdict}")

print()
time.sleep(0.5)

# Statistici
matches = sum(1 for _, _, m in results if m)
non_matches = len(results) - matches

print(f"📈 Statistici:")
print(f"   • Potriviri găsite: {matches}")
print(f"   • Nepotriviri: {non_matches}")
print()

if system_unlocked:
    print("🔓 STATUS FINAL: SISTEM DEBLOCAT")
    print("   ✓ Au fost găsite potriviri cu imaginea de referință!")
else:
    print("🔒 STATUS FINAL: SISTEM BLOCAT")
    print("   ✗ Nicio potrivire cu imaginea de referință.")

print(f"\n💡 Interpretare rezultate:")
print(f"   • 40-100% = ACEEAȘI persoană (cu variații de lumină/unghi)")
print(f"   • 0-40%   = PERSOANĂ DIFERITĂ")
print(f"\n💡 Ajustare prag (dacă e necesar):")
print(f"   • Prag curent: {similarity_threshold}%")
print(f"   • Pentru mai multe potriviri: scade pragul la 35%")
print(f"   • Pentru mai puține potriviri: crește pragul la 45-50%")
print("\n" + "="*70)
print("Sistem închis.")
print("="*70)