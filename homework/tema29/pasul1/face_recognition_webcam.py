import face_recognition
import cv2
import numpy as np

# PASUL 1: Încarcă imaginea ta de referință
print("Se încarcă imaginea de referință...")
reference_image = face_recognition.load_image_file("eu.jpg")

# Extrage encodingul (caracteristicile) feței tale
reference_encoding = face_recognition.face_encodings(reference_image)[0]
print("Fața de referință încărcată cu succes!")

# PASUL 2: Pornește camera web
print("\nPornesc camera web...")
video_capture = cv2.VideoCapture(0)

print("\n=== SISTEM DE RECUNOAȘTERE FACIALĂ ===")
print("Poziționează-te în fața camerei...")
print("Apasă 'q' pentru a ieși\n")

system_unlocked = False

while True:
    # Captează un frame din camera web
    ret, frame = video_capture.read()
    
    if not ret:
        print("Eroare la citirea camerei!")
        break
    
    # Convertește imaginea din BGR (OpenCV) în RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Găsește toate fețele din frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Verifică fiecare față detectată
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compară fața detectată cu fața ta de referință
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        name = "Necunoscut"
        color = (0, 0, 255)  # Roșu pentru necunoscut
        
        # Calculează distanța (cât de similare sunt fețele)
        face_distance = face_recognition.face_distance([reference_encoding], face_encoding)
        
        # Dacă fața se potrivește
        if matches[0]:
            name = "ACCES PERMIS!"
            color = (0, 255, 0)  # Verde pentru recunoscut
            
            if not system_unlocked:
                system_unlocked = True
                print("\n✓ SISTEM DEBLOCAT!")
                print("Bine ai venit!")
        
        # Desenează un dreptunghi în jurul feței
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Afișează numele și scorul de similitudine
        similarity = (1 - face_distance[0]) * 100
        label = f"{name} ({similarity:.1f}%)"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    # Afișează statusul sistemului
    if system_unlocked:
        status_text = "STATUS: DEBLOCAT"
        status_color = (0, 255, 0)
    else:
        status_text = "STATUS: BLOCAT"
        status_color = (0, 0, 255)
    
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Afișează frame-ul
    cv2.imshow('Face Recognition System', frame)
    
    # Apasă 'q' pentru a ieși
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează resursele
video_capture.release()
cv2.destroyAllWindows()
print("\nSistem închis.")