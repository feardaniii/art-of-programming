import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os

# ============================================================================
# TEMA 39-40: YOLO - Comparație Modele (yolov5m, yolov5l, yolov5x)
# Autor: Gigione
# Data: Februarie 2026
# ============================================================================


def test_yolo_model(model_name, video_source=0, duration_seconds=60):
    """
    Testează un model YOLO specific și salvează imagini cu obiecte detectate.
    
    Parametri:
        model_name (str): Numele modelului ('yolov5m', 'yolov5l', 'yolov5x')
        video_source (int): Index cameră (0=laptop, 1=iPhone via Iriun)
        duration_seconds (int): Durata testului în secunde
    
    Returns:
        dict: Dicționar cu metrici (FPS, inferență, imagini salvate, etc.)
    """
    
    # ========================================================================
    # 1. CONFIGURARE FOLDERE ȘI PATHS
    # ========================================================================
    # Determină locația scriptului și creează folder pentru imagini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(script_dir, "detected_images", model_name)
    os.makedirs(images_folder, exist_ok=True)  # Creează folder dacă nu există
    
    # Afișare informații inițiale
    print(f"\n{'='*60}")
    print(f"🔍 Model: {model_name}")
    print(f"📁 Imagini: {images_folder}")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # 2. ÎNCĂRCARE MODEL YOLO
    # ========================================================================
    # Încarcă modelul pre-antrenat (descarcă automat dacă nu există)
    model = YOLO(f'{model_name}.pt')
    
    # ========================================================================
    # 3. CONECTARE CAMERĂ VIDEO
    # ========================================================================
    # Deschide stream video (0=camera laptop, 1=iPhone via Iriun)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Camera nu merge!")
        return None
    
    # ========================================================================
    # 4. INIȚIALIZARE METRICI ȘI CONTOARE
    # ========================================================================
    frame_count = 0              # Total frame-uri procesate
    saved_images = 0             # Imagini salvate (cu detecții)
    total_objects = 0            # Total obiecte detectate
    total_inference_time = 0     # Timp total pentru inferență
    class_counts = {}            # Dicționar pentru contorizare clase
    
    start_time = time.time()     # Timestamp început test
    
    print(f"🎬 Start - {duration_seconds}s | Apasă 'q' pentru stop\n")
    
    # ========================================================================
    # 5. BUCLĂ PRINCIPALĂ - PROCESARE VIDEO
    # ========================================================================
    while True:
        # 5.1 Verificare timeout (dacă s-a depășit durata testului)
        if time.time() - start_time > duration_seconds:
            break
        
        # 5.2 Citire frame din stream video
        ret, frame = cap.read()
        if not ret:  # Dacă nu poate citi frame, oprește
            break
        
        frame_count += 1
        
        # ====================================================================
        # 5.3 DETECȚIE OBIECTE CU YOLO + MĂSURARE TIMP INFERENȚĂ
        # ====================================================================
        inf_start = time.time()                    # Timestamp început inferență
        results = model(frame, verbose=False)      # Rulează detecția YOLO
        inf_time = time.time() - inf_start         # Calculează timp inferență
        total_inference_time += inf_time           # Adună la total
        
        # Numără obiecte detectate în frame-ul curent
        num_objects = len(results[0].boxes)
        
        # ====================================================================
        # 5.4 SALVARE IMAGINE DACĂ SUNT OBIECTE DETECTATE
        # ====================================================================
        if num_objects > 0:
            total_objects += num_objects
            
            # Generare nume fișier cu timestamp și număr obiecte
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{model_name}_{timestamp}_{num_objects}obj.jpg"
            filepath = os.path.join(images_folder, filename)
            
            # Desenează bounding boxes pe imagine
            annotated = results[0].plot()
            
            # Salvează imaginea pe disk
            cv2.imwrite(filepath, annotated)
            saved_images += 1
            
            # Contorizează clase detectate (person, car, etc.)
            for box in results[0].boxes:
                cls = results[0].names[int(box.cls)]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # Afișează progres la fiecare 10 imagini
            if saved_images % 10 == 0:
                print(f"💾 {saved_images} imagini | {total_objects} obiecte")
        
        # ====================================================================
        # 5.5 AFIȘARE LIVE PREVIEW CU STATISTICI
        # ====================================================================
        annotated = results[0].plot()  # Frame cu bounding boxes
        
        # Calculează FPS instantaneu
        fps = 1/inf_time if inf_time > 0 else 0
        
        # Adaugă text pe imagine (nume model, FPS, imagini salvate)
        cv2.putText(annotated, f"{model_name} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Salvate: {saved_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Afișează fereastra video
        cv2.imshow(model_name, annotated)
        
        # Verifică dacă utilizatorul apasă 'q' pentru stop manual
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ========================================================================
    # 6. CLEANUP - ELIBERARE RESURSE
    # ========================================================================
    cap.release()              # Închide stream video
    cv2.destroyAllWindows()    # Închide ferestre OpenCV
    
    # ========================================================================
    # 7. CALCULARE METRICI FINALE
    # ========================================================================
    total_time = time.time() - start_time  # Timp total rulare
    
    # FPS mediu = frame-uri procesate / timp total
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    # Timp inferență mediu în milisecunde
    avg_inf_ms = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
    
    # Rata detecție = procent frame-uri cu obiecte detectate
    detection_rate = (saved_images / frame_count * 100) if frame_count > 0 else 0
    
    # ========================================================================
    # 8. AFIȘARE REZUMAT REZULTATE
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"📊 {model_name}")
    print(f"{'='*60}")
    print(f"⚡ FPS mediu: {avg_fps:.2f}")
    print(f"🕐 Inferență: {avg_inf_ms:.2f}ms")
    print(f"🎯 Detecții: {saved_images}/{frame_count} ({detection_rate:.1f}%)")
    print(f"📦 Obiecte: {total_objects}")
    print(f"💾 Imagini: {saved_images}")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # 9. RETURN REZULTATE CA DICȚIONAR
    # ========================================================================
    return {
        'model_name': model_name,
        'avg_fps': avg_fps,
        'avg_inference_ms': avg_inf_ms,
        'saved_images': saved_images,
        'total_objects': total_objects,
        'detection_rate': detection_rate,
        'class_counts': class_counts
    }


def compare_models(video_source=0, test_duration=60):
    """
    Compară performanța celor 3 modele YOLO cerute în temă.
    
    Parametri:
        video_source (int): Index cameră (0=laptop, 1=iPhone)
        test_duration (int): Durata testului per model (în secunde)
    
    Returns:
        list: Listă cu rezultate pentru fiecare model
    """
    
    # ========================================================================
    # 1. CONFIGURARE MODELE DE TESTAT
    # ========================================================================
    models = ['yolov5m', 'yolov5l', 'yolov5x']  # Modelele din temă
    results = []  # Listă pentru stocare rezultate
    
    # ========================================================================
    # 2. AFIȘARE INFORMAȚII INIȚIALE
    # ========================================================================
    print("\n" + "="*60)
    print("🚀 TEMA 39-40: COMPARAȚIE MODELE")
    print("="*60)
    print(f"Modele: {', '.join(models)}")
    print(f"Durată/model: {test_duration}s")
    print("="*60 + "\n")
    
    # Așteaptă confirmarea utilizatorului
    input("ENTER pentru start...")
    
    # ========================================================================
    # 3. TESTARE SECVENȚIALĂ A MODELELOR
    # ========================================================================
    for model in models:
        # Testează modelul curent
        result = test_yolo_model(model, video_source, test_duration)
        
        # Salvează rezultatul dacă testul a reușit
        if result:
            results.append(result)
        
        # Pauză între modele (pentru a lăsa camera/GPU să se reseteze)
        print("\n⏸️  Pauză 3s...\n")
        time.sleep(3)
    
    # ========================================================================
    # 4. GENERARE RAPORT COMPARATIV TXT
    # ========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(script_dir, f"comparatie_{timestamp}.txt")
    
    # Scriere raport în fișier text
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header raport
        f.write("="*70 + "\n")
        f.write("TEMA 39-40: COMPARAȚIE MODELE YOLO\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Tabel comparativ
        f.write(f"{'Model':<12} {'FPS':<10} {'Inferență':<15} {'Imagini':<12} {'Obiecte':<10}\n")
        f.write("-"*70 + "\n")
        
        # Scriere date pentru fiecare model
        for r in results:
            f.write(f"{r['model_name']:<12} "
                   f"{r['avg_fps']:<10.2f} "
                   f"{r['avg_inference_ms']:<15.2f}ms "
                   f"{r['saved_images']:<12} "
                   f"{r['total_objects']:<10}\n")
        
        # Concluzii
        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUZIE:\n")
        f.write("-"*70 + "\n")
        
        # Găsește cel mai rapid model (FPS maxim)
        fastest = max(results, key=lambda x: x['avg_fps'])
        
        # Găsește modelul cu cele mai multe detecții
        most_detections = max(results, key=lambda x: x['saved_images'])
        
        # Scriere concluzii
        f.write(f"Cel mai RAPID: {fastest['model_name']} ({fastest['avg_fps']:.2f} FPS)\n")
        f.write(f"Cele mai multe DETECȚII: {most_detections['model_name']} ({most_detections['saved_images']} imagini)\n")
    
    # ========================================================================
    # 5. AFIȘARE CONFIRMARE SALVARE RAPORT
    # ========================================================================
    print(f"\n✅ Raport: {report_file}\n")
    
    # ========================================================================
    # 6. RETURN REZULTATE
    # ========================================================================
    return results


# ============================================================================
# PUNCT DE INTRARE - RULARE SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Configurare parametri:
    # video_source=1 → iPhone via Iriun (0=laptop camera)
    # test_duration=60 → 60 secunde per model (total ~3 minute)
    compare_models(video_source=1, test_duration=60)