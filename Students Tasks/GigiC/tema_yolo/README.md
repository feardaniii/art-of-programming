# Tema 39-40: Comparație Modele YOLO

## Descriere

Analiză comparativă a modelelor YOLOv5 (medium, large, extra-large) pentru detecție obiecte în timp real pe stream video de la cameră IP.

## Cerințe Temă

1. Conectare la flux video de la cameră IP
2. Testare modele: yolov5m, yolov5l, yolov5x
3. Comparație performanță (acuratețe + viteză)
4. Salvare imagini cu obiecte detectate (cu timestamp)

## Tehnologii

- Python 3.11
- YOLOv5 (ultralytics)
- OpenCV
- Iriun Webcam (stream iPhone ca IP camera)

## Instalare

**Prerequisite:**
- Python 3.11+
- Iriun Webcam: https://iriun.com/

**Comenzi:**
```bash
pip install -r requirements.txt
```

Modelele YOLO se descarcă automat la prima rulare.

## Utilizare

**Conectare cameră:**
1. Instalează Iriun pe iPhone și PC
2. Conectează pe aceeași WiFi
3. Pornește aplicația pe ambele

**Rulare:**
```bash
python tema_yolo.py
```

**Oprire:** Apasă `q` în fereastra video.

## Rezultate

### Performanță (iPhone 14 via Iriun, CPU)

| Model | FPS | Timp Inferență | Imagini Salvate | Obiecte Detectate | Obiecte/Imagine |
|-------|-----|----------------|-----------------|-------------------|-----------------|
| yolov5m | 2.56 | 363ms | 154 | 2145 | 13.9 |
| yolov5l | 1.43 | 679ms | 86 | 1542 | 17.9 |
| yolov5x | 0.82 | 1204ms | 49 | 921 | 18.8 |

### Concluzii

**Trade-off viteză vs acuratețe:**
- yolov5m: cel mai rapid (2.56 FPS), cele mai multe detecții (154 imagini)
- yolov5l: viteză medie, detecții moderate
- yolov5x: cel mai lent (0.82 FPS), cele mai puține detecții (49 imagini)

**Sensibilitate detecție:**
- yolov5m: mai agresiv, detectează mai multe scene (threshold permisiv)
- yolov5x: mai conservator, detectează doar când e sigur (mai puține false positives)

**Scalare timp inferență:**
```
yolov5m → yolov5l: +87% timp (+315ms)
yolov5l → yolov5x: +77% timp (+525ms)
```
Creștere exponențială, nu liniară.

**Aplicabilitate real-time:**
- Niciun model nu atinge 30 FPS pe CPU
- Cu GPU (NVIDIA RTX 3060): yolov5m ar atinge ~60 FPS

**Recomandare:** yolov5m oferă cel mai bun raport viteză/detecții pentru aplicații CPU.

## Structură Proiect
```
39_40_YOLO/
├── tema_yolo.py                    # Script principal
├── requirements.txt                # Dependențe
├── README.md                       # Documentație
├── detected_images/                # Imagini cu detecții
│   ├── yolov5m/
│   ├── yolov5l/
│   └── yolov5x/
└── comparatie_XXXXXXX.txt          # Raport comparativ
```

## Configurare

**Modificare durată test** (în `tema_yolo.py`, linia finală):
```python
compare_models(video_source=1, test_duration=60)  # 60 secunde
```

**Schimbare sursă video:**
```python
compare_models(video_source=0, ...)  # 0=laptop, 1=iPhone
```

**Modele disponibile:**
- yolov5m - Medium (21M parametri, 40MB)
- yolov5l - Large (46M parametri, 90MB)
- yolov5x - Extra-large (86M parametri, 170MB)

## Troubleshooting

**Nu pot conecta la cameră:**
- Verifică că Iriun rulează pe iPhone și PC
- Ambele dispozitive pe aceeași WiFi
- Testează cu `video_source=0` (camera laptop)

**Eroare "No module named 'ultralytics'":**
```bash
pip install -r requirements.txt
```

**FPS foarte mic (<1):**
Normal pe CPU. Pentru performanță mai bună:
- Folosește GPU (NVIDIA CUDA)
- Testează yolov5n (nano - mai rapid)

## Referințe

- YOLOv5 Documentation: https://docs.ultralytics.com/yolov5/
- Ultralytics GitHub: https://github.com/ultralytics/yolov5
- OpenCV: https://docs.opencv.org/
- Iriun Webcam: https://iriun.com/

## Autor

Gigione  
Tema 39-40 - Machine Learning | YOLO  
Februarie 2026

Proiect educațional - Curs Skillbrain Python & Machine Learning
