import cv2
import time
import threading
import csv
from ultralytics import YOLO

# ================= CONFIG =================

URL = "http://192.168.100.229:8080/video"

MODELS = [
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt"
]

TEST_DURATION = 20      # seconds per model
IMG_SIZE = 416

# ===========================================

cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
running = True


def grab_frames():
    global latest_frame
    while running:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame


threading.Thread(target=grab_frames, daemon=True).start()

results_table = []

for model_name in MODELS:

    print(f"\n===== Benchmarking {model_name} =====")

    model = YOLO(model_name)

    start_time = time.time()

    frames_processed = 0
    total_detections = 0
    confidence_sum = 0
    confidence_count = 0

    while time.time() - start_time < TEST_DURATION:

        if latest_frame is None:
            continue

        frame = latest_frame.copy()
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        results = model(frame, imgsz=IMG_SIZE, verbose=False)

        if results[0].boxes is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            total_detections += len(confs)
            confidence_sum += confs.sum()
            confidence_count += len(confs)

        frames_processed += 1

    avg_fps = frames_processed / TEST_DURATION
    mean_conf = confidence_sum / confidence_count if confidence_count else 0

    print(f"FPS: {avg_fps:.2f}")
    print(f"Detections: {total_detections}")
    print(f"Mean Confidence: {mean_conf:.3f}")

    results_table.append([
        model_name,
        round(avg_fps, 2),
        total_detections,
        round(mean_conf, 3)
    ])

running = False
cap.release()

# Save CSV
with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "avg_fps", "total_detections", "mean_confidence"])
    writer.writerows(results_table)

print("\n✅ Benchmark finished — results saved to benchmark_results.csv")