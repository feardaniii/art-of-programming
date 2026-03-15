import cv2
import time
import threading
import os
from ultralytics import YOLO

# ================= CONFIG =================

URL = "http://192.168.100.229:8080/video"
IMG_SIZE = 416
SAVE_INTERVAL = 1.0   # seconds between saves

# ==========================================

print("Working dir:", os.getcwd())

os.makedirs("detections", exist_ok=True)

model = YOLO("yolov5m.pt")

cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
running = True

# ⭐ cooldown init
last_save = 0


def grab_frames():
    global latest_frame
    while running and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            latest_frame = frame


threading.Thread(target=grab_frames, daemon=True).start()

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

prev = 0

while True:

    if latest_frame is None:
        continue

    original = latest_frame.copy()

    frame = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    results = model(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False)

    boxes = results[0].boxes
    detections = len(boxes) if boxes is not None else 0

    annotated = results[0].plot()

    # ⭐ SAVE WITH COOLDOWN
    if detections > 0 and time.time() - last_save > SAVE_INTERVAL:

        ts = int(time.time() * 1000)
        filename = os.path.join("detections", f"det_{ts}.jpg")

        ok = cv2.imwrite(filename, annotated)
        print("Saved:", ok, filename)

        last_save = time.time()

    # FPS
    now = time.time()
    fps = 1/(now-prev) if prev else 0
    prev = now

    cv2.putText(annotated, f"FPS {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
        break


running = False
time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()