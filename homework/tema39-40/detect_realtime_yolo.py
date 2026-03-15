import cv2
import time
import threading
from ultralytics import YOLO

url = "http://192.168.100.229:8080/video"

model = YOLO("yolov5m.pt")

cap = cv2.VideoCapture(url)
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

cv2.namedWindow("YOLO Realtime", cv2.WINDOW_NORMAL)

prev = 0

while True:

    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    frame = cv2.resize(frame, (416, 416))

    results = model(frame, imgsz=416, verbose=False)
    annotated = results[0].plot()

    now = time.time()
    fps = 1/(now-prev) if prev else 0
    prev = now

    cv2.putText(annotated, f"FPS {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLO Realtime", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.getWindowProperty("YOLO Realtime", cv2.WND_PROP_VISIBLE) < 1:
        break

running = False
cap.release()
cv2.destroyAllWindows()