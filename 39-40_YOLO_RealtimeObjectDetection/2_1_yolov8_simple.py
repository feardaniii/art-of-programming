"""
YOLOv8: Real-Time Object Detection (Minimal Version)
=====================================================

What this does:
    1. Load YOLOv8 model (pre-trained on 80 object classes)
    2. Open webcam
    3. Detect objects in each frame
    4. Draw bounding boxes + labels
    5. Display result

That's it. No fancy analytics, no zones, no tracking.

Installation:
    pip install ultralytics opencv-python

Press 'Q' to quit.
"""

import cv2
from ultralytics import YOLO

# Load model (downloads automatically on first run)
model = YOLO("yolov8n.pt")  # nano = fastest, ~30 FPS on CPU

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame, verbose=False)[0]

    # Draw boxes and labels on frame
    annotated = results.plot()

    # Show result
    cv2.imshow("YOLOv8 Detection", annotated)

    # Quit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()