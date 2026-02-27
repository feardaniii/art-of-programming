"""
AUTONOMOUS DRIVING - Inference Pipeline
========================================
Run your trained KITTI detector on images, video files, or live webcam.

Modes:
  1. Single image   — Load one photo, detect + draw overlay, save result
  2. KITTI samples   — Batch-process sample images from the training set
  3. Video file      — Process dashcam footage frame-by-frame with detection overlay
  4. Live webcam     — Real-time detection from your camera

Prerequisite: Run 1_train_detector.py first to produce 'av_perception_final.keras'

Requirements: pip install tensorflow opencv-python numpy
Run: python 2_inference.py
"""

import os
import sys
import time
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras


# ════════════════════════════════════════════════════════════════
# CONFIGURATION  (must match training script!)
# ════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR   = os.path.join(SCRIPT_DIR, "kitti dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "training", "image_2")

MODEL_PATH = os.path.join(SCRIPT_DIR, "av_perception_final.keras")

IMG_HEIGHT = 224
IMG_WIDTH  = 224

# Must match training
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
ID_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# Colors for bounding boxes (BGR for OpenCV)
CLASS_COLORS_BGR = {
    'Car':        (0, 0, 255),      # Red
    'Pedestrian': (255, 100, 0),    # Blue
    'Cyclist':    (0, 220, 0),      # Green
}

DEFAULT_COLOR = (0, 255, 255)  # Yellow fallback


# ════════════════════════════════════════════════════════════════
# DETECTION FUNCTION
# ════════════════════════════════════════════════════════════════

def detect_objects(image, model, conf_threshold=0.3):
    """
    Run the detection model on a single image (BGR, any size).

    Our model is a SINGLE-OBJECT LOCALIZER:
      - It classifies the MAIN object in the scene (Car / Pedestrian / Cyclist)
      - It predicts ONE bounding box for that object

    This is different from full object detectors like YOLO which find ALL objects.
    See MODEL_ARCHITECTURE.md for discussion.

    Args:
        image:          BGR image (numpy array, any resolution)
        model:          loaded Keras model
        conf_threshold: minimum confidence to accept detection (0-1)

    Returns:
        list of (class_name, confidence, (x1, y1, x2, y2)) tuples
        Length is 0 or 1 for our single-object model.
    """
    img_h, img_w = image.shape[:2]

    # Preprocess: resize, RGB, normalize
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_input = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_input, axis=0)

    # Predict (verbose=0 to suppress per-batch output)
    preds = model.predict(img_batch, verbose=0)

    # Extract classification
    class_probs = preds['classification'][0]
    class_id = int(np.argmax(class_probs))
    confidence = float(class_probs[class_id])
    class_name = ID_TO_CLASS[class_id]

    # Filter low-confidence detections
    if confidence < conf_threshold:
        return []

    # Extract and denormalize bounding box
    bbox_norm = preds['bbox'][0]
    x1 = int(np.clip(bbox_norm[0] * img_w, 0, img_w))
    y1 = int(np.clip(bbox_norm[1] * img_h, 0, img_h))
    x2 = int(np.clip(bbox_norm[2] * img_w, 0, img_w))
    y2 = int(np.clip(bbox_norm[3] * img_h, 0, img_h))

    return [(class_name, confidence, (x1, y1, x2, y2))]


# ════════════════════════════════════════════════════════════════
# DRAWING FUNCTION
# ════════════════════════════════════════════════════════════════

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on an image (modifies in-place).

    Args:
        image:      BGR image (will be modified)
        detections: list from detect_objects()

    Returns:
        The same image with boxes drawn on it
    """
    for class_name, confidence, (x1, y1, x2, y2) in detections:
        color = CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{class_name}: {confidence:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y1 = max(y1 - label_h - 10, 0)
        cv2.rectangle(image, (x1, label_y1), (x1 + label_w + 4, y1), color, -1)

        # Label text
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def draw_fps(image, fps):
    """Draw FPS counter in the top-left corner."""
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


# ════════════════════════════════════════════════════════════════
# MODE 1: SINGLE IMAGE
# ════════════════════════════════════════════════════════════════

def process_single_image(model, img_path):
    """Load one image, detect objects, save annotated result."""
    if not os.path.exists(img_path):
        print(f"  File not found: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"  Could not read image: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"  Image: {os.path.basename(img_path)}  ({w}x{h})")

    # Detect
    start = time.time()
    detections = detect_objects(img, model)
    inference_ms = (time.time() - start) * 1000

    # Draw
    result = draw_detections(img.copy(), detections)

    # Save
    base = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(SCRIPT_DIR, f"detection_{base}.png")
    cv2.imwrite(output_path, result)

    print(f"  Inference time: {inference_ms:.1f} ms")
    if detections:
        for cls, conf, box in detections:
            print(f"  Detected: {cls} ({conf:.0%})  box={box}")
    else:
        print(f"  No confident detection (threshold too high?)")
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════
# MODE 2: KITTI SAMPLE BATCH
# ════════════════════════════════════════════════════════════════

def process_kitti_samples(model):
    """Process a set of KITTI training images to verify the model works."""
    if not os.path.isdir(TRAIN_IMG_DIR):
        print(f"  KITTI images not found at: {TRAIN_IMG_DIR}")
        return

    sample_ids = ['000000', '000010', '000050', '000100', '000200',
                  '000500', '001000', '002000', '003000', '005000']

    print(f"  Processing {len(sample_ids)} KITTI samples...\n")

    for img_id in sample_ids:
        img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
        if not os.path.exists(img_path):
            print(f"  [{img_id}] Not found, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Detect
        start = time.time()
        detections = detect_objects(img, model)
        ms = (time.time() - start) * 1000

        # Draw and save
        result = draw_detections(img.copy(), detections)
        output_path = os.path.join(SCRIPT_DIR, f"detection_{img_id}.png")
        cv2.imwrite(output_path, result)

        det_str = detections[0][0] + f" ({detections[0][1]:.0%})" if detections else "none"
        print(f"  [{img_id}]  {ms:5.1f} ms  |  detection: {det_str}  |  saved: {os.path.basename(output_path)}")

    print(f"\n  Done! Check detection_*.png files in the project folder.")


# ════════════════════════════════════════════════════════════════
# MODE 3: VIDEO FILE
# ════════════════════════════════════════════════════════════════

def process_video(model, video_path, output_path=None):
    """
    Process a video file frame-by-frame with detection overlay.

    Works with any video: dashcam footage, KITTI sequences, YouTube downloads, etc.
    The output video will have bounding boxes and class labels drawn on every frame.
    """
    if not os.path.exists(video_path):
        print(f"  File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(SCRIPT_DIR, f"{base}_detected.mp4")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"  Input:  {os.path.basename(video_path)}")
    print(f"  Size:   {width}x{height} @ {fps} FPS")
    print(f"  Frames: {total_frames}")
    print(f"  Output: {output_path}")
    print()

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect + draw
        detections = detect_objects(frame, model)
        frame_out = draw_detections(frame.copy(), detections)

        # FPS counter
        elapsed = time.time() - start_time
        current_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
        draw_fps(frame_out, current_fps)

        out.write(frame_out)
        frame_count += 1

        # Progress every 30 frames
        if frame_count % 30 == 0:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            det_str = detections[0][0] if detections else "-"
            print(f"  Frame {frame_count:5d}/{total_frames}  ({pct:5.1f}%)  "
                  f"FPS: {current_fps:.1f}  Detection: {det_str}")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print()
    print(f"  Video processing complete!")
    print(f"  Processed {frame_count} frames in {total_time:.1f}s  (avg {avg_fps:.1f} FPS)")
    print(f"  Output saved: {output_path}")


# ════════════════════════════════════════════════════════════════
# MODE 4: LIVE WEBCAM
# ════════════════════════════════════════════════════════════════

def process_webcam(model, camera_index=0):
    """
    Real-time detection from webcam feed.
    Press 'q' to quit, 's' to save current frame.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  Could not open webcam (index {camera_index})")
        print("  Make sure a camera is connected and not in use by another app.")
        return

    print(f"  Webcam opened (index {camera_index})")
    print(f"  Press 'q' to quit, 's' to save a snapshot")
    print()

    fps_history = []
    frame_num = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("  Lost webcam feed")
            break

        # Detect + draw
        detections = detect_objects(frame, model)
        frame_out = draw_detections(frame.copy(), detections)

        # FPS
        elapsed = time.time() - start
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(current_fps)
        avg_fps = np.mean(fps_history[-30:])  # 30-frame rolling average
        draw_fps(frame_out, avg_fps)

        # Display
        cv2.imshow('Autonomous Driving Detector  (q=quit, s=save)', frame_out)
        frame_num += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap_path = os.path.join(SCRIPT_DIR, f"webcam_snapshot_{frame_num:05d}.png")
            cv2.imwrite(snap_path, frame_out)
            print(f"  Snapshot saved: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()

    if fps_history:
        print(f"\n  Session: {frame_num} frames, avg FPS: {np.mean(fps_history):.1f}")


# ════════════════════════════════════════════════════════════════
# MAIN MENU
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("   AUTONOMOUS DRIVING — Inference Pipeline")
    print("=" * 70)
    print()

    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at:")
        print(f"  {MODEL_PATH}")
        print()
        print("Run 1_train_detector.py first to train the model.")
        print("If you saved the model elsewhere, update MODEL_PATH in this script.")
        sys.exit(1)

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"  Loaded: {MODEL_PATH}")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output keys:  {list(model.output_names)}")
    print(f"  Parameters:   {model.count_params():,}")
    print()

    # --- Warm up the model (first prediction is slower due to compilation) ---
    dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    print("  Model warmed up (first prediction compiled)")
    print()

    # --- Interactive menu ---
    while True:
        print()
        print("=" * 60)
        print("  INFERENCE MENU")
        print("=" * 60)
        print("  1. Process a single image")
        print("  2. Process KITTI sample images (batch)")
        print("  3. Process a video file (dashcam, etc.)")
        print("  4. Live webcam detection")
        print("  5. Exit")
        print()

        choice = input("  Select mode (1-5): ").strip()

        if choice == '1':
            print()
            path = input("  Enter image path: ").strip().strip('"').strip("'")
            print()
            process_single_image(model, path)

        elif choice == '2':
            print()
            process_kitti_samples(model)

        elif choice == '3':
            print()
            path = input("  Enter video path: ").strip().strip('"').strip("'")
            output = input("  Output path (Enter for default): ").strip().strip('"').strip("'")
            print()
            process_video(model, path, output if output else None)

        elif choice == '4':
            print()
            cam_str = input("  Camera index (Enter for 0): ").strip()
            cam_idx = int(cam_str) if cam_str.isdigit() else 0
            print()
            process_webcam(model, cam_idx)

        elif choice == '5':
            print("\n  Goodbye!")
            break

        else:
            print("  Invalid choice. Enter 1-5.")


if __name__ == '__main__':
    main()
