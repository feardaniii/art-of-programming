#!/usr/bin/env python3
"""
================================================================================
STEP 7: RUN INFERENCE
================================================================================

Use your trained model to detect objects in real-time!

USAGE:
    # Webcam (default)
    python scripts/07_inference.py

    # Single image
    python scripts/07_inference.py --image path/to/image.jpg

    # Video file
    python scripts/07_inference.py --video path/to/video.mp4

    # Adjust confidence threshold
    python scripts/07_inference.py --conf 0.3

CONTROLS (webcam mode):
    [Q]     Quit
    [S]     Save screenshot
    [+/-]   Adjust confidence threshold
    [SPACE] Pause/resume

================================================================================
"""

import cv2
import time
from pathlib import Path
from collections import deque
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, print_header


def run_webcam_inference(model, conf_threshold: float):
    """Run real-time inference on webcam feed."""
    
    print("Starting webcam inference...")
    print()
    print("Controls:")
    print("  [Q]     Quit")
    print("  [S]     Save screenshot")
    print("  [+/-]   Adjust confidence")
    print("  [SPACE] Pause/resume")
    print()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_queue = deque(maxlen=30)
    paused = False
    frame = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        if frame is None:
            continue
        
        # Run inference
        start_time = time.time()
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        inference_time = time.time() - start_time
        
        # Calculate FPS
        fps_queue.append(1.0 / inference_time if inference_time > 0 else 0)
        avg_fps = sum(fps_queue) / len(fps_queue)
        
        # Get annotated frame
        annotated = results.plot()
        
        # Draw info overlay
        h, w = annotated.shape[:2]
        
        # Top bar
        cv2.rectangle(annotated, (0, 0), (w, 50), (40, 40, 40), -1)
        
        # FPS
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Confidence threshold
        cv2.putText(annotated, f"Conf: {conf_threshold:.2f}", (150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detection count
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(annotated, f"Detections: {num_detections}", (w - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Paused indicator
        if paused:
            cv2.putText(annotated, "PAUSED", (w // 2 - 60, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("YOLOv8 Custom Detection", annotated)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Saved: {filename}")
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()


def run_image_inference(model, image_path: str, conf_threshold: float):
    """Run inference on a single image."""
    
    print(f"Processing: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    # Run inference
    results = model(img, conf=conf_threshold, verbose=False)[0]
    
    # Print detections
    print()
    print("Detections:")
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"  {class_name}: {conf:.1%} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    else:
        print("  No objects detected")
    
    # Show image
    annotated = results.plot()
    cv2.imshow("Detection", annotated)
    
    print()
    print("Press any key to close, 'S' to save")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")
    
    cv2.destroyAllWindows()


def run_video_inference(model, video_path: str, conf_threshold: float):
    """Run inference on a video file."""
    
    print(f"Processing: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print()
    print("Press 'Q' to quit, 'S' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        annotated = results.plot()
        
        # Add frame counter
        cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Video Detection", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main inference function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument('--image', '-i', type=str, help="Image file to process")
    parser.add_argument('--video', '-v', type=str, help="Video file to process")
    parser.add_argument('--conf', '-c', type=float, help="Confidence threshold")
    parser.add_argument('--weights', '-w', type=str, help="Custom weights file")
    
    args = parser.parse_args()
    
    print_header("YOLOV8 INFERENCE")
    
    # Load configuration
    config = load_config()
    models_dir = Path(config['paths']['models'])
    
    # Find weights
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = models_dir / "best.pt"
        if not weights_path.exists():
            weights_path = Path(config['paths']['runs']) / "train" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}")
        print("Train a model first: python scripts/05_train.py")
        return
    
    print(f"Model: {weights_path}")
    
    # Load model
    model = YOLO(str(weights_path))
    
    print(f"Classes: {list(model.names.values())}")
    print()
    
    # Get confidence threshold
    conf_threshold = args.conf or config['inference']['confidence_threshold']
    print(f"Confidence threshold: {conf_threshold}")
    print()
    
    # Run appropriate mode
    if args.image:
        run_image_inference(model, args.image, conf_threshold)
    elif args.video:
        run_video_inference(model, args.video, conf_threshold)
    else:
        run_webcam_inference(model, conf_threshold)
    
    print()
    print("Inference complete!")


if __name__ == "__main__":
    main()
