"""
================================================================================
YOLO CUSTOM MODEL INFERENCE - Simple Webcam Demo
================================================================================

This script loads your trained model and runs inference on webcam.
No training - just detection with your custom model.

USAGE:
    python run_inference.py

Controls:
    - Press 'Q' to quit
    - Press 'S' to save current frame

================================================================================
"""

import cv2
from pathlib import Path
from ultralytics import YOLO


def find_best_weights(base_path: str = "runs") -> Path:
    """
    Search for the most recent best.pt file in the runs directory.

    This handles the case where training output might be in:
    - runs/train/shapes_detector/weights/best.pt
    - runs/detect/runs/train/shapes_detector/weights/best.pt
    - or any other nested structure
    """
    base = Path(base_path)

    # Search for all best.pt files
    weights_files = list(base.rglob("best.pt"))

    if not weights_files:
        raise FileNotFoundError(
            f"No 'best.pt' file found in {base_path}. "
            "Have you trained a model yet?"
        )

    # Get the most recently modified one
    latest_weights = max(weights_files, key=lambda p: p.stat().st_mtime)

    print(f"✓ Found trained model: {latest_weights}")
    print(f"  Modified: {latest_weights.stat().st_mtime}")
    print()

    return latest_weights


def run_webcam_inference(
        weights_path: Path,
        conf_threshold: float = 0.5,
        camera_index: int = 0
):
    """
    Run real-time object detection on webcam feed.

    Args:
        weights_path: Path to trained model weights (best.pt)
        conf_threshold: Minimum confidence to show detection (0.0 to 1.0)
        camera_index: Camera device index (0 for default webcam)
    """
    print()
    print("=" * 70)
    print("CUSTOM YOLO WEBCAM INFERENCE")
    print("=" * 70)
    print()
    print(f"Model: {weights_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Camera: {camera_index}")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save current frame")
    print("  + - Increase confidence threshold")
    print("  - - Decrease confidence threshold")
    print()

    # Load your custom trained model
    model = YOLO(str(weights_path))

    # Print model info
    print(f"Classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    print()
    print("Starting webcam...")
    print()

    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "Try a different camera_index (0, 1, 2, etc.)"
        )

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    save_count = 0
    current_conf = conf_threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Run inference
        results = model(frame, conf=current_conf, verbose=False)[0]

        # Draw results on frame
        annotated_frame = results.plot()

        # Add info overlay
        info_text = [
            f"Frame: {frame_count}",
            f"Conf: {current_conf:.2f}",
            f"Detections: {len(results.boxes)}",
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25

        # Show frame
        cv2.imshow("Custom YOLO Detection - Press Q to quit", annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break

        elif key == ord('s') or key == ord('S'):
            # Save current frame
            save_path = f"detection_save_{save_count:03d}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved: {save_path}")
            save_count += 1

        elif key == ord('+') or key == ord('='):
            # Increase confidence threshold
            current_conf = min(0.95, current_conf + 0.05)
            print(f"Confidence threshold: {current_conf:.2f}")

        elif key == ord('-') or key == ord('_'):
            # Decrease confidence threshold
            current_conf = max(0.05, current_conf - 0.05)
            print(f"Confidence threshold: {current_conf:.2f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"Processed {frame_count} frames")
    print(f"Saved {save_count} frames")
    print("Done!")


def run_image_inference(
        weights_path: Path,
        image_path: str,
        conf_threshold: float = 0.5,
        save_result: bool = True
):
    """
    Run object detection on a single image.

    Args:
        weights_path: Path to trained model weights
        image_path: Path to input image
        conf_threshold: Minimum confidence threshold
        save_result: Whether to save annotated image
    """
    print()
    print("=" * 70)
    print("CUSTOM YOLO IMAGE INFERENCE")
    print("=" * 70)
    print()

    # Load model
    model = YOLO(str(weights_path))

    print(f"Model: {weights_path}")
    print(f"Image: {image_path}")
    print(f"Confidence: {conf_threshold}")
    print()

    # Run inference
    results = model(image_path, conf=conf_threshold)[0]

    # Print detections
    print(f"Found {len(results.boxes)} objects:")
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"  {i + 1}. {class_name} ({confidence:.2f})")
    print()

    # Show result
    annotated = results.plot()
    cv2.imshow("Detection Result - Press any key to close", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if save_result:
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"Saved result to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print()
    print("=" * 70)
    print("YOLO CUSTOM MODEL INFERENCE")
    print("=" * 70)
    print()

    # Find the trained model
    try:
        weights = find_best_weights()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("Make sure you've trained a model first using:")
        print("  python 3_2_custom_yolo_training.py")
        sys.exit(1)

    print()
    print("What would you like to do?")
    print("  1. Webcam inference (real-time)")
    print("  2. Image inference (single image)")
    print("  3. Specify custom weights path")
    print()

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        # Webcam inference
        run_webcam_inference(
            weights_path=weights,
            conf_threshold=0.5,
            camera_index=0
        )

    elif choice == "2":
        # Image inference
        image_path = input("Enter image path: ").strip()
        run_image_inference(
            weights_path=weights,
            image_path=image_path,
            conf_threshold=0.5
        )

    elif choice == "3":
        # Custom path
        custom_path = input("Enter path to weights file: ").strip()
        custom_weights = Path(custom_path)

        if not custom_weights.exists():
            print(f"Error: File not found: {custom_weights}")
            sys.exit(1)

        run_webcam_inference(
            weights_path=custom_weights,
            conf_threshold=0.5
        )

    else:
        print("Invalid choice. Running webcam inference by default...")
        run_webcam_inference(
            weights_path=weights,
            conf_threshold=0.5
        )