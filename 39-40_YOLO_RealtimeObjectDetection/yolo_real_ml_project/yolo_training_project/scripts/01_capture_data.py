#!/usr/bin/env python3
"""
================================================================================
STEP 1: DATA CAPTURE
================================================================================

Capture images for your custom dataset using your webcam.

USAGE:
    python scripts/01_capture_data.py

CONTROLS:
    [SPACE] Capture single image
    [B]     Start/stop burst mode (continuous capture)
    [N]     Next class
    [P]     Previous class
    [Q]     Quit and save

TIPS FOR GOOD DATA:
    - Move the object around (don't keep it in one spot)
    - Vary the background (desk, hand, floor, fabric)
    - Vary lighting (turn lights on/off, move to window)
    - Vary distance (close-up, medium, far)
    - Vary angle (top-down, side view, tilted)
    - Include partial occlusion (object partially hidden)

================================================================================
"""

import cv2
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, print_header


def capture_data():
    """Main capture function."""
    
    print_header("DATA CAPTURE")
    
    # Load configuration
    config = load_config()
    classes = config['classes']
    capture_cfg = config['capture']
    raw_dir = Path(config['paths']['raw_data'])
    
    print(f"Classes to capture: {classes}")
    print(f"Target images per class: {capture_cfg['images_per_class']}")
    print()
    
    # Create directories for each class
    for class_name in classes:
        ensure_dir(raw_dir / class_name)
    
    # Initialize camera
    cap = cv2.VideoCapture(capture_cfg.get('camera_id', 0))
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    # Set resolution
    img_size = capture_cfg.get('image_size', [640, 480])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
    
    # State
    current_class_idx = 0
    image_counts = {c: len(list((raw_dir / c).glob('*.jpg'))) for c in classes}
    burst_mode = False
    last_capture_time = 0
    capture_delay = capture_cfg.get('capture_delay_ms', 500) / 1000.0
    
    print("Controls:")
    print("  [SPACE] Capture image")
    print("  [B] Toggle burst mode")
    print("  [N/P] Next/Previous class")
    print("  [Q] Quit")
    print()
    print("Starting capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_class = classes[current_class_idx]
        count = image_counts[current_class]
        target = capture_cfg['images_per_class']
        
        # Draw UI
        display = frame.copy()
        
        # Top bar - current class
        cv2.rectangle(display, (0, 0), (frame.shape[1], 60), (40, 40, 40), -1)
        cv2.putText(
            display, 
            f"Class: {current_class}", 
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )
        
        # Progress
        progress_text = f"Images: {count}/{target}"
        cv2.putText(
            display,
            progress_text,
            (frame.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        # Progress bar
        bar_width = frame.shape[1] - 40
        bar_height = 20
        progress = min(count / target, 1.0)
        cv2.rectangle(display, (20, 65), (20 + bar_width, 65 + bar_height), (100, 100, 100), -1)
        cv2.rectangle(display, (20, 65), (20 + int(bar_width * progress), 65 + bar_height), (0, 255, 0), -1)
        
        # Burst mode indicator
        if burst_mode:
            cv2.putText(
                display,
                "BURST MODE",
                (frame.shape[1] // 2 - 80, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        # Bottom bar - controls
        cv2.rectangle(display, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (40, 40, 40), -1)
        cv2.putText(
            display,
            "[SPACE] Capture  [B] Burst  [N/P] Class  [Q] Quit",
            (20, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        cv2.imshow("Data Capture", display)
        
        # Handle burst mode
        if burst_mode and time.time() - last_capture_time > capture_delay:
            # Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{current_class}_{timestamp}.jpg"
            filepath = raw_dir / current_class / filename
            cv2.imwrite(str(filepath), frame)
            image_counts[current_class] += 1
            last_capture_time = time.time()
            print(f"  Captured: {filename}")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord(' '):  # Space - capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{current_class}_{timestamp}.jpg"
            filepath = raw_dir / current_class / filename
            cv2.imwrite(str(filepath), frame)
            image_counts[current_class] += 1
            print(f"  Captured: {filename}")
        
        elif key == ord('b'):  # Toggle burst mode
            burst_mode = not burst_mode
            print(f"  Burst mode: {'ON' if burst_mode else 'OFF'}")
        
        elif key == ord('n'):  # Next class
            current_class_idx = (current_class_idx + 1) % len(classes)
            print(f"  Switched to: {classes[current_class_idx]}")
        
        elif key == ord('p'):  # Previous class
            current_class_idx = (current_class_idx - 1) % len(classes)
            print(f"  Switched to: {classes[current_class_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print()
    print_header("CAPTURE SUMMARY")
    total = 0
    for class_name in classes:
        count = image_counts[class_name]
        target = capture_cfg['images_per_class']
        status = "✓" if count >= target else "✗"
        print(f"  {status} {class_name}: {count}/{target} images")
        total += count
    
    print()
    print(f"Total images captured: {total}")
    print(f"Images saved to: {raw_dir}")
    print()
    print("Next step: python scripts/02_label_data.py")


if __name__ == "__main__":
    capture_data()
