#!/usr/bin/env python3
"""
================================================================================
STEP 2: DATA LABELING
================================================================================

Draw bounding boxes around objects in your captured images.

USAGE:
    python scripts/02_label_data.py

CONTROLS:
    [MOUSE]     Click and drag to draw bounding box
    [1-9]       Select class (1=first class, 2=second, etc.)
    [D]         Delete last box on current image
    [C]         Clear all boxes on current image
    [N/→]       Next image
    [P/←]       Previous image
    [S]         Save current annotations
    [Q]         Quit (auto-saves)

LABELING TIPS:
    - Draw boxes TIGHT around objects (no extra padding)
    - Include partially visible objects
    - Label ALL instances of each class
    - Be consistent across images

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, get_image_files, print_header


class LabelingTool:
    """Simple bounding box labeling tool."""
    
    def __init__(self, config: dict):
        self.config = config
        self.classes = config['classes']
        self.raw_dir = Path(config['paths']['raw_data'])
        self.labeled_dir = Path(config['paths']['labeled_data'])
        
        ensure_dir(self.labeled_dir / 'images')
        ensure_dir(self.labeled_dir / 'labels')
        
        # Collect all images
        self.images = []
        for class_name in self.classes:
            class_dir = self.raw_dir / class_name
            if class_dir.exists():
                self.images.extend(get_image_files(class_dir))
        
        self.images = sorted(self.images)
        
        if not self.images:
            raise ValueError(f"No images found in {self.raw_dir}")
        
        # State
        self.current_idx = 0
        self.current_boxes: List[Tuple[int, int, int, int, int]] = []  # (x1, y1, x2, y2, class_id)
        self.current_class = 0
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        # Colors for classes
        np.random.seed(42)
        self.colors = [
            tuple(map(int, np.random.randint(50, 255, 3)))
            for _ in range(len(self.classes))
        ]
        
        # Load existing labels
        self.load_labels()
    
    def get_label_path(self, image_path: Path) -> Path:
        """Get the label file path for an image."""
        return self.labeled_dir / 'labels' / f"{image_path.stem}.txt"
    
    def load_labels(self):
        """Load existing labels for current image."""
        label_path = self.get_label_path(self.images[self.current_idx])
        self.current_boxes = []
        
        if label_path.exists():
            img = cv2.imread(str(self.images[self.current_idx]))
            if img is None:
                return
            h, w = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h
                        
                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)
                        
                        self.current_boxes.append((x1, y1, x2, y2, class_id))
    
    def save_labels(self):
        """Save labels for current image."""
        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]
        
        # Copy image to labeled directory
        dest_img = self.labeled_dir / 'images' / img_path.name
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)
        
        # Save labels
        label_path = self.get_label_path(img_path)
        with open(label_path, 'w') as f:
            for x1, y1, x2, y2, class_id in self.current_boxes:
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1 = min(self.start_point[0], x)
                y1 = min(self.start_point[1], y)
                x2 = max(self.start_point[0], x)
                y2 = max(self.start_point[1], y)
                
                # Only add if box is large enough
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.current_boxes.append((x1, y1, x2, y2, self.current_class))
                
                self.start_point = None
                self.current_point = None
    
    def draw_interface(self, img: np.ndarray) -> np.ndarray:
        """Draw the labeling interface."""
        display = img.copy()
        h, w = display.shape[:2]
        
        # Draw existing boxes
        for x1, y1, x2, y2, class_id in self.current_boxes:
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = self.classes[class_id]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current box being drawn
        if self.drawing and self.start_point and self.current_point:
            color = self.colors[self.current_class % len(self.colors)]
            cv2.rectangle(display, self.start_point, self.current_point, color, 2)
        
        # Top bar - image info
        cv2.rectangle(display, (0, 0), (w, 50), (40, 40, 40), -1)
        img_name = self.images[self.current_idx].name
        cv2.putText(display, f"Image {self.current_idx + 1}/{len(self.images)}: {img_name}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Class selector
        cv2.rectangle(display, (0, 50), (w, 90), (60, 60, 60), -1)
        x_offset = 10
        for i, class_name in enumerate(self.classes):
            color = self.colors[i % len(self.colors)]
            
            # Highlight selected class
            if i == self.current_class:
                cv2.rectangle(display, (x_offset - 3, 55), (x_offset + len(class_name) * 12 + 30, 85), (255, 255, 255), 2)
            
            cv2.rectangle(display, (x_offset, 58), (x_offset + 20, 78), color, -1)
            cv2.putText(display, f"[{i+1}] {class_name}", (x_offset + 25, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x_offset += len(class_name) * 12 + 80
        
        # Bottom bar - controls
        cv2.rectangle(display, (0, h - 30), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "[N/P] Navigate  [D] Delete  [C] Clear  [S] Save  [Q] Quit",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Box count
        cv2.putText(display, f"Boxes: {len(self.current_boxes)}",
                   (w - 100, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display
    
    def run(self):
        """Run the labeling tool."""
        cv2.namedWindow("Labeling Tool")
        cv2.setMouseCallback("Labeling Tool", self.mouse_callback)
        
        while True:
            # Load current image
            img = cv2.imread(str(self.images[self.current_idx]))
            if img is None:
                print(f"Could not load: {self.images[self.current_idx]}")
                self.current_idx = (self.current_idx + 1) % len(self.images)
                continue
            
            # Draw interface
            display = self.draw_interface(img)
            cv2.imshow("Labeling Tool", display)
            
            # Handle keyboard
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                self.save_labels()
                break
            
            elif key == ord('n') or key == 83:  # Next or Right arrow
                self.save_labels()
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self.load_labels()
            
            elif key == ord('p') or key == 81:  # Previous or Left arrow
                self.save_labels()
                self.current_idx = (self.current_idx - 1) % len(self.images)
                self.load_labels()
            
            elif key == ord('d'):  # Delete last box
                if self.current_boxes:
                    self.current_boxes.pop()
            
            elif key == ord('c'):  # Clear all boxes
                self.current_boxes = []
            
            elif key == ord('s'):  # Save
                self.save_labels()
                print(f"Saved labels for {self.images[self.current_idx].name}")
            
            elif ord('1') <= key <= ord('9'):  # Select class
                class_num = key - ord('1')
                if class_num < len(self.classes):
                    self.current_class = class_num
        
        cv2.destroyAllWindows()


def main():
    print_header("DATA LABELING")
    
    config = load_config()
    
    print(f"Classes: {config['classes']}")
    print()
    print("Controls:")
    print("  [MOUSE]  Click and drag to draw box")
    print("  [1-9]    Select class")
    print("  [D]      Delete last box")
    print("  [C]      Clear all boxes")
    print("  [N/→]    Next image")
    print("  [P/←]    Previous image")
    print("  [S]      Save")
    print("  [Q]      Quit")
    print()
    
    try:
        tool = LabelingTool(config)
        print(f"Found {len(tool.images)} images to label")
        print()
        print("Starting labeling tool...")
        tool.run()
    except ValueError as e:
        print(f"Error: {e}")
        print()
        print("Make sure you have captured images first:")
        print("  python scripts/01_capture_data.py")
        return
    
    # Print summary
    labeled_dir = Path(config['paths']['labeled_data'])
    label_files = list((labeled_dir / 'labels').glob('*.txt'))
    labeled_count = sum(1 for f in label_files if f.stat().st_size > 0)
    
    print()
    print_header("LABELING SUMMARY")
    print(f"Images labeled: {labeled_count}/{len(tool.images)}")
    print(f"Labels saved to: {labeled_dir}")
    print()
    print("Next step: python scripts/03_augment_data.py")


if __name__ == "__main__":
    main()
