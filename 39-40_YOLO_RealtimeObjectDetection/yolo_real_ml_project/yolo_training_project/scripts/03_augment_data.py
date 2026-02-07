#!/usr/bin/env python3
"""
================================================================================
STEP 3: DATA AUGMENTATION
================================================================================

Generate augmented versions of your labeled images to expand the dataset.

USAGE:
    python scripts/03_augment_data.py

WHY AUGMENT?
    - 100 original images → 500+ augmented images
    - Model sees more variety → generalizes better
    - Simulates real-world conditions (blur, lighting, angles)
    - Critical for small datasets

AUGMENTATION TYPES APPLIED:
    - Geometric: flip, rotate, scale
    - Color: brightness, contrast, saturation, hue
    - Blur: gaussian, motion blur
    - Noise: gaussian noise, JPEG compression
    - Occlusion: random erasing

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, get_image_files, print_header
from src.augmentation import AugmentationPipeline, BoundingBox


def load_annotations(label_path: Path, img_width: int, img_height: int) -> list:
    """Load YOLO annotations and convert to BoundingBox objects."""
    boxes = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    boxes.append(BoundingBox.from_yolo(line, img_width, img_height))
    
    return boxes


def save_annotations(boxes: list, label_path: Path, img_width: int, img_height: int):
    """Save BoundingBox objects as YOLO annotations."""
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(box.to_yolo(img_width, img_height) + '\n')


def augment_dataset():
    """Main augmentation function."""
    
    print_header("DATA AUGMENTATION")
    
    # Load configuration
    config = load_config()
    aug_config = config['augmentation']
    labeled_dir = Path(config['paths']['labeled_data'])
    augmented_dir = Path(config['paths']['augmented_data'])
    
    # Setup directories
    ensure_dir(augmented_dir / 'images')
    ensure_dir(augmented_dir / 'labels')
    
    # Get labeled images
    labeled_images_dir = labeled_dir / 'images'
    labeled_labels_dir = labeled_dir / 'labels'
    
    if not labeled_images_dir.exists():
        print(f"Error: No labeled images found at {labeled_images_dir}")
        print("Run the labeling step first: python scripts/02_label_data.py")
        return
    
    images = get_image_files(labeled_images_dir)
    
    if not images:
        print("Error: No labeled images found")
        return
    
    print(f"Found {len(images)} labeled images")
    print(f"Augmentation multiplier: {aug_config['multiplier']}x")
    print(f"Expected output: ~{len(images) * (aug_config['multiplier'] + 1)} images")
    print()
    
    # Create augmentation pipeline
    pipeline = AugmentationPipeline(aug_config)
    
    # Process each image
    total_augmented = 0
    
    print("Augmenting images...")
    print()
    
    for img_path in tqdm(images, desc="Processing"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Load annotations
        label_path = labeled_labels_dir / f"{img_path.stem}.txt"
        boxes = load_annotations(label_path, w, h)
        
        if not boxes:
            # Skip images without annotations
            continue
        
        # Save original (copy to augmented directory)
        orig_img_path = augmented_dir / 'images' / img_path.name
        orig_label_path = augmented_dir / 'labels' / f"{img_path.stem}.txt"
        
        cv2.imwrite(str(orig_img_path), img)
        save_annotations(boxes, orig_label_path, w, h)
        total_augmented += 1
        
        # Generate augmented versions
        for aug_idx in range(aug_config['multiplier']):
            # Apply augmentation pipeline
            aug_img, aug_boxes = pipeline.apply(img, boxes)
            
            # Skip if no valid boxes remain
            if not aug_boxes:
                continue
            
            # Save augmented image and labels
            aug_name = f"{img_path.stem}_aug{aug_idx:02d}"
            aug_img_path = augmented_dir / 'images' / f"{aug_name}.jpg"
            aug_label_path = augmented_dir / 'labels' / f"{aug_name}.txt"
            
            cv2.imwrite(str(aug_img_path), aug_img)
            save_annotations(aug_boxes, aug_label_path, w, h)
            total_augmented += 1
    
    # Print summary
    print()
    print_header("AUGMENTATION SUMMARY")
    print(f"Original images: {len(images)}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Expansion factor: {total_augmented / len(images):.1f}x")
    print(f"Output saved to: {augmented_dir}")
    print()
    print("Next step: python scripts/04_prepare_dataset.py")


def visualize_augmentations():
    """
    Visualize augmentation effects on a single image.
    
    Useful for understanding what each augmentation does.
    """
    print_header("AUGMENTATION VISUALIZATION")
    
    config = load_config()
    aug_config = config['augmentation']
    labeled_dir = Path(config['paths']['labeled_data'])
    
    # Get first labeled image
    images = get_image_files(labeled_dir / 'images')
    if not images:
        print("No labeled images found")
        return
    
    img_path = images[0]
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Load annotations
    label_path = labeled_dir / 'labels' / f"{img_path.stem}.txt"
    boxes = load_annotations(label_path, w, h)
    
    # Create pipeline
    pipeline = AugmentationPipeline(aug_config)
    
    print(f"Image: {img_path.name}")
    print(f"Boxes: {len(boxes)}")
    print()
    print("Showing 6 augmented versions...")
    print("Press any key for next, 'Q' to quit")
    print()
    
    def draw_boxes(image, boxes, title):
        """Draw boxes on image."""
        display = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return display
    
    # Show original
    cv2.imshow("Augmentation Demo", draw_boxes(img, boxes, "Original"))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return
    
    # Show augmented versions
    for i in range(6):
        aug_img, aug_boxes = pipeline.apply(img, boxes)
        cv2.imshow("Augmentation Demo", draw_boxes(aug_img, aug_boxes, f"Augmented {i+1}"))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument('--visualize', '-v', action='store_true',
                       help="Visualize augmentations instead of generating")
    args = parser.parse_args()
    
    if args.visualize:
        visualize_augmentations()
    else:
        augment_dataset()
