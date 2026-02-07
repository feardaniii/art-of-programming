#!/usr/bin/env python3
"""
================================================================================
STEP 4: PREPARE DATASET FOR YOLO
================================================================================

Organize augmented data into YOLO's expected folder structure.

USAGE:
    python scripts/04_prepare_dataset.py

WHAT THIS DOES:
    1. Splits data into train/val sets (80/20 by default)
    2. Creates the folder structure YOLO expects
    3. Generates data.yaml configuration file
    4. Validates the dataset (checks for issues)
    5. Prints dataset statistics

YOLO EXPECTED STRUCTURE:
    final/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

================================================================================
"""

import cv2
import yaml
import random
import shutil
from pathlib import Path
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, get_image_files, print_header


def prepare_dataset():
    """Prepare the final dataset for YOLO training."""
    
    print_header("DATASET PREPARATION")
    
    # Load configuration
    config = load_config()
    classes = config['classes']
    dataset_cfg = config['dataset']
    
    augmented_dir = Path(config['paths']['augmented_data'])
    final_dir = Path(config['paths']['final_data'])
    
    # Check if augmented data exists
    augmented_images_dir = augmented_dir / 'images'
    augmented_labels_dir = augmented_dir / 'labels'
    
    if not augmented_images_dir.exists():
        print(f"Error: Augmented images not found at {augmented_images_dir}")
        print("Run augmentation first: python scripts/03_augment_data.py")
        return
    
    # Get all augmented images
    images = get_image_files(augmented_images_dir)
    
    if not images:
        print("Error: No augmented images found")
        return
    
    print(f"Found {len(images)} augmented images")
    print()
    
    # Filter to only images with labels
    valid_images = []
    for img_path in images:
        label_path = augmented_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            valid_images.append(img_path)
    
    print(f"Images with valid labels: {len(valid_images)}")
    
    if not valid_images:
        print("Error: No images have labels")
        return
    
    # Shuffle and split
    if dataset_cfg.get('shuffle', True):
        random.seed(dataset_cfg.get('random_seed', 42))
        random.shuffle(valid_images)
    
    train_ratio = dataset_cfg.get('train_ratio', 0.8)
    split_idx = int(len(valid_images) * train_ratio)
    
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    print(f"Train set: {len(train_images)} images ({train_ratio*100:.0f}%)")
    print(f"Val set: {len(val_images)} images ({(1-train_ratio)*100:.0f}%)")
    print()
    
    # Create final directory structure
    print("Creating directory structure...")
    
    # Clean and create directories
    if final_dir.exists():
        shutil.rmtree(final_dir)
    
    ensure_dir(final_dir / 'images' / 'train')
    ensure_dir(final_dir / 'images' / 'val')
    ensure_dir(final_dir / 'labels' / 'train')
    ensure_dir(final_dir / 'labels' / 'val')
    
    # Copy files
    print("Copying train images...")
    for img_path in train_images:
        label_path = augmented_labels_dir / f"{img_path.stem}.txt"
        
        # Copy image
        shutil.copy(img_path, final_dir / 'images' / 'train' / img_path.name)
        
        # Copy label
        shutil.copy(label_path, final_dir / 'labels' / 'train' / label_path.name)
    
    print("Copying validation images...")
    for img_path in val_images:
        label_path = augmented_labels_dir / f"{img_path.stem}.txt"
        
        # Copy image
        shutil.copy(img_path, final_dir / 'images' / 'val' / img_path.name)
        
        # Copy label
        shutil.copy(label_path, final_dir / 'labels' / 'val' / label_path.name)
    
    # Create data.yaml
    print("Creating data.yaml...")
    
    data_yaml = {
        'path': str(final_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = final_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Validate dataset
    print()
    print("Validating dataset...")
    
    issues = []
    class_counts = Counter()
    
    # Check train set
    train_labels = list((final_dir / 'labels' / 'train').glob('*.txt'))
    for label_path in train_labels:
        img_path = final_dir / 'images' / 'train' / f"{label_path.stem}.jpg"
        
        # Check image exists
        if not img_path.exists():
            # Try png
            img_path = final_dir / 'images' / 'train' / f"{label_path.stem}.png"
            if not img_path.exists():
                issues.append(f"Missing image for {label_path.name}")
                continue
        
        # Check label content
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"{label_path.name}:{line_num} - Invalid format")
                    continue
                
                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(classes):
                    issues.append(f"{label_path.name}:{line_num} - Invalid class ID: {class_id}")
                else:
                    class_counts[classes[class_id]] += 1
    
    # Check val set
    val_labels = list((final_dir / 'labels' / 'val').glob('*.txt'))
    for label_path in val_labels:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(classes):
                        class_counts[classes[class_id]] += 1
    
    # Print results
    print()
    print_header("DATASET SUMMARY")
    
    print("Files created:")
    print(f"  {final_dir / 'data.yaml'}")
    print(f"  {final_dir / 'images/train/'} ({len(train_images)} images)")
    print(f"  {final_dir / 'images/val/'} ({len(val_images)} images)")
    print()
    
    print("Class distribution:")
    total_boxes = sum(class_counts.values())
    for class_name in classes:
        count = class_counts.get(class_name, 0)
        pct = count / total_boxes * 100 if total_boxes > 0 else 0
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {class_name:15} {count:5} ({pct:5.1f}%) {bar}")
    
    print()
    print(f"Total bounding boxes: {total_boxes}")
    
    if issues:
        print()
        print("⚠ Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print()
        print("✓ No issues found - dataset is ready for training!")
    
    print()
    print(f"data.yaml path: {yaml_path.absolute()}")
    print()
    print("Next step: python scripts/05_train.py")


if __name__ == "__main__":
    prepare_dataset()
