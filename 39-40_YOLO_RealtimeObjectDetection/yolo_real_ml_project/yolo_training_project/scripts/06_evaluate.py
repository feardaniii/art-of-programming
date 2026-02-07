#!/usr/bin/env python3
"""
================================================================================
STEP 6: EVALUATE MODEL
================================================================================

Evaluate your trained model's performance and understand the results.

USAGE:
    python scripts/06_evaluate.py

WHAT THIS DOES:
    1. Runs validation on the val set
    2. Calculates metrics (mAP, precision, recall)
    3. Generates confusion matrix
    4. Shows example predictions
    5. Identifies failure cases

UNDERSTANDING METRICS:
    mAP@0.5:        Main metric. >0.7 is good, >0.9 is excellent.
    mAP@0.5:0.95:   Stricter metric. >0.5 is good.
    Precision:      Of detections made, how many were correct?
    Recall:         Of objects present, how many were found?

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, print_header


def evaluate_model():
    """Evaluate the trained model."""
    
    print_header("MODEL EVALUATION")
    
    # Load configuration
    config = load_config()
    final_dir = Path(config['paths']['final_data'])
    models_dir = Path(config['paths']['models'])
    
    # Find model weights
    weights_path = models_dir / "best.pt"
    if not weights_path.exists():
        # Try runs directory
        weights_path = Path(config['paths']['runs']) / "train" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"Error: No trained model found")
        print("Train a model first: python scripts/05_train.py")
        return
    
    data_yaml = final_dir / 'data.yaml'
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        return
    
    print(f"Model: {weights_path}")
    print(f"Dataset: {data_yaml}")
    print()
    
    # Load model
    model = YOLO(str(weights_path))
    
    # Run validation
    print("Running validation...")
    print()
    
    metrics = model.val(
        data=str(data_yaml),
        verbose=True,
        plots=True,
    )
    
    # Print metrics
    print()
    print_header("EVALUATION RESULTS")
    
    print("Overall Metrics:")
    print(f"  mAP@0.5:        {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:   {metrics.box.map:.4f}")
    print(f"  Precision:      {metrics.box.mp:.4f}")
    print(f"  Recall:         {metrics.box.mr:.4f}")
    print()
    
    # Interpret metrics
    print("Interpretation:")
    
    map50 = metrics.box.map50
    if map50 >= 0.9:
        print(f"  ✓ mAP@0.5 = {map50:.2f} - Excellent! Model is very accurate.")
    elif map50 >= 0.7:
        print(f"  ✓ mAP@0.5 = {map50:.2f} - Good. Model should work well.")
    elif map50 >= 0.5:
        print(f"  ~ mAP@0.5 = {map50:.2f} - Moderate. Consider more training data or epochs.")
    else:
        print(f"  ✗ mAP@0.5 = {map50:.2f} - Low. Model needs improvement.")
    
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    if precision < 0.7:
        print(f"  ! Low precision ({precision:.2f}) - Too many false positives.")
        print("    Try: Increase confidence threshold, more negative examples")
    
    if recall < 0.7:
        print(f"  ! Low recall ({recall:.2f}) - Missing too many objects.")
        print("    Try: More training data, lower confidence threshold")
    
    # Per-class metrics
    print()
    print("Per-Class Performance:")
    
    classes = model.names
    for i, class_name in classes.items():
        # Get class-specific metrics if available
        if hasattr(metrics.box, 'ap50'):
            ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            print(f"  {class_name:15} AP@0.5: {ap:.4f}")
    
    print()
    print("Plots saved to validation directory")
    print()
    
    return metrics


def show_predictions():
    """Show model predictions on validation images."""
    
    print_header("PREDICTION EXAMPLES")
    
    # Load configuration
    config = load_config()
    final_dir = Path(config['paths']['final_data'])
    models_dir = Path(config['paths']['models'])
    
    weights_path = models_dir / "best.pt"
    if not weights_path.exists():
        weights_path = Path(config['paths']['runs']) / "train" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print("Error: No trained model found")
        return
    
    # Load model
    model = YOLO(str(weights_path))
    
    # Get validation images
    val_images = list((final_dir / 'images' / 'val').glob('*'))
    
    if not val_images:
        print("No validation images found")
        return
    
    # Show random predictions
    random.shuffle(val_images)
    
    print("Showing predictions on validation images...")
    print("Press any key for next image, 'Q' to quit")
    print()
    
    conf_threshold = config['inference']['confidence_threshold']
    
    for img_path in val_images[:10]:  # Show up to 10
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Run inference
        results = model(img, conf=conf_threshold, verbose=False)[0]
        
        # Get annotated image
        annotated = results.plot()
        
        # Add image name
        cv2.putText(annotated, img_path.name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show detections count
        num_detections = len(results.boxes) if results.boxes is not None else 0
        cv2.putText(annotated, f"Detections: {num_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Predictions", annotated)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


def analyze_failures():
    """Analyze images where the model fails."""
    
    print_header("FAILURE ANALYSIS")
    
    config = load_config()
    final_dir = Path(config['paths']['final_data'])
    models_dir = Path(config['paths']['models'])
    
    weights_path = models_dir / "best.pt"
    if not weights_path.exists():
        weights_path = Path(config['paths']['runs']) / "train" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print("Error: No trained model found")
        return
    
    model = YOLO(str(weights_path))
    
    # Compare predictions to ground truth
    val_images = list((final_dir / 'images' / 'val').glob('*'))
    val_labels = final_dir / 'labels' / 'val'
    
    conf_threshold = config['inference']['confidence_threshold']
    
    false_negatives = []  # Missed objects
    false_positives = []  # Wrong detections
    
    print("Analyzing validation set...")
    
    for img_path in val_images:
        label_path = val_labels / f"{img_path.stem}.txt"
        
        # Count ground truth objects
        gt_count = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                gt_count = len([l for l in f.readlines() if l.strip()])
        
        # Count predictions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        results = model(img, conf=conf_threshold, verbose=False)[0]
        pred_count = len(results.boxes) if results.boxes is not None else 0
        
        if pred_count < gt_count:
            false_negatives.append((img_path, gt_count, pred_count))
        elif pred_count > gt_count:
            false_positives.append((img_path, gt_count, pred_count))
    
    print()
    print(f"Images with missed objects: {len(false_negatives)}")
    print(f"Images with extra detections: {len(false_positives)}")
    print()
    
    # Show worst cases
    if false_negatives:
        print("Top images with missed objects:")
        for img_path, gt, pred in sorted(false_negatives, key=lambda x: x[1]-x[2], reverse=True)[:5]:
            print(f"  {img_path.name}: expected {gt}, got {pred}")
    
    print()
    
    if false_positives:
        print("Top images with false detections:")
        for img_path, gt, pred in sorted(false_positives, key=lambda x: x[2]-x[1], reverse=True)[:5]:
            print(f"  {img_path.name}: expected {gt}, got {pred}")
    
    print()
    print("Tips for improvement:")
    
    if len(false_negatives) > len(val_images) * 0.3:
        print("  - Many missed objects: try more training data, lower confidence threshold")
    
    if len(false_positives) > len(val_images) * 0.3:
        print("  - Many false positives: try higher confidence threshold, more negative examples")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--show', '-s', action='store_true',
                       help="Show prediction examples")
    parser.add_argument('--failures', '-f', action='store_true',
                       help="Analyze failure cases")
    
    args = parser.parse_args()
    
    if args.show:
        show_predictions()
    elif args.failures:
        analyze_failures()
    else:
        evaluate_model()
        print()
        print("Use --show to see prediction examples")
        print("Use --failures to analyze failure cases")
    
    print()
    print("Next step: python scripts/07_inference.py")
