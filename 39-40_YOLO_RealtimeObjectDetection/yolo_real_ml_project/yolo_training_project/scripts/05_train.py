#!/usr/bin/env python3
"""
================================================================================
STEP 5: TRAIN YOLOV8 MODEL
================================================================================

Train YOLOv8 on your prepared custom dataset.

USAGE:
    python scripts/05_train.py

    # With custom settings:
    python scripts/05_train.py --epochs 100 --batch 8 --model s

TRAINING TIME ESTIMATES:
    CPU:        2-8 hours (not recommended)
    GPU (RTX):  15-45 minutes
    Apple M1/2: 30-90 minutes

KEY TRAINING PARAMETERS:
    epochs:     How many times to see all training data (50-300)
    batch:      Images per gradient update (8-32, limited by VRAM)
    imgsz:      Input resolution (640 standard)
    patience:   Stop early if no improvement for N epochs

================================================================================
"""

from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, print_header


def train_model(
    epochs: int = None,
    batch: int = None,
    model_size: str = None,
    imgsz: int = None,
    device: str = None
):
    """Train YOLOv8 on custom dataset."""
    
    print_header("YOLOV8 TRAINING")
    
    # Load configuration
    config = load_config()
    train_cfg = config['training']
    final_dir = Path(config['paths']['final_data'])
    models_dir = Path(config['paths']['models'])
    runs_dir = Path(config['paths']['runs'])
    
    ensure_dir(models_dir)
    
    # Use config values if not overridden
    epochs = epochs or train_cfg['epochs']
    batch = batch or train_cfg['batch_size']
    model_size = model_size or train_cfg['model_size']
    imgsz = imgsz or train_cfg['image_size']
    device = device or train_cfg['device']
    patience = train_cfg.get('patience', 20)
    
    # Check dataset exists
    data_yaml = final_dir / 'data.yaml'
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        print("Run dataset preparation first: python scripts/04_prepare_dataset.py")
        return
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Model:      YOLOv8{model_size}")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Device:     {device}")
    print(f"  Patience:   {patience} (early stopping)")
    print()
    
    # Verify classes
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Classes ({data_config['nc']}):")
    for i, name in enumerate(data_config['names']):
        print(f"  {i}: {name}")
    print()
    
    # Count training images
    train_images = list((final_dir / 'images' / 'train').glob('*'))
    val_images = list((final_dir / 'images' / 'val').glob('*'))
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print()
    
    # Load pre-trained model (transfer learning)
    print(f"Loading YOLOv8{model_size} pre-trained weights...")
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train
    print()
    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print()
    print("Monitor progress in the terminal.")
    print("Training plots will be saved to the runs directory.")
    print()
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        save=True,
        plots=True,
        verbose=True,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
        pretrained=True,
        
        # Optimizer settings
        optimizer=train_cfg.get('optimizer', 'AdamW'),
        lr0=train_cfg.get('learning_rate', 0.001),
        weight_decay=train_cfg.get('weight_decay', 0.0005),
        
        # Data loading
        workers=train_cfg.get('workers', 4),
        
        # Augmentation (YOLO has built-in augmentation too)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=0.0,  # We already did rotation in our augmentation
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,   # Disable mosaic (we have our augmentation)
    )
    
    # Find best weights
    best_weights = runs_dir / "train" / "weights" / "best.pt"
    
    # Copy to models directory
    final_weights = models_dir / "best.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, final_weights)
    
    # Print summary
    print()
    print_header("TRAINING COMPLETE")
    
    print("Results:")
    print(f"  Best weights: {final_weights}")
    print(f"  Training logs: {runs_dir / 'train'}")
    print()
    
    print("Training plots saved:")
    print(f"  {runs_dir / 'train' / 'results.png'}")
    print(f"  {runs_dir / 'train' / 'confusion_matrix.png'}")
    print()
    
    # Show final metrics if available
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("Final Metrics:")
        print(f"  mAP@0.5:      {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision:    {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall:       {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        print()
    
    print("Next step: python scripts/06_evaluate.py")
    
    return str(final_weights)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8")
    parser.add_argument('--epochs', '-e', type=int, help="Number of epochs")
    parser.add_argument('--batch', '-b', type=int, help="Batch size")
    parser.add_argument('--model', '-m', type=str, choices=['n', 's', 'm', 'l', 'x'],
                       help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument('--imgsz', '-i', type=int, help="Image size")
    parser.add_argument('--device', '-d', type=str, help="Device (0 for GPU, cpu, mps)")
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch=args.batch,
        model_size=args.model,
        imgsz=args.imgsz,
        device=args.device
    )
