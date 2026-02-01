from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import shutil
import random
from PIL import Image, ImageDraw
import numpy as np

# ==========================================
# THE NEED: Why Custom Training?
# ==========================================

"""
COCO dataset (80 classes): person, car, dog, laptop, etc.
Great for general detection, but missing:

For Health:
  - Specific food items (broccoli, chicken breast, protein shake)
  - Workout equipment (dumbbells, kettlebells, resistance bands)
  - Portion sizes (small/medium/large servings)
  - Exercise poses (squat, plank, push-up)

For Finance:
  - Receipt types (grocery, restaurant, gas station)
  - Currency notes and coins
  - Financial documents (invoices, bills, statements)
  - Product barcodes and price tags

Custom Training allows you to detect ANYTHING with just 100-500 labeled images!
"""

print("=" * 80)
print("CUSTOM YOLO TRAINING: Detect Anything You Want")
print("=" * 80)
print()


# ==========================================
# STEP 1: Dataset Preparation
# ==========================================

class YOLODatasetBuilder:
    """
    Prepare custom dataset for YOLO training

    YOLO training format:














    """

    def __init__(self, dataset_name, class_names):
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.dataset_path = Path(f"datasets/{dataset_name}")

        self.setup_directories()

    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        print(f"Setting up dataset: {self.dataset_name}")
        print(f"Classes: {', '.join(self.class_names)}")
        print()

        # Create directories
        dirs = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "images" / "val",
            self.dataset_path / "labels" / "train",
            self.dataset_path / "labels" / "val"
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f" Created directory structure at {self.dataset_path}")

    def create_data_yaml(self, train_path, val_path):
        """
        Create data.yaml configuration file

        This file tells YOLO where to find images and what classes to detect
        """
        data_config = {
            'path': str(self.dataset_path.absolute()),
            'train': str(train_path),
            'val': str(val_path),
            'nc': len(self.class_names),  # Number of classes
            'names': self.class_names
        }

        yaml_path = self.dataset_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f" Created {yaml_path}")
        return yaml_path

    def yolo_format_annotation(self, image_width, image_height, bbox, class_id):
        """
        Convert bounding box to YOLO format

        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1]

        Input bbox: [x_min, y_min, x_max, y_max] in pixels
        Output: "class_id x_center y_center width height" (normalized)
        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Format: class x_center y_center width height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def add_labeled_image(self, image_path, annotations, split='train'):
        """
        Add image and its annotations to dataset

        annotations: List of (bbox, class_id) tuples
        bbox: [x_min, y_min, x_max, y_max]
        """
        # Open image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Copy image to dataset
        img_name = Path(image_path).name
        dest_img = self.dataset_path / "images" / split / img_name
        shutil.copy(image_path, dest_img)

        # Create label file
        label_name = Path(image_path).stem + ".txt"
        label_path = self.dataset_path / "labels" / split / label_name

        with open(label_path, 'w') as f:
            for bbox, class_id in annotations:
                yolo_line = self.yolo_format_annotation(img_width, img_height, bbox, class_id)
                f.write(yolo_line + '\n')

        return dest_img, label_path


# ==========================================
# EXAMPLE: Health App - Meal Portion Detection
# ==========================================

def create_meal_portion_dataset():
    """
    Example: Train YOLO to detect meal portions

    Classes:
      0: small_portion
      1: medium_portion
      2: large_portion
      3: protein (chicken, fish, meat)
      4: vegetables (broccoli, salad, carrots)
      5: carbs (rice, pasta, bread)

    Use case: Automatic meal logging for calorie tracking
    """

    print("=" * 80)
    print("HEALTH APPLICATION: Meal Portion Detection Dataset")
    print("=" * 80)
    print()

    classes = ['small_portion', 'medium_portion', 'large_portion',
               'protein', 'vegetables', 'carbs']

    dataset = YOLODatasetBuilder('meal_portions', classes)

    # In real scenario:
    # 1. Collect 300-500 photos of meals
    # 2. Label using tools like LabelImg, Roboflow, or CVAT
    # 3. Export in YOLO format
    # 4. Split 80/20 into train/val

    print("=� Dataset Collection Tips:")
    print(" Take photos from different angles")
    print(" Various lighting conditions")
    print(" Different plate sizes and colors")
    print(" Mix of cuisines and meal types")
    print(" Minimum 50 images per class")
    print()

    # Create data.yaml
    yaml_path = dataset.create_data_yaml('images/train', 'images/val')

    print(" Meal portion dataset ready for labeling!")
    print(f"   Next step: Use LabelImg or Roboflow to annotate images")
    print()

    return yaml_path


# ==========================================
# EXAMPLE: Finance App - Receipt Detection
# ==========================================

def create_receipt_detection_dataset():
    """
    Example: Train YOLO to detect receipt components

    Classes:
      0: receipt (full document)
      1: merchant_name
      2: total_amount
      3: date
      4: line_items
      5: payment_method

    Use case: Automatic expense tracking and categorization
    """

    print("=" * 80)
    print("FINANCE APPLICATION: Receipt Component Detection Dataset")
    print("=" * 80)
    print()

    classes = ['receipt', 'merchant_name', 'total_amount',
               'date', 'line_items', 'payment_method']

    dataset = YOLODatasetBuilder('receipt_components', classes)

    print("=� Dataset Collection Tips:")
    print(" Scan receipts from various stores")
    print(" Include different receipt formats")
    print(" Capture in various lighting conditions")
    print(" Some crumpled or slightly damaged receipts")
    print(" Minimum 200 receipts (more is better)")
    print()

    yaml_path = dataset.create_data_yaml('images/train', 'images/val')

    print(" Receipt detection dataset ready for labeling!")
    print(f"   Next step: Use LabelImg or Roboflow to annotate receipts")
    print()

    return yaml_path


# ==========================================
# STEP 2: Training Custom YOLO Model
# ==========================================

class CustomYOLOTrainer:
    """
    Train custom YOLO model on your dataset

    Training process:
    1. Load pre-trained weights (transfer learning)
    2. Fine-tune on your custom dataset
    3. Validate and evaluate
    4. Export for deployment
    """

    def __init__(self, model_size='n'):
        """
        model_size options:
          'n' (nano): Fastest, good for edge devices
          's' (small): Balanced speed/accuracy
          'm' (medium): Better accuracy, slower
        """
        self.model_size = model_size
        self.model = YOLO(f'yolov8{model_size}.pt')  # Pre-trained COCO weights

    def train(self, data_yaml, epochs=100, imgsz=640, batch=16):
        """
        Train custom model

        Parameters:
          data_yaml: Path to data.yaml configuration
          epochs: Training iterations (100-300 typical)
          imgsz: Image size (640 standard, 320 for faster)
          batch: Batch size (adjust based on GPU memory)

        Transfer Learning:
          - Starts with COCO pre-trained weights
          - Already knows edges, shapes, objects
          - Only learns YOUR specific classes
          - Needs 10x less data than training from scratch
        """

        print("=" * 80)
        print(f"TRAINING CUSTOM YOLOv8{self.model_size.upper()} MODEL")
        print("=" * 80)
        print()
        print(f"Configuration:")
        print(f" Dataset: {data_yaml}")
        print(f" Epochs: {epochs}")
        print(f" Image size: {imgsz}�{imgsz}")
        print(f" Batch size: {batch}")
        print()
        print("Starting training...")
        print("(This will take 1-4 hours depending on dataset size and GPU)")
        print()

        # Train model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=50,  # Early stopping
            save=True,
            device='0',  # GPU 0 (use 'cpu' if no GPU)
            workers=8,
            exist_ok=True,
            pretrained=True,  # Use transfer learning
            optimizer='AdamW',
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True  # Generate training plots
        )

        print()
        print(" Training complete!")
        print(f"   Best model saved to: runs/detect/train/weights/best.pt")
        print()

        return results

    def validate(self, data_yaml, weights='best.pt'):
        """Validate model on test set"""
        metrics = self.model.val(
            data=data_yaml,
            weights=f'runs/detect/train/weights/{weights}'
        )

        print("=" * 80)
        print("VALIDATION METRICS")
        print("=" * 80)
        print(f"mAP@0.5: {metrics.box.map50:.3f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
        print(f"Precision: {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
        print()

        return metrics

    def export_for_deployment(self, format='onnx'):
        """
        Export model for production deployment

        Formats:
          'onnx': Universal format (recommended)
          'torchscript': PyTorch deployment
          'coreml': iOS deployment
          'tflite': Android/mobile deployment
          'engine': TensorRT (NVIDIA GPUs)
        """
        print(f"Exporting model to {format.upper()} format...")

        self.model.export(format=format)

        print(f" Model exported: runs/detect/train/weights/best.{format}")
        print()


# ==========================================
# EXAMPLE TRAINING WORKFLOW
# ==========================================

def train_meal_portion_detector():
    """
    Complete workflow: Train meal portion detection model

    This is what you'd run after collecting and labeling your dataset
    """

    # Step 1: Create dataset (assuming images already labeled)
    data_yaml = create_meal_portion_dataset()

    # Step 2: Train model
    trainer = CustomYOLOTrainer(model_size='n')  # Nano for speed

    # Uncomment to actually train (requires labeled dataset)
    # results = trainer.train(
    #     data_yaml=data_yaml,
    #     epochs=100,
    #     imgsz=640,
    #     batch=16
    # )

    # Step 3: Validate
    # metrics = trainer.validate(data_yaml)

    # Step 4: Export for deployment
    # trainer.export_for_deployment(format='onnx')

    print("=" * 80)
    print("MEAL PORTION DETECTOR: Ready for Production")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Integrate into your health tracking app")
    print("  2. Point camera at meal � Auto-detect portions")
    print("  3. Estimate calories based on portion sizes")
    print("  4. Log to database � Daily nutrition tracking")
    print()


def train_receipt_detector():
    """
    Complete workflow: Train receipt component detector
    """

    data_yaml = create_receipt_detection_dataset()

    trainer = CustomYOLOTrainer(model_size='s')  # Small for accuracy

    # Uncomment to train
    # results = trainer.train(data_yaml, epochs=150)
    # metrics = trainer.validate(data_yaml)
    # trainer.export_for_deployment(format='onnx')

    print("=" * 80)
    print("RECEIPT DETECTOR: Ready for Production")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Integrate into expense tracking app")
    print("  2. Detect receipt � Extract text regions")
    print("  3. OCR on detected regions � Parse amounts, dates")
    print("  4. Auto-categorize expenses � Financial insights")
    print()


# ==========================================
# QUICK START GUIDE
# ==========================================

print("=" * 80)
print("CUSTOM YOLO TRAINING: Quick Start Guide")
print("=" * 80)
print()
print("Step 1: Collect Images")
print(" 50-200 images per class minimum")
print(" Diverse angles, lighting, backgrounds")
print()
print("Step 2: Label Images")
print(" Use LabelImg: pip install labelimg")
print(" Or use Roboflow (web-based, easier)")
print(" Export in YOLO format")
print()
print("Step 3: Train Model")
print(" Run: trainer.train(data_yaml, epochs=100)")
print(" Monitor training: tensorboard --logdir runs")
print(" Wait 1-4 hours for training to complete")
print()
print("Step 4: Deploy")
print(" Export model: trainer.export_for_deployment()")
print(" Integrate into your app")
print(" Start detecting YOUR objects!")
print()

# Example workflows (uncomment to run)
# train_meal_portion_detector()
# train_receipt_detector()

print("=" * 80)
print("PART 3 COMPLETE: You can now train YOLO on anything.")
print("Final Part: Production deployment and optimization.")
print("=" * 80)
