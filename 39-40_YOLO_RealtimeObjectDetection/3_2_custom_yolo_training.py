"""
================================================================================
YOLOv8 CUSTOM TRAINING: Train Your Own Object Detector
================================================================================

Course: The Art of Programming - Computer Vision Module
Lesson: From Zero to Custom Object Detection

LEARNING OBJECTIVES:
    1. Understand the YOLO annotation format
    2. Create a properly structured dataset
    3. Train a model (for real, not just print statements)
    4. Evaluate results and use your trained model

REQUIREMENTS:
    pip install ultralytics opencv-python pyyaml pillow

REALISTIC EXPECTATIONS:
    - Training needs labeled data (50-500 images per class minimum)
    - GPU recommended (CPU works but 10-50x slower)
    - Even small training runs take 10-30 minutes
    - Quality depends heavily on data quality

================================================================================
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# =============================================================================
# PART 1: UNDERSTANDING YOLO ANNOTATION FORMAT
# =============================================================================

"""
YOLO uses a simple text-based annotation format.

For each image "photo.jpg", you need a "photo.txt" with annotations.

Each line in the .txt file represents ONE object:
    <class_id> <x_center> <y_center> <width> <height>

All coordinates are NORMALIZED (0.0 to 1.0, relative to image size).

Example for a 640x480 image with a dog at pixels (100, 120) to (300, 400):

    Image dimensions: 640 x 480
    Bounding box: x_min=100, y_min=120, x_max=300, y_max=400

    x_center = (100 + 300) / 2 / 640 = 0.3125
    y_center = (120 + 400) / 2 / 480 = 0.5417
    width    = (300 - 100) / 640     = 0.3125
    height   = (400 - 120) / 480     = 0.5833

    If dog is class_id=0, the annotation line is:
    0 0.3125 0.5417 0.3125 0.5833
"""


def convert_bbox_to_yolo(
        img_width: int,
        img_height: int,
        bbox: Tuple[int, int, int, int],  # (x_min, y_min, x_max, y_max) in pixels
        class_id: int
) -> str:
    """
    Convert pixel bounding box to YOLO format.

    This is the core conversion you need to understand.
    Every labeling tool does this internally.
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate center point (normalized)
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height

    # Calculate dimensions (normalized)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_yolo_to_bbox(
        img_width: int,
        img_height: int,
        yolo_line: str
) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Convert YOLO format back to pixel bounding box.

    Useful for visualization and verification.
    """
    parts = yolo_line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return class_id, (x_min, y_min, x_max, y_max)


# =============================================================================
# PART 2: DATASET STRUCTURE
# =============================================================================

"""
YOLO expects this exact folder structure:

    my_dataset/
    ├── data.yaml           # Configuration file
    ├── images/
    │   ├── train/          # Training images (80% of data)
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── val/            # Validation images (20% of data)
    │       ├── img101.jpg
    │       └── ...
    └── labels/
        ├── train/          # Annotation files (same names as images)
        │   ├── img001.txt
        │   ├── img002.txt
        │   └── ...
        └── val/
            ├── img101.txt
            └── ...

The data.yaml file contains:
    path: /absolute/path/to/my_dataset
    train: images/train
    val: images/val
    nc: 3                    # Number of classes
    names: ['cat', 'dog', 'bird']  # Class names
"""


@dataclass
class DatasetConfig:
    """Configuration for creating a YOLO dataset."""
    name: str
    class_names: List[str]
    base_path: Path = Path("datasets")
    train_ratio: float = 0.8  # 80% train, 20% val

    @property
    def dataset_path(self) -> Path:
        return self.base_path / self.name

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def create_dataset_structure(config: DatasetConfig) -> Path:
    """
    Create the required folder structure for YOLO training.

    Returns path to data.yaml file.
    """
    dataset_path = config.dataset_path

    # Create directories
    directories = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    data_yaml = {
        "path": str(dataset_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": config.num_classes,
        "names": config.class_names,
    }

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Created dataset structure at: {dataset_path}")
    print(f"Classes ({config.num_classes}): {config.class_names}")
    print(f"Configuration: {yaml_path}")

    return yaml_path


# =============================================================================
# PART 3: SYNTHETIC DATA GENERATOR (For Learning/Testing)
# =============================================================================

"""
In real projects, you'd collect and label real images.
For learning, we'll generate synthetic data to test the pipeline.

This teaches you:
    - How annotations relate to images
    - The full training workflow
    - What to expect from a trained model
"""


def generate_synthetic_dataset(
        config: DatasetConfig,
        num_images: int = 100,
        img_size: Tuple[int, int] = (640, 480),
        objects_per_image: Tuple[int, int] = (1, 5)
) -> Path:
    """
    Generate synthetic training data with colored shapes.

    This creates a COMPLETE, RUNNABLE dataset for testing the training pipeline.

    Classes: Different colored shapes
    """
    print()
    print("=" * 70)
    print("GENERATING SYNTHETIC DATASET")
    print("=" * 70)
    print()

    yaml_path = create_dataset_structure(config)
    dataset_path = config.dataset_path

    # Define colors for each class (BGR format)
    class_colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    # Ensure we have enough colors
    while len(class_colors) < config.num_classes:
        class_colors.append((
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        ))

    # Split into train/val
    num_train = int(num_images * config.train_ratio)
    num_val = num_images - num_train

    def create_image_with_objects(img_idx: int, split: str):
        """Generate one image with random objects."""
        # Create blank image with random background
        bg_color = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        )
        img = np.full((img_size[1], img_size[0], 3), bg_color, dtype=np.uint8)

        # Add some noise/texture
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        annotations = []
        num_objects = random.randint(*objects_per_image)

        for _ in range(num_objects):
            # Random class
            class_id = random.randint(0, config.num_classes - 1)
            color = class_colors[class_id]

            # Random size and position
            obj_w = random.randint(50, 150)
            obj_h = random.randint(50, 150)
            x_min = random.randint(10, img_size[0] - obj_w - 10)
            y_min = random.randint(10, img_size[1] - obj_h - 10)
            x_max = x_min + obj_w
            y_max = y_min + obj_h

            # Draw shape based on class
            if class_id % 3 == 0:  # Rectangle
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, -1)
            elif class_id % 3 == 1:  # Circle
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                radius = min(obj_w, obj_h) // 2
                cv2.circle(img, center, radius, color, -1)
            else:  # Triangle
                pts = np.array([
                    [(x_min + x_max) // 2, y_min],
                    [x_min, y_max],
                    [x_max, y_max]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)

            # Create YOLO annotation
            yolo_line = convert_bbox_to_yolo(
                img_size[0], img_size[1],
                (x_min, y_min, x_max, y_max),
                class_id
            )
            annotations.append(yolo_line)

        # Save image
        img_name = f"img_{img_idx:04d}.jpg"
        img_path = dataset_path / "images" / split / img_name
        cv2.imwrite(str(img_path), img)

        # Save annotations
        label_name = f"img_{img_idx:04d}.txt"
        label_path = dataset_path / "labels" / split / label_name
        with open(label_path, "w") as f:
            f.write("\n".join(annotations))

        return img_name

    # Generate training images
    print(f"Generating {num_train} training images...")
    for i in range(num_train):
        create_image_with_objects(i, "train")
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_train}")

    # Generate validation images
    print(f"Generating {num_val} validation images...")
    for i in range(num_val):
        create_image_with_objects(num_train + i, "val")

    print()
    print(f"✓ Dataset generated: {num_images} images")
    print(f"  Training: {num_train} images")
    print(f"  Validation: {num_val} images")
    print()

    return yaml_path


# =============================================================================
# PART 4: TRAINING
# =============================================================================

"""
Training a YOLO model involves:

1. TRANSFER LEARNING: Start from pre-trained COCO weights
   - Model already knows edges, shapes, textures
   - Only needs to learn YOUR specific classes
   - Requires 10-100x less data than training from scratch

2. FINE-TUNING: Adjust weights for your data
   - Epochs: How many times to see all training data
   - Batch size: Images processed together (limited by GPU memory)
   - Image size: Larger = more accurate but slower

3. VALIDATION: Check performance on held-out data
   - mAP (mean Average Precision): Main metric
   - Precision: Of detections made, how many were correct?
   - Recall: Of objects present, how many were found?
"""


def train_model(
        data_yaml: Path,
        model_size: str = "n",  # n, s, m, l, x
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",  # "0" for GPU, "cpu" for CPU
        project: str = "runs/train",
        name: str = "custom_model"
) -> Path:
    """
    Train YOLOv8 on custom dataset.

    Args:
        data_yaml: Path to data.yaml configuration
        model_size: Model size (n=nano fastest, x=xlarge most accurate)
        epochs: Training iterations (50-300 typical)
        imgsz: Input image size
        batch: Batch size (reduce if out of memory)
        device: "0" for GPU, "cpu" for CPU
        project: Output directory
        name: Experiment name

    Returns:
        Path to best trained weights
    """
    print()
    print("=" * 70)
    print("TRAINING YOLOv8 MODEL")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {'GPU' if device != 'cpu' else 'CPU'}")
    print()

    # Load pre-trained model (transfer learning)
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        patience=20,  # Early stopping patience
        save=True,  # Save checkpoints
        plots=True,  # Generate training plots
        verbose=True,
    )

    # Find best weights
    weights_path = Path(project) / name / "weights" / "best.pt"

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"✓ Best weights saved to: {weights_path}")
    print(f"✓ Training plots saved to: {Path(project) / name}")
    print()

    return weights_path


# =============================================================================
# PART 5: INFERENCE WITH CUSTOM MODEL
# =============================================================================

def run_inference(
        weights_path: Path,
        source: str,  # Image path, video path, or 0 for webcam
        conf_threshold: float = 0.5
):
    """
    Run detection using your trained model.

    Args:
        weights_path: Path to trained weights (best.pt)
        source: Image/video path or 0 for webcam
        conf_threshold: Minimum confidence to show detection
    """
    print()
    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    print()

    # Load YOUR trained model
    model = YOLO(str(weights_path))

    print(f"Model: {weights_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf_threshold}")
    print()

    if source == 0 or source == "0":
        # Webcam - live detection
        print("Starting webcam... Press 'Q' to quit")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold, verbose=False)[0]
            annotated = results.plot()

            cv2.imshow("Custom YOLOv8 Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Single image or video
        results = model(source, conf=conf_threshold)

        for result in results:
            annotated = result.plot()
            cv2.imshow("Detection Result", annotated)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


# =============================================================================
# PART 6: COMPLETE EXAMPLE - RUN THIS
# =============================================================================

def complete_training_example():
    """
    Complete end-to-end example you can actually run.

    This will:
    1. Generate a synthetic dataset (colored shapes)
    2. Train a YOLOv8 model to detect them
    3. Test the trained model on webcam

    Takes about 5-15 minutes depending on your hardware.
    """
    print()
    print("=" * 70)
    print("COMPLETE CUSTOM YOLO TRAINING EXAMPLE")
    print("=" * 70)
    print()
    print("This example trains YOLO to detect colored shapes.")
    print("It demonstrates the FULL workflow from data to deployment.")
    print()

    # Step 1: Define your classes
    config = DatasetConfig(
        name="colored_shapes",
        class_names=[
            "red_rectangle",
            "green_circle",
            "blue_triangle",
        ]
    )

    # Step 2: Generate synthetic dataset
    data_yaml = generate_synthetic_dataset(
        config=config,
        num_images=100,  # More images = better model (try 200-500)
        img_size=(640, 480),
        objects_per_image=(1, 4)
    )

    # Step 3: Train model
    weights_path = train_model(
        data_yaml=data_yaml,
        model_size="n",  # Nano for speed
        epochs=30,  # Increase to 50-100 for better results
        imgsz=640,
        batch=16,  # Reduce to 8 if out of memory
        device="cpu",  # Change to "cpu" if no GPU set up yet ( see CUDA guides but except dependencies and installation struggle)
        name="shapes_detector"
    )

    # Step 4: Test on webcam
    print()
    print("Training complete! Testing on webcam...")
    print("Show colored objects to the camera to test detection.")
    print()

    run_inference(
        weights_path=weights_path,
        source=0,  # Webcam
        conf_threshold=0.5
    )


# =============================================================================
# PART 7: REAL-WORLD WORKFLOW GUIDE
# =============================================================================

def print_realworld_guide():
    """
    Guide for training on REAL data (not synthetic).
    """
    guide = """
    ============================================================================
    REAL-WORLD CUSTOM TRAINING WORKFLOW
    ============================================================================

    STEP 1: COLLECT IMAGES
    ──────────────────────
    • Take 50-500 photos per class (more = better)
    • Vary: angles, lighting, backgrounds, distances
    • Include edge cases (partial occlusion, blur, etc.)
    • Match your deployment environment


    STEP 2: LABEL IMAGES
    ────────────────────
    Option A: LabelImg (Local, Free)
        pip install labelimg
        labelimg                    # Opens GUI
        → Draw boxes → Save as YOLO format

    Option B: Roboflow (Web-based, Easy)
        → Upload images to roboflow.com
        → Label in browser
        → Export as "YOLOv8" format
        → Download and extract

    Option C: CVAT (Self-hosted, Powerful)
        → Best for large teams
        → More features than LabelImg


    STEP 3: ORGANIZE DATASET
    ────────────────────────
    After labeling, ensure this structure:

        my_dataset/
        ├── data.yaml
        ├── images/
        │   ├── train/   (80% of images)
        │   └── val/     (20% of images)
        └── labels/
            ├── train/   (matching .txt files)
            └── val/


    STEP 4: TRAIN
    ─────────────
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # Start with nano
    model.train(
        data="path/to/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
    )


    STEP 5: EVALUATE
    ────────────────
    Check runs/train/experiment/results.png for:
    • Loss curves (should decrease)
    • mAP (should increase, aim for >0.7)
    • Precision/Recall balance


    STEP 6: DEPLOY
    ──────────────
    # Load your trained model
    model = YOLO("runs/train/experiment/weights/best.pt")

    # Use it
    results = model("test_image.jpg")
    results[0].show()


    COMMON ISSUES
    ─────────────
    "CUDA out of memory"
        → Reduce batch size (try 8, 4, or 2)
        → Reduce image size (try 480 or 320)

    "Low mAP after training"
        → Need more training data
        → Check label quality (boxes tight around objects?)
        → Train for more epochs
        → Try larger model (s instead of n)

    "Model detects wrong classes"
        → Classes might look too similar
        → Need more diverse training examples
        → Check that labels are correct

    ============================================================================
    """
    print(guide)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("YOLOv8 CUSTOM TRAINING TUTORIAL")
    print("=" * 70)
    print()
    print("Options:")
    print("  1. Run complete training example (synthetic data)")
    print("  2. Print real-world workflow guide")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        complete_training_example()
    elif choice == "2":
        print_realworld_guide()
    else:
        print("Running complete example by default...")
        complete_training_example()