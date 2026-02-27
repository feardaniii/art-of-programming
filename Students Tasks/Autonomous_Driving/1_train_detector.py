"""
AUTONOMOUS DRIVING - Train Object Detector on KITTI
=====================================================
MobileNetV2 backbone + dual detection heads trained on the KITTI benchmark.

What this script does:
  1. Explores the KITTI dataset (class distribution, sample images with GT boxes)
  2. Builds a MobileNetV2 + dual-head model (classification + bbox regression)
  3. Trains in two stages: frozen backbone -> fine-tuning top layers
  4. Evaluates with accuracy, IoU, and visual predictions vs ground truth

Architecture:
  Input (224x224x3) -> MobileNetV2 (frozen) -> GlobalAvgPool -> Dense layers
    -> Classification head (Car / Pedestrian / Cyclist)
    -> Bounding box head (x1, y1, x2, y2 normalized to [0,1])

Dataset: KITTI Object Detection Benchmark
  - 7,481 training images WITH labels  (we use these)
  - 7,518 testing images WITHOUT labels (held out for official benchmark)
  - We split training 80/20 into our own train/validation sets

Requirements: pip install tensorflow opencv-python matplotlib numpy
Run: python 1_train_detector.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ════════════════════════════════════════════════════════════════
# CONFIGURATION  (edit these if needed)
# ════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR   = os.path.join(SCRIPT_DIR, "kitti dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "training", "image_2")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "training", "label_2")

IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 16

STAGE1_EPOCHS = 15          # Frozen backbone
STAGE2_EPOCHS = 10          # Fine-tuning
FINE_TUNE_LAYERS = 30       # How many backbone layers to unfreeze

# The three object classes we detect
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Map KITTI raw labels -> our 3 classes (None = skip this type)
KITTI_CLASS_MAP = {
    'Car': 'Car',
    'Van': 'Car',                # Vans are close enough to cars
    'Pedestrian': 'Pedestrian',
    'Person_sitting': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Truck': None,               # Too different, skip
    'Tram': None,
    'Misc': None,
    'DontCare': None,
}

# Colors for visualization (matplotlib hex)
CLASS_COLORS = {'Car': '#FF4444', 'Pedestrian': '#4488FF', 'Cyclist': '#44DD44'}


# ════════════════════════════════════════════════════════════════
# KITTI LABEL PARSER
# ════════════════════════════════════════════════════════════════

def parse_kitti_label(label_path):
    """
    Parse a KITTI label .txt file.

    KITTI format (15 space-separated fields per line):
      Type Truncated Occluded Alpha  BBox(L,T,R,B)  3D(H,W,L)  Loc(X,Y,Z)  RotY

    We extract: Type and 2D bounding box (fields 0, 4-7).
    Returns list of dicts with keys: class_name, class_id, bbox, truncated, occluded
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            raw_type = parts[0]
            mapped = KITTI_CLASS_MAP.get(raw_type)
            if mapped is None:
                continue

            objects.append({
                'class_name': mapped,
                'class_id':   CLASS_TO_ID[mapped],
                'bbox':       [float(parts[4]), float(parts[5]),
                               float(parts[6]), float(parts[7])],
                'truncated':  float(parts[1]),
                'occluded':   int(parts[2]),
            })
    return objects


# ════════════════════════════════════════════════════════════════
# IoU (INTERSECTION OVER UNION)
# ════════════════════════════════════════════════════════════════

def calculate_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ════════════════════════════════════════════════════════════════
# DATA GENERATOR
# ════════════════════════════════════════════════════════════════

class KITTIDataGenerator(keras.utils.Sequence):
    """
    Loads KITTI images and labels on-the-fly during training.

    Each sample is (image_path, class_id, raw_bbox_pixels).
    Normalization and augmentation happen inside __getitem__.
    """

    def __init__(self, samples, batch_size=16, img_size=(IMG_WIDTH, IMG_HEIGHT),
                 shuffle=True, augment=False):
        self.samples    = list(samples)
        self.batch_size = batch_size
        self.img_size   = img_size      # (width, height) for cv2.resize
        self.shuffle    = shuffle
        self.augment    = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size : (idx + 1) * self.batch_size]

        images, classes, bboxes = [], [], []
        for img_path, class_id, raw_bbox in batch:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Normalize bounding box to [0, 1] relative to original image
            x1, y1, x2, y2 = raw_bbox
            bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]

            # Resize image to model input size and scale pixels to [0, 1]
            img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0

            # --- Data augmentation ---
            if self.augment:
                # Random horizontal flip (mirror bbox too)
                if np.random.random() > 0.5:
                    img = np.fliplr(img).copy()
                    bx1, by1, bx2, by2 = bbox_norm
                    bbox_norm = [1.0 - bx2, by1, 1.0 - bx1, by2]

                # Random brightness shift
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.7, 1.3)
                    img = np.clip(img * factor, 0.0, 1.0)

            images.append(img)
            classes.append(class_id)
            bboxes.append(bbox_norm)

        return (
            np.array(images,  dtype=np.float32),
            {
                'classification': np.array(classes, dtype=np.int32),
                'bbox':           np.array(bboxes,  dtype=np.float32),
            }
        )


# ════════════════════════════════════════════════════════════════
# BUILD DETECTION MODEL
# ════════════════════════════════════════════════════════════════

def build_detection_model():
    """
    MobileNetV2 backbone + two output heads.

    Think of it like a car mechanic:
      - The BACKBONE (MobileNetV2) is the experienced eye that sees features
      - The CLASSIFICATION HEAD answers: "What is the main object?" (Car/Ped/Cyclist)
      - The BBOX HEAD answers: "Where exactly is it?" (x1, y1, x2, y2)

    Returns: (model, base_model) so we can unfreeze base_model later
    """
    # --- Backbone: pre-trained MobileNetV2 (ImageNet weights) ---
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False   # Frozen for Stage 1

    # --- Detection heads ---
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)   # training=False keeps BatchNorm stable
    x = layers.GlobalAveragePooling2D()(x)

    # Shared dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Head 1: Classification (which class?)
    cls_head = layers.Dense(NUM_CLASSES, activation='softmax',
                            name='classification')(x)

    # Head 2: Bounding box regression (where is it?)
    # sigmoid constrains output to [0,1] — perfect for normalized coordinates
    bbox_head = layers.Dense(4, activation='sigmoid',
                             name='bbox')(x)

    model = keras.Model(
        inputs=inputs,
        outputs={'classification': cls_head, 'bbox': bbox_head},
        name='KITTI_Detector'
    )

    return model, base_model


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("   AUTONOMOUS DRIVING — Object Detection Training Pipeline")
    print("   Model:   MobileNetV2 + Classification Head + BBox Head")
    print("   Dataset: KITTI Object Detection (7,481 labeled images)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Verify dataset exists
    # ------------------------------------------------------------------
    if not os.path.isdir(TRAIN_IMG_DIR):
        print(f"ERROR: Training images not found at:")
        print(f"  {TRAIN_IMG_DIR}")
        print()
        print("Download the KITTI Object Detection dataset from:")
        print("  https://www.cvlibs.net/datasets/kitti/eval_object.php")
        print("  -> Download 'Left color images' and 'Training labels'")
        print("  -> Extract into 'kitti dataset/training/image_2/' and 'label_2/'")
        sys.exit(1)

    if not os.path.isdir(TRAIN_LBL_DIR):
        print(f"ERROR: Training labels not found at:")
        print(f"  {TRAIN_LBL_DIR}")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(TRAIN_LBL_DIR) if f.endswith('.txt')])
    print(f"Found {len(image_files)} training images")
    print(f"Found {len(label_files)} label files")
    print()

    # ==================================================================
    # SECTION 1: EXPLORE THE KITTI DATASET
    # ==================================================================
    print("=" * 70)
    print("  SECTION 1: EXPLORING THE KITTI DATASET")
    print("=" * 70)
    print()

    # --- Explain the dataset split situation ---
    print("WHY DO WE SPLIT THE TRAINING DATA?")
    print("-" * 50)
    print("KITTI provides two sets:")
    print(f"  Training:  {len(image_files):,} images WITH labels  (annotations)")
    print(f"  Testing:   7,518 images WITHOUT labels (for benchmark submission)")
    print()
    print("Since the test set has NO labels, we can't evaluate on it.")
    print("Solution: Split the 7,481 labeled images ourselves:")
    print("  -> 80% (5,985 images) for TRAINING")
    print("  -> 20% (1,496 images) for VALIDATION")
    print()

    # --- Explain the KITTI label format ---
    print("KITTI LABEL FORMAT")
    print("-" * 50)
    print("Each .txt file has one line per object with 15 fields:")
    print()
    print("  Type  Trunc  Occl  Alpha | BBox_L  BBox_T  BBox_R  BBox_B | 3D dims | 3D loc | RotY")
    print("  ^^^^                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("  class                       2D bounding box (pixels) — what we use!")
    print()

    # Show a real example
    sample_label = os.path.join(TRAIN_LBL_DIR, "000000.txt")
    if os.path.exists(sample_label):
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip()
        print(f"  Example (000000.txt):  {first_line}")
        print()

    # --- Parse all labels and gather statistics ---
    print("Parsing all labels...")
    all_objects = []
    objects_per_image = []
    samples = []  # (image_path, class_id, raw_bbox_pixels)
    skipped_images = 0

    for img_file in image_files:
        img_id = img_file.replace('.png', '')
        label_path = os.path.join(TRAIN_LBL_DIR, f"{img_id}.txt")

        if not os.path.exists(label_path):
            skipped_images += 1
            continue

        objects = parse_kitti_label(label_path)
        all_objects.extend(objects)
        objects_per_image.append(len(objects))

        if objects:
            # Pick the LARGEST object (by bbox area) as our training target
            # This is a simplification — real detectors handle ALL objects
            largest = max(objects, key=lambda o:
                (o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1]))

            img_path = os.path.join(TRAIN_IMG_DIR, img_file)
            samples.append((img_path, largest['class_id'], largest['bbox']))
        else:
            skipped_images += 1

    # --- Print dataset statistics ---
    class_counts = Counter(o['class_name'] for o in all_objects)
    sample_class_counts = Counter(s[1] for s in samples)

    print(f"\nDATASET STATISTICS")
    print("-" * 50)
    print(f"Total annotated objects: {len(all_objects):,}")
    print(f"Images with valid objects: {len(samples):,}")
    print(f"Images skipped (no valid objects): {skipped_images}")
    print()

    print("Object counts by class (all annotations):")
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        bar = '#' * (count // 200)
        print(f"  {cls:12s}: {count:5,}  {bar}")
    print()

    print("Training samples by class (largest object per image):")
    for cls in CLASSES:
        count = sample_class_counts.get(CLASS_TO_ID[cls], 0)
        print(f"  {cls:12s}: {count:5,}")
    print()

    avg_objects = np.mean(objects_per_image) if objects_per_image else 0
    print(f"Average objects per image: {avg_objects:.1f}")
    print(f"Max objects in one image:  {max(objects_per_image) if objects_per_image else 0}")
    print()

    # --- Visualize sample images with ground truth boxes ---
    print("Saving dataset exploration plots...")

    sample_indices = [0, 10, 50, 100, 300, 1000]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("KITTI Dataset — Sample Images with Ground Truth Boxes", fontsize=14, fontweight='bold')

    for ax, idx in zip(axes.flat, sample_indices):
        if idx >= len(image_files):
            ax.axis('off')
            continue

        img_file = image_files[idx]
        img_id = img_file.replace('.png', '')
        img_path = os.path.join(TRAIN_IMG_DIR, img_file)
        label_path = os.path.join(TRAIN_LBL_DIR, f"{img_id}.txt")

        img = cv2.imread(img_path)
        if img is None:
            ax.axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects = parse_kitti_label(label_path) if os.path.exists(label_path) else []

        ax.imshow(img)
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            color = CLASS_COLORS[obj['class_name']]
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, obj['class_name'], fontsize=7,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        ax.set_title(f"Image {img_id}  ({len(objects)} objects)", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'kitti_exploration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: kitti_exploration.png")

    # --- Class distribution bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CLASS_COLORS[cls] for cls in CLASSES]
    counts = [class_counts.get(cls, 0) for cls in CLASSES]
    bars = ax.bar(CLASSES, counts, color=colors, edgecolor='black', linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count:,}", ha='center', fontweight='bold', fontsize=11)
    ax.set_title("KITTI — Object Class Distribution (All Annotations)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Objects")
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(SCRIPT_DIR, 'class_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: class_distribution.png")
    print()

    input("Press Enter to build the model and start training...")
    print()

    # ==================================================================
    # SECTION 2: PREPARE DATA + BUILD MODEL
    # ==================================================================
    print("=" * 70)
    print("  SECTION 2: PREPARING DATA AND BUILDING THE MODEL")
    print("=" * 70)
    print()

    # --- Train / Validation split ---
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:]

    print(f"Train/Val split:")
    print(f"  Training:   {len(train_samples):,} samples")
    print(f"  Validation: {len(val_samples):,} samples")
    print()

    # --- Create data generators ---
    train_gen = KITTIDataGenerator(
        train_samples, batch_size=BATCH_SIZE, shuffle=True, augment=True
    )
    val_gen = KITTIDataGenerator(
        val_samples, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )

    print(f"Batches per epoch:  {len(train_gen)} train, {len(val_gen)} val")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # --- Build model ---
    print("Building MobileNetV2 + Detection Heads...")
    model, base_model = build_detection_model()
    print()

    # --- Compile ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'bbox': 'mean_squared_error',
        },
        loss_weights={
            'classification': 1.0,
            'bbox': 5.0,        # Bbox accuracy is critical for detection
        },
        metrics={
            'classification': ['accuracy'],
            'bbox': ['mae'],
        }
    )

    print("MODEL SUMMARY")
    print("-" * 50)
    model.summary(print_fn=lambda x: print(f"  {x}"))
    print()
    print(f"MobileNetV2 backbone parameters: {base_model.count_params():,} (frozen)")
    print(f"Total model parameters:          {model.count_params():,}")
    print()

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            os.path.join(SCRIPT_DIR, 'av_perception_best.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # ==================================================================
    # SECTION 3: TRAINING
    # ==================================================================
    print("=" * 70)
    print("  SECTION 3: TRAINING")
    print("=" * 70)
    print()
    print("STAGE 1 — Frozen backbone (transfer learning)")
    print(f"  Epochs: up to {STAGE1_EPOCHS} (EarlyStopping patience=5)")
    print(f"  Learning rate: 0.001")
    print(f"  MobileNetV2 layers: FROZEN (only training the detection heads)")
    print()

    try:
        history1 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=STAGE1_EPOCHS, callbacks=callbacks, verbose=1
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted! Saving current model...")
        model.save(os.path.join(SCRIPT_DIR, 'av_perception_interrupted.keras'))
        print("  Saved: av_perception_interrupted.keras")
        return

    print()
    print("STAGE 1 COMPLETE")
    stage1_epochs = len(history1.history['loss'])
    print(f"  Ran {stage1_epochs} epochs")
    print(f"  Best val_loss: {min(history1.history['val_loss']):.4f}")
    print()

    # --- Stage 2: Fine-tuning ---
    print("STAGE 2 — Fine-tuning top backbone layers")
    print(f"  Unfreezing last {FINE_TUNE_LAYERS} layers of MobileNetV2")
    print(f"  Learning rate: 0.00001 (100x lower to avoid destroying pretrained features)")
    print()

    base_model.trainable = True
    freeze_until = len(base_model.layers) - FINE_TUNE_LAYERS
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Backbone layers: {trainable} trainable, {frozen} frozen")
    print()

    # Re-compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'bbox': 'mean_squared_error',
        },
        loss_weights={'classification': 1.0, 'bbox': 5.0},
        metrics={'classification': ['accuracy'], 'bbox': ['mae']}
    )

    callbacks_ft = [
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(SCRIPT_DIR, 'av_perception_best.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
    ]

    try:
        history2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=STAGE2_EPOCHS, callbacks=callbacks_ft, verbose=1
        )
    except KeyboardInterrupt:
        print("\n  Fine-tuning interrupted! Saving current model...")
        model.save(os.path.join(SCRIPT_DIR, 'av_perception_interrupted.keras'))
        print("  Saved: av_perception_interrupted.keras")
        return

    print()
    print("STAGE 2 COMPLETE")
    stage2_epochs = len(history2.history['loss'])
    print(f"  Ran {stage2_epochs} epochs")
    print()

    # --- Save final model ---
    model_path = os.path.join(SCRIPT_DIR, 'av_perception_final.keras')
    model.save(model_path)
    print(f"Model saved: {model_path}")
    print()

    # --- Plot training curves ---
    print("Saving training curves...")

    # Combine histories from both stages
    full = {}
    for key in history1.history:
        full[key] = history1.history[key] + history2.history.get(key, [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves (Stage 1: Frozen  |  Stage 2: Fine-tuning)",
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, len(full['loss']) + 1)

    # Total loss
    axes[0].plot(epochs_range, full['loss'], 'b-', label='Train', linewidth=1.5)
    axes[0].plot(epochs_range, full['val_loss'], 'r-', label='Validation', linewidth=1.5)
    axes[0].axvline(x=stage1_epochs + 0.5, color='gray', linestyle='--', alpha=0.5,
                    label='Fine-tune start')
    axes[0].set_title('Total Loss (weighted)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Classification accuracy
    axes[1].plot(epochs_range, full['classification_accuracy'], 'b-', label='Train', linewidth=1.5)
    axes[1].plot(epochs_range, full['val_classification_accuracy'], 'r-', label='Val', linewidth=1.5)
    axes[1].axvline(x=stage1_epochs + 0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Bbox MAE
    axes[2].plot(epochs_range, full['bbox_mae'], 'b-', label='Train', linewidth=1.5)
    axes[2].plot(epochs_range, full['val_bbox_mae'], 'r-', label='Val', linewidth=1.5)
    axes[2].axvline(x=stage1_epochs + 0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('Bbox Mean Absolute Error')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves.png")
    print()

    input("Training complete! Press Enter to evaluate on the validation set...")
    print()

    # ==================================================================
    # SECTION 4: EVALUATION
    # ==================================================================
    print("=" * 70)
    print("  SECTION 4: EVALUATION ON VALIDATION SET")
    print("=" * 70)
    print()

    correct_class = 0
    total = 0
    ious = []
    per_class_correct = Counter()
    per_class_total = Counter()

    print("Evaluating on validation set...")
    for img_path, gt_class_id, gt_bbox in val_samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Prepare input
        img_input = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_input, axis=0)

        # Predict
        preds = model.predict(img_batch, verbose=0)
        pred_class_id = int(np.argmax(preds['classification'][0]))
        pred_bbox_norm = preds['bbox'][0]

        # Denormalize predicted bbox to pixel coords
        pred_bbox_px = [
            pred_bbox_norm[0] * w, pred_bbox_norm[1] * h,
            pred_bbox_norm[2] * w, pred_bbox_norm[3] * h,
        ]

        # Classification accuracy
        is_correct = (pred_class_id == gt_class_id)
        correct_class += int(is_correct)
        total += 1
        per_class_total[gt_class_id] += 1
        if is_correct:
            per_class_correct[gt_class_id] += 1

        # IoU
        iou = calculate_iou(gt_bbox, pred_bbox_px)
        ious.append(iou)

    # --- Print evaluation results ---
    print(f"\nEVALUATION RESULTS ({total} validation samples)")
    print("-" * 50)
    print(f"Overall classification accuracy: {correct_class / total:.1%}")
    print()
    print("Per-class accuracy:")
    for cls_id in range(NUM_CLASSES):
        cls_name = ID_TO_CLASS[cls_id]
        cls_total = per_class_total[cls_id]
        cls_correct = per_class_correct[cls_id]
        if cls_total > 0:
            print(f"  {cls_name:12s}: {cls_correct}/{cls_total}  ({cls_correct / cls_total:.1%})")
        else:
            print(f"  {cls_name:12s}: (no samples)")
    print()
    print(f"Bounding box IoU:")
    print(f"  Mean IoU:    {np.mean(ious):.3f}")
    print(f"  Median IoU:  {np.median(ious):.3f}")
    print(f"  IoU > 0.50:  {sum(1 for i in ious if i > 0.5)}/{total} ({sum(1 for i in ious if i > 0.5) / total:.1%})")
    print(f"  IoU > 0.75:  {sum(1 for i in ious if i > 0.75)}/{total} ({sum(1 for i in ious if i > 0.75) / total:.1%})")
    print()

    # --- Visualize predictions vs ground truth ---
    print("Saving sample predictions...")

    num_vis = min(6, len(val_samples))
    vis_indices = np.linspace(0, len(val_samples) - 1, num_vis, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Predictions vs Ground Truth   (Green = GT,  Red = Prediction)",
                 fontsize=13, fontweight='bold')

    for ax, idx in zip(axes.flat, vis_indices):
        img_path, gt_class_id, gt_bbox = val_samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            ax.axis('off')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Predict
        img_input = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
        preds = model.predict(np.expand_dims(img_input, 0), verbose=0)
        pred_cls = int(np.argmax(preds['classification'][0]))
        pred_conf = float(preds['classification'][0][pred_cls])
        pb = preds['bbox'][0]
        pred_bbox = [pb[0] * w, pb[1] * h, pb[2] * w, pb[3] * h]

        iou = calculate_iou(gt_bbox, pred_bbox)

        ax.imshow(img_rgb)

        # Ground truth box (green)
        gx1, gy1, gx2, gy2 = gt_bbox
        rect_gt = Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                             linewidth=3, edgecolor='#00CC00', facecolor='none',
                             linestyle='--')
        ax.add_patch(rect_gt)

        # Prediction box (red)
        px1, py1, px2, py2 = pred_bbox
        rect_pred = Rectangle((px1, py1), px2 - px1, py2 - py1,
                               linewidth=2, edgecolor='#FF0000', facecolor='none')
        ax.add_patch(rect_pred)

        gt_name = ID_TO_CLASS[gt_class_id]
        pred_name = ID_TO_CLASS[pred_cls]
        ax.set_title(f"GT: {gt_name}  |  Pred: {pred_name} ({pred_conf:.0%})\nIoU: {iou:.2f}",
                     fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sample_predictions.png")
    print()

    # --- Final summary ---
    print("=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Saved files:")
    print(f"  Model:              av_perception_final.keras")
    print(f"  Best checkpoint:    av_perception_best.keras")
    print(f"  Dataset plots:      kitti_exploration.png, class_distribution.png")
    print(f"  Training curves:    training_curves.png")
    print(f"  Predictions:        sample_predictions.png")
    print()
    print("Next step: Run 2_inference.py to test on images, videos, or webcam!")
    print()


if __name__ == '__main__':
    main()
