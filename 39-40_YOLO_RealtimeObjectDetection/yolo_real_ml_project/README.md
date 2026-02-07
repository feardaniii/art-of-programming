# YOLOv8 Custom Training Project

## Real-World Object Detection from Scratch

This project teaches you the **complete pipeline** for training a custom object detector:

```
Capture Images → Label Objects → Augment Data → Train Model → Deploy
```

By the end, you'll have a model that detects YOUR objects in real-time.

---

## Project Structure

```
yolo_training_project/
├── config/
│   └── config.yaml              # All settings in one place
├── scripts/
│   ├── 01_capture_data.py       # Collect images with webcam
│   ├── 02_label_data.py         # Draw bounding boxes on images
│   ├── 03_augment_data.py       # Generate variations (blur, rotate, etc.)
│   ├── 04_prepare_dataset.py    # Split into train/val, create data.yaml
│   ├── 05_train.py              # Train YOLOv8
│   ├── 06_evaluate.py           # Check model performance
│   └── 07_inference.py          # Run your trained model
├── src/
│   ├── augmentation.py          # Augmentation techniques explained
│   ├── dataset.py               # Dataset management utilities
│   ├── labeling.py              # Labeling tool logic
│   └── utils.py                 # Helper functions
├── data/
│   ├── raw/                     # Your captured images go here
│   ├── labeled/                 # Images + annotation files
│   ├── augmented/               # Augmented training data
│   └── final/                   # Ready for YOLO (train/val split)
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python numpy pillow pyyaml albumentations
```

### 2. Configure Your Classes

Edit `config/config.yaml`:

```yaml
classes:
  - phone
  - wallet
  - keys
```

### 3. Run Each Script in Order

```bash
python scripts/01_capture_data.py    # Capture ~50 images per class
python scripts/02_label_data.py      # Label all images
python scripts/03_augment_data.py    # Generate augmented versions
python scripts/04_prepare_dataset.py # Prepare final dataset
python scripts/05_train.py           # Train model (~30 min)
python scripts/06_evaluate.py        # Check results
python scripts/07_inference.py       # Test on webcam!
```

---

## The Pipeline Explained

### Step 1: Data Capture

**Goal:** Collect diverse images of your objects.

**Best Practices:**
- 50-200 images per class minimum
- Vary backgrounds (desk, floor, hand, fabric)
- Vary lighting (bright, dim, shadows)
- Vary angles (top-down, side, tilted)
- Vary distances (close-up, medium, far)
- Include partial occlusion (object partially hidden)

### Step 2: Labeling

**Goal:** Draw bounding boxes around each object.

**YOLO Format:** Each image needs a `.txt` file with:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to 0-1.

### Step 3: Data Augmentation

**Goal:** Artificially expand your dataset with realistic variations.

**Why It Matters:**
- 100 real images → 1000+ augmented images
- Model sees more variety → generalizes better
- Prevents overfitting

**Augmentation Types:**

| Category | Techniques | Purpose |
|----------|------------|---------|
| Geometric | Rotation, flip, scale, crop | Viewpoint invariance |
| Color | Brightness, contrast, saturation | Lighting invariance |
| Blur | Gaussian, motion blur | Handle camera shake |
| Noise | Gaussian noise, compression | Sensor noise robustness |
| Occlusion | Random erasing, cutout | Handle partial visibility |

### Step 4: Dataset Preparation

**Goal:** Organize into YOLO-expected structure.

```
final/
├── data.yaml
├── images/
│   ├── train/  (80%)
│   └── val/    (20%)
└── labels/
    ├── train/
    └── val/
```

### Step 5: Training

**Key Parameters:**
- `epochs`: Training iterations (50-300)
- `batch`: Images per step (8-32, limited by GPU memory)
- `imgsz`: Input resolution (640 standard)
- `patience`: Early stopping (stop if no improvement)

### Step 6: Evaluation

**Metrics to Understand:**
- **mAP@0.5**: Main metric. "Of objects detected, what % were correct?"
- **Precision**: Low = too many false positives
- **Recall**: Low = missing objects
- **Confusion Matrix**: Which classes get confused?

### Step 7: Inference

**Deploy your model:**
```python
from ultralytics import YOLO
model = YOLO("path/to/best.pt")
results = model("image.jpg")
```

---

## Data Augmentation Deep Dive

This is where most beginners skip, but it's **critical** for real-world performance.

### Why Augment?

Your training images were taken in specific conditions. Your model will encounter:
- Different lighting
- Motion blur
- Various angles
- Partial occlusion
- Different backgrounds

Augmentation simulates these conditions.

### Augmentation Examples

**Original image of a phone:**

| Augmentation | Effect | Why It Helps |
|--------------|--------|--------------|
| Horizontal flip | Mirror image | Phone can be either orientation |
| Rotation ±15° | Slight tilt | Objects aren't always perfectly aligned |
| Brightness ±30% | Lighter/darker | Different room lighting |
| Gaussian blur | Soft focus | Camera shake, motion |
| Random crop | Zoom variations | Different distances |
| Color jitter | Hue/saturation shift | Different screens, reflections |
| Noise injection | Grainy image | Low-light camera sensor |

### How Much Augmentation?

Rule of thumb:
- Small dataset (<100 images): Heavy augmentation (10x multiplier)
- Medium dataset (100-500): Moderate augmentation (5x)
- Large dataset (500+): Light augmentation (2-3x)

---

## Common Problems & Solutions

### "Model detects nothing"
- Not enough training data
- Labels might be wrong (check a few manually)
- Confidence threshold too high (try 0.25)

### "Too many false positives"
- Background objects look like your class
- Need more negative examples
- Try higher confidence threshold

### "Good on training, bad on real images"
- Overfitting: training images too similar
- Solution: more augmentation, more diverse data

### "CUDA out of memory"
- Reduce batch size (try 8, 4, 2)
- Reduce image size (try 480, 320)

### "Training is extremely slow"
- CPU training is 10-50x slower than GPU
- Reduce epochs for testing
- Use smaller model (yolov8n)

---

## Hardware Requirements

| Setup | Training Time (100 epochs) | Recommended |
|-------|---------------------------|-------------|
| NVIDIA GPU (RTX 3060+) | 15-30 minutes | ✓ Best |
| Apple M1/M2 | 30-60 minutes | ✓ Good |
| CPU only | 2-8 hours | Works but slow |

---

## Next Steps After This Project

1. **Train on real data**: Replace synthetic shapes with actual objects
2. **Try larger models**: yolov8s, yolov8m for better accuracy
3. **Experiment with augmentation**: Find what works for your use case
4. **Deploy**: Export to ONNX for production use
5. **Edge deployment**: TensorRT for NVIDIA, CoreML for iOS

---

## License

Educational project for The Art of Programming course.
