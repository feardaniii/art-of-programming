# Model Architecture: Autonomous Driving Object Detector

## Table of Contents
1. [The KITTI Dataset](#the-kitti-dataset)
2. [Our Approach: MobileNetV2 + Detection Heads](#our-approach)
3. [Why MobileNetV2?](#why-mobilenetv2)
4. [Alternative Architectures](#alternative-architectures)
5. [KITTI Leaderboard: What the Top Performers Use](#kitti-leaderboard)
6. [Single-Object vs Multi-Object Detection](#single-vs-multi)
7. [How to Extend This Project](#how-to-extend)

---

## The KITTI Dataset

**KITTI** (Karlsruhe Institute of Technology and Toyota Technological Institute) is one of the
most influential benchmarks in autonomous driving research. Recorded from a car driving through
Karlsruhe, Germany, it captures real urban traffic.

### What's in the box

| Split       | Images | Labels | Purpose |
|-------------|--------|--------|---------|
| Training    | 7,481  | 7,481  | We use this — split 80/20 for train/validation |
| Testing     | 7,518  | None   | For official benchmark submission (labels held by organizers) |

### Object classes in KITTI annotations

| Class          | Our mapping     | Notes |
|----------------|-----------------|-------|
| Car            | **Car**         | Standard automobiles |
| Van            | **Car**         | Similar enough to group with cars |
| Truck          | *(skipped)*     | Too different in size/shape from cars |
| Pedestrian     | **Pedestrian**  | People walking |
| Person_sitting | **Pedestrian**  | Seated people — still pedestrians |
| Cyclist        | **Cyclist**     | People on bicycles |
| Tram           | *(skipped)*     | Rare, not relevant for most scenarios |
| Misc           | *(skipped)*     | Catch-all for unusual objects |
| DontCare       | *(skipped)*     | Regions to ignore during evaluation |

### Why do we split training ourselves?

The 7,518 "testing" images have **no labels** — they exist solely for submitting results to
the official KITTI benchmark server. Since we can't evaluate locally without labels, we split
the 7,481 labeled training images into our own **80% train / 20% validation** sets.

### Label format

Each line in a `.txt` label file describes one object with 15 space-separated fields:

```
Type  Truncated  Occluded  Alpha  | BBox_L  BBox_T  BBox_R  BBox_B |  H   W   L  |  X    Y    Z  | RotY
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    2D bounding box (pixels) — what our model uses
```

The 3D information (height/width/length, XYZ location, rotation) is available for 3D detection
tasks, but we only use the 2D bounding box for our camera-only 2D detector.

---

## Our Approach

**Architecture: MobileNetV2 (backbone) + Classification Head + Bounding Box Head**

```
Input Image (224x224x3)
        |
        v
  ┌─────────────────────┐
  │   MobileNetV2        │  Pre-trained on ImageNet (1.4M images, 1000 classes)
  │   (Feature Extractor)│  Produces a 7x7x1280 feature map
  └──────────┬──────────┘
             |
             v
  ┌─────────────────────┐
  │ GlobalAveragePooling │  7x7x1280 → 1280-dim vector
  └──────────┬──────────┘
             |
             v
  ┌─────────────────────┐
  │ Dense(256) + Dropout │  Shared feature processing
  │ Dense(128) + Dropout │
  └─────┬─────────┬─────┘
        |         |
        v         v
  ┌───────────┐ ┌───────────────┐
  │ Dense(3)  │ │ Dense(4)      │
  │ Softmax   │ │ Sigmoid       │
  │           │ │               │
  │ "What?"   │ │ "Where?"      │
  │ Car/Ped/  │ │ x1, y1, x2,  │
  │ Cyclist   │ │ y2 normalized │
  └───────────┘ └───────────────┘
  Classification   Bounding Box
      Head         Regression Head
```

### Training strategy: Two-stage transfer learning

**Stage 1 — Frozen backbone** (15 epochs, lr=0.001)
- MobileNetV2 weights are frozen (no updates)
- Only the detection heads learn
- The backbone acts as a fixed feature extractor

**Stage 2 — Fine-tuning** (10 epochs, lr=0.00001)
- Unfreeze the last 30 layers of MobileNetV2
- Use 100x lower learning rate to avoid destroying pre-trained features
- The backbone adapts its features to driving scenes

### Multi-task loss

```
Total Loss = 1.0 * CrossEntropy(classification) + 5.0 * MSE(bounding box)
```

The bbox loss is weighted 5x higher because accurate localization is critical for driving.

---

## Why MobileNetV2?

| Factor | MobileNetV2 | ResNet50 | EfficientNet-B0 |
|--------|-------------|----------|-----------------|
| Parameters | 3.4M | 25.6M | 5.3M |
| Speed (CPU) | Fast | Slow | Medium |
| ImageNet Top-1 | 71.8% | 76.1% | 77.1% |
| Good for teaching? | Excellent | Too heavy | Good alternative |
| Mobile/edge ready? | Yes (designed for it) | No | Somewhat |

**MobileNetV2** uses **depthwise separable convolutions** — a factorization trick that reduces
computation by 8-9x compared to standard convolutions. It was designed by Google for mobile and
embedded devices, making it perfect for:
- Quick training on CPU/GPU laptops
- Understanding transfer learning concepts
- Potential deployment on edge devices (Raspberry Pi, Jetson Nano)

---

## Alternative Architectures

Here's the landscape of object detection models, from oldest to newest:

### One-Stage Detectors (faster, simpler)

#### YOLO (You Only Look Once)
- **Core idea**: Divide image into grid cells, each cell predicts boxes + classes simultaneously
- **Versions**: YOLOv1 (2016) → YOLOv5 → YOLOv8 → YOLO11 → YOLOv12 (2025)
- **Speed**: Extremely fast (100+ FPS on GPU)
- **KITTI performance**: Good. YOLOv8 can be fine-tuned on KITTI directly
- **When to use**: Real-time applications, embedded systems
- **Library**: `ultralytics` (pip install ultralytics)

#### SSD (Single Shot MultiBox Detector)
- **Core idea**: Predict at multiple feature map scales (handles objects of different sizes)
- **Speed**: Fast (comparable to YOLO)
- **Key innovation**: Multi-scale feature maps for small/medium/large objects
- **When to use**: Good balance of speed and accuracy

#### RetinaNet
- **Core idea**: Uses **Focal Loss** to handle class imbalance (lots of background, few objects)
- **Backbone**: Feature Pyramid Network (FPN) + ResNet
- **Key insight**: Most detection errors come from easy negatives overwhelming the loss
- **When to use**: When class imbalance is a major issue

#### EfficientDet
- **Core idea**: Compound scaling of resolution, backbone, and feature network
- **Backbone**: EfficientNet + BiFPN (Bidirectional Feature Pyramid Network)
- **Variants**: D0 (smallest/fastest) to D7 (largest/most accurate)
- **When to use**: When you want to tune the speed/accuracy trade-off precisely

### Two-Stage Detectors (slower, more accurate)

#### Faster R-CNN
- **Core idea**: Stage 1 proposes regions, Stage 2 classifies and refines them
- **Architecture**: Backbone → Region Proposal Network (RPN) → RoI Pooling → Heads
- **Speed**: Slower (5-15 FPS) but more accurate
- **When to use**: When accuracy matters more than speed (offline analysis)

#### Cascade R-CNN
- **Core idea**: Chain of detection stages with increasing IoU thresholds
- **Improvement**: Better at precise localization (high IoU detections)
- **When to use**: When you need very accurate bounding boxes

### Transformer-Based (newest, powerful)

#### DETR (DEtection TRansformer)
- **Core idea**: Treat detection as a set prediction problem using attention
- **Architecture**: CNN backbone → Transformer encoder/decoder → prediction heads
- **Key innovation**: No anchors, no NMS — end-to-end training
- **When to use**: Research, when simplicity of pipeline matters

#### DINO / Co-DETR
- **Improvements over DETR**: Faster convergence, better small object detection
- **State-of-the-art**: Among the best on COCO benchmark
- **When to use**: When you want top accuracy and have GPU compute

### Summary Comparison

| Model | Type | Speed | Accuracy | Complexity | Best For |
|-------|------|-------|----------|------------|----------|
| **Our model** | Classifier+Loc | Very Fast | Limited* | Low | Learning |
| YOLO (v8/v11) | One-stage | Very Fast | High | Medium | Real-time |
| SSD | One-stage | Fast | Good | Medium | Embedded |
| RetinaNet | One-stage | Medium | High | Medium | Imbalanced data |
| EfficientDet | One-stage | Tunable | High | Medium | Tunable trade-off |
| Faster R-CNN | Two-stage | Slow | Very High | High | Offline analysis |
| DETR | Transformer | Medium | High | High | Research |

*\*Limited because we detect only the single largest object per image (see next section)*

---

## KITTI Leaderboard

The [KITTI 2D Object Detection Leaderboard](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
shows the best-performing models on the official test set.

### Top performers (as of 2025-2026)

| Rank | Model | Car AP (Moderate) | Key Approach |
|------|-------|-------------------|--------------|
| 1 | ViKIENet | 98.06% | Virtual key instance enhanced + LiDAR fusion |
| 2 | ICD-PSOC | 97.83% | Multi-sensor fusion |
| 5 | UDeerPEP | 97.57% | Point-enhanced painting (LiDAR + camera) |
| 7 | VirConv-S | 97.27% | Virtual sparse convolution (CVPR 2023) |
| 11 | GraR-VoI | 96.38% | Graph R-CNN with semantic decoration (ECCV 2022) |

### Key insight: Most top performers use LiDAR

The KITTI benchmark provides both camera images AND LiDAR point clouds. Almost every top-ranked
method fuses LiDAR depth data with camera images. This is fundamentally different from our
camera-only approach.

**Why LiDAR helps so much:**
- Provides exact 3D distance to every point (camera must estimate depth)
- Works in all lighting conditions (camera struggles at night/glare)
- Eliminates ambiguity about object size vs distance

**Camera-only methods** typically achieve lower scores on the KITTI leaderboard, but they are
far more practical for consumer vehicles (LiDAR sensors cost $1,000-$75,000+ vs a $10 camera).
Tesla, for example, famously uses a camera-only approach.

---

## Single-Object vs Multi-Object Detection

### What our model does (Single-Object Localization)

Our model answers: **"What is the main object, and where is it?"**

- Input: full image
- Output: ONE class label + ONE bounding box
- Training: We pick the largest annotated object per image as the target
- Limitation: Cannot detect multiple objects in the same image

This is a **classification + localization** task — a stepping stone toward full detection.

### What YOLO/SSD/Faster R-CNN do (Multi-Object Detection)

These models answer: **"What are ALL the objects, and where is EACH one?"**

- Input: full image
- Output: variable number of (class, bbox, confidence) tuples
- Key mechanisms they use:
  - **Anchor boxes**: Pre-defined box shapes at each grid location
  - **Non-Maximum Suppression (NMS)**: Merge overlapping predictions
  - **Feature Pyramid Networks**: Detect objects at multiple scales

### The gap between our model and real detectors

```
  Our Model                      Full Detector (e.g., YOLO)
  ─────────                      ──────────────────────────
  1 object per image             N objects per image
  Global features only           Multi-scale features
  No anchors                     100s-1000s of anchors
  Simple MSE loss                IoU-based loss (GIoU, CIoU)
  ~3.5M parameters               ~7-60M parameters
  Fast to train                  Needs more data/compute
```

Our model is **the first chapter** of the detection story. It teaches:
- Transfer learning (MobileNetV2 pre-trained features)
- Multi-task learning (classification + regression simultaneously)
- Bounding box regression (predicting coordinates)
- Evaluation metrics (IoU, accuracy)

These concepts are the building blocks used by every full detection architecture.

---

## How to Extend This Project

### Level 1: Quick wins with our current architecture
- Add more augmentation (random rotation, color jitter, random crop)
- Try EfficientNet-B0 as backbone instead of MobileNetV2
- Increase input resolution to 320x320 or 416x416

### Level 2: Move to proper multi-object detection
- **Easiest path**: Fine-tune YOLOv8 on KITTI using Ultralytics
  ```bash
  pip install ultralytics
  # Convert KITTI labels to YOLO format, then:
  yolo train model=yolov8n.pt data=kitti.yaml epochs=50
  ```
- **From scratch path**: Implement SSD with MobileNetV2 backbone
  - Replace GlobalAveragePooling with multi-scale feature maps
  - Add anchor boxes at each feature map location
  - Implement NMS for inference

### Level 3: Advanced techniques
- Implement Feature Pyramid Network (FPN) for multi-scale detection
- Add 3D bounding box prediction using KITTI's 3D annotations
- Train on KITTI's LiDAR point clouds for 3D detection (PointPillars, VoxelNet)
- Deploy with TensorFlow Lite for mobile/edge inference

### Level 4: Production-grade
- Train on larger datasets (nuScenes, Waymo Open Dataset, BDD100K)
- Multi-sensor fusion (camera + LiDAR + radar)
- Temporal tracking (follow objects across frames)
- Uncertainty estimation (know when the model is unsure)

---

## References

- **KITTI Benchmark**: https://www.cvlibs.net/datasets/kitti/eval_object.php
- **MobileNetV2 Paper**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
- **YOLO Origin**: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- **SSD Paper**: Liu et al., "SSD: Single Shot MultiBox Detector" (2016)
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection" (2015)
- **DETR**: Carion et al., "End-to-End Object Detection with Transformers" (2020)
- **EfficientDet**: Tan et al., "EfficientDet: Scalable and Efficient Object Detection" (2020)
- **RetinaNet / Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
