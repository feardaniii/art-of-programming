import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

# ==========================================
# THE PROBLEM: Why Previous Object Detection Was Too Slow
# ==========================================

def demonstrate_detection_speed_crisis():
    """
    The Autonomous Driving Crisis (2015):

    R-CNN (2014):
      - Generate ~2,000 region proposals per image
      - Run CNN on EACH region separately
      - 47 seconds per image
      - For 30 FPS video: Need 1,410 seconds to process 1 second of video

    Fast R-CNN (2015):
      - Share CNN features across regions
      - 2.3 seconds per image
      - Still 13x slower than real-time

    The Need: Autonomous vehicles require:
      - 30 FPS minimum (0.033 seconds per frame)
      - Detect ALL objects simultaneously
      - Pedestrians, vehicles, signs, lanes
      - Miss one frame = potential accident

    YOLO (2016):
      - Single forward pass through network
      - 0.022 seconds per image
      - 45 FPS � Real-time possible!
    """

    print("=" * 80)
    print("THE REAL-TIME OBJECT DETECTION CRISIS")
    print("=" * 80)
    print()

    methods = {
        "R-CNN (2014)": {"time": 47.0, "fps": 0.021},
        "Fast R-CNN (2015)": {"time": 2.3, "fps": 0.43},
        "Faster R-CNN (2015)": {"time": 0.2, "fps": 5.0},
        "YOLO v1 (2016)": {"time": 0.022, "fps": 45.0},
        "YOLO v3 (2018)": {"time": 0.033, "fps": 30.0},
        "YOLO v5 (2020)": {"time": 0.007, "fps": 140.0},
        "YOLO v8 (2023)": {"time": 0.005, "fps": 200.0}
    }

    print(f"{'Method':<25} {'Time/Image':<15} {'FPS':<10} {'Real-time?':<15}")
    print("-" * 80)

    for method, stats in methods.items():
        realtime = " YES" if stats['fps'] >= 30 else "L NO"
        print(f"{method:<25} {stats['time']:<15.3f}s {stats['fps']:<10.1f} {realtime:<15}")

    print()
    print("=� THE BREAKTHROUGH:")
    print("   Instead of looking at 2,000+ regions separately...")
    print("   YOLO looks at the ENTIRE image ONCE")
    print("   Single neural network � All detections simultaneously")
    print()
    print("<� WHY IT MATTERS:")
    print(" Autonomous vehicles: See pedestrians, cars, signs - all at once")
    print(" Health monitoring: Track workout form in real-time")
    print(" Security: Monitor multiple threats simultaneously")
    print(" Finance: Scan receipts and documents instantly")
    print()

demonstrate_detection_speed_crisis()


# ==========================================
# YOLO ARCHITECTURE: The Grid-Based Detection System
# ==========================================

def visualize_yolo_grid_concept():
    """
    YOLO's Core Insight: Divide & Conquer

    1. Divide image into S � S grid (e.g., 7�7)
    2. Each grid cell predicts:
       - B bounding boxes (e.g., 2 boxes)
       - Confidence for each box
       - C class probabilities (e.g., 80 classes)

    3. Output tensor shape: S � S � (B * 5 + C)
       - For 7�7 grid, 2 boxes, 80 classes: 7 � 7 � 90

    This single tensor contains ALL detections!
    """

    print("=" * 80)
    print("YOLO GRID ARCHITECTURE")
    print("=" * 80)
    print()

    # Example: 7x7 grid, 2 bounding boxes per cell, 80 classes
    S = 7  # Grid size
    B = 2  # Bounding boxes per cell
    C = 80  # Number of classes (COCO dataset)

    print(f"Grid Configuration:")
    print(f" Image divided into {S}�{S} = {S*S} cells")
    print(f" Each cell predicts {B} bounding boxes")
    print(f" Each box predicts {C} class probabilities")
    print()

    # Each bounding box predicts: [x, y, w, h, confidence]
    box_params = 5
    cell_predictions = B * box_params + C

    print(f"Predictions per Cell:")
    print(f" Bounding boxes: {B} boxes � {box_params} params = {B * box_params}")
    print(f"    - (x, y): center coordinates")
    print(f"    - (w, h): width and height")
    print(f"    - confidence: P(object) � IOU")
    print(f" Class probabilities: {C} classes")
    print(f" Total per cell: {cell_predictions} values")
    print()

    total_predictions = S * S * cell_predictions
    print(f"Final Output Tensor:")
    print(f" Shape: {S} � {S} � {cell_predictions} = {total_predictions:,} predictions")
    print(f" This single tensor contains ALL objects in the image!")
    print()

    # Visualize the grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create sample image
    img = np.random.rand(224, 224, 3)
    axes[0].imshow(img)
    axes[0].set_title('Original Image\n(224�224 pixels)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Draw grid overlay
    cell_size = 224 // S
    for i in range(S + 1):
        axes[0].axhline(y=i * cell_size, color='cyan', linewidth=2, alpha=0.7)
        axes[0].axvline(x=i * cell_size, color='cyan', linewidth=2, alpha=0.7)

    # Visualize predictions
    axes[1].imshow(img)
    axes[1].set_title(f'YOLO Grid ({S}�{S})\nEach cell predicts {B} boxes + {C} classes',
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Draw sample bounding boxes in a few cells
    import matplotlib.patches as patches
    colors = ['red', 'yellow']

    for cell_i in [2, 3, 5]:
        for cell_j in [2, 4, 5]:
            for b in range(B):
                # Random box within cell
                cx = (cell_j + 0.5) * cell_size + np.random.randn() * 5
                cy = (cell_i + 0.5) * cell_size + np.random.randn() * 5
                w = cell_size * 0.8
                h = cell_size * 0.8

                rect = patches.Rectangle(
                    (cx - w/2, cy - h/2), w, h,
                    linewidth=2, edgecolor=colors[b], facecolor='none', alpha=0.7
                )
                axes[1].add_patch(rect)

    plt.tight_layout()
    plt.savefig('yolo_grid_architecture.png', dpi=300, bbox_inches='tight')
    print(" YOLO grid visualization saved: yolo_grid_architecture.png")
    print()

visualize_yolo_grid_concept()


# ==========================================
# YOLO PREDICTION: From Grid to Bounding Boxes
# ==========================================

class SimpleYOLOPredictor:
    """
    Simplified YOLO prediction logic

    This demonstrates how YOLO converts grid predictions to bounding boxes.
    Real YOLOv5/v8 use anchor boxes and more sophisticated non-max suppression.
    """

    def __init__(self, grid_size=7, num_boxes=2, num_classes=80, conf_threshold=0.5):
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.conf_threshold = conf_threshold

    def parse_predictions(self, predictions):
        """
        Convert YOLO output tensor to bounding boxes

        Input: predictions tensor [S, S, B*5 + C]
        Output: List of detected objects with [x, y, w, h, confidence, class]
        """
        detections = []

        for i in range(self.S):
            for j in range(self.S):
                cell_predictions = predictions[i, j]

                # Each cell predicts B bounding boxes
                for b in range(self.B):
                    # Extract box predictions: [x, y, w, h, confidence]
                    box_idx = b * 5
                    x, y, w, h, conf = cell_predictions[box_idx:box_idx+5]

                    # Convert from grid coordinates to image coordinates
                    # x, y are relative to cell, need to add cell offset
                    abs_x = (j + x) / self.S  # Normalize to [0, 1]
                    abs_y = (i + y) / self.S
                    abs_w = w
                    abs_h = h

                    # Only keep detections above confidence threshold
                    if conf > self.conf_threshold:
                        # Get class probabilities
                        class_probs = cell_predictions[self.B * 5:]
                        class_id = np.argmax(class_probs)
                        class_prob = class_probs[class_id]

                        # Final confidence = box confidence � class probability
                        final_conf = conf * class_prob

                        detections.append({
                            'bbox': [abs_x, abs_y, abs_w, abs_h],
                            'confidence': final_conf,
                            'class_id': class_id
                        })

        return detections

    def non_max_suppression(self, detections, iou_threshold=0.5):
        """
        Remove duplicate detections using Non-Maximum Suppression

        Multiple grid cells might detect the same object.
        NMS keeps only the highest confidence detection.
        """
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while len(detections) > 0:
            # Keep highest confidence detection
            best = detections[0]
            keep.append(best)
            detections = detections[1:]

            # Remove overlapping boxes
            detections = [
                d for d in detections
                if self.calculate_iou(best['bbox'], d['bbox']) < iou_threshold
            ]

        return keep

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU)

        IoU = Area of Overlap / Area of Union
        Measures how much two boxes overlap (0 = no overlap, 1 = perfect overlap)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert center coordinates to corner coordinates
        box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
        box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2

        box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
        box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2

        # Calculate intersection area
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# Test YOLO prediction pipeline
print()
print("=" * 80)
print("YOLO PREDICTION PIPELINE")
print("=" * 80)
print()

# Create random predictions (simulating network output)
predictor = SimpleYOLOPredictor(grid_size=7, num_boxes=2, num_classes=80)
predictions = np.random.rand(7, 7, 2 * 5 + 80)

# Parse predictions
detections = predictor.parse_predictions(predictions)
print(f"Raw detections from grid: {len(detections)}")

# Apply Non-Maximum Suppression
final_detections = predictor.non_max_suppression(detections, iou_threshold=0.5)
print(f"After NMS: {len(final_detections)} unique objects detected")
print()

print("<� WHAT WE LEARNED:")
print("   1. YOLO divides image into grid � Each cell predicts objects")
print("   2. Single forward pass � All predictions simultaneously")
print("   3. Non-Max Suppression � Remove duplicate detections")
print("   4. Result: Fast, accurate, real-time object detection")
print()

print("=" * 80)
print("PART 1 COMPLETE: You understand YOLO's architecture.")
print("Now: Real applications with YOLOv8 and Ultralytics.")
print("=" * 80)
