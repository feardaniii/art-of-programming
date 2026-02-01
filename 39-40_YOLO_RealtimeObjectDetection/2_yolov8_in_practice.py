
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

# ==========================================
# SETUP: Installing and Loading YOLOv8
# ==========================================

"""
Installation:
    pip install ultralytics opencv-python pillow

YOLOv8 comes in 5 sizes:
    - YOLOv8n (nano):     3.2M params, 80 FPS (best for mobile/edge)
    - YOLOv8s (small):    11.2M params, 60 FPS
    - YOLOv8m (medium):   25.9M params, 45 FPS
    - YOLOv8l (large):    43.7M params, 30 FPS
    - YOLOv8x (xlarge):   68.2M params, 25 FPS (best accuracy)

For real-time applications: Use nano or small
For accuracy-critical applications: Use medium or large
"""

# Manually Load pre-trained YOLOv8 model (COCO dataset - 80 classes)
# model = YOLO('yolov8n.pt')  # Nano model for speed

"""
================================================================================
YOLOv8: REAL-TIME OBJECT DETECTION & TRACKING
================================================================================

Course: The Art of Programming - Computer Vision Module
Lesson: Understanding What YOLO Actually Does (And What It Doesn't)

WHAT YOLO DOES:
    ✓ Detects WHAT objects are in a frame (from 80 COCO classes)
    ✓ Locates WHERE objects are (bounding boxes)
    ✓ Tracks objects across frames (assigns persistent IDs)
    ✓ Runs in real-time (~30-80 FPS depending on model size)

WHAT YOLO DOES NOT DO:
    ✗ Pose estimation (use MediaPipe/OpenPose for that)
    ✗ Document/receipt scanning (use edge detection + OCR)
    ✗ Form analysis (that requires skeleton tracking)
    ✗ Detect objects it wasn't trained on (needs custom training)

COCO DATASET CLASSES (80 total):
    People:     person
    Vehicles:   bicycle, car, motorcycle, airplane, bus, train, truck, boat
    Animals:    bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    Sports:     frisbee, skis, snowboard, sports ball, kite, baseball bat/glove,
                skateboard, surfboard, tennis racket
    Kitchen:    bottle, wine glass, cup, fork, knife, spoon, bowl
    Food:       banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
                donut, cake
    Furniture:  chair, couch, potted plant, bed, dining table, toilet
    Electronics: tv, laptop, mouse, remote, keyboard, cell phone
    Other:      book, clock, scissors, teddy bear, hair drier, toothbrush, etc.

Installation:
    pip install ultralytics opencv-python numpy

================================================================================
"""





# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for easy experimentation."""

    # Model settings
    model_size: str = "m"  # n=nano, s=small, m=medium, l=large, x=xlarge
    confidence_threshold: float = 0.5

    # Tracking settings
    track_history_length: int = 30  # How many positions to remember per object

    # Zone detection (for counting objects entering/leaving areas)
    # Format: (x1, y1, x2, y2) as fractions of frame size
    counting_zone: Tuple[float, float, float, float] = (0.3, 0.3, 0.7, 0.7)

    # Display settings
    show_trajectories: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_fps: bool = True
    show_zone: bool = True

    # Colors (BGR format)
    trajectory_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    zone_color: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White

    @property
    def model_path(self) -> str:
        return f"yolov8{self.model_size}.pt"


# =============================================================================
# DETECTION TRACKER
# =============================================================================

@dataclass
class TrackedObject:
    """Represents a single tracked object across frames."""

    track_id: int
    class_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    in_zone: bool = False
    zone_entries: int = 0

    def update(self, center: Tuple[int, int], timestamp: float):
        """Update object position and timing."""
        self.positions.append(center)
        self.last_seen = timestamp

    @property
    def duration_visible(self) -> float:
        """How long this object has been tracked."""
        return self.last_seen - self.first_seen

    @property
    def trajectory(self) -> List[Tuple[int, int]]:
        """Get position history as a list."""
        return list(self.positions)


class ObjectTracker:
    """
    Manages tracking of multiple objects across frames.

    This demonstrates a key YOLO capability: persistent object tracking.
    Each detected object gets a unique ID that persists across frames,
    allowing you to track movement, count entries/exits, measure dwell time, etc.
    """

    def __init__(self, config: Config):
        self.config = config
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.class_names = None  # Will be set from model

        # Statistics
        self.total_unique_objects = 0
        self.zone_entry_count = 0
        self.class_counts: Dict[str, int] = defaultdict(int)

    def update(self, results, frame_shape: Tuple[int, int], timestamp: float):
        """
        Update tracker with new detection results.

        Args:
            results: YOLO detection results with tracking
            frame_shape: (height, width) of the frame
            timestamp: Current time for duration tracking
        """
        if self.class_names is None:
            self.class_names = results.names

        frame_h, frame_w = frame_shape

        # Get zone boundaries in pixels
        z = self.config.counting_zone
        zone_x1 = int(z[0] * frame_w)
        zone_y1 = int(z[1] * frame_h)
        zone_x2 = int(z[2] * frame_w)
        zone_y2 = int(z[3] * frame_h)

        # Track which IDs we see this frame
        current_ids = set()

        # Process each detection
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                current_ids.add(track_id)

                # Calculate center point
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)

                # Check if in zone
                in_zone = (zone_x1 <= center_x <= zone_x2 and
                           zone_y1 <= center_y <= zone_y2)

                # Update or create tracked object
                if track_id in self.tracked_objects:
                    obj = self.tracked_objects[track_id]

                    # Check for zone entry (wasn't in zone, now is)
                    if not obj.in_zone and in_zone:
                        obj.zone_entries += 1
                        self.zone_entry_count += 1

                    obj.in_zone = in_zone
                    obj.update(center, timestamp)
                else:
                    # New object
                    class_name = self.class_names[class_id]
                    obj = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        in_zone=in_zone
                    )
                    obj.positions.append(center)

                    if in_zone:
                        obj.zone_entries = 1
                        self.zone_entry_count += 1

                    self.tracked_objects[track_id] = obj
                    self.total_unique_objects += 1
                    self.class_counts[class_name] += 1

        return current_ids

    def get_statistics(self) -> Dict:
        """Get current tracking statistics."""
        active_objects = [obj for obj in self.tracked_objects.values()
                          if time.time() - obj.last_seen < 1.0]

        return {
            "total_unique": self.total_unique_objects,
            "currently_tracking": len(active_objects),
            "zone_entries": self.zone_entry_count,
            "class_breakdown": dict(self.class_counts),
            "objects_in_zone": sum(1 for obj in active_objects if obj.in_zone)
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """
    Handles all drawing and visualization.

    Separating visualization from detection logic is good practice:
    - Easier to customize appearance
    - Can swap different visualizers
    - Detection code stays clean
    """

    def __init__(self, config: Config):
        self.config = config
        self.fps_history = deque(maxlen=30)

    def draw_trajectories(self, frame: np.ndarray, tracker: ObjectTracker,
                          current_ids: set) -> np.ndarray:
        """Draw movement trails for tracked objects."""
        if not self.config.show_trajectories:
            return frame

        for track_id in current_ids:
            if track_id in tracker.tracked_objects:
                obj = tracker.tracked_objects[track_id]
                trajectory = obj.trajectory

                if len(trajectory) > 1:
                    # Draw trail with fading effect
                    for i in range(1, len(trajectory)):
                        alpha = i / len(trajectory)  # Fade from old to new
                        thickness = int(1 + alpha * 2)

                        pt1 = trajectory[i - 1]
                        pt2 = trajectory[i]

                        # Color based on whether object is in zone
                        if obj.in_zone:
                            color = self.config.zone_color
                        else:
                            color = self.config.trajectory_color

                        cv2.line(frame, pt1, pt2, color, thickness)

        return frame

    def draw_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting/detection zone."""
        if not self.config.show_zone:
            return frame

        h, w = frame.shape[:2]
        z = self.config.counting_zone

        x1, y1 = int(z[0] * w), int(z[1] * h)
        x2, y2 = int(z[2] * w), int(z[3] * h)

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.config.zone_color, 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.config.zone_color, -1)

        # Blend with original (15% opacity for fill)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.zone_color, 2)

        # Label
        cv2.putText(frame, "DETECTION ZONE", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.zone_color, 1)

        return frame

    def draw_statistics(self, frame: np.ndarray, stats: Dict, fps: float) -> np.ndarray:
        """Draw statistics overlay."""
        h, w = frame.shape[:2]

        # Background panel
        panel_h = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Statistics text
        lines = [
            f"FPS: {fps:.1f}",
            f"Unique Objects: {stats['total_unique']}",
            f"Currently Tracking: {stats['currently_tracking']}",
            f"In Zone: {stats['objects_in_zone']}",
            f"Zone Entries: {stats['zone_entries']}",
        ]

        y = 30
        for line in lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.config.text_color, 1)
            y += 22

        return frame

    def draw_class_breakdown(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """Draw detected class breakdown on the right side."""
        h, w = frame.shape[:2]

        if not stats['class_breakdown']:
            return frame

        # Sort by count
        sorted_classes = sorted(stats['class_breakdown'].items(),
                                key=lambda x: x[1], reverse=True)[:8]

        # Background panel
        panel_w = 180
        panel_h = 30 + len(sorted_classes) * 22

        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10), (w - 10, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "DETECTED CLASSES", (w - panel_w, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Class counts
        y = 52
        for class_name, count in sorted_classes:
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (w - panel_w, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.config.text_color, 1)
            y += 22

        return frame

    def draw_controls_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard controls help at the bottom."""
        h, w = frame.shape[:2]

        help_text = "[Q] Quit  [T] Toggle Trails  [Z] Toggle Zone  [S] Screenshot  [R] Reset Stats"

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, help_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class YOLOv8Demo:
    """
    Main application demonstrating YOLOv8 capabilities.

    This class ties together:
    - YOLO model for detection
    - Object tracker for persistence
    - Visualizer for display

    Educational Goals:
    1. Understand what YOLO actually detects
    2. See real-time tracking in action
    3. Learn about zone-based analytics
    4. Observe confidence scores and their meaning
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        print("=" * 70)
        print("YOLOv8 REAL-TIME OBJECT DETECTION & TRACKING")
        print("=" * 70)
        print()

        # Load model
        print(f"Loading YOLOv8{self.config.model_size} model...")
        self.model = YOLO(self.config.model_path)
        print(f"✓ Model loaded: {self.config.model_path}")
        print(f"✓ Classes: 80 (COCO dataset)")
        print(f"✓ Confidence threshold: {self.config.confidence_threshold}")
        print()

        # Initialize components
        self.tracker = ObjectTracker(self.config)
        self.visualizer = Visualizer(self.config)

        # State
        self.running = False
        self.fps_times = deque(maxlen=30)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the detection pipeline.

        Pipeline:
        1. Run YOLO detection with tracking
        2. Update our tracker with results
        3. Draw visualizations
        4. Return annotated frame
        """
        timestamp = time.time()

        # Run YOLO with built-in tracking (ByteTrack algorithm)
        results = self.model.track(
            frame,
            persist=True,  # Keep track IDs across frames
            conf=self.config.confidence_threshold,
            verbose=False
        )[0]

        # Update our tracker
        current_ids = self.tracker.update(results, frame.shape[:2], timestamp)

        # Get annotated frame from YOLO (draws boxes and labels)
        annotated = results.plot()

        # Add our custom visualizations
        annotated = self.visualizer.draw_zone(annotated)
        annotated = self.visualizer.draw_trajectories(annotated, self.tracker, current_ids)

        # Calculate FPS
        self.fps_times.append(timestamp)
        if len(self.fps_times) > 1:
            fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0])
        else:
            fps = 0

        # Draw statistics
        stats = self.tracker.get_statistics()
        annotated = self.visualizer.draw_statistics(annotated, stats, fps)
        annotated = self.visualizer.draw_class_breakdown(annotated, stats)
        annotated = self.visualizer.draw_controls_help(annotated)

        return annotated

    def run_webcam(self):
        """
        Run live detection on webcam feed.

        Controls:
            Q - Quit
            T - Toggle trajectory trails
            Z - Toggle detection zone
            S - Save screenshot
            R - Reset statistics
        """
        print("Starting webcam...")
        print()
        print("Controls:")
        print("  [Q] Quit application")
        print("  [T] Toggle trajectory trails")
        print("  [Z] Toggle detection zone display")
        print("  [S] Save screenshot")
        print("  [R] Reset all statistics")
        print()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return

        # Set resolution (optional - adjust for your needs)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break

            # Process frame
            annotated = self.process_frame(frame)

            # Display
            cv2.imshow("YOLOv8 Detection & Tracking", annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
            elif key == ord('t'):
                self.config.show_trajectories = not self.config.show_trajectories
                print(f"Trajectories: {'ON' if self.config.show_trajectories else 'OFF'}")
            elif key == ord('z'):
                self.config.show_zone = not self.config.show_zone
                print(f"Zone display: {'ON' if self.config.show_zone else 'OFF'}")
            elif key == ord('s'):
                filename = f"yolo_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('r'):
                self.tracker = ObjectTracker(self.config)
                print("✓ Statistics reset")

        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        self.print_session_summary()

    def run_on_image(self, image_path: str):
        """
        Run detection on a single image.

        Useful for understanding YOLO output without real-time pressure.
        """
        print(f"Processing image: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"ERROR: Could not load image: {image_path}")
            return

        # Run detection (no tracking for single image)
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            verbose=False
        )[0]

        # Print detections
        print()
        print("Detections:")
        print("-" * 40)

        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                print(f"  {class_name}: {confidence:.1%} confidence")
                print(f"    Location: ({bbox[0]:.0f}, {bbox[1]:.0f}) to ({bbox[2]:.0f}, {bbox[3]:.0f})")
        else:
            print("  No objects detected")

        # Display
        annotated = results.plot()
        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_session_summary(self):
        """Print summary statistics after session ends."""
        stats = self.tracker.get_statistics()

        print()
        print("=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print()
        print(f"Total unique objects detected: {stats['total_unique']}")
        print(f"Total zone entries: {stats['zone_entries']}")
        print()

        if stats['class_breakdown']:
            print("Objects by class:")
            for class_name, count in sorted(stats['class_breakdown'].items(),
                                            key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")

        print()
        print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point - customize configuration here.

    Try experimenting with:
    - Different model sizes (n, s, m, l, x)
    - Different confidence thresholds
    - Different zone positions
    - Turning features on/off
    """

    # Create configuration
    config = Config(
        model_size="m",  # Start with nano for speed
        confidence_threshold=0.5,  # Adjust based on false positive rate
        show_trajectories=True,  # Show movement trails
        show_zone=True,  # Show detection zone
        counting_zone=(0.25, 0.25, 0.75, 0.75),  # Center zone
    )

    # Create and run demo
    demo = YOLOv8Demo(config)
    demo.run_webcam()


if __name__ == "__main__":
    main()