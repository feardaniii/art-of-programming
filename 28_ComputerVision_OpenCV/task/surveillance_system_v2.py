#!/usr/bin/env python3
"""
🚨 INTELLIGENT SURVEILLANCE & FIRE SAFETY SYSTEM V2 🚨

Smart detection algorithms with visual analysis panel

Features:
- Multi-frame motion confirmation (reduces false alarms)
- Rising smoke detection with centroid tracking
- Visual analysis panel showing what's being detected
- Confidence meters and detection history
- Cooldown periods to prevent alarm spam
- Minimal console output (only real threats)

Controls:
- 'q': Quit
- 's': Save snapshot
- 'r': Reset alarms
- 'm': Toggle motion detection
- 'f': Toggle fire/smoke detection
"""

import cv2
import numpy as np
from datetime import datetime
import os
from collections import deque

# ==================== CONFIGURATION ====================

# Motion Detection Settings
MOTION_MIN_AREA = 1500  # Minimum contour area to consider
MOTION_CONFIRMATION_FRAMES = 3  # Must detect motion for N consecutive frames
MOTION_COOLDOWN_SECONDS = 5  # Don't re-alarm for N seconds
MOTION_THRESHOLD_SENSITIVITY = 30  # Threshold for frame difference

# Smoke Detection Settings
SMOKE_MIN_AREA = 3000  # Minimum smoke region area
SMOKE_CONFIRMATION_FRAMES = 5  # Must detect smoke for N consecutive frames
SMOKE_RISING_THRESHOLD = -10  # Centroid must move up (negative Y) by this many pixels
SMOKE_COOLDOWN_SECONDS = 8  # Don't re-alarm for N seconds
SMOKE_GROWTH_MIN = 1.2  # Minimum growth factor (20% increase)

# HSV Range for Smoke (gray/white, low saturation)
LOWER_SMOKE = np.array([0, 0, 190])
UPPER_SMOKE = np.array([180, 50, 255])

# Visual Settings
SNAPSHOT_DIR = 'surveillance_snapshots'
HISTORY_SIZE = 30  # Keep last N values for graphing

# ==================== HELPER CLASSES ====================

class DetectionState:
    """Track detection state with confirmation and cooldown"""
    def __init__(self, confirmation_frames, cooldown_seconds):
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_detections = 0
        self.alarm_active = False
        self.alarm_count = 0
        self.last_alarm_time = None
        self.confidence = 0.0

    def update(self, detected):
        """Update state based on detection"""
        if detected:
            self.consecutive_detections += 1
            self.confidence = min(100, (self.consecutive_detections / self.confirmation_frames) * 100)
        else:
            self.consecutive_detections = 0
            self.confidence = 0

        # Check if we should trigger alarm
        should_alarm = False
        if self.consecutive_detections >= self.confirmation_frames:
            # Check cooldown
            if self.last_alarm_time is None:
                should_alarm = True
            else:
                elapsed = (datetime.now() - self.last_alarm_time).total_seconds()
                if elapsed > self.cooldown_seconds:
                    should_alarm = True

        if should_alarm and not self.alarm_active:
            self.alarm_active = True
            self.alarm_count += 1
            self.last_alarm_time = datetime.now()
            return True
        elif not detected:
            self.alarm_active = False

        return False

    def reset(self):
        """Reset state"""
        self.consecutive_detections = 0
        self.alarm_active = False
        self.confidence = 0

class SmokeTracker:
    """Track smoke properties over time"""
    def __init__(self, history_size=10):
        self.centroids = deque(maxlen=history_size)
        self.areas = deque(maxlen=history_size)

    def update(self, centroid, area):
        """Add new detection"""
        self.centroids.append(centroid)
        self.areas.append(area)

    def is_rising(self, threshold=-10):
        """Check if smoke is rising (moving up = negative Y)"""
        if len(self.centroids) < 3:
            return False
        recent = self.centroids[-1][1]
        old = self.centroids[0][1]
        return (recent - old) < threshold  # Y decreases = moving up

    def is_growing(self, min_growth=1.2):
        """Check if smoke area is growing"""
        if len(self.areas) < 3:
            return False
        recent_avg = np.mean(list(self.areas)[-3:])
        old_avg = np.mean(list(self.areas)[:3])
        if old_avg < 100:  # Too small to measure growth reliably
            return False
        return recent_avg > old_avg * min_growth

    def reset(self):
        """Clear history"""
        self.centroids.clear()
        self.areas.clear()

class AnalysisPanel:
    """Create visual analysis panel"""
    def __init__(self, width=400, height=600):
        self.width = width
        self.height = height
        self.motion_history = deque(maxlen=HISTORY_SIZE)
        self.smoke_history = deque(maxlen=HISTORY_SIZE)

    def create_panel(self, frame, motion_mask, smoke_mask, motion_state, smoke_state, smoke_tracker):
        """Create comprehensive analysis panel"""
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Title
        cv2.putText(panel, "ANALYSIS PANEL", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        y_offset = 60

        # ===== MOTION DETECTION SECTION =====
        cv2.rectangle(panel, (5, y_offset), (self.width-5, y_offset+180), (50, 50, 50), 1)
        cv2.putText(panel, "MOTION DETECTION", (15, y_offset+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Motion mask thumbnail
        motion_thumb = cv2.resize(motion_mask, (120, 90))
        motion_thumb_rgb = cv2.cvtColor(motion_thumb, cv2.COLOR_GRAY2BGR)
        panel[y_offset+30:y_offset+120, 15:135] = motion_thumb_rgb

        # Motion status
        status_x = 150
        cv2.putText(panel, f"Confidence: {motion_state.confidence:.0f}%",
                   (status_x, y_offset+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Frames: {motion_state.consecutive_detections}/{motion_state.confirmation_frames}",
                   (status_x, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Alarms: {motion_state.alarm_count}",
                   (status_x, y_offset+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Confidence bar
        bar_y = y_offset + 130
        cv2.rectangle(panel, (15, bar_y), (self.width-15, bar_y+20), (100, 100, 100), 1)
        bar_width = int((self.width - 30) * (motion_state.confidence / 100))
        color = (0, 255, 0) if motion_state.alarm_active else (100, 100, 100)
        cv2.rectangle(panel, (15, bar_y), (15 + bar_width, bar_y+20), color, -1)

        # Status text
        status_text = "ALARM!" if motion_state.alarm_active else "Monitoring"
        cv2.putText(panel, status_text, (20, bar_y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += 200

        # ===== SMOKE DETECTION SECTION =====
        cv2.rectangle(panel, (5, y_offset), (self.width-5, y_offset+220), (50, 50, 50), 1)
        cv2.putText(panel, "SMOKE DETECTION", (15, y_offset+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

        # Smoke mask thumbnail
        smoke_thumb = cv2.resize(smoke_mask, (120, 90))
        smoke_thumb_rgb = cv2.cvtColor(smoke_thumb, cv2.COLOR_GRAY2BGR)
        panel[y_offset+30:y_offset+120, 15:135] = smoke_thumb_rgb

        # Smoke status
        cv2.putText(panel, f"Confidence: {smoke_state.confidence:.0f}%",
                   (status_x, y_offset+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Frames: {smoke_state.consecutive_detections}/{smoke_state.confirmation_frames}",
                   (status_x, y_offset+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Alarms: {smoke_state.alarm_count}",
                   (status_x, y_offset+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Smoke properties
        rising = smoke_tracker.is_rising()
        growing = smoke_tracker.is_growing()

        rising_text = "RISING" if rising else "Static"
        rising_color = (0, 255, 255) if rising else (100, 100, 100)
        cv2.putText(panel, f"Motion: {rising_text}",
                   (15, y_offset+140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rising_color, 1)

        growing_text = "GROWING" if growing else "Stable"
        growing_color = (0, 255, 255) if growing else (100, 100, 100)
        cv2.putText(panel, f"Area: {growing_text}",
                   (15, y_offset+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, growing_color, 1)

        # Confidence bar
        bar_y = y_offset + 180
        cv2.rectangle(panel, (15, bar_y), (self.width-15, bar_y+20), (100, 100, 100), 1)
        bar_width = int((self.width - 30) * (smoke_state.confidence / 100))
        color = (0, 100, 255) if smoke_state.alarm_active else (100, 100, 100)
        cv2.rectangle(panel, (15, bar_y), (15 + bar_width, bar_y+20), color, -1)

        # Status text
        status_text = "FIRE ALARM!" if smoke_state.alarm_active else "Monitoring"
        cv2.putText(panel, status_text, (20, bar_y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += 240

        # ===== DETECTION LOGIC EXPLANATION =====
        cv2.rectangle(panel, (5, y_offset), (self.width-5, y_offset+100), (50, 50, 50), 1)
        cv2.putText(panel, "DETECTION LOGIC", (15, y_offset+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Motion explanation
        cv2.putText(panel, "Motion: Frame difference >", (15, y_offset+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(panel, f"{MOTION_CONFIRMATION_FRAMES} frames + area > {MOTION_MIN_AREA}px",
                   (15, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Smoke explanation
        cv2.putText(panel, "Smoke: Gray/white + rising +", (15, y_offset+80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(panel, f"growing over {SMOKE_CONFIRMATION_FRAMES} frames",
                   (15, y_offset+95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        return panel

# ==================== INITIALIZATION ====================

# Create directories
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Find available cameras
print("🎥 Searching for cameras...")
available_cameras = []
for i in range(5):  # Check first 5 camera indices
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, test_frame = test_cap.read()
        if ret:
            available_cameras.append(i)
            print(f"   Camera {i}: Available ({test_frame.shape[1]}x{test_frame.shape[0]})")
        test_cap.release()

if not available_cameras:
    print("❌ ERROR: No cameras found!")
    exit()

# Select camera (prefer laptop camera - usually index 1 on macOS if phone is connected)
camera_index = available_cameras[-1] if len(available_cameras) > 1 else available_cameras[0]
print(f"\n📹 Using camera {camera_index}")
print(f"   💡 If wrong camera, edit line with VideoCapture({camera_index}) to try different index\n")

# Initialize camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera!")
    exit()

# Camera warm-up
for _ in range(10):
    cap.read()

# Get first frame
ret, previous_frame = cap.read()
if not ret:
    print("❌ ERROR: Could not read from camera!")
    exit()

h, w = previous_frame.shape[:2]
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

# Initialize components
motion_state = DetectionState(MOTION_CONFIRMATION_FRAMES, MOTION_COOLDOWN_SECONDS)
smoke_state = DetectionState(SMOKE_CONFIRMATION_FRAMES, SMOKE_COOLDOWN_SECONDS)
smoke_tracker = SmokeTracker(history_size=10)
analysis_panel = AnalysisPanel(width=400, height=600)

# Settings
motion_enabled = True
smoke_enabled = True

# ==================== STARTUP ====================

print("\n" + "="*70)
print("🚨 INTELLIGENT SURVEILLANCE SYSTEM V2 🚨")
print("="*70)
print(f"\n📊 Motion Detection:")
print(f"   Confirmation: {MOTION_CONFIRMATION_FRAMES} consecutive frames")
print(f"   Cooldown: {MOTION_COOLDOWN_SECONDS} seconds")
print(f"   Min area: {MOTION_MIN_AREA} pixels")
print(f"\n🔥 Smoke Detection:")
print(f"   Confirmation: {SMOKE_CONFIRMATION_FRAMES} consecutive frames")
print(f"   Cooldown: {SMOKE_COOLDOWN_SECONDS} seconds")
print(f"   Rising threshold: {SMOKE_RISING_THRESHOLD}px upward")
print(f"   Growth factor: {SMOKE_GROWTH_MIN}x")
print(f"\n🎮 Controls:")
print(f"   q: Quit  |  s: Snapshot  |  r: Reset")
print(f"   m: Toggle motion  |  f: Toggle smoke")
print("\n🟢 System online - Waiting for threats...\n")
print("="*70 + "\n")

# ==================== MAIN LOOP ====================

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        display_frame = frame.copy()

        # ===== MOTION DETECTION =====
        motion_detected = False
        motion_mask = np.zeros((h, w), dtype=np.uint8)

        if motion_enabled:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

            # Frame difference
            frame_diff = cv2.absdiff(previous_gray, current_gray)
            _, motion_thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)
            motion_thresh = cv2.dilate(motion_thresh, None, iterations=2)

            # Find large contours
            contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_motion_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MOTION_MIN_AREA:
                    total_motion_area += area
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)

            motion_detected = total_motion_area > MOTION_MIN_AREA
            motion_mask = motion_thresh
            previous_gray = current_gray

        # Update motion state
        if motion_state.update(motion_detected):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n🚨 INTRUDER ALERT! [{timestamp}]")
            print(f"   Motion confirmed over {MOTION_CONFIRMATION_FRAMES} frames")
            cv2.imwrite(f"{SNAPSHOT_DIR}/intruder_{timestamp.replace(':', '-')}.jpg", frame)
            print(f"   📸 Snapshot saved\n")

        # Visual alarm
        if motion_state.alarm_active:
            cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(display_frame, "!!! INTRUDER DETECTED !!!", (20, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        # ===== SMOKE DETECTION =====
        smoke_detected = False
        smoke_mask = np.zeros((h, w), dtype=np.uint8)

        if smoke_enabled:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            smoke_mask = cv2.inRange(hsv, LOWER_SMOKE, UPPER_SMOKE)

            # Clean noise
            kernel = np.ones((7, 7), np.uint8)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

            # Find smoke regions
            smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_smoke_area = 0
            smoke_centroid = None

            for contour in smoke_contours:
                area = cv2.contourArea(contour)
                if area > SMOKE_MIN_AREA:
                    total_smoke_area += area

                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        smoke_centroid = (cx, cy)

                        # Visualize
                        cv2.drawContours(display_frame, [contour], -1, (0, 255, 255), 2)
                        cv2.circle(display_frame, (cx, cy), 8, (0, 165, 255), -1)

            # Update smoke tracker
            if total_smoke_area > SMOKE_MIN_AREA and smoke_centroid:
                smoke_tracker.update(smoke_centroid, total_smoke_area)

                # Check smoke characteristics
                is_rising = smoke_tracker.is_rising(SMOKE_RISING_THRESHOLD)
                is_growing = smoke_tracker.is_growing(SMOKE_GROWTH_MIN)

                # Smoke detected if area is large AND (rising OR growing)
                smoke_detected = total_smoke_area > SMOKE_MIN_AREA and (is_rising or is_growing)
            else:
                smoke_tracker.reset()

        # Update smoke state
        if smoke_state.update(smoke_detected):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n🔥 FIRE/SMOKE ALERT! [{timestamp}]")
            print(f"   Smoke confirmed over {SMOKE_CONFIRMATION_FRAMES} frames")
            print(f"   Rising: {smoke_tracker.is_rising()}  |  Growing: {smoke_tracker.is_growing()}")
            cv2.imwrite(f"{SNAPSHOT_DIR}/fire_{timestamp.replace(':', '-')}.jpg", frame)
            print(f"   📸 Snapshot saved\n")

        # Visual alarm
        if smoke_state.alarm_active:
            cv2.rectangle(display_frame, (0, 0), (w, h), (0, 100, 255), 8)
            cv2.putText(display_frame, "!!! FIRE/SMOKE DETECTED !!!", (20, 100),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 100, 255), 3)

        # ===== CREATE ANALYSIS PANEL =====
        panel = analysis_panel.create_panel(frame, motion_mask, smoke_mask,
                                            motion_state, smoke_state, smoke_tracker)

        # Combine main view and panel
        # Resize panel to match frame height
        panel_resized = cv2.resize(panel, (400, h))
        combined = np.hstack([display_frame, panel_resized])

        # Show combined view
        cv2.imshow('🚨 Surveillance System + Analysis', combined)

        # ===== KEYBOARD CONTROLS =====
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n🛑 Shutting down...")
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f"{SNAPSHOT_DIR}/manual_{timestamp}.jpg", frame)
            print(f"📸 Manual snapshot saved: manual_{timestamp}.jpg")
        elif key == ord('r'):
            motion_state.reset()
            smoke_state.reset()
            smoke_tracker.reset()
            print("🔄 All alarms reset")
        elif key == ord('m'):
            motion_enabled = not motion_enabled
            print(f"🚶 Motion detection: {'ON' if motion_enabled else 'OFF'}")
        elif key == ord('f'):
            smoke_enabled = not smoke_enabled
            print(f"🔥 Smoke detection: {'ON' if smoke_enabled else 'OFF'}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"Motion alarms: {motion_state.alarm_count}")
    print(f"Smoke alarms: {smoke_state.alarm_count}")
    print(f"Frames processed: {frame_count}")
    print(f"\n✅ System stopped gracefully")
    print("="*70)
