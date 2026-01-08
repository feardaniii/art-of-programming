#!/usr/bin/env python3
# pip install mediapipe opencv-python numpy
"""
====== INTERACTIVE HAND TRACKING LAB ======
Updated for MediaPipe Tasks API (Python 3.12+ compatible)

Learn hand tracking with REAL webcam feedback!
Uses the new stable MediaPipe Tasks API instead of legacy solutions.
"""

import cv2
import numpy as np
import time
import platform
import os
import urllib.request

# MediaPipe Tasks imports (new API)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTrackingLab:
    """
    Interactive hand tracking system using MediaPipe Tasks API.
    Works on Mac, Windows, and Linux with Python 3.12+!
    """
    
    # Hand landmark indices (21 points per hand)
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    # Hand connections for drawing skeleton
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)            # Palm
    ]
    
    def __init__(self, model_path: str = None):
        """
        Initialize hand tracker.
        
        Args:
            model_path: Path to hand_landmarker.task model file.
                       If None, will auto-download.
        """
        self.model_path = model_path or self._ensure_model()
        self.detector = None
        self.latest_result = None
        self.result_timestamp = 0
        
        # Gesture tracking
        self.last_gesture_time = time.time()
        
        # Statistics
        self.frame_count = 0
        self.hands_detected_count = 0
        
        print(f"\n🖐️  MediaPipe Hand Tracking Lab initialized!")
        print(f"💻 Platform: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {platform.python_version()}")
        print(f"📦 MediaPipe: {mp.__version__}")
    
    def _ensure_model(self) -> str:
        """Download the hand landmarker model if not present."""
        model_filename = "hand_landmarker.task"
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        
        if os.path.exists(model_filename):
            print(f"✅ Model found: {model_filename}")
            return model_filename
        
        print(f"📥 Downloading hand landmarker model...")
        try:
            urllib.request.urlretrieve(model_url, model_filename)
            print(f"✅ Model downloaded: {model_filename}")
            return model_filename
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def _result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        """Callback for live stream mode results."""
        self.latest_result = result
        self.result_timestamp = timestamp_ms
    
    def get_finger_status(self, landmarks, handedness: str) -> list:
        """
        Determine which fingers are extended (up).
        
        Args:
            landmarks: List of NormalizedLandmark objects
            handedness: "Left" or "Right"
            
        Returns:
            [thumb, index, middle, ring, pinky] as booleans
        """
        fingers_up = []
        
        # Thumb (special case - horizontal comparison)
        # For right hand: tip.x < ip.x means extended
        # For left hand: tip.x > ip.x means extended
        if handedness == "Right":
            fingers_up.append(landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_IP].x)
        else:
            fingers_up.append(landmarks[self.THUMB_TIP].x > landmarks[self.THUMB_IP].x)
        
        # Other fingers: tip.y < pip.y means extended (y increases downward)
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(landmarks[tip].y < landmarks[pip].y)
        
        return fingers_up
    
    def calculate_distance(self, lm1, lm2) -> float:
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt(
            (lm1.x - lm2.x)**2 +
            (lm1.y - lm2.y)**2 +
            (lm1.z - lm2.z)**2
        )
    
    def detect_gesture(self, landmarks, handedness: str) -> tuple:
        """
        Detect hand gesture from landmarks.
        
        Returns:
            (gesture_name, confidence)
        """
        fingers = self.get_finger_status(landmarks, handedness)
        
        # Calculate pinch distance (thumb tip to index tip)
        pinch_dist = self.calculate_distance(
            landmarks[self.THUMB_TIP],
            landmarks[self.INDEX_TIP]
        )
        
        # GESTURE DETECTION
        
        # 1. PINCH (Thumb + Index close together)
        if pinch_dist < 0.05:
            return "PINCH 🤏", 0.95
        
        # 2. PEACE SIGN (Index + Middle up, others down)
        if fingers == [False, True, True, False, False]:
            return "PEACE ✌️", 0.90
        
        # 3. THUMBS UP (Only thumb up)
        if fingers == [True, False, False, False, False]:
            return "THUMBS UP 👍", 0.90
        
        # 4. ROCK ON (Index + Pinky up, others down)
        if fingers == [False, True, False, False, True]:
            return "ROCK ON 🤘", 0.90
        
        # 5. POINTING (Only index up)
        if fingers == [False, True, False, False, False]:
            return "POINTING ☝️", 0.85
        
        # 6. OPEN PALM (All fingers up)
        if all(fingers):
            return "OPEN PALM 🖐️", 0.85
        
        # 7. FIST (All fingers down)
        if not any(fingers):
            return "FIST ✊", 0.85
        
        # 8. THREE (Thumb + Index + Middle up)
        if fingers == [True, True, True, False, False]:
            return "THREE 🤟", 0.80
        
        # 9. OK SIGN (Thumb + Index circle, others up)
        if pinch_dist < 0.08 and fingers[2:] == [True, True, True]:
            return "OK SIGN 👌", 0.80
        
        return "UNKNOWN", 0.0
    
    def draw_landmarks(self, frame, landmarks, handedness: str, hand_idx: int):
        """
        Draw hand skeleton and info on frame.
        
        Returns:
            (gesture_name, finger_status_string)
        """
        h, w, _ = frame.shape
        
        # Convert landmarks to pixel coordinates
        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))
        
        # Draw connections (skeleton)
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        
        # Draw landmark points
        for i, (px, py) in enumerate(points):
            # Highlight fingertips
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (px, py), 8, (255, 0, 255), -1)
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
        
        # Draw pinch line (thumb to index)
        thumb_px = points[self.THUMB_TIP]
        index_px = points[self.INDEX_TIP]
        pinch_dist = self.calculate_distance(landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP])
        line_color = (0, 255, 0) if pinch_dist < 0.05 else (255, 255, 255)
        cv2.line(frame, thumb_px, index_px, line_color, 2)
        
        # Detect gesture
        gesture, confidence = self.detect_gesture(landmarks, handedness)
        
        # Draw info box
        info_y = 60 + (hand_idx * 130)
        cv2.rectangle(frame, (10, info_y), (380, info_y + 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, info_y), (380, info_y + 120), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Hand {hand_idx + 1}: {handedness}",
                   (20, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Gesture: {gesture}",
                   (20, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                   (20, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Pinch Distance: {pinch_dist:.3f}",
                   (20, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Build finger status string
        fingers = self.get_finger_status(landmarks, handedness)
        finger_icons = ['👍', '☝️', '🖕', '💍', '🤙']
        finger_status = ' '.join([icon if up else '❌' for icon, up in zip(finger_icons, fingers)])
        
        return gesture, finger_status
    
    def run_webcam_demo(self):
        """
        Run interactive webcam demo.
        Shows real-time hand tracking with gesture recognition!
        """
        print("\n" + "="*70)
        print("🎥 STARTING WEBCAM HAND TRACKING")
        print("="*70)
        print("\n📋 Instructions:")
        print("   • Position your hand(s) in front of camera")
        print("   • Try different gestures:")
        print("       - Open palm 🖐️")
        print("       - Peace sign ✌️")
        print("       - Pinch 🤏 (thumb + index)")
        print("       - Thumbs up 👍")
        print("       - Rock on 🤘")
        print("       - Fist ✊")
        print("\n⌨️  Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to save screenshot")
        print("   • Press 'r' to reset statistics")
        print("="*70 + "\n")
        
        # Create hand landmarker with VIDEO mode
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Try different camera indices
        camera_indices = [0, 1, 2]
        cap = None
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ Camera {idx} opened successfully!")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("❌ Error: Could not open any camera!")
            print("\n💡 Troubleshooting:")
            print("   Mac: Check System Preferences > Security & Privacy > Camera")
            print("   Windows: Check Settings > Privacy > Camera")
            print("   Linux: Ensure user is in 'video' group")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_time = time.time()
        fps = 0
        timestamp_ms = 0
        
        with vision.HandLandmarker.create_from_options(options) as detector:
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        print("⚠️  Failed to grab frame")
                        break
                    
                    self.frame_count += 1
                    timestamp_ms += 33  # ~30fps
                    
                    # Flip frame for selfie view
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Detect hand landmarks
                    result = detector.detect_for_video(mp_image, timestamp_ms)
                    
                    # Draw header bar
                    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - fps_time > 0:
                        fps = 1.0 / (current_time - fps_time)
                    fps_time = current_time
                    
                    # Draw FPS and stats
                    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {self.frame_count}", (20, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Process results
                    if result.hand_landmarks:
                        self.hands_detected_count += 1
                        
                        for hand_idx, (landmarks, handedness) in enumerate(
                            zip(result.hand_landmarks, result.handedness)
                        ):
                            hand_label = handedness[0].category_name
                            
                            # Draw landmarks and get gesture
                            gesture, finger_status = self.draw_landmarks(
                                frame, landmarks, hand_label, hand_idx
                            )
                            
                            # Log gestures (debounced)
                            if gesture != "UNKNOWN" and time.time() - self.last_gesture_time > 1.0:
                                print(f"   🎯 {gesture} detected! Fingers: {finger_status}")
                                self.last_gesture_time = time.time()
                    else:
                        # No hands detected message
                        cv2.putText(frame, "No hands detected - Show your hand!",
                                   (w//2 - 220, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.9, (0, 0, 255), 2)
                    
                    # Show help text
                    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot | 'r' to reset",
                               (w - 450, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (180, 180, 180), 1)
                    
                    # Display frame
                    cv2.imshow('Hand Tracking Lab - MediaPipe Tasks', frame)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n👋 Exiting...")
                        break
                    elif key == ord('s'):
                        filename = f'hand_tracking_{int(time.time())}.jpg'
                        cv2.imwrite(filename, frame)
                        print(f"📸 Screenshot saved: {filename}")
                    elif key == ord('r'):
                        self.frame_count = 0
                        self.hands_detected_count = 0
                        print("\n🔄 Statistics reset!")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
                # Print statistics
                print("\n" + "="*70)
                print("📊 SESSION STATISTICS")
                print("="*70)
                print(f"   Total frames processed: {self.frame_count}")
                print(f"   Frames with hands detected: {self.hands_detected_count}")
                if self.frame_count > 0:
                    rate = (self.hands_detected_count / self.frame_count) * 100
                    print(f"   Detection rate: {rate:.1f}%")
                print("="*70 + "\n")


def process_image(image_path: str, model_path: str = None):
    """
    Process a single image file.
    Perfect for testing without a webcam!
    
    Args:
        image_path: Path to image file
        model_path: Optional path to model file
    """
    print(f"\n📸 Processing image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Ensure model exists
    model_file = model_path or "hand_landmarker.task"
    if not os.path.exists(model_file):
        print("📥 Downloading model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_file)
    
    # Create detector for IMAGE mode
    base_options = python.BaseOptions(model_asset_path=model_file)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        # Convert and detect
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            print(f"✅ Detected {len(result.hand_landmarks)} hand(s)!")
            
            h, w, _ = img.shape
            
            for hand_idx, (landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                hand_label = handedness[0].category_name
                print(f"\n📍 Hand {hand_idx + 1}: {hand_label}")
                
                # Convert to pixel coords and draw
                points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                
                # Draw skeleton
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]
                for s, e in connections:
                    cv2.line(img, points[s], points[e], (0, 255, 0), 2)
                
                # Draw points
                for i, pt in enumerate(points):
                    color = (255, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0)
                    cv2.circle(img, pt, 6, color, -1)
                
                # Print some landmarks
                print("   Key landmarks:")
                names = ['Wrist', 'Thumb tip', 'Index tip', 'Middle tip', 'Ring tip', 'Pinky tip']
                indices = [0, 4, 8, 12, 16, 20]
                for name, idx in zip(names, indices):
                    lm = landmarks[idx]
                    print(f"   {name}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
        else:
            print("❌ No hands detected in image")
        
        # Save result
        output_path = 'hand_detection_result.jpg'
        cv2.imwrite(output_path, img)
        print(f"\n✅ Result saved to: {output_path}")
        
        # Display
        cv2.imshow('Hand Detection - Press any key to close', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ====== MAIN ======
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🖐️  MEDIAPIPE HAND TRACKING LAB")
    print("    Updated for Python 3.12+ (MediaPipe Tasks API)")
    print("="*70)
    
    print("\n📚 What you'll learn:")
    print("   • Real-time hand landmark detection (21 points per hand)")
    print("   • Gesture recognition algorithms")
    print("   • Distance calculations in 3D space")
    print("   • Modern MediaPipe Tasks API usage")
    
    print("\n" + "="*70)
    print("🎯 USAGE:")
    print("="*70)
    
    print("\n1️⃣  Webcam mode (RECOMMENDED):")
    print("    lab = HandTrackingLab()")
    print("    lab.run_webcam_demo()")
    
    print("\n2️⃣  Process image file:")
    print("    process_image('hand_photo.jpg')")
    
    print("\n" + "="*70)
    print("🎮 TRY THESE GESTURES:")
    print("="*70)
    print("   🖐️  Open Palm - All fingers extended")
    print("   ✌️  Peace Sign - Index + Middle up")
    print("   👍 Thumbs Up - Only thumb extended")
    print("   🤏 Pinch - Thumb + Index together")
    print("   ✊ Fist - All fingers closed")
    print("   🤘 Rock On - Index + Pinky up")
    
    print("\n" + "="*70)
    print("💡 REQUIREMENTS:")
    print("="*70)
    print("   pip install mediapipe opencv-python numpy")
    print("   (Model auto-downloads on first run)")
    
    print("\n" + "="*70)
    print("\n▶️  Starting webcam demo...")
    print("="*70 + "\n")
    
    # Auto-start webcam demo
    lab = HandTrackingLab()
    lab.run_webcam_demo()