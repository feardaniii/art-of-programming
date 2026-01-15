#!/usr/bin/env python3
# Required install: pip install mediapipe opencv-python numpy
"""
====== INTERACTIVE HAND TRACKING LAB ======
Updated for MediaPipe Tasks API (Python 3.9 - 3.12 compatible)

- Auto-sets working directory to script location.
- Auto-downloads the required AI model.
- LOOPS video correctly without crashing MediaPipe.
"""

import cv2
import numpy as np
import time
import os
import urllib.request

# MediaPipe Tasks imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTrackingLab:
    """
    Interactive hand tracking system using MediaPipe Tasks API.
    """
    
    # Hand landmark indices
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
        self.model_path = model_path or self._ensure_model()
        
        print(f"\n🖐️  MediaPipe Hand Tracking Lab initialized!")
        print(f"📦 MediaPipe Version: {mp.__version__}")
    
    def _ensure_model(self) -> str:
        """Download the hand landmarker model if not present."""
        model_filename = "hand_landmarker.task"
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        
        if os.path.exists(model_filename):
            return model_filename
        
        print(f"📥 Downloading hand landmarker model from Google...")
        try:
            urllib.request.urlretrieve(model_url, model_filename)
            print(f"✅ Model downloaded: {model_filename}")
            return model_filename
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def calculate_distance(self, lm1, lm2) -> float:
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    
    def get_finger_status(self, landmarks, handedness: str) -> list:
        """Returns list of booleans [thumb, index, middle, ring, pinky] indicating if extended."""
        fingers_up = []
        
        # Thumb: Horizontal check based on handedness
        if handedness == "Right":
            fingers_up.append(landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_IP].x)
        else:
            fingers_up.append(landmarks[self.THUMB_TIP].x > landmarks[self.THUMB_IP].x)
        
        # Other fingers: Vertical check (tip higher than pip)
        # Note: In image coordinates, Y=0 is top, so "up" means lower Y value
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(landmarks[tip].y < landmarks[pip].y)
        
        return fingers_up
    
    def detect_gesture(self, landmarks, handedness: str) -> tuple:
        """Simple gesture logic."""
        fingers = self.get_finger_status(landmarks, handedness)
        pinch_dist = self.calculate_distance(landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP])

        if pinch_dist < 0.05: return "PINCH 🤏", 0.95
        if fingers == [False, True, True, False, False]: return "PEACE ✌️", 0.90
        if fingers == [True, False, False, False, False]: return "THUMBS UP 👍", 0.90
        if fingers == [False, True, False, False, True]: return "ROCK ON 🤘", 0.90
        if fingers == [False, True, False, False, False]: return "POINTING ☝️", 0.85
        if all(fingers): return "OPEN PALM 🖐️", 0.85
        if not any(fingers): return "FIST ✊", 0.85
        if fingers == [True, True, True, False, False]: return "THREE 🤟", 0.80
        if pinch_dist < 0.08 and fingers[2:] == [True, True, True]: return "OK SIGN 👌", 0.80
        
        return "UNKNOWN", 0.0
    
    def draw_landmarks(self, frame, landmarks, handedness: str, hand_idx: int):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        # Draw skeleton
        for start, end in self.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        
        # Draw joints
        for i, (px, py) in enumerate(points):
            color = (255, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0) # Tips vs joints
            cv2.circle(frame, (px, py), 5, color, -1)

        # Draw Overlay Info
        gesture, conf = self.detect_gesture(landmarks, handedness)
        info_y = 60 + (hand_idx * 130)
        
        # Semi-transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, info_y), (300, info_y + 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"{handedness} Hand", (20, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{gesture}", (20, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return gesture

    def process_video_stream(self, source=0):
        source_desc = "WEBCAM" if isinstance(source, int) else f"FILE: {source}"
        print(f"\n🎥 STARTING TRACKING ON {source_desc}")
        print("⌨️  Press 'q' to quit.")

        # Configure MediaPipe
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open {source}")
            return

        timestamp_ms = 0
        
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = cap.read()
                
                # --- LOOPING LOGIC START ---
                if not success:
                    # Loop video if it's a file
                    if isinstance(source, str):
                        print("🎬 End of video. Looping...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # CRITICAL FIX: Do NOT reset timestamp_ms to 0. 
                        # MediaPipe needs strictly increasing time.
                        continue
                    else:
                        print("🎬 End of stream.")
                        break
                # --- LOOPING LOGIC END ---

                # For webcam, flip horizontally for mirror effect
                if isinstance(source, int):
                    frame = cv2.flip(frame, 1)

                # Process Frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Timestamp must always increase, even if video loops
                timestamp_ms += 33 
                
                result = detector.detect_for_video(mp_image, timestamp_ms)
                
                # Draw Results
                if result.hand_landmarks:
                    for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                        self.draw_landmarks(frame, landmarks, handedness[0].category_name, i)

                cv2.imshow('Hand Tracking Lab', frame)
                
                # Slow down if playing a file to match approximate speed
                if isinstance(source, str):
                    time.sleep(0.01)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    # Set working directory to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 Working directory changed to: {script_dir}")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Path to image file')
    parser.add_argument('-v', '--video', type=str, help='Force a specific video path')
    args = parser.parse_args()

    lab = HandTrackingLab()

    if args.image:
        print("Image mode selected")
    elif args.video:
        lab.process_video_stream(args.video)
    else:
        # === AUTO DETECT LOGIC ===
        default_video = "video.mp4"
        if os.path.exists(default_video):
            print(f"\n📂 Found '{default_video}' in current folder.")
            lab.process_video_stream(default_video)
        else:
            print(f"\n🚫 '{default_video}' not found in {script_dir}. Switching to Webcam...")
            lab.process_video_stream(0)