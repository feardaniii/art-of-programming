"""
================================================================================
REAL-TIME WORKOUT FORM ANALYZER
================================================================================
Art of Programming - Computer Vision Module

Uses MediaPipe Tasks API (2024+) for pose estimation.

Dependencies:
    pip install mediapipe opencv-python numpy

First run downloads pose model (~4MB) automatically.

Usage:
    python workout_form_analyzer.py
    python workout_form_analyzer.py --exercise bicep_curl
================================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum, auto
import time
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# =============================================================================
# MODEL SETUP
# =============================================================================

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def download_model():
    """Download pose model if not present"""
    if not os.path.exists(MODEL_PATH):
        print(f"  Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}")
    return MODEL_PATH


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ExercisePhase(Enum):
    NEUTRAL = auto()
    ECCENTRIC = auto()   # Going down
    CONCENTRIC = auto()  # Coming up


@dataclass
class JointAngles:
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    spine_angle: float = 0.0


@dataclass
class FormFeedback:
    is_good: bool
    message: str
    severity: str = "info"
    joint: Optional[str] = None


@dataclass
class RepData:
    rep_number: int
    duration_seconds: float
    min_angle: float
    max_angle: float
    form_score: float
    feedback: list = field(default_factory=list)


# =============================================================================
# POSE DETECTOR (New Tasks API)
# =============================================================================

class PoseDetector:
    """
    MediaPipe Pose Landmarker wrapper.

    Detects 33 body landmarks with 3D coordinates.
    """

    # Landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Skeleton connections
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
        (11, 23), (12, 24), (23, 24),                       # Torso
        (23, 25), (25, 27), (24, 26), (26, 28),            # Legs
    ]

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):

        model_path = download_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.landmarks = None
        self.frame_shape = None
        self.timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> bool:
        """Process frame, return True if pose found"""
        self.frame_shape = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp_ms += 33  # ~30 FPS increment
        results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            self.landmarks = results.pose_landmarks[0]
            return True

        self.landmarks = None
        return False

    def get_pixel(self, idx: int) -> Optional[tuple]:
        """Get (x, y) pixel coords for landmark"""
        if self.landmarks is None:
            return None

        lm = self.landmarks[idx]
        if lm.visibility < 0.5:
            return None

        h, w = self.frame_shape[:2]
        return (int(lm.x * w), int(lm.y * h))

    def get_3d(self, idx: int) -> Optional[tuple]:
        """Get normalized (x, y, z) coords"""
        if self.landmarks is None:
            return None

        lm = self.landmarks[idx]
        if lm.visibility < 0.5:
            return None

        return (lm.x, lm.y, lm.z)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw skeleton on frame"""
        if self.landmarks is None:
            return frame

        out = frame.copy()

        # Draw bones
        for start, end in self.CONNECTIONS:
            p1 = self.get_pixel(start)
            p2 = self.get_pixel(end)
            if p1 and p2:
                cv2.line(out, p1, p2, (0, 200, 0), 2)

        # Draw joints
        for i in range(33):
            pos = self.get_pixel(i)
            if pos:
                cv2.circle(out, pos, 4, (0, 255, 0), -1)

        return out


# =============================================================================
# ANGLE CALCULATOR
# =============================================================================

class AngleCalculator:
    """Calculate joint angles using vector math"""

    @staticmethod
    def angle(a: tuple, b: tuple, c: tuple) -> float:
        """
        Angle at point B formed by A-B-C (in degrees).

        Uses: angle = arccos( (BA · BC) / (|BA| × |BC|) )
        """
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])

        ba = a - b
        bc = c - b

        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos = np.clip(cos, -1.0, 1.0)

        return np.degrees(np.arccos(cos))

    @classmethod
    def compute_all(cls, det: PoseDetector) -> Optional[JointAngles]:
        """Compute all joint angles from pose"""
        if det.landmarks is None:
            return None

        get = det.get_3d
        angles = JointAngles()

        # Elbows (shoulder -> elbow -> wrist)
        if all(get(i) for i in [det.LEFT_SHOULDER, det.LEFT_ELBOW, det.LEFT_WRIST]):
            angles.left_elbow = cls.angle(
                get(det.LEFT_SHOULDER), get(det.LEFT_ELBOW), get(det.LEFT_WRIST)
            )

        if all(get(i) for i in [det.RIGHT_SHOULDER, det.RIGHT_ELBOW, det.RIGHT_WRIST]):
            angles.right_elbow = cls.angle(
                get(det.RIGHT_SHOULDER), get(det.RIGHT_ELBOW), get(det.RIGHT_WRIST)
            )

        # Knees (hip -> knee -> ankle)
        if all(get(i) for i in [det.LEFT_HIP, det.LEFT_KNEE, det.LEFT_ANKLE]):
            angles.left_knee = cls.angle(
                get(det.LEFT_HIP), get(det.LEFT_KNEE), get(det.LEFT_ANKLE)
            )

        if all(get(i) for i in [det.RIGHT_HIP, det.RIGHT_KNEE, det.RIGHT_ANKLE]):
            angles.right_knee = cls.angle(
                get(det.RIGHT_HIP), get(det.RIGHT_KNEE), get(det.RIGHT_ANKLE)
            )

        # Hips (shoulder -> hip -> knee)
        if all(get(i) for i in [det.LEFT_SHOULDER, det.LEFT_HIP, det.LEFT_KNEE]):
            angles.left_hip = cls.angle(
                get(det.LEFT_SHOULDER), get(det.LEFT_HIP), get(det.LEFT_KNEE)
            )

        if all(get(i) for i in [det.RIGHT_SHOULDER, det.RIGHT_HIP, det.RIGHT_KNEE]):
            angles.right_hip = cls.angle(
                get(det.RIGHT_SHOULDER), get(det.RIGHT_HIP), get(det.RIGHT_KNEE)
            )

        # Spine angle (forward lean)
        ls, rs = get(det.LEFT_SHOULDER), get(det.RIGHT_SHOULDER)
        lh, rh = get(det.LEFT_HIP), get(det.RIGHT_HIP)

        if all([ls, rs, lh, rh]):
            mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            vertical = (mid_hip[0], mid_hip[1] - 0.3)
            angles.spine_angle = cls.angle(vertical, mid_hip, mid_shoulder)

        return angles


# =============================================================================
# EXERCISE ANALYZER
# =============================================================================

class ExerciseAnalyzer:
    """Track reps and check form for specific exercises"""

    EXERCISES = {
        'squat': {
            'joint': 'knee',
            'top': 160,
            'bottom': 100,
            'form_checks': [('spine_angle', 30, 'Keep back straight')]
        },
        'bicep_curl': {
            'joint': 'elbow',
            'top': 150,
            'bottom': 40,
            'form_checks': []
        },
        'shoulder_press': {
            'joint': 'elbow',
            'top': 160,
            'bottom': 90,
            'form_checks': [('spine_angle', 20, 'Avoid leaning back')]
        }
    }

    def __init__(self, exercise: str = 'squat'):
        if exercise not in self.EXERCISES:
            raise ValueError(f"Unknown: {exercise}. Options: {list(self.EXERCISES.keys())}")

        self.exercise = exercise
        self.config = self.EXERCISES[exercise]

        self.phase = ExercisePhase.NEUTRAL
        self.rep_count = 0
        self.history: list[RepData] = []

        self.angle_buffer = deque(maxlen=5)
        self.phase_buffer = deque(maxlen=3)

        self.rep_start = None
        self.rep_min = 180
        self.rep_max = 0
        self.rep_feedback = []

    def get_angle(self, angles: JointAngles) -> float:
        """Get primary angle for this exercise"""
        joint = self.config['joint']
        if joint == 'knee':
            return (angles.left_knee + angles.right_knee) / 2
        elif joint == 'elbow':
            return (angles.left_elbow + angles.right_elbow) / 2
        return 0

    def update(self, angles: JointAngles) -> Optional[int]:
        """Update with new angles, return rep count if completed"""
        angle = self.get_angle(angles)

        self.angle_buffer.append(angle)
        smooth = np.mean(self.angle_buffer)

        self.rep_min = min(self.rep_min, smooth)
        self.rep_max = max(self.rep_max, smooth)

        # Determine phase
        top, bottom = self.config['top'], self.config['bottom']

        if smooth > top:
            new_phase = ExercisePhase.NEUTRAL
        elif smooth < bottom:
            new_phase = ExercisePhase.ECCENTRIC
        else:
            if self.phase == ExercisePhase.NEUTRAL:
                new_phase = ExercisePhase.ECCENTRIC
            elif self.phase == ExercisePhase.ECCENTRIC:
                new_phase = ExercisePhase.CONCENTRIC
            else:
                new_phase = self.phase

        self.phase_buffer.append(new_phase)

        # Phase change with smoothing
        if len(self.phase_buffer) == 3 and len(set(self.phase_buffer)) == 1:
            if new_phase != self.phase:
                # Rep complete: came back up to neutral
                if self.phase == ExercisePhase.CONCENTRIC and new_phase == ExercisePhase.NEUTRAL:
                    self.rep_count += 1

                    duration = time.time() - self.rep_start if self.rep_start else 0
                    score = self._score(angles)

                    self.history.append(RepData(
                        rep_number=self.rep_count,
                        duration_seconds=duration,
                        min_angle=self.rep_min,
                        max_angle=self.rep_max,
                        form_score=score,
                        feedback=self.rep_feedback.copy()
                    ))

                    self.rep_min, self.rep_max = 180, 0
                    self.rep_feedback = []
                    self.rep_start = time.time()
                    self.phase = new_phase
                    return self.rep_count

                # Starting descent
                if self.phase == ExercisePhase.NEUTRAL and new_phase == ExercisePhase.ECCENTRIC:
                    self.rep_start = time.time()

                self.phase = new_phase

        return None

    def check_form(self, angles: JointAngles) -> list[FormFeedback]:
        """Check form, return feedback"""
        feedback = []

        for check, threshold, msg in self.config['form_checks']:
            if check == 'spine_angle' and angles.spine_angle > threshold:
                feedback.append(FormFeedback(False, msg, 'warning', 'spine'))
                self.rep_feedback.append(msg)

        if not feedback:
            feedback.append(FormFeedback(True, "Good form!", 'info'))

        return feedback

    def _score(self, angles: JointAngles) -> float:
        """Calculate form score 0-100"""
        score = 100.0

        if angles.spine_angle > 30:
            score -= min(30, angles.spine_angle - 30)

        knee_diff = abs(angles.left_knee - angles.right_knee)
        if knee_diff > 10:
            score -= min(20, knee_diff - 10)

        if self.exercise == 'squat' and self.rep_min < 90:
            score += 5

        return max(0, min(100, score))

    def stats(self) -> dict:
        """Get workout stats"""
        if not self.history:
            return {'total_reps': 0}

        return {
            'total_reps': self.rep_count,
            'avg_score': np.mean([r.form_score for r in self.history]),
            'avg_duration': np.mean([r.duration_seconds for r in self.history]),
            'best_depth': min(r.min_angle for r in self.history)
        }


# =============================================================================
# VISUALIZER
# =============================================================================

class Visualizer:
    """Render UI overlay on video frame"""

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    ORANGE = (0, 165, 255)

    def __init__(self):
        self.feedback_queue = []  # (msg, expire_time, severity)

    def add_feedback(self, fb: FormFeedback):
        expire = time.time() + 2.0
        self.feedback_queue.append((fb.message, expire, fb.severity))

    def render(self, frame: np.ndarray, angles: Optional[JointAngles],
               analyzer: ExerciseAnalyzer, detector: PoseDetector) -> np.ndarray:

        out = frame.copy()
        h, w = out.shape[:2]

        # Clean expired feedback
        now = time.time()
        self.feedback_queue = [f for f in self.feedback_queue if f[1] > now]

        # Rep counter (big, center top)
        text = str(analyzer.rep_count)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)

        cv2.rectangle(out, (w//2 - tw//2 - 20, 10), (w//2 + tw//2 + 20, th + 40), (0,0,0), -1)
        cv2.rectangle(out, (w//2 - tw//2 - 20, 10), (w//2 + tw//2 + 20, th + 40), self.ORANGE, 2)
        cv2.putText(out, text, (w//2 - tw//2, th + 25), cv2.FONT_HERSHEY_SIMPLEX, 3, self.GREEN, 5)

        # Exercise name
        cv2.putText(out, analyzer.exercise.upper(), (w//2 - 50, th + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)

        # Phase
        phase_colors = {
            ExercisePhase.NEUTRAL: self.WHITE,
            ExercisePhase.ECCENTRIC: self.YELLOW,
            ExercisePhase.CONCENTRIC: self.GREEN
        }
        cv2.putText(out, analyzer.phase.name, (20, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_colors[analyzer.phase], 2)

        # Angles near joints
        if angles and detector.landmarks:
            lk = detector.get_pixel(detector.LEFT_KNEE)
            if lk:
                cv2.putText(out, f"{angles.left_knee:.0f}°", (lk[0]-40, lk[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)

            rk = detector.get_pixel(detector.RIGHT_KNEE)
            if rk:
                cv2.putText(out, f"{angles.right_knee:.0f}°", (rk[0]+10, rk[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)

            spine_color = self.GREEN if angles.spine_angle < 25 else self.YELLOW
            cv2.putText(out, f"Back: {angles.spine_angle:.0f}°", (20, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, spine_color, 2)

        # Feedback messages
        y = 100
        for msg, _, sev in self.feedback_queue:
            color = self.YELLOW if sev == 'warning' else self.WHITE
            cv2.putText(out, msg, (w-300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 35

        # Form score
        if analyzer.history:
            score = analyzer.history[-1].form_score
            color = self.GREEN if score >= 80 else self.YELLOW if score >= 60 else self.RED
            cv2.putText(out, f"Form: {score:.0f}%", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return out


# =============================================================================
# MAIN SESSION
# =============================================================================

class WorkoutSession:
    """Main application orchestrator"""

    def __init__(self, exercise: str = 'squat', camera: int = 0):
        self.exercise = exercise
        self.camera = camera

        self.detector = PoseDetector()
        self.analyzer = ExerciseAnalyzer(exercise)
        self.viz = Visualizer()

        self.fps_buffer = deque(maxlen=30)
        self.start_time = None

    def run(self):
        print()
        print("=" * 60)
        print(f"  WORKOUT FORM ANALYZER - {self.exercise.upper()}")
        print("=" * 60)
        print()
        print("  q=quit  r=reset  s=screenshot  1/2/3=switch exercise")
        print()

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.start_time = time.time()

        try:
            while True:
                t0 = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror

                # Pipeline
                angles = None
                if self.detector.detect(frame):
                    angles = AngleCalculator.compute_all(self.detector)

                    if angles:
                        completed = self.analyzer.update(angles)
                        if completed:
                            score = self.analyzer.history[-1].form_score
                            print(f"  Rep {completed}! Score: {score:.0f}%")

                        for fb in self.analyzer.check_form(angles):
                            if not fb.is_good:
                                self.viz.add_feedback(fb)

                # Draw
                annotated = self.detector.draw(frame)
                output = self.viz.render(annotated, angles, self.analyzer, self.detector)

                # FPS
                fps = 1.0 / (time.time() - t0 + 1e-6)
                self.fps_buffer.append(fps)
                cv2.putText(output, f"FPS: {np.mean(self.fps_buffer):.0f}",
                           (output.shape[1]-100, output.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

                cv2.imshow('Workout Form Analyzer', output)

                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.analyzer = ExerciseAnalyzer(self.exercise)
                    print("  Reset!")
                elif key == ord('s'):
                    cv2.imwrite(f"workout_{int(time.time())}.jpg", output)
                    print("  Screenshot saved!")
                elif key == ord('1'):
                    self._switch('squat')
                elif key == ord('2'):
                    self._switch('bicep_curl')
                elif key == ord('3'):
                    self._switch('shoulder_press')

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._summary()

    def _switch(self, ex: str):
        self.exercise = ex
        self.analyzer = ExerciseAnalyzer(ex)
        print(f"  Switched to {ex.upper()}")

    def _summary(self):
        duration = time.time() - self.start_time
        stats = self.analyzer.stats()

        print()
        print("=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Exercise:    {self.exercise.upper()}")
        print(f"  Duration:    {duration/60:.1f} min")
        print(f"  Reps:        {stats['total_reps']}")

        if stats['total_reps'] > 0:
            print(f"  Avg Score:   {stats['avg_score']:.0f}%")
            print(f"  Avg Time:    {stats['avg_duration']:.1f}s")
            print(f"  Best Depth:  {stats['best_depth']:.0f}°")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Workout Form Analyzer')
    parser.add_argument('-e', '--exercise',
                       choices=['squat', 'bicep_curl', 'shoulder_press'],
                       default='squat')
    parser.add_argument('-c', '--camera', type=int, default=0)

    args = parser.parse_args()

    WorkoutSession(exercise=args.exercise, camera=args.camera).run()