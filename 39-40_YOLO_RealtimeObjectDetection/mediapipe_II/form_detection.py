"""
================================================================================
REAL-TIME WORKOUT FORM ANALYZER
================================================================================
Art of Programming - Computer Vision Module

This script demonstrates ACTUAL form analysis using pose estimation:
- Joint angle calculation (not just object detection)
- Rep counting based on movement phases
- Form quality scoring with specific feedback
- Exercise pattern recognition

Dependencies:
    pip install mediapipe opencv-python numpy

Architecture:
    1. PoseDetector      - Extracts 33 body landmarks from video frames
    2. AngleCalculator   - Computes joint angles from landmark positions
    3. ExerciseAnalyzer  - Recognizes exercises and counts reps
    4. FormCoach         - Provides real-time form feedback
    5. WorkoutSession    - Orchestrates everything + generates reports

Why MediaPipe over YOLO for this task?
    - YOLO: "There's a person at position (x1,y1,x2,y2)"
    - MediaPipe: "Here are 33 body landmarks with 3D coordinates"

You can't analyze squat depth with a bounding box.
================================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum, auto
import time


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

class ExercisePhase(Enum):
    """Movement phases for rep counting"""
    NEUTRAL = auto()  # Starting/ending position
    ECCENTRIC = auto()  # Lowering phase (e.g., going down in squat)
    CONCENTRIC = auto()  # Lifting phase (e.g., standing up from squat)


@dataclass
class JointAngles:
    """Computed angles for key joints (in degrees)"""
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    spine_angle: float = 0.0  # Forward lean


@dataclass
class FormFeedback:
    """Structured feedback for form correction"""
    is_good: bool
    message: str
    severity: str = "info"  # info, warning, error
    joint: Optional[str] = None


@dataclass
class RepData:
    """Data collected for each rep"""
    rep_number: int
    duration_seconds: float
    min_angle: float  # Deepest point of movement
    max_angle: float  # Top of movement
    form_score: float  # 0-100
    feedback: list = field(default_factory=list)


# =============================================================================
# SECTION 2: POSE DETECTION
# =============================================================================

class PoseDetector:
    """
    Wrapper around MediaPipe Pose for cleaner interface.

    MediaPipe provides 33 landmarks per frame:
    - Face: nose, eyes, ears, mouth
    - Upper body: shoulders, elbows, wrists
    - Torso: hips
    - Lower body: knees, ankles, heels, toes

    Each landmark has (x, y, z, visibility) where:
    - x, y: Normalized coordinates [0, 1]
    - z: Depth relative to hips
    - visibility: Confidence that landmark is visible [0, 1]
    """

    # MediaPipe landmark indices (memorize the important ones)
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

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Args:
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for frame-to-frame tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode (uses tracking)
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,  # Reduces jitter
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarks = None
        self.frame_shape = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Process frame and extract pose landmarks.

        Args:
            frame: BGR image from OpenCV

        Returns:
            True if pose detected, False otherwise
        """
        self.frame_shape = frame.shape

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.landmarks = results.pose_landmarks.landmark
            return True

        self.landmarks = None
        return False

    def get_landmark_pixel(self, landmark_id: int) -> Optional[tuple]:
        """
        Get pixel coordinates for a landmark.

        Args:
            landmark_id: MediaPipe landmark index

        Returns:
            (x, y) pixel coordinates or None if not detected
        """
        if self.landmarks is None:
            return None

        landmark = self.landmarks[landmark_id]

        if landmark.visibility < 0.5:
            return None

        h, w = self.frame_shape[:2]
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        return (x, y)

    def get_landmark_3d(self, landmark_id: int) -> Optional[tuple]:
        """Get normalized 3D coordinates (x, y, z)"""
        if self.landmarks is None:
            return None

        landmark = self.landmarks[landmark_id]

        if landmark.visibility < 0.5:
            return None

        return (landmark.x, landmark.y, landmark.z)

    def draw_landmarks(self, frame: np.ndarray,
                       highlight_joints: list = None) -> np.ndarray:
        """
        Draw pose skeleton on frame.

        Args:
            frame: Image to draw on
            highlight_joints: List of landmark IDs to highlight in different color
        """
        if self.landmarks is None:
            return frame

        annotated = frame.copy()

        # Draw standard skeleton
        self.mp_draw.draw_landmarks(
            annotated,
            mp.solutions.pose.PoseLandmark,  # Landmark positions
            self.mp_pose.POSE_CONNECTIONS,  # Which landmarks to connect
            landmark_drawing_spec=self.mp_draw.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self.mp_draw.DrawingSpec(
                color=(0, 200, 0), thickness=2
            )
        )

        # Highlight specific joints if requested
        if highlight_joints:
            for joint_id in highlight_joints:
                pos = self.get_landmark_pixel(joint_id)
                if pos:
                    cv2.circle(annotated, pos, 10, (0, 0, 255), -1)

        return annotated


# =============================================================================
# SECTION 3: ANGLE CALCULATION
# =============================================================================

class AngleCalculator:
    """
    Calculate joint angles from pose landmarks.

    Angle calculation uses the law of cosines:
    Given three points A, B, C where B is the vertex:

        angle = arccos( (BA · BC) / (|BA| × |BC|) )

    This gives the angle at joint B formed by the limb segments.
    """

    @staticmethod
    def calculate_angle(point_a: tuple, point_b: tuple, point_c: tuple) -> float:
        """
        Calculate angle at point_b formed by points a-b-c.

        Args:
            point_a: First point (x, y) or (x, y, z)
            point_b: Vertex point (the joint)
            point_c: Third point

        Returns:
            Angle in degrees [0, 180]

        Example:
            For elbow angle: shoulder -> elbow -> wrist
            angle = calculate_angle(shoulder, elbow, wrist)
        """
        a = np.array(point_a[:2])  # Use only x, y
        b = np.array(point_b[:2])
        c = np.array(point_c[:2])

        # Vectors from vertex to each point
        ba = a - b
        bc = c - b

        # Cosine of angle using dot product formula
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)  # Handle floating point errors

        angle = np.degrees(np.arccos(cosine))

        return angle

    @classmethod
    def compute_all_angles(cls, detector: PoseDetector) -> Optional[JointAngles]:
        """
        Compute all relevant joint angles from current pose.

        Returns:
            JointAngles dataclass or None if pose not detected
        """
        if detector.landmarks is None:
            return None

        # Helper to get 3D coords
        def get(landmark_id):
            return detector.get_landmark_3d(landmark_id)

        angles = JointAngles()

        # Elbow angles (shoulder -> elbow -> wrist)
        if all(get(i) for i in [PoseDetector.LEFT_SHOULDER,
                                PoseDetector.LEFT_ELBOW,
                                PoseDetector.LEFT_WRIST]):
            angles.left_elbow = cls.calculate_angle(
                get(PoseDetector.LEFT_SHOULDER),
                get(PoseDetector.LEFT_ELBOW),
                get(PoseDetector.LEFT_WRIST)
            )

        if all(get(i) for i in [PoseDetector.RIGHT_SHOULDER,
                                PoseDetector.RIGHT_ELBOW,
                                PoseDetector.RIGHT_WRIST]):
            angles.right_elbow = cls.calculate_angle(
                get(PoseDetector.RIGHT_SHOULDER),
                get(PoseDetector.RIGHT_ELBOW),
                get(PoseDetector.RIGHT_WRIST)
            )

        # Knee angles (hip -> knee -> ankle)
        if all(get(i) for i in [PoseDetector.LEFT_HIP,
                                PoseDetector.LEFT_KNEE,
                                PoseDetector.LEFT_ANKLE]):
            angles.left_knee = cls.calculate_angle(
                get(PoseDetector.LEFT_HIP),
                get(PoseDetector.LEFT_KNEE),
                get(PoseDetector.LEFT_ANKLE)
            )

        if all(get(i) for i in [PoseDetector.RIGHT_HIP,
                                PoseDetector.RIGHT_KNEE,
                                PoseDetector.RIGHT_ANKLE]):
            angles.right_knee = cls.calculate_angle(
                get(PoseDetector.RIGHT_HIP),
                get(PoseDetector.RIGHT_KNEE),
                get(PoseDetector.RIGHT_ANKLE)
            )

        # Hip angles (shoulder -> hip -> knee)
        if all(get(i) for i in [PoseDetector.LEFT_SHOULDER,
                                PoseDetector.LEFT_HIP,
                                PoseDetector.LEFT_KNEE]):
            angles.left_hip = cls.calculate_angle(
                get(PoseDetector.LEFT_SHOULDER),
                get(PoseDetector.LEFT_HIP),
                get(PoseDetector.LEFT_KNEE)
            )

        if all(get(i) for i in [PoseDetector.RIGHT_SHOULDER,
                                PoseDetector.RIGHT_HIP,
                                PoseDetector.RIGHT_KNEE]):
            angles.right_hip = cls.calculate_angle(
                get(PoseDetector.RIGHT_SHOULDER),
                get(PoseDetector.RIGHT_HIP),
                get(PoseDetector.RIGHT_KNEE)
            )

        # Spine angle (vertical alignment)
        # Measured as angle between: vertical line, shoulder-hip line
        left_shoulder = get(PoseDetector.LEFT_SHOULDER)
        right_shoulder = get(PoseDetector.RIGHT_SHOULDER)
        left_hip = get(PoseDetector.LEFT_HIP)
        right_hip = get(PoseDetector.RIGHT_HIP)

        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            # Midpoints
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            # Virtual point directly above hip (vertical reference)
            vertical_point = (mid_hip[0], mid_hip[1] - 0.3)

            angles.spine_angle = cls.calculate_angle(
                vertical_point, mid_hip, mid_shoulder
            )

        return angles


# =============================================================================
# SECTION 4: EXERCISE ANALYSIS
# =============================================================================

class ExerciseAnalyzer:
    """
    Analyze specific exercises using joint angles.

    Each exercise has:
    - Target joints to monitor
    - Angle thresholds for phases
    - Form criteria for feedback

    Currently supported: Squat, Bicep Curl, Shoulder Press
    Extensible: Add new exercises by defining their parameters
    """

    # Exercise definitions
    EXERCISES = {
        'squat': {
            'primary_joint': 'knee',
            'secondary_joint': 'hip',
            'phase_thresholds': {
                'top': 160,  # Standing: knee angle > 160°
                'bottom': 100,  # Deep squat: knee angle < 100°
            },
            'form_checks': [
                ('spine_angle', 30, 'Keep your back straight'),
                ('knee_symmetry', 15, 'Keep knees aligned'),
            ]
        },
        'bicep_curl': {
            'primary_joint': 'elbow',
            'secondary_joint': None,
            'phase_thresholds': {
                'top': 150,  # Arm extended: elbow > 150°
                'bottom': 40,  # Curled: elbow < 40°
            },
            'form_checks': [
                ('shoulder_movement', 10, 'Keep shoulders still'),
            ]
        },
        'shoulder_press': {
            'primary_joint': 'elbow',
            'secondary_joint': 'shoulder',
            'phase_thresholds': {
                'top': 160,  # Arms extended overhead
                'bottom': 90,  # Arms at 90°
            },
            'form_checks': [
                ('spine_angle', 20, 'Avoid leaning back'),
            ]
        }
    }

    def __init__(self, exercise_name: str = 'squat'):
        """
        Args:
            exercise_name: One of the supported exercises
        """
        if exercise_name not in self.EXERCISES:
            raise ValueError(f"Unknown exercise: {exercise_name}. "
                             f"Supported: {list(self.EXERCISES.keys())}")

        self.exercise_name = exercise_name
        self.config = self.EXERCISES[exercise_name]

        # State tracking
        self.current_phase = ExercisePhase.NEUTRAL
        self.rep_count = 0
        self.rep_history: list[RepData] = []

        # Smoothing buffers (reduce noise)
        self.angle_buffer = deque(maxlen=5)
        self.phase_buffer = deque(maxlen=3)

        # Current rep tracking
        self.rep_start_time = None
        self.rep_min_angle = 180
        self.rep_max_angle = 0
        self.rep_feedback = []

    def get_primary_angle(self, angles: JointAngles) -> float:
        """Get the main angle for this exercise"""
        joint = self.config['primary_joint']

        if joint == 'knee':
            return (angles.left_knee + angles.right_knee) / 2
        elif joint == 'elbow':
            return (angles.left_elbow + angles.right_elbow) / 2
        elif joint == 'hip':
            return (angles.left_hip + angles.right_hip) / 2
        elif joint == 'shoulder':
            return (angles.left_shoulder + angles.right_shoulder) / 2

        return 0

    def update(self, angles: JointAngles) -> Optional[int]:
        """
        Process new angle data and update rep count.

        Args:
            angles: Current joint angles

        Returns:
            New rep count if a rep was completed, None otherwise
        """
        primary_angle = self.get_primary_angle(angles)

        # Smooth the angle
        self.angle_buffer.append(primary_angle)
        smoothed_angle = np.mean(self.angle_buffer)

        # Track min/max for current rep
        self.rep_min_angle = min(self.rep_min_angle, smoothed_angle)
        self.rep_max_angle = max(self.rep_max_angle, smoothed_angle)

        # Determine current phase
        thresholds = self.config['phase_thresholds']

        if smoothed_angle > thresholds['top']:
            new_phase = ExercisePhase.NEUTRAL
        elif smoothed_angle < thresholds['bottom']:
            new_phase = ExercisePhase.ECCENTRIC  # At bottom of movement
        else:
            # In transition
            if self.current_phase == ExercisePhase.NEUTRAL:
                new_phase = ExercisePhase.ECCENTRIC  # Going down
            elif self.current_phase == ExercisePhase.ECCENTRIC:
                new_phase = ExercisePhase.CONCENTRIC  # Coming up
            else:
                new_phase = self.current_phase

        # Phase change detection with smoothing
        self.phase_buffer.append(new_phase)

        # Only change phase if consistent
        if len(self.phase_buffer) == 3 and len(set(self.phase_buffer)) == 1:
            if new_phase != self.current_phase:
                # Check for completed rep: went down AND came back up
                if (self.current_phase == ExercisePhase.CONCENTRIC and
                        new_phase == ExercisePhase.NEUTRAL):
                    # Rep completed!
                    self.rep_count += 1

                    # Calculate rep metrics
                    duration = time.time() - self.rep_start_time if self.rep_start_time else 0
                    form_score = self._calculate_form_score(angles)

                    rep_data = RepData(
                        rep_number=self.rep_count,
                        duration_seconds=duration,
                        min_angle=self.rep_min_angle,
                        max_angle=self.rep_max_angle,
                        form_score=form_score,
                        feedback=self.rep_feedback.copy()
                    )
                    self.rep_history.append(rep_data)

                    # Reset for next rep
                    self.rep_min_angle = 180
                    self.rep_max_angle = 0
                    self.rep_feedback = []
                    self.rep_start_time = time.time()

                    self.current_phase = new_phase
                    return self.rep_count

                # Starting new rep
                if (self.current_phase == ExercisePhase.NEUTRAL and
                        new_phase == ExercisePhase.ECCENTRIC):
                    self.rep_start_time = time.time()

                self.current_phase = new_phase

        return None

    def check_form(self, angles: JointAngles) -> list[FormFeedback]:
        """
        Check form against exercise-specific criteria.

        Returns:
            List of feedback items
        """
        feedback = []

        for check_name, threshold, message in self.config['form_checks']:
            if check_name == 'spine_angle':
                if angles.spine_angle > threshold:
                    fb = FormFeedback(
                        is_good=False,
                        message=message,
                        severity='warning',
                        joint='spine'
                    )
                    feedback.append(fb)
                    self.rep_feedback.append(message)

            elif check_name == 'knee_symmetry':
                diff = abs(angles.left_knee - angles.right_knee)
                if diff > threshold:
                    fb = FormFeedback(
                        is_good=False,
                        message=message,
                        severity='warning',
                        joint='knees'
                    )
                    feedback.append(fb)
                    self.rep_feedback.append(message)

            elif check_name == 'shoulder_movement':
                # Would need to track shoulder position change over time
                # Simplified: just check if shoulder angle changes significantly
                pass

        if not feedback:
            feedback.append(FormFeedback(
                is_good=True,
                message="Good form!",
                severity='info'
            ))

        return feedback

    def _calculate_form_score(self, angles: JointAngles) -> float:
        """Calculate form score 0-100 based on criteria"""
        score = 100.0

        # Deduct for spine lean
        if angles.spine_angle > 30:
            score -= min(30, angles.spine_angle - 30)

        # Deduct for asymmetry
        knee_diff = abs(angles.left_knee - angles.right_knee)
        if knee_diff > 10:
            score -= min(20, knee_diff - 10)

        # Bonus for good depth (for squat)
        if self.exercise_name == 'squat':
            if self.rep_min_angle < 90:  # Deep squat
                score += 5

        return max(0, min(100, score))

    def get_stats(self) -> dict:
        """Get workout statistics"""
        if not self.rep_history:
            return {'total_reps': 0}

        return {
            'total_reps': self.rep_count,
            'avg_form_score': np.mean([r.form_score for r in self.rep_history]),
            'avg_rep_duration': np.mean([r.duration_seconds for r in self.rep_history]),
            'best_depth': min(r.min_angle for r in self.rep_history),
            'form_issues': self._aggregate_feedback()
        }

    def _aggregate_feedback(self) -> dict:
        """Count frequency of form issues"""
        all_feedback = []
        for rep in self.rep_history:
            all_feedback.extend(rep.feedback)

        from collections import Counter
        return dict(Counter(all_feedback))


# =============================================================================
# SECTION 5: VISUAL FEEDBACK
# =============================================================================

class FormVisualizer:
    """
    Render form feedback and metrics on video frame.

    Design principles:
    - Critical info (rep count) is large and always visible
    - Form feedback appears briefly when issues detected
    - Angle values shown near relevant joints
    - Color coding: green=good, yellow=warning, red=error
    """

    COLORS = {
        'good': (0, 255, 0),  # Green
        'warning': (0, 255, 255),  # Yellow
        'error': (0, 0, 255),  # Red
        'info': (255, 255, 255),  # White
        'accent': (255, 165, 0),  # Orange
    }

    def __init__(self):
        self.feedback_display_time = 2.0  # Seconds to show feedback
        self.active_feedback = []  # (message, expire_time, severity)

    def add_feedback(self, feedback: FormFeedback):
        """Add feedback to display queue"""
        expire_time = time.time() + self.feedback_display_time
        self.active_feedback.append((feedback.message, expire_time, feedback.severity))

    def render(self, frame: np.ndarray,
               angles: Optional[JointAngles],
               analyzer: ExerciseAnalyzer,
               detector: PoseDetector) -> np.ndarray:
        """
        Render all visual elements on frame.

        Args:
            frame: Video frame to annotate
            angles: Current joint angles
            analyzer: Exercise analyzer for stats
            detector: Pose detector for landmark positions

        Returns:
            Annotated frame
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Clean up expired feedback
        current_time = time.time()
        self.active_feedback = [
            f for f in self.active_feedback if f[1] > current_time
        ]

        # === Rep Counter (top center, large) ===
        rep_text = f"{analyzer.rep_count}"
        font_scale = 3.0
        thickness = 5

        (text_w, text_h), _ = cv2.getTextSize(
            rep_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background box
        cv2.rectangle(output,
                      (w // 2 - text_w // 2 - 20, 10),
                      (w // 2 + text_w // 2 + 20, text_h + 40),
                      (0, 0, 0), -1)
        cv2.rectangle(output,
                      (w // 2 - text_w // 2 - 20, 10),
                      (w // 2 + text_w // 2 + 20, text_h + 40),
                      self.COLORS['accent'], 2)

        # Rep number
        cv2.putText(output, rep_text,
                    (w // 2 - text_w // 2, text_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    self.COLORS['good'], thickness)

        # === Exercise name ===
        cv2.putText(output, analyzer.exercise_name.upper(),
                    (w // 2 - 50, text_h + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.COLORS['info'], 2)

        # === Phase indicator ===
        phase_colors = {
            ExercisePhase.NEUTRAL: self.COLORS['info'],
            ExercisePhase.ECCENTRIC: self.COLORS['warning'],
            ExercisePhase.CONCENTRIC: self.COLORS['good']
        }
        phase_text = analyzer.current_phase.name
        cv2.putText(output, phase_text,
                    (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    phase_colors[analyzer.current_phase], 2)

        # === Joint angles (near joints) ===
        if angles and detector.landmarks:
            # Show knee angle near knees
            left_knee_pos = detector.get_landmark_pixel(PoseDetector.LEFT_KNEE)
            if left_knee_pos:
                cv2.putText(output, f"{angles.left_knee:.0f}°",
                            (left_knee_pos[0] - 40, left_knee_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            self.COLORS['info'], 2)

            right_knee_pos = detector.get_landmark_pixel(PoseDetector.RIGHT_KNEE)
            if right_knee_pos:
                cv2.putText(output, f"{angles.right_knee:.0f}°",
                            (right_knee_pos[0] + 10, right_knee_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            self.COLORS['info'], 2)

            # Spine angle indicator
            spine_color = self.COLORS['good'] if angles.spine_angle < 25 else self.COLORS['warning']
            cv2.putText(output, f"Back: {angles.spine_angle:.0f}°",
                        (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        spine_color, 2)

        # === Form feedback (right side) ===
        y_offset = 100
        for message, _, severity in self.active_feedback:
            color = self.COLORS.get(severity, self.COLORS['info'])
            cv2.putText(output, message,
                        (w - 300, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)
            y_offset += 35

        # === Form score (if reps completed) ===
        if analyzer.rep_history:
            last_score = analyzer.rep_history[-1].form_score
            score_color = (
                self.COLORS['good'] if last_score >= 80 else
                self.COLORS['warning'] if last_score >= 60 else
                self.COLORS['error']
            )
            cv2.putText(output, f"Form: {last_score:.0f}%",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        score_color, 2)

        return output


# =============================================================================
# SECTION 6: MAIN APPLICATION
# =============================================================================

class WorkoutSession:
    """
    Main orchestrator for the workout analysis session.

    Coordinates all components and handles:
    - Video capture
    - Frame processing pipeline
    - User input
    - Session recording and export
    """

    def __init__(self, exercise: str = 'squat', camera_id: int = 0):
        """
        Args:
            exercise: Exercise to track
            camera_id: Camera device ID (0 = default webcam)
        """
        self.exercise = exercise
        self.camera_id = camera_id

        # Initialize components
        self.detector = PoseDetector()
        self.analyzer = ExerciseAnalyzer(exercise)
        self.visualizer = FormVisualizer()

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.session_start = None

    def run(self):
        """
        Main loop - process video frames until quit.

        Controls:
            q - Quit
            r - Reset rep counter
            s - Save screenshot
            1/2/3 - Switch exercise
        """
        print()
        print("=" * 70)
        print(f"  WORKOUT FORM ANALYZER - {self.exercise.upper()}")
        print("=" * 70)
        print()
        print("  Controls:")
        print("    q - Quit and show summary")
        print("    r - Reset rep counter")
        print("    s - Save screenshot")
        print("    1 - Switch to Squat")
        print("    2 - Switch to Bicep Curl")
        print("    3 - Switch to Shoulder Press")
        print()
        print("  Starting camera...")
        print()

        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.session_start = time.time()

        try:
            while True:
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame")
                    break

                # Mirror for more intuitive feedback
                frame = cv2.flip(frame, 1)

                # === Processing Pipeline ===

                # 1. Detect pose
                pose_found = self.detector.detect(frame)

                if pose_found:
                    # 2. Calculate angles
                    angles = AngleCalculator.compute_all_angles(self.detector)

                    if angles:
                        # 3. Analyze exercise
                        completed_rep = self.analyzer.update(angles)

                        if completed_rep:
                            print(f"  Rep {completed_rep} completed! "
                                  f"Score: {self.analyzer.rep_history[-1].form_score:.0f}%")

                        # 4. Check form
                        feedback = self.analyzer.check_form(angles)
                        for fb in feedback:
                            if not fb.is_good:
                                self.visualizer.add_feedback(fb)
                    else:
                        angles = None
                else:
                    angles = None

                # 5. Draw skeleton
                annotated = self.detector.draw_landmarks(frame)

                # 6. Render UI
                output = self.visualizer.render(
                    annotated, angles, self.analyzer, self.detector
                )

                # 7. FPS counter
                fps = 1.0 / (time.time() - frame_start + 1e-6)
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)

                cv2.putText(output, f"FPS: {avg_fps:.1f}",
                            (output.shape[1] - 120, output.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)

                # Display
                cv2.imshow('Workout Form Analyzer', output)

                # Handle input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.analyzer = ExerciseAnalyzer(self.exercise)
                    print("  Rep counter reset!")
                elif key == ord('s'):
                    filename = f"workout_{int(time.time())}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"  Screenshot saved: {filename}")
                elif key == ord('1'):
                    self._switch_exercise('squat')
                elif key == ord('2'):
                    self._switch_exercise('bicep_curl')
                elif key == ord('3'):
                    self._switch_exercise('shoulder_press')

        finally:
            cap.release()
            cv2.destroyAllWindows()

            self._print_summary()

    def _switch_exercise(self, exercise: str):
        """Switch to different exercise"""
        self.exercise = exercise
        self.analyzer = ExerciseAnalyzer(exercise)
        print(f"  Switched to: {exercise.upper()}")

    def _print_summary(self):
        """Print session summary"""
        duration = time.time() - self.session_start
        stats = self.analyzer.get_stats()

        print()
        print("=" * 70)
        print("  SESSION SUMMARY")
        print("=" * 70)
        print()
        print(f"  Exercise:         {self.exercise.upper()}")
        print(f"  Duration:         {duration / 60:.1f} minutes")
        print(f"  Total Reps:       {stats['total_reps']}")

        if stats['total_reps'] > 0:
            print(f"  Avg Form Score:   {stats['avg_form_score']:.1f}%")
            print(f"  Avg Rep Time:     {stats['avg_rep_duration']:.1f}s")
            print(f"  Best Depth:       {stats['best_depth']:.0f}°")

            if stats['form_issues']:
                print()
                print("  Form Issues:")
                for issue, count in stats['form_issues'].items():
                    print(f"    - {issue}: {count} times")

        print()
        print("=" * 70)


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

def main():
    """
    Entry point - run the workout analyzer.

    Usage:
        python workout_form_analyzer.py
        python workout_form_analyzer.py --exercise bicep_curl
    """
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Workout Form Analyzer')
    parser.add_argument('--exercise', '-e',
                        choices=['squat', 'bicep_curl', 'shoulder_press'],
                        default='squat',
                        help='Exercise to track (default: squat)')
    parser.add_argument('--camera', '-c',
                        type=int, default=0,
                        help='Camera device ID (default: 0)')

    args = parser.parse_args()

    session = WorkoutSession(exercise=args.exercise, camera_id=args.camera)
    session.run()


if __name__ == '__main__':
    main()