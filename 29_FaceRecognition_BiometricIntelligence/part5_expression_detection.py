import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from model_downloader import ensure_models

# ====== Load Required Models ======
print("🔍 Checking for required models...")
models = ensure_models('shape_predictor_68_face_landmarks.dat')




# ====== Advanced Landmark Analysis ======
class ExpressionDetector:
    """
    Detect facial expressions using landmark geometry
    Use: Emotion AI, customer satisfaction, mental health monitoring
    """

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR) for smile/expression detection"""
        # Vertical distances
        A = dist.euclidean(mouth_points[2], mouth_points[10])  # 51 to 59
        B = dist.euclidean(mouth_points[4], mouth_points[8])   # 53 to 57

        # Horizontal distance
        C = dist.euclidean(mouth_points[0], mouth_points[6])   # 49 to 55

        mar = (A + B) / (2.0 * C)
        return mar

    def eyebrow_height(self, eyebrow_points, eye_points):
        """Measure eyebrow elevation (surprise, focus)"""
        eyebrow_center = np.mean(eyebrow_points, axis=0)
        eye_center = np.mean(eye_points, axis=0)

        height = eyebrow_center[1] - eye_center[1]
        return height

    def detect_expression(self, frame):
        """Detect facial expression in real-time"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        if len(faces) == 0:
            return frame, "No face detected"

        landmarks = self.predictor(rgb, faces[0])

        # Extract landmark groups
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        left_eyebrow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)])
        right_eyebrow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)])

        # Calculate metrics
        mar = self.mouth_aspect_ratio(mouth)
        left_brow_height = abs(self.eyebrow_height(left_eyebrow, left_eye))
        right_brow_height = abs(self.eyebrow_height(right_eyebrow, right_eye))
        avg_brow_height = (left_brow_height + right_brow_height) / 2

        # Classify expression
        expression = "Neutral"
        emoji = "😐"

        if mar > 0.4:
            expression = "Happy/Smiling"
            emoji = "😊"
        elif mar < 0.2 and avg_brow_height < 15:
            expression = "Sad/Frowning"
            emoji = "😔"
        elif avg_brow_height > 25:
            expression = "Surprised"
            emoji = "😮"

        # Draw visualization
        cv2.putText(frame, f"{emoji} {expression}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, expression

# ====== Head Pose Estimation ======
class HeadPoseEstimator:
    """
    Estimate head pose (pitch, yaw, roll) for gaze tracking
    Use: Driver attention monitoring, accessibility, AR/VR
    """

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # 3D model points of a generic face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_head_pose(self, frame):
        """Calculate head pose angles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        if len(faces) == 0:
            return frame, None

        landmarks = self.predictor(gray, faces[0])

        # 2D image points from landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
        ], dtype="double")

        # Camera internals
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert rotation vector to angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = euler_angles.flatten()[:3]

        # Interpret head pose
        direction = "Center"
        if yaw < -10:
            direction = "Looking Right"
        elif yaw > 10:
            direction = "Looking Left"

        if pitch < -10:
            direction += " / Looking Down"
        elif pitch > 10:
            direction += " / Looking Up"

        # Draw visualization
        cv2.putText(frame, f"Head: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, {'pitch': pitch, 'yaw': yaw, 'roll': roll, 'direction': direction}
















# ====== Gesture-Based Control ======
class GestureController:
    """
    Hands-free gesture control using head movements
    Use: Accessibility, cooking apps, workout apps, presentation control
    """

    def __init__(self, predictor_path):
        self.pose_estimator = HeadPoseEstimator(predictor_path)
        self.gesture_history = []

    def detect_gesture(self, frame):
        """Detect head gesture commands"""
        frame, pose_data = self.pose_estimator.get_head_pose(frame)

        if pose_data is None:
            return frame, None

        yaw = pose_data['yaw']
        pitch = pose_data['pitch']

        gesture = None

        # Define gesture thresholds
        if yaw < -20:
            gesture = "SWIPE_LEFT"
        elif yaw > 20:
            gesture = "SWIPE_RIGHT"
        elif pitch < -15:
            gesture = "SCROLL_DOWN"
        elif pitch > 15:
            gesture = "SCROLL_UP"

        if gesture:
            self.gesture_history.append(gesture)

            # Display gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame, gesture

# ====== Health Application: Mental Health Monitoring ======
class MoodTracker:
    """
    Track mood over time using expression detection
    Use: Mental health apps, wellness tracking, therapy support
    """

    def __init__(self, predictor_path):
        self.expression_detector = ExpressionDetector(predictor_path)
        self.mood_log = []

    def track_mood_session(self, duration_seconds=60):
        """Track mood during a session"""
        import time

        cap = cv2.VideoCapture(0)
        start_time = time.time()

        expression_counts = {
            'Happy/Smiling': 0,
            'Neutral': 0,
            'Sad/Frowning': 0,
            'Surprised': 0
        }

        print(f"\n😊 Mood Tracking Session: {duration_seconds}s")

        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break

            frame, expression = self.expression_detector.detect_expression(frame)

            if expression in expression_counts:
                expression_counts[expression] += 1

            cv2.imshow('Mood Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Calculate mood distribution
        total = sum(expression_counts.values())
        mood_distribution = {k: (v / total * 100) if total > 0 else 0
                            for k, v in expression_counts.items()}

        # Determine overall mood
        dominant_mood = max(mood_distribution, key=mood_distribution.get)
        mood_score = mood_distribution.get('Happy/Smiling', 0) - mood_distribution.get('Sad/Frowning', 0)

        report = {
            'dominant_mood': dominant_mood,
            'mood_score': mood_score,
            'distribution': mood_distribution,
            'recommendation': self.get_wellness_recommendation(mood_score)
        }

        print(f"\n📊 Mood Report:")
        print(f"   Dominant Mood: {dominant_mood}")
        print(f"   Mood Score: {mood_score:.1f}")
        print(f"   Recommendation: {report['recommendation']}")

        return report

    def get_wellness_recommendation(self, mood_score):
        """Generate wellness recommendations based on mood"""
        if mood_score > 30:
            return "✅ Great mood! Keep up the positive energy."
        elif mood_score > 0:
            return "😊 Good mood. Consider activities you enjoy."
        elif mood_score > -30:
            return "😐 Neutral mood. Try a short walk or listen to music."
        else:
            return "💙 Consider reaching out to friends or practicing self-care."

# ====== Finance Application: Gesture-Controlled Receipt Scanner ======
class GestureReceiptScanner:
    """
    Scan receipts using head nod gesture (hands full of shopping)
    Use: Expense tracking, budgeting apps, financial management
    """

    def __init__(self, predictor_path):
        self.gesture_controller = GestureController(predictor_path)
        self.capture_cooldown = 0

    def scan_with_gesture(self):
        """Gesture-controlled receipt scanning"""
        cap = cv2.VideoCapture(0)

        print("\n📸 Gesture-Controlled Receipt Scanner")
        print("   Nod DOWN to capture receipt")
        print("   Look LEFT/RIGHT to navigate")

        captured_images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, gesture = self.gesture_controller.detect_gesture(frame)

            if gesture == "SCROLL_DOWN" and self.capture_cooldown == 0:
                # Capture receipt
                captured_images.append(frame.copy())
                self.capture_cooldown = 30  # 1 second cooldown at 30fps

                cv2.putText(frame, "✅ CAPTURED!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                print(f"✅ Receipt captured! Total: {len(captured_images)}")

            if self.capture_cooldown > 0:
                self.capture_cooldown -= 1

            # Display instruction
            cv2.putText(frame, f"Receipts: {len(captured_images)}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Gesture Receipt Scanner', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return captured_images

print("\n💡 Expressions + Gestures = Human-Computer Interaction")
print("   Landmarks → Geometry → Expressions → Meaningful interaction")
print("   Head pose → Gestures → Hands-free control")
print("\n🎯 You're mastering the INTERFACE between humans and machines!")

# ====== RUNNABLE DEMO ======
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Expression Detection & Gesture Control")
    print("="*60)
    print("\nChoose a demo:")
    print("1. Real-time Expression Detection")
    print("2. Head Pose Estimation & Gaze Tracking")
    print("3. Gesture-Based Control")
    print("4. Mood Tracking Session")

    choice = input("\nEnter your choice (1-4): ").strip()

    predictor_path = models['shape_predictor_68_face_landmarks.dat']

    if choice == "1":
        print("\n" + "-"*60)
        print("Real-time Expression Detection Demo")
        print("-"*60)
        print("\n😊 Starting expression detector...")
        print("The system will detect:")
        print("  - Happy/Smiling")
        print("  - Neutral")
        print("  - Sad/Frowning")
        print("  - Surprised")
        print("\nPress 'q' to quit\n")

        detector = ExpressionDetector(predictor_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            exit(1)

        print("✅ Expression detection started! Try different expressions.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, expression = detector.detect_expression(frame)

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Expression Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n✅ Expression detection demo complete!")

    elif choice == "2":
        print("\n" + "-"*60)
        print("Head Pose Estimation Demo")
        print("-"*60)
        print("\n🎯 Starting head pose estimator...")
        print("The system will track:")
        print("  - Pitch (up/down)")
        print("  - Yaw (left/right)")
        print("  - Roll (tilt)")
        print("\nPress 'q' to quit\n")

        estimator = HeadPoseEstimator(predictor_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            exit(1)

        print("✅ Head pose estimation started! Move your head around.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, pose_data = estimator.get_head_pose(frame)

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Head Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n✅ Head pose estimation demo complete!")

    elif choice == "3":
        print("\n" + "-"*60)
        print("Gesture-Based Control Demo")
        print("-"*60)
        print("\n👋 Starting gesture controller...")
        print("Control gestures:")
        print("  - Look LEFT → SWIPE_LEFT")
        print("  - Look RIGHT → SWIPE_RIGHT")
        print("  - Look UP → SCROLL_UP")
        print("  - Look DOWN → SCROLL_DOWN")
        print("\nPress 'q' to quit\n")

        controller = GestureController(predictor_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            exit(1)

        print("✅ Gesture control started! Use head movements to control.")

        gesture_counts = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, gesture = controller.detect_gesture(frame)

            if gesture:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

            # Display gesture counts
            y_offset = 120
            cv2.putText(frame, "Gesture History:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for g, count in gesture_counts.items():
                y_offset += 25
                cv2.putText(frame, f"{g}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Gesture Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n✅ Gesture control demo complete!")
        print("\n📊 Gesture Summary:")
        for g, count in gesture_counts.items():
            print(f"   {g}: {count} times")

    elif choice == "4":
        print("\n" + "-"*60)
        print("Mood Tracking Session Demo")
        print("-"*60)

        duration = input("\nEnter session duration in seconds (default 30): ").strip()
        try:
            duration = int(duration) if duration else 30
        except ValueError:
            duration = 30

        print(f"\n😊 Starting {duration}-second mood tracking session...")
        print("The system will:")
        print("  - Monitor your facial expressions")
        print("  - Calculate mood distribution")
        print("  - Provide wellness recommendations")
        print("\nPress 'q' to quit early\n")

        tracker = MoodTracker(predictor_path)
        report = tracker.track_mood_session(duration_seconds=duration)

        print("\n✅ Mood tracking complete!")
        print("\n📊 Mood Distribution:")
        for mood, percentage in report['distribution'].items():
            print(f"   {mood}: {percentage:.1f}%")

    else:
        print("❌ Invalid choice. Please run again and select 1, 2, 3, or 4.")

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)