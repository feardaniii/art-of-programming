import cv2
import mediapipe as mp
import numpy as np





# ====== HEALTH APP: Workout Form Analyzer ======
class WorkoutFormAnalyzer:
    """Real-time workout form analysis using MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.squat_count = 0
        self.prev_state = None

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (landmarks)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze_squat_form(self, landmarks):
        """Analyze squat form quality"""
        # Get key landmarks for both sides (for better detection from any angle)
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles for both legs
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Use average for more stable detection
        knee_angle = (left_knee_angle + right_knee_angle) / 2

        # More forgiving thresholds for squat detection
        # Standing: knee angle > 150 degrees
        # Squatting: knee angle < 120 degrees
        if knee_angle > 150:
            state = 'up'
        elif knee_angle < 120:
            state = 'down'
        else:
            state = 'middle'

        # Count rep - only count when going from down to up
        if self.prev_state == 'down' and state == 'up':
            self.squat_count += 1

        self.prev_state = state

        # Form feedback
        feedback = []
        if state == 'down':
            if knee_angle > 110:
                feedback.append("⚠️ Try going deeper!")
            else:
                feedback.append("✅ Good depth")

            # Check knee position (only if front-facing)
            if abs(left_knee[0] - right_knee[0]) > 0.1:  # Side view
                feedback.append("👁️ Side view detected")
            else:  # Front view
                if left_knee[0] > left_ankle[0] + 0.05:
                    feedback.append("⚠️ Keep knees aligned")
                else:
                    feedback.append("✅ Good knee position")
        elif state == 'up':
            feedback.append("⬆️ Standing - Ready!")
        else:
            feedback.append("⬇️ Going down...")

        return {
            'count': self.squat_count,
            'knee_angle': knee_angle,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'state': state,
            'feedback': feedback
        }
    




    

# ====== HAND TRACKING: Finance Receipt Scanner ======
class ReceiptScanner:
    """Use hand tracking to guide receipt photo capture"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

    def detect_pinch_gesture(self, landmarks):
        """Detect pinch gesture (thumb + index finger close)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        return distance < 0.05

# ====== FACE MESH: Emotion & Attention Tracker ======
class FaceAnalyzer:
    """Analyze facial features for emotion and attention"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate eye aspect ratio (EAR) for blink detection"""
        # Vertical eye landmarks
        A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                           np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
        B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                           np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
        # Horizontal eye landmarks
        C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                           np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))

        ear = (A + B) / (2.0 * C)
        return ear

    def analyze_face(self, landmarks):
        """Analyze facial features"""
        # Left eye indices (simplified)
        left_eye = [33, 160, 158, 133, 153, 144]

        try:
            ear = self.calculate_eye_aspect_ratio(landmarks, left_eye)

            # Attention metrics
            if ear < 0.2:
                attention = "😴 Drowsy/Tired"
            elif ear < 0.25:
                attention = "😑 Low attention"
            else:
                attention = "👀 Alert & Focused"

            return {
                'eye_aspect_ratio': ear,
                'attention': attention
            }
        except:
            return {'attention': '🔍 Analyzing...'}


# ====== MAIN INTERACTIVE LEARNING EXPERIENCE ======
def main():
    import argparse

    print("\n" + "="*60)
    print("🚀 WHY MEDIAPIPE IS REVOLUTIONARY")
    print("="*60)
    print("   Building pose estimation from scratch:")
    print("     - Requires PhD-level ML expertise")
    print("     - Years of training data collection")
    print("     - Millions in compute resources")
    print("\n   Using MediaPipe:")
    print("     - 5 lines of code")
    print("     - Works on ANY device")
    print("     - Real-time performance (30+ FPS)")
    print("\n💡 This is the POWER of standing on giants' shoulders!")
    print("="*60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MediaPipe Interactive Learning Experience')
    parser.add_argument('-i', '--image', type=str, help='Path to image file (default: webcam)')
    parser.add_argument('-m', '--mode', type=str, default='pose',
                       choices=['pose', 'hands', 'face', 'all'],
                       help='Detection mode: pose, hands, face, or all')
    args = parser.parse_args()

    # Initialize detectors based on mode
    print(f"\n🎯 Mode: {args.mode.upper()}")

    workout_analyzer = None
    hand_tracker = None
    face_analyzer = None

    if args.mode in ['pose', 'all']:
        workout_analyzer = WorkoutFormAnalyzer()
        print("   ✅ Pose Detection Enabled - Try doing squats!")

    if args.mode in ['hands', 'all']:
        hand_tracker = ReceiptScanner()
        print("   ✅ Hand Tracking Enabled - Show your hands!")

    if args.mode in ['face', 'all']:
        face_analyzer = FaceAnalyzer()
        print("   ✅ Face Mesh Enabled - Show your face!")

    # Setup video source
    if args.image:
        print(f"\n📸 Image mode: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Error: Could not load image from {args.image}")
            return
        use_webcam = False
    else:
        print("\n📹 Webcam mode (Press 'q' to quit, 'c' to capture)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        use_webcam = True

    print("\n⌨️  CONTROLS:")
    print("   'q' - Quit")
    if use_webcam:
        print("   'c' - Capture current frame")
    print("   '1' - Pose detection only")
    print("   '2' - Hand tracking only")
    print("   '3' - Face mesh only")
    print("   '4' - All detections")

    current_mode = args.mode

    while True:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

        # Flip for mirror effect (webcam only)
        if use_webcam:
            frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Create info overlay
        info_y = 30
        cv2.putText(frame, f"Mode: {current_mode.upper()}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        info_y += 30

        # POSE DETECTION
        if current_mode in ['pose', 'all'] and workout_analyzer:
            results = workout_analyzer.pose.process(rgb_frame)

            if results.pose_landmarks:
                # Draw pose landmarks
                workout_analyzer.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    workout_analyzer.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=workout_analyzer.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=workout_analyzer.mp_draw.DrawingSpec(
                        color=(255, 255, 0), thickness=2)
                )

                # Analyze squat form
                analysis = workout_analyzer.analyze_squat_form(
                    results.pose_landmarks.landmark
                )

                # Display analysis with color-coded state
                state_color = {
                    'up': (0, 255, 0),      # Green
                    'down': (0, 165, 255),  # Orange
                    'middle': (0, 255, 255) # Yellow
                }
                current_color = state_color.get(analysis['state'], (255, 255, 255))

                cv2.putText(frame, f"SQUATS: {analysis['count']}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                info_y += 40

                cv2.putText(frame, f"State: {analysis['state'].upper()}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
                info_y += 30

                cv2.putText(frame, f"Avg Knee: {int(analysis['knee_angle'])}°",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 25
                cv2.putText(frame, f"L: {int(analysis['left_knee_angle'])}° | R: {int(analysis['right_knee_angle'])}°",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                info_y += 25

                # Threshold guide
                cv2.putText(frame, "Thresholds: >150°=UP | <120°=DOWN",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                info_y += 20

                for feedback in analysis['feedback']:
                    cv2.putText(frame, feedback, (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    info_y += 25

        # HAND TRACKING
        if current_mode in ['hands', 'all'] and hand_tracker:
            results = hand_tracker.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(250, 44, 250), thickness=2)
                    )

                    # Detect pinch gesture
                    is_pinching = hand_tracker.detect_pinch_gesture(
                        hand_landmarks.landmark
                    )

                    # Get hand label
                    hand_label = results.multi_handedness[idx].classification[0].label

                    # Display hand info
                    cv2.putText(frame, f"{hand_label} Hand", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 44, 250), 2)
                    info_y += 25

                    if is_pinching:
                        cv2.putText(frame, "🤏 PINCH DETECTED!", (10, info_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        info_y += 30

        # FACE MESH
        if current_mode in ['face', 'all'] and face_analyzer:
            results = face_analyzer.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    face_analyzer.mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=1)
                    )

                    # Analyze attention
                    analysis = face_analyzer.analyze_face(face_landmarks.landmark)

                    cv2.putText(frame, f"Attention: {analysis['attention']}",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    info_y += 25

                    if 'eye_aspect_ratio' in analysis:
                        cv2.putText(frame, f"EAR: {analysis['eye_aspect_ratio']:.3f}",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        info_y += 25

        # Display FPS for webcam
        if use_webcam:
            cv2.putText(frame, "Press 'q' to quit | 1-4 to change mode",
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow('MediaPipe Learning Experience', frame)

        # Handle keyboard input
        key = cv2.waitKey(1 if use_webcam else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and use_webcam:
            filename = f'mediapipe_capture_{current_mode}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        elif key == ord('1'):
            current_mode = 'pose'
            print("🎯 Switched to: POSE detection")
        elif key == ord('2'):
            current_mode = 'hands'
            print("🎯 Switched to: HAND tracking")
        elif key == ord('3'):
            current_mode = 'face'
            print("🎯 Switched to: FACE mesh")
        elif key == ord('4'):
            current_mode = 'all'
            print("🎯 Switched to: ALL detections")

        # For image mode, wait for key press
        if not use_webcam:
            break

    # Cleanup
    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Session complete!")
    print("\n📚 WHAT YOU LEARNED:")
    print("   • MediaPipe Pose: Track 33 body landmarks in real-time")
    print("   • MediaPipe Hands: Detect 21 hand landmarks per hand")
    print("   • MediaPipe Face Mesh: Map 468 facial landmarks")
    print("   • All running at 30+ FPS on your local machine!")
    print("\n🎯 REAL-WORLD APPLICATIONS:")
    print("   • Fitness apps (form analysis, rep counting)")
    print("   • Sign language translation")
    print("   • Attention/drowsiness detection")
    print("   • AR filters and effects")
    print("   • Gesture-based UI controls")


if __name__ == "__main__":
    main()