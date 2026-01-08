import dlib
import cv2
import numpy as np
from collections import deque
from model_downloader import ensure_models

# ====== Load Required Models ======
print("🔍 Checking for required models...")
models = ensure_models('shape_predictor_68_face_landmarks.dat')

# ====== Dlib's Correlation Tracker ======
# Efficient tracking algorithm that follows objects between frames
# Much faster than detecting every frame!

class FaceTracker:
    """
    Real-time face tracking system
    Use: Video conferencing, surveillance, interactive installations
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.trackers = []
        self.face_ids = []
        self.next_face_id = 0

        # Performance optimization
        self.detect_every_n_frames = 10  # Detect new faces every N frames
        self.frame_count = 0

    def start_tracking_face(self, frame, face_rect):
        """Initialize tracker for a detected face"""
        tracker = dlib.correlation_tracker()
        tracker.start_track(frame, face_rect)

        self.trackers.append(tracker)
        self.face_ids.append(self.next_face_id)
        self.next_face_id += 1

        return self.next_face_id - 1

    def track_faces(self, video_path):
        """Track all faces in a video"""
        cap = cv2.VideoCapture(video_path)

        tracking_data = {
            'face_positions': {},  # {face_id: [(x, y, w, h, frame_num), ...]}
            'face_durations': {}   # {face_id: num_frames}
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_count += 1

            # Update existing trackers
            active_trackers = []
            active_ids = []

            for tracker, face_id in zip(self.trackers, self.face_ids):
                tracker.update(rgb)
                pos = tracker.get_position()

                # Convert to integer coordinates
                x = int(pos.left())
                y = int(pos.top())
                w = int(pos.width())
                h = int(pos.height())

                # Check if tracking is still valid (face not left frame)
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    active_trackers.append(tracker)
                    active_ids.append(face_id)

                    # Record position
                    if face_id not in tracking_data['face_positions']:
                        tracking_data['face_positions'][face_id] = []
                        tracking_data['face_durations'][face_id] = 0

                    tracking_data['face_positions'][face_id].append((x, y, w, h, self.frame_count))
                    tracking_data['face_durations'][face_id] += 1

                    # Draw tracking box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {face_id}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.trackers = active_trackers
            self.face_ids = active_ids

            # Detect new faces periodically
            if self.frame_count % self.detect_every_n_frames == 0:
                faces = self.detector(rgb, 0)  # No upsampling for speed

                for face_rect in faces:
                    # Check if this is a new face (not already tracked)
                    is_new = True
                    for tracker in self.trackers:
                        tracked_pos = tracker.get_position()

                        # Calculate overlap
                        overlap = self.calculate_overlap(face_rect, tracked_pos)
                        if overlap > 0.5:  # Already tracking this face
                            is_new = False
                            break

                    if is_new:
                        face_id = self.start_tracking_face(rgb, face_rect)
                        print(f"🆕 New face detected: ID {face_id}")

            # Display frame
            cv2.imshow('Face Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return tracking_data

    def calculate_overlap(self, rect1, rect2):
        """Calculate IoU (Intersection over Union) between two rectangles"""
        x1_1, y1_1 = rect1.left(), rect1.top()
        x2_1, y2_1 = rect1.right(), rect1.bottom()

        x1_2, y1_2 = int(rect2.left()), int(rect2.top())
        x2_2, y2_2 = int(rect2.right()), int(rect2.bottom())

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

# ====== Health Application: Posture Monitoring ======
class PostureMonitor:
    """
    Monitor head position to detect poor posture
    Use: Ergonomics, productivity, health tracking
    """

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Calibration (baseline good posture)
        self.baseline_nose_y = None
        self.posture_history = deque(maxlen=100)

    def calibrate(self, frame):
        """Calibrate baseline posture (user sits properly)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 1)

        if len(faces) > 0:
            landmarks = self.predictor(rgb, faces[0])
            nose_tip = (landmarks.part(33).x, landmarks.part(33).y)
            self.baseline_nose_y = nose_tip[1]
            print(f"✅ Posture calibrated at Y: {self.baseline_nose_y}")
            return True
        return False

    def monitor_posture(self, video_source=0):
        """Real-time posture monitoring"""
        cap = cv2.VideoCapture(video_source)

        print("\n🪑 Posture Monitor Active")
        print("   Press 'c' to calibrate good posture")
        print("   Press 'q' to quit")

        poor_posture_alert_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb, 0)

            if len(faces) > 0 and self.baseline_nose_y is not None:
                landmarks = self.predictor(rgb, faces[0])
                nose_tip = (landmarks.part(33).x, landmarks.part(33).y)

                # Calculate deviation from baseline
                y_deviation = nose_tip[1] - self.baseline_nose_y

                # Determine posture quality
                posture_status = "Good"
                color = (0, 255, 0)

                if y_deviation > 50:  # Head too low (slouching)
                    posture_status = "⚠️ SLOUCHING"
                    color = (0, 165, 255)  # Orange
                    poor_posture_alert_frames += 1
                elif y_deviation < -30:  # Head too high (straining)
                    posture_status = "⚠️ HEAD HIGH"
                    color = (0, 255, 255)  # Yellow
                else:
                    poor_posture_alert_frames = 0

                # Alert after sustained poor posture
                if poor_posture_alert_frames > 150:  # ~5 seconds at 30fps
                    cv2.putText(frame, "TAKE A BREAK!", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Draw visualization
                cv2.circle(frame, nose_tip, 5, color, -1)
                cv2.putText(frame, f"Posture: {posture_status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Deviation: {y_deviation:.0f}px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Posture Monitor', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate(frame)

        cap.release()
        cv2.destroyAllWindows()

# ====== Finance Application: Focus Time Tracking ======
class FocusTimeTracker:
    """
    Track time spent focused on screen for productivity billing
    Use: Freelance time tracking, productivity consulting, work-from-home monitoring
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.focused_time = 0
        self.away_time = 0
        self.last_detection_time = None

    def track_focus_session(self, duration_minutes=25):  # Pomodoro timer
        """Track a focus session (Pomodoro technique)"""
        cap = cv2.VideoCapture(0)

        import time
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        print(f"\n⏱️ Focus Session Started: {duration_minutes} minutes")

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed = current_time - start_time
            remaining = end_time - current_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb, 0)

            # Check if person is present
            if len(faces) > 0:
                self.focused_time += 1
                status = "✅ FOCUSED"
                color = (0, 255, 0)
            else:
                self.away_time += 1
                status = "⚠️ AWAY"
                color = (0, 0, 255)

            # Calculate focus percentage
            total_frames = self.focused_time + self.away_time
            focus_percentage = (self.focused_time / total_frames * 100) if total_frames > 0 else 0

            # Display info
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Focus: {focus_percentage:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {int(remaining)}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Focus Time Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Generate report
        total_time = self.focused_time + self.away_time
        focus_rate = (self.focused_time / total_time * 100) if total_time > 0 else 0

        report = {
            'duration_minutes': duration_minutes,
            'focus_percentage': focus_rate,
            'billable_time': (focus_rate / 100) * duration_minutes,
            'quality_score': 'Excellent' if focus_rate > 85 else 'Good' if focus_rate > 70 else 'Needs Improvement'
        }

        print(f"\n📊 Focus Session Report:")
        print(f"   Focus Percentage: {focus_rate:.1f}%")
        print(f"   Billable Time: {report['billable_time']:.1f} minutes")
        print(f"   Quality: {report['quality_score']}")

        return report

print("\n💡 Tracking = Detection + Memory")
print("   Detection finds faces → Tracking follows them")
print("   Much faster than detecting every frame!")
print("\n🎯 Real-time applications demand EFFICIENCY!")

# ====== RUNNABLE DEMO ======
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Real-time Face Tracking")
    print("="*60)
    print("\nChoose a demo:")
    print("1. Real-time Face Tracking (live video with face tracking)")
    print("2. Posture Monitor (monitors your head position)")
    print("3. Focus Time Tracker (track productivity focus time)")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        print("\n" + "-"*60)
        print("Real-time Face Tracking Demo")
        print("-"*60)
        print("\n📹 Starting live face tracking...")
        print("The system will:")
        print("  - Detect faces in the video stream")
        print("  - Assign unique IDs to each face")
        print("  - Track faces as they move")
        print("\nPress 'q' to quit\n")

        tracker = FaceTracker()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            exit(1)

        frame_count = 0
        print("✅ Tracking started! Move around to see the tracking in action.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # Update existing trackers
            active_trackers = []
            active_ids = []

            for track, face_id in zip(tracker.trackers, tracker.face_ids):
                track.update(rgb)
                pos = track.get_position()

                x = int(pos.left())
                y = int(pos.top())
                w = int(pos.width())
                h = int(pos.height())

                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    active_trackers.append(track)
                    active_ids.append(face_id)

                    # Draw tracking box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {face_id}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            tracker.trackers = active_trackers
            tracker.face_ids = active_ids

            # Detect new faces periodically
            if frame_count % tracker.detect_every_n_frames == 0:
                faces = tracker.detector(rgb, 0)

                for face_rect in faces:
                    is_new = True
                    for track in tracker.trackers:
                        tracked_pos = track.get_position()
                        overlap = tracker.calculate_overlap(face_rect, tracked_pos)
                        if overlap > 0.5:
                            is_new = False
                            break

                    if is_new:
                        face_id = tracker.start_tracking_face(rgb, face_rect)
                        print(f"🆕 New face detected: ID {face_id}")

            # Display info
            cv2.putText(frame, f"Tracking: {len(tracker.trackers)} face(s)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Real-time Face Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n✅ Tracking demo complete!")
        print(f"   Total faces tracked: {tracker.next_face_id}")

    elif choice == "2":
        print("\n" + "-"*60)
        print("Posture Monitor Demo")
        print("-"*60)
        print("\n🪑 Starting posture monitor...")
        print("Instructions:")
        print("  1. Press 'c' to calibrate good posture")
        print("  2. The system will monitor your head position")
        print("  3. You'll get alerts if you slouch or strain")
        print("\nPress 'q' to quit\n")

        monitor = PostureMonitor(models['shape_predictor_68_face_landmarks.dat'])
        monitor.monitor_posture(0)

        print("\n✅ Posture monitoring complete!")

    elif choice == "3":
        print("\n" + "-"*60)
        print("Focus Time Tracker Demo")
        print("-"*60)

        duration = input("\nEnter session duration in minutes (default 1): ").strip()
        try:
            duration = int(duration) if duration else 1
        except ValueError:
            duration = 1

        print(f"\n⏱️ Starting {duration}-minute focus session...")
        print("The system will:")
        print("  - Track when you're present at your desk")
        print("  - Calculate your focus percentage")
        print("  - Generate a productivity report")
        print("\nPress 'q' to quit early\n")

        tracker = FocusTimeTracker()
        report = tracker.track_focus_session(duration_minutes=duration)

        print("\n✅ Focus session complete!")

    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)