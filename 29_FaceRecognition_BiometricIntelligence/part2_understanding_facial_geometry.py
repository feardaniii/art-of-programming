import dlib
import cv2
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from model_downloader import ensure_models

# ====== Load the 68-point landmark predictor ======
# Auto-download model if not present
print("🔍 Checking for required models...")
models = ensure_models('shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(models['shape_predictor_68_face_landmarks.dat'])
detector = dlib.get_frontal_face_detector()

# ====== Understanding the 68 Landmarks ======
FACIAL_LANDMARKS_68 = OrderedDict([
    ("jaw", (0, 17)),           # Jawline
    ("right_eyebrow", (17, 22)), # Right eyebrow
    ("left_eyebrow", (22, 27)),  # Left eyebrow
    ("nose_bridge", (27, 31)),   # Nose bridge
    ("nose_tip", (31, 36)),      # Nose tip
    ("right_eye", (36, 42)),     # Right eye
    ("left_eye", (42, 48)),      # Left eye
    ("mouth_outer", (48, 60)),   # Outer mouth
    ("mouth_inner", (60, 68))    # Inner mouth
])

def extract_facial_landmarks(image_path):
    """
    Extract 68 facial landmarks from image
    Each landmark is an (x, y) coordinate
    """
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb, 1)

    if len(faces) == 0:
        print("❌ No faces detected")
        return None

    # Get landmarks for first face
    face_rect = faces[0]
    landmarks = predictor(rgb, face_rect)

    # Convert to NumPy array (68 points, each with x,y)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    print(f"\n🎯 Extracted 68 facial landmarks")
    print(f"   Shape: {coords.shape}")
    print(f"   Left eye center: {coords[36:42].mean(axis=0).astype(int)}")
    print(f"   Right eye center: {coords[42:48].mean(axis=0).astype(int)}")
    print(f"   Nose tip: {coords[33]}")
    print(f"   Mouth center: {coords[48:68].mean(axis=0).astype(int)}")

    return coords, rgb, face_rect

def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    """Visualize the 68 landmarks on image"""
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, color, -1)
    return image

# ====== Health App: Fatigue Detection ======
class FatigueDetector:
    """
    Detect fatigue using Eye Aspect Ratio (EAR)
    Use: Driver safety, workout safety, productivity monitoring
    """

    def __init__(self):
        self.detector = detector
        self.predictor = predictor
        self.EAR_THRESHOLD = 0.25  # Below this = eyes closing
        self.CONSECUTIVE_FRAMES = 20  # Frames before alert
        self.blink_counter = 0

    def eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Where p1-p6 are eye landmark points
        """
        # Vertical eye distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal eye distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def detect_fatigue(self, frame):
        """Analyze frame for signs of fatigue"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        if len(faces) == 0:
            return frame, "No face detected"

        landmarks = self.predictor(rgb, faces[0])

        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])

        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check for fatigue
        status = "Alert"
        color = (0, 255, 0)

        if avg_ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
            if self.blink_counter >= self.CONSECUTIVE_FRAMES:
                status = "⚠️ FATIGUE DETECTED"
                color = (0, 0, 255)
        else:
            self.blink_counter = 0

        # Draw visualization
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, status

# ====== Finance App: Document Signing Verification ======
def verify_face_orientation(landmarks):
    """
    Verify person is looking at camera (for secure document signing)
    Use: Banking apps, legal document signing, identity verification
    """
    # Calculate face orientation using nose and eye positions
    nose_tip = landmarks[33]
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    # Calculate angles
    eye_center = (left_eye + right_eye) / 2
    nose_to_eye_dist = np.linalg.norm(nose_tip - eye_center)

    # Check horizontal symmetry (is face centered?)
    left_dist = np.linalg.norm(nose_tip - left_eye)
    right_dist = np.linalg.norm(nose_tip - right_eye)
    symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)

    is_frontal = symmetry_ratio > 0.85  # Face is looking at camera

    verification_result = {
        'is_frontal_view': is_frontal,
        'symmetry_score': symmetry_ratio,
        'verification_status': '✅ Verified' if is_frontal else '❌ Look at camera',
        'ready_for_signing': is_frontal
    }

    print(f"\n📄 Document Signing Verification:")
    print(f"   Symmetry Score: {symmetry_ratio:.2f}")
    print(f"   Status: {verification_result['verification_status']}")

    return verification_result

# ====== Emotion Detection (Basic) ======
def detect_smile(landmarks):
    """
    Simple smile detection using mouth aspect ratio
    Use: Mood tracking, customer satisfaction, photo capture timing
    """
    mouth_outer = landmarks[48:60]

    # Mouth width (horizontal distance)
    mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])

    # Mouth height (vertical distance)
    mouth_height = np.linalg.norm(mouth_outer[3] - mouth_outer[9])

    # Smile ratio
    smile_ratio = mouth_width / mouth_height

    is_smiling = smile_ratio > 3.0  # Threshold for smile

    return {
        'is_smiling': is_smiling,
        'smile_intensity': min(100, (smile_ratio - 2.0) * 50),  # 0-100 scale
        'mood': '😊 Happy' if is_smiling else '😐 Neutral'
    }

print("\n💡 68 Landmarks = The Language of Faces")
print("   Eyes → Fatigue detection, gaze tracking")
print("   Mouth → Emotion, speech detection")
print("   Jawline → Face shape, gender classification")
print("   Nose → Face orientation, 3D pose estimation")
print("\n🎯 Master landmarks, unlock infinite applications!")

# ====== RUNNABLE DEMO ======
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Facial Landmarks Detection with Webcam Snapshot")
    print("="*60)

    # Initialize webcam
    print("\n📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("   Please ensure your webcam is connected and not in use by another application")
        exit(1)

    print("✅ Webcam ready!")
    print("\nPress SPACE to capture image and detect landmarks")
    print("Press 'q' to quit\n")

    captured = False

    while not captured:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from webcam")
            break

        # Display live feed
        cv2.putText(frame, "Press SPACE to capture", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Facial Landmarks Demo - Press SPACE', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space bar
            # Save captured image
            snapshot_path = 'landmarks_snapshot.jpg'
            cv2.imwrite(snapshot_path, frame)
            print(f"📸 Snapshot saved: {snapshot_path}")
            captured = True

        elif key == ord('q'):
            print("Demo cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # Extract landmarks from captured image
    print("\n🔍 Extracting facial landmarks...")
    result = extract_facial_landmarks('landmarks_snapshot.jpg')

    if result is None:
        print("❌ No face detected in the snapshot. Please try again.")
        exit(1)

    coords, rgb_image, face_rect = result

    # Draw landmarks on image
    result_image = draw_landmarks(rgb_image.copy(), coords)

    # Draw face rectangle
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display result
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Image with landmarks
    axes[1].imshow(result_image)
    axes[1].set_title(f'68 Facial Landmarks Detected')
    axes[1].axis('off')

    plt.tight_layout()

    # Save result
    result_path = 'landmarks_result.jpg'
    plt.savefig(result_path, bbox_inches='tight', dpi=150)
    print(f"\n💾 Result saved: {result_path}")

    # Show the plot
    plt.show()

    # Test smile detection
    print("\n😊 Testing smile detection...")
    smile_result = detect_smile(coords)
    print(f"   Smile detected: {smile_result['is_smiling']}")
    print(f"   Smile intensity: {smile_result['smile_intensity']:.1f}/100")
    print(f"   Mood: {smile_result['mood']}")

    # Test face orientation
    print("\n📐 Testing face orientation...")
    orientation_result = verify_face_orientation(coords)

    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print(f"   Original: landmarks_snapshot.jpg")
    print(f"   Result: {result_path}")
    print("="*60)