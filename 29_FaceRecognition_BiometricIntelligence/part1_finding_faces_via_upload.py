import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== Dlib's Face Detector ======
# HOG (Histogram of Oriented Gradients) - Fast, works on CPU
detector_hog = dlib.get_frontal_face_detector()

# CNN-based detector - More accurate, requires GPU for real-time
# detector_cnn = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def detect_faces_in_image(image_path):
    """
    Detect all faces in an image
    Use case: Security system, photo organization, attendance tracking
    """
    # Load image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces (returns list of rectangles)
    faces = detector_hog(rgb_image, 1)  # 1 = upsample image 1 time (finds smaller faces)

    print(f"\n👤 Found {len(faces)} face(s) in image")

    # Draw rectangles around detected faces
    for i, face_rect in enumerate(faces):
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb_image, f"Person {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"   Face {i+1}: Position ({x}, {y}), Size {w}x{h}px")

    return rgb_image, faces

# ====== Real-World Use Case: Attendance System ======
class AttendanceSystem:
    """
    Automatic attendance tracking using face detection
    Use: Schools, offices, gyms, co-working spaces
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.attendance_log = []

    def check_attendance(self, image_path, timestamp):
        """Check who's present in the image"""
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector(rgb, 1)

        attendance_record = {
            'timestamp': timestamp,
            'people_count': len(faces),
            'face_locations': [(f.left(), f.top(), f.width(), f.height()) for f in faces]
        }

        self.attendance_log.append(attendance_record)

        print(f"\n📸 Attendance Check at {timestamp}")
        print(f"   Detected: {len(faces)} person(s)")

        return attendance_record

    def generate_report(self):
        """Generate attendance summary"""
        total_checks = len(self.attendance_log)
        avg_attendance = np.mean([log['people_count'] for log in self.attendance_log])

        return {
            'total_checks': total_checks,
            'average_attendance': avg_attendance,
            'peak_attendance': max([log['people_count'] for log in self.attendance_log])
        }

# ====== Health App: Focus Monitoring ======
def monitor_focus_session(video_path):
    """
    Monitor if person is present and focused during work session
    Use: Productivity tracking, study sessions, remote work monitoring
    """
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)

    focus_metrics = {
        'total_frames': 0,
        'face_detected_frames': 0,
        'looking_away_frames': 0
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        focus_metrics['total_frames'] += 1

        # Detect face every 10 frames (optimize performance)
        if focus_metrics['total_frames'] % 10 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb, 0)  # 0 = no upsampling (faster)

            if len(faces) > 0:
                focus_metrics['face_detected_frames'] += 1
            else:
                focus_metrics['looking_away_frames'] += 1

    cap.release()

    # Calculate focus score
    focus_percentage = (focus_metrics['face_detected_frames'] /
                       (focus_metrics['face_detected_frames'] + focus_metrics['looking_away_frames'])) * 100

    print(f"\n🎯 Focus Session Analysis:")
    print(f"   Focus Score: {focus_percentage:.1f}%")
    print(f"   Present: {focus_metrics['face_detected_frames']} checks")
    print(f"   Away: {focus_metrics['looking_away_frames']} checks")

    return focus_metrics

print("\n💡 Face Detection is the FOUNDATION of all facial analysis")
print("   No face detection → No landmarks → No recognition → No tracking")
print("\n🎯 Master the foundation, master everything that follows!")

# ====== RUNNABLE DEMO ======
import tkinter as tk
from tkinter import filedialog

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Face Detection from Uploaded Image")
    print("="*60)

    # Hide the main Tkinter window
    root = tk.Tk()
    root.withdraw()

    print("\n📂 Please select an image file...")

    # Open file dialog
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        print("❌ No file selected. Exiting.")
        exit(0)

    print(f"✅ Selected file: {image_path}")

    # Run detection
    print("\n🔍 Detecting faces...")
    result_image, faces = detect_faces_in_image(image_path)

    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result_image)
    plt.title(f'Face Detection Result - {len(faces)} face(s) detected')
    plt.axis('off')
    plt.tight_layout()

    # Save result
    result_path = 'face_detection_result.jpg'
    plt.savefig(result_path, bbox_inches='tight', dpi=150)
    print(f"\n💾 Result saved: {result_path}")

    plt.show()

    print("\n" + "="*60)
    print("✅ DONE!")
    print(f"   Found {len(faces)} face(s)")
    print("="*60)