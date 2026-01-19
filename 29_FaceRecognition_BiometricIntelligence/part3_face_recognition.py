import dlib
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from model_downloader import ensure_models

# ====== Load Required Models ======
print("🔍 Checking for required models...")
models = ensure_models(
    'shape_predictor_68_face_landmarks.dat',
    'dlib_face_recognition_resnet_model_v1.dat'
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(models['shape_predictor_68_face_landmarks.dat'])

# Face recognition model (ResNet-based, trained on millions of faces)
face_rec_model = dlib.face_recognition_model_v1(models['dlib_face_recognition_resnet_model_v1.dat'])

def get_face_encoding(image_path):
    """
    Convert a face to a 128-dimensional vector
    This is the CORE of face recognition
    """
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face
    faces = detector(rgb, 1)
    if len(faces) == 0:
        print("❌ No face detected")
        return None

    # Get landmarks
    landmarks = predictor(rgb, faces[0])

    # Compute 128D face encoding (descriptor)
    # exemplu vector_persoana_referinta = [0.123, -0.234, ..., 0.045]
    # exemplu vector_persoana_verificare = [0.130, -0.220, ..., 0.050]
    face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb, landmarks))

    print(f"\n🔢 Face Encoding Generated:")
    print(f"   Shape: {face_encoding.shape}")  # (128,)
    print(f"   Type: {face_encoding.dtype}")   # float64
    print(f"   First 5 values: {face_encoding[:5]}")
    print(f"\n💡 This 128D vector UNIQUELY represents this person's face!")

    return face_encoding

def compare_faces(encoding1, encoding2, threshold=0.6):
    """
    Compare two face encodings
    Distance < threshold → Same person
    Distance > threshold → Different person
    """
    # Euclidean distance (most common for face recognition)
    euclidean_dist = np.linalg.norm(encoding1 - encoding2)

    # Cosine similarity (alternative metric)
    cosine_sim = cosine_similarity([encoding1], [encoding2])[0][0]

    is_same_person = euclidean_dist < threshold

    result = {
        'euclidean_distance': euclidean_dist,
        'cosine_similarity': cosine_sim,
        'is_same_person': is_same_person,
        'confidence': 1 - (euclidean_dist / 1.0)  # Convert to 0-1 scale
    }

    print(f"\n🔍 Face Comparison:")
    print(f"   Euclidean Distance: {euclidean_dist:.4f}")
    print(f"   Cosine Similarity: {cosine_sim:.4f}")
    print(f"   Result: {'✅ SAME PERSON' if is_same_person else '❌ DIFFERENT PERSON'}")
    print(f"   Confidence: {result['confidence']:.1%}")

    return result

# ====== Production Face Recognition System ======
class FaceRecognitionSystem:
    """
    Production-grade face recognition for authentication
    Use: Banking apps, access control, attendance systems
    """

    def __init__(self, distance_threshold=0.6):
        self.detector = detector
        self.predictor = predictor
        self.face_rec_model = face_rec_model
        self.threshold = distance_threshold

        # Database of known faces (in production: PostgreSQL, MongoDB, etc.)
        self.known_faces = {}  # {name: encoding}
        self.face_metadata = {}  # {name: {last_seen, access_level, etc.}}

    def register_person(self, name, image_path, metadata=None):
        """Register a new person in the system"""
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.detector(rgb, 1)

        if len(faces) != 1:
            print(f"❌ Error: Expected 1 face, found {len(faces)}")
            return False

        # Get face encoding
        landmarks = self.predictor(rgb, faces[0])
        encoding = np.array(self.face_rec_model.compute_face_descriptor(rgb, landmarks))

        # Store in database
        self.known_faces[name] = encoding
        self.face_metadata[name] = metadata or {'registered_date': 'today'}

        print(f"✅ {name} registered successfully")
        print(f"   Encoding shape: {encoding.shape}")
        print(f"   Total registered: {len(self.known_faces)} person(s)")

        return True

    def authenticate(self, image_path):
        """
        Authenticate a person against the database
        Returns: name, confidence, metadata
        """
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.detector(rgb, 1)

        if len(faces) == 0:
            return {'status': 'no_face', 'authenticated': False}

        # Get encoding for detected face
        landmarks = self.predictor(rgb, faces[0])
        unknown_encoding = np.array(self.face_rec_model.compute_face_descriptor(rgb, landmarks))

        # Compare with all known faces
        best_match = None
        best_distance = float('inf')

        for name, known_encoding in self.known_faces.items():
            distance = np.linalg.norm(unknown_encoding - known_encoding)

            if distance < best_distance:
                best_distance = distance
                best_match = name

        # Check if match is good enough
        if best_distance < self.threshold:
            return {
                'status': 'authenticated',
                'name': best_match,
                'confidence': 1 - (best_distance / 1.0),
                'distance': best_distance,
                'metadata': self.face_metadata[best_match],
                'authenticated': True
            }
        else:
            return {
                'status': 'unknown',
                'distance': best_distance,
                'authenticated': False
            }

    def batch_identify(self, image_path):
        """
        Identify all faces in an image
        Use: Group photos, event attendance, security monitoring
        """
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.detector(rgb, 1)

        results = []

        for i, face_rect in enumerate(faces):
            landmarks = self.predictor(rgb, face_rect)
            encoding = np.array(self.face_rec_model.compute_face_descriptor(rgb, landmarks))

            # Find best match
            best_match = "Unknown"
            best_distance = float('inf')

            for name, known_encoding in self.known_faces.items():
                distance = np.linalg.norm(encoding - known_encoding)
                if distance < best_distance and distance < self.threshold:
                    best_distance = distance
                    best_match = name

            results.append({
                'face_number': i + 1,
                'name': best_match,
                'bbox': (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()),
                'confidence': 1 - (best_distance / 1.0) if best_match != "Unknown" else 0
            })

        return results
    















# ====== Finance Application: Secure Payment Authentication ======
class SecurePaymentAuth:
    """
    Facial authentication for financial transactions
    Use: Banking apps, payment platforms, crypto wallets
    """

    def __init__(self):
        self.face_system = FaceRecognitionSystem(distance_threshold=0.5)  # Stricter for finance

    def authorize_transaction(self, user_id, live_image_path, transaction_amount):
        """
        Authorize financial transaction using facial recognition
        Requires: Live image verification, registered user match
        """
        # Authenticate user
        auth_result = self.face_system.authenticate(live_image_path)

        if not auth_result['authenticated']:
            return {
                'authorized': False,
                'reason': 'Face authentication failed',
                'transaction_status': 'DENIED'
            }

        # Additional checks for high-value transactions
        if transaction_amount > 1000 and auth_result['confidence'] < 0.85:
            return {
                'authorized': False,
                'reason': 'Confidence too low for high-value transaction',
                'transaction_status': 'REQUIRES_2FA',
                'confidence': auth_result['confidence']
            }

        return {
            'authorized': True,
            'user': auth_result['name'],
            'confidence': auth_result['confidence'],
            'transaction_status': 'APPROVED',
            'transaction_id': f"TXN_{user_id}_{int(transaction_amount)}"
        }










# ====== Example Usage ======
def demo_face_recognition():
    """Complete face recognition demo"""

    # Initialize system
    system = FaceRecognitionSystem()

    # Register users (in production: bulk import from HR/user database)
    system.register_person("Alice", "users/alice_001.jpg",
                          metadata={'access_level': 'admin', 'employee_id': 'E001'})
    system.register_person("Bob", "users/bob_001.jpg",
                          metadata={'access_level': 'user', 'employee_id': 'E002'})

    # Authenticate new image
    result = system.authenticate("verify/unknown_person.jpg")
    
    if result['authenticated']:
        print(f"\n✅ AUTHENTICATION SUCCESSFUL")
        print(f"   Welcome back, {result['name']}!")
        print(f"   Confidence: {result['confidence']:.1%}")
    else:
        print(f"\n❌ AUTHENTICATION FAILED")
        print(f"   Status: {result['status']}")

print("\n💡 Face Recognition = Linear Algebra in Action")
print("   128D vectors → Distance metrics → Identity verification")
print("   Same concepts as Data:Calculus, now applied to SECURITY!")
print("\n🔐 You're mastering the mathematics of IDENTITY!")

# ====== RUNNABLE DEMO ======
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: Face Recognition System")
    print("="*60)
    print("\nThis demo will:")
    print("1. Capture your reference image (to register you)")
    print("2. Capture a verification image (to test recognition)")
    print("3. Compare the two images and show results")

    # Initialize webcam
    print("\n📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("   Please ensure your webcam is connected and not in use by another application")
        exit(1)

    print("✅ Webcam ready!")

    # ===== STEP 1: Capture reference image =====
    print("\n" + "-"*60)
    print("STEP 1: Capture Reference Image")
    print("-"*60)
    print("Position yourself in front of the camera")
    print("Press SPACE to capture your reference image\n")

    captured = False
    while not captured:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from webcam")
            break

        cv2.putText(frame, "STEP 1: Press SPACE for reference", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Face Recognition Demo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            reference_path = 'face_recognition_reference.jpg'
            cv2.imwrite(reference_path, frame)
            print(f"📸 Reference image saved: {reference_path}")
            captured = True
        elif key == ord('q'):
            print("Demo cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # ===== STEP 2: Capture verification image =====
    print("\n" + "-"*60)
    print("STEP 2: Capture Verification Image")
    print("-"*60)
    print("Change your expression or position slightly")
    print("Press SPACE to capture verification image\n")

    import time
    time.sleep(2)  # Give user time to adjust

    captured = False
    while not captured:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from webcam")
            break

        cv2.putText(frame, "STEP 2: Press SPACE to verify", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Face Recognition Demo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            verify_path = 'face_recognition_verify.jpg'
            cv2.imwrite(verify_path, frame)
            print(f"📸 Verification image saved: {verify_path}")
            captured = True
        elif key == ord('q'):
            print("Demo cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # ===== STEP 3: Compare faces =====
    print("\n" + "-"*60)
    print("STEP 3: Face Recognition Analysis")
    print("-"*60)

    print("\n🔍 Extracting face encoding from reference image...")
    encoding1 = get_face_encoding(reference_path)

    if encoding1 is None:
        print("❌ No face detected in reference image. Please try again.")
        exit(1)

    print("\n🔍 Extracting face encoding from verification image...")
    encoding2 = get_face_encoding(verify_path)

    if encoding2 is None:
        print("❌ No face detected in verification image. Please try again.")
        exit(1)

    print("\n🔬 Comparing face encodings...")
    comparison_result = compare_faces(encoding1, encoding2)

    # ===== Visualize Results =====
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Load and display images
    ref_img = cv2.imread(reference_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    verify_img = cv2.imread(verify_path)
    verify_img = cv2.cvtColor(verify_img, cv2.COLOR_BGR2RGB)

    axes[0].imshow(ref_img)
    axes[0].set_title('Reference Image\n(Registered User)')
    axes[0].axis('off')

    axes[1].imshow(verify_img)
    result_text = "✅ SAME PERSON" if comparison_result['is_same_person'] else "❌ DIFFERENT PERSON"
    axes[1].set_title(f'Verification Image\n{result_text}\nConfidence: {comparison_result["confidence"]:.1%}')
    axes[1].axis('off')

    plt.tight_layout()

    # Save result
    result_path = 'face_recognition_result.jpg'
    plt.savefig(result_path, bbox_inches='tight', dpi=150)
    print(f"\n💾 Result saved: {result_path}")

    plt.show()

    # ===== Summary =====
    print("\n" + "="*60)
    print("✅ FACE RECOGNITION DEMO COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Match: {result_text}")
    print(f"   Euclidean Distance: {comparison_result['euclidean_distance']:.4f}")
    print(f"   Cosine Similarity: {comparison_result['cosine_similarity']:.4f}")
    print(f"   Confidence: {comparison_result['confidence']:.1%}")
    print(f"\n💡 Threshold: Distance < 0.6 = Same Person")
    print(f"   Your distance: {comparison_result['euclidean_distance']:.4f}")
    print("\n" + "="*60)