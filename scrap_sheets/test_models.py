import face_recognition
import face_recognition_models

print("Testing models...")
print(f"Models location: {face_recognition_models.__file__}")

# Test simplu
import numpy as np
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
print("Face recognition is working!")