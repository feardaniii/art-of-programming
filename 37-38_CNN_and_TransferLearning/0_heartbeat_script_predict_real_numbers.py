import tensorflow as tf
import numpy as np
import cv2  # OpenCV for camera
import tkinter as tk  # GUI for drawing
from PIL import Image, ImageDraw, ImageOps  # Image processing for drawing
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ==========================================
# PART 1: Train the Model (Your Code)
# ==========================================
print("Loading data and training model...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=1)

loss, acc = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {acc * 100:.2f}%")
print("-" * 50)


# ==========================================
# Helper Function: Preprocess Image
# ==========================================
def preprocess_image(img_array):
    # Resize to 28x28
    img_resized = cv2.resize(img_array, (28, 28))
    # Reshape to (1, 28, 28, 1) and normalize
    img_final = img_resized.reshape(1, 28, 28, 1) / 255.0
    return img_final


# ==========================================
# PART 2: Option 1 - Webcam Screenshot
# ==========================================
def start_webcam():
    cap = cv2.VideoCapture(0)
    print("WEBCAM INSTRUCTIONS:")
    print("1. Align the number inside the Green Box.")
    print("2. Ensure good lighting and write with a THICK marker.")
    print("3. Press 's' to Scan/Predict.")
    print("4. Press 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define a Region of Interest (ROI) box
        height, width, _ = frame.shape
        # Box coordinates (center of screen)
        x1, y1 = int(width / 2 - 100), int(height / 2 - 100)
        x2, y2 = int(width / 2 + 100), int(height / 2 + 100)

        # Draw the box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place Number Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Webcam Digit Recognizer', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 1. Crop the ROI
            roi = frame[y1:y2, x1:x2]

            # 2. Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 3. Thresholding (Binarize) and Invert
            # We use thresholding to make the background pure black and text pure white
            # cv2.THRESH_BINARY_INV turns white paper black, and black ink white (like MNIST)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

            # Show what the computer sees (for debugging)
            cv2.imshow("Processed Input", cv2.resize(thresh, (200, 200)))

            # 4. Predict
            processed = preprocess_image(thresh)
            prediction = model.predict(processed)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            print(f"PREDICTION: {digit} (Confidence: {confidence * 100:.2f}%)")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# PART 3: Option 2 - Canvas Drawing
# ==========================================
def start_drawing_pad():
    window = tk.Tk()
    window.title("Draw a Digit")

    # We draw on two things:
    # 1. The visible Canvas (for user)
    # 2. A PIL Image object (in memory, for the AI)
    canvas_width, canvas_height = 300, 300
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='black')
    canvas.pack()

    # Create a black image (MNIST style)
    image1 = Image.new("L", (canvas_width, canvas_height), 0)
    draw = ImageDraw.Draw(image1)

    label_result = tk.Label(window, text="Draw a digit and click Predict", font=("Helvetica", 14))
    label_result.pack()

    def paint(event):
        # Brush size
        r = 12
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        # Draw on screen (white on black)
        canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        # Draw on memory image
        draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, 300, 300], fill=0)
        label_result.config(text="Draw a digit...")

    def predict():
        # Resize the PIL image to 28x28
        img_resized = image1.resize((28, 28))
        # Convert to numpy array
        img_array = np.array(img_resized)

        # Predict
        processed = img_array.reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        conf = np.max(prediction)

        label_result.config(text=f"Prediction: {digit} ({conf * 100:.1f}%)")

    canvas.bind("<B1-Motion>", paint)  # Bind mouse drag

    btn_frame = tk.Frame(window)
    btn_frame.pack()
    tk.Button(btn_frame, text="Predict", command=predict).pack(side=tk.LEFT)
    tk.Button(btn_frame, text="Clear", command=clear).pack(side=tk.LEFT)

    window.mainloop()


# ==========================================
# Main Menu
# ==========================================
while True:
    print("\nSelect Mode:")
    print("1. Webcam Scanner (Hold paper)")
    print("2. Drawing Pad (Use cursor)")
    print("3. Exit")
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        start_webcam()
    elif choice == '2':
        start_drawing_pad()
    elif choice == '3':
        break
    else:
        print("Invalid choice")