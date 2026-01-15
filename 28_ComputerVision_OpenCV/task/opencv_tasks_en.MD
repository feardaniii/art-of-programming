# 👁️ OPENCV COMPUTER VISION - SURVEILLANCE & SAFETY SYSTEM

---

## 📚 Introduction

Welcome to **Computer Vision** - where your code learns to "see" the world!

In this chapter, you'll discover that images are just **NumPy arrays in disguise**. Every pixel is a number, and by manipulating these numbers, you can:
- 🎨 **Detect colors** (find all blue objects)
- 🔍 **Track movement** (spot changes between frames)
- 🚨 **Recognize patterns** (detect smoke, fire, faces)
- 📹 **Build real-time systems** (surveillance, safety alerts)

By the end, you'll build a **real-world surveillance and fire safety system** using nothing but pixel math and OpenCV!

---

## 🎯 What is OpenCV?

**OpenCV** (Open Source Computer Vision Library) is the industry standard for computer vision:

```
Camera/Image → NumPy Array → OpenCV Filters → Insights/Actions
     ↓              ↓              ↓              ↓
  Pixels      [0-255 values]  Algorithms    Decisions
```

**Key concepts you'll master:**
- **Color spaces**: RGB (what we see), HSV (what's easier to filter), Grayscale (shapes only)
- **Thresholding**: Converting images to binary (black/white) based on conditions
- **Contours**: Finding boundaries of objects
- **Frame differencing**: Detecting what changed between two images
- **Morphological operations**: Cleaning up noise, filling gaps

---

## 🚀 Setup & Installation

Ensure you have OpenCV installed:
```bash
pip install opencv-python matplotlib numpy
```

**Quick test:**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

print(f"OpenCV version: {cv2.__version__}")

# Create a 100x100 red square
red_square = np.zeros((100, 100, 3), dtype=np.uint8)
red_square[:, :] = [255, 0, 0]  # RGB: Red

plt.imshow(red_square)
plt.title("Your first OpenCV creation!")
plt.axis('off')
plt.show()
```

**💡 Remember:** Images are just 3D arrays: `(height, width, channels)`

---

# PART 1: WARMUP EXERCISES

---

## 📝 EXERCISE 1: Blue Object Detector

**Objective:** Detect all blue objects in an image using HSV color space

### 📖 Concepts you'll learn:

**`Color spaces`**: Different ways to represent colors
- **RGB**: Red-Green-Blue (0-255 each) - how cameras see
- **HSV**: Hue-Saturation-Value - easier for color detection
  - **Hue**: The actual color (0-180 in OpenCV)
  - **Saturation**: Color intensity (0-255)
  - **Value**: Brightness (0-255)

**`cv2.inRange()`**: Creates a mask where pixels in range = white (255), others = black (0)

**`cv2.findContours()`**: Finds boundaries of white regions in binary images

**`cv2.contourArea()`**: Calculates area of a contour in pixels

### 🎯 Your Mission:

1. Load an image (take a photo with your phone of colorful objects!)
2. Convert from BGR to HSV color space
3. Create a mask that isolates ONLY blue pixels
4. Find contours (outlines) of blue objects
5. Filter out small noise (keep only objects > 500 pixels)
6. Draw rectangles around detected blue objects
7. Display: original image, mask, and final detection

### 💻 Starting Code:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image - Practice loading an image in python cv2.


# TODO 1: Convert to HSV

# TODO 2: Define blue color range in HSV
# Blue in HSV: Hue ~100-130, high saturation, decent value
# Research and Adjust these!

# TODO 3: Create mask

# TODO 4: Find contours

# TODO 5: Filter and draw

# TODO 6: Visualize results

```

### 💡 Hints:

- **Finding the right HSV range**: Use trial and error! Common ranges:
  - Blue: `[100, 50, 50]` to `[130, 255, 255]`
  - Green: `[40, 50, 50]` to `[80, 255, 255]`
  - Red: `[0, 50, 50]` to `[10, 255, 255]` OR `[170, 50, 50]` to `[180, 255, 255]`
- **Pro tip**: Create a trackbar to adjust ranges in real-time (advanced!)

### ❓ Reflection Questions:

**1. What happens if you set `area > 100` instead of `area > 500`?**

_Your answer:_ _______________________________________________

**2. Challenge: Modify the code to detect RED objects. What changes?**

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 2: Ancient Text Enhancement with Adaptive Thresholding

**Objective:** Make old, faded text readable again using adaptive thresholding

### 📖 Concepts you'll learn:

**`Thresholding`**: Converting grayscale images to pure black/white
- **Global threshold**: One value for entire image
- **Adaptive threshold**: Different values for different regions (better for uneven lighting!)

**`cv2.adaptiveThreshold()`**: Smart thresholding that handles shadows and lighting variations

**Why this matters:** Scanning old documents, reading receipts, OCR preprocessing

### 🎯 Your Mission:

1. Find an image with text (old book page, handwritten note, receipt)
2. Convert to grayscale
3. Apply regular (global) threshold - observe limitations
4. Apply adaptive threshold - see the magic!
5. Compare results side-by-side

### 💻 Starting Code:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image with text (take a photo of a book page!)
image = cv2.imread('old_text.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"📄 Image loaded: {gray.shape}")

# TODO 1: Apply global threshold
# Threshold value: 127 (middle of 0-255)
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# TODO 2: Apply adaptive threshold (Gaussian method)
adaptive_thresh = cv2.adaptiveThreshold(
    gray,
    255,                              # Max value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Method
    cv2.THRESH_BINARY,                # Type
    11,                               # Block size (must be odd!)
    2                                 # C constant (subtracted from mean)
)

# TODO 3: Try Mean method as well
adaptive_mean = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,       # Different method
    cv2.THRESH_BINARY,
    11,
    2
)

# Visualize all versions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Original Grayscale')
axes[0, 0].axis('off')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('Global Threshold (127)')
axes[0, 1].axis('off')

axes[1, 0].imshow(adaptive_thresh, cmap='gray')
axes[1, 0].set_title('Adaptive Threshold (Gaussian)')
axes[1, 0].axis('off')

axes[1, 1].imshow(adaptive_mean, cmap='gray')
axes[1, 1].set_title('Adaptive Threshold (Mean)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('text_enhancement_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Comparison saved to: text_enhancement_comparison.png")
plt.show()

# Calculate readability score (simple heuristic)
def calculate_readability(binary_image):
    """Higher score = better text/background separation"""
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)
    ratio = min(white_pixels, black_pixels) / max(white_pixels, black_pixels)
    return ratio * 100

print("\n📊 Readability Scores:")
print(f"   Global threshold: {calculate_readability(global_thresh):.1f}/100")
print(f"   Adaptive (Gaussian): {calculate_readability(adaptive_thresh):.1f}/100")
print(f"   Adaptive (Mean): {calculate_readability(adaptive_mean):.1f}/100")
```

### 💡 Hints:

- **Block size**: Larger = smoother (try 11, 15, 21, 31)
- **C constant**: Positive values make it brighter, negative = darker
- **Test with**: Old receipts, whiteboard photos, scanned notes

### 🎓 Advanced Challenge:

Apply **morphological operations** to clean up the result:
```python
# Remove small noise
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
```

### ❓ Reflection Questions:

**1. What else do you think we can accomplish/clean beyond old writings?**

_Your answer:_ _______________________________________________

**2. Real-world application: How could this help with OCR (text recognition)?**

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 3: Contour Area Analysis - Find Large Objects

**Objective:** Extract and analyze objects larger than a specific size

### 📖 Concepts you'll learn:

**`Contours`**: The boundaries of shapes in binary images
**`Morphological operations`**: Mathematical operations on binary images
- **Erosion**: Shrinks objects (removes noise)
- **Dilation**: Expands objects (fills gaps)
- **Opening**: Erosion → Dilation (removes small noise)
- **Closing**: Dilation → Erosion (fills small holes)

### 🎯 Your Mission:

1. Load an image with multiple objects
2. Convert to binary (threshold)
3. Find ALL contours
4. Filter: keep only contours > 500 pixels
5. Draw each contour with a different color
6. Display area, perimeter, and centroid of each object

### 💻 Starting Code:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('objects.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"📸 Image loaded: {image_rgb.shape}")

# TODO 1: Convert to binary

# TODO 2: Optional for better accuray - Clean up with morphological operations

# TODO 3: Find contours

# TODO 4: Filter and analyze - your Work

# Print analysis

# Visualize

```


### ❓ Reflection Questions:

**1. Would your algorithm hold and work if there were 1000 images coming in per second?**

_Your answer:_ _______________________________________________

**2. Challenge: How would you identify circles vs rectangles using contours?**

_Your answer:_ _______________________________________________

---

# PART 2: FINAL PROJECT - SURVEILLANCE & FIRE SAFETY SYSTEM

---

## 🚨 PROJECT: Real-Time Surveillance & Smoke Detection System

**Objective:** Build a complete security system that detects:
1. **Intruders**: Motion detection via frame differencing
2. **Fire hazards**: Smoke detection via color and pattern analysis

This is a **real-world application** combining everything you've learned!

---

## 📋 System Requirements

Your system must:

### 🎯 Core Features:

1. **Webcam Stream**: Capture live video from your camera
2. **Motion Detection**:
   - Compare consecutive frames
   - Calculate pixel differences
   - Trigger alarm when changes exceed threshold
   - Draw bounding boxes around moving objects
3. **Smoke Detection**:
   - Analyze frame colors (smoke = gray/white gradient)
   - Track spreading pattern (smoke rises and expands)
   - Detect gradual color changes over multiple frames
4. **Alarm System**:
   - Visual alerts (red flashing border)
   - Console warnings with timestamps
   - Save snapshots of events

### 🏆 Advanced Features (Optional):

- Ignore small movements (pets, shadows)
- Multi-zone monitoring (different sensitivity per area)
- Event logging to file
- Email/SMS alerts (using `smtplib` or Twilio)

---

## 🧠 Technical Approach

### Part A: Motion Detection Algorithm

```
Step 1: Capture frame_t and frame_t-1
Step 2: Convert both to grayscale
Step 3: Calculate |frame_t - frame_t-1| = difference_map
Step 4: Apply threshold → binary map
Step 5: Find contours in binary map
Step 6: Filter contours by area → large changes = motion!
Step 7: If total_changed_pixels > MOTION_THRESHOLD → ALERT!
```

**Key insight:** Bodies are large objects, so focus on contours > 1000 pixels

### Part B: Smoke Detection Algorithm

```
Step 1: Convert frame to HSV
Step 2: Define smoke color range (gray/white: low saturation, high value)
Step 3: Create smoke mask
Step 4: Track smoke area over last N frames
Step 5: If smoke_area is INCREASING over time → FIRE ALERT!
Step 6: Bonus: Check if smoke "rises" (y-coordinates decrease)
```

**Key insight:** Smoke has:
- Low saturation (not colorful)
- High value (bright/light)
- Expanding pattern (grows over frames)
- Rising motion (moves up)

---

## 💻 Project Starter Code

### Part 1: Motion Detection Only

```python
import cv2
import numpy as np
from datetime import datetime

# Configuration
MOTION_THRESHOLD = 2000  # Pixels changed to trigger alarm
MIN_CONTOUR_AREA = 1000  # Minimum object size (filters noise)

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

print("🎥 Surveillance System Starting...")
print(f"   Motion threshold: {MOTION_THRESHOLD} pixels")
print(f"   Min object size: {MIN_CONTOUR_AREA} pixels")
print("\nPress 'q' to quit\n")

alarm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TODO 1: Prepare current frame

    # TODO 2: Calculate frame difference

    # TODO 3: Threshold the difference

    # TODO 4: Dilate to fill gaps

    # TODO 5: Find contours

    # TODO 6: Analyze motion

    # TODO 7: Trigger alarm

    # TODO 8: Display info

    # Show feeds
    cv2.imshow('Surveillance System', frame)
    # cv2.imshow('Motion Mask', thresh)

    # Update previous frame
    # previous_gray = current_gray

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Surveillance System Stopped")
```

---

### Part 2: Complete System with Smoke Detection

**YOUR TASK:** Extend the above code to include smoke detection!

```python
# Add after motion detection (inside the while loop)

# TODO: SMOKE DETECTION IMPLEMENTATION

# 1. Convert frame to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 2. Define smoke color range
# Smoke characteristics in HSV:
#   - Hue: Any (0-180)
#   - Saturation: LOW (0-50) - not very colorful
#   - Value: HIGH (200-255) - bright/light
lower_smoke = np.array([0, 0, 200])      # TODO: Adjust these values!
upper_smoke = np.array([180, 50, 255])   # TODO: Adjust these values!

# 3. Create smoke mask
smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

# 4. Clean up noise
kernel = np.ones((5, 5), np.uint8)
smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

# 5. Calculate smoke area
smoke_pixels = np.sum(smoke_mask > 0)

# TODO: Track smoke over time (use a list to store last 10 frames)
# smoke_history = []  # Initialize before loop
# smoke_history.append(smoke_pixels)
# if len(smoke_history) > 10:
#     smoke_history.pop(0)

# 6. Check for expanding smoke pattern
SMOKE_THRESHOLD = 5000  # Adjust based on testing
smoke_detected = smoke_pixels > SMOKE_THRESHOLD

# TODO: Advanced - check if smoke is GROWING
# if len(smoke_history) >= 5:
#     recent_avg = np.mean(smoke_history[-5:])
#     old_avg = np.mean(smoke_history[:5])
#     if recent_avg > old_avg * 1.5:  # 50% growth
#         smoke_detected = True

# 7. Fire alarm!
if smoke_detected:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔥 SMOKE DETECTED! [{timestamp}] - {int(smoke_pixels)} smoke pixels")

    # Visual alarm
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 255), 15)
    cv2.putText(frame, "!!! FIRE ALERT !!!", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 4)

    # Save snapshot
    cv2.imwrite(f'smoke_{timestamp.replace(":", "-")}.jpg', frame)

# 8. Display smoke info
cv2.putText(frame, f"Smoke: {int(smoke_pixels)} px", (10, frame.shape[0] - 40),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Show smoke mask
cv2.imshow('Smoke Detection', smoke_mask)
```

---

## 🧪 Testing Your System

### Testing Motion Detection:
1. Start the program
2. Stay still for 3 seconds
3. Wave your hand in front of camera
4. Walk across the frame
5. **Expected**: Green boxes around you, red alert border

### Testing Smoke Detection:
Since real smoke is dangerous, simulate it:
1. **White paper test**: Slowly move a white/gray paper upward
2. **Flashlight test**: Shine a light through white fabric
3. **Steam (safe)**: Boil water and capture rising steam
4. **Image test**: Display smoke image on phone screen, move it

**💡 Pro tip:** Adjust thresholds based on your environment's lighting!

---

## 📊 Deliverables

Your final submission should include:

```
opencv_surveillance_project/
├── surveillance_system.py          # Main code
├── test_results/
│   ├── motion_2024-12-30_14-23-10.jpg
│   ├── smoke_2024-12-30_14-25-33.jpg
│   └── screenshots.png (all 4 windows)
├── README.md                       # Project documentation
└── reflection_answers.md
```

### README.md should contain:
- **How to run** your system
- **Threshold values** you used (and why)
- **Testing methodology** (how you tested motion/smoke)
- **Challenges faced** and solutions
- **Demo video** link (optional but impressive!)

---

## ❓ Final Reflection Questions

### 1. What makes smoke detection harder than motion detection?

_Your answer:_
_______________________________________________________________
_______________________________________________________________

### 2. How would you reduce false alarms (e.g., ignore shadows, lighting changes)?

_Your answer:_
_______________________________________________________________
_______________________________________________________________

### 3. Your system catches an intruder at 3 AM. What additional features would make it more useful?

_Your answer:_
_______________________________________________________________
_______________________________________________________________

### 4. Real-world deployment: What are the ethical considerations of surveillance systems?

_Your answer:_
_______________________________________________________________
_______________________________________________________________

---

## 🎓 Grading Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **Motion Detection** | 30 | Correctly detects and highlights movement |
| **Smoke Detection** | 30 | Identifies smoke-like patterns |
| **Code Quality** | 15 | Clean, commented, organized |
| **Testing** | 10 | Documented testing process with screenshots |
| **Alarms** | 10 | Visual alerts + console logs working |
| **Reflection** | 5 | Thoughtful answers to questions |
| **Bonus Features** | +10 | Event logging, multi-zone, email alerts, etc. |

**Total: 100 points** (+ 10 bonus)

---

## 💡 Useful Resources

- **OpenCV Docs**: https://docs.opencv.org/4.x/
- **Color space converter**: https://colorizer.org/
- **HSV color picker**: https://alloyui.com/examples/color-picker/hsv
- **Frame differencing tutorial**: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
- **Morphological operations**: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

---

---

## ✅ Final Checklist

Before submitting, verify:

- [ ] Motion detection works and triggers alarms
- [ ] Smoke detection identifies smoke-like patterns
- [ ] Both systems can run simultaneously
- [ ] Snapshots are saved correctly
- [ ] Console logs show timestamps
- [ ] Code is well-commented
- [ ] Tested with multiple scenarios
- [ ] Threshold values are documented
- [ ] Reflection questions answered
- [ ] README.md is complete

---

**🎉 Congratulations! 🎉**

You've built a **real surveillance system** using nothing but pixels and math. This is the foundation of:
- 🏠 Smart home security
- 🏭 Industrial safety monitoring
- 🚗 Autonomous vehicle perception
- 📱 Mobile camera features
- 🤖 Robot vision systems

**Remember:** Computer vision is just **NumPy arrays + clever algorithms**. You now have the tools to make computers see!

---

*"The camera is an instrument that teaches people how to see without a camera."* - Dorothea Lange

**Now go build something that sees the world differently!** 👁️✨
