import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================================
# PHASE 1: LOAD & PREPARE IMAGE (IMPROVED)
# ============================================================================

print("🔍 Looking for image files...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"📂 Files in directory: {os.listdir('.')}\n")

# Find image file
image_path = None
for file in os.listdir('.'):
    if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
        image_path = file
        print(f"✅ Found image file: {image_path}")
        break

if image_path is None:
    print("❌ Error: No image files found in this directory!")
    print("   Please place a .jpg, .png, or .bmp file in the same folder as this script.")
    exit()

# Load the image
print(f"\n📸 Attempting to load: {image_path}")
image = cv2.imread(image_path)

# Safety check: did the image load?
if image is None:
    print(f"❌ Error: Could not load image '{image_path}'")
    print("   This might be due to:")
    print("   - File is corrupted")
    print("   - File format not supported")
    print("   - Permission issues")
    exit()

print(f"✅ Image loaded successfully!")
print(f"   Shape: {image.shape} (height × width × channels)")
print(f"   Size: {image.shape[1]} × {image.shape[0]} pixels\n")

# Important: OpenCV uses BGR (not RGB), so convert to RGB for display later
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============================================================================
# PHASE 2: CONVERT TO HSV (The Key Step!)
# ============================================================================

# Convert from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(f"✅ Converted to HSV color space")

# ============================================================================
# PHASE 3: DEFINE BLUE RANGE & CREATE MASK
# ============================================================================

# In HSV, blue is roughly at Hue 100-130
# We give ourselves some margin and use 90-140 to catch various shades
# Saturation should be decent (not washed out) - so 50+
# Value should be visible (not pitch black) - so 50+

# Lower bound of blue range
lower_blue = np.array([90, 50, 50])

# Upper bound of blue range
upper_blue = np.array([140, 255, 255])

# Create a binary mask where:
#   White (255) = pixels within blue range
#   Black (0) = pixels outside blue range
mask = cv2.inRange(hsv, lower_blue, upper_blue)

blue_pixel_count = np.sum(mask > 0)
print(f"✅ Created mask")
print(f"   Blue pixels found: {blue_pixel_count}")
print(f"   Percentage of image: {(blue_pixel_count / (image.shape[0] * image.shape[1]) * 100):.2f}%\n")

# ============================================================================
# PHASE 4: FIND CONTOURS (Object Boundaries)
# ============================================================================

# cv2.findContours() finds all connected white regions in the mask
# It returns: contours (list of all objects), hierarchy (relationship info)
contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_TREE,              # Retrieve all contours
    cv2.CHAIN_APPROX_SIMPLE     # Simplify contour (don't store every pixel)
)

print(f"✅ Found {len(contours)} total contours")

# ============================================================================
# PHASE 5: FILTER BY SIZE & ANALYZE
# ============================================================================

# We'll store information about each valid blue object
blue_objects = []

MIN_AREA = 500  # Only keep objects bigger than 500 pixels
# (smaller = likely noise like compression artifacts)

for contour in contours:
    # Calculate the area of this contour (in pixels)
    area = cv2.contourArea(contour)
    
    # Skip if too small (noise filtering)
    if area < MIN_AREA:
        continue
    
    # This contour is big enough! Let's analyze it.
    
    # Get the bounding rectangle (for drawing boxes)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate centroid (center point)
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Avoid division by zero
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x, y
    
    # Store this object's info
    blue_objects.append({
        'contour': contour,
        'area': area,
        'perimeter': perimeter,
        'x': x, 'y': y, 'w': w, 'h': h,
        'centroid': (cx, cy)
    })

print(f"✅ Filtered to {len(blue_objects)} significant blue objects (area > {MIN_AREA}px)\n")

# ============================================================================
# PHASE 6: DRAW RESULTS ON IMAGE
# ============================================================================

# Make a copy to draw on (don't modify original)
result = image_rgb.copy()

# Draw a box and centroid for each blue object
for i, obj in enumerate(blue_objects):
    x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    cx, cy = obj['centroid']
    area = obj['area']
    
    # Draw rectangle (green box)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw centroid (red dot)
    cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
    
    # Add label with area
    label = f"Blue #{i+1} ({int(area)}px)"
    cv2.putText(result, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"✅ Drew rectangles and centroids on image")

# ============================================================================
# PHASE 7: DISPLAY & ANALYZE
# ============================================================================

# Create a 2x2 grid of images to compare
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Original image
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Top-right: HSV Hue channel (what the code "sees")
axes[0, 1].imshow(hsv[:, :, 0], cmap='hsv')
axes[0, 1].set_title('HSV Hue Channel (0-180)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Bottom-left: The mask (white = blue)
axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title('Blue Mask (white=blue pixels)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Bottom-right: Result with boxes
axes[1, 1].imshow(result)
axes[1, 1].set_title(f'Detection Results ({len(blue_objects)} objects)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('blue_detection_results.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization to: blue_detection_results.png\n")
plt.show()

# ============================================================================
# PHASE 8: PRINT STATISTICS
# ============================================================================

print("="*60)
print("📊 BLUE OBJECT DETECTION RESULTS")
print("="*60)

if len(blue_objects) == 0:
    print("❌ No blue objects detected!")
    print("\n   Debugging steps:")
    print("   1. Check if your image actually has blue objects")
    print("   2. Try adjusting the HSV range:")
    print("      - Increase upper_blue[1] and upper_blue[2] for lighter blues")
    print("      - Decrease lower_blue[1] and lower_blue[2] for more tolerant range")
    print("   3. Reduce MIN_AREA if objects are small")
    print("   4. Look at the 'Blue Mask' panel - white areas = detected blue")
else:
    print(f"\n✅ Detected {len(blue_objects)} blue object(s):\n")
    
    for i, obj in enumerate(blue_objects, 1):
        print(f"   Object #{i}:")
        print(f"      Area: {int(obj['area'])} pixels")
        print(f"      Perimeter: {int(obj['perimeter'])} pixels")
        print(f"      Position (top-left): ({obj['x']}, {obj['y']})")
        print(f"      Size: {obj['w']} × {obj['h']} pixels")
        print(f"      Center: {obj['centroid']}")
        print()

print("="*60)