import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================================
# PHASE 1: LOAD IMAGE & CONVERT TO BINARY
# ============================================================================

print("🔍 Looking for image files...")

# Find image file
image_path = None
for file in os.listdir('.'):
    if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
        image_path = file
        print(f"✅ Found image file: {image_path}")
        break

if image_path is None:
    print("❌ Error: No image files found!")
    print("   Use an image with multiple objects (shapes, letters, etc.)")
    exit()

# Load the image
print(f"📸 Loading: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Error: Could not load image '{image_path}'")
    exit()

# Convert to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"✅ Image loaded! Shape: {image.shape}")
print(f"   Dimensions: {image.shape[1]} × {image.shape[0]} pixels\n")

# ============================================================================
# PHASE 1B: CONVERT TO BINARY (Black/White)
# ============================================================================

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to get binary image
# For colored objects on background, we use a mid-range threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Count white vs black pixels
white_pixels = np.sum(binary > 0)
black_pixels = np.sum(binary == 0)

print(f"✅ Converted to binary image")
print(f"   White pixels (objects): {white_pixels}")
print(f"   Black pixels (background): {black_pixels}")
print(f"   Object coverage: {(white_pixels / (gray.shape[0] * gray.shape[1]) * 100):.1f}%\n")

# ============================================================================
# PHASE 2: MORPHOLOGICAL OPERATIONS (CLEANUP)
# ============================================================================

# Define a kernel (small matrix for morphological operations)
# Larger kernel = more aggressive smoothing
kernel = np.ones((5, 5), np.uint8)

# OPENING: Erosion followed by Dilation
# Effect: Removes small noise while keeping large objects
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# CLOSING: Dilation followed by Erosion
# Effect: Fills small holes inside objects
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

white_after_cleanup = np.sum(cleaned > 0)

print(f"✅ Applied morphological operations")
print(f"   Kernel size: 5×5")
print(f"   White pixels after cleanup: {white_after_cleanup}")
print(f"   Noise removed: {white_pixels - white_after_cleanup} pixels\n")

# ============================================================================
# PHASE 3: FIND CONTOURS
# ============================================================================

# cv2.findContours() finds all white regions in the binary image
# Returns: (contours, hierarchy)
# - contours: list of all objects (each is a list of boundary points)
# - hierarchy: relationships between contours (parent/child)

contours, hierarchy = cv2.findContours(
    cleaned,
    cv2.RETR_TREE,              # Retrieve all contours (including nested)
    cv2.CHAIN_APPROX_SIMPLE     # Simplify contours (reduce points)
)

print(f"✅ Found {len(contours)} total contours\n")

# ============================================================================
# PHASE 4: FILTER & ANALYZE CONTOURS
# ============================================================================

# Configure size filtering
MIN_AREA = 500      # Minimum area to keep (pixels)
MAX_AREA = 100000   # Maximum area (filters very large noise)

# Store analyzed objects
objects = []

print(f"📊 Analyzing contours (filtering for area {MIN_AREA}-{MAX_AREA} pixels):\n")

for i, contour in enumerate(contours):
    # Calculate area
    area = cv2.contourArea(contour)
    
    # Skip if outside size range
    if area < MIN_AREA or area > MAX_AREA:
        continue
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate centroid (center point)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    
    # Calculate circularity (how round is it?)
    # Formula: 4π * area / perimeter²
    # Circle = 1.0, Square ≈ 0.785, very thin = close to 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Aspect ratio (width/height)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Store object info
    obj_info = {
        'contour': contour,
        'area': area,
        'perimeter': perimeter,
        'x': x, 'y': y, 'w': w, 'h': h,
        'centroid': (cx, cy),
        'circularity': circularity,
        'aspect_ratio': aspect_ratio
    }
    objects.append(obj_info)

print(f"✅ Filtered to {len(objects)} significant objects\n")

# ============================================================================
# PHASE 5: PRINT DETAILED STATISTICS
# ============================================================================

if len(objects) == 0:
    print("❌ No objects detected!")
    print("   → Try lowering MIN_AREA")
    print("   → Check if image has clear white objects on dark background")
else:
    print("="*70)
    print("📊 DETECTED OBJECTS - DETAILED ANALYSIS")
    print("="*70)
    
    for i, obj in enumerate(objects, 1):
        print(f"\n   Object #{i}:")
        print(f"      Area:          {int(obj['area'])} pixels")
        print(f"      Perimeter:     {int(obj['perimeter'])} pixels")
        print(f"      Position:      ({obj['x']}, {obj['y']}) (top-left)")
        print(f"      Bounding box:  {obj['w']} × {obj['h']} pixels")
        print(f"      Centroid:      {obj['centroid']}")
        print(f"      Circularity:   {obj['circularity']:.3f} (1.0=circle, 0=thin)")
        print(f"      Aspect ratio:  {obj['aspect_ratio']:.2f} (1.0=square)")
        
        # Describe shape
        if obj['circularity'] > 0.7:
            shape_desc = "🔵 Circular"
        elif 0.4 < obj['circularity'] <= 0.7:
            shape_desc = "⬜ Polygonal"
        else:
            shape_desc = "➖ Elongated/Thin"
        
        if obj['aspect_ratio'] > 2.5:
            shape_desc += " (very wide)"
        elif obj['aspect_ratio'] < 0.4:
            shape_desc += " (very tall)"
        
        print(f"      Shape:         {shape_desc}")
    
    print("\n" + "="*70)

# ============================================================================
# PHASE 6: VISUALIZE RESULTS
# ============================================================================

# Create a result image for drawing
result = image_rgb.copy()

# Assign different colors to different objects
colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
]

# Draw each contour
for i, obj in enumerate(objects):
    color = colors[i % len(colors)]
    contour = obj['contour']
    
    # Draw contour outline
    cv2.drawContours(result, [contour], 0, color, 2)
    
    # Draw bounding rectangle
    x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    
    # Draw centroid
    cx, cy = obj['centroid']
    cv2.circle(result, (cx, cy), 5, color, -1)
    
    # Add label with area
    label = f"#{i+1} ({int(obj['area'])}px)"
    cv2.putText(result, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

print(f"\n✅ Drew {len(objects)} contours on result image\n")

# ============================================================================
# PHASE 7: DISPLAY COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Original image
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Top-right: Binary image
axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('Binary Image (threshold)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Bottom-left: Cleaned binary
axes[1, 0].imshow(cleaned, cmap='gray')
axes[1, 0].set_title('After Morphological Cleanup', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Bottom-right: Result with contours
axes[1, 1].imshow(result)
axes[1, 1].set_title(f'Detected Objects ({len(objects)} found)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('contour_analysis_results.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization to: contour_analysis_results.png\n")
plt.show()

# ============================================================================
# PHASE 8: GENERATE SUMMARY REPORT
# ============================================================================

print("="*70)
print("📈 SUMMARY REPORT")
print("="*70)

if len(objects) > 0:
    areas = [obj['area'] for obj in objects]
    perimeters = [obj['perimeter'] for obj in objects]
    circularities = [obj['circularity'] for obj in objects]
    
    print(f"\nTotal objects detected: {len(objects)}")
    print(f"\nArea Statistics:")
    print(f"   Largest:  {max(areas):.0f} pixels")
    print(f"   Smallest: {min(areas):.0f} pixels")
    print(f"   Average:  {np.mean(areas):.0f} pixels")
    print(f"   Std Dev:  {np.std(areas):.0f} pixels")
    
    print(f"\nPerimeter Statistics:")
    print(f"   Largest:  {max(perimeters):.0f} pixels")
    print(f"   Smallest: {min(perimeters):.0f} pixels")
    print(f"   Average:  {np.mean(perimeters):.0f} pixels")
    
    print(f"\nCircularity Statistics:")
    print(f"   Most round:     {max(circularities):.3f}")
    print(f"   Most elongated: {min(circularities):.3f}")
    print(f"   Average:        {np.mean(circularities):.3f}")
    
    # Shape classification
    round_objects = sum(1 for obj in objects if obj['circularity'] > 0.7)
    polygon_objects = sum(1 for obj in objects if 0.4 < obj['circularity'] <= 0.7)
    thin_objects = sum(1 for obj in objects if obj['circularity'] <= 0.4)
    
    print(f"\nShape Classification:")
    print(f"   🔵 Circular:    {round_objects} objects")
    print(f"   ⬜ Polygonal:   {polygon_objects} objects")
    print(f"   ➖ Elongated:   {thin_objects} objects")

print("\n" + "="*70)

# ============================================================================
# PHASE 9: ADVANCED - FIND CIRCLES VS RECTANGLES
# ============================================================================

print(f"\n🎯 BONUS: Shape Detection\n")

circles = []
rectangles = []

for i, obj in enumerate(objects):
    if obj['circularity'] > 0.7:
        circles.append(i)
    elif obj['aspect_ratio'] > 1.5 or obj['aspect_ratio'] < 0.67:
        rectangles.append(i)

if circles:
    print(f"   🔵 Circles detected: {circles}")
if rectangles:
    print(f"   ⬜ Rectangles detected: {rectangles}")

print("\n" + "="*70)