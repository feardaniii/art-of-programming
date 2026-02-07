import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# ============================================================================
# PHASE 1: LOAD IMAGE & CONVERT TO GRAYSCALE
# ============================================================================

print("🔍 Looking for image files...")

image_path = None
for file in os.listdir('.'):
    if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
        image_path = file
        print(f"✅ Found image file: {image_path}")
        break

if image_path is None:
    print("❌ Error: No image files found!")
    print("   Please place a text image (.jpg, .png, etc.) in this folder")
    print("   Ideas: old book page, receipt, handwritten note, whiteboard")
    exit()

# Load the image
print(f"📸 Loading: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Error: Could not load image '{image_path}'")
    exit()

print(f"✅ Image loaded! Shape: {image.shape}\n")

# Convert to grayscale (single channel, easier to threshold)
# Grayscale: each pixel is a number 0-255 (black to white)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"✅ Converted to grayscale")
print(f"   Value range: {gray.min()} to {gray.max()}")
print(f"   Mean brightness: {gray.mean():.1f}\n")

# ============================================================================
# PHASE 2: APPLY GLOBAL THRESHOLD
# ============================================================================

# Global threshold: one threshold value for entire image
# Syntax: cv2.threshold(image, threshold_value, max_value, method)
# Returns: (threshold_value, result_image)

GLOBAL_THRESHOLD = 127  # Middle of 0-255 range

_, global_thresh = cv2.threshold(
    gray,                      # Input image
    GLOBAL_THRESHOLD,          # Threshold value (0-255)
    255,                       # Max value (what white pixels become)
    cv2.THRESH_BINARY          # Method (black or white, no gray)
)

print(f"✅ Applied global threshold")
print(f"   Threshold value: {GLOBAL_THRESHOLD}")
print(f"   White pixels: {np.sum(global_thresh > 0)}")
print(f"   Black pixels: {np.sum(global_thresh == 0)}\n")

# ============================================================================
# PHASE 3: APPLY ADAPTIVE THRESHOLD (GAUSSIAN METHOD)
# ============================================================================

# Adaptive threshold - Gaussian:
# For each pixel, looks at 11x11 neighborhood
# Compares pixel to weighted average (Gaussian) of that neighborhood
# C = constant subtracted from mean (fine-tuning)

adaptive_thresh = cv2.adaptiveThreshold(
    gray,                              # Input image
    255,                               # Max value (white)
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,    # Method: Gaussian-weighted mean
    cv2.THRESH_BINARY,                 # Output type: black/white only
    11,                                # Block size (must be odd: 3,5,7,11,15...)
    2                                  # C constant (subtracted from mean)
)

print(f"✅ Applied adaptive threshold (Gaussian)")
print(f"   Block size: 11×11 pixels")
print(f"   C constant: 2")
print(f"   White pixels: {np.sum(adaptive_thresh > 0)}")
print(f"   Black pixels: {np.sum(adaptive_thresh == 0)}\n")

# ============================================================================
# PHASE 4: APPLY ADAPTIVE THRESHOLD (MEAN METHOD)
# ============================================================================

# Adaptive threshold - Mean:
# Similar to Gaussian, but uses simple average instead of weighted average
# Usually produces slightly different results, good to compare

adaptive_mean = cv2.adaptiveThreshold(
    gray,                           # Input image
    255,                            # Max value
    cv2.ADAPTIVE_THRESH_MEAN_C,     # Method: simple mean
    cv2.THRESH_BINARY,              # Output type
    11,                             # Block size
    2                               # C constant
)

print(f"✅ Applied adaptive threshold (Mean)")
print(f"   Block size: 11×11 pixels")
print(f"   C constant: 2")
print(f"   White pixels: {np.sum(adaptive_mean > 0)}")
print(f"   Black pixels: {np.sum(adaptive_mean == 0)}\n")

# ============================================================================
# PHASE 5: VISUALIZE & COMPARE ALL VERSIONS
# ============================================================================

# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Top-left: Original grayscale
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('Original Grayscale Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Top-right: Global threshold
axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title(f'Global Threshold (value={GLOBAL_THRESHOLD})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Bottom-left: Adaptive threshold (Gaussian)
axes[1, 0].imshow(adaptive_thresh, cmap='gray')
axes[1, 0].set_title('Adaptive Threshold (Gaussian)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Bottom-right: Adaptive threshold (Mean)
axes[1, 1].imshow(adaptive_mean, cmap='gray')
axes[1, 1].set_title('Adaptive Threshold (Mean)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('text_enhancement_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved comparison to: text_enhancement_comparison.png\n")
plt.show()

# ============================================================================
# PHASE 6: CALCULATE READABILITY SCORE
# ============================================================================

def calculate_readability(binary_image):
    """
    Higher score = better text/background separation
    
    Logic: We want a balanced split between black and white
    - If image is 90% white, text is lost
    - If image is 90% black, background is lost
    - Ideal: close to 50/50 split
    
    Score range: 0-100 (100 = perfect 50/50 balance)
    """
    white_pixels = np.sum(binary_image == 255)
    black_pixels = np.sum(binary_image == 0)
    
    # Ratio of minority to majority color
    ratio = min(white_pixels, black_pixels) / max(white_pixels, black_pixels)
    
    # Convert to 0-100 scale
    return ratio * 100

# Calculate scores
global_score = calculate_readability(global_thresh)
adaptive_gauss_score = calculate_readability(adaptive_thresh)
adaptive_mean_score = calculate_readability(adaptive_mean)

print("="*60)
print("📊 READABILITY SCORES (0-100, higher = better separation)")
print("="*60)
print(f"   Global threshold:      {global_score:.1f}/100")
print(f"   Adaptive (Gaussian):   {adaptive_gauss_score:.1f}/100")
print(f"   Adaptive (Mean):       {adaptive_mean_score:.1f}/100")
print("="*60)

# Recommend best method
best_method = max(
    ("Global", global_score),
    ("Adaptive (Gaussian)", adaptive_gauss_score),
    ("Adaptive (Mean)", adaptive_mean_score),
    key=lambda x: x[1]
)

print(f"\n🏆 Best method: {best_method[0]} with score {best_method[1]:.1f}")

# ============================================================================
# PHASE 7: OPTIONAL - APPLY MORPHOLOGICAL OPERATIONS TO CLEAN UP
# ============================================================================

print(f"\n✨ Applying morphological operations for extra cleanup...\n")

# Define a small kernel (3x3 square of ones)
kernel = np.ones((3, 3), np.uint8)

# Morphological closing: removes small black noise
# (fills small holes in text)
cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

cleaned_score = calculate_readability(cleaned)

print(f"✅ Applied CLOSE and OPEN operations")
print(f"   Cleaned readability score: {cleaned_score:.1f}/100")

# Display cleaned version
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title('Adaptive Threshold (before cleanup)', fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cleaned, cmap='gray')
plt.title('After Morphological Cleanup', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('text_enhancement_with_cleanup.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved cleanup comparison to: text_enhancement_with_cleanup.png\n")
plt.show()

# ============================================================================
# PHASE 8: PRINT SUMMARY
# ============================================================================

print("="*60)
print("✅ TEXT ENHANCEMENT ANALYSIS COMPLETE")
print("="*60)
print(f"""
Key Findings:

1. GLOBAL THRESHOLD ({global_score:.1f}/100)
   - Fast and simple
   - Works well for evenly-lit documents
   - Struggles with shadows and uneven lighting

2. ADAPTIVE THRESHOLD - GAUSSIAN ({adaptive_gauss_score:.1f}/100)
   - Smarter: adapts to local brightness
   - Better for old/damaged documents
   - Gaussian weighting emphasizes nearby pixels

3. ADAPTIVE THRESHOLD - MEAN ({adaptive_mean_score:.1f}/100)
   - Also adaptive, simpler calculation
   - Similar to Gaussian but slightly different results

4. WITH MORPHOLOGICAL CLEANUP ({cleaned_score:.1f}/100)
   - Removes small noise
   - Fills holes in letters
   - Extra processing step for best quality""")
print("="*60)