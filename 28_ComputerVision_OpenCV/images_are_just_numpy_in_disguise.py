import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from your health tracking app (e.g., meal photo)
meal_image = cv2.imread('data-calculus-hero.jpg')
meal_image_2 = cv2.imread('dubai-chocolate.jpg')


# OpenCV reads in BGR, but we think in RGB
meal_rgb = cv2.cvtColor(meal_image, cv2.COLOR_BGR2RGB)

print(f"📸 Image shape: {meal_rgb.shape}")  # e.g., (1080, 1920, 3)
print(f"   Height: {meal_rgb.shape[0]} pixels")
print(f"   Width: {meal_rgb.shape[1]} pixels")
print(f"   Channels: {meal_rgb.shape[2]} (Red, Green, Blue)")
print(f"   Total data points: {meal_rgb.size} numbers!")

# This is just NumPy - everything you learned in Data:Calculus!
print(f"\n🔢 Data type: {meal_rgb.dtype}")  # uint8 (0-255)
print(f"   Memory: {meal_rgb.nbytes / 1024 / 1024:.2f} MB")

# Extract a single pixel - it's just a vector!
center_pixel = meal_rgb[540, 960]  # middle of 1080p image
print(f"\n🎨 Center pixel RGB: {center_pixel}")

# ====== The "AHA!" Moment: Different Color Spaces ======
# RGB: What humans see and cameras capture
rgb_image = meal_rgb.copy()

# Grayscale: Intensity only (great for detecting shapes)
gray_image = cv2.cvtColor(meal_rgb, cv2.COLOR_RGB2GRAY)
print(f"\n🌓 Grayscale shape: {gray_image.shape}")  # (1080, 1920) - 2D only!

# HSV: Hue-Saturation-Value (great for color-based segmentation)
hsv_image = cv2.cvtColor(meal_rgb, cv2.COLOR_RGB2HSV)

# ====== Real Health App Use Case ======
# Heuristic: Analyze color variety in meal - more colors = better nutrition
def analyze_meal_variety(image_rgb):
    """Analyze color variety in meal - more colors = better nutrition"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue_channel = hsv[:, :, 0]

    unique_hues = len(np.unique(hue_channel))
    variety_score = min(100, (unique_hues / 180) * 100)

    return {
        'variety_score': variety_score,
        'unique_colors': unique_hues,
        'recommendation': '✅ Great variety!' if variety_score > 60 else '⚠️ Add more colors'
    }

meal_analysis = analyze_meal_variety(meal_rgb)
print(f"\n🥗 Meal Nutrition Analysis:")
print(f"   Color Variety Score: {meal_analysis['variety_score']:.1f}/100")
print(f"   {meal_analysis['recommendation']}")

print("\n💡 See? Everything connects to Data:Calculus!")