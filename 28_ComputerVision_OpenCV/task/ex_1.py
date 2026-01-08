import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (replace with your own!)
image = cv2.imread('your_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"📸 Image loaded: {image_rgb.shape}")

# TODO 1: Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# TODO 2: Define blue color range in HSV
# Blue in HSV: Hue ~100-130, high saturation, decent value
lower_blue = np.array([100, 50, 50])    # Adjust these!
upper_blue = np.array([130, 255, 255])  # Adjust these!

# TODO 3: Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# TODO 4: Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# TODO 5: Filter and draw
result = image_rgb.copy()
detected_count = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 500:  # Filter small noise
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw on result
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result, f"Blue #{detected_count+1}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detected_count += 1

        print(f"   Object #{detected_count}: {int(area)} pixels at ({x}, {y})")

print(f"\n✅ Detected {detected_count} blue objects!")

# TODO 6: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Blue Mask (White = Blue pixels)')
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title(f'Detection: {detected_count} blue objects')
axes[2].axis('off')

plt.tight_layout()
plt.show()