"""
================================================================================
DATA AUGMENTATION FOR OBJECT DETECTION
================================================================================

This module implements image augmentation techniques used in training
real-world computer vision models.

WHY AUGMENT?
------------
Your camera captures images in specific conditions. But your model will
encounter endless variations:
    - Different lighting (bright sun, dim room, shadows)
    - Motion blur (shaky hands, moving objects)
    - Various angles (not everything is perfectly aligned)
    - Partial occlusion (objects behind other objects)
    - Sensor noise (low-light photography)

Augmentation artificially creates these variations from your original images,
making your model robust to real-world conditions.

THE KEY INSIGHT:
----------------
When you augment an image, you must ALSO transform the bounding boxes.
If you rotate an image 15°, the bounding box coordinates must also rotate.
This is what makes object detection augmentation trickier than classification.

================================================================================
"""

import cv2
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


# =============================================================================
# BOUNDING BOX UTILITIES
# =============================================================================

@dataclass
class BoundingBox:
    """
    Represents a bounding box with its class.
    
    Coordinates are in PIXEL values (not normalized).
    Use to_yolo() and from_yolo() to convert.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_yolo(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format string (normalized coordinates)."""
        x_center = (self.x_min + self.x_max) / 2 / img_width
        y_center = (self.y_min + self.y_max) / 2 / img_height
        width = self.width / img_width
        height = self.height / img_height
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    @classmethod
    def from_yolo(cls, line: str, img_width: int, img_height: int) -> 'BoundingBox':
        """Create from YOLO format string."""
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        return cls(
            x_min=x_center - width / 2,
            y_min=y_center - height / 2,
            x_max=x_center + width / 2,
            y_max=y_center + height / 2,
            class_id=class_id
        )
    
    def clip(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Clip box to image boundaries."""
        return BoundingBox(
            x_min=max(0, min(self.x_min, img_width)),
            y_min=max(0, min(self.y_min, img_height)),
            x_max=max(0, min(self.x_max, img_width)),
            y_max=max(0, min(self.y_max, img_height)),
            class_id=self.class_id
        )
    
    def is_valid(self, min_area: float = 100, min_visibility: float = 0.3) -> bool:
        """Check if box is still valid after transformation."""
        return self.area >= min_area and self.width > 5 and self.height > 5


# =============================================================================
# GEOMETRIC AUGMENTATIONS
# =============================================================================

class GeometricAugmentations:
    """
    Transformations that change the spatial arrangement of pixels.
    
    These require updating bounding box coordinates.
    """
    
    @staticmethod
    def horizontal_flip(
        image: np.ndarray, 
        boxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Flip image horizontally (mirror).
        
        Common augmentation because most objects look similar when mirrored.
        Exception: Text, asymmetric objects.
        
        Box transformation:
            new_x_min = img_width - old_x_max
            new_x_max = img_width - old_x_min
        """
        h, w = image.shape[:2]
        
        # Flip image
        flipped = cv2.flip(image, 1)  # 1 = horizontal
        
        # Transform boxes
        new_boxes = []
        for box in boxes:
            new_boxes.append(BoundingBox(
                x_min=w - box.x_max,
                y_min=box.y_min,
                x_max=w - box.x_min,
                y_max=box.y_max,
                class_id=box.class_id
            ))
        
        return flipped, new_boxes
    
    @staticmethod
    def vertical_flip(
        image: np.ndarray, 
        boxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Flip image vertically.
        
        Less common - only use if your objects can appear upside down.
        """
        h, w = image.shape[:2]
        
        flipped = cv2.flip(image, 0)  # 0 = vertical
        
        new_boxes = []
        for box in boxes:
            new_boxes.append(BoundingBox(
                x_min=box.x_min,
                y_min=h - box.y_max,
                x_max=box.x_max,
                y_max=h - box.y_min,
                class_id=box.class_id
            ))
        
        return flipped, new_boxes
    
    @staticmethod
    def rotate(
        image: np.ndarray, 
        boxes: List[BoundingBox],
        angle: float
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Rotate image by angle (degrees).
        
        This is more complex because:
        1. Rotated rectangle becomes a larger bounding box
        2. Some of the object might go outside the image
        
        We rotate around the image center and keep the same image size.
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Transform boxes
        new_boxes = []
        for box in boxes:
            # Get all 4 corners
            corners = np.array([
                [box.x_min, box.y_min],
                [box.x_max, box.y_min],
                [box.x_max, box.y_max],
                [box.x_min, box.y_max]
            ])
            
            # Rotate corners
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])  # Homogeneous coordinates
            rotated_corners = M @ corners_h.T  # Apply transformation
            rotated_corners = rotated_corners.T
            
            # Get new bounding box (axis-aligned)
            new_x_min = rotated_corners[:, 0].min()
            new_x_max = rotated_corners[:, 0].max()
            new_y_min = rotated_corners[:, 1].min()
            new_y_max = rotated_corners[:, 1].max()
            
            new_box = BoundingBox(
                x_min=new_x_min,
                y_min=new_y_min,
                x_max=new_x_max,
                y_max=new_y_max,
                class_id=box.class_id
            ).clip(w, h)
            
            if new_box.is_valid():
                new_boxes.append(new_box)
        
        return rotated, new_boxes
    
    @staticmethod
    def scale(
        image: np.ndarray, 
        boxes: List[BoundingBox],
        scale_factor: float
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Scale image (zoom in/out).
        
        scale_factor > 1: Zoom in (objects appear larger)
        scale_factor < 1: Zoom out (objects appear smaller)
        
        We scale around the center and crop/pad to original size.
        """
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if scale_factor > 1:
            # Zoom in: crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = resized[start_y:start_y+h, start_x:start_x+w]
            offset_x, offset_y = -start_x, -start_y
        else:
            # Zoom out: pad with border
            result = np.zeros_like(image)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            offset_x, offset_y = start_x, start_y
        
        # Transform boxes
        new_boxes = []
        for box in boxes:
            new_box = BoundingBox(
                x_min=box.x_min * scale_factor + offset_x,
                y_min=box.y_min * scale_factor + offset_y,
                x_max=box.x_max * scale_factor + offset_x,
                y_max=box.y_max * scale_factor + offset_y,
                class_id=box.class_id
            ).clip(w, h)
            
            if new_box.is_valid():
                new_boxes.append(new_box)
        
        return result, new_boxes


# =============================================================================
# COLOR/PHOTOMETRIC AUGMENTATIONS
# =============================================================================

class ColorAugmentations:
    """
    Transformations that change pixel values without moving them.
    
    These do NOT require updating bounding boxes.
    They simulate different lighting conditions and camera settings.
    """
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        factor > 0: Brighter (simulate sunny conditions)
        factor < 0: Darker (simulate dim room)
        
        Range: typically -0.3 to +0.3
        """
        # Convert to float for arithmetic
        img_float = image.astype(np.float32) / 255.0
        
        # Adjust brightness
        img_float = img_float + factor
        
        # Clip and convert back
        img_float = np.clip(img_float, 0, 1)
        return (img_float * 255).astype(np.uint8)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        factor > 1: Higher contrast (more vivid)
        factor < 1: Lower contrast (more washed out)
        
        Range: typically 0.7 to 1.3
        """
        img_float = image.astype(np.float32) / 255.0
        mean = img_float.mean()
        
        # Contrast adjustment: move pixels away from or toward mean
        img_float = (img_float - mean) * factor + mean
        
        img_float = np.clip(img_float, 0, 1)
        return (img_float * 255).astype(np.uint8)
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust color saturation.
        
        factor > 1: More colorful
        factor < 1: More gray
        factor = 0: Grayscale
        
        Range: typically 0.7 to 1.3
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation channel
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_hue(image: np.ndarray, shift: float) -> np.ndarray:
        """
        Shift hue (color tone).
        
        Simulates different white balance or color temperature.
        
        shift: degrees to shift (-180 to 180)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue is in range 0-179 in OpenCV
        hsv[:, :, 0] = (hsv[:, :, 0] + shift / 2) % 180
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# =============================================================================
# BLUR AUGMENTATIONS
# =============================================================================

class BlurAugmentations:
    """
    Blur effects that simulate camera issues.
    
    These help your model work with imperfect images.
    """
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Apply Gaussian blur (general softness).
        
        Simulates:
        - Out of focus camera
        - Low quality lens
        - Distance blur
        
        kernel_size: Must be odd (3, 5, 7, etc.). Larger = more blur.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def motion_blur(image: np.ndarray, kernel_size: int, angle: float = 0) -> np.ndarray:
        """
        Apply motion blur (directional streak).
        
        Simulates:
        - Camera shake
        - Moving object
        - Panning shot
        
        kernel_size: Length of blur
        angle: Direction of motion (degrees)
        """
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel = kernel / kernel_size
        
        # Rotate kernel for different angles
        if angle != 0:
            M = cv2.getRotationMatrix2D(
                (kernel_size / 2, kernel_size / 2), 
                angle, 
                1.0
            )
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()  # Renormalize
        
        return cv2.filter2D(image, -1, kernel)


# =============================================================================
# NOISE AUGMENTATIONS
# =============================================================================

class NoiseAugmentations:
    """
    Add noise to simulate sensor and compression artifacts.
    """
    
    @staticmethod
    def gaussian_noise(image: np.ndarray, variance: float) -> np.ndarray:
        """
        Add Gaussian (random) noise.
        
        Simulates:
        - High ISO camera settings
        - Low light photography
        - Cheap camera sensors
        
        variance: Noise intensity (0.01 to 0.1 typical)
        """
        img_float = image.astype(np.float32) / 255.0
        
        noise = np.random.normal(0, variance ** 0.5, img_float.shape)
        noisy = img_float + noise
        
        noisy = np.clip(noisy, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    @staticmethod
    def jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
        """
        Simulate JPEG compression artifacts.
        
        Simulates:
        - Images from the web
        - Compressed storage
        - Low bandwidth transmission
        
        quality: JPEG quality (0-100, lower = more artifacts)
        """
        # Encode to JPEG in memory
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        
        # Decode back
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    @staticmethod
    def salt_and_pepper(image: np.ndarray, amount: float) -> np.ndarray:
        """
        Add salt-and-pepper noise (random white/black pixels).
        
        Simulates:
        - Dead pixels
        - Transmission errors
        - Sensor defects
        
        amount: Proportion of pixels to affect (0.01 to 0.05 typical)
        """
        noisy = image.copy()
        
        # Salt (white pixels)
        num_salt = int(amount * image.size / 2)
        coords = tuple([
            np.random.randint(0, i - 1, num_salt)
            for i in image.shape[:2]
        ])
        noisy[coords] = 255
        
        # Pepper (black pixels)
        num_pepper = int(amount * image.size / 2)
        coords = tuple([
            np.random.randint(0, i - 1, num_pepper)
            for i in image.shape[:2]
        ])
        noisy[coords] = 0
        
        return noisy


# =============================================================================
# OCCLUSION AUGMENTATIONS
# =============================================================================

class OcclusionAugmentations:
    """
    Simulate partial visibility of objects.
    
    These teach the model to detect objects even when parts are hidden.
    """
    
    @staticmethod
    def random_erasing(
        image: np.ndarray,
        boxes: List[BoundingBox],
        erase_ratio: Tuple[float, float] = (0.02, 0.1),
        aspect_ratio: Tuple[float, float] = (0.5, 2.0)
    ) -> np.ndarray:
        """
        Randomly erase rectangular regions.
        
        Simulates:
        - Objects partially behind other objects
        - Shadows
        - Obstructions
        
        erase_ratio: Size of erased region relative to image
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # Calculate area to erase
        area = h * w
        target_area = random.uniform(erase_ratio[0], erase_ratio[1]) * area
        
        # Random aspect ratio
        aspect = random.uniform(aspect_ratio[0], aspect_ratio[1])
        
        # Calculate dimensions
        erase_h = int((target_area * aspect) ** 0.5)
        erase_w = int((target_area / aspect) ** 0.5)
        
        if erase_h < h and erase_w < w:
            # Random position
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            
            # Fill with random color or gray
            result[top:top+erase_h, left:left+erase_w] = np.random.randint(
                0, 255, (erase_h, erase_w, 3), dtype=np.uint8
            )
        
        return result
    
    @staticmethod
    def cutout(
        image: np.ndarray,
        num_holes: int = 1,
        hole_size: int = 50
    ) -> np.ndarray:
        """
        Cut out square holes from the image.
        
        Simpler version of random erasing with fixed-size squares.
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            y1 = max(0, y - hole_size // 2)
            y2 = min(h, y + hole_size // 2)
            x1 = max(0, x - hole_size // 2)
            x2 = min(w, x + hole_size // 2)
            
            result[y1:y2, x1:x2] = 0  # Black
        
        return result


# =============================================================================
# COMBINED AUGMENTATION PIPELINE
# =============================================================================

class AugmentationPipeline:
    """
    Combines multiple augmentations into a single pipeline.
    
    This is what you actually use during training.
    """
    
    def __init__(self, config: dict):
        """
        Initialize with configuration dictionary.
        
        config should match the augmentation section of config.yaml
        """
        self.config = config
        self.geo = GeometricAugmentations()
        self.color = ColorAugmentations()
        self.blur = BlurAugmentations()
        self.noise = NoiseAugmentations()
        self.occlusion = OcclusionAugmentations()
    
    def apply(
        self, 
        image: np.ndarray, 
        boxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Apply random augmentations based on config probabilities.
        
        Returns augmented image and transformed boxes.
        """
        img = image.copy()
        current_boxes = boxes.copy()
        
        geo_cfg = self.config.get('geometric', {})
        color_cfg = self.config.get('color', {})
        blur_cfg = self.config.get('blur', {})
        noise_cfg = self.config.get('noise', {})
        occlusion_cfg = self.config.get('occlusion', {})
        
        # Geometric augmentations (modify boxes)
        if random.random() < geo_cfg.get('horizontal_flip', 0):
            img, current_boxes = self.geo.horizontal_flip(img, current_boxes)
        
        if random.random() < geo_cfg.get('vertical_flip', 0):
            img, current_boxes = self.geo.vertical_flip(img, current_boxes)
        
        if random.random() < geo_cfg.get('rotation_prob', 0):
            max_angle = geo_cfg.get('rotation_degrees', 15)
            angle = random.uniform(-max_angle, max_angle)
            img, current_boxes = self.geo.rotate(img, current_boxes, angle)
        
        if random.random() < geo_cfg.get('scale_prob', 0):
            scale_range = geo_cfg.get('scale_range', [0.8, 1.2])
            scale = random.uniform(scale_range[0], scale_range[1])
            img, current_boxes = self.geo.scale(img, current_boxes, scale)
        
        # Color augmentations (don't modify boxes)
        if random.random() < color_cfg.get('brightness_prob', 0):
            bright_range = color_cfg.get('brightness_range', [-0.2, 0.2])
            factor = random.uniform(bright_range[0], bright_range[1])
            img = self.color.adjust_brightness(img, factor)
        
        if random.random() < color_cfg.get('contrast_prob', 0):
            contrast_range = color_cfg.get('contrast_range', [0.8, 1.2])
            factor = random.uniform(contrast_range[0], contrast_range[1])
            img = self.color.adjust_contrast(img, factor)
        
        if random.random() < color_cfg.get('saturation_prob', 0):
            sat_range = color_cfg.get('saturation_range', [0.8, 1.2])
            factor = random.uniform(sat_range[0], sat_range[1])
            img = self.color.adjust_saturation(img, factor)
        
        if random.random() < color_cfg.get('hue_shift_prob', 0):
            hue_range = color_cfg.get('hue_shift_range', [-10, 10])
            shift = random.uniform(hue_range[0], hue_range[1])
            img = self.color.adjust_hue(img, shift)
        
        # Blur augmentations
        if random.random() < blur_cfg.get('gaussian_blur_prob', 0):
            kernel = random.choice([3, 5, 7])
            kernel = min(kernel, blur_cfg.get('gaussian_blur_limit', 5))
            img = self.blur.gaussian_blur(img, kernel)
        
        if random.random() < blur_cfg.get('motion_blur_prob', 0):
            kernel = random.choice([3, 5, 7])
            kernel = min(kernel, blur_cfg.get('motion_blur_limit', 7))
            angle = random.uniform(0, 360)
            img = self.blur.motion_blur(img, kernel, angle)
        
        # Noise augmentations
        if random.random() < noise_cfg.get('gaussian_noise_prob', 0):
            variance = noise_cfg.get('gaussian_noise_var', 0.02)
            img = self.noise.gaussian_noise(img, variance)
        
        if random.random() < noise_cfg.get('jpeg_compression_prob', 0):
            quality_range = noise_cfg.get('jpeg_quality_range', [70, 95])
            quality = random.randint(quality_range[0], quality_range[1])
            img = self.noise.jpeg_compression(img, quality)
        
        # Occlusion augmentations
        if random.random() < occlusion_cfg.get('random_erasing_prob', 0):
            ratio = occlusion_cfg.get('random_erasing_ratio', [0.02, 0.1])
            img = self.occlusion.random_erasing(img, current_boxes, ratio)
        
        return img, current_boxes


# =============================================================================
# DEMO: Visualize Augmentations
# =============================================================================

def demo_augmentations(image_path: str):
    """
    Demonstrate each augmentation type on an image.
    
    Run this to see what each augmentation does.
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create sample bounding box (center of image)
    h, w = img.shape[:2]
    boxes = [BoundingBox(w*0.3, h*0.3, w*0.7, h*0.7, 0)]
    
    # Apply each augmentation
    augmentations = [
        ("Original", img, boxes),
        ("H-Flip", *GeometricAugmentations.horizontal_flip(img, boxes)),
        ("Rotate 15°", *GeometricAugmentations.rotate(img, boxes, 15)),
        ("Scale 1.2x", *GeometricAugmentations.scale(img, boxes, 1.2)),
        ("Brightness +", ColorAugmentations.adjust_brightness(img, 0.3), boxes),
        ("Brightness -", ColorAugmentations.adjust_brightness(img, -0.3), boxes),
        ("Contrast +", ColorAugmentations.adjust_contrast(img, 1.3), boxes),
        ("Saturation -", ColorAugmentations.adjust_saturation(img, 0.5), boxes),
        ("Gaussian Blur", BlurAugmentations.gaussian_blur(img, 7), boxes),
        ("Motion Blur", BlurAugmentations.motion_blur(img, 15, 45), boxes),
        ("Gaussian Noise", NoiseAugmentations.gaussian_noise(img, 0.05), boxes),
        ("JPEG Compress", NoiseAugmentations.jpeg_compression(img, 30), boxes),
    ]
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax, (name, aug_img, aug_boxes) in zip(axes, augmentations):
        # Handle both BGR and RGB
        if len(aug_img.shape) == 3:
            display_img = aug_img.copy()
        else:
            display_img = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2RGB)
        
        # Draw boxes
        for box in aug_boxes:
            cv2.rectangle(
                display_img,
                (int(box.x_min), int(box.y_min)),
                (int(box.x_max), int(box.y_max)),
                (255, 0, 0), 2
            )
        
        ax.imshow(display_img)
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_demo.png', dpi=150)
    plt.show()
    print("Saved augmentation_demo.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        demo_augmentations(sys.argv[1])
    else:
        print("Usage: python augmentation.py <image_path>")
        print("This will demonstrate all augmentation types on the image.")
