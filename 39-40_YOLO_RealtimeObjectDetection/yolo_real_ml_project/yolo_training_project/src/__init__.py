"""
YOLOv8 Custom Training Project - Source Module

This module contains:
    - augmentation.py: Data augmentation techniques
    - utils.py: Helper functions
"""

from .augmentation import (
    BoundingBox,
    GeometricAugmentations,
    ColorAugmentations,
    BlurAugmentations,
    NoiseAugmentations,
    OcclusionAugmentations,
    AugmentationPipeline,
)

from .utils import (
    load_config,
    get_project_root,
    ensure_dir,
    get_image_files,
    print_header,
    print_step,
)
