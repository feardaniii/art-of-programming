"""
================================================================================
UTILITY FUNCTIONS
================================================================================

Common functions used throughout the project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is in src/, so parent is project root
    return Path(__file__).parent.parent


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """Get all image files in a directory."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()


def print_step(step_num: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)
