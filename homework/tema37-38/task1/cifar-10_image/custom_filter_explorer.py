import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tensorflow.keras.datasets import cifar10
from pathlib import Path


def rgb_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminance weights."""
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.float32)

    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[..., :3]

    return np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Scale image values to [0, 1] for stable visualization."""
    min_value = image.min()
    max_value = image.max()
    if max_value - min_value < 1e-8:
        return np.zeros_like(image)
    return (image - min_value) / (max_value - min_value)


def ask_image_source() -> tuple[np.ndarray, str]:
    print("Choose image source:")
    print("1) CIFAR-10 by index")
    print("2) Local image path (inside your working folder)")
    choice = input("Type 1 or 2 (default 1): ").strip()

    if choice == "2":
        path_text = input("Enter image path (e.g. images/sample.png): ").strip()
        path_text = path_text.strip("\"'")
        image_path = Path(path_text)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        image = plt.imread(image_path)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image, f"Local image: {image_path}"

    index_text = input("Enter CIFAR-10 train index (default 0): ").strip()
    image_index = int(index_text) if index_text else 0
    (x_train, _), _ = cifar10.load_data()
    if image_index < 0 or image_index >= len(x_train):
        raise ValueError(f"CIFAR-10 index out of range: {image_index}")
    image = x_train[image_index].astype(np.float32) / 255.0
    return image, f"CIFAR-10 train index: {image_index}"


def main() -> None:
    image_rgb, source_description = ask_image_source()
    image_gray = rgb_to_grayscale(image_rgb)

    # 1) Diagonal edge detector (/) - custom 3x3, not Sobel.
    diagonal_filter = np.array(
        [
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2],
        ],
        dtype=np.float32,
    )

    # 2) Blur filter (mean filter).
    blur_filter = np.ones((3, 3), dtype=np.float32) / 9.0

    # 3) Custom design filter (emboss-like effect).
    custom_filter = np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    )

    # Apply filters using convolve2d.
    diagonal_output = convolve2d(image_gray, diagonal_filter, mode="same", boundary="symm")
    blur_output = convolve2d(image_gray, blur_filter, mode="same", boundary="symm")
    custom_output = convolve2d(image_gray, custom_filter, mode="same", boundary="symm")

    # 4-panel visualization: original + 3 filtered results.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[1].imshow(normalize_for_display(diagonal_output), cmap="gray")
    axes[1].set_title("Diagonal edges (/)")
    axes[2].imshow(normalize_for_display(blur_output), cmap="gray")
    axes[2].set_title("Blur (mean 3x3)")
    axes[3].imshow(normalize_for_display(custom_output), cmap="gray")
    axes[3].set_title("Custom emboss-like")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_name = "tema37-38_task1_custom_filters.png"
    plt.savefig(output_name, dpi=150)
    plt.close()

    # Written explanations (2-3 sentences each), printed at the end.
    print("=== TASK 1: CUSTOM FILTER EXPLORER ===")
    print(f"Image source: {source_description}")
    print(f"Saved output PNG: {output_name}")
    print()

    print("1) Diagonal edge detector (/):")
    print(
        "This filter highlights intensity transitions aligned on a diagonal "
        "top-right to bottom-left direction. Uniform regions produce values "
        "close to zero, while diagonal boundaries become bright or dark responses."
    )
    print(
        "It is useful for emphasizing slanted contours that are less visible "
        "with simple horizontal/vertical detectors."
    )
    print()

    print("2) Blur (mean) filter:")
    print(
        "This filter replaces each pixel with the average of its 3x3 neighborhood, "
        "which smooths high-frequency details and local noise. Sharp edges become softer "
        "because neighboring values are blended together."
    )
    print(
        "The result keeps the global object shape but reduces fine texture detail."
    )
    print()

    print("3) Custom filter (emboss-like):")
    print(
        "This custom kernel amplifies directional contrast and creates a raised, "
        "relief-like appearance. Areas with gradual intensity change look flat, "
        "while directional transitions are emphasized strongly."
    )
    print(
        "It behaves like a stylized edge enhancer and is good for visual texture emphasis."
    )


if __name__ == "__main__":
    main()
