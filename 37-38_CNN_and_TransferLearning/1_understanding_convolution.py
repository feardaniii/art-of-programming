import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==========================================
# THE PROBLEM: Why Traditional Neural Networks Fail at Vision
# ==========================================

def demonstrate_parameter_explosion():
    """
    A 224x224 color image has 224 × 224 × 3 = 150,528 pixels

    If we connect this to just 1000 neurons:
    Weights needed = 150,528 × 1,000 = 150 MILLION parameters

    For a simple 3-layer network:
    Layer 1: 150M parameters
    Layer 2: 1M parameters  (1000 × 1000)
    Layer 3: 1M parameters

    Total: ~152 MILLION parameters for ONE small image classifier

    This is:
    - Computationally impossible to train
    - Requires massive memory
    - Overfits immediately (more parameters than training samples)
    - Ignores spatial structure (pixel at (10,10) has no relationship to (10,11))
    """

    print("=" * 70)
    print("THE PARAMETER EXPLOSION PROBLEM")
    print("=" * 70)
    print()

    image_sizes = [28, 64, 128, 224, 512]
    hidden_size = 1000

    print(f"{'Image Size':<15} {'Total Pixels':<15} {'Parameters':<20} {'Memory (GB)':<15}")
    print("-" * 70)

    for size in image_sizes:
        pixels = size * size * 3  # RGB
        params = pixels * hidden_size
        memory_gb = params * 4 / (1024**3)  # 4 bytes per float32

        print(f"{size}x{size}x3 {pixels:>12,} {params:>18,} {memory_gb:>12.2f}")

    print()
    print("💡 THE INSIGHT:")
    print("What if nearby pixels share the same detector?")
    print("What if we use the SAME filter across the whole image?")
    print("This is CONVOLUTION. This is weight sharing. This is CNNs.")
    print()

demonstrate_parameter_explosion()

# ==========================================
# CONVOLUTION: The Sliding Window That Detects Patterns
# ==========================================

def visualize_convolution_operation():
    """
    Convolution = Sliding a small filter over an image

    Each filter detects ONE pattern (edge, corner, texture)
    The SAME filter slides across the ENTIRE image
    This is weight sharing: one detector, many locations
    """

    print("=" * 70)
    print("CONVOLUTION OPERATION: Edge Detection Example")
    print("=" * 70)
    print()

    # Simple 5x5 image
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255]
    ])

    # Vertical edge detector (Sobel-like)
    vertical_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Horizontal edge detector
    horizontal_filter = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Convolve
    vertical_edges = convolve2d(image, vertical_filter, mode='valid')
    horizontal_edges = convolve2d(image, horizontal_filter, mode='valid')

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Vertical filter
    im = axes[0, 1].imshow(vertical_filter, cmap='RdBu', vmin=-2, vmax=2)
    axes[0, 1].set_title('Vertical Edge Filter\n(3x3 kernel)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])

    # Vertical edges detected
    axes[0, 2].imshow(vertical_edges, cmap='hot')
    axes[0, 2].set_title('Vertical Edges Detected', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Horizontal filter
    im = axes[1, 1].imshow(horizontal_filter, cmap='RdBu', vmin=-2, vmax=2)
    axes[1, 1].set_title('Horizontal Edge Filter\n(3x3 kernel)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])

    # Horizontal edges detected
    axes[1, 2].imshow(horizontal_edges, cmap='hot')
    axes[1, 2].set_title('Horizontal Edges Detected', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Hide unused subplot
    axes[1, 0].axis('off')

    plt.tight_layout()
    plt.savefig('convolution_edge_detection.png', dpi=300, bbox_inches='tight')
    print("✅ Convolution visualization saved: convolution_edge_detection.png")
    print()

    print("🔍 WHAT JUST HAPPENED:")
    print("1. Filter slides across image (stride = 1)")
    print("2. At each position: element-wise multiply + sum")
    print("3. Result: Feature map showing where edges exist")
    print("4. ONE filter, MANY locations (weight sharing!)")
    print()

    print("📐 PARAMETER COUNT:")
    print(f"   Traditional NN: {image.size * 9} parameters (fully connected)")
    print(f"   CNN: {vertical_filter.size} parameters (one 3x3 filter)")
    print(f"   Reduction: {image.size * 9 / vertical_filter.size:.0f}x fewer parameters!")
    print()

visualize_convolution_operation()

# ==========================================
# BUILDING A SIMPLE CNN FROM SCRATCH
# ==========================================

class SimpleCNN:
    """
    A minimal CNN to understand the architecture

    Architecture:
    Input (28x28x1)
      → Conv1 (3x3, 8 filters) → ReLU → (26x26x8)
      → MaxPool (2x2) → (13x13x8)
      → Conv2 (3x3, 16 filters) → ReLU → (11x11x16)
      → MaxPool (2x2) → (5x5x16)
      → Flatten → (400)
      → Dense (10) → Softmax

    This is LeNet-5 architecture (1998) simplified.
    """

    def __init__(self):
        # Conv layer 1: 8 filters of size 3x3x1
        self.conv1_filters = np.random.randn(8, 3, 3, 1) * 0.1
        self.conv1_bias = np.zeros(8)

        # Conv layer 2: 16 filters of size 3x3x8
        self.conv2_filters = np.random.randn(16, 3, 3, 8) * 0.1
        self.conv2_bias = np.zeros(16)

        # Dense layer: 400 → 10
        self.dense_weights = np.random.randn(400, 10) * 0.1
        self.dense_bias = np.zeros(10)

    def conv2d(self, image, filters, bias, stride=1):
        """
        Convolution operation

        For each filter:
          Slide across image
          At each position: element-wise multiply + sum
          Add bias
          Apply ReLU
        """
        h, w, c = image.shape
        n_filters, fh, fw, _ = filters.shape

        out_h = (h - fh) // stride + 1
        out_w = (w - fw) // stride + 1

        output = np.zeros((out_h, out_w, n_filters))

        for f in range(n_filters):
            for i in range(0, out_h):
                for j in range(0, out_w):
                    # Extract patch
                    patch = image[i*stride:i*stride+fh, j*stride:j*stride+fw, :]

                    # Convolve: element-wise multiply + sum
                    output[i, j, f] = np.sum(patch * filters[f]) + bias[f]

        return output

    def relu(self, x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    def max_pool(self, x, pool_size=2):
        """
        Max pooling: Take maximum value in each pool_size × pool_size window

        Reduces spatial dimensions while preserving dominant features
        Makes network invariant to small translations
        """
        h, w, c = x.shape
        out_h = h // pool_size
        out_w = w // pool_size

        output = np.zeros((out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                for ch in range(c):
                    patch = x[i*pool_size:(i+1)*pool_size,
                             j*pool_size:(j+1)*pool_size,
                             ch]
                    output[i, j, ch] = np.max(patch)

        return output

    def softmax(self, x):
        """Softmax for multi-class classification"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, x):
        """
        Forward pass through the network

        This is the exact same flow as AlexNet, ResNet, EfficientNet
        Just different number of layers and filters
        """

        # Input shape: (28, 28, 1)
        print(f"Input shape: {x.shape}")

        # Conv layer 1
        conv1 = self.conv2d(x, self.conv1_filters, self.conv1_bias)
        conv1 = self.relu(conv1)
        print(f"After Conv1 + ReLU: {conv1.shape}")

        # Max pool 1
        pool1 = self.max_pool(conv1, pool_size=2)
        print(f"After MaxPool1: {pool1.shape}")

        # Conv layer 2
        conv2 = self.conv2d(pool1, self.conv2_filters, self.conv2_bias)
        conv2 = self.relu(conv2)
        print(f"After Conv2 + ReLU: {conv2.shape}")

        # Max pool 2
        pool2 = self.max_pool(conv2, pool_size=2)
        print(f"After MaxPool2: {pool2.shape}")

        # Flatten
        flattened = pool2.reshape(-1)
        print(f"After Flatten: {flattened.shape}")

        # Dense layer
        dense = flattened @ self.dense_weights + self.dense_bias
        print(f"After Dense: {dense.shape}")

        # Softmax
        output = self.softmax(dense)
        print(f"After Softmax: {output.shape}")

        return output

# Test the CNN
print()
print("=" * 70)
print("SIMPLE CNN FORWARD PASS")
print("=" * 70)
print()

# Create a random 28x28 image
test_image = np.random.randn(28, 28, 1)

# Create CNN
cnn = SimpleCNN()

# Forward pass
predictions = cnn.forward(test_image)

print()
print("Output probabilities (10 classes):")
for i, prob in enumerate(predictions):
    print(f"  Class {i}: {prob:.4f}")

print()
print("🎯 PARAMETER EFFICIENCY:")
conv1_params = 8 * 3 * 3 * 1 + 8  # filters + biases
conv2_params = 16 * 3 * 3 * 8 + 16
dense_params = 400 * 10 + 10
total_params = conv1_params + conv2_params + dense_params

print(f"  Conv1 parameters: {conv1_params:,}")
print(f"  Conv2 parameters: {conv2_params:,}")
print(f"  Dense parameters: {dense_params:,}")
print(f"  Total parameters: {total_params:,}")
print()
print(f"  Traditional fully-connected network: ~{28*28*1000 + 1000*10:,} parameters")
print(f"  CNN reduction: ~{(28*28*1000 + 1000*10) / total_params:.1f}x fewer parameters!")

print()
print("=" * 70)
print("PART 1 COMPLETE: You understand convolution.")
print("You've seen weight sharing. You've built a CNN from scratch.")
print("Now: Real applications with TensorFlow/Keras.")
print("=" * 70)
