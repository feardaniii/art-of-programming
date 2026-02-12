"""
================================================================================
UNDERSTANDING CONVOLUTION: The Foundation of Computer Vision
================================================================================

Course: The Art of Programming - CNN & Transfer Learning (Session 37)
Lesson: What Convolution Really Does (And Why Your Brain Already Knows It)

CNNs didn’t invent vision.
They copied biology.

PREREQUISITES:
    - Completed Module 36 (neurons, backpropagation, TensorFlow/MNIST)
    - Understand: A neuron computes output = sigmoid(w*x + b)
    - Understand: model.fit() trains with automatic differentiation

THE PROBLEM WE SOLVE TODAY:
    In Module 36, we flattened 28x28 MNIST images into 784-pixel vectors.
    A pixel at position (0,0) was treated the same as (27,27).
    We DESTROYED all spatial structure. All neighborhood relationships. Gone.

    Today we fix that. Convolution preserves spatial structure.
    And it does it with 200x FEWER parameters.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tensorflow import keras


# ==============================================================================
# PART 1: THE DETECTIVE'S FLASHLIGHT - What Convolution Really Is
# ==============================================================================

def the_detectives_flashlight():
    """
    ANALOGY: How a detective solves a crime scene.

    A detective does not illuminate the entire room at once.
    They use a small flashlight (3x3 window) and scan it
    across the scene, noting what they find at each position.

    That is EXACTLY what convolution does:
    - The flashlight = the filter/kernel (small, 3x3 or 5x5)
    - The room = the image (large, 28x28 or 224x224)
    - The notes = the feature map (what was detected where)
    """

    print("=" * 70)
    print("PART 1: THE DETECTIVE'S FLASHLIGHT")
    print("What Convolution Really Is")
    print("=" * 70)
    print()

    print("Imagine you are a detective arriving at a dark crime scene.")
    print("You have a small flashlight. The room is huge.")
    print()
    print("What do you do?")
    print()
    print("  1. You do NOT turn on all the lights at once.")
    print("     (That would be a fully-connected network: see EVERYTHING at once)")
    print()
    print("  2. Instead, you SCAN the room with your flashlight:")
    print("     - Top-left corner: nothing unusual")
    print("     - Move right: a scratch on the wall")
    print("     - Move right: nothing")
    print("     - Move down: footprint!")
    print("     - Move right: another footprint!")
    print()
    print("  3. Your flashlight is SMALL (3x3 pixels)")
    print("     The room is LARGE (224x224 pixels)")
    print("     But the SAME flashlight works at every position")
    print()
    print("  4. After scanning, you have a MAP of findings:")
    print("     'Scratches here, footprints there, broken glass over there'")
    print()

    print("THIS is convolution:")
    print("  - The flashlight = the FILTER (3x3 matrix of weights)")
    print("  - The room       = the IMAGE (large matrix of pixels)")
    print("  - The notes       = the FEATURE MAP (what was detected where)")
    print("  - Same flashlight everywhere = WEIGHT SHARING")
    print()

    print("Your brain does this too!")
    print("  Your eye's fovea (sharp center vision) is tiny: ~2 degrees")
    print("  Your visual field is ~180 degrees")
    print("  You scan scenes by moving your fovea (saccades)")
    print("  Your brain assembles the patches into a complete picture")
    print()

    print("Now let's see this with ACTUAL NUMBERS.")
    print()


# ==============================================================================
# PART 2: MANUAL CONVOLUTION - Trace Every Number
# ==============================================================================

def manual_convolution_step_by_step():
    """
    The core operation of convolution, traced by hand.

    We slide a 3x3 filter across a 5x5 image.
    At each position: multiply element-wise, then sum.
    This produces a 3x3 output (the feature map).

    THE GOLDEN RULE OF CONVOLUTION:
    Same filter at every position. This is weight sharing.
    """

    print("=" * 70)
    print("PART 2: MANUAL CONVOLUTION - Trace Every Number")
    print("=" * 70)
    print()

    # A simple 5x5 image with a vertical edge
    image = np.array([
        [0,   0,   0, 255, 255],
        [0,   0,   0, 255, 255],
        [0,   0,   0, 255, 255],
        [0,   0,   0, 255, 255],
        [0,   0,   0, 255, 255]
    ], dtype=float)

    # A simple vertical edge detector
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=float)

    print("Our 5x5 image (dark left, bright right = vertical edge):")
    print()
    for row in image:
        print("  ", [f"{int(v):>3}" for v in row])
    print()

    print("Our 3x3 filter (vertical edge detector):")
    print()
    for row in kernel:
        print("  ", [f"{int(v):>2}" for v in row])
    print()

    print("Now we SLIDE the 3x3 filter across the 5x5 image:")
    print("Output size = (5 - 3 + 1) x (5 - 3 + 1) = 3x3")
    print()

    # Manual convolution with full trace
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch under the flashlight
            patch = image[i:i+kh, j:j+kw]
            # Element-wise multiply and sum
            products = patch * kernel
            value = np.sum(products)
            output[i, j] = value

            print(f"  Position ({i},{j}) - Flashlight covers rows {i}-{i+2}, cols {j}-{j+2}:")
            print(f"    Patch:    {patch.flatten().astype(int).tolist()}")
            print(f"    Filter:   {kernel.flatten().astype(int).tolist()}")
            print(f"    Products: {products.flatten().astype(int).tolist()}")
            print(f"    Sum:      {int(value)}")

            if value > 0:
                print(f"    --> EDGE DETECTED! (positive response)")
            elif value == 0:
                print(f"    --> No edge here (uniform region)")
            else:
                print(f"    --> Reverse edge (negative response)")
            print()

    print("Final feature map (3x3 output):")
    print()
    for row in output:
        print("  ", [f"{int(v):>5}" for v in row])
    print()

    print("THE GOLDEN RULE OF CONVOLUTION:")
    print("  Same 9 weights (the filter) at EVERY position.")
    print("  This is called WEIGHT SHARING.")
    print("  A 224x224 image still uses the same 9 weights.")
    print("  That is why CNNs are efficient: 9 parameters, not millions.")
    print()

    print(f"  Total multiplications: {out_h * out_w} positions x {kh * kw} per position = {out_h * out_w * kh * kw}")
    print(f"  Total parameters: {kh * kw} (just the filter weights!)")
    print()

    return output


# ==============================================================================
# PART 3: THE PARAMETER EXPLOSION PROBLEM
# ==============================================================================

def demonstrate_parameter_explosion():
    """
    NOW you understand convolution. Here is WHY it matters.

    A 224x224 color image has 224 x 224 x 3 = 150,528 pixels.
    Connecting all pixels to 1000 neurons = 150 MILLION parameters.
    That is computationally impossible, memory-hungry, and overfits instantly.

    Convolution solves this: one 3x3 filter = 27 parameters (3x3x3 channels).
    Even 64 such filters = only 1,728 parameters. That is a 100,000x reduction.
    """

    print("=" * 70)
    print("PART 3: THE PARAMETER EXPLOSION PROBLEM")
    print("Why Convolution Is Not Just Clever -- It Is Necessary")
    print("=" * 70)
    print()

    print("In Module 36, we flattened MNIST images: 28x28 -> 784 pixels.")
    print("That worked for small grayscale digits.")
    print("But real images are bigger. And in color.")
    print()

    image_sizes = [28, 64, 128, 224, 512]
    hidden_size = 1000

    print(f"{'Image Size':<15} {'Total Pixels':<15} {'FC Parameters':<20} {'Memory (GB)':<15}")
    print("-" * 70)

    for size in image_sizes:
        pixels = size * size * 3  # RGB
        params = pixels * hidden_size
        memory_gb = params * 4 / (1024**3)  # 4 bytes per float32

        print(f"{size}x{size}x3{'':>3} {pixels:>12,} {params:>18,} {memory_gb:>12.2f}")

    print()
    print("A single layer for a 224x224 image needs 150 MILLION parameters!")
    print("A 3-layer network: 150M + 1M + 1M = 152 MILLION parameters")
    print()

    print("Now compare with convolution:")
    print()
    print(f"{'Approach':<30} {'Parameters':<20} {'Reduction'}")
    print("-" * 65)
    print(f"{'Fully-connected (224x224x3)':<30} {'150,528,000':<20} {'(baseline)'}")
    print(f"{'One 3x3 conv filter':<30} {'27 + 1 = 28':<20} {'5,376,000x fewer'}")
    print(f"{'32 conv filters (3x3)':<30} {'32 x 28 = 896':<20} {'168,000x fewer'}")
    print(f"{'64 conv filters (3x3)':<30} {'64 x 28 = 1,792':<20} {'84,000x fewer'}")
    print()
    print("THE INSIGHT:")
    print("  Convolution exploits the fact that local patterns repeat.")
    print("  An edge is an edge, whether it appears top-left or bottom-right.")
    print("  One detector, applied everywhere. Weight sharing.")
    print()


# ==============================================================================
# PART 4: FILTERS ON REAL IMAGES
# ==============================================================================

def filters_on_real_images():
    """
    Apply edge detection, blur, and sharpen filters to REAL images.
    Using CIFAR-10 (built-in to Keras, no download needed).
    """

    print("=" * 70)
    print("PART 4: FILTERS ON REAL IMAGES")
    print("Edge Detection, Blur, Sharpen -- On Actual Photographs")
    print("=" * 70)
    print()

    # Load a real image from CIFAR-10
    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Pick a recognizable image (a horse, index class 7)
    horse_indices = np.where(y_train.flatten() == 7)[0]
    image_rgb = X_train[horse_indices[5]]  # Pick a clear example
    actual_class = class_names[y_train[horse_indices[5]][0]]

    # Convert to grayscale for filter application
    image_gray = np.mean(image_rgb, axis=2)

    print(f"Loaded a real image: '{actual_class}' from CIFAR-10")
    print(f"Image size: {image_rgb.shape} (32x32 RGB)")
    print(f"Grayscale: {image_gray.shape}")
    print()

    # Define filters
    filters = {
        'Vertical Edge\n(Sobel)': np.array([[-1, 0, 1],
                                              [-2, 0, 2],
                                              [-1, 0, 1]], dtype=float),

        'Horizontal Edge\n(Sobel)': np.array([[-1, -2, -1],
                                                [ 0,  0,  0],
                                                [ 1,  2,  1]], dtype=float),

        'Blur\n(Average)': np.ones((3, 3)) / 9,

        'Sharpen': np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=float),

        'Emboss': np.array([[-2, -1, 0],
                             [-1,  1, 1],
                             [ 0,  1, 2]], dtype=float),

        'Edge Magnitude': None  # Computed from Sobel
    }

    # Apply filters
    results = {}
    for name, kernel in filters.items():
        if kernel is not None:
            results[name] = convolve2d(image_gray, kernel, mode='same', boundary='symm')

    # Edge magnitude: sqrt(vertical^2 + horizontal^2)
    v_edges = results['Vertical Edge\n(Sobel)']
    h_edges = results['Horizontal Edge\n(Sobel)']
    results['Edge Magnitude'] = np.sqrt(v_edges**2 + h_edges**2)

    # Visualize: 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Row 1: Original + first 3 filters
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original (RGB)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    filter_names = list(results.keys())
    for idx, name in enumerate(filter_names[:2]):
        axes[0, idx + 2].imshow(np.abs(results[name]), cmap='hot')
        axes[0, idx + 2].set_title(name, fontsize=11, fontweight='bold')
        axes[0, idx + 2].axis('off')

    # Row 2: Remaining filters
    for idx, name in enumerate(filter_names[2:]):
        if name == 'Blur\n(Average)':
            axes[1, idx].imshow(results[name], cmap='gray')
        else:
            axes[1, idx].imshow(np.abs(results[name]), cmap='hot')
        axes[1, idx].set_title(name, fontsize=11, fontweight='bold')
        axes[1, idx].axis('off')

    plt.suptitle(f'Convolution Filters Applied to a Real Image ("{actual_class}")',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('convolution_filters_real_image.png', dpi=150, bbox_inches='tight')
    print("Visualization saved: convolution_filters_real_image.png")
    print()

    print("WHAT EACH FILTER DOES:")
    print()
    print("  Vertical Edge (Sobel):")
    print("    Detects vertical boundaries (left-right transitions)")
    print("    Bright where the image changes horizontally")
    print()
    print("  Horizontal Edge (Sobel):")
    print("    Detects horizontal boundaries (top-bottom transitions)")
    print("    Bright where the image changes vertically")
    print()
    print("  Blur (Average):")
    print("    Replaces each pixel with the average of its neighbors")
    print("    Smooths out noise, but also destroys details")
    print()
    print("  Sharpen:")
    print("    Amplifies the center pixel, subtracts neighbors")
    print("    Makes edges crisper, amplifies fine details")
    print()
    print("  Emboss:")
    print("    Creates a 3D embossed effect by emphasizing directional change")
    print()
    print("  Edge Magnitude:")
    print("    Combined strength of vertical + horizontal edges")
    print("    Shows ALL edges regardless of direction")
    print()

    print("These filters are HAND-DESIGNED by humans.")
    print("But what if the network could LEARN its own filters?")
    print("What if it could discover detectors we never thought of?")
    print("THAT is what a CNN does. It learns filters through training.")
    print()

    plt.close()


# ==============================================================================
# PART 5: THE HIERARCHY OF FEATURES
# ==============================================================================

def explain_feature_hierarchy():
    """
    CNNs learn features in a hierarchy:
    Layer 1: Edges (simple patterns)
    Layer 2: Textures and corners (combinations of edges)
    Layer 3: Object parts (combinations of textures)
    Layer 4: Whole objects (combinations of parts)

    This hierarchy emerges automatically through training.
    """

    print("=" * 70)
    print("PART 5: THE HIERARCHY OF FEATURES")
    print("How CNNs Build Understanding Layer by Layer")
    print("=" * 70)
    print()

    print("A CNN does NOT recognize a cat in one step.")
    print("It builds understanding LAYER by LAYER:")
    print()
    print("  Layer 1: PIXELS")
    print("    |")
    print("    | Conv 3x3 filters")
    print("    v")
    print("  Layer 2: EDGES (vertical, horizontal, diagonal)")
    print("    |")
    print("    | Conv 3x3 filters")
    print("    v")
    print("  Layer 3: TEXTURES & CORNERS (fur texture, eye corner)")
    print("    |")
    print("    | Conv 3x3 filters")
    print("    v")
    print("  Layer 4: OBJECT PARTS (ear shape, paw, whiskers)")
    print("    |")
    print("    | Conv 3x3 filters")
    print("    v")
    print("  Layer 5: WHOLE OBJECTS (cat! dog! car!)")
    print()

    print("CONCRETE EXAMPLE -- Recognizing a cat:")
    print()
    print("  Layer 1 detects: / | \\ _ (simple edges)")
    print("  Layer 2 combines edges into: triangles, circles, fur patterns")
    print("  Layer 3 combines those into: pointed ears, round eyes, whiskers")
    print("  Layer 4 combines those into: 'this is a cat face'")
    print()

    print("THE KEY INSIGHT:")
    print("  Each layer uses the PREVIOUS layer's output as input.")
    print("  Layer 2 does not see raw pixels -- it sees edges.")
    print("  Layer 3 does not see pixels -- it sees textures.")
    print("  This is HIERARCHICAL FEATURE LEARNING.")
    print()

    print("WHY this matters for architecture design:")
    print("  - Early layers: many spatial details, few filters (32)")
    print("  - Middle layers: less spatial detail, more filters (64)")
    print("  - Late layers: tiny spatial size, many filters (128-256)")
    print("  - MaxPooling between layers: reduce spatial, keep strongest signals")
    print()


# ==============================================================================
# PART 6: BUILDING A CNN FROM SCRATCH (NumPy)
# ==============================================================================

class SimpleCNN:
    """
    A minimal CNN to understand the architecture, built entirely in NumPy.

    Architecture:
    Input (28x28x1)
      -> Conv1 (3x3, 8 filters) -> ReLU -> (26x26x8)
      -> MaxPool (2x2) -> (13x13x8)
      -> Conv2 (3x3, 16 filters) -> ReLU -> (11x11x16)
      -> MaxPool (2x2) -> (5x5x16)
      -> Flatten -> (400)
      -> Dense (10) -> Softmax

    This is a simplified LeNet-5 (1998).
    """

    def __init__(self):
        np.random.seed(42)

        # Conv layer 1: 8 filters of size 3x3x1
        self.conv1_filters = np.random.randn(8, 3, 3, 1) * 0.1
        self.conv1_bias = np.zeros(8)

        # Conv layer 2: 16 filters of size 3x3x8
        self.conv2_filters = np.random.randn(16, 3, 3, 8) * 0.1
        self.conv2_bias = np.zeros(16)

        # Dense layer: 400 -> 10 (we will TRAIN this part)
        self.dense_weights = np.random.randn(400, 10) * 0.01
        self.dense_bias = np.zeros(10)

    def conv2d(self, image, filters, bias, stride=1):
        """
        Convolution operation -- the flashlight scanning the scene.
        Same operation we traced by hand in Part 2, now automated.
        """
        h, w, c = image.shape
        n_filters, fh, fw, _ = filters.shape

        out_h = (h - fh) // stride + 1
        out_w = (w - fw) // stride + 1

        output = np.zeros((out_h, out_w, n_filters))

        for f in range(n_filters):
            for i in range(out_h):
                for j in range(out_w):
                    patch = image[i*stride:i*stride+fh, j*stride:j*stride+fw, :]
                    output[i, j, f] = np.sum(patch * filters[f]) + bias[f]

        return output

    def relu(self, x):
        """ReLU: if feature detected (positive), keep it. If not, zero."""
        return np.maximum(0, x)

    def max_pool(self, x, pool_size=2):
        """
        Max pooling: in each 2x2 window, keep only the strongest signal.
        Reduces size by half. Makes the network robust to small shifts.
        """
        h, w, c = x.shape
        out_h = h // pool_size
        out_w = w // pool_size

        output = np.zeros((out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                for ch in range(c):
                    patch = x[i*pool_size:(i+1)*pool_size,
                              j*pool_size:(j+1)*pool_size, ch]
                    output[i, j, ch] = np.max(patch)

        return output

    def softmax(self, x):
        """Softmax: convert raw scores into probabilities (sum to 1)."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, x, verbose=False):
        """
        Forward pass through the entire network.
        This is the SAME flow as AlexNet, ResNet, EfficientNet.
        Just different numbers of layers and filters.
        """

        if verbose:
            print(f"  Input shape: {x.shape}")

        # Conv layer 1 + ReLU
        self.z1 = self.conv2d(x, self.conv1_filters, self.conv1_bias)
        self.a1 = self.relu(self.z1)
        if verbose:
            print(f"  After Conv1 + ReLU: {self.a1.shape}")

        # Max pool 1
        self.p1 = self.max_pool(self.a1, pool_size=2)
        if verbose:
            print(f"  After MaxPool1: {self.p1.shape}")

        # Conv layer 2 + ReLU
        self.z2 = self.conv2d(self.p1, self.conv2_filters, self.conv2_bias)
        self.a2 = self.relu(self.z2)
        if verbose:
            print(f"  After Conv2 + ReLU: {self.a2.shape}")

        # Max pool 2
        self.p2 = self.max_pool(self.a2, pool_size=2)
        if verbose:
            print(f"  After MaxPool2: {self.p2.shape}")

        # Flatten: 3D feature maps -> 1D vector
        self.flattened = self.p2.reshape(-1)
        if verbose:
            print(f"  After Flatten: {self.flattened.shape}")

        # Dense layer (fully-connected)
        self.logits = self.flattened @ self.dense_weights + self.dense_bias
        if verbose:
            print(f"  After Dense: {self.logits.shape}")

        # Softmax -> probabilities
        self.probs = self.softmax(self.logits)
        if verbose:
            print(f"  After Softmax: {self.probs.shape}")

        return self.probs

    def compute_loss(self, probs, label):
        """Cross-entropy loss: how wrong is our prediction?"""
        # One-hot encode the label
        target = np.zeros(10)
        target[label] = 1.0
        # Cross-entropy: -sum(target * log(probs))
        return -np.sum(target * np.log(probs + 1e-10))

    def train_dense_layer(self, X_train, y_train, epochs=5, lr=0.01):
        """
        Train ONLY the dense layer (keep conv filters fixed).

        Why? Full backprop through conv layers is complex to implement
        from scratch. But training the dense layer shows the KEY idea:
        forward -> loss -> gradient -> update -> repeat.

        In Keras, all layers are trained automatically (Script 2).
        """

        print("Training the dense layer (conv filters are fixed)...")
        print()

        n_samples = len(X_train)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0

            for i in range(n_samples):
                # Forward pass
                probs = self.forward(X_train[i])
                loss = self.compute_loss(probs, y_train[i])
                epoch_loss += loss

                # Check if prediction is correct
                if np.argmax(probs) == y_train[i]:
                    correct += 1

                # Gradient of cross-entropy + softmax w.r.t. logits
                target = np.zeros(10)
                target[y_train[i]] = 1.0
                d_logits = probs - target  # This is the gradient!

                # Gradient w.r.t. dense weights and biases
                d_weights = np.outer(self.flattened, d_logits)
                d_bias = d_logits

                # Update dense layer only
                self.dense_weights -= lr * d_weights
                self.dense_bias -= lr * d_bias

            avg_loss = epoch_loss / n_samples
            accuracy = correct / n_samples * 100
            losses.append(avg_loss)

            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

        print()
        return losses


def build_and_train_scratch_cnn():
    """
    Build our NumPy CNN and train it on a tiny MNIST subset.
    The point is to SEE the training loop work, not to get high accuracy.
    """

    print("=" * 70)
    print("PART 6: CNN FROM SCRATCH - Build It, Train It, Understand It")
    print("=" * 70)
    print()

    # Load MNIST (same dataset from Module 36)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Take a TINY subset (50 images) -- our scratch CNN is slow
    n_train = 50
    X_tiny = X_train[:n_train].reshape(n_train, 28, 28, 1).astype('float32') / 255.0
    y_tiny = y_train[:n_train]

    print(f"Using {n_train} MNIST images for our scratch CNN")
    print(f"Image shape: {X_tiny[0].shape} (28x28x1 grayscale)")
    print()

    # Create CNN
    cnn = SimpleCNN()

    # Show the forward pass for one image
    print("Forward pass through one image:")
    sample = X_tiny[0]
    probs = cnn.forward(sample, verbose=True)
    print()

    print("Output probabilities (10 digit classes):")
    for digit, prob in enumerate(probs):
        marker = " <-- actual" if digit == y_tiny[0] else ""
        print(f"  Digit {digit}: {prob:.4f}{marker}")
    print()

    # Parameter count
    conv1_params = 8 * 3 * 3 * 1 + 8
    conv2_params = 16 * 3 * 3 * 8 + 16
    dense_params = 400 * 10 + 10
    total_params = conv1_params + conv2_params + dense_params

    print("PARAMETER COUNT:")
    print(f"  Conv1: {conv1_params:,} ({8} filters x 3x3x1 + {8} biases)")
    print(f"  Conv2: {conv2_params:,} ({16} filters x 3x3x8 + {16} biases)")
    print(f"  Dense: {dense_params:,} (400 x 10 + 10 biases)")
    print(f"  Total: {total_params:,}")
    print()

    fc_params = 28 * 28 * 1000 + 1000 * 10
    print(f"  If fully-connected: {fc_params:,} parameters")
    print(f"  CNN reduction: {fc_params / total_params:.0f}x fewer parameters!")
    print()

    # Train it!
    print("-" * 50)
    losses = cnn.train_dense_layer(X_tiny, y_tiny, epochs=5, lr=0.01)
    print("-" * 50)

    if losses[-1] < losses[0]:
        print("The loss is DECREASING! The network is LEARNING!")
        print("Even with random conv filters and only 50 images.")
    print()

    print("NOTE: Accuracy is low because:")
    print("  1. We only trained the DENSE layer (conv filters are random)")
    print("  2. We used only 50 training images")
    print("  3. We ran only 5 epochs")
    print()
    print("In Script 2, Keras trains ALL layers on 60,000 images.")
    print("That is where the magic happens.")
    print()


# ==============================================================================
# PART 7: THE BRIDGE TO KERAS
# ==============================================================================

def bridge_to_keras():
    """
    Everything you just built from scratch... Keras does in 10 lines.

    But NOW you understand what those 10 lines actually DO.
    Every Conv2D is a set of filters sliding across the image.
    Every MaxPooling2D takes the max in each 2x2 window.
    Every Dense is a matrix multiplication + bias.
    """

    print("=" * 70)
    print("PART 7: THE BRIDGE TO KERAS")
    print("Your Scratch CNN vs Keras CNN")
    print("=" * 70)
    print()

    print("CORRESPONDENCE TABLE: NumPy → Keras")
    print()

    print(f"{'NumPy-style code':<50} {'Keras equivalent':<45}")
    print("-" * 98)

    print(f"{'self.conv2d(x, filters, bias)':<50} {'layers.Conv2D(filters, (3,3), ...)' :<45}")
    print(f"{'self.relu(x)':<50} {'activation=\"relu\"' :<45}")
    print(f"{'self.max_pool(x, 2)':<50} {'layers.MaxPooling2D((2,2))' :<45}")
    print(f"{'self.flattened = x.reshape(-1)':<50} {'layers.Flatten()' :<45}")
    print(f"{'x @ self.dense_weights + bias':<50} {'layers.Dense(units, ...)' :<45}")
    print(f"{'self.softmax(x)':<50} {'activation=\"softmax\"' :<45}")
    print(f"{'manual gradient + weight update':<50} {'model.fit()  (automatic backprop)' :<45}")

    print()

    # Build the Keras equivalent
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        # Block 1: same as our conv1 + relu + maxpool
        keras.layers.Conv2D(8, (3, 3), activation='relu', name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),

        # Block 2: same as our conv2 + relu + maxpool
        keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),

        # Classification: same as our flatten + dense + softmax
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Keras CNN (same architecture as our scratch CNN):")
    print()
    model.summary()
    print()

    # Quick train to show it works
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print("Training Keras CNN on 60,000 MNIST images (3 epochs)...")
    print("(Our scratch CNN used 50 images. Keras uses ALL of them.)")
    print()

    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print()
    print("With Keras training ALL layers on 60,000 images:")
    print(f"  -> {test_acc*100:.1f}% accuracy in just 3 epochs!")
    print()
    print("vs our scratch CNN with random conv filters on 50 images:")
    print("  -> ~20-30% accuracy")
    print()
    print("THE DIFFERENCE: Keras trains the conv filters too.")
    print("The filters LEARN to detect edges, textures, shapes.")
    print("That is the power of end-to-end training with backpropagation.")
    print()

    return model


# ==============================================================================
# MAIN: Run all parts with pauses for discussion
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("SESSION 37 - SCRIPT 1: UNDERSTANDING CONVOLUTION")
    print("From the Detective's Flashlight to Your First CNN")
    print("=" * 70)
    print()

    # Part 1: The analogy
    the_detectives_flashlight()
    input("Press Enter to continue to Part 2 (Manual Convolution)...")
    print()

    # Part 2: Manual convolution
    manual_convolution_step_by_step()
    input("Press Enter to continue to Part 3 (Parameter Explosion)...")
    print()

    # Part 3: Why convolution matters
    demonstrate_parameter_explosion()
    input("Press Enter to continue to Part 4 (Filters on Real Images)...")
    print()

    # Part 4: Real image filters
    filters_on_real_images()
    input("Press Enter to continue to Part 5 (Feature Hierarchy)...")
    print()

    # Part 5: How CNNs stack layers
    explain_feature_hierarchy()
    input("Press Enter to continue to Part 6 (CNN from Scratch)...")
    print()

    # Part 6: Build and train from scratch
    build_and_train_scratch_cnn()
    input("Press Enter to continue to Part 7 (Bridge to Keras)...")
    print()

    # Part 7: Keras equivalent
    bridge_to_keras()

    # Summary
    print("=" * 70)
    print("SCRIPT 1 COMPLETE: You Understand Convolution")
    print("=" * 70)
    print()
    print("What you learned:")
    print("  1. Convolution is a small detector sliding across a large image")
    print("  2. Weight sharing: same filter at every position = few parameters")
    print("  3. Different filters detect different patterns (edges, blur, sharpen)")
    print("  4. CNNs stack layers to build a hierarchy: edges -> textures -> objects")
    print("  5. You built and trained a CNN from scratch in NumPy")
    print("  6. Keras does the same thing in 10 lines (but now you know WHAT those lines do)")
    print()
    print("Next: Script 2 -- Training a CNN on REAL images with the FULL pipeline.")
    print("      Load -> Explore -> Preprocess -> Augment -> Train -> Evaluate -> Visualize")
    print("=" * 70)


if __name__ == '__main__':
    main()
