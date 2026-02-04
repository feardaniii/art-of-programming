"""
================================================================================
TENSORFLOW: From Manual Backprop to Production Deep Learning
================================================================================

Course: The Art of Programming - Deep Learning Fundamentals
Lesson: Using TensorFlow After Understanding the Fundamentals

PREREQUISITES:
    - Completed Neuron lesson (understand weights, bias, activation)
    - Completed Backpropagation lesson (understand gradients, chain rule)

THE BIG QUESTION:
    We built backprop by hand. Why do we need TensorFlow?

ANSWER:
    1. AUTOMATIC DIFFERENTIATION - No manual gradient math
    2. GPU ACCELERATION - 100x faster on graphics cards
    3. PRE-BUILT COMPONENTS - Layers, optimizers, losses ready to use
    4. PRODUCTION READY - Deploy to web, mobile, servers

THIS LESSON:
    - First: Solve XOR with TensorFlow (compare to our numpy version)
    - Then: See automatic differentiation in action
    - Finally: Train on MNIST (real dataset, 70,000 handwritten digits)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow import
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
print()

# =============================================================================
# PART 1: XOR WITH TENSORFLOW (Bridging from Numpy)
# =============================================================================

"""
In the backpropagation lesson, we solved XOR with numpy.

Let's solve it again with TensorFlow and see the correspondence:

    NUMPY (what we built)          TENSORFLOW (what we'll use)
    ─────────────────────          ──────────────────────────
    W1, b1, W2, b2                  Stored in Dense layers
    forward() method                model(X) or model.predict(X)
    backward() method               model.fit() (automatic!)
    compute_loss()                  loss='binary_crossentropy'
    gradient descent loop           optimizer='adam'
"""


def xor_numpy_vs_tensorflow():
    """
    Side-by-side comparison: Our numpy network vs TensorFlow.
    """
    print("=" * 70)
    print("PART 1: XOR - Numpy vs TensorFlow")
    print("=" * 70)
    print()

    # XOR data
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    print("XOR Truth Table:")
    print("  x₁  x₂  │  target")
    print("  ────────┼────────")
    for i in range(len(X)):
        print(f"   {int(X[i][0])}   {int(X[i][1])}  │    {int(y[i][0])}")
    print()

    # =========================================
    # TENSORFLOW VERSION
    # =========================================
    print("Building TensorFlow model...")
    print()

    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Build model using Keras Sequential API
    model = keras.Sequential([
        # Hidden layer: 4 neurons with sigmoid activation
        # This is equivalent to: z1 = X @ W1 + b1, h = sigmoid(z1)
        keras.layers.Dense(4, activation='sigmoid', input_shape=(2,), name='hidden'),

        # Output layer: 1 neuron with sigmoid activation
        # This is equivalent to: z2 = h @ W2 + b2, out = sigmoid(z2)
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])

    # Show what TensorFlow created
    print("Model Architecture:")
    print("-" * 50)
    model.summary()
    print()

    # Compile model (specify optimizer and loss)
    # This is equivalent to our learning_rate and compute_loss()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=1.0),  # Stochastic Gradient Descent
        loss='binary_crossentropy',  # Better than MSE for classification
        metrics=['accuracy']
    )

    print("Correspondence to our numpy implementation:")
    print("  keras.layers.Dense(4)  →  W1 (2×4), b1 (4,)")
    print("  keras.layers.Dense(1)  →  W2 (4×1), b2 (1,)")
    print("  optimizer=SGD          →  our gradient descent loop")
    print("  loss='binary_crossentropy' → our compute_loss()")
    print("  model.fit()            →  our train() with backward()")
    print()

    # Train the model
    print("Training (this replaces our manual backprop loop)...")
    print("-" * 50)

    history = model.fit(
        X, y,
        epochs=1000,
        verbose=0  # Silent training
    )

    # Show training progress
    print(f"Final loss: {history.history['loss'][-1]:.6f}")
    print(f"Final accuracy: {history.history['accuracy'][-1] * 100:.1f}%")
    print()

    # Test predictions
    print("Predictions:")
    print("  x₁  x₂  │ target │  output  │ prediction")
    print("  ────────┼────────┼──────────┼───────────")

    predictions = model.predict(X, verbose=0)
    for i in range(len(X)):
        pred = 1 if predictions[i, 0] > 0.5 else 0
        correct = "✓" if pred == y[i, 0] else "✗"
        print(
            f"   {int(X[i][0])}   {int(X[i][1])}  │   {int(y[i][0])}    │  {predictions[i, 0]:.4f}  │     {pred}  {correct}")

    print()
    print("✓ XOR solved with TensorFlow!")
    print()
    print("Key insight: We wrote ~100 lines of backprop code.")
    print("TensorFlow does it in ~10 lines, and it's FASTER.")
    print()

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('XOR Training with TensorFlow', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('xor_tensorflow.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model


# =============================================================================
# PART 2: AUTOMATIC DIFFERENTIATION (The Magic Explained)
# =============================================================================

"""
THE KILLER FEATURE: Automatic Differentiation

Remember calculating gradients by hand?
    ∂Loss/∂W2 = (output - target) × sigmoid_derivative × hidden
    ∂Loss/∂W1 = ... (even more complex)

TensorFlow computes ALL gradients AUTOMATICALLY.

How? It builds a "computation graph" and uses the chain rule.
"""


def demonstrate_automatic_differentiation():
    """
    Show how TensorFlow computes gradients automatically.
    """
    print()
    print("=" * 70)
    print("PART 2: Automatic Differentiation")
    print("=" * 70)
    print()

    print("Remember our manual gradient calculation?")
    print()
    print("  # Output layer gradient")
    print("  dL_dout = -(target - output)")
    print("  delta2 = dL_dout * sigmoid_derivative(output)")
    print("  dL_dW2 = hidden.T @ delta2")
    print()
    print("  # Hidden layer gradient (chain rule)")
    print("  dL_dh = delta2 @ W2.T")
    print("  delta1 = dL_dh * sigmoid_derivative(hidden)")
    print("  dL_dW1 = input.T @ delta1")
    print()
    print("TensorFlow does this AUTOMATICALLY with tf.GradientTape():")
    print()

    # Simple example: y = x²
    print("Example: y = x², find dy/dx at x=3")
    print("-" * 50)

    x = tf.Variable(3.0)  # A variable we want gradients for

    with tf.GradientTape() as tape:
        y = x ** 2  # Forward computation

    # Compute gradient automatically!
    dy_dx = tape.gradient(y, x)

    print(f"  x = {x.numpy()}")
    print(f"  y = x² = {y.numpy()}")
    print(f"  dy/dx = 2x = {dy_dx.numpy()}")
    print(f"  (Expected: 2 × 3 = 6) ✓")
    print()

    # More complex example: chain rule
    print("Example: z = (x + y)², find ∂z/∂x and ∂z/∂y at x=2, y=3")
    print("-" * 50)

    x = tf.Variable(2.0)
    y = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        z = (x + y) ** 2  # z = 25

    # Get gradients for BOTH variables at once
    gradients = tape.gradient(z, [x, y])

    print(f"  x = {x.numpy()}, y = {y.numpy()}")
    print(f"  z = (x + y)² = {z.numpy()}")
    print(f"  ∂z/∂x = 2(x + y) = {gradients[0].numpy()}")
    print(f"  ∂z/∂y = 2(x + y) = {gradients[1].numpy()}")
    print(f"  (Expected: 2 × 5 = 10 for both) ✓")
    print()

    # Neural network gradient example
    print("Example: One neuron, compute weight gradients")
    print("-" * 50)

    # Single neuron: output = sigmoid(w*x + b)
    w = tf.Variable(0.5)
    b = tf.Variable(0.1)
    x_input = tf.constant(1.0)
    target = tf.constant(1.0)

    with tf.GradientTape() as tape:
        # Forward pass
        z = w * x_input + b
        output = tf.sigmoid(z)

        # Loss
        loss = 0.5 * (target - output) ** 2

    # Gradients for BOTH w and b
    dL_dw, dL_db = tape.gradient(loss, [w, b])

    print(f"  Input: x = {x_input.numpy()}")
    print(f"  Weight: w = {w.numpy()}")
    print(f"  Bias: b = {b.numpy()}")
    print(f"  Output: σ(wx + b) = {output.numpy():.4f}")
    print(f"  Target: {target.numpy()}")
    print(f"  Loss: {loss.numpy():.6f}")
    print()
    print(f"  ∂Loss/∂w = {dL_dw.numpy():.6f}")
    print(f"  ∂Loss/∂b = {dL_db.numpy():.6f}")
    print()
    print("TensorFlow computed the chain rule for us!")
    print("This is why we don't write backward() by hand anymore.")
    print()


# =============================================================================
# PART 3: UNDERSTANDING KERAS LAYERS
# =============================================================================

"""
KERAS LAYERS = Our neurons, packaged nicely

What's inside a Dense layer?
    - Weights matrix W
    - Bias vector b
    - Activation function
    - Forward pass: output = activation(input @ W + b)

Let's peek inside.
"""


def explore_keras_layers():
    """
    Look inside Keras layers to see they're exactly what we built.
    """
    print()
    print("=" * 70)
    print("PART 3: Inside Keras Layers")
    print("=" * 70)
    print()

    # Create a simple layer
    layer = keras.layers.Dense(3, activation='sigmoid', input_shape=(2,))

    # Build it with dummy input
    dummy_input = np.array([[1.0, 2.0]])
    _ = layer(dummy_input)  # This initializes the weights

    # Extract weights
    weights, biases = layer.get_weights()

    print("A Dense layer with 2 inputs and 3 outputs:")
    print()
    print(f"Weights shape: {weights.shape}")
    print(f"Weights:\n{weights}")
    print()
    print(f"Biases shape: {biases.shape}")
    print(f"Biases: {biases}")
    print()

    # Manual computation vs layer computation
    print("Let's verify the layer does what we expect:")
    print("-" * 50)

    test_input = np.array([[0.5, 0.8]])

    # Manual computation (what we learned)
    z = test_input @ weights + biases
    manual_output = 1 / (1 + np.exp(-z))  # sigmoid

    # Layer computation
    layer_output = layer(test_input).numpy()

    print(f"Input: {test_input}")
    print()
    print(f"Manual computation:")
    print(f"  z = input @ W + b = {z}")
    print(f"  output = sigmoid(z) = {manual_output}")
    print()
    print(f"Layer computation:")
    print(f"  layer(input) = {layer_output}")
    print()
    print(f"Match: {np.allclose(manual_output, layer_output)} ✓")
    print()
    print("Keras layers are EXACTLY our neurons, just packaged nicely.")
    print()


# =============================================================================
# PART 4: MNIST - A REAL DATASET
# =============================================================================

"""
Now let's use TensorFlow for something real: MNIST handwritten digits.

MNIST Dataset:
    - 70,000 images of handwritten digits (0-9)
    - Each image is 28×28 pixels (784 total pixels)
    - This is the "Hello World" of deep learning
    - Real data, real challenge, real results

Why MNIST?
    - It's a benchmark (everyone uses it, you can compare)
    - It's hard enough to need a neural network
    - It's small enough to train quickly
    - It's visual (you can SEE what the network learns)
"""


def train_mnist_classifier():
    """
    Train a neural network to recognize handwritten digits.
    """
    print()
    print("=" * 70)
    print("PART 4: MNIST Digit Classification")
    print("=" * 70)
    print()

    # Load MNIST dataset (built into Keras)
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training images: {X_train.shape[0]}")
    print(f"Test images: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]} (28×28 pixels)")
    print()

    # Show some examples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(f"Label: {y_train[i]}", fontsize=12)
        ax.axis('off')
    plt.suptitle("Sample MNIST Digits", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Preprocess data
    print("Preprocessing...")
    print("-" * 50)

    # Flatten 28×28 images to 784-dimensional vectors
    X_train_flat = X_train.reshape(-1, 784).astype('float32')
    X_test_flat = X_test.reshape(-1, 784).astype('float32')

    # Normalize pixel values from [0, 255] to [0, 1]
    X_train_norm = X_train_flat / 255.0
    X_test_norm = X_test_flat / 255.0

    print(f"Flattened shape: {X_train_flat.shape} (60000 images × 784 pixels)")
    print(f"Normalized range: [{X_train_norm.min()}, {X_train_norm.max()}]")
    print()

    # Build model
    print("Building neural network...")
    print("-" * 50)

    model = keras.Sequential([
        # Input: 784 pixels
        # Hidden layer 1: 128 neurons
        keras.layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_1'),

        # Hidden layer 2: 64 neurons
        keras.layers.Dense(64, activation='relu', name='hidden_2'),

        # Output layer: 10 neurons (one per digit 0-9)
        # softmax converts to probabilities that sum to 1
        keras.layers.Dense(10, activation='softmax', name='output')
    ])

    model.summary()
    print()

    print("Architecture explanation:")
    print("  Input (784)  → 28×28 pixel image flattened")
    print("  Hidden (128) → Learns low-level features (edges, curves)")
    print("  Hidden (64)  → Learns higher-level features (loops, lines)")
    print("  Output (10)  → Probability for each digit 0-9")
    print()
    print("  ReLU activation → max(0, x), faster than sigmoid")
    print("  Softmax output  → converts scores to probabilities")
    print()

    # Compile model
    model.compile(
        optimizer='adam',  # Adaptive learning rate optimizer
        loss='sparse_categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']
    )

    print("Optimizer: Adam (adaptive learning rate, better than plain SGD)")
    print("Loss: Sparse Categorical Cross-Entropy (multi-class version of BCE)")
    print()

    # Train model
    print("Training neural network...")
    print("-" * 50)

    history = model.fit(
        X_train_norm, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,  # Use 10% of training data for validation
        verbose=1
    )

    # Evaluate on test set
    print()
    print("=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)

    test_loss, test_accuracy = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print()

    # Show some predictions
    print("Sample Predictions:")
    print("-" * 50)

    # Get predictions for first 10 test images
    predictions = model.predict(X_test_norm[:10], verbose=0)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_test[i], cmap='gray')

        predicted_digit = np.argmax(predictions[i])
        confidence = predictions[i][predicted_digit] * 100
        actual_digit = y_test[i]

        color = 'green' if predicted_digit == actual_digit else 'red'
        ax.set_title(f"Pred: {predicted_digit} ({confidence:.1f}%)\nActual: {actual_digit}",
                     fontsize=10, color=color)
        ax.axis('off')

    plt.suptitle("MNIST Predictions (Green=Correct, Red=Wrong)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Over Training', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Over Training', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mnist_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    print()
    print(f"✓ Trained digit classifier with {test_accuracy * 100:.1f}% accuracy!")
    print()

    return model, history


# =============================================================================
# PART 5: WHAT'S HAPPENING INSIDE?
# =============================================================================

def visualize_what_network_learned(model):
    """
    Peek inside the trained network to see what it learned.
    """
    print()
    print("=" * 70)
    print("PART 5: What Did the Network Learn?")
    print("=" * 70)
    print()

    # Get weights of first layer
    first_layer = model.layers[0]
    weights, biases = first_layer.get_weights()

    print(f"First layer weights shape: {weights.shape}")
    print(f"  784 inputs × 128 neurons = 100,352 learned parameters!")
    print()

    # Visualize what each neuron "looks for"
    # Each column of weights is a 784-d vector that can be reshaped to 28×28
    print("Visualizing what the first 16 neurons 'look for':")
    print("(Each neuron learns a pattern detector)")
    print()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        # Get weights for neuron i, reshape to 28×28
        neuron_weights = weights[:, i].reshape(28, 28)

        # Display
        ax.imshow(neuron_weights, cmap='RdBu', vmin=-0.2, vmax=0.2)
        ax.axis('off')
        ax.set_title(f'Neuron {i}', fontsize=9)

    plt.suptitle("First Layer Neurons: What Patterns Do They Detect?",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_neuron_weights.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Interpretation:")
    print("  - Red regions: positive weights (neuron activates if these pixels are bright)")
    print("  - Blue regions: negative weights (neuron activates if these pixels are dark)")
    print("  - Each neuron becomes a 'feature detector' for a specific pattern")
    print()

    # Show how a specific digit activates the network
    print("How the network 'sees' a digit:")
    print("-" * 50)

    # Get a test image
    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Find a clear '7'
    idx = np.where(y_test == 7)[0][0]
    test_image = X_test[idx]
    test_input = test_image.reshape(1, 784).astype('float32') / 255.0

    # Get activations at each layer
    layer_outputs = []
    x = test_input
    for layer in model.layers:
        x = layer(x)
        layer_outputs.append(x.numpy())

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    # Original image
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title("Input Image\n(28×28 pixels)", fontsize=10)
    axes[0].axis('off')

    # Hidden layer 1 activations
    axes[1].bar(range(len(layer_outputs[0][0])), layer_outputs[0][0], color='blue', alpha=0.7)
    axes[1].set_title(f"Hidden Layer 1\n({len(layer_outputs[0][0])} neurons)", fontsize=10)
    axes[1].set_xlabel("Neuron")
    axes[1].set_ylabel("Activation")

    # Hidden layer 2 activations
    axes[2].bar(range(len(layer_outputs[1][0])), layer_outputs[1][0], color='green', alpha=0.7)
    axes[2].set_title(f"Hidden Layer 2\n({len(layer_outputs[1][0])} neurons)", fontsize=10)
    axes[2].set_xlabel("Neuron")
    axes[2].set_ylabel("Activation")

    # Output probabilities
    axes[3].bar(range(10), layer_outputs[2][0], color='red', alpha=0.7)
    axes[3].set_title("Output Layer\n(probabilities)", fontsize=10)
    axes[3].set_xlabel("Digit")
    axes[3].set_ylabel("Probability")
    axes[3].set_xticks(range(10))

    predicted = np.argmax(layer_outputs[2][0])
    axes[3].axvline(x=predicted, color='black', linestyle='--', label=f'Predicted: {predicted}')
    axes[3].legend()

    plt.suptitle("How Information Flows Through the Network", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_activations.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"For this '7', the network:")
    print(f"  1. Takes 784 pixel values")
    print(f"  2. Hidden layer 1: Detects {np.sum(layer_outputs[0][0] > 0.1)} active features")
    print(f"  3. Hidden layer 2: Combines into {np.sum(layer_outputs[1][0] > 0.1)} higher features")
    print(f"  4. Output: Highest probability for digit {predicted}")
    print()


# =============================================================================
# PART 6: FROM THEORY TO PRACTICE - KEY TAKEAWAYS
# =============================================================================

def print_summary():
    """
    Summarize the connection between theory and TensorFlow.
    """
    print()
    print("=" * 70)
    print("SUMMARY: What We Learned")
    print("=" * 70)
    print()

    print("THEORY → TENSORFLOW MAPPING:")
    print()
    print("┌─────────────────────────────┬────────────────────────────────────┐")
    print("│ What We Built (Numpy)       │ TensorFlow Equivalent              │")
    print("├─────────────────────────────┼────────────────────────────────────┤")
    print("│ W1, b1, W2, b2              │ keras.layers.Dense()               │")
    print("│ sigmoid(z)                  │ activation='sigmoid'               │")
    print("│ forward() method            │ model(X) or model.predict()        │")
    print("│ backward() + chain rule     │ Automatic via GradientTape         │")
    print("│ compute_loss()              │ loss='binary_crossentropy'         │")
    print("│ gradient descent loop       │ optimizer='sgd' or 'adam'          │")
    print("│ train() with epochs         │ model.fit(epochs=N)                │")
    print("└─────────────────────────────┴────────────────────────────────────┘")
    print()

    print("WHY USE TENSORFLOW?")
    print()
    print("  1. AUTOMATIC DIFFERENTIATION")
    print("     You: Define forward pass")
    print("     TensorFlow: Computes ALL gradients automatically")
    print()
    print("  2. GPU ACCELERATION")
    print("     CPU training: Hours")
    print("     GPU training: Minutes")
    print("     (TensorFlow handles this automatically)")
    print()
    print("  3. PRE-BUILT COMPONENTS")
    print("     Layers: Dense, Conv2D, LSTM, Transformer...")
    print("     Optimizers: SGD, Adam, RMSprop...")
    print("     Losses: MSE, CrossEntropy, Custom...")
    print()
    print("  4. PRODUCTION READY")
    print("     Export to: Web (TensorFlow.js)")
    print("                Mobile (TensorFlow Lite)")
    print("                Server (TensorFlow Serving)")
    print()

    print("THE PATH YOU'VE TRAVELED:")
    print()
    print("  Lesson 1: Neuron")
    print("    → A neuron is a weighted sum + activation")
    print("    → One neuron = one decision boundary (line)")
    print()
    print("  Lesson 2: Backpropagation")
    print("    → Multiple neurons = multiple lines = complex regions")
    print("    → Chain rule assigns 'blame' to each weight")
    print("    → Gradient descent minimizes the error")
    print()
    print("  Lesson 3: TensorFlow (this lesson)")
    print("    → Same concepts, but automated and optimized")
    print("    → Ready for real data and production deployment")
    print()


# =============================================================================
# MAIN: RUN THE COMPLETE LESSON
# =============================================================================

def main():
    """Run the complete TensorFlow lesson."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "TENSORFLOW: From Theory to Practice".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Part 1: XOR comparison
    xor_model = xor_numpy_vs_tensorflow()
    input("\nPress Enter to learn about automatic differentiation...")

    # Part 2: Automatic differentiation
    demonstrate_automatic_differentiation()
    input("\nPress Enter to explore Keras layers...")

    # Part 3: Inside Keras layers
    explore_keras_layers()
    input("\nPress Enter to train on MNIST...")

    # Part 4: MNIST
    mnist_model, history = train_mnist_classifier()
    input("\nPress Enter to see what the network learned...")

    # Part 5: Visualize learned features
    visualize_what_network_learned(mnist_model)
    input("\nPress Enter for summary...")

    # Part 6: Summary
    print_summary()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "LESSON COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("You now understand:")
    print("  ✓ How TensorFlow relates to the theory you learned")
    print("  ✓ Automatic differentiation (no manual gradients!)")
    print("  ✓ What's inside Keras layers (exactly what you built)")
    print("  ✓ How to train on real data (MNIST)")
    print("  ✓ How to visualize what networks learn")
    print()
    print("Next steps:")
    print("  - Convolutional Neural Networks (for images)")
    print("  - Recurrent Neural Networks (for sequences)")
    print("  - Transfer Learning (use pre-trained models)")
    print()


if __name__ == "__main__":
    main()