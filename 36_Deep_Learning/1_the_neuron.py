import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# THE ARTIFICIAL NEURON - McCulloch & Pitts
# ==========================================

class Neuron:
    """
    A single artificial neuron.

    Biology → Math:
    - Dendrites (inputs) → x1, x2, ..., xn
    - Synaptic weights → w1, w2, ..., wn
    - Cell body (soma) → weighted sum + bias
    - Axon (output) → activation function
    """

    def __init__(self, n_inputs):
        # Initialize small random weights (Xavier initialization concept)
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = np.random.randn() * 0.1

    def forward(self, inputs, activation='sigmoid'):
        """
        Forward propagation through the neuron.

        z = w·x + b  (Linear Algebra: dot product)
        a = σ(z)     (Activation function)
        """
        # Step 1: Weighted sum (your Linear Algebra in action)
        z = np.dot(self.weights, inputs) + self.bias

        # Step 2: Activation function (introduces non-linearity)
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))  # Squashes to [0, 1]
        elif activation == 'relu':
            a = np.maximum(0, z)       # Rectified Linear Unit
        elif activation == 'tanh':
            a = np.tanh(z)             # Squashes to [-1, 1]
        else:
            a = z                      # Linear (no activation)

        return a, z  # Return both activation and pre-activation

# ==========================================
# ACTIVATION FUNCTIONS VISUALIZED
# ==========================================

def visualize_activations():
    """Why do we need activation functions? Let's see."""

    z = np.linspace(-5, 5, 200)

    sigmoid = 1 / (1 + np.exp(-z))
    relu = np.maximum(0, z)
    tanh = np.tanh(z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sigmoid
    axes[0].plot(z, sigmoid, 'b-', linewidth=2)
    axes[0].set_title('Sigmoid: σ(z) = 1/(1+e^(-z))', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('z (weighted sum)')
    axes[0].set_ylabel('activation')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary')
    axes[0].legend()

    # ReLU
    axes[1].plot(z, relu, 'g-', linewidth=2)
    axes[1].set_title('ReLU: max(0, z)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('z (weighted sum)')
    axes[1].set_ylabel('activation')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Activation threshold')
    axes[1].legend()

    # Tanh
    axes[2].plot(z, tanh, 'orange', linewidth=2)
    axes[2].set_title('Tanh: (e^z - e^(-z))/(e^z + e^(-z))', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('z (weighted sum)')
    axes[2].set_ylabel('activation')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero crossing')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    print("✅ Activation functions visualized")
    print()
    print("WHY NON-LINEAR ACTIVATIONS?")
    print("Without them: stacking neurons = still just linear transformation")
    print("With them: universal approximation - can learn ANY pattern")

visualize_activations()

# ==========================================
# MANUAL AND GATE - 1943 Style
# ==========================================

def demo_and_gate():
    """
    McCulloch-Pitts 1943: Let's build an AND gate with a neuron.

    Truth table:
    x1 | x2 | output
    ---|----+-------
    0  | 0  | 0
    0  | 1  | 0
    1  | 0  | 0
    1  | 1  | 1
    """

    # Manual weights that solve AND
    neuron = Neuron(n_inputs=2)
    neuron.weights = np.array([0.5, 0.5])  # Both inputs matter equally
    neuron.bias = -0.7                      # High threshold

    print("=" * 50)
    print("MANUAL AND GATE (1943 McCulloch-Pitts style)")
    print("=" * 50)

    test_cases = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]

    print(f"\nWeights: {neuron.weights}")
    print(f"Bias: {neuron.bias}")
    print(f"\nTesting AND gate:")
    print(f"{'x1':<5} {'x2':<5} {'Expected':<10} {'Got':<10} {'Raw output':<15}")
    print("-" * 50)

    for inputs, expected in test_cases:
        activation, z = neuron.forward(inputs, activation='sigmoid')
        prediction = 1 if activation > 0.5 else 0

        print(f"{inputs[0]:<5} {inputs[1]:<5} {expected:<10} {prediction:<10} {activation:.4f}")

    print("\n✅ The neuron learned logic! (Actually, we hand-coded it)")
    print("Next: Make it LEARN the weights itself...")

demo_and_gate()

# ==========================================
# LEARNING: ADJUST WEIGHTS FROM EXAMPLES
# ==========================================

def train_and_gate_from_scratch():
    """
    1958 Rosenblatt: The Perceptron Learning Rule

    If prediction is wrong:
        weights += learning_rate * error * input
        bias += learning_rate * error

    Simple. Elegant. Revolutionary.
    """

    # Training data for AND gate
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y_train = np.array([0, 0, 0, 1])  # Expected outputs

    # Initialize random neuron
    neuron = Neuron(n_inputs=2)
    learning_rate = 0.1
    epochs = 20

    print("\n" + "=" * 50)
    print("TRAINING AND GATE FROM RANDOM WEIGHTS")
    print("=" * 50)

    print(f"\nInitial weights: {neuron.weights}")
    print(f"Initial bias: {neuron.bias}")

    # Training loop
    for epoch in range(epochs):
        total_error = 0

        for i in range(len(X_train)):
            # Forward pass
            inputs = X_train[i]
            target = y_train[i]

            activation, z = neuron.forward(inputs, activation='sigmoid')
            prediction = 1 if activation > 0.5 else 0

            # Calculate error
            error = target - activation
            total_error += abs(error)

            # Update weights (gradient descent, simplified)
            neuron.weights += learning_rate * error * inputs
            neuron.bias += learning_rate * error

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Total Error = {total_error:.4f}")

    print(f"\nFinal weights: {neuron.weights}")
    print(f"Final bias: {neuron.bias}")

    # Test learned neuron
    print("\nTesting learned AND gate:")
    print(f"{'x1':<5} {'x2':<5} {'Expected':<10} {'Predicted':<10}")
    print("-" * 40)

    for i in range(len(X_train)):
        activation, _ = neuron.forward(X_train[i], activation='sigmoid')
        prediction = 1 if activation > 0.5 else 0

        print(f"{X_train[i][0]:<5} {X_train[i][1]:<5} {y_train[i]:<10} {prediction:<10}")

    print("\n✅ IT LEARNED! The machine adjusted its own weights!")
    print("This is 1958. This is the Perceptron. This is the beginning.")

train_and_gate_from_scratch()

# ==========================================
# THE LIMITATION: XOR PROBLEM (1969)
# ==========================================

def demonstrate_xor_failure():
    """
    XOR truth table:
    x1 | x2 | output
    ---|----+-------
    0  | 0  | 0
    0  | 1  | 1
    1  | 0  | 1
    1  | 1  | 0

    A single perceptron CANNOT learn this.
    It's linearly inseparable.
    This killed AI research for 15 years.
    """

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])  # XOR pattern

    neuron = Neuron(n_inputs=2)
    learning_rate = 0.1
    epochs = 100

    print("\n" + "=" * 50)
    print("TRYING TO LEARN XOR (This will fail)")
    print("=" * 50)

    for epoch in range(0, epochs, 20):
        for i in range(len(X_train)):
            activation, z = neuron.forward(X_train[i], activation='sigmoid')
            error = y_train[i] - activation
            neuron.weights += learning_rate * error * X_train[i]
            neuron.bias += learning_rate * error

        # Check accuracy
        correct = 0
        for i in range(len(X_train)):
            activation, _ = neuron.forward(X_train[i], activation='sigmoid')
            prediction = 1 if activation > 0.5 else 0
            if prediction == y_train[i]:
                correct += 1

        accuracy = correct / len(X_train) * 100
        print(f"Epoch {epoch:3d}: Accuracy = {accuracy:.1f}%")

    print("\n❌ Single neuron CANNOT learn XOR")
    print("Minsky & Papert proved this in 1969")
    print("Solution: MULTIPLE LAYERS (next part!)")

demonstrate_xor_failure()

print("\n" + "=" * 70)
print("PART 1 COMPLETE: You've built neurons. You've seen them learn.")
print("You've hit the XOR wall. Now you need... DEPTH.")
print("=" * 70)
