import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MULTI-LAYER NEURAL NETWORK
# ==========================================

class NeuralNetwork:
    """
    A simple 2-layer neural network (1 hidden layer).

    Architecture:
    Input Layer → Hidden Layer → Output Layer
       (2)     →      (3)      →     (1)

    This solves XOR. This ends the AI winter.
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Layer 1: Input → Hidden
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: Hidden → Output
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # For storing intermediate values (needed for backprop)
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for numerical stability

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        """
        Forward propagation through the network.

        Layer 1:
            z1 = X @ W1 + b1
            a1 = sigmoid(z1)

        Layer 2:
            z2 = a1 @ W2 + b2
            a2 = sigmoid(z2)
        """
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, learning_rate):
        """
        Backpropagation: The Chain Rule in Action

        Goal: Compute ∂Loss/∂W for each weight

        Chain rule:
            ∂Loss/∂W2 = ∂Loss/∂a2 × ∂a2/∂z2 × ∂z2/∂W2
            ∂Loss/∂W1 = ∂Loss/∂a2 × ∂a2/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂W1

        This is your CALCULUS becoming LEARNING.
        """
        m = X.shape[0]  # Number of samples

        # ============================================
        # STEP 1: Output layer gradients
        # ============================================

        # Loss = 1/2 * (y - a2)^2  (Mean Squared Error)
        # ∂Loss/∂a2 = -(y - a2)
        dLoss_da2 = -(y - self.a2)

        # ∂a2/∂z2 = sigmoid'(z2)
        da2_dz2 = self.sigmoid_derivative(self.z2)

        # Combine: ∂Loss/∂z2
        delta2 = dLoss_da2 * da2_dz2

        # ∂z2/∂W2 = a1  (from z2 = a1 @ W2 + b2)
        dLoss_dW2 = self.a1.T @ delta2 / m
        dLoss_db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # ============================================
        # STEP 2: Hidden layer gradients (BACKPROP!)
        # ============================================

        # ∂Loss/∂a1 = ∂Loss/∂z2 × ∂z2/∂a1 = delta2 @ W2.T
        dLoss_da1 = delta2 @ self.W2.T

        # ∂a1/∂z1 = sigmoid'(z1)
        da1_dz1 = self.sigmoid_derivative(self.z1)

        # Combine: ∂Loss/∂z1
        delta1 = dLoss_da1 * da1_dz1

        # ∂z1/∂W1 = X
        dLoss_dW1 = X.T @ delta1 / m
        dLoss_db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # ============================================
        # STEP 3: Gradient descent update
        # ============================================

        self.W2 -= learning_rate * dLoss_dW2
        self.b2 -= learning_rate * dLoss_db2
        self.W1 -= learning_rate * dLoss_dW1
        self.b1 -= learning_rate * dLoss_db1

    def train(self, X, y, epochs, learning_rate):
        """Training loop with loss tracking."""
        losses = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Calculate loss
            loss = np.mean((y - predictions) ** 2)
            losses.append(loss)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Print progress
            if epoch % 500 == 0:
                accuracy = np.mean((predictions > 0.5) == y) * 100
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.1f}%")

        return losses

# ==========================================
# SOLVING XOR WITH NEURAL NETWORK
# ==========================================

def solve_xor():
    """
    1986: The XOR problem is TRIVIAL with backpropagation.
    Two layers. Chain rule. Done.
    """

    print("=" * 60)
    print("SOLVING XOR WITH MULTI-LAYER NETWORK (1986 Renaissance)")
    print("=" * 60)
    print()

    # XOR data
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Create network: 2 inputs → 3 hidden → 1 output
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

    # Train
    print("Training neural network on XOR...")
    print()
    losses = nn.train(X, y, epochs=5000, learning_rate=0.5)

    # Test
    print()
    print("=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    predictions = nn.forward(X)

    print(f"\n{'x1':<5} {'x2':<5} {'Expected':<10} {'Predicted':<12} {'Raw Output':<15}")
    print("-" * 60)
    for i in range(len(X)):
        pred_class = 1 if predictions[i, 0] > 0.5 else 0
        print(f"{X[i, 0]:<5} {X[i, 1]:<5} {y[i, 0]:<10} {pred_class:<12} {predictions[i, 0]:.6f}")

    accuracy = np.mean((predictions > 0.5) == y) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.title('XOR Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, alpha=0.3)
    plt.savefig('xor_training_loss.png', dpi=300, bbox_inches='tight')
    print("\n✅ Loss curve saved: xor_training_loss.png")

    print()
    print("🎉 XOR SOLVED! The 1969 AI Winter is over.")
    print("Backpropagation + Multi-layer = Universal Approximation")
    print("Any pattern. Any function. Just add more neurons.")

solve_xor()

# ==========================================
# VISUALIZING DECISION BOUNDARIES
# ==========================================

def visualize_decision_boundary():
    """
    Let's SEE how the network learned to separate XOR.
    This is geometry. This is your Linear Algebra visualized.
    """

    # Train a network
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=5000, learning_rate=0.5)

    # Create mesh grid
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in mesh
    mesh_input = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(mesh_input)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))

    # Contour plot (decision boundary)
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Network Output')

    # Plot XOR points
    colors = ['red' if label == 0 else 'blue' for label in y.flatten()]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

    # Labels
    for i, (x1, x2) in enumerate(X):
        plt.annotate(f'({x1},{x2})→{y[i,0]}',
                    xy=(x1, x2),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')

    plt.title('XOR Decision Boundary - Neural Network', fontsize=14, fontweight='bold')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.savefig('xor_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("\n✅ Decision boundary visualized: xor_decision_boundary.png")
    print()
    print("Notice: The boundary is NON-LINEAR")
    print("A single perceptron could never draw this curve")
    print("But with hidden layers, we bend space itself")

visualize_decision_boundary()

print("\n" + "=" * 70)
print("PART 2 COMPLETE: You've mastered backpropagation.")
print("You've seen the chain rule become learning.")
print("You've solved XOR. You've ended the AI winter.")
print("Now: Real applications with TensorFlow.")
print("=" * 70)
