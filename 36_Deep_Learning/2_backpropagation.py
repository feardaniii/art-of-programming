"""
================================================================================
NEURAL NETWORKS & BACKPROPAGATION
================================================================================

Course: The Art of Programming - Deep Learning Fundamentals
Lesson: From Single Neuron to Multi-Layer Networks

PREREQUISITES:
    - Completed the Neuron lesson
    - Understand: A single neuron draws ONE line
    - Understand: XOR cannot be separated by one line

THE BIG IDEA:
    One neuron = one line = limited
    Multiple neurons = multiple lines = can separate ANYTHING

    But how do we train multiple neurons together?
    Answer: BACKPROPAGATION (the chain rule from calculus)

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# PART 1: WHY DO WE NEED MULTIPLE LAYERS?
# =============================================================================

"""
RECAP FROM NEURON LESSON:

    Single neuron computes: output = σ(w₁x₁ + w₂x₂ + b)

    This is ONE LINE in 2D space.

    XOR needs (0,0) and (1,1) on one side, (0,1) and (1,0) on the other.
    NO single line can do this.

THE SOLUTION:

    What if we use TWO neurons to draw TWO lines?
    Then a THIRD neuron to combine them?

    Layer 1 (Hidden): Draw multiple lines
    Layer 2 (Output): Combine lines into regions

    Let's see this visually...
"""


def visualize_xor_solution():
    """
    Show how two lines can solve XOR when combined.
    """
    print("=" * 70)
    print("WHY MULTIPLE LAYERS? Solving XOR with Two Lines")
    print("=" * 70)
    print()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    x_range = np.linspace(-0.5, 1.5, 100)

    # Panel 1: The XOR problem
    ax = axes[0]
    ax.scatter([0, 1], [0, 1], c='red', s=200, label='Output = 0', zorder=5)
    ax.scatter([0, 1], [1, 0], c='green', s=200, label='Output = 1', zorder=5)
    ax.set_title('XOR Problem\n(Impossible with 1 line)', fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')

    # Panel 2: Hidden neuron 1 (line 1)
    ax = axes[1]
    ax.scatter([0, 1], [0, 1], c='red', s=200, zorder=5)
    ax.scatter([0, 1], [1, 0], c='green', s=200, zorder=5)
    # Line 1: x1 + x2 = 0.5 (separates (0,0) from rest)
    line1_y = 0.5 - x_range
    ax.plot(x_range, line1_y, 'b-', linewidth=2, label='Line 1: x₁+x₂=0.5')
    ax.fill_between(x_range, -0.5, line1_y, alpha=0.2, color='blue')
    ax.set_title('Hidden Neuron 1\nSeparates (0,0)', fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')

    # Panel 3: Hidden neuron 2 (line 2)
    ax = axes[2]
    ax.scatter([0, 1], [0, 1], c='red', s=200, zorder=5)
    ax.scatter([0, 1], [1, 0], c='green', s=200, zorder=5)
    # Line 2: x1 + x2 = 1.5 (separates (1,1) from rest)
    line2_y = 1.5 - x_range
    ax.plot(x_range, line2_y, 'orange', linewidth=2, label='Line 2: x₁+x₂=1.5')
    ax.fill_between(x_range, line2_y, 2, alpha=0.2, color='orange')
    ax.set_title('Hidden Neuron 2\nSeparates (1,1)', fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')

    # Panel 4: Combined (the XOR region)
    ax = axes[3]
    ax.scatter([0, 1], [0, 1], c='red', s=200, zorder=5)
    ax.scatter([0, 1], [1, 0], c='green', s=200, zorder=5)
    ax.plot(x_range, 0.5 - x_range, 'b-', linewidth=2)
    ax.plot(x_range, 1.5 - x_range, 'orange', linewidth=2)
    # Fill the region BETWEEN the two lines
    ax.fill_between(x_range, np.maximum(-0.5, 0.5 - x_range),
                    np.minimum(1.5, 1.5 - x_range),
                    alpha=0.3, color='green', label='Output = 1')
    ax.set_title('Combined: Output Layer\nRegion BETWEEN lines = 1', fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('xor_two_lines.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("The insight:")
    print("  - Hidden neuron 1: Is the point ABOVE line 1? (x₁+x₂ > 0.5)")
    print("  - Hidden neuron 2: Is the point BELOW line 2? (x₁+x₂ < 1.5)")
    print("  - Output neuron: Are BOTH conditions true?")
    print()
    print("  (0,0): Below line 1 → No")
    print("  (1,1): Above line 2 → No")
    print("  (0,1): Between both lines → Yes!")
    print("  (1,0): Between both lines → Yes!")
    print()
    print("Two lines + combination = XOR solved!")
    print()


# =============================================================================
# PART 2: NETWORK ARCHITECTURE
# =============================================================================

"""
NETWORK STRUCTURE FOR XOR:

    INPUT          HIDDEN           OUTPUT
    LAYER          LAYER            LAYER

     x₁ ----→  [Neuron 1] ---→
         \   /              \
          \ /                →  [Neuron 3] ---→  output
          / \                →
         /   \              /
     x₂ ----→  [Neuron 2] ---→


    - Input layer: 2 values (x₁, x₂)
    - Hidden layer: 2 neurons (draw 2 lines)
    - Output layer: 1 neuron (combine the lines)

WHAT EACH LAYER COMPUTES:

    Hidden layer:
        h₁ = σ(w₁₁·x₁ + w₁₂·x₂ + b₁)  ← "Is point above line 1?"
        h₂ = σ(w₂₁·x₁ + w₂₂·x₂ + b₂)  ← "Is point above line 2?"

    Output layer:
        out = σ(v₁·h₁ + v₂·h₂ + c)     ← "Combine the answers"

THE PARAMETERS WE NEED TO LEARN:
    - W₁: weights from input to hidden (2×2 = 4 weights)
    - b₁: biases for hidden layer (2 biases)
    - W₂: weights from hidden to output (2×1 = 2 weights)
    - b₂: bias for output layer (1 bias)

    Total: 9 parameters to learn
"""


def show_network_architecture():
    """
    Visualize the network structure.
    """
    print()
    print("=" * 70)
    print("NETWORK ARCHITECTURE: Input → Hidden → Output")
    print("=" * 70)
    print()

    print("For XOR, we use a 2-2-1 network:")
    print()
    print("    INPUT          HIDDEN           OUTPUT")
    print("    LAYER          LAYER            LAYER")
    print()
    print("                   ┌─────┐")
    print("     x₁ ─────────→│  h₁  │─────────┐")
    print("          ╲      ╱ └─────┘          │    ┌─────┐")
    print("           ╲    ╱                   ├───→│ out │───→  prediction")
    print("            ╲  ╱                    │    └─────┘")
    print("             ╲╱                     │")
    print("             ╱╲                     │")
    print("            ╱  ╲                    │")
    print("           ╱    ╲   ┌─────┐         │")
    print("     x₂ ─────────→│  h₂  │─────────┘")
    print("                   └─────┘")
    print()
    print("Parameters:")
    print("  W₁ (2×2): Weights from input to hidden")
    print("  b₁ (2):   Biases for hidden neurons")
    print("  W₂ (2×1): Weights from hidden to output")
    print("  b₂ (1):   Bias for output neuron")
    print()
    print("Total: 4 + 2 + 2 + 1 = 9 learnable parameters")
    print()


# =============================================================================
# PART 3: FORWARD PROPAGATION (Step by Step)
# =============================================================================

"""
FORWARD PROPAGATION: How data flows through the network

Given input (x₁, x₂), compute the output step by step:

Step 1: Input to Hidden
    z₁ = W₁ · x + b₁        (weighted sums for hidden layer)
    h = σ(z₁)               (apply activation)

Step 2: Hidden to Output  
    z₂ = W₂ · h + b₂        (weighted sum for output)
    out = σ(z₂)             (apply activation)

Let's trace through with actual numbers...
"""


def forward_propagation_example():
    """
    Trace forward propagation with specific numbers.
    """
    print()
    print("=" * 70)
    print("FORWARD PROPAGATION: Tracing Data Through the Network")
    print("=" * 70)
    print()

    # Define a simple network with known weights
    # These weights approximately solve XOR
    W1 = np.array([[1, 1],  # Weights for h1: both inputs matter equally
                   [1, 1]])  # Weights for h2: same
    b1 = np.array([-0.5, -1.5])  # Biases: different thresholds!

    W2 = np.array([[1],  # h1 contributes positively
                   [-1]])  # h2 contributes negatively
    b2 = np.array([-0.5])

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    print("Network weights (pre-set for demonstration):")
    print(f"  W₁ = [[{W1[0, 0]}, {W1[0, 1]}],   b₁ = [{b1[0]}, {b1[1]}]")
    print(f"        [{W1[1, 0]}, {W1[1, 1]}]]")
    print()
    print(f"  W₂ = [[{W2[0, 0]}],               b₂ = [{b2[0]}]")
    print(f"        [{W2[1, 0]}]]")
    print()

    # Test all XOR inputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = [0, 1, 1, 0]

    print("Tracing each input through the network:")
    print("-" * 70)

    for (x1, x2), target in zip(test_inputs, expected):
        x = np.array([x1, x2])

        print(f"\nInput: x = [{x1}, {x2}]  (target: {target})")

        # Hidden layer
        z1 = W1 @ x + b1
        h = sigmoid(z1)

        print(f"  Hidden layer:")
        print(f"    z₁ = W₁·x + b₁")
        print(f"       = [{W1[0, 0]}·{x1} + {W1[0, 1]}·{x2} + {b1[0]}, "
              f"{W1[1, 0]}·{x1} + {W1[1, 1]}·{x2} + {b1[1]}]")
        print(f"       = [{z1[0]:.2f}, {z1[1]:.2f}]")
        print(f"    h  = σ(z₁) = [{h[0]:.4f}, {h[1]:.4f}]")

        # Interpretation
        print(f"    → h₁={h[0]:.2f}: Point is {'ABOVE' if h[0] > 0.5 else 'BELOW'} line 1")
        print(f"    → h₂={h[1]:.2f}: Point is {'ABOVE' if h[1] > 0.5 else 'BELOW'} line 2")

        # Output layer
        z2 = W2.T @ h + b2
        out = sigmoid(z2)

        print(f"  Output layer:")
        print(f"    z₂ = W₂·h + b₂")
        print(f"       = {W2[0, 0]}·{h[0]:.4f} + {W2[1, 0]}·{h[1]:.4f} + {b2[0]}")
        print(f"       = {z2[0]:.4f}")
        print(f"    out = σ(z₂) = {out[0]:.4f}")

        prediction = 1 if out[0] > 0.5 else 0
        correct = "✓" if prediction == target else "✗"
        print(f"  Prediction: {prediction} {correct}")

    print()
    print("-" * 70)
    print("Key insight: The hidden layer creates a NEW representation!")
    print("  Original space: (x₁, x₂) where XOR is not linearly separable")
    print("  Hidden space:   (h₁, h₂) where XOR IS linearly separable")
    print()


# =============================================================================
# PART 4: THE LOSS FUNCTION
# =============================================================================

"""
THE LOSS FUNCTION: Measuring How Wrong We Are

For a single prediction:
    Loss = ½(target - output)²

Why squared?
    - Always positive (wrong is wrong, doesn't matter which direction)
    - Larger errors are penalized more
    - Nice derivative: ∂Loss/∂output = -(target - output)

For multiple samples:
    Total Loss = (1/n) × Σ ½(target_i - output_i)²

OUR GOAL: Adjust weights to MINIMIZE the loss.
"""


def visualize_loss_function():
    """
    Show what the loss function looks like.
    """
    print()
    print("=" * 70)
    print("THE LOSS FUNCTION: How Wrong Is Our Network?")
    print("=" * 70)
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Loss vs output for target=1
    ax = axes[0]
    outputs = np.linspace(0, 1, 100)
    target = 1
    losses = 0.5 * (target - outputs) ** 2

    ax.plot(outputs, losses, 'b-', linewidth=2)
    ax.axvline(x=target, color='green', linestyle='--', label=f'Target = {target}')
    ax.scatter([0.3], [0.5 * (1 - 0.3) ** 2], c='red', s=100, zorder=5,
               label=f'Output=0.3, Loss={0.5 * (1 - 0.3) ** 2:.3f}')
    ax.scatter([0.7], [0.5 * (1 - 0.7) ** 2], c='orange', s=100, zorder=5,
               label=f'Output=0.7, Loss={0.5 * (1 - 0.7) ** 2:.3f}')
    ax.scatter([1.0], [0], c='green', s=100, zorder=5,
               label=f'Output=1.0, Loss=0 (perfect!)')

    ax.set_xlabel('Network Output')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Function (Target = 1)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss surface for a simple case
    ax = axes[1]

    # Imagine we only have one weight to adjust
    weights = np.linspace(-2, 2, 100)
    # Fake loss surface (usually bowl-shaped)
    fake_loss = 0.5 * (weights - 0.8) ** 2 + 0.1

    ax.plot(weights, fake_loss, 'b-', linewidth=2)
    ax.scatter([0.8], [0.1], c='green', s=100, zorder=5, label='Minimum (optimal weight)')
    ax.scatter([-0.5], [0.5 * (-0.5 - 0.8) ** 2 + 0.1], c='red', s=100, zorder=5,
               label='Current weight')

    # Arrow showing gradient descent direction
    ax.annotate('', xy=(0.3, 0.35), xytext=(-0.3, 0.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(-0.1, 0.55, 'Gradient\nDescent', fontsize=9, ha='center')

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Surface (Finding the Best Weight)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_function.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("The loss function is like a landscape.")
    print("  - High loss = mountain tops (bad weights)")
    print("  - Low loss = valleys (good weights)")
    print("  - We want to reach the lowest valley")
    print()
    print("Gradient descent: Roll downhill!")
    print("  - Calculate the slope (gradient)")
    print("  - Take a step in the downhill direction")
    print("  - Repeat until we reach the bottom")
    print()


# =============================================================================
# PART 5: BACKPROPAGATION - THE CHAIN RULE
# =============================================================================

"""
BACKPROPAGATION: Assigning Blame for Errors

The key question: How much should we change each weight?

For the OUTPUT layer weights, it's easy:
    Output is directly connected to the error.

For the HIDDEN layer weights, it's harder:
    Hidden neurons affect the output, which affects the error.
    We need to trace back the influence: ERROR → OUTPUT → HIDDEN

This is the CHAIN RULE from calculus!

    ∂Loss/∂W₁ = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂W₁

              = (how output affects loss)
                × (how hidden affects output)
                × (how W₁ affects hidden)

INTUITION: Blame Assignment
    - Output neuron made a mistake
    - Which hidden neurons contributed to that mistake?
    - Which weights caused those hidden neurons to fire?
    - Adjust those weights proportionally to their contribution
"""


def explain_chain_rule():
    """
    Explain the chain rule with a concrete example.
    """
    print()
    print("=" * 70)
    print("BACKPROPAGATION: The Chain Rule as Blame Assignment")
    print("=" * 70)
    print()

    print("Imagine a factory assembly line:")
    print()
    print("  Raw materials → Worker A → Worker B → Worker C → Final product")
    print("       (input)     (hidden)   (hidden)   (output)    (output)")
    print()
    print("The final product has a defect. Who is responsible?")
    print()
    print("  Worker C (output): Directly touched the product")
    print("  Worker B (hidden): Passed work to C")
    print("  Worker A (hidden): Passed work to B who passed to C")
    print()
    print("Blame flows BACKWARD through the chain!")
    print()
    print("In math, this is the CHAIN RULE:")
    print()
    print("  ∂Loss     ∂Loss    ∂output   ∂hidden₂   ∂hidden₁")
    print("  ───── = ──────── × ─────── × ──────── × ────────")
    print("  ∂W₁     ∂output    ∂hidden₂   ∂hidden₁    ∂W₁")
    print()
    print("We multiply the 'blame' at each step as we go backward.")
    print()


def backprop_step_by_step():
    """
    Show backpropagation with actual numbers.
    """
    print()
    print("=" * 70)
    print("BACKPROPAGATION: Step-by-Step with Numbers")
    print("=" * 70)
    print()

    # Simple example: single training sample
    x = np.array([1, 0])  # Input
    target = 1  # Expected output (this is XOR of 1,0)

    # Initialize small random weights
    np.random.seed(42)
    W1 = np.array([[0.5, 0.3],
                   [0.2, 0.4]])
    b1 = np.array([0.1, -0.1])

    W2 = np.array([[0.6],
                   [0.5]])
    b2 = np.array([0.1])

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(s):
        """Derivative: σ'(z) = σ(z)·(1-σ(z)), and s=σ(z)"""
        return s * (1 - s)

    print("Given:")
    print(f"  Input x = {x}")
    print(f"  Target = {target}")
    print()

    # =========================================
    # FORWARD PASS
    # =========================================
    print("=" * 50)
    print("FORWARD PASS")
    print("=" * 50)

    # Hidden layer
    z1 = W1 @ x + b1
    h = sigmoid(z1)

    print(f"\nHidden layer:")
    print(f"  z₁ = W₁·x + b₁ = {z1}")
    print(f"  h  = σ(z₁)     = {h}")

    # Output layer
    z2 = W2.T @ h + b2
    output = sigmoid(z2)

    print(f"\nOutput layer:")
    print(f"  z₂  = W₂·h + b₂ = {z2}")
    print(f"  out = σ(z₂)     = {output}")

    # Loss
    loss = 0.5 * (target - output[0]) ** 2
    print(f"\nLoss = ½(target - output)² = {loss:.6f}")

    # =========================================
    # BACKWARD PASS
    # =========================================
    print()
    print("=" * 50)
    print("BACKWARD PASS (Backpropagation)")
    print("=" * 50)

    # Step 1: Output layer gradients
    print("\nStep 1: Output layer gradients")
    print("-" * 40)

    # ∂Loss/∂output = -(target - output)
    dL_dout = -(target - output[0])
    print(f"  ∂Loss/∂output = -(target - output) = {dL_dout:.6f}")

    # ∂output/∂z₂ = σ'(z₂) = output·(1-output)
    dout_dz2 = sigmoid_derivative(output[0])
    print(f"  ∂output/∂z₂ = output·(1-output) = {dout_dz2:.6f}")

    # δ₂ = ∂Loss/∂z₂ (we call this "delta" - the error signal)
    delta2 = dL_dout * dout_dz2
    print(f"  δ₂ = ∂Loss/∂z₂ = ∂Loss/∂output × ∂output/∂z₂ = {delta2:.6f}")

    # Gradients for W₂ and b₂
    # ∂z₂/∂W₂ = h, so ∂Loss/∂W₂ = δ₂ × h
    dL_dW2 = delta2 * h
    dL_db2 = delta2
    print(f"\n  ∂Loss/∂W₂ = δ₂ × h = {dL_dW2}")
    print(f"  ∂Loss/∂b₂ = δ₂     = {dL_db2:.6f}")

    # Step 2: Hidden layer gradients (this is the key!)
    print("\nStep 2: Hidden layer gradients (backprop through hidden layer)")
    print("-" * 40)

    # ∂Loss/∂h = ∂Loss/∂z₂ × ∂z₂/∂h = δ₂ × W₂
    dL_dh = delta2 * W2.flatten()
    print(f"  ∂Loss/∂h = δ₂ × W₂ = {dL_dh}")
    print(f"    This tells us how much each hidden neuron contributed to the error")

    # ∂h/∂z₁ = σ'(z₁) = h·(1-h)
    dh_dz1 = sigmoid_derivative(h)
    print(f"\n  ∂h/∂z₁ = h·(1-h) = {dh_dz1}")

    # δ₁ = ∂Loss/∂z₁
    delta1 = dL_dh * dh_dz1
    print(f"  δ₁ = ∂Loss/∂h × ∂h/∂z₁ = {delta1}")

    # Gradients for W₁ and b₁
    # ∂z₁/∂W₁ = x, so ∂Loss/∂W₁ = δ₁ × x
    dL_dW1 = np.outer(x, delta1)
    dL_db1 = delta1
    print(f"\n  ∂Loss/∂W₁ = outer(x, δ₁) =")
    print(f"    {dL_dW1}")
    print(f"  ∂Loss/∂b₁ = δ₁ = {dL_db1}")

    # Step 3: Update weights
    print("\nStep 3: Update weights (gradient descent)")
    print("-" * 40)

    learning_rate = 0.5
    print(f"  Learning rate = {learning_rate}")
    print()
    print(f"  W₂_new = W₂ - lr × ∂Loss/∂W₂")
    print(f"         = {W2.flatten()} - {learning_rate} × {dL_dW2}")
    W2_new = W2.flatten() - learning_rate * dL_dW2
    print(f"         = {W2_new}")
    print()
    print(f"  W₁_new = W₁ - lr × ∂Loss/∂W₁")
    W1_new = W1 - learning_rate * dL_dW1.T
    print(f"         = (shown as matrix)")
    print(f"         {W1_new}")

    print()
    print("=" * 50)
    print("That's ONE step of backpropagation!")
    print("Repeat for all training samples, many times (epochs)")
    print("=" * 50)


# =============================================================================
# PART 6: THE COMPLETE NEURAL NETWORK CLASS
# =============================================================================

class NeuralNetwork:
    """
    A 2-layer neural network (1 hidden layer).

    Architecture: Input → Hidden → Output

    This implementation prioritizes clarity over efficiency.
    Every step is explicit and commented.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the network with random weights.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
        """
        # Xavier initialization: scale by sqrt(fan_in)
        # Prevents vanishing/exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros(output_size)

        # Store intermediate values for backprop
        self.z1 = None  # Pre-activation of hidden layer
        self.h = None  # Activation of hidden layer (hidden outputs)
        self.z2 = None  # Pre-activation of output layer
        self.out = None  # Final output

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: σ(z) = 1/(1+e^(-z))"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, s: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: σ'(z) = σ(z)·(1-σ(z))"""
        return s * (1 - s)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Args:
            X: Input data of shape (n_samples, input_size)

        Returns:
            Output predictions of shape (n_samples, output_size)
        """
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.h = self.sigmoid(self.z1)

        # Output layer
        self.z2 = self.h @ self.W2 + self.b2
        self.out = self.sigmoid(self.z2)

        return self.out

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        """
        Backward propagation (backprop).

        Computes gradients and updates weights.

        Args:
            X: Input data of shape (n_samples, input_size)
            y: True labels of shape (n_samples, output_size)
            learning_rate: Step size for gradient descent
        """
        n_samples = X.shape[0]

        # === Output layer gradients ===
        # ∂Loss/∂out = -(y - out)
        dL_dout = -(y - self.out)

        # δ₂ = ∂Loss/∂z₂ = ∂Loss/∂out × ∂out/∂z₂
        delta2 = dL_dout * self.sigmoid_derivative(self.out)

        # Gradients for W₂ and b₂
        dL_dW2 = (self.h.T @ delta2) / n_samples
        dL_db2 = np.mean(delta2, axis=0)

        # === Hidden layer gradients ===
        # ∂Loss/∂h = δ₂ × W₂ᵀ
        dL_dh = delta2 @ self.W2.T

        # δ₁ = ∂Loss/∂z₁ = ∂Loss/∂h × ∂h/∂z₁
        delta1 = dL_dh * self.sigmoid_derivative(self.h)

        # Gradients for W₁ and b₁
        dL_dW1 = (X.T @ delta1) / n_samples
        dL_db1 = np.mean(delta1, axis=0)

        # === Update weights (gradient descent) ===
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def compute_loss(self, y: np.ndarray) -> float:
        """Compute mean squared error loss."""
        return np.mean((y - self.out) ** 2)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int, learning_rate: float,
              verbose: bool = True) -> List[float]:
        """
        Train the network.

        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of training iterations
            learning_rate: Step size
            verbose: Print progress

        Returns:
            List of loss values at each epoch
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            self.forward(X)

            # Compute loss
            loss = self.compute_loss(y)
            losses.append(loss)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Print progress
            if verbose and epoch % (epochs // 10) == 0:
                accuracy = np.mean((self.out > 0.5) == y) * 100
                print(f"Epoch {epoch:5d}: Loss = {loss:.6f}, Accuracy = {accuracy:.1f}%")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (forward pass only)."""
        return self.forward(X)


# =============================================================================
# PART 7: TRAINING ON XOR
# =============================================================================

def train_xor_network():
    """
    Train a neural network to solve XOR.
    """
    print()
    print("=" * 70)
    print("TRAINING: Learning XOR with Backpropagation")
    print("=" * 70)
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

    print("Training data (XOR):")
    print("  x₁  x₂  │  target")
    print("  ────────┼────────")
    for i in range(len(X)):
        print(f"   {X[i][0]}   {X[i][1]}  │    {y[i][0]}")
    print()

    # Create network
    np.random.seed(42)
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    print("Network: 2 inputs → 4 hidden neurons → 1 output")
    print()

    # Train
    print("Training for 5000 epochs...")
    print("-" * 50)
    losses = nn.train(X, y, epochs=5000, learning_rate=1.0)

    # Test
    print()
    print("-" * 50)
    print("\nFinal results:")
    print("  x₁  x₂  │ target │  output  │ prediction")
    print("  ────────┼────────┼──────────┼───────────")

    predictions = nn.predict(X)
    for i in range(len(X)):
        pred = 1 if predictions[i, 0] > 0.5 else 0
        correct = "✓" if pred == y[i, 0] else "✗"
        print(f"   {X[i][0]}   {X[i][1]}  │   {y[i][0]}    │  {predictions[i, 0]:.4f}  │     {pred}  {correct}")

    accuracy = np.mean((predictions > 0.5) == y) * 100
    print(f"\nFinal accuracy: {accuracy:.0f}%")

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('XOR Learning Curve', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale shows the decrease better
    plt.savefig('xor_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✓ XOR solved! The network learned to separate the classes.")

    return nn


# =============================================================================
# PART 8: VISUALIZING THE DECISION BOUNDARY
# =============================================================================

def visualize_decision_boundary(nn: NeuralNetwork):
    """
    Visualize what the trained network learned.
    """
    print()
    print("=" * 70)
    print("VISUALIZATION: The Learned Decision Boundary")
    print("=" * 70)
    print()

    # Create a grid of points
    x1_range = np.linspace(-0.5, 1.5, 200)
    x2_range = np.linspace(-0.5, 1.5, 200)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Flatten grid for prediction
    grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])

    # Predict for all grid points
    predictions = nn.predict(grid_points)
    predictions = predictions.reshape(xx1.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Contour plot
    contour = ax.contourf(xx1, xx2, predictions, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(contour, label='Network Output')

    # Decision boundary (0.5 contour)
    ax.contour(xx1, xx2, predictions, levels=[0.5], colors='black', linewidths=2)

    # Plot XOR points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    for i in range(len(X)):
        color = 'red' if y[i] == 0 else 'blue'
        ax.scatter(X[i, 0], X[i, 1], c=color, s=300, edgecolors='black',
                   linewidth=2, zorder=5)
        ax.annotate(f'({X[i, 0]},{X[i, 1]})→{y[i]}',
                    xy=(X[i, 0], X[i, 1]),
                    xytext=(15, 10),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold')

    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('XOR Decision Boundary\n(Black line = decision boundary at 0.5)',
                 fontweight='bold', fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('xor_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Notice: The decision boundary is CURVED!")
    print("A single neuron can only draw straight lines.")
    print("But multiple neurons combine to create curves.")
    print()
    print("This is the power of depth.")
    print()


# =============================================================================
# MAIN: RUN THE COMPLETE LESSON
# =============================================================================

def main():
    """Run the complete backpropagation lesson."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "NEURAL NETWORKS & BACKPROPAGATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Part 1: Why multiple layers?
    visualize_xor_solution()
    input("\nPress Enter to continue to network architecture...")

    # Part 2: Network architecture
    show_network_architecture()
    input("\nPress Enter to see forward propagation...")

    # Part 3: Forward propagation
    forward_propagation_example()
    input("\nPress Enter to understand the loss function...")

    # Part 4: Loss function
    visualize_loss_function()
    input("\nPress Enter to learn about backpropagation...")

    # Part 5: Backpropagation explanation
    explain_chain_rule()
    input("\nPress Enter to see backprop step-by-step...")

    backprop_step_by_step()
    input("\nPress Enter to train the network...")

    # Part 7: Train on XOR
    nn = train_xor_network()
    input("\nPress Enter to visualize what the network learned...")

    # Part 8: Visualize decision boundary
    visualize_decision_boundary(nn)

    # Summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "LESSON COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("What you learned:")
    print("  1. Multiple neurons draw multiple lines")
    print("  2. Hidden layers create new representations")
    print("  3. Backpropagation = chain rule = blame assignment")
    print("  4. Error signal flows backward through the network")
    print("  5. Each weight is adjusted proportional to its 'blame'")
    print()
    print("The 1969 XOR problem is solved.")
    print("Any pattern can now be learned with enough neurons and layers.")
    print()
    print("Next lesson: Deep Networks & Modern Architectures")
    print()


if __name__ == "__main__":
    main()