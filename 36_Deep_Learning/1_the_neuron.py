"""
================================================================================
THE NEURON: The Building Block of Neural Networks
================================================================================

Course: The Art of Programming - Deep Learning Fundamentals
Lesson: Understanding the Artificial Neuron from First Principles

LEARNING OBJECTIVES:
    1. Understand what a neuron computes (it's simpler than you think)
    2. See how weights and bias create a decision boundary
    3. Watch learning happen step-by-step (no magic, just math)
    4. Understand why a single neuron has limits

THE CORE IDEA:
    A neuron is a tiny decision-maker.
    It takes inputs, weighs their importance, and outputs a decision.

    That's it. Everything else is details.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# =============================================================================
# PART 1: WHAT IS A NEURON?
# =============================================================================

"""
ANALOGY: Should I go outside today?

You consider multiple factors:
    - Is it sunny?        (input 1)
    - Is it warm?         (input 2)
    - Do I have free time? (input 3)

But not all factors matter equally:
    - Sunny matters a lot to you     (weight = 0.8)
    - Warm matters somewhat          (weight = 0.5)
    - Free time is essential         (weight = 1.0)

You mentally "score" the day:
    score = (sunny × 0.8) + (warm × 0.5) + (free_time × 1.0)

If score > some threshold → GO OUTSIDE
Otherwise                 → STAY IN

THIS IS EXACTLY WHAT A NEURON DOES.
"""


def intuition_example():
    """
    Concrete example: Should I go outside?

    This shows exactly what a neuron computes, no code abstraction.
    """
    print("=" * 70)
    print("INTUITION: A Neuron is a Decision Maker")
    print("=" * 70)
    print()

    # Our inputs (each is 0 or 1)
    sunny = 1  # Yes, it's sunny
    warm = 1  # Yes, it's warm
    free_time = 0  # No, I'm busy

    # How much each factor matters to me (weights)
    weight_sunny = 0.8
    weight_warm = 0.5
    weight_free_time = 1.0

    # My threshold for going outside (bias, but negative)
    threshold = 1.5  # Need at least this much "score" to go out

    # Calculate the weighted sum
    score = (sunny * weight_sunny) + (warm * weight_warm) + (free_time * weight_free_time)

    print("Inputs:")
    print(f"  Sunny?     {sunny}  × weight {weight_sunny} = {sunny * weight_sunny}")
    print(f"  Warm?      {warm}  × weight {weight_warm} = {warm * weight_warm}")
    print(f"  Free time? {free_time}  × weight {weight_free_time} = {free_time * weight_free_time}")
    print()
    print(f"Weighted Sum (score): {score}")
    print(f"Threshold: {threshold}")
    print()

    if score >= threshold:
        print(f"Decision: GO OUTSIDE! (score {score} >= threshold {threshold})")
    else:
        print(f"Decision: STAY IN (score {score} < threshold {threshold})")

    print()
    print("This IS a neuron. The math is:")
    print("  output = 1 if (w₁x₁ + w₂x₂ + w₃x₃) >= threshold else 0")
    print()
    print("Or equivalently (moving threshold to the other side):")
    print("  output = 1 if (w₁x₁ + w₂x₂ + w₃x₃ - threshold) >= 0 else 0")
    print()
    print("That '-threshold' term is what we call the BIAS (b).")
    print()


# =============================================================================
# PART 2: THE MATH (It's Just a Line)
# =============================================================================

"""
A neuron with 2 inputs computes:

    z = w₁x₁ + w₂x₂ + b

This is the equation of a LINE in 2D space!

    w₁x₁ + w₂x₂ + b = 0   ←  This line divides space into two regions

Points on one side → output 1 (positive class)
Points on other side → output 0 (negative class)

THE WEIGHTS DETERMINE THE LINE'S ORIENTATION.
THE BIAS DETERMINES THE LINE'S POSITION.
"""


def visualize_decision_boundary():
    """
    Show how weights and bias create a line that separates classes.
    """
    print()
    print("=" * 70)
    print("VISUALIZATION: The Decision Boundary is a Line")
    print("=" * 70)
    print()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common setup
    x_range = np.linspace(-0.5, 1.5, 100)

    # Example 1: AND gate decision boundary
    # We need: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
    ax = axes[0]
    ax.set_title("AND Gate Decision Boundary", fontsize=12, fontweight='bold')

    # Points
    ax.scatter([0, 0, 1], [0, 1, 0], c='red', s=200, marker='o', label='Output = 0', zorder=5)
    ax.scatter([1], [1], c='green', s=200, marker='o', label='Output = 1', zorder=5)

    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # With w1=1, w2=1, b=-1.5: x1 + x2 = 1.5
    w1, w2, b = 1, 1, -1.5
    boundary_y = (-w1 * x_range - b) / w2
    ax.plot(x_range, boundary_y, 'b-', linewidth=2, label=f'Line: x₁ + x₂ = 1.5')

    # Shade regions
    ax.fill_between(x_range, boundary_y, 2, alpha=0.2, color='green')
    ax.fill_between(x_range, -0.5, boundary_y, alpha=0.2, color='red')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Annotate points
    ax.annotate('(0,0)', (0, 0), textcoords="offset points", xytext=(10, -15))
    ax.annotate('(0,1)', (0, 1), textcoords="offset points", xytext=(10, 5))
    ax.annotate('(1,0)', (1, 0), textcoords="offset points", xytext=(10, -15))
    ax.annotate('(1,1)', (1, 1), textcoords="offset points", xytext=(10, 5))

    # Example 2: OR gate decision boundary
    ax = axes[1]
    ax.set_title("OR Gate Decision Boundary", fontsize=12, fontweight='bold')

    ax.scatter([0], [0], c='red', s=200, marker='o', label='Output = 0', zorder=5)
    ax.scatter([0, 1, 1], [1, 0, 1], c='green', s=200, marker='o', label='Output = 1', zorder=5)

    # Decision boundary: x1 + x2 = 0.5
    w1, w2, b = 1, 1, -0.5
    boundary_y = (-w1 * x_range - b) / w2
    ax.plot(x_range, boundary_y, 'b-', linewidth=2, label=f'Line: x₁ + x₂ = 0.5')

    ax.fill_between(x_range, boundary_y, 2, alpha=0.2, color='green')
    ax.fill_between(x_range, -0.5, boundary_y, alpha=0.2, color='red')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Example 3: XOR - impossible with one line!
    ax = axes[2]
    ax.set_title("XOR: No Single Line Works!", fontsize=12, fontweight='bold')

    ax.scatter([0, 1], [0, 1], c='red', s=200, marker='o', label='Output = 0', zorder=5)
    ax.scatter([0, 1], [1, 0], c='green', s=200, marker='o', label='Output = 1', zorder=5)

    # Try a few lines - none work
    ax.plot(x_range, 1 - x_range, 'b--', linewidth=2, alpha=0.5, label='Try 1: fails')
    ax.plot(x_range, x_range, 'r--', linewidth=2, alpha=0.5, label='Try 2: fails')
    ax.axhline(0.5, color='purple', linestyle='--', alpha=0.5, label='Try 3: fails')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax.text(0.5, -0.3, "No single line can separate\ngreen from red!",
            ha='center', fontsize=10, color='darkred')

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Key insight: A single neuron draws ONE LINE.")
    print("If your data can be separated by one line → neuron works")
    print("If not (like XOR) → you need multiple neurons (layers)")
    print()


# =============================================================================
# PART 3: THE ACTIVATION FUNCTION (Making It Smooth)
# =============================================================================

"""
So far our neuron outputs 0 or 1 with a hard threshold.

Problem: Hard thresholds are difficult to learn from.
         Small change in weights → no change in output (until threshold crossed)

Solution: Use a SMOOTH function instead.

SIGMOID: σ(z) = 1 / (1 + e^(-z))
    - Smoothly transitions from 0 to 1
    - Output can be interpreted as probability
    - Small weight changes → small output changes (learnable!)
"""


def visualize_activation_functions():
    """
    Show what activation functions do and why we need them.
    """
    print()
    print("=" * 70)
    print("ACTIVATION FUNCTIONS: From Hard Threshold to Smooth Curves")
    print("=" * 70)
    print()

    z = np.linspace(-6, 6, 200)

    # Hard threshold (step function)
    step = np.where(z >= 0, 1, 0)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))

    # ReLU
    relu = np.maximum(0, z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Step function
    ax = axes[0]
    ax.plot(z, step, 'b-', linewidth=2)
    ax.set_title('Step Function (Hard Threshold)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z = w·x + b')
    ax.set_ylabel('output')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.text(0.5, 0.5, 'Problem:\nNo gradient\nat most points', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Sigmoid
    ax = axes[1]
    ax.plot(z, sigmoid, 'g-', linewidth=2)
    ax.set_title('Sigmoid: σ(z) = 1/(1+e⁻ᶻ)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z = w·x + b')
    ax.set_ylabel('output')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision at 0.5')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.text(2, 0.3, 'Smooth!\nGradient\neverywhere', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    # ReLU
    ax = axes[2]
    ax.plot(z, relu, 'orange', linewidth=2)
    ax.set_title('ReLU: max(0, z)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z = w·x + b')
    ax.set_ylabel('output')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.text(2, 1, 'Simple!\nFast to compute\nUsed in deep nets', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='moccasin'))

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Why sigmoid for learning?")
    print("  - Output between 0 and 1 (like a probability)")
    print("  - Smooth curve means we can compute derivatives")
    print("  - Derivative tells us: 'which direction should I adjust weights?'")
    print()


# =============================================================================
# PART 4: THE NEURON CLASS (Clean Implementation)
# =============================================================================

class Neuron:
    """
    A single artificial neuron.

    Computes: output = sigmoid(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

    Attributes:
        weights: How much each input matters
        bias: The threshold (shifted to the other side of equation)
    """

    def __init__(self, n_inputs: int):
        """
        Initialize with small random weights.

        Why random? If all weights start at 0, all neurons learn the same thing.
        Why small? Large weights → sigmoid saturates → learning stops.
        """
        self.weights = np.random.randn(n_inputs) * 0.5
        self.bias = np.random.randn() * 0.5

        # Store these for visualization
        self.n_inputs = n_inputs

    def forward(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass: compute the output for given input.

        Args:
            x: Input array of shape (n_inputs,)

        Returns:
            output: The sigmoid activation (between 0 and 1)
            z: The pre-activation value (weighted sum)
        """
        # Step 1: Weighted sum
        z = np.dot(self.weights, x) + self.bias

        # Step 2: Sigmoid activation
        output = self._sigmoid(z)

        return output, z

    def _sigmoid(self, z: float) -> float:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, output: float) -> float:
        """
        Derivative of sigmoid.

        Beautiful fact: σ'(z) = σ(z) × (1 - σ(z))
        We can compute it from the output alone!
        """
        return output * (1 - output)

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias:.4f})"


# =============================================================================
# PART 5: LEARNING (The Magic Demystified)
# =============================================================================

"""
HOW DOES A NEURON LEARN?

1. Make a prediction
2. Compare to correct answer (compute error)
3. Ask: "How should I change weights to reduce error?"
4. Update weights slightly in that direction
5. Repeat

The key question: "How should I change weights?"

Answer: GRADIENT DESCENT
    - Gradient = direction of steepest increase
    - We want to DECREASE error, so go OPPOSITE to gradient
    - Update rule: w = w - learning_rate × gradient

THE GRADIENT FOR A SINGLE NEURON:
    ∂Error/∂w = (prediction - target) × sigmoid_derivative × input

This tells us EXACTLY how to adjust each weight!
"""


def train_neuron_step_by_step():
    """
    Train a neuron to learn AND gate, showing every step.
    """
    print()
    print("=" * 70)
    print("LEARNING: Training a Neuron Step by Step")
    print("=" * 70)
    print()

    # Training data: AND gate
    # Input: [x1, x2]  Output: x1 AND x2
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])  # AND outputs

    print("Training data (AND gate):")
    print("  x₁  x₂  │  target")
    print("  ────────┼────────")
    for i in range(len(X)):
        print(f"   {X[i][0]}   {X[i][1]}  │    {y[i]}")
    print()

    # Create neuron with random weights
    np.random.seed(42)  # For reproducibility
    neuron = Neuron(n_inputs=2)

    print(f"Initial neuron:")
    print(f"  weights = [{neuron.weights[0]:.4f}, {neuron.weights[1]:.4f}]")
    print(f"  bias    = {neuron.bias:.4f}")
    print()

    # Training parameters
    learning_rate = 0.5
    epochs = 100 # 20 before, failed to grasp 1-1

    # Track history for plotting
    error_history = []
    weight_history = []

    print("Training (showing first 5 epochs in detail):")
    print("-" * 70)

    for epoch in range(epochs):
        total_error = 0

        if epoch < 5:
            print(f"\nEpoch {epoch + 1}:")

        for i in range(len(X)):
            x = X[i]
            target = y[i]

            # Forward pass
            output, z = neuron.forward(x)

            # Compute error
            error = target - output
            total_error += error ** 2

            # Compute gradient
            # ∂Error/∂w = -2 × error × σ'(z) × x
            # (The -2 is absorbed into learning rate for simplicity)
            sigmoid_deriv = neuron._sigmoid_derivative(output)
            gradient_w = error * sigmoid_deriv * x
            gradient_b = error * sigmoid_deriv

            if epoch < 5:
                print(f"  Sample {i}: x={x}, target={target}")
                print(f"    output={output:.4f}, error={error:.4f}")
                print(f"    gradient_w={gradient_w}, gradient_b={gradient_b:.4f}")

            # Update weights (gradient ASCENT because error = target - output)
            neuron.weights += learning_rate * gradient_w
            neuron.bias += learning_rate * gradient_b

        error_history.append(total_error)
        weight_history.append(neuron.weights.copy())

        if epoch < 5:
            print(f"  → Updated weights: [{neuron.weights[0]:.4f}, {neuron.weights[1]:.4f}]")
            print(f"  → Updated bias: {neuron.bias:.4f}")
            print(f"  → Total squared error: {total_error:.4f}")

    print()
    print("-" * 70)
    print(f"\nAfter {epochs} epochs:")
    print(f"  Final weights: [{neuron.weights[0]:.4f}, {neuron.weights[1]:.4f}]")
    print(f"  Final bias: {neuron.bias:.4f}")
    print()

    # Test the trained neuron
    print("Testing trained neuron:")
    print("  x₁  x₂  │ target │ output │ prediction")
    print("  ────────┼────────┼────────┼───────────")

    all_correct = True
    for i in range(len(X)):
        output, _ = neuron.forward(X[i])
        prediction = 1 if output > 0.5 else 0
        correct = "✓" if prediction == y[i] else "✗"
        if prediction != y[i]:
            all_correct = False
        print(f"   {X[i][0]}   {X[i][1]}  │   {y[i]}    │ {output:.4f} │     {prediction}  {correct}")

    print()
    if all_correct:
        print("✓ The neuron learned AND successfully!")
    else:
        print("✗ Still learning... (try more epochs)")

    # Plot learning curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(range(1, epochs + 1), error_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Squared Error')
    ax.set_title('Learning Curve: Error Decreases Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    weight_history = np.array(weight_history)
    ax.plot(range(1, epochs + 1), weight_history[:, 0], 'r-', linewidth=2, label='w₁', marker='o', markersize=4)
    ax.plot(range(1, epochs + 1), weight_history[:, 1], 'g-', linewidth=2, label='w₂', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.set_title('Weight Evolution During Training', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    return neuron


# =============================================================================
# PART 6: THE LIMITATION (XOR Problem)
# =============================================================================

def demonstrate_xor_limitation():
    """
    Show why a single neuron cannot learn XOR.
    """
    print()
    print("=" * 70)
    print("THE LIMITATION: XOR Cannot Be Learned by One Neuron")
    print("=" * 70)
    print()

    # XOR truth table
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    print("XOR truth table:")
    print("  x₁  x₂  │  target")
    print("  ────────┼────────")
    for i in range(len(X)):
        print(f"   {X[i][0]}   {X[i][1]}  │    {y[i]}")
    print()

    # Try to train a neuron
    np.random.seed(42)
    neuron = Neuron(n_inputs=2)
    learning_rate = 0.5

    print("Training for 1000 epochs...")

    best_accuracy = 0
    for epoch in range(1000):
        for i in range(len(X)):
            output, _ = neuron.forward(X[i])
            error = y[i] - output
            sigmoid_deriv = neuron._sigmoid_derivative(output)
            neuron.weights += learning_rate * error * sigmoid_deriv * X[i]
            neuron.bias += learning_rate * error * sigmoid_deriv

        # Check accuracy
        correct = sum(1 for i in range(len(X))
                      if (1 if neuron.forward(X[i])[0] > 0.5 else 0) == y[i])
        accuracy = correct / len(X)
        best_accuracy = max(best_accuracy, accuracy)

    print(f"Best accuracy achieved: {best_accuracy * 100:.0f}%")
    print()

    print("Testing final neuron:")
    print("  x₁  x₂  │ target │ output │ prediction")
    print("  ────────┼────────┼────────┼───────────")

    for i in range(len(X)):
        output, _ = neuron.forward(X[i])
        prediction = 1 if output > 0.5 else 0
        correct = "✓" if prediction == y[i] else "✗"
        print(f"   {X[i][0]}   {X[i][1]}  │   {y[i]}    │ {output:.4f} │     {prediction}  {correct}")

    print()
    print("=" * 70)
    print("WHY XOR FAILS:")
    print("=" * 70)
    print()
    print("A single neuron computes: w₁x₁ + w₂x₂ + b = 0")
    print("This is a STRAIGHT LINE in 2D space.")
    print()
    print("For XOR, we need (0,1) and (1,0) on one side,")
    print("          and (0,0) and (1,1) on the other side.")
    print()
    print("No single straight line can do this!")
    print()
    print("SOLUTION: Stack multiple neurons → NEURAL NETWORK")
    print("  - First layer: draw multiple lines")
    print("  - Second layer: combine them into curves")
    print("  - This can separate ANY pattern!")
    print()


# =============================================================================
# PART 7: INTERACTIVE EXPLORATION
# =============================================================================

def interactive_neuron_explorer():
    """
    Let students experiment with weights and see the decision boundary move.
    """
    print()
    print("=" * 70)
    print("INTERACTIVE: See How Weights Affect the Decision Boundary")
    print("=" * 70)
    print()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Different weight configurations
    configs = [
        ([1, 1], -0.5, "w=[1,1], b=-0.5\nOR Gate"),
        ([1, 1], -1.5, "w=[1,1], b=-1.5\nAND Gate"),
        ([1, 1], 0.5, "w=[1,1], b=0.5\nMost inputs activate"),
        ([2, 1], -1.5, "w=[2,1], b=-1.5\nx₁ matters more"),
        ([1, 2], -1.5, "w=[1,2], b=-1.5\nx₂ matters more"),
        ([-1, 1], 0, "w=[-1,1], b=0\nNOT x₁ AND x₂"),
    ]

    x_range = np.linspace(-0.5, 1.5, 100)

    for ax, (weights, bias, title) in zip(axes.flatten(), configs):
        w1, w2 = weights

        # Decision boundary: w1*x1 + w2*x2 + b = 0
        # → x2 = (-w1*x1 - b) / w2
        if w2 != 0:
            boundary_y = (-w1 * x_range - bias) / w2
            ax.plot(x_range, boundary_y, 'b-', linewidth=2)

            # Shade regions
            ax.fill_between(x_range, boundary_y, 2, alpha=0.3, color='green', label='Output ≈ 1')
            ax.fill_between(x_range, -0.5, boundary_y, alpha=0.3, color='red', label='Output ≈ 0')

        # Plot the four corners
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                z = w1 * x1 + w2 * x2 + bias
                output = 1 / (1 + np.exp(-z))
                color = 'green' if output > 0.5 else 'red'
                ax.scatter(x1, x2, c=color, s=200, zorder=5, edgecolor='black')
                ax.annotate(f'{output:.2f}', (x1, x2), textcoords="offset points",
                            xytext=(10, 5), fontsize=9)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('weight_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Key observations:")
    print("  - Changing weights rotates the line")
    print("  - Changing bias shifts the line")
    print("  - Higher weight for x₁ → line more vertical (x₁ matters more)")
    print("  - Negative weights flip the direction")
    print()


# =============================================================================
# MAIN: RUN THE LESSON
# =============================================================================

def main():
    """Run the complete neuron lesson."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "THE NEURON: Building Block of Neural Networks".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Part 1: Intuition
    intuition_example()
    input("\nPress Enter to continue to visualization...")

    # Part 2: Decision boundary
    visualize_decision_boundary()
    input("\nPress Enter to continue to activation functions...")

    # Part 3: Activation functions
    visualize_activation_functions()
    input("\nPress Enter to continue to learning...")

    # Part 4 & 5: Learning
    trained_neuron = train_neuron_step_by_step()
    input("\nPress Enter to see the XOR limitation...")

    # Part 6: XOR limitation
    demonstrate_xor_limitation()
    input("\nPress Enter to explore weight configurations...")

    # Part 7: Interactive exploration
    interactive_neuron_explorer()

    # Summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "LESSON COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("What you learned:")
    print("  1. A neuron computes: output = σ(w·x + b)")
    print("  2. This draws a LINE in input space")
    print("  3. Learning = adjusting weights to reduce error")
    print("  4. One neuron can only draw ONE line")
    print("  5. For complex patterns (XOR), we need MULTIPLE neurons → Networks")
    print()
    print("Next lesson: Building a Neural Network (multiple layers)")
    print()


if __name__ == "__main__":
    main()