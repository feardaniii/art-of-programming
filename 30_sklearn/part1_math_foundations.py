import numpy as np
import matplotlib.pyplot as plt

"""
🧮 MATHEMATICAL FOUNDATIONS OF MACHINE LEARNING

Before we use sklearn, let's understand the MATH that powers it.
These concepts aren't theoretical - they're what makes machines learn!
"""

# ===== CONCEPT 1: DERIVATIVES = "How fast is it changing?" =====
print("="*70)
print("📐 DERIVATIVES: The Foundation of Learning")
print("="*70)

def simple_function(x):
    """A simple quadratic function: f(x) = x²"""
    return x ** 2

def derivative_at_point(x, h=0.0001):
    """
    Calculate derivative using limit definition
    This is what calculus actually computes!
    """
    # Derivative = (f(x+h) - f(x)) / h as h approaches 0
    return (simple_function(x + h) - simple_function(x)) / h

# Test at x = 3
x_point = 3
slope = derivative_at_point(x_point)

print(f"\nAt x = {x_point}:")
print(f"   f(x) = {simple_function(x_point)}")
print(f"   Derivative (slope) = {slope:.4f}")
print(f"   Analytical derivative = {2*x_point}")  # For f(x)=x², derivative is 2x

print("\n💡 WHY THIS MATTERS FOR ML:")
print("   In machine learning, the 'function' is your model's error")
print("   The derivative tells us: 'Should I increase or decrease parameters?'")
print("   This is literally how neural networks learn!")

# ===== CONCEPT 2: GRADIENT DESCENT = "Rolling Down Hill" =====
print("\n" + "="*70)
print("⛰️  GRADIENT DESCENT: How Machines Find Best Fit")
print("="*70)

def loss_function(w):
    """
    A loss function: measures how 'wrong' our model is
    Goal: find w that minimizes this!
    """
    return (w - 5)**2 + 10  # Minimum at w = 5

def gradient_of_loss(w):
    """Derivative of loss function"""
    return 2 * (w - 5)

# Gradient Descent Algorithm
w = 0  # Start at random point
learning_rate = 0.1
history = [w]

print(f"\nStarting at w = {w}")
print(f"   Loss: {loss_function(w):.4f}\n")

for step in range(20):
    # Calculate gradient (direction of steepest ascent)
    grad = gradient_of_loss(w)

    # Move OPPOSITE to gradient (descent!)
    w = w - learning_rate * grad
    history.append(w)

    if step % 5 == 0:
        print(f"Step {step:2d}: w = {w:.4f}, Loss = {loss_function(w):.4f}")

print(f"\n✅ Converged to w = {w:.4f} (true minimum: 5.0)")
print(f"   Final loss: {loss_function(w):.4f}")

print("\n💡 THIS IS LITERALLY HOW SKLEARN TRAINS MODELS:")
print("   1. Start with random parameters")
print("   2. Calculate how wrong you are (loss)")
print("   3. Calculate derivative (gradient)")
print("   4. Update parameters in opposite direction")
print("   5. Repeat until convergence")

# ===== CONCEPT 3: COSINE SIMILARITY = "How similar are two things?" =====
print("\n" + "="*70)
print("📏 COSINE SIMILARITY: Measuring Similarity in High Dimensions")
print("="*70)

def cosine_similarity(vec1, vec2):
    """
    Measures similarity between two vectors
    Result: -1 (opposite) to +1 (identical)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Example: Movie preferences
# Features: [Action, Comedy, Drama, Horror, Romance]
user_a = np.array([5, 2, 1, 0, 3])  # Loves action & romance
user_b = np.array([4, 1, 2, 0, 4])  # Similar taste!
user_c = np.array([1, 5, 4, 3, 1])  # Prefers comedy & drama

sim_ab = cosine_similarity(user_a, user_b)
sim_ac = cosine_similarity(user_a, user_c)

print("\nMovie Preference Vectors:")
print(f"   User A: {user_a} (Action & Romance fan)")
print(f"   User B: {user_b} (Similar taste)")
print(f"   User C: {user_c} (Comedy & Drama fan)")

print(f"\nSimilarity Scores:")
print(f"   A vs B: {sim_ab:.4f} → Similar!")
print(f"   A vs C: {sim_ac:.4f} → Different")

print("\n💡 REAL-WORLD USES:")
print("   • Netflix: 'Users similar to you watched...'")
print("   • Spotify: Song recommendation")
print("   • Document similarity")
print("   • Face recognition (compare face vectors)")

# ===== CONCEPT 4: DISTANCE METRICS = "How far apart are they?" =====
print("\n" + "="*70)
print("📐 DISTANCE METRICS: Different Ways to Measure 'Distance'")
print("="*70)

def euclidean_distance(p1, p2):
    """Straight-line distance (like measuring with a ruler)"""
    return np.sqrt(np.sum((p1 - p2)**2))

def manhattan_distance(p1, p2):
    """City-block distance (like walking on a grid)"""
    return np.sum(np.abs(p1 - p2))

# Example: Two houses with features [sqm, price_k, age_years]
house_a = np.array([80, 200, 5])   # 80m², 200k€, 5 years old
house_b = np.array([82, 205, 6])   # Similar house
house_c = np.array([120, 350, 1])  # Different house

euclidean_ab = euclidean_distance(house_a, house_b)
euclidean_ac = euclidean_distance(house_a, house_c)

manhattan_ab = manhattan_distance(house_a, house_b)
manhattan_ac = manhattan_distance(house_a, house_c)

print("\nHouse Comparison:")
print(f"   House A: {house_a} (80m², 200k, 5yr)")
print(f"   House B: {house_b} (Similar)")
print(f"   House C: {house_c} (Luxury)")

print(f"\nEuclidean Distance:")
print(f"   A → B: {euclidean_ab:.2f} (close)")
print(f"   A → C: {euclidean_ac:.2f} (far)")

print(f"\nManhattan Distance:")
print(f"   A → B: {manhattan_ab:.2f}")
print(f"   A → C: {manhattan_ac:.2f}")

print("\n💡 WHEN TO USE WHICH:")
print("   Euclidean: Default for most ML (k-NN, k-Means)")
print("   Manhattan: Grid-based problems, city distances")
print("   Cosine: High-dimensional data, text, recommendations")

# ===== VISUALIZATION: Gradient Descent in Action =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left plot: Loss function with descent path
w_range = np.linspace(-2, 12, 100)
loss_range = loss_function(w_range)

axes[0].plot(w_range, loss_range, 'b-', linewidth=2, label='Loss Function')
axes[0].plot(history, [loss_function(w) for w in history],
            'ro-', markersize=4, linewidth=1.5, label='Gradient Descent Path')
axes[0].axvline(x=5, color='g', linestyle='--', alpha=0.5, label='True Minimum')
axes[0].set_xlabel('Parameter w', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Gradient Descent: Rolling Down to Minimum', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right plot: Cosine similarity visualization
angles = np.linspace(0, 2*np.pi, 100)
unit_circle_x = np.cos(angles)
unit_circle_y = np.sin(angles)

# User vectors (normalized for visualization)
user_a_norm = user_a[:2] / np.linalg.norm(user_a[:2])  # Use first 2 dims
user_b_norm = user_b[:2] / np.linalg.norm(user_b[:2])
user_c_norm = user_c[:2] / np.linalg.norm(user_c[:2])

axes[1].plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.3)
axes[1].arrow(0, 0, user_a_norm[0], user_a_norm[1], head_width=0.05,
             head_length=0.05, fc='red', ec='red', linewidth=2, label='User A')
axes[1].arrow(0, 0, user_b_norm[0], user_b_norm[1], head_width=0.05,
             head_length=0.05, fc='blue', ec='blue', linewidth=2, label='User B')
axes[1].arrow(0, 0, user_c_norm[0], user_c_norm[1], head_width=0.05,
             head_length=0.05, fc='green', ec='green', linewidth=2, label='User C')
axes[1].set_xlabel('Dimension 1 (e.g., Action)', fontsize=12)
axes[1].set_ylabel('Dimension 2 (e.g., Comedy)', fontsize=12)
axes[1].set_title('Cosine Similarity: Angle Between Vectors', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.savefig('ml_math_foundations.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: ml_math_foundations.png")

print("\n" + "="*70)
print("🎓 YOU NOW UNDERSTAND THE MATH BEHIND ML!")
print("="*70)
print("   Derivatives → Tell us which direction to optimize")
print("   Gradient Descent → The algorithm that finds best fit")
print("   Cosine Similarity → Measures similarity in high dimensions")
print("   Distance Metrics → Foundation of clustering & classification")
print("\n🚀 NOW you're ready for sklearn - with UNDERSTANDING, not memorization!")
