import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

"""
🧠 CAN WE BEAT ML WITH SIMPLE IF-ELSE?

Spoiler: For Iris, YES. Here's why:
- Setosa has tiny petals (easy to spot)
- Versicolor vs Virginica: slightly trickier, but still separable

Let's find the magic thresholds!
"""

print("=" * 70)
print("🔬 RULE-BASED CLASSIFICATION: No ML, Just Logic")
print("=" * 70)

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create DataFrame for exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# ===== STEP 1: Find Natural Boundaries =====
print("\n📊 STEP 1: Explore Feature Ranges by Species")
print("-" * 70)

for i, feature in enumerate(feature_names):
    print(f"\n{feature}:")
    for species in [0, 1, 2]:
        values = X[y == species, i]
        name = iris.target_names[species]
        print(f"  {name:12s}: {values.min():.1f} - {values.max():.1f}  (mean: {values.mean():.2f})")

# ===== STEP 2: Discover Key Insights =====
print("\n\n💡 STEP 2: Key Observations")
print("-" * 70)

print("""
Looking at the data:

1. PETAL LENGTH is the MAGIC FEATURE:
   • Setosa:     1.0 - 1.9 cm  ← CLEARLY SEPARATED!
   • Versicolor: 3.0 - 5.1 cm
   • Virginica:  4.5 - 6.9 cm  ← Some overlap with Versicolor

2. PETAL WIDTH also helps:
   • Setosa:     0.1 - 0.6 cm  ← TINY!
   • Versicolor: 1.0 - 1.8 cm
   • Virginica:  1.4 - 2.5 cm

3. Sepal measurements have MORE OVERLAP → less useful alone
""")

# ===== STEP 3: Build Rule-Based Classifier =====
print("\n🔧 STEP 3: Build If-Else Classifier")
print("-" * 70)


def classify_iris_v1(sepal_length, sepal_width, petal_length, petal_width):
    """
    Version 1: Simple 2-rule classifier
    """
    # Rule 1: Setosa has tiny petals
    if petal_length < 2.5:
        return 0  # Setosa

    # Rule 2: Separate Versicolor vs Virginica
    if petal_width < 1.75:
        return 1  # Versicolor
    else:
        return 2  # Virginica


def classify_iris_v2(sepal_length, sepal_width, petal_length, petal_width):
    """
    Version 2: More refined rules
    """
    # Rule 1: Setosa - unmistakable tiny petals
    if petal_length < 2.5:
        return 0  # Setosa (100% confident)

    # Rule 2: Large petal width = Virginica
    if petal_width >= 1.8:
        return 2  # Virginica

    # Rule 3: Small petal width = Versicolor
    if petal_width < 1.35:
        return 1  # Versicolor

    # Rule 4: The tricky middle ground (1.35 <= width < 1.8)
    # Use petal length as tiebreaker
    if petal_length < 4.9:
        return 1  # Versicolor
    else:
        return 2  # Virginica


def classify_iris_v3(sepal_length, sepal_width, petal_length, petal_width):
    """
    Version 3: Optimized thresholds with combined features
    """
    # Rule 1: Setosa - tiny petals (this is 100% accurate)
    if petal_length < 2.5:
        return 0

    # Rule 2: For non-Setosa, use petal area approximation
    petal_area = petal_length * petal_width

    if petal_area < 7.5:
        return 1  # Versicolor (smaller petal area)
    elif petal_area > 10:
        return 2  # Virginica (larger petal area)
    else:
        # Middle ground: use width as tiebreaker
        if petal_width < 1.55:
            return 1
        else:
            return 2


# Print the rules
print("""
📋 CLASSIFIER V1 (2 Simple Rules):

   if petal_length < 2.5:
       return "Setosa"
   elif petal_width < 1.75:
       return "Versicolor"
   else:
       return "Virginica"

📋 CLASSIFIER V2 (Refined Rules):

   if petal_length < 2.5:
       return "Setosa"           # Tiny petals = always Setosa
   elif petal_width >= 1.8:
       return "Virginica"        # Wide petals = Virginica
   elif petal_width < 1.35:
       return "Versicolor"       # Narrow petals = Versicolor
   elif petal_length < 4.9:
       return "Versicolor"       # Shorter = Versicolor
   else:
       return "Virginica"        # Longer = Virginica

📋 CLASSIFIER V3 (Petal Area):

   if petal_length < 2.5:
       return "Setosa"

   petal_area = petal_length * petal_width

   if petal_area < 7.5:
       return "Versicolor"
   elif petal_area > 10:
       return "Virginica"
   else:
       # Tiebreaker
       return "Versicolor" if petal_width < 1.55 else "Virginica"
""")

# ===== STEP 4: Test All Versions =====
print("\n📊 STEP 4: Test on Full Dataset")
print("-" * 70)

# Test each classifier
classifiers = {
    "V1 (2 rules)": classify_iris_v1,
    "V2 (refined)": classify_iris_v2,
    "V3 (petal area)": classify_iris_v3
}

for name, clf in classifiers.items():
    predictions = [clf(*x) for x in X]
    acc = accuracy_score(y, predictions)
    print(f"{name}: {acc * 100:.2f}% accuracy")

# ===== STEP 5: Proper Train-Test Evaluation =====
print("\n\n🧪 STEP 5: Train-Test Split Evaluation")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Test on unseen data
print("\nTest Set Performance:")
for name, clf in classifiers.items():
    test_preds = [clf(*x) for x in X_test]
    acc = accuracy_score(y_test, test_preds)
    print(f"  {name}: {acc * 100:.2f}%")

# Compare to KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print(f"\n  KNN (K=5): {knn_acc * 100:.2f}%")

# ===== STEP 6: Detailed Analysis of Best Rule-Based =====
print("\n\n🔍 STEP 6: Detailed Analysis (V2 Classifier)")
print("-" * 70)

best_clf = classify_iris_v2
y_pred_rules = [best_clf(*x) for x in X_test]

print("\nClassification Report (Rule-Based V2):")
print(classification_report(y_test, y_pred_rules, target_names=iris.target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rules)
print(f"              Predicted")
print(f"              Set  Ver  Vir")
print(f"Actual Setosa  {cm[0]}")
print(f"       Versic  {cm[1]}")
print(f"       Virgin  {cm[2]}")

# Find misclassified samples
print("\n❌ Misclassified Samples:")
misclassified = []
for i, (actual, pred) in enumerate(zip(y_test, y_pred_rules)):
    if actual != pred:
        misclassified.append({
            'features': X_test[i],
            'actual': iris.target_names[actual],
            'predicted': iris.target_names[pred]
        })

if misclassified:
    for m in misclassified:
        print(f"  Features: {m['features']}")
        print(f"  Actual: {m['actual']} → Predicted: {m['predicted']}")
        print()
else:
    print("  None! Perfect classification!")

# ===== STEP 7: Visualization =====
print("\n\n📈 STEP 7: Visualize Decision Boundaries")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Petal dimensions with decision boundaries
ax = axes[0, 0]
colors = ['red', 'green', 'blue']
for species in [0, 1, 2]:
    mask = y == species
    ax.scatter(X[mask, 2], X[mask, 3], c=colors[species],
               label=iris.target_names[species], alpha=0.7, s=50, edgecolors='black')

# Draw decision boundaries
ax.axvline(x=2.5, color='black', linestyle='--', linewidth=2, label='Rule: petal_length=2.5')
ax.axhline(y=1.75, color='purple', linestyle='--', linewidth=2, label='Rule: petal_width=1.75')
ax.axhline(y=1.35, color='orange', linestyle=':', linewidth=2, label='Rule: petal_width=1.35')

ax.set_xlabel('Petal Length (cm)', fontsize=11)
ax.set_ylabel('Petal Width (cm)', fontsize=11)
ax.set_title('🎯 Petal Dimensions: Natural Separability', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Annotate regions
ax.annotate('SETOSA\nZone', xy=(1.5, 0.3), fontsize=12, fontweight='bold', color='red')
ax.annotate('VERSICOLOR\nZone', xy=(3.5, 1.1), fontsize=10, fontweight='bold', color='green')
ax.annotate('VIRGINICA\nZone', xy=(5.5, 2.2), fontsize=10, fontweight='bold', color='blue')

# Plot 2: Accuracy comparison
ax = axes[0, 1]
methods = ['V1\n(2 rules)', 'V2\n(refined)', 'V3\n(area)', 'KNN\n(K=5)']
accs = [
    accuracy_score(y_test, [classify_iris_v1(*x) for x in X_test]) * 100,
    accuracy_score(y_test, [classify_iris_v2(*x) for x in X_test]) * 100,
    accuracy_score(y_test, [classify_iris_v3(*x) for x in X_test]) * 100,
    knn_acc * 100
]
bars = ax.bar(methods, accs, color=['lightcoral', 'coral', 'tomato', 'steelblue'], edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('📊 Rule-Based vs KNN: Who Wins?', fontsize=12, fontweight='bold')
ax.set_ylim([85, 102])
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Feature importance (which features matter?)
ax = axes[1, 0]

# Calculate "separability score" for each feature
# (ratio of between-class variance to within-class variance)
separability = []
for i, feat in enumerate(feature_names):
    class_means = [X[y == c, i].mean() for c in [0, 1, 2]]
    overall_mean = X[:, i].mean()

    # Between-class variance
    between = sum(50 * (m - overall_mean) ** 2 for m in class_means)

    # Within-class variance
    within = sum(X[y == c, i].var() * 50 for c in [0, 1, 2])

    separability.append(between / (within + 0.001))

short_names = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W']
colors = ['gray', 'gray', 'green', 'green']
bars = ax.barh(short_names, separability, color=colors, edgecolor='black')
ax.set_xlabel('Separability Score (higher = easier to classify)', fontsize=11)
ax.set_title('🏆 Which Features Matter Most?', fontsize=12, fontweight='bold')
ax.axvline(x=np.mean(separability), color='red', linestyle='--', label='Average')

# Plot 4: The decision tree (visualized as flowchart)
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('🌳 Our If-Else "Decision Tree"', fontsize=12, fontweight='bold')

# Draw boxes and arrows
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Root question
box1 = FancyBboxPatch((3, 8), 4, 1.5, boxstyle="round,pad=0.1",
                      facecolor='lightyellow', edgecolor='black', linewidth=2)
ax.add_patch(box1)
ax.text(5, 8.75, 'petal_length < 2.5?', ha='center', va='center', fontsize=10, fontweight='bold')

# Yes -> Setosa
ax.annotate('', xy=(2, 6.5), xytext=(4, 8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(2.5, 7.3, 'YES', fontsize=9, color='green', fontweight='bold')

box2 = FancyBboxPatch((0.5, 5.5), 3, 1, boxstyle="round,pad=0.1",
                      facecolor='lightcoral', edgecolor='black', linewidth=2)
ax.add_patch(box2)
ax.text(2, 6, '🌸 SETOSA', ha='center', va='center', fontsize=10, fontweight='bold')

# No -> Next question
ax.annotate('', xy=(6, 6.5), xytext=(6, 8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(6.3, 7.3, 'NO', fontsize=9, color='red', fontweight='bold')

box3 = FancyBboxPatch((4, 5), 4, 1.5, boxstyle="round,pad=0.1",
                      facecolor='lightyellow', edgecolor='black', linewidth=2)
ax.add_patch(box3)
ax.text(6, 5.75, 'petal_width < 1.75?', ha='center', va='center', fontsize=10, fontweight='bold')

# Yes -> Versicolor
ax.annotate('', xy=(4.5, 3.5), xytext=(5.5, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(4.5, 4.3, 'YES', fontsize=9, color='green', fontweight='bold')

box4 = FancyBboxPatch((2.5, 2.5), 3, 1, boxstyle="round,pad=0.1",
                      facecolor='lightgreen', edgecolor='black', linewidth=2)
ax.add_patch(box4)
ax.text(4, 3, '🌿 VERSICOLOR', ha='center', va='center', fontsize=10, fontweight='bold')

# No -> Virginica
ax.annotate('', xy=(7.5, 3.5), xytext=(6.5, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(7.5, 4.3, 'NO', fontsize=9, color='red', fontweight='bold')

box5 = FancyBboxPatch((6, 2.5), 3, 1, boxstyle="round,pad=0.1",
                      facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(box5)
ax.text(7.5, 3, '💜 VIRGINICA', ha='center', va='center', fontsize=10, fontweight='bold')

# Add note
ax.text(5, 0.8, '✨ Just 2 if-else statements = 90%+ accuracy!',
        ha='center', fontsize=11, fontweight='bold', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('rule_based_classifier.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: rule_based_classifier.png")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("🎯 CONCLUSION: When Do You Need ML?")
print("=" * 70)

print("""
📊 RESULTS SUMMARY:
   Rule-Based V1 (2 rules):  ~93-97%
   Rule-Based V2 (refined):  ~97-100%
   K-NN (K=5):               ~97-100%

🤔 SO... IS ML OVERKILL HERE?

For Iris specifically: YES! Simple rules work great because:
   ✓ Low dimensions (only 4 features)
   ✓ Clear class separation
   ✓ No noise in the data
   ✓ Features have physical meaning

⚠️ BUT ML WINS WHEN:
   • High dimensions (images: millions of pixels)
   • Complex, non-linear boundaries
   • Features interact in unexpected ways
   • You can't visualize the data
   • Patterns are subtle or noisy

💡 REAL INSIGHT:
   "If you can draw the boundary by hand, you might not need ML."

   Iris flowers: ✓ Easy to visualize, easy rules
   Cat vs Dog:  ✗ Which pixel values make a cat? No simple rule exists!
""")

print("\n🔑 THE TAKEAWAY:")
print("-" * 50)
print("""
   Rule-based systems are:
   ✓ Interpretable (you can explain every decision)
   ✓ Fast (no model loading, just comparisons)
   ✓ Debuggable (fix wrong predictions easily)
   ✓ No training needed

   ML is better when:
   ✗ Patterns are too complex for humans to find
   ✗ Too many features to visualize
   ✗ Relationships are non-linear
   ✗ Data changes over time (ML can retrain)
""")