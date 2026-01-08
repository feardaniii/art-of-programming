import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

"""
🤖 K-NEAREST NEIGHBORS: The Algorithm That Thinks Like Humans

When you see a dog, you don't analyze pixels - you compare it to dogs
you've seen before. KNN does exactly this: classify based on similarity
to known examples.

This is 'lazy learning' - the model doesn't really 'learn' during training.
It just memorizes examples and compares new cases to them. Simple, powerful.
"""

print("="*70)
print("🤖 BUILDING YOUR FIRST CLASSIFIER: K-Nearest Neighbors")
print("="*70)

# ===== STEP 1: Load and Prepare Data =====
print("\n📊 STEP 1: Preparing the Iris Dataset")
print("-" * 70)

iris = load_iris()
X = iris.data  # Features: 4 measurements
y = iris.target  # Labels: 0, 1, 2 (species)

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# ===== STEP 2: The Golden Rule - Train-Test Split =====
print("\n✂️  STEP 2: Train-Test Split")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% for testing
    random_state=42,  # For reproducibility
    stratify=y  # Keep class proportions in both sets
)

print(f"Training set: {len(X_train)} flowers")
print(f"Test set: {len(X_test)} flowers")

print("\n💡 WHY SPLIT?")
print("   Training data: Teach the model patterns")
print("   Test data: Evaluate on UNSEEN flowers (simulates real world)")
print("   → Never test on training data - that's cheating!")

print("\nClass distribution in train/test:")
print(f"   Training: {np.bincount(y_train)}")
print(f"   Testing:  {np.bincount(y_test)}")
print("   → Balanced due to stratify=y")

# ===== STEP 3: Understanding K-Nearest Neighbors =====
print("\n\n🧠 STEP 3: How K-NN Actually Works")
print("-" * 70)

print("\n📐 The Algorithm (Step by Step):")
print("   1. Choose K (number of neighbors to check)")
print("   2. For a new flower:")
print("      a) Calculate distance to ALL training flowers")
print("      b) Find K nearest neighbors")
print("      c) Count species among those K neighbors")
print("      d) Majority vote wins!")

print("\n💡 INTUITION:")
print("   'You are the average of the 5 people you spend most time with'")
print("   K-NN applies this to classification:")
print("   'A flower is likely the species of its K nearest neighbors'")

print("\n🎯 CHOOSING K:")
print("   K=1: Too sensitive to noise (nearest neighbor might be outlier)")
print("   K=3 or 5: Good balance")
print("   K=large: Too general (might include wrong species)")

# ===== STEP 4: Training K-NN (Actually Just Memorizing) =====
print("\n\n🎓 STEP 4: 'Training' K-NN Model")
print("-" * 70)

# Try different K values
k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    # Create and train model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Evaluate
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    results.append({
        'k': k,
        'train_acc': train_accuracy,
        'test_acc': test_accuracy
    })

    print(f"\nK = {k}:")
    print(f"   Training accuracy: {train_accuracy*100:.2f}%")
    print(f"   Test accuracy:     {test_accuracy*100:.2f}%")

# Find best K
results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df['test_acc'].idxmax(), 'k']
print(f"\n✅ Best K = {int(best_k)} (highest test accuracy)")

# ===== STEP 5: Deep Dive with Best Model =====
print("\n\n🔍 STEP 5: Detailed Analysis with K=5")
print("-" * 70)

# Train with best K
best_knn = KNeighborsClassifier(n_neighbors=5)
best_knn.fit(X_train, y_train)

# Make predictions
y_pred = best_knn.predict(X_test)

print("\nFirst 10 predictions:")
print("-" * 50)
for i in range(min(10, len(X_test))):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]
    match = "✓" if y_test[i] == y_pred[i] else "✗"

    print(f"{i+1}. Flower: [{X_test[i][0]:.1f}, {X_test[i][1]:.1f}, "
          f"{X_test[i][2]:.1f}, {X_test[i][3]:.1f}]")
    print(f"   Actual: {actual:12s} | Predicted: {predicted:12s} {match}")

# ===== STEP 6: Confusion Matrix - Where Does It Fail? =====
print("\n\n📊 STEP 6: Confusion Matrix Analysis")
print("-" * 70)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                Predicted")
print("              Set  Ver  Vir")
print("Actual Setosa ", cm[0])
print("       Versic ", cm[1])
print("       Virgin ", cm[2])

print("\n💡 HOW TO READ:")
print("   Diagonal = Correct predictions")
print("   Off-diagonal = Mistakes")

# Analyze mistakes
total_errors = len(y_test) - np.trace(cm)
if total_errors > 0:
    print(f"\n⚠️  Total errors: {total_errors}/{len(y_test)}")
    if cm[1, 2] > 0 or cm[2, 1] > 0:
        print("   Most confusion: Versicolor ↔ Virginica")
        print("   → Expected! These species have overlapping features")
else:
    print("\n🎉 Perfect classification on test set!")

# ===== STEP 7: Classification Report =====
print("\n\n📈 STEP 7: Detailed Classification Report")
print("-" * 70)

print("\n" + classification_report(
    y_test, y_pred,
    target_names=iris.target_names,
    digits=4
))

print("💡 METRICS EXPLAINED:")
print("   PRECISION: Of all predicted setosa, how many were actually setosa?")
print("   RECALL: Of all actual setosa, how many did we find?")
print("   F1-SCORE: Harmonic mean of precision & recall")
print("   SUPPORT: Number of flowers of each species in test set")

# ===== STEP 8: The Real Test - New Flower Classification =====
print("\n\n🌸 STEP 8: Classify a New Mystery Flower")
print("-" * 70)

# Example: new flower found in the wild
new_flower = np.array([[5.8, 2.7, 5.1, 1.9]])

print("\nBotanist finds a flower with measurements:")
print(f"   Sepal: {new_flower[0][0]} cm × {new_flower[0][1]} cm")
print(f"   Petal: {new_flower[0][2]} cm × {new_flower[0][3]} cm")

# Get prediction
predicted_class = best_knn.predict(new_flower)[0]
predicted_species = iris.target_names[predicted_class]

# Get probability distribution
probabilities = best_knn.predict_proba(new_flower)[0]

print(f"\n🤖 K-NN Classification Result:")
print(f"   Predicted species: {predicted_species}")
print(f"\n   Confidence breakdown:")
for i, species in enumerate(iris.target_names):
    prob = probabilities[i]
    bar = "█" * int(prob * 20)
    print(f"   {species:12s}: {prob*100:5.1f}% {bar}")

# Explain the classification
distances, indices = best_knn.kneighbors(new_flower)
print(f"\n📏 5 Nearest Neighbors (and their species):")
for i, idx in enumerate(indices[0]):
    neighbor_species = iris.target_names[y_train[idx]]
    dist = distances[0][i]
    print(f"   {i+1}. Distance: {dist:.3f} → {neighbor_species}")

# ===== VISUALIZATION =====
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: K value comparison
axes[0, 0].plot(results_df['k'], results_df['train_acc'], 'o-', label='Training', linewidth=2)
axes[0, 0].plot(results_df['k'], results_df['test_acc'], 's-', label='Testing', linewidth=2)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best K={int(best_k)}')
axes[0, 0].set_xlabel('K (Number of Neighbors)', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Finding Optimal K: Train vs Test Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.85, 1.05])

# Plot 2: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=iris.target_names,
           yticklabels=iris.target_names,
           ax=axes[0, 1], cbar_kws={'label': 'Count'})
axes[0, 1].set_xlabel('Predicted Species', fontsize=11)
axes[0, 1].set_ylabel('Actual Species', fontsize=11)
axes[0, 1].set_title('Confusion Matrix: Where K-NN Makes Mistakes', fontsize=12, fontweight='bold')

# Plot 3: Decision boundary (2D projection - petal dimensions)
# Create mesh for decision boundary
h = 0.02  # step size
x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train KNN on just petal dimensions for visualization
knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X[:, 2:4], y)

# Predict on mesh
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
axes[1, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
axes[1, 0].scatter(X[:, 2], X[:, 3], c=y, cmap='viridis',
                  edgecolors='black', linewidth=0.5, s=50, alpha=0.7)
axes[1, 0].set_xlabel('Petal Length (cm)', fontsize=11)
axes[1, 0].set_ylabel('Petal Width (cm)', fontsize=11)
axes[1, 0].set_title('K-NN Decision Boundary (K=5)', fontsize=12, fontweight='bold')

# Plot 4: Prediction confidence bar chart
species_names = iris.target_names
axes[1, 1].barh(species_names, probabilities, color=['red', 'green', 'blue'], alpha=0.7)
axes[1, 1].set_xlabel('Probability', fontsize=11)
axes[1, 1].set_title(f'New Flower Classification: Predicted as {predicted_species}',
                    fontsize=12, fontweight='bold')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('knn_classifier_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: knn_classifier_analysis.png")

print("\n" + "="*70)
print("🎉 CONGRATULATIONS! You built a working classifier!")
print("="*70)
print("\nWhat you achieved:")
print("   ✓ Trained K-NN on 120 iris flowers")
print("   ✓ Tested on 30 unseen flowers")
print(f"   ✓ Achieved {best_knn.score(X_test, y_test)*100:.2f}% accuracy")
print("   ✓ Classified new mystery flower")
print("   ✓ Understood confusion matrix")
print("\n💡 This exact algorithm (with more features) is used for:")
print("   • Medical diagnosis (patient symptoms → disease)")
print("   • Customer segmentation (behavior → customer type)")
print("   • Fraud detection (transaction patterns → fraud/legit)")
print("   • Recommendation systems (user similarity → recommendations)")
