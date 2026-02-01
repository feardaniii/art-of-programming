import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
🌸 THE IRIS DATASET: Where Machine Learning Classification Begins

1936: Ronald Fisher publishes what becomes the most famous dataset in ML.
Not because it's complex - but because it's PERFECT for learning pattern recognition.

Today: You'll understand why THIS dataset changed statistics forever.
"""

print("="*70)
print("🌸 THE IRIS DATASET: Fisher's 1936 Classification Challenge")
print("="*70)

# ===== STEP 1: Load the Historical Dataset =====
print("\n📊 STEP 1: Loading the Iris Dataset")
print("-" * 70)

# Load Fisher's original data
iris = load_iris()

# Convert to DataFrame for better exploration
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

print(f"Loaded {len(df)} iris flowers from Fisher's original study")
print(f"\nSpecies distribution:")
print(df['species_name'].value_counts().sort_index())

print("\n💡 PERFECT BALANCE:")
print("   50 samples per species → No class imbalance problems")
print("   This is rare in real-world data!")

print("\n\nFirst 5 flowers:")
print(df.head())

# ===== STEP 2: Understanding the Features =====
print("\n\n🔍 STEP 2: What Do These Measurements Mean?")
print("-" * 70)

print("\n🌸 Anatomy of an Iris Flower:")
print("   SEPAL: Outer protective leaf (usually green)")
print("      • sepal length (cm): Length of outer petal")
print("      • sepal width (cm): Width of outer petal")
print("\n   PETAL: Inner colorful part (the 'flower' you see)")
print("      • petal length (cm): Length of colored petal")
print("      • petal width (cm): Width of colored petal")

print("\n💡 WHY THESE 4 MEASUREMENTS?")
print("   Fisher discovered these 4 numbers contain enough information")
print("   to distinguish species. Not too few (can't separate), not too")
print("   many (overfitting). This is feature selection at its finest.")

# ===== STEP 3: Statistical Overview =====
print("\n\n📈 STEP 3: Statistical Summary")
print("-" * 70)

print("\nOverall Statistics (all 150 flowers):")
print(df[iris.feature_names].describe().round(2))

print("\n\nSpecies-Specific Statistics:")
for species in ['Setosa', 'Versicolor', 'Virginica']:
    print(f"\n{species}:")
    species_data = df[df['species_name'] == species][iris.feature_names]
    print(species_data.describe().loc[['mean', 'std']].round(2))

# ===== STEP 4: The Key Insight - Separability =====
print("\n\n🎯 STEP 4: Fisher's Key Discovery - Class Separability")
print("-" * 70)

# Calculate feature importance for separation
for feature in iris.feature_names:
    setosa_mean = df[df['species_name'] == 'Setosa'][feature].mean()
    versicolor_mean = df[df['species_name'] == 'Versicolor'][feature].mean()
    virginica_mean = df[df['species_name'] == 'Virginica'][feature].mean()

    # Calculate variance between species
    species_variance = np.var([setosa_mean, versicolor_mean, virginica_mean])

    print(f"\n{feature}:")
    print(f"   Setosa:     {setosa_mean:.2f} cm")
    print(f"   Versicolor: {versicolor_mean:.2f} cm")
    print(f"   Virginica:  {virginica_mean:.2f} cm")
    print(f"   → Separation power: {species_variance:.3f}")

print("\n💡 OBSERVATION:")
print("   Petal measurements have HIGH variance between species")
print("   → These features are most discriminative!")
print("   Sepal width has LOW variance")
print("   → Less useful for classification")

# ===== STEP 5: Visual Pattern Recognition =====
print("\n\n👁️ STEP 5: Can YOU See the Patterns?")
print("-" * 70)

print("\nLet's visualize what Fisher discovered in 1936...")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Petal length vs width - THE MONEY SHOT
colors = ['red', 'green', 'blue']
species_names = ['Setosa', 'Versicolor', 'Virginica']

for idx, species in enumerate(species_names):
    species_data = df[df['species_name'] == species]
    axes[0, 0].scatter(
        species_data['petal length (cm)'],
        species_data['petal width (cm)'],
        c=colors[idx],
        label=species,
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )

axes[0, 0].set_xlabel('Petal Length (cm)', fontsize=11)
axes[0, 0].set_ylabel('Petal Width (cm)', fontsize=11)
axes[0, 0].set_title('The Money Shot: Petal Dimensions Separate Species!',
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Add annotation showing perfect separation
axes[0, 0].annotate('Setosa\n(clearly separated)',
                   xy=(1.5, 0.3), fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
axes[0, 0].annotate('Versicolor & Virginica\n(overlapping)',
                   xy=(5, 1.5), fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))

# Plot 2: Sepal dimensions - less clear separation
for idx, species in enumerate(species_names):
    species_data = df[df['species_name'] == species]
    axes[0, 1].scatter(
        species_data['sepal length (cm)'],
        species_data['sepal width (cm)'],
        c=colors[idx],
        label=species,
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )

axes[0, 1].set_xlabel('Sepal Length (cm)', fontsize=11)
axes[0, 1].set_ylabel('Sepal Width (cm)', fontsize=11)
axes[0, 1].set_title('Sepal Dimensions: More Overlap', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of petal length
for idx, species in enumerate(species_names):
    species_data = df[df['species_name'] == species]['petal length (cm)']
    axes[1, 0].hist(species_data, bins=15, alpha=0.5,
                   label=species, color=colors[idx])

axes[1, 0].set_xlabel('Petal Length (cm)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Petal Length Distribution by Species',
                     fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Correlation heatmap
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
           cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": .8},
           ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlations: Which Features Move Together?',
                    fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('iris_exploration.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: iris_exploration.png")

# ===== STEP 6: The Classification Challenge =====
print("\n\n🎯 STEP 6: The Challenge Fisher Solved")
print("-" * 70)

print("\nGiven a NEW iris flower with measurements:")
mystery_flower = [5.8, 2.7, 5.1, 1.9]
print(f"   Sepal Length: {mystery_flower[0]} cm")
print(f"   Sepal Width:  {mystery_flower[1]} cm")
print(f"   Petal Length: {mystery_flower[2]} cm")
print(f"   Petal Width:  {mystery_flower[3]} cm")

print("\nQuestion: Which species is it?")
print("   A) Setosa")
print("   B) Versicolor")
print("   C) Virginica")

print("\n💡 HOW HUMANS SOLVE IT:")
print("   1. Compare measurements to known flowers")
print("   2. Find the 'closest' matches")
print("   3. Majority vote wins")

print("\n🤖 HOW MACHINES SOLVE IT (K-Nearest Neighbors):")
print("   1. Calculate distance in 4D space to all 150 flowers")
print("   2. Find K nearest neighbors")
print("   3. Species that appears most among neighbors wins")

print("\n" + "="*70)
print("🎓 YOU NOW UNDERSTAND THE IRIS DATASET!")
print("="*70)
print("\nWhat makes it special:")
print("   ✓ Perfect balance (50-50-50)")
print("   ✓ Clear patterns (petal measurements separate well)")
print("   ✓ Challenging edge cases (versicolor/virginica overlap)")
print("   ✓ Low dimensional (4 features - visualizable)")
print("   ✓ Real problem (actual botanical classification)")
print("\n🚀 Next: Build a K-Nearest Neighbors classifier to solve it!")
