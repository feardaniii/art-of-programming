import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 2: Pairplot pentru Iris Dataset")
print("="*70)

# ==================== STEP 1: Load Iris Dataset ====================
print("\n🌸 STEP 1: Încărcare dataset Iris")
print("="*70)

# Load the famous Iris dataset (flower measurements)
df_iris = sns.load_dataset("iris")

print(f"✓ Dataset încărcat: {df_iris.shape[0]} flori, {df_iris.shape[1]} coloane")

# Explore the data
print("\nPrimele 10 rânduri:")
print(df_iris.head(10))

print("\n📋 Informații despre coloane:")
print(df_iris.info())

print("\n🔍 Coloanele disponibile:")
print(f"  Numerice: {df_iris.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"  Categorice: {df_iris.select_dtypes(exclude=[np.number]).columns.tolist()}")

print(f"\n🌺 Specii de iris: {df_iris['species'].unique()}")
print(f"  Număr flori per specie:")
print(df_iris['species'].value_counts())

# ==================== STEP 2: Basic Statistics ====================
print("\n" + "="*70)
print("📊 STEP 2: Statistici descriptive")
print("="*70)

print("\nStatistici generale:")
print(df_iris.describe())

print("\n📈 Statistici per specie:")
for species in df_iris['species'].unique():
    print(f"\n{species.upper()}:")
    print(df_iris[df_iris['species'] == species].describe().T[['mean', 'min', 'max']])

# ==================== STEP 3: Simple Pairplot (Without Hue) ====================
print("\n" + "="*70)
print("🎨 STEP 3: Pairplot simplu (fără hue)")
print("="*70)

# Set style
sns.set_theme(style="ticks")

# Create basic pairplot
print("⏳ Se generează pairplot-ul (poate dura câteva secunde)...")
pairplot_simple = sns.pairplot(df_iris)

plt.suptitle('Iris Dataset - Pairplot Simplu', 
             y=1.02, fontsize=14, fontweight='bold')

plt.tight_layout()
print("✓ Pairplot simplu creat (toate florile într-o culoare)")

# Save
plt.savefig('iris_pairplot_simple.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'iris_pairplot_simple.png'")

plt.show()

# ==================== STEP 4: Pairplot WITH Hue (THE MAGIC!) ====================
print("\n" + "="*70)
print("✨ STEP 4: Pairplot cu HUE (culori per specie) - MAGIC!")
print("="*70)

# Create pairplot with species colored
print("⏳ Se generează pairplot colorat...")
pairplot_hue = sns.pairplot(
    df_iris,
    hue='species',  # Color by species - THIS IS THE MAGIC! 🎨
    palette='Set2',  # Beautiful color scheme
    diag_kind='kde',  # Density curves on diagonal instead of histograms
    plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'black', 'linewidth': 0.5},
    diag_kws={'alpha': 0.7, 'linewidth': 2}
)

# Add title
pairplot_hue.fig.suptitle('Iris Dataset - Explorare Completă (Colored by Species)', 
                          y=1.01, fontsize=14, fontweight='bold')

plt.tight_layout()
print("✓ Pairplot cu hue creat")
print("  🔴 setosa")
print("  🟢 versicolor")
print("  🔵 virginica")

# Save
plt.savefig('iris_pairplot_with_hue.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'iris_pairplot_with_hue.png'")

plt.show()

# ==================== STEP 5: Custom Pairplot (Advanced) ====================
print("\n" + "="*70)
print("🎯 STEP 5: Pairplot personalizat (doar variabile selective)")
print("="*70)

# Create pairplot with only specific variables
print("⏳ Se generează pairplot personalizat...")
pairplot_custom = sns.pairplot(
    df_iris,
    vars=['petal_length', 'petal_width'],  # Only these 2 variables
    hue='species',
    palette='husl',
    height=3,  # Size of each subplot
    markers=['o', 's', 'D'],  # Different markers per species
    plot_kws={'s': 80, 'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 0.8}
)

pairplot_custom.fig.suptitle('Iris - Focus pe Petale', 
                             y=1.02, fontsize=14, fontweight='bold')

plt.tight_layout()
print("✓ Pairplot personalizat (doar petal_length și petal_width)")

# Save
plt.savefig('iris_pairplot_petals.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'iris_pairplot_petals.png'")

plt.show()

# ==================== STEP 6: Analysis - What Do We See? ====================
print("\n" + "="*70)
print("🔍 STEP 6: Analiză - Ce Observăm?")
print("="*70)

# Calculate correlations
print("\n📊 Matricea de corelații:")
correlations = df_iris.drop('species', axis=1).corr()
print(correlations)

# Find strongest correlations
print("\n💡 Cele mai puternice corelații:")
# Get upper triangle of correlation matrix
mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
upper_triangle = correlations.where(mask)

# Find top correlations
strong_corr = []
for column in upper_triangle.columns:
    for index in upper_triangle.index:
        value = upper_triangle.loc[index, column]
        if pd.notna(value) and abs(value) > 0.9:
            strong_corr.append((index, column, value))

for var1, var2, corr in strong_corr:
    print(f"  • {var1} ↔ {var2}: {corr:.3f}")

# ==================== STEP 7: Key Insights ====================
print("\n" + "="*70)
print("💡 DESCOPERIRI IMPORTANTE")
print("="*70)

print("\n🌸 Ce am descoperit din Pairplot:")
print("  1. Setosa este CLAR separată de celelalte 2 specii")
print("     → Petalele ei sunt mult mai mici")
print("  2. Petal_length și petal_width sunt FOARTE corelate")
print("     → Dacă una crește, și cealaltă crește")
print("  3. Versicolor și Virginica au suprapunere")
print("     → Mai greu de separat")
print("  4. Sepal_width are corelație slabă cu restul")
print("     → Nu e foarte util pentru clasificare")

# Verify these insights with numbers
print("\n📊 Verificare numerică:")
print(f"  Setosa - petal_length medie: {df_iris[df_iris['species']=='setosa']['petal_length'].mean():.2f} cm")
print(f"  Versicolor - petal_length medie: {df_iris[df_iris['species']=='versicolor']['petal_length'].mean():.2f} cm")
print(f"  Virginica - petal_length medie: {df_iris[df_iris['species']=='virginica']['petal_length'].mean():.2f} cm")

corr_petal = df_iris['petal_length'].corr(df_iris['petal_width'])
print(f"\n  Corelație petal_length ↔ petal_width: {corr_petal:.3f} (FOARTE PUTERNICĂ!)")

# ==================== STEP 8: Comparison - Pairplot vs Manual ====================
print("\n" + "="*70)
print("⚖️ STEP 8: Comparație - Pairplot vs Manual")
print("="*70)

print("\n🤯 Câte plot-uri am creat cu pairplot?")
n_vars = 4  # 4 numeric columns
n_plots = n_vars * n_vars
print(f"  • {n_vars} variabile → {n_plots} plot-uri în matrice!")
print(f"  • {n_vars} histograme pe diagonală")
print(f"  • {n_plots - n_vars} scatter plots off-diagonal")

print("\n⏱️ Timp necesar:")
print("  Cu Pairplot:    1 linie de cod, 5 secunde")
print("  Manual:         ~50+ linii de cod, 20+ minute")
print("\n  → Pairplot e de 100x mai rapid! 🚀")