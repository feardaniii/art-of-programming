import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 3: Heatmap de Corelații - Penguins Dataset")
print("="*70)

# ==================== STEP 1: Load Penguins Dataset ====================
print("\n🐧 STEP 1: Încărcare dataset Penguins")
print("="*70)

# Load penguins dataset
df_penguins = sns.load_dataset("penguins")

print(f"✓ Dataset încărcat: {df_penguins.shape[0]} pinguini, {df_penguins.shape[1]} coloane")

# Explore the data
print("\nPrimele 5 rânduri:")
print(df_penguins.head())

print("\n📋 Informații despre coloane:")
print(df_penguins.info())

print("\n🔍 Coloane disponibile:")
numeric_cols = df_penguins.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_penguins.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"  Numerice: {numeric_cols}")
print(f"  Categorice: {categorical_cols}")

print(f"\n🐧 Specii de pinguini: {df_penguins['species'].unique()}")
print(f"  Număr pinguini per specie:")
print(df_penguins['species'].value_counts())

# Check for missing values
print(f"\n⚠️ Valori lipsă:")
missing = df_penguins.isnull().sum()
print(missing[missing > 0])

# ==================== STEP 2: Calculate Correlation Matrix ====================
print("\n" + "="*70)
print("🔢 STEP 2: Calculare matrice de corelații")
print("="*70)

# Calculate correlations (only numeric columns)
correlation_matrix = df_penguins.corr(numeric_only=True)

print("✓ Matrice de corelații calculată")
print("\n📊 Matricea de corelații:")
print(correlation_matrix)

# Find strongest correlations
print("\n💡 Cele mai puternice corelații (|r| > 0.5):")
# Get upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
upper = correlation_matrix.where(mask)

strong_corrs = []
for col in upper.columns:
    for idx in upper.index:
        value = upper.loc[idx, col]
        if pd.notna(value) and abs(value) > 0.5:
            strong_corrs.append((idx, col, value))

# Sort by absolute value
strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

for var1, var2, corr in strong_corrs:
    emoji = "📈" if corr > 0 else "📉"
    print(f"  {emoji} {var1} ↔ {var2}: {corr:.3f}")

# ==================== STEP 3: Basic Heatmap ====================
print("\n" + "="*70)
print("🎨 STEP 3: Heatmap simplu")
print("="*70)

# Set style
sns.set_theme(style="white")

# Create figure
plt.figure(figsize=(10, 8))

# Basic heatmap
sns.heatmap(correlation_matrix, 
            cmap='coolwarm',  # Blue-white-red color scheme
            center=0,  # Center colormap at 0
            square=True,  # Square cells
            linewidths=1,  # Lines between cells
            cbar_kws={"shrink": 0.8})  # Colorbar size

plt.title('Penguins - Matrice de Corelații (Basic)', 
          fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
print("✓ Heatmap simplu creat")

# Save
plt.savefig('penguins_heatmap_basic.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'penguins_heatmap_basic.png'")

plt.show()

# ==================== STEP 4: Heatmap with Annotations (BONUS) ====================
print("\n" + "="*70)
print("✨ STEP 4: Heatmap cu ADNOTĂRI (valorile numerice)")
print("="*70)

# Create figure
plt.figure(figsize=(12, 9))

# Heatmap with values displayed
sns.heatmap(correlation_matrix,
            annot=True,  # Show correlation values - THE MAGIC! ✨
            fmt='.2f',  # Format: 2 decimal places
            cmap='coolwarm',  # Color scheme
            center=0,
            square=True,
            linewidths=2,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "label": "Corelație"},
            annot_kws={"size": 11, "weight": "bold"})  # Annotation styling

plt.title('Penguins - Matrice de Corelații (cu Valori Numerice)', 
          fontsize=15, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
print("✓ Heatmap cu adnotări creat")
print("  📊 Valorile sunt afișate în fiecare celulă!")

# Save
plt.savefig('penguins_heatmap_annotated.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'penguins_heatmap_annotated.png'")

plt.show()

# ==================== STEP 5: Different Color Schemes ====================
print("\n" + "="*70)
print("🎨 STEP 5: Comparație scheme de culori")
print("="*70)

# Create figure with 4 different color schemes
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Color scheme 1: coolwarm (blue-red)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=axes[0, 0],
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[0, 0].set_title('Coolwarm (Blue-Red)', fontweight='bold', fontsize=12)

# Color scheme 2: RdYlGn (red-yellow-green)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='RdYlGn', center=0, ax=axes[0, 1],
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[0, 1].set_title('RdYlGn (Red-Yellow-Green)', fontweight='bold', fontsize=12)

# Color scheme 3: viridis (purple-yellow)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='viridis', ax=axes[1, 0],
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[1, 0].set_title('Viridis (Purple-Yellow)', fontweight='bold', fontsize=12)

# Color scheme 4: rocket (dark-light)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='rocket', ax=axes[1, 1],
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
axes[1, 1].set_title('Rocket (Dark-Light)', fontweight='bold', fontsize=12)

fig.suptitle('Penguins Correlations - Different Color Schemes', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
print("✓ 4 scheme de culori create pentru comparație")

# Save
plt.savefig('penguins_heatmap_colorschemes.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'penguins_heatmap_colorschemes.png'")

plt.show()

# ==================== STEP 6: Masked Heatmap (Lower Triangle Only) ====================
print("\n" + "="*70)
print("🎯 STEP 6: Heatmap mascat (doar jumătate - elegant!)")
print("="*70)

# Create mask for upper triangle (we only need one half - matrix is symmetric!)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create figure
plt.figure(figsize=(12, 9))

# Heatmap with mask
sns.heatmap(correlation_matrix,
            mask=mask,  # Hide upper triangle - MORE ELEGANT! ✨
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=2,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "label": "Coeficient de Corelație"},
            annot_kws={"size": 12, "weight": "bold"})

plt.title('Penguins - Matrice de Corelații (Triangular - Elegant)', 
          fontsize=15, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
print("✓ Heatmap mascat creat (doar triunghi inferior)")
print("  💡 Mai puțin clutter - matricea e simetrică oricum!")

# Save
plt.savefig('penguins_heatmap_masked.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'penguins_heatmap_masked.png'")

plt.show()

# ==================== STEP 7: Analysis & Insights ====================
print("\n" + "="*70)
print("🔍 STEP 7: Analiză și Descoperiri")
print("="*70)

print("\n📊 Interpretarea Corelațiilor:")
print("\n1️⃣ CORELAȚII POZITIVE PUTERNICE (r > 0.7):")
for var1, var2, corr in strong_corrs:
    if corr > 0.7:
        print(f"   📈 {var1} ↔ {var2}: {corr:.3f}")
        print(f"      → Când {var1} crește, {var2} crește și el!")

print("\n2️⃣ CORELAȚII NEGATIVE MODERATE (r < -0.4):")
for var1, var2, corr in strong_corrs:
    if corr < -0.4:
        print(f"   📉 {var1} ↔ {var2}: {corr:.3f}")
        print(f"      → Când {var1} crește, {var2} scade!")

print("\n3️⃣ CORELAȚII SLABE (|r| < 0.3):")
weak_corrs = []
for col in upper.columns:
    for idx in upper.index:
        value = upper.loc[idx, col]
        if pd.notna(value) and abs(value) < 0.3:
            weak_corrs.append((idx, col, value))

if weak_corrs:
    for var1, var2, corr in weak_corrs[:3]:  # Show first 3
        print(f"   ➖ {var1} ↔ {var2}: {corr:.3f}")
        print(f"      → Aproape nicio relație între ele")

# ==================== STEP 8: Practical Implications ====================
print("\n" + "="*70)
print("💡 IMPLICAȚII PRACTICE")
print("="*70)

print("\n🎯 Ce înseamnă aceste corelații:")
print("\n✅ VARIABILE REDUNDANTE:")
print("   Dacă două variabile au r > 0.9, una poate fi eliminată")
print("   → Economisim resurse, simplificăm modelul")

print("\n✅ FEATURE SELECTION:")
print("   Variabilele cu corelații puternice față de target sunt importante")
print("   → Le păstrăm în model pentru predicții")

print("\n✅ MULTICOLINIARITY:")
print("   Variabile foarte corelate între ele (r > 0.8) pot cauza probleme")
print("   → În modele de regresie, una trebuie eliminată")