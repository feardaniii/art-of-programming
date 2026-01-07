import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("="*70)
print("EXERCIȚIUL 1: Histogramă Vârsta Pasagerilor Titanic")
print("="*70)

# ==================== STEP 1: Load Titanic Dataset ====================
print("\n📊 STEP 1: Încărcare dataset Titanic")
print("="*70)

# Load the Titanic dataset (built into Seaborn!)
df_titanic = sns.load_dataset("titanic")

print(f"✓ Dataset încărcat: {df_titanic.shape[0]} pasageri, {df_titanic.shape[1]} coloane")

# Explore the data
print("\nPrimele 5 rânduri:")
print(df_titanic.head())

print("\n📋 Informații despre coloane:")
print(df_titanic.info())

print("\n🔍 Verificare coloane importante:")
print(f"  - Age (vârsta): {df_titanic['age'].notna().sum()} valori valide")
print(f"  - Survived (supraviețuire): {df_titanic['survived'].notna().sum()} valori valide")
print(f"  - Survived values: {df_titanic['survived'].unique()} (0=died, 1=survived)")

# ==================== STEP 2: Basic Histogram (Without Hue) ====================
print("\n" + "="*70)
print("📈 STEP 2: Histogramă simplă (fără hue)")
print("="*70)

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Create figure
plt.figure(figsize=(10, 6))

# Create basic histogram
sns.histplot(data=df_titanic, x="age", bins=20, kde=False)

# Customize
plt.title("Distribuția Vârstei Pasagerilor Titanic", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Vârsta (ani)", fontsize=12, fontweight='bold')
plt.ylabel("Număr de Pasageri", fontsize=12, fontweight='bold')

# Add grid for readability
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
print("✓ Histogramă simplă creată")

# Save
plt.savefig('titanic_age_simple.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'titanic_age_simple.png'")

plt.show()

# ==================== STEP 3: Histogram WITH Hue (BONUS) ====================
print("\n" + "="*70)
print("🎨 STEP 3: Histogramă cu HUE (supraviețuitori vs decedați)")
print("="*70)

# Create new figure
plt.figure(figsize=(12, 6))

# Create histogram with HUE
sns.histplot(data=df_titanic, 
             x="age", 
             hue="survived",  # THIS IS THE MAGIC! 🎨
             bins=20,
             kde=False,
             palette={0: '#E74C3C', 1: '#2ECC71'},  # Red for died, green for survived
             alpha=0.6,  # Transparency so we can see overlap
             edgecolor='black',
             linewidth=0.5)

# Customize
plt.title("Distribuția Vârstei Pasagerilor Titanic (Supraviețuitori vs Decedați)", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Vârsta (ani)", fontsize=12, fontweight='bold')
plt.ylabel("Număr de Pasageri", fontsize=12, fontweight='bold')

# Customize legend
plt.legend(title='Supraviețuire', labels=['Decedat (0)', 'Supraviețuit (1)'], 
           fontsize=10, title_fontsize=11)

# Add grid
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
print("✓ Histogramă cu hue creată")
print("  - Roșu: pasageri decedați")
print("  - Verde: pasageri supraviețuitori")

# Save
plt.savefig('titanic_age_with_hue.png', dpi=300, bbox_inches='tight')
print("💾 Salvat ca 'titanic_age_with_hue.png'")

plt.show()

# ==================== STEP 4: Analysis & Statistics ====================
print("\n" + "="*70)
print("📊 STEP 4: Analiză și Statistici")
print("="*70)

# Overall age statistics
print("\n📈 Statistici generale vârstă:")
print(df_titanic['age'].describe())

# Statistics by survival
print("\n🔍 Statistici vârstă per grup:")
survival_stats = df_titanic.groupby('survived')['age'].describe()
print(survival_stats)

# Age comparison
mean_survived = df_titanic[df_titanic['survived'] == 1]['age'].mean()
mean_died = df_titanic[df_titanic['survived'] == 0]['age'].mean()

print(f"\n💡 Observații interesante:")
print(f"  - Vârsta medie supraviețuitori: {mean_survived:.1f} ani")
print(f"  - Vârsta medie decedați: {mean_died:.1f} ani")
print(f"  - Diferență: {abs(mean_survived - mean_died):.1f} ani")

# Count by survival
survived_count = df_titanic['survived'].value_counts()
print(f"\n📊 Număr pasageri:")
print(f"  - Decedați: {survived_count[0]} ({survived_count[0]/len(df_titanic)*100:.1f}%)")
print(f"  - Supraviețuitori: {survived_count[1]} ({survived_count[1]/len(df_titanic)*100:.1f}%)")

# ==================== STEP 5: Additional Visualizations (EXTRA) ====================
print("\n" + "="*70)
print("🎁 STEP 5: Vizualizări suplimentare (BONUS)")
print("="*70)

# Create figure with multiple variations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Basic histogram
sns.histplot(data=df_titanic, x="age", bins=20, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Histogramă Simplă', fontweight='bold')
axes[0, 0].set_xlabel('Vârsta')
axes[0, 0].set_ylabel('Număr Pasageri')

# Top-right: With KDE (density curve)
sns.histplot(data=df_titanic, x="age", bins=20, kde=True, ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Cu Curba de Densitate (KDE)', fontweight='bold')
axes[0, 1].set_xlabel('Vârsta')
axes[0, 1].set_ylabel('Număr Pasageri')

# Bottom-left: With hue (stacked)
sns.histplot(data=df_titanic, x="age", hue="survived", bins=20, 
             multiple="stack", ax=axes[1, 0], palette={0: '#E74C3C', 1: '#2ECC71'})
axes[1, 0].set_title('Cu Hue (Stacked)', fontweight='bold')
axes[1, 0].set_xlabel('Vârsta')
axes[1, 0].set_ylabel('Număr Pasageri')
axes[1, 0].legend(title='Survived', labels=['No', 'Yes'])

# Bottom-right: With hue (dodge - side by side)
sns.histplot(data=df_titanic, x="age", hue="survived", bins=20, 
             multiple="dodge", ax=axes[1, 1], palette={0: '#E74C3C', 1: '#2ECC71'})
axes[1, 1].set_title('Cu Hue (Side-by-Side)', fontweight='bold')
axes[1, 1].set_xlabel('Vârsta')
axes[1, 1].set_ylabel('Număr Pasageri')
axes[1, 1].legend(title='Survived', labels=['No', 'Yes'])

plt.tight_layout()
plt.savefig('titanic_age_variations.png', dpi=300, bbox_inches='tight')
print("✓ 4 variații de histograme create")
print("💾 Salvat ca 'titanic_age_variations.png'")

plt.show()