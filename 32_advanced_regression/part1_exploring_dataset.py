import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ========================================
# PARTEA 1: CREAREA DATASET-ULUI REALIST
# ========================================

# Creăm un dataset care simulează datele de pe OLX/Imobiliare.ro
np.random.seed(42)
n_samples = 500

# Zone din București cu prețuri diferite
zone = ['Floreasca', 'Pipera', 'Militari', 'Drumul Taberei', 'Titan',
        'Vitan', 'Berceni', 'Pantelimon']
zone_prices = {
    'Floreasca': 2000, 'Pipera': 1900, 'Militari': 1200,
    'Drumul Taberei': 1300, 'Titan': 1100, 'Vitan': 1150,
    'Berceni': 900, 'Pantelimon': 950
}

data = {
    'zona': np.random.choice(zone, n_samples),
    'suprafata': np.random.randint(35, 120, n_samples),
    'numar_camere': np.random.randint(1, 5, n_samples),
    'etaj': np.random.randint(0, 11, n_samples),
    'an_constructie': np.random.randint(1970, 2024, n_samples),
    'balcon': np.random.choice(['da', 'nu'], n_samples),
    'parcare': np.random.choice(['da', 'nu'], n_samples)
}

df = pd.DataFrame(data)

# Calculăm prețul bazat pe features (cu variație realistă)
df['pret'] = df.apply(lambda row:
    zone_prices[row['zona']] * row['suprafata'] +
    row['numar_camere'] * 5000 +
    (2024 - row['an_constructie']) * -200 +
    (10000 if row['balcon'] == 'da' else 0) +
    (15000 if row['parcare'] == 'da' else 0) +
    np.random.normal(0, 15000),
    axis=1
)

# Rotunjim prețurile
df['pret'] = df['pret'].round(-3)  # Rotunjim la mii

print("📊 DATASET CREAT:")
print(f"Număr de apartamente: {len(df)}")
print(f"\nPrimele 5 rânduri:")
print(df.head())

# ========================================
# PARTEA 2: INTRODUCEREA VALORILOR LIPSĂ
# ========================================

# Simulăm missing values (ca în realitate!)
missing_indices_suprafata = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
df.loc[missing_indices_suprafata, 'suprafata'] = np.nan

missing_indices_etaj = np.random.choice(df.index, size=int(0.10 * len(df)), replace=False)
df.loc[missing_indices_etaj, 'etaj'] = np.nan

missing_indices_zona = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
df.loc[missing_indices_zona, 'zona'] = np.nan

missing_indices_balcon = np.random.choice(df.index, size=int(0.12 * len(df)), replace=False)
df.loc[missing_indices_balcon, 'balcon'] = np.nan

print("\n❓ VALORI LIPSĂ INTRODUSE:")
print(df.isnull().sum())
print(f"\nProcent total missing: {df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")

# ========================================
# PARTEA 3: EXPLORAREA DATELOR
# ========================================

print("\n🔍 STATISTICI DESCRIPTIVE:")
print(df.describe())

# Verifică distribuția prețurilor
print("\n💰 DISTRIBUȚIA PREȚURILOR:")
print(f"Min: {df['pret'].min():,.0f} €")
print(f"Max: {df['pret'].max():,.0f} €")
print(f"Medie: {df['pret'].mean():,.0f} €")
print(f"Mediană: {df['pret'].median():,.0f} €")

# Verifică distribuția pe zone
print("\n🏘️ PREȚURI MEDII PE ZONE:")
print(df.groupby('zona')['pret'].mean().sort_values(ascending=False).round(0))

# ========================================
# PARTEA 4: VIZUALIZĂRI EXPLORATORII
# ========================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribuția prețurilor
axes[0, 0].hist(df['pret'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['pret'].mean(), color='red', linestyle='--', linewidth=2, label=f'Medie: {df["pret"].mean():,.0f}€')
axes[0, 0].axvline(df['pret'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediană: {df["pret"].median():,.0f}€')
axes[0, 0].set_xlabel('Preț (€)', fontsize=12)
axes[0, 0].set_ylabel('Frecvență', fontsize=12)
axes[0, 0].set_title('Distribuția Prețurilor Apartamentelor', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Preț vs Suprafață
axes[0, 1].scatter(df['suprafata'], df['pret'], alpha=0.5, color='coral')
axes[0, 1].set_xlabel('Suprafață (mp)', fontsize=12)
axes[0, 1].set_ylabel('Preț (€)', fontsize=12)
axes[0, 1].set_title('Preț în Funcție de Suprafață', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Prețuri medii pe zone
zone_avg = df.groupby('zona')['pret'].mean().sort_values()
axes[1, 0].barh(zone_avg.index, zone_avg.values, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Preț Mediu (€)', fontsize=12)
axes[1, 0].set_title('Prețuri Medii pe Zone', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Heatmap missing values
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_data / len(df) * 100).round(1)
axes[1, 1].barh(missing_data.index, missing_percent.values, color='indianred', edgecolor='black')
axes[1, 1].set_xlabel('Procent Missing (%)', fontsize=12)
axes[1, 1].set_title('Valori Lipsă pe Coloane', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('real_estate_exploration.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafic salvat: real_estate_exploration.png")

# ========================================
# PARTEA 5: SALVAREA DATASET-ULUI
# ========================================

df.to_csv('apartamente_bucuresti.csv', index=False)
print("\n✅ Dataset salvat: apartamente_bucuresti.csv")

print("""
\n🎯 CE AM ÎNVĂȚAT:

1. CREAREA DATASET-ULUI REALIST:
   - Zone cu prețuri diferite
   - Variație naturală în date
   - Multiple features (numerical + categorical)

2. MISSING VALUES (cum în realitate!):
   - 15% missing în suprafață
   - 10% missing în etaj
   - 8% missing în zonă
   - 12% missing în balcon

3. EXPLORAREA DATELOR:
   - Statistici descriptive
   - Distribuții
   - Corelații vizuale
   - Identificarea pattern-urilor

4. VIZUALIZĂRI:
   - Histograme pentru distribuții
   - Scatter plots pentru relații
   - Bar charts pentru comparații
   - Missing value analysis

🚀 URMĂTORUL PAS: ColumnTransformer pentru preprocessing!
""")