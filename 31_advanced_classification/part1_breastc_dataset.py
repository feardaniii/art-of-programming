import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========================================
# PARTEA 1: ÎNCĂRCAREA DATASET-ULUI
# ========================================

# Încarcă dataset-ul breast cancer de la sklearn
cancer_data = load_breast_cancer()

print("📊 INFORMAȚII DESPRE DATASET:")
print(f"Număr de sample: {cancer_data.data.shape[0]}")
print(f"Număr de features: {cancer_data.data.shape[1]}")
print(f"Clase: {cancer_data.target_names}")
print()

# Creează DataFrame pentru o vizualizare mai bună
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

print("🔍 PRIMELE 5 RÂNDURI:")
print(df.head())
print()

# ========================================
# PARTEA 2: EXPLORAREA DATELOR
# ========================================

print("📈 STATISTICI DESCRIPTIVE:")
print(df.describe())
print()

# Verifică distribuția claselor
print("⚖️ DISTRIBUȚIA CLASELOR:")
print(f"Malignă (0): {sum(cancer_data.target == 0)} paciente")
print(f"Benignă (1): {sum(cancer_data.target == 1)} paciente")
print()

# Vizualizare: Distribuția primelor 4 features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features_to_plot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 2, idx % 2]

    # Histogramă pentru fiecare clasă
    df[df['target'] == 0][feature].hist(ax=ax, alpha=0.5, label='Malignă',
                                         color='red', bins=30)
    df[df['target'] == 1][feature].hist(ax=ax, alpha=0.5, label='Benignă',
                                         color='green', bins=30)

    ax.set_xlabel(feature)
    ax.set_ylabel('Frecvență')
    ax.set_title(f'Distribuția: {feature}')
    ax.legend()

plt.tight_layout()
plt.savefig('breast_cancer_features_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Grafic salvat: breast_cancer_features_distribution.png")
print()

# ========================================
# PARTEA 3: PREGĂTIREA DATELOR
# ========================================

# Separare features (X) și target (y)
X = cancer_data.data
y = cancer_data.target

# Split în train și test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📦 SPLIT TRAIN-TEST:")
print(f"Training set: {X_train.shape[0]} sample")
print(f"Test set: {X_test.shape[0]} sample")
print()

# ========================================
# PARTEA 4: NORMALIZARE (CRUCIAL!)
# ========================================

# IMPORTANT: fit_transform() pe train, doar transform() pe test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🔧 NORMALIZARE COMPLETĂ:")
print(f"Înainte - Mean prima feature train: {X_train[:, 0].mean():.2f}")
print(f"După - Mean prima feature train: {X_train_scaled[:, 0].mean():.2f}")
print(f"După - Std prima feature train: {X_train_scaled[:, 0].std():.2f}")
print()

# Vizualizare: Efect normalizare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Înainte de normalizare
ax1.boxplot([X_train[:, i] for i in range(5)], tick_labels=cancer_data.feature_names[:5])
ax1.set_title('Înainte de Normalizare', fontsize=14, fontweight='bold')
ax1.set_ylabel('Valoare')
ax1.tick_params(axis='x', rotation=45)

# După normalizare
ax2.boxplot([X_train_scaled[:, i] for i in range(5)], tick_labels=cancer_data.feature_names[:5])
ax2.set_title('După Normalizare (StandardScaler)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Valoare Normalizată')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('normalization_effect.png', dpi=300, bbox_inches='tight')
print("✅ Grafic salvat: normalization_effect.png")

# ========================================
# DE CE NORMALIZARE?
# ========================================
print("""
🎯 DE CE ESTE NORMALIZAREA CRUCIALĂ?

1. SCARA DIFERITĂ A FEATURES:
   - 'mean radius': 6-28 (diferență de ~22)
   - 'mean area': 143-2501 (diferență de ~2358)

   Fără normalizare, 'mean area' ar domina modelul!

2. ALGORITMI SENSIBILI:
   - SVM: bazat pe distanțe → trebuie scale similar
   - KNN: distanța Euclideană → trebuie scale similar
   - Neural Networks: converge mai repede cu date normalizate

3. INTERPRETARE:
   - După normalizare: toate features au contribuție echitabilă
   - Coeficienții modelului sunt comparabili

🔒 REGULA DE AUR: fit_transform() DOAR pe TRAIN!
   - Test set-ul NU TREBUIE să influențeze media/std
   - Altfel → DATA LEAKAGE → rezultate false
""")