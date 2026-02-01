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

























from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

# ========================================
# MODEL 1: SUPPORT VECTOR MACHINE (SVM)
# ========================================

print("=" * 60)
print("🔵 MODEL 1: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

# Creează și antrenează modelul SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predicții
svm_predictions = svm_model.predict(X_test_scaled)

# Evaluare
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"\n✅ Acuratețe SVM: {svm_accuracy:.4f} ({svm_accuracy * 100:.2f}%)")

print("\n📋 Classification Report SVM:")
print(classification_report(y_test, svm_predictions,
                          target_names=['Malignă', 'Benignă']))

# Confusion Matrix
svm_cm = confusion_matrix(y_test, svm_predictions)
print("\n🎯 Confusion Matrix SVM:")
print(svm_cm)

# ========================================
# MODEL 2: RANDOM FOREST
# ========================================

print("\n" + "=" * 60)
print("🟢 MODEL 2: RANDOM FOREST")
print("=" * 60)

# Creează și antrenează Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Predicții
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluare
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"\n✅ Acuratețe Random Forest: {rf_accuracy:.4f} ({rf_accuracy * 100:.2f}%)")

print("\n📋 Classification Report Random Forest:")
print(classification_report(y_test, rf_predictions,
                          target_names=['Malignă', 'Benignă']))

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
print("\n🎯 Confusion Matrix Random Forest:")
print(rf_cm)

# Feature Importance (bonus pentru Random Forest)
print("\n⭐ TOP 10 CELE MAI IMPORTANTE FEATURES:")
feature_importance = pd.DataFrame({
    'feature': cancer_data.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# ========================================
# MODEL 3: K-NEAREST NEIGHBORS (KNN)
# ========================================

print("\n" + "=" * 60)
print("🟣 MODEL 3: K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# Creează și antrenează KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

# Predicții
knn_predictions = knn_model.predict(X_test_scaled)

# Evaluare
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"\n✅ Acuratețe KNN: {knn_accuracy:.4f} ({knn_accuracy * 100:.2f}%)")

print("\n📋 Classification Report KNN:")
print(classification_report(y_test, knn_predictions,
                          target_names=['Malignă', 'Benignă']))

# Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_predictions)
print("\n🎯 Confusion Matrix KNN:")
print(knn_cm)

# ========================================
# COMPARAȚIE FINALĂ
# ========================================

print("\n" + "=" * 60)
print("📊 COMPARAȚIE FINALĂ")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'KNN'],
    'Acuratețe': [svm_accuracy, rf_accuracy, knn_accuracy]
}).sort_values('Acuratețe', ascending=False)

print(comparison)
print(f"\n🏆 CÂȘTIGĂTORUL: {comparison.iloc[0]['Model']} cu {comparison.iloc[0]['Acuratețe']:.4f}")

# ========================================
# VIZUALIZARE: CONFUSION MATRICES
# ========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# SVM
disp1 = ConfusionMatrixDisplay(confusion_matrix=svm_cm,
                               display_labels=['Malignă', 'Benignă'])
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title(f'SVM\nAcuratețe: {svm_accuracy:.4f}', fontsize=14, fontweight='bold')

# Random Forest
disp2 = ConfusionMatrixDisplay(confusion_matrix=rf_cm,
                               display_labels=['Malignă', 'Benignă'])
disp2.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title(f'Random Forest\nAcuratețe: {rf_accuracy:.4f}', fontsize=14, fontweight='bold')

# KNN
disp3 = ConfusionMatrixDisplay(confusion_matrix=knn_cm,
                               display_labels=['Malignă', 'Benignă'])
disp3.plot(ax=axes[2], cmap='Purples', values_format='d')
axes[2].set_title(f'KNN\nAcuratețe: {knn_accuracy:.4f}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('classifiers_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafic salvat: classifiers_confusion_matrices.png")

# ========================================
# ÎNȚELEGEREA METRICILOR
# ========================================

print("""
\n📚 ÎNȚELEGEREA METRICILOR DE CLASIFICARE:

1. ACCURACY (Acuratețe):
   - (TP + TN) / Total
   - Câte predicții corecte din total
   - ⚠️ Poate fi înșelătoare cu clase imbalanced!

2. PRECISION (Precizie):
   - TP / (TP + FP)
   - Din ce am zis că e pozitiv, câte chiar sunt?
   - Important când False Positive e costisitor

3. RECALL (Sensitivitate):
   - TP / (TP + FN)
   - Din toate pozitivele reale, câte am găsit?
   - Important când False Negative e costisitor

4. F1-SCORE:
   - 2 * (Precision * Recall) / (Precision + Recall)
   - Medie armonică între Precision și Recall
   - Bună pentru evaluare overall

🏥 PENTRU DETECTAREA CANCERULUI:
   - RECALL e crucial! (nu vrem să ratăm cancer = False Negative)
   - False Negative (cancer ratat) >> False Positive (alarmă falsă)
""")