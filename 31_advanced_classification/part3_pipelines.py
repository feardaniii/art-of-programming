from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Presupunem că avem X_train, X_test, y_train, y_test (NON-scaled!)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



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
# PIPELINE 1: SVM CU STANDARD SCALER
# ========================================

print("=" * 60)
print("🔵 PIPELINE 1: StandardScaler → SVM")
print("=" * 60)

# Creează pipeline-ul
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True))
])

# Antrenează pipeline-ul (scalare + antrenare într-un singur pas!)
svm_pipeline.fit(X_train, y_train)

# Predicție (scalare + predicție automat!)
svm_pred = svm_pipeline.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"\n✅ Acuratețe SVM Pipeline: {svm_accuracy:.4f}")
print(f"\n📋 Pași în pipeline: {[name for name, _ in svm_pipeline.steps]}")

# ========================================
# PIPELINE 2: RANDOM FOREST
# ========================================

print("\n" + "=" * 60)
print("🟢 PIPELINE 2: StandardScaler → Random Forest")
print("=" * 60)

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10,
                                         random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\n✅ Acuratețe Random Forest Pipeline: {rf_accuracy:.4f}")

# ========================================
# PIPELINE 3: KNN
# ========================================

print("\n" + "=" * 60)
print("🟣 PIPELINE 3: StandardScaler → KNN")
print("=" * 60)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)
knn_pred = knn_pipeline.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"\n✅ Acuratețe KNN Pipeline: {knn_accuracy:.4f}")

# ========================================
# BENEFICIILE PIPELINE-URILOR
# ========================================

print("\n" + "=" * 60)
print("🎯 BENEFICIILE PIPELINE-URILOR")
print("=" * 60)

print("""
1. 🔒 ZERO DATA LEAKAGE:
   - Scaler învață DOAR din train
   - Test nu influențează niciodată transformările

2. 📝 COD MAI CURAT:
   - fit() și predict() într-un singur apel
   - Nu mai ai nevoie de variabile separate pentru scaled data

3. 🔄 REPRODUCIBILITATE:
   - Întregul workflow într-un singur obiect
   - Poți salva pipeline-ul și îl folosești identic mai târziu

4. 🚀 DEPLOYMENT MAI UȘOR:
   - Un singur obiect de salvat: pickle.dump(pipeline, file)
   - În producție: pickle.load() → pipeline.predict()

5. 🛠️ COMPATIBIL CU GRIDSERCHCV:
   - Poți optimiza hyperparametri pentru TOȚI pașii
   - Cross-validation corectă automat
""")

# ========================================
# SALVARE ȘI ÎNCĂRCARE PIPELINE
# ========================================

print("\n" + "=" * 60)
print("💾 SALVARE ȘI ÎNCĂRCARE PIPELINE")
print("=" * 60)

import joblib

# Salvează cel mai bun pipeline
best_pipeline = svm_pipeline
joblib.dump(best_pipeline, 'breast_cancer_classifier_pipeline.pkl')
print("\n✅ Pipeline salvat: breast_cancer_classifier_pipeline.pkl")

# Încarcă pipeline-ul
loaded_pipeline = joblib.load('breast_cancer_classifier_pipeline.pkl')
print("✅ Pipeline încărcat cu succes!")

# Testează că funcționează identic
loaded_pred = loaded_pipeline.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_pred)
print(f"\n🔍 Acuratețe pipeline încărcat: {loaded_accuracy:.4f}")
print(f"✅ Match cu original: {loaded_accuracy == svm_accuracy}")

# ========================================
# PREDICȚIE PE DATE NOI (SIMULARE)
# ========================================

print("\n" + "=" * 60)
print("🏥 PREDICȚIE PE DATE NOI - SIMULARE SPITAL")
print("=" * 60)

# Simulează un pacient nou (30 features)
new_patient = np.array([cancer_data.data[0]])  # folosim prima sample ca exemplu

print("\n📋 Date pacient nou (primele 5 features):")
print(new_patient[0][:5])

# Predicție cu pipeline (scalare automată!)
prediction = loaded_pipeline.predict(new_patient)
prediction_proba = loaded_pipeline.predict_proba(new_patient)

diagnosis = "Malignă 🔴" if prediction[0] == 0 else "Benignă 🟢"
confidence = max(prediction_proba[0]) * 100

print(f"\n🏥 DIAGNOSTIC: {diagnosis}")
print(f"📊 Confidence: {confidence:.2f}%")
print(f"\n📈 Probabilități:")
print(f"   - Malignă: {prediction_proba[0][0] * 100:.2f}%")
print(f"   - Benignă: {prediction_proba[0][1] * 100:.2f}%")

if prediction[0] == 0:
    print("\n⚠️ RECOMANDARE: Consultare oncolog urgentă + biopsie suplimentară")
else:
    print("\n✅ RECOMANDARE: Control de rutină peste 6 luni")

# ========================================
# CROSS-VALIDATION CU PIPELINE
# ========================================

print("\n" + "=" * 60)
print("🔄 CROSS-VALIDATION CU PIPELINE (5-Fold)")
print("=" * 60)

# Cross-validation pe SVM pipeline
cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=5, scoring='accuracy')

print(f"\n📊 Scoruri pentru fiecare fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")

print(f"\n✅ Media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"\n💡 Interpretare:")
print(f"   - Modelul are ~{cv_scores.mean() * 100:.2f}% acuratețe pe date nevăzute")
print(f"   - Variație mică ({cv_scores.std():.4f}) = model stabil!")

print("""
\n🎓 DE CE CROSS-VALIDATION?
   - Un singur test set poate fi norocos/nenorocos
   - CV testează pe 5 părți diferite → estimare mai realistă
   - Detectează overfitting: dacă train score >> CV score
""")

