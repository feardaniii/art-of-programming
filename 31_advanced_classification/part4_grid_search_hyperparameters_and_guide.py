from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Presupunem că avem X_train, X_test, y_train, y_test

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




# ========================================
# GRIDSEARCHCV PENTRU SVM
# ========================================

print("=" * 60)
print("🔵 GRIDSEARCHCV PENTRU SVM")
print("=" * 60)

# Creează pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Definește grila de parametri
# NOTĂ: Pentru pipeline, folosim 'classifier__parametru'
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 0.001, 0.01, 0.1]
}

print(f"\n📊 Număr total de combinații: {len(param_grid_svm['classifier__C']) * len(param_grid_svm['classifier__kernel']) * len(param_grid_svm['classifier__gamma'])}")
print("\n🔍 Parametri de testat:")
for param, values in param_grid_svm.items():
    print(f"   {param}: {values}")

# Creează GridSearchCV
grid_search_svm = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid_svm,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # folosește toate核心ele CPU
    verbose=2  # afișează progres
)

print("\n⏳ Antrenare în curs... (poate dura câteva minute)")
grid_search_svm.fit(X_train, y_train)

# Rezultate
print("\n" + "=" * 60)
print("✅ ANTRENARE COMPLETĂ!")
print("=" * 60)

print(f"\n🏆 CELE MAI BUNE PARAMETRI:")
for param, value in grid_search_svm.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Cel mai bun score (CV): {grid_search_svm.best_score_:.4f}")

# Testează pe test set
test_score = grid_search_svm.score(X_test, y_test)
print(f"📊 Score pe test set: {test_score:.4f}")

# ========================================
# GRIDSEARCHCV PENTRU RANDOM FOREST
# ========================================

print("\n" + "=" * 60)
print("🟢 GRIDSEARCHCV PENTRU RANDOM FOREST")
print("=" * 60)

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

print(f"\n📊 Număr total de combinații: {len(param_grid_rf['classifier__n_estimators']) * len(param_grid_rf['classifier__max_depth']) * len(param_grid_rf['classifier__min_samples_split']) * len(param_grid_rf['classifier__min_samples_leaf'])}")

grid_search_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\n⏳ Antrenare Random Forest...")
grid_search_rf.fit(X_train, y_train)

print(f"\n🏆 CELE MAI BUNE PARAMETRI:")
for param, value in grid_search_rf.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Cel mai bun score (CV): {grid_search_rf.best_score_:.4f}")
print(f"📊 Score pe test set: {grid_search_rf.score(X_test, y_test):.4f}")

# ========================================
# GRIDSEARCHCV PENTRU KNN
# ========================================

print("\n" + "=" * 60)
print("🟣 GRIDSEARCHCV PENTRU KNN")
print("=" * 60)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}

print(f"\n📊 Număr total de combinații: {len(param_grid_knn['classifier__n_neighbors']) * len(param_grid_knn['classifier__weights']) * len(param_grid_knn['classifier__metric'])}")

grid_search_knn = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid_knn,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("\n⏳ Antrenare KNN...")
grid_search_knn.fit(X_train, y_train)

print(f"\n🏆 CELE MAI BUNE PARAMETRI:")
for param, value in grid_search_knn.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Cel mai bun score (CV): {grid_search_knn.best_score_:.4f}")
print(f"📊 Score pe test set: {grid_search_knn.score(X_test, y_test):.4f}")

# ========================================
# COMPARAȚIE FINALĂ
# ========================================

print("\n" + "=" * 60)
print("🏆 COMPARAȚIE FINALĂ - DUPĂ TUNING")
print("=" * 60)

comparison_results = pd.DataFrame({
    'Model': ['SVM (tuned)', 'Random Forest (tuned)', 'KNN (tuned)'],
    'CV Score': [
        grid_search_svm.best_score_,
        grid_search_rf.best_score_,
        grid_search_knn.best_score_
    ],
    'Test Score': [
        grid_search_svm.score(X_test, y_test),
        grid_search_rf.score(X_test, y_test),
        grid_search_knn.score(X_test, y_test)
    ]
}).sort_values('Test Score', ascending=False)

print(comparison_results)
print(f"\n🥇 CÂȘTIGĂTORUL: {comparison_results.iloc[0]['Model']}")
print(f"   CV Score: {comparison_results.iloc[0]['CV Score']:.4f}")
print(f"   Test Score: {comparison_results.iloc[0]['Test Score']:.4f}")

# ========================================
# SALVARE MODEL FINAL
# ========================================

print("\n" + "=" * 60)
print("💾 SALVARE MODEL FINAL")
print("=" * 60)

import joblib

# Salvează cel mai bun model (presupunem că e SVM)
best_model = grid_search_svm.best_estimator_
joblib.dump(best_model, 'breast_cancer_best_model.pkl')

print("\n✅ Model salvat: breast_cancer_best_model.pkl")
print(f"\nℹ️ Modelul salvat include:")
print("   1. StandardScaler cu parametrii antrenați")
print("   2. SVM cu hyperparametri optimizați")
print("   3. Gata de deployment în producție!")

# ========================================
# ANALIZĂ DETALIATĂ REZULTATE GRIDSEARCH
# ========================================

print("\n" + "=" * 60)
print("📊 ANALIZA DETALIATĂ - TOP 10 COMBINAȚII SVM")
print("=" * 60)

cv_results = pd.DataFrame(grid_search_svm.cv_results_)
top_10 = cv_results.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]

print(top_10.to_string(index=False))

print("""
\n💡 INTERPRETARE:
   - mean_test_score: Media acurateței pe 5 folduri
   - std_test_score: Deviația standard (variabilitate)
   - rank_test_score: Ranking (1 = cel mai bun)

   🎯 Căutăm: score mare + std mic = model stabil și performant!
""")





# =============================================================================
# BIG PICTURE (one sentence)
# Trained several models, tried many settings for each, and GridSearchCV
# automatically picked the settings that worked best based on cross-validation accuracy.
# That's it. Everything else is details.
# =============================================================================

# =============================================================================
# SVM - INTUITION
# =============================================================================
# SVM tries to draw a line/curve that best separates: malignant vs benign tumors
#
# kernel = rbf → "We allow curved decision boundaries"
#   - linear → straight line
#   - rbf → flexible, curved boundary
#   - 📌 Why it won: the data is not linearly separable
#
# C = 10 → "How strict are we about misclassifying points?"
#   - small C → relaxed, simpler boundary
#   - big C → stricter, fits data more closely
#   - 📌 C = 10 = good balance between underfitting and overfitting

# =============================================================================
# RANDOM FOREST PARAMETERS (demystified)
# =============================================================================
# classifier__n_estimators: 50 | classifier__max_depth: 10
# classifier__min_samples_split: 5 | classifier__min_samples_leaf: 2
#
# What Random Forest is: A committee of decision trees voting together 🌳🌳🌳
#
# n_estimators = 50 → "How many trees are in the forest"
#   - more trees = more stability, diminishing returns after a point
#   - 📌 50 is efficient and stable
#
# max_depth = 10 → "How deep can each tree grow?"
#   - shallow → underfit | too deep → memorizes noise
#   - 📌 Depth 10 = controlled complexity
#
# min_samples_split = 5 → "A node needs at least 5 samples to split"
#   - Prevents silly splits on tiny noise
#
# min_samples_leaf = 2 → "Each leaf must have at least 2 samples"
#   - Stops extreme overfitting

# =============================================================================
# SUMMARY BY LEVEL
# =============================================================================
# Level 1: "We tried many models and settings. The computer tested them fairly and chose the best one."
# Level 2: "GridSearch tested different hyperparameters using cross-validation to avoid overfitting."
# Level 3: "SVM with RBF kernel, C=10 and gamma=0.01 gave the best bias-variance tradeoff."