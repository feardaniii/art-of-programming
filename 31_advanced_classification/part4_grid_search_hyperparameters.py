from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Presupunem că avem X_train, X_test, y_train, y_test

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