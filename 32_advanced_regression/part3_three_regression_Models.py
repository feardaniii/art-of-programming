import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Presupunem că avem X_train_processed, X_test_processed, y_train, y_test
# din partea anterioară (după ColumnTransformer)

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Încarcă dataset-ul
df = pd.read_csv('apartamente_bucuresti.csv')

# ========================================
# PARTEA 1: SEPARAREA FEATURES & TARGET
# ========================================

# Features (X) și Target (y)
X = df.drop('pret', axis=1)
y = df['pret']

print("📊 STRUCTURA DATELOR:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nColoane în X:\n{X.columns.tolist()}")

# ========================================
# PARTEA 2: IDENTIFICAREA TIPURILOR
# ========================================

# Identifică automat coloanele numerice și categorice
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Future-proof: include both 'object' and 'string' dtypes
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"\n🔢 NUMERICAL FEATURES ({len(numerical_features)}):")
print(numerical_features)

print(f"\n🏷️ CATEGORICAL FEATURES ({len(categorical_features)}):")
print(categorical_features)

# ========================================
# PARTEA 3: CREAREA TRANSFORMERS
# ========================================

# Transformer pentru features numerice
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),    # Completează cu media
    ('scaler', StandardScaler())                     # Normalizează
])

# Transformer pentru features categorice
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Completează cu cel mai frecvent
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encodare
])

print("\n✅ TRANSFORMERS CREAȚI:")
print("  1. Numerical: SimpleImputer(mean) → StandardScaler")
print("  2. Categorical: SimpleImputer(most_frequent) → OneHotEncoder")

# ========================================
# PARTEA 4: COLUMN TRANSFORMER
# ========================================

# Combinăm transformers-ii folosind ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Drop orice altă coloană nespecificată
)

print("\n🔧 COLUMN TRANSFORMER CREAT!")
print(f"  - Va procesa {len(numerical_features)} numerical features")
print(f"  - Va procesa {len(categorical_features)} categorical features")

# ========================================
# PARTEA 5: TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📦 TRAIN-TEST SPLIT:")
print(f"Training set: {X_train.shape[0]} apartamente ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} apartamente ({X_test.shape[0]/len(X)*100:.1f}%)")

# ========================================
# PARTEA 6: FIT & TRANSFORM
# ========================================

# Fit preprocessor pe training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n🎯 PREPROCESSING COMPLET:")
print(f"  Înainte: {X_train.shape} → După: {X_train_processed.shape}")
print(f"  Features create: {X_train_processed.shape[1]}")

# ========================================
# PARTEA 7: ÎNȚELEGEREA OUTPUT-ULUI
# ========================================

# Obține numele features după OneHotEncoding
feature_names = []

# Numerical features (same names)
feature_names.extend(numerical_features)

# Categorical features (get encoded names from OneHotEncoder)
cat_encoder = preprocessor.named_transformers_['cat']['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names.extend(cat_feature_names)































# ========================================
# MODEL 1: LINEAR REGRESSION (BASELINE)
# ========================================

print("=" * 60)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 60)

# Creează și antrenează modelul
lr_model = LinearRegression()
lr_model.fit(X_train_processed, y_train)

# Predicții
y_train_pred_lr = lr_model.predict(X_train_processed)
y_test_pred_lr = lr_model.predict(X_test_processed)

# Evaluare
r2_train_lr = r2_score(y_train, y_train_pred_lr)
r2_test_lr = r2_score(y_test, y_test_pred_lr)
rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)

print(f"\n📊 REZULTATE LINEAR REGRESSION:")
print(f"  R² Train: {r2_train_lr:.4f}")
print(f"  R² Test:  {r2_test_lr:.4f}")
print(f"  RMSE:     {rmse_test_lr:,.0f} €")
print(f"  MAE:      {mae_test_lr:,.0f} €")

# ========================================
# MODEL 2: RIDGE REGRESSION + GRID SEARCH
# ========================================

print("\n" + "=" * 60)
print("MODEL 2: RIDGE REGRESSION (cu GridSearchCV)")
print("=" * 60)

# Definește parametrii pentru Grid Search
param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000]
}

# Creează Ridge cu GridSearchCV
ridge = Ridge()
grid_search = GridSearchCV(
    ridge,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Antrenează cu Grid Search
grid_search.fit(X_train_processed, y_train)

# Best model
best_ridge = grid_search.best_estimator_
best_alpha = grid_search.best_params_['alpha']

print(f"\n🔍 GRID SEARCH RESULTS:")
print(f"  Best alpha: {best_alpha}")
print(f"  Best CV score: {-grid_search.best_score_:,.0f} (MSE)")

# Predicții cu best model
y_train_pred_ridge = best_ridge.predict(X_train_processed)
y_test_pred_ridge = best_ridge.predict(X_test_processed)

# Evaluare
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
rmse_test_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
mae_test_ridge = mean_absolute_error(y_test, y_test_pred_ridge)

print(f"\n📊 REZULTATE RIDGE REGRESSION:")
print(f"  R² Train: {r2_train_ridge:.4f}")
print(f"  R² Test:  {r2_test_ridge:.4f}")
print(f"  RMSE:     {rmse_test_ridge:,.0f} €")
print(f"  MAE:      {mae_test_ridge:,.0f} €")

# ========================================
# MODEL 3: RANDOM FOREST REGRESSOR
# ========================================

print("\n" + "=" * 60)
print("MODEL 3: RANDOM FOREST REGRESSOR")
print("=" * 60)

# Creează și antrenează Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,    # 100 de arbori
    max_depth=15,        # Limită adâncime pentru a preveni overfitting
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

# Predicții
y_train_pred_rf = rf_model.predict(X_train_processed)
y_test_pred_rf = rf_model.predict(X_test_processed)

# Evaluare
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

print(f"\n📊 REZULTATE RANDOM FOREST:")
print(f"  R² Train: {r2_train_rf:.4f}")
print(f"  R² Test:  {r2_test_rf:.4f}")
print(f"  RMSE:     {rmse_test_rf:,.0f} €")
print(f"  MAE:      {mae_test_rf:,.0f} €")

# ========================================
# COMPARAȚIE FINALĂ
# ========================================

print("\n" + "=" * 60)
print("📊 COMPARAȚIE FINALĂ - TOATE MODELELE")
print("=" * 60)

# Creează tabel comparativ
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
    'R² Train': [r2_train_lr, r2_train_ridge, r2_train_rf],
    'R² Test': [r2_test_lr, r2_test_ridge, r2_test_rf],
    'RMSE (€)': [rmse_test_lr, rmse_test_ridge, rmse_test_rf],
    'MAE (€)': [mae_test_lr, mae_test_ridge, mae_test_rf]
})

print("\n" + comparison.to_string(index=False))

# Identifică best model
best_model_idx = comparison['R² Test'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Test: {comparison.loc[best_model_idx, 'R² Test']:.4f}")
print(f"   RMSE: {comparison.loc[best_model_idx, 'RMSE (€)']:,.0f} €")

# ========================================
# VIZUALIZARE COMPARAȚIE
# ========================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. Comparație R² scores
models = comparison['Model']
r2_train_scores = comparison['R² Train']
r2_test_scores = comparison['R² Test']

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, r2_train_scores, width, label='R² Train', color='lightblue', edgecolor='black')
axes[0].bar(x + width/2, r2_test_scores, width, label='R² Test', color='coral', edgecolor='black')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Comparație R² Scores', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, 1)

# 2. Comparație RMSE
rmse_scores = comparison['RMSE (€)']
colors = ['lightblue', 'lightgreen', 'coral']
axes[1].bar(models, rmse_scores, color=colors, edgecolor='black')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('RMSE (€)', fontsize=12)
axes[1].set_title('Comparație RMSE (Lower is Better)', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(models, rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)

# Adaugă valori pe bare
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + max(rmse_scores)*0.02, f'{v:,.0f}€', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafic salvat: models_comparison.png")

# ========================================
# INTERPRETARE & CONCLUZII
# ========================================

print("""
\n💡 INTERPRETARE REZULTATE:

1. R² SCORE (0 to 1, higher is better):
   - Măsoară cât de bine modelul explică variația în preț
   - R² = 0.85 înseamnă: modelul explică 85% din variația prețurilor
   - Compară Train vs Test pentru a detecta overfitting

2. RMSE (Root Mean Squared Error):
   - Eroarea medie în unități de preț (€)
   - RMSE = 15.000€ înseamnă: predicțiile greșesc în medie cu ±15.000€
   - Sensibil la outlieri (errori mari sunt penalizate mai mult)

3. MAE (Mean Absolute Error):
   - Eroarea medie absolută în preț
   - MAE = 12.000€ înseamnă: deviația absolută medie este 12.000€
   - Mai robust la outlieri decât RMSE

🎯 ALEGEREA MODELULUI:
   - Dacă Train R² >> Test R² → Overfitting
   - Dacă ambele R² sunt similare → Good generalization
   - Alege modelul cu cel mai bun R² Test și RMSE mic

🚀 URMĂTORUL PAS: Vizualizarea predicțiilor!
""")