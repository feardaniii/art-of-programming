import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Presupunem că avem predicțiile din partea anterioară
# y_test, y_test_pred_lr, y_test_pred_ridge, y_test_pred_rf

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
# PARTEA 8: TRAINING MODELS
# ========================================

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_processed, y_train)
y_test_pred_lr = lr_model.predict(X_test_processed)

# 2. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_processed, y_train)
y_test_pred_ridge = ridge_model.predict(X_test_processed)

# 3. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_processed, y_train)
y_test_pred_rf = rf_model.predict(X_test_processed)

print("\n🤖 MODELS TRAINED:")
print("  ✓ Linear Regression")
print("  ✓ Ridge Regression (α=1.0)")
print("  ✓ Random Forest (100 trees, depth=10)")



# Part 1-7: Preprocessing ✓ (you have this)
# Part 8: Train models ✗ (MISSING - add above)
# Part 9: Visualizations ✓ (you have this, but it crashes without Part 8)










# ========================================
# FIGURA 1: PREDICTION VS ACTUAL (3 MODELE)
# ========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_predictions = [
    ('Linear Regression', y_test_pred_lr, 'blue'),
    ('Ridge Regression', y_test_pred_ridge, 'purple'),
    ('Random Forest', y_test_pred_rf, 'green')
]

for idx, (model_name, predictions, color) in enumerate(models_predictions):
    ax = axes[idx]

    # Scatter plot: actual vs predicted
    ax.scatter(y_test, predictions, alpha=0.5, color=color, edgecolors='black', s=50)

    # Linia perfectă y=x
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Labels și titlu
    ax.set_xlabel('Preț Real (€)', fontsize=12)
    ax.set_ylabel('Preț Prezis (€)', fontsize=12)
    ax.set_title(f'{model_name}\nPrediction vs Actual', fontsize=13, fontweight='bold')

    # Calculează R² pentru display
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("✅ Grafic salvat: prediction_vs_actual.png")

# ========================================
# FIGURA 2: RESIDUALS ANALYSIS
# ========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (model_name, predictions, color) in enumerate(models_predictions):
    # Calculează residuals
    residuals = y_test - predictions

    # Plot 1: Residuals vs Predicted (detectează heterocedasticitate)
    ax1 = axes[0, idx]
    ax1.scatter(predictions, residuals, alpha=0.5, color=color, edgecolors='black', s=50)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Preț Prezis (€)', fontsize=11)
    ax1.set_ylabel('Residuals (€)', fontsize=11)
    ax1.set_title(f'{model_name}\nResiduals vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Plot 2: Distribution of Residuals (verifică normalitatea)
    ax2 = axes[1, idx]
    ax2.hist(residuals, bins=30, color=color, alpha=0.7, edgecolor='black')

    # Suprapune curba normală
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max()-residuals.min())/30,
             'r-', linewidth=2, label='Normal Distribution')

    # Linie verticală la 0
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Residuals (€)', fontsize=11)
    ax2.set_ylabel('Frecvență', fontsize=11)
    ax2.set_title(f'{model_name}\nDistribuția Residuals', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Adaugă statistici
    ax2.text(0.05, 0.95, f'Mean: {mu:,.0f}€\nStd: {sigma:,.0f}€',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Grafic salvat: residuals_analysis.png")

# ========================================
# FIGURA 3: FEATURE IMPORTANCE (RANDOM FOREST)
# ========================================

# Doar Random Forest are feature_importances_
importances = rf_model.feature_importances_

# Creează DataFrame pentru sortare
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Selectează top 15
top_features = feature_importance_df.head(15)

# Plot
plt.figure(figsize=(12, 8))
colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
plt.barh(top_features['feature'], top_features['importance'], color=colors_gradient, edgecolor='black')
plt.xlabel('Importance', fontsize=13)
plt.ylabel('Feature', fontsize=13)
plt.title('Top 15 Feature Importance - Random Forest Regressor', fontsize=15, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Adaugă valori pe bare
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['importance'] + 0.005, i, f"{row['importance']:.3f}",
             va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Grafic salvat: feature_importance.png")

# ========================================
# RAPORT FINAL - DIAGNOSTIC
# ========================================

print("""
\n📊 INTERPRETAREA VIZUALIZĂRILOR:

1. PREDICTION VS ACTUAL:
   ✓ Puncte aproape de linia roșie = predicții bune
   ✗ Puncte departe de linie = erori mari
   → Verifică: sunt erorile aleatorii sau sistematice?

2. RESIDUALS VS PREDICTED:
   ✓ Residuals random scatter around 0 = GOOD (homoscedasticity)
   ✗ Pattern în residuals (funnel shape) = BAD (heteroscedasticity)
   → Dacă vezi funnel: modelul greșește mai mult la prețuri mari/mici

3. DISTRIBUȚIA RESIDUALS:
   ✓ Distribuție normală centrată pe 0 = IDEAL
   ✗ Skewed la stânga/dreapta = Bias sistematic
   ✗ Heavy tails = Multe outlieri
   → Normal distribution = model capturează bine pattern-urile

4. FEATURE IMPORTANCE:
   → Identifică ce features sunt cele mai importante
   → Poate elimina features irelevante
   → Poate sugera noi features de creat

🎯 ACȚIUNI BAZATE PE GRAFICE:

Dacă vezi:
  • Funnel shape în residuals → Log transform la target
  • Skewed residuals → Verifică outlieri
  • Low R² dar residuals OK → Adaugă mai multe features
  • Train R² >> Test R² → Reduce complexity (regularization)

🏆 CONCLUZIE:
   Metricile spun "CÂT DE BINE".
   Vizualizările spun "UNDE și DE CE".
   Împreună = COMPLETE MODEL DIAGNOSTIC.
""")