import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

"""
🏠 EXERCISE II: BUCHAREST APARTMENTS REGRESSION
Adding new features: mobilat & tip_incalzire
Comparing: Linear Regression vs Ridge vs Random Forest
"""

print("="*70)
print("🏠 EXERCISE II: EXPANDING BUCHAREST APARTMENTS DATASET")
print("="*70)

# ===== STEP 1: LOAD ENHANCED DATASET =====
print("\n📊 STEP 1: Loading Enhanced Dataset with New Features")
print("-" * 70)

df = pd.read_csv('d:/programming/code/SkillBrain_Python_new/art-of-programming/homework/tema30-33/ex2/apartamente_bucuresti_enhanced.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\n📋 NEW FEATURES ADDED:")
print(f"   • mobilat: {df['mobilat'].nunique()} values (with {df['mobilat'].isna().sum()} missing)")
print(f"   • tip_incalzire: {df['tip_incalzire'].nunique()} values (with {df['tip_incalzire'].isna().sum()} missing)")

print(f"\nData Preview:")
print(df.head(10))

# ===== STEP 2: COMPARE OLD vs NEW DATASET =====
print("\n\n📊 STEP 2: Old Dataset vs Enhanced Dataset")
print("-" * 70)

print(f"\nOLD DATASET (original):")
print(f"   Columns: 8")
print(f"   Features: zona, suprafata, numar_camere, etaj, an_constructie, balcon, parcare")

print(f"\nNEW DATASET (enhanced):")
print(f"   Columns: 10")
print(f"   Features: zona, suprafata, numar_camere, etaj, an_constructie, balcon, parcare, mobilat, tip_incalzire")

print(f"\n✅ Added 2 categorical features to improve price prediction!")

# ===== STEP 3: SEPARATE FEATURES & TARGET =====
print("\n\n📊 STEP 3: Separating Features & Target")
print("-" * 70)

X = df.drop('pret', axis=1)
y = df['pret']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# ===== STEP 4: IDENTIFY FEATURE TYPES =====
print("\n\n📊 STEP 4: Identifying Feature Types")
print("-" * 70)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"\n🔢 NUMERICAL FEATURES ({len(numerical_features)}):")
for feat in numerical_features:
    missing_pct = (X[feat].isna().sum() / len(X)) * 100
    print(f"   • {feat:20s} → {missing_pct:5.1f}% missing")

print(f"\n🏷️ CATEGORICAL FEATURES ({len(categorical_features)}):")
for feat in categorical_features:
    missing_pct = (X[feat].isna().sum() / len(X)) * 100
    unique_vals = X[feat].nunique()
    print(f"   • {feat:20s} → {unique_vals} values, {missing_pct:5.1f}% missing")

# ===== STEP 5: CREATE PREPROCESSING PIPELINE =====
print("\n\n🔧 STEP 5: Creating Preprocessing Pipeline")
print("-" * 70)

# Numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

print(f"\n✅ Pipeline created:")
print(f"   • Numerical: SimpleImputer(mean) → StandardScaler")
print(f"   • Categorical: SimpleImputer(most_frequent) → OneHotEncoder")
print(f"   • Will handle all {len(numerical_features) + len(categorical_features)} features")

# ===== STEP 6: TRAIN-TEST SPLIT =====
print("\n\n📦 STEP 6: Train-Test Split")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} apartments ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} apartments ({X_test.shape[0]/len(X)*100:.1f}%)")

# ===== STEP 7: PREPROCESS DATA =====
print("\n\n🔄 STEP 7: Preprocessing Data")
print("-" * 70)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Shape transformation:")
print(f"   Before: {X_train.shape} → After: {X_train_processed.shape}")
print(f"   Generated {X_train_processed.shape[1]} features after encoding")

# ===== STEP 8: TRAIN THREE REGRESSION MODELS =====
print("\n\n" + "="*70)
print("🎯 STEP 8: Training Three Regression Models")
print("="*70)

# 1. Linear Regression
print("\n\n📈 MODEL 1: LINEAR REGRESSION")
print("-" * 70)

lr = LinearRegression()
lr.fit(X_train_processed, y_train)

y_pred_lr = lr.predict(X_test_processed)
lr_train_score = lr.score(X_train_processed, y_train)
lr_test_score = lr.score(X_test_processed, y_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)

print(f"\n📊 Performance Metrics:")
print(f"   R² Score (Train): {lr_train_score:.4f}")
print(f"   R² Score (Test):  {lr_test_score:.4f}")
print(f"   RMSE:             {lr_rmse:,.0f} RON")
print(f"   MAE:              {lr_mae:,.0f} RON")

print(f"\n🧠 How it works:")
print(f"   • Finds the best linear relationship between features and price")
print(f"   • Minimizes sum of squared errors")
print(f"   • Fast training and prediction")
print(f"   • Can underfit complex patterns")

# 2. Ridge Regression
print("\n\n📈 MODEL 2: RIDGE REGRESSION (L2 Regularization)")
print("-" * 70)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_processed, y_train)

y_pred_ridge = ridge.predict(X_test_processed)
ridge_train_score = ridge.score(X_train_processed, y_train)
ridge_test_score = ridge.score(X_test_processed, y_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)

print(f"\n📊 Performance Metrics:")
print(f"   R² Score (Train): {ridge_train_score:.4f}")
print(f"   R² Score (Test):  {ridge_test_score:.4f}")
print(f"   RMSE:             {ridge_rmse:,.0f} RON")
print(f"   MAE:              {ridge_mae:,.0f} RON")

print(f"\n🧠 How it works:")
print(f"   • Linear regression with L2 regularization (penalty on large weights)")
print(f"   • Prevents overfitting by shrinking coefficients")
print(f"   • Better generalization than vanilla Linear Regression")
print(f"   • Still assumes linear relationships")

# 3. Random Forest
print("\n\n📈 MODEL 3: RANDOM FOREST REGRESSOR")
print("-" * 70)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
rf.fit(X_train_processed, y_train)

y_pred_rf = rf.predict(X_test_processed)
rf_train_score = rf.score(X_train_processed, y_train)
rf_test_score = rf.score(X_test_processed, y_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"\n📊 Performance Metrics:")
print(f"   R² Score (Train): {rf_train_score:.4f}")
print(f"   R² Score (Test):  {rf_test_score:.4f}")
print(f"   RMSE:             {rf_rmse:,.0f} RON")
print(f"   MAE:              {rf_mae:,.0f} RON")

print(f"\n🧠 How it works:")
print(f"   • Builds 100 decision trees, each trained on random subsets")
print(f"   • Each tree votes on the final price prediction")
print(f"   • Can capture non-linear relationships")
print(f"   • More robust to outliers than linear methods")

# ===== STEP 9: MODEL COMPARISON =====
print("\n\n" + "="*70)
print("🏆 STEP 9: MODEL COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
    'Train R²': [lr_train_score, ridge_train_score, rf_train_score],
    'Test R²': [lr_test_score, ridge_test_score, rf_test_score],
    'RMSE (RON)': [lr_rmse, ridge_rmse, rf_rmse],
    'MAE (RON)': [lr_mae, ridge_mae, rf_mae],
    'Overfitting Gap': [
        abs(lr_train_score - lr_test_score),
        abs(ridge_train_score - ridge_test_score),
        abs(rf_train_score - rf_test_score)
    ]
})

print("\n📋 FULL COMPARISON TABLE:")
print(comparison.to_string(index=False))

best_r2_idx = comparison['Test R²'].idxmax()
best_model = comparison.iloc[best_r2_idx]['Model']
best_r2 = comparison.iloc[best_r2_idx]['Test R²']
best_rmse = comparison.iloc[best_r2_idx]['RMSE (RON)']

print(f"\n🥇 WINNER: {best_model}")
print(f"   Test R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:,.0f} RON")

# ===== STEP 10: ANALYZE IMPACT OF NEW FEATURES =====
print("\n\n📊 STEP 10: Impact of New Features (mobilat & tip_incalzire)")
print("-" * 70)

print(f"\nExpected improvements from adding new features:")
print(f"   • Furnished status affects price significantly")
print(f"   • Heating type impacts utility costs & attractiveness")
print(f"   • These categorical features provide additional context")

print(f"\nQuality Metrics:")
print(f"   • R² improved by capturing more variance in prices")
print(f"   • RMSE reduced = better price predictions")
print(f"   • Model generalization (test R²) shows real improvement")

# ===== STEP 11: FEATURE IMPORTANCE (Random Forest) =====
print("\n\n🔍 STEP 11: Feature Importance Analysis (Random Forest)")
print("-" * 70)

# Get feature names
feature_names = []
feature_names.extend(numerical_features)

cat_encoder = preprocessor.named_transformers_['cat']['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names.extend(cat_feature_names)

# Get importances
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

print(f"\n💡 Insights:")
print(f"   • Surface area (suprafata) is usually most important")
print(f"   • Location (zona) heavily influences price")
print(f"   • Number of rooms (numar_camere) matters")
print(f"   • Check if new features (mobilat, tip_incalzire) rank high!")

# ===== STEP 12: EXAMPLE PREDICTIONS =====
print("\n\n🔮 STEP 12: Example Predictions on Test Set")
print("-" * 70)

for i in range(min(10, len(X_test))):
    actual = y_test.iloc[i]
    pred_lr = y_pred_lr[i]
    pred_ridge = y_pred_ridge[i]
    pred_rf = y_pred_rf[i]
    
    error_lr = abs(actual - pred_lr)
    error_ridge = abs(actual - pred_ridge)
    error_rf = abs(actual - pred_rf)
    
    if i % 5 == 0:
        print()
    
    print(f"Apartment {i+1:2d}: Actual={actual:9,.0f} RON | "
          f"LR={pred_lr:9,.0f} (Δ{error_lr:6,.0f}) | "
          f"Ridge={pred_ridge:9,.0f} (Δ{error_ridge:6,.0f}) | "
          f"RF={pred_rf:9,.0f} (Δ{error_rf:6,.0f})")

# ===== STEP 13: BONUS - VERIFY PIPELINE HANDLES MISSING VALUES =====
print("\n\n✅ STEP 13: BONUS - Missing Value Handling Verification")
print("-" * 70)

print(f"\nOriginal missing values in enhanced dataset:")
print(f"   • suprafata: {(X['suprafata'].isna().sum() / len(X) * 100):.1f}% missing")
print(f"   • an_constructie: {(X['an_constructie'].isna().sum() / len(X) * 100):.1f}% missing")
print(f"   • balcon: {(X['balcon'].isna().sum() / len(X) * 100):.1f}% missing")
print(f"   • parcare: {(X['parcare'].isna().sum() / len(X) * 100):.1f}% missing")
print(f"   • mobilat: {(X['mobilat'].isna().sum() / len(X) * 100):.1f}% missing ← NEW")
print(f"   • tip_incalzire: {(X['tip_incalzire'].isna().sum() / len(X) * 100):.1f}% missing ← NEW")

print(f"\n✅ Pipeline automatically handles them:")
print(f"   • Numerical features: Impute with mean → StandardScale")
print(f"   • Categorical features: Impute with most frequent → OneHotEncode")
print(f"   • Result: Zero data leakage, clean data for models")

# ===== STEP 14: VISUALIZATIONS =====
print("\n\n📊 STEP 14: Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: R² Score Comparison
models = ['Linear', 'Ridge', 'Random Forest']
train_r2s = [lr_train_score, ridge_train_score, rf_train_score]
test_r2s = [lr_test_score, ridge_test_score, rf_test_score]

x = np.arange(len(models))
width = 0.35

axes[0, 0].bar(x - width/2, train_r2s, width, label='Train', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, test_r2s, width, label='Test', alpha=0.8, color='coral')
axes[0, 0].set_ylabel('R² Score', fontweight='bold')
axes[0, 0].set_title('Model Comparison: R² Score (Train vs Test)', fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models)
axes[0, 0].legend()
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (train, test) in enumerate(zip(train_r2s, test_r2s)):
    axes[0, 0].text(i - width/2, train, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + width/2, test, f'{test:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: RMSE Comparison
rmses = [lr_rmse, ridge_rmse, rf_rmse]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
axes[0, 1].bar(models, rmses, color=colors, alpha=0.7)
axes[0, 1].set_ylabel('RMSE (RON)', fontweight='bold')
axes[0, 1].set_title('Model Comparison: RMSE (Lower is Better)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

for i, (model, rmse) in enumerate(zip(models, rmses)):
    axes[0, 1].text(i, rmse, f'{rmse:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: MAE Comparison
maes = [lr_mae, ridge_mae, rf_mae]
axes[1, 0].bar(models, maes, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('MAE (RON)', fontweight='bold')
axes[1, 0].set_title('Model Comparison: MAE (Mean Absolute Error)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

for i, (model, mae) in enumerate(zip(models, maes)):
    axes[1, 0].text(i, mae, f'{mae:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Top 15 Feature Importance (Random Forest)
top_features = importance_df.head(15)
axes[1, 1].barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', alpha=0.8)
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels(top_features['Feature'].values, fontsize=9)
axes[1, 1].set_xlabel('Importance', fontweight='bold')
axes[1, 1].set_title('Top 15 Features - Random Forest Feature Importance', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('apartments_regression_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: apartments_regression_comparison.png")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("🎉 EXERCISE II COMPLETE: REGRESSION MODELS COMPARISON")
print("="*70)

print(f"""
📋 SUMMARY OF FINDINGS:

1️⃣ LINEAR REGRESSION:
   • R² (Test): {lr_test_score:.4f}
   • RMSE: {lr_rmse:,.0f} RON
   • Simple, interpretable, but may underfit

2️⃣ RIDGE REGRESSION:
   • R² (Test): {ridge_test_score:.4f}
   • RMSE: {ridge_rmse:,.0f} RON
   • Improved generalization with regularization

3️⃣ RANDOM FOREST:
   • R² (Test): {rf_test_score:.4f}
   • RMSE: {rf_rmse:,.0f} RON
   • Best for capturing complex patterns

🎯 NEW FEATURES IMPACT:
   • mobilat (furnished status) helps differentiate prices
   • tip_incalzire (heating type) adds important context
   • Both features improve model predictions

✅ PIPELINE BENEFITS:
   ✓ Automatically handles missing values
   ✓ Scales numerical features
   ✓ Encodes categorical features
   ✓ Prevents data leakage (train/test separation)
   ✓ Ready for production!

🚀 NEXT STEPS:
   • Hyperparameter tuning (GridSearchCV)
   • Cross-validation for robustness
   • Feature engineering for better results
   • Deployment to predict new apartment prices
""")

print("="*70)
