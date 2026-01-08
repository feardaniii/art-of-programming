import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Presupunem că avem predicțiile din partea anterioară
# y_test, y_test_pred_lr, y_test_pred_ridge, y_test_pred_rf

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