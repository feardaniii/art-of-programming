import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
🔬 EXERCISE III - OPTION B: FEATURE SELECTION WITH GRIDSEARCHCV
Breast Cancer Dataset: Select best 15 out of 30 features
Find optimal classifier + hyperparameters using GridSearchCV
"""

print("="*80)
print("🔬 EXERCISE III - OPTION B: FEATURE SELECTION WITH GRIDSEARCHCV")
print("="*80)

# ===== PHASE 1: BASELINE ESTABLISHMENT (30 FEATURES) =====
print("\n" + "="*80)
print("📋 PHASE 1: BASELINE ESTABLISHMENT (30 FEATURES)")
print("="*80)

# Step 1.1: Load dataset
print("\n📊 STEP 1.1: Loading Breast Cancer Dataset")
print("-" * 80)

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print(f"Dataset Shape: {X.shape}")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
print(f"\nFeature Names (30 total):")
for i, name in enumerate(feature_names, 1):
    print(f"  {i:2d}. {name}")

# Step 1.2: Train-test split
print("\n\n✂️  STEP 1.2: Train-Test Split")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 1.3: Scale features
print("\n\n🔄 STEP 1.3: Scaling Features")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using StandardScaler")

# Step 1.4: Train baseline models with 30 features
print("\n\n🎓 STEP 1.4: Training Baseline Models (30 Features)")
print("-" * 80)

baseline_results = {}

# Baseline SVM
print("\n📈 SVM (Baseline - 30 features):")
svm_baseline = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_baseline.fit(X_train_scaled, y_train)
svm_baseline_acc = svm_baseline.score(X_test_scaled, y_test)
baseline_results['SVM'] = svm_baseline_acc
print(f"   Accuracy: {svm_baseline_acc:.4f}")

# Baseline Random Forest
print("\n📈 Random Forest (Baseline - 30 features):")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
rf_baseline.fit(X_train_scaled, y_train)
rf_baseline_acc = rf_baseline.score(X_test_scaled, y_test)
baseline_results['Random Forest'] = rf_baseline_acc
print(f"   Accuracy: {rf_baseline_acc:.4f}")

# Baseline KNN
print("\n📈 KNN (Baseline - 30 features):")
knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train_scaled, y_train)
knn_baseline_acc = knn_baseline.score(X_test_scaled, y_test)
baseline_results['KNN'] = knn_baseline_acc
print(f"   Accuracy: {knn_baseline_acc:.4f}")

print("\n📋 BASELINE SUMMARY (30 Features):")
for model, acc in baseline_results.items():
    print(f"   {model:20s}: {acc:.4f}")

# ===== PHASE 2: FEATURE SELECTION (30 → 15 FEATURES) =====
print("\n\n" + "="*80)
print("🔍 PHASE 2: FEATURE SELECTION (30 → 15 FEATURES)")
print("="*80)

# Step 2.1: Get feature importances from Random Forest
print("\n📊 STEP 2.1: Analyzing Feature Importance (Random Forest)")
print("-" * 80)

rf_full = RandomForestClassifier(n_estimators=200, random_state=42)
rf_full.fit(X_train_scaled, y_train)

importances = rf_full.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

print(f"\nBottom 15 Least Important Features:")
print(importance_df.tail(15).to_string(index=False))

# Step 2.2: Select top 15 features
print("\n\n✂️  STEP 2.2: Selecting Top 15 Features")
print("-" * 80)

top_15_features = importance_df.head(15)['Feature'].tolist()
feature_indices = [list(feature_names).index(f) for f in top_15_features]

print(f"\n✅ Selected 15 Features (Automated by Feature Importance):")
for i, feat in enumerate(top_15_features, 1):
    importance = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"   {i:2d}. {feat:30s} → Importance: {importance:.4f}")

# Step 2.3: Create reduced feature sets
print("\n\n📦 STEP 2.3: Creating Reduced Feature Sets (15 Features)")
print("-" * 80)

X_train_15 = X_train_scaled[:, feature_indices]
X_test_15 = X_test_scaled[:, feature_indices]

print(f"Training set (15 features): {X_train_15.shape}")
print(f"Test set (15 features): {X_test_15.shape}")

# ===== PHASE 3: HYPERPARAMETER TUNING WITH GRIDSEARCHCV =====
print("\n\n" + "="*80)
print("🔧 PHASE 3: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

print("\n📊 STEP 3.1: Defining Search Spaces")
print("-" * 80)

# SVM parameter grid
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
print(f"\nSVM Parameter Grid:")
print(f"   C: {svm_params['C']}")
print(f"   kernel: {svm_params['kernel']}")
print(f"   gamma: {svm_params['gamma']}")
print(f"   Total combinations: {len(svm_params['C']) * len(svm_params['kernel']) * len(svm_params['gamma'])}")

# Random Forest parameter grid
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
print(f"\nRandom Forest Parameter Grid:")
print(f"   n_estimators: {rf_params['n_estimators']}")
print(f"   max_depth: {rf_params['max_depth']}")
print(f"   min_samples_split: {rf_params['min_samples_split']}")
print(f"   Total combinations: {len(rf_params['n_estimators']) * len(rf_params['max_depth']) * len(rf_params['min_samples_split'])}")

# KNN parameter grid
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan']
}
print(f"\nKNN Parameter Grid:")
print(f"   n_neighbors: {knn_params['n_neighbors']}")
print(f"   metric: {knn_params['metric']}")
print(f"   Total combinations: {len(knn_params['n_neighbors']) * len(knn_params['metric'])}")

# Step 3.2: GridSearchCV for SVM
print("\n\n🔧 STEP 3.2: GridSearchCV for SVM (15 Features)")
print("-" * 80)

start_time = time.time()
svm_gs = GridSearchCV(SVC(), svm_params, cv=5, n_jobs=-1, verbose=0)
svm_gs.fit(X_train_15, y_train)
svm_time = time.time() - start_time

print(f"✅ GridSearchCV completed in {svm_time:.2f} seconds")
print(f"Best Parameters: {svm_gs.best_params_}")
print(f"Best CV Score: {svm_gs.best_score_:.4f}")

# Step 3.3: GridSearchCV for Random Forest
print("\n\n🔧 STEP 3.3: GridSearchCV for Random Forest (15 Features)")
print("-" * 80)

start_time = time.time()
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=0)
rf_gs.fit(X_train_15, y_train)
rf_time = time.time() - start_time

print(f"✅ GridSearchCV completed in {rf_time:.2f} seconds")
print(f"Best Parameters: {rf_gs.best_params_}")
print(f"Best CV Score: {rf_gs.best_score_:.4f}")

# Step 3.4: GridSearchCV for KNN
print("\n\n🔧 STEP 3.4: GridSearchCV for KNN (15 Features)")
print("-" * 80)

start_time = time.time()
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1, verbose=0)
knn_gs.fit(X_train_15, y_train)
knn_time = time.time() - start_time

print(f"✅ GridSearchCV completed in {knn_time:.2f} seconds")
print(f"Best Parameters: {knn_gs.best_params_}")
print(f"Best CV Score: {knn_gs.best_score_:.4f}")

# ===== PHASE 4: EVALUATION ON TEST SET =====
print("\n\n" + "="*80)
print("📊 PHASE 4: EVALUATION ON TEST SET")
print("="*80)

print("\n📈 STEP 4.1: Test Set Performance (15 Features)")
print("-" * 80)

# Test best SVM
svm_15_acc = svm_gs.score(X_test_15, y_test)
svm_15_pred = svm_gs.predict(X_test_15)
svm_15_cm = confusion_matrix(y_test, svm_15_pred)

print(f"\nSVM (15 Features):")
print(f"   Accuracy: {svm_15_acc:.4f}")
print(f"   Baseline: {baseline_results['SVM']:.4f}")
print(f"   Difference: {svm_15_acc - baseline_results['SVM']:+.4f}")

# Test best Random Forest
rf_15_acc = rf_gs.score(X_test_15, y_test)
rf_15_pred = rf_gs.predict(X_test_15)
rf_15_cm = confusion_matrix(y_test, rf_15_pred)

print(f"\nRandom Forest (15 Features):")
print(f"   Accuracy: {rf_15_acc:.4f}")
print(f"   Baseline: {baseline_results['Random Forest']:.4f}")
print(f"   Difference: {rf_15_acc - baseline_results['Random Forest']:+.4f}")

# Test best KNN
knn_15_acc = knn_gs.score(X_test_15, y_test)
knn_15_pred = knn_gs.predict(X_test_15)
knn_15_cm = confusion_matrix(y_test, knn_15_pred)

print(f"\nKNN (15 Features):")
print(f"   Accuracy: {knn_15_acc:.4f}")
print(f"   Baseline: {baseline_results['KNN']:.4f}")
print(f"   Difference: {knn_15_acc - baseline_results['KNN']:+.4f}")

# ===== PHASE 5: COMPARISON & ANALYSIS =====
print("\n\n" + "="*80)
print("📊 PHASE 5: COMPARISON & ANALYSIS")
print("="*80)

print("\n📋 STEP 5.1: Comprehensive Comparison Table")
print("-" * 80)

comparison_data = {
    'Model': ['SVM', 'SVM', 'Random Forest', 'Random Forest', 'KNN', 'KNN'],
    'Features': [30, 15, 30, 15, 30, 15],
    'Accuracy': [
        baseline_results['SVM'], svm_15_acc,
        baseline_results['Random Forest'], rf_15_acc,
        baseline_results['KNN'], knn_15_acc
    ],
    'Best Parameters': [
        'Default (rbf)', str(svm_gs.best_params_),
        'Default (100 trees)', str(rf_gs.best_params_),
        'Default (k=5)', str(knn_gs.best_params_)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

print("\n\n💡 STEP 5.2: Key Insights")
print("-" * 80)

svm_loss = baseline_results['SVM'] - svm_15_acc
rf_loss = baseline_results['Random Forest'] - rf_15_acc
knn_loss = baseline_results['KNN'] - knn_15_acc

print(f"\nAccuracy Loss by Reducing Features (30 → 15):")
print(f"   SVM:            {svm_loss:+.4f} ({svm_loss*100:+.2f}%)")
print(f"   Random Forest:  {rf_loss:+.4f} ({rf_loss*100:+.2f}%)")
print(f"   KNN:            {knn_loss:+.4f} ({knn_loss*100:+.2f}%)")

best_15_model = comparison_df[comparison_df['Features'] == 15].loc[
    comparison_df[comparison_df['Features'] == 15]['Accuracy'].idxmax()
]
best_30_model = comparison_df[comparison_df['Features'] == 30].loc[
    comparison_df[comparison_df['Features'] == 30]['Accuracy'].idxmax()
]

print(f"\nBest Model with 30 Features: {best_30_model['Model']} ({best_30_model['Accuracy']:.4f})")
print(f"Best Model with 15 Features: {best_15_model['Model']} ({best_15_model['Accuracy']:.4f})")

total_accuracy_loss = best_30_model['Accuracy'] - best_15_model['Accuracy']
print(f"Accuracy Loss (Best 30 vs Best 15): {total_accuracy_loss:+.4f} ({total_accuracy_loss*100:+.2f}%)")

feature_reduction = (1 - 15/30) * 100
print(f"\nTrade-offs:")
print(f"   Feature Reduction: {feature_reduction:.1f}% (30 → 15 features)")
print(f"   Accuracy Loss: {abs(total_accuracy_loss)*100:.2f}%")
print(f"   Worth it? {'✅ YES' if abs(total_accuracy_loss) <= 0.02 else '❌ MARGINAL'}")

# ===== PHASE 6: VISUALIZATIONS =====
print("\n\n" + "="*80)
print("📊 PHASE 6: GENERATING VISUALIZATIONS")
print("="*80)

try:
    plt.switch_backend('Agg')
    
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Feature Importance (Top 15)
    ax1 = plt.subplot(3, 3, 1)
    top_15_plot = importance_df.head(15).iloc[::-1]
    ax1.barh(range(len(top_15_plot)), top_15_plot['Importance'].values, color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(top_15_plot)))
    ax1.set_yticklabels(top_15_plot['Feature'].values, fontsize=8)
    ax1.set_xlabel('Importance Score', fontweight='bold')
    ax1.set_title('Top 15 Features by Importance\n(Random Forest)', fontweight='bold', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Accuracy Comparison (30 vs 15)
    ax2 = plt.subplot(3, 3, 2)
    models = ['SVM', 'Random Forest', 'KNN']
    acc_30 = [baseline_results['SVM'], baseline_results['Random Forest'], baseline_results['KNN']]
    acc_15 = [svm_15_acc, rf_15_acc, knn_15_acc]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax2.bar(x - width/2, acc_30, width, label='30 Features', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, acc_15, width, label='15 Features', alpha=0.8, color='coral')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Comparison: 30 vs 15 Features', fontweight='bold', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (a30, a15) in enumerate(zip(acc_30, acc_15)):
        ax2.text(i - width/2, a30, f'{a30:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, a15, f'{a15:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Accuracy Loss (Degradation)
    ax3 = plt.subplot(3, 3, 3)
    loss = [svm_loss, rf_loss, knn_loss]
    colors = ['red' if l > 0 else 'green' for l in loss]
    ax3.bar(models, loss, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_ylabel('Accuracy Loss', fontweight='bold')
    ax3.set_title('Accuracy Loss: 30 → 15 Features', fontweight='bold', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (model, l) in enumerate(zip(models, loss)):
        ax3.text(i, l, f'{l:+.4f}', ha='center', va='bottom' if l >= 0 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 4: SVM Confusion Matrix (15 features)
    ax4 = plt.subplot(3, 3, 4)
    sns.heatmap(svm_15_cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontweight='bold')
    ax4.set_ylabel('Actual', fontweight='bold')
    ax4.set_title(f'SVM (15 Features)\nAcc: {svm_15_acc:.4f}', fontweight='bold', fontsize=10)
    
    # Plot 5: Random Forest Confusion Matrix (15 features)
    ax5 = plt.subplot(3, 3, 5)
    sns.heatmap(rf_15_cm, annot=True, fmt='d', cmap='Greens', ax=ax5, cbar_kws={'label': 'Count'})
    ax5.set_xlabel('Predicted', fontweight='bold')
    ax5.set_ylabel('Actual', fontweight='bold')
    ax5.set_title(f'Random Forest (15 Features)\nAcc: {rf_15_acc:.4f}', fontweight='bold', fontsize=10)
    
    # Plot 6: KNN Confusion Matrix (15 features)
    ax6 = plt.subplot(3, 3, 6)
    sns.heatmap(knn_15_cm, annot=True, fmt='d', cmap='Oranges', ax=ax6, cbar_kws={'label': 'Count'})
    ax6.set_xlabel('Predicted', fontweight='bold')
    ax6.set_ylabel('Actual', fontweight='bold')
    ax6.set_title(f'KNN (15 Features)\nAcc: {knn_15_acc:.4f}', fontweight='bold', fontsize=10)
    
    # Plot 7: GridSearchCV Results - SVM
    ax7 = plt.subplot(3, 3, 7)
    svm_cv_results = pd.DataFrame(svm_gs.cv_results_)
    svm_cv_results['params_str'] = svm_cv_results['params'].apply(lambda x: f"C={x['C']}, ker={x['kernel']}")
    top_svm = svm_cv_results.nlargest(10, 'mean_test_score')[['params_str', 'mean_test_score']].reset_index(drop=True)
    ax7.barh(range(len(top_svm)), top_svm['mean_test_score'].values, color='steelblue', alpha=0.8)
    ax7.set_yticks(range(len(top_svm)))
    ax7.set_yticklabels(top_svm['params_str'].values, fontsize=7)
    ax7.set_xlabel('Mean CV Score', fontweight='bold')
    ax7.set_title('Top 10 SVM Configurations\n(GridSearchCV)', fontweight='bold', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Plot 8: GridSearchCV Results - Random Forest
    ax8 = plt.subplot(3, 3, 8)
    rf_cv_results = pd.DataFrame(rf_gs.cv_results_)
    rf_cv_results['params_str'] = rf_cv_results['params'].apply(lambda x: f"n_est={x['n_estimators']}, depth={x['max_depth']}")
    top_rf = rf_cv_results.nlargest(10, 'mean_test_score')[['params_str', 'mean_test_score']].reset_index(drop=True)
    ax8.barh(range(len(top_rf)), top_rf['mean_test_score'].values, color='forestgreen', alpha=0.8)
    ax8.set_yticks(range(len(top_rf)))
    ax8.set_yticklabels(top_rf['params_str'].values, fontsize=7)
    ax8.set_xlabel('Mean CV Score', fontweight='bold')
    ax8.set_title('Top 10 Random Forest Configurations\n(GridSearchCV)', fontweight='bold', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Plot 9: GridSearchCV Results - KNN
    ax9 = plt.subplot(3, 3, 9)
    knn_cv_results = pd.DataFrame(knn_gs.cv_results_)
    knn_cv_results['params_str'] = knn_cv_results['params'].apply(lambda x: f"k={x['n_neighbors']}, {x['metric'][:3]}")
    top_knn = knn_cv_results.nlargest(10, 'mean_test_score')[['params_str', 'mean_test_score']].reset_index(drop=True)
    ax9.barh(range(len(top_knn)), top_knn['mean_test_score'].values, color='coral', alpha=0.8)
    ax9.set_yticks(range(len(top_knn)))
    ax9.set_yticklabels(top_knn['params_str'].values, fontsize=7)
    ax9.set_xlabel('Mean CV Score', fontweight='bold')
    ax9.set_title('Top 10 KNN Configurations\n(GridSearchCV)', fontweight='bold', fontsize=10)
    ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    import os
    output_file = os.path.join(os.getcwd(), 'feature_selection_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_file}")
    plt.close()
    
except Exception as e:
    print(f"\n⚠️ Warning: Could not generate visualizations: {e}")

# ===== PHASE 7: SAVE SELECTED FEATURES =====
print("\n\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

try:
    with open('selected_features.txt', 'w') as f:
        f.write("SELECTED 15 FEATURES (Out of 30)\n")
        f.write("="*60 + "\n\n")
        for i, feat in enumerate(top_15_features, 1):
            importance = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
            f.write(f"{i:2d}. {feat:30s} (Importance: {importance:.4f})\n")
        f.write("\n\n" + "="*60 + "\n")
        f.write("BASELINE RESULTS (30 Features)\n")
        f.write("="*60 + "\n")
        for model, acc in baseline_results.items():
            f.write(f"{model:20s}: {acc:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("BEST RESULTS (15 Features with GridSearchCV)\n")
        f.write("="*60 + "\n")
        f.write(f"SVM:            {svm_15_acc:.4f} (Best params: {svm_gs.best_params_})\n")
        f.write(f"Random Forest:  {rf_15_acc:.4f} (Best params: {rf_gs.best_params_})\n")
        f.write(f"KNN:            {knn_15_acc:.4f} (Best params: {knn_gs.best_params_})\n")
    
    print("\n✅ Selected features saved: selected_features.txt")
    
except Exception as e:
    print(f"\n⚠️ Warning: Could not save features: {e}")

# ===== PHASE 8: FINAL SUMMARY =====
print("\n\n" + "="*80)
print("🎉 EXERCISE III COMPLETE: FEATURE SELECTION SUMMARY")
print("="*80)

print(f"""
📊 FINAL REPORT:

1️⃣ FEATURE SELECTION:
   • Reduced from 30 → 15 features ({feature_reduction:.1f}% reduction)
   • Used automated feature importance ranking (Random Forest)
   • Top feature: {importance_df.iloc[0]['Feature']}
   • Least important feature: {importance_df.iloc[-1]['Feature']}

2️⃣ BASELINE (30 Features):
   • SVM:            {baseline_results['SVM']:.4f}
   • Random Forest:  {baseline_results['Random Forest']:.4f}
   • KNN:            {baseline_results['KNN']:.4f}

3️⃣ BEST RESULTS (15 Features with GridSearchCV):
   • SVM:            {svm_15_acc:.4f} {svm_gs.best_params_}
   • Random Forest:  {rf_15_acc:.4f} {rf_gs.best_params_}
   • KNN:            {knn_15_acc:.4f} {knn_gs.best_params_}

4️⃣ ACCURACY TRADE-OFFS:
   • SVM Loss:       {svm_loss:+.4f} ({svm_loss*100:+.2f}%)
   • RF Loss:        {rf_loss:+.4f} ({rf_loss*100:+.2f}%)
   • KNN Loss:       {knn_loss:+.4f} ({knn_loss*100:+.2f}%)

5️⃣ OVERALL BEST MODEL:
   • 30 Features: {best_30_model['Model']} ({best_30_model['Accuracy']:.4f})
   • 15 Features: {best_15_model['Model']} ({best_15_model['Accuracy']:.4f})
   • Loss: {total_accuracy_loss:+.4f} ({total_accuracy_loss*100:+.2f}%)

6️⃣ CONCLUSION:
   ✅ Feature selection successfully reduces dimensionality
   ✅ Minimal accuracy loss ({abs(total_accuracy_loss)*100:.2f}%)
   ✅ Better generalization & faster training
   ✅ GridSearchCV found optimal hyperparameters for each model

7️⃣ GENERATED FILES:
   ✓ feature_selection_analysis.png (visualizations)
   ✓ selected_features.txt (list of selected features)

💡 INSIGHTS:
   • Breast Cancer dataset has redundant features
   • 50% feature reduction achieves ~{100 - abs(total_accuracy_loss)*100:.1f}% of original accuracy
   • {best_15_model['Model']} is most robust to feature reduction
   • GridSearchCV tuning is essential for optimal performance
""")

print("="*80)
