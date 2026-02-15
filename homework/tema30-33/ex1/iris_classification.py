import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

"""
🤖 COMPARING THREE CLASSIFIERS: KNN vs SVM vs Random Forest

Different algorithms, different philosophies:
• K-NN: "You are your neighbors"
• SVM: "Find the best boundary between classes"
• Random Forest: "Let many trees vote"

Today we test all three and find the champion!
"""

print("="*70)
print("🏆 COMPARING THREE CLASSIFIERS ON IRIS DATASET")
print("="*70)

# ===== STEP 1: Load and Prepare Data =====
print("\n📊 STEP 1: Preparing the Iris Dataset")
print("-" * 70)

iris = load_iris()
X = iris.data  # Features: 4 measurements
y = iris.target  # Labels: 0, 1, 2 (species)

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# ===== STEP 2: Train-Test Split (Same as KNN) =====
print("\n✂️  STEP 2: Train-Test Split")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {len(X_train)} flowers")
print(f"Test set: {len(X_test)} flowers")
print(f"Class distribution (train): {np.bincount(y_train)}")
print(f"Class distribution (test):  {np.bincount(y_test)}")

# ===== STEP 3: K-NEAREST NEIGHBORS (Baseline) =====
print("\n\n🤖 CLASSIFIER 1: K-NEAREST NEIGHBORS (K=5)")
print("-" * 70)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)
knn_train_acc = knn.score(X_train, y_train)
knn_test_acc = knn.score(X_test, y_test)

print(f"\n📊 Performance:")
print(f"   Training accuracy: {knn_train_acc*100:.2f}%")
print(f"   Test accuracy:     {knn_test_acc*100:.2f}%")

print(f"\n🧠 How KNN works:")
print(f"   • Lazy learner - doesn't learn a model, just memorizes")
print(f"   • Classifies by checking 5 nearest neighbors")
print(f"   • Fast to train, slow to predict")

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f"\n🎯 Confusion Matrix:")
print(f"   {cm_knn}")

# ===== STEP 4: SUPPORT VECTOR MACHINE (SVM) =====
print("\n\n🎯 CLASSIFIER 2: SUPPORT VECTOR MACHINE (SVM)")
print("-" * 70)

print(f"\n🧠 How SVM works:")
print(f"   • Finds the optimal hyperplane that separates classes")
print(f"   • Maximizes margin between classes")
print(f"   • Uses kernel trick for non-linear boundaries")
print(f"   • Good for high-dimensional data")

svm = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Predictions
y_pred_svm = svm.predict(X_test)
svm_train_acc = svm.score(X_train, y_train)
svm_test_acc = svm.score(X_test, y_test)

print(f"\n📊 Performance:")
print(f"   Training accuracy: {svm_train_acc*100:.2f}%")
print(f"   Test accuracy:     {svm_test_acc*100:.2f}%")

cm_svm = confusion_matrix(y_test, y_pred_svm)
print(f"\n🎯 Confusion Matrix:")
print(f"   {cm_svm}")

# ===== STEP 5: RANDOM FOREST =====
print("\n\n🌲 CLASSIFIER 3: RANDOM FOREST")
print("-" * 70)

print(f"\n🧠 How Random Forest works:")
print(f"   • Builds 100 decision trees (ensemble method)")
print(f"   • Each tree votes on the classification")
print(f"   • Takes majority vote as final prediction")
print(f"   • Robust to outliers and overfitting")

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

print(f"\n📊 Performance:")
print(f"   Training accuracy: {rf_train_acc*100:.2f}%")
print(f"   Test accuracy:     {rf_test_acc*100:.2f}%")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\n🎯 Confusion Matrix:")
print(f"   {cm_rf}")

# ===== STEP 6: MODEL COMPARISON =====
print("\n\n" + "="*70)
print("🏆 MODEL COMPARISON & WINNER")
print("="*70)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Algorithm': ['K-Nearest Neighbors', 'Support Vector Machine', 'Random Forest'],
    'Train Accuracy': [knn_train_acc, svm_train_acc, rf_train_acc],
    'Test Accuracy': [knn_test_acc, svm_test_acc, rf_test_acc],
    'Overfitting Gap': [abs(knn_train_acc - knn_test_acc), 
                         abs(svm_train_acc - svm_test_acc),
                         abs(rf_train_acc - rf_test_acc)]
})

print("\n📈 Accuracy Comparison:")
print(comparison.to_string(index=False))

best_model_idx = comparison['Test Accuracy'].idxmax()
best_model_name = comparison.iloc[best_model_idx]['Algorithm']
best_test_acc = comparison.iloc[best_model_idx]['Test Accuracy']

print(f"\n🥇 WINNER: {best_model_name}")
print(f"   Test Accuracy: {best_test_acc*100:.2f}%")

# ===== STEP 7: DETAILED CLASSIFICATION REPORTS =====
print("\n\n📊 DETAILED CLASSIFICATION REPORTS")
print("-" * 70)

print("\n🤖 K-NEAREST NEIGHBORS:")
print("-" * 50)
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names, digits=4))

print("\n🎯 SUPPORT VECTOR MACHINE:")
print("-" * 50)
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names, digits=4))

print("\n🌲 RANDOM FOREST:")
print("-" * 50)
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names, digits=4))

# ===== STEP 8: FEATURE IMPORTANCE (Random Forest) =====
print("\n\n🔍 FEATURE IMPORTANCE (Random Forest)")
print("-" * 70)

feature_importance = rf.feature_importances_
feature_names = iris.feature_names

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n" + importance_df.to_string(index=False))

print("\n💡 What this means:")
print("   • Petal Length & Width are most important for classification")
print("   • Sepal measurements contribute less")
print("   • Random Forest tells us which features matter most")

# ===== STEP 9: WHY EACH ALGORITHM WORKS =====
print("\n\n💡 WHY EACH ALGORITHM EXCELS/FAILS")
print("-" * 70)

print("\n🤖 K-NEAREST NEIGHBORS (K=5):")
print(f"   ✓ Strength: Simple, intuitive, no training needed")
print(f"   ✓ Strength: Good for this small 4D dataset")
print(f"   ✗ Weakness: Slow predictions (check all training samples)")
print(f"   ✗ Weakness: Sensitive to irrelevant features")
print(f"   📊 Test Accuracy: {knn_test_acc*100:.2f}%")

print("\n🎯 SUPPORT VECTOR MACHINE:")
print(f"   ✓ Strength: Great for high-dimensional data")
print(f"   ✓ Strength: Robust hyperplane boundaries")
print(f"   ✗ Weakness: Slower training")
print(f"   ✗ Weakness: Less interpretable than trees")
print(f"   📊 Test Accuracy: {svm_test_acc*100:.2f}%")

print("\n🌲 RANDOM FOREST:")
print(f"   ✓ Strength: Shows which features matter (feature importance)")
print(f"   ✓ Strength: Handles complex patterns through ensemble voting")
print(f"   ✓ Strength: Fast predictions (parallel tree evaluation)")
print(f"   ✗ Weakness: Can overfit on small datasets")
print(f"   📊 Test Accuracy: {rf_test_acc*100:.2f}%")

# ===== STEP 10: PREDICTION EXAMPLES =====
print("\n\n🌸 EXAMPLE PREDICTIONS ON TEST SET")
print("-" * 70)

for i in range(min(15, len(X_test))):
    actual = iris.target_names[y_test[i]]
    pred_knn = iris.target_names[y_pred_knn[i]]
    pred_svm = iris.target_names[y_pred_svm[i]]
    pred_rf = iris.target_names[y_pred_rf[i]]
    
    match_knn = "✓" if y_test[i] == y_pred_knn[i] else "✗"
    match_svm = "✓" if y_test[i] == y_pred_svm[i] else "✗"
    match_rf = "✓" if y_test[i] == y_pred_rf[i] else "✗"
    
    if i % 5 == 0:
        print()
    
    print(f"Sample {i+1:2d}: Actual={actual:12s} | "
          f"KNN={pred_knn:12s}{match_knn} | "
          f"SVM={pred_svm:12s}{match_svm} | "
          f"RF={pred_rf:12s}{match_rf}")

# ===== VISUALIZATION =====
print("\n\n📊 GENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Confusion Matrices
cm_list = [cm_knn, cm_svm, cm_rf]
titles = ['K-Nearest Neighbors', 'Support Vector Machine', 'Random Forest']
for idx, (cm, title) in enumerate(zip(cm_list, titles)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=iris.target_names,
               yticklabels=iris.target_names,
               ax=axes[0, idx], cbar_kws={'label': 'Count'})
    axes[0, idx].set_xlabel('Predicted')
    axes[0, idx].set_ylabel('Actual')
    axes[0, idx].set_title(f'{title}\nConfusion Matrix', fontweight='bold')

# Row 2: Accuracy Comparison
models = ['K-NN', 'SVM', 'Random Forest']
train_accs = [knn_train_acc, svm_train_acc, rf_train_acc]
test_accs = [knn_test_acc, svm_test_acc, rf_test_acc]

x = np.arange(len(models))
width = 0.35

bars1 = axes[1, 0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
bars2 = axes[1, 0].bar(x + width/2, test_accs, width, label='Test', alpha=0.8)

axes[1, 0].set_xlabel('Model', fontweight='bold')
axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
axes[1, 0].set_title('Accuracy Comparison (Train vs Test)', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models)
axes[1, 0].legend()
axes[1, 0].set_ylim([0.9, 1.02])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)

# Feature Importance (Random Forest)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.7)
axes[1, 1].set_xlabel('Importance', fontweight='bold')
axes[1, 1].set_title('Random Forest: Feature Importance', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

# Overfitting Gap Analysis
gaps = [abs(knn_train_acc - knn_test_acc), 
        abs(svm_train_acc - svm_test_acc),
        abs(rf_train_acc - rf_test_acc)]

axes[1, 2].bar(models, gaps, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[1, 2].set_ylabel('Train-Test Gap', fontweight='bold')
axes[1, 2].set_title('Overfitting Analysis\n(Lower = Better Generalization)', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (model, gap) in enumerate(zip(models, gaps)):
    axes[1, 2].text(i, gap, f'{gap*100:.2f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('iris_classifier_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: iris_classifier_comparison.png")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("🎉 EXERCISE I COMPLETE: THREE CLASSIFIERS COMPARED")
print("="*70)

print("\n📋 SUMMARY:")
print(f"\n   K-Nearest Neighbors (K=5)")
print(f"      • Test Accuracy: {knn_test_acc*100:.2f}%")
print(f"      • Training Time: Very Fast")
print(f"      • Prediction Time: Slow")

print(f"\n   Support Vector Machine (RBF Kernel)")
print(f"      • Test Accuracy: {svm_test_acc*100:.2f}%")
print(f"      • Training Time: Medium")
print(f"      • Prediction Time: Fast")

print(f"\n   Random Forest (100 trees)")
print(f"      • Test Accuracy: {rf_test_acc*100:.2f}%")
print(f"      • Training Time: Medium")
print(f"      • Prediction Time: Very Fast")

print(f"\n🏆 WINNER: {best_model_name} ({best_test_acc*100:.2f}%)")
print("\n✅ All models trained, evaluated, and compared!")
print("✅ Confusion matrices generated for each model")
print("✅ Feature importance analyzed for Random Forest")
print("✅ Visualization saved!")