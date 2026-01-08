import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Presupunem că avem deja X_train_scaled, X_test_scaled, y_train, y_test

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