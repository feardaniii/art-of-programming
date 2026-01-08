from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Presupunem că avem X_train, X_test, y_train, y_test (NON-scaled!)

# ========================================
# PIPELINE 1: SVM CU STANDARD SCALER
# ========================================

print("=" * 60)
print("🔵 PIPELINE 1: StandardScaler → SVM")
print("=" * 60)

# Creează pipeline-ul
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
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

