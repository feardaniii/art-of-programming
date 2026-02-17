import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# ========================================
# 1. LOAD DATA
# ========================================
print("="*60)
print("🧬 MAN VS MACHINE: BREAST CANCER DETECTION")
print("="*60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = Malignant, 1 = Benign

# Split (Standard 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test Set Size: {len(X_test)} patients")

# ========================================
# 2. THE "IF/ELSE" DOCTOR (ALGORITHMIC)
# ========================================
def manual_doctor_predict(row):
    """
    A hard-coded algorithmic approach based on medical intuition.
    
    Logic:
    Malignant tumors are typically:
    1. Larger (Mean Radius)
    2. Rougher (Mean Texture)
    3. More irregular (Concavity)
    
    Thresholds derived from dataset statistics (mean values).
    """

    # Vote System
    votes_malignant = 0

    # Rule 1: Size (Radius)
    # Mean radius is ~14. If > 15, highly suspicious.
    if row['mean radius'] > 15.0:
        votes_malignant += 1

    # Rule 2: Texture
    # Mean texture is ~19. If > 21, suspicious.
    if row['mean texture'] > 21.0:
        votes_malignant += 1
        
    # Rule 3: Concavity (The "dents" in the cell)
    # Mean is 0.08. If > 0.10, very suspicious.
    if row['mean concavity'] > 0.10:
        votes_malignant += 1

    # Rule 4: Area (Mass)
    # Mean is 655. If > 800, suspicious.
    if row['mean area'] > 800:
        votes_malignant += 1

    # Rule 5: Worst Perimeter (The edge of the largest cell)
    # If > 110, almost certainly malignant.
    if row['worst perimeter'] > 110:
        votes_malignant += 2 # Strong vote

    # DECISION:
    # If we have 2 or more "bad signs", call it Malignant (0).
    # Otherwise Benign (1).
    if votes_malignant >= 2:
        return 0 # Malignant
    else:
        return 1 # Benign

print("\n👨‍⚕️  Running Manual Algorithmic Diagnosis...")
y_pred_algo = X_test.apply(manual_doctor_predict, axis=1)

# ========================================
# 3. THE MACHINE LEARNING MODEL (RF)
# ========================================
print("🤖 Running Random Forest Model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_ml = rf.predict(X_test)

# ========================================
# 4. COMPARISON
# ========================================

acc_algo = accuracy_score(y_test, y_pred_algo)
acc_ml = accuracy_score(y_test, y_pred_ml)

print("\n" + "="*60)
print("🏆 FINAL SCOREBOARD")
print("="*60)
print(f"1. Machine Learning (Random Forest): {acc_ml*100:.2f}% Accuracy")
print(f"2. Algorithmic Logic (If/Else):      {acc_algo*100:.2f}% Accuracy")

print("\n" + "-"*30)
print("📊 DETAILED BREAKDOWN (IF/ELSE APPROACH)")
print("-" * 30)
cm = confusion_matrix(y_test, y_pred_algo)
print("Confusion Matrix:")
print(f"True Malignant identified as Malignant: {cm[0][0]} (Recall: {cm[0][0]/(cm[0][0]+cm[0][1])*100:.1f}%)")
print(f"True Benign identified as Benign:       {cm[1][1]}")
print(f"Missed Cancer (False Negatives):        {cm[0][1]} ⚠️ CRITICAL")
print(f"False Alarm (False Positives):          {cm[1][0]}")

print("\n💡 CONCLUSION:")
if acc_ml > acc_algo:
    print("   The Machine won. Why?")
    print("   1. Non-Linear Interactions: RF considers how texture changes *relative* to size.")
    print("   2. High Dimensions: RF looks at 30 features, we only looked at 5.")
    print("   3. Optimality: RF thresholds (e.g., radius > 14.32) are mathematically precise,")
    print("      whereas our thresholds (e.g., 15.0) were human guesses.")
else:
    print("   The Human Logic won! (Rare)")
    print("   Simple rules can sometimes beat complex models if the signal is very strong.")
