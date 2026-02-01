import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# REAL MEDICAL DATA: Breast Cancer Detection
# ==========================================
# 569 patients, 30 features (tumor measurements)
# Binary classification: malignant (1) or benign (0)

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

print("="*70)
print("BREAST CANCER DETECTION DATASET")
print("="*70)
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]} (tumor measurements)")
print(f"Classes: {len(np.unique(y))} (malignant=0, benign=1)")
print(f"Class distribution: {np.bincount(y)}")
print()

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL WITHOUT DROPOUT (Baseline)
# ==========================================

model_no_dropout = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(30,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='no_dropout')

model_no_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("="*70)
print("TRAINING MODEL WITHOUT DROPOUT")
print("="*70)

history_no_dropout = model_no_dropout.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# ==========================================
# MODEL WITH DROPOUT (Advanced)
# ==========================================

model_with_dropout = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(30,)),
    layers.Dropout(0.3),  # Drop 30% of neurons

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),  # Another 30% dropout

    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),  # Lighter dropout in later layers

    layers.Dense(1, activation='sigmoid')
], name='with_dropout')

model_with_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*70)
print("TRAINING MODEL WITH DROPOUT")
print("="*70)

history_with_dropout = model_with_dropout.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# ==========================================
# COMPARISON: The Overfitting Test
# ==========================================

print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)

# Evaluate both models
no_dropout_train_acc = history_no_dropout.history['accuracy'][-1]
no_dropout_val_acc = history_no_dropout.history['val_accuracy'][-1]

with_dropout_train_acc = history_with_dropout.history['accuracy'][-1]
with_dropout_val_acc = history_with_dropout.history['val_accuracy'][-1]

print("\nWithout Dropout:")
print(f"  Training Accuracy: {no_dropout_train_acc:.4f}")
print(f"  Validation Accuracy: {no_dropout_val_acc:.4f}")
print(f"  Gap (Overfitting): {no_dropout_train_acc - no_dropout_val_acc:.4f}")

print("\nWith Dropout:")
print(f"  Training Accuracy: {with_dropout_train_acc:.4f}")
print(f"  Validation Accuracy: {with_dropout_val_acc:.4f}")
print(f"  Gap (Overfitting): {with_dropout_train_acc - with_dropout_val_acc:.4f}")

# Test set evaluation
no_dropout_test_loss, no_dropout_test_acc = model_no_dropout.evaluate(
    X_test, y_test, verbose=0
)
with_dropout_test_loss, with_dropout_test_acc = model_with_dropout.evaluate(
    X_test, y_test, verbose=0
)

print("\n" + "="*70)
print("TEST SET PERFORMANCE (UNSEEN DATA)")
print("="*70)
print(f"Without Dropout: {no_dropout_test_acc:.4f} ({no_dropout_test_acc*100:.2f}%)")
print(f"With Dropout: {with_dropout_test_acc:.4f} ({with_dropout_test_acc*100:.2f}%)")
print()

if with_dropout_test_acc > no_dropout_test_acc:
    print("✅ DROPOUT WINS: Better generalization to unseen data")
    print("   Dropout prevented overfitting to training set patterns")
else:
    print("⚠️  Results may vary—dropout adds randomness")
    print("   Try running multiple times or adjusting dropout rate")

# ==========================================
# VISUALIZATION: Training Dynamics
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
ax1.plot(history_no_dropout.history['accuracy'],
         label='No Dropout - Train', color='red', linestyle='--')
ax1.plot(history_no_dropout.history['val_accuracy'],
         label='No Dropout - Val', color='red')
ax1.plot(history_with_dropout.history['accuracy'],
         label='With Dropout - Train', color='blue', linestyle='--')
ax1.plot(history_with_dropout.history['val_accuracy'],
         label='With Dropout - Val', color='blue')
ax1.set_title('Accuracy: Dropout Effect on Overfitting', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss comparison
ax2.plot(history_no_dropout.history['loss'],
         label='No Dropout - Train', color='red', linestyle='--')
ax2.plot(history_no_dropout.history['val_loss'],
         label='No Dropout - Val', color='red')
ax2.plot(history_with_dropout.history['loss'],
         label='With Dropout - Train', color='blue', linestyle='--')
ax2.plot(history_with_dropout.history['val_loss'],
         label='With Dropout - Val', color='blue')
ax2.set_title('Loss: Training Stability', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dropout_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'dropout_comparison.png'")

# ==========================================
# KEY INSIGHTS
# ==========================================

print("\n" + "="*70)
print("KEY INSIGHTS: WHEN TO USE DROPOUT")
print("="*70)
print("""
1. WHEN TO USE DROPOUT:
   - Deep networks (3+ hidden layers)
   - Large number of parameters relative to data
   - You observe training accuracy >> validation accuracy
   - High-stakes applications (medical, financial)

2. DROPOUT RATES:
   - Typical: 0.2-0.5 (20-50% of neurons dropped)
   - Start with 0.3 (30%) and adjust
   - Can use different rates per layer
   - Later layers often use lower dropout

3. DURING PREDICTION:
   - Dropout is AUTOMATICALLY DISABLED
   - Keras handles this for you
   - All neurons are active during inference

4. THE TRADE-OFF:
   - Lower training accuracy (expected!)
   - Better generalization to new data
   - Worth it for production deployment
""")