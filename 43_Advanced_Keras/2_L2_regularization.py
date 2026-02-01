import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers

# ==========================================
# L2 REGULARIZATION EXAMPLE
# ==========================================

# Model with L2 regularization
model_l2 = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01),  # Penalty coefficient
        input_shape=(30,)
    ),
    layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dense(1, activation='sigmoid')
])

model_l2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("="*70)
print("MODEL WITH L2 REGULARIZATION")
print("="*70)
model_l2.summary()

# Train
history_l2 = model_l2.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# Evaluate
l2_test_loss, l2_test_acc = model_l2.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*70)
print("L2 REGULARIZATION RESULTS")
print("="*70)
print(f"Test Accuracy: {l2_test_acc:.4f} ({l2_test_acc*100:.2f}%)")
print(f"Test Loss: {l2_test_loss:.4f}")

# ==========================================
# COMBINING DROPOUT + L2 (Best Practice)
# ==========================================

model_combined = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01),
        input_shape=(30,)
    ),
    layers.Dropout(0.3),  # Dropout after Dense

    layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dropout(0.3),

    layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid')
])

model_combined.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*70)
print("TRAINING WITH BOTH DROPOUT + L2 REGULARIZATION")
print("="*70)

history_combined = model_combined.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

combined_test_loss, combined_test_acc = model_combined.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy: {combined_test_acc:.4f} ({combined_test_acc*100:.2f}%)")
print(f"Test Loss: {combined_test_loss:.4f}")

# ==========================================
# COMPARISON: All Regularization Strategies
# ==========================================

print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON")
print("="*70)
print(f"No Regularization:    {no_dropout_test_acc:.4f}")
print(f"Dropout Only:         {with_dropout_test_acc:.4f}")
print(f"L2 Only:              {l2_test_acc:.4f}")
print(f"Dropout + L2:         {combined_test_acc:.4f}")
print()

best_acc = max(no_dropout_test_acc, with_dropout_test_acc, l2_test_acc, combined_test_acc)
if combined_test_acc == best_acc:
    print("✅ BEST: Combined approach (Dropout + L2)")
    print("   Professional ML teams often use both techniques together")
elif with_dropout_test_acc == best_acc:
    print("✅ BEST: Dropout regularization")
elif l2_test_acc == best_acc:
    print("✅ BEST: L2 regularization")
else:
    print("⚠️  No regularization performed best (may indicate undertting)")
    print("   Consider reducing regularization or increasing model capacity")