# ==========================================
# PRODUCTION MODEL OPTIMIZATION
# ==========================================
# Taking our Health & Finance models from previous sessions
# and preparing them for cloud deployment

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

print("="*70)
print("SESSION 47: PRODUCTION MODEL FINE-TUNING")
print("="*70)
print()

# ==========================================
# SCENARIO 1: DIABETES RISK PREDICTION MODEL
# ==========================================

print("SCENARIO 1: Healthcare - Diabetes Risk Prediction")
print("="*70)
print("Real-World Need: Rural clinics need fast, reliable diabetes screening")
print("Model must: 1) Run in <100ms  2) Handle missing data  3) Be explainable")
print()

# Load diabetes dataset (built-in scikit-learn)
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Binary classification: high risk (>140) vs normal
y_binary = (y > 140).astype(int)

print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
print(f"High risk patients: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
print()

# ==========================================
# DATA PREPROCESSING (PRODUCTION-GRADE)
# ==========================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Standardization (save scaler for deployment!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data preprocessed and split")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print()

# ==========================================
# BUILD PRODUCTION-OPTIMIZED MODEL
# ==========================================

def build_production_model(input_dim, model_name="health_predictor"):
    """
    Build a production-optimized model:
    - Small size (deployable to edge devices)
    - Fast inference (<100ms)
    - Robust to missing data
    - L2 regularization to prevent overfitting
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        # Layer 1: Dense with L2 regularization
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='features_layer_1'
        ),
        layers.BatchNormalization(),  # Improves training stability
        layers.Dropout(0.3),

        # Layer 2: Smaller for efficiency
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='features_layer_2'
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Layer 3: Even smaller
        layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            name='features_layer_3'
        ),

        # Output: Binary classification
        layers.Dense(1, activation='sigmoid', name='risk_probability')
    ], name=model_name)

    return model

# Build model
model = build_production_model(X_train.shape[1])

print("="*70)
print("PRODUCTION MODEL ARCHITECTURE")
print("="*70)
model.summary()
print()

# Count parameters
total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
print(f"Estimated model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
print()

# ==========================================
# COMPILE WITH PRODUCTION CONSIDERATIONS
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("✅ Model compiled with production metrics")
print("   - Accuracy: Overall correctness")
print("   - Precision: Of predicted high-risk, how many actually are?")
print("   - Recall: Of all high-risk cases, how many did we catch?")
print("   - AUC: Overall discrimination ability")
print()

# ==========================================
# TRAIN WITH EARLY STOPPING & CHECKPOINTING
# ==========================================

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'health_model_best.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("="*70)
print("TRAINING PRODUCTION MODEL")
print("="*70)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,  # Will stop early if no improvement
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print()
print("✅ Training complete!")
print()

# ==========================================
# EVALUATE ON TEST SET
# ==========================================

print("="*70)
print("PRODUCTION MODEL EVALUATION")
print("="*70)

test_results = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")
print(f"Test AUC: {test_results[4]:.4f}")
print()

# ==========================================
# INFERENCE SPEED TEST
# ==========================================

import time

# Warmup
_ = model.predict(X_test_scaled[:1], verbose=0)

# Time 100 predictions
inference_times = []
for _ in range(100):
    start = time.time()
    _ = model.predict(X_test_scaled[:1], verbose=0)
    inference_times.append((time.time() - start) * 1000)  # ms

avg_inference = np.mean(inference_times)
p95_inference = np.percentile(inference_times, 95)

print("="*70)
print("INFERENCE PERFORMANCE")
print("="*70)
print(f"Average inference time: {avg_inference:.2f} ms")
print(f"95th percentile: {p95_inference:.2f} ms")
print(f"Throughput: {1000/avg_inference:.0f} predictions/second")
print()

if avg_inference < 100:
    print("✅ PRODUCTION READY: Inference time < 100ms target")
else:
    print("⚠️  WARNING: Inference time exceeds 100ms target")
print()

# ==========================================
# SAVE PRODUCTION ARTIFACTS
# ==========================================

# Save model
model.save('health_predictor_production.keras')
print("✅ Model saved: health_predictor_production.keras")

# Save scaler (CRITICAL for deployment!)
import joblib
joblib.dump(scaler, 'health_scaler.pkl')
print("✅ Scaler saved: health_scaler.pkl")

# Save metadata
metadata = {
    'model_name': 'Diabetes Risk Predictor',
    'version': '1.0.0',
    'input_features': diabetes.feature_names,
    'target': 'diabetes_progression_binary',
    'threshold': 0.5,
    'metrics': {
        'accuracy': float(test_results[1]),
        'precision': float(test_results[2]),
        'recall': float(test_results[3]),
        'auc': float(test_results[4])
    },
    'inference_time_ms': float(avg_inference)
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata saved: model_metadata.json")
print()

print("="*70)
print("PRODUCTION ARTIFACTS READY FOR DEPLOYMENT")
print("="*70)
print("""
Files created:
1. health_predictor_production.keras - The trained model
2. health_scaler.pkl - Preprocessing scaler (MUST deploy with model!)
3. model_metadata.json - Model information and performance metrics

These 3 files contain everything needed to deploy the model to production.
Next step: Wrap in API and containerize!
""")