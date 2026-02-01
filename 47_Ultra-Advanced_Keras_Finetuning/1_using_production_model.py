# ==========================================
# SESSION 47 PART 2: USING YOUR PRODUCTION MODEL
# ==========================================
# This is the DEPLOYMENT side - what happens AFTER training
# You ship the .keras and .pkl files, and THIS script runs in production

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import json
import joblib
import os

# ==========================================
# SETUP: Create artifacts if they don't exist
# (So this script can run standalone for learning)
# ==========================================
if not os.path.exists('health_predictor_production.keras'):
    print("📦 Creating model artifacts for demo...")
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Quick model creation
    diabetes = load_diabetes()
    X, y = diabetes.data, (diabetes.target > 140).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, verbose=0)

    model.save('health_predictor_production.keras')
    joblib.dump(scaler, 'health_scaler.pkl')
    print("✅ Artifacts created!\n")

print("=" * 70)
print("PART 2: LOADING & USING YOUR PRODUCTION MODEL")
print("=" * 70)
print()
print("🎯 THE BIG PICTURE:")
print("-" * 70)
print("""
What you built in Part 1:
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING PHASE (runs once, maybe weekly/monthly)              │
│                                                                 │
│  Raw Data → Preprocessing → Model Training → Save Artifacts    │
│                                                                 │
│  Output: 3 files that contain EVERYTHING needed                │
│    1. health_predictor_production.keras  (the trained brain)   │
│    2. health_scaler.pkl                  (data transformer)    │
│    3. model_metadata.json                (documentation)       │
└─────────────────────────────────────────────────────────────────┘

What this script does (Part 2):
┌─────────────────────────────────────────────────────────────────┐
│  INFERENCE PHASE (runs thousands of times per day)             │
│                                                                 │
│  Load Artifacts → Receive Patient Data → Predict → Return Risk │
│                                                                 │
│  This is what runs in the clinic's computer!                   │
└─────────────────────────────────────────────────────────────────┘
""")
input("Press Enter to continue...")
print()

# ==========================================
# STEP 1: LOAD THE PRODUCTION ARTIFACTS
# ==========================================
print("=" * 70)
print("STEP 1: LOADING PRODUCTION ARTIFACTS")
print("=" * 70)
print()

# Load the trained model
print("Loading model...")
model = keras.models.load_model('health_predictor_production.keras')
print("✅ Model loaded!")
print()

# Load the scaler (CRITICAL!)
print("Loading scaler...")
scaler = joblib.load('health_scaler.pkl')
print("✅ Scaler loaded!")
print()

print("💡 WHY BOTH FILES ARE ESSENTIAL:")
print("-" * 70)
print("""
The model expects data in a SPECIFIC format - the same format it saw 
during training. The scaler transforms raw patient data into that format.

WITHOUT the scaler:
  Raw data: [150, 35, 0.8, ...]  → Model: "??? These numbers make no sense!"

WITH the scaler:
  Raw data: [150, 35, 0.8, ...] → Scaler → [-0.5, 1.2, 0.3, ...] → Model: "Ah! 73% risk"

This is why you MUST deploy both files together!
""")
input("Press Enter to continue...")
print()

# ==========================================
# STEP 2: UNDERSTAND THE MODEL'S INPUT
# ==========================================
print("=" * 70)
print("STEP 2: UNDERSTANDING MODEL INPUT")
print("=" * 70)
print()

# The diabetes dataset features
FEATURE_NAMES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
FEATURE_DESCRIPTIONS = {
    'age': 'Age (normalized)',
    'sex': 'Sex (normalized)',
    'bmi': 'Body mass index',
    'bp': 'Average blood pressure',
    's1': 'Total serum cholesterol (tc)',
    's2': 'Low-density lipoproteins (ldl)',
    's3': 'High-density lipoproteins (hdl)',
    's4': 'Total cholesterol / HDL (tch)',
    's5': 'Log of serum triglycerides (ltg)',
    's6': 'Blood sugar level (glu)'
}

print("This model expects 10 patient measurements:")
print("-" * 50)
for i, name in enumerate(FEATURE_NAMES):
    print(f"  {i + 1}. {name:5} - {FEATURE_DESCRIPTIONS[name]}")
print()

print(f"Model input shape: {model.input_shape}")
print(f"Model output: Single probability (0.0 to 1.0)")
print()
input("Press Enter to continue...")
print()

# ==========================================
# STEP 3: MAKE A SINGLE PREDICTION
# ==========================================
print("=" * 70)
print("STEP 3: MAKING A SINGLE PREDICTION")
print("=" * 70)
print()

# Simulate a new patient (using realistic-ish values)
# Note: These are normalized values from the original dataset
new_patient_raw = np.array([[
    0.05,  # age - slightly above average
    0.05,  # sex
    0.06,  # bmi - slightly high BMI
    0.02,  # bp - slightly elevated blood pressure
    0.03,  # s1 - cholesterol
    0.02,  # s2 - LDL
    -0.03,  # s3 - HDL (negative = lower, which is worse)
    0.05,  # s4 - ratio
    0.04,  # s5 - triglycerides
    0.05  # s6 - blood sugar
]])

print("📋 NEW PATIENT DATA (raw):")
print("-" * 50)
for name, value in zip(FEATURE_NAMES, new_patient_raw[0]):
    print(f"  {name:5}: {value:+.3f}")
print()

# Step 3a: Scale the data (transform to what model expects)
print("Step 3a: Apply scaler transformation...")
patient_scaled = scaler.transform(new_patient_raw)
print(f"  Raw values:    {new_patient_raw[0][:3]}...")
print(f"  Scaled values: {patient_scaled[0][:3]}...")
print()

# Step 3b: Make prediction
print("Step 3b: Model prediction...")
risk_probability = model.predict(patient_scaled, verbose=0)[0][0]
print()

# Step 3c: Interpret result
print("=" * 70)
print("📊 PREDICTION RESULT")
print("=" * 70)
risk_percentage = risk_probability * 100
print(f"  Risk Probability: {risk_percentage:.1f}%")
print()

# Clinical interpretation
if risk_probability < 0.3:
    risk_level = "LOW RISK"
    color = "🟢"
    recommendation = "Continue routine monitoring. Maintain healthy lifestyle."
elif risk_probability < 0.6:
    risk_level = "MODERATE RISK"
    color = "🟡"
    recommendation = "Schedule follow-up in 3 months. Consider lifestyle modifications."
else:
    risk_level = "HIGH RISK"
    color = "🔴"
    recommendation = "Immediate clinical consultation recommended. Further testing needed."

print(f"  {color} Classification: {risk_level}")
print(f"  📝 Recommendation: {recommendation}")
print()
input("Press Enter to continue...")
print()

# ==========================================
# STEP 4: BATCH PREDICTIONS (MULTIPLE PATIENTS)
# ==========================================
print("=" * 70)
print("STEP 4: BATCH PREDICTIONS")
print("=" * 70)
print()

print("In production, you often predict for many patients at once.")
print("This is MORE EFFICIENT than one-by-one predictions.")
print()

# Simulate 5 patients
batch_patients = np.array([
    [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01],  # Low risk profile
    [0.05, 0.05, 0.08, 0.06, 0.05, 0.04, -0.04, 0.06, 0.07, 0.06],  # High risk profile
    [-0.02, -0.01, -0.02, -0.01, -0.01, -0.01, 0.03, -0.01, -0.02, -0.01],  # Very low
    [0.03, 0.02, 0.04, 0.03, 0.02, 0.02, 0.00, 0.03, 0.03, 0.03],  # Moderate
    [0.07, 0.06, 0.09, 0.07, 0.06, 0.05, -0.05, 0.07, 0.08, 0.07],  # Very high
])

# Process entire batch
batch_scaled = scaler.transform(batch_patients)
batch_predictions = model.predict(batch_scaled, verbose=0)

print("📋 BATCH RESULTS:")
print("-" * 60)
print(f"{'Patient':<10} {'Risk %':<10} {'Level':<15} {'Action'}")
print("-" * 60)

for i, prob in enumerate(batch_predictions):
    risk_pct = prob[0] * 100
    if prob[0] < 0.3:
        level, action = "🟢 Low", "Routine"
    elif prob[0] < 0.6:
        level, action = "🟡 Moderate", "Follow-up"
    else:
        level, action = "🔴 High", "Urgent"
    print(f"Patient {i + 1:<3} {risk_pct:>6.1f}%    {level:<15} {action}")

print("-" * 60)
print()
input("Press Enter to continue...")
print()

# ==========================================
# STEP 5: THE COMPLETE INFERENCE FUNCTION
# ==========================================
print("=" * 70)
print("STEP 5: PRODUCTION-READY INFERENCE FUNCTION")
print("=" * 70)
print()

print("Here's what you'd actually deploy in production:")
print()


# This is the function you'd put in your API/service
def predict_diabetes_risk(patient_data, model, scaler, threshold=0.5):
    """
    Production inference function for diabetes risk prediction.

    Args:
        patient_data: numpy array of shape (n_patients, 10) or (10,)
                     Features: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
        model: loaded Keras model
        scaler: loaded StandardScaler
        threshold: classification threshold (default 0.5)

    Returns:
        dict with predictions and metadata
    """
    # Ensure 2D array
    if patient_data.ndim == 1:
        patient_data = patient_data.reshape(1, -1)

    # Validate input
    if patient_data.shape[1] != 10:
        raise ValueError(f"Expected 10 features, got {patient_data.shape[1]}")

    # Scale the data
    scaled_data = scaler.transform(patient_data)

    # Predict
    probabilities = model.predict(scaled_data, verbose=0).flatten()

    # Classify
    classifications = ['HIGH_RISK' if p >= threshold else 'LOW_RISK'
                       for p in probabilities]

    # Build response
    results = []
    for i, (prob, classification) in enumerate(zip(probabilities, classifications)):
        results.append({
            'patient_index': i,
            'risk_probability': float(prob),
            'risk_percentage': float(prob * 100),
            'classification': classification,
            'requires_followup': bool(prob >= 0.3)  # Convert numpy bool to Python bool
        })

    return {
        'predictions': results,
        'model_version': '1.0.0',
        'threshold_used': threshold,
        'n_patients': len(results)
    }


# Demo the function
print("def predict_diabetes_risk(patient_data, model, scaler, threshold=0.5):")
print('    """Production inference function"""')
print("    # ... (see full code above)")
print()

# Test it
test_result = predict_diabetes_risk(batch_patients[1], model, scaler)
print("Example call:")
print("  predict_diabetes_risk(patient_data, model, scaler)")
print()
print("Example response:")
print(json.dumps(test_result, indent=2))
print()
input("Press Enter to continue...")
print()

# ==========================================
# STEP 6: INFERENCE TIMING ANALYSIS
# ==========================================
print("=" * 70)
print("STEP 6: PERFORMANCE BENCHMARKS")
print("=" * 70)
print()

import time

# Single prediction timing
times_single = []
for _ in range(50):
    start = time.time()
    _ = predict_diabetes_risk(new_patient_raw, model, scaler)
    times_single.append((time.time() - start) * 1000)

# Batch prediction timing (5 patients)
times_batch = []
for _ in range(50):
    start = time.time()
    _ = predict_diabetes_risk(batch_patients, model, scaler)
    times_batch.append((time.time() - start) * 1000)

print("⚡ PERFORMANCE RESULTS:")
print("-" * 50)
print(f"Single patient prediction:")
print(f"  Average: {np.mean(times_single):.2f} ms")
print(f"  95th percentile: {np.percentile(times_single, 95):.2f} ms")
print()
print(f"Batch prediction (5 patients):")
print(f"  Average: {np.mean(times_batch):.2f} ms")
print(f"  Per patient: {np.mean(times_batch) / 5:.2f} ms")
print()
print("💡 Notice: Batch is more efficient per-patient!")
print("   This is why APIs often support batch endpoints.")
print()
input("Press Enter to continue...")
print()

# ==========================================
# SUMMARY: WHAT YOU LEARNED
# ==========================================
print("=" * 70)
print("🎓 SESSION 47 SUMMARY: THE COMPLETE ML PIPELINE")
print("=" * 70)
print()

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    THE ML PRODUCTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TRAINING (Part 1 - runs periodically)                             │
│  ─────────────────────────────────────                             │
│  1. Load & preprocess data                                         │
│  2. Build model architecture                                        │
│  3. Compile with optimizer & metrics                               │
│  4. Train with callbacks (early stopping, checkpoints)             │
│  5. Evaluate on test set                                           │
│  6. Save artifacts: .keras + .pkl + metadata                       │
│                                                                     │
│                         ↓ Ship these files                         │
│                                                                     │
│  INFERENCE (Part 2 - runs constantly in production)                │
│  ──────────────────────────────────────────────────                │
│  1. Load model and scaler                                          │
│  2. Receive patient data                                           │
│  3. Scale data with saved scaler                                   │
│  4. model.predict() → probability                                  │
│  5. Apply threshold → classification                               │
│  6. Return structured response                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

KEY CONCEPTS MASTERED:
──────────────────────
✅ Model serialization (.keras format)
✅ Preprocessing artifact preservation (.pkl scaler)
✅ Inference vs training mode
✅ Batch predictions for efficiency
✅ Performance benchmarking
✅ Production-ready function design
✅ Structured API responses

NEXT STEPS (future sessions):
────────────────────────────
📦 Wrap in FastAPI for REST endpoint
🐳 Containerize with Docker
☁️ Deploy to cloud (GKE, since you're using that!)
📊 Add monitoring and logging
🔄 Set up model retraining pipeline
""")

print()
print("=" * 70)
print("🚀 You now understand the FULL lifecycle of a production ML model!")
print("=" * 70)