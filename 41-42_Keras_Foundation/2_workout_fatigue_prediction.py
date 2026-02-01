import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# HEALTH DATA: Your Tracked Lifestyle Metrics
# The Problem: Should You Work Out Today?
# ==========================================

# Simulated 30 days of health tracking data
np.random.seed(42)
n_days = 30

health_data = pd.DataFrame({
    'sleep_hours': np.random.uniform(4, 9, n_days),
    'sleep_quality': np.random.randint(1, 11, n_days),  # 1-10 scale
    'calories_consumed': np.random.randint(1500, 3000, n_days),
    'protein_grams': np.random.randint(50, 200, n_days),
    'stress_level': np.random.randint(1, 11, n_days),  # 1-10 scale
    'previous_workout_intensity': np.random.randint(1, 11, n_days),  # 1-10
    'water_liters': np.random.uniform(1, 4, n_days),
})

# Target: Workout fatigue (1 = high fatigue, 0 = ready to train)
# Logic: Fatigue is high if sleep < 6, stress > 7, or previous intensity > 8
health_data['high_fatigue'] = (
    (health_data['sleep_hours'] < 6) |
    (health_data['stress_level'] > 7) |
    (health_data['previous_workout_intensity'] > 8)
).astype(int)

print("="*70)
print("HEALTH TRACKING DATA (First 10 Days)")
print("="*70)
print(health_data.head(10))
print()
print(f"Fatigue distribution: {health_data['high_fatigue'].value_counts().to_dict()}")
print()

# ==========================================
# MODEL: Predict Workout Fatigue
# ==========================================

# Separate features and target
X = health_data.drop('high_fatigue', axis=1)
y = health_data['high_fatigue']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Build neural network
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(7,)),  # 7 features
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary: fatigue yes/no
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("="*70)
print("TRAINING FATIGUE PREDICTION MODEL")
print("="*70)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.2,
    verbose=0  # Silent training
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()

# ==========================================
# PREDICTION: Should You Work Out Today?
# ==========================================

# Example: Today's metrics
today = pd.DataFrame({
    'sleep_hours': [5.5],
    'sleep_quality': [4],
    'calories_consumed': [1800],
    'protein_grams': [90],
    'stress_level': [8],
    'previous_workout_intensity': [9],
    'water_liters': [2.0]
})

today_scaled = scaler.transform(today)
fatigue_prob = model.predict(today_scaled, verbose=0)[0][0]

print("="*70)
print("TODAY'S WORKOUT RECOMMENDATION")
print("="*70)
print("Your Metrics:")
for col in today.columns:
    print(f"  {col}: {today[col].values[0]}")
print()
print(f"Fatigue Probability: {fatigue_prob:.2%}")
print()

if fatigue_prob > 0.6:
    print("🛑 HIGH FATIGUE DETECTED")
    print("   Recommendation: Rest day or light activity")
    print("   Focus on: Sleep, hydration, stress management")
elif fatigue_prob > 0.3:
    print("⚠️  MODERATE FATIGUE")
    print("   Recommendation: Light to moderate workout")
    print("   Avoid: High-intensity training")
else:
    print("✅ READY TO TRAIN")
    print("   Recommendation: Full intensity workout")
    print("   Your body is recovered and ready!")
print()

# ==========================================
# INSIGHTS: What Drives Fatigue?
# ==========================================

print("="*70)
print("KEY INSIGHTS FROM YOUR DATA")
print("="*70)

# Feature importance (simplified: look at correlation)
correlations = health_data.corr()['high_fatigue'].sort_values(ascending=False)
print("\nFactors most correlated with fatigue:")
for feature, corr in correlations.items():
    if feature != 'high_fatigue':
        print(f"  {feature}: {corr:+.3f}")

# Save model
model.save('health_fatigue_predictor.h5')
print("\nModel saved as 'health_fatigue_predictor.h5'")
print("You can now use this model daily to optimize your training!")