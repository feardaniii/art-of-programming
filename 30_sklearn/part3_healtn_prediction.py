import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

"""
⚡ HEALTH PREDICTION: Will Tomorrow Be a High-Energy Day?

You track: sleep hours, workout intensity, water intake, screen time before bed.
Question: Can ML predict tomorrow's energy level (1-10)?
Answer: Absolutely! Let's build it.
"""

print("="*70)
print("⚡ HEALTH ENERGY PREDICTION: ML for Your Wellbeing")
print("="*70)

# ===== STEP 1: Create Health Dataset =====
print("\n📊 STEP 1: Your Health Tracking Data")
print("-" * 70)

np.random.seed(42)
n_days = 180  # 6 months of tracking

# Generate realistic health data
sleep_hours = np.random.normal(7, 1.2, n_days)  # Average 7hrs, std 1.2
sleep_hours = np.clip(sleep_hours, 4, 10)  # Realistic range

workout_mins = np.random.choice([0, 30, 45, 60, 90], n_days, p=[0.2, 0.3, 0.2, 0.2, 0.1])
water_liters = np.random.normal(2.0, 0.5, n_days)
water_liters = np.clip(water_liters, 0.5, 4)

screen_before_bed = np.random.uniform(0, 180, n_days)  # minutes
stress_level = np.random.uniform(1, 10, n_days)
meal_quality = np.random.uniform(1, 10, n_days)  # 1-10 nutrition score

# Energy formula (what we want to learn!)
energy_level = (
    sleep_hours * 0.8               # Sleep is crucial
    + workout_mins * 0.02           # Exercise helps (diminishing returns)
    + water_liters * 0.5            # Hydration matters
    - screen_before_bed * 0.01      # Screen time hurts sleep quality
    - stress_level * 0.3            # Stress drains energy
    + meal_quality * 0.4            # Nutrition fuels you
    + np.random.normal(0, 0.5, n_days)  # Daily variation
)

# Normalize to 1-10 scale
energy_level = np.clip(energy_level, 1, 10)

# Create DataFrame
df = pd.DataFrame({
    'sleep_hours': sleep_hours,
    'workout_mins': workout_mins,
    'water_liters': water_liters,
    'screen_before_bed_mins': screen_before_bed,
    'stress_level': stress_level,
    'meal_quality_score': meal_quality,
    'energy_level': energy_level
})

print(f"Analyzing {len(df)} days of health data")
print("\nSample week:")
print(df.head(7).round(2))

print("\n📈 Your Health Averages:")
stats = df[['sleep_hours', 'workout_mins', 'water_liters', 'energy_level']].describe()
print(stats.loc[['mean', 'min', 'max']].round(2))

# ===== STEP 2: Feature Engineering =====
print("\n🔧 STEP 2: Feature Engineering")
print("-" * 70)

# Create derived features
df['sleep_quality'] = df['sleep_hours'] - (df['screen_before_bed_mins'] / 60)
df['exercise_hydration'] = df['workout_mins'] * df['water_liters'] / 100
df['wellness_score'] = df['meal_quality_score'] - df['stress_level']

print("Created 3 derived features:")
print("   • sleep_quality: sleep - (screen_time/60)")
print("   • exercise_hydration: combined workout & water effect")
print("   • wellness_score: nutrition - stress")

# ===== STEP 3: Train-Test Split =====
print("\n✂️  STEP 3: Preparing Data")
print("-" * 70)

feature_cols = ['sleep_hours', 'workout_mins', 'water_liters',
                'screen_before_bed_mins', 'stress_level', 'meal_quality_score',
                'sleep_quality', 'exercise_hydration', 'wellness_score']

X = df[feature_cols]
y = df['energy_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training days: {len(X_train)}")
print(f"Test days: {len(X_test)}")

# ===== STEP 4: Compare Models =====
print("\n🎓 STEP 4: Training & Comparing Models")
print("-" * 70)

# Model 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("\n1️⃣  Linear Regression:")
print(f"   R² Score: {r2_lr:.4f}")
print(f"   RMSE: {rmse_lr:.4f} energy points")

# Model 2: Random Forest (more complex, can capture non-linear patterns)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n2️⃣  Random Forest:")
print(f"   R² Score: {r2_rf:.4f}")
print(f"   RMSE: {rmse_rf:.4f} energy points")

if r2_rf > r2_lr:
    print("\n   ✅ Random Forest performs better!")
    print("   → Health patterns are non-linear (expected!)")
    best_model = model_rf
    best_pred = y_pred_rf
else:
    print("\n   ✅ Linear model is sufficient!")
    best_model = model_lr
    best_pred = y_pred_lr

# ===== STEP 5: Feature Importance =====
print("\n🔍 STEP 5: What Matters Most for Energy?")
print("-" * 70)

if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop factors affecting your energy:")
    for idx, row in importance.head(5).iterrows():
        print(f"   {row['feature']:25s}: {row['importance']*100:5.2f}%")
else:
    # For linear regression, use coefficients
    coefficients = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': best_model.coef_
    }).sort_values('coefficient', ascending=False, key=abs)

    print("\nTop factors affecting your energy:")
    for idx, row in coefficients.head(5).iterrows():
        impact = "↑ Increases" if row['coefficient'] > 0 else "↓ Decreases"
        print(f"   {row['feature']:25s}: {impact} energy")

# ===== STEP 6: Actionable Insights =====
print("\n💡 STEP 6: Actionable Recommendations")
print("-" * 70)

# Calculate optimal conditions from training data
optimal_conditions = X_train[y_train == y_train.max()].iloc[0]

print("\n🌟 Your BEST day profile (from historical data):")
print(f"   Sleep: {optimal_conditions['sleep_hours']:.1f} hours")
print(f"   Workout: {optimal_conditions['workout_mins']:.0f} minutes")
print(f"   Water: {optimal_conditions['water_liters']:.1f} liters")
print(f"   Screen before bed: {optimal_conditions['screen_before_bed_mins']:.0f} min")
print(f"   Stress level: {optimal_conditions['stress_level']:.1f}/10")
print(f"   Meal quality: {optimal_conditions['meal_quality_score']:.1f}/10")

# ===== STEP 7: Tomorrow's Prediction =====
print("\n🔮 STEP 7: Predict Tomorrow's Energy")
print("-" * 70)

print("\nToday's stats:")
today = {
    'sleep_hours': 6.5,
    'workout_mins': 30,
    'water_liters': 1.8,
    'screen_before_bed_mins': 90,
    'stress_level': 7,
    'meal_quality_score': 6
}

for key, value in today.items():
    print(f"   {key}: {value}")

# Calculate derived features
today['sleep_quality'] = today['sleep_hours'] - (today['screen_before_bed_mins'] / 60)
today['exercise_hydration'] = today['workout_mins'] * today['water_liters'] / 100
today['wellness_score'] = today['meal_quality_score'] - today['stress_level']

today_features = np.array([[today[col] for col in feature_cols]])
predicted_energy = best_model.predict(today_features)[0]

print(f"\n   Predicted energy tomorrow: {predicted_energy:.1f}/10")

if predicted_energy < 5:
    print("   ⚠️  LOW energy predicted!")
    print("\n   💪 Recommendations:")
    print("      • Aim for 8+ hours sleep tonight")
    print("      • Reduce screen time before bed (<30min)")
    print("      • Stay hydrated (2.5L+ water)")
    print("      • Light workout tomorrow (30min)")
elif predicted_energy < 7:
    print("   😐 MEDIUM energy predicted")
    print("\n   💪 To boost it:")
    print("      • Sleep 7-8 hours")
    print("      • 45min workout")
    print("      • Manage stress (meditation?)")
else:
    print("   ✅ HIGH energy predicted!")
    print("   → Great day for intense workout or important tasks!")

# ===== VISUALIZATION =====
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Top left: Actual vs Predicted
axes[0, 0].scatter(y_test, best_pred, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()],
               [y_test.min(), y_test.max()],
               'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual Energy Level', fontsize=11)
axes[0, 0].set_ylabel('Predicted Energy Level', fontsize=11)
axes[0, 0].set_title('Energy Prediction Accuracy', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Top right: Sleep vs Energy
axes[0, 1].scatter(df['sleep_hours'], df['energy_level'], alpha=0.5, s=30)
z = np.polyfit(df['sleep_hours'], df['energy_level'], 1)
p = np.poly1d(z)
axes[0, 1].plot(df['sleep_hours'].sort_values(),
               p(df['sleep_hours'].sort_values()),
               "r-", linewidth=2, label=f'Trend')
axes[0, 1].set_xlabel('Sleep Hours', fontsize=11)
axes[0, 1].set_ylabel('Energy Level', fontsize=11)
axes[0, 1].set_title('Sleep Impact on Energy', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Bottom left: Workout vs Energy
workout_groups = df.groupby('workout_mins')['energy_level'].mean()
axes[1, 0].bar(workout_groups.index, workout_groups.values, color='skyblue', alpha=0.7)
axes[1, 0].set_xlabel('Workout Duration (minutes)', fontsize=11)
axes[1, 0].set_ylabel('Average Energy Level', fontsize=11)
axes[1, 0].set_title('Exercise Impact on Energy', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Bottom right: Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importance_plot = importance.head(6)
    axes[1, 1].barh(importance_plot['feature'], importance_plot['importance'], color='coral')
    axes[1, 1].set_xlabel('Importance', fontsize=11)
    axes[1, 1].set_title('What Matters Most for Energy?', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('health_energy_prediction.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: health_energy_prediction.png")

print("\n" + "="*70)
print("🎉 YOU BUILT AN AI HEALTH COACH!")
print("="*70)
print("\nWhat you can do with this:")
print("   • Optimize sleep for maximum energy")
print("   • Plan workouts on predicted high-energy days")
print("   • Understand personal health patterns")
print("   • Make data-driven wellness decisions")
print("\n💡 This is REAL machine learning applied to YOUR life!")
print("   Not abstract exercises - actual tools you'll use daily.")
