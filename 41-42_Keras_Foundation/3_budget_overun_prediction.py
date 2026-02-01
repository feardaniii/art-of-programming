import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# FINANCE DATA: Your Spending Patterns
# The Problem: Will You Go Over Budget This Month?
# ==========================================

# Simulated 6 months of spending tracking (weekly data)
np.random.seed(42)
n_weeks = 24  # 6 months

finance_data = pd.DataFrame({
    'week_of_month': np.tile([1, 2, 3, 4], n_weeks // 4),
    'groceries_spent': np.random.uniform(50, 200, n_weeks),
    'restaurants_spent': np.random.uniform(0, 150, n_weeks),
    'entertainment_spent': np.random.uniform(0, 100, n_weeks),
    'transport_spent': np.random.uniform(20, 80, n_weeks),
    'shopping_spent': np.random.uniform(0, 200, n_weeks),
    'avg_daily_transactions': np.random.randint(1, 10, n_weeks),
    'high_value_purchases': np.random.randint(0, 4, n_weeks),  # >$100
})

# Monthly budget: $1500
monthly_budget = 1500

# Target: Will total monthly spend exceed budget?
# Estimate monthly total from weekly spending
finance_data['estimated_monthly_total'] = (
    finance_data['groceries_spent'] +
    finance_data['restaurants_spent'] +
    finance_data['entertainment_spent'] +
    finance_data['transport_spent'] +
    finance_data['shopping_spent']
) * 4  # Multiply weekly by 4 for monthly estimate

finance_data['will_exceed_budget'] = (
    finance_data['estimated_monthly_total'] > monthly_budget
).astype(int)

print("="*70)
print("SPENDING TRACKING DATA (First 10 Weeks)")
print("="*70)
print(finance_data.head(10))
print()
print(f"Budget overrun distribution: {finance_data['will_exceed_budget'].value_counts().to_dict()}")
print()

# ==========================================
# MODEL: Predict Budget Overrun Risk
# ==========================================

# Features: all except target and estimated total
X = finance_data.drop(['will_exceed_budget', 'estimated_monthly_total'], axis=1)
y = finance_data['will_exceed_budget']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Build model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(7,)),
    layers.Dropout(0.2),  # Regularization to prevent overfitting
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("="*70)
print("TRAINING BUDGET OVERRUN PREDICTOR")
print("="*70)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.2,
    verbose=0
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print()

# ==========================================
# PREDICTION: This Week's Spending Alert
# ==========================================

# Current week data (week 2 of month)
this_week = pd.DataFrame({
    'week_of_month': [2],
    'groceries_spent': [180],
    'restaurants_spent': [120],
    'entertainment_spent': [80],
    'transport_spent': [60],
    'shopping_spent': [150],
    'avg_daily_transactions': [7],
    'high_value_purchases': [2]
})

this_week_scaled = scaler.transform(this_week)
overrun_prob = model.predict(this_week_scaled, verbose=0)[0][0]

print("="*70)
print("THIS WEEK'S BUDGET ALERT")
print("="*70)
print("Current Week Spending:")
for col in this_week.columns:
    if col != 'week_of_month':
        print(f"  {col}: {this_week[col].values[0]:.2f}")
print()
weekly_total = this_week.drop('week_of_month', axis=1).sum(axis=1).values[0]
print(f"Total this week: {weekly_total:.2f}")
print(f"Projected monthly (×4): {weekly_total * 4:.2f}")
print(f"Monthly budget: {monthly_budget:.2f}")
print()
print(f"Budget Overrun Risk: {overrun_prob:.2%}")
print()

if overrun_prob > 0.7:
    print("🚨 HIGH RISK OF BUDGET OVERRUN")
    print("   Recommendation: Immediate spending freeze on non-essentials")
    print("   Action: Review and cut discretionary spending")
    print("   Focus: Groceries only, no restaurants/entertainment")
elif overrun_prob > 0.4:
    print("⚠️  MODERATE RISK")
    print("   Recommendation: Reduce discretionary spending")
    print("   Action: Limit restaurants and entertainment")
    print("   Track daily to avoid creeping costs")
else:
    print("✅ ON TRACK")
    print("   Status: Spending within healthy limits")
    print("   Continue monitoring, maintain discipline")
print()

# ==========================================
# INSIGHTS: Spending Pattern Analysis
# ==========================================

print("="*70)
print("SPENDING PATTERN INSIGHTS")
print("="*70)

# Correlations with budget overrun
correlations = finance_data.corr()['will_exceed_budget'].sort_values(ascending=False)
print("\nFactors most correlated with budget overruns:")
for feature, corr in correlations.items():
    if feature not in ['will_exceed_budget', 'estimated_monthly_total']:
        print(f"  {feature}: {corr:+.3f}")

print("\nKey Takeaway:")
print("  The categories with highest positive correlation are your")
print("  biggest budget risks. Focus cutbacks there first.")
print()

# Save model
model.save('finance_budget_predictor.h5')
print("Model saved as 'finance_budget_predictor.h5'")