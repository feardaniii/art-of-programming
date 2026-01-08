import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

"""
🏠 REAL-WORLD PROJECT: Should I Buy This Apartment?

You're apartment hunting. The realtor shows you a 75m² place for 180,000€.
Is that a good deal? Let's build a model that predicts fair prices!
"""

print("="*70)
print("🏠 HOUSE PRICE PREDICTION: Your First ML Model")
print("="*70)

# ===== STEP 1: Create Realistic Dataset =====
print("\n📊 STEP 1: Loading Housing Data")
print("-" * 70)

# Real housing data from your city (simulated but realistic)
np.random.seed(42)
n_houses = 200

# Generate realistic features
square_meters = np.random.uniform(40, 150, n_houses)
distance_center = np.random.uniform(1, 20, n_houses)  # km from city center
floor_level = np.random.randint(0, 10, n_houses)
age_years = np.random.uniform(0, 50, n_houses)
has_parking = np.random.choice([0, 1], n_houses, p=[0.3, 0.7])

# Price formula (what we're trying to learn!)
# Base: 2500€/m² + location penalty - age penalty + parking bonus
base_price_per_sqm = 2500
price = (
    base_price_per_sqm * square_meters
    - distance_center * 3000  # Further from center = cheaper
    + floor_level * 2000       # Higher floor = slightly more expensive
    - age_years * 500          # Older = cheaper
    + has_parking * 15000      # Parking adds value
    + np.random.normal(0, 15000, n_houses)  # Random variation
)

# Create DataFrame
df = pd.DataFrame({
    'square_meters': square_meters,
    'distance_km': distance_center,
    'floor': floor_level,
    'age_years': age_years,
    'has_parking': has_parking,
    'price': price
})

print(f"Loaded {len(df)} apartments from the market")
print("\nFirst 5 listings:")
print(df.head())

print("\n📈 Quick Statistics:")
print(df.describe()[['square_meters', 'price']].round(2))

# ===== STEP 2: Split Data (The Golden Rule) =====
print("\n✂️  STEP 2: Train-Test Split")
print("-" * 70)

# Separate features (X) from target (y)
X = df[['square_meters', 'distance_km', 'floor', 'age_years', 'has_parking']]
y = df['price']

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} apartments")
print(f"Test set: {len(X_test)} apartments")
print("\n💡 WHY SPLIT?")
print("   Training data: Teach the model")
print("   Test data: Evaluate on unseen data (simulates real world)")
print("   → Never test on training data! That's cheating!")

# ===== STEP 3: Train the Model =====
print("\n🎓 STEP 3: Training Linear Regression Model")
print("-" * 70)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained!")
print("\nWhat did it learn?")
print(f"   Coefficient for square_meters: {model.coef_[0]:.2f}€/m²")
print(f"   → Each extra m² adds {model.coef_[0]:.2f}€ to price")
print(f"\n   Coefficient for distance_km: {model.coef_[1]:.2f}€/km")
print(f"   → Each km further from center subtracts {abs(model.coef_[1]):.2f}€")
print(f"\n   Coefficient for parking: {model.coef_[4]:.2f}€")
print(f"   → Parking adds ~{model.coef_[4]/1000:.1f}k€ to value")

print(f"\n   Intercept (base price): {model.intercept_:.2f}€")

# ===== STEP 4: Make Predictions =====
print("\n🔮 STEP 4: Making Predictions")
print("-" * 70)

y_pred = model.predict(X_test)

# Show some examples
print("\nReal vs Predicted (first 5 test cases):")
print("-" * 50)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100

    print(f"\nApartment {i+1}:")
    print(f"   {X_test.iloc[i]['square_meters']:.0f}m², "
          f"{X_test.iloc[i]['distance_km']:.1f}km from center, "
          f"Floor {X_test.iloc[i]['floor']}, "
          f"{X_test.iloc[i]['age_years']:.0f} years old")
    print(f"   Actual price:    {actual:>10,.0f}€")
    print(f"   Predicted price: {predicted:>10,.0f}€")
    print(f"   Error: {error:,.0f}€ ({error_pct:.1f}%)")

# ===== STEP 5: Evaluate Model =====
print("\n📊 STEP 5: Model Evaluation")
print("-" * 70)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📏 Evaluation Metrics:")
print(f"   RMSE (Root Mean Squared Error): {rmse:,.0f}€")
print(f"      → Average prediction error: ±{rmse:,.0f}€")
print(f"\n   MAE (Mean Absolute Error): {mae:,.0f}€")
print(f"      → Typical error: {mae:,.0f}€")
print(f"\n   R² Score: {r2:.4f}")
print(f"      → Model explains {r2*100:.2f}% of price variation")

print("\n💡 WHAT DO THESE MEAN?")
print(f"   RMSE = {rmse/1000:.1f}k€: On average, predictions are off by this much")
print(f"   R² = {r2:.2f}: Close to 1.0 is excellent (1.0 = perfect predictions)")
print(f"   MAE = {mae/1000:.1f}k€: Typical error in real-world terms")

# ===== STEP 6: The Real Test =====
print("\n🎯 STEP 6: The Real-World Test")
print("-" * 70)
print("\nYou found an apartment:")
print("   • 75m², 3km from center, Floor 4, 10 years old, with parking")
print("   • Asking price: 180,000€")
print("\nIs it a good deal?")

new_apartment = np.array([[75, 3, 4, 10, 1]])  # Features as array
predicted_price = model.predict(new_apartment)[0]

print(f"\n   Model's fair price: {predicted_price:,.0f}€")
print(f"   Asking price: 180,000€")
difference = predicted_price - 180000
if difference > 0:
    print(f"   → {difference:,.0f}€ BELOW fair value! ✅ Good deal!")
else:
    print(f"   → {abs(difference):,.0f}€ ABOVE fair value! ⚠️ Overpriced!")

# ===== VISUALIZATION =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, s=50)
axes[0].plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price (€)', fontsize=12)
axes[0].set_ylabel('Predicted Price (€)', fontsize=12)
axes[0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Residuals (errors)
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, s=50)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price (€)', fontsize=12)
axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_title('Residual Plot: Are Errors Random?', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_prediction.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved: house_price_prediction.png")

print("\n" + "="*70)
print("🎉 CONGRATULATIONS! You built a working ML model!")
print("="*70)
print("\nWhat you just did:")
print("   1. Loaded and explored real-world data")
print("   2. Split data (train/test - the golden rule)")
print("   3. Trained a Linear Regression model")
print("   4. Made predictions on new data")
print("   5. Evaluated performance (RMSE, R², MAE)")
print("   6. Applied it to a real decision (buy apartment?)")
print("\n💡 This exact process scales to:")
print("   • Predicting stock prices")
print("   • Diagnosing diseases")
print("   • Recommending products")
print("   • Optimizing ad spend")
print("\n🚀 You're not just learning sklearn - you're learning to make")
print("   data-driven decisions that could save you thousands of euros!")
