import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt


print("=" * 70)
print("Random Forest: Predictia temperaturii unei stele")
print("=" * 70)

# ===== STEP 1: Incarcare date =====
print("\n📊 PASUL 1: Incarcarea datelor")
print("-" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "star_dataset.csv")

df = pd.read_csv(file_path)
print("Date incarcate cu succes!")
print("Dimensiune initiala:", df.shape)
print(df.head())

# ===== STEP 2: Curatire date =====
print("\n🧹 PASUL 2: Curatare minimala")
print("-" * 70)

required_cols = [
    "Distance (ly)",
    "Luminosity (L/Lo)",
    "Radius (R/Ro)",
    "Temperature (K)",
    "Spectral Class",
]

df = df.dropna(subset=required_cols).copy()

# Eliminam valori ne-fizice / invalide
df = df[df["Distance (ly)"] > 0]
df = df[df["Radius (R/Ro)"] > 0]
df = df[df["Luminosity (L/Lo)"] > 0]
df = df[df["Temperature (K)"] > 0]

print("Dimensiune dupa curatare:", df.shape)

# ===== STEP 3: Definim X (intrarile) si y (target) =====
print("\n📌 PASUL 3: Definirea lui X si y")
print("-" * 70)

X = df[["Distance (ly)", "Luminosity (L/Lo)", "Radius (R/Ro)", "Spectral Class"]]
y = df["Temperature (K)"]

print("X:", X.shape, " y:", y.shape)

# ===== STEP 4: Splitarea datelor =====
print("\n🔀 PASUL 4: Splitarea datelor")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("y_train:", y_train.shape, " y_test:", y_test.shape)

# ===== STEP 5: Preprocessing + Model =====
print("\n🧩 PASUL 5: Preprocesare + Random Forest")
print("-" * 70)

numeric_features = ["Distance (ly)", "Luminosity (L/Lo)", "Radius (R/Ro)"]
categorical_features = ["Spectral Class"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", rf)
    ]
)

# ===== STEP 6: Antrenare =====
print("\n🎓 PASUL 6: Antrenare")
print("-" * 70)

pipeline.fit(X_train, y_train)
print("Model antrenat cu succes!")

# ===== STEP 7: Evaluare =====
print("\n📈 PASUL 7: Evaluare")
print("-" * 70)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  = {mae:.2f} K")
print(f"RMSE = {rmse:.2f} K")
print(f"R²   = {r2:.3f}")

# ===== STEP 8: Exemple =====
print("\n🔎 Exemple (Real vs Prezis) - primele 10 din setul de test")
print("-" * 70)

comparison = pd.DataFrame({
    "Real_Temp(K)": y_test.values[:10],
    "Pred_Temp(K)": y_pred[:10],
})
comparison["Abs_Error(K)"] = (comparison["Real_Temp(K)"] - comparison["Pred_Temp(K)"]).abs()
print(comparison)

# ===== (Optional) Interpretare =====
print("\n⭐ PAS OPTIONAL: Feature importance (interpretare)")
print("-" * 70)

# Extragerea caracteristicilor (numeric + one-hot)
ohe = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names

importances = pipeline.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.close("all")

# 📊 Graficul 1: Temperatură reală vs prezisă

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Temperatura reală (K)")
plt.ylabel("Temperatura prezisă (K)")
plt.title("Real vs Prezis – Random Forest")
plt.tight_layout()

# 📊 Graficul 2: Distribuția erorilor (histogramă)

errors = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.hist(errors, bins=30)
plt.xlabel("Eroare (K)")
plt.ylabel("Număr de stele")
plt.title("Distribuția erorilor – Random Forest")
plt.tight_layout()

# 📊 Graficul 3: Feature importance

plt.figure(figsize=(8, 5))
fi.head(10).plot(kind="barh", ax=plt.gca())
plt.xlabel("Importanță")
plt.title("Importanța caracteristicilor – Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 15 feature-uri importante:")
print(fi.head(15))


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

results = pd.DataFrame({
    "Real_Temp": y_test.values,
    "Pred_Temp": y_pred,
})

results["Error"] = results["Real_Temp"] - results["Pred_Temp"]
results["Abs_Error"] = results["Error"].abs()

results_path = os.path.join(out_dir, "results_rf.csv")
results.to_csv(results_path, index=False)

print(f"Rezultate salvate in: {results_path}")

# Introducere date de la tastatura
print("\n⭐ Predictie pentru o stea noua (input de la tastatura)")
print("-" * 70)

try:
    dist_ly = float(input("Distanta (ly): "))
    lum = float(input("Luminosity (L/Lo): "))
    rad = float(input("Radius (R/Ro): "))
    sp_class = input("Spectral Class (ex: A7V, B2III, M2Iab): ").strip()

    new_star = pd.DataFrame([{
        "Distance (ly)": dist_ly,
        "Luminosity (L/Lo)": lum,
        "Radius (R/Ro)": rad,
        "Spectral Class": sp_class
    }])

    pred_temp = pipeline.predict(new_star)[0]
    print(f"\n✅ Temperatura estimata: {pred_temp:.2f} K")

except ValueError:
    print("❌ Eroare: te rog introdu valori numerice valide pentru distanta/luminozitate/raza.")
except Exception as e:
    print(f"❌ Eroare neasteptata: {e}")
