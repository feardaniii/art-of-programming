import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


"""
Vreau să estimez temperatura unei stele pe baza luminozitatii și razelor.
Am folosit date publice din misiunea Gaia a ESA, respectiv din NASA Exoplanet Archive. 
Sunt surse standard în astronomie, utilizate frecvent în cercetare.Datele au fost descarcate de pe Kaggle.
"""

print("="*70)
print("Primul ML Model: Determinarea temperaturii unei stele")
print("="*70)

# ===== STEP 1: Incarcam dataset-ul =====
print("\n📊 PASUL 1: Incarcarea datelor")
print("-" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "star_dataset.csv")

try:
    df = pd.read_csv(file_path)
    print("Date incarcate cu succes!")
    print(df.head())
except FileNotFoundError:
    print(f"❌ EROARE: Fisierul '{file_path}' nu a fost gasit!")
    print("   Verifica daca fisierul este in acelasi folder de unde rulezi scriptul.")
    df = None
except Exception as e:
    print(f"❌ EROARE: A aparut o problema la citirea fisierului: {e}")
    df = None

print("Dimensiunea dataset-ului:", df.shape) #verificam daca a incarcat tot fisierul

# ==== STEP 2: Definim pe X (intrari) si y (target) 

if df is None:
    raise SystemExit("Oprire: nu s-au incarcat datele.")
print("\n📌 PASUL 2: Definirea lui X (intrari) si y (target)")
print("-" * 70)
print("Coloane disponibile:", list(df.columns))


# facem o curatire in date
df = df.copy()
df = df.dropna(subset=["Temperature (K)", "Distance (ly)", "Luminosity (L/Lo)", "Radius (R/Ro)", "Spectral Class"])
df = df[df["Luminosity (L/Lo)"] > 0]
df = df[df["Radius (R/Ro)"] > 0]
df = df[df["Distance (ly)"] > 0]

# definim pe X si y
y = df["Temperature (K)"]

X = df[[
    "Distance (ly)",
    "Luminosity (L/Lo)",
    "Radius (R/Ro)",
    "Spectral Class"
]]

# verificare inainte de pasul 3 - splitare
print("Dimensiune X:", X.shape)
print("Dimensiune y:", y.shape)
print("Exemplu X (primele 3 randuri):")
print(X.head(3))
print("Exemplu y (primele 3 valori):")
print(y.head(3).to_list())

#===== STEP 3: Splitarea datelor (train/test)
print("\n🔀 PASUL 3: Splitarea datelor (train / test)")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # invat pe 80% din stele si testez pe 20%
    random_state=42 # impartirea va fi aceeasi de fiecare data cand se ruleaza scriptul. Permite sa compari modelele, sa refaci scriptul, etc.
)
print("Dimensiuni dupa split:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


print("\n🧩 PASUL 3.5: Definirea preprocesarii (numeric + categoric)")
print("-" * 70)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_features = [
    "Distance (ly)",
    "Luminosity (L/Lo)",
    "Radius (R/Ro)"
]

categorical_features = [
    "Spectral Class"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

print("Preprocesor definit cu succes!")


#===== STEP 4: Antrenarea modelului cu Pipeline

print("\n🎓 STEP 4: Training Linear Regression Model (with Pipeline)")
print("-" * 70)

# Definim modelul
model = LinearRegression()

# Construim pipeline-ul complet
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

# Antrenarea propriu-zisa
pipeline.fit(X_train, y_train)

print("Model antrenat cu succes!")


#===== STEP 5: Evaluarea modelului

print("\n📈 STEP 5: Evaluarea modelului")
print("-" * 70)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  = {mae:.2f} K")
print(f"RMSE = {rmse:.2f} K")
print(f"R²   = {r2:.3f}")

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
plt.title("Real vs Prezis – Linear Regression")
plt.tight_layout()

# 📊 Graficul 2: Distribuția erorilor (histogramă)

errors = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.hist(errors, bins=30)
plt.xlabel("Eroare (K)")
plt.ylabel("Număr de stele")
plt.title("Distribuția erorilor – Linear Regression")
plt.tight_layout()
plt.show()



#===== STEP 5.1: Exemple

print("\n🔎 Exemple (Real vs Prezis) - primele 10 din setul de test")
print("-" * 70)

comparison = pd.DataFrame({
    "Real_Temp(K)": y_test.values[:10],
    "Pred_Temp(K)": y_pred[:10]
})

comparison["Abs_Error(K)"] = (comparison["Real_Temp(K)"] - comparison["Pred_Temp(K)"]).abs()
print(comparison)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

results = pd.DataFrame({
    "Real_Temp": y_test.values,
    "Pred_Temp": y_pred,
})

results["Error"] = results["Real_Temp"] - results["Pred_Temp"]
results["Abs_Error"] = results["Error"].abs()

results_path = os.path.join(out_dir, "results_linear.csv")
results.to_csv(results_path, index=False)

print(f"Rezultate salvate in: {results_path}")