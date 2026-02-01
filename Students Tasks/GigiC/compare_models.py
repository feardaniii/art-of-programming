import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "outputs")

lin_path = os.path.join(out_dir, "results_linear.csv")
rf_path = os.path.join(out_dir, "results_rf.csv")

lin = pd.read_csv(lin_path)
rf = pd.read_csv(rf_path)

# Verificare: real values trebuie sa fie identice
if not (lin["Real_Temp"].values == rf["Real_Temp"].values).all():
    raise ValueError("Seturile de test NU coincid. Verifica random_state/curatarea/split-ul in cele 2 scripturi.")

y_true = lin["Real_Temp"].values
y_pred_lin = lin["Pred_Temp"].values
y_pred_rf = rf["Pred_Temp"].values

# Metrici
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae_l, rmse_l, r2_l = metrics(y_true, y_pred_lin)
mae_r, rmse_r, r2_r = metrics(y_true, y_pred_rf)

print("=" * 70)
print("Comparatie modele: Linear Regression vs Random Forest")
print("=" * 70)
print(f"Linear Regression: MAE={mae_l:.2f} K | RMSE={rmse_l:.2f} K | R²={r2_l:.3f}")
print(f"Random Forest   : MAE={mae_r:.2f} K | RMSE={rmse_r:.2f} K | R²={r2_r:.3f}")

# Grafic 1: Real vs Pred (ambele modele)
plt.close("all")
plt.figure(figsize=(7, 7))
plt.scatter(y_true, y_pred_lin, alpha=0.35, label="Linear Regression")
plt.scatter(y_true, y_pred_rf, alpha=0.35, label="Random Forest")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle="--")
plt.xlabel("Temperatura reală (K)")
plt.ylabel("Temperatura prezisă (K)")
plt.title("Real vs Prezis – Linear vs Random Forest")
plt.legend()
plt.tight_layout()

# Grafic 2: Histograma erorilor (ambele modele)
err_lin = y_true - y_pred_lin
err_rf = y_true - y_pred_rf

plt.figure(figsize=(8, 4))
plt.hist(err_lin, bins=30, alpha=0.5, label="Linear Regression")
plt.hist(err_rf, bins=30, alpha=0.5, label="Random Forest")
plt.xlabel("Eroare (K)")
plt.ylabel("Număr de stele")
plt.title("Distribuția erorilor – Comparatie modele")
plt.legend()
plt.tight_layout()

plt.show()
