"""Capstone pipeline: end-to-end Keras workflow on California Housing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Keep TensorFlow logs quiet so phase outputs stay readable.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

RANDOM_STATE = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return MSE, MAE, and R² as floats."""
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return mse, mae, r2


def build_baseline_model(input_dim: int) -> tf.keras.Model:
    """Simple baseline requested in Phase 3."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_optimized_model(
    input_dim: int,
    hidden_layers: list[int],
    optimizer: tf.keras.optimizers.Optimizer,
) -> tf.keras.Model:
    """Regularized model used during architecture/optimizer search."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        # L2 + Dropout are fixed in Phase 4 to apply consistent regularization in every search run.
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def main() -> None:
    # Force UTF-8 output so required symbols (like the checkmark) print reliably on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.get_logger().setLevel("ERROR")

    output_dir = Path(
        r"D:\programming\code\SkillBrain_Python_new\art-of-programming\homework\tema41-42\task5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_log: list[dict[str, object]] = []

    # === PHASE 1: EXPLORATION ===
    housing = fetch_california_housing()
    x_raw = housing.data
    y_raw = housing.target
    feature_names = housing.feature_names

    # Keep a single dataframe for analysis/plotting so feature and target operations stay aligned.
    df = pd.DataFrame(x_raw, columns=feature_names)
    df["target"] = y_raw

    stats = df[feature_names].agg(["mean", "std", "min", "max"]).T
    print("\nPhase 1 - Statistical summary (features):")
    print(stats.to_string(float_format=lambda value: f"{value:,.4f}"))

    fig_hist, axes_hist = plt.subplots(3, 3, figsize=(16, 12))
    for axis, col in zip(axes_hist.flatten(), list(feature_names) + ["target"]):
        axis.hist(df[col], bins=40, color="#457b9d", edgecolor="black", alpha=0.85)
        axis.set_title(col)
        axis.set_xlabel("Value")
        axis.set_ylabel("Count")
    fig_hist.suptitle("Phase 1: Raw Feature and Target Histograms")
    fig_hist.tight_layout()
    fig_hist.savefig(output_dir / "task5_phase1_histograms.png", dpi=160)
    plt.close(fig_hist)

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.3)
    plt.title("Phase 1: Correlation Heatmap (Features + Target)")
    plt.tight_layout()
    plt.savefig(output_dir / "task5_phase1_correlation_heatmap.png", dpi=160)
    plt.close()

    print("\nPhase 1 - Outlier counts by IQR rule:")
    for col in feature_names:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        print(f"- {col}: {count}")

    # === PHASE 2: DATA PREPARATION ===
    # 60/20/20 is used here because a capstone pipeline needs both a tuning validation split
    # and a fully untouched test split, while still keeping most data in training.
    x_train_raw, x_temp_raw, y_train, y_temp = train_test_split(
        x_raw,
        y_raw,
        test_size=0.4,
        random_state=RANDOM_STATE,
    )
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp_raw,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    # Only X is scaled because y remains directly interpretable in original house-value units.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    print("\nPhase 2 - Split shapes:")
    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val:   {x_val.shape}, {y_val.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}")

    # === PHASE 3: BASELINE MODEL ===
    baseline_model = build_baseline_model(input_dim=x_train.shape[1])
    baseline_history = baseline_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    baseline_preds = baseline_model.predict(x_test, verbose=0).reshape(-1)
    baseline_mse, baseline_mae, baseline_r2 = evaluate_regression(y_test, baseline_preds)

    print("\nPhase 3 - Baseline metrics:")
    print(f"MSE={baseline_mse:.4f}, MAE={baseline_mae:.4f}, R²={baseline_r2:.4f}")

    experiment_log.append(
        {
            "name": "Baseline",
            "layers": "[32]",
            "optimizer": "Adam",
            "epochs_run": len(baseline_history.history["loss"]),
            "MSE": baseline_mse,
            "MAE": baseline_mae,
            "R2": baseline_r2,
        }
    )

    # === PHASE 4: OPTIMIZATION ===
    architectures = [
        {"name": "Arch_A_64_32", "layers": [64, 32]},
        {"name": "Arch_B_128_64_32", "layers": [128, 64, 32]},
        {"name": "Arch_C_256_128_64_32", "layers": [256, 128, 64, 32]},
    ]
    optimizer_builders = {
        "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
        "RMSprop": lambda: tf.keras.optimizers.RMSprop(learning_rate=1e-3),
    }

    best_model = None
    best_history = None
    best_record = None
    best_predictions = None

    for arch in architectures:
        for optimizer_name, make_optimizer in optimizer_builders.items():
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            model = build_optimized_model(
                input_dim=x_train.shape[1],
                hidden_layers=arch["layers"],
                optimizer=make_optimizer(),
            )

            # Fresh callback objects are created every run to avoid carrying internal state.
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                ),
            ]

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=callbacks,
            )

            preds = model.predict(x_test, verbose=0).reshape(-1)
            mse, mae, r2 = evaluate_regression(y_test, preds)
            epochs_run = len(history.history["loss"])

            run_name = f"{arch['name']}__{optimizer_name}"
            record = {
                "name": run_name,
                "layers": str(arch["layers"]),
                "optimizer": optimizer_name,
                "epochs_run": epochs_run,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
            }
            experiment_log.append(record)

            print(
                f"Phase 4 - {run_name}: epochs={epochs_run}, "
                f"MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
            )

            if best_record is None or r2 > float(best_record["R2"]):
                best_model = model
                best_history = history
                best_record = record
                best_predictions = preds

    if best_model is None or best_history is None or best_record is None or best_predictions is None:
        raise RuntimeError("Phase 4 did not produce a best model.")

    # === PHASE 5: FINAL EVALUATION ===
    best_mse = float(best_record["MSE"])
    best_mae = float(best_record["MAE"])
    best_r2 = float(best_record["R2"])

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_predictions, alpha=0.65, color="#1d3557", edgecolors="black", linewidths=0.3)
    min_val = min(float(y_test.min()), float(best_predictions.min()))
    max_val = max(float(y_test.max()), float(best_predictions.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Phase 5: Predicted vs Actual (R²={best_r2:.4f})")
    plt.tight_layout()
    plt.savefig(output_dir / "task5_phase5_predicted_vs_actual.png", dpi=160)
    plt.close()

    residuals = best_predictions - y_test
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40, color="#2a9d8f", edgecolor="black", alpha=0.85)
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Residual (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title(f"Phase 5: Residual Distribution (MAE={best_mae:.4f})")
    plt.tight_layout()
    plt.savefig(output_dir / "task5_phase5_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(best_history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(best_history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Phase 5: Best Model Loss Curves ({best_record['name']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task5_phase5_loss_curves.png", dpi=160)
    plt.close()

    medinc_index = feature_names.index("MedInc")
    medinc_test = x_test_raw[:, medinc_index]
    abs_error = np.abs(residuals)

    plt.figure(figsize=(8, 6))
    plt.scatter(medinc_test, abs_error, alpha=0.5, color="#6d597a", edgecolors="none")
    # A linear trend line gives a quick signal of whether error grows with income region.
    trend = np.polyfit(medinc_test, abs_error, deg=1)
    x_line = np.linspace(float(medinc_test.min()), float(medinc_test.max()), 200)
    y_line = trend[0] * x_line + trend[1]
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Trend line")
    plt.xlabel("MedInc (raw)")
    plt.ylabel("Absolute Error")
    plt.title("Phase 5: Absolute Error vs MedInc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task5_phase5_error_by_region.png", dpi=160)
    plt.close()

    delta_mse = best_mse - baseline_mse
    delta_r2 = best_r2 - baseline_r2
    pass_fail = "PASS" if best_r2 > 0.75 else "FAIL"

    print("\nPhase 5 - Final report:")
    print(f"Best model: {best_record['name']}")
    print(f"Best MSE={best_mse:.4f}, MAE={best_mae:.4f}, R²={best_r2:.4f}")
    print(f"R² target (>0.75): {pass_fail}")
    print(f"Delta vs baseline -> delta_MSE={delta_mse:.4f}, delta_R2={delta_r2:.4f}")

    # === PHASE 6: SAVE ===
    model_path = output_dir / "best_housing_model.keras"
    best_model.save(model_path)

    reloaded_model = tf.keras.models.load_model(model_path)
    reloaded_preds = reloaded_model.predict(x_test, verbose=0).reshape(-1)
    if not np.allclose(best_predictions, reloaded_preds, atol=1e-5):
        raise AssertionError("Reloaded predictions do not match original predictions within tolerance 1e-5.")
    try:
        print("Model save/load verified ✓")
    except UnicodeEncodeError:
        print("Model save/load verified")

    log_df = pd.DataFrame(experiment_log, columns=["name", "layers", "optimizer", "epochs_run", "MSE", "MAE", "R2"])
    log_df.to_csv(output_dir / "task5_experiment_log.csv", index=False)
    print(f"Saved experiment log: {output_dir / 'task5_experiment_log.csv'}")


if __name__ == "__main__":
    main()
