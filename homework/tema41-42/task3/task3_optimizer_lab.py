"""Optimizer and learning-rate grid lab on California Housing."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set env flags before importing TensorFlow.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adagrad, Adam, RMSprop, SGD

RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

OPTIMIZER_CLASSES = {
    "SGD": SGD,
    "Adam": Adam,
    "RMSprop": RMSprop,
    "Adagrad": Adagrad,
}
LEARNING_RATES = [0.1, 0.01, 0.001]


def build_model(optimizer: tf.keras.optimizers.Optimizer) -> tf.keras.Model:
    """Build a fresh regression model with fixed architecture."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def sanitize_r2(value: float) -> float:
    """Convert invalid/diverged R² values to 0, as required."""
    if not np.isfinite(value) or value < -1.0:
        return 0.0
    return float(value)


def print_summary_table(sorted_items: list[tuple[tuple[str, float], dict[str, object]]]) -> None:
    """Print all 12 runs sorted by R² descending."""
    print("\n| Rank | Optimizer | Learning Rate | R² |")
    print("|---|---|---|---|")
    for rank, ((optimizer_name, learning_rate), result) in enumerate(sorted_items, start=1):
        print(f"| {rank} | {optimizer_name} | {learning_rate:.3f} | {result['r2']:.3f} |")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.get_logger().setLevel("ERROR")

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing()
    x = dataset.data
    y = dataset.target.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y).reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y_scaled,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results: dict[tuple[str, float], dict[str, object]] = {}

    for optimizer_name in OPTIMIZER_CLASSES:
        for learning_rate in LEARNING_RATES:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            optimizer_cls = OPTIMIZER_CLASSES[optimizer_name]
            optimizer = optimizer_cls(learning_rate=learning_rate)
            model = build_model(optimizer=optimizer)

            history = model.fit(
                x_train,
                y_train,
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

            try:
                y_pred = model.predict(x_test, verbose=0).reshape(-1)
                raw_r2 = float(r2_score(y_test, y_pred))
            except Exception:
                raw_r2 = float("nan")

            safe_r2 = sanitize_r2(raw_r2)
            results[(optimizer_name, learning_rate)] = {
                "r2": safe_r2,
                "raw_r2": raw_r2,
                "history": history.history,
            }

            print(f"Training {optimizer_name} lr={learning_rate}... R²={safe_r2:.3f}")

    r2_matrix = np.zeros((len(OPTIMIZER_CLASSES), len(LEARNING_RATES)), dtype=float)
    optimizer_names = list(OPTIMIZER_CLASSES.keys())
    for row_idx, optimizer_name in enumerate(optimizer_names):
        for col_idx, learning_rate in enumerate(LEARNING_RATES):
            r2_matrix[row_idx, col_idx] = float(results[(optimizer_name, learning_rate)]["r2"])

    r2_display = np.clip(r2_matrix, 0.0, None)

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        r2_display,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=[f"{lr:.3f}" for lr in LEARNING_RATES],
        yticklabels=optimizer_names,
        cbar_kws={"label": "R² (clamped at 0 for display)"},
    )
    plt.title("California Housing: R² Heatmap (Optimizer vs Learning Rate)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Optimizer")
    plt.tight_layout()
    plt.savefig(output_dir / "task3_heatmap.png", dpi=160)
    plt.close()

    sorted_results = sorted(results.items(), key=lambda item: float(item[1]["r2"]), reverse=True)
    top3 = sorted_results[:3]

    plt.figure(figsize=(10, 6))
    colors = ["#1d3557", "#e63946", "#2a9d8f"]
    for color, ((optimizer_name, learning_rate), result) in zip(colors, top3):
        losses = result["history"]["loss"]
        r2_val = float(result["r2"])
        label = f"{optimizer_name} lr={learning_rate:.3f} (R²={r2_val:.3f})"
        plt.plot(losses, color=color, linewidth=2, label=label)

    plt.title("Top-3 Configurations: Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task3_top3_loss_curves.png", dpi=160)
    plt.close()

    print_summary_table(sorted_results)

    best_key, best_result = sorted_results[0]
    best_optimizer, best_learning_rate = best_key
    print(
        f"\nBest combination: {best_optimizer} lr={best_learning_rate:.3f} "
        f"with R²={float(best_result['r2']):.3f}"
    )


if __name__ == "__main__":
    main()


# In this run, the best optimizer/LR combination was Adagrad with learning rate 0.100, reaching
# an R² of 0.790 on the test set. The next best settings, RMSprop at 0.001 (R²=0.785) and Adam at
# 0.001 (R²=0.784), were very close, which suggests multiple adaptive methods can perform well when
# their step size is tuned. SGD behaves differently at high learning rates because a global step can
# overshoot minima repeatedly, making training unstable and potentially collapsing generalization, as
# seen with SGD at 0.100 producing a poor score. The heatmap reveals a clear sensitivity pattern:
# many optimizers improve around 0.01 to 0.001, while some high-LR cells degrade sharply or appear
# near the bottom, indicating unstable update dynamics. Combined with the top-3 loss curves, this
# shows that the best runs are not only accurate at the end, but also maintain smoother optimization.
