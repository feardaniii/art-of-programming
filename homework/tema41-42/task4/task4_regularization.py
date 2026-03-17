"""Regularization strategy comparison on California Housing."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configure TensorFlow behavior before importing it.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
L2_FACTOR = 1e-4

STRATEGIES = [
    {"name": "Baseline", "dropout": False, "l2_reg": False, "batchnorm": False},
    {"name": "A - Dropout", "dropout": True, "l2_reg": False, "batchnorm": False},
    {"name": "B - L2", "dropout": False, "l2_reg": True, "batchnorm": False},
    {"name": "C - BatchNorm", "dropout": False, "l2_reg": False, "batchnorm": True},
    {"name": "D - Dropout+L2", "dropout": True, "l2_reg": True, "batchnorm": False},
    {"name": "E - All three", "dropout": True, "l2_reg": True, "batchnorm": True},
]


def build_model(dropout: bool = False, l2_reg: bool = False, batchnorm: bool = False) -> tf.keras.Model:
    """Build the fixed large architecture with optional regularization modules."""
    kernel_reg = l2(L2_FACTOR) if l2_reg else None

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(8,)))

    for units in (256, 256, 128, 64):
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=kernel_reg))
        if batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout:
            model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


def strategy_description(strategy: dict[str, object]) -> str:
    """Create a compact textual description for table display."""
    parts = []
    if strategy["dropout"]:
        parts.append("Dropout")
    if strategy["l2_reg"]:
        parts.append("L2")
    if strategy["batchnorm"]:
        parts.append("BatchNorm")
    return "+".join(parts) if parts else "None"


def flatten_hidden_kernels(model: tf.keras.Model) -> np.ndarray:
    """Collect and flatten all hidden-layer dense kernels from a trained model."""
    kernels = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.units != 1:
            weights = layer.get_weights()
            if weights:
                kernels.append(weights[0].ravel())
    if not kernels:
        return np.array([])
    return np.concatenate(kernels)


def print_comparison_table(results: list[dict[str, object]]) -> None:
    """Print strategy comparison table."""
    print("\n| Strategy | Description | Test MSE | Gap |")
    print("|---|---|---|---|")
    for item in results:
        print(
            f"| {item['name']} | {item['description']} | "
            f"{float(item['test_mse']):.4f} | {float(item['gap']):.4f} |"
        )


def plot_training_curves(results: list[dict[str, object]], output_path: Path) -> None:
    """Plot train/validation loss curves for all six strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    flat_axes = axes.flatten()

    for axis, item in zip(flat_axes, results):
        history = item["history"]
        axis.plot(history["loss"], label="Train Loss", linewidth=1.8)
        axis.plot(history["val_loss"], label="Val Loss", linewidth=1.8)
        axis.set_title(f"{item['name']} | Gap={float(item['gap']):.4f}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE Loss")
        axis.grid(alpha=0.2)
        axis.legend()

    fig.suptitle("Regularization Strategies: Train vs Validation Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_weight_histograms(baseline_weights: np.ndarray, l2_weights: np.ndarray, output_path: Path) -> None:
    """Plot weight distributions for Baseline and L2 strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for axis, weights, name, color in (
        (axes[0], baseline_weights, "Baseline", "#457b9d"),
        (axes[1], l2_weights, "B - L2", "#2a9d8f"),
    ):
        axis.hist(weights, bins=50, color=color, edgecolor="black", alpha=0.85)
        axis.set_title(f"{name} Hidden Kernels\nmean={weights.mean():.5f}, std={weights.std():.5f}")
        axis.set_xlabel("Weight Value")
        axis.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.get_logger().setLevel("ERROR")

    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing()
    x = dataset.data
    y = dataset.target.reshape(-1, 1)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.fit_transform(y_train).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test).reshape(-1)

    results: list[dict[str, object]] = []
    baseline_weights = None
    l2_weights = None

    for strategy in STRATEGIES:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        model = build_model(
            dropout=bool(strategy["dropout"]),
            l2_reg=bool(strategy["l2_reg"]),
            batchnorm=bool(strategy["batchnorm"]),
        )
        history_obj = model.fit(
            x_train_scaled,
            y_train_scaled,
            validation_data=(x_val_scaled, y_val_scaled),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        test_mse, _ = model.evaluate(x_test_scaled, y_test_scaled, verbose=0)
        train_loss_final = float(history_obj.history["loss"][-1])
        val_loss_final = float(history_obj.history["val_loss"][-1])
        gap = train_loss_final - val_loss_final

        strategy_name = str(strategy["name"])
        results.append(
            {
                "name": strategy_name,
                "description": strategy_description(strategy),
                "test_mse": float(test_mse),
                "gap": float(gap),
                "history": history_obj.history,
            }
        )

        if strategy_name == "Baseline":
            baseline_weights = flatten_hidden_kernels(model)
        elif strategy_name == "B - L2":
            l2_weights = flatten_hidden_kernels(model)

        print(f"Training {strategy_name}... Test MSE={float(test_mse):.3f}, Gap={gap:.3f}")

    print_comparison_table(results)

    plot_training_curves(results, output_dir / "task4_training_curves.png")

    if baseline_weights is None or l2_weights is None:
        raise RuntimeError("Could not collect baseline and L2 hidden kernels for histogram plotting.")
    plot_weight_histograms(baseline_weights, l2_weights, output_dir / "task4_weight_histograms.png")


if __name__ == "__main__":
    main()


# In this run, D - Dropout+L2 reduced overfitting the most by Gap, ending at 0.0011, which is the
# closest-to-zero train/validation difference among all strategies. The lowest test MSE, however,
# came from A - Dropout at 0.2032 with Gap=-0.0212, showing a strong generalization result with only
# a small residual mismatch between train and validation loss. Baseline had Test MSE=0.2453 and
# Gap=-0.2139, so the large-capacity model clearly behaved less balanced without regularization.
# L2 alone improved the gap relative to baseline (Gap=-0.1409) but did not improve test MSE here
# (0.2624), which highlights that smaller weights do not automatically mean better predictive error.
# BatchNorm alone performed worst in this setup, with Test MSE=0.4106 and Gap=-0.3283, suggesting
# normalization by itself was not enough to control this model's behavior on this split. Combining
# all three methods (E - All three) produced Gap=-0.0258 but a weaker Test MSE=0.3043, so adding
# every regularizer together helped gap stability more than final accuracy. This shows the practical
# trade-off: Dropout and L2 can reduce overfitting signals, but too much combined constraint can hurt
# the error objective even if train/val curves appear better aligned. The weight histograms reinforce
# the expected L2 effect by showing a tighter concentration around zero for the L2 model compared with
# Baseline, which indicates reduced weight magnitude and fewer extreme kernel values. In short, this
# experiment suggests choosing A - Dropout for best MSE, or D - Dropout+L2 when minimizing the gap is
# the primary goal.
