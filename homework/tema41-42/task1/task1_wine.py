"""Train and compare Wine classifiers with top-2 vs full features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
EPOCHS = 100
TEST_SIZE = 0.2
OUTPUT_PLOT_PATH = Path("homework/tema41-42/task1_decision_boundary.png")


def build_model(input_dim: int) -> tf.keras.Model:
    """Create a 3-class MLP classifier with the requested architecture."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_accuracy_table(model_2f_acc: float, model_full_acc: float, top2_names: list[str]) -> None:
    """Print a markdown-style comparison table."""
    print("\n| Model | Features | Test Accuracy |")
    print("|---|---|---|")
    print(f"| model_2f | 2 ({top2_names[0]}, {top2_names[1]}) | {model_2f_acc:.4f} |")
    print(f"| model_full | 13 (all features) | {model_full_acc:.4f} |")


def plot_decision_boundary(
    model: tf.keras.Model,
    scaler: StandardScaler,
    x_space_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    target_names: np.ndarray,
    output_path: Path,
) -> None:
    """Plot and save a decision boundary for a 2D input model."""
    x0_min, x0_max = x_space_raw[:, 0].min(), x_space_raw[:, 0].max()
    x1_min, x1_max = x_space_raw[:, 1].min(), x_space_raw[:, 1].max()
    x0_pad = (x0_max - x0_min) * 0.1
    x1_pad = (x1_max - x1_min) * 0.1

    xx, yy = np.meshgrid(
        np.linspace(x0_min - x0_pad, x0_max + x0_pad, 350),
        np.linspace(x1_min - x1_pad, x1_max + x1_pad, 350),
    )

    grid_points_raw = np.column_stack([xx.ravel(), yy.ravel()])
    grid_points_scaled = scaler.transform(grid_points_raw)
    pred_classes = np.argmax(model.predict(grid_points_scaled, verbose=0), axis=1).reshape(xx.shape)

    region_cmap = ListedColormap(["#a8dadc", "#ffe29a", "#f4a6a6"])
    point_cmap = ListedColormap(["#1d3557", "#e07a5f", "#6a4c93"])

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, pred_classes, levels=np.arange(-0.5, 3.5, 1), cmap=region_cmap, alpha=0.75)
    scatter = plt.scatter(
        x_test_raw[:, 0],
        x_test_raw[:, 1],
        c=y_test,
        cmap=point_cmap,
        edgecolors="black",
        linewidths=0.5,
        s=60,
    )

    handles, _ = scatter.legend_elements()
    plt.legend(handles, list(target_names), title="Wine class", loc="best")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Wine Decision Boundary (Top-2 Features, Test Points)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Not all TensorFlow installations expose deterministic-op control.
        pass

    wine = load_wine()
    x_full = wine.data
    y = wine.target
    feature_names = np.array(wine.feature_names)

    f_scores, _ = f_classif(x_full, y)
    top2_idx = np.argsort(f_scores)[-2:][::-1]
    top2_names = feature_names[top2_idx].tolist()

    print("Top 2 features selected by f_classif:")
    for idx in top2_idx:
        print(f"- {feature_names[idx]}: {f_scores[idx]:.4f}")

    x_2f = x_full[:, top2_idx]

    sample_idx = np.arange(x_full.shape[0])
    train_idx, test_idx = train_test_split(
        sample_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train_full_raw = x_full[train_idx]
    x_test_full_raw = x_full[test_idx]
    x_train_2f_raw = x_2f[train_idx]
    x_test_2f_raw = x_2f[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler_2f = StandardScaler()
    x_train_2f = scaler_2f.fit_transform(x_train_2f_raw)
    x_test_2f = scaler_2f.transform(x_test_2f_raw)

    scaler_full = StandardScaler()
    x_train_full = scaler_full.fit_transform(x_train_full_raw)
    x_test_full = scaler_full.transform(x_test_full_raw)

    model_2f = build_model(input_dim=x_train_2f.shape[1])
    model_full = build_model(input_dim=x_train_full.shape[1])

    model_2f.fit(x_train_2f, y_train, epochs=EPOCHS, verbose=0)
    model_full.fit(x_train_full, y_train, epochs=EPOCHS, verbose=0)

    _, acc_2f = model_2f.evaluate(x_test_2f, y_test, verbose=0)
    _, acc_full = model_full.evaluate(x_test_full, y_test, verbose=0)

    print_accuracy_table(model_2f_acc=acc_2f, model_full_acc=acc_full, top2_names=top2_names)

    plot_decision_boundary(
        model=model_2f,
        scaler=scaler_2f,
        x_space_raw=x_2f,
        x_test_raw=x_test_2f_raw,
        y_test=y_test,
        feature_names=top2_names,
        target_names=wine.target_names,
        output_path=OUTPUT_PLOT_PATH,
    )
    print(f"\nSaved decision boundary plot to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()


# The selected two features outperform the rest because f_classif ranks them by how strongly
# each feature separates classes compared to within-class noise. The best pair has the largest
# between-class mean differences, so even a simple nonlinear model can place clean boundaries.
# In the Wine dataset, these top variables capture major chemical differences that are highly
# characteristic of each cultivar, so overlap between classes is reduced in 2D space. Lower-ranked
# features still carry signal, but many are weaker individually or partially redundant, which means
# they add less separability when used in isolation. As a result, this top-2 projection retains a
# surprising amount of discriminatory power while being much easier to visualize and interpret.
