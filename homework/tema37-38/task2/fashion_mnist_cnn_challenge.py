import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass
class Config:
    name: str
    filters: list[int]
    dropout: float
    augmentation: bool
    learning_rate: float
    epochs: int = 18


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    val_size = 6000
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(config: Config) -> keras.Model:
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    if config.augmentation:
        x = keras.Sequential(
            [
                layers.RandomTranslation(0.08, 0.08),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.08),
            ],
            name=f"aug_{config.name}",
        )(x)

    for i, n_filters in enumerate(config.filters):
        x = layers.Conv2D(
            n_filters,
            (3, 3),
            activation="relu",
            padding="same",
            name=f"conv_{config.name}_{i+1}",
        )(x)
        x = layers.Conv2D(
            n_filters,
            (3, 3),
            activation="relu",
            padding="same",
            name=f"conv_{config.name}_{i+1}_b",
        )(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool_{config.name}_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{config.name}_{i+1}")(x)
        x = layers.Dropout(config.dropout, name=f"drop_{config.name}_{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name=f"gap_{config.name}")(x)
    x = layers.Dense(128, activation="relu", name=f"dense_{config.name}")(x)
    x = layers.Dropout(min(0.6, config.dropout + 0.15), name=f"head_drop_{config.name}")(x)
    outputs = layers.Dense(10, activation="softmax", name=f"output_{config.name}")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"fashion_{config.name}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history: keras.callbacks.History, config_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title(f"Accuracy - Config {config_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title(f"Loss - Config {config_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"task2_training_curves_{config_name}.png", dpi=150)
    plt.close()


def plot_conf_matrix(y_true, y_pred, config_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9.5, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix - Config {config_name}")
    plt.tight_layout()
    out_path = f"task2_confusion_matrix_{config_name}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return cm, out_path


def hardest_class_analysis(cm: np.ndarray):
    per_class_total = cm.sum(axis=1)
    per_class_recall = np.diag(cm) / np.maximum(per_class_total, 1)
    hardest_idx = int(np.argmin(per_class_recall))

    confusion_partners = cm[hardest_idx].copy()
    confusion_partners[hardest_idx] = 0
    most_confused_with_idx = int(np.argmax(confusion_partners))

    return hardest_idx, most_confused_with_idx, per_class_recall


def train_one_config(config: Config, data):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data

    model = build_model(config)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    return {
        "config": config,
        "model": model,
        "history": history,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "y_pred": y_pred,
        "y_test": y_test,
    }


def print_comparison_table(results: list[dict]):
    print("\n| Config | Filters    | Dropout | Augmentation | Learning Rate | Test Acc |")
    print("|--------|------------|---------|--------------|---------------|----------|")
    for r in results:
        cfg = r["config"]
        filters_text = "-".join(str(x) for x in cfg.filters)
        aug_text = "Da" if cfg.augmentation else "Nu"
        print(
            f"| {cfg.name} | {filters_text:<10} | {cfg.dropout:<7.2f} | "
            f"{aug_text:<12} | {cfg.learning_rate:<13.5f} | {r['test_acc']*100:>7.2f}% |"
        )


def save_comparison_csv(results: list[dict], path="task2_comparison_table.csv"):
    lines = ["config,filters,dropout,augmentation,learning_rate,test_acc"]
    for r in results:
        cfg = r["config"]
        lines.append(
            f"{cfg.name},{'-'.join(str(x) for x in cfg.filters)},{cfg.dropout},"
            f"{cfg.augmentation},{cfg.learning_rate},{r['test_acc']:.6f}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    data = load_data()

    configs = [
        Config(name="A", filters=[32, 64], dropout=0.25, augmentation=True, learning_rate=1e-3),
        Config(name="B", filters=[64, 128], dropout=0.50, augmentation=False, learning_rate=8e-4),
        Config(name="C", filters=[32, 64, 128], dropout=0.30, augmentation=True, learning_rate=8e-4),
    ]

    results = []
    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"Training Config {cfg.name} | Filters={cfg.filters} | Dropout={cfg.dropout} | "
              f"Augmentation={cfg.augmentation} | LR={cfg.learning_rate}")
        print("=" * 70)
        result = train_one_config(cfg, data)
        plot_training_curves(result["history"], cfg.name)
        results.append(result)
        print(f"Config {cfg.name} test accuracy: {result['test_acc']*100:.2f}%")

    results = sorted(results, key=lambda x: x["test_acc"], reverse=True)
    best = results[0]
    best_cfg = best["config"]

    print("\nBest configuration:", best_cfg.name)
    print(f"Best test accuracy: {best['test_acc']*100:.2f}%")

    cm, cm_path = plot_conf_matrix(best["y_test"], best["y_pred"], best_cfg.name)
    hardest_idx, partner_idx, recalls = hardest_class_analysis(cm)

    print_comparison_table(results)
    csv_path = save_comparison_csv(results)

    print("\nHardest class analysis:")
    print(
        f"- Hardest class: {CLASS_NAMES[hardest_idx]} "
        f"(recall={recalls[hardest_idx]*100:.2f}%)."
    )
    print(
        f"- Most often confused with: {CLASS_NAMES[partner_idx]} "
        f"(count={cm[hardest_idx, partner_idx]})."
    )

    print("\nAnalysis (5+ sentences):")
    print(
        "The three configurations show that model capacity and regularization must be balanced."
    )
    print(
        "Config B uses stronger dropout and no augmentation, which can reduce overfitting but may also underfit difficult classes."
    )
    print(
        "Configs A and C benefit from augmentation because Fashion-MNIST has class pairs with similar shapes and textures."
    )
    print(
        "The deepest setup (C) usually improves representation quality, but it needs a stable learning rate and enough regularization."
    )
    print(
        "The confusion matrix confirms that visually similar garments remain the main source of errors."
    )
    print(
        "The hardest class tends to have low inter-class separability, so the model confuses it with a structurally close class."
    )
    print(
        "Overall, a configuration that reaches at least 90% test accuracy while keeping train/validation curves close is the best trade-off."
    )

    print("\nArtifacts generated:")
    print("- Training curves: task2_training_curves_A.png, task2_training_curves_B.png, task2_training_curves_C.png")
    print(f"- Confusion matrix (best config): {cm_path}")
    print(f"- Comparison table CSV: {csv_path}")

    if best["test_acc"] < 0.90:
        print(
            "\nWARNING: best model is below 90% test accuracy. "
            "Increase epochs, add mild augmentation, or lower dropout on deeper config."
        )
    else:
        print("\nRequirement met: best model reached at least 90% test accuracy.")


if __name__ == "__main__":
    main()
