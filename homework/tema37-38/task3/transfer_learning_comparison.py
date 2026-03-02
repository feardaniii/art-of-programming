import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Three visually distinct CIFAR-10 classes:
# airplane (0), frog (6), truck (9)
SELECTED_CLASSES = [0, 6, 9]
CLASS_LABELS = {0: "airplane", 6: "frog", 9: "truck"}
CLASS_TO_NEW = {old: new for new, old in enumerate(SELECTED_CLASSES)}


@dataclass
class RunResult:
    model_name: str
    parameters: int
    train_time_sec: float
    test_acc: float


def subset_cifar10(samples_per_class_train=300, input_size=(160, 160)):
    (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.cifar10.load_data()
    y_train_all = y_train_all.flatten()
    y_test_all = y_test_all.flatten()

    train_indices = []
    for c in SELECTED_CLASSES:
        idx = np.where(y_train_all == c)[0][:samples_per_class_train]
        train_indices.append(idx)
    train_indices = np.concatenate(train_indices)

    # Use all test samples from the same classes for stable evaluation.
    test_indices = np.concatenate([np.where(y_test_all == c)[0] for c in SELECTED_CLASSES])

    x_train = x_train_all[train_indices]
    y_train = np.array([CLASS_TO_NEW[v] for v in y_train_all[train_indices]], dtype=np.int32)
    x_test = x_test_all[test_indices]
    y_test = np.array([CLASS_TO_NEW[v] for v in y_test_all[test_indices]], dtype=np.int32)

    x_train = tf.image.resize(x_train, input_size).numpy().astype("float32") / 255.0
    x_test = tf.image.resize(x_test, input_size).numpy().astype("float32") / 255.0

    # Validation split from train subset.
    val_size = int(0.15 * len(x_train))
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(model_name: str, input_shape=(160, 160, 3), n_classes=3):
    if model_name == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base.trainable = False

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name=f"{model_name}_transfer",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_experiment(model_name: str, data, epochs=8):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    model = build_model(model_name)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    start = time.time()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=callbacks,
    )
    train_time = time.time() - start

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return RunResult(
        model_name=model_name,
        parameters=model.count_params(),
        train_time_sec=train_time,
        test_acc=float(test_acc),
    )


def save_bar_chart(results: list[RunResult], out_name="task3_model_comparison.png"):
    names = [r.model_name for r in results]
    params_m = [r.parameters / 1_000_000 for r in results]
    times = [r.train_time_sec for r in results]
    accs = [r.test_acc * 100 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(names, params_m, color=["#2e86de", "#16a085"])
    axes[0].set_title("Parameters (Millions)")
    axes[0].set_ylabel("M params")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, times, color=["#2e86de", "#16a085"])
    axes[1].set_title("Training Time (s)")
    axes[1].set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(names, accs, color=["#2e86de", "#16a085"])
    axes[2].set_title("Test Accuracy (%)")
    axes[2].set_ylabel("%")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()
    return out_name


def print_table(results: list[RunResult]):
    print("\n| Model       | Parameters | Train Time | Test Acc |")
    print("|-------------|-----------:|-----------:|---------:|")
    for r in results:
        print(
            f"| {r.model_name:<11} | {r.parameters:>10} | {r.train_time_sec:>9.1f}s | {r.test_acc*100:>7.2f}% |"
        )


def deployment_recommendation(results: list[RunResult]):
    by_acc = sorted(results, key=lambda r: r.test_acc, reverse=True)
    best = by_acc[0]
    other = by_acc[1]
    acc_gap = (best.test_acc - other.test_acc) * 100

    if best.model_name == "MobileNetV2":
        rec = (
            "I would deploy MobileNetV2 because it is the most accurate in this run and "
            "it is also much lighter for inference."
        )
    else:
        if acc_gap < 1.5:
            rec = (
                "I would deploy MobileNetV2 because the accuracy gap is small while inference "
                "cost is much lower than ResNet50."
            )
        else:
            rec = (
                "I would deploy ResNet50 because it gives a clear accuracy advantage that may justify "
                "the larger compute and memory footprint."
            )
    return rec


def main():
    print("Task 3 - Transfer Learning Comparison")
    print("Selected classes:", ", ".join(CLASS_LABELS[c] for c in SELECTED_CLASSES))
    print("Train subset: 300 images per class, same epochs for both models.\n")

    data = subset_cifar10(samples_per_class_train=300, input_size=(160, 160))
    epochs = 8

    results = [
        run_experiment("MobileNetV2", data, epochs=epochs),
        run_experiment("ResNet50", data, epochs=epochs),
    ]

    print_table(results)
    chart_path = save_bar_chart(results)
    rec = deployment_recommendation(results)

    print("\nRecommendation:")
    print(rec)
    print(f"\nSaved chart: {chart_path}")


if __name__ == "__main__":
    main()
