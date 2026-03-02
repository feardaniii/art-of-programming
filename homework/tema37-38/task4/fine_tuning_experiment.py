import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class StrategyResult:
    name: str
    trainable_params: int
    final_val_acc: float
    overfit: bool
    train_time_sec: float
    history: keras.callbacks.History


def load_cats_vs_dogs_from_cifar10(samples_per_class_train=500, image_size=(128, 128)):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.flatten()
    y_test_full = y_test_full.flatten()

    cat_label = 3
    dog_label = 5

    cat_train_idx = np.where(y_train_full == cat_label)[0][:samples_per_class_train]
    dog_train_idx = np.where(y_train_full == dog_label)[0][:samples_per_class_train]
    train_idx = np.concatenate([cat_train_idx, dog_train_idx])

    cat_test_idx = np.where(y_test_full == cat_label)[0]
    dog_test_idx = np.where(y_test_full == dog_label)[0]
    test_idx = np.concatenate([cat_test_idx, dog_test_idx])

    x_train = x_train_full[train_idx]
    y_train = (y_train_full[train_idx] == dog_label).astype(np.float32)
    x_test = x_test_full[test_idx]
    y_test = (y_test_full[test_idx] == dog_label).astype(np.float32)

    x_train = tf.image.resize(x_train, image_size).numpy().astype(np.float32)
    x_test = tf.image.resize(x_test, image_size).numpy().astype(np.float32)

    val_size = int(0.15 * len(x_train))
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def count_trainable_params(model: keras.Model) -> int:
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def build_model(unfreeze_last_n: int, input_shape=(128, 128, 3), learning_rate=1e-4):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    if unfreeze_last_n > 0:
        base.trainable = True
        for layer in base.layers[:-unfreeze_last_n]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=f"mobilenetv2_last{unfreeze_last_n}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_strategy(name: str, unfreeze_last_n: int, data, epochs=10) -> StrategyResult:
    (x_train, y_train), (x_val, y_val), _ = data
    learning_rate = 1e-3 if unfreeze_last_n == 0 else 1e-4
    model = build_model(unfreeze_last_n=unfreeze_last_n, learning_rate=learning_rate)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    final_val_acc = float(max(history.history["val_accuracy"]))
    final_train_acc = float(max(history.history["accuracy"]))
    overfit = (final_train_acc - final_val_acc) > 0.06

    return StrategyResult(
        name=name,
        trainable_params=count_trainable_params(model),
        final_val_acc=final_val_acc,
        overfit=overfit,
        train_time_sec=elapsed,
        history=history,
    )


def plot_all_val_accuracy(results: list[StrategyResult], out_path="task4_val_accuracy_comparison.png"):
    plt.figure(figsize=(10, 5))
    for r in results:
        plt.plot(r.history.history["val_accuracy"], marker="o", linewidth=2, label=r.name)
    plt.title("Validation Accuracy - Fine-Tuning Strategies")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def print_table(results: list[StrategyResult]):
    print("\n| Strategy   | Trainable Params | Final Val Acc | Overfit? |")
    print("|------------|-----------------:|--------------:|:--------:|")
    for r in results:
        overfit_text = "Da" if r.overfit else "Nu"
        print(
            f"| {r.name:<10} | {r.trainable_params:>16} | {r.final_val_acc*100:>11.2f}% | {overfit_text:^8} |"
        )


def print_analysis(results: list[StrategyResult]):
    by_acc = sorted(results, key=lambda r: r.final_val_acc, reverse=True)
    best = by_acc[0]

    print("\nAnaliza (diminishing returns vs risc de overfitting):")
    print("1. Strategia A antreneaza doar head-ul si are cel mai mic risc de overfitting.")
    print("2. Strategia B permite adaptare partiala a reprezentarilor si de obicei creste acuratetea.")
    print("3. Strategia C creste numarul de parametri trainabili semnificativ si poate aduce castig suplimentar.")
    print("4. Totusi, castigul de la B la C este adesea mai mic decat castigul de la A la B.")
    print("5. Acesta este efectul de diminishing returns: mai mult efort pentru castig incremental mai mic.")
    print("6. Pe seturi mici, C poate memoriza mai usor pattern-uri specifice si creste riscul de overfitting.")
    print("7. `ReduceLROnPlateau` ajuta deoarece reduce pasul de invatare cand val_loss stagneaza.")
    print("8. Astfel, fine-tuning-ul ramane stabil si evita degradarea rapida a modelului pre-antrenat.")
    print("9. Daca diferenta train vs val devine mare, strategia este prea agresiva pentru volumul de date.")
    print("10. Daca B este aproape de C ca performanta, B este de regula alegerea mai eficienta in productie.")
    print("11. Decizia finala trebuie sa echilibreze acuratetea, timpul de antrenare si robustetea la overfitting.")
    print("12. In acest run, strategia recomandata este cea cu cel mai bun raport val_acc / overfitting / cost.")
    print(f"\nRecomandare pentru acest experiment: {best.name} (cea mai buna val_accuracy).")


def main():
    print("Task 4 - Fine-Tuning Experiment (MobileNetV2, Cats vs Dogs from CIFAR-10)")
    print("Strategii: A=head frozen base, B=last10 unfrozen, C=last30 unfrozen\n")

    data = load_cats_vs_dogs_from_cifar10(samples_per_class_train=500, image_size=(128, 128))

    results = [
        run_strategy("A: Head", unfreeze_last_n=0, data=data, epochs=10),
        run_strategy("B: Last10", unfreeze_last_n=10, data=data, epochs=10),
        run_strategy("C: Last30", unfreeze_last_n=30, data=data, epochs=10),
    ]

    chart_path = plot_all_val_accuracy(results)
    print_table(results)

    print("\nTimp de antrenare:")
    for r in results:
        print(f"- {r.name}: {r.train_time_sec:.1f}s")

    print_analysis(results)
    print(f"\nGrafic salvat: {chart_path}")


if __name__ == "__main__":
    main()
