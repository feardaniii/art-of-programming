import os
import random

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Make runs more stable/reproducible.
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def build_model(activation: str = "relu") -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation=activation),
            tf.keras.layers.Dense(8, activation=activation),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_prepare_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def segment_1_base_model(X_train, X_test, y_train, y_test):
    print("\n=== Segment 1: Base Iris Classifier (TensorFlow) ===")

    model = build_model(activation="relu")
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train Accuracy: {train_acc:.4f} | Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f} | Test Loss:  {test_loss:.4f}")


def segment_2_activation_compare(X_train, X_test, y_train, y_test):
    print("\n=== Segment 2: Activation Function Comparison ===")

    activations = ["relu", "tanh", "sigmoid"]
    results = []

    for activation in activations:
        model = build_model(activation=activation)
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results.append((activation, test_acc, test_loss))

    print("Activation | Test Accuracy | Test Loss")
    print("-" * 39)
    for activation, acc, loss in results:
        print(f"{activation:<10} | {acc:<13.4f} | {loss:.4f}")


def segment_3_epoch_impact(X_train, X_test, y_train, y_test):
    print("\n=== Segment 3: Epoch Impact (activation='relu') ===")

    epochs_list = [20, 50, 100]
    results = []

    for n_epochs in epochs_list:
        model = build_model(activation="relu")
        history = model.fit(
            X_train,
            y_train,
            epochs=n_epochs,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]

        results.append((n_epochs, test_acc, test_loss, final_train_loss, final_val_loss))

    print("Epochs | Test Accuracy | Test Loss | Final Train Loss | Final Val Loss")
    print("-" * 72)
    for n_epochs, acc, test_loss, train_loss, val_loss in results:
        print(
            f"{n_epochs:<6} | {acc:<13.4f} | {test_loss:<9.4f} | "
            f"{train_loss:<16.4f} | {val_loss:.4f}"
        )


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    segment_1_base_model(X_train, X_test, y_train, y_test)
    segment_2_activation_compare(X_train, X_test, y_train, y_test)
    segment_3_epoch_impact(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
