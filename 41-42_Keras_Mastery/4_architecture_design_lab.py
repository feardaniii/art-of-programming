"""
================================================================================
ARCHITECTURE DESIGN LAB: How Many Layers? How Many Neurons? Callbacks Mastery
================================================================================

Course: The Art of Programming - Keras Mastery (Sessions 41-42)
Lesson: Designing Neural Networks That Actually Work

PREREQUISITES:
    - Script 1: Keras Decoded (loss, activations)
    - Script 2: Multiclass & Regression (problem types)
    - Script 3: Practitioner's Toolkit (regularization, optimizers)

THE BIG QUESTIONS:
    "How many layers should I use?"  — Every beginner asks this.
    "How many neurons per layer?"    — No one teaches this properly.
    "When do I stop training?"       — Callbacks answer this.

    Today we run EXPERIMENTS to find answers. No guessing. No rules of thumb.
    Systematic comparison. Data-driven decisions.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# PART 1: THE ARCHITECTURE DECISION FRAMEWORK
# ==============================================================================

def architecture_framework():
    """
    ANALOGY: Building a house.
    Before you lay a single brick, you ask: How big is the family? What's the
    budget? You do NOT start with a skyscraper and tear floors off.
    """
    print("=" * 70)
    print("PART 1: THE ARCHITECTURE DECISION FRAMEWORK")
    print("Three Questions Before Building ANY Model")
    print("=" * 70)
    print()
    print("Every beginner asks: 'How many layers? How many neurons?'")
    print("Every expert answers: 'It depends.' ...which is useless.")
    print("Today we replace 'it depends' with a SYSTEMATIC FRAMEWORK.")
    print()
    print("-" * 60)
    print("BEFORE BUILDING ANY MODEL, ASK:")
    print("-" * 60)
    print()
    print("  Question 1: HOW COMPLEX is my problem?")
    print("      -> Linear relationship?   1-2 layers is enough.")
    print("      -> Complex non-linear?    Start with 2-3 layers.")
    print("      -> Image/sequence data?   Specialized layers (Conv, LSTM).")
    print()
    print("  Question 2: HOW MUCH DATA do I have?")
    print("      -> <1,000 samples:   Keep it SMALL (1-2 layers, 32-64 neurons)")
    print("      -> 1,000-100,000:    Medium (2-3 layers, 64-256 neurons)")
    print("      -> >100,000:         Go deep (3+ layers, experiment with width)")
    print()
    print("  Question 3: WHAT'S MY BASELINE?")
    print("      -> Always start with the simplest model.")
    print("      -> Add complexity ONLY when justified by validation performance.")
    print("      -> 'Start simple, add complexity when justified'")
    print()
    print("-" * 60)
    print("THE GOLDEN RULE OF ARCHITECTURE:")
    print("  Your model should be JUST complex enough to capture the pattern,")
    print("  and NO MORE. Extra capacity = overfitting = bad generalization.")
    print("  Think of it like luggage: pack what you need, not everything you own.")
    print("-" * 60)
    print()
    print("Today's experiment: California Housing dataset")
    print("  - 20,640 samples | 8 features | Regression (predict house price)")
    print("  -> Framework says: 2-3 layers, 64-128 neurons, funnel shape.")
    print("  Let's see if the DATA agrees...")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare_data():
    """Load California Housing, scale features, split into train/test."""
    print("\n" + "=" * 70)
    print("LOADING CALIFORNIA HOUSING DATASET")
    print("=" * 70)

    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print(f"\n  Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"  Target: Median house value ($100,000s), range [{y.min():.2f}, {y.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Scaled: StandardScaler")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# PART 2: WIDTH VS DEPTH EXPERIMENT
# ==============================================================================

def build_architecture(name, input_dim):
    """Build one of our 4 experimental architectures."""
    configs = {
        "Wide-Shallow": [256],
        "Balanced":     [64, 64],
        "Deep-Narrow":  [32, 32, 32, 32],
        "Funnel":       [128, 64, 32],
    }
    model = keras.Sequential()
    for i, units in enumerate(configs[name]):
        if i == 0:
            model.add(layers.Dense(units, activation='relu', input_shape=(input_dim,)))
        else:
            model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def width_vs_depth(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Same budget, different house layouts.
    You have 4,000 bricks. Build a bungalow, a 2-story, a tower, or a pyramid.
    Same materials, different shape. Which is strongest?
    """
    print("\n" + "=" * 70)
    print("PART 2: WIDTH VS DEPTH EXPERIMENT")
    print("Same Budget, Different Layouts - Which Architecture Wins?")
    print("=" * 70)
    print()
    print("We build 4 architectures. Same data, optimizer, epochs. Only SHAPE differs.")
    print()

    architectures = ["Wide-Shallow", "Balanced", "Deep-Narrow", "Funnel"]
    descs = [
        "Dense(256) -> Out                         | One massive layer",
        "Dense(64) -> Dense(64) -> Out             | Two medium layers",
        "Dense(32) x4 -> Out                       | Four small layers",
        "Dense(128) -> Dense(64) -> Dense(32) -> Out | Narrowing funnel",
    ]
    for n, d in zip(architectures, descs):
        print(f"  {n:15s}: {d}")
    print()

    results, histories = {}, {}
    for name in architectures:
        print(f"--- Training: {name} ---")
        model = build_architecture(name, X_train.shape[1])
        n_params = model.count_params()
        n_layers = len([l for l in model.layers if isinstance(l, layers.Dense)]) - 1

        start = time.time()
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=100,
                            batch_size=64, verbose=0,
                            callbacks=[callbacks.EarlyStopping(
                                monitor='val_loss', patience=15, restore_best_weights=True)])
        elapsed = time.time() - start
        ep = len(history.history['loss'])

        y_pred = model.predict(X_test, verbose=0).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'layers': n_layers, 'params': n_params, 'mse': mse,
                         'r2': r2, 'time': elapsed, 'epochs': ep, 'y_pred': y_pred}
        histories[name] = history.history
        print(f"  Epoch {ep}, MSE: {mse:.4f}, R2: {r2:.4f}, Time: {elapsed:.1f}s")

    # --- Comparison Table ---
    print("\n" + "-" * 75)
    print(f"{'Architecture':<16} {'Layers':>6} {'Params':>8} {'Test MSE':>10} {'R2':>8} {'Time':>8}")
    print("-" * 75)
    for name in architectures:
        r = results[name]
        print(f"{name:<16} {r['layers']:>6} {r['params']:>8,} {r['mse']:>10.4f} "
              f"{r['r2']:>8.4f} {r['time']:>7.1f}s")
    print("-" * 75)

    best = min(architectures, key=lambda n: results[n]['mse'])
    print(f"\n  WINNER: {best} (lowest test MSE)")
    print("  INSIGHT: Funnel (narrowing) often works best for tabular data.")
    print("  It compresses information progressively - details to essentials.")

    # --- Plot 1: Validation Loss Curves ---
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    plt.figure(figsize=(10, 6))
    for name, c in zip(architectures, colors):
        plt.plot(histories[name]['val_loss'], label=name, color=c, linewidth=2)
    plt.xlabel('Epoch', fontsize=12); plt.ylabel('Validation Loss (MSE)', fontsize=12)
    plt.title('Width vs Depth: Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'width_vs_depth_val_loss.png')
    plt.savefig(p, dpi=120); plt.close()
    print(f"\n  Plot saved: {p}")

    # --- Plot 2: Predicted vs Actual (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name, c in zip(axes.flat, architectures, colors):
        ax.scatter(y_test, results[name]['y_pred'], alpha=0.3, s=8, color=c)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'k--', linewidth=1.5, label='Perfect')
        ax.set_xlabel('Actual ($100k)'); ax.set_ylabel('Predicted ($100k)')
        ax.set_title(f"{name} (R2={results[name]['r2']:.3f})", fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle('Predicted vs Actual: Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(SCRIPT_DIR, 'width_vs_depth_pred_vs_actual.png')
    plt.savefig(p2, dpi=120); plt.close()
    print(f"  Plot saved: {p2}")


# ==============================================================================
# PART 3: CALLBACKS MASTERCLASS
# ==============================================================================

def callbacks_masterclass(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Callbacks are your TRAINING AUTOPILOT.
    - EarlyStopping:     "Pull over if we're going in circles."
    - ModelCheckpoint:   "Save a photo of the best scenic view."
    - ReduceLROnPlateau: "Slow down when the road gets tricky."
    """
    print("\n" + "=" * 70)
    print("PART 3: CALLBACKS MASTERCLASS")
    print("Your Training Autopilot")
    print("=" * 70)
    print()
    print("Callbacks are functions Keras calls AUTOMATICALLY during training.")
    print("They monitor metrics, make decisions, save you from babysitting.")
    print()

    # --- 3A: EarlyStopping ---
    print("-" * 60)
    print("CALLBACK 1: EarlyStopping")
    print("'Stop when validation loss stops improving. Don't waste time.'")
    print("-" * 60)
    print()
    print("  Without: trains all 150 epochs, overfits after epoch 50.")
    print("  With patience=15: stops at ~65, restores best weights from epoch 50.")
    print()

    def build_model():
        m = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)])
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return m

    # WITHOUT EarlyStopping
    print("  Training WITHOUT EarlyStopping (150 epochs)...")
    model_no = build_model()
    t0 = time.time()
    h_no = model_no.fit(X_train, y_train, validation_split=0.2, epochs=150,
                         batch_size=64, verbose=0)
    t_no = time.time() - t0
    mse_no = mean_squared_error(y_test, model_no.predict(X_test, verbose=0).flatten())
    print(f"  -> 150 epochs, MSE: {mse_no:.4f}, Time: {t_no:.1f}s")

    # WITH EarlyStopping
    print("  Training WITH EarlyStopping (patience=15)...")
    model_es = build_model()
    t0 = time.time()
    h_es = model_es.fit(X_train, y_train, validation_split=0.2, epochs=150,
                         batch_size=64, verbose=0,
                         callbacks=[callbacks.EarlyStopping(
                             monitor='val_loss', patience=15, restore_best_weights=True)])
    t_es = time.time() - t0
    ep_es = len(h_es.history['loss'])
    mse_es = mean_squared_error(y_test, model_es.predict(X_test, verbose=0).flatten())
    print(f"  -> Stopped at epoch {ep_es}, MSE: {mse_es:.4f}, Time: {t_es:.1f}s")
    print(f"  TIME SAVED: {t_no - t_es:.1f}s ({(1 - t_es/t_no)*100:.0f}% faster)")
    print(f"  EarlyStopping: saves time AND often gets better results!")

    # --- 3B: ModelCheckpoint ---
    print()
    print("-" * 60)
    print("CALLBACK 2: ModelCheckpoint")
    print("'Save your best model, even if training goes bad later.'")
    print("-" * 60)
    print("  Like hiking and saving photos: keep the best view automatically.")
    checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model.keras')
    print(f"  Will save to: {checkpoint_path}")

    # --- 3C: ReduceLROnPlateau ---
    print()
    print("-" * 60)
    print("CALLBACK 3: ReduceLROnPlateau")
    print("'When stuck, try smaller steps.'")
    print("-" * 60)
    print("  Searching for a coin: BIG steps at first, TINY steps when close.")
    print("  Halves learning rate after 5 epochs with no improvement.")
    print()

    # --- 3D: ALL THREE TOGETHER ---
    print("-" * 60)
    print("ALL THREE CALLBACKS WORKING TOGETHER")
    print("-" * 60)

    model_all = build_model()
    lr_history = []

    class LRTracker(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr_history.append(float(self.model.optimizer.learning_rate))

    print("  Training with all 3 callbacks (verbose=1)...\n")
    h_all = model_all.fit(
        X_train, y_train, validation_split=0.2, epochs=200, batch_size=64, verbose=1,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                     save_best_only=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=5, min_lr=1e-6, verbose=0),
            LRTracker()])

    ep_all = len(h_all.history['loss'])
    mse_all = mean_squared_error(y_test, model_all.predict(X_test, verbose=0).flatten())
    print(f"\n  Stopped at epoch {ep_all}, Test MSE: {mse_all:.4f}")

    # --- Plot: Training Curves + LR ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(h_all.history['loss'], label='Train', color='#3498db', linewidth=2)
    ax1.plot(h_all.history['val_loss'], label='Val', color='#e74c3c', linewidth=2)
    best_ep = np.argmin(h_all.history['val_loss']) + 1
    ax1.axvline(x=best_ep, color='green', ls='--', alpha=0.7, label=f'Best ({best_ep})')
    ax1.axvline(x=ep_all, color='red', ls=':', alpha=0.7, label=f'Stop ({ep_all})')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Curves with Callbacks', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    if lr_history:
        ax2.plot(lr_history, color='#9b59b6', linewidth=2)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate (ReduceLROnPlateau)', fontweight='bold')
        ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = os.path.join(SCRIPT_DIR, 'callbacks_training_curves.png')
    plt.savefig(p3, dpi=120); plt.close()
    print(f"  Plot saved: {p3}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Cleaned up: {checkpoint_path}")


# ==============================================================================
# PART 4: CUSTOM CALLBACKS
# ==============================================================================

def custom_callbacks_demo(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Factory floor monitors.
    Standard safety systems exist (smoke alarms = built-in callbacks).
    Custom monitors = your own gauges and clocks on the training floor.
    """
    print("\n" + "=" * 70)
    print("PART 4: CUSTOM CALLBACKS")
    print("Build Your Own Training Monitors")
    print("=" * 70)
    print()
    print("Subclass keras.callbacks.Callback and override on_epoch_end, etc.")
    print()

    class TrainingMonitor(keras.callbacks.Callback):
        """Prints a clean summary every N epochs."""
        def __init__(self, print_every=10):
            super().__init__()
            self.print_every = print_every
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.print_every == 0:
                print(f"    Epoch {epoch:>3d}: loss={logs['loss']:.4f}, "
                      f"val_loss={logs['val_loss']:.4f}, mae={logs['mae']:.4f}")

    class TimeEstimator(keras.callbacks.Callback):
        """Estimates and prints remaining training time."""
        def __init__(self, total_epochs, print_every=20):
            super().__init__()
            self.total_epochs = total_epochs
            self.print_every = print_every
            self.start_time = None
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print(f"    Timer started. Target: {self.total_epochs} epochs.")
        def on_epoch_end(self, epoch, logs=None):
            if epoch > 0 and epoch % self.print_every == 0:
                elapsed = time.time() - self.start_time
                per_ep = elapsed / (epoch + 1)
                remaining = per_ep * (self.total_epochs - epoch - 1)
                print(f"    [TIME] Epoch {epoch}: {elapsed:.1f}s elapsed, "
                      f"~{remaining:.1f}s remaining ({per_ep:.2f}s/epoch)")
        def on_train_end(self, logs=None):
            print(f"    [TIME] Complete in {time.time() - self.start_time:.1f}s.")

    print("  Training with BOTH custom callbacks (80 epochs)...\n")
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, validation_split=0.2, epochs=80, batch_size=64,
              verbose=0, callbacks=[
                  TrainingMonitor(print_every=10),
                  TimeEstimator(total_epochs=80, print_every=20),
                  callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                          restore_best_weights=True)])

    y_pred = model.predict(X_test, verbose=0).flatten()
    print(f"\n  Final: MSE={mean_squared_error(y_test, y_pred):.4f}, "
          f"R2={r2_score(y_test, y_pred):.4f}")
    print("  Custom callbacks give you full control over the training experience!")


# ==============================================================================
# PART 5: THE COMPLETE PIPELINE TEMPLATE
# ==============================================================================

def complete_pipeline(X_train, X_test, y_train, y_test):
    """
    ANALOGY: An assembly line in a factory.
    Frame -> Engine -> Wiring -> Body -> Paint -> Quality check.
    Every step depends on the previous one. This is your 8-step ML assembly line.
    """
    print("\n" + "=" * 70)
    print("PART 5: THE COMPLETE PIPELINE TEMPLATE")
    print("Everything from Sessions 41-42 in ONE Pipeline")
    print("=" * 70)
    print()
    print("8 steps. No shortcuts. Each one matters.")
    print()

    # STEP 1-2: EXPLORE & PREPROCESS
    print("STEP 1-2: EXPLORE & PREPROCESS")
    print("-" * 40)
    print(f"  {X_train.shape[0]} train | {X_test.shape[0]} test | {X_train.shape[1]} features")
    print(f"  Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print("  Scaled with StandardScaler (fit on train only)")
    print()

    # STEP 3: BUILD
    print("STEP 3: BUILD (Funnel + Dropout + BatchNorm)")
    print("-" * 40)
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)])
    model.summary()

    # STEP 4: COMPILE
    print("\nSTEP 4: COMPILE (Adam + MSE)")
    print("-" * 40)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    print("  Adam(lr=0.001) | MSE loss | MAE metric")

    # STEP 5: CALLBACKS
    print("\nSTEP 5: CALLBACKS")
    print("-" * 40)
    ckpt = os.path.join(SCRIPT_DIR, 'pipeline_best.keras')
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
          callbacks.ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=0)]
    print("  EarlyStopping(patience=15) + ReduceLR(factor=0.5) + ModelCheckpoint")

    # STEP 6: TRAIN
    print("\nSTEP 6: TRAIN (verbose=1)")
    print("-" * 40)
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=200,
                         batch_size=64, verbose=1, callbacks=cb)
    train_time = time.time() - t0
    n_ep = len(history.history['loss'])
    print(f"\n  Done: {n_ep} epochs in {train_time:.1f}s")

    # STEP 7: EVALUATE
    print("\nSTEP 7: EVALUATE")
    print("-" * 40)
    y_pred = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}  (avg error in $100k)")
    print(f"  R2   = {r2:.4f}  (1.0=perfect, 0.0=guessing mean)")
    print(f"  In plain English: predictions off by ~${mae*100000:.0f} on average.")

    # STEP 8: VISUALIZE
    print("\nSTEP 8: VISUALIZE")
    print("-" * 40)

    # Plot A: Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train', color='#3498db', lw=2)
    axes[0].plot(history.history['val_loss'], label='Val', color='#e74c3c', lw=2)
    best_ep = np.argmin(history.history['val_loss']) + 1
    axes[0].axvline(x=best_ep, color='green', ls='--', alpha=0.6, label=f'Best ({best_ep})')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train', color='#3498db', lw=2)
    axes[1].plot(history.history['val_mae'], label='Val', color='#e74c3c', lw=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Curves', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle('Complete Pipeline: Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(SCRIPT_DIR, 'pipeline_training_curves.png')
    plt.savefig(p1, dpi=120); plt.close()
    print(f"  Saved: {p1}")

    # Plot B: Predicted vs Actual
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='#3498db')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect')
    plt.xlabel('Actual ($100k)'); plt.ylabel('Predicted ($100k)')
    plt.title(f'Predicted vs Actual (R2={r2:.3f})', fontsize=14, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p2 = os.path.join(SCRIPT_DIR, 'pipeline_predicted_vs_actual.png')
    plt.savefig(p2, dpi=120); plt.close()
    print(f"  Saved: {p2}")

    # Plot C: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
    plt.axvline(x=0, color='red', ls='--', lw=2, label='Zero Error')
    plt.xlabel('Residual (Actual - Predicted)'); plt.ylabel('Count')
    plt.title('Residual Distribution (Centered at 0 = Good)', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    p3 = os.path.join(SCRIPT_DIR, 'pipeline_residual_distribution.png')
    plt.savefig(p3, dpi=120); plt.close()
    print(f"  Saved: {p3}")

    print("\n  PIPELINE COMPLETE! 8 steps. Systematic. Reproducible. Professional.")
    if os.path.exists(ckpt):
        os.remove(ckpt)


# ==============================================================================
# PART 6: SESSIONS 41-42 SUMMARY REFERENCE CARD
# ==============================================================================

def summary_reference_card():
    """The cheat sheet you tape to your monitor."""
    print("\n" + "=" * 65)
    print("  SESSIONS 41-42 PRACTITIONER'S REFERENCE CARD")
    print("=" * 65)
    print("""
  PROBLEM -> OUTPUT LAYER:
    Binary classification  -> 1 neuron, sigmoid, binary_crossentropy
    Multiclass (N classes) -> N neurons, softmax, sparse_categorical_crossentropy
    Regression             -> 1 neuron, linear, mse

  ARCHITECTURE RULES:
    Start simple -> Add complexity ONLY when validation improves
    Funnel pattern for tabular: Wide -> Narrow -> Output
    <1k samples: 1-2 layers | 1k-100k: 2-3 layers | >100k: 3+

  ACTIVATION FUNCTIONS:
    Hidden: ReLU (default), LeakyReLU (dying neuron fix)
    Output: sigmoid (binary), softmax (multiclass), linear (regression)

  REGULARIZATION ORDER:
    1. Dropout (0.2-0.5)  2. L2 (0.001-0.01)  3. BatchNorm  4. Less capacity

  OPTIMIZER DEFAULT: Adam(learning_rate=0.001)
    Stuck? ReduceLROnPlateau(factor=0.5, patience=5)

  CALLBACKS (always use):
    EarlyStopping(patience=10-20, restore_best_weights=True)
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ModelCheckpoint('best.keras', save_best_only=True)

  PREPROCESSING:
    ALWAYS scale with StandardScaler (fit on train, transform both)
    NEVER fit on full dataset (data leakage!)

  DEBUGGING:
    Loss flat?             -> Check learning rate, data scaling
    Val >> Train loss?     -> Overfitting: add dropout, reduce capacity
    Both losses high?      -> Underfitting: more capacity
    NaN loss?              -> Lower learning rate, check data for NaN""")
    print()
    print("-" * 65)
    print("  Coming next in Session 43:")
    print("  Functional API: multi-input models, skip connections, custom layers")
    print("-" * 65)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("ARCHITECTURE DESIGN LAB")
    print("How Many Layers? How Many Neurons? Callbacks Mastery")
    print("=" * 70)

    architecture_framework()
    input("\nPress Enter for the Width vs Depth experiment...")

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    width_vs_depth(X_train, X_test, y_train, y_test)
    input("\nPress Enter for Callbacks Masterclass...")

    callbacks_masterclass(X_train, X_test, y_train, y_test)
    input("\nPress Enter for Custom Callbacks...")

    custom_callbacks_demo(X_train, X_test, y_train, y_test)
    input("\nPress Enter for the Complete Pipeline Template...")

    complete_pipeline(X_train, X_test, y_train, y_test)
    input("\nPress Enter for the Summary Reference Card...")

    summary_reference_card()

    print("\n" + "=" * 70)
    print("SESSIONS 41-42 COMPLETE!")
    print("=" * 70)
    print()
    print("You now have a complete toolkit for tabular deep learning:")
    print("  - Architecture framework (3 Questions)")
    print("  - Width vs Depth experiment results")
    print("  - Callbacks mastery (EarlyStopping, Checkpoint, ReduceLR)")
    print("  - Custom callbacks for monitoring")
    print("  - Complete 8-step pipeline template")
    print("  - Reference card for quick lookup")
    print()
    print("Next session: The Functional API opens doors to ANYTHING.")


if __name__ == "__main__":
    main()
