"""
================================================================================
THE PRACTITIONER'S TOOLKIT: Overfitting, Regularization, and Optimizers
================================================================================

Course: The Art of Programming - Keras Mastery (Sessions 41-42)
Lesson: From Theory to Practice -- Fighting Overfitting and Choosing Optimizers

PREREQUISITES:
    - Script 1: Keras Decoded (loss, activations, training curves)
    - Script 2: Multiclass & Regression (problem types, scaling)

THE CHALLENGE:
    Your model gets 99% on training data but 60% on test data.
    It MEMORIZED the answers instead of LEARNING the patterns.

    This is the #1 problem in deep learning: OVERFITTING.

    Today you learn the weapons against it:
    - Dropout (randomly disable neurons)
    - L2 Regularization (penalize large weights)
    - BatchNorm (stabilize training)
    - Optimizer choice (SGD vs Adam vs RMSProp)
    - Learning rate tuning

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import fetch_california_housing, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# PART 1: THE MEMORIZER VS THE LEARNER
# ==============================================================================

def memorizer_vs_learner():
    """
    ANALOGY: Two students prepare for an exam.
    Student A memorizes every answer (OVERFITTING).
    Student B understands the concepts (GENERALIZATION).
    Give them a NEW question -- Student B wins every time.
    """
    print("=" * 70)
    print("PART 1: THE MEMORIZER VS THE LEARNER")
    print("=" * 70)
    print()
    print("Two students study for an exam.")
    print("  Student A: Memorizes every answer word-for-word.")
    print("             Textbook question? 100%. New question? Lost.")
    print("  Student B: Understands the concepts.")
    print("             Textbook question? 85%. New question? 82%.")
    print()
    print("Student A is OVERFITTING. Student B is GENERALIZING.")
    print("Let's see this in action with neural networks.\n")

    # --- make_moons: small dataset, easy to memorize ---
    print("Generating make_moons dataset (200 samples, noise=0.2)")
    print("Small dataset = easy to memorize, hard to generalize\n")
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"  Training: {X_train.shape[0]}  |  Validation: {X_val.shape[0]}  |  Features: 2\n")

    # --- THE MEMORIZER: way too much capacity ---
    print("MODEL 1: THE MEMORIZER (4 x 128 neurons on 140 points!)")
    overfit_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(2,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    overfit_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training the Memorizer (100 epochs)...")
    hist_over = overfit_model.fit(X_train, y_train, epochs=100, batch_size=16,
                                  validation_data=(X_val, y_val), verbose=1)

    # --- THE UNDERFITTER: not enough capacity ---
    print("\nMODEL 2: THE UNDERFITTER (1 x 8 neurons)")
    simple_model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(2,)),
        layers.Dense(1, activation='sigmoid')
    ])
    simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training the Underfitter (100 epochs)...")
    hist_simple = simple_model.fit(X_train, y_train, epochs=100, batch_size=16,
                                   validation_data=(X_val, y_val), verbose=1)

    # --- Decision boundary plots ---
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                          np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model, title in zip(axes, [overfit_model, simple_model],
                                 ["MEMORIZER (Overfit)", "UNDERFITTER (Too Simple)"]):
        preds = model.predict(grid, verbose=0).reshape(xx.shape)
        ax.contourf(xx, yy, preds, levels=50, cmap='RdBu', alpha=0.7)
        ax.scatter(X_val[y_val == 0, 0], X_val[y_val == 0, 1],
                   c='blue', edgecolors='k', s=40, label='Class 0')
        ax.scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1],
                   c='red', edgecolors='k', s=40, label='Class 1')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
    plt.suptitle("Decision Boundaries: Overfitting vs Underfitting", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part1_decision_boundaries.png'), dpi=150)
    plt.close()
    print("\n  Saved: part1_decision_boundaries.png")

    # --- Loss curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, title in zip(axes, [hist_over, hist_simple],
                                ["MEMORIZER: Loss Curves", "UNDERFITTER: Loss Curves"]):
        ax.plot(hist.history['loss'], label='Train Loss', linewidth=2)
        ax.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Training vs Validation Loss", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part1_loss_curves.png'), dpi=150)
    plt.close()
    print("  Saved: part1_loss_curves.png\n")

    # --- Diagnosis ---
    print("DIAGNOSIS:")
    print(f"  Memorizer  -> Train: {hist_over.history['loss'][-1]:.4f}  Val: {hist_over.history['val_loss'][-1]:.4f}")
    print(f"  Underfitter -> Train: {hist_simple.history['loss'][-1]:.4f}  Val: {hist_simple.history['val_loss'][-1]:.4f}")
    print()
    print("  OVERFITTING  = Train loss WAY down, Val loss UP. Gap is HUGE.")
    print("  UNDERFITTING = Both losses stay HIGH. Cannot even learn training data.")
    print("  GOOD FIT     = Both losses go down together and stay close.")


# ==============================================================================
# PART 2: CALIFORNIA HOUSING -- OUR LABORATORY
# ==============================================================================

def california_housing_lab():
    """
    ANALOGY: A medical laboratory. Before testing any drug, you need
    a well-understood patient population and baseline measurements.
    California Housing is our patient population: 20,640 houses, 8 features.
    """
    print("\n" + "=" * 70)
    print("PART 2: CALIFORNIA HOUSING -- OUR LABORATORY")
    print("=" * 70)
    print()
    print("Before testing any treatment, a doctor needs baseline measurements.")
    print("Before testing regularization, WE need a baseline model.\n")

    print("Loading California Housing dataset...")
    print("  (First run may download ~600KB from scikit-learn)")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    print(f"\n  Samples: {X.shape[0]:,}  |  Features: {X.shape[1]}")
    print(f"  Feature names:")
    for i, name in enumerate(housing.feature_names):
        print(f"    [{i}] {name:12s}  range=[{X[:, i].min():.1f}, {X[:, i].max():.1f}]  mean={X[:, i].mean():.2f}")
    print(f"\n  Target (median house value, $100k): mean={y.mean():.2f}  std={y.std():.2f}")

    # --- Price histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel("Median House Value ($100k)", fontsize=12)
    plt.ylabel("Number of Districts", fontsize=12)
    plt.title("California Housing: Price Distribution", fontsize=14, fontweight='bold')
    plt.axvline(y.mean(), color='red', linestyle='--', lw=2, label=f'Mean: ${y.mean()*100:.0f}k')
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part2_price_distribution.png'), dpi=150)
    plt.close()
    print("\n  Saved: part2_price_distribution.png")

    # --- Feature correlations ---
    print("\n  Feature correlations with house price:")
    for i, name in enumerate(housing.feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        bar = ("+" if corr > 0 else "-") * int(abs(corr) * 30)
        print(f"    {name:12s}  r={corr:+.3f}  {bar}")

    # --- Scale and split ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\n  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  (scaled with StandardScaler)")

    # --- Baseline model ---
    print("\n" + "=" * 50)
    print("BASELINE MODEL: 2 x Dense(64), no tricks")
    print("=" * 50)
    baseline = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    baseline.compile(optimizer='adam', loss='mse', metrics=['mae'])
    baseline.summary()
    print("\nTraining baseline (50 epochs)...")
    baseline.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.15, verbose=1)

    y_pred = baseline.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nBASELINE: MSE={mse:.4f}  MAE={mae:.4f} (${mae*100:.0f}k error)  R2={r2:.4f}")
    print("This is our benchmark. Every technique gets compared to this.\n")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# HELPER: Evaluate and print metrics
# ==============================================================================

def _eval_model(model, X_test, y_test, label):
    """Evaluate a model and return dict of metrics."""
    pred = model.predict(X_test, verbose=0).flatten()
    return {
        'label': label,
        'mse': mean_squared_error(y_test, pred),
        'mae': mean_absolute_error(y_test, pred),
        'r2': r2_score(y_test, pred),
    }

def _print_comparison(results):
    """Print a comparison table from list of result dicts."""
    print("-" * 55)
    print(f"  {'Model':<20s} {'MSE':>8s} {'MAE':>8s} {'R2':>8s}")
    print("-" * 55)
    for r in results:
        print(f"  {r['label']:<20s} {r['mse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}")
    print("-" * 55)

def _plot_two_histories(h1, h2, t1, t2, suptitle, filename):
    """Plot side-by-side loss curves for two histories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, h, t in zip(axes, [h1, h2], [t1, t2]):
        ax.plot(h.history['loss'], label='Train Loss', linewidth=2)
        ax.plot(h.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(suptitle, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# PART 3: DROPOUT EXPERIMENT
# ==============================================================================

def dropout_experiment(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Studying without your notes.
    Dropout randomly turns OFF 30% of neurons each training step.
    The network CANNOT rely on any single neuron -- it must spread
    knowledge across many neurons. Result: better generalization.
    """
    print("=" * 70)
    print("PART 3: DROPOUT -- Studying Without Your Notes")
    print("=" * 70)
    print()
    print("Imagine studying with 10 friends. You always rely on the same 2-3.")
    print("DROPOUT: Each session, 3 random friends are ABSENT.")
    print("Now you CANNOT rely on specific friends. You must actually learn.\n")
    print("For neural networks:")
    print("  - Each step, randomly set 30% of neurons to 0")
    print("  - Forces redundant representations")
    print("  - At test time: all neurons active (scaled appropriately)\n")

    # --- No Dropout ---
    print("Model A: NO Dropout (3 x 128 neurons)")
    no_drop = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    no_drop.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_no = no_drop.fit(X_train, y_train, epochs=80, batch_size=64,
                        validation_split=0.15, verbose=1)

    # --- With Dropout ---
    print("\nModel B: WITH Dropout(0.3) after each hidden layer")
    w_drop = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    w_drop.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_dr = w_drop.fit(X_train, y_train, epochs=80, batch_size=64,
                       validation_split=0.15, verbose=1)

    _plot_two_histories(h_no, h_dr, "NO Dropout", "WITH Dropout(0.3)",
                        "Dropout Experiment", 'part3_dropout_comparison.png')

    r1 = _eval_model(no_drop, X_test, y_test, "No Dropout")
    r2 = _eval_model(w_drop, X_test, y_test, "Dropout(0.3)")
    print("\nDROPOUT COMPARISON (test data):")
    _print_comparison([r1, r2])

    gap_no = abs(h_no.history['loss'][-1] - h_no.history['val_loss'][-1])
    gap_dr = abs(h_dr.history['loss'][-1] - h_dr.history['val_loss'][-1])
    print(f"\n  Train-Val gap without dropout: {gap_no:.4f}")
    print(f"  Train-Val gap with dropout:    {gap_dr:.4f}")
    if gap_dr < gap_no:
        print("  -> Dropout REDUCED the gap. Less overfitting!")
    print("\nPRACTICAL GUIDE:")
    print("  0.2 = light  |  0.3 = typical (start here)  |  0.5 = aggressive  |  >0.5 = too much\n")


# ==============================================================================
# PART 4: L2 REGULARIZATION
# ==============================================================================

def l2_regularization(X_train, X_test, y_train, y_test):
    """
    ANALOGY: A tax on large weights.
    Buildings with no size limit grow MASSIVE. A "size tax" shrinks them.
    L2 adds a penalty: Loss_total = Loss_data + lambda * SUM(weights^2).
    Smaller weights = simpler model = less overfitting.
    """
    print("=" * 70)
    print("PART 4: L2 REGULARIZATION -- The Weight Tax")
    print("=" * 70)
    print()
    print("Without regulation, weights can grow as large as they want.")
    print("Large weights = extreme sensitivity = overfitting.\n")
    print("L2 adds a PENALTY:  Loss_total = Loss_data + lambda * SUM(weight^2)")
    print("lambda=0.01: for every unit of weight^2, add 0.01 to the loss.\n")

    # --- No L2 ---
    print("Model A: NO Regularization (3 x 128)")
    no_reg = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    no_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_no = no_reg.fit(X_train, y_train, epochs=80, batch_size=64,
                       validation_split=0.15, verbose=1)

    # --- With L2 ---
    print("\nModel B: WITH L2(0.01) on every hidden layer")
    w_l2 = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1)
    ])
    w_l2.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_l2 = w_l2.fit(X_train, y_train, epochs=80, batch_size=64,
                     validation_split=0.15, verbose=1)

    _plot_two_histories(h_no, h_l2, "NO Regularization", "WITH L2(0.01)",
                        "L2 Regularization Experiment", 'part4_l2_comparison.png')

    # --- Weight distribution histograms ---
    w_no = np.concatenate([l.get_weights()[0].flatten() for l in no_reg.layers if l.get_weights()])
    w_l2_arr = np.concatenate([l.get_weights()[0].flatten() for l in w_l2.layers if l.get_weights()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(w_no, bins=80, color='salmon', edgecolor='black', alpha=0.8)
    axes[0].set_title("NO Regularization", fontsize=12, fontweight='bold')
    axes[0].axvline(0, color='black', linestyle='--'); axes[0].set_xlim(-1.5, 1.5)
    axes[1].hist(w_l2_arr, bins=80, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].set_title("WITH L2", fontsize=12, fontweight='bold')
    axes[1].axvline(0, color='black', linestyle='--'); axes[1].set_xlim(-1.5, 1.5)
    for ax in axes:
        ax.set_xlabel("Weight Value"); ax.set_ylabel("Count")
    plt.suptitle("L2 Pushes Weights Toward Zero", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part4_weight_distributions.png'), dpi=150)
    plt.close()
    print("  Saved: part4_weight_distributions.png")

    print(f"\n  Weight stats:")
    print(f"    No Reg -> std={w_no.std():.4f}  max|w|={np.abs(w_no).max():.4f}")
    print(f"    L2     -> std={w_l2_arr.std():.4f}  max|w|={np.abs(w_l2_arr).max():.4f}")

    r1 = _eval_model(no_reg, X_test, y_test, "No Regularization")
    r2 = _eval_model(w_l2, X_test, y_test, "L2(0.01)")
    print("\nL2 COMPARISON (test data):")
    _print_comparison([r1, r2])
    print("\nKEY INSIGHT: L2 pushes weights toward 0.")
    print("  Smaller weights = simpler boundary = less overfitting.\n")


# ==============================================================================
# PART 5: OPTIMIZER SHOWDOWN
# ==============================================================================

def optimizer_showdown(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Three chefs, same kitchen, same ingredients.
    Chef SGD: turns the dial by a fixed amount (slow, reliable).
    Chef Adam: adapts each burner individually (fast, adaptive).
    Chef RMSProp: Adam's cousin (popular for recurrent nets).
    """
    print("=" * 70)
    print("PART 5: OPTIMIZER SHOWDOWN -- Three Chefs, Same Kitchen")
    print("=" * 70)
    print()
    print("  Chef SGD:     Fixed step size. Slow and steady.")
    print("  Chef Adam:    Adaptive per-parameter. Fast learner. DEFAULT CHOICE.")
    print("  Chef RMSProp: Also adaptive. Popular for RNNs.\n")

    def build_model(opt):
        m = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        m.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return m

    optimizers = {
        'SGD':     keras.optimizers.SGD(learning_rate=0.01),
        'Adam':    keras.optimizers.Adam(learning_rate=0.001),
        'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    }
    results, histories = {}, {}

    for name, opt in optimizers.items():
        print(f"--- Training with {name} ---")
        model = build_model(opt)
        t0 = time.time()
        hist = model.fit(X_train, y_train, epochs=50, batch_size=64,
                         validation_split=0.15, verbose=1)
        elapsed = time.time() - t0
        r = _eval_model(model, X_test, y_test, name)
        r['time'] = elapsed
        results[name] = r
        histories[name] = hist
        print()

    # --- All optimizers on one plot ---
    plt.figure(figsize=(10, 6))
    colors = {'SGD': '#e74c3c', 'Adam': '#2ecc71', 'RMSprop': '#3498db'}
    for name, hist in histories.items():
        plt.plot(hist.history['val_loss'], label=name, linewidth=2.5, color=colors[name])
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss (MSE)")
    plt.title("Optimizer Showdown: Who Converges Fastest?", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part5_optimizer_showdown.png'), dpi=150)
    plt.close()
    print("  Saved: part5_optimizer_showdown.png")

    print("\nOPTIMIZER COMPARISON:")
    print("-" * 65)
    print(f"  {'Optimizer':<12s} {'MSE':>10s} {'MAE':>10s} {'R2':>8s} {'Time':>10s}")
    print("-" * 65)
    for r in results.values():
        print(f"  {r['label']:<12s} {r['mse']:>10.4f} {r['mae']:>10.4f} {r['r2']:>8.4f} {r['time']:>9.1f}s")
    print("-" * 65)

    # === Learning Rate Impact ===
    print("\n" + "=" * 70)
    print("LEARNING RATE: The Most Important Hyperparameter")
    print("=" * 70)
    print()
    print("  Too HIGH (0.1):    Overshoots. Loss bounces wildly.")
    print("  Just RIGHT (0.001): Smooth convergence.")
    print("  Too LOW (0.00001): Barely moves. Takes forever.\n")

    lr_configs = {'lr=0.1 (TOO HIGH)': 0.1, 'lr=0.001 (JUST RIGHT)': 0.001,
                  'lr=0.00001 (TOO LOW)': 0.00001}
    lr_histories = {}
    for label, lr in lr_configs.items():
        print(f"  Training Adam with {label}...")
        model = build_model(keras.optimizers.Adam(learning_rate=lr))
        lr_histories[label] = model.fit(X_train, y_train, epochs=50, batch_size=64,
                                        validation_split=0.15, verbose=1)
        print()

    plt.figure(figsize=(10, 6))
    for (label, hist), c in zip(lr_histories.items(), ['#e74c3c', '#2ecc71', '#3498db']):
        plt.plot(hist.history['val_loss'], label=label, linewidth=2.5, color=c)
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss (MSE)")
    plt.title("Learning Rate: Too High vs Just Right vs Too Low", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
    plt.ylim(0, min(5.0, max(max(lr_histories['lr=0.1 (TOO HIGH)'].history['val_loss']), 3.0)))
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'part5_learning_rate_impact.png'), dpi=150)
    plt.close()
    print("  Saved: part5_learning_rate_impact.png\n")

    print("GOLDEN RULES:")
    print("  1. Adam is the DEFAULT. Start with lr=0.001.")
    print("  2. SGD + momentum(0.9) for fine control.")
    print("  3. Learning rate is THE most important hyperparameter.")
    print("  4. Loss explodes? REDUCE lr.  Loss plateaus? INCREASE lr slightly.\n")


# ==============================================================================
# PART 6: BATCH NORMALIZATION
# ==============================================================================

def batch_normalization(X_train, X_test, y_train, y_test):
    """
    ANALOGY: Recalibrating instruments between measurements.
    As data flows through deep networks, value distributions SHIFT.
    BatchNorm normalizes outputs at each layer: mean=0, std=1,
    then applies learned scale and shift. Training becomes stable.
    """
    print("=" * 70)
    print("PART 6: BATCH NORMALIZATION -- Recalibrating the Instruments")
    print("=" * 70)
    print()
    print("Problem: In deep networks, value distributions SHIFT at each layer.")
    print("Layer 5 sees very different ranges than Layer 1. Training becomes unstable.\n")
    print("Solution: BatchNormalization after each Dense layer.")
    print("  1. Normalize: (x - mean) / std")
    print("  2. Apply learned scale (gamma) and shift (beta)")
    print("  3. Result: stable values at every layer\n")

    # --- Without BatchNorm ---
    print("Model A: WITHOUT BatchNorm (4 x 128)")
    no_bn = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    no_bn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_no = no_bn.fit(X_train, y_train, epochs=80, batch_size=64,
                      validation_split=0.15, verbose=1)

    # --- With BatchNorm ---
    print("\nModel B: WITH BatchNorm (Dense -> BN -> ReLU)")
    w_bn = keras.Sequential([
        layers.Dense(128, input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Dense(1)
    ])
    w_bn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    h_bn = w_bn.fit(X_train, y_train, epochs=80, batch_size=64,
                     validation_split=0.15, verbose=1)

    _plot_two_histories(h_no, h_bn, "WITHOUT BatchNorm", "WITH BatchNorm",
                        "Batch Normalization: Stability Comparison", 'part6_batchnorm_comparison.png')

    r1 = _eval_model(no_bn, X_test, y_test, "No BatchNorm")
    r2 = _eval_model(w_bn, X_test, y_test, "With BatchNorm")
    print("\nBATCH NORMALIZATION COMPARISON (test data):")
    _print_comparison([r1, r2])

    # --- Smoothness analysis ---
    smooth_no = np.mean(np.abs(np.diff(h_no.history['val_loss'])))
    smooth_bn = np.mean(np.abs(np.diff(h_bn.history['val_loss'])))
    print(f"\n  Curve smoothness (avg epoch-to-epoch change):")
    print(f"    Without BN: {smooth_no:.4f}  |  With BN: {smooth_bn:.4f}")
    if smooth_bn < smooth_no:
        print("    -> BatchNorm produces SMOOTHER training!")
    print("\nWHEN TO USE: Deep networks (4+), unstable training, place AFTER Dense BEFORE Activation.\n")


# ==============================================================================
# PART 7: REGULARIZATION HIERARCHY
# ==============================================================================

def regularization_hierarchy():
    """
    A decision guide for practitioners. When your model overfits,
    follow this hierarchy from easiest to hardest interventions.
    """
    print("=" * 70)
    print("PART 7: THE REGULARIZATION DECISION GUIDE")
    print("=" * 70)
    print()
    print("You trained a model. It overfits. What do you do?")
    print("Follow this hierarchy, from easiest to hardest:\n")
    print("""
    Step 1: IS YOUR MODEL ACTUALLY OVERFITTING?
            Train loss << Val loss?  YES -> Overfitting.
            Both losses HIGH?        NO  -> Underfitting. ADD capacity first.

    Step 2: TRY DROPOUT (easiest, most common)
            model.add(Dropout(0.3))   # after hidden layers

    Step 3: ADD L2 REGULARIZATION
            Dense(128, kernel_regularizer=regularizers.l2(0.01))

    Step 4: ADD BATCH NORMALIZATION (for stability)
            Dense(128) -> BatchNormalization() -> Activation('relu')

    Step 5: REDUCE MODEL SIZE
            Fewer layers, fewer neurons per layer.

    Step 6: NUCLEAR OPTION -- GET MORE DATA
            More data > any regularization trick.

    COMBO RECIPE (what practitioners actually use):
        Dense(128, kernel_regularizer=l2(0.001))
        BatchNormalization()
        Activation('relu')
        Dropout(0.3)
        Dense(1)  # output -- NO regularization here
    """)

    print("SUMMARY TABLE:")
    print("-" * 65)
    print(f"  {'Technique':<20s} {'Effect':<25s} {'When to Use'}")
    print("-" * 65)
    print(f"  {'Dropout':<20s} {'Random neuron disabling':<25s} First line of defense")
    print(f"  {'L2 Regularization':<20s} {'Penalizes large weights':<25s} With or after dropout")
    print(f"  {'BatchNorm':<20s} {'Stabilizes activations':<25s} Deep networks (4+)")
    print(f"  {'Reduce Size':<20s} {'Less capacity':<25s} When all else fails")
    print(f"  {'More Data':<20s} {'Better generalization':<25s} Always helps")
    print("-" * 65)
    print()
    print("OPTIMIZER QUICK REFERENCE:")
    print("-" * 65)
    print(f"  {'Optimizer':<15s} {'Learning Rate':<15s} {'Best For'}")
    print("-" * 65)
    print(f"  {'Adam':<15s} {'0.001':<15s} Default choice. Works on most problems.")
    print(f"  {'SGD+Momentum':<15s} {'0.01':<15s} Fine-tuning. Often best final result.")
    print(f"  {'RMSprop':<15s} {'0.001':<15s} Recurrent networks (LSTM, GRU).")
    print("-" * 65)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("THE PRACTITIONER'S TOOLKIT")
    print("Overfitting, Regularization, and Optimizers")
    print("=" * 70)
    print()
    print("Today we learn to FIGHT overfitting and CHOOSE optimizers.")
    print("These are the skills that separate beginners from practitioners.\n")

    memorizer_vs_learner()
    input("\nPress Enter to enter the California Housing laboratory...")

    X_train, X_test, y_train, y_test = california_housing_lab()
    input("\nPress Enter for the Dropout experiment...")

    dropout_experiment(X_train, X_test, y_train, y_test)
    input("\nPress Enter for L2 regularization...")

    l2_regularization(X_train, X_test, y_train, y_test)
    input("\nPress Enter for the Optimizer Showdown...")

    optimizer_showdown(X_train, X_test, y_train, y_test)
    input("\nPress Enter for Batch Normalization...")

    batch_normalization(X_train, X_test, y_train, y_test)
    input("\nPress Enter for the Regularization Hierarchy...")

    regularization_hierarchy()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("  1. OVERFITTING = memorizing, not learning. Diagnose: train << val loss.")
    print("  2. DROPOUT = random neuron disabling. Start with 0.3.")
    print("  3. L2 = weight penalty. Pushes weights toward 0.")
    print("  4. ADAM is the default optimizer. lr=0.001.")
    print("  5. LEARNING RATE is the MOST important hyperparameter.")
    print("  6. BATCHNORM stabilizes deep networks. Dense -> BN -> Activation.")
    print("  7. COMBO: Dropout + L2 + BatchNorm = maximum defense.")
    print()
    print("=" * 70)
    print("Next: Architecture Design Lab (how many layers? callbacks?) ->")
    print("=" * 70)


if __name__ == "__main__":
    main()
