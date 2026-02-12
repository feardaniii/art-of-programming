"""
================================================================================
KERAS DECODED: What Really Happens Inside model.compile() and model.fit()
================================================================================

Course: The Art of Programming - Keras Mastery (Sessions 41-42)
Lesson: Opening the Black Box of Deep Learning Frameworks

PREREQUISITES:
    - Session 36: First neuron, MNIST introduction
    - Sessions 37-38: CNN pipeline, Fashion-MNIST, callbacks basics

THE PROMISE:
    You've used model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    But do you REALLY know what those words mean?
    Today we open the black box. Every term. Every choice. Every WHY.
 
DATASET: Wine dataset (sklearn) - 178 samples, 13 chemical features, 3 classes

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# PART 1: THE RESTAURANT KITCHEN ANALOGY
# ==============================================================================

def restaurant_kitchen_analogy():
    """
    ANALOGY: Keras is a Restaurant Kitchen

    THE KITCHEN STAFF:
        Layers      = The COOKS (each transforms ingredients)
        Activations = COOKING TECHNIQUES (grill, saute, bake)
        Loss        = The TASTE TESTER (measures how far from perfect)
        Optimizer   = The HEAD CHEF (adjusts recipes based on taste test)
        Metrics     = The FOOD CRITIC (independent quality score for YOU)
        Epochs      = SERVICE ROUNDS (repeat until mastered)
        Batch Size  = TABLE SIZE (serve N customers at a time)
    """
    print("=" * 70)
    print("PART 1: THE RESTAURANT KITCHEN ANALOGY")
    print("=" * 70)
    print()
    print("Building a neural network is like running a RESTAURANT KITCHEN.")
    print()
    print("THE KITCHEN STAFF:")
    print("-" * 60)
    print(f"  {'Keras Component':<28} {'Kitchen Role':<30}")
    print(f"  {'=' * 28} {'=' * 30}")
    print(f"  {'Layers (Dense, Conv2D)':<28} {'The COOKS':<30}")
    print(f"  {'Activation Functions':<28} {'COOKING TECHNIQUES':<30}")
    print(f"  {'Loss Function':<28} {'The TASTE TESTER':<30}")
    print(f"  {'Optimizer (Adam, SGD)':<28} {'The HEAD CHEF':<30}")
    print(f"  {'Metrics (accuracy)':<28} {'The FOOD CRITIC':<30}")
    print(f"  {'Epochs':<28} {'SERVICE ROUNDS':<30}")
    print(f"  {'Batch Size':<28} {'TABLE SIZE (customers/batch)':<30}")
    print()
    print("THE THREE SACRED COMMANDS:")
    print("-" * 60)
    print("  model.compile()  = Hiring your staff (loss, optimizer, metrics)")
    print("  model.fit()      = Opening the restaurant (training rounds)")
    print("  model.evaluate() = The Michelin review (test on unseen data)")
    print()
    print("Today we decode EVERY ingredient in this kitchen.")
    print("No more magic strings. After today, you CHOOSE with understanding.")


# ==============================================================================
# PART 2: MEET THE WINE DATASET
# ==============================================================================

def explore_wine_dataset():
    """
    ANALOGY: We are wine detectives. 13 chemical measurements, 3 vineyards.
    Given the alcohol level, color intensity, and 11 other clues,
    can we identify which vineyard made it? That is our task.
    """
    print()
    print("=" * 70)
    print("PART 2: MEET THE WINE DATASET")
    print("=" * 70)
    print()
    print("Imagine you are a master sommelier. Someone hands you a glass.")
    print("You cannot see the label. But you CAN measure 13 chemical properties.")
    print("Can you tell which of 3 vineyards produced it?")
    print()

    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(target_names)} classes")
    print()
    print("THE 13 CHEMICAL FEATURES:")
    print("-" * 40)
    for i, name in enumerate(feature_names):
        print(f"  {i + 1:>2}. {name}")
    print()

    unique, counts = np.unique(y, return_counts=True)
    print("CLASS DISTRIBUTION:")
    print("-" * 40)
    for cls, count in zip(unique, counts):
        bar = "#" * count
        print(f"  Class {cls} ({target_names[cls]:>10}): {count:>3} samples  {bar}")
    print(f"  {'TOTAL':>24}: {len(y):>3} samples")
    print()

    # --- Bar chart ---
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(target_names, counts, color=colors, edgecolor='black')
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                str(c), ha='center', fontweight='bold', fontsize=13)
    ax.set_xlabel('Wine Class')
    ax.set_ylabel('Samples')
    ax.set_title('Wine Dataset: Class Distribution', fontweight='bold')
    ax.set_ylim(0, max(counts) + 15)
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'wine_class_distribution.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"[SAVED] {path}")

    # --- Feature histograms ---
    sel = [0, 1, 6, 9, 10, 11, 12, 4]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for idx, feat in enumerate(sel):
        ax = axes.flat[idx]
        for cls in range(3):
            ax.hist(X[y == cls, feat], bins=15, alpha=0.6,
                    label=target_names[cls], color=colors[cls], edgecolor='black')
        ax.set_title(feature_names[feat], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
    fig.suptitle('Wine Features by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'wine_feature_histograms.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"[SAVED] {path}")

    # --- Correlation heatmap ---
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(13)); ax.set_yticks(range(13))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    for i in range(13):
        for j in range(13):
            c = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i,j]:.1f}', ha='center', va='center',
                    fontsize=6, color=c)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'wine_correlation_heatmap.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"[SAVED] {path}")

    # --- Preprocess ---
    print()
    print("PREPROCESSING:")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Split: {len(X_train)} train, {len(X_test)} test (80/20)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"  StandardScaler applied (mean=0, std=1 per feature)")
    print(f"  Before scaling - Feature 0 range: "
          f"[{wine.data[:, 0].min():.1f}, {wine.data[:, 0].max():.1f}]")
    print(f"  After scaling  - Feature 0 range: "
          f"[{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print()
    print("  WHY SCALE? Neural networks learn faster when all features")
    print("  live on the same scale. Alcohol (11-14) vs Proline (278-1680)")
    print("  would confuse the optimizer if left raw.")

    return X_train, X_test, y_train, y_test, scaler


# ==============================================================================
# PART 3: LOSS FUNCTIONS DECODED
# ==============================================================================

def loss_functions_decoded():
    """
    Loss = How WRONG is our prediction?

    ANALOGY: The taste tester says 'This is 0.73 units from perfect.'
    The head chef (optimizer) uses that score to adjust the recipe.
    Loss of 0 = perfect. Higher = worse.
    """
    print()
    print("=" * 70)
    print("PART 3: LOSS FUNCTIONS DECODED")
    print("=" * 70)
    print()
    print("What IS a loss function?")
    print("  It answers: HOW WRONG IS OUR PREDICTION?")
    print("  Loss = 0 -> perfect.  Loss = 2.3 -> way off.")
    print("  The taste tester scores every dish. The chef adjusts.")
    print()

    # --- Binary Cross-Entropy ---
    print("-" * 60)
    print("LOSS #1: Binary Cross-Entropy (2-class problems)")
    print("-" * 60)
    print("  Formula: BCE = -[y * log(p) + (1-y) * log(1-p)]")
    print()

    y_true = 1.0
    p_good, p_bad = 0.9, 0.1
    bce_good = -(y_true * np.log(p_good) + (1 - y_true) * np.log(1 - p_good))
    bce_bad = -(y_true * np.log(p_bad) + (1 - y_true) * np.log(1 - p_bad))

    print(f"  TRUE = 1, Predicted = 0.9 -> BCE = {bce_good:.4f}  (low = good!)")
    print(f"  TRUE = 1, Predicted = 0.1 -> BCE = {bce_bad:.4f}  (high = bad!)")
    print(f"  Confident WRONG = {bce_bad/bce_good:.0f}x HARSHER penalty!")
    print()

    # Verify with Keras
    keras_val = keras.losses.BinaryCrossentropy()(
        tf.constant([1.0]), tf.constant([0.9])).numpy()
    print(f"  Keras BinaryCE([1],[0.9]) = {keras_val:.4f} | Ours = {bce_good:.4f} "
          f"| Match: {'YES' if abs(keras_val - bce_good) < 0.001 else 'NO'}")
    print()

    # --- Categorical Cross-Entropy ---
    print("-" * 60)
    print("LOSS #2: Categorical Cross-Entropy (one-hot labels)")
    print("-" * 60)
    print("  Labels are one-hot: Class 0->[1,0,0], Class 1->[0,1,0], Class 2->[0,0,1]")
    print()

    y_oh = np.array([0, 1, 0])
    p_pred = np.array([0.1, 0.8, 0.1])
    cce = -np.sum(y_oh * np.log(p_pred + 1e-15))
    p_bad_m = np.array([0.7, 0.1, 0.2])
    cce_bad = -np.sum(y_oh * np.log(p_bad_m + 1e-15))

    print(f"  True: {y_oh}, Predicted: {p_pred} -> CCE = {cce:.4f}")
    print(f"  True: {y_oh}, Predicted: {p_bad_m} -> CCE = {cce_bad:.4f} (much worse!)")
    print()

    # --- Sparse Categorical Cross-Entropy ---
    print("-" * 60)
    print("LOSS #3: Sparse Categorical Cross-Entropy (integer labels)")
    print("-" * 60)
    print("  SAME math, but labels are integers, not one-hot vectors.")
    print("  Categorical CE: y = [0,1,0]  |  Sparse CE: y = 1")
    print()

    pred = tf.constant([[0.1, 0.8, 0.1]])
    sparse_l = keras.losses.SparseCategoricalCrossentropy()(
        tf.constant([1]), pred).numpy()
    cat_l = keras.losses.CategoricalCrossentropy()(
        tf.constant([[0.0, 1.0, 0.0]]), pred).numpy()
    print(f"  Sparse CE: {sparse_l:.4f}  |  Categorical CE: {cat_l:.4f}  "
          f"|  Same? {'YES' if abs(sparse_l - cat_l) < 0.001 else 'NO'}")
    print()

    # --- MSE ---
    print("-" * 60)
    print("LOSS #4: Mean Squared Error (regression)")
    print("-" * 60)
    y_reg = np.array([3.0, 5.0, 7.0])
    p_reg = np.array([2.8, 5.5, 6.0])
    mse = np.mean((y_reg - p_reg) ** 2)
    print(f"  True: {y_reg}  Pred: {p_reg}")
    print(f"  MSE = mean({(y_reg - p_reg)**2}) = {mse:.4f}")
    print()

    # --- Decision Table ---
    print("=" * 60)
    print("WHEN TO USE EACH LOSS (The Golden Rule)")
    print("=" * 60)
    print(f"  {'Problem':<28} {'Loss':<28} {'Output Layer':<20}")
    print(f"  {'='*28} {'='*28} {'='*20}")
    print(f"  {'2 classes':<28} {'Binary CE':<28} {'1 neuron + sigmoid':<20}")
    print(f"  {'N classes (one-hot)':<28} {'Categorical CE':<28} {'N neurons + softmax':<20}")
    print(f"  {'N classes (integers)':<28} {'Sparse Categorical CE':<28} {'N neurons + softmax':<20}")
    print(f"  {'Predict a number':<28} {'MSE or MAE':<28} {'1 neuron + linear':<20}")
    print()
    print("  Wine dataset: 3 classes, integer labels -> Sparse CE + softmax")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    p_range = np.linspace(0.01, 0.99, 200)
    axes[0].plot(p_range, -np.log(p_range), 'b-', lw=2, label='True = 1')
    axes[0].plot(p_range, -np.log(1 - p_range), 'r-', lw=2, label='True = 0')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Binary CE: Confident Wrong = HUGE Penalty', fontweight='bold')
    axes[0].legend(); axes[0].set_ylim(0, 5); axes[0].grid(True, alpha=0.3)

    names = ['BCE\ngood', 'BCE\nbad', 'CCE\ngood', 'CCE\nbad', 'MSE']
    vals = [bce_good, bce_bad, cce, cce_bad, mse]
    bar_c = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c', '#3498db']
    bars = axes[1].bar(names, vals, color=bar_c, edgecolor='black')
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                     f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].set_ylabel('Loss Value')
    axes[1].set_title('Loss Values: Good vs Bad', fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'loss_functions_comparison.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"\n[SAVED] {path}")


# ==============================================================================
# PART 4: ACTIVATION FUNCTIONS DECODED
# ==============================================================================

def activation_functions_decoded():
    """
    Activation functions = cooking techniques.
    Without them, layers collapse into one linear transform. No power.

    ReLU    = The BOUNCER (blocks negatives, passes positives)
    Sigmoid = The SQUISHER (output in [0, 1])
    Softmax = The VOTE COUNTER (N outputs sum to 1.0)
    Tanh    = The CENTERED SQUISHER (output in [-1, 1])
    """
    print()
    print("=" * 70)
    print("PART 4: ACTIVATION FUNCTIONS DECODED")
    print("=" * 70)
    print()
    print("Without activation, Dense(64)->Dense(32)->Dense(3) collapses")
    print("into a SINGLE linear transformation. All those layers = one layer.")
    print("Activation adds NON-LINEARITY: the power to learn curves.")
    print()

    x = np.linspace(-5, 5, 500)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    print("ReLU:    f(x) = max(0, x)  -- The BOUNCER")
    print(f"         ReLU(-3)={max(0,-3)}, ReLU(0)={max(0,0)}, ReLU(4)={max(0,4)}")
    print("         Used in: HIDDEN layers (default, fast, effective)")
    print()
    print("Sigmoid: f(x) = 1/(1+e^-x)  -- The PROBABILITY SQUISHER")
    print(f"         sigmoid(-5)={1/(1+np.exp(5)):.4f}, sigmoid(0)=0.5, "
          f"sigmoid(5)={1/(1+np.exp(-5)):.4f}")
    print("         Used in: OUTPUT for BINARY classification")
    print()
    print("Tanh:    f(x) = (e^x-e^-x)/(e^x+e^-x)  -- CENTERED SQUISHER")
    print(f"         tanh(-5)={np.tanh(-5):.4f}, tanh(0)={np.tanh(0):.1f}, "
          f"tanh(5)={np.tanh(5):.4f}")
    print("         Used in: Sometimes hidden layers, RNNs")
    print()

    raw = np.array([2.0, 1.0, 0.1])
    sm = np.exp(raw) / np.sum(np.exp(raw))
    print("Softmax: softmax(x_i) = e^x_i / sum(e^x_j)  -- VOTE COUNTER")
    print(f"         Raw: {raw} -> Softmax: [{sm[0]:.4f}, {sm[1]:.4f}, {sm[2]:.4f}]")
    print(f"         Sum = {sm.sum():.4f} (always 1.0!)")
    print("         Used in: OUTPUT for MULTICLASS classification")
    print()

    print("THE OUTPUT LAYER RULE (memorize this!):")
    print(f"  {'Task':<30} {'Neurons':<15} {'Activation':<12}")
    print(f"  {'='*30} {'='*15} {'='*12}")
    print(f"  {'Binary classification':<30} {'1':<15} {'sigmoid':<12}")
    print(f"  {'Multiclass classification':<30} {'N':<15} {'softmax':<12}")
    print(f"  {'Regression':<30} {'1':<15} {'linear/none':<12}")

    # --- 2x2 visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(x, relu, 'b-', lw=2.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.fill_between(x, relu, where=(x < 0), alpha=0.1, color='red')
    ax.fill_between(x, relu, where=(x >= 0), alpha=0.1, color='blue')
    ax.set_title('ReLU: The Bouncer', fontweight='bold')
    ax.set_ylim(-1, 5); ax.grid(True, alpha=0.3)
    ax.text(-4, 3.5, 'BLOCKED', color='red', ha='center', fontweight='bold')
    ax.text(3, 3.5, 'PASSED', color='blue', ha='center', fontweight='bold')

    ax = axes[0, 1]
    ax.plot(x, sigmoid, 'r-', lw=2.5)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.set_title('Sigmoid: Probability Squisher [0,1]', fontweight='bold')
    ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, tanh, 'g-', lw=2.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Tanh: Centered Squisher [-1,1]', fontweight='bold')
    ax.set_ylim(-1.3, 1.3); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cls_labels = ['Class 0', 'Class 1', 'Class 2']
    b = ax.bar(cls_labels, sm, color=['#2ecc71', '#3498db', '#e74c3c'],
               edgecolor='black')
    for bi, v in zip(b, sm):
        ax.text(bi.get_x() + bi.get_width()/2, bi.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold')
    ax.set_title(f'Softmax: Scores {list(raw)} -> probs sum to 1.0',
                 fontweight='bold')
    ax.set_ylim(0, 0.85); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'activation_functions.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"\n[SAVED] {path}")


# ==============================================================================
# PART 5: BUILD THE WINE CLASSIFIER
# ==============================================================================

def build_wine_classifier(X_train, X_test, y_train, y_test):
    """
    Assemble the kitchen: choose cooks (layers), techniques (activations),
    taste tester (loss), head chef (optimizer). Then open for business.
    """
    print()
    print("=" * 70)
    print("PART 5: BUILD THE WINE CLASSIFIER")
    print("=" * 70)
    print()
    print("Time to assemble our kitchen. Every choice is deliberate.")
    print()

    print("STEP 1: Architecture")
    print("-" * 50)
    model = keras.Sequential(name="WineClassifier")

    model.add(layers.Dense(64, activation='relu', input_shape=(13,), name='hidden_1'))
    print(f"  Dense(64, relu, input_shape=(13,))")
    print(f"    13 features in -> 64 neurons -> ReLU -> params: {13*64+64}")

    model.add(layers.Dense(32, activation='relu', name='hidden_2'))
    print(f"  Dense(32, relu)")
    print(f"    64 in -> 32 neurons -> ReLU -> params: {64*32+32}")

    model.add(layers.Dense(3, activation='softmax', name='output'))
    print(f"  Dense(3, softmax)")
    print(f"    32 in -> 3 outputs (one per class) -> params: {32*3+3}")

    total = (13*64+64) + (64*32+32) + (32*3+3)
    print(f"\n  TOTAL: {total} trainable parameters")
    print()

    print("STEP 2: model.summary()")
    print("-" * 50)
    model.summary()
    print()

    print("STEP 3: model.compile() - Hiring the staff")
    print("-" * 50)
    print("  optimizer='adam'")
    print("    Adam = Adaptive Moment Estimation. Smart head chef who")
    print("    adjusts learning rate PER parameter. Default lr=0.001.")
    print()
    print("  loss='sparse_categorical_crossentropy'")
    print("    Sparse = integer labels (0,1,2). Categorical = multi-class.")
    print("    Cross-Entropy = gap between predicted and true distribution.")
    print()
    print("  metrics=['accuracy']")
    print("    The food critic. Does NOT affect training!")
    print("    Loss drives learning. Accuracy is YOUR dashboard.")

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("\n  Compiled!")
    print()

    print("STEP 4: model.fit() - Open the restaurant!")
    print("-" * 50)
    print("  epochs=100       -> 100 full passes through training data")
    print("  batch_size=16    -> process 16 samples at a time")
    print("  validation_split -> hold 20% for overfitting detection")
    print()
    print("  Training now. Watch loss DROP and accuracy RISE:")
    print("=" * 70)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=16,
                        verbose=1)
    print("=" * 70)
    print()

    print("STEP 5: model.evaluate() - The Michelin review")
    print("-" * 50)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    if test_acc > 0.90:
        print("  VERDICT: Excellent generalization!")
    elif test_acc > 0.75:
        print("  VERDICT: Good. Room for improvement.")
    else:
        print("  VERDICT: Needs work. More data/epochs/capacity.")

    return model, history


# ==============================================================================
# PART 6: READING THE TRAINING REPORT
# ==============================================================================

def read_training_report(model, history, X_test, y_test):
    """
    Training curves are a doctor's X-ray for neural networks.
    Learn to read them: learning, overfitting, or underfitting.
    """
    print()
    print("=" * 70)
    print("PART 6: READING THE TRAINING REPORT")
    print("=" * 70)
    print()

    ep = range(1, len(history.history['loss']) + 1)
    tl = history.history['loss']
    vl = history.history['val_loss']
    ta = history.history['accuracy']
    va = history.history['val_accuracy']

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, tl, 'b-', lw=2, label='Train Loss')
    axes[0].plot(ep, vl, 'r-', lw=2, label='Val Loss')
    best_ep = np.argmin(vl) + 1
    axes[0].axvline(best_ep, color='green', ls='--', alpha=0.7)
    axes[0].annotate(f'Best: epoch {best_ep}', xy=(best_ep, min(vl)),
                     xytext=(best_ep+10, min(vl)+0.3),
                     arrowprops=dict(arrowstyle='->'), color='green',
                     fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, ta, 'b-', lw=2, label='Train Acc')
    axes[1].plot(ep, va, 'r-', lw=2, label='Val Acc')
    best_acc_ep = np.argmax(va) + 1
    axes[1].axvline(best_acc_ep, color='green', ls='--', alpha=0.7)
    axes[1].annotate(f'Best: {max(va):.1%}', xy=(best_acc_ep, max(va)),
                     xytext=(best_acc_ep+10, max(va)-0.15),
                     arrowprops=dict(arrowstyle='->'), color='green',
                     fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'training_curves.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"[SAVED] {path}")
    print()

    # --- Diagnostic guide ---
    print("HOW TO READ TRAINING CURVES:")
    print("=" * 55)
    print("  Train DOWN + Val DOWN  -> LEARNING (good!)")
    print("  Train DOWN + Val UP    -> OVERFITTING (memorizing)")
    print("    Fix: dropout, smaller model, early stopping")
    print("  Both HIGH + FLAT       -> UNDERFITTING (too small)")
    print("    Fix: bigger model, more epochs, less regularization")
    print("  Both CONVERGE          -> GOOD FIT (this is the goal)")
    print()

    gap = abs(vl[-1] - tl[-1])
    print(f"  Our model: train_loss={tl[-1]:.4f}, val_loss={vl[-1]:.4f}, gap={gap:.4f}")
    if gap < 0.1 and vl[-1] < 0.5:
        print("  DIAGNOSIS: Healthy generalization.")
    elif vl[-1] > tl[-1] * 1.5:
        print("  DIAGNOSIS: Overfitting. Consider regularization.")
    else:
        print("  DIAGNOSIS: Reasonable fit.")
    print()

    # --- Confusion Matrix ---
    print("CONFUSION MATRIX:")
    print("-" * 50)
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    names = load_wine().target_names

    print(f"  {'':>12} {'Predicted':>30}")
    print(f"  {'':>12} {names[0]:>10}{names[1]:>10}{names[2]:>10}")
    for i in range(3):
        print(f"  True {names[i]:>5} |{cm[i,0]:>10}{cm[i,1]:>10}{cm[i,2]:>10}")
    print()
    print("  PER-CLASS ACCURACY:")
    for i in range(3):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"    {names[i]:>10}: {correct}/{total} = {correct/total:.1%}")
    print()

    # --- Confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix', fontweight='bold')
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            c = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=c)
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=120); plt.close()
    print(f"[SAVED] {path}")
    print()

    # --- Classification report ---
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=names))

    # --- Sample predictions ---
    print("SAMPLE PREDICTIONS (first 5):")
    print("-" * 50)
    for i in range(min(5, len(y_test))):
        p = y_probs[i]
        pc, tc = y_pred[i], y_test[i]
        ok = "CORRECT" if pc == tc else "WRONG"
        print(f"  #{i+1}: True={names[tc]}, Pred={names[pc]} ({ok})")
        print(f"       Probs: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
              f"Confidence: {p[pc]:.1%}")


# ==============================================================================
# PART 7: MAIN
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("  KERAS DECODED")
    print("  What Really Happens Inside model.compile() and model.fit()")
    print("=" * 70)
    print()
    print("  Course: The Art of Programming - Keras Mastery (Sessions 41-42)")
    print()
    print("  ROADMAP:")
    print("    Part 1: The Restaurant Kitchen Analogy")
    print("    Part 2: Meet the Wine Dataset")
    print("    Part 3: Loss Functions Decoded (by hand!)")
    print("    Part 4: Activation Functions Decoded (visualized)")
    print("    Part 5: Build the Wine Classifier")
    print("    Part 6: Reading the Training Report")
    print("=" * 70)

    restaurant_kitchen_analogy()
    input("\nPress Enter to meet the Wine dataset...")

    X_train, X_test, y_train, y_test, scaler = explore_wine_dataset()
    input("\nPress Enter to decode loss functions...")

    loss_functions_decoded()
    input("\nPress Enter to decode activation functions...")

    activation_functions_decoded()
    input("\nPress Enter to build the Wine classifier...")

    model, history = build_wine_classifier(X_train, X_test, y_train, y_test)
    input("\nPress Enter to read the training report...")

    read_training_report(model, history, X_test, y_test)

    print()
    print("=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("  1. LOSS = how WRONG (taste tester). Drives training.")
    print("  2. OPTIMIZER = adjusts weights to reduce loss (head chef). Adam default.")
    print("  3. METRICS = YOUR scoreboard (food critic). Does NOT affect training.")
    print("  4. ACTIVATION = shapes layer output. Hidden: ReLU. Output: see table.")
    print("  5. TRAINING CURVES = your diagnostic X-ray.")
    print("  6. CONFUSION MATRIX = WHERE the model struggles.")
    print()
    print("  THE GOLDEN CHOICE TABLE:")
    print("  " + "-" * 55)
    print(f"  {'Task':<25} {'Loss':<25} {'Output':<15}")
    print("  " + "-" * 55)
    print(f"  {'Binary (2 classes)':<25} {'binary_crossentropy':<25} {'sigmoid':<15}")
    print(f"  {'Multi (int labels)':<25} {'sparse_categorical_ce':<25} {'softmax':<15}")
    print(f"  {'Multi (one-hot)':<25} {'categorical_ce':<25} {'softmax':<15}")
    print(f"  {'Regression':<25} {'mse / mae':<25} {'linear':<15}")
    print("  " + "-" * 55)
    print()
    print("  Next: Multiclass deep-dive + your first REGRESSION model")
    print("=" * 70)


if __name__ == "__main__":
    main()
