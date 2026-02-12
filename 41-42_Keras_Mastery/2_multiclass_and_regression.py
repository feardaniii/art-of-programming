"""
================================================================================
MULTICLASS & REGRESSION: Two Problems, One Framework
================================================================================

Course: The Art of Programming - Keras Mastery (Sessions 41-42)
Lesson: Classification Meets Regression -- The Complete Problem-Type Toolkit

PREREQUISITES:
    - Script 1: Keras Decoded (loss, activations, training curves)

THE NEW TERRITORY:
    In Script 1, we classified wine into 3 classes. That's MULTICLASS.
    But what if you need to predict a NUMBER? Not "which class" but "how much"?

    - Predicting house prices = regression
    - Predicting disease progression = regression
    - Predicting temperature tomorrow = regression

    Same Keras. Different output layer. Different loss. Let's master BOTH.

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
from sklearn.datasets import load_wine, load_diabetes, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)

# Directory where this script lives (all PNGs saved here)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# PART 1: THE HOTEL ANALOGY
# ==============================================================================

def hotel_analogy():
    """
    ANALOGY: Hotel Reviews

    Imagine you work for a travel company. You have thousands of hotel reviews.
    Your boss asks TWO different questions about the same reviews:

    Question A: "Which hotel CHAIN is this review about?"
        → Marriott, Hilton, or Holiday Inn
        → This is CLASSIFICATION (discrete categories)

    Question B: "What STAR RATING does this hotel deserve?"
        → 3.7 stars, 4.2 stars, 1.9 stars
        → This is REGRESSION (continuous number)

    Same data. Different question. Different answer format.
    And therefore: different output layer, different loss function.
    """

    print()
    print("=" * 70)
    print("PART 1: THE HOTEL ANALOGY")
    print("Same Data, Two Different Questions")
    print("=" * 70)
    print()

    print("Imagine you work at a travel company. You have 10,000 hotel reviews.")
    print("Your boss asks TWO different questions about the SAME data:")
    print()

    print("-" * 70)
    print("QUESTION A: 'Which hotel CHAIN wrote this review?'")
    print("-" * 70)
    print("  Answer:  Marriott, Hilton, or Holiday Inn")
    print("  Type:    CLASSIFICATION (pick a category)")
    print("  Output:  3 neurons with softmax (probabilities that sum to 1)")
    print("  Loss:    Categorical cross-entropy")
    print("  Example: [0.85, 0.10, 0.05] -> 'Marriott' (85% confident)")
    print()

    print("-" * 70)
    print("QUESTION B: 'What STAR RATING does this hotel deserve?'")
    print("-" * 70)
    print("  Answer:  3.7 stars (a continuous number)")
    print("  Type:    REGRESSION (predict a number)")
    print("  Output:  1 neuron with NO activation (linear output)")
    print("  Loss:    Mean Squared Error (MSE)")
    print("  Example: 3.72 -> 'This hotel is about 3.7 stars'")
    print()

    print("=" * 70)
    print("          CLASSIFICATION vs REGRESSION: Side by Side")
    print("=" * 70)
    print()
    print(f"  {'Feature':<22} {'CLASSIFICATION':<24} {'REGRESSION':<24}")
    print(f"  {'-'*22} {'-'*24} {'-'*24}")
    print(f"  {'Question':<22} {'Which category?':<24} {'How much?':<24}")
    print(f"  {'Answer type':<22} {'Discrete label':<24} {'Continuous number':<24}")
    print(f"  {'Output neurons':<22} {'N (one per class)':<24} {'1':<24}")
    print(f"  {'Output activation':<22} {'softmax':<24} {'none (linear)':<24}")
    print(f"  {'Loss function':<22} {'cross-entropy':<24} {'MSE or MAE':<24}")
    print(f"  {'Metric':<22} {'accuracy':<24} {'R-squared, MAE':<24}")
    print(f"  {'Example':<22} {'Cat vs Dog vs Bird':<24} {'Predict temperature':<24}")
    print()
    print("  KEY INSIGHT: The architecture is almost identical!")
    print("  The ONLY differences are the last layer and the loss function.")
    print("  Everything else (hidden layers, training loop, etc.) stays the same.")
    print()


# ==============================================================================
# PART 2: MULTICLASS DEEP-DIVE WITH WINE
# ==============================================================================

def multiclass_deep_dive():
    """
    Multiclass classification: predicting one of N categories.

    ANALOGY: A wine sommelier tasting wine and saying:
    "This is a Class 0 (Barolo), or a Class 1 (Grignolino), or a Class 2 (Barbera)."

    The model doesn't just pick one class -- it gives CONFIDENCE scores:
    "I'm 92% sure it's Barolo, 5% Grignolino, 3% Barbera."

    That confidence distribution is what softmax gives us.
    """

    print()
    print("=" * 70)
    print("PART 2: MULTICLASS DEEP-DIVE WITH WINE")
    print("From Probabilities to Predictions")
    print("=" * 70)
    print()

    # --- Load and prepare data ---
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names

    print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(class_names)} (3 Italian wine cultivars)")
    print(f"Class distribution: {np.bincount(y)} samples per class")
    print()

    # --- One-hot encoding explained ---
    print("-" * 70)
    print("ONE-HOT ENCODING: How computers understand categories")
    print("-" * 70)
    print()
    print("  Humans say:   'This wine is Class 2'")
    print("  Computer says: [0, 0, 1]")
    print()
    print("  Why? Because neural networks output PROBABILITIES for each class.")
    print("  The true label needs to be in the same format:")
    print()
    print("  Integer label  ->  One-hot vector")
    print("  Class 0        ->  [1, 0, 0]    ('definitely class 0')")
    print("  Class 1        ->  [0, 1, 0]    ('definitely class 1')")
    print("  Class 2        ->  [0, 0, 1]    ('definitely class 2')")
    print()

    # Show both label formats
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y)

    print("First 5 samples -- integer vs one-hot:")
    for i in range(5):
        print(f"  Sample {i}: integer = {y[i]}  ->  one-hot = {y_onehot[i]}")
    print()

    print("KERAS SHORTCUT: You don't always need to one-hot encode manually!")
    print("  - loss='sparse_categorical_crossentropy' accepts INTEGER labels (0, 1, 2)")
    print("  - loss='categorical_crossentropy' requires ONE-HOT labels ([0,0,1])")
    print("  We'll use SPARSE (integer labels) -- it's simpler.")
    print()

    # --- Scale and split ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
    print()

    # --- Build multiclass model ---
    print("-" * 70)
    print("BUILDING THE MULTICLASS MODEL")
    print("-" * 70)
    print()
    print("Architecture:")
    print("  Input(13 features) -> Dense(64, relu) -> Dense(32, relu) -> Dense(3, softmax)")
    print()
    print("  The final Dense(3, softmax) is the KEY:")
    print("  - 3 neurons = 3 classes")
    print("  - softmax = outputs sum to 1.0 (probabilities)")
    print()

    tf.random.set_seed(42)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(13,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print()

    # --- Train ---
    print("Training the multiclass model...")
    print()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Test Loss:     {test_loss:.4f}")
    print()

    # --- Show prediction PROBABILITIES ---
    print("-" * 70)
    print("PREDICTION PROBABILITIES: What softmax actually outputs")
    print("-" * 70)
    print()

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print("The model doesn't just say 'Class 1'. It says HOW SURE it is:")
    print()
    num_show = 8
    for i in range(num_show):
        probs = y_probs[i]
        pred = y_pred[i]
        actual = y_test[i]
        match = "correct" if pred == actual else "WRONG"
        print(f"  Sample {i+1}: ", end="")
        for c in range(3):
            print(f"Class {c}: {probs[c]*100:5.1f}%  ", end="")
        print(f"  -> Predicted: {pred}  Actual: {actual}  ({match})")
    print()

    # --- Horizontal bar chart of prediction probabilities ---
    fig, axes = plt.subplots(5, 1, figsize=(8, 10))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx in range(5):
        ax = axes[idx]
        probs = y_probs[idx]
        pred = y_pred[idx]
        actual = y_test[idx]
        bar_colors = [colors[c] for c in range(3)]

        bars = ax.barh([f"Class {c}\n({class_names[c]})" for c in range(3)],
                       probs * 100, color=bar_colors, edgecolor='black', height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Probability (%)")

        match_str = "Correct!" if pred == actual else "WRONG!"
        ax.set_title(f"Sample {idx+1}: Predicted={pred}, Actual={actual} ({match_str})",
                     fontsize=10, fontweight='bold')

        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{prob*100:.1f}%", va='center', fontsize=9, fontweight='bold')

    plt.suptitle("Multiclass Prediction Probabilities (Wine Dataset)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "multiclass_probabilities.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Probability bar chart saved: multiclass_probabilities.png")
    print()

    # --- Confusion matrix heatmap ---
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Confusion Matrix: Wine Multiclass", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('Actual Class', fontsize=11)
    ax.set_xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "multiclass_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: multiclass_confusion_matrix.png")
    print()

    # --- Per-class metrics ---
    print("-" * 70)
    print("PER-CLASS METRICS: Precision, Recall, F1-Score")
    print("-" * 70)
    print()
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    print()


# ==============================================================================
# PART 3: REGRESSION WITH DIABETES
# ==============================================================================

def regression_with_diabetes():
    """
    Regression: predicting a continuous number.

    ANALOGY: A doctor looking at patient data and predicting:
    "This patient's disease progression score will be about 152."

    Not a CATEGORY. A NUMBER. How far will the disease progress in one year?

    The key differences from classification:
    - Output layer: 1 neuron, NO activation (linear output)
    - Loss: MSE (mean squared error) instead of cross-entropy
    - Metrics: R-squared, MAE instead of accuracy
    """

    print()
    print("=" * 70)
    print("PART 3: REGRESSION WITH DIABETES")
    print("Predicting Numbers, Not Categories")
    print("=" * 70)
    print()

    # --- Load and explore ---
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names

    print("Diabetes Dataset (sklearn built-in):")
    print(f"  Samples:  {X.shape[0]} patients")
    print(f"  Features: {X.shape[1]} medical measurements")
    print(f"  Target:   Disease progression after one year (continuous)")
    print()
    print("Features:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name:<8}  range: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]")
    print()

    print("Target distribution:")
    print(f"  Min:    {y.min():.1f}")
    print(f"  Max:    {y.max():.1f}")
    print(f"  Mean:   {y.mean():.1f}")
    print(f"  Median: {np.median(y):.1f}")
    print(f"  Std:    {y.std():.1f}")
    print()
    print("  This is NOT a class label. It's a continuous score.")
    print("  Our model must output a NUMBER, not a probability distribution.")
    print()

    # --- Target distribution histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axvline(y.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {y.mean():.1f}')
    ax.axvline(np.median(y), color='orange', linestyle='--', linewidth=2,
               label=f'Median = {np.median(y):.1f}')
    ax.set_xlabel("Disease Progression Score", fontsize=11)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_title("Diabetes: Target Distribution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "regression_target_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Target distribution saved: regression_target_distribution.png")
    print()

    # --- Prepare data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
    print()

    # --- Build regression model ---
    print("-" * 70)
    print("BUILDING THE REGRESSION MODEL")
    print("-" * 70)
    print()
    print("Architecture:")
    print("  Input(10 features) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)")
    print()
    print("  CRITICAL DIFFERENCE from classification:")
    print("  - Last layer: Dense(1) with NO ACTIVATION")
    print("  - No softmax, no sigmoid. Just a raw linear output.")
    print("  - This lets the model output ANY number (negative, positive, large, small)")
    print()
    print("  Loss: MSE (Mean Squared Error)")
    print("    = average of (predicted - actual)^2 for all samples")
    print("    Punishes big errors MORE than small errors (squared!)")
    print()

    tf.random.set_seed(42)
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # NO activation! Linear output for regression
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    print()

    # --- Train ---
    print("Training the regression model...")
    print()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss (MSE)", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE', color='#e74c3c', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', color='#3498db', linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Training & Validation MAE", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Diabetes Regression: Training Curves", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "regression_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print()
    print(f"Training curves saved: regression_training_curves.png")
    print()

    # --- Predictions ---
    y_pred = model.predict(X_test, verbose=0).flatten()

    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    print("-" * 70)
    print("REGRESSION METRICS")
    print("-" * 70)
    print()
    print(f"  MSE  (Mean Squared Error):  {mse_val:.2f}")
    print(f"  RMSE (Root Mean Squared):   {np.sqrt(mse_val):.2f}")
    print(f"  MAE  (Mean Absolute Error): {mae_val:.2f}")
    print(f"  R^2  (R-squared):           {r2_val:.4f}")
    print()
    print("  INTERPRETING R-squared:")
    print(f"    R^2 = {r2_val:.4f} means the model explains {r2_val*100:.1f}% of the variance.")
    print("    R^2 = 1.0 -> perfect predictions")
    print("    R^2 = 0.0 -> no better than predicting the mean every time")
    print("    R^2 < 0   -> worse than just guessing the mean (bad model!)")
    print()
    print(f"  MAE = {mae_val:.1f} means on average, our prediction is off by")
    print(f"  about {mae_val:.0f} points (target range: {y.min():.0f} to {y.max():.0f}).")
    print()

    # --- Predicted vs Actual scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.6, color='#3498db', edgecolor='black',
               linewidth=0.5, s=60, label='Predictions')

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min()) - 10
    max_val = max(y_test.max(), y_pred.max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')

    ax.set_xlabel("Actual Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title("Predicted vs Actual (Diabetes Regression)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Annotate with metrics
    textstr = f"R$^2$ = {r2_val:.3f}\nMAE = {mae_val:.1f}\nRMSE = {np.sqrt(mse_val):.1f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "regression_predicted_vs_actual.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predicted vs Actual plot saved: regression_predicted_vs_actual.png")
    print(f"  (Points on the red diagonal = perfect predictions)")
    print()

    # --- Residual distribution ---
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=25, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Mean = {residuals.mean():.1f}')
    ax.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Residual Distribution (Should be centered at 0)", fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "regression_residuals.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Residual distribution saved: regression_residuals.png")
    print(f"  (Centered around 0 = no systematic bias)")
    print()

    print("  WHAT RESIDUALS TELL YOU:")
    print(f"    Mean residual: {residuals.mean():.2f}  (close to 0 = no bias)")
    print(f"    Std residual:  {residuals.std():.2f}  (smaller = more consistent)")
    print("    If the histogram is bell-shaped and centered at 0, the model")
    print("    is making 'honest' errors -- not systematically over/under-predicting.")
    print()


# ==============================================================================
# PART 4: THE GOLDEN RULES TABLE
# ==============================================================================

def golden_rules_table():
    """
    The cheat sheet you'll reference for EVERY project.

    ANALOGY: A mechanic's wrench chart.
    You don't memorize which wrench fits which bolt -- you check the chart.
    This is YOUR chart for neural network output design.
    """

    print()
    print("=" * 70)
    print("PART 4: THE GOLDEN RULES OF OUTPUT DESIGN")
    print("Your Forever Reference Table")
    print("=" * 70)
    print()

    # Print the golden rules table
    print("+" + "=" * 68 + "+")
    print("|" + "THE GOLDEN RULES OF OUTPUT DESIGN".center(68) + "|")
    print("+" + "=" * 18 + "+" + "=" * 15 + "+" + "=" * 15 + "+" + "=" * 17 + "+")
    print(f"| {'Problem Type':<16} | {'Output Layer':<13} | {'Activation':<13} | {'Loss Function':<15} |")
    print("+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 17 + "+")
    print(f"| {'Binary':<16} | {'1 neuron':<13} | {'sigmoid':<13} | {'binary_crossent':<15} |")
    print(f"| {'Multiclass':<16} | {'N neurons':<13} | {'softmax':<13} | {'sparse_cat_CE':<15} |")
    print(f"| {'Multi-label':<16} | {'N neurons':<13} | {'sigmoid':<13} | {'binary_crossent':<15} |")
    print(f"| {'Regression':<16} | {'1 neuron':<13} | {'none/linear':<13} | {'mse or mae':<15} |")
    print(f"| {'Bounded Regress.':<16} | {'1 neuron':<13} | {'sigmoid':<13} | {'mse':<15} |")
    print("+" + "=" * 18 + "+" + "=" * 15 + "+" + "=" * 15 + "+" + "=" * 17 + "+")
    print()

    print("ROW-BY-ROW EXPLANATION:")
    print()
    print("  1. BINARY CLASSIFICATION")
    print("     Example: Email spam detector (spam or not spam)")
    print("     sigmoid squishes output to [0, 1]: one probability for 'yes'")
    print()
    print("  2. MULTICLASS CLASSIFICATION")
    print("     Example: Digit recognition (0, 1, 2, ... 9)")
    print("     softmax gives a probability for EACH class, they sum to 1")
    print("     Use 'sparse_categorical_crossentropy' with integer labels")
    print("     Use 'categorical_crossentropy' with one-hot labels")
    print()
    print("  3. MULTI-LABEL CLASSIFICATION")
    print("     Example: Movie genre tags (a movie can be Action AND Comedy)")
    print("     sigmoid on EACH neuron independently (not softmax!)")
    print("     Each output is a separate yes/no decision")
    print()
    print("  4. REGRESSION")
    print("     Example: House price prediction ($347,500)")
    print("     No activation = linear output. Can be ANY number.")
    print()
    print("  5. BOUNDED REGRESSION")
    print("     Example: Probability of rain (must be between 0 and 1)")
    print("     sigmoid constrains output to [0, 1] range")
    print("     Multiply by range if needed (e.g., sigmoid * 100 for percentages)")
    print()

    # --- Visual decision flowchart ---
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color='#3498db', fontsize=9, fontweight='bold'):
        """Draw a rounded box with text."""
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, linewidth=2,
                              edgecolor='black', facecolor=color, alpha=0.85,
                              zorder=2, joinstyle='round')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, zorder=3, wrap=True)

    def draw_arrow(x1, y1, x2, y2, label='', color='black'):
        """Draw arrow between boxes with optional label."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color=color),
                    zorder=1)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x + 0.15, mid_y, label, fontsize=8, fontstyle='italic',
                    color='#555555', ha='left', va='center')

    # Title
    ax.text(6, 9.5, "OUTPUT DESIGN DECISION FLOWCHART",
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Start question
    draw_box(6, 8.3, 4.5, 0.8, "What are you predicting?", color='#f39c12')

    # Branch: Category or Number?
    draw_arrow(6, 7.9, 3.5, 7.2)
    draw_arrow(6, 7.9, 8.5, 7.2)
    ax.text(4.2, 7.7, "Category", fontsize=9, fontstyle='italic', color='#2c3e50')
    ax.text(7.6, 7.7, "Number", fontsize=9, fontstyle='italic', color='#2c3e50')

    # Left branch: Classification
    draw_box(3.5, 6.8, 3.2, 0.7, "How many classes?", color='#e8daef')

    # Binary
    draw_arrow(2.5, 6.45, 1.8, 5.7)
    ax.text(1.3, 6.2, "2", fontsize=9, fontstyle='italic', color='#2c3e50')
    draw_box(1.8, 5.3, 2.8, 0.7, "BINARY\n1 neuron, sigmoid\nbinary_crossentropy",
             color='#aed6f1', fontsize=8)

    # Multiclass
    draw_arrow(3.5, 6.45, 3.5, 5.7)
    ax.text(3.7, 6.1, "3+", fontsize=9, fontstyle='italic', color='#2c3e50')
    draw_box(3.5, 5.3, 2.8, 0.7, "MULTICLASS\nN neurons, softmax\nsparse_cat_CE",
             color='#aed6f1', fontsize=8)

    # Multi-label
    draw_arrow(4.5, 6.45, 5.5, 5.7)
    ax.text(5.3, 6.2, "multiple\nper sample", fontsize=7, fontstyle='italic',
            color='#2c3e50')
    draw_box(5.5, 5.3, 2.8, 0.7, "MULTI-LABEL\nN neurons, sigmoid\nbinary_crossentropy",
             color='#aed6f1', fontsize=8)

    # Right branch: Regression
    draw_box(8.5, 6.8, 3.0, 0.7, "Is output bounded?", color='#e8daef')

    # Unbounded
    draw_arrow(7.7, 6.45, 7.5, 5.7)
    ax.text(6.9, 6.2, "No", fontsize=9, fontstyle='italic', color='#2c3e50')
    draw_box(7.5, 5.3, 2.8, 0.7, "REGRESSION\n1 neuron, linear\nmse / mae",
             color='#a9dfbf', fontsize=8)

    # Bounded
    draw_arrow(9.3, 6.45, 10.0, 5.7)
    ax.text(10.0, 6.2, "Yes", fontsize=9, fontstyle='italic', color='#2c3e50')
    draw_box(10.0, 5.3, 2.8, 0.7, "BOUNDED REGR.\n1 neuron, sigmoid\nmse",
             color='#a9dfbf', fontsize=8)

    # Examples row
    ax.text(6, 4.3, "EXAMPLES:", ha='center', fontsize=11, fontweight='bold')
    examples = [
        (1.8, "Spam?\nYes/No"),
        (3.5, "Digit?\n0-9"),
        (5.5, "Genres?\nAction+Comedy"),
        (7.5, "Price?\n$347k"),
        (10.0, "Rain chance?\n0.0-1.0")
    ]
    for ex_x, ex_text in examples:
        draw_box(ex_x, 3.5, 2.4, 0.7, ex_text, color='#fdebd0', fontsize=8,
                 fontweight='normal')

    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "golden_rules_flowchart.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Decision flowchart saved: golden_rules_flowchart.png")
    print()
    print("  MEMORIZE THIS TABLE. It will guide every project you build.")
    print("  When in doubt, check the Golden Rules.")
    print()


# ==============================================================================
# PART 5: THE SCALING EXPERIMENT
# ==============================================================================

def scaling_experiment():
    """
    Does scaling features matter? Let's PROVE it with real experiments.

    ANALOGY: Imagine a group project where one person SHOUTS (range 0-1000)
    and another whispers (range 0-1). The teacher (gradient descent) only
    hears the shouter. Scaling makes everyone speak at the same volume.
    """

    print()
    print("=" * 70)
    print("PART 5: THE SCALING EXPERIMENT")
    print("Does StandardScaler Actually Matter? Let's PROVE It.")
    print("=" * 70)
    print()

    print("ANALOGY: A group project with a SHOUTER and a WHISPERER")
    print("  Feature 'Proline' ranges from 278 to 1680")
    print("  Feature 'Nonflavanoid phenols' ranges from 0.13 to 0.66")
    print()
    print("  Without scaling, gradient descent only 'hears' Proline.")
    print("  The small features are INVISIBLE to the learning process.")
    print("  Scaling puts everyone on equal footing.")
    print()

    # --- Experiment 1: Wine Classification ---
    print("=" * 50)
    print("EXPERIMENT 1: Wine Classification")
    print("=" * 50)
    print()

    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model builder (same architecture both times)
    def build_wine_model():
        tf.random.set_seed(42)
        m = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(13,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        m.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    # --- WITHOUT scaling ---
    print("Training WITHOUT StandardScaler (raw features)...")
    print()
    model_raw = build_wine_model()
    history_raw = model_raw.fit(
        X_train, y_train,
        epochs=50, batch_size=16,
        validation_split=0.2, verbose=1
    )
    raw_loss, raw_acc = model_raw.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"  WITHOUT scaling -> Test Accuracy: {raw_acc*100:.1f}%")
    print()

    # --- WITH scaling ---
    print("Training WITH StandardScaler...")
    print()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scaled = build_wine_model()
    history_scaled = model_scaled.fit(
        X_train_scaled, y_train,
        epochs=50, batch_size=16,
        validation_split=0.2, verbose=1
    )
    scaled_loss, scaled_acc = model_scaled.evaluate(X_test_scaled, y_test, verbose=0)
    print()
    print(f"  WITH scaling    -> Test Accuracy: {scaled_acc*100:.1f}%")
    print()

    # --- Comparison ---
    print("-" * 50)
    print("WINE CLASSIFICATION RESULTS:")
    print("-" * 50)
    print(f"  WITHOUT scaling: {raw_acc*100:.1f}% accuracy")
    print(f"  WITH scaling:    {scaled_acc*100:.1f}% accuracy")
    diff = (scaled_acc - raw_acc) * 100
    if diff > 0:
        print(f"  Improvement:     +{diff:.1f} percentage points!")
    print()

    # --- Side-by-side training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_raw.history['accuracy'], label='Train Acc', color='#e74c3c',
                 linewidth=2)
    axes[0].plot(history_raw.history['val_accuracy'], label='Val Acc', color='#3498db',
                 linewidth=2)
    axes[0].set_title(f"WITHOUT Scaling (Test: {raw_acc*100:.1f}%)", fontweight='bold',
                      fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(history_scaled.history['accuracy'], label='Train Acc', color='#e74c3c',
                 linewidth=2)
    axes[1].plot(history_scaled.history['val_accuracy'], label='Val Acc', color='#3498db',
                 linewidth=2)
    axes[1].set_title(f"WITH Scaling (Test: {scaled_acc*100:.1f}%)", fontweight='bold',
                      fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("The Scaling Experiment: Wine Classification",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "scaling_experiment_wine.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Side-by-side training curves saved: scaling_experiment_wine.png")
    print()

    # --- Experiment 2: Diabetes Regression ---
    print("=" * 50)
    print("EXPERIMENT 2: Diabetes Regression")
    print("=" * 50)
    print()

    diabetes = load_diabetes()
    X_d, y_d = diabetes.data, diabetes.target
    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42
    )

    def build_regression_model():
        tf.random.set_seed(42)
        m = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(10,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return m

    # Without scaling
    print("Training WITHOUT scaling...")
    model_r_raw = build_regression_model()
    history_r_raw = model_r_raw.fit(
        X_d_train, y_d_train,
        epochs=100, batch_size=32,
        validation_split=0.2, verbose=1
    )
    y_pred_raw = model_r_raw.predict(X_d_test, verbose=0).flatten()
    mse_raw = mean_squared_error(y_d_test, y_pred_raw)
    mae_raw = mean_absolute_error(y_d_test, y_pred_raw)
    r2_raw = r2_score(y_d_test, y_pred_raw)
    print()

    # With scaling
    print("Training WITH scaling...")
    scaler_d = StandardScaler()
    X_d_train_s = scaler_d.fit_transform(X_d_train)
    X_d_test_s = scaler_d.transform(X_d_test)

    model_r_scaled = build_regression_model()
    history_r_scaled = model_r_scaled.fit(
        X_d_train_s, y_d_train,
        epochs=100, batch_size=32,
        validation_split=0.2, verbose=1
    )
    y_pred_scaled = model_r_scaled.predict(X_d_test_s, verbose=0).flatten()
    mse_scaled = mean_squared_error(y_d_test, y_pred_scaled)
    mae_scaled = mean_absolute_error(y_d_test, y_pred_scaled)
    r2_scaled = r2_score(y_d_test, y_pred_scaled)
    print()

    print("-" * 50)
    print("DIABETES REGRESSION RESULTS:")
    print("-" * 50)
    print(f"                    {'WITHOUT Scaling':<20} {'WITH Scaling':<20}")
    print(f"  MSE:              {mse_raw:<20.2f} {mse_scaled:<20.2f}")
    print(f"  MAE:              {mae_raw:<20.2f} {mae_scaled:<20.2f}")
    print(f"  R-squared:        {r2_raw:<20.4f} {r2_scaled:<20.4f}")
    print()

    print("WHY SCALING MATTERS:")
    print("  Features with range [0, 1000] dominate features with range [0, 1].")
    print("  Gradient descent adjusts weights proportional to feature magnitude.")
    print("  Scaling (StandardScaler) puts ALL features on equal footing:")
    print("    mean = 0, standard deviation = 1 for every feature.")
    print()
    print("  RULE: ALWAYS scale your features before feeding them to a neural network.")
    print("  The only exception: image pixels already in [0, 1] after /255.")
    print()


# ==============================================================================
# PART 6: METRICS DEEP-DIVE -- ACCURACY ISN'T ENOUGH
# ==============================================================================

def metrics_deep_dive():
    """
    Why accuracy can LIE to you, and what to use instead.

    ANALOGY: Bank fraud detection.
    Only 1% of transactions are fraud. If your model ALWAYS predicts
    'not fraud', it's 99% accurate. But it catches ZERO fraud.
    Accuracy says 99%. Reality says useless.

    You need precision, recall, and F1-score to see the REAL picture.
    """

    print()
    print("=" * 70)
    print("PART 6: METRICS DEEP-DIVE -- ACCURACY ISN'T ENOUGH")
    print("When 99% Accuracy is Actually Terrible")
    print("=" * 70)
    print()

    print("THE TRAP: Bank Fraud Detection")
    print()
    print("  Your bank processes 10,000 transactions per day.")
    print("  Only 100 are fraudulent (1%).")
    print()
    print("  A 'model' that ALWAYS predicts 'NOT FRAUD':")
    print("    Accuracy = 9,900 / 10,000 = 99.0%")
    print("    Fraud caught = 0 / 100 = 0%")
    print()
    print("  99% accuracy. Zero fraud detected. USELESS.")
    print()
    print("  This is why accuracy alone is DANGEROUS on imbalanced data.")
    print()

    # --- Generate imbalanced dataset ---
    print("-" * 70)
    print("LET'S SEE THIS IN ACTION")
    print("-" * 70)
    print()
    print("Creating an imbalanced dataset: 90% class 0, 10% class 1")
    print("(Think: 90% legitimate transactions, 10% fraud)")
    print()

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
        flip_y=0.05
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Class 0 = {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.0f}%), "
          f"Class 1 = {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.0f}%)")
    print()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train model ---
    print("Training a model on this imbalanced data...")
    print()

    tf.random.set_seed(42)
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(20,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    print()

    print("-" * 70)
    print("THE ACCURACY TRAP IN ACTION")
    print("-" * 70)
    print()
    print(f"  Model Accuracy: {test_acc*100:.1f}%")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("  Confusion Matrix:")
    print(f"                        Predicted: 0    Predicted: 1")
    print(f"    Actual: 0 (legit)      {tn:4d} (TN)       {fp:4d} (FP)")
    print(f"    Actual: 1 (fraud)      {fn:4d} (FN)       {tp:4d} (TP)")
    print()

    total_fraud = fn + tp
    if total_fraud > 0:
        fraud_recall = tp / total_fraud
        print(f"  Fraud detected: {tp} out of {total_fraud} actual fraud cases "
              f"({fraud_recall*100:.1f}%)")
    if tp + fp > 0:
        fraud_precision = tp / (tp + fp)
        print(f"  Of all fraud ALERTS: {tp} out of {tp + fp} were real fraud "
              f"({fraud_precision*100:.1f}%)")
    print()

    # --- Confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Oranges')
    ax.set_title("Confusion Matrix: Imbalanced Classification", fontsize=12,
                 fontweight='bold')
    plt.colorbar(im, ax=ax)

    class_labels = ['Class 0 (Legit)', 'Class 1 (Fraud)']
    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_labels)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')

    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "imbalanced_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: imbalanced_confusion_matrix.png")
    print()

    # --- Precision, Recall, F1 explained ---
    print("-" * 70)
    print("PRECISION, RECALL, F1-SCORE: The Metrics That Don't Lie")
    print("-" * 70)
    print()
    print("  PRECISION (for fraud class):")
    print('    "Of all the transactions I FLAGGED as fraud, how many were real fraud?"')
    print("     Formula: TP / (TP + FP)")
    print(f"     Think: How many false alarms am I sending?")
    print()
    print("  RECALL (for fraud class):")
    print('    "Of all ACTUAL fraud transactions, how many did I catch?"')
    print("     Formula: TP / (TP + FN)")
    print(f"     Think: How much fraud am I missing?")
    print()
    print("  F1-SCORE:")
    print('    "Balanced score between precision and recall"')
    print("     Formula: 2 * (precision * recall) / (precision + recall)")
    print(f"     Think: A single number that captures both concerns.")
    print()

    print("  BANK FRAUD ANALOGY:")
    print("    High precision, low recall = Few false alarms, but misses most fraud")
    print("    Low precision, high recall = Catches most fraud, but too many false alarms")
    print("    F1 helps you find the sweet spot between the two.")
    print()

    # --- Classification report ---
    print("-" * 70)
    print("FULL CLASSIFICATION REPORT")
    print("-" * 70)
    print()
    report = classification_report(y_test, y_pred,
                                   target_names=['Legit (0)', 'Fraud (1)'])
    print(report)
    print()

    # --- Precision/Recall bar chart ---
    report_dict = classification_report(y_test, y_pred,
                                        target_names=['Legit (0)', 'Fraud (1)'],
                                        output_dict=True)

    classes = ['Legit (0)', 'Fraud (1)']
    precisions = [report_dict[c]['precision'] for c in classes]
    recalls = [report_dict[c]['recall'] for c in classes]
    f1_scores = [report_dict[c]['f1-score'] for c in classes]

    x_pos = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x_pos - width, precisions, width, label='Precision',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x_pos, recalls, width, label='Recall',
                   color='#e74c3c', edgecolor='black')
    bars3 = ax.bar(x_pos + width, f1_scores, width, label='F1-Score',
                   color='#2ecc71', edgecolor='black')

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision, Recall, F1-Score per Class (Imbalanced Data)",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "precision_recall_f1_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision/Recall/F1 chart saved: precision_recall_f1_chart.png")
    print()

    print("TAKEAWAYS:")
    print("  1. Accuracy can LIE on imbalanced datasets")
    print("  2. Always check the confusion matrix to see WHERE errors happen")
    print("  3. Precision answers: 'Are my alerts trustworthy?'")
    print("  4. Recall answers: 'Am I catching everything important?'")
    print("  5. F1-score is the balanced summary of both")
    print("  6. For fraud/medical: RECALL is often more important than precision")
    print("     (Better to flag 10 false alarms than miss 1 real fraud)")
    print()


# ==============================================================================
# PART 7: MAIN FUNCTION
# ==============================================================================

def main():
    """
    Orchestrates the complete lesson: from hotel analogy to metrics mastery.
    Each section builds on the previous one, with pauses for comprehension.
    """

    print()
    print("=" * 70)
    print("MULTICLASS & REGRESSION: Two Problems, One Framework")
    print("The Art of Programming - Keras Mastery (Sessions 41-42)")
    print("=" * 70)
    print()
    print("In this lesson, you will:")
    print("  1. Understand classification vs regression through analogy")
    print("  2. Master multiclass prediction probabilities")
    print("  3. Build your first regression model")
    print("  4. Learn the Golden Rules of output design")
    print("  5. PROVE that feature scaling matters")
    print("  6. Discover why accuracy lies on imbalanced data")
    print()

    hotel_analogy()
    input("\nPress Enter for multiclass deep-dive...")

    multiclass_deep_dive()
    input("\nPress Enter for your first REGRESSION model...")

    regression_with_diabetes()
    input("\nPress Enter for the Golden Rules...")

    golden_rules_table()
    input("\nPress Enter for the Scaling Experiment...")

    scaling_experiment()
    input("\nPress Enter for Metrics Deep-Dive...")

    metrics_deep_dive()

    print()
    print("=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("  1. Classification = WHICH category  ->  softmax + cross-entropy")
    print("  2. Regression = HOW MUCH             ->  linear output + MSE")
    print("  3. THE GOLDEN RULES table is your cheat sheet forever")
    print("  4. ALWAYS scale your features (StandardScaler)")
    print("  5. Accuracy LIES on imbalanced data  ->  check precision/recall")
    print()
    print("  PNGs saved in this session:")
    print("    - multiclass_probabilities.png")
    print("    - multiclass_confusion_matrix.png")
    print("    - regression_target_distribution.png")
    print("    - regression_training_curves.png")
    print("    - regression_predicted_vs_actual.png")
    print("    - regression_residuals.png")
    print("    - golden_rules_flowchart.png")
    print("    - scaling_experiment_wine.png")
    print("    - imbalanced_confusion_matrix.png")
    print("    - precision_recall_f1_chart.png")
    print()
    print("Next: The Practitioner's Toolkit (overfitting, regularization, optimizers) ->")
    print()


if __name__ == "__main__":
    main()
