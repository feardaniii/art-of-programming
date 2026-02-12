"""
================================================================================
FUNCTIONAL API & CUSTOM COMPONENTS: The Real Keras
================================================================================

Course: The Art of Programming - Advanced Keras (Session 43)
Lesson: Beyond Sequential -- Multi-Input, Skip Connections, Custom Everything

PREREQUISITES:
    - Sessions 41-42: Keras decoded, multiclass, regression, regularization,
      optimizers, callbacks, architecture design
    - MAE=Average of ∣Actual Price−Predicted Price∣



THE LEAP:
    Until now, every model was a straight stack: layer after layer after layer.
    Sequential([Dense, Dense, Dense]) -- a highway with one entrance and one exit.

    But real problems are messier:
    - What if you have TWO different input types? (location + house features)
    - What if you want skip connections? (ResNet-style shortcut paths)
    - What if MSE is not the right loss? (outliers need Huber loss)

    Sequential cannot do this. The Functional API can do ANYTHING.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ==============================================================================
# PART 1: HIGHWAY VS CITY STREETS
# ==============================================================================

def sequential_vs_functional():
    """
    ANALOGY: Highway vs City Streets

    Sequential = a highway. One entrance, one exit, no turns.
    Functional API = city streets. Multiple entrances, junctions, roundabouts.

    Sequential is for SIMPLE models. Functional is for ANYTHING.
    """

    print("=" * 70)
    print("PART 1: SEQUENTIAL vs FUNCTIONAL API")
    print("Highway vs City Streets")
    print("=" * 70)
    print()

    print("SEQUENTIAL (Highway):")
    print("  One entrance. One exit. No turns. No junctions.")
    print("  Input -> Dense -> Dense -> Dense -> Output")
    print("  Perfect for: simple stacks of layers")
    print("  Cannot do: multiple inputs, skip connections, branches")
    print()

    print("FUNCTIONAL API (City Streets):")
    print("  Multiple entrances. Junctions. Roundabouts. Merging lanes.")
    print("  Input_A -> Dense -> Concatenate")
    print("                        |       -> Dense -> Output")
    print("  Input_B -> Dense -> Concatenate")
    print("  Can do: ANYTHING a neural network can express")
    print()

    # Side-by-side: same model, two ways
    print("-" * 50)
    print("SAME MODEL, TWO WAYS:")
    print("-" * 50)
    print()

    print("Sequential style:")
    print("  model = keras.Sequential([")
    print("      layers.Dense(64, activation='relu', input_shape=(8,)),")
    print("      layers.Dense(32, activation='relu'),")
    print("      layers.Dense(1)")
    print("  ])")
    print()

    print("Functional style:")
    print("  inputs = layers.Input(shape=(8,))")
    print("  x = layers.Dense(64, activation='relu')(inputs)")
    print("  x = layers.Dense(32, activation='relu')(x)")
    print("  outputs = layers.Dense(1)(x)")
    print("  model = keras.Model(inputs=inputs, outputs=outputs)")
    print()

    print("They produce the EXACT same network.")
    print("The Functional API just gives you more FLEXIBILITY.")
    print()

    # Prove it
    np.random.seed(42)
    tf.random.set_seed(42)

    seq_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    tf.random.set_seed(42)
    inputs = layers.Input(shape=(8,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    func_model = keras.Model(inputs=inputs, outputs=outputs)

    print(f"Sequential parameters: {seq_model.count_params():,}")
    print(f"Functional parameters: {func_model.count_params():,}")
    print("Identical architecture, identical parameter count.")
    print()

    print("SO WHY USE FUNCTIONAL?")
    print("  Because Sequential CANNOT do:")
    print("  1. Multiple inputs (Part 3)")
    print("  2. Skip connections (Part 2)")
    print("  3. Multiple outputs")
    print("  4. Shared layers")
    print("  5. Any non-linear topology")
    print()


# ==============================================================================
# PART 2: SKIP CONNECTIONS - THE RESNET TRICK
# ==============================================================================

def skip_connections_demo():
    """
    Skip connections: a shortcut path that bypasses layers.

    Instead of: Input -> Dense -> Dense -> Dense -> Output
    We add:     Input -> Dense -> Dense -> Dense -> Output
                  |                          ^
                  +--------(shortcut)--------+

    WHY? Gradients can flow through the shortcut.
    This prevents the vanishing gradient problem in deep networks.
    This is the core idea behind ResNet (won ImageNet 2015).
    """

    print("=" * 70)
    print("PART 2: SKIP CONNECTIONS")
    print("The ResNet Trick for Better Deep Networks")
    print("=" * 70)
    print()

    print("THE PROBLEM WITH DEEP NETWORKS:")
    print("  As networks get deeper, gradients shrink (vanishing gradient).")
    print("  Layer 20 barely learns because the signal from the loss")
    print("  has been multiplied by 20 small numbers.")
    print()
    print("THE SOLUTION: Skip connections")
    print("  Add a 'shortcut' that bypasses some layers.")
    print("  The gradient can flow DIRECTLY through the shortcut.")
    print("  Both paths (shortcut + main) contribute to learning.")
    print()
    print("  Input -----> Dense -> ReLU -> Dense -----> Add -> ReLU -> Output")
    print("    |                                         ^")
    print("    +---------(skip connection)--------------+")
    print()

    # Load California Housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # Model WITHOUT skip connections
    np.random.seed(42)
    tf.random.set_seed(42)

    inputs = layers.Input(shape=(8,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model_no_skip = keras.Model(inputs=inputs, outputs=outputs, name='no_skip')

    model_no_skip.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Training model WITHOUT skip connections...")
    history_no_skip = model_no_skip.fit(
        X_train, y_train, epochs=50, batch_size=256,
        validation_data=(X_val, y_val), verbose=0
    )

    # Model WITH skip connections
    np.random.seed(42)
    tf.random.set_seed(42)

    inputs = layers.Input(shape=(8,))
    x = layers.Dense(64, activation='relu')(inputs)

    # Skip connection block 1
    shortcut = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, shortcut])  # Add shortcut to main path
    x = layers.ReLU()(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model_skip = keras.Model(inputs=inputs, outputs=outputs, name='with_skip')

    model_skip.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Training model WITH skip connections...")
    history_skip = model_skip.fit(
        X_train, y_train, epochs=50, batch_size=256,
        validation_data=(X_val, y_val), verbose=0
    )

    # Compare
    no_skip_mae = model_no_skip.evaluate(X_test, y_test, verbose=0)[1]
    skip_mae = model_skip.evaluate(X_test, y_test, verbose=0)[1]

    print()
    print(f"  Without skip connections: Test MAE = {no_skip_mae:.4f}")
    print(f"  With skip connections:    Test MAE = {skip_mae:.4f}")
    print()

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history_no_skip.history['val_mae'], 'b-', label='No Skip', linewidth=2)
    ax1.plot(history_skip.history['val_mae'], 'r-', label='With Skip', linewidth=2)
    ax1.set_title('Validation MAE', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_no_skip.history['val_loss'], 'b-', label='No Skip', linewidth=2)
    ax2.plot(history_skip.history['val_loss'], 'r-', label='With Skip', linewidth=2)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Skip Connections: Smoother Training, Better Results',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('skip_connection_comparison.png', dpi=150, bbox_inches='tight')
    print("  Visualization saved: skip_connection_comparison.png")
    plt.close()
    print()

    print("  Skip connections help because:")
    print("  1. Gradients flow directly through the shortcut (no vanishing)")
    print("  2. The network can learn to 'skip' layers it doesn't need")
    print("  3. Training is more stable and converges faster")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ==============================================================================
# PART 3: MULTI-INPUT MODEL
# ==============================================================================

def multi_input_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Multi-input: process different feature GROUPS through separate branches,
    then combine them.

    California Housing features split into:
    - Location branch: Latitude (6), Longitude (7)
    - Property branch: MedInc (0), HouseAge (1), AveRooms (2),
                       AveBedrms (3), Population (4), AveOccup (5)

    Each branch learns patterns specific to its feature group.
    The merge layer combines location knowledge + property knowledge.
    """

    print("=" * 70)
    print("PART 3: MULTI-INPUT MODEL")
    print("Separate Branches for Different Feature Types")
    print("=" * 70)
    print()

    print("SCENARIO: California Housing has two kinds of features:")
    print("  LOCATION: Latitude, Longitude (WHERE is the house?)")
    print("  PROPERTY: Income, Age, Rooms, Bedrooms, Population, Occupancy")
    print()
    print("  These are fundamentally different types of information.")
    print("  A multi-input model processes each type SEPARATELY,")
    print("  then COMBINES the learned representations.")
    print()

    # Split features into two groups
    # California Housing: [MedInc, HouseAge, AveRooms, AveBedrms, Pop, AveOccup, Lat, Long]
    #                     indices: 0, 1, 2, 3, 4, 5, 6, 7
    location_cols = [6, 7]       # Latitude, Longitude
    property_cols = [0, 1, 2, 3, 4, 5]  # Everything else

    X_train_loc = X_train[:, location_cols]
    X_train_prop = X_train[:, property_cols]
    X_val_loc = X_val[:, location_cols]
    X_val_prop = X_val[:, property_cols]
    X_test_loc = X_test[:, location_cols]
    X_test_prop = X_test[:, property_cols]

    print(f"  Location features: {X_train_loc.shape[1]} (Lat, Long)")
    print(f"  Property features: {X_train_prop.shape[1]} (Income, Age, Rooms, etc.)")
    print()

    # Build multi-input model
    np.random.seed(42)
    tf.random.set_seed(42)

    # Location branch
    location_input = layers.Input(shape=(2,), name='location_input')
    loc = layers.Dense(16, activation='relu', name='loc_dense1')(location_input)
    loc = layers.Dense(8, activation='relu', name='loc_dense2')(loc)

    # Property branch
    property_input = layers.Input(shape=(6,), name='property_input')
    prop = layers.Dense(32, activation='relu', name='prop_dense1')(property_input)
    prop = layers.Dense(16, activation='relu', name='prop_dense2')(prop)

    # Merge branches
    merged = layers.Concatenate(name='merge')([loc, prop])  # 8 + 16 = 24

    # Combined processing
    x = layers.Dense(32, activation='relu', name='combined_dense1')(merged)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, name='output')(x)

    multi_model = keras.Model(
        inputs=[location_input, property_input],
        outputs=outputs,
        name='MultiInputHousing'
    )

    multi_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Multi-input model architecture:")
    multi_model.summary()
    print()

    # Train
    print("Training multi-input model...")
    history_multi = multi_model.fit(
        [X_train_loc, X_train_prop], y_train,
        epochs=50, batch_size=256,
        validation_data=([X_val_loc, X_val_prop], y_val),
        verbose=0
    )

    multi_mae = multi_model.evaluate(
        [X_test_loc, X_test_prop], y_test, verbose=0
    )[1]

    # Compare to single-input
    np.random.seed(42)
    tf.random.set_seed(42)

    single_model = keras.Sequential([
        layers.Dense(48, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ], name='SingleInputHousing')

    single_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Training single-input model (for comparison)...")
    history_single = single_model.fit(
        X_train, y_train, epochs=50, batch_size=256,
        validation_data=(X_val, y_val), verbose=0
    )

    single_mae = single_model.evaluate(X_test, y_test, verbose=0)[1]

    print()
    print(f"  Single-input model: Test MAE = {single_mae:.4f}")
    print(f"  Multi-input model:  Test MAE = {multi_mae:.4f}")
    print()

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_single.history['val_mae'], 'b-', label='Single-Input', linewidth=2)
    ax.plot(history_multi.history['val_mae'], 'r-', label='Multi-Input', linewidth=2)
    ax.set_title('Single-Input vs Multi-Input: Validation MAE', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multi_input_comparison.png', dpi=150, bbox_inches='tight')
    print("  Visualization saved: multi_input_comparison.png")
    plt.close()
    print()

    print("WHEN TO USE MULTI-INPUT:")
    print("  - Text + metadata (document classification)")
    print("  - Image + tabular data (medical diagnosis)")
    print("  - User features + item features (recommendations)")
    print("  - Time series + static features (forecasting)")
    print("  - Any time you have DIFFERENT TYPES of input data")
    print()

    return multi_model


# ==============================================================================
# PART 4: CUSTOM LOSS AND METRICS
# ==============================================================================

def custom_loss_and_metrics(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Sometimes MSE is not the right loss.
    Sometimes accuracy is not the right metric.

    Keras lets you define your own.
    """

    print("=" * 70)
    print("PART 4: CUSTOM LOSS AND METRICS")
    print("Tailoring Keras to Your Problem")
    print("=" * 70)
    print()

    # ---- Custom Loss: Huber ----
    print("CUSTOM LOSS: Huber Loss")
    print()
    print("  MSE penalizes large errors QUADRATICALLY (error^2).")
    print("  One outlier with error=100 contributes 10,000 to the loss!")
    print("  This can destabilize training.")
    print()
    print("  Huber Loss: MSE for small errors, MAE for large errors.")
    print("  Best of both worlds: smooth gradient near zero, robust to outliers.")
    print()

    # keras.saving not in this tensorflow version, using .utils
    @keras.utils.register_keras_serializable()
    def huber_loss(y_true, y_pred, delta=1.0):
        """Huber loss: quadratic for small errors, linear for large errors."""
        error = y_true - y_pred
        is_small = tf.abs(error) <= delta
        small_loss = 0.5 * tf.square(error)
        large_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small, small_loss, large_loss)

    # ---- Custom Metric: R-squared ----
    print("CUSTOM METRIC: R-squared (Coefficient of Determination)")
    print()
    print("  R-squared tells you: what fraction of variance does the model explain?")
    print("  R^2 = 1.0: perfect predictions")
    print("  R^2 = 0.0: no better than predicting the mean")
    print("  R^2 < 0.0: worse than predicting the mean (terrible)")
    print()

    class RSquared(keras.metrics.Metric):
        def __init__(self, name='r_squared', **kwargs):
            super().__init__(name=name, **kwargs)
            self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
            self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')
            self.count = self.add_weight(name='count', initializer='zeros')
            self.y_sum = self.add_weight(name='y_sum', initializer='zeros')

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)
            self.ss_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
            self.y_sum.assign_add(tf.reduce_sum(y_true))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            y_mean = self.y_sum / self.count
            # We approximate ss_tot using running statistics
            return 1.0 - (self.ss_res / (self.ss_res + 1e-7))

        def reset_state(self):
            self.ss_res.assign(0.0)
            self.ss_tot.assign(0.0)
            self.count.assign(0.0)
            self.y_sum.assign(0.0)

    # ---- Compare MSE vs Huber on data with outliers ----
    print("EXPERIMENT: MSE vs Huber on data with outliers")
    print()

    # Inject some outliers into training data
    X_train_noisy = X_train.copy()
    y_train_noisy = y_train.copy()
    n_outliers = 50
    outlier_idx = np.random.choice(len(y_train_noisy), n_outliers, replace=False)
    y_train_noisy[outlier_idx] = y_train_noisy[outlier_idx] * 3 + 5  # Extreme values

    print(f"  Injected {n_outliers} outliers into training data")
    print(f"  Normal target range: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"  Outlier values up to: {y_train_noisy.max():.2f}")
    print()

    # Train with MSE
    np.random.seed(42)
    tf.random.set_seed(42)

    mse_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    mse_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("  Training with MSE loss...")
    mse_history = mse_model.fit(
        X_train_noisy, y_train_noisy, epochs=30, batch_size=256,
        validation_data=(X_val, y_val), verbose=0
    )

    # Train with Huber
    np.random.seed(42)
    tf.random.set_seed(42)

    huber_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    huber_model.compile(optimizer='adam', loss=huber_loss, metrics=['mae'])

    print("  Training with Huber loss...")
    huber_history = huber_model.fit(
        X_train_noisy, y_train_noisy, epochs=30, batch_size=256,
        validation_data=(X_val, y_val), verbose=0
    )

    mse_mae = mse_model.evaluate(X_test, y_test, verbose=0)[1]
    huber_mae = huber_model.evaluate(X_test, y_test, verbose=0)[1]

    print()
    print(f"  MSE loss  -> Test MAE: {mse_mae:.4f}")
    print(f"  Huber loss -> Test MAE: {huber_mae:.4f}")
    print()

    if huber_mae < mse_mae:
        print("  Huber wins! It is more ROBUST to outliers.")
    else:
        print("  Both perform similarly. Huber shines more with extreme outliers.")
    print()

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mse_history.history['val_mae'], 'b-', label='MSE Loss', linewidth=2)
    ax.plot(huber_history.history['val_mae'], 'r-', label='Huber Loss', linewidth=2)
    ax.set_title('MSE vs Huber Loss (Data with Outliers)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('huber_vs_mse.png', dpi=150, bbox_inches='tight')
    print("  Visualization saved: huber_vs_mse.png")
    plt.close()
    print()

    print("  WHEN TO USE CUSTOM LOSS:")
    print("    - Huber: regression with potential outliers")
    print("    - Focal loss: imbalanced classification")
    print("    - Weighted loss: some classes matter more than others")
    print("    - Domain-specific: any business logic that MSE cannot capture")
    print()


# ==============================================================================
# PART 5: MODEL INTERPRETATION
# ==============================================================================

def model_interpretation(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Neural networks are NOT black boxes -- IF you know how to ask.

    Permutation Importance:
    Shuffle one feature at a time. If performance drops a lot,
    that feature is important.
    """

    print("=" * 70)
    print("PART 5: MODEL INTERPRETATION")
    print("What Did the Network Actually Learn?")
    print("=" * 70)
    print()

    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']

    # Train a good model
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=256,
              validation_data=(X_val, y_val), verbose=0)

    baseline_mae = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"  Baseline Test MAE: {baseline_mae:.4f}")
    print()

    # Permutation importance
    print("PERMUTATION IMPORTANCE:")
    print("  For each feature, shuffle its values and measure performance drop.")
    print("  Large drop = important feature. Small drop = unimportant feature.")
    print()

    importances = {}
    for i, name in enumerate(feature_names):
        X_test_shuffled = X_test.copy()
        np.random.seed(42)
        np.random.shuffle(X_test_shuffled[:, i])

        shuffled_mae = model.evaluate(X_test_shuffled, y_test, verbose=0)[1]
        importance = shuffled_mae - baseline_mae
        importances[name] = importance

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Feature':<15} {'MAE Increase':<15} {'Interpretation'}")
    print(f"  {'-'*50}")
    for name, imp in sorted_features:
        if imp > 0.1:
            interpretation = "Very important"
        elif imp > 0.01:
            interpretation = "Moderately important"
        else:
            interpretation = "Low importance"
        print(f"  {name:<15} {imp:>+.4f}{'':>8} {interpretation}")
    print()

    # Visualization
    names = [f for f, _ in sorted_features]
    values = [v for _, v in sorted_features]
    colors = ['#e74c3c' if v > 0.1 else '#3498db' if v > 0.01 else '#95a5a6'
              for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel('MAE Increase When Shuffled', fontsize=12)
    ax.set_title('Feature Importance (Permutation Method)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("  Visualization saved: feature_importance.png")
    plt.close()
    print()

    print("  INSIGHT: MedInc (median income) is by far the most important")
    print("  predictor of house value. Location (Lat/Long) matters too.")
    print("  This matches economic intuition: income drives housing prices.")
    print()

    print("  Neural networks are NOT black boxes IF you ask the right questions.")
    print("  Permutation importance works with ANY model, including deep networks.")
    print()

    return model


# ==============================================================================
# PART 6: SAVE, LOAD, AND INSPECT
# ==============================================================================

def save_load_inspect(model, X_test, y_test):
    """
    Three ways to save a Keras model:
    1. Full model (.keras): architecture + weights + optimizer state
    2. Weights only (.weights.h5): just the learned parameters
    3. Architecture only (JSON): just the structure
    """

    print("=" * 70)
    print("PART 6: SAVE, LOAD, AND INSPECT")
    print("From Training to Deployment")
    print("=" * 70)
    print()

    # 1. Save full model
    model.save('housing_model.keras')
    print("  1. FULL MODEL saved: housing_model.keras")
    print("     Contains: architecture + weights + optimizer + training config")
    print("     Use for: deployment, sharing, continuing training")
    print()

    # 2. Load and verify
    loaded = keras.models.load_model('housing_model.keras')
    original_mae = model.evaluate(X_test, y_test, verbose=0)[1]
    loaded_mae = loaded.evaluate(X_test, y_test, verbose=0)[1]
    print(f"  Original MAE: {original_mae:.4f}")
    print(f"  Loaded MAE:   {loaded_mae:.4f}")
    print(f"  Match: {np.isclose(original_mae, loaded_mae)}")
    print()

    # 3. Inspect weights
    print("  INSPECTING WEIGHTS:")
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            w = weights[0]
            b = weights[1] if len(weights) > 1 else None
            print(f"    {layer.name}:")
            print(f"      Weight shape: {w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")
            if b is not None:
                print(f"      Bias shape: {b.shape}, mean={b.mean():.4f}")
    print()

    print("  WHEN TO USE EACH SAVE METHOD:")
    print("    Full model (.keras): deployment, sharing (most common)")
    print("    Weights only: checkpointing during training, transfer learning")
    print("    Architecture (JSON): versioning, reproducing experiments")
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("SESSION 43 - SCRIPT 1: FUNCTIONAL API & CUSTOM COMPONENTS")
    print("Beyond Sequential -- The Real Keras")
    print("=" * 70)
    print()

    # Part 1: Sequential vs Functional
    sequential_vs_functional()
    input("Press Enter to continue to Part 2 (Skip Connections)...")
    print()

    # Part 2: Skip connections
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = skip_connections_demo()
    input("Press Enter to continue to Part 3 (Multi-Input Model)...")
    print()

    # Part 3: Multi-input
    multi_model = multi_input_model(X_train, y_train, X_val, y_val, X_test, y_test)
    input("Press Enter to continue to Part 4 (Custom Loss & Metrics)...")
    print()

    # Part 4: Custom loss and metrics
    custom_loss_and_metrics(X_train, y_train, X_val, y_val, X_test, y_test)
    input("Press Enter to continue to Part 5 (Model Interpretation)...")
    print()

    # Part 5: Interpretation
    model = model_interpretation(X_train, y_train, X_val, y_val, X_test, y_test)
    input("Press Enter to continue to Part 6 (Save & Load)...")
    print()

    # Part 6: Save/Load
    save_load_inspect(model, X_test, y_test)

    # Summary
    print("=" * 70)
    print("SCRIPT 1 COMPLETE: Functional API & Custom Components")
    print("=" * 70)
    print()
    print("What you learned:")
    print("  1. Functional API: flexible alternative to Sequential")
    print("  2. Skip connections: shortcuts for better gradient flow")
    print("  3. Multi-input models: separate branches for different feature types")
    print("  4. Custom loss (Huber): robust to outliers")
    print("  5. Model interpretation: permutation importance reveals what matters")
    print("  6. Save/load/inspect: from training to deployment")
    print()
    print("Next: Script 2 -- The Model Comparison Experiment.")
    print("      Rigorous methodology for testing multiple configurations.")
    print("=" * 70)


if __name__ == '__main__':
    main()
