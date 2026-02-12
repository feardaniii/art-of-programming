"""
================================================================================
MODEL COMPARISON EXPERIMENT: The Scientific Method for Machine Learning
================================================================================

Course: The Art of Programming - Advanced Keras (Session 43)
Lesson: Running Rigorous Experiments -- Multiple Seeds, Systematic Comparison

PREREQUISITES:
    - Sessions 41-42: Complete Keras toolkit (losses, optimizers, regularization,
      callbacks, architecture design)
    - Script 1: Functional API, custom components

THE PROBLEM WITH "I GOT 95%!":
    You train a model. Get 95% accuracy. Celebrate.
    Your friend trains the SAME model. Gets 88%.
    What happened? RANDOM INITIALIZATION.
    Neural networks start with random weights. Different random seed =
    different starting point = different result.
    One run means NOTHING. Science requires REPRODUCIBILITY.
    Today you learn to run PROPER experiments.

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
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score, mean_absolute_error)
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_all_seeds(seed):
    """Set numpy and tensorflow seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==============================================================================
# PART 1: THE SCIENTIFIC METHOD FOR ML
# ==============================================================================

def scientific_method_intro():
    """
    ANALOGY: Clinical Trials.
    You don't test a drug on ONE patient. You test on MANY, with a control
    group, and report the AVERAGE outcome with its UNCERTAINTY.
    ML experiments follow the SAME logic.
    """
    print("=" * 70)
    print("PART 1: THE SCIENTIFIC METHOD FOR ML")
    print("Clinical Trials for Neural Networks")
    print("=" * 70)
    print()
    print("ANALOGY: Clinical Trials")
    print("-" * 40)
    print("  A pharmaceutical company creates a new drug.")
    print("  They test it on ONE patient. Patient recovers!")
    print("  'Our drug works!' they announce.")
    print()
    print("  Would you trust that? Of course not.")
    print("  Maybe the patient would have recovered anyway.")
    print("  Real science demands: many patients, a control group, statistics.")
    print("  ML is NO different.")
    print()
    print("THE ML EXPERIMENT PROTOCOL:")
    print("=" * 40)
    print("  1. DEFINE your question ('Is architecture A better than B?')")
    print("  2. CONTROL variables (same data split, preprocessing, epochs)")
    print("  3. VARY only ONE thing (architecture, OR optimizer, OR LR)")
    print("  4. REPEAT with multiple seeds (minimum 3, ideally 5-10)")
    print("  5. REPORT mean +/- std (not just the best single run)")
    print("  6. INCLUDE a BASELINE (simple model or even random)")
    print()

    # --- Demonstrate the seed problem ---
    print("-" * 70)
    print("DEMO: Same Model, Different Seeds, Different Results")
    print("-" * 70)
    print()
    wine = load_wine()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(wine.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, wine.target, test_size=0.2, random_state=0, stratify=wine.target)

    seeds = [42, 123, 7, 999, 2024]
    accuracies = []
    print("  Training the EXACT SAME architecture 5 times...")
    print("  Only difference: random weight initialization seed.")
    print()
    for seed in seeds:
        set_all_seeds(seed)
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=80, batch_size=16, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(acc * 100)
        print(f"  Run (seed={seed:>4}): {acc*100:.1f}%")

    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)
    print(f"\n  Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%")
    print(f"  Spread: {max(accuracies) - min(accuracies):.1f} percentage points!")
    print("\n  THAT is why single runs are MEANINGLESS.")
    print()


# ==============================================================================
# PART 2: WINE CLASSIFICATION TOURNAMENT
# ==============================================================================

WINE_CONFIGS = {
    "Tiny": lambda s: keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=s),
        layers.Dense(3, activation='softmax')]),
    "Small": lambda s: keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=s),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')]),
    "Medium": lambda s: keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=s),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')]),
    "Large": lambda s: keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=s),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')]),
    "With L2": lambda s: keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=s,
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(3, activation='softmax')]),
    "With BN": lambda s: keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=s),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')]),
}


def wine_classification_tournament():
    """
    ANALOGY: MasterChef Tournament.
    Six chefs (architectures) compete on the same dish (Wine dataset).
    Each chef cooks three times (3 seeds) to prove consistency.
    We judge on AVERAGE performance, not one lucky plate.
    """
    print("=" * 70)
    print("PART 2: WINE CLASSIFICATION TOURNAMENT")
    print("6 Architectures Compete. 3 Seeds Each. May the Best Model Win.")
    print("=" * 70)
    print()
    print("  Dataset: Wine (sklearn) -- 178 samples, 13 features, 3 classes")
    print("  Seeds: 42, 123, 7  |  Epochs: 100  |  Metric: Test accuracy")
    print()

    wine = load_wine()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(wine.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, wine.target, test_size=0.2, random_state=0, stratify=wine.target)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0, stratify=y_train)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print()

    configs = list(WINE_CONFIGS.keys())
    seeds = [42, 123, 7]
    shape = (X_train.shape[1],)

    # Show contenders
    print("THE CONTENDERS:")
    print("-" * 60)
    for name in configs:
        set_all_seeds(42)
        m = WINE_CONFIGS[name](shape)
        print(f"  {name:<12} | {m.count_params():>6} params")
    print()

    # Run tournament
    print("TOURNAMENT IN PROGRESS...")
    print("=" * 60)
    results = {}
    all_results = []

    for name in configs:
        results[name] = {'acc': [], 'loss': [], 'times': [], 'params': 0}
        print(f"\n  [{name}]")
        for seed in seeds:
            set_all_seeds(seed)
            model = WINE_CONFIGS[name](shape)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            results[name]['params'] = model.count_params()
            t0 = time.time()
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=100, batch_size=16, verbose=0)
            elapsed = time.time() - t0
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            results[name]['acc'].append(acc * 100)
            results[name]['loss'].append(loss)
            results[name]['times'].append(elapsed)
            all_results.append({
                'experiment': 'Wine Classification', 'model_name': name,
                'seed': seed, 'accuracy': acc * 100, 'loss': loss,
                'r2': None, 'mse': None, 'mae': None,
                'params': model.count_params(), 'epochs_used': 100,
                'time_sec': round(elapsed, 2)})
            print(f"    seed={seed:>4}: acc={acc*100:.1f}%, loss={loss:.4f}, "
                  f"time={elapsed:.1f}s")

    # Scoreboard
    print("\n")
    print("+" + "=" * 74 + "+")
    print("|{:^74}|".format("WINE CLASSIFICATION TOURNAMENT RESULTS"))
    print("+" + "=" * 74 + "+")
    print(f"| {'Model':<12} | {'Acc(mean)':>9} | {'Acc(std)':>9} | "
          f"{'Best':>7} | {'Worst':>7} | {'Params':>7} |")
    print("+" + "-" * 74 + "+")

    best_mean, winner = -1, ""
    for name in configs:
        accs = results[name]['acc']
        m, s = np.mean(accs), np.std(accs)
        if m > best_mean:
            best_mean, winner = m, name
        print(f"| {name:<12} | {m:>8.1f}% | +/-{s:>5.1f}% | "
              f"{max(accs):>6.1f}% | {min(accs):>6.1f}% | {results[name]['params']:>7} |")
    print("+" + "=" * 74 + "+")

    w = results[winner]
    print(f"\n  WINNER: {winner}!")
    print(f"  Mean: {np.mean(w['acc']):.1f}% +/- {np.std(w['acc']):.1f}%  "
          f"({w['params']} params)")
    print("  - Winner chosen by MEAN accuracy, not best single run.")
    print("  - Lower std = more RELIABLE (consistent performance).")
    print("  - More parameters != better results on small datasets.")
    print()

    # Visualization 1: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [np.mean(results[c]['acc']) for c in configs]
    stds = [np.std(results[c]['acc']) for c in configs]
    colors = ['#2ecc71' if c == winner else '#3498db' for c in configs]
    bars = ax.bar(configs, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Wine Tournament -- Mean Accuracy +/- Std (3 seeds)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(max(0, min(means) - 10), 105)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'wine_tournament_bar.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # Visualization 2: Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([results[c]['acc'] for c in configs], labels=configs,
                    patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], configs):
        patch.set_facecolor('#2ecc71' if c == winner else '#85c1e9')
        patch.set_alpha(0.7)
    for i, c in enumerate(configs):
        ax.scatter(np.ones(len(results[c]['acc'])) * (i+1), results[c]['acc'],
                   color='#e74c3c', zorder=5, s=50, edgecolors='black', linewidth=0.5)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Wine -- Accuracy Distribution Across Seeds',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'wine_tournament_box.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}\n")

    return all_results


# ==============================================================================
# PART 3: CALIFORNIA HOUSING REGRESSION TOURNAMENT
# ==============================================================================

HOUSING_CONFIGS = {
    "Linear": lambda s: keras.Sequential([
        layers.Dense(1, input_shape=s)]),
    "Simple": lambda s: keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=s),
        layers.Dense(1)]),
    "Funnel": lambda s: keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=s),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)]),
    "Reg. Funnel": lambda s: keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=s,
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)]),
    "Deep": lambda s: keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=s),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)]),
}


def housing_regression_tournament():
    """
    ANALOGY: Engineering Stress Test.
    Five bridge designs (architectures) tested under the same load
    (California Housing). Each built three times (3 seeds) to test
    manufacturing consistency. We measure deflection (error) and R-squared.
    """
    print("=" * 70)
    print("PART 3: CALIFORNIA HOUSING REGRESSION TOURNAMENT")
    print("Same Rigor, Different Problem Type")
    print("=" * 70)
    print()
    print("  Dataset: California Housing -- 20,640 samples, 8 features")
    print("  Seeds: 42, 123, 7  |  Epochs: 80  |  Metrics: MSE, MAE, R2")
    print()

    housing = fetch_california_housing()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(housing.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, housing.target, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print()

    configs = list(HOUSING_CONFIGS.keys())
    seeds = [42, 123, 7]
    shape = (X_train.shape[1],)

    print("THE CONTENDERS:")
    print("-" * 50)
    for name in configs:
        set_all_seeds(42)
        m = HOUSING_CONFIGS[name](shape)
        print(f"  {name:<14} | {m.count_params():>6} params")
    print()

    print("TOURNAMENT IN PROGRESS...")
    print("=" * 60)
    results = {}
    all_results = []

    for name in configs:
        results[name] = {'mse': [], 'mae': [], 'r2': [], 'times': [], 'params': 0}
        print(f"\n  [{name}]")
        for seed in seeds:
            set_all_seeds(seed)
            model = HOUSING_CONFIGS[name](shape)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            results[name]['params'] = model.count_params()
            t0 = time.time()
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=80, batch_size=256, verbose=0)
            elapsed = time.time() - t0
            y_pred = model.predict(X_test, verbose=0).flatten()
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name]['mse'].append(mse)
            results[name]['mae'].append(mae)
            results[name]['r2'].append(r2)
            results[name]['times'].append(elapsed)
            all_results.append({
                'experiment': 'Housing Regression', 'model_name': name,
                'seed': seed, 'accuracy': None, 'loss': mse,
                'r2': r2, 'mse': mse, 'mae': mae,
                'params': model.count_params(), 'epochs_used': 80,
                'time_sec': round(elapsed, 2)})
            print(f"    seed={seed:>4}: R2={r2:.4f}, MSE={mse:.4f}, "
                  f"MAE={mae:.4f}, time={elapsed:.1f}s")

    # Scoreboard
    print("\n")
    print("+" + "=" * 76 + "+")
    print("|{:^76}|".format("HOUSING REGRESSION TOURNAMENT RESULTS"))
    print("+" + "=" * 76 + "+")
    print(f"| {'Model':<14} | {'R2(mean)':>9} | {'R2(std)':>8} | "
          f"{'MSE(mean)':>10} | {'MAE(mean)':>10} | {'Params':>7} |")
    print("+" + "-" * 76 + "+")

    best_r2, winner = -999, ""
    for name in configs:
        mr2, sr2 = np.mean(results[name]['r2']), np.std(results[name]['r2'])
        mmse = np.mean(results[name]['mse'])
        mmae = np.mean(results[name]['mae'])
        if mr2 > best_r2:
            best_r2, winner = mr2, name
        print(f"| {name:<14} | {mr2:>9.4f} | +/-{sr2:>.4f} | "
              f"{mmse:>10.4f} | {mmae:>10.4f} | {results[name]['params']:>7} |")
    print("+" + "=" * 76 + "+")

    w = results[winner]
    print(f"\n  WINNER: {winner}!")
    print(f"  Mean R2: {np.mean(w['r2']):.4f} +/- {np.std(w['r2']):.4f}  "
          f"({w['params']} params)")
    print("  - Linear baseline shows what a simple model can do.")
    print("  - Low R2 std = consistent. High std = unstable.")
    print("  - Regularized models trade peak performance for CONSISTENCY.")
    print()

    # Visualization 1: R2 and MSE side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label, higher_better in [
        (axes[0], 'r2', 'R-squared', True),
        (axes[1], 'mse', 'MSE', False)
    ]:
        vals = [np.mean(results[c][metric]) for c in configs]
        errs = [np.std(results[c][metric]) for c in configs]
        clrs = ['#2ecc71' if c == winner else
                ('#3498db' if higher_better else '#e74c3c') for c in configs]
        bars = ax.bar(configs, vals, yerr=errs, capsize=6, color=clrs,
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_ylabel(label, fontsize=12)
        hb = "higher" if higher_better else "lower"
        ax.set_title(f'{label} ({hb} is better)', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')
    fig.suptitle('Housing Regression Tournament (3 seeds)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'housing_tournament_metrics.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")

    # Visualization 2: Predicted vs Actual for winner
    best_seed = seeds[int(np.argmax(w['r2']))]
    set_all_seeds(best_seed)
    wm = HOUSING_CONFIGS[winner](shape)
    wm.compile(optimizer='adam', loss='mse', metrics=['mae'])
    wm.fit(X_train, y_train, validation_data=(X_val, y_val),
           epochs=80, batch_size=256, verbose=0)
    y_pred = wm.predict(X_test, verbose=0).flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#3498db')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual House Value ($100k)', fontsize=12)
    ax.set_ylabel('Predicted House Value ($100k)', fontsize=12)
    ax.set_title(f'Winner: {winner}\nR2={r2_score(y_test, y_pred):.4f}, '
                 f'MAE={mean_absolute_error(y_test, y_pred):.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'housing_predicted_vs_actual.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}\n")

    return all_results


# ==============================================================================
# PART 4: EXPERIMENT LOG TEMPLATE
# ==============================================================================

def experiment_log(wine_results, housing_results):
    """
    ANALOGY: Lab Notebook.
    Every scientist keeps a lab notebook -- every experiment, every variable,
    every result. The notebook IS the science. Without it, you have anecdotes.
    In ML, the experiment log is a DataFrame.
    """
    print("=" * 70)
    print("PART 4: EXPERIMENT LOG -- Your Lab Notebook")
    print("Real ML Teams Log EVERY Experiment")
    print("=" * 70)
    print()

    all_rows = wine_results + housing_results
    for i, row in enumerate(all_rows):
        row['experiment_id'] = f"EXP-{i+1:03d}"

    df = pd.DataFrame(all_rows)
    col_order = ['experiment_id', 'experiment', 'model_name', 'seed',
                 'accuracy', 'r2', 'mse', 'mae', 'loss', 'params',
                 'epochs_used', 'time_sec']
    df = df[col_order]

    # Print classification log
    wine_df = df[df['experiment'] == 'Wine Classification']
    print("  WINE CLASSIFICATION EXPERIMENTS:")
    print("  " + "-" * 62)
    print(f"  {'ID':<10} {'Model':<12} {'Seed':>5} {'Acc(%)':>8} "
          f"{'Loss':>8} {'Params':>7} {'Time':>6}")
    print("  " + "-" * 62)
    for _, r in wine_df.iterrows():
        print(f"  {r['experiment_id']:<10} {r['model_name']:<12} "
              f"{r['seed']:>5} {r['accuracy']:>8.1f} "
              f"{r['loss']:>8.4f} {r['params']:>7} {r['time_sec']:>6.1f}s")
    print()

    # Print regression log
    housing_df = df[df['experiment'] == 'Housing Regression']
    print("  HOUSING REGRESSION EXPERIMENTS:")
    print("  " + "-" * 70)
    print(f"  {'ID':<10} {'Model':<14} {'Seed':>5} {'R2':>8} "
          f"{'MSE':>8} {'MAE':>8} {'Params':>7} {'Time':>6}")
    print("  " + "-" * 70)
    for _, r in housing_df.iterrows():
        print(f"  {r['experiment_id']:<10} {r['model_name']:<14} "
              f"{r['seed']:>5} {r['r2']:>8.4f} {r['mse']:>8.4f} "
              f"{r['mae']:>8.4f} {r['params']:>7} {r['time_sec']:>6.1f}s")
    print()

    # Summary stats
    print("  SUMMARY (Wine -- Accuracy):")
    ws = wine_df.groupby('model_name')['accuracy'].agg(['mean','std','min','max']).round(2)
    ws.columns = ['Mean', 'Std', 'Min', 'Max']
    print(ws.to_string(index=True))
    print()
    print("  SUMMARY (Housing -- R2):")
    hs = housing_df.groupby('model_name')['r2'].agg(['mean','std','min','max']).round(4)
    hs.columns = ['Mean', 'Std', 'Min', 'Max']
    print(hs.to_string(index=True))
    print()

    # Save CSV
    csv_path = os.path.join(SCRIPT_DIR, 'experiment_log.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Experiment log saved: {csv_path}")
    print()

    # Interpretation
    print("READING THE EXPERIMENT LOG:")
    print("=" * 40)
    print("  -> Low std  = consistent model (good!)")
    print("  -> High std = unstable, needs more regularization")
    print("  -> More params does NOT always mean better")
    print("  -> The baseline is ALWAYS included for perspective")
    print("  -> Time matters: 0.1% better but 10x slower may not be worth it")
    print()

    # Visualization: Parameters vs Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sub_df, metric, ylabel, title in [
        (axes[0], wine_df, 'accuracy', 'Mean Accuracy (%)', 'Wine: Params vs Accuracy'),
        (axes[1], housing_df, 'r2', 'Mean R-squared', 'Housing: Params vs R2')
    ]:
        grp = sub_df.groupby('model_name').agg(
            {metric: ['mean', 'std'], 'params': 'first'})
        grp.columns = ['val_mean', 'val_std', 'params']
        ax.scatter(grp['params'], grp['val_mean'], s=150,
                   c='#3498db' if metric == 'accuracy' else '#e74c3c',
                   edgecolors='black', zorder=5)
        ax.errorbar(grp['params'], grp['val_mean'], yerr=grp['val_std'],
                    fmt='none', ecolor='gray', capsize=5, zorder=4)
        for name, row in grp.iterrows():
            ax.annotate(name, (row['params'], row['val_mean']),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight='bold')
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    fig.suptitle('More Parameters != Better Performance',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(SCRIPT_DIR, 'params_vs_performance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")
    print()
    print("  KEY TAKEAWAY: Bigger is not always better.")
    print("  The sweet spot = large ENOUGH to learn, small ENOUGH to generalize.")
    print("  Regularization lets you use bigger models without penalty.")
    print()


# ==============================================================================
# PART 5: BRIDGE TO COMPLETE PROJECTS
# ==============================================================================

def bridge_to_projects():
    """Summary of the Keras journey and practitioner's checklist."""
    print("=" * 70)
    print("PART 5: YOUR KERAS JOURNEY -- A PRACTITIONER'S CHECKLIST")
    print("=" * 70)
    print()
    print("  Sessions 41-42: The Complete Keras Toolkit")
    print("  " + "-" * 50)
    print("  [x] Loss functions decoded (when to use each)")
    print("  [x] Activation functions mastered (output layer rules)")
    print("  [x] Classification AND regression pipelines")
    print("  [x] Scaling (always StandardScaler for features)")
    print("  [x] Overfitting weapons (Dropout, L2, BatchNorm)")
    print("  [x] Optimizer choice (Adam default, SGD for fine control)")
    print("  [x] Architecture design (funnel pattern, width vs depth)")
    print("  [x] Callbacks (EarlyStopping, ReduceLR, ModelCheckpoint)")
    print()
    print("  Session 43: Advanced Keras & Experiments")
    print("  " + "-" * 50)
    print("  [x] Functional API (multi-input, skip connections)")
    print("  [x] Custom loss functions and metrics")
    print("  [x] Model interpretation (permutation importance)")
    print("  [x] Rigorous experiment methodology (multiple seeds)")
    print("  [x] Tournament-style model comparison")
    print("  [x] Experiment logging (DataFrame + CSV)")
    print("  [x] Statistical reporting (mean +/- std, not best run)")
    print()
    print("  THE GOLDEN RULES OF ML EXPERIMENTS:")
    print("  " + "=" * 50)
    print("  1. NEVER report a single run as your result")
    print("  2. ALWAYS include a baseline (even a dumb one)")
    print("  3. CONTROL everything except the variable you test")
    print("  4. USE the same data splits across all experiments")
    print("  5. REPORT mean +/- std (and min/max for context)")
    print("  6. LOG every experiment (you will thank yourself later)")
    print("  7. PLOT your results (bar charts with error bars)")
    print()
    print("  WHAT'S NEXT (Sessions 44-46):")
    print("  " + "-" * 50)
    print("  -> Complete end-to-end projects")
    print("  -> Real-world data (messy, missing values, categorical features)")
    print("  -> Model deployment and serving")
    print()
    print("  Be the practitioner, not the student who celebrates one lucky run.")
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("MODEL COMPARISON EXPERIMENT")
    print("The Scientific Method for Machine Learning")
    print("=" * 70)
    print()

    scientific_method_intro()
    input("\nPress Enter for the Wine Classification Tournament...")
    print()

    wine_results = wine_classification_tournament()
    input("\nPress Enter for the Housing Regression Tournament...")
    print()

    housing_results = housing_regression_tournament()
    input("\nPress Enter for the Experiment Log...")
    print()

    experiment_log(wine_results, housing_results)
    input("\nPress Enter for the final summary...")
    print()

    bridge_to_projects()

    print("=" * 70)
    print("SESSION 43 COMPLETE!")
    print("=" * 70)
    print()
    print("You can now run RIGOROUS ML experiments.")
    print("No more 'I got 95% once!' -- now it's 'Mean: 93.2% +/- 1.4%'")
    print()
    print("Plots saved:")
    for f in ['wine_tournament_bar.png', 'wine_tournament_box.png',
              'housing_tournament_metrics.png', 'housing_predicted_vs_actual.png',
              'params_vs_performance.png']:
        print(f"  - {os.path.join(SCRIPT_DIR, f)}")
    print(f"\nExperiment log: {os.path.join(SCRIPT_DIR, 'experiment_log.csv')}")
    print("\nYou are ready for Complete Projects in Sessions 44-46!")
    print("=" * 70)


if __name__ == '__main__':
    main()
