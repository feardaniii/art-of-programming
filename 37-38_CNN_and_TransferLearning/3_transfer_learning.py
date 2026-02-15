"""
================================================================================
TRANSFER LEARNING: Standing on Giants' Shoulders
================================================================================

Course: The Art of Programming - CNN & Transfer Learning (Session 38)
Lesson: How to Get 80-90% Accuracy with Only 500 Images

PREREQUISITES:
    - Script 1: Understand convolution and CNN architecture
    - Script 2: Trained a CNN from scratch on Fashion-MNIST

THE REVELATION:
    In Script 2, you trained a CNN on 60,000 images for 15 epochs.
    What if you only had 500 images? You would get terrible accuracy.

    Unless... you borrowed knowledge from someone who already did the work.

    This is transfer learning.
    This is the single most important technique
    in practical computer vision.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0

import time


# ==============================================================================
# PART 1: THE MOVING TO A NEW CITY ANALOGY
# ==============================================================================

def the_transfer_learning_revelation():
    """
    ANALOGY: Moving to a new city.

    You move from Bucharest to Berlin.
    Do you re-learn how to:
      - Walk? No.
      - Read signs? No (you learn German, but you already know HOW to read).
      - Recognize faces? No.
      - Drive? No (you learn new streets, but driving skill transfers).

    You keep ALL your general skills.
    You only learn the NEW specifics.

    Transfer learning works the same way:
      - Keep the general skills: edge detection, texture recognition, shape understanding
      - Learn only the new specifics: YOUR classes (cats vs dogs, disease vs healthy)
    """

    print("=" * 70)
    print("PART 1: TRANSFER LEARNING - The Paradigm Shift")
    print("=" * 70)
    print()

    print("ANALOGY: Moving to a new city")
    print()
    print("  You move from Bucharest to Berlin.")
    print("  Do you re-learn everything from scratch?")
    print()
    print("    Walking?         NO. You already know how.")
    print("    Reading signs?   PARTIALLY. Learn German, but reading skill transfers.")
    print("    Recognizing faces? NO. Same skill, different faces.")
    print("    Navigating streets? YES. New city, new map.")
    print()
    print("  Transfer learning is the same idea:")
    print("    Edge detection?    NO retraining needed. Edges are edges everywhere.")
    print("    Texture patterns?  NO retraining. Fur texture, metal texture -- universal.")
    print("    Shape recognition? PARTIAL. Adapt to your specific shapes.")
    print("    Final classification? YES. Train this for YOUR specific classes.")
    print()

    print("KNOWLEDGE TRANSFER HIERARCHY:")
    print()
    print("  ImageNet Model (trained on 14 MILLION images):")
    print("    Layer 1-5:   Edge detectors, texture patterns")
    print("                 --> REUSABLE for ANY vision task")
    print("    Layer 6-10:  Shape combinations, object parts")
    print("                 --> REUSABLE for similar domains")
    print("    Layer 11-15: ImageNet-specific objects (cats, cars, ships...)")
    print("                 --> REPLACE with your task")
    print("    Layer 16:    1000-class classification")
    print("                 --> REPLACE with your classes")
    print()
    print("  Your Custom Task (500 images):")
    print("    Layer 1-5:   FROZEN (use ImageNet weights)")
    print("    Layer 6-10:  FROZEN initially, fine-tune later")
    print("    Layer 11-15: TRAIN from scratch")
    print("    Layer 16:    NEW classification head")
    print()

    # Efficiency comparison table
    print("EFFICIENCY GAINS:")
    print()
    scenarios = [
        ("Train from scratch (500 imgs)", 500, 100, 0.62, "7 days", "$$$$"),
        ("Train from scratch (100K imgs)", 100000, 200, 0.94, "21 days", "$$$$$$$$"),
        ("Transfer learning (frozen)", 500, 10, 0.88, "30 min", "$"),
        ("Transfer learning (fine-tuned)", 500, 20, 0.95, "2 hours", "$$"),
    ]

    print(f"{'Method':<40} {'Images':<8} {'Epochs':<8} {'Accuracy':<10} {'Time':<10} {'Cost'}")
    print("-" * 90)
    for method, imgs, epochs, acc, t, cost in scenarios:
        print(f"{method:<40} {imgs:<8} {epochs:<8} {acc:<10.2f} {t:<10} {cost}")

    print()
    print("500 images + transfer learning = BETTER than 100,000 images from scratch!")
    print("This is the most important technique in practical computer vision.")
    print()


# ==============================================================================
# PART 2: LOADING A PRE-TRAINED MODEL
# ==============================================================================

def load_pretrained_model():
    """
    Load MobileNetV2 pre-trained on ImageNet (14 million images, 1000 classes).

    Why MobileNetV2?
    - Only 3.4 million parameters (vs 25M for ResNet50)
    - Trains fast even on CPU
    - Designed for mobile/edge deployment
    - Still achieves excellent accuracy

    The model already knows edges, textures, shapes, object parts.
    We just need to teach it OUR specific classes.
    """

    print("=" * 70)
    print("PART 2: LOADING A PRE-TRAINED MODEL")
    print("3.4 Million Parameters, Trained for Weeks, Free for You")
    print("=" * 70)
    print()

    # Load MobileNetV2 without the top classification layer
    print("Loading MobileNetV2 pre-trained on ImageNet...")
    base_model = MobileNetV2(
        weights='imagenet',     # Use ImageNet-trained weights
        include_top=False,      # Remove the 1000-class classification layer
        input_shape=(96, 96, 3) # Our input size (minimum 96x96 for MobileNetV2)
    )

    # Freeze all layers -- we do NOT want to change ImageNet knowledge
    base_model.trainable = False

    print(f"  Model: MobileNetV2")
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Total parameters: {base_model.count_params():,}")
    print(f"  Trainable: {base_model.trainable} (ALL layers frozen)")
    print()

    print("WHAT THIS MODEL ALREADY KNOWS:")
    print("  It was trained on ImageNet (14 million images, 1000 categories)")
    print("  It spent WEEKS on expensive GPUs learning:")
    print("    - How to detect edges in any orientation")
    print("    - How to recognize textures (fur, metal, wood, fabric)")
    print("    - How to identify shapes (circles, rectangles, curves)")
    print("    - How to combine parts into objects")
    print()
    print("  We get ALL of that knowledge for FREE.")
    print("  We only need to add a small classification head for OUR task.")
    print()

    return base_model


# ==============================================================================
# PART 3: PREPARING A SMALL DATASET
# ==============================================================================

def prepare_small_dataset():
    """
    Simulate a real-world scenario: you only have 500 images per class.

    We use CIFAR-10 (built-in to Keras) and extract just 2 classes:
    - Cats (class 3)
    - Dogs (class 5)
    - 500 training images per class = 1000 total

    Then resize to 96x96 for MobileNetV2.
    """

    print("=" * 70)
    print("PART 3: PREPARING A SMALL DATASET")
    print("The Real-World Scenario: Limited Data")
    print("=" * 70)
    print()

    # Load CIFAR-10
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.cifar10.load_data()
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    print("SCENARIO:")
    print("  You are building a Cat vs Dog classifier.")
    print("  You only have 500 images of each. Total: 1000 images.")
    print("  That is NOT enough to train a CNN from scratch.")
    print("  But it IS enough with transfer learning.")
    print()

    # Extract cats (3) and dogs (5)
    cat_label, dog_label = 3, 5
    n_per_class = 500

    # Training data
    cat_train_idx = np.where(y_train_full.flatten() == cat_label)[0][:n_per_class]
    dog_train_idx = np.where(y_train_full.flatten() == dog_label)[0][:n_per_class]
    train_idx = np.concatenate([cat_train_idx, dog_train_idx])

    X_train = X_train_full[train_idx]
    y_train = (y_train_full[train_idx].flatten() == dog_label).astype('float32')
    # cats = 0, dogs = 1

    # Test data (use ALL available test images for reliable evaluation)
    cat_test_idx = np.where(y_test_full.flatten() == cat_label)[0]
    dog_test_idx = np.where(y_test_full.flatten() == dog_label)[0]
    test_idx = np.concatenate([cat_test_idx, dog_test_idx])

    X_test = X_test_full[test_idx]
    y_test = (y_test_full[test_idx].flatten() == dog_label).astype('float32')

    print(f"Training images: {len(X_train)} ({n_per_class} cats + {n_per_class} dogs)")
    print(f"Test images: {len(X_test)} ({len(cat_test_idx)} cats + {len(dog_test_idx)} dogs)")
    print(f"Original image size: {X_train.shape[1:]} (32x32 RGB)")
    print()

    # Resize to 96x96 for MobileNetV2
    print("Resizing images from 32x32 to 96x96 for MobileNetV2...")
    X_train_resized = tf.image.resize(X_train, (96, 96)).numpy()
    X_test_resized = tf.image.resize(X_test, (96, 96)).numpy()

    # Normalize to [0, 1]
    X_train_resized = X_train_resized.astype('float32') / 255.0
    X_test_resized = X_test_resized.astype('float32') / 255.0

    print(f"Resized image size: {X_train_resized.shape[1:]}")
    print()

    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # Cats
        cat_idx = np.where(y_train == 0)[0][i]
        axes[0, i].imshow(X_train_resized[cat_idx])
        axes[0, i].set_title('Cat', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        # Dogs
        dog_idx = np.where(y_train == 1)[0][i]
        axes[1, i].imshow(X_train_resized[dog_idx])
        axes[1, i].set_title('Dog', fontsize=11, fontweight='bold')
        axes[1, i].axis('off')

    plt.suptitle(f'Cat vs Dog Dataset: Only {n_per_class} Per Class',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cats_dogs_samples.png', dpi=150, bbox_inches='tight')
    print("Sample images saved: cats_dogs_samples.png")
    plt.close()
    print()

    # Create validation split
    val_size = 100
    X_val = X_train_resized[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_resized[:-val_size]
    y_train_final = y_train[:-val_size]

    print(f"Final split:")
    print(f"  Training:   {len(X_train_final)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Test:       {len(X_test_resized)} images")
    print()

    return X_train_final, y_train_final, X_val, y_val, X_test_resized, y_test


# ==============================================================================
# PART 4: THE EXPERIMENT - From Scratch vs Transfer Learning
# ==============================================================================

def the_experiment(base_model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    THE HEART OF THIS SCRIPT.

    Same data. Same number of epochs. Two approaches:
    A) Train a CNN from scratch -> ~55-65% accuracy
    B) Transfer learning with MobileNetV2 -> ~85-95% accuracy

    The difference is DRAMATIC and proves why transfer learning matters.
    """

    print("=" * 70)
    print("PART 4: THE EXPERIMENT")
    print("From Scratch vs Transfer Learning -- Same Data, Same Epochs")
    print("=" * 70)
    print()

    epochs = 10
    batch_size = 32

    # =========================================
    # MODEL A: Train from scratch (the failure)
    # =========================================
    print("-" * 50)
    print("MODEL A: Training from SCRATCH")
    print("-" * 50)
    print()
    print("Building a small CNN from scratch (no pre-trained knowledge)...")

    scratch_model = keras.Sequential([
        layers.Input(shape=(96, 96, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='ScratchCNN')

    scratch_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    scratch_params = scratch_model.count_params()
    print(f"  Parameters: {scratch_params:,}")
    print(f"  Training on {len(X_train)} images for {epochs} epochs...")
    print()

    start = time.time()
    scratch_history = scratch_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    scratch_time = time.time() - start

    scratch_loss, scratch_acc = scratch_model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"  FROM SCRATCH RESULT: {scratch_acc*100:.1f}% test accuracy")
    print(f"  Training time: {scratch_time:.1f} seconds")
    print()

    if scratch_acc < 0.70:
        print("  That is BARELY better than flipping a coin (50%).")
    elif scratch_acc < 0.80:
        print("  Mediocre. Not good enough for any real application.")
    else:
        print("  Decent, but we can do MUCH better with transfer learning.")
    print()

    # =========================================
    # MODEL B: Transfer learning (the revelation)
    # =========================================
    print("-" * 50)
    print("MODEL B: Transfer Learning with MobileNetV2")
    print("-" * 50)
    print()
    print("Using MobileNetV2's ImageNet knowledge + a small custom head...")

    # Make sure base is frozen
    base_model.trainable = False

    transfer_model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name='TransferCNN')

    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    transfer_trainable = sum(
        tf.size(w).numpy() for w in transfer_model.trainable_weights
    )
    total_params = transfer_model.count_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {transfer_trainable:,}")
    print(f"  Frozen (reused) parameters: {total_params - transfer_trainable:,}")
    print(f"  We only train {transfer_trainable / total_params * 100:.1f}% of the model!")
    print()
    print(f"  Training on the SAME {len(X_train)} images for {epochs} epochs...")
    print()

    start = time.time()
    transfer_history = transfer_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    transfer_time = time.time() - start

    transfer_loss, transfer_acc = transfer_model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"  TRANSFER LEARNING RESULT: {transfer_acc*100:.1f}% test accuracy")
    print(f"  Training time: {transfer_time:.1f} seconds")
    print()

    # =========================================
    # COMPARISON
    # =========================================
    print("=" * 60)
    print("THE VERDICT")
    print("=" * 60)
    print()
    print(f"  {'Metric':<25} {'From Scratch':<18} {'Transfer Learning'}")
    print(f"  {'-'*60}")
    print(f"  {'Test Accuracy':<25} {scratch_acc*100:<18.1f} {transfer_acc*100:.1f}%")
    print(f"  {'Training Time':<25} {scratch_time:<18.1f} {transfer_time:.1f}s")
    print(f"  {'Training Images':<25} {len(X_train):<18} {len(X_train)}")
    print(f"  {'Epochs':<25} {epochs:<18} {epochs}")
    print()

    improvement = transfer_acc - scratch_acc
    print(f"  Transfer learning improved accuracy by {improvement*100:.1f} percentage points!")
    print()
    print("  SAME data. SAME epochs. SAME hardware.")
    print("  The ONLY difference: borrowed knowledge from ImageNet.")
    print("  This is the power of transfer learning.")
    print()

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(scratch_history.history['accuracy'], 'b--', label='Scratch (train)', linewidth=2)
    ax1.plot(scratch_history.history['val_accuracy'], 'b-', label='Scratch (val)', linewidth=2)
    ax1.plot(transfer_history.history['accuracy'], 'r--', label='Transfer (train)', linewidth=2)
    ax1.plot(transfer_history.history['val_accuracy'], 'r-', label='Transfer (val)', linewidth=2)
    ax1.set_title('Accuracy: Scratch vs Transfer', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])

    # Loss
    ax2.plot(scratch_history.history['loss'], 'b--', label='Scratch (train)', linewidth=2)
    ax2.plot(scratch_history.history['val_loss'], 'b-', label='Scratch (val)', linewidth=2)
    ax2.plot(transfer_history.history['loss'], 'r--', label='Transfer (train)', linewidth=2)
    ax2.plot(transfer_history.history['val_loss'], 'r-', label='Transfer (val)', linewidth=2)
    ax2.set_title('Loss: Scratch vs Transfer', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'The Transfer Learning Advantage ({len(X_train)} Training Images)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scratch_vs_transfer_learning.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved: scratch_vs_transfer_learning.png")
    plt.close()
    print()

    return transfer_model, base_model, scratch_history, transfer_history


# ==============================================================================
# PART 5: FINE-TUNING - Squeezing Out More Accuracy
# ==============================================================================

def fine_tune_model(transfer_model, base_model, X_train, y_train, X_val, y_val,
                    X_test, y_test, transfer_history):
    """
    Fine-tuning: Unfreeze the last few layers of the base model
    and train them with a VERY small learning rate.

    This adapts the pre-trained features to YOUR specific data.
    Think of it as: the model already knows textures in general,
    but now it refines its understanding of CAT fur vs DOG fur.

    IMPORTANT BEST PRACTICES:
    1. ALWAYS train the custom head FIRST (frozen base)
    2. Use a learning rate 10x smaller for fine-tuning
    3. Unfreeze only the LAST 20-30% of layers
    4. Monitor for overfitting (you have limited data)
    """

    print("=" * 70)
    print("PART 5: FINE-TUNING")
    print("Adapting Pre-Trained Features to Your Specific Data")
    print("=" * 70)
    print()

    print("The transfer model is already good.")
    print("But we can squeeze out MORE accuracy by fine-tuning.")
    print()
    print("FINE-TUNING = Unfreeze some pre-trained layers + train slowly")
    print()

    # Unfreeze the last 20 layers of MobileNetV2
    base_model.trainable = True
    n_layers = len(base_model.layers)
    n_unfreeze = 20

    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)

    print(f"  Total base model layers: {n_layers}")
    print(f"  Unfrozen layers: {trainable_count} (last {n_unfreeze})")
    print(f"  Frozen layers: {n_layers - trainable_count}")
    print()

    # Recompile with LOWER learning rate
    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 10x smaller!
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    trainable_params = sum(tf.size(w).numpy() for w in transfer_model.trainable_weights)
    total_params = transfer_model.count_params()
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Learning rate: 0.0001 (10x smaller than initial training)")
    print()

    print("FINE-TUNING BEST PRACTICES:")
    print("  1. ALWAYS train custom head first (Phase 1 = frozen base)")
    print("  2. Use learning rate 10-100x SMALLER for fine-tuning")
    print("  3. Unfreeze only the LAST 20-30% of layers")
    print("  4. Monitor validation loss -- overfitting risk is high!")
    print("  5. Use data augmentation aggressively")
    print("  6. Use early stopping")
    print()

    print("Fine-tuning for 5 epochs...")
    print("-" * 50)

    finetune_history = transfer_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    print("-" * 50)

    finetune_loss, finetune_acc = transfer_model.evaluate(X_test, y_test, verbose=0)
    print()
    print(f"  Fine-tuned accuracy: {finetune_acc*100:.1f}%")
    print()

    # 3-curve comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Transfer learning curve
    transfer_val_acc = transfer_history.history['val_accuracy']
    # Fine-tuning continues from where transfer left off
    finetune_val_acc = finetune_history.history['val_accuracy']
    combined_epochs = list(range(1, len(transfer_val_acc) + 1))
    finetune_epochs = list(range(len(transfer_val_acc) + 1,
                                  len(transfer_val_acc) + len(finetune_val_acc) + 1))

    ax.plot(combined_epochs, transfer_val_acc, 'b-o',
            label='Phase 1: Frozen Base', linewidth=2, markersize=6)
    ax.plot(finetune_epochs, finetune_val_acc, 'r-s',
            label='Phase 2: Fine-Tuned', linewidth=2, markersize=6)

    # Add vertical line at phase boundary
    ax.axvline(x=len(transfer_val_acc), color='gray', linestyle='--', alpha=0.5)
    ax.text(len(transfer_val_acc) + 0.2, 0.5, 'Unfreeze layers',
            fontsize=10, color='gray', rotation=90, va='bottom')

    ax.set_title('Transfer Learning: Frozen vs Fine-Tuned', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig('fine_tuning_comparison.png', dpi=150, bbox_inches='tight')
    print("Fine-tuning plot saved: fine_tuning_comparison.png")
    plt.close()
    print()

    return transfer_model, finetune_acc


# ==============================================================================
# PART 6: COMPARING ARCHITECTURES
# ==============================================================================

def compare_architectures():
    """
    Quick reference: popular pre-trained models for transfer learning.

    Choose based on your constraints:
    - MobileNetV2: fast, small, good for mobile/edge
    - ResNet50: balanced, widely used, good default
    - EfficientNet: best accuracy per parameter, state-of-the-art
    - VGG16: simple, large, good for understanding (but slow)
    """

    print("=" * 70)
    print("PART 6: CHOOSING THE RIGHT PRE-TRAINED MODEL")
    print("Quick Reference Guide")
    print("=" * 70)
    print()

    architectures = [
        ("MobileNetV2", "2018", "3.4M", "Light, fast", "Mobile/edge deployment"),
        ("ResNet50", "2015", "25.6M", "Skip connections", "General-purpose, balanced"),
        ("EfficientNetB0", "2019", "5.3M", "Compound scaling", "Best accuracy/efficiency"),
        ("VGG16", "2014", "138M", "Simple, deep", "Teaching/baseline"),
    ]

    print(f"{'Model':<18} {'Year':<6} {'Params':<10} {'Key Innovation':<22} {'Best For'}")
    print("-" * 85)
    for name, year, params, innovation, best_for in architectures:
        print(f"{name:<18} {year:<6} {params:<10} {innovation:<22} {best_for}")

    print()
    print("WHEN TO USE EACH:")
    print()
    print("  MobileNetV2:")
    print("    - You need FAST inference (real-time, mobile apps)")
    print("    - Training on CPU (no GPU available)")
    print("    - Deploying to phones or edge devices")
    print()
    print("  ResNet50:")
    print("    - General-purpose vision tasks")
    print("    - Good balance of speed and accuracy")
    print("    - When in doubt, start here")
    print()
    print("  EfficientNet:")
    print("    - Need the BEST accuracy possible")
    print("    - Willing to trade some speed for quality")
    print("    - State-of-the-art results required")
    print()
    print("  VGG16:")
    print("    - Learning/teaching purposes")
    print("    - Simple architecture, easy to understand")
    print("    - Do NOT use in production (too large and slow)")
    print()


# ==============================================================================
# PART 7: SAVING AND LOADING THE MODEL
# ==============================================================================

def save_and_load_model(model, X_test, y_test):
    """
    Your trained model is valuable. Save it so you can:
    - Use it later without retraining
    - Deploy it to a server or app
    - Share it with colleagues
    """

    print("=" * 70)
    print("PART 7: SAVING AND LOADING YOUR MODEL")
    print("From Training to Deployment")
    print("=" * 70)
    print()

    # Save
    model_path = 'cat_vs_dog_classifier.keras'
    model.save(model_path)
    print(f"  Model saved to: {model_path}")
    print()

    # Load
    loaded_model = keras.models.load_model(model_path)
    print(f"  Model loaded from: {model_path}")
    print()

    # Verify: same predictions
    original_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    loaded_acc = loaded_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"  Original model accuracy: {original_acc*100:.1f}%")
    print(f"  Loaded model accuracy:   {loaded_acc*100:.1f}%")
    print(f"  Match: {np.isclose(original_acc, loaded_acc)}")
    print()

    # Demonstrate prediction on a single image
    sample = X_test[0:1]
    prediction = loaded_model.predict(sample, verbose=0)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"  Example prediction:")
    print(f"    Actual: {'Dog' if y_test[0] == 1 else 'Cat'}")
    print(f"    Predicted: {predicted_class} (confidence: {confidence*100:.1f}%)")
    print()
    print("  Your model is saved. It can be deployed to:")
    print("    - A web server (Flask, FastAPI)")
    print("    - A mobile app (TensorFlow Lite)")
    print("    - An edge device (Raspberry Pi, Jetson)")
    print("    - A cloud service (AWS, GCP, Azure)")
    print()


# ==============================================================================
# MAIN: The Complete Transfer Learning Journey
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("SESSION 38 - SCRIPT 3: TRANSFER LEARNING")
    print("Standing on Giants' Shoulders")
    print("=" * 70)
    print()

    # Part 1: The concept
    the_transfer_learning_revelation()
    input("Press Enter to continue to Part 2 (Load Pre-Trained Model)...")
    print()

    # Part 2: Load MobileNetV2
    base_model = load_pretrained_model()
    input("Press Enter to continue to Part 3 (Prepare Dataset)...")
    print()

    # Part 3: Prepare cats vs dogs
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_small_dataset()
    input("Press Enter to continue to Part 4 (THE EXPERIMENT)...")
    print()

    # Part 4: From scratch vs transfer learning
    transfer_model, base_model, scratch_hist, transfer_hist = the_experiment(
        base_model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    input("Press Enter to continue to Part 5 (Fine-Tuning)...")
    print()

    # Part 5: Fine-tuning
    final_model, final_acc = fine_tune_model(
        transfer_model, base_model, X_train, y_train,
        X_val, y_val, X_test, y_test, transfer_hist
    )
    input("Press Enter to continue to Part 6 (Architecture Comparison)...")
    print()

    # Part 6: Architecture comparison
    compare_architectures()
    input("Press Enter to continue to Part 7 (Save & Load)...")
    print()

    # Part 7: Save and load
    save_and_load_model(final_model, X_test, y_test)

    # Summary
    print("=" * 70)
    print("SCRIPT 3 COMPLETE: Transfer Learning Mastered")
    print("=" * 70)
    print()
    print("What you accomplished:")
    print("  1. Understood WHY transfer learning works (knowledge hierarchy)")
    print("  2. Loaded a pre-trained model (MobileNetV2, 3.4M parameters)")
    print("  3. Prepared a small dataset (500 cats + 500 dogs)")
    print("  4. PROVED transfer learning works: same data, same epochs, dramatically better")
    print("  5. Fine-tuned for even higher accuracy")
    print("  6. Know which architecture to choose for different scenarios")
    print("  7. Saved your model for deployment")
    print()
    print("THE KEY TAKEAWAY:")
    print("  With 500 images, training from scratch is HOPELESS.")
    print("  With transfer learning, you get PRODUCTION-QUALITY results.")
    print("  This is how 90% of real-world computer vision is done today.")
    print()
    print("Next: Script 4 -- Beyond CNNs: Style Transfer, GANs, CLIP, Transformers.")
    print("=" * 70)


if __name__ == '__main__':
    main()
