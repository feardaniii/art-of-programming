"""
================================================================================
CNN WITH REAL IMAGES: The Complete Pipeline
================================================================================

Course: The Art of Programming - CNN & Transfer Learning (Session 37)
Lesson: Building, Training, and Evaluating a Real CNN

PREREQUISITES:
    - Script 1: Understand convolution, weight sharing, feature hierarchy
    - Module 36: TensorFlow basics, model.fit(), loss curves

THE DIFFERENCE FROM SCRIPT 1:
    Script 1: You understood the THEORY (manual convolution, scratch CNN)
    Script 2: You build a REAL system (Keras CNN, real data, real results)

    This is like the difference between understanding how an engine works
    and actually driving a car on real roads.

WHAT WE BUILD:
    A complete CNN pipeline on Fashion-MNIST:
    Load -> Explore -> Preprocess -> Augment -> Train -> Evaluate -> Visualize

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import time


# ==============================================================================
# PART 1: LOADING AND EXPLORING FASHION-MNIST
# ==============================================================================

def load_and_explore_data():
    """
    Fashion-MNIST: 70,000 grayscale images of clothing items.
    10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
                Sandal, Shirt, Sneaker, Bag, Ankle boot

    WHY Fashion-MNIST instead of MNIST digits?
    - MNIST digits are TOO EASY. A simple dense network gets 97%.
    - Fashion-MNIST is harder: shirts look like t-shirts, sneakers like boots.
    - Your CNN needs to EARN its accuracy here.
    """

    print("=" * 70)
    print("PART 1: LOADING AND EXPLORING FASHION-MNIST")
    print("=" * 70)
    print()

    # Class names (in order of label index 0-9)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Load dataset (built-in to Keras, no download needed)
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    print("Dataset loaded!")
    print()
    print(f"Training images: {X_train.shape} ({X_train.shape[0]:,} images)")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images:     {X_test.shape} ({X_test.shape[0]:,} images)")
    print(f"Test labels:     {y_test.shape}")
    print()

    print(f"Image size: {X_train.shape[1]}x{X_train.shape[2]} pixels (grayscale)")
    print(f"Pixel value range: [{X_train.min()}, {X_train.max()}]")
    print(f"Data type: {X_train.dtype}")
    print()

    # Class distribution
    print("Class distribution (training set):")
    for i, name in enumerate(class_names):
        count = np.sum(y_train == i)
        print(f"  {i}: {name:<15} {count:,} images ({count/len(y_train)*100:.1f}%)")
    print()
    print("Perfectly balanced! 6,000 images per class.")
    print()

    # Visualize: 2 rows x 5 columns = one example per class
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        # Find first image of this class
        idx = np.where(y_train == i)[0][0]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(f"{class_names[i]}\n(label={i})", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Fashion-MNIST: One Example Per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
    print("Visualization saved: fashion_mnist_samples.png")
    plt.close()
    print()

    print("CHALLENGE AHEAD:")
    print("  Some classes look very similar:")
    print("  - T-shirt (0) vs Shirt (6): both are upper-body garments")
    print("  - Pullover (2) vs Coat (4): both are long-sleeved")
    print("  - Sneaker (7) vs Ankle boot (9): both are shoes")
    print()
    print("  Can your CNN tell them apart? Let's find out.")
    print()

    return (X_train, y_train), (X_test, y_test), class_names


# ==============================================================================
# PART 2: DATA PREPROCESSING PIPELINE
# ==============================================================================

def preprocess_data(X_train, y_train, X_test, y_test):
    """
    THE PREPROCESSING CHECKLIST:
    1. Normalize pixel values (0-255 -> 0-1)
    2. Reshape to add channel dimension (Conv2D needs it)
    3. Create validation split (for monitoring overfitting)

    Every step has a reason. Skip one, and training suffers.
    """

    print("=" * 70)
    print("PART 2: DATA PREPROCESSING PIPELINE")
    print("The Preprocessing Checklist")
    print("=" * 70)
    print()

    # Step 1: Normalize
    print("Step 1: NORMALIZE pixel values")
    print(f"  Before: pixels in [{X_train.min()}, {X_train.max()}] (integers)")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    print(f"  After:  pixels in [{X_train.min():.1f}, {X_train.max():.1f}] (floats)")
    print()
    print("  WHY? Neural networks learn faster when inputs are small numbers.")
    print("  Gradient descent with values like 255 causes huge, unstable updates.")
    print("  Values near 0-1 keep gradients well-behaved.")
    print()

    # Step 2: Reshape
    print("Step 2: RESHAPE to add channel dimension")
    print(f"  Before: {X_train.shape} (batch, height, width)")
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    print(f"  After:  {X_train.shape} (batch, height, width, channels)")
    print()
    print("  WHY? Conv2D expects a channel dimension.")
    print("  Grayscale = 1 channel. RGB would be 3 channels.")
    print("  Even though our images are 2D, the conv layer needs that extra dimension.")
    print()

    # Step 3: Validation split
    print("Step 3: CREATE VALIDATION SPLIT")
    val_size = 6000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"  Training:   {X_train.shape[0]:,} images")
    print(f"  Validation: {X_val.shape[0]:,} images (10% of training)")
    print(f"  Test:       {X_test.shape[0]:,} images (untouched until final evaluation)")
    print()
    print("  WHY split training into train + validation?")
    print("  Training data: the model learns from this")
    print("  Validation data: we monitor overfitting with this (model never trains on it)")
    print("  Test data: final evaluation ONLY (touched once at the very end)")
    print()

    print("THE PREPROCESSING CHECKLIST (memorize this):")
    print("  [x] Normalize: 0-255 -> 0.0-1.0")
    print("  [x] Reshape: add channel dimension for Conv2D")
    print("  [x] Validate: split 10% for overfitting monitoring")
    print("  [ ] Augment: coming in Part 3!")
    print()

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# PART 3: DATA AUGMENTATION
# ==============================================================================

def explain_and_visualize_augmentation(X_train):
    """
    Data augmentation: create variations of your training images.

    A slightly rotated T-shirt is still a T-shirt.
    A horizontally flipped sneaker is still a sneaker.
    A slightly zoomed dress is still a dress.

    Augmentation gives you MORE training data for FREE.
    The model sees different perspectives of the same object,
    which makes it more robust to real-world variations.
    """

    print("=" * 70)
    print("PART 3: DATA AUGMENTATION")
    print("Free Training Data Through Smart Transformations")
    print("=" * 70)
    print()

    print("THE PROBLEM:")
    print("  60,000 training images sounds like a lot.")
    print("  But the real world has INFINITE variation:")
    print("    - Different angles")
    print("    - Different lighting")
    print("    - Slightly shifted position")
    print("    - Slightly different size")
    print()
    print("THE SOLUTION: Data augmentation")
    print("  Take each training image and create VARIATIONS:")
    print("    - Flip it horizontally (mirror)")
    print("    - Rotate it slightly (up to 10 degrees)")
    print("    - Zoom in slightly (up to 10%)")
    print("    - Shift it left/right/up/down (up to 10%)")
    print()
    print("  A flipped T-shirt is still a T-shirt!")
    print("  This effectively MULTIPLIES your dataset.")
    print()

    # Define augmentation layers
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),      # up to 10 degrees
        layers.RandomZoom(0.1),           # up to 10% zoom
        layers.RandomTranslation(0.1, 0.1),  # up to 10% shift
    ], name='data_augmentation')

    # Visualize augmentations of one image
    sample_image = X_train[0]  # A T-shirt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # First row: original
    axes[0, 0].imshow(sample_image[:, :, 0], cmap='gray')
    axes[0, 0].set_title('ORIGINAL', fontsize=11, fontweight='bold', color='blue')
    axes[0, 0].axis('off')

    # Generate augmented versions
    for i in range(1, 5):
        augmented = data_augmentation(tf.expand_dims(sample_image, 0), training=True)
        axes[0, i].imshow(augmented[0, :, :, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f'Augmented {i}', fontsize=11)
        axes[0, i].axis('off')

    # Second row: another image
    sample_image_2 = X_train[1]  # A Trouser
    axes[1, 0].imshow(sample_image_2[:, :, 0], cmap='gray')
    axes[1, 0].set_title('ORIGINAL', fontsize=11, fontweight='bold', color='blue')
    axes[1, 0].axis('off')

    for i in range(1, 5):
        augmented = data_augmentation(tf.expand_dims(sample_image_2, 0), training=True)
        axes[1, i].imshow(augmented[0, :, :, 0].numpy(), cmap='gray')
        axes[1, i].set_title(f'Augmented {i}', fontsize=11)
        axes[1, i].axis('off')

    plt.suptitle('Data Augmentation: Same Label, Different Perspective',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data_augmentation_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved: data_augmentation_demo.png")
    plt.close()
    print()

    print("Each image generates INFINITE variations during training.")
    print("The model never sees the exact same image twice!")
    print()

    return data_augmentation


# ==============================================================================
# PART 4: BUILDING THE CNN - Layer by Layer
# ==============================================================================

def build_cnn(data_augmentation):
    """
    Build a CNN for Fashion-MNIST classification.

    Architecture designed with purpose:
    - Block 1: Detect edges and simple textures (32 filters)
    - Block 2: Combine edges into shapes and patterns (64 filters)
    - Block 3: Combine patterns into clothing-level features (128 filters)
    - Classification head: Convert features to class probabilities

    Each block follows the pattern: Conv -> Conv -> Pool -> BatchNorm
    (Two conv layers per block capture richer patterns than one)
    """

    print("=" * 70)
    print("PART 4: BUILDING THE CNN - Layer by Layer")
    print("Every Layer Has a Purpose")
    print("=" * 70)
    print()

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),

        # Data augmentation (only active during training)
        data_augmentation,

        # ---- Block 1: Edge and texture detection ----
        # 32 flashlights, each 3x3, scanning for basic patterns
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        # Reduce from 28x28 to 14x14, keep strongest signals
        layers.MaxPooling2D((2, 2), name='pool1'),
        # Normalize activations for stable training
        layers.BatchNormalization(name='bn1'),

        # ---- Block 2: Shape and pattern detection ----
        # 64 flashlights operating on edge features
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        # Reduce from 14x14 to 7x7
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.BatchNormalization(name='bn2'),
        # Drop 25% of neurons randomly -> reduces overfitting
        layers.Dropout(0.25, name='dropout1'),

        # ---- Block 3: High-level feature combination ----
        # 128 flashlights combining shapes into clothing components
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        # Global average: take the average of each 7x7 feature map -> one number per filter
        layers.GlobalAveragePooling2D(name='gap'),

        # ---- Classification head ----
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout2'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='FashionCNN')

    print("ARCHITECTURE EXPLAINED:")
    print()
    print("  Input: 28x28x1 (grayscale clothing image)")
    print("    |")
    print("  Augmentation: random flip, rotate, zoom, shift (training only)")
    print("    |")
    print("  Block 1: Conv2D(32) + Conv2D(32) + MaxPool -> 14x14x32")
    print("    What it learns: edges, basic textures")
    print("    Why 32 filters: start small, learn simple patterns")
    print("    |")
    print("  Block 2: Conv2D(64) + Conv2D(64) + MaxPool -> 7x7x64")
    print("    What it learns: shapes, patterns (sleeves, collars)")
    print("    Why 64 filters: more complex patterns need more detectors")
    print("    |")
    print("  Block 3: Conv2D(128) + GlobalAveragePool -> 128")
    print("    What it learns: clothing components (full shoe shape, dress silhouette)")
    print("    Why GAP: more efficient than Flatten, reduces overfitting")
    print("    |")
    print("  Dense(128) -> Dropout(0.5) -> Dense(10, softmax)")
    print("    What it does: convert features into 10 class probabilities")
    print()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print()

    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    print(f"That is {total_params / 1e6:.2f} million parameters.")
    print(f"A fully-connected network for 28x28: {28*28*1000 + 1000*10:,} parameters")
    print(f"Our CNN is {(28*28*1000 + 1000*10) / total_params:.0f}x more efficient!")
    print()

    return model


# ==============================================================================
# PART 5: TRAINING - Watch the Network Learn
# ==============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    ACTUALLY TRAIN the CNN. Watch loss decrease and accuracy increase.

    Callbacks:
    - EarlyStopping: stop if validation loss stops improving (overfitting)
    - ReduceLROnPlateau: reduce learning rate when progress stalls
    """

    print("=" * 70)
    print("PART 5: TRAINING - Watch the Network Learn")
    print("=" * 70)
    print()

    print("Training configuration:")
    print(f"  Epochs: 15 (with early stopping)")
    print(f"  Batch size: 64")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: Sparse Categorical Crossentropy")
    print(f"  Training images: {X_train.shape[0]:,}")
    print(f"  Validation images: {X_val.shape[0]:,}")
    print()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print("WHAT TO WATCH:")
    print("  - 'loss' should DECREASE each epoch")
    print("  - 'accuracy' should INCREASE each epoch")
    print("  - 'val_loss' and 'val_accuracy' = performance on unseen data")
    print("  - If train accuracy >> val accuracy: OVERFITTING")
    print()
    print("Training begins...")
    print("-" * 50)

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    print("-" * 50)
    print(f"Training completed in {training_time:.1f} seconds")
    print()

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.0])

    # Loss curves
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('CNN Training Curves: Fashion-MNIST', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved: cnn_training_curves.png")
    plt.close()
    print()

    # Interpret the curves
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc

    print("INTERPRETING THE CURVES:")
    print(f"  Final training accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.1f}%)")
    print(f"  Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
    print(f"  Gap: {gap:.4f} ({gap*100:.1f}%)")
    print()

    if gap < 0.03:
        print("  Gap < 3%: Good generalization! No significant overfitting.")
    elif gap < 0.08:
        print("  Gap 3-8%: Mild overfitting. Acceptable for this model size.")
    else:
        print("  Gap > 8%: Significant overfitting! Consider more dropout/augmentation.")
    print()

    return history


# ==============================================================================
# PART 6: EVALUATION - Confusion Matrix and Per-Class Analysis
# ==============================================================================

def evaluate_model(model, X_test, y_test, class_names):
    """
    The REAL test: how well does the CNN perform on images it has NEVER seen?

    We look at:
    1. Overall accuracy
    2. Per-class precision, recall, F1
    3. Confusion matrix (which classes get confused?)
    4. Specific misclassified examples (WHY did the model fail?)
    """

    print("=" * 70)
    print("PART 6: EVALUATION - The Real Test")
    print("=" * 70)
    print()

    # Overall accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test set accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Test set loss: {test_loss:.4f}")
    print()

    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("PER-CLASS PERFORMANCE:")
    print("-" * 65)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=9)

    ax.set_title('Confusion Matrix: Fashion-MNIST CNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved: cnn_confusion_matrix.png")
    plt.close()
    print()

    # Find the most confused pair
    # Zero out the diagonal (correct predictions)
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    max_confusion = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
    confused_true = class_names[max_confusion[0]]
    confused_pred = class_names[max_confusion[1]]
    confused_count = cm_off_diag[max_confusion]

    print("MOST CONFUSED PAIR:")
    print(f"  '{confused_true}' misclassified as '{confused_pred}': {confused_count} times")
    print(f"  (Do YOU find them hard to distinguish?)")
    print()

    # Show misclassified examples
    misclassified = np.where(y_pred != y_test)[0]

    if len(misclassified) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flatten()):
            if i < len(misclassified):
                idx = misclassified[i]
                ax.imshow(X_test[idx, :, :, 0], cmap='gray')
                ax.set_title(
                    f"True: {class_names[y_test[idx]]}\n"
                    f"Pred: {class_names[y_pred[idx]]}\n"
                    f"Conf: {y_pred_probs[idx][y_pred[idx]]:.2f}",
                    fontsize=9, color='red'
                )
            ax.axis('off')

        plt.suptitle('Misclassified Examples (Red = Wrong Prediction)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cnn_misclassified.png', dpi=150, bbox_inches='tight')
        print("Misclassified examples saved: cnn_misclassified.png")
        plt.close()
    print()

    return y_pred, y_pred_probs


# ==============================================================================
# PART 7: VISUALIZING WHAT THE CNN LEARNED
# ==============================================================================

def visualize_learned_features(model, X_test):
    """
    THE MAGIC: What did the CNN actually learn?

    1. First-layer filters: the learned 'flashlights' (edge detectors, etc.)
    2. Feature maps: what each filter 'sees' when processing an image

    This is how we verify that the hierarchy actually emerges:
    Layer 1 should show edges
    Layer 2 should show textures and shapes
    Layer 3 should show high-level features
    """

    print("=" * 70)
    print("PART 7: VISUALIZING WHAT THE CNN LEARNED")
    print("Opening the Black Box")
    print("=" * 70)
    print()

    # ---- Part A: First layer filters ----
    print("A) LEARNED FILTERS (First Conv Layer)")
    print()

    # Find the first Conv2D layer
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            first_conv = layer
            break

    if first_conv is not None:
        filters = first_conv.get_weights()[0]  # Shape: (3, 3, 1, 32)
        n_filters = filters.shape[-1]

        print(f"  First conv layer has {n_filters} filters of size {filters.shape[0]}x{filters.shape[1]}")
        print("  Each filter is a learned 'flashlight' -- a pattern detector.")
        print()

        # Plot filters in a grid
        cols = 8
        rows = n_filters // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))

        for i, ax in enumerate(axes.flatten()):
            if i < n_filters:
                ax.imshow(filters[:, :, 0, i], cmap='gray', interpolation='none')
                ax.set_title(f'Filter {i}', fontsize=8)
            ax.axis('off')

        plt.suptitle('Learned Filters: First Convolutional Layer\n(Each is a 3x3 pattern detector)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cnn_learned_filters.png', dpi=150, bbox_inches='tight')
        print("  Learned filters saved: cnn_learned_filters.png")
        plt.close()

        print("  Notice: some filters look like edge detectors (vertical, horizontal)")
        print("  Others detect corners, gradients, or textures.")
        print("  The network DISCOVERED these patterns through training!")
        print()

    # ---- Part B: Feature maps ----
    print("B) FEATURE MAPS (What each layer sees)")
    print()

    # Get conv layer outputs
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]

    if len(conv_layers) > 0:
        # Create a model that outputs feature maps
        layer_outputs = [layer.output for layer in conv_layers]
        activation_model = keras.Model(inputs=model.inputs, outputs=layer_outputs)

        # Pick a test image
        sample = X_test[0:1]  # Keep batch dimension
        activations = activation_model.predict(sample, verbose=0)

        fig, all_axes = plt.subplots(len(activations), 8, figsize=(16, len(activations) * 2))

        for layer_idx, activation in enumerate(activations):
            n_maps = min(8, activation.shape[-1])
            for i in range(8):
                ax = all_axes[layer_idx, i] if len(activations) > 1 else all_axes[i]
                if i < n_maps:
                    ax.imshow(activation[0, :, :, i], cmap='viridis')
                    if i == 0:
                        ax.set_ylabel(f'Layer {layer_idx + 1}\n{activation.shape[1]}x{activation.shape[2]}',
                                     fontsize=9)
                ax.axis('off')

        plt.suptitle('Feature Maps: What Each Layer Sees\n(From simple edges to complex patterns)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cnn_feature_maps.png', dpi=150, bbox_inches='tight')
        print("  Feature maps saved: cnn_feature_maps.png")
        plt.close()
        print()

        print("  OBSERVE THE HIERARCHY:")
        for layer_idx, (activation, layer) in enumerate(zip(activations, conv_layers)):
            print(f"    Layer {layer_idx + 1} ({layer.name}): "
                  f"{activation.shape[1]}x{activation.shape[2]} spatial, "
                  f"{activation.shape[-1]} feature maps")

        print()
        print("  Layer 1: Large spatial size, detects LOW-LEVEL features (edges)")
        print("  Layer 2-3: Medium spatial, detects MID-LEVEL features (textures, shapes)")
        print("  Layer 4-5: Small spatial, detects HIGH-LEVEL features (object parts)")
        print()
        print("  The hierarchy EMERGED from training. We did not design it.")
        print("  We only stacked layers. The data taught the network WHAT to detect.")
        print()


# ==============================================================================
# MAIN: The Complete Pipeline
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("SESSION 37 - SCRIPT 2: CNN WITH REAL IMAGES")
    print("The Complete Pipeline: Load -> Preprocess -> Augment -> Train -> Evaluate")
    print("=" * 70)
    print()

    # Part 1: Load and explore
    (X_train, y_train), (X_test, y_test), class_names = load_and_explore_data()
    input("Press Enter to continue to Part 2 (Preprocessing)...")
    print()

    # Part 2: Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    input("Press Enter to continue to Part 3 (Data Augmentation)...")
    print()

    # Part 3: Augmentation
    data_augmentation = explain_and_visualize_augmentation(X_train)
    input("Press Enter to continue to Part 4 (Build CNN)...")
    print()

    # Part 4: Build
    model = build_cnn(data_augmentation)
    input("Press Enter to continue to Part 5 (Training)...")
    print()

    # Part 5: Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    input("Press Enter to continue to Part 6 (Evaluation)...")
    print()

    # Part 6: Evaluate
    y_pred, y_pred_probs = evaluate_model(model, X_test, y_test, class_names)
    input("Press Enter to continue to Part 7 (Visualize Features)...")
    print()

    # Part 7: Visualize
    visualize_learned_features(model, X_test)

    # Summary
    print("=" * 70)
    print("SCRIPT 2 COMPLETE: You Built a Real CNN Pipeline")
    print("=" * 70)
    print()
    print("What you accomplished:")
    print("  1. Loaded and explored Fashion-MNIST (60,000 real images)")
    print("  2. Preprocessed: normalized, reshaped, split")
    print("  3. Applied data augmentation (free training data!)")
    print("  4. Built a CNN with purpose: edge -> texture -> shape -> classification")
    print("  5. TRAINED it and watched the loss decrease")
    print("  6. Evaluated: confusion matrix, per-class metrics, misclassified examples")
    print("  7. Opened the black box: saw learned filters and feature maps")
    print()
    print("THE FULL PIPELINE:")
    print("  Load -> Explore -> Preprocess -> Augment -> Train -> Evaluate -> Visualize")
    print("  This pipeline works for ANY image classification problem.")
    print("  Change the data, keep the pipeline.")
    print()
    print("Next: Script 3 -- Transfer Learning.")
    print("      What if you only had 500 images instead of 60,000?")
    print("      Transfer learning makes the impossible possible.")
    print("=" * 70)


if __name__ == '__main__':
    main()
