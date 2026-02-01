from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from datetime import datetime

# ==========================================
# CALLBACK 1: EarlyStopping
# ==========================================
# Stops training when validation performance stops improving

early_stopping = EarlyStopping(
    monitor='val_loss',        # Watch validation loss
    patience=10,               # Wait 10 epochs before stopping
    restore_best_weights=True, # Revert to best weights (critical!)
    verbose=1
)

# ==========================================
# CALLBACK 2: ModelCheckpoint
# ==========================================
# Automatically saves best model during training

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # Modern .keras format
    monitor='val_accuracy',       # Watch validation accuracy
    save_best_only=True,          # Only save when improving
    mode='max',                   # Maximize accuracy
    verbose=1
)

# ==========================================
# CALLBACK 3: ReduceLROnPlateau
# ==========================================
# Reduces learning rate when training plateaus

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # Multiply LR by 0.5 when plateau detected
    patience=5,              # Wait 5 epochs before reducing
    min_lr=1e-7,            # Don't go below this learning rate
    verbose=1
)

# ==========================================
# CALLBACK 4: TensorBoard
# ==========================================
# Real-time visualization of training

log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # Log weight distributions
    write_graph=True,        # Save model graph
    update_freq='epoch'      # Update after each epoch
)

# ==========================================
# BUILD MODEL WITH ALL CALLBACKS
# ==========================================

model_professional = keras.Sequential([
    layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01),
        input_shape=(30,)
    ),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),

    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid')
])

model_professional.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# TRAIN WITH PROFESSIONAL CALLBACKS
# ==========================================

print("="*70)
print("PROFESSIONAL TRAINING WITH CALLBACKS")
print("="*70)
print("Callbacks active:")
print("  - EarlyStopping: Stops when validation stops improving")
print("  - ModelCheckpoint: Saves best model automatically")
print("  - ReduceLROnPlateau: Adjusts learning rate dynamically")
print("  - TensorBoard: Real-time training visualization")
print("="*70)
print()

history_professional = model_professional.fit(
    X_train, y_train,
    epochs=200,  # Set high—EarlyStopping will stop us
    batch_size=16,
    validation_split=0.2,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr,
        tensorboard
    ],
    verbose=1
)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Stopped at epoch: {len(history_professional.history['loss'])}")
print(f"Best model saved to: best_model.keras")
print(f"TensorBoard logs: {log_dir}")
print()
print("To view TensorBoard:")
print(f"  tensorboard --logdir {log_dir}")
print("  Then open: http://localhost:6006")

# ==========================================
# LOAD BEST MODEL & EVALUATE
# ==========================================

print("\n" + "="*70)
print("EVALUATING BEST MODEL")
print("="*70)

# Load the best saved model
best_model = keras.models.load_model('best_model.keras')

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best Model Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Best Model Test Loss: {test_loss:.4f}")

# ==========================================
# CUSTOM CALLBACK EXAMPLE
# ==========================================

class CustomCallback(keras.callbacks.Callback):
    """
    Custom callback for production monitoring.
    Example: Log to external monitoring system, send alerts, etc.
    """

    def on_epoch_end(self, epoch, logs=None):
        # Called at end of each epoch
        if logs.get('val_accuracy') > 0.95:
            print(f"\n🎉 MILESTONE: Validation accuracy exceeded 95% at epoch {epoch+1}")

        # Example: Send to monitoring system
        # monitoring_system.log(epoch, logs)

    def on_train_end(self, logs=None):
        print("\n✅ Training completed successfully")

# Use custom callback
custom_monitor = CustomCallback()

# Add to callbacks list when training:
# callbacks=[early_stopping, checkpoint, reduce_lr, tensorboard, custom_monitor]