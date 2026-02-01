import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# STEP 1: Load and Prepare Data
# ==========================================

# Load Iris dataset
iris = load_iris()
X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
y = (iris.target == 0).astype(int)  # Binary: 1 if Setosa, 0 otherwise

print(f"Dataset shape: {X.shape}")  # (150, 4)
print(f"Targets: {np.unique(y, return_counts=True)}")  # [0, 1]: [100, 50]

# ==========================================
# STEP 2: Preprocessing
# ==========================================

# Standardize features (mean=0, std=1)
# WHY? Neural networks converge faster with normalized inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==========================================
# STEP 3: Build the Model (The Magic!)
# ==========================================

model = keras.Sequential([
    # Layer 1: Input layer (implicit) + First hidden layer
    layers.Dense(8, activation='relu', input_shape=(4,)),
    # 8 neurons, ReLU activation, expects 4 input features

    # Layer 2: Second hidden layer
    layers.Dense(4, activation='relu'),
    # 4 neurons, reduces dimensionality

    # Layer 3: Output layer
    layers.Dense(1, activation='sigmoid')
    # 1 neuron, sigmoid outputs probability [0, 1]
])

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()

# ==========================================
# STEP 4: Compile the Model
# ==========================================

model.compile(
    optimizer='adam',              # Adaptive learning rate optimizer
    loss='binary_crossentropy',    # Binary classification loss
    metrics=['accuracy']           # Track accuracy during training
)

# ==========================================
# STEP 5: Train the Model
# ==========================================

print("\n" + "="*60)
print("TRAINING")
print("="*60)

history = model.fit(
    X_train, y_train,
    epochs=50,                     # 50 complete passes through data
    batch_size=10,                 # Update weights after every 10 samples
    validation_split=0.2,          # Use 20% of training data for validation
    verbose=1                      # Show progress
)

# ==========================================
# STEP 6: Evaluate on Test Set
# ==========================================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ==========================================
# STEP 7: Make Predictions
# ==========================================

# Predict on first 5 test samples
predictions_prob = model.predict(X_test[:5])
predictions_class = (predictions_prob > 0.5).astype(int)

print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"  Probability: {predictions_prob[i][0]:.4f}")
    print(f"  Predicted Class: {predictions_class[i][0]}")
    print(f"  Actual Class: {y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]}")
    print()

# ==========================================
# STEP 8: Save the Model
# ==========================================

model.save('iris_model.h5')
print("Model saved as 'iris_model.h5'")

# To load later:
# loaded_model = keras.models.load_model('iris_model.h5')