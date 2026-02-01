import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# APPLICATION 1: HEALTH - DISEASE PREDICTION
# ==========================================

def health_disease_prediction():
    """
    HEALTH PROJECT: Predict disease risk from symptoms & vitals

    Features: Age, Blood Pressure, Cholesterol, BMI, Glucose, Family History
    Target: Disease Risk (0 = Low, 1 = High)

    This is your Linear Algebra (weights) + Calculus (gradients) +
    NumPy (data) converging into MEDICAL AI.
    """

    print("=" * 70)
    print("HEALTH APPLICATION: Disease Risk Prediction Neural Network")
    print("=" * 70)
    print()

    # Simulated patient data
    np.random.seed(42)
    n_patients = 1000

    # Features
    age = np.random.randint(20, 80, n_patients)
    bp_systolic = np.random.randint(90, 180, n_patients)
    cholesterol = np.random.randint(150, 300, n_patients)
    bmi = np.random.uniform(18, 40, n_patients)
    glucose = np.random.randint(70, 200, n_patients)
    family_history = np.random.choice([0, 1], n_patients)  # 0=No, 1=Yes

    # Create risk (complex non-linear relationship)
    risk_score = (
        (age - 20) * 0.02 +
        (bp_systolic - 90) * 0.01 +
        (cholesterol - 150) * 0.005 +
        (bmi - 18) * 0.05 +
        (glucose - 70) * 0.008 +
        family_history * 0.5 +
        np.random.randn(n_patients) * 0.1  # Noise
    )

    # Binary classification (threshold at median)
    disease_risk = (risk_score > np.median(risk_score)).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'bp_systolic': bp_systolic,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'glucose': glucose,
        'family_history': family_history,
        'disease_risk': disease_risk
    })

    print("Patient Dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"High risk patients: {disease_risk.sum()} ({disease_risk.mean()*100:.1f}%)")
    print()

    # Prepare data
    X = df.drop('disease_risk', axis=1).values
    y = df['disease_risk'].values

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (important for neural networks!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data Preparation:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print()

    # ==========================================
    # BUILD NEURAL NETWORK WITH TENSORFLOW
    # ==========================================

    print("Building Neural Network Architecture...")
    print()

    model = keras.Sequential([
        # Input layer (6 features)
        keras.layers.Dense(16, activation='relu', input_shape=(6,), name='hidden_layer_1'),
        keras.layers.Dense(8, activation='relu', name='hidden_layer_2'),
        keras.layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    # Display architecture
    model.summary()
    print()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Model compiled with:")
    print("  Optimizer: Adam (adaptive learning rate)")
    print("  Loss: Binary Cross-Entropy (better than MSE for classification)")
    print("  Metrics: Accuracy")
    print()

    # Train model
    print("Training neural network...")
    print()

    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print()

    # Make predictions
    predictions = model.predict(X_test_scaled[:5], verbose=0)

    print("Sample Predictions:")
    print("Patient  Age   BP    Chol   BMI    Gluc   FamHist  Pred Risk  Actual")
    print("-" * 80)

    for i in range(5):
        patient_data = X_test[i]
        pred_prob = predictions[i, 0]
        pred_class = 1 if pred_prob > 0.5 else 0
        actual = y_test[i]

        print(f"{i+1}  {patient_data[0]:.0f}  {patient_data[1]:.0f}  {patient_data[2]:.0f}  " +
              f"{patient_data[3]:.1f}  {patient_data[4]:.0f}  {patient_data[5]:.0f}  " +
              f"{pred_prob*100:.1f}%  {actual}")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('health_disease_nn_training.png', dpi=300, bbox_inches='tight')
    print("\n✅ Training curves saved: health_disease_nn_training.png")

    print()
    print("🏥 HEALTH AI DEPLOYED!")
    print("This network can predict disease risk from patient vitals")
    print("Your calculus gradients just became medical intelligence")

health_disease_prediction()

# ==========================================
# APPLICATION 2: FINANCE - CREDIT RISK
# ==========================================

def finance_credit_risk():
    """
    FINANCE PROJECT: Predict loan default risk

    Features: Income, Age, Debt-to-Income, Credit Score, Loan Amount, Employment Years
    Target: Default Risk (0 = Safe, 1 = Risky)

    This is the same mathematics. Different domain.
    Same convergence of Linear Algebra + Calculus.
    """

    print()
    print("=" * 70)
    print("FINANCE APPLICATION: Credit Risk Neural Network")
    print("=" * 70)
    print()

    # Simulated loan application data
    np.random.seed(42)
    n_applications = 1200

    # Features
    income = np.random.uniform(20000, 150000, n_applications)
    age = np.random.randint(18, 70, n_applications)
    debt_to_income = np.random.uniform(0, 0.8, n_applications)
    credit_score = np.random.randint(300, 850, n_applications)
    loan_amount = np.random.uniform(5000, 50000, n_applications)
    employment_years = np.random.randint(0, 30, n_applications)

    # Create default risk (non-linear relationship)
    risk_score = (
        -income / 50000 +
        debt_to_income * 3 +
        (850 - credit_score) / 100 +
        loan_amount / 10000 +
        -employment_years * 0.05 +
        np.random.randn(n_applications) * 0.2
    )

    default_risk = (risk_score > np.median(risk_score)).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'age': age,
        'debt_to_income': debt_to_income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'employment_years': employment_years,
        'default_risk': default_risk
    })

    print("Loan Applications Dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"Risky applications: {default_risk.sum()} ({default_risk.mean()*100:.1f}%)")
    print()

    # Prepare data
    X = df.drop('default_risk', axis=1).values
    y = df['default_risk'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    print("Building Credit Risk Neural Network...")
    print()

    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        keras.layers.Dropout(0.2),  # Regularization to prevent overfitting
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    model.summary()
    print()

    # Train
    print("Training credit risk model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    print()
    print("=" * 70)
    print("CREDIT RISK MODEL EVALUATION")
    print("=" * 70)

    test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    print()

    # Sample predictions
    predictions = model.predict(X_test_scaled[:5], verbose=0)

    print("Sample Loan Application Predictions:")
    print("App   Income      Age   DTI    Credit    Loan Amt    Risk %      Decision")
    print("-" * 85)

    for i in range(5):
        app_data = X_test[i]
        pred_prob = predictions[i, 0]
        decision = "RISKY" if pred_prob > 0.5 else "SAFE"

        print(f"{i+1}  {app_data[0]:.0f}  {app_data[1]:.0f}  {app_data[2]:.2f}  " +
              f"{app_data[3]:.0f}  {app_data[4]:.0f}  {pred_prob*100:.1f}%  {decision}")

    print()
    print("💰 FINANCE AI DEPLOYED!")
    print("Neural network predicting credit risk for loan approvals")
    print("Same backpropagation. Same chain rule. Different billions.")

finance_credit_risk()

print()
print("=" * 80)
print("SESSION 36 COMPLETE - THE CONVERGENCE")
print("=" * 80)
print()
print("You started with:")
print("  📐 Linear Algebra → Became weight matrices")
print("  📊 Calculus → Became gradient descent")
print("  🐍 NumPy → Became data pipelines")
print("  🧮 Pandas → Became feature engineering")
print()
print("You built:")
print("  🧠 Neurons → From McCulloch-Pitts 1943")
print("  🔗 Networks → From Rosenblatt 1958")
print("  🔄 Backprop → From Rumelhart 1986")
print("  🚀 TensorFlow → Production ML 2012+")
print()
print("You deployed:")
print("  🏥 Health: Disease risk prediction")
print("  💰 Finance: Credit risk assessment")
print()
print("This is not magic. This is mathematics.")
print("This is not memorization. This is understanding.")
print("This is why you learned the foundations.")
print()
print("Next: Deeper networks. CNNs. RNNs. Transformers.")
print("But you have the core. Everything else is architecture.")
print()
print("🎓 You've mastered Deep Learning fundamentals.")
print("=" * 80)
