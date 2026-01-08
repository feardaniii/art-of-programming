# =========== EX NO 2 ===============

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid activation function
    Formula: 1 / (1 + e^(-x))
    Maps any real number to (0, 1)
    """
    return 1 / (1 + np.exp(-x))

# Set seed for reproducibility
np.random.seed(42)

# Generate 100 random values between -10 and 10
date_aleatorii = np.random.uniform(-10, 10, 100)

# Apply sigmoid (vectorized - no loop needed!)
rezultate = sigmoid(date_aleatorii)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(date_aleatorii, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribuția valorilor inițiale', fontsize=14, fontweight='bold')
plt.xlabel('Valoare')
plt.ylabel('Frecvență')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(rezultate, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribuția după aplicarea sigmoid', fontsize=14, fontweight='bold')
plt.xlabel('Valoare')
plt.ylabel('Frecvență')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics
print("="*60)
print("STATISTICI")
print("="*60)
print(f"\nDate inițiale:")
print(f"  Media:    {np.mean(date_aleatorii):.3f}")
print(f"  Min/Max:  {np.min(date_aleatorii):.3f} / {np.max(date_aleatorii):.3f}")

print(f"\nDupă sigmoid:")
print(f"  Media:    {np.mean(rezultate):.3f}")
print(f"  Min/Max:  {np.min(rezultate):.3f} / {np.max(rezultate):.3f}")

print("\n✓ Observație: Toate valorile sunt acum în intervalul (0, 1)!")



# ======================== EX NO. 1 ========================


class NeuronSimplu:
    def __init__(self, numar_intrari):
        """
        Initialize neuron with random weights and bias
        """
        # Random weights for each input
        self.weights = np.random.randn(numar_intrari)
        # Random bias
        self.bias = np.random.randn()
        
        print(f"✓ Neuron initialized with {numar_intrari} inputs")
        print(f"  Weights: {self.weights}")
        print(f"  Bias: {self.bias:.4f}")
    
    def relu(self, x):
        """
        ReLU activation: max(0, x)
        """
        return np.maximum(0, x)
    
    def forward(self, intrari):
        """
        Forward pass through neuron
        output = ReLU(weights · inputs + bias)
        """
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, intrari) + self.bias
        
        print(f"\n{'='*50}")
        print(f"FORWARD PASS")
        print(f"{'='*50}")
        print(f"Inputs:        {intrari}")
        print(f"Weights:       {self.weights}")
        print(f"Bias:          {self.bias:.4f}")
        print(f"Weighted sum:  {weighted_sum:.4f}")
        
        # Apply activation
        output = self.relu(weighted_sum)
        
        print(f"After ReLU:    {output:.4f}")
        print(f"{'='*50}")
        
        return output

# ==================== TESTING ====================

# Set seed for reproducibility
np.random.seed(42)

# Create neuron with 3 inputs
print("Creating neuron...")
neuron = NeuronSimplu(3)

# Test input
intrare_test = np.array([1.0, 2.0, -0.5])

# Get output
output = neuron.forward(intrare_test)

print(f"\n✓ Final output: {output:.4f}")



# ==================== EX NO. 3 ============================


class ReteasNeuronala:
    def __init__(self, dim_intrare, dim_ascuns, dim_iesire):
        """
        Initialize 2-layer neural network
        """
        print(f"🧠 Creating neural network:")
        print(f"   Input:  {dim_intrare} → Hidden: {dim_ascuns} → Output: {dim_iesire}")
        
        # Layer 1: input → hidden (with ReLU)
        self.W1 = np.random.randn(dim_intrare, dim_ascuns) * 0.1
        self.b1 = np.zeros((1, dim_ascuns))
        
        # Layer 2: hidden → output (with sigmoid)
        self.W2 = np.random.randn(dim_ascuns, dim_iesire) * 0.1
        self.b2 = np.zeros((1, dim_iesire))
        
        print(f"✓ Network initialized")
        print(f"  W1: {self.W1.shape}, b1: {self.b1.shape}")
        print(f"  W2: {self.W2.shape}, b2: {self.b2.shape}")
    
    def relu(self, Z):
        """ReLU: max(0, Z)"""
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        """Sigmoid: 1 / (1 + e^(-Z))"""
        return 1 / (1 + np.exp(-Z))
    
    def forward(self, X):
        """
        Forward pass through network
        X (batch_size, dim_intrare) → A2 (batch_size, dim_iesire)
        """
        # Layer 1: X → Z1 → A1 (with ReLU)
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # Layer 2: A1 → Z2 → A2 (with sigmoid)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        return A2
    
    def prezice(self, X):
        """
        Make binary predictions (0 or 1)
        """
        output = self.forward(X)
        return (output > 0.5).astype(int)


# ==================== TESTING ====================

# Set seed for reproducibility
np.random.seed(42)

print("="*70)
print("NEURAL NETWORK TEST")
print("="*70)

# Create network: 4 inputs → 5 hidden → 2 outputs
retea = ReteasNeuronala(dim_intrare=4, dim_ascuns=5, dim_iesire=2)

# Create test data: 10 examples, 4 features each
X_test = np.random.randn(10, 4)
print(f"\n📊 Test data created: {X_test.shape}")
print(f"   (10 examples, 4 features each)")

# Get predictions
print(f"\n{'='*70}")
print("MAKING PREDICTIONS")
print(f"{'='*70}")

predictii = retea.forward(X_test)
clase_prezise = retea.prezice(X_test)

# Display results
print(f"\n📈 RESULTS:")
print(f"{'='*70}")
print(f"Input shape:  {X_test.shape}")
print(f"Output shape: {predictii.shape}")

print(f"\n🎲 First 3 examples:")
print(f"{'='*70}")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"  Input:        {X_test[i]}")
    print(f"  Probabilities: {predictii[i]}")
    print(f"  Predictions:   {clase_prezise[i]}")

print(f"\n{'='*70}")
print("✓ Network test complete!")
print(f"{'='*70}")

# Verify dimensions
print(f"\n🔍 DIMENSION VERIFICATION:")
print(f"{'='*70}")
print(f"W1: {retea.W1.shape} - connects {retea.W1.shape[0]} inputs to {retea.W1.shape[1]} hidden neurons")
print(f"b1: {retea.b1.shape}")
print(f"W2: {retea.W2.shape} - connects {retea.W2.shape[0]} hidden to {retea.W2.shape[1]} outputs")
print(f"b2: {retea.b2.shape}")
print(f"{'='*70}")