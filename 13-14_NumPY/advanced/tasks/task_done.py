"""
Tema 15: Introducere în Rețele Neuronale cu NumPy
===================================================
Implementare completă cu explicații detaliate și research notes

Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Setăm seed pentru reproducibilitate
np.random.seed(42)

print("=" * 80)
print("TEMA 15: INTRODUCERE ÎN REȚELE NEURONALE CU NUMPY")
print("=" * 80)

# =============================================================================
# EXERCIȚIUL 1: Neuron Simplu cu Activare ReLU
# =============================================================================

print("\n" + "=" * 80)
print("EXERCIȚIUL 1: Neuron Simplu cu Activare ReLU")
print("=" * 80)

"""
Research Notes - Ce este un neuron?
-----------------------------------
Un neuron artificial simulează comportamentul unui neuron biologic:
- Primește mai multe intrări (dendrite)
- Fiecare intrare are o pondere (weight) - importanța semnalului
- Se adaugă un bias (prag de activare)
- Se aplică o funcție de activare (ReLU în cazul nostru)
- Produce un output

De ce ReLU?
-----------
ReLU (Rectified Linear Unit) = max(0, x)
- Simplu de calculat și derivat
- Evită problema "vanishing gradient" (față de sigmoid/tanh)
- Introduce non-linearitate (esențial pentru învățare complexă)
- Biologic plauzibil (neuronii fie se activează, fie nu)

Problema "Dying ReLU":
- Dacă neuronul primește valori negative constant, gradientul devine 0
- Neuronul "moare" și nu mai învață
- Soluții: Leaky ReLU, ELU, etc.
"""


class NeuronSimplu:
    def __init__(self, numar_intrari):
        """
        Inițializează un neuron cu parametri aleatorii

        Parameters:
        -----------
        numar_intrari : int
            Numărul de conexiuni de intrare ale neuronului
        """
        # Inițializăm weights folosind distribuția normală standard
        # Multiplicăm cu 0.1 pentru valori mici inițiale (evită saturarea)
        self.weights = np.random.randn(numar_intrari) * 0.1

        # Bias-ul începe de la o valoare mică aleatorie
        # Reprezintă pragul de activare al neuronului
        self.bias = np.random.randn() * 0.1

        print(f"Neuron inițializat cu {numar_intrari} intrări")
        print(f"Weights inițiale: {self.weights}")
        print(f"Bias inițial: {self.bias:.4f}")

    def relu(self, x):
        """
        Implementează funcția de activare ReLU
        ReLU(x) = max(0, x)

        Intuiție: Dacă suma ponderată e negativă, neuronul nu se activează (output 0)
                  Dacă e pozitivă, transmite semnalul proporțional cu intensitatea
        """
        return np.maximum(0, x)

    def forward(self, intrari):
        """
        Calculează output-ul neuronului (forward pass)

        Formula: output = ReLU(Σ(wi * xi) + b)
        unde:
        - wi = weight pentru intrarea i
        - xi = valoarea intrării i
        - b = bias

        Parameters:
        -----------
        intrari : numpy.ndarray
            Vector cu valorile de intrare

        Returns:
        --------
        float
            Output-ul neuronului după aplicarea ReLU
        """
        # Pasul 1: Calculăm suma ponderată (dot product)
        # Aceasta reprezintă "potențialul" neuronului
        suma_ponderata = np.dot(self.weights, intrari) + self.bias

        # Debugging: afișăm calculul intermediar
        print(f"\nCalcul detaliat:")
        print(f"Intrări: {intrari}")
        print(f"Weights: {self.weights}")
        print(f"Suma ponderată (înainte de bias): {np.dot(self.weights, intrari):.4f}")
        print(f"Suma ponderată (cu bias): {suma_ponderata:.4f}")

        # Pasul 2: Aplicăm funcția de activare ReLU
        output = self.relu(suma_ponderata)

        print(f"Output după ReLU: {output:.4f}")

        return output


# Testare Exercițiul 1
print("\nTESTARE NEURON SIMPLU:")
print("-" * 40)

neuron = NeuronSimplu(3)
intrare_test = np.array([1.0, 2.0, -0.5])

print(f"\nIntrare de test: {intrare_test}")
output = neuron.forward(intrare_test)

print(f"\n{'=' * 40}")
print(f"OUTPUT FINAL NEURON: {output:.4f}")
print(f"{'=' * 40}")

# Test adițional cu valori care vor produce output negativ (pentru a vedea ReLU în acțiune)
print("\nTest cu valori negative mari:")
intrare_negativa = np.array([-5.0, -3.0, -2.0])
output_negativ = neuron.forward(intrare_negativa)
print(f"Output pentru intrări negative: {output_negativ:.4f} (demonstrează efectul ReLU)")

# =============================================================================
# EXERCIȚIUL 2: Funcția Sigmoid pe Dataset Aleatoriu
# =============================================================================

print("\n" + "=" * 80)
print("EXERCIȚIUL 2: Funcția Sigmoid pe Dataset Aleatoriu")
print("=" * 80)

"""
Research Notes - Funcția Sigmoid
---------------------------------
Sigmoid: σ(x) = 1 / (1 + e^(-x))

Caracteristici:
- Mapează orice valoare reală în intervalul (0, 1)
- Formă de "S" - smooth și diferențiabilă peste tot
- Derivata: σ'(x) = σ(x) * (1 - σ(x)) - convenabilă pentru backpropagation
- Output interpretabil ca probabilitate

Utilizări:
- Output layer pentru clasificare binară
- Gates în LSTM/GRU
- Istoric: foarte populară înainte de ReLU

Probleme:
- Vanishing gradient: pentru |x| mare, gradientul → 0
- Computational expensive (exponențială)
- Output-ul nu e zero-centered
"""


def sigmoid(x):
    """
    Implementează funcția sigmoid: f(x) = 1 / (1 + e^(-x))

    Vectorizată - funcționează pe scalari, vectori și matrici

    Pentru valori foarte mari/mici, folosim np.clip pentru stabilitate numerică
    """
    # Evităm overflow pentru valori foarte mari negative
    # Clipping la [-500, 500] previne exp() overflow
    x_safe = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_safe))


# Demonstrație vizuală a funcției sigmoid
print("\nVIZUALIZARE FUNCȚIA SIGMOID:")
print("-" * 40)

# Creăm un range de valori pentru a vedea forma sigmoid
x_demo = np.linspace(-10, 10, 1000)
y_sigmoid = sigmoid(x_demo)

plt.figure(figsize=(12, 4))

# Subplot 1: Funcția sigmoid
plt.subplot(1, 3, 1)
plt.plot(x_demo, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Funcția Sigmoid')
plt.legend()
plt.ylim(-0.1, 1.1)

# Subplot 2: Comparație cu ReLU
plt.subplot(1, 3, 2)
y_relu = np.maximum(0, x_demo)
plt.plot(x_demo, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
plt.plot(x_demo, y_relu / 10, 'g-', linewidth=2, label='ReLU (scalat /10)')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sigmoid vs ReLU')
plt.legend()
plt.xlim(-10, 10)

# Subplot 3: Derivata sigmoid
plt.subplot(1, 3, 3)
y_derivative = y_sigmoid * (1 - y_sigmoid)
plt.plot(x_demo, y_derivative, 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("σ'(x)")
plt.title('Derivata Sigmoid\nσ\'(x) = σ(x)(1-σ(x))')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Generează 100 de valori aleatorii între -10 și 10
print("\nAPLICARE PE DATASET ALEATORIU:")
print("-" * 40)

date_aleatorii = np.random.uniform(-10, 10, 100)
print(f"Am generat 100 de valori aleatorii în intervalul [-10, 10]")

# Aplică sigmoid pe toate valorile (vectorizat - fără loop!)
rezultate = sigmoid(date_aleatorii)

# Vizualizare distribuții
plt.figure(figsize=(14, 6))

# Subplot 1: Distribuția valorilor inițiale
plt.subplot(2, 3, 1)
plt.hist(date_aleatorii, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.title('Distribuția valorilor inițiale')
plt.xlabel('Valoare')
plt.ylabel('Frecvență')

# Subplot 2: Distribuția după sigmoid
plt.subplot(2, 3, 2)
plt.hist(rezultate, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
plt.title('Distribuția după aplicarea sigmoid')
plt.xlabel('Valoare')
plt.ylabel('Frecvență')
plt.xlim(0, 1)

# Subplot 3: Scatter plot pentru a vedea transformarea
plt.subplot(2, 3, 3)
plt.scatter(date_aleatorii, rezultate, alpha=0.5)
plt.plot(x_demo, sigmoid(x_demo), 'r-', linewidth=2, alpha=0.5)
plt.xlabel('Valori inițiale')
plt.ylabel('După sigmoid')
plt.title('Transformarea sigmoid')
plt.grid(True, alpha=0.3)

# Subplot 4: Box plots comparative
plt.subplot(2, 3, 4)
plt.boxplot([date_aleatorii, rezultate], labels=['Inițial', 'După sigmoid'])
plt.title('Comparație Box Plot')
plt.ylabel('Valoare')
plt.grid(True, alpha=0.3)

# Subplot 5: CDF (Cumulative Distribution Function)
plt.subplot(2, 3, 5)
sorted_initial = np.sort(date_aleatorii)
sorted_sigmoid = np.sort(rezultate)
plt.plot(sorted_initial, np.arange(len(sorted_initial)) / len(sorted_initial),
         'b-', label='Inițial', linewidth=2)
plt.plot(sorted_sigmoid, np.arange(len(sorted_sigmoid)) / len(sorted_sigmoid),
         'g-', label='După sigmoid', linewidth=2)
plt.xlabel('Valoare')
plt.ylabel('CDF')
plt.title('Funcții de Distribuție Cumulative')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 6: Q-Q plot
plt.subplot(2, 3, 6)
plt.scatter(np.sort(date_aleatorii), np.sort(rezultate), alpha=0.5)
plt.xlabel('Quantile inițiale')
plt.ylabel('Quantile după sigmoid')
plt.title('Q-Q Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistici detaliate
print("\nSTATISTICI COMPARATIVE:")
print("-" * 40)
print(f"{'Metrică':<20} {'Inițial':>15} {'După Sigmoid':>15}")
print("-" * 50)
print(f"{'Media':<20} {np.mean(date_aleatorii):>15.3f} {np.mean(rezultate):>15.3f}")
print(f"{'Mediana':<20} {np.median(date_aleatorii):>15.3f} {np.median(rezultate):>15.3f}")
print(f"{'Deviația standard':<20} {np.std(date_aleatorii):>15.3f} {np.std(rezultate):>15.3f}")
print(f"{'Minim':<20} {np.min(date_aleatorii):>15.3f} {np.min(rezultate):>15.3f}")
print(f"{'Maxim':<20} {np.max(date_aleatorii):>15.3f} {np.max(rezultate):>15.3f}")
print(f"{'Percentila 25':<20} {np.percentile(date_aleatorii, 25):>15.3f} {np.percentile(rezultate, 25):>15.3f}")
print(f"{'Percentila 75':<20} {np.percentile(date_aleatorii, 75):>15.3f} {np.percentile(rezultate, 75):>15.3f}")

# Observații despre transformare
print("\nOBSERVAȚII CHEIE:")
print("-" * 40)
print("1. Sigmoid comprimă TOATE valorile în intervalul (0, 1)")
print("2. Valorile extreme (±10) devin foarte aproape de 0 sau 1")
print("3. Valorile din jurul lui 0 sunt mapate în jurul lui 0.5")
print("4. Distribuția devine mai 'concentrată' în mijloc")
print("5. Relațiile de ordine sunt păstrate (monotonie)")

# =============================================================================
# EXERCIȚIUL 3: Rețea Neuronală cu 2 Straturi (Fully Connected)
# =============================================================================

print("\n" + "=" * 80)
print("EXERCIȚIUL 3: Rețea Neuronală cu 2 Straturi")
print("=" * 80)

"""
Research Notes - Rețele Neuronale Multi-Layer
----------------------------------------------
De ce avem nevoie de mai multe straturi?
- Un singur neuron = doar decizii liniare
- Mai multe straturi = pot învăța funcții complexe, non-liniare
- Teorema aproximării universale: o rețea cu 1 strat ascuns poate aproxima orice funcție

Fully Connected (Dense) Layers:
- Fiecare neuron din stratul N e conectat la TOȚI neuronii din stratul N+1
- Număr parametri între straturi: (neuroni_strat_N × neuroni_strat_N+1) + neuroni_strat_N+1

Forward Propagation:
1. Input → Linear transformation (W1·x + b1) → Activare → Hidden Layer
2. Hidden → Linear transformation (W2·h + b2) → Activare → Output

Inițializarea parametrilor:
- Prea mici → semnalul dispare (vanishing)
- Prea mari → semnalul explodează (exploding)
- Soluții: Xavier/He initialization

Broadcasting în NumPy:
- Permite operații eficiente pe batch-uri
- Ex: (batch_size, features) × (features, neurons) = (batch_size, neurons)
"""


class ReteasNeuronala:
    def __init__(self, dim_intrare, dim_ascuns, dim_iesire):
        """
        Inițializează o rețea neuronală cu 2 straturi dense

        Arhitectura:
        Input (dim_intrare) → Hidden (dim_ascuns) → Output (dim_iesire)

        Parameters:
        -----------
        dim_intrare : int
            Numărul de features de intrare
        dim_ascuns : int
            Numărul de neuroni în stratul ascuns
        dim_iesire : int
            Numărul de neuroni de ieșire (clase)
        """
        print(f"Inițializare rețea neuronală:")
        print(f"  Arhitectură: {dim_intrare} → {dim_ascuns} → {dim_iesire}")

        # STRATUL 1: Input → Hidden
        # W1 shape: (dim_intrare, dim_ascuns)
        # Inițializare Xavier/Glorot pentru ReLU
        self.W1 = np.random.randn(dim_intrare, dim_ascuns) * np.sqrt(2.0 / dim_intrare)
        self.b1 = np.zeros((1, dim_ascuns))  # Shape (1, dim_ascuns) pentru broadcasting

        # STRATUL 2: Hidden → Output
        # W2 shape: (dim_ascuns, dim_iesire)
        self.W2 = np.random.randn(dim_ascuns, dim_iesire) * np.sqrt(2.0 / dim_ascuns)
        self.b2 = np.zeros((1, dim_iesire))

        # Calculăm numărul total de parametri
        total_params = (dim_intrare * dim_ascuns + dim_ascuns +  # Stratul 1
                        dim_ascuns * dim_iesire + dim_iesire)  # Stratul 2

        print(f"  Total parametri antrenabili: {total_params}")
        print(f"  Dimensiuni matrici:")
        print(f"    W1: {self.W1.shape}, b1: {self.b1.shape}")
        print(f"    W2: {self.W2.shape}, b2: {self.b2.shape}")

    def relu(self, Z):
        """
        Implementează ReLU vectorizat
        Funcționează pe matrici de orice dimensiune
        """
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        """
        Implementează sigmoid vectorizat
        Include clipping pentru stabilitate numerică
        """
        Z_safe = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_safe))

    def forward(self, X):
        """
        Forward pass prin rețea - COMPLET VECTORIZAT

        Parameters:
        -----------
        X : numpy.ndarray
            Matrice de intrare cu shape (batch_size, dim_intrare)
            Fiecare rând = un exemplu

        Returns:
        --------
        numpy.ndarray
            Probabilități de ieșire cu shape (batch_size, dim_iesire)

        Proces:
        -------
        1. Z1 = X @ W1 + b1        (transformare liniară)
        2. A1 = ReLU(Z1)          (activare non-liniară)
        3. Z2 = A1 @ W2 + b2      (transformare liniară)
        4. A2 = Sigmoid(Z2)       (activare → probabilități)
        """
        # STRATUL 1: Input → Hidden cu ReLU
        # Matrix multiplication: (batch_size, dim_intrare) @ (dim_intrare, dim_ascuns)
        # Result: (batch_size, dim_ascuns)
        Z1 = np.dot(X, self.W1) + self.b1  # Broadcasting adaugă bias la fiecare exemplu
        A1 = self.relu(Z1)  # Activare element-wise

        # Salvăm pentru debugging/vizualizare
        self.Z1 = Z1
        self.A1 = A1

        # STRATUL 2: Hidden → Output cu Sigmoid
        # Matrix multiplication: (batch_size, dim_ascuns) @ (dim_ascuns, dim_iesire)
        # Result: (batch_size, dim_iesire)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)

        # Salvăm pentru debugging
        self.Z2 = Z2
        self.A2 = A2

        return A2

    def prezice(self, X):
        """
        Returnează predicții binare (0 sau 1)
        Folosește pragul standard de 0.5 pentru clasificare binară
        """
        output = self.forward(X)
        return (output > 0.5).astype(int)

    def vizualizeaza_activari(self, X):
        """
        Funcție helper pentru a vizualiza activările prin rețea
        """
        _ = self.forward(X)  # Rulăm forward pass

        print("\nVIZUALIZARE ACTIVĂRI:")
        print("-" * 40)
        print(f"Input shape: {X.shape}")
        print(f"După stratul 1 (înainte de ReLU): {self.Z1.shape}")
        print(f"După ReLU: {self.A1.shape}")
        print(f"După stratul 2 (înainte de sigmoid): {self.Z2.shape}")
        print(f"Output final: {self.A2.shape}")

        # Statistici despre activări
        print(f"\nStatistici activări:")
        print(f"Neuroni activi în hidden layer: {np.mean(self.A1 > 0) * 100:.1f}%")
        print(f"Sparsitate hidden layer: {np.mean(self.A1 == 0) * 100:.1f}%")

        return self.A1, self.A2


# Testare Rețea Neuronală
print("\nTESTARE REȚEA NEURONALĂ:")
print("-" * 40)

# Creează o rețea: 4 intrări → 5 neuroni ascunși → 2 ieșiri
retea = ReteasNeuronala(dim_intrare=4, dim_ascuns=5, dim_iesire=2)

# Date de test: 10 exemple cu 4 features fiecare
# Simulăm un mini-batch de date
X_test = np.random.randn(10, 4)
print(f"\nDate de test generate: {X_test.shape[0]} exemple, {X_test.shape[1]} features")

# Obține predicțiile
print("\nRulare forward pass...")
predictii = retea.forward(X_test)
clase_prezise = retea.prezice(X_test)

print("\n" + "=" * 60)
print("REZULTATE:")
print("=" * 60)
print(f"Forma intrării: {X_test.shape}")
print(f"Forma ieșirii: {predictii.shape}")

print("\nPrimele 5 exemple:")
print("-" * 40)
for i in range(5):
    print(f"Exemplu {i + 1}:")
    print(f"  Input: {X_test[i]}")
    print(f"  Probabilități: {predictii[i]}")
    print(f"  Clasă prezisă: {clase_prezise[i]}")

# Verifică vectorizarea
print("\n" + "=" * 60)
print("VERIFICARE VECTORIZARE:")
print("=" * 60)
print(f"Dimensiuni parametri:")
print(f"  W1: {retea.W1.shape} = {retea.W1.shape[0]}×{retea.W1.shape[1]} = {np.prod(retea.W1.shape)} parametri")
print(f"  b1: {retea.b1.shape} = {np.prod(retea.b1.shape)} parametri")
print(f"  W2: {retea.W2.shape} = {retea.W2.shape[0]}×{retea.W2.shape[1]} = {np.prod(retea.W2.shape)} parametri")
print(f"  b2: {retea.b2.shape} = {np.prod(retea.b2.shape)} parametri")
print(
    f"  TOTAL: {np.prod(retea.W1.shape) + np.prod(retea.b1.shape) + np.prod(retea.W2.shape) + np.prod(retea.b2.shape)} parametri")

# Vizualizare activări
activari_hidden, activari_output = retea.vizualizeaza_activari(X_test)

# Test de performanță - verificăm că e într-adevăr vectorizat
print("\n" + "=" * 60)
print("TEST PERFORMANȚĂ VECTORIZARE:")
print("=" * 60)

import time

# Test cu batch mic
X_small = np.random.randn(100, 4)
start = time.time()
_ = retea.forward(X_small)
time_small = time.time() - start

# Test cu batch mare
X_large = np.random.randn(10000, 4)
start = time.time()
_ = retea.forward(X_large)
time_large = time.time() - start

print(f"Timp pentru 100 exemple: {time_small * 1000:.2f} ms")
print(f"Timp pentru 10,000 exemple: {time_large * 1000:.2f} ms")
print(f"Speedup: {(time_small * 100) / (time_large):.1f}x (ideal ar fi ~1x)")
print("(Dacă speedup-ul e aproape de 1, vectorizarea funcționează corect!)")

# Vizualizare distribuția output-urilor
plt.figure(figsize=(15, 5))

# Subplot 1: Histograma probabilităților pentru clasa 1
plt.subplot(1, 3, 1)
plt.hist(predictii[:, 0], bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Prag decizie')
plt.xlabel('Probabilitate')
plt.ylabel('Frecvență')
plt.title('Distribuția probabilităților\npentru Clasa 0')
plt.legend()

# Subplot 2: Histograma probabilităților pentru clasa 2
plt.subplot(1, 3, 2)
plt.hist(predictii[:, 1], bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Prag decizie')
plt.xlabel('Probabilitate')
plt.ylabel('Frecvență')
plt.title('Distribuția probabilităților\npentru Clasa 1')
plt.legend()

# Subplot 3: Scatter plot probabilități clasa 0 vs clasa 1
plt.subplot(1, 3, 3)
colors = ['red' if c[0] == 1 else 'blue' for c in clase_prezise]
plt.scatter(predictii[:, 0], predictii[:, 1], c=colors, alpha=0.6, s=50)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Probabilitate Clasa 0')
plt.ylabel('Probabilitate Clasa 1')
plt.title('Spațiul probabilităților\n(roșu=prezis clasa 0, albastru=prezis clasa 1)')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# =============================================================================
# SUMAR ȘI CONCLUZII
# =============================================================================

print("\n" + "=" * 80)
print("SUMAR ȘI CONCLUZII")
print("=" * 80)

print("""
Ce am învățat în această temă:
-------------------------------

1. NEURON SIMPLU (Exercițiul 1):
   ✓ Un neuron calculează o sumă ponderată și aplică o funcție de activare
   ✓ ReLU introduce non-linearitate păstrând eficiența computațională
   ✓ Weights și bias determină comportamentul neuronului

2. FUNCȚIA SIGMOID (Exercițiul 2):
   ✓ Sigmoid mapează valori în (0,1) - perfect pentru probabilități
   ✓ Are derivată convenabilă dar suferă de vanishing gradient
   ✓ NumPy vectorizează automat operațiile pe arrays

3. REȚEA MULTI-LAYER (Exercițiul 3):
   ✓ Mai multe straturi = capacitate de a învăța funcții complexe
   ✓ Vectorizarea permite procesare eficientă de batch-uri
   ✓ Broadcasting-ul NumPy elimină necesitatea loop-urilor

Concepte NumPy esențiale demonstrate:
-------------------------------------
• np.dot() - multiplicare matriceală pentru propagare forward
• np.maximum() - operații element-wise pentru funcții de activare  
• Broadcasting - adăugare eficientă de bias la batch-uri întregi
• Vectorizare - procesare simultană a mai multor exemple

Întrebări pentru explorare viitoare:
------------------------------------
? Cum se antrenează acești parametri? (Backpropagation)
? Cum alegem arhitectura optimă? (Hyperparameter tuning)
? Ce alte funcții de activare există? (Leaky ReLU, Swish, GELU)
? Cum prevenim overfitting? (Dropout, regularizare)
? Cum inițializăm parametrii optim? (Xavier, He, LSUV)

Această implementare reprezintă FUNDAMENTUL pentru:
- Clasificare (binary/multiclass)
- Regresie
- Deep Learning frameworks (PyTorch, TensorFlow)
- Înțelegerea rețelelor moderne (CNNs, Transformers)
""")

print("\n" + "=" * 80)
print("TEMĂ COMPLETATĂ CU SUCCES! 🎉")
print("=" * 80)