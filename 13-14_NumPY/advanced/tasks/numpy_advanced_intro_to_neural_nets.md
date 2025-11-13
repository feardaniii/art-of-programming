# Tema 15: Introducere în Rețele Neuronale cu NumPy

**Temă comună pentru înțelegerea conceptelor de bază ale rețelelor neuronale**

---

## 📚 Ghid de Abordare - Research First, Code Second

### Filosofia acestei teme: **Don't just Code - but Evolve your Knowledge**

Această temă urmărește **două obiective paralele**:
1. **Stăpânirea NumPy** - înțelegerea operațiilor vectorizate, broadcasting, și manipularea eficientă a matricelor
2. **Fundamentele Rețelelor Neuronale** - înțelegerea conceptuală a cum "învață" o mașină

### 🔍 Metodologia de lucru recomandată

Pentru **FIECARE** exercițiu, urmați această abordare în 4 pași:

#### Pasul 1: Research (15-30 minute per exercițiu)
Înainte de a scrie orice linie de cod, cercetați:
- **Ce este conceptul?** (ex: Ce este un neuron? De ce ReLU?)
- **De ce există?** (Ce problemă rezolvă? Care e intuiția?)
- **Cum funcționează matematic?** (Formulele, dar și intuiția din spate)
- **Exemple vizuale** (Căutați grafice, animații, diagrame)

**Resurse de start pentru research:**
- 3Blue1Brown - Neural Network series (pentru intuiție vizuală)
- Papers with Code (pentru implementări practice)
- NumPy documentation (pentru operații specifice)

#### Pasul 2: Implementare (20-30 minute)
- Acum doar începeți să scrieți codul
- Comentați FIECARE linie cu ce face ȘI de ce
- Verificați dimensiunile matricelor la fiecare pas

#### Pasul 3: Reflecție și Documentare (10-15 minute)
- Scrieți un paragraf despre ce ați învățat
- Notați 2-3 întrebări noi care v-au apărut
- Documentați orice "aha!" moment

### 🎯 Exemple de întrebări pentru research

**Pentru Exercițiul 1 (Neuron cu ReLU):**
- De ce ReLU și nu o funcție liniară? (hint: non-linearitate)
- Ce se întâmplă cu "dying ReLU problem"?
- Cum arată ReLU vs Sigmoid vs Tanh vizual?
- Ce reprezintă weights și bias în lumea reală?

**Pentru Exercițiul 2 (Sigmoid):**
- De ce sigmoid mapează în (0,1)? Pentru ce e util asta?
- Care e derivata lui sigmoid și de ce e importantă?
- Ce e "vanishing gradient problem"?
- Când folosim sigmoid vs softmax?

**Pentru Exercițiul 3 (Rețea cu 2 straturi):**
- Ce înseamnă "fully connected"? 
- De ce avem nevoie de straturi multiple?
- Ce e forward propagation vs backward propagation?
- Cum aleg numărul de neuroni în stratul ascuns?

### 💡 Anti-patterns de evitat

❌ **Nu faceți:** Copy-paste din ChatGPT fără înțelegere
✅ **Faceți:** Folosiți AI pentru clarificări, apoi implementați singuri

❌ **Nu faceți:** Săriți direct la cod
✅ **Faceți:** Desenați pe hârtie ce vreți să faceți întâi

❌ **Nu faceți:** Implementați tot deodată
✅ **Faceți:** Baby steps - testați fiecare funcție izolat

### 📊 Ce înseamnă "înțelegere profundă"

Știți că ați înțeles cu adevărat când puteți:
1. Explica conceptul unui coleg în cuvinte simple
2. Desena pe tablă cum circulă datele prin neuron/rețea
3. Prezice ce se întâmplă dacă schimbați un parametru
4. Identifica când și de ce ar eșua implementarea

### 🔗 Conexiunea NumPy - Neural Networks

În timp ce lucrați, observați:
- **Dot product** (`np.dot`) = suma ponderată în neuroni
- **Broadcasting** = aplicare eficientă a bias-ului
- **Vectorizare** = procesare batch (mai multe exemple simultan)
- **Reshape** = pregătirea datelor pentru layere diferite

---

## I. Implementează un neuron simplu cu activare ReLU

Creează o clasă `NeuronSimplu` care să simuleze comportamentul unui neuron cu funcția de activare ReLU.

```python
import numpy as np

class NeuronSimplu:
    def __init__(self, numar_intrari):
        # Inițializează weights și bias aleatoriu
        self.weights = np.random.randn(numar_intrari)
        self.bias = np.random.randn()
    
    def relu(self, x):
        # Implementează funcția ReLU
        # ReLU(x) = max(0, x)
        pass
    
    def forward(self, intrari):
        # Calculează output = ReLU(weights · intrari + bias)
        pass

# Testare
neuron = NeuronSimplu(3)
intrare_test = np.array([1.0, 2.0, -0.5])
output = neuron.forward(intrare_test)
print(f"Output neuron: {output}")
```

**Cerințe:**
- Funcția `relu()` trebuie să returneze 0 pentru valori negative și valoarea însăși pentru valori pozitive
- Metoda `forward()` trebuie să calculeze suma ponderată și să aplice ReLU
- Afișează weights, bias și output-ul final

## II. Creează și aplică funcția sigmoid pe un dataset aleatoriu

Implementează funcția sigmoid și aplică-o pe 100 de valori generate aleatoriu.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Implementează funcția sigmoid: f(x) = 1 / (1 + e^(-x))
    pass

# Generează 100 de valori aleatorii între -10 și 10
date_aleatorii = np.random.uniform(-10, 10, 100)

# Aplică sigmoid pe toate valorile
rezultate = date_aleatorii # aplică sigmoid aici

# Vizualizare (opțional dar recomandat)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(date_aleatorii, bins=20, alpha=0.7, color='blue')
plt.title('Distribuția valorilor inițiale')

plt.subplot(1, 2, 2)
plt.hist(rezultate, bins=20, alpha=0.7, color='green')
plt.title('Distribuția după aplicarea sigmoid')
plt.show()

# Statistici
print(f"Media inițială: {np.mean(date_aleatorii):.3f}")
print(f"Media după sigmoid: {np.mean(rezultate):.3f}")
print(f"Min/Max inițial: {np.min(date_aleatorii):.3f}/{np.max(date_aleatorii):.3f}")
print(f"Min/Max după sigmoid: {np.min(rezultate):.3f}/{np.max(rezultate):.3f}")
```

**Cerințe:**
- Funcția sigmoid trebuie să mapeze orice valoare reală în intervalul (0, 1)
- Calculează și afișează media, minimul și maximul înainte și după aplicarea sigmoid
- Observă cum sigmoid "comprimă" valorile în intervalul (0, 1)

## III. Simulează o rețea neuronală cu 2 straturi complet vectorizată

Implementează o rețea neuronală simplă cu 2 straturi dense (fully connected) folosind doar NumPy și operații vectorizate.

```python
import numpy as np

class ReteasNeuronala:
    def __init__(self, dim_intrare, dim_ascuns, dim_iesire):
        # Inițializare parametri pentru 2 straturi
        # Stratul 1: dim_intrare -> dim_ascuns
        self.W1 = np.random.randn(dim_intrare, dim_ascuns) * 0.1
        self.b1 = np.zeros((1, dim_ascuns))
        
        # Stratul 2: dim_ascuns -> dim_iesire
        self.W2 = np.random.randn(dim_ascuns, dim_iesire) * 0.1
        self.b2 = np.zeros((1, dim_iesire))
    
    def relu(self, Z):
        # Implementează ReLU vectorizat
        pass
    
    def sigmoid(self, Z):
        # Implementează sigmoid vectorizat
        pass
    
    def forward(self, X):
        """
        Forward pass prin rețea
        X: matrice de intrare (batch_size, dim_intrare)
        """
        # Stratul 1 cu activare ReLU
        Z1 = # calculează X · W1 + b1
        A1 = # aplică ReLU
        
        # Stratul 2 cu activare sigmoid
        Z2 = # calculează A1 · W2 + b2
        A2 = # aplică sigmoid
        
        return A2
    
    def prezice(self, X):
        """Returnează predicții binare (0 sau 1)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Testare cu date simulate
np.random.seed(42)

# Creează o rețea: 4 intrări -> 5 neuroni ascunși -> 2 ieșiri
retea = ReteasNeuronala(dim_intrare=4, dim_ascuns=5, dim_iesire=2)

# Date de test: 10 exemple cu 4 features fiecare
X_test = np.random.randn(10, 4)

# Obține predicțiile
predictii = retea.forward(X_test)
clase_prezise = retea.prezice(X_test)

print("Forma intrării:", X_test.shape)
print("Forma ieșirii:", predictii.shape)
print("\nPrimele 3 probabilități de ieșire:")
print(predictii[:3])
print("\nPrimele 3 clase prezise:")
print(clase_prezise[:3])

# Verifică vectorizarea
print(f"\nDimensiuni parametri:")
print(f"W1: {retea.W1.shape}, b1: {retea.b1.shape}")
print(f"W2: {retea.W2.shape}, b2: {retea.b2.shape}")
```

**Cerințe:**
- Toate operațiile trebuie să fie **complet vectorizate** (fără bucle for explicite)
- Stratul 1 folosește activare ReLU, stratul 2 folosește sigmoid
- Rețeaua trebuie să poată procesa un batch de mai multe exemple simultan
- Afișează dimensiunile tuturor matricelor pentru a verifica corectitudinea
- Metoda `prezice()` trebuie să returneze clase binare (0 sau 1) bazate pe un prag de 0.5
[
## Criterii de evaluare

1. **Corectitudinea implementării** (40%)
   - Funcțiile matematice sunt implementate corect
   - Operațiile matriceale sunt făcute corect

2. **Vectorizare** (30%)
   - Codul folosește operații NumPy vectorizate
   - Nu există bucle for inutile

3. **Claritatea codului** (20%)
   - Cod bine comentat și organizat
   - Variabile cu nume sugestive

4. **Testing și validare** (10%)
   - Testarea funcțiilor cu date de exemplu]()
   - Afișarea rezultatelor într-un mod clar

## Note importante

- Folosiți `np.random.seed()` pentru reproducibilitate
- Atenție la dimensiunile matricelor în operațiile matriceale
- Broadcasting-ul NumPy vă poate ajuta să evitați bucle
- Pentru debugging, afișați formele (shapes) matricelor intermediare

## Resurse utile

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Funcții de activare](https://en.wikipedia.org/wiki/Activation_function)
- [Vectorizare în NumPy](https://numpy.org/doc/stable/user/quickstart.html#vectorization)

**Format livrare:** Jupyter Notebook sau script Python cu output-ul rulării