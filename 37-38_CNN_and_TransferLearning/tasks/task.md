# CNN & Transfer Learning - Tasks
## Sesiunile 37-38 / Sessions 37-38

---

### Task 1: Custom Filter Explorer (Session 37 - Warmup)
**Dificultate / Difficulty: Easy | Timp / Time: 30 min**

**RO:** Creaza 3 filtre custom 3x3 (nu Sobel) si aplica-le pe o imagine
din CIFAR-10. Vizualizeaza rezultatele. Explica CE detecteaza fiecare filtru.

**EN:** Create 3 custom 3x3 filters (not Sobel) and apply them to a
CIFAR-10 image. Visualize the results. Explain WHAT each filter detects.

**Cerinte / Requirements:**
- Un filtru care detecteaza margini diagonale (/)
- Un filtru care aplica blur (medie)
- Un filtru de design propriu (experimenteaza!)
- Vizualizare: 4 panouri (original + 3 filtrate)
- Explicatie scrisa (2-3 propozitii per filtru)

**Output asteptat / Expected output:** PNG cu 4 imagini, explicatie printata

**Evaluare / Evaluation:**
- [ ] Cele 3 filtre sunt definite corect ca matrice 3x3
- [ ] Filtrele sunt aplicate cu `convolve2d` sau manual
- [ ] Vizualizarea arata clar diferentele
- [ ] Explicatia e corecta (ce pattern detecteaza fiecare filtru)

---

### Task 2: Fashion-MNIST CNN Challenge (Session 37 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Antreneaza un CNN pe Fashion-MNIST si obtine cel putin 90% accuracy
pe test set. Experimenteaza cu: numarul de filtre, dropout rate,
data augmentation, learning rate.

**EN:** Train a CNN on Fashion-MNIST and achieve at least 90% accuracy
on the test set. Experiment with: number of filters, dropout rate,
data augmentation, learning rate.

**Cerinte / Requirements:**
- Minimum 90% test accuracy
- Grafic cu curbele de antrenament (accuracy + loss, training + validation)
- Confusion matrix vizualizata
- Tabel comparativ cu cel putin 3 configuratii diferite:

```
| Config | Filters    | Dropout | Augmentation | Test Acc |
|--------|-----------|---------|--------------|----------|
| A      | 32-64     | 0.25    | Da           | ??%      |
| B      | 64-128    | 0.5     | Nu           | ??%      |
| C      | 32-64-128 | 0.3     | Da           | ??%      |
```

- Identifica clasa cea mai grea si explica DE CE

**Evaluare / Evaluation:**
- [ ] Modelul obtine 90%+ test accuracy
- [ ] Curbele de antrenament nu arata overfitting sever
- [ ] Confusion matrix vizualizata si analizata
- [ ] Cel putin 3 experimente comparate in tabel
- [ ] Analiza scrisa a rezultatelor (5+ propozitii)

---




### Task 3: Transfer Learning Comparison (Session 38 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Compara performanta a doua modele pre-antrenate diferite
(MobileNetV2 si ResNet50) pe un subset de CIFAR-10 (3 clase la alegere,
300 imagini per clasa). Raporteaza: accuracy, timp de antrenare,
numar de parametri.

**EN:** Compare the performance of two different pre-trained models
(MobileNetV2 and ResNet50) on a CIFAR-10 subset (3 classes of your choice,
300 images per class). Report: accuracy, training time, parameter count.

**Cerinte / Requirements:**
- Selecteaza 3 clase CIFAR-10 (care sunt vizual distincte)
- Antreneaza ambele modele pentru acelasi numar de epoci
- Inregistreaza si compara: accuracy, timp, marime model

```
| Model       | Parameters | Train Time | Test Acc |
|-------------|-----------|------------|----------|
| MobileNetV2 | ??        | ??s        | ??%      |
| ResNet50    | ??        | ??s        | ??%      |
```

- Vizualizare: bar chart comparativ
- Recomandare scrisa: ce model ai deploya si de ce?

**Evaluare / Evaluation:**
- [ ] Ambele modele antrenate si evaluate corect
- [ ] Comparatie corecta (aceleasi date, aceleasi epoci)
- [ ] Tabel cu cel putin 3 metrici
- [ ] Grafic comparativ
- [ ] Recomandare scrisa cu justificare

---




### Task 4: Fine-Tuning Experiment (Session 38 - Advanced)
**Dificultate / Difficulty: Hard | Timp / Time: 90 min**

**RO:** Incepe cu un model MobileNetV2 frozen pe Cats vs Dogs (CIFAR-10).
Implementeaza 3 strategii de fine-tuning:
(A) doar head-ul custom (frozen base),
(B) ultimele 10 straturi unfrozen,
(C) ultimele 30 de straturi unfrozen.
Compara rezultatele. Adauga learning rate scheduling.

**EN:** Start with a frozen MobileNetV2 on Cats vs Dogs (CIFAR-10).
Implement 3 fine-tuning strategies:
(A) custom head only (frozen base),
(B) last 10 layers unfrozen,
(C) last 30 layers unfrozen.
Compare results. Add learning rate scheduling.

**Cerinte / Requirements:**
- 3 run-uri separate de antrenare, bine documentate
- Foloseste `ReduceLROnPlateau` callback
- Toate 3 curbele de accuracy pe acelasi grafic
- Numar de parametri trainabili pentru fiecare strategie

```
| Strategy | Trainable Params | Final Val Acc | Overfit? |
|----------|-----------------|---------------|----------|
| A: Head  | ??              | ??%           | Da/Nu    |
| B: Last10| ??              | ??%           | Da/Nu    |
| C: Last30| ??              | ??%           | Da/Nu    |
```

- Analiza: diminishing returns vs risc de overfitting

**Evaluare / Evaluation:**
- [ ] 3 strategii de fine-tuning implementate corect
- [ ] Learning rate scheduling folosit
- [ ] Toate 3 curbele pe un singur grafic
- [ ] Tabel comparativ cu parametri
- [ ] Analiza scrisa a trade-off-urilor (10+ propozitii)

---

### Task 5: Mini-Proiect -- Build Your Own Image Classifier (Session 38 - Capstone)
**Dificultate / Difficulty: Hard | Timp / Time: 120 min**

**RO:** Alege 5 clase din CIFAR-10 si construieste un clasificator complet
folosind transfer learning. Pipeline complet:
load -> explore -> preprocess -> augment -> train -> evaluate -> visualize -> save.
Salveaza modelul si demonstreaza predictii pe imagini noi.

**EN:** Choose 5 classes from CIFAR-10 and build a complete classifier
using transfer learning. Full pipeline:
load -> explore -> preprocess -> augment -> train -> evaluate -> visualize -> save.
Save the model and demonstrate predictions on new images.

**Cerinte / Requirements:**
- Pipeline complet intr-un singur fisier Python
- Data augmentation aplicata si vizualizata
- Transfer learning cu fine-tuning
- Confusion matrix
- Feature map visualization (cel putin un layer)
- Model salvat pe disc (`model.save()`)
- Functie de predictie care primeste o imagine si returneaza clasa + confidence
- Sectiune README (ca comentarii) care explica alegerile de design

**Evaluare / Evaluation:**
- [ ] Pipeline-ul este complet si ruleaza end-to-end
- [ ] Data augmentation aplicata si vizualizata
- [ ] Modelul obtine 85%+ accuracy
- [ ] Confusion matrix analizata
- [ ] Feature maps vizualizate
- [ ] Modelul salvat si loadable
- [ ] Functia de predictie functioneaza
- [ ] Codul e curat si bine comentat

---




### Ghid de notare / Grading Guide

| Criteriu | Punctaj |
|----------|---------|
| Task 1 (Filter Explorer) | 10 pct |
| Task 2 (Fashion-MNIST CNN) | 25 pct |
| Task 3 (Transfer Learning Comparison) | 25 pct |
| Task 4 (Fine-Tuning Experiment) | 20 pct |
| Task 5 (Mini-Proiect Capstone) | 20 pct |

