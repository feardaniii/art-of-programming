# Keras Mastery - Tasks
## Sesiunile 41-42 / Sessions 41-42

---

### Task 1: Wine Expert (Session 41 - Warmup)
**Dificultate / Difficulty: Easy | Timp / Time: 30 min**

**RO:** Antreneaza un clasificator pe Wine dataset folosind doar 2 features
(alege cele mai importante). Vizualizeaza granita de decizie in 2D.
Compara rezultatul cu modelul full (13 features).

**EN:** Train a classifier on the Wine dataset using only 2 features
(choose the most important ones). Visualize the decision boundary in 2D.
Compare the result with the full model (13 features).

**Cerinte / Requirements:**
- Analizeaza corelatia features cu target-ul si alege 2 features
- Antreneaza un model Keras pe cele 2 features
- Vizualizeaza granita de decizie cu `meshgrid` (scatter + regiuni colorate)
- Compara accuracy: 2 features vs 13 features
- Explica: de ce unele features sunt mai utile?

```
| Model         | Features | Test Accuracy |
|---------------|----------|---------------|
| 2-Feature     | ??       | ??%           |
| Full (13)     | All      | ??%           |
```

**Output asteptat / Expected output:** PNG cu decision boundary, tabel comparativ

**Evaluare / Evaluation:**
- [ ] Alegerea features-urilor e justificata (nu random)
- [ ] Modelul 2-feature antreneaza si evalueaza corect
- [ ] Decision boundary vizualizata in 2D
- [ ] Comparatie corecta cu modelul full
- [ ] Explicatie scrisa (3+ propozitii)

---

### Task 2: Diabetes Regression Challenge (Session 41 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Antreneaza un model de regresie pe Diabetes dataset si obtine
MAE < 50. Compara cel putin 3 arhitecturi diferite. Vizualizeaza
predicted vs actual si distributia erorilor.

**EN:** Train a regression model on the Diabetes dataset and achieve
MAE < 50. Compare at least 3 different architectures. Visualize
predicted vs actual and error distribution.

**Cerinte / Requirements:**
- Foloseste StandardScaler pe features
- Antreneaza cel putin 3 arhitecturi (e.g., mic/mediu/mare)
- Fiecare model: vizualizeaza curbele de antrenament (loss curves)
- Scatter plot: predicted vs actual (cu linia diagonala perfecta)
- Histograma erorilor (residuals) pentru cel mai bun model
- Tabel comparativ:

```
| Architecture    | Layers     | Test MSE | Test MAE | R²    |
|-----------------|-----------|----------|----------|-------|
| Simple          | 32        | ??       | ??       | ??    |
| Medium          | 64→32     | ??       | ??       | ??    |
| Large           | 128→64→32 | ??       | ??       | ??    |
```

**Evaluare / Evaluation:**
- [ ] StandardScaler aplicat corect
- [ ] Cel putin 3 arhitecturi comparate
- [ ] MAE < 50 atins pe cel putin un model
- [ ] Scatter plot predicted vs actual generat
- [ ] Histograma residuals generata
- [ ] Tabel comparativ complet
- [ ] Analiza scrisa: care arhitectura e mai buna si de ce (5+ propozitii)

---

### Task 3: Optimizer & Learning Rate Lab (Session 42 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Ruleaza un experiment sistematic: 4 optimizers × 3 learning rates
pe California Housing. Creeaza un heatmap cu rezultatele (R² sau MSE).
Identifica cea mai buna combinatie.

**EN:** Run a systematic experiment: 4 optimizers × 3 learning rates
on California Housing. Create a heatmap with the results (R² or MSE).
Identify the best combination.

**Cerinte / Requirements:**
- Optimizers: SGD, Adam, RMSprop, Adagrad
- Learning rates: 0.1, 0.01, 0.001
- Aceeasi arhitectura pentru toate (e.g., Dense(64) → Dense(32) → Dense(1))
- Aceleasi date (fix aceeasi split, acelasi scaling)
- Antreneaza fiecare combinatie (12 total), colecteaza test R²
- Creeaza un heatmap (4×3 grid) cu culori — salvat ca PNG

```
           LR=0.1    LR=0.01   LR=0.001
SGD        ??        ??        ??
Adam       ??        ??        ??
RMSprop    ??        ??        ??
Adagrad    ??        ??        ??
```

- Grafic suplimentar: loss curves pentru top-3 combinatii pe acelasi plot
- Concluzie scrisa

**Evaluare / Evaluation:**
- [ ] Toate 12 combinatii antrenate corect
- [ ] Heatmap generat si lizibil
- [ ] Variabilele controlate (aceeasi arhitectura, aceleasi date)
- [ ] Loss curves pentru top-3
- [ ] Concluzie corecta si argumentata (5+ propozitii)
- [ ] Cod organizat (loop-uri, nu copy-paste)

---

### Task 4: Regularization Recipe (Session 42 - Advanced)
**Dificultate / Difficulty: Hard | Timp / Time: 90 min**

**RO:** Creeaza deliberat un model care face overfit pe California Housing
(retea mare, fara regularizare). Apoi aplica sistematic combinatii de
regularizare si compara rezultatele.

**EN:** Deliberately create a model that overfits on California Housing
(large network, no regularization). Then systematically apply
regularization combinations and compare results.

**Cerinte / Requirements:**
- Model overfit: Dense(256) → Dense(256) → Dense(128) → Dense(64) → Dense(1), fara regularizare
- Dovedeste ca face overfit: train loss << val loss (grafic)
- Aplica 5 strategii de regularizare:

```
| Strategy      | Description                          | Test MSE | Gap*  |
|---------------|--------------------------------------|----------|-------|
| Baseline      | No regularization                    | ??       | ??    |
| A: Dropout    | Dropout(0.3) dupa fiecare layer      | ??       | ??    |
| B: L2         | l2(0.01) pe fiecare layer            | ??       | ??    |
| C: BatchNorm  | BatchNorm dupa fiecare layer         | ??       | ??    |
| D: Dropout+L2 | Combinatie                           | ??       | ??    |
| E: All three  | Dropout + L2 + BatchNorm             | ??       | ??    |
```

*Gap = diferenta intre train loss si val loss (overfit measure)

- Grafic: 6 training curves (all on one plot, sau grid 2×3)
- Histograma distributiei weights: baseline vs L2 (side-by-side)
- Concluzie: care combinatie e cea mai eficace?

**Evaluare / Evaluation:**
- [ ] Modelul baseline demonstreaza clar overfitting
- [ ] Cel putin 5 strategii aplicate si comparate
- [ ] Training curves vizualizate
- [ ] Weight histograms pentru baseline vs L2
- [ ] Tabel comparativ complet cu metrici
- [ ] Analiza scrisa a trade-off-urilor (10+ propozitii)
- [ ] Cod modular (functie care construieste model parametrizat)

---

### Task 5: Complete Tabular Pipeline Capstone (Session 42 - Capstone)
**Dificultate / Difficulty: Hard | Timp / Time: 120 min**

**RO:** Construieste un pipeline complet end-to-end pe California Housing:
de la explorare la model final optimizat. Foloseste TOATE tehnicile
din sesiunile 41-42. Documenteaza FIECARE decizie.

**EN:** Build a complete end-to-end pipeline on California Housing:
from exploration to optimized final model. Use ALL techniques from
sessions 41-42. Document EVERY decision.

**Cerinte / Requirements:**
- **Faza 1: Explorare** (20 min)
  - Distributia fiecarui feature (histograme)
  - Correlation heatmap
  - Identifica outliers
  - Print rezumat statistic

- **Faza 2: Pregatire date** (15 min)
  - StandardScaler
  - Train/val/test split (60/20/20)
  - Justifica alegerile

- **Faza 3: Baseline** (15 min)
  - Model simplu (1-2 layers)
  - Raporteaza MSE, MAE, R²

- **Faza 4: Optimizare** (40 min)
  - Experimenteaza cu cel putin 3 arhitecturi
  - Aplica regularizare (Dropout + L2)
  - Foloseste callbacks: EarlyStopping + ReduceLROnPlateau
  - Alege cel mai bun optimizer (Adam vs altele)
  - Documeteaza fiecare experiment

- **Faza 5: Evaluare finala** (20 min)
  - Predicted vs actual scatter plot
  - Distributia erorilor
  - Training curves (loss + metrics)
  - Confusion analysis: unde greseste cel mai mult?
  - Print final report cu toate metricile

- **Faza 6: Salvare** (10 min)
  - Salveaza modelul: `model.save('best_housing_model.keras')`
  - Incarca modelul si verifica ca predictiile sunt identice
  - Salveaza experiment log ca CSV

**Output asteptat:**
- Minimum 8 PNG-uri (explorare, training, evaluare)
- Un CSV cu log-ul experimentelor
- Model salvat pe disc

**Evaluare / Evaluation:**
- [ ] Pipeline complet: load → explore → preprocess → train → evaluate → save
- [ ] Explorarea datelor e completa (histograme, correlatie, outliers)
- [ ] StandardScaler aplicat corect
- [ ] Baseline definit si raportat
- [ ] Cel putin 3 arhitecturi comparate
- [ ] Regularizare aplicata (Dropout si/sau L2)
- [ ] Callbacks folosite (EarlyStopping + ReduceLR)
- [ ] Predicted vs Actual plot generat
- [ ] Experiment log salvat ca CSV
- [ ] Modelul salvat si loadable
- [ ] Fiecare decizie e documentata in comentarii
- [ ] R² > 0.75 pe test set

---

### Ghid de notare / Grading Guide

| Criteriu | Punctaj |
|----------|---------|
| Task 1 (Wine Expert) | 10 pct |
| Task 2 (Diabetes Regression) | 20 pct |
| Task 3 (Optimizer & LR Lab) | 20 pct |
| Task 4 (Regularization Recipe) | 25 pct |
| Task 5 (Complete Pipeline Capstone) | 25 pct |

**Sesiunea 41:** Tasks 1-2 (30 pct)
**Sesiunea 42:** Tasks 3-5 (70 pct)

**Termen de predare:** la finalul sesiunii respective sau la inceputul urmatoarei.
