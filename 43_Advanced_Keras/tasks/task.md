# Advanced Keras - Tasks
## Sesiunea 43 / Session 43

---

### Task 1: Functional API Rebuild (Session 43 - Warmup)
**Dificultate / Difficulty: Easy | Timp / Time: 30 min**

**RO:** Ia cel mai bun model din sesiunile 41-42 (California Housing)
si reconstruieste-l cu Functional API. Adauga un skip connection
(ResNet-style). Compara performanta: Sequential vs Functional vs
Functional + Skip.

**EN:** Take your best model from sessions 41-42 (California Housing)
and rebuild it with the Functional API. Add a skip connection
(ResNet-style). Compare performance: Sequential vs Functional vs
Functional + Skip.

**Cerinte / Requirements:**
- Model A: Sequential API (original din 41-42)
- Model B: Functional API (identic ca arhitectura)
- Model C: Functional API + skip connection
- Aceleasi date, acelasi preprocessing, aceleasi epoci
- Tabel comparativ:

```
| Model              | API        | Skip? | Test MSE | R²    |
|--------------------|-----------|-------|----------|-------|
| A: Sequential      | Sequential | Nu    | ??       | ??    |
| B: Functional      | Functional | Nu    | ??       | ??    |
| C: Func + Skip     | Functional | Da    | ??       | ??    |
```

- Training curves pentru toate 3 pe acelasi grafic
- Vizualizeaza arhitectura cu `model.summary()` si `keras.utils.plot_model()`

**Evaluare / Evaluation:**
- [ ] Cele 3 modele construite si antrenate corect
- [ ] Skip connection implementat corect (add layer)
- [ ] Comparatie pe aceleasi date
- [ ] Training curves vizualizate
- [ ] model.summary() afisat pentru fiecare
- [ ] Concluzie: skip connection ajuta? (3+ propozitii)

---

### Task 2: Multi-Input Housing Model (Session 43 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Construieste un model multi-input pe California Housing cu
3 branch-uri: (1) features de locatie (latitude, longitude),
(2) features ale casei (rooms, bedrooms, age), (3) features economice
(income, population, occupancy). Combina-le si prezice pretul.

**EN:** Build a multi-input model on California Housing with 3 branches:
(1) location features (latitude, longitude), (2) house features
(rooms, bedrooms, age), (3) economic features (income, population,
occupancy). Combine them and predict price.

**Cerinte / Requirements:**
- Imparte cele 8 features in 3 grupuri logice
- Construieste 3 sub-retele separate (branch-uri)
- Combina cu `concatenate` inainte de output layers
- Compara cu un model single-input (toate features-urile intr-un singur input)

```
| Model          | Inputs | Test MSE | R²    | Parameters |
|----------------|--------|----------|-------|------------|
| Single-Input   | 1      | ??       | ??    | ??         |
| Multi-Input 3B | 3      | ??       | ??    | ??         |
```

- Vizualizeaza arhitectura cu `keras.utils.plot_model()`
- Predicted vs Actual scatter plot pentru ambele modele
- Experimenteaza: ce se intampla daca un branch e mai mare?

**Evaluare / Evaluation:**
- [ ] Features-urile impartite logic in 3 grupuri
- [ ] Model multi-input construit cu Functional API
- [ ] Concatenarea functioneaza corect
- [ ] Comparatie cu single-input model
- [ ] Plot arhitectura generat
- [ ] Scatter plots generate
- [ ] Analiza: multi-input ajuta? De ce / de ce nu? (5+ propozitii)

---

### Task 3: Grand Tournament Capstone (Session 43 - Capstone)
**Dificultate / Difficulty: Hard | Timp / Time: 90 min**

**RO:** Organizeaza un turneu complet: 8+ configuratii de modele,
fiecare antrenata cu 3 seed-uri diferite. Raporteaza mean ± std.
Include un model liniar ca baseline. Genereaza un raport complet
in format CSV.

**EN:** Organize a complete tournament: 8+ model configurations,
each trained with 3 different seeds. Report mean ± std.
Include a linear model as baseline. Generate a complete report
in CSV format.

**Cerinte / Requirements:**
- **Dataset:** California Housing (regression)
- **Minimum 8 configuratii:**
  1. Linear baseline (Dense(1), no activation)
  2. Simple Sequential (1 hidden layer)
  3. Medium Sequential (2 hidden layers)
  4. Large Sequential (3+ hidden layers)
  5. Sequential + Dropout
  6. Sequential + L2 + BatchNorm
  7. Functional API cu skip connection
  8. Multi-input (cel putin 2 branch-uri)
  - Bonus: adauga configuratii proprii!

- **Protocol experimental:**
  - Fiecare configuratie: 3 run-uri (seeds: 42, 123, 7)
  - Aceleasi date, acelasi preprocessing, aceleasi epoci
  - Callbacks: EarlyStopping(patience=15)
  - Colecteaza: MSE, MAE, R², timp de antrenare, numar parametri

- **Raportare:**
  - Tabel cu mean ± std pentru fiecare metrica
  - Bar chart cu error bars (R² per model) salvat ca PNG
  - Box plot al R² distributions salvat ca PNG
  - Scatter: Parameters vs R² (mai multi parametri = mai bine?) salvat ca PNG
  - Scatter: Training Time vs R² salvat ca PNG
  - CSV cu toate rezultatele individuale

```
| Config         | R² (mean±std)  | MSE (mean±std) | Params | Time  |
|----------------|---------------|----------------|--------|-------|
| Linear         | 0.52±0.00     | 0.65±0.00      | 9      | 2.1s  |
| Simple         | 0.71±0.02     | 0.39±0.02      | 577    | 4.3s  |
| Medium         | 0.78±0.01     | 0.29±0.01      | 2,657  | 6.1s  |
| ...            | ...           | ...            | ...    | ...   |
```

- **Analiza scrisa** (minimum 15 propozitii):
  - Care model a castigat si de ce?
  - Exista diminishing returns? (mai mare != mai bun?)
  - Care model are cel mai bun raport performanta/complexitate?
  - Skip connections au ajutat? Multi-input a ajutat?
  - Ce ai alege pentru productie si de ce?

**Evaluare / Evaluation:**
- [ ] Minimum 8 configuratii implementate
- [ ] Fiecare cu 3 seed-uri (24+ run-uri total)
- [ ] Baseline liniar inclus
- [ ] Mean ± std raportate corect
- [ ] Bar chart cu error bars generat
- [ ] Box plot generat
- [ ] Parameters vs Performance scatter generat
- [ ] CSV cu toate rezultatele salvat
- [ ] Analiza scrisa completa (15+ propozitii)
- [ ] Cod modular: functie `build_model(config)` parametrizata
- [ ] Reproducibil: setting seeds functioneaza

---

### Ghid de notare / Grading Guide

| Criteriu | Punctaj |
|----------|---------|
| Task 1 (Functional API Rebuild) | 20 pct |
| Task 2 (Multi-Input Housing) | 30 pct |
| Task 3 (Grand Tournament Capstone) | 50 pct |

**Sesiunea 43:** Toate tasks-urile

**Termen de predare:** la finalul sesiunii sau la inceputul urmatoarei.

**Nota:** Task 3 este un proiect substantial. Studentii pot folosi cod
din scripturile 1-6 ca punct de plecare, dar TREBUIE sa adapteze
si sa extinda (nu copy-paste).
