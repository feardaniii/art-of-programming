# Temă Sklearn — Capitolele 30–33

Temă care acoperă tot ce am învățat în capitolele **sklearn**:
fundamente matematice, clasificare (Breast Cancer & Iris), regresie (apartamente București), pipeline-uri și optimizare de hiperparametri.

Continuați de unde am rămas în scripturi — codul și dataset-urile sunt deja acolo, gata de folosit.

---

## Exercițiul I — Clasificare Iris (continuare din Capitolul 33)

> Referință: [`33_real_world_classification_iris/`](../../31_advanced_classification/33_real_world_classification_iris/)
> Deja făcut: clasificatorul KNN în `part2_k_nearest_neighbours.py`

**Cerință:** Antrenați clasificatoarele **SVM** și **Random Forest** pe dataset-ul Iris.

- Folosiți același train/test split și preprocesare din scriptul KNN existent.
- Evaluați fiecare model (accuracy, classification report, confusion matrix).
- Comparați toate trei modelele — care performează cel mai bine și de ce?

---

## Exercițiul II — Extinderea Dataset-ului Apartamente București (continuare din Capitolul 32)

> Referință: [`32_advanced_regression/`](../)
> Dataset: `apartamente_bucuresti.csv` | Pipeline: `part2_pipeline.py`

**Cerință:** Adăugați **2 coloane categoriale noi** în dataset:

1. **`mobilat`** — Mobilat / Nemobilat
2. **`tip_incalzire`** — Tip încălzire (ex. centrală proprie, termoficare, etc.)

Apoi:
- Actualizați pipeline-ul de preprocesare pentru a gestiona noile feature-uri.
- Re-antrenați toate trei modelele de regresie (Linear, Ridge, Random Forest).
- Comparați noile metrici cu cele vechi — au ajutat feature-urile adăugate?
- Explorați care model funcționează cel mai bine acum.

**Bonus:** Creați o versiune a dataset-ului cu valori lipsă intenționate în noile coloane și verificați că pipeline-ul le gestionează corect.

---

## Exercițiul III — La Alegere (opțional)

Alegeți **una** din următoarele variante:

### Opțiunea A — Bate ML-ul cu Python pur

Alegeți orice algoritm/dataset din capitolele anterioare și încercați să obțineți rezultate comparabile folosind **zero ML** — doar `if/else`, loop-uri și matematică de bază.

- Fără `train_test_split` — întregul dataset devine setul vostru de test.
- Comparați acuratețea codului vostru cu acuratețea modelului ML.
- Reflectați: unde cedează abordarea bazată pe reguli?

### Opțiunea B — Selecția Feature-urilor pe Breast Cancer

> Referință: [`31_advanced_classification/`](../../31_advanced_classification/)
> Punct de plecare: `part4_grid_search_hyperparameters_and_guide.py`

Dataset-ul Breast Cancer are **30 de feature-uri**. Cerința voastră:

- Selectați doar **15 feature-uri** (la alegere — folosiți cunoștințe de domeniu sau feature importance).
- Folosiți **GridSearchCV** cu **Pipelines** pentru a găsi cea mai bună combinație de:
  - Care 15 feature-uri să păstrați
  - Ce clasificator (SVM, RF, KNN)
  - Ce hiperparametri pentru acel clasificator
- Raportați cea mai bună configurație și comparați-o cu rezultatele pe toate cele 30 de feature-uri.

---

**Baftă!**
