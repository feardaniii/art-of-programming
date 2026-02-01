# Sklearn Homework — Chapters 30–33

Homework covering everything we learned across the **sklearn** chapters:
math foundations, classification (Breast Cancer & Iris), regression (Bucharest apartments), pipelines, and hyperparameter tuning.

Pick up from where we left off in the scripts — the code and datasets are already there for you to build on.

---

## Exercise I — Iris Classification (continue from Chapter 33)

> Reference: [`33_real_world_classification_iris/`](../../31_advanced_classification/33_real_world_classification_iris/)
> Already done: KNN classifier in `part2_k_nearest_neighbours.py`

**Task:** Train **SVM** and **Random Forest** classifiers on the Iris dataset.

- Use the same train/test split and preprocessing from the existing KNN script.
- Evaluate each model (accuracy, classification report, confusion matrix).
- Compare all three models — which one performs best and why?

---

## Exercise II — Expand the Bucharest Apartments Dataset (continue from Chapter 32)

> Reference: [`32_advanced_regression/`](../)
> Dataset: `apartamente_bucuresti.csv` | Pipeline: `part2_pipeline.py`

**Task:** Add **2 new categorical columns** to the dataset:

1. **`mobilat`** — Mobilat / Nemobilat (furnished or not)
2. **`tip_incalzire`** — Tip incalzire (e.g. centrala proprie, termoficare, etc.)

Then:
- Update the preprocessing pipeline to handle the new features.
- Re-train all three regression models (Linear, Ridge, Random Forest).
- Compare the new metrics with the old ones — did the extra features help?
- Explore which model works best now.

**Bonus:** Create a version of the dataset with some intentional missing values in the new columns and verify that your pipeline handles them correctly.

---

## Exercise III — Pick One (optional, la alegere)

Choose **one** of the following:

### Option A — Beat ML with Pure Python

Pick any algorithm/dataset from the previous chapters and try to achieve comparable results using **zero ML** — only `if/else`, loops, and basic math.

- No `train_test_split` — the entire dataset is your test set.
- Compare your hand-coded accuracy against the ML model's accuracy.
- Reflect: where does the rule-based approach break down?

### Option B — Feature Selection on Breast Cancer

> Reference: [`31_advanced_classification/`](../../31_advanced_classification/)
> Starting point: `part4_grid_search_hyperparameters_and_guide.py`

The Breast Cancer dataset has **30 features**. Your task:

- Select only **15 features** (your choice — use domain knowledge or feature importance).
- Use **GridSearchCV** with **Pipelines** to find the best combination of:
  - Which 15 features to keep
  - Which classifier (SVM, RF, KNN)
  - Which hyperparameters for that classifier
- Report the best configuration and compare it to the full 30-feature results.

---

**Bafta!**
