# EV Battery SoH Analysis — Beginner-Friendly Guide

This note explains the machine learning in `ev_battery_soh_analysis.py` in **simple terms**. You do not need to know every equation to understand what each part does and what each output file means.

---

## What problem are we solving?

We have a table of **charging sessions** (one row per session). Each row has things like:

- How full the battery was (SOC), voltage, current  
- Battery temperature, ambient temperature  
- How many **charge cycles** the pack has seen  
- Categorical details (charging mode, battery type, EV model, …)

From those inputs, we ask two ML questions:

1. **Regression:** “What is the battery’s **State of Health (SoH)** as a percentage?” (higher ≈ healthier.)  
2. **Classification:** “Should we flag this pack for **maintenance / a closer check**?” — here defined as **SoH below 80%** (a common rule-of-thumb style threshold in teaching examples).

The CSV does **not** contain a real measured SoH column. The script builds a **simulated SoH** from stress-like signals (temperature, cycles, current, etc.) plus random noise, so we can practice ML **without** using columns that would “cheat” (see below).

---

## Words you will see everywhere

| Term | Plain meaning |
|------|----------------|
| **Features** | The columns the model is **allowed** to look at when making a prediction (inputs). |
| **Target / label** | The thing we want to predict (output). SoH is a number; “maintenance yes/no” is a category. |
| **Training set** | Data the model **learns from** (like practice problems with answers). |
| **Validation set** | Extra held-out data used to **sanity-check** or tune ideas (here: logged as AUC; the script keeps the story simple). |
| **Test set** | Data the model **never** saw during training. Final scores on the test set tell you how well the model might generalize. |
| **Leakage** | Using information in training that you would not really have at prediction time, or that duplicates the answer. That makes scores look “too good” and misleading. |

**Columns we deliberately do *not* use as features** (because they duplicate or shortcut the synthetic “health story” in this dataset):

- Degradation rate, charging duration, optimal duration class  
- **Efficiency (%)** — in *this* file it lines up almost perfectly with degradation, so it would make the task unrealistically easy.

---

## How the data is split (60% / 20% / 20%)

All rows are split into three groups:

- **60% train** — fit the models.  
- **20% validation** — optional check (AUC is saved in `metrics_summary.json`).  
- **20% test** — only used at the end for honest scores and plots.

Splits are **stratified** on the maintenance label: each split keeps roughly the same fraction of “maintenance” vs “OK” sessions, so one split is not accidentally all one class.

---

## What happens to the data before modeling?

Think of this as **getting the spreadsheet into a shape sklearn understands**:

1. **Numbers** — missing values filled with the median; then **scaled** so big numbers (e.g. current) do not dominate small ones (e.g. normalized voltages) just because of units.  
2. **Categories** (e.g. “Fast” / “Normal” / “Slow”) — turned into **one-hot** columns (separate 0/1 flags per category).

That bundle is called a **pipeline** with a **preprocessor** + a **model** step.

---

## The ML methods (one by one)

### 1. Ridge regression (for SoH — a number)

- **Type:** Regression (predicts a **continuous** value, e.g. 87.3%).  
- **Idea:** Like linear regression, it learns **weights** for each input direction, but **Ridge** adds a penalty so weights do not explode and the model stays a bit more stable when features are correlated.  
- **Beginner takeaway:** A smooth, global “weighted sum” style predictor — good baseline for numeric targets.

### 2. Decision tree regressor (for SoH)

- **Type:** Regression.  
- **Idea:** A flowchart of **if–else rules** (“if temperature is high *and* cycles are high → predict lower SoH”). Leaf nodes output a **single average-like** prediction for everyone who lands there.  
- **Beginner takeaway:** Very interpretable, can capture **non-linear** patterns, but can overfit if the tree is too deep.

### 3. Decision tree classifier (maintenance yes/no)

- **Type:** Classification (predicts **classes**: 0 = OK, 1 = maintenance).  
- **Same flowchart idea**, but leaves predict **class** (often majority vote in that region) and can output **probabilities** for ROC curves.  
- **Here:** `class_weight="balanced"` nudges the tree to care more about the **minority** class (maintenance) so it is not ignored.

### 4. Gaussian Naive Bayes (maintenance yes/no)

- **Type:** Classification.  
- **Idea:** For each class, assume numeric features look somewhat **bell-shaped (Gaussian)**. It estimates simple per-feature statistics, then asks: “Which class makes this session’s feature values more likely?”  
- **“Naive”** means it **pretends features are independent** — often wrong in reality, but fast and sometimes surprisingly OK.  
- **Here:** **Equal class priors** (50% / 50%) so the rare “maintenance” class still gets a fair chance in the math.

### 5. k-Nearest Neighbors, kNN (maintenance yes/no)

- **Type:** Classification.  
- **Idea:** No explicit training formula. To classify a new session, find the **k** past sessions in the training set that look **most similar** in feature space, and **vote** (here with **distance weighting** — closer neighbors count more).  
- **Beginner takeaway:** “Tell me who my neighbors are” — flexible, but sensitive to scaling and irrelevant noise.

---

## How we measure “good” (metrics in beginner language)

### Regression (SoH)

| Metric | Plain meaning |
|--------|----------------|
| **R² (R-squared)** | Roughly: how much variance the model explains vs predicting the average every time. **1.0 = perfect** on that set; **0 =** no better than always guessing the mean; **negative** = worse than the mean on that test set. |
| **MAE (mean absolute error)** | Average size of mistake in **percentage points** of SoH (e.g. MAE 6 means “off by about 6% on average”). Lower is better. |

### Classification (maintenance)

| Metric | Plain meaning |
|--------|----------------|
| **Accuracy** | Fraction of rows where predicted class equals true class. **Misleading** if one class is very rare (always predicting “OK” can look accurate). |
| **Precision / recall / F1** (in the JSON report) | For the “maintenance” class: **precision** = “when we say maintenance, how often right?” **recall** = “of all true maintenance cases, how many did we catch?” **F1** = balance of both. |
| **AUC (ROC area)** | A single number summarizing **trade-offs** between catching true maintenance (TPR) vs crying wolf on healthy packs (FPR) across all probability thresholds. **0.5 ≈ random**; **1.0 = perfect ranking**. |

### ROC curve — TPR, FPR, thresholds

- **ROC curve** plots **TPR (true positive rate)** on the y-axis vs **FPR (false positive rate)** on the x-axis as you change the **decision threshold** on predicted probability.  
- **Threshold** — e.g. “flag maintenance if predicted probability ≥ 0.35.” Lower threshold → more alarms → higher TPR but usually higher FPR.  
- **Why it matters:** You choose an operating point that fits your **business tolerance** (miss fewer bad packs vs annoy fewer customers with false alarms).

---

## Output files (what each one is for)

All are written to the folder **`ev_battery_soh_outputs/`** after you run:

```bash
python ev_battery_soh_analysis.py
```

| File | What it shows (beginner view) |
|------|-------------------------------|
| **`metrics_summary.json`** | One place for numbers: excluded columns, class balance, regression R²/MAE, each classifier’s test AUC and accuracy, validation AUC, and a snippet of ROC-related lists. Good for reports or slides. |
| **`roc_maintenance_risk.png`** | One chart with **ROC curves** for decision tree, Naive Bayes, and kNN. Higher and more “north-west” is generally better. The dashed diagonal is **random guessing**. |
| **`confusion_maintenance_best.png`** | A **2×2 table** for the best test-AUC model: how often OK vs maintenance was confused. Diagonal = correct. |
| **`soh_regression_residuals.png`** | For Ridge and tree regression: each point’s **error** (actual minus predicted) vs predicted SoH. Ideally cloud centered on zero; patterns can hint at “model is missing something.” |
| **`soh_actual_vs_predicted_ridge.png`** | Scatter of **true SoH** vs **Ridge prediction**. Points near the red diagonal line = good predictions. |
| **`decision_tree_feature_importance.png`** | Which **engineered input directions** (including one-hot category splits) the tree relied on most for maintenance prediction. Useful for storytelling (“cycles and temperature showed up a lot”). |
| **`tpr_fpr_thresholds_<model>.csv`** | Table of **FPR**, **TPR**, and **threshold** along the ROC for the best classifier — lets you pick a concrete alarm rule instead of only looking at the default 0.5 cutoff. |

---

## How to read this project in one minute

1. We **hide** leaky / duplicate columns so the model cannot memorize shortcuts.  
2. We **simulate** a realistic SoH-style target from allowed inputs + noise.  
3. We **train** simple, standard models (Ridge, trees, Naive Bayes, kNN).  
4. We **score** on held-out test data and save **plots + JSON + CSV** so you can explain trade-offs (ROC) and errors (residuals, confusion matrix).

If you want to go deeper later, the natural order is: **train/test split → preprocessing → one model (Ridge) → ROC/thresholds → then trees and kNN.**

---

*This guide matches the behavior of `ev_battery_soh_analysis.py` in this repository. Numbers in your `metrics_summary.json` will change slightly if you change the script, random seed, or data file.*
