# Battery ML pipeline — changes and additions

This document summarizes updates to `battery_ml_project.py` for the EV battery charging ML coursework: what changed, what was added, and why it matters for your report.

---

## 1. Degradation regression — leakage fix (important)

### What was wrong

When **Charging Duration (min)** and **Optimal Charging Duration Class** were included as inputs to predict **Degradation Rate (%)**, models (especially linear regression) could achieve unrealistically strong scores (for example, R² very close to 1 and MAE near zero). In this synthetic dataset, duration and the derived duration class correlate strongly with degradation, so they act as **indirect labels** rather than independent sensors or context.

### What we changed

For **`run_regression_degradation` only**, those two columns are now **excluded** from the feature list:

- `Charging Duration (min)`
- `Optimal Charging Duration Class`

The model instead uses electrical, thermal, usage, and categorical context (SOC, voltage, current, temperatures, cycles, charging mode, battery type, EV model, etc.), plus **Efficiency (%)** remains excluded as before (it is tightly tied to degradation in the dataset description).

### Why it matters for your write-up

You can honestly describe the task as **predicting degradation from operational and environmental indicators**, not from outcomes that essentially encode the same charging session length bucket. Metrics after this change are more suitable for discussing **generalization** and **limitations** (noise, nonlinearity, proxy targets).

---

## 2. Two data-split modes

### New configuration

| Variable       | Meaning |
|----------------|---------|
| `SPLIT_MODE`   | `"60_20_20"` (default) or `"70_30"` |
| `CV_FOLDS`     | Number of folds (default **5**) when using cross-validation on the training set |

### `"60_20_20"` (default)

- **60%** train, **20%** validation, **20%** test.
- Stratified splits are used where the target is discrete (classification).
- Model selection and ROC threshold tuning use the **held-out validation** set.
- Final model is refit on **train + validation** before the **test** evaluation (consistent holdout test).

### `"70_30"`

- **70%** train, **30%** test — **no separate validation split**.
- To avoid tuning on the same rows used for fitting, the script uses **stratified K-fold cross-validation on the training portion only** for:
  - **Binary tasks:** comparing models by mean **ROC-AUC** across folds; **ROC curve and probability threshold** are built from **out-of-fold** predicted probabilities (`cross_val_predict`), then the chosen model is refit on all training data and evaluated on the test set.
  - **Degradation regression:** comparing regressors by mean **MAE** across **KFold** on train.
  - **Multiclass duration class:** comparing models by mean **macro-F1** across folds on train.
  - **RUL proxy regression:** reports **mean MAE +/- std** on train via CV when there is no validation set.

### How to switch

At the top of `battery_ml_project.py`:

```python
SPLIT_MODE = "60_20_20"  # or "70_30"
```

Or, without editing the file:

```bash
python -c "import battery_ml_project as b; b.SPLIT_MODE='70_30'; b.main()"
```

`main()` also prints a short reminder of which split mode ran.

---

## 3. New and updated functions

| Piece | Role |
|--------|------|
| `split_data(X, y, stratify=...)` | Single entry point: returns `(X_train, X_val, X_test, y_train, y_val, y_test)` with `X_val` / `y_val` set to **`None`** when `SPLIT_MODE == "70_30"`. |
| `split_train_val_test(...)` | Unchanged logic; used internally for the 60/20/20 path. |
| `run_binary_task` | Uses `split_data`; branches on whether validation exists; uses **`StratifiedKFold`**, **`cross_val_score`**, and **`cross_val_predict`** for the 70/30 path. |
| `run_regression_degradation` | Dropped leaky columns; uses **`KFold`** + **`cross_val_score`** for 70/30; **rebuilds** the best pipeline from `best_name` after the loop so the final estimator is always correct (including after CV-only iterations). |
| `run_multiclass_optimal_duration` | Uses `split_data` and CV macro-F1 on train when `X_val` is `None`. |
| `run_rul_proxy_regression` | Uses `split_data`; optional CV MAE on train for 70/30; refits on **train+validation** when a validation set exists, else on full train. |

---

## 4. Preprocessing — Naive Bayes compatibility

**`OneHotEncoder`** is now created with **`sparse_output=False`** so the design matrix passed to **Gaussian Naive Bayes** is dense. That avoids sparse-matrix issues in the pipeline across sklearn versions.

---

## 5. ROC, TPR, FPR, and thresholds (binary tasks)

Behavior is unchanged in spirit, with clearer naming:

- The script still prints **`roc_curve`**-style **FPR**, **TPR**, and **threshold** samples via `roc_tpr_fpr_table`.
- A **distance-to-corner** heuristic picks a single operating threshold (Youden-style corner on the ROC).
- Test-set reporting labels refer to a **“tuning ROC”** (validation or out-of-fold train), not only “validation,” so it stays accurate for both split modes.

ROC figures are still written as:

- `roc_maintenance_needed.png`
- `roc_replace_needed.png`

---

## 6. Console output

Cross-validation uncertainty is reported with **ASCII `+/-`** instead of the Unicode plus-minus character, so Windows terminals are less likely to show garbled symbols.

---

## 7. What did *not* change (baseline behavior)

- **CSV path**, **random seed**, and **maintenance / replacement degradation thresholds** (`MAINT_DEG_THRESHOLD`, `REPLACE_DEG_THRESHOLD`) work as before.
- **Binary targets** are still derived from degradation vs those thresholds; **multiclass** still targets **Optimal Charging Duration Class** with **charging duration** removed from features to limit leakage for that task.
- **SoH proxy** after degradation regression is still **`clip(100 - predicted_degradation, 0, 100)`** — a coursework-style proxy, not a laboratory SoH from capacity fade.
- **RUL** is still a **synthetic proxy** built from cycles and degradation (`RUL_CYCLE_CAP`, `RUL_DEG_SCALE`); it should be described in reports as a **methodology illustration**, not a measured remaining useful life.

---

## 8. How to run

From the project directory (where `ev_battery_charging_data.csv` lives):

```bash
python battery_ml_project.py
```

Dependencies are typical for the script: **pandas**, **numpy**, **matplotlib**, **scikit-learn**.

---

## 9. Suggested wording for your report

1. **Leakage:** Explain why duration and optimal-duration **class** were removed from **degradation** inputs but may still appear in other tasks where the question is different (e.g., predicting duration class without using raw duration).
2. **Splits:** State whether you used **60/20/20** or **70/30** and, for **70/30**, that model comparison and ROC thresholds used **cross-validation on the training set** and the test set was used **once** for final metrics.
3. **SoH / RUL:** Clearly separate **proxy / synthetic** targets from **physical** definitions (capacity fade, cycles to 80% SoH, etc.).

If you sync this repo with another machine or branch, ensure `battery_ml_project.py` includes the same `split_data`, `SPLIT_MODE`, and degradation `drop_cols` as described above so this note stays accurate.
