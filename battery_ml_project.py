"""
EV battery health & performance — ML pipeline (synthetic dataset).

Tasks:
  1) Regression: predict Degradation Rate (%) from electrical/thermal/context features.
  2) Binary: maintenance alert (degradation > threshold) — ROC, TPR, FPR, thresholds.
  3) Binary: replacement alert (stricter degradation threshold).
  4) Multiclass: Optimal Charging Duration Class (0/1/2); charging duration excluded to avoid leakage.
  5) SoH proxy: SoH ≈ 100% − predicted degradation (coursework proxy; real SoH needs capacity tests).
  6) RUL proxy: regression on a synthetic remaining-cycle target (methodology demo, not physical RUL).

Splits (set SPLIT_MODE):
  - "60_20_20": train / validation / test with stratified class splits.
  - "70_30": train / test only; model choice and ROC thresholds use stratified 5-fold CV on the training set
    (out-of-fold scores, so metrics are not inflated by reusing the same rows for fitting and thresholding).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = "ev_battery_charging_data.csv"
RANDOM_STATE = 42
# "60_20_20" — held-out validation set; "70_30" — train/test + 5-fold CV on train for selection / thresholds
SPLIT_MODE = "70_30"
CV_FOLDS = 5

MAINT_DEG_THRESHOLD = 10.0   # % — service / inspection alert
REPLACE_DEG_THRESHOLD = 15.0  # % — stronger replacement / major service flag

# Synthetic RUL proxy: cycles remaining before a notional end-of-life (coursework demonstration).
RUL_CYCLE_CAP = 1200.0
RUL_DEG_SCALE = 12.0  # higher degradation shrinks effective remaining life in proxy

# Realism controls for synthetic-data demonstrations:
# - Slight feature noise mimics sensor uncertainty.
# - Small label flips mimic annotation/decision noise.
REALISM_MODE = True
FEATURE_NOISE_FRAC = 0.03  # 3% of each numeric feature std
BINARY_LABEL_FLIP_RATE = 0.03  # flip 3% of binary labels


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def add_feature_noise(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """Inject small Gaussian noise into numeric features for realism on synthetic data."""
    if not REALISM_MODE:
        return df
    out = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    numeric_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    for col in numeric_cols:
        std = float(out[col].std(ddof=0))
        if std > 0:
            out[col] = out[col] + rng.normal(0.0, FEATURE_NOISE_FRAC * std, size=len(out))
    return out


def flip_binary_labels(y: pd.Series, flip_rate: float, seed_offset: int = 0) -> pd.Series:
    """Randomly flip a small proportion of binary labels to reduce synthetic perfection."""
    if not REALISM_MODE or flip_rate <= 0:
        return y
    out = y.astype(int).copy()
    rng = np.random.default_rng(RANDOM_STATE + seed_offset)
    mask = rng.random(len(out)) < flip_rate
    out.loc[mask] = 1 - out.loc[mask]
    return out


def split_train_val_test(X, y, stratify=None):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=stratify
    )
    strat2 = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat2
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(X, y, stratify=None):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test); X_val/y_val are None when SPLIT_MODE is 70_30."""
    if SPLIT_MODE == "60_20_20":
        return split_train_val_test(X, y, stratify=stratify)
    if SPLIT_MODE == "70_30":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify
        )
        return X_train, None, X_test, y_train, None, y_test
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE!r} (use '60_20_20' or '70_30')")


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def print_split_sizes(name: str, X_train, X_val, X_test):
    print(f"\n=== {name} (split={SPLIT_MODE}) ===")
    if X_val is not None:
        print(f"Train: {X_train.shape}  Validation: {X_val.shape}  Test: {X_test.shape}")
    else:
        print(f"Train: {X_train.shape}  Test: {X_test.shape}  (CV on train: {CV_FOLDS} folds)")


def roc_tpr_fpr_table(fpr, tpr, thresholds, max_rows: int = 12):
    """Print a compact table of FPR, TPR, threshold (sklearn roc_curve convention)."""
    idx = np.linspace(0, len(thresholds) - 1, num=min(max_rows, len(thresholds)), dtype=int)
    print(f"{'Threshold':>14} {'FPR':>10} {'TPR':>10}")
    for i in idx:
        th = thresholds[i] if i < len(thresholds) else float("nan")
        print(f"{th:14.6g} {fpr[i]:10.4f} {tpr[i]:10.4f}")


def run_binary_task(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    categorical_cols: list[str],
    task_title: str,
):
    X = df[feature_cols]
    y = flip_binary_labels(df[target_col].astype(int), BINARY_LABEL_FLIP_RATE, seed_offset=17)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    pre = make_preprocessor(categorical_cols, numeric_cols)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=y)
    print_split_sizes(task_title, X_train, X_val, X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Gaussian NB": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    }

    results = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, est in classifiers.items():
        pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
        if X_val is not None:
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results[name] = {
                "est": est,
                "auc": roc_auc_score(y_val, proba),
                "acc": accuracy_score(y_val, pred),
                "f1": f1_score(y_val, pred, zero_division=0),
            }
        else:
            pipe_cv = Pipeline([("preprocess", clone(pre)), ("model", clone(est))])
            aucs = cross_val_score(
                pipe_cv, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            results[name] = {
                "est": est,
                "auc": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "acc": float("nan"),
                "f1": float("nan"),
            }

    print("\nModel selection summary:")
    if X_val is not None:
        print("(held-out validation; default threshold 0.5)")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True):
            print(f"  {name:22s}  AUC {m['auc']:.3f}  Acc {m['acc']:.3f}  F1 {m['f1']:.3f}")
    else:
        print(f"({CV_FOLDS}-fold CV ROC-AUC on training set)")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True):
            print(
                f"  {name:22s}  AUC {m['auc']:.3f} +/- {m['auc_std']:.3f}"
            )

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_est = classifiers[best_name]
    print(f"\nBest model: {best_name}")

    best_pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(best_est))])
    if X_val is not None:
        best_pipe.fit(X_train, y_train)
        y_tune_proba = best_pipe.predict_proba(X_val)[:, 1]
        y_tune = y_val
        tune_label = "Validation"
    else:
        y_tune_proba = cross_val_predict(
            clone(best_pipe), X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        y_tune = y_train
        tune_label = "Train (out-of-fold)"

    fpr, tpr, thresholds = roc_curve(y_tune, y_tune_proba)
    auc_tune = roc_auc_score(y_tune, y_tune_proba)
    dist = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    best_idx = int(np.argmin(dist))
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"\n{tune_label} ROC  AUC={auc_tune:.4f}")
    print(
        f"Youden-style corner threshold ~ {best_th:.6g}  "
        f"TPR={tpr[best_idx]:.4f}  FPR={fpr[best_idx]:.4f}"
    )
    print("Sample (FPR, TPR, threshold) along curve:")
    roc_tpr_fpr_table(fpr, tpr, thresholds)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_tune:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], s=50, zorder=5, label="Chosen threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (TPR / Recall+)")
    plt.title(task_title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"roc_{target_col}.png", dpi=150)
    plt.close()

    if X_val is not None:
        X_trv = pd.concat([X_train, X_val], axis=0)
        y_trv = pd.concat([y_train, y_val], axis=0)
        best_pipe.fit(X_trv, y_trv)
    else:
        best_pipe.fit(X_train, y_train)

    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_th).astype(int)
    print("\n--- Test set (threshold from tuning ROC corner) ---")
    print(f"AUC:       {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall/TPR:{recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))


def run_regression_degradation(df: pd.DataFrame):
    """Predict degradation from electrical/thermal/context features (avoid duration/class leakage)."""
    drop_cols = [
        "Degradation Rate (%)",
        "Efficiency (%)",
        "Charging Duration (min)",
        "Optimal Charging Duration Class",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    categorical_cols = ["Charging Mode", "Battery Type", "EV Model"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    X = df[feature_cols]
    y = df["Degradation Rate (%)"]

    pre = make_preprocessor(categorical_cols, numeric_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=None)
    print_split_sizes("Regression — Degradation Rate (%)", X_train, X_val, X_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Reg": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
        "KNN Reg (k=7)": KNeighborsRegressor(n_neighbors=7),
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print("\nModel selection metrics:")
    best = None
    best_name = None
    for name, est in regressors.items():
        if X_val is not None:
            pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
            pipe.fit(X_train, y_train)
            p_val = pipe.predict(X_val)
            mae = mean_absolute_error(y_val, p_val)
            rmse = mean_squared_error(y_val, p_val) ** 0.5
            r2 = r2_score(y_val, p_val)
            print(f"  {name:20s}  MAE {mae:.4f}  RMSE {rmse:.4f}  R² {r2:.4f}  (validation)")
        else:
            pipe_cv = Pipeline([("preprocess", clone(pre)), ("model", clone(est))])
            maes = -cross_val_score(
                pipe_cv,
                X_train,
                y_train,
                cv=kf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            mae = float(np.mean(maes))
            mae_std = float(np.std(maes))
            print(f"  {name:20s}  MAE {mae:.4f} +/- {mae_std:.4f}  ({CV_FOLDS}-fold CV on train)")
        if best is None or mae < best:
            best = mae
            best_name = name

    print(f"\nBest (lowest MAE): {best_name}")
    best_est = regressors[best_name]
    best_pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(best_est))])
    if X_val is not None:
        X_trv = pd.concat([X_train, X_val], axis=0)
        y_trv = pd.concat([y_train, y_val], axis=0)
        best_pipe.fit(X_trv, y_trv)
    else:
        best_pipe.fit(X_train, y_train)
    p_test = best_pipe.predict(X_test)
    print("--- Test ---")
    print(f"MAE:  {mean_absolute_error(y_test, p_test):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, p_test) ** 0.5:.4f}")
    print(f"R²:   {r2_score(y_test, p_test):.4f}")

    # SoH-style proxy: assume nominal 100% at 0% degradation, linear penalty (simplified).
    soh_est = np.clip(100.0 - p_test, 0.0, 100.0)
    print("\nSoH proxy from test predictions (100 - predicted degradation), clipped to [0,100]:")
    print(f"  mean={soh_est.mean():.2f}%  std={soh_est.std():.2f}%")


def run_multiclass_optimal_duration(df: pd.DataFrame):
    target = "Optimal Charging Duration Class"
    # Class labels are duration-based; exclude measured duration to avoid trivial leakage.
    drop_cols = [target, "Charging Duration (min)"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    categorical_cols = ["Charging Mode", "Battery Type", "EV Model"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    X = df[feature_cols]
    y = df[target].astype(int)

    pre = make_preprocessor(categorical_cols, numeric_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=y)
    print_split_sizes("Multiclass — Optimal Charging Duration Class", X_train, X_val, X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Gaussian NB": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_f1 = -1.0
    best_name = None
    print("\nModel selection (macro-F1):")
    for name, est in models.items():
        if X_val is not None:
            pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            acc = accuracy_score(y_val, pred)
            print(f"  {name:36s}  Acc {acc:.3f}  macro-F1 {f1:.3f}  (validation)")
        else:
            pipe_cv = Pipeline([("preprocess", clone(pre)), ("model", clone(est))])
            f1s = cross_val_score(
                pipe_cv, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1
            )
            f1 = float(np.mean(f1s))
            f1_std = float(np.std(f1s))
            print(f"  {name:36s}  macro-F1 {f1:.3f} +/- {f1_std:.3f}  ({CV_FOLDS}-fold CV on train)")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    print(f"\nBest (macro-F1): {best_name}")
    best_est = models[best_name]
    best_pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(best_est))])
    if X_val is not None:
        X_trv = pd.concat([X_train, X_val], axis=0)
        y_trv = pd.concat([y_train, y_val], axis=0)
        best_pipe.fit(X_trv, y_trv)
    else:
        best_pipe.fit(X_train, y_train)
    pred_test = best_pipe.predict(X_test)
    print("--- Test ---")
    print(f"Accuracy:   {accuracy_score(y_test, pred_test):.4f}")
    print(f"macro-F1:   {f1_score(y_test, pred_test, average='macro', zero_division=0):.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred_test))
    print(classification_report(y_test, pred_test, digits=3))


def run_rul_proxy_regression(df: pd.DataFrame):
    """
    Synthetic RUL proxy (cycles): decreases with more cycles and higher degradation.
    Not a physical RUL label — for methodology demo only.
    """
    cycles = df["Charging Cycles"].astype(float)
    deg = df["Degradation Rate (%)"].astype(float)
    y = np.maximum(0.0, RUL_CYCLE_CAP - cycles * (1.0 + deg / RUL_DEG_SCALE))

    feature_cols = [
        c
        for c in df.columns
        if c
        not in (
            "Degradation Rate (%)",
            "Efficiency (%)",
            "Optimal Charging Duration Class",
        )
    ]
    categorical_cols = ["Charging Mode", "Battery Type", "EV Model"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    X = df[feature_cols]

    pre = make_preprocessor(categorical_cols, numeric_cols)
    y = pd.Series(y, index=X.index)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, stratify=None)
    print_split_sizes("Regression — RUL proxy (synthetic cycles remaining)", X_train, X_val, X_test)

    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10)),
        ]
    )
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if X_val is not None:
        pipe.fit(X_train, y_train)
        p_val = pipe.predict(X_val)
        print("\nValidation  MAE:", mean_absolute_error(y_val, p_val))
    else:
        maes = -cross_val_score(
            clone(pipe), X_train, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        print(f"\nTrain CV MAE: {float(np.mean(maes)):.4f} +/- {float(np.std(maes)):.4f}")
    if X_val is not None:
        X_trv = pd.concat([X_train, X_val], axis=0)
        y_trv = pd.concat([y_train, y_val], axis=0)
        pipe.fit(X_trv, y_trv)
    else:
        pipe.fit(X_train, y_train)
    p_test = pipe.predict(X_test)
    print("Test        MAE:", mean_absolute_error(y_test, p_test))
    print("Test        R²: ", r2_score(y_test, p_test))


def main():
    df = load_data(CSV_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    if REALISM_MODE:
        print(
            f"Realism mode ON: feature_noise={FEATURE_NOISE_FRAC:.1%}, "
            f"binary_label_flip={BINARY_LABEL_FLIP_RATE:.1%}"
        )
    else:
        print("Realism mode OFF: using raw synthetic data.")

    df = add_feature_noise(
        df,
        exclude_cols=[
            "Degradation Rate (%)",
            "Efficiency (%)",
            "Optimal Charging Duration Class",
        ],
    )

    # --- Derived binary targets ---
    df = df.copy()
    df["maintenance_needed"] = (df["Degradation Rate (%)"] > MAINT_DEG_THRESHOLD).astype(int)
    df["replace_needed"] = (df["Degradation Rate (%)"] > REPLACE_DEG_THRESHOLD).astype(int)

    # Features for health binaries: do not use degradation or efficiency (efficiency tied to degradation in metadata).
    binary_feature_cols = [
        c
        for c in df.columns
        if c
        not in (
            "Degradation Rate (%)",
            "Efficiency (%)",
            "maintenance_needed",
            "replace_needed",
            "Optimal Charging Duration Class",
        )
    ]
    cat_binary = ["Charging Mode", "Battery Type", "EV Model"]

    run_regression_degradation(df)

    run_binary_task(
        df,
        "maintenance_needed",
        binary_feature_cols,
        cat_binary,
        "Binary — maintenance (degradation > {:.1f}%)".format(MAINT_DEG_THRESHOLD),
    )

    run_binary_task(
        df,
        "replace_needed",
        binary_feature_cols,
        cat_binary,
        "Binary — replacement alert (degradation > {:.1f}%)".format(REPLACE_DEG_THRESHOLD),
    )

    run_multiclass_optimal_duration(df)
    run_rul_proxy_regression(df)

    print("\nDone. ROC figures saved as roc_maintenance_needed.png and roc_replace_needed.png")
    print(f"Split mode was {SPLIT_MODE!r}. Set SPLIT_MODE = '70_30' or '60_20_20' at top of file to switch.")


if __name__ == "__main__":
    main()
