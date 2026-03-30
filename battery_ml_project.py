"""
EV battery health & performance — ML pipeline (synthetic dataset).

Tasks:
  1) Regression: predict Degradation Rate (%) (proxy for health stress).
  2) Binary: maintenance alert (degradation > threshold) — ROC, TPR, FPR, thresholds.
  3) Binary: replacement alert (stricter degradation threshold).
  4) Multiclass: Optimal Charging Duration Class (0/1/2); Charging Duration excluded to avoid trivial leakage.
  5) SoH-style readout from predicted degradation (engineering proxy, not measured capacity).
  6) RUL proxy: regression on synthetic remaining-cycle target derived from cycles + degradation (documented).

Split: 60% train / 20% validation / 20% test (stratified for classification).
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
from sklearn.model_selection import train_test_split
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

MAINT_DEG_THRESHOLD = 10.0   # % — service / inspection alert
REPLACE_DEG_THRESHOLD = 15.0  # % — stronger replacement / major service flag

# Synthetic RUL proxy: cycles remaining before a notional end-of-life (coursework demonstration).
RUL_CYCLE_CAP = 1200.0
RUL_DEG_SCALE = 12.0  # higher degradation shrinks effective remaining life in proxy


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_val_test(X, y, stratify=None):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=stratify
    )
    strat2 = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat2
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def print_split_sizes(name: str, X_train, X_val, X_test):
    print(f"\n=== {name} ===")
    print(f"Train: {X_train.shape}  Validation: {X_val.shape}  Test: {X_test.shape}")


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
    y = df[target_col].astype(int)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    pre = make_preprocessor(categorical_cols, numeric_cols)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, stratify=y)
    print_split_sizes(task_title, X_train, X_val, X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Gaussian NB": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    }

    results = {}
    for name, est in classifiers.items():
        pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "pipe": pipe,
            "auc": roc_auc_score(y_val, proba),
            "acc": accuracy_score(y_val, pred),
            "f1": f1_score(y_val, pred, zero_division=0),
        }

    print("\nValidation summary (default threshold 0.5):")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True):
        print(f"  {name:22s}  AUC {m['auc']:.3f}  Acc {m['acc']:.3f}  F1 {m['f1']:.3f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_pipe = results[best_name]["pipe"]
    print(f"\nBest model (by val AUC): {best_name}")

    y_val_proba = best_pipe.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    auc_val = roc_auc_score(y_val, y_val_proba)
    dist = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    best_idx = int(np.argmin(dist))
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"\nValidation ROC  AUC={auc_val:.4f}")
    print(
        f"Youden-style corner threshold ~ {best_th:.6g}  "
        f"TPR={tpr[best_idx]:.4f}  FPR={fpr[best_idx]:.4f}"
    )
    print("Sample (FPR, TPR, threshold) along curve:")
    roc_tpr_fpr_table(fpr, tpr, thresholds)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], s=50, zorder=5, label="Chosen threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (TPR / Recall+)")
    plt.title(task_title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"roc_{target_col}.png", dpi=150)
    plt.close()

    X_trv = pd.concat([X_train, X_val], axis=0)
    y_trv = pd.concat([y_train, y_val], axis=0)
    best_pipe.fit(X_trv, y_trv)

    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_th).astype(int)
    print("\n--- Test set (threshold from validation corner) ---")
    print(f"AUC:       {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall/TPR:{recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))


def run_regression_degradation(df: pd.DataFrame):
    """Predict degradation from charge/thermal/context features (exclude direct leakage)."""
    drop_cols = ["Degradation Rate (%)", "Efficiency (%)"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    categorical_cols = ["Charging Mode", "Battery Type", "EV Model"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    X = df[feature_cols]
    y = df["Degradation Rate (%)"]

    pre = make_preprocessor(categorical_cols, numeric_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, stratify=None)
    print_split_sizes("Regression — Degradation Rate (%)", X_train, X_val, X_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Reg": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
        "KNN Reg (k=7)": KNeighborsRegressor(n_neighbors=7),
    }

    print("\nValidation metrics:")
    best = None
    best_name = None
    best_pipe = None
    for name, est in regressors.items():
        pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, p_val)
        rmse = mean_squared_error(y_val, p_val) ** 0.5
        r2 = r2_score(y_val, p_val)
        print(f"  {name:20s}  MAE {mae:.4f}  RMSE {rmse:.4f}  R² {r2:.4f}")
        if best is None or mae < best:
            best = mae
            best_name = name
            best_pipe = pipe

    print(f"\nBest on validation MAE: {best_name}")
    X_trv = pd.concat([X_train, X_val], axis=0)
    y_trv = pd.concat([y_train, y_val], axis=0)
    best_pipe.fit(X_trv, y_trv)
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
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, stratify=y)
    print_split_sizes("Multiclass — Optimal Charging Duration Class", X_train, X_val, X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "Gaussian NB": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    }

    best_f1 = -1.0
    best_name = None
    best_pipe = None
    print("\nValidation macro-F1:")
    for name, est in models.items():
        pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        print(f"  {name:36s}  Acc {acc:.3f}  macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipe = pipe

    print(f"\nBest (macro-F1): {best_name}")
    X_trv = pd.concat([X_train, X_val], axis=0)
    y_trv = pd.concat([y_train, y_val], axis=0)
    best_pipe.fit(X_trv, y_trv)
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
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, stratify=None)
    print_split_sizes("Regression — RUL proxy (synthetic cycles remaining)", X_train, X_val, X_test)

    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10)),
        ]
    )
    pipe.fit(X_train, y_train)
    p_val = pipe.predict(X_val)
    p_test = pipe.predict(X_test)
    print("\nValidation  MAE:", mean_absolute_error(y_val, p_val))
    print("Test        MAE:", mean_absolute_error(y_test, p_test))
    print("Test        R²: ", r2_score(y_test, p_test))


def main():
    df = load_data(CSV_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

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


if __name__ == "__main__":
    main()
