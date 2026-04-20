"""
EV battery health ML (new standalone analysis).

Engineering focus: estimate State of Health (SoH, %) and a maintenance/replacement
binary risk flag from charging-session measurements only.

Excluded from inputs (synthetic coupling / leakage in this dataset):
  - Degradation Rate (%)
  - Charging Duration (min)
  - Optimal Charging Duration Class
  - Efficiency (%)  # numerically identical inverse of Degradation Rate in this CSV

Target SoH is a simulated latent health score (not a column in the file) built from
stress proxies (cycles, temperature, current, SOC, voltage) plus noise, so models
see realistic but non-trivial scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from sklearn.linear_model import Ridge
except ImportError:  # pragma: no cover
    from sklearn.linear_model import RidgeCV as Ridge  # type: ignore

RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
DATA_PATH = Path(__file__).resolve().parent / "ev_battery_charging_data.csv"
OUT_DIR = Path(__file__).resolve().parent / "ev_battery_soh_outputs"


def _json_float(x: float) -> float | None:
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return float(x)


def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, encoding="utf-8")
    drop_cols = [
        "Degradation Rate (%)",
        "Charging Duration (min)",
        "Optimal Charging Duration Class",
        "Efficiency (%)",
    ]
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from CSV: {missing}")
    X = df.drop(columns=drop_cols)
    return X, drop_cols


def simulated_soh_percent(X: pd.DataFrame) -> pd.Series:
    """
    Latent SoH in [~55, 100]: higher wear → lower SoH.
    Uses only non-excluded measurements + categoricals (encoded minimally here).
    """
    soc = X["SOC (%)"].to_numpy(dtype=float)
    v = X["Voltage (V)"].to_numpy(dtype=float)
    cur = X["Current (A)"].to_numpy(dtype=float)
    tbat = X["Battery Temp (°C)"].to_numpy(dtype=float)
    tamb = X["Ambient Temp (°C)"].to_numpy(dtype=float)
    cycles = X["Charging Cycles"].to_numpy(dtype=float)

    soc_n = (soc - 50.0) / 50.0
    cycles_n = cycles / max(float(np.max(cycles)), 1.0)
    t_hot = np.clip((tbat - 24.0) / 18.0, 0.0, None)
    tamb_n = (tamb - 22.0) / 12.0
    cur_n = cur / 100.0
    volt_n = (v - 3.85) / 0.35

    mode = X["Charging Mode"].astype(str).to_numpy()
    fast = (mode == "Fast").astype(float)
    slow = (mode == "Slow").astype(float)

    btype = X["Battery Type"].astype(str).to_numpy()
    li_ion = (btype == "Li-ion").astype(float)

    wear = (
        0.20 * np.abs(soc_n)
        + 0.26 * cycles_n
        + 0.18 * (t_hot**1.35)
        + 0.10 * np.abs(tamb_n)
        + 0.12 * cur_n
        + 0.08 * np.abs(volt_n)
        + 0.06 * fast
        - 0.04 * slow
        + 0.05 * li_ion
    )
    # Scale + bounded noise: enough structure for useful regression, while the
    # 80% SoH rule keeps a realistic minority class (~22–28% in practice).
    latent = np.clip(0.30 * wear + RNG.normal(0.0, 0.075, size=len(X)), 0.0, 0.33)
    soh = 100.0 * (1.0 - latent)
    return pd.Series(soh, index=X.index, name="SoH_simulated_pct")


def make_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    num_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_frame.columns if c not in num_cols]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )


def split_train_val_test(
    X: pd.DataFrame, y_cls: np.ndarray, test_size: float = 0.2, val_size: float = 0.2
):
    """60 / 20 / 20 stratified on binary maintenance label."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y_cls,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, excluded = load_features(DATA_PATH)
    y_soh = simulated_soh_percent(X_raw)
    # Industry-style alert: SoH below 80% of nominal ⇒ schedule maintenance / capacity test
    maintenance = (y_soh < 80.0).astype(int)
    y_maint = maintenance.astype(int)

    X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls = split_train_val_test(
        X_raw, y_maint.to_numpy()
    )

    y_train_soh = y_soh.loc[X_train.index].to_numpy()
    y_val_soh = y_soh.loc[X_val.index].to_numpy()
    y_test_soh = y_soh.loc[X_test.index].to_numpy()

    # --- Regression: SoH ---
    ridge = Pipeline(
        steps=[
            ("prep", make_preprocessor(X_raw)),
            ("model", Ridge(alpha=1.8)),
        ]
    )
    tree_reg = Pipeline(
        steps=[
            ("prep", make_preprocessor(X_raw)),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_leaf=12,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    ridge.fit(X_train, y_train_soh)
    tree_reg.fit(X_train, y_train_soh)

    reg_rows = []
    for name, model in [("ridge_regression", ridge), ("decision_tree_regression", tree_reg)]:
        pred = model.predict(X_test)
        reg_rows.append(
            {
                "model": name,
                "r2": float(r2_score(y_test_soh, pred)),
                "mae": float(mean_absolute_error(y_test_soh, pred)),
            }
        )

    # --- Classification: maintenance risk ---
    classifiers: dict[str, Pipeline] = {
        "decision_tree": Pipeline(
            steps=[
                ("prep", make_preprocessor(X_raw)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gaussian_nb": Pipeline(
            steps=[
                ("prep", make_preprocessor(X_raw)),
                (
                    "model",
                    GaussianNB(var_smoothing=8e-3, priors=np.array([0.5, 0.5])),
                ),
            ]
        ),
        "knn_21": Pipeline(
            steps=[
                ("prep", make_preprocessor(X_raw)),
                ("model", KNeighborsClassifier(n_neighbors=21, weights="distance")),
            ]
        ),
    }

    cls_report: dict[str, dict] = {}
    roc_payload: dict[str, dict] = {}

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_cls)
        prob_pos = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test_cls, preds)
        try:
            auc = roc_auc_score(y_test_cls, prob_pos)
        except ValueError:
            auc = float("nan")

        fpr, tpr, thresholds = roc_curve(y_test_cls, prob_pos)
        roc_payload[name] = {
            "auc": float(auc),
            "accuracy": float(acc),
            "n_thresholds": int(len(thresholds)),
            "sample": {
                "fpr_head": [_json_float(float(v)) for v in fpr[:5]],
                "tpr_head": [_json_float(float(v)) for v in tpr[:5]],
                "thresholds_head": [_json_float(float(v)) for v in thresholds[:5]],
                "fpr_tail": [_json_float(float(v)) for v in fpr[-5:]],
                "tpr_tail": [_json_float(float(v)) for v in tpr[-5:]],
                "thresholds_tail": [_json_float(float(v)) for v in thresholds[-5:]],
            },
        }

        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        cls_report[name] = classification_report(
            y_test_cls, preds, digits=3, output_dict=True, zero_division=0
        )

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC: maintenance risk (SoH < 80%)")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    fig_roc.tight_layout()
    fig_roc.savefig(OUT_DIR / "roc_maintenance_risk.png", dpi=150)
    plt.close(fig_roc)

    # Confusion matrix for best AUC model
    best_name = max(roc_payload, key=lambda k: roc_payload[k]["auc"])
    best_clf = classifiers[best_name]
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_clf,
        X_test,
        y_test_cls,
        display_labels=["OK", "Maintenance"],
        cmap="Blues",
        ax=ax_cm,
        colorbar=False,
    )
    ax_cm.set_title(f"Confusion matrix — {best_name}")
    fig_cm.tight_layout()
    fig_cm.savefig(OUT_DIR / "confusion_maintenance_best.png", dpi=150)
    plt.close(fig_cm)

    # Residuals for regression
    fig_res, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (mname, model) in zip(
        axes, [("Ridge SoH", ridge), ("Tree SoH", tree_reg)], strict=True
    ):
        pred = model.predict(X_test)
        resid = y_test_soh - pred
        ax.scatter(pred, resid, alpha=0.35, s=12)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("Predicted SoH (%)")
        ax.set_ylabel("Residual (actual − pred)")
        ax.set_title(mname)
    fig_res.suptitle("Regression residuals on held-out test set")
    fig_res.tight_layout()
    fig_res.savefig(OUT_DIR / "soh_regression_residuals.png", dpi=150)
    plt.close(fig_res)

    # Actual vs predicted SoH (ridge)
    pred_ridge = ridge.predict(X_test)
    fig_avp, ax_avp = plt.subplots(figsize=(5.5, 5.5))
    ax_avp.scatter(y_test_soh, pred_ridge, alpha=0.35, s=14)
    lims = min(y_test_soh.min(), pred_ridge.min()), max(y_test_soh.max(), pred_ridge.max())
    ax_avp.plot(lims, lims, "r--", linewidth=1, label="ideal")
    ax_avp.set_xlabel("Simulated SoH (%) — actual")
    ax_avp.set_ylabel("Predicted SoH (%) — Ridge")
    ax_avp.set_title("SoH regression (test)")
    ax_avp.legend()
    ax_avp.set_aspect("equal", adjustable="box")
    fig_avp.tight_layout()
    fig_avp.savefig(OUT_DIR / "soh_actual_vs_predicted_ridge.png", dpi=150)
    plt.close(fig_avp)

    # Decision tree importances (interpretable drivers of the maintenance flag)
    dt_clf = classifiers["decision_tree"]
    prep = dt_clf.named_steps["prep"]
    tree_model = dt_clf.named_steps["model"]
    feat_names = prep.get_feature_names_out()
    imp = tree_model.feature_importances_
    top_k = 12
    order = np.argsort(imp)[::-1][:top_k]
    fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
    ax_imp.barh(np.arange(top_k), imp[order][::-1])
    ax_imp.set_yticks(np.arange(top_k))
    ax_imp.set_yticklabels(feat_names[order][::-1], fontsize=8)
    ax_imp.set_xlabel("Importance (Gini decrease)")
    ax_imp.set_title("Decision tree — top features for maintenance risk")
    fig_imp.tight_layout()
    fig_imp.savefig(OUT_DIR / "decision_tree_feature_importance.png", dpi=150)
    plt.close(fig_imp)

    val_auc: dict[str, float] = {}
    for name, clf in classifiers.items():
        pval = clf.predict_proba(X_val)[:, 1]
        val_auc[name] = float(roc_auc_score(y_val_cls, pval))

    summary = {
        "excluded_input_columns": excluded,
        "note": (
            "SoH is simulated from non-leaky features + noise; Efficiency dropped "
            "because it is a perfect affine transform of Degradation Rate in this CSV."
        ),
        "split": "60/20/20 train/val/test (stratified on maintenance label)",
        "n_samples": int(len(X_raw)),
        "class_balance_maintenance_pct": float(100.0 * y_maint.mean()),
        "regression_test_metrics": reg_rows,
        "classification_test": roc_payload,
        "classification_val_auc": val_auc,
        "classification_report_test_best_auc_model": cls_report.get(best_name),
        "best_classifier_by_test_auc": best_name,
    }

    (OUT_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Human-readable threshold table for best model
    best_prob = best_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_cls, best_prob)
    thr_col = np.full(len(fpr), np.nan, dtype=float)
    n_thr = min(len(thresholds), len(fpr) - 1)
    if n_thr > 0:
        thr_col[1 : 1 + n_thr] = thresholds[:n_thr]
    tbl = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_col})
    tbl.to_csv(OUT_DIR / f"tpr_fpr_thresholds_{best_name}.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
