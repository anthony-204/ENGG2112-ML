"""
Simple ML demo: linear regression to predict Efficiency (%).

Why R2 = 1 is "too perfect" on an all-columns model:
  In this synthetic CSV, Efficiency (%) and Degradation Rate (%) have Pearson
  correlation -1.0 — they are exact linear transforms of each other. Predicting
  efficiency while feeding in degradation (and other columns built from the same
  generator) gives R2 ~ 1 by construction, not because the model discovered
  hidden physics.

This script:
  1) Prints that correlation and a one-feature "degradation only" demo (R2 ~ 1).
  2) WEAK model: SOC + two temperatures only (modest R2).
  3) STRONG model: richer honest inputs — all columns except the target and
     columns we omit as trivially collinear / leakage-prone for this target:
     degradation, charging duration, optimal duration class.

Outputs: metrics to stdout and simple_efficiency_regression.png (strong model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = "ev_battery_charging_data.csv"
RANDOM_STATE = 42
TEST_FRACTION = 0.20
OUT_PNG = "simple_efficiency_regression.png"

TARGET = "Efficiency (%)"
CATEGORICAL = ("Charging Mode", "Battery Type", "EV Model")
# Omit from strong model: exact linear twin + duration-derived fields tied to the same synthetic recipe.
OMIT_FOR_HONEST_EFFICIENCY = (
    "Degradation Rate (%)",
    "Charging Duration (min)",
    "Optimal Charging Duration Class",
)


def _ambient_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "Ambient" in c and "Temp" in c:
            return c
    raise KeyError("Ambient temperature column not found")


def _battery_temp_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "Battery" in c and "Temp" in c:
            return c
    raise KeyError("Battery temperature column not found")


def build_weak_X(df: pd.DataFrame) -> pd.DataFrame:
    amb = _ambient_col(df)
    bat = _battery_temp_col(df)
    return df[["SOC (%)", amb, bat]].astype(float)


def build_strong_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Features for an 'honest' strong model: everything except target and OMIT_FOR_HONEST_EFFICIENCY."""
    drop = {TARGET, *OMIT_FOR_HONEST_EFFICIENCY}
    feature_cols = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return df[feature_cols], num_cols, cat_cols


def make_model_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return Pipeline([("prep", pre), ("lin", LinearRegression())])


def report_metrics(split: str, y_true, y_hat) -> None:
    mae = mean_absolute_error(y_true, y_hat)
    rmse = mean_squared_error(y_true, y_hat) ** 0.5
    r2 = r2_score(y_true, y_hat)
    print(f"  {split:6s}  MAE {mae:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}")


def print_why_r2_can_be_one(df: pd.DataFrame) -> None:
    """Show that efficiency vs degradation is perfectly linear in this file."""
    if "Degradation Rate (%)" not in df.columns or TARGET not in df.columns:
        return
    r = df[TARGET].corr(df["Degradation Rate (%)"])
    print("=" * 60)
    print("Why an 'all-features' efficiency model can show R2 = 1")
    print(f"  Pearson corr( Efficiency (%), Degradation Rate (%) ) = {r:.4f}")
    print("  So degradation alone is an exact linear code for efficiency here.")
    X = df[["Degradation Rate (%)"]].astype(float)
    y = df[TARGET].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    print("  Trivial demo - LinearRegression, target ~ degradation only:")
    report_metrics("Test", yte, pred)
    print("  -> R2 ~ 1 is expected and not a useful claim of 'super accuracy'.")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y = df[TARGET].astype(float)

    print_why_r2_can_be_one(df)

    # --- Weak 3-feature baseline ---
    X_weak = build_weak_X(df)
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        X_weak, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    weak_model = Pipeline(
        [("scale", StandardScaler()), ("lin", LinearRegression())]
    )
    weak_model.fit(Xw_train, yw_train)

    print("=" * 60)
    print("WEAK model: linear regression, 3 features (SOC + two temperatures)")
    print(f"Features: {list(X_weak.columns)}")
    report_metrics("Train", yw_train, weak_model.predict(Xw_train))
    report_metrics("Test", yw_test, weak_model.predict(Xw_test))

    # --- Strong honest model (no degradation / duration leakage) ---
    X_strong, num_cols, cat_cols = build_strong_X(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_strong, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )

    strong_model = make_model_pipeline(num_cols, cat_cols)
    strong_model.fit(X_train, y_train)

    pred_train = strong_model.predict(X_train)
    pred_test = strong_model.predict(X_test)

    print("=" * 60)
    print("STRONG (honest) model: linear regression + scaled numerics + one-hot categories")
    print("  Omitted inputs (avoid trivial R2 and synthetic leakage for this target):")
    for name in OMIT_FOR_HONEST_EFFICIENCY:
        print(f"    - {name}")
    print(f"Numeric ({len(num_cols)}): {num_cols}")
    print(f"Categorical (one-hot): {cat_cols}")
    print(f"Rows: train={len(y_train)}  test={len(y_test)}  (test_fraction={TEST_FRACTION})")
    print("Metrics:")
    report_metrics("Train", y_train, pred_train)
    report_metrics("Test", y_test, pred_test)

    baseline = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    r2_base = r2_score(y_test, baseline)
    print(f"  Baseline (predict train-mean efficiency): test R2 = {r2_base:.4f}")

    plt.figure(figsize=(5.5, 5.5))
    lo = min(y_test.min(), pred_test.min()) - 0.15
    hi = max(y_test.max(), pred_test.max()) + 0.15
    plt.scatter(y_test, pred_test, alpha=0.45, edgecolors="none", s=28, label="Test points")
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect prediction")
    plt.xlabel("Actual efficiency (%)")
    plt.ylabel("Predicted efficiency (%)")
    plt.title("Strong model (no degradation / duration-class inputs)")
    plt.axis("square")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"\nSaved figure: {OUT_PNG}  (strong model: actual vs predicted)")


if __name__ == "__main__":
    main()
