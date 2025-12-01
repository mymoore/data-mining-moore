#!/usr/bin/env python3
"""
coffee_multi_reg.py — Multivariate linear regression for coffee ratings.

Features: aroma, acidity, body, flavor, aftertaste, price (if present)
Target: rating

Outputs:
  - data/coffee/multi_reg_metrics.txt
  - data/coffee/multi_reg_pred_vs_actual.png
  - data/coffee/multi_reg_residuals.png
  - data/coffee/multi_reg_coefficients.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["aroma", "acidity", "body", "flavor", "aftertaste", "price"]
TARGET = "rating"


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def plot_pred_vs_actual(y_true, y_pred, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidths=0.3)
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    plt.plot(lims, lims, "r--", label="Ideal")
    plt.xlabel("Actual rating")
    plt.ylabel("Predicted rating")
    plt.title("Predicted vs Actual")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residuals(y_true, y_pred, out_path: Path) -> None:
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.6, edgecolor="k", linewidths=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted rating")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coefficients(coefs: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    coefs.sort_values().plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title("Linear Regression Coefficients")
    plt.xlabel("Coefficient (rating units per feature unit)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/coffee/coffee.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/coffee"))
    args = ap.parse_args()

    df = load_df(args.input)
    available_feats = [c for c in FEATURES if c in df.columns]
    # Drop columns that are all NA
    available_feats = [c for c in available_feats if df[c].notna().any()]
    cols = available_feats + [TARGET]
    data = df[cols].dropna()
    if data.empty:
        raise SystemExit("No rows after dropping NA for selected features/target.")
    if len(available_feats) < 2:
        raise SystemExit("Need at least 2 predictor columns with data.")

    X = data[[c for c in FEATURES if c in data.columns]]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = out_dir / "multi_reg_metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"Samples (train/test): {len(X_train)}/{len(X_test)}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("Coefficients:\n")
        for name, coef in zip(X.columns, model.coef_):
            f.write(f"  {name}: {coef:.4f}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")

    plot_pred_vs_actual(y_test, y_pred, out_dir / "multi_reg_pred_vs_actual.png")
    plot_residuals(y_test, y_pred, out_dir / "multi_reg_residuals.png")

    coefs = pd.Series(model.coef_, index=X.columns)
    plot_coefficients(coefs, out_dir / "multi_reg_coefficients.png")

    print(metrics_path.read_text())
    print("Wrote plots to", out_dir)


if __name__ == "__main__":
    main()
