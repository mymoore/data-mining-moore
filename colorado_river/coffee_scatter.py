#!/usr/bin/env python3
"""
coffee_scatter.py — scatter plots of flavor metrics vs. rating.

Inputs:
  - coffee dataset CSV/Parquet (default: data/coffee/coffee.parquet)
Outputs (written beside the data by default):
  - scatter_rating_vs_<metric>.png for each metric in FEATURES
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FEATURES = ["aroma", "acidity", "body", "flavor", "aftertaste"]


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for feat in FEATURES:
        if feat not in df.columns or "rating" not in df.columns:
            print(f"Skipping {feat}: column missing")
            continue
        subset = df[[feat, "rating"]].dropna()
        if subset.empty:
            print(f"Skipping {feat}: no data after dropping NA")
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(subset[feat], subset["rating"], alpha=0.5, edgecolor="k", linewidths=0.3)
        # Add simple linear regression line
        x = subset[feat].to_numpy()
        y = subset["rating"].to_numpy()
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.array([5, 10])
            plt.plot(xs, m * xs + b, color="red", linewidth=1.5, label="Linear fit")
            lines.append((feat, m, b))
        plt.xlabel(feat.capitalize())
        plt.ylabel("Rating")
        plt.xlim(5, 10)
        plt.ylim(75, 100)
        plt.title(f"{feat.capitalize()} vs Rating")
        plt.grid(True, linestyle="--", alpha=0.3)
        if len(x) > 1:
            plt.legend()
        out_path = out_dir / f"scatter_rating_vs_{feat}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote {out_path}")

    # Combined plot of regression lines only
    if lines:
        plt.figure(figsize=(7, 5))
        xs = np.linspace(5, 10, 100)
        for feat, m, b in lines:
            plt.plot(xs, m * xs + b, linewidth=2, label=feat.capitalize())
        plt.xlim(5, 10)
        plt.ylim(75, 100)
        plt.xlabel("Sensory score")
        plt.ylabel("Rating")
        plt.title("Rating vs Sensory (Regression Lines)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        out_path = out_dir / "scatter_rating_vs_all_lines.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/coffee/coffee.parquet"), help="Coffee dataset CSV/Parquet")
    ap.add_argument("--out-dir", type=Path, default=Path("data/coffee"), help="Directory for plots")
    args = ap.parse_args()

    df = load_df(args.input)
    plot_scatter(df, args.out_dir)


if __name__ == "__main__":
    main()
