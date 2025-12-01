#!/usr/bin/env python3
"""
coffee_pca.py — Run PCA on tasting attributes from the coffee dataset.

Inputs:
  - CSV or Parquet normalized by coffee_kaggle.py (default: data/coffee/coffee.parquet)

Outputs (written to data/coffee/ by default):
  - coffee_pca_loadings.csv  : feature loadings for each principal component
  - coffee_pca_variance.csv  : per-component variance and cumulative ratio
  - coffee_pca_scores.csv    : first 4 PCs for each cup (with name/roaster/origin)

Example:
  python coffee_pca.py
  python coffee_pca.py --input data/coffee/kaggle/coffee_clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path("data/coffee/coffee.parquet")
DEFAULT_OUT_DIR = Path("data/coffee")
FEATURE_COLS = ["aroma", "acidity", "body", "flavor", "aftertaste", "rating"]
CONTEXT_COLS = ["name", "roaster", "origin"]


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def compute_pca(df: pd.DataFrame, features: Iterable[str], n_components: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (loadings, variance_summary, scores)
    """
    features = list(features)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}")

    # Keep rows where all feature columns are present
    clean = df[list(features)].dropna()
    if clean.empty:
        raise SystemExit("No rows left after dropping NAs for selected features.")

    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    pca = PCA(n_components=min(n_components, len(features)))
    scores_arr = pca.fit_transform(X)

    comp_labels = [f"PC{i+1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=comp_labels,
    )

    variance = pd.DataFrame(
        {
            "component": comp_labels,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    variance["cumulative_ratio"] = variance["explained_variance_ratio"].cumsum()

    scores = pd.DataFrame(scores_arr, columns=comp_labels)
    if CONTEXT_COLS[0] in df.columns:
        context = df[CONTEXT_COLS].loc[clean.index].reset_index(drop=True)
        scores = pd.concat([context, scores], axis=1)
    return loadings, variance, scores


def write_outputs(out_dir: Path, loadings: pd.DataFrame, variance: pd.DataFrame, scores: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(out_dir / "coffee_pca_loadings.csv")
    variance.to_csv(out_dir / "coffee_pca_variance.csv", index=False)
    scores.to_csv(out_dir / "coffee_pca_scores.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV or Parquet coffee dataset")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for PCA outputs")
    args = ap.parse_args()

    df = load_frame(args.input)
    loadings, variance, scores = compute_pca(df, FEATURE_COLS)
    write_outputs(args.out_dir, loadings, variance, scores)

    print(f"PCA run on {len(scores)} rows with features: {', '.join(FEATURE_COLS)}")
    print("\nExplained variance ratio (cumulative):")
    for comp, ratio, cum in variance[["component", "explained_variance_ratio", "cumulative_ratio"]].itertuples(index=False):
        print(f"  {comp}: {ratio:.3f} (cum {cum:.3f})")
    print("\nTop loadings (PC1..PC4):")
    print(loadings)
    print("\nSample scores:")
    print(scores.head())


if __name__ == "__main__":
    main()
