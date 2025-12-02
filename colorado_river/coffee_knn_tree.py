#!/usr/bin/env python3
"""
coffee_knn_tree.py — Train/test KNN and Decision Tree classifiers on coffee ratings (binned).

Target classes: rating buckets (Low/Med/High) derived from numeric ratings.
Train/test split: 80/20
Outputs:
  - data/coffee/knn_tree_metrics.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["aroma", "acidity", "body", "flavor", "aftertaste", "price"]
TARGET = "rating"


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def bin_rating(r: float) -> str:
    if r < 92:
        return "Low"
    if r < 94.5:
        return "Med"
    return "High"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/coffee/coffee.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/coffee/knn_tree_metrics.txt"))
    args = ap.parse_args()

    df = load_df(args.input)
    feats = [c for c in FEATURES if c in df.columns and df[c].notna().any()]
    if len(feats) < 2 or TARGET not in df.columns:
        raise SystemExit("Need at least two feature columns and rating to proceed.")

    data = df[feats + [TARGET]].dropna()
    if len(data) < 50:
        raise SystemExit("Not enough rows after dropping NA.")

    X = data[feats]
    y = data[TARGET].apply(bin_rating)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

    out_lines = []
    def add_metrics(name: str, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=3)
        out_lines.append(f"{name} Accuracy: {acc:.4f}")
        out_lines.append(report)

    add_metrics("KNN", y_test, y_pred_knn)
    add_metrics("DecisionTree", y_test, y_pred_tree)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines))
    print(args.out.read_text())
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
