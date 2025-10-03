#!/usr/bin/env python3
"""
coffee_eda.py — tiny helpers to explore the normalized coffee dataset.
Usage:
  python coffee_eda.py --input data/coffee/coffee.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Not found: {path}")
    return pd.read_parquet(p) if path.endswith(".parquet") else pd.read_csv(p)

def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    out = num.agg(["count","mean","std","min","median","max"]).T
    out = out.sort_index()
    return out

def top_terms(df: pd.DataFrame, col: str = "notes", n: int = 20) -> pd.Series:
    if col not in df or df[col].isna().all():
        return pd.Series(dtype=int)
    tokens = (
        df[col].astype(str).str.lower()
           .str.replace(r"[^a-z0-9\s]", " ", regex=True)
           .str.split()
           .explode()
    )
    stop = set("the a an of and to for in with on at from by is are it this that these those very".split())
    tokens = tokens[~tokens.isin(stop)]
    return tokens.value_counts().head(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV or Parquet produced by coffee_kaggle.py or coffee_scraper.py")
    args = ap.parse_args()

    df = load_df(args.input)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))

    prof = basic_profile(df)
    print("\n=== Numeric profile ===")
    print(prof)

    if "notes" in df.columns:
        print("\n=== Top terms in notes ===")
        print(top_terms(df, "notes", 25))

if __name__ == "__main__":
    main()
