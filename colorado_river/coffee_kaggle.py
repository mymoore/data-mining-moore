#!/usr/bin/env python3
"""
coffee_kaggle.py — Load & normalize the Kaggle CoffeeReview dataset
Dataset: hanifalirsyad/coffee-scrap-coffeereview

Flow:
  - download/cache → ./data/coffee/kaggle/
  - normalize (15 columns) → ./data/coffee/coffee.parquet

Use either:
  * kagglehub  (pip install kagglehub)
  * Kaggle CLI (pip install kaggle; set up ~/.kaggle/kaggle.json)

Examples:
  python coffee_kaggle.py --method kagglehub
  python coffee_kaggle.py --method cli
"""
from __future__ import annotations
import shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List
import pandas as pd

DATA_DIR = Path("data/coffee")
KAGGLE_CACHE = DATA_DIR / "kaggle"
DATA_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)

DATASET = "hanifalirsyad/coffee-scrap-coffeereview"
OUT_PARQUET = DATA_DIR / "coffee.parquet"

# Keep EXACTLY these columns (≤15 if some are missing)
COLUMN_WHITELIST = [
    "name","roaster","rating","aroma","acidity","body","flavor","aftertaste",
    "origin","process","variety","roast","price","review_date","notes",
]

# Map dataset column variants into our canonical names
ALIASES = {
    "name":        ["coffee_name","title","coffee","coffeename","reviewtitle","name"],
    "roaster":     ["roaster","company","brand","producer"],
    "rating":      ["rating","overall","score"],
    "aroma":       ["aroma"],
    "flavor":      ["flavor","flavour","taste"],
    "body":        ["body"],
    "aftertaste":  ["aftertaste","finish"],
    "acidity":     ["acidity","sour","acid","brightness"],  # 'sour' → 'acidity'
    "sweetness":   ["sweetness"],                           # not in whitelist; fine to ignore
    "origin":      ["origin","country","region"],
    "process":     ["process","processing","method"],
    "variety":     ["variety","varietal","cultivar"],
    "roast":       ["roast","roast_level"],
    "price":       ["price","cost"],
    "review_date": ["review_date","date","published","reviewed"],
    "url":         ["url","link","source"],                 # not in whitelist
    "notes":       ["notes","description","review_text","summary"],
}

@dataclass
class LoadResult:
    raw_files: List[str]
    out_path: str

def _download_with_kagglehub(dst: Path) -> list[str]:
    """
    Use kagglehub cache. IMPORTANT: don't pass path= (that means 'download a file named ...').
    """
    try:
        import kagglehub  # pip install kagglehub
    except Exception as e:
        raise SystemExit("kagglehub not installed. Run: pip install kagglehub") from e

    cache_dir = Path(kagglehub.dataset_download(DATASET))
    files = [p for p in cache_dir.rglob("*") if p.suffix.lower() in {".csv",".json"}]
    if not files:
        files = [p for p in cache_dir.rglob("*") if p.is_file()]

    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in files:
        target = dst / p.name
        shutil.copy2(p, target)
        out.append(str(target))
    return out

def _download_with_cli(dst: Path) -> list[str]:
    """
    Requires Kaggle CLI with token at ~/.kaggle/kaggle.json (chmod 600).
    """
    cmd = ["kaggle","datasets","download","-d",DATASET,"-p",str(dst),"-o"]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        raise SystemExit("Kaggle CLI not found. Install with: pip install kaggle") from None
    # Unzip any archives
    for z in Path(dst).glob("*.zip"):
        shutil.unpack_archive(str(z), str(dst))
    return [str(p) for p in dst.rglob("*") if p.suffix.lower() in {".csv",".json"}]

def _coalesce_files(files: Iterable[str]) -> list[str]:
    files = list(files)
    csvs = [f for f in files if f.lower().endswith(".csv")]
    return csvs if csvs else [f for f in files if f.lower().endswith(".json")]

def _try_read(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_json(path, lines=False)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

def _first_existing(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for n in names:
        if n in cols:
            return n
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Build canonical frame
    cols = {}
    for canon, candidates in ALIASES.items():
        chosen = _first_existing(df, candidates)
        cols[canon] = df[chosen] if chosen else pd.Series([None]*len(df), dtype="object")
    out = pd.DataFrame(cols)

    # Numeric
    for num in ["rating","aroma","flavor","body","aftertaste","acidity","sweetness","price"]:
        if num in out.columns:
            out[num] = pd.to_numeric(out[num], errors="coerce")

    # Dates
    if "review_date" in out.columns:
        out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce", utc=True)

    # Strings
    for s in ["name","roaster","origin","process","variety","roast","url","notes"]:
        if s in out.columns:
            out[s] = out[s].astype("string").str.strip()

    # Enforce 15-column schema & order
    out = out[[c for c in COLUMN_WHITELIST if c in out.columns]]
    return out

def load(method: str = "kagglehub") -> LoadResult:
    KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)
    if method == "kagglehub":
        files = _download_with_kagglehub(KAGGLE_CACHE)
    elif method == "cli":
        files = _download_with_cli(KAGGLE_CACHE)
    else:
        raise SystemExit("--method must be 'kagglehub' or 'cli'")

    files = _coalesce_files(files)
    if not files:
        raise SystemExit("No CSV/JSON files found in the Kaggle dataset.")

    frames = []
    for f in files:
        df = _try_read(f)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise SystemExit("Failed to read any usable file from dataset.")

    raw = pd.concat(frames, ignore_index=True)
    norm = normalize(raw)
    norm.to_parquet(OUT_PARQUET, index=False)
    return LoadResult(raw_files=files, out_path=str(OUT_PARQUET))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["kagglehub","cli"], default="kagglehub")
    args = ap.parse_args()
    res = load(method=args.method)
    print("Wrote:", res.out_path)
    print("Raw files:", *res.raw_files, sep="\n  - ")

if __name__ == "__main__":
    main()
