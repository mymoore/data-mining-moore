#!/usr/bin/env python3
"""
Streamlit Coffee EDA App (Kaggle → clean Parquet → interactive charts)

What this app does
------------------
- Ensures a local, clean dataset exists at `data/coffee/coffee.parquet`.
  - If not found, it tries to download & normalize the Kaggle dataset
    `hanifalirsyad/coffee-scrap-coffeereview` via coffee_kaggle.py.
- Loads the table and gives you filters for roaster/origin/process/variety,
  date range (if present), and rating thresholds.
- Shows quick EDA: rating distribution, top roasters/origins by average rating,
  sensory scatter (aroma/flavor/body/aftertaste/acidity/sweetness), and
  top tasting-note terms.

How to run
----------
  pip install streamlit pandas numpy altair pyarrow
  streamlit run app_coffee.py

If Parquet isn't there yet, the app will attempt:
  from coffee_kaggle import load; load("kagglehub")  # or "cli"
You can also pre-run:
  python coffee_kaggle.py --method kagglehub
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


PARQUET = Path("data/coffee/coffee.parquet")
SENSORY_COLS = ["aroma", "flavor", "body", "aftertaste", "acidity", "sweetness"]


# ------------------------- Helpers -------------------------

def ensure_dataset() -> Path | None:
    """
    Make sure data/coffee/coffee.parquet exists.
    Try to fetch via coffee_kaggle.load if missing.
    """
    if PARQUET.exists():
        return PARQUET

    try:
        from coffee_kaggle import load
        with st.spinner("Downloading Kaggle dataset via kagglehub..."):
            res = load(method="kagglehub")
            path = Path(res.out_path)
            if path.exists():
                return path
    except Exception as e:
        st.warning(f"KaggleHub path failed: {e}")

    # Fallback to CLI
    try:
        from coffee_kaggle import load
        with st.spinner("Downloading Kaggle dataset via Kaggle CLI..."):
            res = load(method="cli")
            path = Path(res.out_path)
            if path.exists():
                return path
    except Exception as e:
        st.error(
            "Could not download the Kaggle dataset automatically. "
            "Install kagglehub **or** Kaggle CLI and try again.\n\n"
            "Kaggle CLI quickstart:\n"
            "  pip install kaggle\n"
            "  Place kaggle.json in ~/.kaggle/ (from kaggle.com/settings → Create New Token)\n"
            "  chmod 600 ~/.kaggle/kaggle.json\n"
            "Then rerun this app."
        )
        st.exception(e)

    return None


@st.cache_data(show_spinner=False)
def load_df(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # basic cleanup
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)
        # drop tz for Altair/Streamlit serialization
        df["review_date"] = df["review_date"].dt.tz_convert("America/Denver").dt.tz_localize(None)
    # standardize text columns
    for col in ["name", "roaster", "origin", "process", "variety", "roast", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    # coercions
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # computed sensory mean if not present
    present = [c for c in SENSORY_COLS if c in df.columns]
    if present and "sensory_mean" not in df.columns:
        df["sensory_mean"] = df[present].mean(axis=1, skipna=True)
    return df


def top_terms(series: pd.Series, n: int = 25) -> pd.Series:
    if series is None or series.dropna().empty:
        return pd.Series(dtype=int)
    tokens = (
        series.fillna("")
              .astype(str)
              .str.lower()
              .str.replace(r"[^a-z0-9\s]", " ", regex=True)
              .str.split()
              .explode()
    )
    stop = set("the a an of and to for in with on at from by is are it this that these those very coffee".split())
    tokens = tokens[~tokens.isin(stop)]
    vc = tokens.value_counts()
    return vc.head(n)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    return num.agg(["count", "mean", "std", "min", "median", "max"]).T.sort_index()


# ------------------------- UI -------------------------

st.set_page_config(page_title="Coffee Review EDA", layout="wide")
st.title("☕ Coffee Review Explorer")

st.caption(
    "Source: Kaggle dataset scraped from CoffeeReview. "
    "This app loads a normalized Parquet and provides quick filters + charts."
)

parquet_path = ensure_dataset()
if parquet_path is None or not Path(parquet_path).exists():
    st.stop()

df = load_df(str(parquet_path))

with st.sidebar:
    st.header("Filters")

    # Rating filter
    min_rating = float(np.nanmin(df["rating"])) if "rating" in df else 0.0
    max_rating = float(np.nanmax(df["rating"])) if "rating" in df else 100.0
    r_min, r_max = st.slider(
        "Rating range",
        min_value=float(np.floor(min_rating if np.isfinite(min_rating) else 0.0)),
        max_value=float(np.ceil(max_rating if np.isfinite(max_rating) else 100.0)),
        value=(
            float(np.floor(min_rating if np.isfinite(min_rating) else 0.0)),
            float(np.ceil(max_rating if np.isfinite(max_rating) else 100.0)),
        ),
        step=0.5,
    )

    # Multi-select helpers
    def multiselect_for(col: str, label: str):
        if col in df.columns:
            opts = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip()])
            return st.multiselect(label, opts, default=[])
        return []

    sel_roaster = multiselect_for("roaster", "Roaster(s)")
    sel_origin  = multiselect_for("origin",  "Origin(s)")
    sel_proc    = multiselect_for("process", "Process(es)")
    sel_var     = multiselect_for("variety", "Variety/Varietal(s)")

    date_rng = None
    if "review_date" in df.columns and df["review_date"].notna().any():
        dmin = pd.to_datetime(df["review_date"]).min()
        dmax = pd.to_datetime(df["review_date"]).max()
        date_rng = st.date_input(
            "Review date range",
            value=(dmin.date(), dmax.date()),
            min_value=dmin.date(),
            max_value=dmax.date(),
        )

# Apply filters
mask = pd.Series(True, index=df.index)
if "rating" in df.columns:
    mask &= df["rating"].between(r_min, r_max)

def _in_sel(col: str, selected: list[str]) -> pd.Series:
    if not selected or col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].isin(selected)

mask &= _in_sel("roaster", sel_roaster)
mask &= _in_sel("origin", sel_origin)
mask &= _in_sel("process", sel_proc)
mask &= _in_sel("variety", sel_var)

if date_rng and "review_date" in df.columns:
    d0, d1 = date_rng
    d0 = pd.to_datetime(d0)
    d1 = pd.to_datetime(d1) + pd.Timedelta(days=1)  # inclusive end
    mask &= df["review_date"].between(d0, d1)

fdf = df[mask].copy()

# ------------------------- Main Panels -------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Ratings", "Sensory", "Notes"])

with tab1:
    st.subheader("Dataset overview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"Rows after filters: **{len(fdf)}**  •  Columns: **{fdf.shape[1]}**")
        st.dataframe(fdf.head(200))
    with c2:
        prof = numeric_profile(fdf)
        if not prof.empty:
            st.write("**Numeric profile**")
            st.dataframe(prof)

with tab2:
    st.subheader("Ratings")
    if "rating" not in fdf.columns or fdf["rating"].dropna().empty:
        st.info("No rating column found.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            hist = (
                alt.Chart(fdf.dropna(subset=["rating"]))
                .mark_bar()
                .encode(x=alt.X("rating:Q", bin=alt.Bin(maxbins=30)), y="count()")
                .properties(height=260)
            )
            st.altair_chart(hist, use_container_width=True)
        with c2:
            # top roasters by mean rating
            if "roaster" in fdf.columns:
                agg = (
                    fdf.dropna(subset=["rating", "roaster"])
                    .groupby("roaster", as_index=False)["rating"]
                    .mean()
                    .sort_values("rating", ascending=False)
                    .head(20)
                )
                bar = (
                    alt.Chart(agg)
                    .mark_bar()
                    .encode(y=alt.Y("roaster:N", sort="-x"), x="rating:Q")
                    .properties(height=500)
                )
                st.altair_chart(bar, use_container_width=True)

        if "origin" in fdf.columns:
            st.markdown("**Average rating by origin (top 20)**")
            agg = (
                fdf.dropna(subset=["rating", "origin"])
                .groupby("origin", as_index=False)["rating"]
                .mean()
                .sort_values("rating", ascending=False)
                .head(20)
            )
            bar = alt.Chart(agg).mark_bar().encode(y=alt.Y("origin:N", sort="-x"), x="rating:Q").properties(height=500)
            st.altair_chart(bar, use_container_width=True)

with tab3:
    st.subheader("Sensory relationships")
    present = [c for c in SENSORY_COLS if c in fdf.columns]
    if not present:
        st.info("No sensory columns found.")
    else:
        c1, c2 = st.columns(2)
        target = "rating" if "rating" in fdf.columns else "sensory_mean"
        # scatter aroma vs flavor colored by rating
        num = fdf.dropna(subset=["aroma", "flavor"])
        if not num.empty:
            sc = (
                alt.Chart(num)
                .mark_circle(opacity=0.6)
                .encode(
                    x="aroma:Q",
                    y="flavor:Q",
                    size=alt.Size(target + ":Q", legend=None),
                    tooltip=["name", "roaster", "origin", "aroma", "flavor", target],
                )
                .properties(height=300)
            )
            st.altair_chart(sc, use_container_width=True)

        # radar-style substitute: multiple small bars per attribute avg
        melted = (
            fdf[present]
            .mean()
            .reset_index()
            .rename(columns={"index": "attribute", 0: "mean"})
        )
        bar2 = (
            alt.Chart(melted)
            .mark_bar()
            .encode(x=alt.X("mean:Q"), y=alt.Y("attribute:N", sort="-x"))
            .properties(height=220)
        )
        st.altair_chart(bar2, use_container_width=True)

with tab4:
    st.subheader("Tasting notes")
    if "notes" not in fdf.columns or fdf["notes"].dropna().empty:
        st.info("No textual notes found.")
    else:
        n = st.slider("Top terms to show", 10, 50, 25, 5)
        freq = top_terms(fdf["notes"], n=n)
        if not freq.empty:
            chart = (
                alt.Chart(freq.rename_axis("term").reset_index(name="count"))
                .mark_bar()
                .encode(y=alt.Y("term:N", sort="-x"), x="count:Q")
                .properties(height=500)
            )
            st.altair_chart(chart, use_container_width=True)
        st.write("Sample notes")
        st.dataframe(fdf[["name", "roaster", "origin", "notes"]].head(50))

st.caption("Built for the Kaggle CoffeeReview dataset → normalized to Parquet for fast, repeatable analysis.")
