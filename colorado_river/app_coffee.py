
#!/usr/bin/env python3
# app_coffee.py — Streamlit app for CoffeeReview Kaggle dataset
# Run: streamlit run app_coffee.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union
import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Config ----------
st.set_page_config(page_title="Coffee Analysis", page_icon="☕", layout="wide")
PARQUET = Path("data/coffee/coffee.parquet")

# We lock the UI to these 15 columns (only those present will show)
WANTED = [
    "name","roaster","rating","aroma","acidity","body","flavor","aftertaste",
    "origin","roast","review_date","dec1"
]

# ---------- Utilities ----------
def ensure_coffee_parquet() -> Path:
    #Make sure data/coffee/coffee.parquet exists. Attempt to fetch/normalize via coffee_kaggle.py.
    if PARQUET.exists():
        return PARQUET
    try:
        from coffee_kaggle import load
    except Exception as e:
        st.error(
            "Missing parquet and could not import `coffee_kaggle`.\n\n"
            "Create it with:\n\n"
            "    python coffee_kaggle.py --method kagglehub\n\n"
            "or set up Kaggle CLI and run `--method cli`.\n\n"
            f"Details: {e}"
        )
        st.stop()
    try:
        res = load("kagglehub")
        st.success(f"Downloaded & normalized dataset → {res.out_path}")
    except Exception as e:
        st.warning(f"kagglehub failed ({e}). Trying Kaggle CLI...")
        try:
            res = load("cli")
            st.success(f"Downloaded & normalized dataset via CLI → {res.out_path}")
        except Exception as e2:
            st.error(f"Failed to obtain dataset. Please run coffee_kaggle.py manually.\n\n{e2}")
            st.stop()
    return PARQUET

@st.cache_data(show_spinner=True)
def load_df() -> pd.DataFrame:
    #Load, coerce types, and restrict to the 15-column view.\"\"\"
    path = ensure_coffee_parquet()
    df = pd.read_parquet(path)

    # Type fixes
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce", utc=True)

    # Map 'sour' → 'acidity' if needed (defensive if parquet was built differently)
    if "acidity" not in df.columns and "sour" in df.columns:
        df["acidity"] = pd.to_numeric(df["sour"], errors="coerce")

    # Keep only the intended columns (and order)
    keep = [c for c in WANTED if c in df.columns]
    df = df[keep]
    return df

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    out = df.copy()

    # Rating range
    if "rating" in out.columns and out["rating"].notna().any():
        rmin = float(np.nanmin(out["rating"]))
        rmax = float(np.nanmax(out["rating"]))
        rsel = st.sidebar.slider("Rating", min_value=int(np.floor(rmin)), max_value=int(np.ceil(rmax)),
                                 value=(int(np.floor(rmin)), int(np.ceil(rmax))))
        out = out[(out["rating"] >= rsel[0]) & (out["rating"] <= rsel[1])]

    # Date range (TZ-safe): compare by DATE (not Timestamp) to avoid tz-naive vs tz-aware issues
    if "review_date" in out.columns and out["review_date"].notna().any():
        dates_utc = pd.to_datetime(out["review_date"], errors="coerce", utc=True)
        dmin = dates_utc.min().date()
        dmax = dates_utc.max().date()

        dsel: Union[_dt.date, Tuple[_dt.date, _dt.date]] = st.sidebar.date_input(
            "Review date", value=(dmin, dmax), min_value=dmin, max_value=dmax
        )

        if isinstance(dsel, tuple) and len(dsel) == 2:
            start_date, end_date = dsel
        else:
            # If a single date is returned, use it as both start and end
            start_date = end_date = dsel if isinstance(dsel, _dt.date) else dmin

        rd = dates_utc.dt.date
        out = out[(rd >= start_date) & (rd <= end_date)]

    def ms(label: str) -> List[str]:
        if label not in out.columns:
            return []
        opts = sorted([x for x in out[label].dropna().astype(str).unique() if x.strip()])
        return st.sidebar.multiselect(label.capitalize(), opts)

    for col in ["roaster", "origin", "process", "variety", "roast"]:
        choices = ms(col)
        if choices:
            out = out[out[col].astype(str).isin(choices)]

    return out

def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["rating","aroma","acidity","body","flavor","aftertaste","price"] if c in df.columns]
    if not numeric_cols:
        return pd.DataFrame()
    prof = df[numeric_cols].agg(["count","mean","std","min","median","max"]).T
    prof = prof.rename_axis("metric").reset_index()
    return prof

def top_terms(df: pd.DataFrame, col: str = "dec1", n: int = 25) -> pd.DataFrame:
    if col not in df.columns or df[col].isna().all():
        return pd.DataFrame(columns=["term","count"])
    tokens = (
        df[col].astype(str).str.lower()
          .str.replace(r"[^a-z0-9\s]", " ", regex=True)
          .str.split()
          .explode()
    )
    stop = set("the a an of and to for in with on at from by is are it this that these those very".split())
    tokens = tokens[~tokens.isin(stop)]
    vc = tokens.value_counts().head(n)
    return vc.rename_axis("term").reset_index(name="count")

def chart_hist(series: pd.Series, title: str):
    base = pd.DataFrame({"x": series.dropna()})
    chart = alt.Chart(base).mark_bar().encode(
        x=alt.X("x:Q", bin=alt.Bin(maxbins=30), title=title),
        y=alt.Y("count()", title="Count")
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

def chart_bar(df: pd.DataFrame, x: str, y: str, title: str, sort_desc: bool = True, limit: int = 15):
    use = df[[x, y]].dropna()
    if sort_desc:
        order = use.groupby(x)[y].mean().sort_values(ascending=False).head(limit).index.tolist()
    else:
        order = use.groupby(x)[y].mean().sort_values(ascending=True).head(limit).index.tolist()
    agg = use.groupby(x, as_index=False)[y].mean()
    agg = agg[agg[x].isin(order)]
    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X(f"{x}:N", sort=order, title=x.capitalize()),
        y=alt.Y(f"{y}:Q", title=f"Avg {y}"),
        tooltip=[x, alt.Tooltip(y, format=".2f")]
    ).properties(height=300, title=title)
    st.altair_chart(chart, use_container_width=True)

# ---------- App ----------
st.title("☕ Coffee Analysis (Kaggle CoffeeReview)")
st.caption("Loads `data/coffee/coffee.parquet` (15 columns). Use the sidebar to filter.")

df = load_df()
filtered = sidebar_filters(df)

tab_overview, tab_ratings, tab_sensory, tab_notes = st.tabs(["Data Overview", "Ratings", "Sensory", "Notes"])

with tab_overview:
    st.subheader("Data Overview")
    st.write(f"Rows: **{len(filtered):,}**  |  Columns: **{len(filtered.columns)}**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    prof = numeric_profile(filtered)
    if not prof.empty:
        st.markdown("**Numeric profile**")
        st.dataframe(prof, use_container_width=True, hide_index=True)

    # Download filtered CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, "coffee_filtered.csv", "text/csv")

with tab_ratings:
    st.subheader("Ratings")
    if "rating" in filtered.columns and filtered["rating"].notna().any():
        chart_hist(filtered["rating"], "Rating")
        if "roaster" in filtered.columns:
            chart_bar(filtered, "roaster", "rating", "Top Roasters by Average Rating", sort_desc=True, limit=15)
        if "origin" in filtered.columns:
            chart_bar(filtered, "origin", "rating", "Top Origins by Average Rating", sort_desc=True, limit=15)
    else:
        st.info("No rating data available.")
 
with tab_sensory:
    st.subheader("Sensory")
    present = [c for c in ["aroma","acidity","body","flavor","aftertaste"] if c in filtered.columns]
    if present:
        # Average bars for each sensory attribute
        means = filtered[present].mean(numeric_only=True).reset_index()
        means.columns = ["attribute","mean"]
        chart = alt.Chart(means).mark_bar().encode(
            x=alt.X("attribute:N", sort=present, title="Attribute"),
            y=alt.Y("mean:Q", title="Mean (filtered)"),
            tooltip=[alt.Tooltip("mean:Q", format=".2f")]
        ).properties(height=300, title="Average Sensory Scores")
        st.altair_chart(chart, use_container_width=True)

        # Scatter: aroma vs flavor if both present
        if all(c in filtered.columns for c in ["aroma","flavor"]):
            scat = alt.Chart(filtered.dropna(subset=["aroma","flavor"])).mark_circle().encode(
                x=alt.X("aroma:Q"),
                y=alt.Y("flavor:Q"),
                tooltip=["name","roaster","rating","aroma","flavor"]
            ).properties(height=320, title="Aroma vs Flavor")
            st.altair_chart(scat, use_container_width=True)
    else:
        st.info("No sensory columns available.")

with tab_notes:
    st.subheader("Notes")
    tt = top_terms(filtered, "notes", 25)
    if not tt.empty:
        chart = alt.Chart(tt).mark_bar().encode(
            x=alt.X("term:N", sort="-y"),
            y=alt.Y("count:Q"),
            tooltip=["term","count"]
        ).properties(height=320, title="Top Terms in Notes (filtered)")
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Sample notes**")
        if "name" in filtered.columns and "roaster" in filtered.columns:
            sample_cols = [c for c in ["name","roaster","origin","rating","notes"] if c in filtered.columns]
            st.dataframe(filtered[sample_cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.dataframe(filtered[["notes"]].head(25), use_container_width=True, hide_index=True)
    else:
        st.info("No 'notes' text available.")
