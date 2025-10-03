"""Exploratory data analysis helpers for the uranium dataset.

This module contains small, composable functions that operate on
pandas DataFrames loaded from :func:`~uranium_dataset.data_loader.load_dataset`.
These helpers encapsulate common exploratory steps, such as counting
records per state, summarising deposit types, and computing basic
statistics about numeric fields.  The goal is to keep analysis logic
separate from data loading and user interfaces, facilitating reuse in
both Python scripts and notebooks.

Each function is documented with a summary of its purpose and usage
examples.  Many functions return pandas DataFrames to enable easy
chaining with further analysis or visualisation libraries.  Inline
comments explain the intent of key operations, making the code
suitable for teaching or self‑study.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def count_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of records per state.

    Parameters
    ----------
    df : pandas.DataFrame
        The uranium dataset.  Must include a ``state`` column.

    Returns
    -------
    pandas.DataFrame
        A two‑column DataFrame with ``state`` and ``count``.  The
        rows are sorted in descending order of count.

    Examples
    --------
    >>> from uranium_dataset.data_loader import load_dataset
    >>> from uranium_dataset.eda import count_by_state
    >>> df = load_dataset()
    >>> count_by_state(df).head()

    Notes
    -----
    This function uses pandas grouping and aggregation to compute
    counts.  It returns a new DataFrame rather than modifying the
    input.  Sorting ensures that the largest counts appear first.
    """

    # Group the DataFrame by state and count the number of rows per group.
    counts = (
        df.groupby("state", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    return counts


def count_by_dep_type(df: pd.DataFrame) -> pd.DataFrame:
    """Count records by deposit type (``dep_type``).

    Parameters
    ----------
    df : pandas.DataFrame
        The uranium dataset.  Must include a ``dep_type`` column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with ``dep_type`` and ``count`` columns, sorted
        descending by count.  Missing values are represented with
        ``<missing>`` for clarity.

    Examples
    --------
    >>> df = load_dataset(usecols=["dep_type"])
    >>> count_by_dep_type(df)

    Notes
    -----
    Many MRDS records have blank deposit types.  Converting NaNs to
    the string ``<missing>`` before counting can make the resulting
    table easier to interpret in downstream notebooks.
    """

    # Replace missing values with a placeholder string.
    dep_series = df["dep_type"].fillna("<missing>")

    counts = (
        dep_series.value_counts()
        .reset_index()
        .rename(columns={"index": "dep_type", 0: "count"})
        .sort_values(by="count", ascending=False)
    )

    return counts


def numeric_summary(df: pd.DataFrame, numeric_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Compute summary statistics for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The uranium dataset.  Numeric columns (e.g. latitude,
        longitude) will be summarised.
    numeric_columns : list of str, optional
        A list of specific numeric columns to summarise.  If omitted
        ``df.select_dtypes`` is used to infer numeric columns.

    Returns
    -------
    pandas.DataFrame
        A DataFrame whose index contains the column names and whose
        columns contain summary statistics (count, mean, std, min,
        quartiles, max).  See :meth:`pandas.DataFrame.describe` for
        details on the computed metrics.

    Examples
    --------
    >>> df = load_dataset()
    >>> numeric_summary(df, numeric_columns=["latitude", "longitude"])
    """

    # Determine which columns to summarise.  If none supplied, select
    # columns with a numeric dtype automatically.  The ``include``
    # parameter accepts numpy/pandas dtypes or Python primitives.
    if numeric_columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
    else:
        numeric_cols = numeric_columns

    summary = df[numeric_cols].describe()
    return summary


__all__ = ["count_by_state", "count_by_dep_type", "numeric_summary"]
