"""Data loading utilities for the uranium mining dataset.

This module encapsulates logic for reading configuration values from
``config.json`` and loading the uranium dataset into a pandas
``DataFrame``.  It mirrors the pattern used in the ``colorado_river``
package: configuration is kept separate from code, and a small set of
well‑defined functions provide high‑level access to the underlying data.

Users of this module should call :func:`load_dataset` to obtain a
pandas DataFrame containing all records.  The default configuration
expects a ``data/uranium_mines.csv`` file living inside the
``uranium_dataset`` package directory.  Should you wish to override
this location or supply additional configuration values (e.g. API
keys), edit ``config.json`` accordingly.

In addition to the simple loader, this module provides helper
functions to read and cache the configuration.  All public functions
carry extensive docstrings and inline comments to aid readability and
teaching purposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def load_config(path: Optional[str | Path] = None) -> dict:
    """Load configuration from ``config.json``.

    Parameters
    ----------
    path : str or Path, optional
        The path to the configuration JSON file.  If omitted, the
        function will look for ``config.json`` in the same directory
        as this module.  Passing an explicit path is useful for
        testing or when running notebooks from a different working
        directory.

    Returns
    -------
    dict
        A dictionary representation of the JSON contents.  Keys
        typically include ``dataset_path``, ``openai_api_key`` and
        optionally ``states`` or ``permit_dataset_path``.

    Notes
    -----
    This function separates configuration concerns from the rest of
    the code.  By reading a JSON file rather than hard‑coding values
    in Python, you can change dataset locations or API keys without
    modifying source code.  The pattern is borrowed from the
    ``colorado_river`` package, which loads its site catalogue from
    ``config.json``【470012308847619†L0-L69】.
    """

    # Determine the default configuration path relative to this file.
    if path is None:
        # __file__ points to the current module (e.g., ``data_loader.py``).
        here = Path(__file__).resolve().parent
        path = here / "config.json"
    else:
        path = Path(path)

    # Read the file contents.  Using UTF‑8 ensures compatibility with
    # international characters in configuration values.
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def load_dataset(
    config: Optional[dict] = None,
    usecols: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load the uranium dataset into a pandas DataFrame.

    Parameters
    ----------
    config : dict, optional
        A configuration dictionary obtained via :func:`load_config`.
        If omitted, :func:`load_config` will be called with its
        default behaviour to locate ``config.json``.  The dataset
        path will be derived from the ``dataset_path`` key.
    usecols : sequence of str, optional
        A list or other iterable of column names to read from the
        CSV.  This can speed up loading when only a subset of fields
        is required.  See ``pandas.read_csv`` for details.
    **read_csv_kwargs
        Additional keyword arguments passed directly to
        :func:`pandas.read_csv`.  You might use these to specify
        delimiter, encoding, row filtering or data type hints.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the requested columns of the uranium
        dataset.  Column names come directly from the CSV header.

    Examples
    --------
    Basic usage:

    >>> from uranium_dataset.data_loader import load_dataset
    >>> df = load_dataset()
    >>> df.head()

    Selecting just a few columns:

    >>> df = load_dataset(usecols=["state", "dep_type"])

    Notes
    -----
    The dataset is read lazily: no caching is performed by default.
    If you find yourself repeatedly loading the same dataset in a
    performance‑critical context (e.g. inside a Streamlit app), you
    could memoise this function or save the DataFrame to a binary
    format such as Parquet.  For the purposes of this educational
    package, we emphasise clarity over micro‑optimisations.
    """

    # Load configuration if not supplied by caller.
    if config is None:
        config = load_config()

    # Derive the dataset path.  We resolve relative to this module's
    # directory so that notebooks executed from other working
    # directories still locate the dataset correctly.
    here = Path(__file__).resolve().parent
    dataset_path = here / config.get("dataset_path", "data/uranium_mines.csv")

    # Use pandas to load the CSV file into a DataFrame.  We pass
    # through any additional keyword arguments to allow customisation.
    df = pd.read_csv(dataset_path, usecols=usecols, **read_csv_kwargs)

    return df


__all__ = ["load_config", "load_dataset"]
