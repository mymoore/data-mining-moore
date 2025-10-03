"""Command‑line interface for uranium dataset metadata.

This module provides a small CLI that prints basic information
about the uranium dataset.  It mirrors the ``meta.py`` script found
in the ``colorado_river`` package【63721919611428†L0-L15】 by exposing
data discovery functionality outside of a UI context.  Users can
invoke this script directly from the command line to quickly
inspect high‑level statistics such as the number of records, unique
states, and deposit types.

The script uses Python's :mod:`argparse` to define command‑line
options.  Additional subcommands could be added in the future (for
example, to dump the dataset schema or to export a subset).  The
functions defined here can also be imported and reused programmatically
in other contexts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .data_loader import load_config, load_dataset
from .eda import count_by_state, count_by_dep_type


def print_overview() -> None:
    """Print a brief overview of the uranium dataset.

    This function loads the dataset using the default configuration
    and prints to stdout the number of rows, number of columns,
    available states and a sample of deposit types.  It is called
    when the CLI is invoked without subcommands.
    """

    # Load the dataset using the default configuration.  The call
    # returns a pandas DataFrame.
    df = load_dataset()

    # Compute high‑level statistics.  We convert sets to sorted lists
    # for consistent output ordering.
    n_rows, n_cols = df.shape
    states = sorted(set(df["state"].dropna()))
    dep_types = sorted(set(df["dep_type"].dropna()))

    print(f"Uranium dataset overview")
    print("------------------------")
    print(f"Records: {n_rows}")
    print(f"Columns: {n_cols}")
    print(f"States ({len(states)}): {', '.join(states)}")
    print(f"Deposit types ({len(dep_types)}): {', '.join(dep_types[:10])}")
    if len(dep_types) > 10:
        print("  ...")


def print_counts_by_state() -> None:
    """Print the number of records per state in descending order."""
    df = load_dataset(usecols=["state"])
    counts = count_by_state(df)
    print(counts.to_string(index=False))


def print_counts_by_dep_type() -> None:
    """Print the number of records per deposit type."""
    df = load_dataset(usecols=["dep_type"])
    counts = count_by_dep_type(df)
    print(counts.to_string(index=False))


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the CLI.

    Parameters
    ----------
    argv : iterable of str, optional
        The command‑line arguments, excluding the program name.
        Supplying this argument makes the function testable.

    Returns
    -------
    int
        An exit status (0 for success, non‑zero for errors).
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect and summarise the uranium mining dataset. "
            "By default, prints a high‑level overview; use subcommands "
            "for more detailed counts."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for counting by state.
    subparsers.add_parser(
        "state", help="List the number of records per state"
    )
    # Subcommand for counting by deposit type.
    subparsers.add_parser(
        "type", help="List the number of records per deposit type"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # Dispatch based on chosen subcommand.
    if args.command == "state":
        print_counts_by_state()
    elif args.command == "type":
        print_counts_by_dep_type()
    else:
        print_overview()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
