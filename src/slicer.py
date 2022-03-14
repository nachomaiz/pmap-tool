from typing import Callable, Literal, Optional
import pandas as pd

Slicer = Callable[[pd.DataFrame, pd.Index, pd.Index], Optional[pd.DataFrame]]


def row_slicer(X: pd.DataFrame, rows: pd.Index, cols: pd.Index) -> Optional[pd.DataFrame]:
    """Slices supplementary rows out of data based on row and column index."""
    if not rows.empty:
        if cols.empty:
            return X.loc[rows]
        return X.loc[rows, ~X.columns.isin(cols)]
    return None


def column_slicer(
    X: pd.DataFrame, rows: pd.Index, cols: pd.Index
) -> Optional[pd.DataFrame]:
    """Slices supplementary columns out of data based on row and column index."""
    if not cols.empty:
        if rows.empty:
            return X.loc[:, cols]
        return X.loc[~X.index.isin(rows), cols]
    return None


def core_slicer(X: pd.DataFrame, rows: pd.Index, cols: pd.Index) -> pd.DataFrame:
    """Slices supplementary columns out of data based on row and column index."""
    if not cols.empty or not rows.empty:
        return X.drop(index=rows, columns=cols)
    return X


def get_slicer(axis: Literal["index", "rows", 0, "columns", 1, "core"]) -> Slicer:
    """Returns the appropriate row or column slicer based on axis"""
    if axis in ["index", "rows", 0]:
        return row_slicer
    elif axis in ["columns", 1]:
        return column_slicer
    elif axis == "core":
        return core_slicer
    else:
        raise ValueError(f"{str(axis)} is not a valid DataFrame index.")


def supp_slice(
    X: pd.DataFrame,
    rows: Optional[pd.Index],
    cols: Optional[pd.Index],
    axis: Literal["index", "rows", 0, "columns", 1, "core"] = "index",
) ->  Optional[pd.DataFrame]:
    """Slice DataFrame based on supplementary data."""

    if rows is None:
        rows = pd.Index([])

    if cols is None:
        cols = pd.Index([])

    slicer = get_slicer(axis)
    return slicer(X, rows, cols)
