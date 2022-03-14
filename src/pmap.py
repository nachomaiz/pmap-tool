from typing import Any, Literal, Optional, Union
from matplotlib.axes import Axes
from pandas.core.indexing import Index

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np
from prince.ca import CA
from prince import util, plot
import sklearn.utils.validation
from sklearn.exceptions import NotFittedError
from sklearn.base import TransformerMixin

from src.rotator import PmapRotator
import src.slicer as slc
from src.utils import isin_index, invert_plot_axis


class NoSupplementaryDataError(Exception):
    """Exception for when PMAP has no supplementary data."""


class Pmap(TransformerMixin):
    """Perceptual Map object based on prince Correspondence Analysis (CA).

    Parameters
    ----------
    n_components: int, default = 5
        Number of reduced dimensions to calculate.

    CA kwargs
    ---------
    n_iter: int, default = 10
        Maximum number of iterations for dimensionality reduction.
    copy: bool, default = True
        Copy inputs so data is not overwritten.
    check_input: bool, default = True
        Check inputs for positive values.
    benzecri: bool, default = False
        Benzecri (not sure what this does. Default from prince package is False)
    random_state: int, default = None
        Set random state for reproducible results.
    engine: str, default = auto
        Sets engine for dimensionality reduction. See prince reference materials.

    Attributes
    ----------
    data : pandas.DataFrame
        DataFrame passed to fit function.
    supp_rows, supp_cols : pandas.Index
        Index of supplementary rows/columns in the data df.
    estimator : prince.CA
        Correspondence Analysis estimator.
    core : pandas.DataFrame | None
        DataFrame containing data used in fit function,
        excluding supplementary data.
        Returns None if estimator is not fitted.
    supp_rows_slice, supp_cols_slice : pandas.DataFrame | None
        Sliced supplementary rows/columns. Helpful for using transform.
        Returns None if estimator is not fitted.
    result_dict : dict[str, pandas.DataFrame]
        Dictionary of data name and DataFrame.
        Useful for iterating.
    result : pandas.DataFrame
        Equivalent to running transform method on fit data.

    Methods
    -------
    fit : Fits Pmap with a pandas DataFrame and optional supplementary data.

    transform : Transforms pandas DataFrame returning row, column and supplementary
        coordinates.

    fit_transform : Equivalent to using fit, then transform.



    """

    def __init__(
        self,
        n_components: int = 5,
        rotation: Optional[str] = None,
        rotation_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        self.rotation = rotation
        self._rotation_kwargs = rotation_kwargs

        self.estimator = CA(n_components=n_components, **kwargs)
        self.n_components = self.estimator.n_components

        self.data: Union[pd.DataFrame, None] = None
        self.supp_rows: Union[pd.DataFrame, None] = None
        self.supp_cols: Union[pd.DataFrame, None] = None
        self.rows_rotator: Union[PmapRotator, None] = None
        self.cols_rotator: Union[PmapRotator, None] = None

    @property
    def rotation_kwargs(self) -> dict[str, Any]:
        """
        Modifies rotation keyword arguments to approximate SPSS rotation outputs.

        Explicitly passing 'rotation_kwargs' with the appropriate arguments will override SPSS parameters.

        Will add new conditions as needed and tested against SPSS.
        """

        if self.data is None:
            msg = f"This {self.__class__.__name__} instance is not fitted yet."
            raise NotFittedError(msg)

        spss_preset: dict[str, Any] = {}

        if self.rotation == "equamax":
            spss_preset.update({"kappa": self.n_components / (2 * self.data.shape[1])})

        if self._rotation_kwargs is not None:
            return (
                spss_preset | self._rotation_kwargs
            )  # Keeps passed kwargs over preset.

        return spss_preset

    def _check_is_fitted(self) -> None:
        """Convenience method for checking estimator is fitted."""
        sklearn.utils.validation.check_is_fitted(self.estimator, "total_inertia_")

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        supp_rows: Optional[Union[int, list, Index]] = None,
        supp_cols: Optional[Union[int, list, Index]] = None,
    ):
        """Fits PMAP to dataframe, handling supplementary data separately.

        Parameters
        ----------
        X : dataframe to fit PMAP
            This dataframe should have cases by row and attributes by columns.
            Case labels should be reflected in the index.
        y : (dependent) None
            Scikit-learn design pattern. Attribute is ignored.
        supp_rows, supp_cols : list | Index, default = None
            Rows and/or columns that should be treated as supplementary data.
            Accepts list of index/column values or pandas Index object.

        Returns
        -------
        PMAP object fitted to X.
        """
        self.data = X

        if self.rotation:
            self.rows_rotator = PmapRotator(self.rotation, **self.rotation_kwargs)
            self.cols_rotator = PmapRotator(self.rotation, **self.rotation_kwargs)

        if supp_rows is not None:
            if isinstance(supp_rows, int):
                supp_rows = X.index[-supp_rows:]
            elif isinstance(supp_rows, list):
                supp_rows = pd.Index(supp_rows)
            isin_index(X.index, supp_rows, "index")

        if supp_cols is not None:
            if isinstance(supp_cols, int):
                supp_cols = X.columns[-supp_cols:]
            if isinstance(supp_cols, list):
                supp_cols = pd.Index(supp_cols)
            isin_index(X.columns, supp_cols, "columns")

        self.supp_rows = supp_rows
        self.supp_cols = supp_cols

        self.estimator = self.estimator.fit(self.core)

        if self.rotation is not None:
            self.rows_rotator = PmapRotator(self.rotation, **self.rotation_kwargs).fit(
                self.estimator.row_coordinates(self.core)
            )
            self.cols_rotator = PmapRotator(self.rotation, **self.rotation_kwargs).fit(
                self.estimator.column_coordinates(self.core)
            )

        return self

    def _rotate_rows(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rotate compoment loadings."""
        self._check_is_fitted()

        return pd.DataFrame(
            self.rows_rotator.transform(X), index=X.index, columns=X.columns
        )

    def _rotate_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rotate compoment loadings."""
        self._check_is_fitted()

        return pd.DataFrame(
            self.cols_rotator.transform(X), index=X.index, columns=X.columns
        )

    def row_coords(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for estimator row_coordinates function."""
        res = self.estimator.row_coordinates(X)

        if self.rotation is not None:
            return self._rotate_rows(res)

        return res

    def col_coords(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for estimator column_coordinates function."""
        res = self.estimator.column_coordinates(X)

        if self.rotation is not None:
            return self._rotate_cols(res)

        return res

    def _result_dict(
        self,
        X: pd.DataFrame,
        rows: Optional[Index],
        cols: Optional[Index],
    ) -> dict[str, pd.DataFrame]:
        """Generate dictionary of results from arbitrary inputs."""

        self._check_is_fitted()

        result: dict[str, pd.DataFrame] = {}

        if not X.columns.equals(cols) and not X.index.equals(rows):

            isin_index(X.index, self.core.index, "index")
            isin_index(X.columns, self.core.columns, "columns")

            core = slc.supp_slice(X, rows, cols, "core")

            result["Rows"] = self.row_coords(core)
            result["Columns"] = self.col_coords(core)

        supp_rows = slc.supp_slice(X, rows, cols, "rows")
        supp_cols = slc.supp_slice(X, rows, cols, "columns")

        if supp_rows is not None and supp_rows.columns.equals(self.core.columns):
            result["Supp Rows"] = self.row_coords(supp_rows)
        else:
            result["Supp Rows"] = None

        if supp_cols is not None and supp_cols.index.equals(self.core.index):
            result["Supp Cols"] = self.col_coords(supp_cols)
        else:
            result["Supp Cols"] = None

        return result

    @property
    def result_dict(self) -> dict[str, pd.DataFrame]:
        """Dictionary of results from fitted data."""
        return self._result_dict(self.data, self.supp_rows, self.supp_cols)

    def _transform(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Transforms input data based on fit parameters.
        Rows or columns not in the original core data will be
        treated as supplementary data.
        """

        self._check_is_fitted()

        diff_rows = X.index.difference(self.core.index)
        diff_cols = X.columns.difference(self.core.columns)

        return self._result_dict(X, diff_rows, diff_cols)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transforms input data based on fit parameters.
        Rows or columns not in the original core data will be
        treated as supplementary data.
        """
        return pd.concat(self._transform(X))

    ## ATTRIBUTES
    @property
    def core(self) -> Optional[pd.DataFrame]:
        """Data for model fitting.
        Stripped of supplementary data if it exists."""

        if self.data is None:
            return None
        return slc.supp_slice(self.data, self.supp_rows, self.supp_cols, "core")

    @property
    def supp_rows_slice(self) -> Optional[pd.DataFrame]:
        """Transformed supplementary rows from fit method."""

        if self.supp_rows is None:
            return None
        return slc.supp_slice(self.data, self.supp_rows, self.supp_cols, "rows")

    @property
    def supp_cols_slice(self) -> Optional[pd.DataFrame]:
        """Transformed supplementary cols from fit method."""

        if self.supp_cols is None:
            return None
        return slc.supp_slice(self.data, self.supp_rows, self.supp_cols, "columns")

    @property
    def result(self) -> Optional[pd.DataFrame]:
        """DataFrame of result."""
        return pd.concat(self.result_dict)

    @property
    def has_supp_rows(self) -> bool:
        """Check if supplementary rows exist."""
        if self.supp_rows is None:
            return False
        if self.supp_rows.empty:
            return False
        return True

    @property
    def has_supp_cols(self) -> bool:
        """Check if supplementary rows exist."""
        if self.supp_cols is None:
            return False
        if self.supp_cols.empty:
            return False
        return True

    @property
    def has_supp(self) -> bool:
        """Check if supplementary data exist."""
        if self.has_supp_rows or self.has_supp_cols:
            return True
        return False

    ## PLOTTING
    def plot_eigenvalues(self) -> tuple[Axes, Axes]:
        """Plot eigenvalues from fitted data.

        Parameters
        ----------
        X : pd.DataFrame
            _description_

        Returns
        -------
        tuple[Figure, Axes, Axes]
            matplotlib Figure and Axes objects
        """
        f, ax1 = plt.subplots()

        eigen = self.estimator.eigenvalues_

        ax1.bar(x=range(self.n_components), height=eigen, color=["red"])
        ax1.set_ylabel("Eigenvalues")

        ax2 = plt.twinx()

        ax2.set_ylim(0, 1.05)

        ax2.plot(
            range(self.n_components), np.cumsum([val / sum(eigen) for val in eigen])
        )
        ax2.set_ylabel("Cumulative Variance Explained")

        return ax1, ax2

    @staticmethod
    def _plot_map(
        X: pd.DataFrame,
        ax: Axes,
        data_name: str,
        color: Optional[str],
        labels: bool = True,
        only_labels: bool = False,
        **kwargs,
    ) -> None:
        """Refactored plot function to enable loops."""
        _, names, _, _ = util.make_labels_and_names(X)

        # Plot coordinates
        x = X.iloc[:, 0]
        y = X.iloc[:, 1]

        scatter = ax.scatter(x, y, label=data_name, color=color, **kwargs)

        # Add labels
        if labels:
            if not only_labels:
                annot_kwargs = {"ha": "left", "va": "bottom"}
            else:
                annot_kwargs = {"ha": "center", "va": "center", "color": color}
                scatter.remove()

            for xi, yi, lab in zip(x, y, names):
                ax.annotate(str(lab), (xi, yi), **annot_kwargs)

    def plot(self, *args, **kwargs) -> plt.Axes:
        """Generic plotting function for pipelines."""
        return self.plot_map(*args, **kwargs)

    def plot_map(
        self,
        X: Optional[pd.DataFrame] = None,
        x_component: int = 0,
        y_component: int = 1,
        supp: Union[bool, Literal["only"]] = False,
        figsize: tuple[int, int] = (16, 9),
        ax: Optional[plt.Axes] = None,
        show_labels: Optional[list[bool]] = None,
        only_labels: bool = True,
        invert_ax: Optional[Literal["x", "y", "b"]] = None,
        stylize: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plots perceptual map from trained self.

        Parameters
        ----------
        figsize : tuple(int)
            Size of the returned plot. Ignored if ax is not None
        ax : matplotlib Axis, default = None
            The axis to plot into. Defaults to None, creating a new ax
        x_component, y_component : int
            Component from the trained self to use as x and y axis
            in the perceptual map.
        supp : bool, 'only'
            Plot supplementary data (if present).
            'only' will suppress core data and show supplementary data instead.
            Ignored if no supplementary data exists.
        show_labels : list[bool], default = None
            [bool, bool] = show labels for [rows, columns].
                If supp = True, shows all rows or columns
            [bool, bool, bool, bool] = only if supp == True,
                show labels for [rows, columns, supp rows, supp columns]
            If None, defaults to [True, True]
        only_labels : bool, default = True
            Only plot labels. Labels will be centered on the original coordinates.
        invert_ax : str, default = None
            'x' = invert x axis
            'y' = invert y axis
            'b' = invert both axis
        stylize: bool, default = True
            Add axis origin lines to ax.
        **kwargs
            Additional arguments to pass to matplotlib plotting function.

        Returns
        -------
        ax : matplotlib Axes
            Perceptual Map plot
        """

        if X is not None:
            result = self._transform(X)
        else:
            result = self.result_dict

        if ax is None:
            f, ax = plt.subplots(figsize=figsize)

        if show_labels is None:
            show_labels = [True, True]

        if stylize:
            ax = plot.stylize_axis(ax, grid=False)

        if supp:
            if self.supp_rows is None and self.supp_cols is None:
                raise NoSupplementaryDataError("PMAP has no supplementary data.")

        # Change inputs based on function parameters
        # if supp == "only" or supp == False all tuples are len 2, if supp == True all tuples are len 4
        if supp == "only" or supp == False:
            if len(show_labels) != 2:
                raise ValueError("Length of show_labels expected to be 2")
        elif supp == True:
            if len(show_labels) not in (2, 4):
                raise ValueError("Length of show_labels expected to be 2 or 4")
            show_labels = show_labels * 2 if len(show_labels) == 2 else show_labels
        else:
            raise ValueError("supp must be True, False or 'only'")

        # for only_labels
        legend_handles = []

        # Main plotting loop
        for (name, df), l in zip(result.items(), show_labels):
            if df is not None:
                color = next(ax._get_lines.prop_cycler)["color"]
                self._plot_map(
                    df.loc[:, [x_component, y_component]],
                    ax,
                    name,
                    color,
                    l,
                    only_labels,
                    **kwargs,
                )
                if only_labels and l:
                    legend_handles.append(mpatches.Patch(color=color, label=name))

        if invert_ax is not None:
            ax = invert_plot_axis(ax, invert_ax)

        if only_labels:
            ax.legend(handles=legend_handles)
        else:
            ax.legend()

        # Text
        ei = self.estimator.explained_inertia_
        ax.set_title(
            f"Principal Coordinates ({(ei[y_component] + ei[x_component]):.2%} total inertia)"
        )
        ax.set_xlabel(f"Component {x_component} ({ei[x_component]:.2%} inertia)")
        ax.set_ylabel(f"Component {y_component} ({ei[y_component]:.2%} inertia)")

        return ax
