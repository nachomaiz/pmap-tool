from dataclasses import dataclass
from typing import Self

import pandas as pd
import streamlit as st
from prince import CA

from app.io import pickle_serialize
from app.params import ModelParams, SuppParams
from src.rotator import PmapRotator


@dataclass
class Model:
    ca: CA
    rotator: PmapRotator | None
    supp: SuppParams

    @classmethod
    def build(
        cls, params: ModelParams, n_features: int, supp_params: SuppParams
    ) -> Self:
        ca = CA(params.n_components, params.n_iter)

        if params.rotation:
            kappa = params.n_components // (2 * n_features)
            rotator = PmapRotator(params.rotation, kappa=kappa)
        else:
            rotator = None

        return cls(ca, rotator, supp_params)

    def fit(self, data: pd.DataFrame) -> Self:
        self.ca.fit(data.drop(index=self.supp.rows, columns=self.supp.cols))
        if self.rotator is not None:
            col_coords = self.ca.column_coordinates(data.drop(index=self.supp.rows))
            self.rotator.fit(col_coords)

        return self

    def row_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        res = self.ca.row_coordinates(data)

        if self.rotator is not None:
            return pd.DataFrame(
                self.rotator.transform(res), index=res.index, columns=res.columns
            )

        return res

    def column_coordinates(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for estimator column_coordinates function."""
        res = self.ca.column_coordinates(X)

        if self.rotator is not None:
            return pd.DataFrame(
                self.rotator.transform(res), index=res.index, columns=res.columns
            )
        return res


@st.cache_data(hash_funcs={Model: pickle_serialize})
def get_ca_model(data: pd.DataFrame, params: ModelParams, supp: SuppParams) -> Model:
    return Model.build(params, data.drop(columns=supp.cols).shape[1], supp).fit(data)


@st.cache_data(hash_funcs={Model: pickle_serialize})
def get_plot_coords(model: Model, data: pd.DataFrame, supp: SuppParams) -> pd.DataFrame:
    row_coords = model.row_coordinates(data).rename(columns=str)
    col_coords = model.column_coordinates(data).rename(columns=str)
    coord_groups = {
        "Rows": row_coords.drop(index=supp.rows),
        "Columns": col_coords.drop(index=supp.cols),
        "Supplementary rows": row_coords.loc[supp.rows],
        "Supplementary columns": col_coords.loc[supp.cols],
    }
    return pd.concat(coord_groups).rename_axis(index=["group", "item"])
