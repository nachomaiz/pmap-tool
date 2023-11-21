import altair as alt
import pandas as pd
import streamlit as st

from app.io import pickle_serialize
from app.model import Model
from app.params import PlotParams


@st.cache_data(hash_funcs={Model: pickle_serialize})
def plot_map(coords: pd.DataFrame, plot_params: PlotParams) -> alt.Chart:
    domain_x = (
        coords[plot_params.x_component].min() * 1.05,
        coords[plot_params.x_component].max() * 1.05,
    )
    domain_y = (
        coords[plot_params.y_component].min() * 1.05,
        coords[plot_params.y_component].max() * 1.05,
    )

    if plot_params.invert_x:
        domain_x = [domain_x[1], domain_x[0]]

    if plot_params.invert_y:
        domain_y = [domain_y[1], domain_y[0]]

    plot = (
        alt.Chart(coords.reset_index())
        .mark_text()
        .encode(
            x=alt.X(f"{plot_params.x_component}:Q", scale=alt.Scale(domain=domain_x)),
            y=alt.Y(f"{plot_params.y_component}:Q", scale=alt.Scale(domain=domain_y)),
            text="item",
            color="group:O",
            tooltip=alt.Tooltip(
                [
                    "item",
                    "group",
                    f"{plot_params.x_component}:Q",
                    f"{plot_params.y_component}:Q",
                ]
            ),
        )
        .configure_axis(grid=False)
        .interactive()
    )

    plot.height = 800
    return plot


def render(coords: pd.DataFrame, plot_params: PlotParams) -> None:
    st.altair_chart(plot_map(coords, plot_params), use_container_width=True)


def get_plot_params(plot_coords: pd.DataFrame) -> PlotParams:
    col1, col2 = st.columns(2)
    x_component = str(col1.slider("X component", 0, plot_coords.shape[1] - 1, 0, 1))
    y_component = str(col2.slider("Y component", 0, plot_coords.shape[1] - 1, 1, 1))
    invert_x = col1.toggle("Invert X")
    invert_y = col2.toggle("Invert Y")

    return PlotParams(x_component, y_component, invert_x, invert_y)
