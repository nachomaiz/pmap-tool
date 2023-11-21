import pandas as pd
import streamlit as st

from app.io import download_button
from app.params import PlotParams


def render(coords: pd.DataFrame, plot_params: PlotParams) -> None:
    def highlight_columns(s: pd.Series) -> list[str]:
        return [
            "background-color: #FF4B4B;" if s.name in to_highlight else "" for _ in s
        ]

    plot_components = (plot_params.x_component, plot_params.y_component)

    st.subheader(
        "Download plot data", help="Will only download columns highlighted in red."
    )
    col1, col2 = st.columns(2)
    download_raw = col1.toggle(
        "Download raw data", False, help="Ignores inverted axes."
    )
    to_highlight = coords.columns if download_raw else set(plot_components)

    to_invert = [
        col
        for col, inv in zip(
            plot_components, (plot_params.invert_x, plot_params.invert_y)
        )
        if inv
    ]

    if col2.toggle("Apply axis inversions", not download_raw, disabled=download_raw):
        coords = coords.assign(**{col: coords[col] * -1 for col in to_invert})

    st.dataframe(coords.style.apply(highlight_columns), use_container_width=True)

    download_button(
        "💾 Download map result",
        coords
        if download_raw
        else coords[[plot_params.x_component, plot_params.y_component]],
        "output.xlsx",
        type="primary",
        use_container_width=True,
    )
