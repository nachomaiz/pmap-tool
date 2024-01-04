import pandas as pd
import streamlit as st

from app.io import download_button
from app.params import PlotParams, maybe_invert_coords


def render(coords: pd.DataFrame, plot_params: PlotParams) -> None:
    def highlight_columns(s: pd.Series) -> list[str]:
        return [
            "background-color: #FF4B4B;" if s.name in to_highlight else "" for _ in s
        ]

    plot_components = (plot_params.x_component, plot_params.y_component)

    st.subheader(
        "Download plot data",
        help="Selected components (highlighted in red) will be downloaded.",
    )
    col1, col2 = st.columns(2)
    download_raw = col1.toggle("Download all components", False)
    to_highlight = coords.columns if download_raw else set(plot_components)

    if col2.toggle("Apply axis inversions", True):
        coords = maybe_invert_coords(coords, plot_params)

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
