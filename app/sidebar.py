import altair as alt
import pandas as pd
import streamlit as st
from prince import CA

from app.backend import AppState
from app.io import load_data
from app.params import ModelParams, Rotation, SuppParams

# Minimum to draw map
MIN_ROWS = 3
MIN_COLS = 2


def get_model_params(state: AppState, data: pd.DataFrame) -> ModelParams:
    with st.form("form_model_params"):
        n_iter = int(st.number_input("Number of iterations", 1, 100, 10, 5))

        n_components = int(st.slider("Number of components", 2, data.shape[1], 5))

        rotation = st.selectbox("Rotation", tuple(Rotation), index=None)

        st.form_submit_button(
            "Run",
            on_click=state.set,
            args=("model_params_set", True),
            use_container_width=True,
        )

        return ModelParams(n_iter, n_components, rotation)


def get_supp_params(state: AppState, data: pd.DataFrame) -> SuppParams:
    with st.form("form_supp_params"):
        supp_rows: list[str] = st.multiselect(
            "Supplementary rows",
            data.index,
            max_selections=data.shape[0] - MIN_ROWS,
            placeholder="Choose option(s)",
        )
        supp_cols = st.multiselect(
            "Supplementary columns",
            data.columns,
            max_selections=data.shape[1] - MIN_COLS,
            placeholder="Choose option(s)",
        )

        st.form_submit_button(
            "Confirm",
            on_click=state.set,
            args=("pmap_data_loaded", True),
            use_container_width=True,
        )

    return SuppParams(supp_rows, supp_cols)


def plot_eigenvalues(data: pd.DataFrame) -> alt.Chart:
    model = CA(data.shape[1]).fit(data)
    eigen = model._eigenvalues_summary.reset_index()  # pylint: disable=protected-access
    eigen = eigen.assign(
        **{
            "Number of components": eigen["component"].astype(int) + 1,
            "% of variance": eigen["% of variance"] / 100,
        }
    )
    return (
        alt.Chart(eigen)
        .mark_line()
        .encode(x="Number of components:O", y=r"% of variance:Q")
        .configure_axis(grid=False)
    )


def render(
    state: AppState,
) -> tuple[pd.DataFrame | None, SuppParams | None, ModelParams | None]:
    with st.expander("Load data", expanded=not state.pmap_data_loaded):
        data = load_data("Upload perceptual map data:", "uploader_data")

        if data is None:
            return None, None, None

        supp_params = get_supp_params(state, data)

    params_form = st.container()

    st.subheader(
        "Elbow Plot", help="Use to determine the number of components for rotations"
    )
    st.altair_chart(
        plot_eigenvalues(data.drop(index=supp_params.rows, columns=supp_params.cols)),
        use_container_width=True,
    )

    with params_form:
        st.header("Model parameters")
        model_params = get_model_params(state, data)

    if not state.all_completed():
        return data, supp_params, None

    return data, supp_params, model_params
