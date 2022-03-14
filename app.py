from typing import Optional, Any

import matplotlib.pyplot as plt
import pandas as pd
import cycler

import streamlit as st

from src.rotator import ROTATIONS
from src.pmap import Pmap
from src.utils import clean_pct, format_data_to_plot, download_button

SAMPLE_PATH = "data/brand data.xlsx"

KwargDict = dict[str, Any]


def ready_state(value: bool) -> None:
    """Set ready state in session."""
    st.session_state.__setitem__("ready", value)


def load_data() -> Optional[pd.DataFrame]:
    """Load sample data or upload file."""
    upload_disabled = False

    if st.sidebar.checkbox("Load sample data", on_change=ready_state, args=(False,)):
        upload_disabled = False
        return pd.read_excel(SAMPLE_PATH, index_col=0)

    file = st.sidebar.file_uploader(
        "Please upload your Excel file:",
        type=["xls", "xlsx"],
        accept_multiple_files=False,
        disabled=upload_disabled,
    )
    if file:
        return pd.read_excel(file, index_col=0)


def get_model_params(n_cols: int) -> tuple:
    """Get main parameters for Pmap object."""
    n_iter = st.number_input("# of iterations", 1, 100, 10, step=5)

    n_comp_help = """
        For performance reasons the max is 10. 
        Allows one less than the total number of data columns.
        
        4 or more are recommended for rotations.
        """

    if n_cols > 3:
        n_components = st.slider(
            "Number of components (default 5)",
            2,
            10 if n_cols > 10 else n_cols - 1,
            5,
            help=n_comp_help,
        )
    else:
        n_components = 2
        st.info("Max number of components for 3 columns = 2")

    rot_help = """`varimax` rotation recommended."""
    rot = st.selectbox("Rotation", ["None"] + ROTATIONS, help=rot_help)
    if rot == "None":
        rotation = None
    else:
        rotation = rot

    return n_components, rotation, n_iter


def get_supp_params(
    data: pd.DataFrame, n_rows: int, n_cols: int
) -> tuple[Optional[list], Optional[list]]:
    """Get Supplementary data parameters."""

    supp_rows, supp_cols = [], []
    max_supp_rows, max_supp_cols = [i - 3 for i in data.shape]

    # Show only if each data shape dimension is large enough
    if n_rows > 3:
        supp_rows = st.multiselect(
            "Supplementary rows",
            data.index,
            format_func=clean_pct,
            help=f"Max: {max_supp_rows}",
        )

        if len(supp_rows) > max_supp_rows:
            st.error("Must leave at least 3 rows as core data.")

    if n_cols > 3:
        supp_cols = st.multiselect(
            "Supplementary columns",
            data.columns,
            format_func=clean_pct,
            help=f"Max: {max_supp_cols}",
        )

        if len(supp_cols) > max_supp_cols:
            st.error("Must leave at least 3 columns as core data.")

    return supp_rows, supp_cols


def get_pmap_model(data: pd.DataFrame, n_rows: int, n_cols: int) -> Pmap:
    """App sidebar to fit Pmap model."""

    n_rows, n_cols = data.shape

    # Warning for small datasets
    if n_rows < 4 or n_cols < 4:
        dim = "rows" if n_rows < 4 else "columns"
        comps = "components or" if n_cols < 4 else ""
        st.warning(
            f"""Your data can be turned into a perceptual map, but it only has 3 {dim},
            so you cannot pick {comps} supplementary {dim}."""
        )

    # Model parameters

    n_components, rotation, n_iter = get_model_params(n_cols)

    return Pmap(n_components, rotation, n_iter=n_iter)


def sidebar() -> tuple[Pmap, pd.DataFrame]:
    """App sidebar."""
    st.sidebar.title("Perceptual Map Setup")

    data = load_data()

    if data is None:
        st.stop()

    try:
        data = data.astype("float")
    except ValueError:
        st.sidebar.error("Only numeric types are supported.")
        st.stop()

    st.sidebar.info(
        f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns."
    )

    n_rows, n_cols = data.shape

    with st.sidebar.form("parameters"):
        model = get_pmap_model(data, n_rows, n_cols)
        supp_rows, supp_cols = get_supp_params(data, n_rows, n_cols)
        st.form_submit_button("Run", on_click=ready_state, args=(True,))

    if not st.session_state["ready"]:
        st.stop()
    else:
        st.sidebar.success("Success!")

    model = model.fit(data, supp_rows=supp_rows, supp_cols=supp_cols)

    return model, data


def get_plot_params(model: Pmap) -> tuple[KwargDict, KwargDict]:
    """Get plot parameters."""
    with st.expander("Plot Parameters"):

        plot_colors = {
            "Dark": {"bg": "#0E1117", "fg": "#FAFAFA"},
            "Light": {"bg": "#FFFFFF", "fg": "#0E1117"},
        }

        plot_theme = st.select_slider("Plot theme", ["Dark", "Light"], "Dark")

        context_params = {
            "figure.facecolor": plot_colors[plot_theme]["bg"],
            "text.color": plot_colors[plot_theme]["fg"],
            "axes.facecolor": plot_colors[plot_theme]["bg"],
            "axes.edgecolor": plot_colors[plot_theme]["fg"],
            "axes.labelcolor": plot_colors[plot_theme]["fg"],
            "xtick.color": plot_colors[plot_theme]["fg"],
            "xtick.labelcolor": plot_colors[plot_theme]["fg"],
            "ytick.color": plot_colors[plot_theme]["fg"],
            "ytick.labelcolor": plot_colors[plot_theme]["fg"],
            "axes.prop_cycle": cycler.cycler("color", plt.cm.Dark2(range(0, 4))),
            "font.size": "12.0",
        }

        plot_params = {}

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            plot_params["x_component"] = st.slider(
                "Horizontal component", 0, model.n_components, 0
            )

        with col1_2:
            plot_params["y_component"] = st.slider(
                "Vertical component", 0, model.n_components, 1
            )

        plot_params["supp"] = st.select_slider(
            "Show Supplementary",
            [False, True],
            True if model.has_supp else False,
            format_data_to_plot,
            disabled=not model.has_supp,
        )

        st.write("Show Labels:")
        col3_1, col3_2, col3_3, col3_4 = st.columns(4)
        with col3_1:
            row_l = st.checkbox("Rows", True)
        with col3_2:
            col_l = st.checkbox("Columns", True)
        with col3_3:
            srow_l = st.checkbox(
                "Supp Rows", model.has_supp_rows, disabled=not model.has_supp_rows
            )
        with col3_4:
            scol_l = st.checkbox(
                "Supp Columns", model.has_supp_cols, disabled=not model.has_supp_cols
            )

        plot_params["show_labels"] = [row_l, col_l, srow_l, scol_l]

        if not plot_params["supp"]:
            plot_params["show_labels"] = [row_l, col_l]

        st.write("Style:")
        col4_1, col4_2 = st.columns(2)
        with col4_1:
            plot_params["only_labels"] = st.checkbox("Only Labels", True)
            plot_params["stylize"] = st.checkbox("Show Origin", True)
        with col4_2:
            invert_x = st.checkbox("Invert x axis")
            invert_y = st.checkbox("Invert y axis")

        plot_params["invert_ax"] = (
            "b"
            if invert_x and invert_y
            else "x"
            if invert_x
            else "y"
            if invert_y
            else None
        )

    return plot_params, context_params


def get_pmap_data(
    pmap: Pmap,
    x_component: int = 0,
    y_component: int = 1,
    invert_ax: Optional[str] = None,
) -> pd.DataFrame:
    """Returns two dimensions of multi-indexed, invert-compatible fitted data"""
    d = pmap.result.iloc[:, [x_component, y_component]]
    if invert_ax is not None:
        if invert_ax == "x":
            d.iloc[:, x_component] = d.iloc[:, x_component] * -1
        elif invert_ax == "y":
            d.iloc[:, y_component] = d.iloc[:, y_component] * -1
        elif invert_ax == "b":
            d = d * -1
        else:
            raise ValueError("invert_ax must be 'x', 'y' or 'b' for both")

    return d


def main():
    """Main Streamlit App"""

    st.set_page_config(page_title="Perceptual Map Tool", page_icon="💡")

    st.write(
        """
        # Perceptual Map Tool
        You can use this tool to create perceptual maps, download the results, 
        and supports supplementary data.\n
        [More info](https://en.wikipedia.org/wiki/Perceptual_mapping) on perceptual maps.\n
        Make sure the data is in the right format: 
        - Names on left-most column
        - All numerical columns labeled

        Load sample data provided to see an example of the correct data format.
        """
    )

    model, data = sidebar()

    if model.rotation and model.has_supp:
        max_supp_rows, max_supp_cols = (model.n_components - 4, model.n_components - 3)
        model_sr, model_sc = len(model.supp_rows), len(model.supp_cols)

        if (model_sr > max_supp_rows) or (model_sc > max_supp_cols):
            st.warning(
                f"""
                Max supplementary rows and/or columns for rotations exceeded:
                - Supplementary rows: {model_sr} (max {max_supp_rows})
                - Supplementary columns: {model_sc} (max {max_supp_cols})
                """,
            )
            st.stop()

    st.info(
        f"""
        Components: {model.n_components},
        Rotation: {model.rotation},
        Supp Rows: {len(model.supp_rows)},
        Supp Cols: {len(model.supp_cols)}
        """
    )

    with st.expander("Show Data"):
        st.dataframe(data)

    plot_params, context_params = get_plot_params(model)

    with plt.style.context(context_params):
        fig, ax = plt.subplots(figsize=(16, 9))
        model.plot_map(ax=ax, **plot_params)
    st.pyplot(fig)

    st.write(
        "You can download the coordinates to build your charts, or copy the image above."
    )

    out_data = get_pmap_data(
        model,
        x_component=plot_params["x_component"],
        y_component=plot_params["y_component"],
        invert_ax=plot_params["invert_ax"],
    )

    with st.expander("Download Output"):
        st.markdown(
            download_button(
                out_data.reset_index(), "pmap-output.xlsx", "Download output as excel"
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(out_data)

    st.info(
        """Developed with ❤ by [nachomaiz](https://github.com/nachomaiz)
            based on the [prince](https://github.com/MaxHalford/prince) and
            [factor_analyzer](https://github.com/ets/factor_analyzer) packages. 
            [GitHub repo](https://github.com/nachomaiz/pmap).
            """
    )


if __name__ == "__main__":
    main()
