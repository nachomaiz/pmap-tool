import streamlit as st

from src import output, plot, sidebar
from src.backend import get_state, set_default_state
from src.model import get_ca_model, get_plot_coords

ABOUT_INFO = """pmap-tool is a free online tool to create perceptual maps.

Check out the project on [GitHub](https://github.com/nachomaiz/pmap-tool/)."""

APP_INFO = """Create perceptual maps from a table of items as rows and associations
or attributes as columns, and download the perceptual map coordinates.

Based on Correspondence Analysis, with support for supplementary rows and columns,
as well as rotation of the results.

Perceptual maps can be a great tool to understand the relationships of various items,
based on a common set of attributes, to each other as well as the attributes
most associated with them. Read more on perceptual maps from
[Wikipedia](https://en.wikipedia.org/wiki/Perceptual_mapping).

## Instructions

To use the tool, go to the sidebar and:

1. Upload your perceptual map data (see Input Format below). Once the data is uploaded,
you will be able to inspect it in the main container of the app.
2. Select your supplementary rows and columns, if any, and click `👍 Confirm`.
3. Select your model parameters, and click `🚀 Run`. The resulting map will display in
the main container. You can use the `Plot parameters` section to customize the plot.
4. Select your download parameters and click the `💾 Download map result` button."""

INPUT_FORMAT = """Example input data:

| Item | Attribute 1 | Attribute 2 | Attribute 3 | ... |
| :--- | ----------: | ----------: | ----------: | :-- |
| One  |        25.3 |        24.3 |        66.2 | ... |
| Two  |        15.4 |         6.6 |        32.9 | ... |
| ...  |         ... |         ... |         ... | ... |

Make sure the data is in the right format: 
- Names on left-most column
- All numerical columns labeled"""

FOOTER = """Made with ❤ by [nachomaiz](https://github.com/nachomaiz).
Based on [`streamlit`](https://streamlit.io/),
[`prince`](https://maxhalford.github.io/prince/)
and [`factor_analyzer`](https://factor-analyzer.readthedocs.io/en/latest/index.html).
[GitHub repo](https://github.com/nachomaiz/pmap-tool/)."""


def main() -> None:
    # app initialization
    st.set_page_config(
        page_title="Perceptual Map Tool",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": ABOUT_INFO},
    )
    st.title("💡 Perceptual Map Tool 🗺")
    set_default_state()

    app_state = get_state()

    # wireframe
    info_container = st.container()
    loaded_data_container = st.container()
    plot_container = st.container()
    plot_params_container = st.container()
    output_container = st.container()
    footer_container = st.container()

    # static elements
    if not app_state.all_completed():
        info_container.markdown(APP_INFO)
        with info_container.expander("Input format"):
            st.markdown(INPUT_FORMAT)
    else:
        with info_container.expander("App info"):
            st.markdown(APP_INFO)
            st.subheader("Input format")
            st.markdown(INPUT_FORMAT)

    footer_container.info(FOOTER)

    # sidebar
    with st.sidebar:
        st.title("Perceptual Map Setup")
        data, supp_params, model_params = sidebar.render(app_state)

    if data is None:
        return

    # loaded data
    with loaded_data_container.expander(
        "Loaded data", expanded=not app_state.model_params_set
    ):
        st.dataframe(data, use_container_width=True)

    if model_params is None or supp_params is None:
        return

    model = get_ca_model(data, model_params, supp_params)
    coords = get_plot_coords(model, data, supp_params)

    # plot parameters
    with plot_params_container.expander("Plot parameters"):
        plot_params = plot.get_plot_params(coords)

    # plot
    with plot_container:
        plot.render(coords, model.explained_variance, plot_params)

    # output
    with output_container:
        output.render(coords, plot_params)
