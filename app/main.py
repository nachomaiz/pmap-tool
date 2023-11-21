import streamlit as st

from app import output, plot, sidebar
from app.backend import get_state, set_default_state
from app.model import get_ca_model, get_plot_coords

ABOUT_INFO = "pmap-tool is a free online tool to create perceptual maps."
APP_INFO = """# Perceptual Map Tool

Create perceptual maps from a table of items as rows and associations or attributes
as columns, and download the perceptual map coordinates.

Based on Correspondende Analysis
(powered by [`prince`](https://github.com/MaxHalford/prince)) 
and [`factor_analyzer`](https://github.com/EducationalTestingService/factor_analyzer)
for rotation support.

Supports supplementary rows and columns.

[More info](https://en.wikipedia.org/wiki/Perceptual_mapping) on perceptual maps."""

EXAMPLE = """Example input data:

| Item | Attribute 1 | Attribute 2 | Attribute 3 | ... |
| :--- | ----------: | ----------: | ----------: | :-- |
| One  |        25.3 |        24.3 |        66.2 | ... |
| Two  |        15.4 |         6.6 |        32.9 | ... |
| ...  |         ... |         ... |         ... | ... |

Make sure the data is in the right format: 
- Names on left-most column
- All numerical columns labeled

"""
FOOTER_MSG = (
    "Made with ❤ by nachomaiz. Based on `streamlit`, `prince` and `factor_analyzer`."
)


def main():
    st.set_page_config(
        page_title="Perceptual Map Tool",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": ABOUT_INFO},
    )
    st.title("💡 Perceptual Maps 🗺")
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
            st.markdown(EXAMPLE)
    else:
        with info_container.expander("Info"):
            st.markdown(APP_INFO)
            st.subheader("Input format")
            st.markdown(EXAMPLE)

    footer_container.info(FOOTER_MSG)

    # sidebar
    with st.sidebar:
        st.title("Perceptual Map Setup")
        data, supp_params, model_params = sidebar.render(app_state)

    if data is None:
        return

    # loaded data inspector
    with loaded_data_container.expander(
        "Loaded data", expanded=not app_state.model_params_set
    ):
        st.dataframe(data, use_container_width=True)

    if model_params is None or supp_params is None:
        return

    model = get_ca_model(data, model_params, supp_params)
    coords = get_plot_coords(model, data, supp_params)

    # Plot Parameters
    with plot_params_container.expander("Plot parameters"):
        plot_params = plot.get_plot_params(coords)

    # Plot
    with plot_container:
        plot.render(coords, plot_params)

    # Output
    with output_container:
        output.render(coords, plot_params)
