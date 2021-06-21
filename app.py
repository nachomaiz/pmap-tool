import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import cycler

from src import pmap, utils

# Styling for the Perceptual Map plot
mplparams = {
    'figure.facecolor' : '#0E1117',
    'text.color' : '#FAFAFA',
    'axes.facecolor' : '#0E1117',
    'axes.edgecolor' : '#FAFAFA',
    'axes.labelcolor' : '#FAFAFA',
    'xtick.color' : '#FAFAFA',
    'xtick.labelcolor' : '#FAFAFA',
    'ytick.color' : '#FAFAFA',
    'ytick.labelcolor' : '#FAFAFA',
    'axes.prop_cycle' : cycler.cycler('color',plt.cm.Dark2(range(0,4))),
    'font.size' : '14.0'
}

st.set_page_config(
    page_title="Perceptual Map Tool",
    page_icon="💡"
)

# Cached data and model load for faster interactivity
@st.cache(allow_output_mutation=True)
def build_pmap_model(X, supp, n_components, n_iter):
    return pmap.PMAP(n_components, n_iter).fit(X, supp)

@st.cache()
def load_data(data):
    return pd.read_excel(data, index_col=0)

"""
# Perceptual Map Tool
You can use this tool to create perceptual maps, download the results, 
and supports supplementary data. \n
Make sure the data is in the right format: 
- Names on left-most column
- All columns labeled
- Supplementary rows at bottom of table
- Supplementary columns right of the table

Load sample data below to ensure you're using the right data format.
"""

st.sidebar.title("Perceptual Map Setup")

uploaded_file = st.sidebar.file_uploader(
    "Please upload your Excel file:",
    type=['xls','xlsx']
)

# sample data functionality which displays filename in sidebar, overrides file upload for referencing.
if sample := st.checkbox('Load sample data'):
    uploaded_file = 'data/brand data.xlsx'
    filename = uploaded_file.split('/')[-1]
    st.info('Try using one supplementary column on this dataset.')
    if filename in uploaded_file:
        filewrap = st.sidebar.empty()
    filewrap.info(f'Using sample data: {filename}')

# Main app once data is ready
if uploaded_file:

    # handles most cases of bad files.
    try:
        data = load_data(uploaded_file)
        for c in data.columns:
            if data[c].dtype not in ('float','int'):
                st.stop()
    except:
        st.error("Your file could not be loaded.")
        st.stop()
    
    n_rows, n_cols = data.shape

    # handles data being too small
    if n_rows < 3 or n_cols < 3:
        st.error("Data is too small for perceptual map. Needs at least 3 rows and columns.")
        st.stop()
    with st.beta_expander("Show Data"):
        st.dataframe(data)

    # Warning for small datasets
    if n_rows < 4 or n_cols < 4:
        st.sidebar.warning("Your data can be turned into a perceptual map, but it only has 3 {dim}, so you cannot pick {comps} supplementary {dim}."
            .format(dim="rows" if n_rows < 4 else "columns",
                    comps="components or" if n_cols < 4 else ""
                    )
                )
        
    # Model parameters
    n_iter = st.sidebar.number_input("# of iterations", 1, 100, 10)
    if n_cols > 3:
        n_components = st.sidebar.slider("Number of components (default 2)", 2, 10 if n_cols > 10 else n_cols - 1, 2, help="For performance reasons the max is 10. Allows one less than the total number of data columns")
    else: 
        n_components = 2
        st.sidebar.info("Number of components = 2")
    x_component = st.sidebar.slider("Horizontal component", 0, n_components - 1, 0)
    y_component = st.sidebar.slider("Vertical component", 0, n_components - 1, 1)

    # Chart options
    invert_x = st.sidebar.checkbox("Invert x axis", False)
    invert_y = st.sidebar.checkbox("Invert y axis", False)
    
    # Supplementary Data

    if st.sidebar.checkbox("Supplementary data", help="For plotting grouped averages, factors, etc. Only the final rows and columns can be supplementary"):
        supp = [0,0]

        # Show only if each data shape dimension is large enough
        if n_rows > 3:
            supp[0] = st.sidebar.slider("Supplementary rows", 0, n_rows - 3, 0, 1, help="Must leave at least 3 rows as core data")

        if n_cols > 3:
            supp[1] = st.sidebar.slider("Supplementary columns", 0, n_cols - 3, 1 if sample else 0, 1, help="Must leave at least 3 columns as core data") # default value is 1 if sample data is loaded
        
        # set supp to tuple or None if both supp values are zero
        supp = None if supp == [0,0] else tuple(supp)
        
        plot_supp_idx = 1 if supp is None else 0
        plot_supp = st.sidebar.selectbox("Plot supplementary data",[True,False,'only'], index=plot_supp_idx)

    else:
        supp = None
        plot_supp = False
    
    # Parse invert axis for PMAP functions
    invert_ax = 'b' if invert_x and invert_y else 'x' if invert_x else 'y' if invert_y else None

    model = build_pmap_model(data, supp, n_components, n_iter)

    # Plot Perceptual Map
    with plt.style.context(mplparams):
        fig, ax = plt.subplots(figsize=(16,9))
        model.plot_coordinates(x_component=x_component, y_component=y_component, supp=plot_supp, ax=ax, invert_ax=invert_ax)
    ax.grid(False) # force grid off

    st.pyplot(fig)

    st.write("You can download the coordinates to build your charts, or copy the image above.")

    out_data = model.get_chart_data(x_component=x_component, y_component=y_component, invert_ax=invert_ax)

    with st.beta_expander("Download Data"):
        st.markdown(utils.download_button(out_data, 'pmap-output.xlsx', 'Download output as excel'), unsafe_allow_html=True)
        st.dataframe(out_data)

st.info("Developed with ❤ by [nachomaiz](https://github.com/nachomaiz) based on the [prince](https://github.com/MaxHalford/prince) package. [GitHub repo](https://github.com/nachomaiz/pmap).")