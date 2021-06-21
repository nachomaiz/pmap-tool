import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import cycler

from src import pmap, utils

# Styling for the Perceptual Map plot
mplparams = {
    'figure.facecolor' : '#0E1117',
    'text.color' : 'white',
    'axes.facecolor' : '#0E1117',
    'axes.edgecolor' : 'white',
    'axes.labelcolor' : 'white',
    'axes.grid' : False,
    'xtick.color' : 'white',
    'xtick.labelcolor' : 'white',
    'ytick.color' : 'white',
    'ytick.labelcolor' : 'white',
    'axes.prop_cycle' : cycler.cycler('color',plt.cm.Dark2(range(0,5))),
    'font.size' : '14.0'
}

"""
# Create your own Perceptual Map
You can use this tool to create perceptual maps, download the results, 
and supports supplementary data.
"""
with st.beta_expander("Learn more about Perceptual Maps"):
    st.write("Coming soon")

"""
Make sure the data is in the right format: 
- Names on left-most column
- All columns labeled
- Supplementary rows at bottom of table
- Supplementary columns right of the table

Load sample data below to ensure you're using the right format.
"""

st.sidebar.title("Perceptual Map Setup")

uploaded_file = st.sidebar.file_uploader(
    "Please upload your Excel file:",
    type=['xls','xlsx']
)

# sample data functionality which displays filename in sidebar
if sample := st.checkbox('Load sample data'):
    uploaded_file = 'data/brand data.xlsx'
    filename = uploaded_file.split('/')[-1]
    st.info('Try using one supplementary column on this dataset.')
    if filename in uploaded_file:
        filewrap = st.sidebar.empty()
    filewrap.info(f'Using sample data: {filename}')

# Main app once data is ready
if uploaded_file:
    data = pd.read_excel(uploaded_file, index_col=0)
    n_cols = data.shape[1]
    with st.beta_expander("Show Data"):
        st.dataframe(data)
        
    # Model parameters
    n_components = st.sidebar.slider("Number of components (default 2)", 2, 10 if n_cols >= 10 else n_cols - 1, 2, help="For performance reasons the max is 10. Allows one less than the total number of data columns")
    x_component = st.sidebar.slider("Horizontal component", 0, n_components - 1, 0)
    y_component = st.sidebar.slider("Vertical component", 0, n_components - 1, 1)
    
    # Supplementary Data
    if st.sidebar.checkbox("Supplementary data", help="For plotting grouped averages, factors, etc. Only the final rows and columns can be supplementary"):
        supp = (
            st.sidebar.slider("Supplementary rows", 0, data.shape[0] - 3, 0, 1, help="Must leave at least 3 rows as core data"),
            st.sidebar.slider("Supplementary columns", 0, n_cols - 3, 1 if sample else 0, 1, help="Must leave at least 3 columns as core data") # default value is 1 if sample data is loaded
        )
        plot_supp = st.sidebar.selectbox("Plot supplementary data",[True,False,'only'])
    else:
        supp = None
        plot_supp = False
    
    # Advanced options
    if st.sidebar.checkbox("Advanced Options", help="Additional options that generally don't need to change"):
        n_iter = st.sidebar.number_input("# of iterations", 1, 100, 10)
        invert_x = st.sidebar.checkbox("Invert x axis", False)
        invert_y = st.sidebar.checkbox("Invert y axis", False)
    else: 
        n_iter = 10
        invert_x = False
        invert_y = False
    
    # Parse invert axis for PMAP functions
    invert_ax = 'b' if invert_x and invert_y else 'x' if invert_x else 'y' if invert_y else None

    # Initialize and fit PMAP
    @st.cache(allow_output_mutation=True)
    def build_pmap_model(n_components, n_iter):
        return pmap.PMAP(n_components=n_components, n_iter=n_iter)

    model = build_pmap_model(n_components, n_iter)
    model = model.fit(data, supp=supp)

    # Plot Perceptual Map
    with plt.style.context(mplparams):
        fig, ax = plt.subplots(figsize=(16,9))
        model.plot_coordinates(x_component=x_component, y_component=y_component, supp=plot_supp, ax=ax, invert_ax=invert_ax)
    ax.grid(False) # force grid off

    st.pyplot(fig)

    st.write("You can download the coordinates to build your charts, or copy the image below.")

    out_data = model.get_chart_data(x_component=x_component, y_component=y_component, invert_ax=invert_ax)

    with st.beta_expander("Download Data"):
        st.markdown(utils.download_button(out_data, 'pmap-output.xlsx', 'Download output as excel'), unsafe_allow_html=True)
        st.dataframe(out_data)

st.info("Developed with ❤ by [nachomaiz](https://github.com/nachomaiz) based on the [prince](https://github.com/MaxHalford/prince) package")