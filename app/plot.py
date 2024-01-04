import altair as alt
import pandas as pd
import streamlit as st

from app.params import PlotParams, maybe_invert_coords

COLOR_PALETTE: dict[str, str] = {
    "Rows": "#FF4B4B",
    "Columns": "#337CA0",
    "Supplementary rows": "#F9AB55",
    "Supplementary columns": "#91C499",
}


@st.cache_data
def plot_map(
    coords: pd.DataFrame, explained_variance: pd.Series, plot_params: PlotParams
) -> alt.Chart:
    x_component, y_component = plot_params.x_component, plot_params.y_component
    coords = maybe_invert_coords(coords, plot_params)

    x_domain = (
        coords[x_component].min() * 1.05,
        coords[x_component].max() * 1.05,
    )
    y_domain = (
        coords[y_component].min() * 1.05,
        coords[y_component].max() * 1.05,
    )

    color_params = {"domain": [], "range": []}
    for group in coords.index.get_level_values("group").unique():
        color_params["domain"].append(group)
        color_params["range"].append(COLOR_PALETTE[group])

    color_scale = alt.Scale(**color_params)
    color = alt.Color("group:N", scale=color_scale)

    x_var = explained_variance.loc[int(x_component)]
    y_var = explained_variance.loc[int(y_component)]

    x_title = f"Component {x_component} ({x_var:.2%} of variance)"
    y_title = f"Component {y_component} ({y_var:.2%} of variance)"

    plot = (
        alt.Chart(coords.reset_index())
        .mark_text(align="center", baseline="middle", fontWeight="bold")
        .encode(
            x=alt.X(
                f"{x_component}:Q", scale=alt.Scale(domain=x_domain), title=x_title
            ),
            y=alt.Y(
                f"{y_component}:Q", scale=alt.Scale(domain=y_domain), title=y_title
            ),
            text="item",
            color=color,
            tooltip=[
                "item",
                "group",
                alt.Tooltip(f"{x_component}:Q", format=".4f"),
                alt.Tooltip(f"{y_component}:Q", format=".4f"),
            ],
        )
        .configure_axis(grid=False)
        .properties(
            height=800, title=f"Perceptual Map ({x_var + y_var:.2%} variance explained)"
        )
        .configure_title(anchor="middle")
        .interactive()
    )

    return plot


def render(
    coords: pd.DataFrame, explained_variance: pd.Series, plot_params: PlotParams
) -> None:
    st.subheader("Perceptual Map")
    st.altair_chart(
        plot_map(coords, explained_variance, plot_params), use_container_width=True
    )


def get_plot_params(plot_coords: pd.DataFrame) -> PlotParams:
    col1, col2 = st.columns(2)
    max_component = plot_coords.shape[1] - 1
    x_component = str(col1.slider("X component", 0, max_component, 0, 1))
    y_component = str(col2.slider("Y component", 0, max_component, 1, 1))
    invert_x = col1.toggle("Invert X")
    invert_y = col2.toggle("Invert Y")

    return PlotParams(x_component, y_component, invert_x, invert_y)
