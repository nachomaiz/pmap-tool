from typing import Literal

import io
import base64
import json
import pickle
import uuid
import re

import pandas as pd
import matplotlib.pyplot as plt


### PANDAS INDEX FUNCTIONS ###


def clean_pct(index: pd.Index, keep_tag: bool = False) -> pd.Index:
    """Clean pct tags from pandas Index"""
    index = index.str.replace(r"_", " ")
    if not keep_tag:
        index = index.str.replace(r" PCT", "")
        index = index.str.replace(r" pct", "")
    return index


def isin_index(
    index: pd.Index, other: pd.Index, axis: Literal["index", "columns"] = "index"
) -> None:
    """Check if all members of a pandas Index is in another,
    raising a KeyError with any missing values."""

    if axis not in ["index", "columns"]:
        raise ValueError("axis must be 'index' or 'columns'.")

    if any(key := other[~other.isin(index)]):
        raise KeyError(f"None of {key} are in the {axis}")


def invert_plot_axis(ax: plt.Axes, axis: Literal["x", "y", "b", "both"]) -> plt.Axes:
    """Invert Axes axis direction."""
    if axis == "x":
        ax.invert_xaxis()
    elif axis == "y":
        ax.invert_yaxis()
    elif axis == "b" or axis == "both":
        ax.invert_xaxis()
        ax.invert_yaxis()
    else:
        raise ValueError("invert_ax must be 'x', 'y' or 'both'('b').")

    return ax

# Download button for streamlit app
def download_button(object_to_download, download_filename : str, button_text : str, pickle_it : bool = False) -> str:
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError:
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            #object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: #0E1117;
                color: rgb(255, 255, 255);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link
