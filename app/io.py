import functools
import io
import json
import pickle
from typing import Callable

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

READERS: dict[str, Callable[..., pd.DataFrame]] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": pd.read_excel,
    "text/csv": pd.read_csv,
}
DL_MIME_TYPE: dict[str, str] = {
    "json": "application/json",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@st.cache_data
def load_pandas(file: UploadedFile) -> pd.DataFrame:
    return READERS[file.type](file, index_col=0)


def load_data(label: str, key: str | None = None) -> pd.DataFrame | None:
    file = st.file_uploader(
        label, type=["xlsx", "csv"], key=key, accept_multiple_files=False
    )

    return load_pandas(file) if file is not None else None


def serialize(obj: object, pickle_it: bool = False) -> bytes:
    write_buf = io.BytesIO()
    if pickle_it:
        pickle.dump(obj, write_buf)
    elif isinstance(obj, pd.DataFrame):
        obj.to_excel(write_buf)
    else:
        json.dump(obj, io.TextIOWrapper(write_buf))

    write_buf.seek(0)
    return write_buf.read()


pickle_serialize = functools.partial(serialize, pickle_it=True)


def download_button(
    label: str,
    data: object,
    file_name: str,
    pickle_it: bool = False,
    **kwargs,
) -> None:
    label_slug = (
        label.strip()
        .replace(" ", "_")
        .lower()
        .encode("latin-1", "ignore")
        .decode("latin-1")
        .strip("_")
    )
    st.download_button(
        label,
        serialize(data, pickle_it=pickle_it),
        file_name,
        DL_MIME_TYPE[file_name.rpartition(".")[2]],
        key=f"button_{label_slug}_{file_name}",
        **kwargs,
    )
