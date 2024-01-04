from dataclasses import dataclass

import streamlit as st


@dataclass(slots=True)
class AppState:
    pmap_data_loaded: bool = False
    model_params_set: bool = False

    def set(self, state_name: str, value: bool) -> None:
        setattr(self, state_name, value)

    def all_completed(self) -> bool:
        return self.pmap_data_loaded and self.model_params_set


def set_default_state(override: bool = False) -> None:
    default_state = {"app_state": AppState()}

    for key, value in default_state.items():
        if key not in st.session_state or override:
            st.session_state[key] = value


def get_state() -> AppState:
    return st.session_state["app_state"]
