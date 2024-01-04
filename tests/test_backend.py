from unittest import mock

from app.backend import AppState, get_state, set_default_state


def test_set():
    state = AppState()
    state.set("pmap_data_loaded", True)

    assert state.pmap_data_loaded


def test_all_completed():
    state = AppState(True, True)
    assert state.all_completed()


def test_set_default_state():
    mock_state = {}

    with mock.patch("app.backend.st.session_state", mock_state):
        set_default_state()

    assert "app_state" in mock_state


def test_set_default_state_exists():
    mock_state = {"app_state": AppState(True, True)}

    with mock.patch("app.backend.st.session_state", mock_state):
        set_default_state()

    assert mock_state["app_state"].all_completed()


def test_set_default_state_override():
    mock_state = {"app_state": AppState(True, True)}

    with mock.patch("app.backend.st.session_state", mock_state):
        set_default_state(override=True)

    assert not mock_state["app_state"].all_completed()


def test_get_state():
    mock_state = {"app_state": AppState(True, True)}

    with mock.patch("app.backend.st.session_state", mock_state):
        state = get_state()

    assert state.all_completed()
