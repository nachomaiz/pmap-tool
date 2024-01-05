import warnings
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.model import Model, get_ca_model, get_plot_coords
from src.params import ModelParams, Rotation, SuppParams


def test_model_build():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    model = Model.build(params, 10, supp_params)

    assert isinstance(model, Model)
    assert model.ca.n_components == 2
    assert model.ca.n_iter == 10

    assert model.rotator is None


def test_model_build_with_rotation():
    params = ModelParams(10, 2, Rotation.EQUAMAX)
    supp_params = SuppParams([], [])
    model = Model.build(params, 10, supp_params)

    assert isinstance(model, Model)
    assert model.ca.n_components == 2
    assert model.ca.n_iter == 10

    assert model.rotator
    assert model.rotator.kappa == (2 // (2 * 10))


def test_model_fit():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams(["A"], ["a"])
    model = Model.build(params, 3, supp_params)
    df = pd.DataFrame(
        {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3], "c": [0, 1, 2, 3]},
        index=["A", "B", "C", "D"],
    )

    with mock.patch("app.model.CA.fit", return_value=None) as mock_fit:
        model.fit(df)

    assert_frame_equal(mock_fit.call_args[0][0], df.drop(index=["A"], columns=["a"]))


def test_model_fit_rotation():
    params = ModelParams(10, 2, Rotation.EQUAMAX)
    supp_params = SuppParams([], [])
    model = Model.build(params, 10, supp_params)
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )

    with mock.patch(
        "app.model.TransformRotator.fit", return_value=None
    ) as mock_rotator_fit:
        model.fit(df)

    col_coords = model.ca.column_coordinates(df)

    assert_frame_equal(mock_rotator_fit.call_args[0][0], col_coords)


def test_model_row_coordinates():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)

    with mock.patch(
        "app.model.CA.row_coordinates", return_value=None
    ) as mock_row_coordinates:
        model.row_coordinates(df)

    mock_row_coordinates.assert_called_once_with(df)


def test_model_row_coordinates_rotation():
    params = ModelParams(10, 2, Rotation.EQUAMAX)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)

    with mock.patch(
        "app.model.TransformRotator.transform", return_value=None
    ) as mock_transform:
        model.row_coordinates(df)

    assert_frame_equal(mock_transform.call_args[0][0], model.ca.row_coordinates(df))


def test_model_column_coordinates():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)

    with mock.patch(
        "app.model.CA.column_coordinates", return_value=None
    ) as mock_column_coordinates:
        model.column_coordinates(df)

    mock_column_coordinates.assert_called_once_with(df)


def test_model_column_coordinates_rotation():
    params = ModelParams(10, 2, Rotation.EQUAMAX)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)

    with mock.patch(
        "app.model.TransformRotator.transform", return_value=None
    ) as mock_transform:
        model.column_coordinates(df)

    assert_frame_equal(mock_transform.call_args[0][0], model.ca.column_coordinates(df))


def test_explained_variance():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)

    assert_series_equal(
        model.explained_variance, model.ca._eigenvalues_summary["% of variance"] / 100
    )


def test_get_ca_model():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    with warnings.catch_warnings():
        model = get_ca_model(df, params, supp_params)

    assert model.supp == supp_params
    assert model.ca.n_components == 2
    assert model.ca.n_iter == 10
    assert not model.rotator


def test_get_plot_coords():
    params = ModelParams(10, 2, None)
    supp_params = SuppParams([], [])
    df = pd.DataFrame(
        {"a": [0, 1, 1, 3, 4], "b": [1, 1, 2, 1, 4], "c": [3, 1, 2, 3, 3]},
        index=["A", "B", "C", "D", "E"],
    )
    model = Model.build(params, 3, supp_params).fit(df)
    coords = get_plot_coords(model, df, supp_params)

    assert coords.shape == (8, 2)
