# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=pointless-statement
# pylint: disable=protected-access

import unittest
import unittest.mock

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import sklearn.exceptions

from ..pmap import NoSupplementaryDataError, Pmap

test_data = pd.DataFrame(
    data=[
        [326, 38, 241, 110, 3],
        [688, 116, 584, 188, 4],
        [343, 84, 909, 412, 26],
        [98, 48, 403, 681, 85],
    ],
    columns=pd.Series(["Fair", "Red", "Medium", "Dark", "Black"]).rename("Hair color"),
    index=pd.Series(["Blue", "Light", "Medium", "Dark"]).rename("Eye color"),
)


class TestFit(unittest.TestCase):
    def setUp(self):
        self.X = test_data
        self.pmap = Pmap()

    def test_fit(self):
        self.assertIsInstance(self.pmap.fit(self.X), Pmap)

    def test_fit_supp(self):
        self.assertIsInstance(
            self.pmap.fit(self.X, supp_rows=["Blue"], supp_cols=["Fair"]), Pmap
        )

    def test_fit_supp_ints(self):
        self.assertIsInstance(self.pmap.fit(self.X, supp_rows=1, supp_cols=1), Pmap)

    def test_bad_index(self):
        with self.assertRaises(KeyError):
            self.pmap.fit(self.X, supp_rows=["Bad"])

    def test_nones(self):
        self.assertIsNone(self.pmap.core)
        self.assertIsNone(self.pmap.supp_rows)
        self.assertIsNone(self.pmap.supp_cols)
        self.assertIsNone(self.pmap.supp_rows_slice)
        self.assertIsNone(self.pmap.supp_cols_slice)
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            self.pmap.rotation_kwargs

    def test_rotation_kwargs_exist(self):
        pmap = Pmap(rotation_kwargs={}).fit(self.X)
        self.assertEqual(pmap.rotation_kwargs, {})

    def test_rotation_kwargs_exist_equamax(self):
        pmap = Pmap(n_components=2, rotation="equamax", rotation_kwargs={}).fit(self.X)
        self.assertEqual(pmap.rotation_kwargs, {"kappa": 0.2})

    def test_not_fit(self):
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            self.pmap.result


class TestNoSupp(unittest.TestCase):
    def setUp(self):
        self.X = test_data
        self.pmap = Pmap(n_components=4).fit(self.X)

    def test_supp(self):
        self.assertIsNone(self.pmap.supp_rows)
        self.assertIsNone(self.pmap.supp_cols)

    def test_transform(self):
        self.assertTrue(self.pmap.transform(self.X).shape == (9, 4))

    def test_fit_transform(self):
        self.assertTrue(self.pmap.fit_transform(self.X).shape == (9, 4))

    def test_result_dict(self):
        for name, df in (items := self.pmap.result_dict.items()):
            self.assertTrue(name in ["Rows", "Columns"])
            self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(items) == 2)

    def test_result(self):
        self.assertTrue(self.pmap.result.equals(self.pmap.transform(self.X)))

    def test_plot_no_supp_error(self):
        with self.assertRaises(NoSupplementaryDataError):
            self.pmap.plot(supp=True)


class TestSupp(unittest.TestCase):
    def setUp(self):
        self.X = test_data
        self.pmap = Pmap()

    def test_only_rows(self):
        self.pmap.fit(self.X, supp_rows=["Medium"])
        self.assertTrue(self.pmap.core.shape == (3, 5))
        self.assertTrue(self.pmap.supp_rows_slice.shape == (1, 5))
        self.assertTrue(self.pmap.result.equals(self.pmap.transform(self.X)))

    def test_only_cols(self):
        self.pmap.fit(self.X, supp_cols=["Medium"])
        self.assertTrue(self.pmap.core.shape == (4, 4))
        self.assertTrue(self.pmap.supp_cols_slice.shape == (4, 1))
        self.assertTrue(self.pmap.result.equals(self.pmap.transform(self.X)))

    def test_rows_cols(self):
        self.pmap.fit(self.X, supp_rows=["Medium"], supp_cols=["Medium"])
        self.assertIsInstance(self.pmap.core, pd.DataFrame)
        self.assertIsInstance(self.pmap.supp_rows_slice, pd.DataFrame)
        self.assertIsInstance(self.pmap.supp_cols_slice, pd.DataFrame)
        self.assertTrue(self.pmap.result.equals(self.pmap.transform(self.X)))

    def test_result_dict(self):
        self.pmap.fit(self.X, supp_rows=["Medium"], supp_cols=["Medium"])
        for name, df in (items := self.pmap.result_dict.items()):
            self.assertTrue(name in ["Rows", "Columns", "Supp Rows", "Supp Cols"])
            self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(items) == 4)

    def test_transform_supp_only(self):
        self.pmap.fit(self.X, supp_rows=["Medium"], supp_cols=["Medium"])
        self.assertIsInstance(self.pmap.transform(self.X.loc[["Medium"]]), pd.DataFrame)
        self.assertIsInstance(
            self.pmap.transform(self.X.loc[:, ["Medium"]]), pd.DataFrame
        )


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.X = test_data

    def test_rotation_varimax(self):
        self.pmap = Pmap(rotation="varimax").fit(self.X)

        with unittest.mock.patch(
            "factor_analyzer.rotator.Rotator.fit_transform"
        ) as mock_fit_transform:
            self.pmap.row_coords(self.X)
            self.pmap.col_coords(self.X)
            mock_fit_transform.assert_called()

    def test_rotation_equamax(self):
        self.pmap = Pmap(rotation="equamax").fit(self.X)

        with unittest.mock.patch(
            "factor_analyzer.rotator.Rotator.fit_transform"
        ) as mock_fit_transform:
            self.pmap.row_coords(self.X)
            self.pmap.col_coords(self.X)
            mock_fit_transform.assert_called()

    def tearDown(self):
        self.pmap = None


def plt_axes_test(func):
    """Decorator to mock `matplotlib` `annotate`, `scatter` and `legend` functions."""

    def wrapper(self):
        with (
            unittest.mock.patch("matplotlib.pyplot.Axes.annotate") as mock_annotate,
            unittest.mock.patch("matplotlib.pyplot.Axes.scatter") as mock_scatter,
            unittest.mock.patch("matplotlib.pyplot.Axes.legend") as mock_legend,
        ):
            func(self)

            mock_annotate.assert_called()
            mock_scatter.assert_called()
            mock_legend.assert_called()

    return wrapper


class TestPlotEigenvalues(unittest.TestCase):
    def setUp(self):
        self.X = test_data
        self.pmap = Pmap(n_components=3).fit(self.X, supp_rows=["Medium"])

    @unittest.mock.patch("matplotlib.pyplot")
    def test_plot_eigenvalues(self, mock_pyplot):
        with unittest.mock.patch("matplotlib.pyplot.Axes") as mock_Axes:
            self.assertIsInstance(self.pmap.plot_eigenvalues()[0], Axes)

    def tearDown(self) -> None:
        plt.close("all")


class TestPlotMap(unittest.TestCase):
    def setUp(self):
        self.X = test_data
        self.pmap = Pmap().fit(self.X, supp_rows=["Medium"])

    @plt_axes_test
    def test_plot_stored(self):
        self.pmap.plot_map()

    @plt_axes_test
    def test_plot_new(self):
        self.pmap.plot_map(self.X)

    @plt_axes_test
    def test_plot_supp(self):
        self.pmap.plot_map(supp=True)

    @plt_axes_test
    def test_plot_scatter(self):
        self.pmap.plot_map(only_labels=False)

    @plt_axes_test
    def test_plot_invert(self):
        self.pmap.plot_map(invert_ax="y")

    @plt_axes_test
    def test_plot_generic(self):
        self.pmap.plot(supp=True)

    def test_plot_bad_show_labels(self):
        with self.assertRaises(ValueError):
            self.pmap.plot(supp=True, show_labels=[True])

    def test_plot_bad_supp(self):
        with self.assertRaises(ValueError):
            self.pmap.plot(supp=1)  # Intentional wrong type

    def tearDown(self) -> None:
        plt.close("all")


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
