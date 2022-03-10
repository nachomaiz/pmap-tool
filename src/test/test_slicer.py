# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=pointless-statement
# pylint: disable=protected-access

import unittest

import pandas as pd

from ..slicer import (
    core_slicer,
    get_slicer,
    row_slicer,
    column_slicer,
    supp_slice,
)

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


class TestGetSlicer(unittest.TestCase):
    def setUp(self):
        self.factory_func = get_slicer

    def test_rows_factory(self):
        self.assertIs(self.factory_func("index"), row_slicer)
        self.assertIs(self.factory_func("rows"), row_slicer)
        self.assertIs(self.factory_func(0), row_slicer)

    def test_columns_factory(self):
        self.assertIs(self.factory_func("columns"), column_slicer)
        self.assertIs(self.factory_func(1), column_slicer)

    def test_core_factory(self):
        self.assertIs(self.factory_func("core"), core_slicer)

    def test_bad_axis(self):
        with self.assertRaises(ValueError):
            self.factory_func("blah")  # purposeful ValueError


class TestRowSlicer(unittest.TestCase):
    def setUp(self) -> None:
        self.X = test_data
        self.slicer = row_slicer

    def test_no_rows(self):
        self.assertIsNone(self.slicer(self.X, pd.Index([]), pd.Index([])))

    def test_only_rows(self):
        sliced = self.slicer(self.X, pd.Index(["Medium"]), pd.Index([]))
        self.assertTrue(sliced.shape == (1, 5))

    def test_rows_cols(self):
        sliced = self.slicer(self.X, pd.Index(["Medium"]), pd.Index(["Medium"]))
        self.assertTrue(sliced.shape == (1, 4))


class TestColumnSlicer(unittest.TestCase):
    def setUp(self) -> None:
        self.X = test_data
        self.slicer = column_slicer

    def test_no_cols(self):
        self.assertIsNone(self.slicer(self.X, pd.Index([]), pd.Index([])))

    def test_only_cols(self):
        sliced = self.slicer(self.X, pd.Index([]), pd.Index(["Medium"]))
        self.assertTrue(sliced.shape == (4, 1))

    def test_rows_cols(self):
        sliced = self.slicer(self.X, pd.Index(["Medium"]), pd.Index(["Medium"]))
        self.assertTrue(sliced.shape == (3, 1))


class TestCoreSlicer(unittest.TestCase):
    def setUp(self) -> None:
        self.X = test_data
        self.slicer = core_slicer

    def test_no_supp(self):
        sliced = self.slicer(self.X, pd.Index([]), pd.Index([]))
        self.assertTrue(sliced.shape == (4, 5))
        self.assertTrue(sliced.equals(self.X))

    def test_rows_cols(self):
        sliced = self.slicer(self.X, pd.Index(["Medium"]), pd.Index(["Medium"]))
        self.assertTrue(sliced.shape == (3, 4))


class TestSuppSlice(unittest.TestCase):
    def setUp(self) -> None:
        self.X = test_data

    def test_no_supp(self):
        sliced = supp_slice(self.X, None, None, "core")
        self.assertTrue(sliced.shape == (4, 5))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
