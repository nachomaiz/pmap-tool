# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=pointless-statement
# pylint: disable=protected-access

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import (
    clean_pct,
    invert_plot_axis,
    isin_index,
)

test_data = pd.DataFrame(
    data=[
        [326, 38, 241, 110, 3],
        [688, 116, 584, 188, 4],
        [343, 84, 909, 412, 26],
        [98, 48, 403, 681, 85],
    ],
    columns=pd.Series(["Fair", "Red", "Medium", "Dark", "Black_pct"]).rename(
        "Hair color"
    ),
    index=pd.Series(["Blue", "Light", "Medium", "Dark"]).rename("Eye color"),
)


class TestCleanPct(unittest.TestCase):
    def test_caps(self):
        test_str = "Test_PCT"
        self.assertFalse("PCT" in clean_pct(test_str))
        self.assertFalse("_" in clean_pct(test_str))

    def test_no_caps(self):
        test_str = "Test_pct"
        self.assertFalse("pct" in clean_pct(test_str))


class TestInvertPlotAxis(unittest.TestCase):
    def test_invert_axis(self):
        f, ax = plt.subplots()
        with self.assertRaises(ValueError):
            invert_plot_axis(ax, "blah")
        for i in ["x", "y", "b", "both"]:
            invert_plot_axis(ax, i)

    def tearDown(self):
        plt.close("all")


class TestIsinIndex(unittest.TestCase):
    def test_bad_axis(self):
        i = pd.Index([])
        with self.assertRaises(ValueError):
            isin_index(i, i, 1)  # intentional value error

    def test_missing_value(self):
        i = pd.Index(["B", "C"])
        o = pd.Index(["A", "B", "C"])
        with self.assertRaises(KeyError) as e:
            isin_index(i, o, "index")
        self.assertTrue("Index(['A']" in str(e.exception))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
