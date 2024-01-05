import pandas as pd

from src.params import PlotParams, maybe_invert_coords


def test_maybe_invert_coords():
    params = PlotParams("0", "1", True, True)
    coords = pd.DataFrame({"0": [1, 1], "1": [1, 1]})

    assert maybe_invert_coords(coords, params).equals(coords * -1)
