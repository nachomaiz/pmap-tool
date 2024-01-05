from dataclasses import dataclass
from enum import StrEnum, auto

import pandas as pd


class Rotation(StrEnum):
    VARIMAX = auto()
    OBLIMAX = auto()
    QUARTIMAX = auto()
    EQUAMAX = auto()
    GEOMIN_ORT = auto()
    OBLIMIN = auto()
    QUARTIMIN = auto()
    GEOMIN_OBL = auto()


@dataclass(slots=True)
class ModelParams:
    n_iter: int
    n_components: int
    rotation: Rotation | None


@dataclass(slots=True)
class SuppParams:
    rows: list[str]
    cols: list[str]


@dataclass(slots=True)
class PlotParams:
    x_component: str
    y_component: str
    invert_x: bool
    invert_y: bool


def maybe_invert_coords(coords: pd.DataFrame, plot_params: PlotParams) -> pd.DataFrame:
    if plot_params.invert_x:
        coords = coords.assign(
            **{plot_params.x_component: coords[plot_params.x_component] * -1}
        )

    if plot_params.invert_y:
        coords = coords.assign(
            **{plot_params.y_component: coords[plot_params.y_component] * -1}
        )

    return coords
