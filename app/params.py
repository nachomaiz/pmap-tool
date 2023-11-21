from dataclasses import dataclass
from enum import StrEnum, auto


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
