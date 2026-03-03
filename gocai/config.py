from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


class GridworldInstance(BaseModel):
    """JSON-serializable Gridworld instance.

    Matches the Go struct fields used by gw.go:
      - Grid: 2D list of ints [width][height]
      - PSucc: action success probability
      - RScale: reward scaling
      - S0: start state as continuous (x,y)
      - Xmin, Xmax, Dx, Ymin, Ymax, Dy: geometry
    """
    Grid: List[List[int]]
    PSucc: float
    RScale: float = 1.0

    S0: List[float] = Field(default_factory=lambda: [-1.0, -1.0])
    Xmin: float = -1.0
    Xmax: float = 1.0
    Dx: float = 1.0
    Ymin: float = -1.0
    Ymax: float = 1.0
    Dy: float = 1.0

    @classmethod
    def load(cls, path: Path) -> "GridworldInstance":
        return cls.model_validate_json(path.read_text())


@dataclass(frozen=True)
class RunConfig:
    # Optimization
    niter: int = 100_000
    nplateau: int = 0
    nwarmup: int = 1000
    rate: float = 2e-5
    minrate: float = 0.0
    decay: float = 0.0
    cosinelr: bool = False

    # SMC / evaluation
    nsteps: int = 20
    msteps: int = 0
    nparticles: int = 10
    mparticles: int = 10_000
    mbins: int = 0

    # Annealing / resampling
    temp: float = 1.0
    annealing: float = 1.0
    miness: float = 0.2
    no_deterministic_policy: bool = False
    no_shared_dynamics: bool = False

    # Policy net
    nlayers: int = 2
    lrwidth: int = 64

    # Domain overrides
    rscale: float = 1.0
    psucc: float = 0.0
    startloc: tuple[int, int] | None = None  # (i,j) in grid coords, 0-based

    # Output
    imgpath: Path = Path(".")
    savegrid: bool = False
    showguide: bool = True
