from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Cell kinds (match Go)
PAVEMENT = 0
MEADOW = 1
DIRT = 2
WALL = 3
SWAMP = 4
GOAL = 5

# Actions (match Go)
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
NACTIONS = 4

R_CELL: Dict[int, float] = {
    PAVEMENT: 0.0,
    MEADOW: 0.5,
    DIRT: -0.5,
    WALL: -1.0,
    SWAMP: -10.0,
    GOAL: 5.0,
}
R_ACTION: Dict[int, float] = {RIGHT: -0.1, UP: -0.1, LEFT: -0.1, DOWN: -0.1}


@dataclass
class Gridworld:
    grid: List[List[int]]  # [width][height]
    psucc: float
    rscale: float

    # derived geometry (continuous state is (x,y) in [-1,1]^2 centered on cells)
    s0: Tuple[float, float]
    xmin: float
    xmax: float
    dx: float
    ymin: float
    ymax: float
    dy: float

    @classmethod
    def new_builtin(cls, width: int, height: int, landscape: Dict[int, List[Tuple[int, int]]], psucc: float,
                    rscale: float) -> "Gridworld":
        grid = [[PAVEMENT for _ in range(height)] for __ in range(width)]
        for kind, locs in landscape.items():
            for (i, j) in locs:
                grid[i][j] = kind

        dx = 2.0 / float(width)
        dy = 2.0 / float(height)
        xmin = -1.0 + dx / 2.0
        ymin = -1.0 + dy / 2.0
        xmax = xmin + float(width - 1) * dx
        ymax = ymin + float(height - 1) * dy
        s0 = (xmin, ymin)
        return cls(grid=grid, psucc=psucc, rscale=rscale, s0=s0, xmin=xmin, xmax=xmax, dx=dx, ymin=ymin, ymax=ymax,
                   dy=dy)

    @property
    def width(self) -> int:
        return len(self.grid)

    @property
    def height(self) -> int:
        return len(self.grid[0])

    def state_to_loc(self, state: Tuple[float, float]) -> Tuple[int, int]:
        x, y = state
        i = int(round((x - self.xmin) / self.dx))
        j = int(round((y - self.ymin) / self.dy))
        return i, j

    def loc_to_state(self, i: int, j: int) -> Tuple[float, float]:
        x = self.xmin + float(i) * self.dx
        y = self.ymin + float(j) * self.dy
        return (x, y)

    def skey(self, state: Tuple[float, float]) -> int:
        i, j = self.state_to_loc(state)
        return i * self.height + j

    def reward(self, state: Tuple[float, float], action: int, state2: Tuple[float, float]) -> float:
        i, j = self.state_to_loc(state)
        if self.grid[i][j] in (GOAL, SWAMP):
            return 0.0
        i2, j2 = self.state_to_loc(state2)
        r = R_ACTION[action] + R_CELL[self.grid[i2][j2]]
        return r * self.rscale

    def step(self, state: Tuple[float, float], action: int) -> Tuple[float, float]:
        # mirrors Go's stepU with fresh random u,v
        u = random.random()
        v = random.random()
        x, y = state
        i, j = self.state_to_loc(state)
        if self.grid[i][j] in (GOAL, SWAMP):
            return (x, y)  # absorbing

        if self.psucc < u:
            # slip left/right relative to intended direction
            if 0.5 < v:
                action = {RIGHT: UP, UP: LEFT, LEFT: DOWN, DOWN: RIGHT}[action]
            else:
                action = {RIGHT: DOWN, UP: RIGHT, LEFT: UP, DOWN: LEFT}[action]

        if action == RIGHT:
            x += self.dx
        elif action == UP:
            y += self.dy
        elif action == LEFT:
            x -= self.dx
        elif action == DOWN:
            y -= self.dy
        else:
            raise ValueError(f"illegal action {action}")

        x = min(max(x, self.xmin), self.xmax)
        y = min(max(y, self.ymin), self.ymax)

        # block walls (cannot enter WALL cells)
        i2, j2 = self.state_to_loc((x, y))
        if self.grid[i2][j2] == WALL:
            return state

        return (x, y)
