from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .mdp import BranchedFiniteHorizonMDP, FiniteHorizonMDP


RIGHT, UP, LEFT, DOWN = range(4)
ACTION_DELTAS = {
    RIGHT: (0, 1),
    UP: (-1, 0),
    LEFT: (0, -1),
    DOWN: (1, 0),
}
LEFT_OF = {RIGHT: UP, UP: LEFT, LEFT: DOWN, DOWN: RIGHT}
RIGHT_OF = {RIGHT: DOWN, DOWN: LEFT, LEFT: UP, UP: RIGHT}


@dataclass(frozen=True)
class GridWorldSpec:
    rows: int
    cols: int
    start: tuple[int, int]
    goals: tuple[tuple[int, int], ...]
    walls: tuple[tuple[int, int], ...] = ()
    hazards: tuple[tuple[int, int], ...] = ()
    slip_probability: float = 0.2
    horizon: int = 40
    step_reward: float = -0.1
    goal_reward: float = 5.0
    hazard_reward: float = -5.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("grid dimensions must be at least 2x2")
        if not 0.0 <= self.slip_probability <= 1.0:
            raise ValueError("slip_probability must lie in [0, 1]")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        cells = {self.start, *self.goals, *self.walls, *self.hazards}
        if any(not (0 <= r < self.rows and 0 <= c < self.cols) for r, c in cells):
            raise ValueError("all cells must lie inside the grid")
        if not self.goals:
            raise ValueError("at least one goal is required")
        if self.start in set(self.walls) | set(self.goals) | set(self.hazards):
            raise ValueError("start must be a nonterminal traversable cell")
        if set(self.walls) & (set(self.goals) | set(self.hazards)):
            raise ValueError("walls cannot also be terminal cells")


def state_of(spec: GridWorldSpec, cell: tuple[int, int]) -> int:
    return cell[0] * spec.cols + cell[1]


def cell_of(spec: GridWorldSpec, state: int) -> tuple[int, int]:
    return divmod(int(state), spec.cols)


def _move(
    spec: GridWorldSpec,
    cell: tuple[int, int],
    action: int,
    walls: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    candidate = (cell[0] + dr, cell[1] + dc)
    if not (0 <= candidate[0] < spec.rows and 0 <= candidate[1] < spec.cols):
        return cell
    if candidate in (set(spec.walls) if walls is None else walls):
        return cell
    return candidate


def gridworld_mdp(spec: GridWorldSpec) -> FiniteHorizonMDP:
    n_states = spec.rows * spec.cols
    next_state = np.zeros((n_states, 4, 3), dtype=np.int64)
    probability = np.zeros((n_states, 4, 3), dtype=np.float64)
    branch_reward = np.zeros((n_states, 4, 3), dtype=np.float64)
    walls = set(spec.walls)
    goals = set(spec.goals)
    hazards = set(spec.hazards)
    terminals = goals | hazards | walls

    for state in range(n_states):
        cell = cell_of(spec, state)
        if cell in terminals:
            next_state[state, :, :] = state
            probability[state, :, 0] = 1.0
            continue
        for action in range(4):
            outcomes = (
                (action, 1.0 - spec.slip_probability),
                (LEFT_OF[action], spec.slip_probability / 2.0),
                (RIGHT_OF[action], spec.slip_probability / 2.0),
            )
            successor_probability: dict[int, float] = {}
            for realized_action, outcome_probability in outcomes:
                if outcome_probability == 0.0:
                    continue
                next_cell = _move(spec, cell, realized_action, walls)
                successor = state_of(spec, next_cell)
                successor_probability[successor] = (
                    successor_probability.get(successor, 0.0) + outcome_probability
                )
            next_state[state, action, :] = state
            for branch, (successor, outcome_probability) in enumerate(
                sorted(successor_probability.items())
            ):
                next_state[state, action, branch] = successor
                probability[state, action, branch] = outcome_probability
                next_cell = cell_of(spec, successor)
                cell_reward = (
                    spec.goal_reward
                    if next_cell in goals
                    else spec.hazard_reward
                    if next_cell in hazards
                    else 0.0
                )
                branch_reward[state, action, branch] = spec.step_reward + cell_reward

    terminal_mask = np.asarray([cell_of(spec, s) in terminals for s in range(n_states)], dtype=bool)
    decision_states = tuple(int(s) for s in np.flatnonzero(~terminal_mask))
    initial = np.zeros(n_states, dtype=np.float64)
    initial[state_of(spec, spec.start)] = 1.0
    return BranchedFiniteHorizonMDP(
        next_state=next_state,
        probability=probability,
        branch_reward=branch_reward,
        initial=initial,
        horizon=spec.horizon,
        terminal=terminal_mask,
        decision_states=decision_states,
        gamma=spec.gamma,
    )


def has_path(spec: GridWorldSpec) -> bool:
    walls = set(spec.walls)
    goals = set(spec.goals)
    terminals = goals | set(spec.hazards)
    queue: deque[tuple[int, int]] = deque([spec.start])
    seen = {spec.start}
    while queue:
        cell = queue.popleft()
        if cell in goals:
            return True
        if cell in terminals:
            continue
        for action in range(4):
            neighbor = _move(spec, cell, action, walls)
            if neighbor not in walls and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


@dataclass(frozen=True)
class GridCharacteristics:
    traversable_cells: int
    reachable_cells: int
    reachable_fraction: float
    shortest_path_length: int
    shortest_path_count: int
    single_cell_bottlenecks: tuple[tuple[int, int], ...]


def characterize_gridworld(spec: GridWorldSpec) -> GridCharacteristics:
    """Return deterministic graph diagnostics for a frozen GridWorld map."""

    walls = set(spec.walls)
    goals = set(spec.goals)
    terminals = goals | set(spec.hazards)

    def distances_and_counts(
        extra_wall: tuple[int, int] | None = None,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        blocked = walls | ({extra_wall} if extra_wall is not None else set())
        if spec.start in blocked:
            return {}, {}
        distances = {spec.start: 0}
        counts = {spec.start: 1}
        queue: deque[tuple[int, int]] = deque([spec.start])
        while queue:
            cell = queue.popleft()
            if cell in terminals:
                continue
            for action in range(4):
                neighbor = _move(spec, cell, action, blocked)
                if neighbor == cell or neighbor in blocked:
                    continue
                proposed_distance = distances[cell] + 1
                if neighbor not in distances:
                    distances[neighbor] = proposed_distance
                    counts[neighbor] = counts[cell]
                    queue.append(neighbor)
                elif distances[neighbor] == proposed_distance:
                    counts[neighbor] += counts[cell]
        return distances, counts

    distances, counts = distances_and_counts()
    reachable_goals = [goal for goal in goals if goal in distances]
    if not reachable_goals:
        raise ValueError("GridWorld has no terminal-respecting path to a goal")
    shortest_length = min(distances[goal] for goal in reachable_goals)
    shortest_count = sum(
        counts[goal] for goal in reachable_goals if distances[goal] == shortest_length
    )
    traversable = spec.rows * spec.cols - len(walls)
    candidates = [
        cell for cell in distances if cell != spec.start and cell not in terminals
    ]
    bottlenecks = []
    for cell in candidates:
        removed_distances, _ = distances_and_counts(cell)
        if not any(goal in removed_distances for goal in goals):
            bottlenecks.append(cell)
    return GridCharacteristics(
        traversable_cells=traversable,
        reachable_cells=len(distances),
        reachable_fraction=len(distances) / float(traversable),
        shortest_path_length=shortest_length,
        shortest_path_count=shortest_count,
        single_cell_bottlenecks=tuple(sorted(bottlenecks)),
    )


@dataclass(frozen=True)
class GridDifficulty:
    name: str
    rows: int
    cols: int
    obstacle_density: float
    slip_probability: float
    horizon: int


GRID_DIFFICULTIES = {
    "smoke": GridDifficulty("smoke", 4, 4, 0.10, 0.10, 24),
    "easy": GridDifficulty("easy", 8, 8, 0.10, 0.10, 64),
    "medium": GridDifficulty("medium", 16, 16, 0.20, 0.20, 160),
    "hard": GridDifficulty("hard", 32, 32, 0.30, 0.25, 384),
}


def generate_gridworld(
    difficulty: GridDifficulty | str,
    seed: int,
    max_attempts: int = 1_000,
) -> GridWorldSpec:
    if isinstance(difficulty, str):
        difficulty = GRID_DIFFICULTIES[difficulty]
    rng = np.random.default_rng(seed)
    start = (difficulty.rows - 1, 0)
    goal = (0, difficulty.cols - 1)
    candidates = [
        (r, c)
        for r in range(difficulty.rows)
        for c in range(difficulty.cols)
        if (r, c) not in (start, goal)
    ]
    n_walls = int(round(difficulty.obstacle_density * difficulty.rows * difficulty.cols))
    n_walls = min(n_walls, len(candidates))
    for _ in range(max_attempts):
        indices = rng.choice(len(candidates), size=n_walls, replace=False)
        walls = tuple(sorted(candidates[int(index)] for index in indices))
        spec = GridWorldSpec(
            rows=difficulty.rows,
            cols=difficulty.cols,
            start=start,
            goals=(goal,),
            walls=walls,
            slip_probability=difficulty.slip_probability,
            horizon=difficulty.horizon,
        )
        if has_path(spec):
            return spec
    raise RuntimeError(f"failed to generate a solvable {difficulty.name} grid in {max_attempts} attempts")
