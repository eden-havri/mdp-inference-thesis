from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _sample_cdf(probabilities: np.ndarray, u: float) -> int:
    """Sample an index using an explicit U(0, 1) variate."""

    cdf = np.cumsum(probabilities, dtype=np.float64)
    return int(np.searchsorted(cdf, min(float(u), np.nextafter(1.0, 0.0)), side="right"))


@dataclass(frozen=True)
class TapeBank:
    """Common random numbers used to evaluate policies on identical noise.

    ``initial_u[k]`` samples the initial state for replica ``k`` and
    ``transition_u[k, t]`` samples its transition at time ``t``.  A bank is
    immutable and therefore defines a deterministic sample-average objective.
    """

    initial_u: np.ndarray
    transition_u: np.ndarray

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_u, dtype=np.float64)
        transition = np.asarray(self.transition_u, dtype=np.float64)
        if initial.ndim != 1:
            raise ValueError("initial_u must have shape [K]")
        if transition.ndim != 2 or transition.shape[0] != initial.shape[0]:
            raise ValueError("transition_u must have shape [K, H]")
        if np.any((initial < 0.0) | (initial >= 1.0)):
            raise ValueError("initial_u values must lie in [0, 1)")
        if np.any((transition < 0.0) | (transition >= 1.0)):
            raise ValueError("transition_u values must lie in [0, 1)")
        object.__setattr__(self, "initial_u", initial)
        object.__setattr__(self, "transition_u", transition)

    @property
    def num_replicas(self) -> int:
        return int(self.initial_u.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.transition_u.shape[1])

    @classmethod
    def sample(cls, num_replicas: int, horizon: int, seed: int) -> "TapeBank":
        if num_replicas <= 0 or horizon <= 0:
            raise ValueError("num_replicas and horizon must be positive")
        rng = np.random.default_rng(seed)
        return cls(rng.random(num_replicas), rng.random((num_replicas, horizon)))


@dataclass(frozen=True)
class FiniteHorizonMDP:
    """Finite MDP with transition-conditioned rewards.

    ``transition[s, a, s_next]`` contains transition probabilities and
    ``reward[s, a, s_next]`` the corresponding immediate reward. Terminal
    states are treated as absorbing with zero subsequent reward. Policies are
    deterministic and stationary; actions at non-decision states are ignored.
    """

    transition: np.ndarray
    reward: np.ndarray
    initial: np.ndarray
    horizon: int
    terminal: np.ndarray | None = None
    decision_states: tuple[int, ...] | None = None
    gamma: float = 1.0

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=np.float64)
        reward = np.asarray(self.reward, dtype=np.float64)
        initial = np.asarray(self.initial, dtype=np.float64)
        if transition.ndim != 3:
            raise ValueError("transition must have shape [S, A, S]")
        n_states, n_actions, n_states_2 = transition.shape
        if n_states != n_states_2 or reward.shape != transition.shape:
            raise ValueError("reward must match square transition tensor")
        if initial.shape != (n_states,):
            raise ValueError("initial must have shape [S]")
        if n_actions < 1 or self.horizon < 1:
            raise ValueError("the MDP must have actions and a positive horizon")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if np.any(transition < -1e-12):
            raise ValueError("transition probabilities must be nonnegative")
        if not np.allclose(transition.sum(axis=2), 1.0, atol=1e-10):
            raise ValueError("each transition row must sum to one")
        if np.any(initial < -1e-12) or not np.isclose(initial.sum(), 1.0, atol=1e-10):
            raise ValueError("initial distribution must be nonnegative and sum to one")

        terminal = (
            np.zeros(n_states, dtype=bool)
            if self.terminal is None
            else np.asarray(self.terminal, dtype=bool)
        )
        if terminal.shape != (n_states,):
            raise ValueError("terminal must have shape [S]")
        decision_states = (
            tuple(int(s) for s in np.flatnonzero(~terminal))
            if self.decision_states is None
            else tuple(int(s) for s in self.decision_states)
        )
        if len(set(decision_states)) != len(decision_states):
            raise ValueError("decision_states must be unique")
        if any(s < 0 or s >= n_states or terminal[s] for s in decision_states):
            raise ValueError("decision states must be valid nonterminal states")

        # Make terminal dynamics explicitly absorbing and reward-free. This
        # prevents accidental reward accumulation after termination.
        transition = transition.copy()
        reward = reward.copy()
        for state in np.flatnonzero(terminal):
            transition[state, :, :] = 0.0
            transition[state, :, state] = 1.0
            reward[state, :, :] = 0.0

        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(self, "decision_states", decision_states)

    @property
    def num_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.transition.shape[1])

    @property
    def num_decisions(self) -> int:
        return len(self.decision_states or ())

    @property
    def state_to_decision(self) -> dict[int, int]:
        return {state: index for index, state in enumerate(self.decision_states or ())}

    @property
    def transition_model_kind(self) -> str:
        return "dense"

    @property
    def transition_storage_bytes(self) -> int:
        return int(self.transition.nbytes + self.reward.nbytes)

    def sample_initial(self, u: float) -> int:
        return _sample_cdf(self.initial, u)

    def sample_transition(self, state: int, action: int, u: float) -> tuple[int, float]:
        next_state = _sample_cdf(self.transition[state, action], u)
        return next_state, float(self.reward[state, action, next_state])

    def successor_probabilities(
        self, state: int, action: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the positive-probability successor distribution.

        Successor state IDs are strictly increasing.  The returned arrays are
        copies, so callers cannot mutate the MDP through this inspection API.
        """

        probabilities = self.transition[state, action]
        successors = np.flatnonzero(probabilities > 0.0).astype(np.int64, copy=False)
        return successors.copy(), probabilities[successors].copy()

    def action_backups(self, next_value: np.ndarray) -> np.ndarray:
        next_value = np.asarray(next_value, dtype=np.float64)
        if next_value.shape != (self.num_states,):
            raise ValueError("next_value must have shape [S]")
        backups = np.sum(
            self.transition * (self.reward + self.gamma * next_value[None, None, :]),
            axis=2,
        )
        backups[self.terminal] = 0.0
        return backups

    def policy_from_decisions(self, actions: Sequence[int]) -> np.ndarray:
        if len(actions) != self.num_decisions:
            raise ValueError(f"expected {self.num_decisions} decision actions")
        policy = np.zeros(self.num_states, dtype=np.int64)
        for state, action in zip(self.decision_states or (), actions):
            if action < 0 or action >= self.num_actions:
                raise ValueError(f"invalid action {action}")
            policy[state] = int(action)
        return policy

    def validate_policy(self, policy: Sequence[int]) -> np.ndarray:
        policy_array = np.asarray(policy, dtype=np.int64)
        if policy_array.shape != (self.num_states,):
            raise ValueError("policy must have shape [S]")
        for state in self.decision_states or ():
            action = int(policy_array[state])
            if action < 0 or action >= self.num_actions:
                raise ValueError(f"invalid action {action} at state {state}")
        return policy_array

    def expected_return(self, policy: Sequence[int]) -> float:
        policy_array = self.validate_policy(policy)
        value = np.zeros(self.num_states, dtype=np.float64)
        states = np.arange(self.num_states)
        for _ in range(self.horizon):
            backups = self.action_backups(value)
            value = backups[states, policy_array]
            value[self.terminal] = 0.0
        return float(np.dot(self.initial, value))

    def rollout_return_and_steps(
        self,
        policy: Sequence[int],
        initial_u: float,
        transition_u: Sequence[float],
    ) -> tuple[float, int]:
        policy_array = self.validate_policy(policy)
        transition_u_array = np.asarray(transition_u, dtype=np.float64)
        if transition_u_array.shape != (self.horizon,):
            raise ValueError("transition_u must have shape [H]")
        state = self.sample_initial(initial_u)
        total = 0.0
        discount = 1.0
        steps = 0
        for time in range(self.horizon):
            if self.terminal[state]:
                break
            action = int(policy_array[state])
            next_state, reward = self.sample_transition(state, action, transition_u_array[time])
            total += discount * reward
            steps += 1
            state = next_state
            discount *= self.gamma
        return float(total), steps

    def rollout_return_steps_visited(
        self,
        policy: Sequence[int],
        initial_u: float,
        transition_u: Sequence[float],
    ) -> tuple[float, int, frozenset[int]]:
        policy_array = self.validate_policy(policy)
        transition_u_array = np.asarray(transition_u, dtype=np.float64)
        if transition_u_array.shape != (self.horizon,):
            raise ValueError("transition_u must have shape [H]")
        state = self.sample_initial(initial_u)
        total = 0.0
        discount = 1.0
        steps = 0
        visited: set[int] = set()
        for time in range(self.horizon):
            if self.terminal[state]:
                break
            visited.add(state)
            action = int(policy_array[state])
            next_state, reward = self.sample_transition(state, action, transition_u_array[time])
            total += discount * reward
            steps += 1
            state = next_state
            discount *= self.gamma
        return float(total), steps, frozenset(visited)

    def rollout_return(self, policy: Sequence[int], initial_u: float, transition_u: Sequence[float]) -> float:
        return self.rollout_return_and_steps(policy, initial_u, transition_u)[0]

    def tape_returns(self, policy: Sequence[int], tapes: TapeBank) -> np.ndarray:
        if tapes.horizon != self.horizon:
            raise ValueError("tape horizon must match the MDP")
        return np.asarray(
            [
                self.rollout_return(policy, tapes.initial_u[k], tapes.transition_u[k])
                for k in range(tapes.num_replicas)
            ],
            dtype=np.float64,
        )

    def sample_rollouts(self, policy: Sequence[int], num_rollouts: int, rng: np.random.Generator) -> np.ndarray:
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        tapes = TapeBank(rng.random(num_rollouts), rng.random((num_rollouts, self.horizon)))
        return self.tape_returns(policy, tapes)

    def sample_rollouts_with_steps(
        self,
        policy: Sequence[int],
        num_rollouts: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        initial = rng.random(num_rollouts)
        transitions = rng.random((num_rollouts, self.horizon))
        results = [
            self.rollout_return_and_steps(policy, initial[k], transitions[k])
            for k in range(num_rollouts)
        ]
        return (
            np.asarray([result[0] for result in results], dtype=np.float64),
            int(sum(result[1] for result in results)),
        )

    def sample_rollouts_with_metadata(
        self,
        policy: Sequence[int],
        num_rollouts: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int, frozenset[int]]:
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        initial = rng.random(num_rollouts)
        transitions = rng.random((num_rollouts, self.horizon))
        results = [
            self.rollout_return_steps_visited(policy, initial[k], transitions[k])
            for k in range(num_rollouts)
        ]
        visited = frozenset().union(*(result[2] for result in results))
        return (
            np.asarray([result[0] for result in results], dtype=np.float64),
            int(sum(result[1] for result in results)),
            visited,
        )


class BranchedFiniteHorizonMDP(FiniteHorizonMDP):
    """Finite MDP with at most ``B`` explicit successor branches per action.

    The class intentionally implements the same public algorithmic interface
    as :class:`FiniteHorizonMDP` while avoiding dense ``[S, A, S]`` tensors.
    Zero-probability padding branches are ignored.
    """

    def __init__(
        self,
        next_state: np.ndarray,
        probability: np.ndarray,
        branch_reward: np.ndarray,
        initial: np.ndarray,
        horizon: int,
        terminal: np.ndarray | None = None,
        decision_states: tuple[int, ...] | None = None,
        gamma: float = 1.0,
    ) -> None:
        next_state_array = np.asarray(next_state, dtype=np.int64)
        probability_array = np.asarray(probability, dtype=np.float64)
        reward_array = np.asarray(branch_reward, dtype=np.float64)
        initial_array = np.asarray(initial, dtype=np.float64)
        if next_state_array.ndim != 3:
            raise ValueError("next_state must have shape [S, A, B]")
        if probability_array.shape != next_state_array.shape or reward_array.shape != next_state_array.shape:
            raise ValueError("probability and branch_reward must match next_state")
        n_states, n_actions, n_branches = next_state_array.shape
        if n_states < 1 or n_actions < 1 or n_branches < 1 or horizon < 1:
            raise ValueError("the branched MDP dimensions must be positive")
        if initial_array.shape != (n_states,):
            raise ValueError("initial must have shape [S]")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if np.any((next_state_array < 0) | (next_state_array >= n_states)):
            raise ValueError("branch successor is outside the state space")
        if np.any(probability_array < -1e-12):
            raise ValueError("branch probabilities must be nonnegative")
        if not np.allclose(probability_array.sum(axis=2), 1.0, atol=1e-10):
            raise ValueError("each branch row must sum to one")
        if np.any(initial_array < -1e-12) or not np.isclose(initial_array.sum(), 1.0, atol=1e-10):
            raise ValueError("initial distribution must be nonnegative and sum to one")

        terminal_array = (
            np.zeros(n_states, dtype=bool)
            if terminal is None
            else np.asarray(terminal, dtype=bool)
        )
        if terminal_array.shape != (n_states,):
            raise ValueError("terminal must have shape [S]")
        decision_tuple = (
            tuple(int(s) for s in np.flatnonzero(~terminal_array))
            if decision_states is None
            else tuple(int(s) for s in decision_states)
        )
        if len(set(decision_tuple)) != len(decision_tuple):
            raise ValueError("decision_states must be unique")
        if any(s < 0 or s >= n_states or terminal_array[s] for s in decision_tuple):
            raise ValueError("decision states must be valid nonterminal states")

        next_state_array = next_state_array.copy()
        probability_array = probability_array.copy()
        reward_array = reward_array.copy()
        for state in np.flatnonzero(terminal_array):
            next_state_array[state, :, :] = state
            probability_array[state, :, :] = 0.0
            probability_array[state, :, 0] = 1.0
            reward_array[state, :, :] = 0.0

        object.__setattr__(self, "next_state", next_state_array)
        object.__setattr__(self, "probability", probability_array)
        object.__setattr__(self, "branch_reward", reward_array)
        object.__setattr__(self, "initial", initial_array)
        object.__setattr__(self, "horizon", int(horizon))
        object.__setattr__(self, "terminal", terminal_array)
        object.__setattr__(self, "decision_states", decision_tuple)
        object.__setattr__(self, "gamma", float(gamma))

    @property
    def num_states(self) -> int:
        return int(self.next_state.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.next_state.shape[1])

    @property
    def num_branches(self) -> int:
        return int(self.next_state.shape[2])

    @property
    def transition_model_kind(self) -> str:
        return "fixed_branch"

    @property
    def transition_storage_bytes(self) -> int:
        return int(self.next_state.nbytes + self.probability.nbytes + self.branch_reward.nbytes)

    def sample_transition(self, state: int, action: int, u: float) -> tuple[int, float]:
        branch = _sample_cdf(self.probability[state, action], u)
        return (
            int(self.next_state[state, action, branch]),
            float(self.branch_reward[state, action, branch]),
        )

    def successor_probabilities(
        self, state: int, action: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the aggregated positive-probability successor distribution."""

        branch_probabilities = self.probability[state, action]
        positive = branch_probabilities > 0.0
        aggregated = np.zeros(self.num_states, dtype=np.float64)
        np.add.at(
            aggregated,
            self.next_state[state, action, positive],
            branch_probabilities[positive],
        )
        successors = np.flatnonzero(aggregated > 0.0).astype(np.int64, copy=False)
        return successors.copy(), aggregated[successors].copy()

    def action_backups(self, next_value: np.ndarray) -> np.ndarray:
        next_value = np.asarray(next_value, dtype=np.float64)
        if next_value.shape != (self.num_states,):
            raise ValueError("next_value must have shape [S]")
        continuation = next_value[self.next_state]
        backups = np.sum(
            self.probability * (self.branch_reward + self.gamma * continuation),
            axis=2,
        )
        backups[self.terminal] = 0.0
        return backups


def enumerate_decision_vectors(num_decisions: int, num_actions: int) -> Iterable[tuple[int, ...]]:
    """Yield deterministic decision vectors in lexicographic order."""

    if num_decisions < 0 or num_actions < 1:
        raise ValueError("invalid enumeration dimensions")
    if num_decisions == 0:
        yield ()
        return
    yield from np.ndindex(*(num_actions for _ in range(num_decisions)))
