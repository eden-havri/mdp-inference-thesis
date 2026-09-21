from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank


@dataclass(frozen=True)
class PolicyMHConfig:
    beta: float = 1.0
    iterations: int = 10_000
    burn_in: int = 2_000
    thinning: int = 10
    seed: int = 0


@dataclass(frozen=True)
class PolicyMHResult:
    policies: np.ndarray
    estimated_returns: np.ndarray
    accepted_proposals: int
    changed_return_proposals: int
    value_evaluations: int
    simulator_steps: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_proposals / float(max(self.value_evaluations - 1, 1))

    @property
    def return_effective_sample_size(self) -> float:
        return autocorrelation_effective_sample_size(self.estimated_returns)

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        marginals = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for state in mdp.decision_states or ():
            marginals[state] = np.bincount(
                self.policies[:, state], minlength=mdp.num_actions
            ) / float(len(self.policies))
        return marginals


def autocorrelation_effective_sample_size(values: np.ndarray) -> float:
    """Conservative single-chain ESS using the initial positive sequence."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty vector")
    if len(values) == 1 or np.var(values) == 0.0:
        return float(len(values))
    centered = values - values.mean()
    variance = float(np.dot(centered, centered) / len(values))
    correlation_sum = 0.0
    for lag in range(1, len(values)):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (len(values) - lag))
        correlation = covariance / variance
        if correlation <= 0.0:
            break
        correlation_sum += correlation
    return float(max(1.0, min(len(values), len(values) / (1.0 + 2.0 * correlation_sum))))


def run_single_site_policy_mh(
    mdp: FiniteHorizonMDP,
    initial_policy: np.ndarray,
    config: PolicyMHConfig,
    tapes: TapeBank | None = None,
) -> PolicyMHResult:
    """Single-site Metropolis-Hastings for the policy Gibbs posterior.

    The proposal chooses a decision state uniformly and then a different action
    uniformly, so it is symmetric and the uniform policy reference cancels.
    With exact model returns this targets the intended posterior exactly.  With
    a fixed tape bank it targets the corresponding sample-average posterior.
    """

    if config.beta <= 0.0:
        raise ValueError("beta must be positive")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.burn_in < 0 or config.burn_in >= config.iterations:
        raise ValueError("burn_in must lie in [0, iterations)")
    if config.thinning <= 0:
        raise ValueError("thinning must be positive")
    if mdp.num_actions < 2 or mdp.num_decisions == 0:
        raise ValueError("policy MH requires at least one decision and two actions")
    if tapes is not None and tapes.horizon != mdp.horizon:
        raise ValueError("tape horizon must match the MDP")

    rng = np.random.default_rng(config.seed)
    policy = mdp.validate_policy(initial_policy).copy()
    policy[mdp.terminal] = 0
    simulator_steps = 0
    value_evaluations = 0

    def estimate(candidate: np.ndarray) -> float:
        nonlocal simulator_steps, value_evaluations
        value_evaluations += 1
        if tapes is None:
            return mdp.expected_return(candidate)
        values = []
        for replica in range(tapes.num_replicas):
            value, steps = mdp.rollout_return_and_steps(
                candidate,
                tapes.initial_u[replica],
                tapes.transition_u[replica],
            )
            values.append(value)
            simulator_steps += steps
        return float(np.mean(values))

    current_return = estimate(policy)
    accepted = 0
    changed_return = 0
    samples: list[np.ndarray] = []
    sample_returns: list[float] = []
    decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
    for iteration in range(config.iterations):
        state = int(decision_states[int(rng.integers(len(decision_states)))])
        old_action = int(policy[state])
        proposed_action = int(rng.integers(mdp.num_actions - 1))
        if proposed_action >= old_action:
            proposed_action += 1
        proposal = policy.copy()
        proposal[state] = proposed_action
        proposed_return = estimate(proposal)
        if not np.isclose(proposed_return, current_return, atol=1e-15, rtol=0.0):
            changed_return += 1
        log_acceptance = config.beta * (proposed_return - current_return)
        if np.log(rng.random()) < min(0.0, log_acceptance):
            policy = proposal
            current_return = proposed_return
            accepted += 1
        if iteration >= config.burn_in and (iteration - config.burn_in) % config.thinning == 0:
            samples.append(policy.copy())
            sample_returns.append(current_return)

    return PolicyMHResult(
        policies=np.stack(samples, axis=0),
        estimated_returns=np.asarray(sample_returns, dtype=np.float64),
        accepted_proposals=accepted,
        changed_return_proposals=changed_return,
        value_evaluations=value_evaluations,
        simulator_steps=simulator_steps,
    )
