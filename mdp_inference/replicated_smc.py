from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank


def systematic_resample(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard systematic resampling with one offset in [0, 1/N)."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    n = len(probabilities)
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / float(n)
    return np.searchsorted(np.cumsum(probabilities), positions, side="right").astype(np.int64)


@dataclass(frozen=True)
class ReplicatedSMCConfig:
    num_particles: int = 256
    beta: float = 1.0
    ess_threshold_ratio: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class ReplicatedSMCResult:
    policies: np.ndarray
    empirical_returns: np.ndarray
    log_normalizer_estimate: float
    ess_history: np.ndarray
    resampled: np.ndarray
    root_ancestor_count: int
    simulator_steps: int

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        result = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for state in mdp.decision_states or ():
            counts = np.bincount(self.policies[:, state], minlength=mdp.num_actions)
            result[state] = counts / float(len(self.policies))
        return result


def _validated_probabilities(
    probabilities: np.ndarray | None,
    mdp: FiniteHorizonMDP,
    name: str,
) -> np.ndarray:
    if probabilities is None:
        return np.full((mdp.num_decisions, mdp.num_actions), 1.0 / mdp.num_actions)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (mdp.num_decisions, mdp.num_actions):
        raise ValueError(f"{name} must have shape [D, A]")
    if np.any(probabilities <= 0.0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError(f"{name} rows must be positive and sum to one")
    return probabilities


def run_replicated_policy_smc(
    mdp: FiniteHorizonMDP,
    tapes: TapeBank,
    config: ReplicatedSMCConfig,
    proposal_probabilities: np.ndarray | None = None,
    prior_probabilities: np.ndarray | None = None,
) -> ReplicatedSMCResult:
    """SMC for a fixed sample-average expected-return posterior.

    Each policy particle is evaluated on all ``K`` transition tapes. The mean
    reward across replicas is added to the particle log-weight at every time
    step, so averaging happens before exponentiation. Policy actions are
    sampled lazily once per state and shared by every replica of that particle.
    """

    if tapes.horizon != mdp.horizon:
        raise ValueError("tape horizon must match the MDP")
    if config.num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if not 0.0 < config.ess_threshold_ratio <= 1.0:
        raise ValueError("ess_threshold_ratio must lie in (0, 1]")

    proposal = _validated_probabilities(proposal_probabilities, mdp, "proposal_probabilities")
    prior = _validated_probabilities(prior_probabilities, mdp, "prior_probabilities")
    state_to_decision = mdp.state_to_decision
    rng = np.random.default_rng(config.seed)
    n = config.num_particles
    k_count = tapes.num_replicas

    policies = np.full((n, mdp.num_states), -1, dtype=np.int64)
    states = np.empty((n, k_count), dtype=np.int64)
    initial_states = np.asarray([mdp.sample_initial(u) for u in tapes.initial_u], dtype=np.int64)
    states[:] = initial_states[None, :]
    log_weights = np.zeros(n, dtype=np.float64)
    empirical_returns = np.zeros(n, dtype=np.float64)
    root_ancestors = np.arange(n, dtype=np.int64)
    ess_history: list[float] = []
    resampled_history: list[bool] = []
    log_normalizer = 0.0
    simulator_steps = 0

    for time in range(mdp.horizon):
        step_rewards = np.zeros((n, k_count), dtype=np.float64)
        for particle in range(n):
            active_states = sorted({int(s) for s in states[particle] if not mdp.terminal[int(s)]})
            for state in active_states:
                if policies[particle, state] >= 0:
                    continue
                decision = state_to_decision[state]
                action = int(rng.choice(mdp.num_actions, p=proposal[decision]))
                policies[particle, state] = action
                log_weights[particle] += np.log(prior[decision, action]) - np.log(proposal[decision, action])

            for replica in range(k_count):
                state = int(states[particle, replica])
                if mdp.terminal[state]:
                    continue
                action = int(policies[particle, state])
                next_state, reward = mdp.sample_transition(
                    state,
                    action,
                    tapes.transition_u[replica, time],
                )
                step_rewards[particle, replica] = reward
                states[particle, replica] = next_state
                simulator_steps += 1

        discounted_mean_reward = (mdp.gamma**time) * step_rewards.mean(axis=1)
        empirical_returns += discounted_mean_reward
        log_weights += config.beta * discounted_mean_reward

        maximum = float(log_weights.max())
        unnormalized = np.exp(log_weights - maximum)
        probabilities = unnormalized / unnormalized.sum()
        ess = 1.0 / float(np.sum(probabilities**2))
        ess_history.append(ess)
        do_resample = ess < config.ess_threshold_ratio * n or time == mdp.horizon - 1
        resampled_history.append(do_resample)
        if do_resample:
            log_normalizer += maximum + float(np.log(unnormalized.mean()))
            ancestors = systematic_resample(probabilities, rng)
            policies = policies[ancestors].copy()
            states = states[ancestors].copy()
            empirical_returns = empirical_returns[ancestors].copy()
            root_ancestors = root_ancestors[ancestors].copy()
            log_weights.fill(0.0)

    # Actions that were irrelevant to all K rollouts have posterior equal to
    # the prior. Completing them from the prior avoids unnecessary importance
    # weight variance while returning complete deterministic policies.
    for particle in range(n):
        for decision, state in enumerate(mdp.decision_states or ()):
            if policies[particle, state] < 0:
                policies[particle, state] = int(rng.choice(mdp.num_actions, p=prior[decision]))
    policies[:, mdp.terminal] = 0

    return ReplicatedSMCResult(
        policies=policies,
        empirical_returns=empirical_returns,
        log_normalizer_estimate=float(log_normalizer),
        ess_history=np.asarray(ess_history, dtype=np.float64),
        resampled=np.asarray(resampled_history, dtype=bool),
        root_ancestor_count=int(np.unique(root_ancestors).size),
        simulator_steps=simulator_steps,
    )
