from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank
from .policy_mcmc import autocorrelation_effective_sample_size


@dataclass(frozen=True)
class PolicyTemperingConfig:
    """Replica-exchange MCMC configuration for a policy Gibbs target."""

    beta: float = 1.0
    num_temperatures: int = 8
    ladder_power: float = 2.0
    iterations: int = 10_000
    burn_in: int = 2_000
    thinning: int = 10
    swap_interval: int = 1
    guide_strength: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class PolicyTemperingResult:
    policies: np.ndarray
    estimated_returns: np.ndarray
    betas: np.ndarray
    local_proposals: np.ndarray
    local_accepts: np.ndarray
    swap_proposals: np.ndarray
    swap_accepts: np.ndarray
    value_evaluations: int
    simulator_steps: int

    @property
    def return_effective_sample_size(self) -> float:
        return autocorrelation_effective_sample_size(self.estimated_returns)

    @property
    def local_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.local_accepts,
            self.local_proposals,
            out=np.zeros_like(self.local_accepts, dtype=np.float64),
            where=self.local_proposals > 0,
        )

    @property
    def swap_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.swap_accepts,
            self.swap_proposals,
            out=np.zeros_like(self.swap_accepts, dtype=np.float64),
            where=self.swap_proposals > 0,
        )

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        marginals = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for state in mdp.decision_states or ():
            marginals[state] = np.bincount(
                self.policies[:, state], minlength=mdp.num_actions
            ) / float(len(self.policies))
        return marginals


def power_beta_ladder(beta: float, num_temperatures: int, power: float) -> np.ndarray:
    """Return an increasing ladder from the policy prior (0) to ``beta``."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if num_temperatures < 2:
        raise ValueError("num_temperatures must be at least two")
    if power <= 0.0:
        raise ValueError("ladder_power must be positive")
    fractions = np.linspace(0.0, 1.0, num_temperatures, dtype=np.float64)
    ladder = beta * fractions**power
    ladder[0] = 0.0
    ladder[-1] = beta
    return ladder


def _validate_guide_probabilities(
    guide_probabilities: np.ndarray | None,
    mdp: FiniteHorizonMDP,
    guide_strength: float,
) -> np.ndarray:
    if not 0.0 <= guide_strength < 1.0:
        raise ValueError("guide_strength must lie in [0, 1)")
    uniform = np.full(
        (mdp.num_decisions, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )
    if guide_probabilities is None or guide_strength == 0.0:
        return uniform
    guide = np.asarray(guide_probabilities, dtype=np.float64)
    if guide.shape == (mdp.num_states, mdp.num_actions):
        guide = guide[np.asarray(mdp.decision_states, dtype=np.int64)]
    if guide.shape != (mdp.num_decisions, mdp.num_actions):
        raise ValueError("guide_probabilities must have shape [D, A] or [S, A]")
    if np.any(guide < 0.0) or not np.all(np.isfinite(guide)):
        raise ValueError("guide probabilities must be finite and nonnegative")
    if not np.allclose(guide.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("guide probability rows must sum to one")
    # The uniform component gives every local move positive probability.  The
    # guide changes efficiency only; the Hastings ratio preserves the target.
    return guide_strength * guide + (1.0 - guide_strength) * uniform


def run_replica_exchange_policy_mh(
    mdp: FiniteHorizonMDP,
    initial_policy: np.ndarray,
    config: PolicyTemperingConfig,
    tapes: TapeBank | None = None,
    guide_probabilities: np.ndarray | None = None,
) -> PolicyTemperingResult:
    """Sample the exact or fixed-tape policy target by replica exchange.

    With ``tapes=None``, every policy value is evaluated by dynamic programming
    and the cold replica has invariant density
    ``mu(pi) * exp(config.beta * J(pi))`` for a uniform policy reference.
    With an immutable tape bank, the same algorithm exactly targets the stated
    sample-average approximation.  PPO probabilities may guide local proposals
    but are fully corrected by the Metropolis--Hastings ratio.
    """

    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.burn_in < 0 or config.burn_in >= config.iterations:
        raise ValueError("burn_in must lie in [0, iterations)")
    if config.thinning <= 0:
        raise ValueError("thinning must be positive")
    if config.swap_interval <= 0:
        raise ValueError("swap_interval must be positive")
    if mdp.num_actions < 2 or mdp.num_decisions == 0:
        raise ValueError("policy tempering requires decisions and at least two actions")
    if tapes is not None and tapes.horizon != mdp.horizon:
        raise ValueError("tape horizon must match the MDP")

    betas = power_beta_ladder(
        config.beta,
        config.num_temperatures,
        config.ladder_power,
    )
    proposal_probabilities = _validate_guide_probabilities(
        guide_probabilities,
        mdp,
        config.guide_strength,
    )
    rng = np.random.default_rng(config.seed)
    decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
    base_policy = mdp.validate_policy(initial_policy).copy()
    base_policy[mdp.terminal] = 0

    policies = np.zeros(
        (config.num_temperatures, mdp.num_states),
        dtype=np.int64,
    )
    # Overdispersed hot replicas reduce dependence on the learned initializer;
    # only the target-temperature replica starts at the supplied policy.
    for replica in range(config.num_temperatures - 1):
        for state in decision_states:
            policies[replica, state] = int(rng.integers(mdp.num_actions))
    policies[-1] = base_policy

    value_evaluations = 0
    simulator_steps = 0

    def estimate(policy: np.ndarray) -> float:
        nonlocal value_evaluations, simulator_steps
        value_evaluations += 1
        if tapes is None:
            return float(mdp.expected_return(policy))
        values = np.empty(tapes.num_replicas, dtype=np.float64)
        for tape_index in range(tapes.num_replicas):
            values[tape_index], steps = mdp.rollout_return_and_steps(
                policy,
                tapes.initial_u[tape_index],
                tapes.transition_u[tape_index],
            )
            simulator_steps += steps
        return float(values.mean())

    returns = np.asarray([estimate(policy) for policy in policies], dtype=np.float64)
    local_proposals = np.zeros(config.num_temperatures, dtype=np.int64)
    local_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    swap_proposals = np.zeros(config.num_temperatures - 1, dtype=np.int64)
    swap_accepts = np.zeros(config.num_temperatures - 1, dtype=np.int64)
    samples: list[np.ndarray] = []
    sample_returns: list[float] = []

    for iteration in range(config.iterations):
        for replica, replica_beta in enumerate(betas):
            decision = int(rng.integers(mdp.num_decisions))
            state = int(decision_states[decision])
            old_action = int(policies[replica, state])
            row = proposal_probabilities[decision]
            forward = row.copy()
            forward[old_action] = 0.0
            forward /= forward.sum()
            new_action = int(rng.choice(mdp.num_actions, p=forward))

            proposal = policies[replica].copy()
            proposal[state] = new_action
            proposed_return = estimate(proposal)

            reverse_probability = row[old_action] / (1.0 - row[new_action])
            forward_probability = row[new_action] / (1.0 - row[old_action])
            log_acceptance = (
                replica_beta * (proposed_return - returns[replica])
                + np.log(reverse_probability)
                - np.log(forward_probability)
            )
            local_proposals[replica] += 1
            if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                policies[replica] = proposal
                returns[replica] = proposed_return
                local_accepts[replica] += 1

        if (iteration + 1) % config.swap_interval == 0:
            swap_round = (iteration + 1) // config.swap_interval
            parity = (swap_round - 1) % 2
            for lower in range(parity, config.num_temperatures - 1, 2):
                upper = lower + 1
                log_acceptance = (betas[upper] - betas[lower]) * (
                    returns[lower] - returns[upper]
                )
                swap_proposals[lower] += 1
                if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                    policies[[lower, upper]] = policies[[upper, lower]]
                    returns[[lower, upper]] = returns[[upper, lower]]
                    swap_accepts[lower] += 1

        if iteration >= config.burn_in and (
            iteration - config.burn_in
        ) % config.thinning == 0:
            samples.append(policies[-1].copy())
            sample_returns.append(float(returns[-1]))

    return PolicyTemperingResult(
        policies=np.stack(samples, axis=0),
        estimated_returns=np.asarray(sample_returns, dtype=np.float64),
        betas=betas,
        local_proposals=local_proposals,
        local_accepts=local_accepts,
        swap_proposals=swap_proposals,
        swap_accepts=swap_accepts,
        value_evaluations=value_evaluations,
        simulator_steps=simulator_steps,
    )
