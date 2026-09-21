from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank


@dataclass(frozen=True)
class PolicyBankPosterior:
    """Exact Gibbs posterior restricted to a finite, deduplicated policy bank."""

    policies: np.ndarray
    estimated_returns: np.ndarray
    probabilities: np.ndarray
    restricted_log_normalizer: float
    beta: float
    simulator_steps: int

    def __post_init__(self) -> None:
        if self.policies.ndim != 2:
            raise ValueError("policies must have shape [M, S]")
        count = len(self.policies)
        if count == 0:
            raise ValueError("the policy bank cannot be empty")
        if self.estimated_returns.shape != (count,) or self.probabilities.shape != (count,):
            raise ValueError("policy-bank arrays have inconsistent shapes")
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError("policy-bank probabilities must sum to one")

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / float(np.sum(self.probabilities**2))

    @property
    def entropy(self) -> float:
        positive = self.probabilities[self.probabilities > 0.0]
        return -float(np.sum(positive * np.log(positive)))

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        marginals = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for policy, probability in zip(self.policies, self.probabilities):
            for state in mdp.decision_states or ():
                marginals[state, int(policy[state])] += probability
        return marginals

    def exact_return_moments(self, mdp: FiniteHorizonMDP) -> tuple[float, float]:
        values = np.asarray([mdp.expected_return(policy) for policy in self.policies])
        mean = float(np.dot(self.probabilities, values))
        variance = float(np.dot(self.probabilities, (values - mean) ** 2))
        return mean, float(np.sqrt(max(variance, 0.0)))


def _deduplicated_valid_policies(
    mdp: FiniteHorizonMDP, candidate_policies: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(candidate_policies, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != mdp.num_states:
        raise ValueError("candidate_policies must have shape [M, S]")
    if len(candidates) == 0:
        raise ValueError("candidate_policies cannot be empty")
    validated = np.stack([mdp.validate_policy(policy) for policy in candidates], axis=0)
    # Duplicates are one policy in the variational support, not extra prior mass.
    return np.unique(validated, axis=0)


def restricted_policy_bank_posterior(
    mdp: FiniteHorizonMDP,
    candidate_policies: np.ndarray,
    beta: float,
    tapes: TapeBank | None = None,
) -> PolicyBankPosterior:
    """Optimize categorical ELBO weights over a fixed finite support.

    With a uniform reference measure over deterministic policies, the optimum
    on a fixed bank is proportional to ``exp(beta * J_hat(policy))``.  When
    ``tapes`` is supplied, every candidate is evaluated on the same fixed tape
    bank and the result targets the restricted sample-average posterior.
    Omitting ``tapes`` uses exact model expectations and consumes no simulator
    transitions.
    """

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    policies = _deduplicated_valid_policies(mdp, candidate_policies)
    simulator_steps = 0
    if tapes is None:
        estimated_returns = np.asarray(
            [mdp.expected_return(policy) for policy in policies], dtype=np.float64
        )
    else:
        if tapes.horizon != mdp.horizon:
            raise ValueError("tape horizon must match the MDP")
        estimates: list[float] = []
        for policy in policies:
            returns = []
            for replica in range(tapes.num_replicas):
                value, steps = mdp.rollout_return_and_steps(
                    policy,
                    tapes.initial_u[replica],
                    tapes.transition_u[replica],
                )
                returns.append(value)
                simulator_steps += steps
            estimates.append(float(np.mean(returns)))
        estimated_returns = np.asarray(estimates, dtype=np.float64)

    logits = beta * estimated_returns
    maximum = float(np.max(logits))
    shifted = np.exp(logits - maximum)
    probabilities = shifted / shifted.sum()
    global_uniform_log_prior = -mdp.num_decisions * np.log(float(mdp.num_actions))
    restricted_log_normalizer = (
        global_uniform_log_prior + maximum + float(np.log(shifted.sum()))
    )
    return PolicyBankPosterior(
        policies=policies,
        estimated_returns=estimated_returns,
        probabilities=probabilities,
        restricted_log_normalizer=restricted_log_normalizer,
        beta=beta,
        simulator_steps=simulator_steps,
    )
