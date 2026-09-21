from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank, enumerate_decision_vectors


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    normalizer = maximum + float(np.log(shifted.sum()))
    return shifted / shifted.sum(), normalizer


@dataclass(frozen=True)
class GibbsPosterior:
    policies: np.ndarray
    returns: np.ndarray
    probabilities: np.ndarray
    log_normalizer: float
    beta: float
    source: str

    def __post_init__(self) -> None:
        if self.policies.ndim != 2:
            raise ValueError("policies must have shape [P, S]")
        n = self.policies.shape[0]
        if self.returns.shape != (n,) or self.probabilities.shape != (n,):
            raise ValueError("posterior arrays have inconsistent shapes")
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError("posterior probabilities must sum to one")

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        marginals = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for policy, probability in zip(self.policies, self.probabilities):
            for state in mdp.decision_states or ():
                marginals[state, int(policy[state])] += float(probability)
        return marginals


def _enumerated_policies(mdp: FiniteHorizonMDP) -> np.ndarray:
    policy_count = mdp.num_actions ** mdp.num_decisions
    if policy_count > 2_000_000:
        raise ValueError(
            f"exact enumeration would require {policy_count:,} policies; use an approximate method"
        )
    return np.stack(
        [mdp.policy_from_decisions(actions) for actions in enumerate_decision_vectors(mdp.num_decisions, mdp.num_actions)],
        axis=0,
    )


def _prior_log_probabilities(
    mdp: FiniteHorizonMDP,
    policies: np.ndarray,
    prior_probabilities: np.ndarray | None,
) -> np.ndarray:
    if prior_probabilities is None:
        return np.full(len(policies), -mdp.num_decisions * np.log(mdp.num_actions), dtype=np.float64)
    prior = np.asarray(prior_probabilities, dtype=np.float64)
    if prior.shape != (mdp.num_decisions, mdp.num_actions):
        raise ValueError("prior_probabilities must have shape [D, A]")
    if np.any(prior <= 0.0) or not np.allclose(prior.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each prior row must be positive and sum to one")
    result = np.zeros(len(policies), dtype=np.float64)
    for i, policy in enumerate(policies):
        result[i] = sum(
            np.log(prior[d, int(policy[state])])
            for d, state in enumerate(mdp.decision_states or ())
        )
    return result


def exact_gibbs_posterior(
    mdp: FiniteHorizonMDP,
    beta: float = 1.0,
    prior_probabilities: np.ndarray | None = None,
) -> GibbsPosterior:
    """Enumerate the intended expected-return Gibbs posterior."""

    policies = _enumerated_policies(mdp)
    returns = np.asarray([mdp.expected_return(policy) for policy in policies], dtype=np.float64)
    log_prior = _prior_log_probabilities(mdp, policies, prior_probabilities)
    probabilities, log_normalizer = _normalize_log_weights(log_prior + beta * returns)
    return GibbsPosterior(policies, returns, probabilities, log_normalizer, beta, "expected_return")


def exact_saa_posterior(
    mdp: FiniteHorizonMDP,
    tapes: TapeBank,
    beta: float = 1.0,
    prior_probabilities: np.ndarray | None = None,
) -> GibbsPosterior:
    """Enumerate the Gibbs posterior for a fixed sample-average objective."""

    policies = _enumerated_policies(mdp)
    returns = np.asarray([mdp.tape_returns(policy, tapes).mean() for policy in policies], dtype=np.float64)
    log_prior = _prior_log_probabilities(mdp, policies, prior_probabilities)
    probabilities, log_normalizer = _normalize_log_weights(log_prior + beta * returns)
    return GibbsPosterior(policies, returns, probabilities, log_normalizer, beta, "sample_average")


def exact_exponential_utility_posterior(
    mdp: FiniteHorizonMDP,
    tapes: TapeBank,
    beta: float = 1.0,
    prior_probabilities: np.ndarray | None = None,
) -> GibbsPosterior:
    """Enumerate the finite-tape exponential-utility target for diagnostics.

    This is intentionally a diagnostic, not the main method. It computes
    ``log mean exp(beta * G)`` per policy and exposes the target changed by
    exponentiating individual stochastic returns.
    """

    policies = _enumerated_policies(mdp)
    risk_values = []
    for policy in policies:
        values = beta * mdp.tape_returns(policy, tapes)
        maximum = float(values.max())
        risk_values.append(maximum + np.log(np.exp(values - maximum).mean()))
    risk_values_array = np.asarray(risk_values, dtype=np.float64)
    log_prior = _prior_log_probabilities(mdp, policies, prior_probabilities)
    probabilities, log_normalizer = _normalize_log_weights(log_prior + risk_values_array)
    return GibbsPosterior(
        policies,
        risk_values_array / beta if beta != 0.0 else risk_values_array,
        probabilities,
        log_normalizer,
        beta,
        "exponential_utility",
    )
