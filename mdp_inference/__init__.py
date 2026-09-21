"""Correct expected-return policy inference for finite-horizon MDPs.

The package deliberately keeps the mathematical target separate from any
particular optimizer:

    p_beta(pi) propto mu(pi) exp(beta * E[G | pi]).

Small finite MDP utilities provide exact oracles, while ``replicated_smc``
implements a sample-average approximation using common random-number tapes.
"""

from .exact import GibbsPosterior, exact_gibbs_posterior, exact_saa_posterior
from .mdp import BranchedFiniteHorizonMDP, FiniteHorizonMDP, TapeBank

__all__ = [
    "FiniteHorizonMDP",
    "BranchedFiniteHorizonMDP",
    "TapeBank",
    "GibbsPosterior",
    "exact_gibbs_posterior",
    "exact_saa_posterior",
]
