from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mdp import FiniteHorizonMDP, TapeBank
from .policy_mcmc import autocorrelation_effective_sample_size


@dataclass(frozen=True)
class PolicyTemperingConfig:
    """Replica-exchange MCMC configuration for a policy Gibbs target."""

    beta: float = 1.0
    num_temperatures: int = 8
    ladder_power: float = 2.0
    beta_ladder: tuple[float, ...] | None = None
    iterations: int = 10_000
    burn_in: int = 2_000
    thinning: int = 10
    swap_interval: int = 1
    swap_sweeps: int = 4
    guide_strength: float = 0.5
    global_refresh_probability: float = 0.10
    path_refresh_probability: float = 0.0
    block_refresh_probability: float = 0.0
    block_repeats: tuple[int, ...] | None = None
    global_map_weight: float = 0.25
    global_guide_weight: float = 0.70
    global_uniform_weight: float = 0.05
    lazy_probability: float = 0.05
    prior_initialize_hot_replicas: bool = False
    seed: int = 0


@dataclass(frozen=True)
class PolicyTemperingResult:
    policies: np.ndarray
    estimated_returns: np.ndarray
    betas: np.ndarray
    local_proposals: np.ndarray
    local_accepts: np.ndarray
    global_proposals: np.ndarray
    global_accepts: np.ndarray
    global_move_accepts: np.ndarray
    path_proposals: np.ndarray
    path_accepts: np.ndarray
    path_move_accepts: np.ndarray
    block_proposals: np.ndarray
    block_accepts: np.ndarray
    block_move_accepts: np.ndarray
    swap_proposals: np.ndarray
    swap_accepts: np.ndarray
    post_burn_local_proposals: np.ndarray
    post_burn_local_accepts: np.ndarray
    post_burn_global_proposals: np.ndarray
    post_burn_global_accepts: np.ndarray
    post_burn_global_move_accepts: np.ndarray
    post_burn_path_proposals: np.ndarray
    post_burn_path_accepts: np.ndarray
    post_burn_path_move_accepts: np.ndarray
    post_burn_block_proposals: np.ndarray
    post_burn_block_accepts: np.ndarray
    post_burn_block_move_accepts: np.ndarray
    post_burn_swap_proposals: np.ndarray
    post_burn_swap_accepts: np.ndarray
    walker_temperature_visits: np.ndarray
    walker_endpoint_visits: np.ndarray
    walker_endpoint_transitions: np.ndarray
    walker_round_trips: np.ndarray
    lazy_iterations: int
    value_evaluations: int
    score_computations: int
    score_cache_hits: int
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

    @property
    def global_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.global_accepts,
            self.global_proposals,
            out=np.zeros_like(self.global_accepts, dtype=np.float64),
            where=self.global_proposals > 0,
        )

    @property
    def block_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.block_accepts,
            self.block_proposals,
            out=np.zeros_like(self.block_accepts, dtype=np.float64),
            where=self.block_proposals > 0,
        )

    @property
    def path_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.path_accepts,
            self.path_proposals,
            out=np.zeros_like(self.path_accepts, dtype=np.float64),
            where=self.path_proposals > 0,
        )

    @property
    def post_burn_local_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.post_burn_local_accepts,
            self.post_burn_local_proposals,
            out=np.zeros_like(self.post_burn_local_accepts, dtype=np.float64),
            where=self.post_burn_local_proposals > 0,
        )

    @property
    def post_burn_swap_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.post_burn_swap_accepts,
            self.post_burn_swap_proposals,
            out=np.zeros_like(self.post_burn_swap_accepts, dtype=np.float64),
            where=self.post_burn_swap_proposals > 0,
        )

    @property
    def post_burn_global_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.post_burn_global_accepts,
            self.post_burn_global_proposals,
            out=np.zeros_like(self.post_burn_global_accepts, dtype=np.float64),
            where=self.post_burn_global_proposals > 0,
        )

    @property
    def post_burn_block_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.post_burn_block_accepts,
            self.post_burn_block_proposals,
            out=np.zeros_like(self.post_burn_block_accepts, dtype=np.float64),
            where=self.post_burn_block_proposals > 0,
        )

    @property
    def post_burn_path_acceptance_rates(self) -> np.ndarray:
        return np.divide(
            self.post_burn_path_accepts,
            self.post_burn_path_proposals,
            out=np.zeros_like(self.post_burn_path_accepts, dtype=np.float64),
            where=self.post_burn_path_proposals > 0,
        )

    def marginal_action_probabilities(self, mdp: FiniteHorizonMDP) -> np.ndarray:
        marginals = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
        for state in mdp.decision_states or ():
            marginals[state] = np.bincount(
                self.policies[:, state], minlength=mdp.num_actions
            ) / float(len(self.policies))
        return marginals

    def policy_occupancy_diagnostics(
        self, mdp: FiniteHorizonMDP
    ) -> PolicyOccupancyDiagnostics:
        """Diagnose mixing of the sampled complete policies.

        Return ESS alone can look perfect while the chain is trapped among
        distinct policies with the same return.  These diagnostics instead use
        the categorical action trace at every decision state.  Indicator ESS
        is undefined for a constant trace, so those entries are represented by
        ``NaN`` and the conservative per-state ESS is defined as zero.  This is
        an explicit warning about an untested occupancy probability, not a
        claim that the true posterior must put mass on another action.
        """

        decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
        actions = self.policies[:, decision_states]
        state_action_ess = _categorical_indicator_ess(actions, mdp.num_actions)
        varying = np.isfinite(state_action_ess)
        state_ess = np.zeros(mdp.num_decisions, dtype=np.float64)
        for decision in range(mdp.num_decisions):
            if np.any(varying[decision]):
                state_ess[decision] = float(
                    np.min(state_action_ess[decision, varying[decision]])
                )

        if len(actions) <= 1:
            switch_rates = np.zeros(mdp.num_decisions, dtype=np.float64)
            mean_hamming_jump = 0.0
        else:
            changed = actions[1:] != actions[:-1]
            switch_rates = changed.mean(axis=0)
            mean_hamming_jump = float(changed.mean())
        unique_policy_count = int(np.unique(actions, axis=0).shape[0])
        return PolicyOccupancyDiagnostics(
            state_action_effective_sample_sizes=state_action_ess,
            state_effective_sample_sizes=state_ess,
            state_action_switch_rates=switch_rates,
            unique_policy_count=unique_policy_count,
            unique_policy_fraction=unique_policy_count / float(len(actions)),
            mean_policy_hamming_jump=mean_hamming_jump,
        )

    @property
    def temperature_visit_fractions(self) -> np.ndarray:
        totals = self.walker_temperature_visits.sum(axis=1, keepdims=True)
        return np.divide(
            self.walker_temperature_visits,
            totals,
            out=np.zeros_like(self.walker_temperature_visits, dtype=np.float64),
            where=totals > 0,
        )

    @property
    def cold_walker_count(self) -> int:
        """Number of distinct configuration walkers observed at target beta."""

        return int(np.count_nonzero(self.walker_endpoint_visits[:, 1]))


@dataclass(frozen=True)
class PolicyOccupancyDiagnostics:
    """Mixing diagnostics for categorical complete-policy samples."""

    state_action_effective_sample_sizes: np.ndarray
    state_effective_sample_sizes: np.ndarray
    state_action_switch_rates: np.ndarray
    unique_policy_count: int
    unique_policy_fraction: float
    mean_policy_hamming_jump: float

    @property
    def conservative_effective_sample_size(self) -> float:
        return float(np.min(self.state_effective_sample_sizes))

    @property
    def states_without_action_switches(self) -> int:
        return int(np.count_nonzero(self.state_action_switch_rates == 0.0))

    @property
    def effective_sample_size_quantiles(self) -> np.ndarray:
        return np.quantile(
            self.state_effective_sample_sizes,
            [0.0, 0.1, 0.5, 0.9, 1.0],
        )


def _categorical_indicator_ess(
    actions: np.ndarray,
    num_actions: int,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return indicator ESS for every state/action with bounded memory use.

    Constant Bernoulli traces have no estimable autocorrelation and are
    returned as ``NaN``.  Variable traces use the same initial-positive-
    sequence convention as :func:`autocorrelation_effective_sample_size`, with
    FFT autocovariances so large grids remain practical.
    """

    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("actions must be a nonempty [samples, decisions] array")
    if num_actions < 2:
        raise ValueError("num_actions must be at least two")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    num_samples, num_decisions = actions.shape
    output = np.full((num_decisions, num_actions), np.nan, dtype=np.float64)
    if num_samples == 1:
        return output

    total_traces = num_decisions * num_actions
    fft_length = 1 << (2 * num_samples - 1).bit_length()
    lag_denominators = np.arange(
        num_samples, 0, -1, dtype=np.float64
    )[:, None]
    flat_output = output.reshape(-1)
    for start in range(0, total_traces, chunk_size):
        stop = min(start + chunk_size, total_traces)
        trace_indices = np.arange(start, stop, dtype=np.int64)
        state_indices = trace_indices // num_actions
        action_indices = trace_indices % num_actions
        indicators = actions[:, state_indices] == action_indices[None, :]
        variable = np.any(indicators, axis=0) & ~np.all(indicators, axis=0)
        if not np.any(variable):
            continue

        centered = indicators[:, variable].astype(np.float64)
        centered -= centered.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(centered, n=fft_length, axis=0)
        autocovariances = np.fft.irfft(
            spectrum.conj() * spectrum,
            n=fft_length,
            axis=0,
        )[:num_samples]
        autocovariances /= lag_denominators
        correlations = autocovariances[1:] / autocovariances[0]
        positive_prefix = np.logical_and.accumulate(correlations > 0.0, axis=0)
        correlation_sums = np.sum(
            np.where(positive_prefix, correlations, 0.0), axis=0
        )
        effective_sizes = num_samples / (1.0 + 2.0 * correlation_sums)
        effective_sizes = np.clip(effective_sizes, 1.0, float(num_samples))
        flat_output[trace_indices[variable]] = effective_sizes
    return output


def power_beta_ladder(beta: float, num_temperatures: int, power: float) -> np.ndarray:
    """Return an increasing ladder from the policy prior (0) to ``beta``."""

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if num_temperatures < 2:
        raise ValueError("num_temperatures must be at least two")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("ladder_power must be finite and positive")
    fractions = np.linspace(0.0, 1.0, num_temperatures, dtype=np.float64)
    ladder = beta * fractions**power
    ladder[0] = 0.0
    ladder[-1] = beta
    if np.any(np.diff(ladder) <= 0.0):
        raise ValueError("ladder_power does not yield distinct temperatures")
    return ladder


def resolve_beta_ladder(config: PolicyTemperingConfig) -> np.ndarray:
    """Return and validate the fixed ladder used by a tempering run.

    A custom ladder is an efficiency setting selected during development; its
    endpoints remain the reference distribution at zero and the declared
    scientific target at ``config.beta``.  Confirmatory runs must keep it fixed.
    """

    if config.beta_ladder is None:
        return power_beta_ladder(
            config.beta,
            config.num_temperatures,
            config.ladder_power,
        )
    ladder = np.asarray(config.beta_ladder, dtype=np.float64)
    if ladder.ndim != 1 or len(ladder) != config.num_temperatures:
        raise ValueError("beta_ladder length must equal num_temperatures")
    if not np.all(np.isfinite(ladder)):
        raise ValueError("beta_ladder entries must be finite")
    if ladder[0] != 0.0 or not np.isclose(
        ladder[-1], config.beta, rtol=0.0, atol=1e-12
    ):
        raise ValueError("beta_ladder endpoints must be 0 and beta")
    if np.any(np.diff(ladder) <= 0.0):
        raise ValueError("beta_ladder must be strictly increasing")
    ladder[0] = 0.0
    ladder[-1] = config.beta
    return ladder.copy()


def normalize_frozen_guide_probabilities(
    guide_probabilities: np.ndarray | None,
    mdp: FiniteHorizonMDP,
) -> np.ndarray:
    """Validate and canonically normalize the frozen proposal guide.

    A supplied guide keeps its ``[S, A]`` or ``[D, A]`` shape so the exact
    matrix used by an experiment can be stored and hashed.  The sampler slices
    full-state guides to decision states only after this normalization.
    """

    uniform = np.full(
        (mdp.num_decisions, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )
    if guide_probabilities is None:
        return uniform
    guide = np.asarray(guide_probabilities, dtype=np.float64)
    valid_shapes = {
        (mdp.num_states, mdp.num_actions),
        (mdp.num_decisions, mdp.num_actions),
    }
    if guide.shape not in valid_shapes:
        raise ValueError("guide_probabilities must have shape [D, A] or [S, A]")
    if np.any(guide < 0.0) or not np.all(np.isfinite(guide)):
        raise ValueError("guide probabilities must be finite and nonnegative")
    row_sums = guide.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("guide probability rows must sum to one")
    # Canonicalize tiny accepted round-off before using these same rows both to
    # draw proposals and to evaluate their Hastings probabilities.  Keep every
    # entry except the largest fixed and let that entry absorb the round-off.
    # Reapplying this operation is bitwise idempotent because the sum of the
    # other entries does not depend on the previously corrected entry.
    normalized = guide.copy()
    for row_index in np.flatnonzero(row_sums != 1.0):
        row = normalized[row_index]
        largest = int(np.argmax(row))
        other_sum = math.fsum(
            float(value)
            for action, value in enumerate(row)
            if action != largest
        )
        corrected = 1.0 - other_sum
        if not 0.0 <= corrected <= 1.0:
            raise ValueError("guide probability row could not be normalized")
        row[largest] = corrected
    return normalized


def _local_proposal_probabilities(
    frozen_guide_probabilities: np.ndarray,
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
    # The uniform component gives every local move positive probability.  The
    # guide changes efficiency only; the Hastings ratio preserves the target.
    return (
        guide_strength * frozen_guide_probabilities
        + (1.0 - guide_strength) * uniform
    )


def _validate_global_mixture_weights(config: PolicyTemperingConfig) -> np.ndarray:
    weights = np.asarray(
        [
            config.global_map_weight,
            config.global_guide_weight,
            config.global_uniform_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("global mixture weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("global mixture weights must sum to one")
    # Normalize the final few floating-point bits for ``Generator.choice``.
    return weights / weights.sum()


def _stable_logsumexp(log_terms: np.ndarray) -> float:
    """Return ``log(sum(exp(log_terms)))`` without overflow or underflow."""

    maximum = float(np.max(log_terms))
    if np.isneginf(maximum):
        return -np.inf
    return maximum + float(np.log(np.exp(log_terms - maximum).sum()))


def _log_action_mixture_probability(
    actions: np.ndarray,
    map_actions: np.ndarray,
    guide_probabilities: np.ndarray,
    num_actions: int,
    mixture_weights: np.ndarray,
) -> float:
    """Log probability under an overlapping map/guide/uniform mixture."""

    actions = np.asarray(actions, dtype=np.int64)
    map_actions = np.asarray(map_actions, dtype=np.int64)
    guide_probabilities = np.asarray(guide_probabilities, dtype=np.float64)
    if map_actions.shape != actions.shape or guide_probabilities.shape != (
        len(actions),
        num_actions,
    ):
        raise ValueError("proposal actions and guide rows have incompatible shapes")
    map_weight, guide_weight, uniform_weight = mixture_weights
    log_terms = np.full(3, -np.inf, dtype=np.float64)
    if map_weight > 0.0 and np.array_equal(actions, map_actions):
        log_terms[0] = np.log(map_weight)
    if guide_weight > 0.0:
        selected = guide_probabilities[np.arange(len(actions)), actions]
        if np.all(selected > 0.0):
            log_terms[1] = np.log(guide_weight) + float(np.log(selected).sum())
    if uniform_weight > 0.0:
        log_terms[2] = np.log(uniform_weight) - len(actions) * np.log(num_actions)
    result = _stable_logsumexp(log_terms)
    if not np.isfinite(result):
        raise ValueError("mixture proposal does not have full support")
    return result


def _log_global_proposal_probability(
    policy: np.ndarray,
    map_policy: np.ndarray,
    frozen_guide_probabilities: np.ndarray,
    decision_states: np.ndarray,
    num_actions: int,
    mixture_weights: np.ndarray,
) -> float:
    """Log probability under the overlapping full-policy proposal mixture."""

    return _log_action_mixture_probability(
        np.asarray(policy, dtype=np.int64)[decision_states],
        np.asarray(map_policy, dtype=np.int64)[decision_states],
        frozen_guide_probabilities,
        num_actions,
        mixture_weights,
    )


def _sample_global_policy(
    rng: np.random.Generator,
    map_policy: np.ndarray,
    frozen_guide_probabilities: np.ndarray,
    decision_states: np.ndarray,
    num_actions: int,
    mixture_weights: np.ndarray,
    terminal: np.ndarray,
) -> np.ndarray:
    """Sample from the map/guide/uniform policy mixture."""

    component = int(rng.choice(3, p=mixture_weights))
    proposal = map_policy.copy()
    if component == 1:
        uniforms = rng.random(len(decision_states))
        cumulative = np.cumsum(frozen_guide_probabilities, axis=1)
        actions = np.sum(
            uniforms[:, None] > cumulative,
            axis=1,
            dtype=np.int64,
        )
        # Protect against a last cumulative probability infinitesimally below 1.
        actions = np.minimum(actions, num_actions - 1)
        proposal[decision_states] = actions
    elif component == 2:
        proposal[decision_states] = rng.integers(
            num_actions,
            size=len(decision_states),
        )
    proposal[terminal] = 0
    return proposal


def _validate_block_catalog(
    block_catalog: np.ndarray | None,
    num_decisions: int,
    required: bool,
) -> np.ndarray | None:
    if block_catalog is None:
        if required:
            raise ValueError("block_catalog is required when block refresh is enabled")
        return None
    raw = np.asarray(block_catalog)
    if raw.ndim != 2 or raw.shape[1] != num_decisions or raw.shape[0] == 0:
        raise ValueError("block_catalog must have shape [blocks, D]")
    if raw.dtype.kind not in "biu" or np.any((raw != 0) & (raw != 1)):
        raise ValueError("block_catalog must contain Boolean membership values")
    catalog = raw.astype(bool, copy=True)
    if np.any(catalog.sum(axis=1) == 0):
        raise ValueError("every proposal block must contain at least one state")
    return catalog


def _resolve_block_repeats(
    configured_repeats: tuple[int, ...] | None,
    num_blocks: int,
) -> np.ndarray:
    """Return the fixed number of MH substeps for every catalog block."""

    if num_blocks == 0:
        if configured_repeats not in (None, (), []):
            raise ValueError("block_repeats requires a nonempty block catalog")
        return np.empty(0, dtype=np.int64)
    if configured_repeats is None:
        return np.ones(num_blocks, dtype=np.int64)
    raw = np.asarray(configured_repeats)
    if raw.dtype.kind not in "iu" or raw.shape != (num_blocks,):
        raise ValueError("block_repeats must contain one integer per block")
    repeats = raw.astype(np.int64, copy=True)
    if np.any(repeats <= 0):
        raise ValueError("block_repeats must be positive")
    return repeats


def _block_reverse_mixture_weights(mixture_weights: np.ndarray) -> np.ndarray:
    """Remove the MAP atom and renormalize the full-support reverse mixture."""

    reverse_mass = float(mixture_weights[1] + mixture_weights[2])
    if reverse_mass <= 0.0:
        raise ValueError("block map-toggle reverse proposal has no non-MAP mass")
    return np.asarray(
        [0.0, mixture_weights[1] / reverse_mass, mixture_weights[2] / reverse_mass],
        dtype=np.float64,
    )


def _log_conditional_nonmap_probability(
    actions: np.ndarray,
    map_actions: np.ndarray,
    guide_probabilities: np.ndarray,
    num_actions: int,
    reverse_mixture_weights: np.ndarray,
) -> float:
    """Log R(actions), where R is the reverse mixture conditioned off MAP."""

    if np.array_equal(actions, map_actions):
        raise ValueError("conditional non-MAP probability is undefined at MAP")
    log_probability = _log_action_mixture_probability(
        actions,
        map_actions,
        guide_probabilities,
        num_actions,
        reverse_mixture_weights,
    )
    log_map_probability = _log_action_mixture_probability(
        map_actions,
        map_actions,
        guide_probabilities,
        num_actions,
        reverse_mixture_weights,
    )
    map_probability = float(np.exp(log_map_probability))
    if not 0.0 <= map_probability < 1.0:
        raise ValueError("block reverse proposal must put mass outside MAP")
    return log_probability - float(np.log1p(-map_probability))


def _sample_map_toggle_block_policy(
    rng: np.random.Generator,
    current_policy: np.ndarray,
    map_policy: np.ndarray,
    frozen_guide_probabilities: np.ndarray,
    decision_states: np.ndarray,
    block: np.ndarray,
    num_actions: int,
    mixture_weights: np.ndarray,
    terminal: np.ndarray,
) -> np.ndarray:
    """Propose the MAP block, or a full-support reverse draw when at MAP.

    This star-shaped proposal deliberately attempts the rare useful move on
    every non-MAP visit.  Its reverse distribution is the guide/uniform
    mixture conditioned not to equal MAP; the caller applies the exact
    Hastings correction in both directions.
    """

    block_indices = np.flatnonzero(block)
    states = decision_states[block_indices]
    proposal = np.asarray(current_policy, dtype=np.int64).copy()
    map_actions = map_policy[states]
    if not np.array_equal(current_policy[states], map_actions):
        proposal[states] = map_policy[states]
    else:
        reverse_weights = _block_reverse_mixture_weights(mixture_weights)
        while True:
            component = int(rng.choice(3, p=reverse_weights))
            if component == 1:
                uniforms = rng.random(len(block_indices))
                cumulative = np.cumsum(
                    frozen_guide_probabilities[block_indices],
                    axis=1,
                )
                actions = np.sum(
                    uniforms[:, None] > cumulative,
                    axis=1,
                    dtype=np.int64,
                )
                actions = np.minimum(actions, num_actions - 1)
            else:
                actions = rng.integers(num_actions, size=len(block_indices))
            if not np.array_equal(actions, map_actions):
                proposal[states] = actions
                break
    proposal[terminal] = 0
    return proposal


def _sample_independence_block_policy(
    rng: np.random.Generator,
    current_policy: np.ndarray,
    map_policy: np.ndarray,
    frozen_guide_probabilities: np.ndarray,
    decision_states: np.ndarray,
    block: np.ndarray,
    num_actions: int,
    mixture_weights: np.ndarray,
    terminal: np.ndarray,
) -> np.ndarray:
    """Refresh one block from the overlapping MAP/guide/uniform mixture."""

    block_indices = np.flatnonzero(block)
    states = decision_states[block_indices]
    component = int(rng.choice(3, p=mixture_weights))
    proposal = np.asarray(current_policy, dtype=np.int64).copy()
    if component == 0:
        proposal[states] = map_policy[states]
    elif component == 1:
        uniforms = rng.random(len(block_indices))
        cumulative = np.cumsum(
            frozen_guide_probabilities[block_indices], axis=1
        )
        actions = np.sum(
            uniforms[:, None] > cumulative,
            axis=1,
            dtype=np.int64,
        )
        proposal[states] = np.minimum(actions, num_actions - 1)
    else:
        proposal[states] = rng.integers(num_actions, size=len(block_indices))
    proposal[terminal] = 0
    return proposal


def run_replica_exchange_policy_mh(
    mdp: FiniteHorizonMDP,
    initial_policy: np.ndarray,
    config: PolicyTemperingConfig,
    tapes: TapeBank | None = None,
    guide_probabilities: np.ndarray | None = None,
    block_catalog: np.ndarray | None = None,
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
    if (
        not isinstance(config.swap_sweeps, (int, np.integer))
        or config.swap_sweeps <= 0
    ):
        raise ValueError("swap_sweeps must be a positive integer")
    if not 0.0 <= config.global_refresh_probability <= 1.0:
        raise ValueError("global_refresh_probability must lie in [0, 1]")
    if not 0.0 <= config.path_refresh_probability <= 1.0:
        raise ValueError("path_refresh_probability must lie in [0, 1]")
    if not 0.0 <= config.block_refresh_probability <= 1.0:
        raise ValueError("block_refresh_probability must lie in [0, 1]")
    if (
        config.global_refresh_probability
        + config.path_refresh_probability
        + config.block_refresh_probability
        > 1.0
    ):
        raise ValueError(
            "global, path, and block refresh probabilities must not exceed one"
        )
    if not 0.0 < config.lazy_probability < 1.0:
        raise ValueError("lazy_probability must lie in (0, 1)")
    if mdp.num_actions < 2 or mdp.num_decisions == 0:
        raise ValueError("policy tempering requires decisions and at least two actions")
    if tapes is not None and tapes.horizon != mdp.horizon:
        raise ValueError("tape horizon must match the MDP")

    betas = resolve_beta_ladder(config)
    frozen_guide_probabilities = normalize_frozen_guide_probabilities(
        guide_probabilities,
        mdp,
    )
    if frozen_guide_probabilities.shape == (mdp.num_states, mdp.num_actions):
        frozen_guide_probabilities = frozen_guide_probabilities[
            np.asarray(mdp.decision_states, dtype=np.int64)
        ]
    proposal_probabilities = _local_proposal_probabilities(
        frozen_guide_probabilities,
        mdp,
        config.guide_strength,
    )
    global_mixture_weights = _validate_global_mixture_weights(config)
    if global_mixture_weights[2] == 0.0 and not (
        global_mixture_weights[1] > 0.0
        and np.all(frozen_guide_probabilities > 0.0)
    ):
        raise ValueError("global proposal must have full policy support")
    rng = np.random.default_rng(config.seed)
    decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
    frozen_block_catalog = _validate_block_catalog(
        block_catalog,
        mdp.num_decisions,
        required=(
            config.path_refresh_probability > 0.0
            or config.block_refresh_probability > 0.0
        ),
    )
    num_blocks = 0 if frozen_block_catalog is None else len(frozen_block_catalog)
    block_repeats = _resolve_block_repeats(config.block_repeats, num_blocks)
    base_policy = mdp.validate_policy(initial_policy).copy()
    base_policy[mdp.terminal] = 0
    # The atom is frozen before sampling.  In the main method ``initial_policy``
    # is the action-wise MAP of the learned guide.
    map_policy = base_policy.copy()

    policies = np.zeros(
        (config.num_temperatures, mdp.num_states),
        dtype=np.int64,
    )
    policies[:] = base_policy
    if config.prior_initialize_hot_replicas:
        # This option is useful as a deliberately overdispersed convergence
        # check.  Sparse-reward grids normally warm-start every replica, then
        # rely on the beta=0 chain to forget that initializer during burn-in.
        for replica in range(config.num_temperatures - 1):
            for state in decision_states:
                policies[replica, state] = int(rng.integers(mdp.num_actions))

    value_evaluations = 0
    score_computations = 0
    score_cache_hits = 0
    simulator_steps = 0
    score_cache: dict[bytes, float] = {}
    key_dtype = np.min_scalar_type(max(mdp.num_actions - 1, 0))

    def estimate(policy: np.ndarray) -> float:
        nonlocal value_evaluations, score_computations, score_cache_hits
        nonlocal simulator_steps
        value_evaluations += 1
        cache_key = np.asarray(
            policy[decision_states], dtype=key_dtype
        ).tobytes()
        cached = score_cache.get(cache_key)
        if cached is not None:
            score_cache_hits += 1
            return cached
        score_computations += 1
        if tapes is None:
            value = float(mdp.expected_return(policy))
        else:
            values = np.empty(tapes.num_replicas, dtype=np.float64)
            for tape_index in range(tapes.num_replicas):
                values[tape_index], steps = mdp.rollout_return_and_steps(
                    policy,
                    tapes.initial_u[tape_index],
                    tapes.transition_u[tape_index],
                )
                simulator_steps += steps
            value = float(values.mean())
        score_cache[cache_key] = value
        return value

    returns = np.asarray([estimate(policy) for policy in policies], dtype=np.float64)
    local_proposals = np.zeros(config.num_temperatures, dtype=np.int64)
    local_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    global_proposals = np.zeros(config.num_temperatures, dtype=np.int64)
    global_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    global_move_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    path_proposals = np.zeros(config.num_temperatures, dtype=np.int64)
    path_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    path_move_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    block_proposals = np.zeros(
        (num_blocks, config.num_temperatures), dtype=np.int64
    )
    block_accepts = np.zeros_like(block_proposals)
    block_move_accepts = np.zeros_like(block_proposals)
    swap_proposals = np.zeros(config.num_temperatures - 1, dtype=np.int64)
    swap_accepts = np.zeros(config.num_temperatures - 1, dtype=np.int64)
    post_burn_local_proposals = np.zeros(config.num_temperatures, dtype=np.int64)
    post_burn_local_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    post_burn_global_proposals = np.zeros(
        config.num_temperatures, dtype=np.int64
    )
    post_burn_global_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    post_burn_global_move_accepts = np.zeros(
        config.num_temperatures, dtype=np.int64
    )
    post_burn_path_proposals = np.zeros(
        config.num_temperatures, dtype=np.int64
    )
    post_burn_path_accepts = np.zeros(config.num_temperatures, dtype=np.int64)
    post_burn_path_move_accepts = np.zeros(
        config.num_temperatures, dtype=np.int64
    )
    post_burn_block_proposals = np.zeros_like(block_proposals)
    post_burn_block_accepts = np.zeros_like(block_proposals)
    post_burn_block_move_accepts = np.zeros_like(block_proposals)
    post_burn_swap_proposals = np.zeros(
        config.num_temperatures - 1, dtype=np.int64
    )
    post_burn_swap_accepts = np.zeros(
        config.num_temperatures - 1, dtype=np.int64
    )
    # A walker label travels with its configuration during swaps.  Tracking
    # labels is observational only: it consumes no randomness and cannot
    # change the Markov transition kernel.
    walker_ids = np.arange(config.num_temperatures, dtype=np.int64)
    walker_temperature_visits = np.zeros(
        (config.num_temperatures, config.num_temperatures), dtype=np.int64
    )
    walker_endpoint_visits = np.zeros(
        (config.num_temperatures, 2), dtype=bool
    )
    walker_endpoint_transitions = np.zeros(config.num_temperatures, dtype=np.int64)
    walker_round_trips = np.zeros(config.num_temperatures, dtype=np.int64)
    walker_last_endpoint = np.full(config.num_temperatures, -1, dtype=np.int8)
    samples: list[np.ndarray] = []
    sample_returns: list[float] = []
    lazy_iterations = 0
    active_iterations = 0
    swap_rounds = 0

    def record_endpoint_visits() -> None:
        """Record every observed endpoint hit, including within swap sub-sweeps."""

        for temperature, endpoint in (
            (0, 0),
            (config.num_temperatures - 1, 1),
        ):
            walker = int(walker_ids[temperature])
            walker_endpoint_visits[walker, endpoint] = True
            previous_endpoint = int(walker_last_endpoint[walker])
            if previous_endpoint < 0:
                walker_last_endpoint[walker] = endpoint
            elif previous_endpoint != endpoint:
                walker_endpoint_transitions[walker] += 1
                if walker_endpoint_transitions[walker] % 2 == 0:
                    walker_round_trips[walker] += 1
                walker_last_endpoint[walker] = endpoint

    def attempt_structural_block(
        replica: int,
        replica_beta: float,
        block_index: int,
    ) -> tuple[bool, bool]:
        """Apply one exact structural MH sub-kernel to one replica."""

        if frozen_block_catalog is None:
            raise RuntimeError("block proposal catalog is unavailable")
        block = frozen_block_catalog[block_index]
        block_decisions = np.flatnonzero(block)
        block_states = decision_states[block_decisions]
        if block_index == 0:
            # The long path has a broad high-return mode; an independence
            # proposal can land near that mode without requiring the single
            # exact MAP assignment.
            proposal = _sample_independence_block_policy(
                rng,
                policies[replica],
                map_policy,
                frozen_guide_probabilities,
                decision_states,
                block,
                mdp.num_actions,
                global_mixture_weights,
                mdp.terminal,
            )
            current_log_q = _log_action_mixture_probability(
                policies[replica, block_states],
                map_policy[block_states],
                frozen_guide_probabilities[block_decisions],
                mdp.num_actions,
                global_mixture_weights,
            )
            proposed_log_q = _log_action_mixture_probability(
                proposal[block_states],
                map_policy[block_states],
                frozen_guide_probabilities[block_decisions],
                mdp.num_actions,
                global_mixture_weights,
            )
            log_hastings = current_log_q - proposed_log_q
        else:
            current_is_map = np.array_equal(
                policies[replica, block_states],
                map_policy[block_states],
            )
            proposal = _sample_map_toggle_block_policy(
                rng,
                policies[replica],
                map_policy,
                frozen_guide_probabilities,
                decision_states,
                block,
                mdp.num_actions,
                global_mixture_weights,
                mdp.terminal,
            )
            reverse_weights = _block_reverse_mixture_weights(
                global_mixture_weights
            )
            if current_is_map:
                log_hastings = -_log_conditional_nonmap_probability(
                    proposal[block_states],
                    map_policy[block_states],
                    frozen_guide_probabilities[block_decisions],
                    mdp.num_actions,
                    reverse_weights,
                )
            else:
                log_hastings = _log_conditional_nonmap_probability(
                    policies[replica, block_states],
                    map_policy[block_states],
                    frozen_guide_probabilities[block_decisions],
                    mdp.num_actions,
                    reverse_weights,
                )
        proposed_return = estimate(proposal)
        log_acceptance = (
            replica_beta * (proposed_return - returns[replica]) + log_hastings
        )
        if not np.isfinite(log_acceptance):
            raise FloatingPointError("block-refresh log acceptance is non-finite")
        changes_policy = not np.array_equal(
            proposal[block_states],
            policies[replica, block_states],
        )
        accepted = np.log(rng.random()) < min(0.0, float(log_acceptance))
        if accepted:
            policies[replica] = proposal
            returns[replica] = proposed_return
        return bool(accepted), bool(changes_policy)

    for iteration in range(config.iterations):
        # Mix every complete replica-exchange sweep with the identity kernel.
        # This gives every augmented chain state a positive self-transition,
        # including two-action domains where a forced-different proposal can
        # otherwise be periodic.  The stay decision is independent of state,
        # so it changes neither the product target nor its cold marginal.
        if rng.random() < config.lazy_probability:
            lazy_iterations += 1
        else:
            active_iterations += 1
            for replica, replica_beta in enumerate(betas):
                branch_draw = rng.random()
                if branch_draw < config.global_refresh_probability:
                    proposal = _sample_global_policy(
                        rng,
                        map_policy,
                        frozen_guide_probabilities,
                        decision_states,
                        mdp.num_actions,
                        global_mixture_weights,
                        mdp.terminal,
                    )
                    proposed_return = estimate(proposal)
                    current_log_q = _log_global_proposal_probability(
                        policies[replica],
                        map_policy,
                        frozen_guide_probabilities,
                        decision_states,
                        mdp.num_actions,
                        global_mixture_weights,
                    )
                    proposed_log_q = _log_global_proposal_probability(
                        proposal,
                        map_policy,
                        frozen_guide_probabilities,
                        decision_states,
                        mdp.num_actions,
                        global_mixture_weights,
                    )
                    log_acceptance = (
                        replica_beta * (proposed_return - returns[replica])
                        + current_log_q
                        - proposed_log_q
                    )
                    changes_policy = not np.array_equal(
                        proposal[decision_states],
                        policies[replica, decision_states],
                    )
                    global_proposals[replica] += 1
                    if iteration >= config.burn_in:
                        post_burn_global_proposals[replica] += 1
                    if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                        policies[replica] = proposal
                        returns[replica] = proposed_return
                        global_accepts[replica] += 1
                        if changes_policy:
                            global_move_accepts[replica] += 1
                        if iteration >= config.burn_in:
                            post_burn_global_accepts[replica] += 1
                            if changes_policy:
                                post_burn_global_move_accepts[replica] += 1
                elif branch_draw < (
                    config.global_refresh_probability
                    + config.path_refresh_probability
                ):
                    if frozen_block_catalog is None:  # guarded above
                        raise RuntimeError("block proposal catalog is unavailable")
                    accepted, changes_policy = attempt_structural_block(
                        replica,
                        float(replica_beta),
                        0,
                    )
                    path_proposals[replica] += 1
                    if iteration >= config.burn_in:
                        post_burn_path_proposals[replica] += 1
                    if accepted:
                        path_accepts[replica] += 1
                        if changes_policy:
                            path_move_accepts[replica] += 1
                        if iteration >= config.burn_in:
                            post_burn_path_accepts[replica] += 1
                            if changes_policy:
                                post_burn_path_move_accepts[replica] += 1
                elif branch_draw < (
                    config.global_refresh_probability
                    + config.path_refresh_probability
                    + config.block_refresh_probability
                ):
                    if frozen_block_catalog is None:  # guarded above
                        raise RuntimeError("block proposal catalog is unavailable")
                    # A refresh is a deterministic composition of the fixed
                    # block kernels in catalog order.  Each sub-kernel is an
                    # individually Hastings-corrected MH transition, so their
                    # composition preserves the replica target even though the
                    # complete sweep need not itself be reversible.  Ordering
                    # lets a path repair be followed immediately by its
                    # recovery-fringe repair before replica swaps intervene.
                    for block_index in range(num_blocks):
                        for _ in range(int(block_repeats[block_index])):
                            accepted, changes_policy = attempt_structural_block(
                                replica,
                                float(replica_beta),
                                block_index,
                            )
                            block_proposals[block_index, replica] += 1
                            if iteration >= config.burn_in:
                                post_burn_block_proposals[block_index, replica] += 1
                            if accepted:
                                block_accepts[block_index, replica] += 1
                                if changes_policy:
                                    block_move_accepts[block_index, replica] += 1
                                if iteration >= config.burn_in:
                                    post_burn_block_accepts[
                                        block_index, replica
                                    ] += 1
                                    if changes_policy:
                                        post_burn_block_move_accepts[
                                            block_index, replica
                                        ] += 1
                else:
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
                    proposal[mdp.terminal] = 0
                    proposed_return = estimate(proposal)

                    reverse_probability = row[old_action] / (1.0 - row[new_action])
                    forward_probability = row[new_action] / (1.0 - row[old_action])
                    log_acceptance = (
                        replica_beta * (proposed_return - returns[replica])
                        + np.log(reverse_probability)
                        - np.log(forward_probability)
                    )
                    local_proposals[replica] += 1
                    if iteration >= config.burn_in:
                        post_burn_local_proposals[replica] += 1
                    if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                        policies[replica] = proposal
                        returns[replica] = proposed_return
                        local_accepts[replica] += 1
                        if iteration >= config.burn_in:
                            post_burn_local_accepts[replica] += 1

            if active_iterations % config.swap_interval == 0:
                if iteration >= config.burn_in:
                    record_endpoint_visits()
                for _ in range(config.swap_sweeps):
                    parity = swap_rounds % 2
                    swap_rounds += 1
                    for lower in range(
                        parity, config.num_temperatures - 1, 2
                    ):
                        upper = lower + 1
                        log_acceptance = (betas[upper] - betas[lower]) * (
                            returns[lower] - returns[upper]
                        )
                        swap_proposals[lower] += 1
                        if iteration >= config.burn_in:
                            post_burn_swap_proposals[lower] += 1
                        if np.log(rng.random()) < min(0.0, float(log_acceptance)):
                            policies[[lower, upper]] = policies[[upper, lower]]
                            returns[[lower, upper]] = returns[[upper, lower]]
                            walker_ids[[lower, upper]] = walker_ids[[upper, lower]]
                            swap_accepts[lower] += 1
                            if iteration >= config.burn_in:
                                post_burn_swap_accepts[lower] += 1
                    if iteration >= config.burn_in:
                        record_endpoint_visits()

        if iteration >= config.burn_in:
            walker_temperature_visits[walker_ids, np.arange(config.num_temperatures)] += 1
            record_endpoint_visits()

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
        global_proposals=global_proposals,
        global_accepts=global_accepts,
        global_move_accepts=global_move_accepts,
        path_proposals=path_proposals,
        path_accepts=path_accepts,
        path_move_accepts=path_move_accepts,
        block_proposals=block_proposals,
        block_accepts=block_accepts,
        block_move_accepts=block_move_accepts,
        swap_proposals=swap_proposals,
        swap_accepts=swap_accepts,
        post_burn_local_proposals=post_burn_local_proposals,
        post_burn_local_accepts=post_burn_local_accepts,
        post_burn_global_proposals=post_burn_global_proposals,
        post_burn_global_accepts=post_burn_global_accepts,
        post_burn_global_move_accepts=post_burn_global_move_accepts,
        post_burn_path_proposals=post_burn_path_proposals,
        post_burn_path_accepts=post_burn_path_accepts,
        post_burn_path_move_accepts=post_burn_path_move_accepts,
        post_burn_block_proposals=post_burn_block_proposals,
        post_burn_block_accepts=post_burn_block_accepts,
        post_burn_block_move_accepts=post_burn_block_move_accepts,
        post_burn_swap_proposals=post_burn_swap_proposals,
        post_burn_swap_accepts=post_burn_swap_accepts,
        walker_temperature_visits=walker_temperature_visits,
        walker_endpoint_visits=walker_endpoint_visits,
        walker_endpoint_transitions=walker_endpoint_transitions,
        walker_round_trips=walker_round_trips,
        lazy_iterations=lazy_iterations,
        value_evaluations=value_evaluations,
        score_computations=score_computations,
        score_cache_hits=score_cache_hits,
        simulator_steps=simulator_steps,
    )
