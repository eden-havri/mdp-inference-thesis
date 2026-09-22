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
    swap_proposals: np.ndarray
    swap_accepts: np.ndarray
    walker_temperature_visits: np.ndarray
    walker_endpoint_transitions: np.ndarray
    walker_round_trips: np.ndarray
    lazy_iterations: int
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

        return int(np.count_nonzero(self.walker_temperature_visits[:, -1]))


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
    if not 0.0 < config.lazy_probability < 1.0:
        raise ValueError("lazy_probability must lie in (0, 1)")
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
    policies[:] = base_policy
    if config.prior_initialize_hot_replicas:
        # This option is useful as a deliberately overdispersed convergence
        # check.  Sparse-reward grids normally warm-start every replica, then
        # rely on the beta=0 chain to forget that initializer during burn-in.
        for replica in range(config.num_temperatures - 1):
            for state in decision_states:
                policies[replica, state] = int(rng.integers(mdp.num_actions))

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
    # A walker label travels with its configuration during swaps.  Tracking
    # labels is observational only: it consumes no randomness and cannot
    # change the Markov transition kernel.
    walker_ids = np.arange(config.num_temperatures, dtype=np.int64)
    walker_temperature_visits = np.zeros(
        (config.num_temperatures, config.num_temperatures), dtype=np.int64
    )
    walker_endpoint_transitions = np.zeros(config.num_temperatures, dtype=np.int64)
    walker_round_trips = np.zeros(config.num_temperatures, dtype=np.int64)
    walker_last_endpoint = np.full(config.num_temperatures, -1, dtype=np.int8)
    samples: list[np.ndarray] = []
    sample_returns: list[float] = []
    lazy_iterations = 0
    active_iterations = 0

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

            if active_iterations % config.swap_interval == 0:
                swap_round = active_iterations // config.swap_interval
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
                        walker_ids[[lower, upper]] = walker_ids[[upper, lower]]
                        swap_accepts[lower] += 1

        if iteration >= config.burn_in:
            walker_temperature_visits[walker_ids, np.arange(config.num_temperatures)] += 1
            for temperature, endpoint in ((0, 0), (config.num_temperatures - 1, 1)):
                walker = int(walker_ids[temperature])
                previous_endpoint = int(walker_last_endpoint[walker])
                if previous_endpoint < 0:
                    walker_last_endpoint[walker] = endpoint
                elif previous_endpoint != endpoint:
                    walker_endpoint_transitions[walker] += 1
                    if walker_endpoint_transitions[walker] % 2 == 0:
                        walker_round_trips[walker] += 1
                    walker_last_endpoint[walker] = endpoint

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
        walker_temperature_visits=walker_temperature_visits,
        walker_endpoint_transitions=walker_endpoint_transitions,
        walker_round_trips=walker_round_trips,
        lazy_iterations=lazy_iterations,
        value_evaluations=value_evaluations,
        simulator_steps=simulator_steps,
    )
