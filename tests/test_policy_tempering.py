from __future__ import annotations

import numpy as np
import pytest

from mdp_inference.exact import exact_gibbs_posterior
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.policy_tempering import (
    PolicyTemperingConfig,
    PolicyTemperingResult,
    _log_action_mixture_probability,
    _log_conditional_nonmap_probability,
    _log_global_proposal_probability,
    normalize_frozen_guide_probabilities,
    power_beta_ladder,
    resolve_beta_ladder,
    run_replica_exchange_policy_mh,
)


def test_power_beta_ladder_has_exact_endpoints() -> None:
    ladder = power_beta_ladder(beta=16.0, num_temperatures=5, power=2.0)
    assert ladder[0] == 0.0
    assert ladder[-1] == 16.0
    assert np.all(np.diff(ladder) > 0.0)


def test_frozen_guide_normalization_is_strict_and_idempotent() -> None:
    mdp = two_step_choice_mdp()
    guide = np.full((mdp.num_decisions, mdp.num_actions), 0.5)
    guide[0, 0] += 5e-11

    normalized = normalize_frozen_guide_probabilities(guide, mdp)

    assert np.array_equal(
        normalize_frozen_guide_probabilities(normalized, mdp), normalized
    )
    assert np.allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)

    invalid = guide.copy()
    invalid[0, 0] += 2e-10
    with pytest.raises(ValueError, match="sum to one"):
        normalize_frozen_guide_probabilities(invalid, mdp)


@pytest.mark.parametrize("beta,power", [(np.nan, 2.0), (16.0, np.inf), (16.0, 1e6)])
def test_power_beta_ladder_rejects_nonfinite_or_collapsed_values(
    beta: float, power: float
) -> None:
    with pytest.raises(ValueError):
        power_beta_ladder(beta=beta, num_temperatures=5, power=power)


def test_custom_beta_ladder_is_used_exactly() -> None:
    config = PolicyTemperingConfig(
        beta=16.0,
        num_temperatures=5,
        beta_ladder=(0.0, 0.5, 1.0, 4.0, 16.0),
    )
    assert resolve_beta_ladder(config).tolist() == [0.0, 0.5, 1.0, 4.0, 16.0]


@pytest.mark.parametrize(
    "ladder, message",
    [
        ((0.0, 1.0, 16.0), "length"),
        ((0.1, 1.0, 4.0, 8.0, 16.0), "endpoints"),
        ((0.0, 1.0, 1.0, 8.0, 16.0), "strictly increasing"),
    ],
)
def test_custom_beta_ladder_validation(
    ladder: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_beta_ladder(
            PolicyTemperingConfig(
                beta=16.0,
                num_temperatures=5,
                beta_ladder=ladder,
            )
        )


def test_replica_exchange_matches_exact_policy_marginals() -> None:
    mdp = two_step_choice_mdp()
    beta = 1.4
    exact = exact_gibbs_posterior(mdp, beta=beta)
    guide = np.full((mdp.num_states, mdp.num_actions), 0.5)
    guide[0] = [0.85, 0.15]
    guide[1] = [0.10, 0.90]
    initial_policy = mdp.policy_from_decisions((0, 1))
    initial_policy[mdp.terminal] = 1
    sampled = run_replica_exchange_policy_mh(
        mdp,
        initial_policy,
        PolicyTemperingConfig(
            beta=beta,
            num_temperatures=4,
            iterations=30_000,
            burn_in=3_000,
            thinning=3,
            swap_interval=1,
            swap_sweeps=4,
            guide_strength=0.7,
            global_refresh_probability=0.35,
            seed=912,
        ),
        guide_probabilities=guide,
    )
    assert sampled.simulator_steps == 0
    assert sampled.lazy_iterations > 0
    active_iterations = 30_000 - sampled.lazy_iterations
    assert np.all(
        sampled.local_proposals
        + sampled.global_proposals
        + sampled.path_proposals
        + sampled.block_proposals.sum(axis=0)
        == active_iterations
    )
    assert np.all(sampled.local_proposals > 0)
    assert np.all(sampled.global_proposals > 0)
    assert np.all(sampled.post_burn_local_proposals <= sampled.local_proposals)
    assert np.all(sampled.post_burn_global_proposals <= sampled.global_proposals)
    assert np.all(sampled.post_burn_swap_proposals <= sampled.swap_proposals)
    assert np.all(sampled.post_burn_local_acceptance_rates > 0.0)
    assert np.all(sampled.post_burn_global_acceptance_rates > 0.0)
    assert np.all(sampled.post_burn_swap_acceptance_rates > 0.0)
    assert sampled.value_evaluations == 4 + int(
        sampled.local_proposals.sum()
        + sampled.global_proposals.sum()
        + sampled.path_proposals.sum()
        + sampled.block_proposals.sum()
    )
    assert sampled.score_computations + sampled.score_cache_hits == (
        sampled.value_evaluations
    )
    assert np.all(sampled.local_acceptance_rates > 0.0)
    assert np.all(sampled.global_acceptance_rates > 0.0)
    assert np.all(sampled.swap_acceptance_rates > 0.0)
    assert np.all(sampled.swap_proposals == 2 * active_iterations)
    monitored_iterations = 30_000 - 3_000
    assert np.all(sampled.walker_temperature_visits.sum(axis=1) == monitored_iterations)
    assert sampled.cold_walker_count == 4
    assert sampled.walker_round_trips.sum() > 0
    assert np.all(sampled.policies[:, mdp.terminal] == 0)
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.025,
    )


def test_global_proposal_probability_sums_over_overlapping_components() -> None:
    mdp = two_step_choice_mdp()
    decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
    guide = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    map_policy = mdp.policy_from_decisions((0, 1))
    weights = np.asarray([0.25, 0.70, 0.05], dtype=np.float64)

    map_log_q = _log_global_proposal_probability(
        map_policy,
        map_policy,
        guide,
        decision_states,
        mdp.num_actions,
        weights,
    )
    non_map_policy = mdp.policy_from_decisions((1, 1))
    non_map_log_q = _log_global_proposal_probability(
        non_map_policy,
        map_policy,
        guide,
        decision_states,
        mdp.num_actions,
        weights,
    )

    assert np.exp(map_log_q) == pytest.approx(
        0.25 + 0.70 * 0.8 * 0.75 + 0.05 / 4.0
    )
    assert np.exp(non_map_log_q) == pytest.approx(
        0.70 * 0.2 * 0.75 + 0.05 / 4.0
    )


def test_block_proposal_probability_sums_over_overlapping_components() -> None:
    guide = np.asarray([[0.8, 0.2]], dtype=np.float64)
    weights = np.asarray([0.25, 0.70, 0.05], dtype=np.float64)

    map_log_q = _log_action_mixture_probability(
        np.asarray([0]),
        np.asarray([0]),
        guide,
        2,
        weights,
    )
    non_map_log_q = _log_action_mixture_probability(
        np.asarray([1]),
        np.asarray([0]),
        guide,
        2,
        weights,
    )

    assert np.exp(map_log_q) == pytest.approx(0.25 + 0.70 * 0.8 + 0.05 / 2.0)
    assert np.exp(non_map_log_q) == pytest.approx(0.70 * 0.2 + 0.05 / 2.0)


def test_block_reverse_distribution_is_normalized_off_map() -> None:
    guide = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    map_actions = np.asarray([0, 1], dtype=np.int64)
    reverse_weights = np.asarray([0.0, 0.70 / 0.75, 0.05 / 0.75])
    probability = 0.0
    for first in range(2):
        for second in range(2):
            actions = np.asarray([first, second], dtype=np.int64)
            if np.array_equal(actions, map_actions):
                continue
            probability += np.exp(
                _log_conditional_nonmap_probability(
                    actions,
                    map_actions,
                    guide,
                    2,
                    reverse_weights,
                )
            )
    assert probability == pytest.approx(1.0)


def test_pure_global_refresh_matches_exact_target_with_map_overlap() -> None:
    mdp = two_step_choice_mdp()
    beta = 1.4
    exact = exact_gibbs_posterior(mdp, beta=beta)
    guide = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 1)),
        PolicyTemperingConfig(
            beta=beta,
            num_temperatures=2,
            iterations=25_000,
            burn_in=2_500,
            thinning=3,
            swap_sweeps=4,
            global_refresh_probability=1.0,
            seed=331,
        ),
        guide_probabilities=guide,
    )

    assert np.all(sampled.local_proposals == 0)
    assert np.all(sampled.global_proposals > 0)
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.03,
    )


def test_pure_block_refresh_matches_exact_target_with_map_overlap() -> None:
    mdp = two_step_choice_mdp()
    beta = 1.4
    exact = exact_gibbs_posterior(mdp, beta=beta)
    guide = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 1)),
        PolicyTemperingConfig(
            beta=beta,
            num_temperatures=2,
            iterations=25_000,
            burn_in=2_500,
            thinning=3,
            swap_sweeps=4,
            global_refresh_probability=0.0,
            block_refresh_probability=1.0,
            block_repeats=(1, 2),
            seed=332,
        ),
        guide_probabilities=guide,
        block_catalog=np.eye(mdp.num_decisions, dtype=bool),
    )

    assert np.all(sampled.local_proposals == 0)
    assert np.all(sampled.global_proposals == 0)
    assert np.all(sampled.block_proposals > 0)
    assert np.array_equal(
        sampled.block_proposals[1], 2 * sampled.block_proposals[0]
    )
    assert np.all(sampled.block_move_accepts <= sampled.block_accepts)
    active_iterations = 25_000 - sampled.lazy_iterations
    assert np.all(sampled.block_proposals[0] == active_iterations)
    assert np.all(sampled.block_proposals[1] == 2 * active_iterations)
    assert sampled.value_evaluations == 2 + int(sampled.block_proposals.sum())
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.03,
    )


def test_tempering_rejects_a_non_full_support_guide_strength() -> None:
    mdp = two_step_choice_mdp()
    with pytest.raises(ValueError, match="guide_strength"):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                guide_strength=1.0,
            ),
            guide_probabilities=np.full(
                (mdp.num_decisions, mdp.num_actions),
                1.0 / mdp.num_actions,
            ),
        )


def test_tempering_requires_a_positive_whole_sweep_lazy_probability() -> None:
    mdp = two_step_choice_mdp()
    with pytest.raises(ValueError, match="lazy_probability"):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                lazy_probability=0.0,
            ),
        )


@pytest.mark.parametrize(
    "config, message",
    [
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                global_refresh_probability=1.1,
            ),
            "global_refresh_probability",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                block_refresh_probability=-0.1,
            ),
            "block_refresh_probability",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                path_refresh_probability=-0.1,
            ),
            "path_refresh_probability",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                global_refresh_probability=0.6,
                path_refresh_probability=0.1,
                block_refresh_probability=0.5,
            ),
            "must not exceed one",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                global_map_weight=0.20,
            ),
            "sum to one",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                global_map_weight=-0.10,
                global_guide_weight=1.05,
            ),
            "nonnegative",
        ),
        (
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                swap_sweeps=0,
            ),
            "swap_sweeps",
        ),
    ],
)
def test_tempering_validates_global_kernel_settings(
    config: PolicyTemperingConfig, message: str
) -> None:
    mdp = two_step_choice_mdp()
    with pytest.raises(ValueError, match=message):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 1)),
            config,
        )


def test_global_proposal_requires_full_support() -> None:
    mdp = two_step_choice_mdp()
    guide_without_full_support = np.asarray(
        [[1.0, 0.0], [1.0, 0.0]], dtype=np.float64
    )
    with pytest.raises(ValueError, match="full policy support"):
        run_replica_exchange_policy_mh(
            mdp,
            mdp.policy_from_decisions((0, 0)),
            PolicyTemperingConfig(
                num_temperatures=2,
                iterations=10,
                burn_in=1,
                global_map_weight=0.25,
                global_guide_weight=0.75,
                global_uniform_weight=0.0,
            ),
            guide_probabilities=guide_without_full_support,
        )


def test_overdispersed_initialization_does_not_change_the_exact_target() -> None:
    mdp = two_step_choice_mdp()
    exact = exact_gibbs_posterior(mdp, beta=1.0)
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((1, 1)),
        PolicyTemperingConfig(
            beta=1.0,
            num_temperatures=3,
            iterations=20_000,
            burn_in=2_000,
            thinning=3,
            prior_initialize_hot_replicas=True,
            seed=73,
        ),
    )
    assert np.allclose(
        sampled.marginal_action_probabilities(mdp),
        exact.marginal_action_probabilities(mdp),
        atol=0.035,
    )


def test_policy_occupancy_diagnostics_flag_a_stuck_decision_state() -> None:
    mdp = two_step_choice_mdp()
    policies = np.stack(
        [
            mdp.policy_from_decisions((0, 0)),
            mdp.policy_from_decisions((1, 0)),
            mdp.policy_from_decisions((0, 0)),
            mdp.policy_from_decisions((1, 0)),
        ]
    )
    sampled = PolicyTemperingResult(
        policies=policies,
        estimated_returns=np.ones(4),
        betas=np.asarray([0.0, 1.0]),
        local_proposals=np.ones(2, dtype=np.int64),
        local_accepts=np.ones(2, dtype=np.int64),
        global_proposals=np.ones(2, dtype=np.int64),
        global_accepts=np.ones(2, dtype=np.int64),
        global_move_accepts=np.ones(2, dtype=np.int64),
        path_proposals=np.ones(2, dtype=np.int64),
        path_accepts=np.ones(2, dtype=np.int64),
        path_move_accepts=np.ones(2, dtype=np.int64),
        block_proposals=np.ones((1, 2), dtype=np.int64),
        block_accepts=np.ones((1, 2), dtype=np.int64),
        block_move_accepts=np.ones((1, 2), dtype=np.int64),
        swap_proposals=np.ones(1, dtype=np.int64),
        swap_accepts=np.ones(1, dtype=np.int64),
        post_burn_local_proposals=np.ones(2, dtype=np.int64),
        post_burn_local_accepts=np.ones(2, dtype=np.int64),
        post_burn_global_proposals=np.ones(2, dtype=np.int64),
        post_burn_global_accepts=np.ones(2, dtype=np.int64),
        post_burn_global_move_accepts=np.ones(2, dtype=np.int64),
        post_burn_path_proposals=np.ones(2, dtype=np.int64),
        post_burn_path_accepts=np.ones(2, dtype=np.int64),
        post_burn_path_move_accepts=np.ones(2, dtype=np.int64),
        post_burn_block_proposals=np.ones((1, 2), dtype=np.int64),
        post_burn_block_accepts=np.ones((1, 2), dtype=np.int64),
        post_burn_block_move_accepts=np.ones((1, 2), dtype=np.int64),
        post_burn_swap_proposals=np.ones(1, dtype=np.int64),
        post_burn_swap_accepts=np.ones(1, dtype=np.int64),
        walker_temperature_visits=np.asarray([[2, 2], [2, 2]]),
        walker_endpoint_visits=np.ones((2, 2), dtype=bool),
        walker_endpoint_transitions=np.asarray([1, 1]),
        walker_round_trips=np.asarray([0, 0]),
        lazy_iterations=0,
        value_evaluations=0,
        score_computations=0,
        score_cache_hits=0,
        simulator_steps=0,
    )

    diagnostics = sampled.policy_occupancy_diagnostics(mdp)

    assert sampled.return_effective_sample_size == 4.0
    assert diagnostics.conservative_effective_sample_size == 0.0
    assert diagnostics.states_without_action_switches == 1
    assert diagnostics.unique_policy_count == 2
    assert diagnostics.unique_policy_fraction == 0.5
    assert diagnostics.state_action_switch_rates.tolist() == [1.0, 0.0]
    assert diagnostics.mean_policy_hamming_jump == 0.5
    assert diagnostics.state_action_effective_sample_sizes[0].tolist() == [4.0, 4.0]
    assert np.all(np.isnan(diagnostics.state_action_effective_sample_sizes[1]))


def test_walker_flow_accounting_is_post_burn_in_and_normalized() -> None:
    mdp = two_step_choice_mdp()
    iterations = 400
    burn_in = 100
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 0)),
        PolicyTemperingConfig(
            beta=0.01,
            num_temperatures=3,
            iterations=iterations,
            burn_in=burn_in,
            thinning=10,
            swap_interval=1,
            guide_strength=0.0,
            seed=112,
        ),
    )

    assert np.all(
        sampled.walker_temperature_visits.sum(axis=1) == iterations - burn_in
    )
    assert np.allclose(sampled.temperature_visit_fractions.sum(axis=1), 1.0)
    assert sampled.cold_walker_count == 3
    assert sampled.walker_round_trips.sum() > 0
    assert np.all(sampled.walker_endpoint_transitions >= 2 * sampled.walker_round_trips)


def test_endpoint_flow_is_not_missed_between_swap_sub_sweeps() -> None:
    mdp = two_step_choice_mdp()
    sampled = run_replica_exchange_policy_mh(
        mdp,
        mdp.policy_from_decisions((0, 0)),
        PolicyTemperingConfig(
            beta=1e-12,
            num_temperatures=2,
            iterations=100,
            burn_in=10,
            thinning=10,
            swap_interval=1,
            swap_sweeps=4,
            global_refresh_probability=0.0,
            guide_strength=0.0,
            seed=918,
        ),
    )

    # With two temperatures, four checkerboard sub-sweeps swap each walker to
    # the opposite endpoint and back before the outer iteration is observed.
    assert sampled.walker_endpoint_transitions.sum() > 0
    assert sampled.walker_round_trips.sum() > 0
