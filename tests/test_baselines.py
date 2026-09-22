from __future__ import annotations

import json

import numpy as np
import torch

import mdp_inference.baselines as baseline_module
from mdp_inference.artifacts import write_grid_spec
from mdp_inference.baselines import (
    CEMConfig,
    DiscreteSACConfig,
    DoubleQConfig,
    PPOConfig,
    ReinforceConfig,
    train_cem,
    train_discrete_sac,
    train_double_q,
    train_ppo,
    train_reinforce,
)
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.experiment import ExperimentConfig, run_experiment
from mdp_inference.gridworld import GridWorldSpec
from mdp_inference.oracles import evaluate_stationary_stochastic_policy, uniform_random_policy_value


def test_all_learning_baselines_beat_random_on_gate_mdp() -> None:
    mdp = two_step_choice_mdp()
    random_value = uniform_random_policy_value(mdp)
    runs = (
        train_reinforce(mdp, ReinforceConfig(transition_budget=1_000, seed=1)),
        train_ppo(mdp, PPOConfig(transition_budget=1_000, batch_transitions=128, seed=1)),
        train_discrete_sac(
            mdp,
            DiscreteSACConfig(
                transition_budget=1_000,
                learning_starts=64,
                batch_size=64,
                seed=1,
            ),
        ),
        train_cem(
            mdp,
            CEMConfig(
                transition_budget=1_000,
                population_size=16,
                elite_fraction=0.25,
                rollouts_per_policy=1,
                seed=1,
            ),
        ),
        train_double_q(
            mdp,
            DoubleQConfig(
                transition_budget=1_000,
                learning_rate=0.1,
                seed=1,
            ),
        ),
    )
    for result in runs:
        value = evaluate_stationary_stochastic_policy(mdp, result.action_probabilities)
        assert 1_000 <= result.transitions < 1_000 + mdp.horizon
        assert value > random_value + 0.15


def test_double_q_target_selects_with_one_table_and_evaluates_with_the_other() -> None:
    selection_q = np.array([[0.0, 0.0], [1.0, 4.0]], dtype=np.float64)
    evaluation_q = np.array([[0.0, 0.0], [9.0, -3.0]], dtype=np.float64)
    rng = np.random.default_rng(7)
    target = baseline_module._double_q_td_target(
        selection_q,
        evaluation_q,
        next_state=1,
        reward=2.0,
        done=False,
        gamma=0.5,
        rng=rng,
    )
    assert target == 0.5
    terminal_target = baseline_module._double_q_td_target(
        selection_q,
        evaluation_q,
        next_state=1,
        reward=2.0,
        done=True,
        gamma=0.5,
        rng=rng,
    )
    assert terminal_target == 2.0


def test_double_q_target_uses_the_correct_finite_horizon_slice() -> None:
    # The same physical next state has a different greedy action and value at
    # the two times.  A state-only table would necessarily alias these targets.
    selection_q = np.array(
        [
            [[8.0, 1.0]],
            [[1.0, 8.0]],
        ],
        dtype=np.float64,
    )
    evaluation_q = np.array(
        [
            [[3.0, 30.0]],
            [[40.0, 5.0]],
        ],
        dtype=np.float64,
    )
    first = baseline_module._double_q_td_target(
        selection_q,
        evaluation_q,
        next_state=0,
        reward=1.0,
        done=False,
        gamma=0.5,
        rng=np.random.default_rng(1),
        next_time=0,
    )
    second = baseline_module._double_q_td_target(
        selection_q,
        evaluation_q,
        next_state=0,
        reward=1.0,
        done=False,
        gamma=0.5,
        rng=np.random.default_rng(1),
        next_time=1,
    )
    assert first == 2.5
    assert second == 3.5


def test_discrete_sac_target_uses_the_correct_finite_horizon_slice() -> None:
    actor_logits = torch.zeros((1, 2), dtype=torch.float64)
    target_q1 = torch.tensor(
        [[[2.0, 2.0]], [[7.0, 7.0]]], dtype=torch.float64
    )
    target_q2 = target_q1.clone()
    targets = baseline_module._discrete_sac_td_targets(
        actor_logits=actor_logits,
        target_q1=target_q1,
        target_q2=target_q2,
        next_times=torch.tensor([0, 1, 2]),
        next_states=torch.tensor([0, 0, 0]),
        rewards=torch.tensor([1.0, 1.0, 4.0], dtype=torch.float64),
        dones=torch.tensor([False, False, True]),
        gamma=0.5,
        alpha=0.0,
    )
    assert torch.allclose(targets, torch.tensor([2.0, 4.5, 4.0], dtype=torch.float64))


def test_stationary_double_q_projection_uses_observed_time_occupancy() -> None:
    q_values = np.array(
        [
            [[4.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [0.0, 5.0]],
        ],
        dtype=np.float64,
    )
    visits = np.array([[1, 0], [3, 2]], dtype=np.int64)
    probabilities = baseline_module._stationary_double_q_projection(q_values, visits)
    assert np.array_equal(probabilities, np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_cem_elite_update_uses_smoothed_frequencies_and_probability_floor() -> None:
    probabilities = np.full((2, 2), 0.5, dtype=np.float64)
    elite_decisions = np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int64)
    updated = baseline_module._cem_elite_update(
        probabilities,
        elite_decisions,
        smoothing=1.0,
        min_action_probability=0.1,
    )
    expected = np.array([[19.0 / 30.0, 11.0 / 30.0], [0.1, 0.9]])
    assert np.allclose(updated, expected)
    assert np.allclose(probabilities, 0.5)


def test_new_tabular_baselines_are_reproducible_and_share_budget_accounting() -> None:
    mdp = two_step_choice_mdp()
    cem_config = CEMConfig(
        transition_budget=250,
        population_size=8,
        rollouts_per_policy=1,
        seed=19,
    )
    cem_first = train_cem(mdp, cem_config)
    cem_second = train_cem(mdp, cem_config)
    assert 250 <= cem_first.transitions < 250 + mdp.horizon
    assert cem_first.transitions == cem_second.transitions
    assert np.array_equal(cem_first.action_probabilities, cem_second.action_probabilities)
    assert cem_first.history == cem_second.history

    double_q_config = DoubleQConfig(
        transition_budget=250,
        learning_rate=0.1,
        seed=19,
    )
    double_q_first = train_double_q(mdp, double_q_config)
    double_q_second = train_double_q(mdp, double_q_config)
    assert double_q_first.transitions == 250
    assert double_q_second.transitions == 250
    assert np.array_equal(
        double_q_first.action_probabilities,
        double_q_second.action_probabilities,
    )
    assert double_q_first.history == double_q_second.history

    sac_config = DiscreteSACConfig(
        transition_budget=250,
        learning_starts=16,
        batch_size=16,
        updates_per_transition=1,
        seed=19,
    )
    sac_first = train_discrete_sac(mdp, sac_config)
    sac_second = train_discrete_sac(mdp, sac_config)
    assert 250 <= sac_first.transitions < 250 + mdp.horizon
    assert sac_first.transitions == sac_second.transitions
    assert np.array_equal(sac_first.action_probabilities, sac_second.action_probabilities)
    assert sac_first.history == sac_second.history


def test_new_tabular_baselines_use_the_shared_experiment_budget(tmp_path) -> None:
    spec = GridWorldSpec(
        rows=2,
        cols=2,
        start=(1, 0),
        goals=((0, 1),),
        slip_probability=0.0,
        horizon=3,
    )
    map_path = tmp_path / "map.json"
    write_grid_spec(map_path, spec)
    budget = 60
    for method in ("cem", "double_q"):
        output = run_experiment(
            ExperimentConfig(
                run_id=f"{method}-budget-smoke",
                method=method,
                map_path=str(map_path),
                output_root=str(tmp_path / "results"),
                evaluation_policy_samples=8,
                transition_budget=budget,
                learning_rate=0.1,
                training_seed=5,
            )
        )
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert result["transition_budget"] == budget
        assert budget <= result["simulator_steps"] < budget + spec.horizon
        assert result["transition_budget_semantics"] == "minimum_counted_training_transitions"
