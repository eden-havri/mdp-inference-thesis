from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mdp_inference.experiment as experiment_module
from mdp_inference.artifacts import load_grid_spec
from mdp_inference.baselines import BaselineResult
from mdp_inference.experiment import ExperimentConfig, _array_sha256, run_experiment
from mdp_inference.gridworld import RIGHT, UP, gridworld_mdp, state_of
from mdp_inference.recovery_blocks import RECOVERY_BLOCK_CONSTRUCTION


MAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "maps"
    / "smoke-seed0.json"
)


def _goal_reaching_ppo_result(mdp) -> BaselineResult:
    probabilities = np.full(
        (mdp.num_states, mdp.num_actions),
        0.01 / float(mdp.num_actions - 1),
        dtype=np.float64,
    )
    probabilities[:, RIGHT] = 0.99
    spec, _ = load_grid_spec(MAP_PATH)
    path_actions = {
        (3, 0): UP,
        (2, 0): UP,
        (1, 0): RIGHT,
        (1, 1): RIGHT,
        (1, 2): RIGHT,
        (1, 3): UP,
    }
    for cell, action in path_actions.items():
        state = state_of(spec, cell)
        probabilities[state] = 0.01 / float(mdp.num_actions - 1)
        probabilities[state, action] = 0.99
    probabilities[mdp.terminal] = 1.0 / mdp.num_actions
    return BaselineResult(
        action_probabilities=probabilities,
        transitions=0,
        history=[],
    )


def test_enabled_block_refresh_persists_catalog_and_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment_module,
        "train_ppo",
        lambda mdp, _config: _goal_reaching_ppo_result(mdp),
    )
    output = run_experiment(
        ExperimentConfig(
            run_id="block-artifacts",
            method="ppo_policy_tempering",
            map_path=str(MAP_PATH),
            output_root=str(tmp_path),
            transition_budget=1,
            evaluation_policy_samples=4,
            mcmc_iterations=12,
            mcmc_burn_in=4,
            mcmc_thinning=2,
            mcmc_temperatures=2,
            mcmc_global_refresh_probability=0.0,
            mcmc_block_refresh_probability=1.0,
            mcmc_seed=23,
        )
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    catalog = np.load(output / "mcmc_block_catalog.npy", allow_pickle=False)
    spec, _ = load_grid_spec(MAP_PATH)
    mdp = gridworld_mdp(spec)

    assert result["result_schema_version"] == 4
    assert result["mcmc_block_refresh_probability"] == 1.0
    assert result["mcmc_block_repeats"] == [1] * len(catalog)
    assert result["mcmc_block_catalog_artifact"] == "mcmc_block_catalog.npy"
    assert result["mcmc_block_catalog_kind"] == RECOVERY_BLOCK_CONSTRUCTION
    assert catalog.dtype == np.bool_
    assert catalog.shape[1] == mdp.num_decisions
    assert len(catalog) >= 2
    assert result["mcmc_block_sizes"] == catalog.sum(axis=1).tolist()
    assert result["mcmc_block_catalog_sha256"] == _array_sha256(catalog)
    for key in (
        "mcmc_block_proposals",
        "mcmc_block_accepts",
        "mcmc_block_move_accepts",
        "mcmc_block_acceptance_rates",
        "mcmc_post_burn_block_proposals",
        "mcmc_post_burn_block_accepts",
        "mcmc_post_burn_block_move_accepts",
        "mcmc_post_burn_block_acceptance_rates",
    ):
        assert np.asarray(result[key]).shape == (len(catalog), 2)
    proposals = np.asarray(result["mcmc_block_proposals"])
    assert np.all(proposals == proposals[:1])


def test_refresh_probabilities_cannot_exceed_one_in_total(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global refresh mixture"):
        run_experiment(
            ExperimentConfig(
                run_id="invalid-proposal-probabilities",
                method="ppo_policy_tempering",
                map_path=str(MAP_PATH),
                output_root=str(tmp_path),
                mcmc_global_refresh_probability=0.7,
                mcmc_block_refresh_probability=0.4,
            )
        )
