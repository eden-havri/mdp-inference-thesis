from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mdp_inference.chain_diagnostics import (
    TemperingGateThresholds,
    _array_sha256,
    _realized_tape_seed,
    audit_tempering_chains,
    audit_tempering_probe,
    audit_tempering_rescue,
    coordinate_total_variation,
    split_rhat,
)
from mdp_inference.artifacts import load_grid_spec
from mdp_inference.experiment import ExperimentConfig, run_experiment
from mdp_inference.gridworld import gridworld_mdp


MAP_PATH = Path(__file__).resolve().parents[1] / "experiments" / "maps" / "smoke-seed0.json"


def _write_probe_run(
    root: Path,
    name: str,
    mcmc_seed: int,
    min_swap: float,
    endpoint_transitions: int,
    source_hash: str = "same-source",
) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    spec, _ = load_grid_spec(MAP_PATH)
    mdp = gridworld_mdp(spec)
    config = {
        "run_id": name,
        "output_root": str(root),
        "method": "ppo_policy_tempering",
        "training_seed": 5,
        "transition_budget": 100,
        "beta": 1.0,
        "mcmc_tapes": 0,
        "mcmc_temperatures": 2,
        "mcmc_iterations": 10,
        "mcmc_burn_in": 2,
        "mcmc_thinning": 1,
        "mcmc_swap_interval": 1,
        "mcmc_guide_strength": 0.5,
        "mcmc_lazy_probability": 0.05,
        "mcmc_seed": mcmc_seed,
        "mcmc_prior_initialize_hot_replicas": False,
    }
    result = {
        "method": "ppo_policy_tempering",
        "mcmc_target": "exact_expected_return",
        "mcmc_seed": mcmc_seed,
        "mcmc_tape_seed": 5_000_008,
        "mcmc_beta_ladder": [0.0, 1.0],
        "mcmc_swap_acceptance_rates": [min_swap],
        "mcmc_post_burn_swap_acceptance_rates": [min_swap],
        "mcmc_walker_endpoint_transitions": [endpoint_transitions, 0],
        "mcmc_total_round_trips": 1,
        "mcmc_cold_walker_count": 2,
        "mcmc_unique_policy_fraction": 0.125,
        "mcmc_mean_policy_hamming_jump": 0.0,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"source_snapshot_sha256": source_hash}), encoding="utf-8"
    )
    shutil.copyfile(MAP_PATH, run_dir / "map.json")
    np.save(
        run_dir / "policy_particles.npy",
        np.zeros((8, mdp.num_states), dtype=np.int64),
        allow_pickle=False,
    )
    (run_dir / "DONE").write_text("ok\n", encoding="utf-8")


def _upgrade_probe_run_to_schema_v2(root: Path, name: str) -> None:
    run_dir = root / name
    spec, _ = load_grid_spec(MAP_PATH)
    mdp = gridworld_mdp(spec)
    guide = np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )
    np.save(run_dir / "mcmc_guide_probabilities.npy", guide, allow_pickle=False)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "result_schema_version": 2,
            "mcmc_samples": 8,
            "mcmc_temperatures": 2,
            "mcmc_ladder_kind": "power",
            "mcmc_diagnostic_window": "post_burn",
            "mcmc_guide_probabilities_sha256": _array_sha256(guide),
            "mcmc_guide_probabilities_artifact": "mcmc_guide_probabilities.npy",
            "mcmc_local_proposals": [8, 8],
            "mcmc_local_accepts": [4, 4],
            "mcmc_local_acceptance_rates": [0.5, 0.5],
            "mcmc_swap_proposals": [8],
            "mcmc_swap_accepts": [2],
            "mcmc_swap_acceptance_rates": [0.25],
            "mcmc_post_burn_local_proposals": [6, 6],
            "mcmc_post_burn_local_accepts": [3, 3],
            "mcmc_post_burn_local_acceptance_rates": [0.5, 0.5],
            "mcmc_post_burn_swap_proposals": [4],
            "mcmc_post_burn_swap_accepts": [1],
            "mcmc_post_burn_swap_acceptance_rates": [0.25],
        }
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")


def _upgrade_probe_run_to_schema_v3(root: Path, name: str) -> None:
    _upgrade_probe_run_to_schema_v2(root, name)
    run_dir = root / name
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "mcmc_swap_sweeps": 4,
            "mcmc_global_refresh_probability": 0.10,
            "mcmc_global_map_weight": 0.25,
            "mcmc_global_guide_weight": 0.70,
            "mcmc_global_uniform_weight": 0.05,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "result_schema_version": 3,
            "mcmc_swap_interval": 1,
            "mcmc_swap_sweeps": 4,
            "mcmc_global_refresh_probability": 0.10,
            "mcmc_global_mixture_weights": {
                "map": 0.25,
                "guide_product": 0.70,
                "uniform": 0.05,
            },
            "mcmc_global_proposals": [2, 2],
            "mcmc_global_accepts": [1, 2],
            "mcmc_global_move_accepts": [1, 2],
            "mcmc_global_acceptance_rates": [0.5, 1.0],
            "mcmc_post_burn_global_proposals": [1, 1],
            "mcmc_post_burn_global_accepts": [0, 1],
            "mcmc_post_burn_global_move_accepts": [0, 1],
            "mcmc_post_burn_global_acceptance_rates": [0.0, 1.0],
            "mcmc_walker_temperature_visit_fractions": [
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            "mcmc_walker_endpoint_visits": [[True, True], [True, True]],
            "mcmc_walker_endpoint_transitions": [1, 1],
            "mcmc_walker_round_trips": [0, 0],
            "mcmc_total_round_trips": 0,
            "mcmc_cold_walker_count": 2,
            "mcmc_unique_policy_samples": 2,
        }
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")


def test_split_rhat_detects_between_chain_shift() -> None:
    rng = np.random.default_rng(7)
    agreeing = rng.normal(size=(4, 2_000))
    shifted = agreeing.copy()
    shifted[-1] += 2.0
    assert split_rhat(agreeing) < 1.01
    assert split_rhat(shifted) > 1.1


def test_coordinate_total_variation_reports_policy_disagreement() -> None:
    policies = np.zeros((3, 100, 4), dtype=np.int64)
    policies[0, :50, 1] = 1
    policies[1, :50, 1] = 1
    policies[2, :, 1] = 1
    distances = coordinate_total_variation(policies, (1, 2), num_actions=2)
    assert distances.shape == (3, 2)
    assert np.allclose(distances[:, 1], 0.0)
    assert np.isclose(distances[0, 0], 0.0)
    assert np.isclose(distances[1, 0], 0.5)
    assert np.isclose(distances[2, 0], 0.5)


def test_realized_tape_seed_is_part_of_the_audited_target() -> None:
    config = {"mcmc_tapes": 8, "mcmc_tape_seed": None, "training_seed": 3}
    assert _realized_tape_seed(config, {}) == 5_000_006
    assert _realized_tape_seed(config, {"mcmc_tape_seed": 71}) == 71
    assert _realized_tape_seed({"mcmc_tapes": 0}, {}) is None


def test_probe_audit_encodes_the_predeclared_ladder_decision(tmp_path: Path) -> None:
    _write_probe_run(tmp_path, "warm", 11, min_swap=0.25, endpoint_transitions=1)
    _write_probe_run(
        tmp_path, "overdispersed", 12, min_swap=0.21, endpoint_transitions=2
    )
    audit = audit_tempering_probe(tmp_path)
    assert audit["passed"] is True
    assert audit["scope"] == "ladder_probe_only_not_full_mixing_validation"

    second_result_path = tmp_path / "warm" / "result.json"
    second_result = json.loads(second_result_path.read_text(encoding="utf-8"))
    second_result["mcmc_post_burn_swap_acceptance_rates"] = [0.19]
    second_result_path.write_text(json.dumps(second_result), encoding="utf-8")
    failed = audit_tempering_probe(tmp_path)
    assert failed["passed"] is False
    assert failed["checks"]["post_burn_swap_acceptance"] is False


def test_rescue_audit_requires_global_jump_final_swap_and_multiple_walkers(
    tmp_path: Path,
) -> None:
    _write_probe_run(tmp_path, "overdispersed", 13, 0.25, 0)
    _upgrade_probe_run_to_schema_v3(tmp_path, "overdispersed")

    audit = audit_tempering_rescue(tmp_path)

    assert audit["passed"] is True
    assert audit["metrics"]["post_burn_high_beta_global_move_accepts"] == 1
    assert audit["metrics"]["post_burn_final_edge_swap_accepts"] == 1
    assert audit["metrics"]["cold_unique_policy_samples"] == 2

    result_path = tmp_path / "overdispersed" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["mcmc_post_burn_global_move_accepts"] = [0, 0]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    failed = audit_tempering_rescue(tmp_path)
    assert failed["passed"] is False
    assert (
        failed["checks"]["accepted_policy_changing_high_beta_global_refresh"]
        is False
    )

    result["mcmc_post_burn_global_move_accepts"] = [0, 1]
    result["mcmc_unique_policy_samples"] = 1
    result_path.write_text(json.dumps(result), encoding="utf-8")
    failed = audit_tempering_rescue(tmp_path)
    assert failed["passed"] is False
    assert failed["checks"]["cold_policy_movement"] is False


def test_audit_rejects_mixed_source_snapshots(tmp_path: Path) -> None:
    _write_probe_run(tmp_path, "first", 21, 0.3, 1, source_hash="source-a")
    _write_probe_run(tmp_path, "second", 22, 0.3, 1, source_hash="source-b")
    with pytest.raises(ValueError, match="source snapshot"):
        audit_tempering_probe(tmp_path)


def test_full_gate_rejects_chains_jointly_stuck_at_one_policy(tmp_path: Path) -> None:
    _write_probe_run(tmp_path, "first", 31, 0.5, 2)
    _write_probe_run(tmp_path, "second", 32, 0.5, 2)
    audit = audit_tempering_chains(
        tmp_path,
        TemperingGateThresholds(
            min_return_ess=0.0,
            min_swap_acceptance=0.0,
            min_cold_walkers=0,
            min_round_trips_per_chain=0,
            max_coordinate_tv=1.0,
        ),
    )
    assert audit["passed"] is False
    assert audit["checks"]["within_chain_policy_movement"] is False


def test_schema_v2_requires_every_raw_counter_group(tmp_path: Path) -> None:
    for name, seed in (("first", 41), ("second", 42)):
        _write_probe_run(tmp_path, name, seed, 0.25, 1)
        _upgrade_probe_run_to_schema_v2(tmp_path, name)
        result_path = tmp_path / name / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for key in (
            "mcmc_post_burn_local_proposals",
            "mcmc_post_burn_local_accepts",
            "mcmc_post_burn_local_acceptance_rates",
        ):
            result.pop(key)
        result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="schema v2 requires diagnostic counters"):
        audit_tempering_probe(tmp_path)


def test_schema_v2_verifies_frozen_guide_artifact_hash(tmp_path: Path) -> None:
    for name, seed in (("first", 51), ("second", 52)):
        _write_probe_run(tmp_path, name, seed, 0.25, 1)
        _upgrade_probe_run_to_schema_v2(tmp_path, name)

    assert audit_tempering_probe(tmp_path)["passed"] is True

    guide_path = tmp_path / "second" / "mcmc_guide_probabilities.npy"
    guide = np.load(guide_path, allow_pickle=False)
    guide[0, 0] += 0.01
    guide[0, 1] -= 0.01
    np.save(guide_path, guide, allow_pickle=False)
    with pytest.raises(ValueError, match="does not match its declared SHA-256"):
        audit_tempering_probe(tmp_path)


def test_schema_v2_rejects_post_burn_counters_exceeding_whole_run(
    tmp_path: Path,
) -> None:
    for name, seed in (("first", 61), ("second", 62)):
        _write_probe_run(tmp_path, name, seed, 0.25, 1)
        _upgrade_probe_run_to_schema_v2(tmp_path, name)
    result_path = tmp_path / "first" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["mcmc_post_burn_local_proposals"] = [9, 9]
    result["mcmc_post_burn_local_accepts"] = [3, 3]
    result["mcmc_post_burn_local_acceptance_rates"] = [1.0 / 3.0, 1.0 / 3.0]
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="post-burn local counters exceed"):
        audit_tempering_probe(tmp_path)


def test_fresh_schema_v3_tempering_outputs_are_auditable(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    for name, mcmc_seed in (("first", 71), ("second", 72)):
        run_experiment(
            ExperimentConfig(
                run_id=name,
                method="ppo_policy_tempering",
                map_path=str(MAP_PATH),
                output_root=str(output_root),
                training_seed=9,
                transition_budget=32,
                evaluation_policy_samples=4,
                ppo_batch_transitions=16,
                ppo_update_epochs=1,
                ppo_minibatch_size=8,
                mcmc_iterations=20,
                mcmc_burn_in=4,
                mcmc_thinning=2,
                mcmc_temperatures=2,
                mcmc_seed=mcmc_seed,
            )
        )

    audit = audit_tempering_probe(output_root)
    assert audit["scope"] == "ladder_probe_only_not_full_mixing_validation"
