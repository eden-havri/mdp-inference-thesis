from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_grid_spec
from .gridworld import gridworld_mdp
from .mdp import TapeBank
from .policy_mcmc import autocorrelation_effective_sample_size
from .recovery_blocks import (
    RECOVERY_BLOCK_CONSTRUCTION,
    RECOVERY_FRINGE_BLOCK_SIZE,
)


@dataclass(frozen=True)
class TemperingGateThresholds:
    max_split_return_rhat: float = 1.05
    min_return_ess: float = 100.0
    min_swap_acceptance: float = 0.10
    min_cold_walkers: int = 4
    min_round_trips_per_chain: int = 1
    max_coordinate_tv: float = 0.15
    min_unique_policy_fraction: float = 0.10
    min_mean_policy_hamming_jump: float = 0.01
    min_pooled_nonmodal_mass_for_switch: float = 0.02
    max_unswitched_nondegenerate_states_per_chain: int = 0


@dataclass(frozen=True)
class TemperingProbeThresholds:
    required_chain_count: int = 2
    min_post_burn_swap_acceptance: float = 0.20
    min_endpoint_transitions_per_chain: int = 1


@dataclass(frozen=True)
class TemperingRescueThresholds:
    required_chain_count: int = 1
    high_beta_fraction: float = 0.25
    min_post_burn_high_beta_global_move_accepts: int = 1
    min_post_burn_final_edge_swap_accepts: int = 1
    min_cold_walkers: int = 2
    min_cold_unique_policies: int = 2
    structural_bridge_beta_min: float = 1.9
    structural_bridge_beta_max: float = 13.5
    min_post_burn_bridge_block_proposals_per_block: int = 50
    min_post_burn_path_block_move_accepts: int = 1
    min_post_burn_fringe_block_move_coverage: float = 0.50
    min_post_burn_all_edge_swap_acceptance: float = 0.20
    min_post_burn_endpoint_transitions: int = 1


@dataclass(frozen=True)
class _TemperingArtifacts:
    run_dirs: list[Path]
    configs: list[dict[str, Any]]
    results: list[dict[str, Any]]
    policies: list[np.ndarray]
    mdp: Any


_COUNTER_GROUPS = (
    (
        "mcmc_local_proposals",
        "mcmc_local_accepts",
        "mcmc_local_acceptance_rates",
        0,
        2,
    ),
    (
        "mcmc_swap_proposals",
        "mcmc_swap_accepts",
        "mcmc_swap_acceptance_rates",
        -1,
        2,
    ),
    (
        "mcmc_post_burn_local_proposals",
        "mcmc_post_burn_local_accepts",
        "mcmc_post_burn_local_acceptance_rates",
        0,
        2,
    ),
    (
        "mcmc_global_proposals",
        "mcmc_global_accepts",
        "mcmc_global_acceptance_rates",
        0,
        3,
    ),
    (
        "mcmc_post_burn_global_proposals",
        "mcmc_post_burn_global_accepts",
        "mcmc_post_burn_global_acceptance_rates",
        0,
        3,
    ),
    (
        "mcmc_path_proposals",
        "mcmc_path_accepts",
        "mcmc_path_acceptance_rates",
        0,
        4,
    ),
    (
        "mcmc_post_burn_path_proposals",
        "mcmc_post_burn_path_accepts",
        "mcmc_post_burn_path_acceptance_rates",
        0,
        4,
    ),
    (
        "mcmc_post_burn_swap_proposals",
        "mcmc_post_burn_swap_accepts",
        "mcmc_post_burn_swap_acceptance_rates",
        -1,
        2,
    ),
)


def _array_sha256(value: np.ndarray) -> str:
    """Match the canonical NumPy artifact digest written by experiment.py."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def split_rhat(chains: np.ndarray) -> float:
    """Classical split-Rhat for equal-length scalar chains."""

    values = np.asarray(chains, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("chains must have shape [at least 2 chains, at least 4 draws]")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    means = np.mean(split, axis=1)
    if within == 0.0:
        return 1.0 if np.allclose(means, means[0], atol=0.0, rtol=0.0) else float("inf")
    between = half * float(np.var(means, ddof=1))
    variance = ((half - 1.0) / half) * within + between / half
    return float(np.sqrt(max(variance / within, 0.0)))


def coordinate_total_variation(
    policy_chains: np.ndarray,
    decision_states: tuple[int, ...],
    num_actions: int,
) -> np.ndarray:
    """Pairwise empirical TV distances for each policy coordinate."""

    policies = np.asarray(policy_chains, dtype=np.int64)
    if policies.ndim != 3 or policies.shape[0] < 2 or policies.shape[1] == 0:
        raise ValueError("policy_chains must have shape [at least 2 chains, draws, states]")
    states = np.asarray(decision_states, dtype=np.int64)
    if len(states) == 0 or np.any(states < 0) or np.any(states >= policies.shape[2]):
        raise ValueError("decision_states must index at least one valid state")
    if num_actions < 2 or np.any((policies[:, :, states] < 0) | (policies[:, :, states] >= num_actions)):
        raise ValueError("policy actions must lie in the declared action space")

    probabilities = np.empty((policies.shape[0], len(states), num_actions), dtype=np.float64)
    for chain in range(policies.shape[0]):
        for decision, state in enumerate(states):
            probabilities[chain, decision] = np.bincount(
                policies[chain, :, state], minlength=num_actions
            ) / float(policies.shape[1])

    distances = []
    for first in range(policies.shape[0]):
        for second in range(first + 1, policies.shape[0]):
            distances.append(
                0.5 * np.abs(probabilities[first] - probabilities[second]).sum(axis=1)
            )
    return np.asarray(distances, dtype=np.float64)


def _realized_tape_seed(
    config: dict[str, Any], result: dict[str, Any]
) -> int | None:
    tapes_count = int(config.get("mcmc_tapes", 0))
    if tapes_count == 0:
        return None
    tape_seed = result.get("mcmc_tape_seed")
    if tape_seed is None:
        tape_seed = config.get("mcmc_tape_seed")
    if tape_seed is None:
        tape_seed = int(config["training_seed"]) + 5_000_003
    return int(tape_seed)


def _target_scores(
    mdp,
    policies: np.ndarray,
    config: dict[str, Any],
    result: dict[str, Any],
) -> np.ndarray:
    tapes_count = int(config.get("mcmc_tapes", 0))
    if tapes_count == 0:
        return np.asarray([mdp.expected_return(policy) for policy in policies])
    tape_seed = _realized_tape_seed(config, result)
    assert tape_seed is not None
    tapes = TapeBank.sample(tapes_count, mdp.horizon, tape_seed)
    return np.asarray([mdp.tape_returns(policy, tapes).mean() for policy in policies])


def _protocol_signature(config: dict[str, Any]) -> str:
    allowed_differences = {
        "run_id",
        "output_root",
        "mcmc_seed",
        "mcmc_prior_initialize_hot_replicas",
    }
    fixed = {
        key: value for key, value in config.items() if key not in allowed_differences
    }
    return json.dumps(fixed, sort_keys=True, separators=(",", ":"))


def _result_schema_version(result: dict[str, Any]) -> int:
    version = result.get("result_schema_version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("result_schema_version must be a nonnegative integer")
    if version > 4:
        raise ValueError(f"unsupported result_schema_version: {version}")
    return version


def _validate_recorded_counters(
    result: dict[str, Any], num_temperatures: int, schema_version: int
) -> None:
    validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (
        proposal_key,
        accept_key,
        rate_key,
        length_offset,
        required_version,
    ) in _COUNTER_GROUPS:
        expected_length = num_temperatures + length_offset
        availability = [key in result for key in (proposal_key, accept_key, rate_key)]
        if not any(availability):
            if schema_version >= required_version:
                raise ValueError(
                    f"schema v{schema_version} requires diagnostic counters for {rate_key}"
                )
            continue
        if availability == [False, False, True]:
            if schema_version >= required_version:
                raise ValueError(
                    f"schema v{schema_version} requires raw counters for {rate_key}"
                )
            rates = np.asarray(result[rate_key], dtype=np.float64)
            if rates.shape != (expected_length,) or np.any(~np.isfinite(rates)):
                raise ValueError(f"invalid legacy rate-only diagnostics for {rate_key}")
            if np.any((rates < 0.0) | (rates > 1.0)):
                raise ValueError(f"acceptance rates lie outside [0, 1] for {rate_key}")
            continue
        if not all(availability):
            raise ValueError(f"incomplete diagnostic counter group for {rate_key}")
        raw_proposals = np.asarray(result[proposal_key])
        raw_accepts = np.asarray(result[accept_key])
        if raw_proposals.dtype.kind not in "iu" or raw_accepts.dtype.kind not in "iu":
            raise ValueError(f"diagnostic counters must be integers for {rate_key}")
        proposals = raw_proposals.astype(np.int64, copy=False)
        accepts = raw_accepts.astype(np.int64, copy=False)
        rates = np.asarray(result[rate_key], dtype=np.float64)
        if proposals.shape != (expected_length,) or accepts.shape != proposals.shape:
            raise ValueError(f"invalid counter shape for {rate_key}")
        if rates.shape != proposals.shape or np.any(proposals < 0):
            raise ValueError(f"proposal counts must be nonnegative for {rate_key}")
        if np.any(accepts < 0) or np.any(accepts > proposals):
            raise ValueError(f"invalid acceptance counts for {rate_key}")
        expected_rates = np.divide(
            accepts,
            proposals,
            out=np.zeros_like(accepts, dtype=np.float64),
            where=proposals > 0,
        )
        if not np.allclose(rates, expected_rates, rtol=0.0, atol=1e-12):
            raise ValueError(f"recorded rates disagree with counts for {rate_key}")
        validated[proposal_key] = (proposals, accepts)

    for kind in ("local", "global", "path", "swap"):
        full = validated.get(f"mcmc_{kind}_proposals")
        post_burn = validated.get(f"mcmc_post_burn_{kind}_proposals")
        if full is not None and post_burn is not None:
            if np.any(post_burn[0] > full[0]) or np.any(post_burn[1] > full[1]):
                raise ValueError(f"post-burn {kind} counters exceed whole-run counters")


def _validate_schema_v3_diagnostics(
    config: dict[str, Any], result: dict[str, Any], num_temperatures: int
) -> None:
    """Validate diagnostics introduced with the global-refresh protocol."""

    required_config_fields = (
        "mcmc_swap_interval",
        "mcmc_swap_sweeps",
        "mcmc_global_refresh_probability",
        "mcmc_global_map_weight",
        "mcmc_global_guide_weight",
        "mcmc_global_uniform_weight",
    )
    required_result_fields = (
        "mcmc_swap_interval",
        "mcmc_swap_sweeps",
        "mcmc_global_refresh_probability",
        "mcmc_global_mixture_weights",
        "mcmc_global_move_accepts",
        "mcmc_post_burn_global_move_accepts",
        "mcmc_walker_temperature_visit_fractions",
        "mcmc_walker_endpoint_visits",
        "mcmc_walker_endpoint_transitions",
        "mcmc_walker_round_trips",
        "mcmc_total_round_trips",
        "mcmc_cold_walker_count",
        "mcmc_unique_policy_samples",
    )
    missing_config = [key for key in required_config_fields if key not in config]
    missing_result = [key for key in required_result_fields if key not in result]
    if missing_config or missing_result:
        missing = missing_config + missing_result
        raise ValueError(
            "schema v3 requires global-refresh and walker diagnostics: "
            + ", ".join(missing)
        )

    for move_key, accept_key in (
        ("mcmc_global_move_accepts", "mcmc_global_accepts"),
        (
            "mcmc_post_burn_global_move_accepts",
            "mcmc_post_burn_global_accepts",
        ),
    ):
        raw_moves = np.asarray(result[move_key])
        if raw_moves.dtype.kind not in "iu" or raw_moves.shape != (num_temperatures,):
            raise ValueError(f"{move_key} must contain one integer count per replica")
        moves = raw_moves.astype(np.int64, copy=False)
        accepts = np.asarray(result[accept_key], dtype=np.int64)
        if np.any(moves < 0) or np.any(moves > accepts):
            raise ValueError(f"{move_key} must lie between zero and global accepts")

    if np.any(
        np.asarray(result["mcmc_post_burn_global_move_accepts"], dtype=np.int64)
        > np.asarray(result["mcmc_global_move_accepts"], dtype=np.int64)
    ):
        raise ValueError("post-burn global move counts exceed whole-run counts")

    endpoint_visits = np.asarray(result["mcmc_walker_endpoint_visits"])
    if endpoint_visits.shape != (num_temperatures, 2) or endpoint_visits.dtype.kind not in "biu":
        raise ValueError("walker endpoint visits must have shape [temperatures, 2]")
    if np.any((endpoint_visits != 0) & (endpoint_visits != 1)):
        raise ValueError("walker endpoint visits must be Boolean")
    endpoint_visits = endpoint_visits.astype(bool, copy=False)

    raw_transitions = np.asarray(result["mcmc_walker_endpoint_transitions"])
    raw_round_trips = np.asarray(result["mcmc_walker_round_trips"])
    if (
        raw_transitions.dtype.kind not in "iu"
        or raw_round_trips.dtype.kind not in "iu"
        or raw_transitions.shape != (num_temperatures,)
        or raw_round_trips.shape != (num_temperatures,)
    ):
        raise ValueError("walker flow counts must contain one integer per walker")
    transitions = raw_transitions.astype(np.int64, copy=False)
    round_trips = raw_round_trips.astype(np.int64, copy=False)
    if np.any(transitions < 0) or np.any(round_trips < 0):
        raise ValueError("walker flow counts must be nonnegative")
    if np.any(round_trips != transitions // 2):
        raise ValueError("walker round trips disagree with endpoint transitions")
    if np.any((transitions > 0) != np.all(endpoint_visits, axis=1)):
        raise ValueError("walker endpoint visits disagree with endpoint transitions")

    total_round_trips = result["mcmc_total_round_trips"]
    cold_walker_count = result["mcmc_cold_walker_count"]
    if (
        isinstance(total_round_trips, bool)
        or not isinstance(total_round_trips, int)
        or total_round_trips != int(round_trips.sum())
    ):
        raise ValueError("total round trips disagree with per-walker counts")
    expected_cold_walkers = int(np.count_nonzero(endpoint_visits[:, 1]))
    if (
        isinstance(cold_walker_count, bool)
        or not isinstance(cold_walker_count, int)
        or cold_walker_count != expected_cold_walkers
    ):
        raise ValueError("cold-walker count disagrees with endpoint visits")

    unique_policy_samples = result["mcmc_unique_policy_samples"]
    if (
        isinstance(unique_policy_samples, bool)
        or not isinstance(unique_policy_samples, int)
        or unique_policy_samples < 1
        or unique_policy_samples > int(result.get("mcmc_samples", 0))
    ):
        raise ValueError("invalid cold-replica unique-policy count")

    visit_fractions = np.asarray(
        result["mcmc_walker_temperature_visit_fractions"], dtype=np.float64
    )
    if (
        visit_fractions.shape != (num_temperatures, num_temperatures)
        or not np.all(np.isfinite(visit_fractions))
        or np.any((visit_fractions < 0.0) | (visit_fractions > 1.0))
        or not np.allclose(
            visit_fractions.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("invalid walker temperature-visit fractions")


def _validate_block_counter_group(
    result: dict[str, Any],
    *,
    num_blocks: int,
    num_temperatures: int,
    post_burn: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one schema-v4 block counter matrix.

    JSON cannot retain the second dimension of a NumPy array with zero rows,
    so the disabled-kernel representation ``[]`` is accepted only when the
    declared block catalog is empty.
    """

    prefix = "mcmc_post_burn_block" if post_burn else "mcmc_block"
    proposal_key = f"{prefix}_proposals"
    accept_key = f"{prefix}_accepts"
    move_key = f"{prefix}_move_accepts"
    rate_key = f"{prefix}_acceptance_rates"
    required = (proposal_key, accept_key, move_key, rate_key)
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            "schema v4 requires block diagnostic counters: " + ", ".join(missing)
        )

    expected_shape = (num_blocks, num_temperatures)
    integer_arrays: dict[str, np.ndarray] = {}
    for key in (proposal_key, accept_key, move_key):
        raw = np.asarray(result[key])
        if num_blocks == 0 and raw.shape == (0,):
            raw = np.empty(expected_shape, dtype=np.int64)
        if raw.dtype.kind not in "iu" or raw.shape != expected_shape:
            raise ValueError(
                f"{key} must have integer shape [blocks, temperatures]"
            )
        integer_arrays[key] = raw.astype(np.int64, copy=False)

    proposals = integer_arrays[proposal_key]
    accepts = integer_arrays[accept_key]
    moves = integer_arrays[move_key]
    if np.any(proposals < 0):
        raise ValueError(f"{proposal_key} must be nonnegative")
    if np.any(accepts < 0) or np.any(accepts > proposals):
        raise ValueError(f"{accept_key} must lie between zero and proposals")
    if np.any(moves < 0) or np.any(moves > accepts):
        raise ValueError(f"{move_key} must lie between zero and accepts")

    rates = np.asarray(result[rate_key], dtype=np.float64)
    if num_blocks == 0 and rates.shape == (0,):
        rates = np.empty(expected_shape, dtype=np.float64)
    if rates.shape != expected_shape or np.any(~np.isfinite(rates)):
        raise ValueError(
            f"{rate_key} must have finite shape [blocks, temperatures]"
        )
    expected_rates = np.divide(
        accepts,
        proposals,
        out=np.zeros_like(accepts, dtype=np.float64),
        where=proposals > 0,
    )
    if not np.allclose(rates, expected_rates, rtol=0.0, atol=1e-12):
        raise ValueError(f"recorded rates disagree with counts for {rate_key}")
    return proposals, accepts, moves


def _validate_schema_v4_diagnostics(
    config: dict[str, Any],
    result: dict[str, Any],
    *,
    num_temperatures: int,
    num_blocks: int,
) -> None:
    """Validate diagnostics introduced with fixed structural block refreshes."""

    if (
        "mcmc_block_refresh_probability" not in config
        or "mcmc_path_refresh_probability" not in config
    ):
        raise ValueError("schema v4 requires structural refresh probabilities")
    required_result = (
        "mcmc_block_refresh_probability",
        "mcmc_path_refresh_probability",
        "mcmc_block_repeats",
        "mcmc_block_sizes",
        "mcmc_lazy_iterations",
        "mcmc_value_evaluations",
        "mcmc_score_computations",
        "mcmc_score_cache_hits",
    )
    missing = [key for key in required_result if key not in result]
    if missing:
        raise ValueError(
            "schema v4 requires block-refresh diagnostics: " + ", ".join(missing)
        )

    block_probability = config["mcmc_block_refresh_probability"]
    path_probability = config["mcmc_path_refresh_probability"]
    if (
        isinstance(block_probability, bool)
        or not isinstance(block_probability, (int, float))
        or not np.isfinite(block_probability)
        or not 0.0 <= float(block_probability) <= 1.0
    ):
        raise ValueError("mcmc_block_refresh_probability must lie in [0, 1]")
    if (
        isinstance(path_probability, bool)
        or not isinstance(path_probability, (int, float))
        or not np.isfinite(path_probability)
        or not 0.0 <= float(path_probability) <= 1.0
    ):
        raise ValueError("mcmc_path_refresh_probability must lie in [0, 1]")
    if result["mcmc_block_refresh_probability"] != block_probability:
        raise ValueError(
            "recorded block-refresh probability disagrees with configuration"
        )
    if result["mcmc_path_refresh_probability"] != path_probability:
        raise ValueError(
            "recorded path-refresh probability disagrees with configuration"
        )
    global_probability = float(config["mcmc_global_refresh_probability"])
    if (
        global_probability + float(path_probability) + float(block_probability)
        > 1.0 + 1e-15
    ):
        raise ValueError("structural refresh probabilities exceed one")

    full_proposals, full_accepts, full_moves = _validate_block_counter_group(
        result,
        num_blocks=num_blocks,
        num_temperatures=num_temperatures,
        post_burn=False,
    )
    post_proposals, post_accepts, post_moves = _validate_block_counter_group(
        result,
        num_blocks=num_blocks,
        num_temperatures=num_temperatures,
        post_burn=True,
    )
    if (
        np.any(post_proposals > full_proposals)
        or np.any(post_accepts > full_accepts)
        or np.any(post_moves > full_moves)
    ):
        raise ValueError("post-burn block counters exceed whole-run counters")

    structural_enabled = (
        float(block_probability) > 0.0 or float(path_probability) > 0.0
    )
    if not structural_enabled:
        if num_blocks != 0 or np.any(full_proposals != 0):
            raise ValueError(
                "disabled structural refresh must not record a block catalog or proposals"
            )
    elif num_blocks == 0:
        raise ValueError("enabled structural refresh requires a block catalog")

    expected_repeats = (
        []
        if num_blocks == 0
        else (
            [1] * num_blocks
            if config.get("mcmc_block_repeats") is None
            else config["mcmc_block_repeats"]
        )
    )
    recorded_repeats = result["mcmc_block_repeats"]
    if recorded_repeats != expected_repeats:
        raise ValueError("recorded block repeats disagree with configuration")
    if num_blocks == 0:
        repeats = np.empty(0, dtype=np.int64)
    else:
        raw_repeats = np.asarray(recorded_repeats)
        if (
            raw_repeats.dtype.kind not in "iu"
            or raw_repeats.shape != (num_blocks,)
            or np.any(raw_repeats <= 0)
        ):
            raise ValueError("block repeats must be positive integers")
        repeats = raw_repeats.astype(np.int64, copy=False)
        if (
            np.any(full_proposals % repeats[:, None] != 0)
            or np.any(post_proposals % repeats[:, None] != 0)
        ):
            raise ValueError("block proposal counts disagree with fixed repeats")
        full_sweeps = full_proposals // repeats[:, None]
        post_sweeps = post_proposals // repeats[:, None]
        if (
            np.any(full_sweeps != full_sweeps[:1])
            or np.any(post_sweeps != post_sweeps[:1])
        ):
            raise ValueError("ordered block sweeps have inconsistent proposal counts")

    local_proposals = np.asarray(result["mcmc_local_proposals"], dtype=np.int64)
    global_proposals = np.asarray(result["mcmc_global_proposals"], dtype=np.int64)
    path_proposals = np.asarray(result["mcmc_path_proposals"], dtype=np.int64)
    path_accepts = np.asarray(result["mcmc_path_accepts"], dtype=np.int64)
    path_moves = np.asarray(result["mcmc_path_move_accepts"], dtype=np.int64)
    post_local = np.asarray(
        result["mcmc_post_burn_local_proposals"], dtype=np.int64
    )
    post_global = np.asarray(
        result["mcmc_post_burn_global_proposals"], dtype=np.int64
    )
    post_path = np.asarray(
        result["mcmc_post_burn_path_proposals"], dtype=np.int64
    )
    post_path_accepts = np.asarray(
        result["mcmc_post_burn_path_accepts"], dtype=np.int64
    )
    post_path_moves = np.asarray(
        result["mcmc_post_burn_path_move_accepts"], dtype=np.int64
    )
    if (
        np.any(path_moves > path_accepts)
        or np.any(post_path_moves > post_path_accepts)
        or np.any(post_path_moves > path_moves)
    ):
        raise ValueError("path move counts disagree with path acceptances")
    if float(path_probability) == 0.0 and np.any(path_proposals != 0):
        raise ValueError("disabled path refresh recorded proposals")
    if float(block_probability) == 0.0 and np.any(full_proposals != 0):
        raise ValueError("disabled block sweep recorded proposals")
    block_by_replica = (
        np.zeros(num_temperatures, dtype=np.int64)
        if num_blocks == 0
        else full_proposals[0] // repeats[0]
    )
    post_block_by_replica = (
        np.zeros(num_temperatures, dtype=np.int64)
        if num_blocks == 0
        else post_proposals[0] // repeats[0]
    )
    branch_totals = (
        local_proposals + global_proposals + path_proposals + block_by_replica
    )
    post_branch_totals = (
        post_local + post_global + post_path + post_block_by_replica
    )
    if np.any(branch_totals != branch_totals[0]):
        raise ValueError("whole-run proposal branches disagree across replicas")
    if np.any(post_branch_totals != post_branch_totals[0]):
        raise ValueError("post-burn proposal branches disagree across replicas")

    lazy_iterations = result["mcmc_lazy_iterations"]
    if (
        isinstance(lazy_iterations, bool)
        or not isinstance(lazy_iterations, int)
        or not 0 <= lazy_iterations <= int(config["mcmc_iterations"])
    ):
        raise ValueError("invalid whole-sweep lazy-iteration count")
    expected_active = int(config["mcmc_iterations"]) - lazy_iterations
    if int(branch_totals[0]) != expected_active:
        raise ValueError("proposal branches disagree with active iteration count")
    value_evaluations = result["mcmc_value_evaluations"]
    expected_value_evaluations = (
        num_temperatures
        + int(local_proposals.sum())
        + int(global_proposals.sum())
        + int(path_proposals.sum())
        + int(full_proposals.sum())
    )
    if (
        isinstance(value_evaluations, bool)
        or not isinstance(value_evaluations, int)
        or value_evaluations != expected_value_evaluations
    ):
        raise ValueError("value-evaluation count disagrees with proposal accounting")
    score_computations = result["mcmc_score_computations"]
    score_cache_hits = result["mcmc_score_cache_hits"]
    if (
        isinstance(score_computations, bool)
        or not isinstance(score_computations, int)
        or score_computations < 1
        or isinstance(score_cache_hits, bool)
        or not isinstance(score_cache_hits, int)
        or score_cache_hits < 0
        or score_computations + score_cache_hits != value_evaluations
    ):
        raise ValueError("score-cache accounting disagrees with value evaluations")
    post_iterations = int(config["mcmc_iterations"]) - int(config["mcmc_burn_in"])
    if int(post_branch_totals[0]) > post_iterations:
        raise ValueError("post-burn proposal branches exceed post-burn iterations")


def _validate_realized_ladder(
    config: dict[str, Any], result: dict[str, Any]
) -> tuple[float, ...]:
    num_temperatures = int(config["mcmc_temperatures"])
    beta = float(config["beta"])
    ladder = np.asarray(result["mcmc_beta_ladder"], dtype=np.float64)
    if ladder.shape != (num_temperatures,) or not np.all(np.isfinite(ladder)):
        raise ValueError("realized beta ladder has the wrong shape or non-finite entries")
    if ladder[0] != 0.0 or not np.isclose(ladder[-1], beta, rtol=0.0, atol=1e-12):
        raise ValueError("realized beta ladder endpoints do not match the target")
    if np.any(np.diff(ladder) <= 0.0):
        raise ValueError("realized beta ladder must be strictly increasing")

    custom = config.get("mcmc_beta_ladder")
    if custom is None:
        fractions = np.linspace(0.0, 1.0, num_temperatures, dtype=np.float64)
        expected = beta * fractions ** float(config.get("mcmc_ladder_power", 2.0))
        expected[0] = 0.0
        expected[-1] = beta
        expected_kind = "power"
    else:
        expected = np.asarray(custom, dtype=np.float64)
        expected_kind = "custom"
    if expected.shape != ladder.shape or not np.allclose(
        ladder, expected, rtol=0.0, atol=1e-12
    ):
        raise ValueError("realized beta ladder disagrees with the configured ladder")
    if _result_schema_version(result) >= 2:
        if result.get("mcmc_ladder_kind") != expected_kind:
            raise ValueError("recorded ladder kind disagrees with the configuration")
        if int(result.get("mcmc_temperatures", -1)) != num_temperatures:
            raise ValueError("recorded temperature count disagrees with the configuration")
    return tuple(float(value) for value in ladder)


def _validate_guide_artifact(
    run_dir: Path,
    result: dict[str, Any],
    expected_shape: tuple[int, int],
) -> str | None:
    declared_hash = result.get("mcmc_guide_probabilities_sha256")
    artifact_name = result.get("mcmc_guide_probabilities_artifact")
    required = _result_schema_version(result) >= 2
    if declared_hash is None and artifact_name is None and not required:
        return None
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError("missing or invalid frozen-guide SHA-256")
    if not isinstance(artifact_name, str):
        raise ValueError("missing frozen-guide artifact name")
    relative = Path(artifact_name)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("frozen-guide artifact must be a file inside its run directory")
    artifact_path = run_dir / relative
    if not artifact_path.is_file():
        raise ValueError(f"missing frozen-guide artifact: {artifact_path}")
    guide = np.load(artifact_path, allow_pickle=False)
    if guide.shape != expected_shape or not np.issubdtype(guide.dtype, np.floating):
        raise ValueError("frozen-guide artifact has the wrong shape or dtype")
    if np.any(guide < 0.0) or not np.all(np.isfinite(guide)):
        raise ValueError("frozen-guide probabilities must be finite and nonnegative")
    if not np.allclose(guide.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("frozen-guide probability rows must sum to one")
    if _array_sha256(guide) != declared_hash:
        raise ValueError("frozen-guide artifact does not match its declared SHA-256")
    return declared_hash


def _validate_block_catalog_artifact(
    run_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    expected_decisions: int,
) -> tuple[str | None, int]:
    """Validate the immutable structural block catalog recorded by schema v4."""

    if _result_schema_version(result) < 4:
        return None, 0
    probability = float(config["mcmc_block_refresh_probability"])
    path_probability = float(config.get("mcmc_path_refresh_probability", 0.0))
    enabled = probability > 0.0 or path_probability > 0.0
    metadata_keys = (
        "mcmc_block_catalog_sha256",
        "mcmc_block_catalog_artifact",
        "mcmc_block_catalog_kind",
    )
    availability = [key in result for key in metadata_keys]
    sizes = result.get("mcmc_block_sizes")
    if not enabled:
        if any(availability):
            raise ValueError("disabled block refresh must not record a block catalog")
        if sizes != []:
            raise ValueError("disabled block refresh must record empty block sizes")
        return None, 0

    if not all(availability):
        missing = [key for key, present in zip(metadata_keys, availability) if not present]
        raise ValueError(
            "enabled block refresh requires catalog metadata: " + ", ".join(missing)
        )
    declared_hash = result["mcmc_block_catalog_sha256"]
    artifact_name = result["mcmc_block_catalog_artifact"]
    catalog_kind = result["mcmc_block_catalog_kind"]
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise ValueError("missing or invalid block-catalog SHA-256")
    try:
        int(declared_hash, 16)
    except ValueError as error:
        raise ValueError("missing or invalid block-catalog SHA-256") from error
    if catalog_kind != RECOVERY_BLOCK_CONSTRUCTION:
        raise ValueError("unsupported block-catalog construction")
    if not isinstance(artifact_name, str):
        raise ValueError("missing block-catalog artifact name")
    relative = Path(artifact_name)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("block-catalog artifact must be a file inside its run directory")
    artifact_path = run_dir / relative
    if not artifact_path.is_file():
        raise ValueError(f"missing block-catalog artifact: {artifact_path}")

    if (
        not isinstance(sizes, list)
        or len(sizes) < 2
        or any(isinstance(size, bool) or not isinstance(size, int) for size in sizes)
        or any(size <= 0 for size in sizes)
        or any(size > RECOVERY_FRINGE_BLOCK_SIZE for size in sizes[1:])
    ):
        raise ValueError("invalid modal-path paired-fringe block sizes")
    catalog = np.load(artifact_path, allow_pickle=False)
    if catalog.dtype != np.dtype(bool) or catalog.shape != (len(sizes), expected_decisions):
        raise ValueError("block-catalog artifact must have Boolean shape [blocks, decisions]")
    realized_sizes = catalog.sum(axis=1, dtype=np.int64).tolist()
    if realized_sizes != sizes:
        raise ValueError("block-catalog sizes disagree with the saved artifact")
    if np.any(catalog.sum(axis=0) > 1):
        raise ValueError("modal-path paired-fringe blocks must be disjoint")
    if _array_sha256(catalog) != declared_hash:
        raise ValueError("block-catalog artifact does not match its declared SHA-256")
    return declared_hash, len(sizes)


def _load_tempering_artifacts(
    output_root: Path, *, min_chains: int = 2
) -> _TemperingArtifacts:
    run_dirs = sorted(path.parent for path in output_root.glob("*/DONE"))
    if len(run_dirs) < min_chains:
        raise ValueError(
            f"at least {min_chains} completed chain directories are required"
        )

    configs = [
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        for path in run_dirs
    ]
    results = [
        json.loads((path / "result.json").read_text(encoding="utf-8"))
        for path in run_dirs
    ]
    environments = [
        json.loads((path / "environment.json").read_text(encoding="utf-8"))
        for path in run_dirs
    ]
    policies = [np.load(path / "policy_particles.npy") for path in run_dirs]

    if {config.get("method") for config in configs} != {"ppo_policy_tempering"}:
        raise ValueError("all audited runs must use ppo_policy_tempering")
    if {result.get("method") for result in results} != {"ppo_policy_tempering"}:
        raise ValueError("result methods do not match policy tempering")
    if len({_protocol_signature(config) for config in configs}) != 1:
        raise ValueError("chains use incompatible sampler or guide protocols")

    source_hashes = {
        environment.get("source_snapshot_sha256") for environment in environments
    }
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError("chains must use the same recorded source snapshot")
    realized_mcmc_seeds = [int(result["mcmc_seed"]) for result in results]
    if len(set(realized_mcmc_seeds)) != len(realized_mcmc_seeds):
        raise ValueError("independent chains must use distinct realized MCMC seeds")

    schema_signatures = {
        (
            _result_schema_version(result),
            tuple(
                key in result
                for group in _COUNTER_GROUPS
                for key in group[:3]
            ),
        )
        for result in results
    }
    if len(schema_signatures) != 1:
        raise ValueError("chains use mixed diagnostic schemas")

    sample_counts = {len(chain) for chain in policies}
    if len(sample_counts) != 1 or next(iter(sample_counts)) == 0:
        raise ValueError("all chains must contain the same positive number of draws")
    if any(chain.ndim != 2 for chain in policies):
        raise ValueError("policy particles must have shape [draws, states]")
    state_counts = {chain.shape[1] for chain in policies}
    if len(state_counts) != 1:
        raise ValueError("policy particles must have compatible [draws, states] shapes")

    map_paths = [path / "map.json" for path in run_dirs]
    specs_and_hashes = [load_grid_spec(path) for path in map_paths]
    map_hashes = {digest for _, digest in specs_and_hashes}
    if len(map_hashes) != 1:
        raise ValueError("all chains must use the same map")
    mdp = gridworld_mdp(specs_and_hashes[0][0])
    if state_counts != {mdp.num_states}:
        raise ValueError("policy particle state dimension does not match the map")

    guide_hashes = {
        _validate_guide_artifact(
            run_dir,
            result,
            (mdp.num_states, mdp.num_actions),
        )
        for run_dir, result in zip(run_dirs, results)
    }
    if len(guide_hashes) != 1:
        raise ValueError("chains must use the same frozen PPO guide")

    block_metadata = [
        _validate_block_catalog_artifact(
            run_dir,
            config,
            result,
            mdp.num_decisions,
        )
        for run_dir, config, result in zip(run_dirs, configs, results)
    ]
    if len({metadata[0] for metadata in block_metadata}) != 1:
        raise ValueError("chains must use the same frozen block catalog")

    target_signatures = {
        (
            result.get("mcmc_target"),
            float(config["beta"]),
            int(config.get("mcmc_tapes", 0)),
            _realized_tape_seed(config, result),
            _validate_realized_ladder(config, result),
        )
        for config, result in zip(configs, results)
    }
    if len(target_signatures) != 1:
        raise ValueError("chains must use the same target and realized beta ladder")

    for config, result, policies_for_chain, (_, num_blocks) in zip(
        configs, results, policies, block_metadata
    ):
        expected_target = (
            "exact_expected_return"
            if int(config.get("mcmc_tapes", 0)) == 0
            else "fixed_tape_sample_average"
        )
        if result.get("mcmc_target") != expected_target:
            raise ValueError("recorded MCMC target disagrees with the configuration")
        schema_version = _result_schema_version(result)
        if schema_version >= 2:
            if result.get("mcmc_diagnostic_window") != "post_burn":
                raise ValueError("schema v2 requires post-burn diagnostics")
            if int(result.get("mcmc_samples", -1)) != len(policies_for_chain):
                raise ValueError("recorded MCMC sample count disagrees with artifacts")
        if schema_version >= 3:
            _validate_schema_v3_diagnostics(
                config,
                result,
                int(config["mcmc_temperatures"]),
            )
            if result.get("mcmc_swap_interval") != config.get("mcmc_swap_interval"):
                raise ValueError("recorded swap interval disagrees with configuration")
            if result.get("mcmc_swap_sweeps") != config.get("mcmc_swap_sweeps"):
                raise ValueError("recorded swap-sweep count disagrees with configuration")
            if result.get("mcmc_global_refresh_probability") != config.get(
                "mcmc_global_refresh_probability"
            ):
                raise ValueError(
                    "recorded global-refresh probability disagrees with configuration"
                )
            expected_weights = {
                "map": config.get("mcmc_global_map_weight"),
                "guide_product": config.get("mcmc_global_guide_weight"),
                "uniform": config.get("mcmc_global_uniform_weight"),
            }
            if result.get("mcmc_global_mixture_weights") != expected_weights:
                raise ValueError(
                    "recorded global-refresh weights disagree with configuration"
                )
        if schema_version >= 4:
            _validate_schema_v4_diagnostics(
                config,
                result,
                num_temperatures=int(config["mcmc_temperatures"]),
                num_blocks=num_blocks,
            )
        _validate_recorded_counters(
            result,
            int(config["mcmc_temperatures"]),
            schema_version=schema_version,
        )

    return _TemperingArtifacts(
        run_dirs=run_dirs,
        configs=configs,
        results=results,
        policies=policies,
        mdp=mdp,
    )


def audit_tempering_rescue(
    output_root: Path,
    thresholds: TemperingRescueThresholds = TemperingRescueThresholds(),
) -> dict[str, Any]:
    """Audit the one-chain sparse-plateau rescue before a two-chain probe."""

    if not 0.0 < thresholds.high_beta_fraction <= 1.0:
        raise ValueError("high_beta_fraction must lie in (0, 1]")
    artifacts = _load_tempering_artifacts(output_root, min_chains=1)
    if len(artifacts.run_dirs) != thresholds.required_chain_count:
        raise ValueError(
            f"rescue gate requires exactly {thresholds.required_chain_count} completed chain"
        )
    result = artifacts.results[0]
    if _result_schema_version(result) < 3:
        raise ValueError("rescue gate requires schema-v3 global-refresh diagnostics")

    if _result_schema_version(result) >= 4 and float(
        result["mcmc_block_refresh_probability"]
    ) > 0.0:
        block_moves = np.asarray(
            result["mcmc_post_burn_block_move_accepts"], dtype=np.int64
        )
        num_blocks, num_temperatures = block_moves.shape
        betas = np.asarray(result["mcmc_beta_ladder"], dtype=np.float64)
        bridge = (betas >= thresholds.structural_bridge_beta_min) & (
            betas <= thresholds.structural_bridge_beta_max
        )
        if bridge.shape != (num_temperatures,) or not np.any(bridge):
            raise ValueError("structural bridge beta interval misses the ladder")
        if num_blocks == 0:
            raise ValueError("structural rescue requires at least one block")
        if (
            not np.isfinite(thresholds.structural_bridge_beta_min)
            or not np.isfinite(thresholds.structural_bridge_beta_max)
            or thresholds.structural_bridge_beta_min
            >= thresholds.structural_bridge_beta_max
            or thresholds.min_post_burn_bridge_block_proposals_per_block < 1
            or thresholds.min_post_burn_path_block_move_accepts < 1
            or not 0.0
            < thresholds.min_post_burn_fringe_block_move_coverage
            <= 1.0
            or not 0.0
            < thresholds.min_post_burn_all_edge_swap_acceptance
            <= 1.0
            or thresholds.min_post_burn_endpoint_transitions < 1
        ):
            raise ValueError("invalid structural rescue thresholds")

        block_proposals = np.asarray(
            result["mcmc_post_burn_block_proposals"], dtype=np.int64
        )
        block_accepts = np.asarray(
            result["mcmc_post_burn_block_accepts"], dtype=np.int64
        )
        bridge_proposals = block_proposals[:, bridge].sum(axis=1)
        bridge_accepts = block_accepts[:, bridge].sum(axis=1)
        bridge_move_accepts = block_moves[:, bridge].sum(axis=1)
        path_only_bridge_moves = int(
            np.asarray(
                result["mcmc_post_burn_path_move_accepts"], dtype=np.int64
            )[bridge].sum()
        )
        total_path_bridge_moves = int(
            path_only_bridge_moves + bridge_move_accepts[0]
        )
        fringe_move_coverage = float(
            np.count_nonzero(bridge_move_accepts[1:])
            / max(len(bridge_move_accepts) - 1, 1)
        )
        swap_proposals = np.asarray(
            result["mcmc_post_burn_swap_proposals"], dtype=np.int64
        )
        swap_accepts = np.asarray(
            result["mcmc_post_burn_swap_accepts"], dtype=np.int64
        )
        swap_rates = np.divide(
            swap_accepts,
            swap_proposals,
            out=np.zeros_like(swap_accepts, dtype=np.float64),
            where=swap_proposals > 0,
        )
        minimum_swap_edge = int(np.argmin(swap_rates))
        minimum_swap_rate = float(swap_rates[minimum_swap_edge])
        endpoint_transitions = int(
            np.asarray(
                result["mcmc_walker_endpoint_transitions"], dtype=np.int64
            ).sum()
        )
        cold_walkers = int(result["mcmc_cold_walker_count"])
        cold_unique_policies = int(result["mcmc_unique_policy_samples"])
        metrics = {
            "run_id": artifacts.run_dirs[0].name,
            "structural_bridge_beta_min": thresholds.structural_bridge_beta_min,
            "structural_bridge_beta_max": thresholds.structural_bridge_beta_max,
            "post_burn_bridge_block_proposals": bridge_proposals.tolist(),
            "post_burn_bridge_block_accepts": bridge_accepts.tolist(),
            "post_burn_bridge_block_move_accepts": (
                bridge_move_accepts.tolist()
            ),
            "post_burn_path_only_bridge_move_accepts": path_only_bridge_moves,
            "post_burn_total_path_bridge_move_accepts": total_path_bridge_moves,
            "post_burn_fringe_block_move_coverage": fringe_move_coverage,
            "post_burn_minimum_swap_edge": minimum_swap_edge,
            "post_burn_minimum_swap_acceptance": minimum_swap_rate,
            "post_burn_endpoint_transitions": endpoint_transitions,
            "cold_walker_count": cold_walkers,
            "cold_unique_policy_samples": cold_unique_policies,
        }
        checks = {
            "enough_bridge_block_proposals_for_every_block": bool(
                np.all(
                    bridge_proposals
                    >= thresholds.min_post_burn_bridge_block_proposals_per_block
                )
            ),
            "accepted_policy_changing_path_refresh": total_path_bridge_moves
            >= thresholds.min_post_burn_path_block_move_accepts,
            "accepted_policy_changing_fringe_refreshes": fringe_move_coverage
            >= thresholds.min_post_burn_fringe_block_move_coverage,
            "all_edges_have_usable_swap_acceptance": minimum_swap_rate
            >= thresholds.min_post_burn_all_edge_swap_acceptance,
            "endpoint_transition": endpoint_transitions
            >= thresholds.min_post_burn_endpoint_transitions,
            "multiple_cold_walkers": cold_walkers >= thresholds.min_cold_walkers,
            "cold_policy_movement": cold_unique_policies
            >= thresholds.min_cold_unique_policies,
        }
        return {
            "passed": bool(all(checks.values())),
            "scope": "single_chain_structural_block_rescue_only",
            "checks": checks,
            "thresholds": asdict(thresholds),
            "metrics": metrics,
        }

    global_proposals = np.asarray(
        result["mcmc_post_burn_global_proposals"], dtype=np.int64
    )
    global_accepts = np.asarray(
        result["mcmc_post_burn_global_accepts"], dtype=np.int64
    )
    global_move_accepts = np.asarray(
        result["mcmc_post_burn_global_move_accepts"], dtype=np.int64
    )
    swap_proposals = np.asarray(
        result["mcmc_post_burn_swap_proposals"], dtype=np.int64
    )
    swap_accepts = np.asarray(
        result["mcmc_post_burn_swap_accepts"], dtype=np.int64
    )
    high_beta_start = int(
        np.floor((1.0 - thresholds.high_beta_fraction) * len(global_accepts))
    )
    high_beta_start = min(max(high_beta_start, 0), len(global_accepts) - 1)
    high_beta_proposals = int(global_proposals[high_beta_start:].sum())
    high_beta_accepts = int(global_accepts[high_beta_start:].sum())
    high_beta_move_accepts = int(global_move_accepts[high_beta_start:].sum())
    final_edge_proposals = int(swap_proposals[-1])
    final_edge_accepts = int(swap_accepts[-1])
    cold_walkers = int(result["mcmc_cold_walker_count"])
    cold_unique_policies = int(result["mcmc_unique_policy_samples"])
    metrics = {
        "run_id": artifacts.run_dirs[0].name,
        "high_beta_start_index": high_beta_start,
        "post_burn_high_beta_global_proposals": high_beta_proposals,
        "post_burn_high_beta_global_accepts": high_beta_accepts,
        "post_burn_high_beta_global_move_accepts": high_beta_move_accepts,
        "post_burn_high_beta_global_acceptance": (
            high_beta_accepts / high_beta_proposals
            if high_beta_proposals > 0
            else 0.0
        ),
        "post_burn_final_edge_swap_proposals": final_edge_proposals,
        "post_burn_final_edge_swap_accepts": final_edge_accepts,
        "post_burn_final_edge_swap_acceptance": (
            final_edge_accepts / final_edge_proposals
            if final_edge_proposals > 0
            else 0.0
        ),
        "cold_walker_count": cold_walkers,
        "cold_unique_policy_samples": cold_unique_policies,
    }
    checks = {
        "accepted_policy_changing_high_beta_global_refresh": high_beta_move_accepts
        >= thresholds.min_post_burn_high_beta_global_move_accepts,
        "accepted_final_edge_swap": final_edge_accepts
        >= thresholds.min_post_burn_final_edge_swap_accepts,
        "multiple_cold_walkers": cold_walkers >= thresholds.min_cold_walkers,
        "cold_policy_movement": cold_unique_policies
        >= thresholds.min_cold_unique_policies,
    }
    return {
        "passed": bool(all(checks.values())),
        "scope": "single_chain_sparse_plateau_rescue_only",
        "checks": checks,
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }


def audit_tempering_probe(
    output_root: Path,
    thresholds: TemperingProbeThresholds = TemperingProbeThresholds(),
) -> dict[str, Any]:
    """Apply the cheap, predeclared ladder-flow probe without a full mixing claim."""

    artifacts = _load_tempering_artifacts(output_root)
    if len(artifacts.run_dirs) != thresholds.required_chain_count:
        raise ValueError(
            f"probe requires exactly {thresholds.required_chain_count} completed chains"
        )
    if not all(
        "mcmc_post_burn_swap_acceptance_rates" in result
        for result in artifacts.results
    ):
        raise ValueError("probe requires post-burn swap-acceptance diagnostics")

    per_chain_min_swap = [
        float(np.min(result["mcmc_post_burn_swap_acceptance_rates"]))
        for result in artifacts.results
    ]
    per_chain_endpoint_transitions = [
        int(sum(result["mcmc_walker_endpoint_transitions"]))
        for result in artifacts.results
    ]
    metrics = {
        "chain_count": len(artifacts.run_dirs),
        "run_ids": [path.name for path in artifacts.run_dirs],
        "per_chain_min_post_burn_swap_acceptance": per_chain_min_swap,
        "min_post_burn_swap_acceptance": float(min(per_chain_min_swap)),
        "per_chain_endpoint_transitions": per_chain_endpoint_transitions,
        "min_endpoint_transitions_per_chain": int(
            min(per_chain_endpoint_transitions)
        ),
    }
    checks = {
        "post_burn_swap_acceptance": metrics["min_post_burn_swap_acceptance"]
        >= thresholds.min_post_burn_swap_acceptance,
        "endpoint_transitions": metrics["min_endpoint_transitions_per_chain"]
        >= thresholds.min_endpoint_transitions_per_chain,
    }
    return {
        "passed": bool(all(checks.values())),
        "scope": "ladder_probe_only_not_full_mixing_validation",
        "checks": checks,
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }


def audit_tempering_chains(
    output_root: Path,
    thresholds: TemperingGateThresholds = TemperingGateThresholds(),
) -> dict[str, Any]:
    """Audit completed replica-exchange chains in one output directory."""

    artifacts = _load_tempering_artifacts(output_root)
    run_dirs = artifacts.run_dirs
    configs = artifacts.configs
    results = artifacts.results
    policies = artifacts.policies
    mdp = artifacts.mdp
    sample_counts = {len(chain) for chain in policies}
    score_chains = np.stack(
        [
            _target_scores(mdp, chain, config, result)
            for chain, config, result in zip(policies, configs, results)
        ],
        axis=0,
    )
    policy_array = np.stack(policies, axis=0)
    coordinate_tv = coordinate_total_variation(
        policy_array, tuple(mdp.decision_states or ()), mdp.num_actions
    )
    per_chain_ess = [autocorrelation_effective_sample_size(scores) for scores in score_chains]
    swap_rate_keys = [
        "mcmc_post_burn_swap_acceptance_rates"
        if "mcmc_post_burn_swap_acceptance_rates" in result
        else "mcmc_swap_acceptance_rates"
        for result in results
    ]
    min_swap_rates = [
        float(np.min(result[key])) for result, key in zip(results, swap_rate_keys)
    ]
    round_trips = [int(result["mcmc_total_round_trips"]) for result in results]
    endpoint_transitions = [
        int(sum(result["mcmc_walker_endpoint_transitions"])) for result in results
    ]
    cold_walkers = [int(result["mcmc_cold_walker_count"]) for result in results]
    unique_policy_fractions = [
        float(result["mcmc_unique_policy_fraction"]) for result in results
    ]
    mean_hamming_jumps = [
        float(result["mcmc_mean_policy_hamming_jump"]) for result in results
    ]

    decision_states = np.asarray(mdp.decision_states, dtype=np.int64)
    pooled_actions = policy_array[:, :, decision_states].reshape(
        -1, mdp.num_decisions
    )
    pooled_probabilities = np.empty(
        (mdp.num_decisions, mdp.num_actions), dtype=np.float64
    )
    for decision in range(mdp.num_decisions):
        pooled_probabilities[decision] = np.bincount(
            pooled_actions[:, decision], minlength=mdp.num_actions
        ) / float(len(pooled_actions))
    pooled_nonmodal_mass = 1.0 - pooled_probabilities.max(axis=1)
    nondegenerate = (
        pooled_nonmodal_mass >= thresholds.min_pooled_nonmodal_mass_for_switch
    )
    unswitched_nondegenerate = []
    for chain in policy_array:
        actions = chain[:, decision_states]
        switched = np.any(actions[1:] != actions[:-1], axis=0)
        unswitched_nondegenerate.append(int(np.count_nonzero(nondegenerate & ~switched)))

    metrics = {
        "chain_count": len(run_dirs),
        "draws_per_chain": int(next(iter(sample_counts))),
        "run_ids": [path.name for path in run_dirs],
        "split_return_rhat": split_rhat(score_chains),
        "per_chain_return_ess": per_chain_ess,
        "min_return_ess": float(min(per_chain_ess)),
        "per_chain_min_swap_acceptance": min_swap_rates,
        "min_swap_acceptance": float(min(min_swap_rates)),
        "swap_acceptance_window": (
            "post_burn"
            if all(key == "mcmc_post_burn_swap_acceptance_rates" for key in swap_rate_keys)
            else "includes_burn_in_for_legacy_results"
        ),
        "per_chain_cold_walkers": cold_walkers,
        "min_cold_walkers": int(min(cold_walkers)),
        "per_chain_round_trips": round_trips,
        "min_round_trips_per_chain": int(min(round_trips)),
        "per_chain_endpoint_transitions": endpoint_transitions,
        "min_endpoint_transitions_per_chain": int(min(endpoint_transitions)),
        "per_chain_unique_policy_fraction": unique_policy_fractions,
        "min_unique_policy_fraction": float(min(unique_policy_fractions)),
        "per_chain_mean_policy_hamming_jump": mean_hamming_jumps,
        "min_mean_policy_hamming_jump": float(min(mean_hamming_jumps)),
        "pooled_nondegenerate_state_count": int(np.count_nonzero(nondegenerate)),
        "per_chain_unswitched_nondegenerate_states": unswitched_nondegenerate,
        "max_unswitched_nondegenerate_states_per_chain": int(
            max(unswitched_nondegenerate)
        ),
        "max_coordinate_tv": float(np.max(coordinate_tv)),
        "coordinate_tv_quantiles": np.quantile(
            coordinate_tv, [0.0, 0.5, 0.9, 0.99, 1.0]
        ).tolist(),
        "states_with_pairwise_tv_above_0_1": int(
            np.count_nonzero(np.max(coordinate_tv, axis=0) > 0.1)
        ),
    }
    checks = {
        "split_return_rhat": metrics["split_return_rhat"]
        <= thresholds.max_split_return_rhat,
        "return_ess": metrics["min_return_ess"] >= thresholds.min_return_ess,
        "swap_acceptance": metrics["min_swap_acceptance"]
        >= thresholds.min_swap_acceptance,
        "cold_walkers": metrics["min_cold_walkers"] >= thresholds.min_cold_walkers,
        "round_trips": metrics["min_round_trips_per_chain"]
        >= thresholds.min_round_trips_per_chain,
        "coordinate_agreement": metrics["max_coordinate_tv"]
        <= thresholds.max_coordinate_tv,
        "within_chain_policy_diversity": metrics["min_unique_policy_fraction"]
        >= thresholds.min_unique_policy_fraction,
        "within_chain_policy_movement": metrics["min_mean_policy_hamming_jump"]
        >= thresholds.min_mean_policy_hamming_jump,
        "nondegenerate_coordinate_switches": metrics[
            "max_unswitched_nondegenerate_states_per_chain"
        ]
        <= thresholds.max_unswitched_nondegenerate_states_per_chain,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }
