from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_grid_spec, software_environment, source_tree_sha256
from .baselines import (
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
from .gridworld import gridworld_mdp, state_of
from .mdp import TapeBank
from .oracles import (
    evaluate_deterministic_policy_goal_probability,
    evaluate_nonstationary_deterministic_policy_goal_probability,
    evaluate_stationary_stochastic_policy_goal_probability,
    evaluate_stationary_stochastic_policy,
    finite_horizon_optimal_control,
    finite_horizon_optimal_goal_reaching_control,
    uniform_random_policy_value,
)
from .policy import DirectELBOTrainConfig, train_direct_elbo
from .policy_bank import restricted_policy_bank_posterior
from .policy_mcmc import (
    PolicyMHConfig,
    autocorrelation_effective_sample_size,
    run_single_site_policy_mh,
)
from .policy_tempering import (
    PolicyTemperingConfig,
    run_replica_exchange_policy_mh,
)
from .replicated_smc import ReplicatedSMCConfig, run_replicated_policy_smc


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    method: str
    map_path: str
    output_root: str = "results"
    training_seed: int = 0
    transition_budget: int = 10_000
    beta: float = 1.0
    evaluation_policy_samples: int = 2_000
    direct_policy_samples: int = 32
    direct_rollouts_per_policy: int = 4
    learning_rate: float = 0.02
    smc_particles: int = 256
    smc_tapes: int = 16
    ess_threshold_ratio: float = 0.5
    proposal_uniform_mix: float = 0.05
    warm_start_fraction: float = 0.5
    ppo_count_bonus: float = 0.0
    bank_snapshot_interval: int = 5
    bank_evaluation_tapes: int = 64
    mcmc_iterations: int = 5_000
    mcmc_burn_in: int = 1_000
    mcmc_thinning: int = 4
    mcmc_tapes: int = 0
    mcmc_temperatures: int = 8
    mcmc_ladder_power: float = 2.0
    mcmc_swap_interval: int = 1
    mcmc_guide_strength: float = 0.5
    mcmc_prior_initialize_hot_replicas: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _complete_probability_matrix(mdp, decision_probabilities: np.ndarray) -> np.ndarray:
    full = np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)
    for decision, state in enumerate(mdp.decision_states or ()):
        full[state] = decision_probabilities[decision]
    return full


def _committed_policy_value(mdp, policies: np.ndarray) -> tuple[float, float]:
    values = np.asarray([mdp.expected_return(policy) for policy in policies], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def _committed_policy_goal_probability(
    mdp,
    policies: np.ndarray,
    goal_states: tuple[int, ...],
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Exact environment-success moments over committed deterministic policies."""

    policies = np.asarray(policies, dtype=np.int64)
    if policies.ndim != 2 or policies.shape[1] != mdp.num_states or len(policies) == 0:
        raise ValueError("committed policies must have nonempty shape [N, S]")
    unique_policies, inverse = np.unique(policies, axis=0, return_inverse=True)
    unique_values = np.asarray(
        [
            evaluate_deterministic_policy_goal_probability(mdp, policy, goal_states)
            for policy in unique_policies
        ],
        dtype=np.float64,
    )
    values = unique_values[inverse]
    if weights is None:
        return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("committed-policy weights must be nonnegative and sum to one")
    mean = float(np.dot(weights, values))
    variance = float(np.dot(weights, (values - mean) ** 2))
    return mean, float(np.sqrt(max(variance, 0.0)))


def _map_policy_value(mdp, probabilities: np.ndarray) -> float:
    return float(mdp.expected_return(np.argmax(probabilities, axis=1).astype(np.int64)))


def _sample_committed_policies(
    mdp,
    probabilities: np.ndarray,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    policies = np.zeros((num_samples, mdp.num_states), dtype=np.int64)
    for sample in range(num_samples):
        for state in mdp.decision_states or ():
            policies[sample, state] = int(rng.choice(mdp.num_actions, p=probabilities[state]))
    mean, std = _committed_policy_value(mdp, policies)
    return policies, mean, std


def run_experiment(config: ExperimentConfig) -> Path:
    allowed_methods = {
        "random",
        "oracle",
        "direct_elbo",
        "ppo_warmstart_elbo",
        "ppo_policy_bank",
        "ppo_policy_mh",
        "ppo_policy_tempering",
        "replicated_smc",
        "ppo_guided_smc",
        "reinforce",
        "ppo",
        "sac",
        "cem",
        "double_q",
    }
    if config.method not in allowed_methods:
        raise ValueError(f"unknown method {config.method!r}")
    if not 0.0 < config.proposal_uniform_mix < 1.0:
        raise ValueError("proposal_uniform_mix must lie in (0, 1)")
    if not 0.0 < config.warm_start_fraction < 1.0:
        raise ValueError("warm_start_fraction must lie in (0, 1)")
    if config.ppo_count_bonus < 0.0:
        raise ValueError("ppo_count_bonus must be nonnegative")
    if config.bank_snapshot_interval <= 0 or config.bank_evaluation_tapes <= 0:
        raise ValueError("policy-bank snapshot interval and tape count must be positive")
    if config.mcmc_iterations <= 0 or config.mcmc_thinning <= 0 or config.mcmc_tapes < 0:
        raise ValueError("invalid policy-MCMC iteration, thinning, or tape count")
    if not 0 <= config.mcmc_burn_in < config.mcmc_iterations:
        raise ValueError("mcmc_burn_in must lie in [0, mcmc_iterations)")
    if config.mcmc_temperatures < 2 or config.mcmc_ladder_power <= 0.0:
        raise ValueError("invalid policy-tempering ladder")
    if config.mcmc_swap_interval <= 0 or not 0.0 <= config.mcmc_guide_strength < 1.0:
        raise ValueError("invalid policy-tempering swap interval or guide strength")
    map_path = Path(config.map_path).resolve()
    spec, map_hash = load_grid_spec(map_path)
    mdp = gridworld_mdp(spec)
    if config.method in {"reinforce", "ppo", "sac"} and not np.isclose(mdp.gamma, 1.0):
        raise NotImplementedError(
            "discounted REINFORCE/PPO/SAC semantics are not yet validated; use gamma=1"
        )
    goal_states = tuple(state_of(spec, cell) for cell in spec.goals)
    final_dir = Path(config.output_root).resolve() / config.run_id
    if (final_dir / "DONE").exists():
        return final_dir
    if final_dir.exists():
        raise FileExistsError(f"partial output already exists: {final_dir}")
    work_dir = final_dir.with_name(f"{final_dir.name}.inprogress.{os.getpid()}")
    if work_dir.exists():
        raise FileExistsError(f"temporary output already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    started = time.time()
    (work_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    (work_dir / "map.json").write_text(map_path.read_text(encoding="utf-8"), encoding="utf-8")
    environment = {
        **software_environment(),
        **_git_revision(),
        "map_hash": map_hash,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    source_hash, source_file_count = source_tree_sha256(Path(__file__).resolve().parents[1])
    environment["source_snapshot_sha256"] = source_hash
    environment["source_snapshot_file_count"] = source_file_count
    (work_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    try:
        uniform_probabilities = np.full(
            (mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions, dtype=np.float64
        )
        random_policies, random_committed_mean, random_committed_std = _sample_committed_policies(
            mdp,
            uniform_probabilities,
            config.evaluation_policy_samples,
            config.training_seed + 7_000_003,
        )
        random_goal_mean, random_goal_std = _committed_policy_goal_probability(
            mdp, random_policies, goal_states
        )
        result: dict[str, Any] = {
            "method": config.method,
            "random_marginal_value": uniform_random_policy_value(mdp),
            "random_committed_value": random_committed_mean,
            "random_committed_value_std": random_committed_std,
            "random_committed_value_se": random_committed_std
            / np.sqrt(config.evaluation_policy_samples),
            "random_marginal_goal_probability": (
                evaluate_stationary_stochastic_policy_goal_probability(
                    mdp, uniform_probabilities, goal_states
                )
            ),
            "random_committed_goal_probability": random_goal_mean,
            "random_committed_goal_probability_std": random_goal_std,
            "random_committed_goal_probability_se": random_goal_std
            / np.sqrt(config.evaluation_policy_samples),
            "transition_budget": config.transition_budget,
            "transition_budget_semantics": "minimum_counted_training_transitions",
            "evaluation_policy_samples": config.evaluation_policy_samples,
            "transition_model_kind": mdp.transition_model_kind,
            "transition_storage_bytes": mdp.transition_storage_bytes,
            "ppo_count_bonus": config.ppo_count_bonus,
            "goal_probability_event": "hit_any_goal_at_or_before_horizon",
            "goal_probability_horizon": mdp.horizon,
        }
        history: list[dict[str, float]] = []
        policy_probabilities_artifact: np.ndarray | None = None
        policy_particles_artifact: np.ndarray | None = None
        policy_weights_artifact: np.ndarray | None = None
        primary_committed_policies: np.ndarray | None = None
        primary_committed_weights: np.ndarray | None = None
        if config.method == "random":
            result["value"] = result["random_committed_value"]
            result["marginal_action_value"] = result["random_marginal_value"]
            result["committed_policy_value"] = result["random_committed_value"]
            result["committed_policy_value_std"] = result["random_committed_value_std"]
            result["committed_policy_value_se"] = result["random_committed_value_se"]
            result["simulator_steps"] = 0
            policy_probabilities_artifact = uniform_probabilities
            primary_committed_policies = random_policies
        elif config.method == "oracle":
            oracle = finite_horizon_optimal_control(mdp)
            result["value"] = oracle.initial_value
            result["oracle_kind"] = "nonstationary_finite_horizon"
            result["simulator_steps"] = 0
        elif config.method in {"direct_elbo", "replicated_smc"}:
            distribution, history = train_direct_elbo(
                mdp,
                DirectELBOTrainConfig(
                    iterations=10**9,
                    num_policy_samples=config.direct_policy_samples,
                    rollouts_per_policy=config.direct_rollouts_per_policy,
                    beta=config.beta,
                    learning_rate=config.learning_rate,
                    seed=config.training_seed,
                    transition_budget=config.transition_budget,
                ),
            )
            proposal = distribution.probabilities().detach().cpu().numpy()
            full_probabilities = _complete_probability_matrix(mdp, proposal)
            result["marginal_action_value"] = evaluate_stationary_stochastic_policy(mdp, full_probabilities)
            torch_generator = __import__("torch").Generator(device=distribution.logits.device)
            torch_generator.manual_seed(config.training_seed + 1_000_003)
            sampled_decisions, _ = distribution.sample_decisions(
                config.evaluation_policy_samples, torch_generator
            )
            sampled_policies = distribution.full_policies(sampled_decisions)
            primary_committed_policies = sampled_policies
            committed_mean, committed_std = _committed_policy_value(mdp, sampled_policies)
            result["committed_policy_value"] = committed_mean
            result["committed_policy_value_std"] = committed_std
            result["committed_policy_value_se"] = committed_std / np.sqrt(
                config.evaluation_policy_samples
            )
            result["map_policy_value"] = _map_policy_value(mdp, full_probabilities)
            result["training_simulator_steps"] = int(history[-1]["simulator_steps"])
            policy_probabilities_artifact = full_probabilities
            if config.method == "direct_elbo":
                result["value"] = committed_mean
                result["simulator_steps"] = result["training_simulator_steps"]
            else:
                tapes = TapeBank.sample(config.smc_tapes, mdp.horizon, config.training_seed + 2_000_003)
                smc = run_replicated_policy_smc(
                    mdp,
                    tapes,
                    ReplicatedSMCConfig(
                        num_particles=config.smc_particles,
                        beta=config.beta,
                        ess_threshold_ratio=config.ess_threshold_ratio,
                        seed=config.training_seed + 3_000_003,
                    ),
                    proposal_probabilities=proposal,
                )
                smc_mean, smc_std = _committed_policy_value(mdp, smc.policies)
                result.update(
                    {
                        "value": smc_mean,
                        "marginal_action_value": evaluate_stationary_stochastic_policy(
                            mdp, smc.marginal_action_probabilities(mdp)
                        ),
                        "committed_policy_value_std": smc_std,
                        "committed_policy_value_se": smc_std / np.sqrt(len(smc.policies)),
                        "smc_simulator_steps": smc.simulator_steps,
                        "simulator_steps": result["training_simulator_steps"] + smc.simulator_steps,
                        "smc_log_normalizer": smc.log_normalizer_estimate,
                        "smc_min_ess": float(smc.ess_history.min()),
                        "smc_resampling_events": int(smc.resampled.sum()),
                        "smc_unique_roots": smc.root_ancestor_count,
                    }
                )
                policy_probabilities_artifact = smc.marginal_action_probabilities(mdp)
                policy_particles_artifact = smc.policies
                primary_committed_policies = smc.policies
                result["map_policy_value"] = _map_policy_value(
                    mdp, policy_probabilities_artifact
                )
        elif config.method == "ppo_guided_smc":
            trained = train_ppo(
                mdp,
                PPOConfig(
                    transition_budget=config.transition_budget,
                    learning_rate=config.learning_rate,
                    count_bonus_coefficient=config.ppo_count_bonus,
                    seed=config.training_seed,
                ),
            )
            history = trained.history
            proposal_full = trained.action_probabilities
            proposal_decisions = proposal_full[np.asarray(mdp.decision_states, dtype=np.int64)]
            uniform_decisions = np.full_like(proposal_decisions, 1.0 / mdp.num_actions)
            guided_proposal = (
                (1.0 - config.proposal_uniform_mix) * proposal_decisions
                + config.proposal_uniform_mix * uniform_decisions
            )
            guided_full = _complete_probability_matrix(mdp, guided_proposal)
            _, proposal_committed_mean, proposal_committed_std = _sample_committed_policies(
                mdp,
                proposal_full,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            result.update(
                {
                    "proposal_uniform_mix": config.proposal_uniform_mix,
                    "proposal_min_action_probability": float(guided_proposal.min()),
                    "proposal_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, proposal_full
                    ),
                    "proposal_committed_policy_value": proposal_committed_mean,
                    "proposal_committed_policy_value_std": proposal_committed_std,
                    "proposal_committed_policy_value_se": proposal_committed_std
                    / np.sqrt(config.evaluation_policy_samples),
                    "proposal_map_policy_value": _map_policy_value(mdp, proposal_full),
                    "guided_proposal_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, guided_full
                    ),
                    "training_simulator_steps": trained.transitions,
                }
            )
            tapes = TapeBank.sample(config.smc_tapes, mdp.horizon, config.training_seed + 2_000_003)
            smc = run_replicated_policy_smc(
                mdp,
                tapes,
                ReplicatedSMCConfig(
                    num_particles=config.smc_particles,
                    beta=config.beta,
                    ess_threshold_ratio=config.ess_threshold_ratio,
                    seed=config.training_seed + 3_000_003,
                ),
                proposal_probabilities=guided_proposal,
            )
            smc_mean, smc_std = _committed_policy_value(mdp, smc.policies)
            result.update(
                {
                    "value": smc_mean,
                    "marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, smc.marginal_action_probabilities(mdp)
                    ),
                    "committed_policy_value": smc_mean,
                    "committed_policy_value_std": smc_std,
                    "committed_policy_value_se": smc_std / np.sqrt(len(smc.policies)),
                    "smc_simulator_steps": smc.simulator_steps,
                    "simulator_steps": trained.transitions + smc.simulator_steps,
                    "smc_log_normalizer": smc.log_normalizer_estimate,
                    "smc_min_ess": float(smc.ess_history.min()),
                    "smc_resampling_events": int(smc.resampled.sum()),
                    "smc_unique_roots": smc.root_ancestor_count,
                }
            )
            policy_probabilities_artifact = smc.marginal_action_probabilities(mdp)
            policy_particles_artifact = smc.policies
            primary_committed_policies = smc.policies
            result["map_policy_value"] = _map_policy_value(mdp, policy_probabilities_artifact)
        elif config.method == "ppo_warmstart_elbo":
            warm_start_budget = max(1, int(config.transition_budget * config.warm_start_fraction))
            refinement_budget = max(1, config.transition_budget - warm_start_budget)
            warm_start = train_ppo(
                mdp,
                PPOConfig(
                    transition_budget=warm_start_budget,
                    learning_rate=config.learning_rate,
                    count_bonus_coefficient=config.ppo_count_bonus,
                    seed=config.training_seed,
                ),
            )
            warm_start_probabilities = warm_start.action_probabilities
            warm_start_decisions = warm_start_probabilities[
                np.asarray(mdp.decision_states, dtype=np.int64)
            ]
            torch = __import__("torch")
            initial_logits = torch.log(
                torch.as_tensor(
                    np.clip(warm_start_decisions, 1e-12, 1.0),
                    dtype=torch.float64,
                )
            )
            distribution, refinement_history = train_direct_elbo(
                mdp,
                DirectELBOTrainConfig(
                    iterations=10**9,
                    num_policy_samples=config.direct_policy_samples,
                    rollouts_per_policy=config.direct_rollouts_per_policy,
                    beta=config.beta,
                    learning_rate=config.learning_rate,
                    seed=config.training_seed + 1_000_003,
                    transition_budget=refinement_budget,
                ),
                initial_logits=initial_logits,
            )
            history = []
            for row in warm_start.history:
                history.append(
                    {
                        **row,
                        "stage": 0.0,
                        "simulator_steps": float(row["transitions"]),
                    }
                )
            for row in refinement_history:
                history.append(
                    {
                        **row,
                        "stage": 1.0,
                        "simulator_steps": float(warm_start.transitions + row["simulator_steps"]),
                    }
                )
            final_decisions = distribution.probabilities().detach().cpu().numpy()
            final_probabilities = _complete_probability_matrix(mdp, final_decisions)
            _, warm_committed_mean, warm_committed_std = _sample_committed_policies(
                mdp,
                warm_start_probabilities,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            torch_generator = torch.Generator(device=distribution.logits.device)
            torch_generator.manual_seed(config.training_seed + 9_000_003)
            sampled_decisions, _ = distribution.sample_decisions(
                config.evaluation_policy_samples, torch_generator
            )
            sampled_policies = distribution.full_policies(sampled_decisions)
            primary_committed_policies = sampled_policies
            committed_mean, committed_std = _committed_policy_value(mdp, sampled_policies)
            total_training_steps = warm_start.transitions + int(
                refinement_history[-1]["simulator_steps"]
            )
            result.update(
                {
                    "value": committed_mean,
                    "marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, final_probabilities
                    ),
                    "committed_policy_value": committed_mean,
                    "committed_policy_value_std": committed_std,
                    "committed_policy_value_se": committed_std
                    / np.sqrt(config.evaluation_policy_samples),
                    "map_policy_value": _map_policy_value(mdp, final_probabilities),
                    "warm_start_fraction": config.warm_start_fraction,
                    "warm_start_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, warm_start_probabilities
                    ),
                    "warm_start_committed_policy_value": warm_committed_mean,
                    "warm_start_committed_policy_value_std": warm_committed_std,
                    "warm_start_committed_policy_value_se": warm_committed_std
                    / np.sqrt(config.evaluation_policy_samples),
                    "warm_start_map_policy_value": _map_policy_value(
                        mdp, warm_start_probabilities
                    ),
                    "warm_start_simulator_steps": warm_start.transitions,
                    "refinement_simulator_steps": int(
                        refinement_history[-1]["simulator_steps"]
                    ),
                    "simulator_steps": total_training_steps,
                }
            )
            policy_probabilities_artifact = final_probabilities
        elif config.method == "ppo_policy_bank":
            proposal = train_ppo(
                mdp,
                PPOConfig(
                    transition_budget=config.transition_budget,
                    learning_rate=config.learning_rate,
                    count_bonus_coefficient=config.ppo_count_bonus,
                    snapshot_interval_updates=config.bank_snapshot_interval,
                    seed=config.training_seed,
                ),
            )
            if proposal.policy_snapshots is None or len(proposal.policy_snapshots) == 0:
                raise RuntimeError("PPO produced no policy-bank snapshots")
            history = proposal.history
            candidate_policies = np.argmax(proposal.policy_snapshots, axis=2).astype(np.int64)
            tapes = TapeBank.sample(
                config.bank_evaluation_tapes,
                mdp.horizon,
                config.training_seed + 4_000_003,
            )
            bank = restricted_policy_bank_posterior(
                mdp,
                candidate_policies,
                beta=config.beta,
                tapes=tapes,
            )
            bank_probabilities = bank.marginal_action_probabilities(mdp)
            committed_mean, committed_std = bank.exact_return_moments(mdp)
            exact_candidate_values = np.asarray(
                [mdp.expected_return(policy) for policy in bank.policies], dtype=np.float64
            )
            best_index = int(np.argmax(bank.probabilities))
            _, proposal_committed_mean, proposal_committed_std = _sample_committed_policies(
                mdp,
                proposal.action_probabilities,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            result.update(
                {
                    "value": committed_mean,
                    "marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, bank_probabilities
                    ),
                    "committed_policy_value": committed_mean,
                    "committed_policy_value_std": committed_std,
                    "map_policy_value": float(exact_candidate_values[best_index]),
                    "proposal_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, proposal.action_probabilities
                    ),
                    "proposal_committed_policy_value": proposal_committed_mean,
                    "proposal_committed_policy_value_std": proposal_committed_std,
                    "proposal_map_policy_value": _map_policy_value(
                        mdp, proposal.action_probabilities
                    ),
                    "bank_snapshot_interval": config.bank_snapshot_interval,
                    "bank_evaluation_tapes": config.bank_evaluation_tapes,
                    "bank_candidate_count_before_deduplication": int(len(candidate_policies)),
                    "bank_unique_policy_count": int(len(bank.policies)),
                    "bank_effective_sample_size": bank.effective_sample_size,
                    "bank_entropy": bank.entropy,
                    "bank_restricted_log_normalizer": bank.restricted_log_normalizer,
                    "bank_best_candidate_value": float(exact_candidate_values.max()),
                    "bank_uniform_candidate_value": float(exact_candidate_values.mean()),
                    "bank_saa_return_rmse": float(
                        np.sqrt(np.mean((bank.estimated_returns - exact_candidate_values) ** 2))
                    ),
                    "training_simulator_steps": proposal.transitions,
                    "bank_simulator_steps": bank.simulator_steps,
                    "simulator_steps": proposal.transitions + bank.simulator_steps,
                }
            )
            policy_probabilities_artifact = bank_probabilities
            policy_particles_artifact = bank.policies
            policy_weights_artifact = bank.probabilities
            primary_committed_policies = bank.policies
            primary_committed_weights = bank.probabilities
        elif config.method == "ppo_policy_mh":
            proposal = train_ppo(
                mdp,
                PPOConfig(
                    transition_budget=config.transition_budget,
                    learning_rate=config.learning_rate,
                    count_bonus_coefficient=config.ppo_count_bonus,
                    seed=config.training_seed,
                ),
            )
            history = proposal.history
            initial_policy = np.argmax(proposal.action_probabilities, axis=1).astype(np.int64)
            mcmc_tapes = (
                None
                if config.mcmc_tapes == 0
                else TapeBank.sample(
                    config.mcmc_tapes,
                    mdp.horizon,
                    config.training_seed + 5_000_003,
                )
            )
            mcmc_started = time.time()
            chain = run_single_site_policy_mh(
                mdp,
                initial_policy,
                PolicyMHConfig(
                    beta=config.beta,
                    iterations=config.mcmc_iterations,
                    burn_in=config.mcmc_burn_in,
                    thinning=config.mcmc_thinning,
                    seed=config.training_seed + 6_000_003,
                ),
                tapes=mcmc_tapes,
            )
            mcmc_elapsed = time.time() - mcmc_started
            chain_probabilities = chain.marginal_action_probabilities(mdp)
            committed_mean, committed_std = _committed_policy_value(mdp, chain.policies)
            exact_chain_values = np.asarray(
                [mdp.expected_return(policy) for policy in chain.policies], dtype=np.float64
            )
            exact_return_ess = autocorrelation_effective_sample_size(exact_chain_values)
            _, proposal_committed_mean, proposal_committed_std = _sample_committed_policies(
                mdp,
                proposal.action_probabilities,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            result.update(
                {
                    "value": committed_mean,
                    "marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, chain_probabilities
                    ),
                    "committed_policy_value": committed_mean,
                    "committed_policy_value_std": committed_std,
                    "committed_policy_value_se": committed_std / np.sqrt(exact_return_ess),
                    "map_policy_value": float(exact_chain_values.max()),
                    "marginal_map_policy_value": _map_policy_value(
                        mdp, chain_probabilities
                    ),
                    "proposal_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, proposal.action_probabilities
                    ),
                    "proposal_committed_policy_value": proposal_committed_mean,
                    "proposal_committed_policy_value_std": proposal_committed_std,
                    "proposal_map_policy_value": _map_policy_value(
                        mdp, proposal.action_probabilities
                    ),
                    "mcmc_target": "exact_expected_return"
                    if mcmc_tapes is None
                    else "fixed_tape_sample_average",
                    "mcmc_tapes": config.mcmc_tapes,
                    "mcmc_iterations": config.mcmc_iterations,
                    "mcmc_burn_in": config.mcmc_burn_in,
                    "mcmc_thinning": config.mcmc_thinning,
                    "mcmc_samples": int(len(chain.policies)),
                    "mcmc_acceptance_rate": chain.acceptance_rate,
                    "mcmc_target_return_ess": chain.return_effective_sample_size,
                    "mcmc_exact_return_ess": exact_return_ess,
                    "mcmc_changed_return_fraction": chain.changed_return_proposals
                    / float(config.mcmc_iterations),
                    "mcmc_value_evaluations": chain.value_evaluations,
                    "mcmc_simulator_steps": chain.simulator_steps,
                    "mcmc_elapsed_seconds": mcmc_elapsed,
                    "training_simulator_steps": proposal.transitions,
                    "simulator_steps": proposal.transitions + chain.simulator_steps,
                }
            )
            policy_probabilities_artifact = chain_probabilities
            policy_particles_artifact = chain.policies
            primary_committed_policies = chain.policies
        elif config.method == "ppo_policy_tempering":
            proposal = train_ppo(
                mdp,
                PPOConfig(
                    transition_budget=config.transition_budget,
                    learning_rate=config.learning_rate,
                    count_bonus_coefficient=config.ppo_count_bonus,
                    seed=config.training_seed,
                ),
            )
            history = proposal.history
            initial_policy = np.argmax(proposal.action_probabilities, axis=1).astype(np.int64)
            mcmc_tapes = (
                None
                if config.mcmc_tapes == 0
                else TapeBank.sample(
                    config.mcmc_tapes,
                    mdp.horizon,
                    config.training_seed + 5_000_003,
                )
            )
            mcmc_started = time.time()
            chain = run_replica_exchange_policy_mh(
                mdp,
                initial_policy,
                PolicyTemperingConfig(
                    beta=config.beta,
                    num_temperatures=config.mcmc_temperatures,
                    ladder_power=config.mcmc_ladder_power,
                    iterations=config.mcmc_iterations,
                    burn_in=config.mcmc_burn_in,
                    thinning=config.mcmc_thinning,
                    swap_interval=config.mcmc_swap_interval,
                    guide_strength=config.mcmc_guide_strength,
                    prior_initialize_hot_replicas=(
                        config.mcmc_prior_initialize_hot_replicas
                    ),
                    seed=config.training_seed + 6_000_003,
                ),
                tapes=mcmc_tapes,
                guide_probabilities=proposal.action_probabilities,
            )
            mcmc_elapsed = time.time() - mcmc_started
            chain_probabilities = chain.marginal_action_probabilities(mdp)
            committed_mean, committed_std = _committed_policy_value(mdp, chain.policies)
            exact_chain_values = np.asarray(
                [mdp.expected_return(policy) for policy in chain.policies], dtype=np.float64
            )
            exact_return_ess = autocorrelation_effective_sample_size(exact_chain_values)
            occupancy = chain.policy_occupancy_diagnostics(mdp)
            _, proposal_committed_mean, proposal_committed_std = _sample_committed_policies(
                mdp,
                proposal.action_probabilities,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            result.update(
                {
                    "value": committed_mean,
                    "marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, chain_probabilities
                    ),
                    "committed_policy_value": committed_mean,
                    "committed_policy_value_std": committed_std,
                    "committed_policy_value_se": committed_std / np.sqrt(exact_return_ess),
                    "map_policy_value": float(exact_chain_values.max()),
                    "marginal_map_policy_value": _map_policy_value(
                        mdp, chain_probabilities
                    ),
                    "proposal_marginal_action_value": evaluate_stationary_stochastic_policy(
                        mdp, proposal.action_probabilities
                    ),
                    "proposal_committed_policy_value": proposal_committed_mean,
                    "proposal_committed_policy_value_std": proposal_committed_std,
                    "proposal_map_policy_value": _map_policy_value(
                        mdp, proposal.action_probabilities
                    ),
                    "mcmc_target": "exact_expected_return"
                    if mcmc_tapes is None
                    else "fixed_tape_sample_average",
                    "mcmc_tapes": config.mcmc_tapes,
                    "mcmc_iterations": config.mcmc_iterations,
                    "mcmc_burn_in": config.mcmc_burn_in,
                    "mcmc_thinning": config.mcmc_thinning,
                    "mcmc_samples": int(len(chain.policies)),
                    "mcmc_temperatures": config.mcmc_temperatures,
                    "mcmc_prior_initialize_hot_replicas": (
                        config.mcmc_prior_initialize_hot_replicas
                    ),
                    "mcmc_beta_ladder": chain.betas.tolist(),
                    "mcmc_local_acceptance_rates": chain.local_acceptance_rates.tolist(),
                    "mcmc_swap_acceptance_rates": chain.swap_acceptance_rates.tolist(),
                    "mcmc_target_return_ess": chain.return_effective_sample_size,
                    "mcmc_exact_return_ess": exact_return_ess,
                    "mcmc_policy_occupancy_ess": (
                        occupancy.conservative_effective_sample_size
                    ),
                    "mcmc_policy_occupancy_ess_quantiles": (
                        occupancy.effective_sample_size_quantiles.tolist()
                    ),
                    "mcmc_states_without_action_switches": (
                        occupancy.states_without_action_switches
                    ),
                    "mcmc_min_state_action_switch_rate": float(
                        occupancy.state_action_switch_rates.min()
                    ),
                    "mcmc_mean_state_action_switch_rate": float(
                        occupancy.state_action_switch_rates.mean()
                    ),
                    "mcmc_unique_policy_samples": occupancy.unique_policy_count,
                    "mcmc_unique_policy_fraction": occupancy.unique_policy_fraction,
                    "mcmc_mean_policy_hamming_jump": (
                        occupancy.mean_policy_hamming_jump
                    ),
                    "mcmc_walker_temperature_visit_fractions": (
                        chain.temperature_visit_fractions.tolist()
                    ),
                    "mcmc_walker_endpoint_transitions": (
                        chain.walker_endpoint_transitions.tolist()
                    ),
                    "mcmc_walker_round_trips": chain.walker_round_trips.tolist(),
                    "mcmc_total_round_trips": int(chain.walker_round_trips.sum()),
                    "mcmc_cold_walker_count": chain.cold_walker_count,
                    "mcmc_value_evaluations": chain.value_evaluations,
                    "mcmc_simulator_steps": chain.simulator_steps,
                    "mcmc_elapsed_seconds": mcmc_elapsed,
                    "training_simulator_steps": proposal.transitions,
                    "simulator_steps": proposal.transitions + chain.simulator_steps,
                }
            )
            policy_probabilities_artifact = chain_probabilities
            policy_particles_artifact = chain.policies
            primary_committed_policies = chain.policies
        else:
            if config.method == "reinforce":
                trained = train_reinforce(
                    mdp,
                    ReinforceConfig(
                        transition_budget=config.transition_budget,
                        learning_rate=config.learning_rate,
                        seed=config.training_seed,
                    ),
                )
            elif config.method == "ppo":
                trained = train_ppo(
                    mdp,
                    PPOConfig(
                        transition_budget=config.transition_budget,
                        learning_rate=config.learning_rate,
                        count_bonus_coefficient=config.ppo_count_bonus,
                        seed=config.training_seed,
                    ),
                )
            elif config.method == "sac":
                trained = train_discrete_sac(
                    mdp,
                    DiscreteSACConfig(
                        transition_budget=config.transition_budget,
                        learning_rate=config.learning_rate,
                        seed=config.training_seed,
                    ),
                )
            elif config.method == "cem":
                trained = train_cem(
                    mdp,
                    CEMConfig(
                        transition_budget=config.transition_budget,
                        seed=config.training_seed,
                    ),
                )
            elif config.method == "double_q":
                trained = train_double_q(
                    mdp,
                    DoubleQConfig(
                        transition_budget=config.transition_budget,
                        learning_rate=config.learning_rate,
                        seed=config.training_seed,
                    ),
                )
            else:
                raise AssertionError(f"unhandled learned method {config.method!r}")
            history = trained.history
            result["marginal_action_value"] = evaluate_stationary_stochastic_policy(
                mdp, trained.action_probabilities
            )
            sampled_policies, committed_mean, committed_std = _sample_committed_policies(
                mdp,
                trained.action_probabilities,
                config.evaluation_policy_samples,
                config.training_seed + 8_000_003,
            )
            primary_committed_policies = sampled_policies
            result["value"] = committed_mean
            result["committed_policy_value"] = committed_mean
            result["committed_policy_value_std"] = committed_std
            result["committed_policy_value_se"] = committed_std / np.sqrt(
                config.evaluation_policy_samples
            )
            result["map_policy_value"] = _map_policy_value(mdp, trained.action_probabilities)
            result["simulator_steps"] = trained.transitions
            policy_probabilities_artifact = trained.action_probabilities

        if not np.isfinite(float(result["value"])):
            raise FloatingPointError("experiment produced a non-finite value")
        return_oracle = finite_horizon_optimal_control(mdp)
        goal_oracle = finite_horizon_optimal_goal_reaching_control(mdp, goal_states)
        result["oracle_value"] = return_oracle.initial_value
        result["oracle_goal_probability"] = goal_oracle.initial_value
        if policy_probabilities_artifact is not None:
            result["marginal_action_goal_probability"] = (
                evaluate_stationary_stochastic_policy_goal_probability(
                    mdp, policy_probabilities_artifact, goal_states
                )
            )
            result["map_policy_goal_probability"] = (
                evaluate_deterministic_policy_goal_probability(
                    mdp,
                    np.argmax(policy_probabilities_artifact, axis=1).astype(np.int64),
                    goal_states,
                )
            )
        if primary_committed_policies is not None:
            committed_goal_mean, committed_goal_std = _committed_policy_goal_probability(
                mdp,
                primary_committed_policies,
                goal_states,
                primary_committed_weights,
            )
            result["committed_policy_goal_probability"] = committed_goal_mean
            result["committed_policy_goal_probability_std"] = committed_goal_std

        if config.method == "oracle":
            result["goal_reaching_probability"] = (
                evaluate_nonstationary_deterministic_policy_goal_probability(
                    mdp, return_oracle.policy, goal_states
                )
            )
            result["goal_reaching_probability_semantics"] = (
                "return_optimal_nonstationary_deterministic_policy"
            )
        else:
            result["goal_reaching_probability"] = result[
                "committed_policy_goal_probability"
            ]
            result["goal_reaching_probability_semantics"] = "policy_sample_then_commit"
        result["primary_value_semantics"] = (
            "return_optimal_nonstationary_deterministic_policy"
            if config.method == "oracle"
            else "policy_sample_then_commit"
        )
        result["elapsed_seconds"] = time.time() - started
        if policy_probabilities_artifact is not None:
            policy_path = work_dir / "policy_probabilities.npy"
            np.save(policy_path, policy_probabilities_artifact, allow_pickle=False)
            saved_probabilities = np.load(policy_path, allow_pickle=False)
            if saved_probabilities.shape != (mdp.num_states, mdp.num_actions):
                raise ValueError("saved policy probabilities have the wrong shape")
            if not np.isfinite(saved_probabilities).all():
                raise FloatingPointError("saved policy probabilities are non-finite")
            result["policy_probabilities_artifact"] = policy_path.name
        if policy_particles_artifact is not None:
            particles_path = work_dir / "policy_particles.npy"
            np.save(particles_path, policy_particles_artifact, allow_pickle=False)
            saved_particles = np.load(particles_path, allow_pickle=False)
            if saved_particles.shape != policy_particles_artifact.shape:
                raise ValueError("saved policy particles have the wrong shape")
            result["policy_particles_artifact"] = particles_path.name
        if policy_weights_artifact is not None:
            weights_path = work_dir / "policy_weights.npy"
            np.save(weights_path, policy_weights_artifact, allow_pickle=False)
            saved_weights = np.load(weights_path, allow_pickle=False)
            if saved_weights.shape != policy_weights_artifact.shape:
                raise ValueError("saved policy weights have the wrong shape")
            if not np.isclose(saved_weights.sum(), 1.0, atol=1e-10):
                raise ValueError("saved policy weights do not sum to one")
            result["policy_weights_artifact"] = weights_path.name
        with (work_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in history:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        (work_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        # Read every final JSON artifact before declaring success.
        for name in ("config.json", "map.json", "environment.json", "result.json"):
            json.loads((work_dir / name).read_text(encoding="utf-8"))
        (work_dir / "DONE").write_text("ok\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.replace(final_dir)
        return final_dir
    except Exception as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "elapsed_seconds": time.time() - started,
        }
        (work_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        if not final_dir.exists():
            work_dir.replace(final_dir)
        raise
