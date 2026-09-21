from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .mdp import FiniteHorizonMDP
from .simulation import discounted_returns_to_go, simulate_episode


def _full_policy_probabilities(actor_logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(actor_logits, dim=-1).detach().cpu().numpy()


def _assert_finite(module: nn.Module, label: str) -> None:
    for parameter in module.parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite parameter in {label}")


@dataclass(frozen=True)
class BaselineResult:
    action_probabilities: np.ndarray
    transitions: int
    history: list[dict[str, float]]
    policy_snapshots: np.ndarray | None = None


class TabularActorCritic(nn.Module):
    def __init__(self, mdp: FiniteHorizonMDP):
        super().__init__()
        self.actor_logits = nn.Parameter(
            torch.zeros((mdp.num_states, mdp.num_actions), dtype=torch.float64)
        )
        self.values = nn.Parameter(torch.zeros(mdp.num_states, dtype=torch.float64))


@dataclass(frozen=True)
class ReinforceConfig:
    transition_budget: int = 20_000
    episodes_per_update: int = 16
    learning_rate: float = 3e-2
    value_learning_rate: float = 5e-2
    seed: int = 0
    log_every_updates: int = 10


def train_reinforce(mdp: FiniteHorizonMDP, config: ReinforceConfig) -> BaselineResult:
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    model = TabularActorCritic(mdp)
    actor_optimizer = torch.optim.Adam([model.actor_logits], lr=config.learning_rate)
    value_optimizer = torch.optim.Adam([model.values], lr=config.value_learning_rate)
    transitions = 0
    update = 0
    history: list[dict[str, float]] = []
    while transitions < config.transition_budget:
        states: list[int] = []
        actions: list[int] = []
        targets: list[float] = []
        episode_returns: list[float] = []
        probabilities = _full_policy_probabilities(model.actor_logits)
        for _ in range(config.episodes_per_update):
            episode = simulate_episode(mdp, lambda state: probabilities[state], rng)
            if episode.length == 0:
                continue
            returns_to_go = discounted_returns_to_go(episode.rewards, mdp.gamma)
            states.extend(episode.states.tolist())
            actions.extend(episode.actions.tolist())
            targets.extend(returns_to_go.tolist())
            episode_returns.append(episode.total_return)
            transitions += episode.length
            if transitions >= config.transition_budget:
                break
        if not states:
            raise RuntimeError("REINFORCE collected no transitions")
        state_tensor = torch.as_tensor(states, dtype=torch.long)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        target_tensor = torch.as_tensor(targets, dtype=torch.float64)

        log_probabilities = torch.log_softmax(model.actor_logits[state_tensor], dim=-1)
        selected_log_probabilities = log_probabilities.gather(1, action_tensor[:, None]).squeeze(1)
        advantages = target_tensor - model.values[state_tensor].detach()
        actor_loss = -(advantages * selected_log_probabilities).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        value_loss = torch.mean((model.values[state_tensor] - target_tensor) ** 2)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        _assert_finite(model, "REINFORCE")

        if update % config.log_every_updates == 0 or transitions >= config.transition_budget:
            history.append(
                {
                    "update": float(update),
                    "transitions": float(transitions),
                    "mean_episode_return": float(np.mean(episode_returns)),
                    "actor_loss": float(actor_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                }
            )
        update += 1
    return BaselineResult(_full_policy_probabilities(model.actor_logits), transitions, history)


@dataclass(frozen=True)
class PPOConfig:
    transition_budget: int = 20_000
    batch_transitions: int = 512
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 1e-2
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    count_bonus_coefficient: float = 0.0
    snapshot_interval_updates: int = 0
    seed: int = 0


def train_ppo(mdp: FiniteHorizonMDP, config: PPOConfig) -> BaselineResult:
    if config.count_bonus_coefficient < 0.0:
        raise ValueError("count_bonus_coefficient must be nonnegative")
    if config.snapshot_interval_updates < 0:
        raise ValueError("snapshot_interval_updates must be nonnegative")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    model = TabularActorCritic(mdp)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    transitions = 0
    update = 0
    state_visit_counts = np.zeros(mdp.num_states, dtype=np.int64)
    history: list[dict[str, float]] = []
    snapshots: list[np.ndarray] = []
    while transitions < config.transition_budget:
        states: list[int] = []
        actions: list[int] = []
        returns: list[float] = []
        old_log_probabilities: list[float] = []
        episode_returns: list[float] = []
        probabilities = _full_policy_probabilities(model.actor_logits)
        old_log_probs_all = torch.log_softmax(model.actor_logits.detach(), dim=-1).cpu().numpy()
        target_batch = min(config.batch_transitions, config.transition_budget - transitions)
        while len(states) < target_batch:
            episode = simulate_episode(mdp, lambda state: probabilities[state], rng)
            if episode.length == 0:
                continue
            training_rewards = episode.rewards.copy()
            if config.count_bonus_coefficient > 0.0:
                for index, next_state in enumerate(episode.next_states):
                    state_visit_counts[next_state] += 1
                    training_rewards[index] += config.count_bonus_coefficient / np.sqrt(
                        float(state_visit_counts[next_state])
                    )
            returns_to_go = discounted_returns_to_go(training_rewards, mdp.gamma)
            states.extend(episode.states.tolist())
            actions.extend(episode.actions.tolist())
            returns.extend(returns_to_go.tolist())
            old_log_probabilities.extend(
                old_log_probs_all[episode.states, episode.actions].tolist()
            )
            episode_returns.append(episode.total_return)
        transitions += len(states)

        states_t = torch.as_tensor(states, dtype=torch.long)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        returns_t = torch.as_tensor(returns, dtype=torch.float64)
        old_log_t = torch.as_tensor(old_log_probabilities, dtype=torch.float64)
        with torch.no_grad():
            advantages = returns_t - model.values[states_t]
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        indices = np.arange(len(states))
        last_policy_loss = torch.zeros((), dtype=torch.float64)
        last_value_loss = torch.zeros((), dtype=torch.float64)
        last_clip_fraction = torch.zeros((), dtype=torch.float64)
        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), config.minibatch_size):
                batch = torch.as_tensor(indices[start : start + config.minibatch_size], dtype=torch.long)
                batch_states = states_t[batch]
                batch_actions = actions_t[batch]
                logits = model.actor_logits[batch_states]
                log_probs = torch.log_softmax(logits, dim=-1)
                chosen_log = log_probs.gather(1, batch_actions[:, None]).squeeze(1)
                ratio = torch.exp(chosen_log - old_log_t[batch])
                unclipped = ratio * advantages[batch]
                clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.mean((model.values[batch_states] - returns_t[batch]) ** 2)
                probabilities_t = torch.softmax(logits, dim=-1)
                entropy = -(probabilities_t * log_probs).sum(dim=-1).mean()
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_clip_fraction = (torch.abs(ratio - 1.0) > config.clip_ratio).double().mean().detach()
        _assert_finite(model, "PPO")
        history.append(
            {
                "update": float(update),
                "transitions": float(transitions),
                "mean_episode_return": float(np.mean(episode_returns)),
                "policy_loss": float(last_policy_loss),
                "value_loss": float(last_value_loss),
                "clip_fraction": float(last_clip_fraction),
                "visited_states": float(np.count_nonzero(state_visit_counts)),
            }
        )
        if config.snapshot_interval_updates > 0 and (
            update % config.snapshot_interval_updates == 0
            or transitions >= config.transition_budget
        ):
            snapshots.append(_full_policy_probabilities(model.actor_logits))
        update += 1
    snapshot_array = np.stack(snapshots, axis=0) if snapshots else None
    return BaselineResult(
        _full_policy_probabilities(model.actor_logits),
        transitions,
        history,
        snapshot_array,
    )


@dataclass(frozen=True)
class DiscreteSACConfig:
    transition_budget: int = 20_000
    replay_capacity: int = 100_000
    learning_starts: int = 256
    batch_size: int = 128
    learning_rate: float = 1e-2
    alpha: float = 0.1
    tau: float = 0.01
    updates_per_transition: int = 1
    seed: int = 0
    log_every: int = 1_000


def train_discrete_sac(mdp: FiniteHorizonMDP, config: DiscreteSACConfig) -> BaselineResult:
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    actor_logits = nn.Parameter(torch.zeros((mdp.num_states, mdp.num_actions), dtype=torch.float64))
    q1 = nn.Parameter(torch.zeros((mdp.num_states, mdp.num_actions), dtype=torch.float64))
    q2 = nn.Parameter(torch.zeros((mdp.num_states, mdp.num_actions), dtype=torch.float64))
    target_q1 = q1.detach().clone()
    target_q2 = q2.detach().clone()
    actor_optimizer = torch.optim.Adam([actor_logits], lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam([q1, q2], lr=config.learning_rate)
    replay: deque[tuple[int, int, float, int, bool]] = deque(maxlen=config.replay_capacity)
    transitions = 0
    history: list[dict[str, float]] = []
    episode_returns: list[float] = []
    last_q_loss = torch.zeros((), dtype=torch.float64)
    last_actor_loss = torch.zeros((), dtype=torch.float64)

    while transitions < config.transition_budget:
        probabilities = _full_policy_probabilities(actor_logits)
        episode = simulate_episode(mdp, lambda state: probabilities[state], rng)
        episode_returns.append(episode.total_return)
        for state, action, reward, next_state, done in zip(
            episode.states, episode.actions, episode.rewards, episode.next_states, episode.dones
        ):
            replay.append((int(state), int(action), float(reward), int(next_state), bool(done)))
            transitions += 1
            if transitions >= config.learning_starts and len(replay) >= config.batch_size:
                for _ in range(config.updates_per_transition):
                    indices = rng.integers(0, len(replay), size=config.batch_size)
                    batch = [replay[int(index)] for index in indices]
                    states_t = torch.as_tensor([x[0] for x in batch], dtype=torch.long)
                    actions_t = torch.as_tensor([x[1] for x in batch], dtype=torch.long)
                    rewards_t = torch.as_tensor([x[2] for x in batch], dtype=torch.float64)
                    next_states_t = torch.as_tensor([x[3] for x in batch], dtype=torch.long)
                    dones_t = torch.as_tensor([x[4] for x in batch], dtype=torch.float64)
                    with torch.no_grad():
                        next_log_probs = torch.log_softmax(actor_logits[next_states_t], dim=-1)
                        next_probs = torch.softmax(actor_logits[next_states_t], dim=-1)
                        next_min_q = torch.minimum(target_q1[next_states_t], target_q2[next_states_t])
                        next_v = torch.sum(
                            next_probs * (next_min_q - config.alpha * next_log_probs), dim=-1
                        )
                        target = rewards_t + mdp.gamma * (1.0 - dones_t) * next_v
                    q1_selected = q1[states_t].gather(1, actions_t[:, None]).squeeze(1)
                    q2_selected = q2[states_t].gather(1, actions_t[:, None]).squeeze(1)
                    q_loss = torch.mean((q1_selected - target) ** 2 + (q2_selected - target) ** 2)
                    critic_optimizer.zero_grad()
                    q_loss.backward()
                    critic_optimizer.step()

                    log_probs = torch.log_softmax(actor_logits[states_t], dim=-1)
                    probs = torch.softmax(actor_logits[states_t], dim=-1)
                    min_q = torch.minimum(q1[states_t], q2[states_t]).detach()
                    actor_loss = torch.sum(probs * (config.alpha * log_probs - min_q), dim=-1).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    with torch.no_grad():
                        target_q1.mul_(1.0 - config.tau).add_(config.tau * q1)
                        target_q2.mul_(1.0 - config.tau).add_(config.tau * q2)
                    last_q_loss = q_loss.detach()
                    last_actor_loss = actor_loss.detach()
            if transitions % config.log_every == 0:
                history.append(
                    {
                        "transitions": float(transitions),
                        "mean_episode_return_100": float(np.mean(episode_returns[-100:])),
                        "q_loss": float(last_q_loss),
                        "actor_loss": float(last_actor_loss),
                        "replay_size": float(len(replay)),
                    }
                )
        if episode.length == 0:
            raise RuntimeError("SAC collected no transitions")
        if transitions >= config.transition_budget and (
            not history or history[-1]["transitions"] != float(transitions)
        ):
            history.append(
                {
                    "transitions": float(transitions),
                    "mean_episode_return_100": float(np.mean(episode_returns[-100:])),
                    "q_loss": float(last_q_loss),
                    "actor_loss": float(last_actor_loss),
                    "replay_size": float(len(replay)),
                }
            )
        if not torch.isfinite(actor_logits).all() or not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
            raise FloatingPointError("non-finite parameter in discrete SAC")
    return BaselineResult(_full_policy_probabilities(actor_logits), transitions, history)


@dataclass(frozen=True)
class CEMConfig:
    """Configuration for categorical cross-entropy policy search."""

    transition_budget: int = 20_000
    population_size: int = 64
    elite_fraction: float = 0.2
    rollouts_per_policy: int = 4
    smoothing: float = 0.7
    min_action_probability: float = 0.0
    seed: int = 0


def _validate_cem_config(mdp: FiniteHorizonMDP, config: CEMConfig) -> None:
    if config.transition_budget <= 0:
        raise ValueError("transition_budget must be positive")
    if config.population_size < 2:
        raise ValueError("population_size must be at least two")
    if not 0.0 < config.elite_fraction <= 1.0:
        raise ValueError("elite_fraction must lie in (0, 1]")
    if config.rollouts_per_policy <= 0:
        raise ValueError("rollouts_per_policy must be positive")
    if not 0.0 < config.smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    if not 0.0 <= config.min_action_probability < 1.0 / mdp.num_actions:
        raise ValueError("min_action_probability must lie in [0, 1 / num_actions)")


def _cem_elite_update(
    probabilities: np.ndarray,
    elite_decisions: np.ndarray,
    smoothing: float,
    min_action_probability: float,
) -> np.ndarray:
    num_actions = probabilities.shape[1]
    frequencies = np.stack(
        [np.mean(elite_decisions == action, axis=0) for action in range(num_actions)],
        axis=1,
    )
    updated = (1.0 - smoothing) * probabilities + smoothing * frequencies
    if min_action_probability > 0.0:
        residual = 1.0 - num_actions * min_action_probability
        updated = min_action_probability + residual * updated
    return updated / updated.sum(axis=1, keepdims=True)


def _deterministic_action_probabilities(
    mdp: FiniteHorizonMDP,
    policy: np.ndarray,
) -> np.ndarray:
    probabilities = np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )
    for state in mdp.decision_states or ():
        probabilities[state] = 0.0
        probabilities[state, int(policy[state])] = 1.0
    return probabilities


def train_cem(mdp: FiniteHorizonMDP, config: CEMConfig) -> BaselineResult:
    """Search for a deterministic stationary policy with tabular CEM.

    Each complete generation samples policies from an independent categorical
    distribution over decision states and updates that distribution from the
    highest-return elite policies.  An incomplete final generation is not used
    for an update, but every simulator transition it consumed is still counted.
    The returned policy is the best completely evaluated policy encountered.
    """

    _validate_cem_config(mdp, config)
    rng = np.random.default_rng(config.seed)
    probabilities = np.full(
        (mdp.num_decisions, mdp.num_actions),
        1.0 / mdp.num_actions,
        dtype=np.float64,
    )
    transitions = 0
    generation = 0
    history: list[dict[str, float]] = []
    best_policy: np.ndarray | None = None
    best_score = -np.inf

    while transitions < config.transition_budget:
        generation_start = transitions
        candidate_decisions: list[np.ndarray] = []
        candidate_scores: list[float] = []

        for _ in range(config.population_size):
            decisions = np.asarray(
                [
                    rng.choice(mdp.num_actions, p=probabilities[decision])
                    for decision in range(mdp.num_decisions)
                ],
                dtype=np.int64,
            )
            policy = mdp.policy_from_decisions(decisions)
            rollout_returns: list[float] = []
            for _ in range(config.rollouts_per_policy):
                returns, steps = mdp.sample_rollouts_with_steps(policy, 1, rng)
                transitions += steps
                rollout_returns.append(float(returns[0]))
                if transitions >= config.transition_budget:
                    break

            if len(rollout_returns) == config.rollouts_per_policy:
                score = float(np.mean(rollout_returns))
                candidate_decisions.append(decisions)
                candidate_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_policy = policy.copy()
            if transitions >= config.transition_budget:
                break

        completed_population = len(candidate_scores)
        updated = completed_population == config.population_size
        elite_mean = 0.0
        if updated:
            score_array = np.asarray(candidate_scores, dtype=np.float64)
            elite_count = max(1, int(np.ceil(config.elite_fraction * config.population_size)))
            elite_indices = np.argsort(score_array, kind="stable")[-elite_count:]
            elite_decisions = np.asarray(candidate_decisions, dtype=np.int64)[elite_indices]
            probabilities = _cem_elite_update(
                probabilities,
                elite_decisions,
                config.smoothing,
                config.min_action_probability,
            )
            elite_mean = float(score_array[elite_indices].mean())

        if completed_population:
            score_array = np.asarray(candidate_scores, dtype=np.float64)
            mean_score = float(score_array.mean())
            generation_best = float(score_array.max())
        else:
            mean_score = 0.0
            generation_best = best_score if np.isfinite(best_score) else 0.0
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)))
        history.append(
            {
                "generation": float(generation),
                "transitions": float(transitions),
                "completed_population": float(completed_population),
                "updated": float(updated),
                "mean_policy_return": mean_score,
                "best_policy_return": generation_best,
                "elite_mean_return": elite_mean,
                "sampling_entropy": float(entropy),
            }
        )
        if transitions == generation_start:
            raise RuntimeError("CEM collected no transitions")
        generation += 1

    if best_policy is None:
        mode_decisions = np.argmax(probabilities, axis=1)
        best_policy = mdp.policy_from_decisions(mode_decisions)
    return BaselineResult(
        _deterministic_action_probabilities(mdp, best_policy),
        transitions,
        history,
    )


@dataclass(frozen=True)
class DoubleQConfig:
    """Configuration for online tabular Double Q-learning."""

    transition_budget: int = 20_000
    learning_rate: float = 0.1
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    exploration_fraction: float = 0.5
    seed: int = 0
    log_every: int = 1_000


def _validate_double_q_config(config: DoubleQConfig) -> None:
    if config.transition_budget <= 0:
        raise ValueError("transition_budget must be positive")
    if not 0.0 < config.learning_rate <= 1.0:
        raise ValueError("learning_rate must lie in (0, 1]")
    if not 0.0 <= config.final_epsilon <= config.initial_epsilon <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")
    if not 0.0 < config.exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must lie in (0, 1]")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximizers = np.flatnonzero(values == np.max(values))
    return int(rng.choice(maximizers))


def _double_q_td_target(
    selection_q: np.ndarray,
    evaluation_q: np.ndarray,
    next_state: int,
    reward: float,
    done: bool,
    gamma: float,
    rng: np.random.Generator,
) -> float:
    if done:
        return float(reward)
    next_action = _random_argmax(selection_q[next_state], rng)
    return float(reward + gamma * evaluation_q[next_state, next_action])


def _epsilon_at_transition(config: DoubleQConfig, transitions: int) -> float:
    anneal_steps = max(1, int(np.ceil(config.exploration_fraction * config.transition_budget)))
    fraction = min(1.0, transitions / float(anneal_steps))
    return float(
        config.initial_epsilon
        + fraction * (config.final_epsilon - config.initial_epsilon)
    )


def _greedy_q_probabilities(q_values: np.ndarray) -> np.ndarray:
    maxima = np.max(q_values, axis=1, keepdims=True)
    maximizers = q_values == maxima
    return maximizers / maximizers.sum(axis=1, keepdims=True)


def train_double_q(mdp: FiniteHorizonMDP, config: DoubleQConfig) -> BaselineResult:
    """Train a stationary policy with online tabular Double Q-learning."""

    _validate_double_q_config(config)
    if float(mdp.initial[~mdp.terminal].sum()) <= 0.0:
        raise RuntimeError("Double Q-learning cannot collect nonterminal transitions")
    rng = np.random.default_rng(config.seed)
    q1 = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
    q2 = np.zeros_like(q1)
    transitions = 0
    history: list[dict[str, float]] = []
    completed_returns: deque[float] = deque(maxlen=100)
    last_td_error = 0.0
    last_epsilon = config.initial_epsilon

    while transitions < config.transition_budget:
        state = mdp.sample_initial(float(rng.random()))
        if mdp.terminal[state]:
            continue
        episode_return = 0.0
        discount = 1.0
        for time in range(mdp.horizon):
            epsilon = _epsilon_at_transition(config, transitions)
            if rng.random() < epsilon:
                action = int(rng.integers(mdp.num_actions))
            else:
                action = _random_argmax(q1[state] + q2[state], rng)
            next_state, reward = mdp.sample_transition(state, action, float(rng.random()))
            done = bool(mdp.terminal[next_state] or time == mdp.horizon - 1)

            if rng.random() < 0.5:
                target = _double_q_td_target(
                    q1,
                    q2,
                    next_state,
                    reward,
                    done,
                    mdp.gamma,
                    rng,
                )
                last_td_error = target - q1[state, action]
                q1[state, action] += config.learning_rate * last_td_error
            else:
                target = _double_q_td_target(
                    q2,
                    q1,
                    next_state,
                    reward,
                    done,
                    mdp.gamma,
                    rng,
                )
                last_td_error = target - q2[state, action]
                q2[state, action] += config.learning_rate * last_td_error

            episode_return += discount * reward
            discount *= mdp.gamma
            transitions += 1
            last_epsilon = epsilon
            state = next_state
            if done:
                completed_returns.append(float(episode_return))
            if transitions % config.log_every == 0:
                history.append(
                    {
                        "transitions": float(transitions),
                        "mean_episode_return_100": float(np.mean(completed_returns))
                        if completed_returns
                        else 0.0,
                        "epsilon": epsilon,
                        "td_error": float(last_td_error),
                    }
                )
            if done or transitions >= config.transition_budget:
                break

    if not history or history[-1]["transitions"] != float(transitions):
        history.append(
            {
                "transitions": float(transitions),
                "mean_episode_return_100": float(np.mean(completed_returns))
                if completed_returns
                else 0.0,
                "epsilon": float(last_epsilon),
                "td_error": float(last_td_error),
            }
        )
    if not np.isfinite(q1).all() or not np.isfinite(q2).all():
        raise FloatingPointError("non-finite value in Double Q-learning")
    return BaselineResult(_greedy_q_probabilities(q1 + q2), transitions, history)
