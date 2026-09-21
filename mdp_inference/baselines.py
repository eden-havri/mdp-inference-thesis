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
