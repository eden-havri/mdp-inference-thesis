from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .exact import _enumerated_policies
from .mdp import FiniteHorizonMDP


class TabularPolicyDistribution(nn.Module):
    """Mean-field distribution over deterministic stationary policies."""

    def __init__(self, mdp: FiniteHorizonMDP, initial_logits: torch.Tensor | None = None):
        super().__init__()
        shape = (mdp.num_decisions, mdp.num_actions)
        if initial_logits is None:
            initial_logits = torch.zeros(shape, dtype=torch.float64)
        if tuple(initial_logits.shape) != shape:
            raise ValueError(f"initial_logits must have shape {shape}")
        self.logits = nn.Parameter(initial_logits.detach().clone())
        self.decision_states = tuple(mdp.decision_states or ())
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions

    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def sample_decisions(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        probabilities = self.probabilities()
        # torch.multinomial treats each row independently. Transpose to [M, D].
        decisions = torch.multinomial(
            probabilities,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        ).transpose(0, 1)
        return decisions, self.log_prob_decisions(decisions)

    def log_prob_decisions(self, decisions: torch.Tensor) -> torch.Tensor:
        if decisions.ndim != 2 or decisions.shape[1] != len(self.decision_states):
            raise ValueError("decisions must have shape [M, D]")
        log_probabilities = torch.log_softmax(self.logits, dim=-1)
        expanded = log_probabilities.unsqueeze(0).expand(decisions.shape[0], -1, -1)
        return expanded.gather(2, decisions.to(torch.long).unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def log_prob_terms(self, decisions: torch.Tensor) -> torch.Tensor:
        if decisions.ndim != 2 or decisions.shape[1] != len(self.decision_states):
            raise ValueError("decisions must have shape [M, D]")
        log_probabilities = torch.log_softmax(self.logits, dim=-1)
        expanded = log_probabilities.unsqueeze(0).expand(decisions.shape[0], -1, -1)
        return expanded.gather(2, decisions.to(torch.long).unsqueeze(-1)).squeeze(-1)

    def negative_kl_to_uniform_prior(self) -> torch.Tensor:
        probabilities = self.probabilities()
        log_probabilities = torch.log_softmax(self.logits, dim=-1)
        entropy = -torch.sum(probabilities * log_probabilities)
        return entropy - len(self.decision_states) * np.log(float(self.num_actions))

    def full_policies(self, decisions: torch.Tensor) -> np.ndarray:
        decision_array = decisions.detach().cpu().numpy()
        policies = np.zeros((decision_array.shape[0], self.num_states), dtype=np.int64)
        for decision_index, state in enumerate(self.decision_states):
            policies[:, state] = decision_array[:, decision_index]
        return policies


@dataclass(frozen=True)
class DirectELBOSample:
    surrogate: torch.Tensor
    estimated_elbo: float
    mean_return: float
    learning_signal_std: float
    simulator_steps: int


def direct_elbo_sample(
    distribution: TabularPolicyDistribution,
    mdp: FiniteHorizonMDP,
    num_policy_samples: int,
    rollouts_per_policy: int,
    beta: float,
    rng: np.random.Generator,
    torch_generator: torch.Generator | None = None,
    leave_one_out_baseline: bool = True,
) -> DirectELBOSample:
    """Unbiased score-gradient sample for the expected-return ELBO.

    Returns are averaged *before* entering the learning signal. The detached
    score-function surrogate has the exact ELBO gradient in expectation. A
    leave-one-out baseline is unbiased because each sample's baseline depends
    only on the other policy samples.
    """

    decisions, log_q = distribution.sample_decisions(num_policy_samples, torch_generator)
    log_q_terms = distribution.log_prob_terms(decisions)
    policies = distribution.full_policies(decisions)
    returns_list: list[float] = []
    simulator_steps = 0
    visited_mask = torch.zeros_like(log_q_terms)
    state_to_decision = mdp.state_to_decision
    for policy in policies:
        rollout_returns, rollout_steps, visited_states = mdp.sample_rollouts_with_metadata(
            policy, rollouts_per_policy, rng
        )
        returns_list.append(float(rollout_returns.mean()))
        simulator_steps += rollout_steps
        sample_index = len(returns_list) - 1
        for state in visited_states:
            visited_mask[sample_index, state_to_decision[state]] = 1.0
    returns = np.asarray(returns_list, dtype=np.float64)
    returns_tensor = torch.as_tensor(returns, dtype=log_q.dtype, device=log_q.device)
    reward_signal = beta * returns_tensor

    if leave_one_out_baseline and num_policy_samples > 1:
        baseline = (reward_signal.sum() - reward_signal) / float(num_policy_samples - 1)
    else:
        baseline = torch.zeros_like(reward_signal)
    visited_log_q = torch.sum(visited_mask * log_q_terms, dim=1)
    regularizer = distribution.negative_kl_to_uniform_prior()
    surrogate = ((reward_signal - baseline).detach() * visited_log_q).mean() + regularizer
    estimated_elbo_tensor = reward_signal.mean() + regularizer.detach()
    return DirectELBOSample(
        surrogate=surrogate,
        estimated_elbo=float(estimated_elbo_tensor),
        mean_return=float(returns.mean()),
        learning_signal_std=float(reward_signal.std(unbiased=False).detach()),
        simulator_steps=simulator_steps,
    )


def exact_factorized_elbo(
    distribution: TabularPolicyDistribution,
    mdp: FiniteHorizonMDP,
    beta: float,
) -> torch.Tensor:
    """Differentiable exact ELBO, restricted to tiny enumeratable MDPs."""

    policies = _enumerated_policies(mdp)
    decisions = torch.as_tensor(
        policies[:, list(mdp.decision_states or ())],
        dtype=torch.long,
        device=distribution.logits.device,
    )
    log_q = distribution.log_prob_decisions(decisions)
    q = torch.exp(log_q)
    returns = torch.as_tensor(
        [mdp.expected_return(policy) for policy in policies],
        dtype=distribution.logits.dtype,
        device=distribution.logits.device,
    )
    log_prior = -float(mdp.num_decisions) * np.log(float(mdp.num_actions))
    return torch.sum(q * (beta * returns + log_prior - log_q))


@dataclass(frozen=True)
class DirectELBOTrainConfig:
    iterations: int = 1_000
    num_policy_samples: int = 32
    rollouts_per_policy: int = 8
    beta: float = 1.0
    learning_rate: float = 3e-2
    seed: int = 0
    transition_budget: int | None = None


def train_direct_elbo(
    mdp: FiniteHorizonMDP,
    config: DirectELBOTrainConfig,
    initial_logits: torch.Tensor | None = None,
) -> tuple[TabularPolicyDistribution, list[dict[str, float]]]:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    distribution = TabularPolicyDistribution(mdp, initial_logits=initial_logits)
    optimizer = torch.optim.Adam(distribution.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=distribution.logits.device)
    torch_generator.manual_seed(config.seed)
    history: list[dict[str, float]] = []
    total_simulator_steps = 0
    for iteration in range(config.iterations):
        sample = direct_elbo_sample(
            distribution,
            mdp,
            config.num_policy_samples,
            config.rollouts_per_policy,
            config.beta,
            rng,
            torch_generator,
        )
        optimizer.zero_grad()
        (-sample.surrogate).backward()
        optimizer.step()
        total_simulator_steps += sample.simulator_steps
        history.append(
            {
                "iteration": float(iteration),
                "estimated_elbo": sample.estimated_elbo,
                "mean_return": sample.mean_return,
                "learning_signal_std": sample.learning_signal_std,
                "simulator_steps": float(total_simulator_steps),
            }
        )
        if config.transition_budget is not None and total_simulator_steps >= config.transition_budget:
            break
    return distribution, history
