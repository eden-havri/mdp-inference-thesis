from __future__ import annotations

import numpy as np
import torch

import mdp_inference.policy as policy_module
from mdp_inference.examples import two_step_choice_mdp
from mdp_inference.policy import (
    DirectELBOTrainConfig,
    TabularPolicyDistribution,
    exact_factorized_elbo,
    train_direct_elbo,
)


def test_exact_factorized_elbo_has_finite_gradient() -> None:
    mdp = two_step_choice_mdp()
    distribution = TabularPolicyDistribution(mdp)
    objective = exact_factorized_elbo(distribution, mdp, beta=1.0)
    objective.backward()
    assert torch.isfinite(objective)
    assert distribution.logits.grad is not None
    assert torch.isfinite(distribution.logits.grad).all()
    # At state 1, action 1 dominates action 0.
    assert distribution.logits.grad[1, 1] > 0.0
    assert distribution.logits.grad[1, 0] < 0.0


def test_enumerated_score_gradient_matches_exact_elbo_gradient() -> None:
    mdp = two_step_choice_mdp()
    initial_logits = torch.tensor([[0.3, -0.2], [-0.4, 0.7]], dtype=torch.float64)

    exact_distribution = TabularPolicyDistribution(mdp, initial_logits)
    exact_objective = exact_factorized_elbo(exact_distribution, mdp, beta=0.8)
    exact_objective.backward()
    exact_gradient = exact_distribution.logits.grad.detach().clone()

    score_distribution = TabularPolicyDistribution(mdp, initial_logits)
    probabilities = score_distribution.probabilities().detach().numpy()
    expected_surrogate = torch.zeros((), dtype=torch.float64)
    log_prior = -mdp.num_decisions * np.log(mdp.num_actions)
    for first in range(mdp.num_actions):
        for second in range(mdp.num_actions):
            decisions = torch.tensor([[first, second]], dtype=torch.long)
            log_q = score_distribution.log_prob_decisions(decisions)[0]
            sampling_probability = probabilities[0, first] * probabilities[1, second]
            policy = mdp.policy_from_decisions((first, second))
            signal = 0.8 * mdp.expected_return(policy) + log_prior - float(log_q.detach())
            expected_surrogate = expected_surrogate + sampling_probability * signal * log_q
    expected_surrogate.backward()
    score_gradient = score_distribution.logits.grad.detach()
    assert torch.allclose(score_gradient, exact_gradient, atol=1e-12, rtol=1e-12)


def test_direct_elbo_training_improves_exact_objective() -> None:
    mdp = two_step_choice_mdp()
    initial = TabularPolicyDistribution(mdp)
    initial_objective = float(exact_factorized_elbo(initial, mdp, beta=1.0).detach())
    trained, _ = train_direct_elbo(
        mdp,
        DirectELBOTrainConfig(
            iterations=500,
            num_policy_samples=64,
            rollouts_per_policy=1,
            learning_rate=2e-2,
            seed=7,
        ),
    )
    final_objective = float(exact_factorized_elbo(trained, mdp, beta=1.0).detach())
    probabilities = trained.probabilities().detach().numpy()
    assert final_objective > initial_objective + 0.2
    assert probabilities[1, 1] > 0.75
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_direct_elbo_warm_start_is_observed_before_update_without_mutating_input(
    monkeypatch,
) -> None:
    mdp = two_step_choice_mdp()
    initial_logits = torch.tensor(
        [[0.75, -0.25], [-1.5, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    initial_snapshot = initial_logits.detach().clone()
    observed_logits: list[torch.Tensor] = []

    def capture_first_objective(distribution, *_args, **_kwargs):
        observed_logits.append(distribution.logits.detach().clone())
        return policy_module.DirectELBOSample(
            surrogate=distribution.logits.sum(),
            estimated_elbo=0.0,
            mean_return=0.0,
            learning_signal_std=0.0,
            simulator_steps=0,
        )

    monkeypatch.setattr(policy_module, "direct_elbo_sample", capture_first_objective)
    trained, _ = policy_module.train_direct_elbo(
        mdp,
        DirectELBOTrainConfig(
            iterations=1,
            num_policy_samples=1,
            rollouts_per_policy=1,
            learning_rate=0.1,
            seed=3,
        ),
        initial_logits=initial_logits,
    )

    assert len(observed_logits) == 1
    assert torch.equal(observed_logits[0], initial_snapshot)
    assert torch.equal(initial_logits.detach(), initial_snapshot)
    assert initial_logits.grad is None
    assert trained.logits.data_ptr() != initial_logits.data_ptr()
    assert not torch.equal(trained.logits.detach(), initial_snapshot)
