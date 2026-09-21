from __future__ import annotations

from mdp_inference.baselines import (
    DiscreteSACConfig,
    PPOConfig,
    ReinforceConfig,
    train_discrete_sac,
    train_ppo,
    train_reinforce,
)
from mdp_inference.examples import two_step_choice_mdp
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
    )
    for result in runs:
        value = evaluate_stationary_stochastic_policy(mdp, result.action_probabilities)
        assert 1_000 <= result.transitions < 1_000 + mdp.horizon
        assert value > random_value + 0.15
