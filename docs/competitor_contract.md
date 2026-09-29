# Competitor and fairness contract

## Competitors

| Reported name | Implemented variant | Role |
|---|---|---|
| REINFORCE | Monte Carlo policy gradient with a learned state-value baseline | On-policy score-function baseline |
| PPO-Clip | Clipped surrogate with Monte Carlo advantages and an entropy bonus | On-policy clipped-surrogate baseline |
| Discrete SAC | Categorical actor, twin critics, target critics, fixed entropy temperature, replay | Off-policy entropy-regularized baseline |
| Categorical CEM | Cross-entropy search over complete deterministic stationary policies | Direct black-box policy-search baseline |
| Double Q-learning | Online tabular Double Q-learning with annealed epsilon-greedy exploration | Value-learning baseline |

Uniform random and exact finite-horizon dynamic programming are controls, not
learned competitors. The exact dynamic-programming policy is nonstationary and
is therefore an upper control rather than a like-for-like stationary-policy
competitor.

## What "standard" means here

The baselines retain the published core update rules, but are implemented in a
common tabular code path so that environment semantics, seeds, transition
accounting, and artifacts are identical. They are not byte-for-byte copies of
one software library. Each variant must be named precisely in the paper; for
example, the PPO implementation does not claim GAE or value clipping, and SAC
uses a fixed rather than automatically tuned entropy temperature.

The finite horizon is not included in the reported policy input. Consequently
all learned methods return stationary physical-state policies, matching the
thesis policy class. SAC nevertheless uses time-indexed twin critics and replay
records, and Double Q uses time-indexed value tables before projecting their
greedy choices into a stationary policy with observed state--time occupancy.
This prevents incorrect finite-horizon Bellman targets without giving either
competitor a richer reported policy. REINFORCE and PPO retain state-only value
baselines; this can increase gradient variance on recurrent states but does not
change the policy-gradient expectation. A time-indexed-policy study, if added,
must augment the policy input for every method.

## Fair-comparison rules

1. The committed-policy study samples one complete deterministic policy and
   retains it for the episode. This changes the native deployment of stochastic
   baselines. Practical-control tables must also report native per-visit
   stochastic execution as a co-primary metric for REINFORCE, PPO, and SAC.
   Never describe committed-policy baseline scores as native behavior.
2. Every method uses the same maps, training-seed list, horizons, rewards, and
   evaluation streams.
3. Tuning allowance is matched in both environment transitions and core-hours.
   Method-specific grids are allowed; one global learning rate is not treated
   as fair tuning.
4. All simulator interactions used by proposal training, SMC, policy-bank
   evaluation, or sample-average MCMC count toward the proposed method's total.
   Exact dynamic-programming evaluations are logged separately with wall time
   and value-evaluation count.
5. A generally applicable aid, such as count-based exploration, must be offered
   to compatible competitors or reported as a separate shared exploration
   ablation. It cannot silently benefit only the proposed method.
6. Baseline defects are fixed; competitors are never weakened. Improvements
   specific to the policy-inference target--importance correction, temperature
   exchange, complete-policy commitment, and target diagnostics--belong to the
   proposed method and are isolated in ablations.
7. Failed, timed-out, and numerically invalid runs remain in the accounting.

## Information-access strata

Results are never pooled across unequal information access:

- **Known-model target fidelity:** exact dynamic programming inside policy
  tempering is compared with exact enumeration or other known-model controls.
  This is the primary test of whether the desired policy distribution is
  sampled correctly.
- **Simulator-only control:** direct policy ELBO (including simulator-trained
  PPO initialization), or fixed-tape sample-average tempering without
  exact-model structural construction and learned competitors receive the same
  generative-simulator access and matched total allowance. This is the control comparison,
  while explicitly recognizing that finite-tape inference targets an
  approximation to the expected-return Gibbs distribution.
- **Model-assisted rollout scoring:** fixed-tape runs with the current modal
  path/recovery catalog also query exact successor probabilities. They belong
  to their own model-assisted stratum, not the simulator-only comparison.
- **Cross-stratum results:** exact-model tempering may be shown beside
  simulator-only learners to explain attainable behavior or computational
  tradeoffs, but it is labeled descriptive and cannot support a sample-efficiency
  superiority claim.

The nonstationary dynamic-programming oracle remains an upper control in both
strata. Model evaluations, simulator transitions, and guide-training
transitions are reported in separate columns rather than converted into an
arbitrary common unit.

The existing `transition_budget` limits PPO guide training before MCMC. It is
not a total-method cap. New result metadata labels this explicitly; matched
total-budget dispatch is still required. Include the frozen guide and continued
PPO under the same total cost as attribution controls.

For direct ELBO, PPO-warm-start ELBO and the five baselines, new configurations
can now set `strict_transition_budget: true`. This caps the entire method's
simulator training, including the guide, and reports unused allowance. The
baseline update implementations remain unchanged; complete-episode trainers
receive a conservatively reserved stopping threshold. MCMC/SMC experiment
branches currently reject strict mode. See `strict_budget_contract.md`.

## Current audit limitations

- All current GridWorld experiments use `gamma = 1`. Discounted-policy-gradient
  semantics must be retested before using `gamma < 1`.
- All reported policies are stationary even though the finite-horizon dynamic-
  programming upper control is nonstationary. That control is therefore an
  upper bound, not a same-class oracle.
- PPO count bonuses modify training rewards. Results with this aid are labeled
  exploration-assisted and are not compared against unassisted competitors as
  if the training conditions were identical.
