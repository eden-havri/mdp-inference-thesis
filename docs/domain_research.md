# Domain-extension decision

## Decision

Keep stochastic GridWorld as the complete primary study. Add one cheap
non-spatial domain only after the GridWorld method and baselines pass their
medium-tier gates. The best cheap addition is tabular Blackjack. If time and
compute remain, add Tireworld as the stronger planning-domain extension.

Do not add several environments merely to increase the table size. A second
domain is useful only if it tests a failure mode that GridWorld cannot.

## Evidence-based comparison

| Domain | What it adds | Fit to current policy target | Implementation and experiment cost | Decision |
|---|---|---|---|---|
| Blackjack-v1 | Short stochastic episodes; non-spatial state; high return variance; exact tabular dynamics can be derived | Very good: discrete states/actions and deterministic stationary policies | Low. Two actions and small observations; exact evaluation is practical | Add after GridWorld calibration |
| Tireworld | Stochastic failures, recovery resources, sparse success, long-term planning, multiple routes | Strong scientifically; directly tests whether averaging return before exponentiation matters under risk | Medium-high. Requires a factored-state adapter, action masks, instances, and a trusted simulator | Best second headline domain if schedule permits |
| Taxi-v3 | Action masks, pickup/drop-off subgoals, 500 discrete states | Technically easy, but current standard version is deterministic and still spatial | Low-medium | Optional smoke test, not a headline domain |
| Academic Advising | True stochasticity, multivalued variables, concurrent actions, factored state | Interesting but no longer a small tabular extension; requires structured policies and constraints | High | Stretch/future work only |
| FrozenLake / Cliff Walking | Familiar sparse-reward controls | Mostly duplicates the existing GridWorld axes | Low | Do not add |
| Continuous-control domains | Continuous state/action generalization | Requires a new reference measure, policy representation, and inference machinery | Very high | Out of scope for this thesis version |

## Why Blackjack first

The official Gymnasium formulation has observations consisting of player sum,
dealer up-card, and usable-ace status; actions are hit or stand; rewards are
win/loss/draw; and card draws are stochastic. This creates a compact,
non-geometric test in which return variance is intrinsic rather than introduced
by movement slip. It is cheap enough to run exact controls and many seeds, so a
negative result is inexpensive and informative.

Source: Gymnasium's official Blackjack tutorial and environment documentation:
https://gymnasium.farama.org/tutorials/training_agents/blackjack_q_learning/

## Why Tireworld second

Tireworld is a recognized probabilistic-planning benchmark designed to require
planning ahead under uncertainty. A flat tire can force recovery decisions and
the state includes resources, so it tests more than obstacle navigation. It is
also naturally compatible with a generative simulator. The pyRDDLGym project
provides a maintained Gymnasium-compatible route from RDDL factored MDPs to a
simulator, and its repository supports established planning and RL baselines.

Primary/official sources:

- Bonet and Geffner's probabilistic-planning work and Tireworld description:
  https://www.cs.cmu.edu/afs/cs.cmu.edu/project/jair/pub/volume24/bonet05a.pdf
- pyRDDLGym paper: https://arxiv.org/abs/2211.05939
- pyRDDLGym documentation: https://pyrddlgym.readthedocs.io/en/latest/start.html

Before adopting Tireworld, a diagnostic instance must pass transition,
termination, reward, action-mask, and exact-small-instance checks. RDDL/PROST
integration should not be allowed to delay the primary GridWorld confirmation.

## Why not Academic Advising now

The published domain was created specifically to add true stochasticity,
multivalued variables, and concurrent actions to probabilistic-planning
benchmarks. Those properties make it scientifically valuable, but they also
move the project from a tabular complete-policy distribution to structured
factored policies and constrained concurrent actions. That is a separate method
extension rather than a small extra experiment.

Source: Guerin et al., *The Academic Advising Planning Domain*:
https://www.research.ed.ac.uk/en/publications/the-academic-advising-planning-domain/

## Fail-fast extension order

1. Finish exact and sample-average GridWorld validation.
2. Freeze competitor implementations and medium-tier settings.
3. Implement Blackjack dynamics plus exact return/success tests.
4. Run one seed per method at a tiny budget; stop if semantics disagree.
5. Run three-seed timing and learning gates; add it to the paper only if at
   least the controls and two learned methods behave sensibly.
6. Start Tireworld only after the GridWorld confirmatory manifest is frozen.
7. Treat Taxi and Academic Advising as optional diagnostics/future work, not as
   requirements for thesis completion.
