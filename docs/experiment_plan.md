# Fail-fast experiment and Slurm plan

## Purpose and decision rule

The experiment program should answer two separate questions:

1. **Correctness:** does the current method implement the claimed distribution over complete policies and produce valid estimates?
2. **Performance:** under the same simulator access and number of environment transitions, does it learn policies that are better or more reliable than standard policy-gradient and entropy-regularized baselines?

No large cluster run should start until the corresponding smaller stage passes. The primary resource budget is the number of **logical environment transitions represented by training**, with wall-clock time, peak memory, and simulator calls logged as additional costs. For a particle method, a sweep with `N` particles and horizon `H` consumes `N * H` logical transitions even if vectorization or shared randomness reduces the number of function invocations.

The confirmatory result is fixed-budget performance on held-out maps. Hyperparameter pruning and early stopping belong only to smoke tests and validation; they must not select favorable stopping points on the final test set.

Comparisons are stratified by information access. Exact-model tempering is used
for target-fidelity and mixing claims against exact known-model controls.
Simulator-only fixed-tape tempering is compared with the five learned
competitors under matched transition budgets. Exact-model results may be shown
next to model-free learners only as a descriptive reference, never as evidence
of superior sample efficiency. Model evaluations, simulator transitions, and
guide-training transitions remain separate cost columns.

## Experimental contract to freeze first

Before implementing more algorithms, write one machine-readable experiment contract that fixes:

- transition, wall, terminal, timeout, reward, and action-success semantics;
- whether rewards are discounted and the exact discount factor;
- the horizon rule for each map;
- the training-transition accounting rule for every method;
- the policy object being evaluated;
- the map-generation and map-acceptance rules;
- the checkpoint schedule and primary metric;
- the validation maps, test maps, and all seed lists.

The inverse temperature is part of the scientific target, not a free optimizer
setting to choose after seeing control scores. Exact diagnostic maps report a
predeclared beta sensitivity grid. The headline GridWorld comparison freezes
one beta on development map seed 0 before any held-out run; current development
work uses `beta = 64`. Any secondary beta is labeled a target-sensitivity
analysis and carries its own mixing and compute diagnostics.

The policy-evaluation definition needs special care. Report these as distinct quantities when the method represents a distribution over deterministic policies:

- **Complete-policy rollout:** sample one deterministic action for every state once per episode and retain it on revisits.
- **Statewise stochastic rollout:** resample an action from the policy distribution on every visit.

These are generally different in environments with revisitation. The complete-policy rollout is the primary evaluation for a claim about distributions over complete deterministic policies; the stochastic rollout is a useful secondary comparison shared with standard baselines.

## Stage gates

| Stage | Scope | Provisional compute | Pass condition | Stop condition |
|---|---|---:|---|---|
| 0. Environment and evaluator | Local, diagnostic maps only | 5–30 minutes | Hand-computed transitions/rewards agree; exact oracle bounds all sampled policies; both evaluation modes are tested | Any disagreement in state indexing, terminal handling, horizon, rewards, or transition probabilities |
| 1. Determinism and artifacts | One diagnostic and one easy map, two repeated runs | 10–30 minutes | Same configuration and seed reproduce metrics and checkpoint hashes; different seeds change stochastic traces; output paths never collide | Missing seed control, non-reproducible run, overwritten artifact, or unseeded evaluator |
| 2. Algorithm smoke tests | One easy map, 3 training seeds, tiny budgets | 1–4 CPU-hours total | All methods finish; values remain finite; learned policies show improvement direction over random; oracle remains an upper bound | NaN/Inf, invalid probabilities, zero/absent gradients, no learning signal, or learned return outside valid bounds |
| 3. Cluster timing pilot | One fixed map per GridWorld tier, 3 seeds, all learned methods | 54 learning jobs plus cheap controls | At least 90% clean completion; measured throughput, peak memory, and output size permit a credible final budget | Repeated timeout/OOM, unstable result schema, excessive per-checkpoint I/O, or unexplained local/cluster mismatch |
| 4. GridWorld calibration | Two maps per tier, 5 seeds, validation only | 180 learning jobs for 6 learned methods | Hyperparameters are frozen; easy cases are solved; medium and hard cases show nontrivial separation from both random and oracle | Baselines cannot pass easy maps, hard maps are indistinguishable from random for every method, or maps are trivially solved by every method |
| 5. GridWorld confirmation | Three held-out maps per tier, 10 seeds, frozen settings | 540 learning jobs for 6 learned methods | Predeclared metrics and intervals are computed from all valid runs; no post-hoc tuning | Any change to maps, seeds, rewards, budgets, or hyperparameters after inspecting test outcomes |
| 6. Tireworld extension | Only after Stage 5 is stable | Start with 12–36 jobs | Small instance matches exact calculations; semantics and cost accounting reuse the same contract | Unvalidated action masking, terminal semantics, or transition probabilities |

The job counts above assume the proposed method plus REINFORCE, PPO, discrete
SAC, categorical CEM, and Double Q-learning. Random and oracle controls are
computed once per map and evaluation setting rather than retrained for each
seed.

## Baseline implementation order

Implement and validate baselines in this order. Do not begin tuning the next learned baseline until the previous gate passes.

1. **Random policy.** This establishes reward scale, chance success, and evaluator health. Evaluate exactly when feasible.
2. **Exact finite-horizon oracle.** Dynamic programming gives the attainable upper bound under the same transition, reward, horizon, and policy class. Also retain an exact soft/entropy-regularized oracle as a diagnostic when comparing action marginals.
3. **REINFORCE.** Use a learned or state-dependent baseline if desired, but record the variant explicitly. Its simplicity makes it the first end-to-end gradient and budget-accounting test.
4. **PPO.** Add only after REINFORCE learns the easy tier. Log clipping rate, approximate divergence, policy entropy, and value loss.
5. **Double Q-learning.** Use time-indexed value tables for correct finite-horizon targets and report the declared stationary projection.
6. **Discrete SAC.** Add last because it introduces replay, target updates, entropy-temperature behavior, and more ways for budget accounting to diverge. Count only newly collected environment transitions as sample cost; log replay updates separately.
7. **Categorical CEM.** Treat complete deterministic policies as the search objects and count every rollout used to score a candidate.

The proposed particle method participates in every learned-method stage. For it, additionally log particle count, horizon, sweep count, effective sample size, resampling events, weight entropy, unique ancestors, and both logical transitions and actual simulator invocations.

## GridWorld suite

### Diagnostic tier: correctness, not headline performance

- Sizes: 2x2, 2x3, and 3x3.
- Enumerate all deterministic policies when tractable.
- Compute exact expected return for each policy and the normalized target distribution.
- Compare estimated state-action marginals and policy probabilities using total variation, Jensen–Shannon divergence, and calibration plots.
- Include at least one map that revisits a state under stochastic dynamics; this exposes the difference between complete-policy and statewise-stochastic evaluation.

### Easy tier

- Frozen size: 8x8.
- Obstacles: 6 walls, or 9.4% of cells.
- Action success probability: 0.90.
- Audited shortest-path length: 14 steps; audited reachability: 100%.
- Horizon: 64.
- Purpose: integration tests, fast tuning, and confirmation that every learned baseline can beat random.

### Medium tier

- Frozen size: 16x16.
- Obstacles: 51 walls, or 19.9% of cells.
- Action success probability: 0.80.
- Audited shortest-path length: 30–32 steps; audited reachable fraction: 97.6–100%.
- Horizon: 160.
- The frozen maps include one to five single-cell graph bottlenecks and many
  distinct shortest action sequences.
- Purpose: expose stochasticity, revisitation, multiple useful routes, and optimizer stability.

### Hard tier

- Frozen size: 32x32.
- Obstacles: 307 walls, or 30.0% of cells.
- Action success probability: 0.75.
- Audited shortest-path length: 62–72 steps; audited reachable fraction: 94.8–97.8%.
- Horizon: 384.
- The frozen maps span one to seventeen single-cell graph bottlenecks.
- Purpose: measure scaling, particle degeneracy, sparse-success behavior, and wall-clock cost.

Every frozen map has a content hash and a separate audited-characteristics
record containing reachability, shortest-path length and count, and the number
of single-cell start--goal bottlenecks. Never regenerate a frozen reported map
from a seed. Seed 0 is development-only; seeds 1--3 are the three frozen
held-out maps for each reported tier. Future candidates are rejected unless start and goal are
connected and their declared tier constraints are met. A single primary action-
success level per tier keeps the core design affordable; extra stochasticity or
obstacle-density levels are one-factor stress tests after confirmation.

## Budgets and checkpoints

Use geometric checkpoints so one run supplies a complete learning curve:

`10k, 30k, 100k, 300k, 1M, 3M` training transitions.

Provisional maximum budgets are:

| Tier | Smoke maximum | Calibration maximum | Confirmatory maximum |
|---|---:|---:|---:|
| Easy | 30k | 100k | 250k |
| Medium | 100k | 300k | 1M |
| Hard | 300k | 1M | 3M |

These maxima are starting points, not promises. Stage 3 timing data should determine whether they fit the available core-hour budget. Train once to the maximum and checkpoint along the way; do not launch a separate job for every checkpoint.

For the proposed 540-run confirmation, these maxima imply about 765 million training transitions:

- easy: `3 maps * 10 seeds * 6 methods * 250k = 45M`;
- medium: `3 * 10 * 6 * 1M = 180M`;
- hard: `3 * 10 * 6 * 3M = 540M`.

If measured end-to-end throughput is `q` training transitions per second, the
lower-bound CPU time is `765M / q`. For example, the training portion is about
1,063 core-hours at 200 transitions/s, 213 core-hours at 1,000 transitions/s,
or 106 core-hours at 2,000 transitions/s, before evaluation and scheduler
overhead. Replace these planning examples with measured per-method estimates
after Stage 3; never extrapolate one method's speed to another.

For each homogeneous job class, set requested wall time to `1.5 * observed p95 runtime + evaluation allowance` and requested memory to `1.25 * observed p95 peak memory`, rounded up to scheduler units. Begin timing pilots with CPU jobs; request accelerators only if a controlled benchmark shows a worthwhile end-to-end speedup at the actual batch sizes.

## Seeds and experimental units

Keep independent, explicit seeds for:

- `map_seed` for candidate map construction only;
- `initialization_seed` for parameters;
- `training_seed` for trajectories, particles, replay sampling, and minibatches;
- `evaluation_seed` for common evaluation noise;
- `policy_sample_seed` for complete-policy evaluation.

Derive them deterministically from a stable run identifier and store the realized integer values in the manifest and result. The same ordered training-seed list must be used by all methods on a map. Use 3 seeds for smoke/timing, 5 for calibration, and 10 for the confirmatory suite. Expand to 20 only if the calibration variance or a prospective power calculation shows that 10 is inadequate.

Maps and training seeds are the inferential units; thousands of rollouts from one trained policy are evaluation precision, not thousands of independent experimental samples. Prefer exact finite-horizon evaluation on tabular maps. Otherwise use a fixed, sufficiently large common evaluation stream at every checkpoint and report its Monte Carlo standard error.

## Metrics and comparisons

Primary metrics:

- normalized oracle gap, `(J_oracle - J_method) / max(epsilon, J_oracle - J_random)`;
- area under the learning curve versus training transitions;
- final expected return at the tier's fixed budget;
- goal-reaching probability within the horizon.

Secondary metrics:

- wall-clock time and peak memory;
- transitions to reach fixed return or success thresholds;
- return standard deviation and lower-tail return;
- policy entropy and state visitation coverage;
- run failure/timeout rate;
- particle effective sample size, unique ancestry, and resampling count for the proposed method;
- total variation or Jensen–Shannon error against the exact policy distribution on diagnostic maps.

Use paired comparisons because every method sees the same map and seed list. Report per-map curves, the median paired difference, an interval from a hierarchical bootstrap over maps then seeds, and an aggregate robust score such as the interquartile mean. Do not pool evaluation episodes as independent samples. Report all failed runs in the denominator and predeclare how safety-stopped runs are scored.

## Hyperparameter search without budget leakage

Use validation maps only and give every learned method the same total tuning allowance, expressed both in training transitions and core-hours. A cost-effective successive-halving template is:

1. Eight configurations, one easy validation map, one seed, 100k transitions.
2. Top three configurations, two validation maps, three seeds, up to 300k transitions.
3. One selected configuration per tier, checked on all calibration maps and five seeds.
4. Freeze the configuration and code revision before generating confirmatory results.

Selection should use predeclared normalized-gap AUC, with failure assigned the worst score. The final comparison includes the cost of tuning in a separate compute table even though tuning transitions are not part of the per-run learning curve.

## Early stopping and failure policy

### Immediate safety stop for any run

Exit nonzero and write `failure.json` if any of the following occurs:

- a NaN/Inf appears in parameters, loss, gradients, probabilities, returns, or particle weights;
- probabilities are negative or cannot be normalized;
- the environment produces an invalid state/action or return outside proven bounds;
- memory crosses a configured limit or no progress heartbeat is written for a configured interval;
- a checkpoint or metrics record cannot be written and reread;
- the process receives a scheduler preemption signal and cannot save a resumable checkpoint.

### Validation-only futility pruning

After at least 30% of the candidate's maximum budget, prune a tuning candidate if it remains statistically indistinguishable from or worse than random at three consecutive checkpoints, or is clearly dominated by another candidate on both AUC and runtime. Retain the records and label the run `pruned`, not `complete`.

### Stage-level stops

- Do not continue past Stage 0 if random and oracle do not bracket independently simulated policies.
- Do not continue past Stage 2 if REINFORCE cannot improve on random on the easy map after a small, documented tuning pass; first diagnose environment, gradients, and evaluation.
- Do not scale a job class if more than 10% of its timing-pilot runs fail or exceed the estimate.
- Redesign the hard maps if every method is either at random performance or at the oracle ceiling; either case provides little comparative information.
- If calibration shows the target effect is smaller than the detectable effect at 10 seeds, increase seeds based on a prospective calculation or explicitly state that the planned study is underpowered.

Confirmatory runs use fixed budgets. They stop early only for the safety rules above, not because a current result looks good or bad. The primary endpoint is the last scheduled checkpoint, and AUC uses the full predeclared range.

## Required per-run artifacts

Every run directory should be self-contained and append-only until finalization:

```text
results/<study>/<code_revision>/<run_id>/
  config.json
  manifest_row.json
  map.json
  environment.json
  metrics.jsonl
  evaluation.jsonl
  resources.json
  checkpoint_latest.*
  diagnostic_trace.json
  stdout.log
  stderr.log
  failure.json          # only on failure/pruning
  DONE                  # written last, only after validation
```

Record the code revision and dirty-state indicator, command line, host, scheduler job/array IDs, start/end timestamps, all seeds, map hash, software versions, CPU/thread settings, training transitions, evaluation transitions, simulator calls, elapsed time, CPU time, and peak resident memory. Method-specific telemetry should include gradient norms, action entropy, update counts, replay statistics, or particle diagnostics as applicable.

`diagnostic_trace.json` should contain a bounded trace from the first episode/sweep and one late episode/sweep: states, actions, rewards, terminals, cumulative return, and relevant policy/weight values. It must be small enough to retain for every run. Store a few evaluation trajectories and a state-visitation heatmap at each major checkpoint, but avoid writing an image on every iteration.

Write into a temporary run directory, validate the schema, then atomically rename it and create `DONE`. A rerun must skip a valid `DONE` directory, resume a compatible partial checkpoint, or create a new attempt directory; it must never overwrite prior evidence. Logs and manifests must not contain credentials or other secrets.

## Slurm array workflow

### Manifest-driven jobs

Generate a CSV or JSONL manifest locally. One row equals one independent training run and includes at least:

`run_id, study, method, tier, map_path, map_hash, budget, checkpoint_schedule, all seeds, hyperparameter_set, code_revision, result_path`.

An array task selects exactly one row using `SLURM_ARRAY_TASK_ID`, validates it, prints a redacted summary, and invokes the runner. This is safer and easier to audit than constructing configurations inside the batch script.

### Homogeneous arrays

Split arrays by method, tier, and resource class so that one slow hard job does not force every easy job to reserve the same resources. Recommended launch sequence:

1. environment/oracle audit;
2. one-task cluster smoke per resource class;
3. 3-seed timing arrays;
4. validation arrays in baseline order;
5. an `afterany` audit job that lists missing, failed, and malformed outputs;
6. resubmission manifest containing only unresolved rows;
7. confirmatory arrays after settings are frozen;
8. aggregation only after the audit declares the manifest complete.

Start with an array concurrency cap of 2--4. Raise it only after confirming that
shared storage, logging, and scheduler policy tolerate the I/O pattern. Keep CPU
thread counts equal to allocated CPUs to avoid hidden oversubscription. Use a
preemption signal with enough notice to write a checkpoint, and test checkpoint
resume before relying on it.

The first measured medium pilots used one effective CPU core and under 300 MB
RAM. Current array rows therefore request the cluster minimum allocation with
2 GB RAM and 2 GB local scratch on the `main` CPU partition. Recompute requests
from the observed p95 for each final job class instead of copying these pilot
values to hard or extension domains.

### Preflight and audit checks

Before each submission:

- validate that every manifest row has a unique `run_id` and output path;
- verify that every map hash and code revision exists;
- estimate total transitions, array tasks, core-hours, and storage;
- run the first and last manifest rows locally or in a single cluster smoke task;
- ensure the working tree state and dependency snapshot are recorded;
- make the batch wrapper fail on any command error;
- inspect scheduler accounting for exit code, elapsed time, CPU utilization, and peak memory.

After an array completes, compare expected run IDs with valid `DONE` markers. Aggregate by manifest identity rather than by globbing whichever files happen to exist.

## Tireworld: deliberately later

Tireworld is a valuable second domain because stochastic failures, recovery resources, revisitation, sparse goal success, and route multimodality test behavior that rectangular grids can hide. It also adds action masking and more complex terminal/reward semantics, so introducing it before the GridWorld harness is trustworthy would multiply debugging ambiguity.

Proceed only after GridWorld confirmation:

1. Add transition-table unit tests and verify every state's legal actions sum to probability one.
2. On the smallest instance, compute exact finite-horizon value and goal probability for random, selected deterministic policies, and the oracle.
3. Verify complete-policy versus statewise-stochastic evaluation on states that can be revisited.
4. Run the proposed method and REINFORCE on 3 seeds at 100k transitions.
5. Add PPO, then discrete SAC, only after the first two methods pass.
6. Scale to medium instances at 500k and 2M checkpoints; choose larger instances only if success is neither zero nor saturated.

Treat Tireworld as a preregistered extension, not as a replacement for failed GridWorld results. Keep its statistics separate, then report a cross-domain aggregate only as a secondary analysis.

## Useful work in parallel with implementation

- Decide whether the thesis's primary claim is posterior fidelity over complete policies, expected-return optimization, or both; this determines the primary endpoint.
- Approve the reward, discount, horizon, terminal, and evaluation semantics in the experiment contract.
- Review the frozen maps visually and confirm that the proposed easy/medium/hard distinctions match the intended scientific question.
- Confirm cluster partitions, per-job limits, array limits, account quota, storage quota, and data-retention policy without placing credentials in files or chat logs.
- Set a hard maximum for core-hours and storage; the calibration stage can then choose seeds and budgets transparently.
- Confirm the desired thesis deadline and reserve time for failed-job recovery, analysis, and figure generation.
- Choose the exact success threshold and the smallest practically meaningful normalized-gap improvement before confirmatory runs.
- Prepare a short table of intended algorithmic variants and hyperparameters so that names such as “PPO” or “discrete SAC” do not conceal material implementation differences.

## Minimum useful study versus expansion

If time or allocation is tight, the minimum defensible study is the diagnostic
exact-policy suite plus two frozen maps per GridWorld tier, five seeds, the
proposed method, all five declared competitors, and the random/oracle controls
under matched accounting. That is sufficient to find correctness failures and
estimate effect sizes, but it should be labeled exploratory.

The confirmatory study uses three held-out maps per tier and ten seeds as specified above. Add extra obstacle densities, stochasticity levels, 20 seeds, and Tireworld only in that order and only when the preceding evidence justifies the additional cost.
