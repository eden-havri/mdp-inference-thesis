# GridWorld scalability plan

## Executive decision

The current 16x16 medium grid is feasible but already wastes minutes per reported run, and the 32x32 hard grid is not economical with the default evaluator. The immediate problem is not the tabular policy: it is the dense `[S, A, S]` transition and reward representation combined with repeated exact evaluation of 2,000 sampled complete policies.

Use a fixed-branch transition representation for every generated GridWorld before launching medium/hard arrays. Keep the dense MDP only as a tiny-domain reference and compatibility path. Do not attempt exact policy enumeration outside diagnostic domains.

No production code is changed by this document.

## Evidence from the current code

The critical paths are:

- `mdp_inference/gridworld.py`: `gridworld_mdp` allocates both `transition` and `reward` as float64 arrays of shape `[S, 4, S]`; it also scans every length-`S` row while building rewards.
- `mdp_inference/mdp.py`: construction validates and then copies both dense tensors. Every sampled transition builds a cumulative sum over a length-`S` row. `expected_return` propagates a dense state distribution for every horizon step.
- `mdp_inference/oracles.py`: the finite-horizon oracle and stochastic-policy evaluator perform a length-`S` dot product for every `(time, state, action)` combination.
- `mdp_inference/experiment.py`: every run recomputes random controls and the oracle. It evaluates 2,000 random committed policies, then normally another 2,000 learned committed policies; particle methods also evaluate their returned particles exactly.
- `mdp_inference/exact.py`: policy enumeration is exponential in the number of decision states. Its two-million-policy guard still permits a dangerously large transient list/stack.
- `mdp_inference/replicated_smc.py`: each simulated particle transition samples from a dense length-`S` row, and every resampling event copies an `[N, S]` policy matrix.

### Measured reference timings

Single-process timings on the current workstation, using the existing `seed0` maps, provide a regression reference rather than a portable performance promise:

| Tier | `S` | Decision states | `H` | Dense tensors | Build | One committed-policy exact value | Random stochastic value | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy 8x8 | 64 | 57 | 64 | 0.25 MiB | 0.003 s | 0.0125 s | 0.049 s | 0.058 s |
| Medium 16x16 | 256 | 204 | 160 | 4.00 MiB | 0.030 s | 0.109 s | 0.476 s | 0.583 s |
| Hard 32x32 | 1,024 | 716 | 384 | 64.0 MiB | 0.445 s | 0.837 s | 5.03 s | 6.67 s |

The dense-tensor column counts the steady-state transition and reward arrays only. Construction peaks at roughly twice that amount because `FiniteHorizonMDP.__post_init__` copies both inputs.

At the hard-grid measured rate, 2,000 exact committed-policy values take about 28 minutes. A learned-method run evaluates a random distribution and a learned distribution, projecting to about 56 minutes before training, marginal evaluation, oracle computation, or particle-policy evaluation. The corresponding projection is about 7.3 minutes on medium and 50 seconds on easy. This repeated evaluation is the first bottleneck to remove.

Sampling 50,000 ordinary transitions currently sustains approximately 22.6k, 19.8k, and 19.6k transitions/s on easy, medium, and hard respectively. Dense-row sampling is therefore a secondary but important cost at multi-million-transition budgets and inside the Python particle loops.

## Complexity and breakpoints

Let:

- `S = rows * cols` under the current indexing, including walls;
- `A = 4` actions;
- `B <= 3` distinct stochastic branches per GridWorld state-action;
- `H` be the horizon;
- `P` be the number of complete policies evaluated;
- `N` be the number of particles and `K` the number of transition tapes.

| Operation | Current dense cost | Fixed-branch GridWorld cost |
|---|---:|---:|
| Transition/reward storage | `O(A S^2)` | `O(A B S)` |
| Grid construction/validation | `O(A S^2)` plus repeated wall-set creation | `O(A B S)` |
| One sampled transition | `O(S)` and a temporary CDF | `O(B)` |
| One deterministic-policy exact value | `O(H S^2)` | `O(H B S)` |
| Stochastic-policy exact value | `O(H A S^2)` | `O(H A B S)` |
| Finite-horizon oracle | `O(H A S^2)` | `O(H A B S)` |
| `P` committed-policy values | `O(P H S^2)` | `O(P H B S)` before deduplication/batching |
| Particle simulation | `O(N K H S)` sampling work | `O(N K H B)` sampling work |
| Policy particles | `O(N S)` | initially unchanged; later `O(N D)` |

With two float64 dense tensors and four actions, storage is `64 S^2` bytes:

| Grid | `S` | Current steady tensors | Approximate construction peak | Fixed branches (`int32`, two float64 arrays, `B=3`) |
|---|---:|---:|---:|---:|
| 8x8 | 64 | 0.25 MiB | 0.5 MiB | 0.015 MiB |
| 16x16 | 256 | 4 MiB | 8 MiB | 0.059 MiB |
| 32x32 | 1,024 | 64 MiB | 128 MiB | 0.234 MiB |
| 50x50 | 2,500 | 381 MiB | 763 MiB | 0.572 MiB |
| 64x64 | 4,096 | 1.0 GiB | 2.0 GiB | 0.938 MiB |
| 100x100 | 10,000 | 5.96 GiB | 11.9 GiB | 2.29 MiB |

These estimates exclude allocator overhead and oracle/evaluation workspaces. The hard 32x32 map is not an immediate memory failure, but its quadratic time makes a full array costly. Grids around 45x45 already cross 256 MiB steady dense storage, and 64x64 is unsuitable on ordinary CPU job allocations.

Exact enumeration is a different limit: it requires `A^D` policies for `D` decision states. With four actions, `D=10` already means 1,048,576 policies and `D=11` means 4,194,304. Easy, medium, and hard currently have 57, 204, and 716 decision states, so enumeration must never be part of their run path.

## Minimal refactor

### 1. Add a transition-model API

Stop letting algorithms index `.transition[s, a]` and `.reward[s, a, s_next]` directly. Add four operations to the MDP abstraction:

```text
sample_initial(u) -> state
sample_transition(state, action, u) -> (next_state, reward)
action_backups(next_value[S]) -> q[S, A]
expected_return(policy[S]) -> scalar
```

The dense implementation can provide these operations unchanged for tiny generic MDPs. `simulation.py`, `replicated_smc.py`, and `oracles.py` should call the API, which removes their dependence on a particular storage layout.

Precompute the initial CDF once. GridWorld has a deterministic initial state and can return it in `O(1)`.

### 2. Add a fixed-branch GridWorld model

Represent local dynamics with:

```text
next_state[S, A, B]       int32
probability[S, A, B]      float64
reward[S, A, B]           float64
branch_count[S, A]        uint8, or zero-padded probabilities
```

Build rows directly from the intended action and two slip actions. Merge duplicate next states caused by walls/boundaries. To preserve current common-random-number semantics exactly, sort merged branches by ascending `next_state`: the dense sampler's CDF is currently ordered by global state index. Terminal rows contain one self-transition with probability one and reward zero.

Construct `walls`, `goals`, and `hazards` once. The current `_move` recreates `set(spec.walls)` on every call; pass the precomputed wall set or close over it.

Do not compress away wall states in the first refactor. Keeping the existing `cell <-> state` mapping lowers correctness risk. Traversable-state compression is optional after parity and performance gates pass.

### 3. Express dynamic programming as branch gathers

For a supplied next-value vector, compute all state-action backups as:

```text
q[s, a] = sum_b probability[s, a, b]
                 * (reward[s, a, b] + gamma * next_value[next_state[s, a, b]])
```

This vectorizes over `[S, A, B]` and supports:

- oracle recurrence: `value = max_a q`;
- stationary stochastic evaluation: `value = sum_a policy_probability * q`;
- committed deterministic evaluation: `value = q[all_states, policy]`.

Iterate backward for `H` steps and take the initial expectation. This uses `O(S)` rolling value memory for evaluation. The oracle may retain its `H x S` policy table only when a caller requests the full nonstationary policy; ordinary experiment runs need just the initial oracle value.

### 4. Remove redundant experiment-level evaluation

Create a map-control artifact keyed by map hash, horizon, gamma/reward semantics, evaluator version, and fixed evaluation seed. Compute random controls and the oracle once per map, not once per method and training seed.

For learned complete-policy evaluation:

1. evaluate in chunks of 128 policies;
2. deduplicate identical policies and weight their exact values by multiplicity, especially after particle resampling;
3. maintain an online mean/variance;
4. after a minimum of 256 sampled policies, stop when the 95% half-width is below both a declared absolute tolerance and 1% of the random-to-oracle range;
5. retain 2,000 as a cap, not an unconditional count.

The exact value of each retained complete policy should remain the primary value when the scientific claim requires action commitment. Monte Carlo trajectories can be a cheap smoke-test evaluator, but silently replacing complete-policy exact values with statewise stochastic values would change the estimand.

The current particle path computes proposal committed-policy statistics before particle refinement and then computes particle-policy values. Retain both only if both are reported under distinct names; otherwise skip the unused proposal evaluation. Ensure the final `committed_policy_value` and its standard deviation refer to the same final policy population.

### 5. Tighten exact-enumeration guards

Default exact enumeration to at most nine four-action decision states (`4^9 = 262,144`) and require both:

- an estimated peak below 256 MiB; and
- an explicit diagnostic-mode opt-in above 100,000 policies.

Generate policies in chunks rather than building a Python list of all per-policy arrays before `np.stack`. Enumeration stays a tiny-domain validation tool; it is not an evaluator for the experimental GridWorld tiers.

### 6. Defer second-order optimizations

After the sparse transition/evaluation gates pass, profile again before changing particle ancestry or replay storage. Likely next targets are:

- store particle actions only for decision states (`[N, D]`, not `[N, S]`);
- avoid copying the full policy matrix at every resampling event through ancestry/copy-on-write storage;
- validate a complete policy once per batch rather than once per rollout;
- cache `state_to_decision` rather than rebuilding the dictionary in repeated objective calls;
- replace Python tuple replay storage only if it becomes the measured memory leader.

None of these should delay the fixed-branch refactor; they do not remove the current quadratic bottleneck.

## Hard limits and dispatch rules

Add fail-fast estimates before allocating arrays or starting a run:

- Generated GridWorld always uses fixed-branch storage, regardless of size.
- Dense generic storage requires an explicit override if the two tensors exceed 128 MiB (`S > 1,448` for four actions), and must refuse if the projected construction peak exceeds 50% of the job memory limit.
- Exact policy enumeration is disabled by default above 100,000 policies or 256 MiB estimated peak, and absolutely disabled in non-diagnostic experiment manifests.
- Dense committed-policy evaluation refuses a request when `P * H * S^2 > 1e9` scalar branch contributions; the caller must use the compact model or lower/adapt `P`.
- Any evaluator must print its estimated work, chosen policy-sample cap, and cache key before running.
- An experiment aborts before training if the map-control artifact cannot be computed within 10% of the requested job wall time.

These are conservative safety rails, not claims that work just below a threshold is efficient.

## Fail-fast benchmark ladder

Add a standalone benchmark command or test marker; do not run performance gates in the ordinary unit-test suite.

### Gate A: semantic parity on tiny grids

For at least 20 generated 2x2 to 5x5 maps, compare dense and fixed-branch models:

- every `(state, action)` successor distribution and transition-conditioned reward;
- terminal absorption and zero post-terminal reward;
- sampled next state for boundary `u` values and 1,000 seeded uniform values per row;
- 100 deterministic-policy exact values;
- random stochastic value and oracle initial value;
- complete trajectories under identical tapes.

Pass thresholds: exact successor/reward agreement, identical seeded next states, and value error `<= 1e-10` in float64.

### Gate B: representation and construction

Measure median of at least five warm runs for existing easy/medium/hard maps plus an allocation-only 64x64 map.

Pass thresholds:

- no `[S, A, S]` array exists in a GridWorld object;
- hard transition-model storage is below 1 MiB;
- 64x64 transition-model storage is below 2 MiB;
- hard construction is at least 4x faster than the 0.445 s dense reference;
- 64x64 construction completes in under 2 seconds and below 64 MiB peak process growth.

### Gate C: sampling and dynamic programming

Benchmark 100,000 seeded transitions, one committed-policy value, random stochastic value, and the oracle.

Pass thresholds on the current workstation:

- sampling throughput is at least 20k transitions/s and no slower than 90% of the dense reference;
- hard committed-policy evaluation is below 0.10 s (at least 8x faster than 0.837 s);
- hard random stochastic evaluation is below 0.50 s (at least 10x faster than 5.03 s);
- hard oracle value is below 0.65 s (at least 10x faster than 6.67 s);
- all values match the dense reference within `1e-10`.

If hardware changes, retain the relative speedup gates and record new absolute references.

### Gate D: evaluation scaling

On hard, evaluate 16, 64, 256, and 1,024 complete policies. Record unique-policy count, time, peak memory, and confidence-interval half-width.

Pass thresholds:

- time is approximately linear in the number of unique policies;
- 256 policies complete in under 30 seconds;
- no evaluation batch exceeds 256 MiB working memory;
- adaptive evaluation normally stops before the 2,000-policy cap on a stable distribution;
- cached random/oracle controls take under one second to load and validate.

### Gate E: end-to-end canaries

Run one 5k-transition job per method on easy, then one 100k-transition job for the proposed method and one baseline on medium and hard.

Pass thresholds:

- results match pre-refactor values within declared stochastic uncertainty;
- evaluation consumes less than 25% of end-to-end hard-job time;
- actual peak memory is below 50% of requested Slurm memory;
- output records transition-model kind, storage bytes, control-cache key, unique evaluated policies, evaluation time, training time, and peak RSS;
- no fallback materializes dense transitions.

Only after Gate E should medium/hard job arrays be released. A single 64x64 canary should then verify that the quadratic allocation path is impossible, even though 64x64 is not required for the main study.

## Recommended implementation order

1. Add parity tests and benchmark capture around the current dense code.
2. Add the transition-model API to the dense MDP without changing results.
3. Implement the fixed-branch model and switch GridWorld construction to it.
4. Rewrite the oracle and both exact policy evaluators with branch backups.
5. Update simulation and particle sampling to call `sample_transition`.
6. Add control caching, policy deduplication, and adaptive evaluation.
7. Run Gates A–E and archive their JSON results.
8. Profile again; only then consider particle-policy copy-on-write or traversable-state compression.

This order keeps each change testable, preserves a dense reference oracle for tiny maps, and attacks the measured evaluation cost before more invasive particle optimizations.
