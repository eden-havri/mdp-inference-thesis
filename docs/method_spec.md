# Method Specification: Expected-Return Gibbs Policy Inference

Status: normative design note. No large-scale experiment is authorized until every
required gate in this document passes.

## 1. Scope and semantic contract

This document specifies inference over deterministic Markov policies for a finite-
horizon MDP. Its central contract is that environment randomness is averaged **before**
the policy score is exponentiated.

Let

- \(S\) be a finite or countable state space;
- \(A\) be a finite action space;
- \(H < \infty\) be the rollout horizon;
- \(\rho_0\) be the initial-state distribution;
- \(T(\cdot\mid s,a)\) be the transition kernel;
- \(r(s,a,s')\) be the reward;
- \(\Pi\) be the deterministic stationary policy class; and
- \(\mu\) be a proper reference probability measure on \(\Pi\), uniform in the
  finite tabular case unless explicitly stated otherwise.

A simulator tape \(\xi\) contains all exogenous randomness needed to run the MDP.
For a fixed policy and tape, define

\[
G(\pi,\xi)=\sum_{t=0}^{H-1} r(s_t,\pi(s_t),s_{t+1}),
\qquad
J(\pi)=\mathbb E_{\xi}[G(\pi,\xi)].
\]

The required Gibbs posterior at inverse preference temperature \(\beta>0\) is

\[
p_\beta(d\pi)
=
\frac{\exp\{\beta J(\pi)\}\,\mu(d\pi)}
     {Z_\beta},
\qquad
Z_\beta=\int_\Pi \exp\{\beta J(\pi)\}\,\mu(d\pi).
\tag{1}
\]

For finite \(\Pi\), this is simply a normalized finite sum. Its modes are exactly
the maximizers of expected return, up to ties and any non-uniform mass in \(\mu\).
Positive reward scaling changes posterior concentration but not the ordering of
policies. A constant added to every policy's *total* return cancels from the
posterior.

The induced action distribution is

\[
p_\beta(a\mid s)
=\Pr_{\pi\sim p_\beta}[\pi(s)=a].
\tag{2}
\]

Equation (2) is a posterior marginal. The execution protocol must be named
explicitly because the following are different objects:

1. sample one deterministic policy per episode and reuse its action on revisits;
2. resample from the marginal \(p_\beta(\cdot\mid s)\) on every visit;
3. execute the marginal-mode action; or
4. execute a MAP deterministic policy.

All reported results must identify which protocol was used.

### 1.1 Forbidden target substitution

The following target is not equivalent to (1):

\[
\bar p_\beta(d\pi)
\propto
\mathbb E_\xi\!\left[\exp\{\beta G(\pi,\xi)\}\right]\mu(d\pi).
\tag{3}
\]

In general,

\[
\mathbb E[\exp\{\beta G\}]
\ne
\exp\{\beta\mathbb E[G]\}.
\]

Equation (3) is an entropic/risk-sensitive criterion and may rank policies
differently from expected return. A single realized return is an unbiased estimate
of \(J(\pi)\), but exponentiating that realization does **not** give an unbiased
estimate of \(\exp\{\beta J(\pi)\}\). Common random numbers can reduce variance in
policy comparisons; they do not repair this order-of-operations error.

Any implementation whose particle weight contains `exp(beta * realized_return)`
without first constructing a controlled approximation to \(J(\pi)\) must be
treated as implementing (3), not (1).

## 2. Assumptions

The core results and tests use the following assumptions.

### A1. Finite horizon and bounded return

There are constants \(g_-\) and \(g_+\) such that

\[
g_-\le G(\pi,\xi)\le g_+
\]

almost surely for every \(\pi\). Finite horizon and bounded one-step rewards are
sufficient. This guarantees \(0<Z_\beta<\infty\) for a proper \(\mu\).

### A2. Correct simulator tapes

Independent tape identifiers produce independent simulator randomness. Tape
\(k\) contains one initial-state uniform and one transition uniform for each
time index. Given the current \((s,a)\), inverse-CDF sampling maps that fixed
uniform to a successor from \(T(\cdot\mid s,a)\). Re-evaluating a policy on the
same tape is deterministic, while different tape identifiers are independent.
The precise coupling is part of the configuration and must not change within a
reported study.

### A3. Policy consistency

A deterministic policy assigns an action once per state. The same assignment is
used on every revisit and across every tape used to evaluate that policy.

### A4. Proposal support

For every policy with positive reference mass and finite target density,

\[
q_\theta(\pi)>0.
\]

Complementary Candidate A enforces this directly in \(q_\theta\). Complementary
Candidate B enforces it with the explicit uniform mixture in Section 6.2, even if
its learned guide assigns zero probability to an action. The primary sampler uses
the analogous mixture in (RE-5) and the exact conditional proposal in (RE-6).
All actual proposal probabilities must be computed stably and must cover the
reference support.

### A5. RNG separation

Policy sampling, transition tapes, resampling, training diagnostics, and final
evaluation use separately seeded random streams. Changing evaluation frequency or
sample count must not change a training trajectory.

### A6. Declared variational family

The factorization and context of \(q_\theta\) are part of the method definition.
A state-only mean-field family cannot generally represent posterior correlations
between action assignments at different states. Validation therefore distinguishes
target error from variational-family error.

## 3. Exact target oracle for finite policy spaces

Every algorithm is first validated against an exact finite-policy oracle.

For each \(\pi\in\Pi\):

1. compute \(J(\pi)\) exactly by finite-horizon dynamic programming;
2. compute \(u_\pi=\log\mu(\pi)+\beta J(\pi)\);
3. normalize with log-sum-exp; and
4. derive \(Z_\beta\), posterior probabilities, action marginals, posterior mean
   return, and the return of each declared execution protocol.

The dynamic program for a fixed deterministic policy is

\[
V_H^\pi(s)=0,
\qquad
V_t^\pi(s)
=
\sum_{s'}T(s'\mid s,\pi(s))
\left[r(s,\pi(s),s')+V_{t+1}^\pi(s')\right],
\tag{4}
\]

with

\[
J(\pi)=\mathbb E_{s_0\sim\rho_0}[V_0^\pi(s_0)].
\]

This oracle is the source of truth. Monte Carlo agreement with another Monte Carlo
implementation is not sufficient validation.

## 4. Primary sampler: replica-exchange policy Metropolis--Hastings

The primary corrected sampler operates on **complete deterministic policies**. It
uses replica exchange (parallel tempering) to preserve the required Gibbs target
while allowing hot replicas to cross return barriers. The learned guide affects
proposal efficiency only; every asymmetric proposal is corrected by the
Metropolis--Hastings ratio.

### 4.1 Exact-model and fixed-tape targets

Write \(\widetilde J\) for the score used throughout one Markov-chain run. There
are exactly two supported semantics:

\[
\widetilde J(\pi)=
\begin{cases}
J(\pi), & \text{exact-model mode, computed by (4)},\\[2mm]
\widehat J_K(\pi;\Xi_K), & \text{fixed-tape simulator mode.}
\end{cases}
\tag{RE-1}
\]

In exact-model mode, the cold replica targets (1) exactly. This variant is
model-based because it evaluates policies by dynamic programming. In fixed-tape
mode, one immutable tape bank \(\Xi_K\) is created before the chain starts and the
cold replica targets the conditional SAA posterior

\[
p_{\beta,K}(d\pi\mid\Xi_K)
\propto
\exp\{\beta\widehat J_K(\pi;\Xi_K)\}\mu(d\pi).
\tag{RE-2}
\]

The same tapes must be reused for every proposed and current policy evaluation in
that run. Redrawing tapes inside an acceptance ratio produces a noisy-MH kernel and
is outside this specification. Finite \(K\) makes (RE-2) a random approximation to
(1), not an exact or unbiased pseudo-marginal implementation of it. Independent
tape banks are therefore independent SAA problem instances and are not additional
samples from one Markov chain.

### 4.2 Temperature ladder and joint invariant distribution

Use \(L\ge2\) inverse temperatures

\[
0=\beta_0<\beta_1<\cdots<\beta_{L-1}=\beta.
\]

The implemented power ladder is

\[
\beta_\ell
=
\beta\left(\frac{\ell}{L-1}\right)^p,
\qquad p>0.
\tag{RE-3}
\]

Replica \(\ell\) targets

\[
p_{\beta_\ell}^{\widetilde J}(d\pi)
\propto
\exp\{\beta_\ell\widetilde J(\pi)\}\mu(d\pi),
\tag{RE-4}
\]

and the joint target is the product of (RE-4) over replicas. Thus the hottest
replica samples the reference measure and the coldest samples the declared target.
The number and placement of temperatures are efficiency choices, not target
changes. They must be selected on pilot instances and frozen before confirmatory
runs.

### 4.3 Guided single-site proposal and local acceptance

Let \(D\) be the number of decision states. At a local update, select one decision
state \(s\) uniformly. Let \(g_\phi(a\mid s)\) be a frozen stochastic guide and
define the full-support row

\[
r_s(a)
=
\lambda g_\phi(a\mid s)
+(1-\lambda)u_s(a),
\qquad 0\le\lambda<1,
\tag{RE-5}
\]

where \(u_s\) is uniform over the prior-supported legal actions at \(s\). Given
the current action \(a=\pi(s)\), propose a different action \(a'\) from

\[
q_s(a'\mid a)
=
\frac{r_s(a')}{1-r_s(a)},
\qquad a'\ne a.
\tag{RE-6}
\]

Every state included in the decision set must have at least two prior-supported
legal actions; a one-action state is fixed and excluded from local proposals.

For policies \(\pi\) and \(\pi'\) that differ only at \(s\), the complete proposal
probability is \(q(\pi'\mid\pi)=D^{-1}q_s(a'\mid a)\). Replica \(\ell\) accepts the
proposal with probability

\[
\alpha_\ell(\pi,\pi')
=
\min\!\left\{
1,
\exp\!\left[\beta_\ell
  \{\widetilde J(\pi')-\widetilde J(\pi)\}\right]
\frac{\mu(\pi')}{\mu(\pi)}
\frac{q(\pi\mid\pi')}{q(\pi'\mid\pi)}
\right\}.
\tag{RE-7}
\]

For the forced-different proposal in (RE-6), its nontrivial Hastings factor is

\[
\frac{q(\pi\mid\pi')}{q(\pi'\mid\pi)}
=
\frac{r_s(a)/(1-r_s(a'))}
     {r_s(a')/(1-r_s(a))}.
\tag{RE-8}
\]

The reference-measure ratio cancels only for the implemented uniform tabular
reference. The exact probability in (RE-6), including its renormalization after
excluding the current action, must be used in (RE-8). Omitting (RE-8), using the
unmixed guide, or changing the guide during a run changes the transition kernel
and invalidates the stated guarantee. With full support, repeated single-site
moves connect the finite Cartesian policy space. If aperiodicity is not otherwise
proved for a configured domain, an explicit lazy stay-put step must be added before
claiming convergence from arbitrary initialization.

### 4.4 Adjacent replica swaps

At a declared interval, propose exchanges between disjoint adjacent replica pairs,
alternating even and odd edges. For adjacent states \(\pi_\ell\) and
\(\pi_{\ell+1}\), accept their exchange with probability

\[
\alpha_{\mathrm{swap}}
=
\min\!\left\{
1,
\exp\!\left[
(\beta_{\ell+1}-\beta_\ell)
\{\widetilde J(\pi_\ell)-\widetilde J(\pi_{\ell+1})\}
\right]
\right\}.
\tag{RE-9}
\]

Equation (RE-9) assumes every replica uses the same \(\mu\), score function, and
fixed tape bank. Local moves satisfying (RE-7) and swaps satisfying (RE-9) each
leave the joint product target invariant; samples retained from replica \(L-1\)
after burn-in therefore have the cold target as their invariant marginal.

### 4.5 Initialization, burn-in, and retained samples

All replicas may be initialized at a guide-derived policy, such as the guide MAP.
Alternatively, hot replicas may be initialized from \(\mu\) as an overdispersed
diagnostic. Initialization changes neither (RE-7), (RE-9), nor the invariant
distribution, but it can materially affect finite-run bias and time to stationarity.
The \(\beta_0=0\) replica must be allowed to forget a warm start during burn-in;
warm initialization is not evidence of convergence.

Burn-in, thinning, and the retained cold-chain indices are fixed in the manifest.
Thinning may reduce stored autocorrelation but does not reduce computation and must
not be used to hide poor mixing. Independent chains use independent MCMC seeds. In
fixed-tape studies, whether chains share a tape bank or use independent banks is a
separate, explicit experimental factor.

### 4.6 Diagnostics and fail-fast sampler gates

Return ESS alone is insufficient because distinct policies can have identical or
nearly identical returns. Every pilot records at least:

- local proposal and acceptance counts at every temperature;
- swap proposal and acceptance counts on every ladder edge;
- walker temperature-visit fractions, endpoint transitions, completed round trips,
  and the number of distinct walkers reaching the cold replica;
- cold-chain return ESS and action-indicator ESS for every decision-state/action;
- per-state action-switch rates, distinct-policy count, and mean Hamming jump; and
- agreement across independently seeded warm and overdispersed chains, including a
  predeclared multichain diagnostic where estimable.

Before a scale run, exact tiny domains must verify detailed balance or the full
transition matrix numerically, cold posterior probabilities, action marginals, and
expectations against enumeration. A ladder pilot is `NO-GO` if a required edge has
no effective exchange, walkers do not traverse the ladder, complete-policy
occupancy remains untested, or independent initializations disagree beyond their
predeclared Monte Carlo uncertainty. Acceptance rate by itself is not a pass
criterion. Diagnostic thresholds, run length, and allowed ladder retuning rounds
must be set before confirmatory outcomes are viewed.

### 4.7 Cost accounting

With \(I\) iterations, the current implementation performs
\(L(I+1)\) policy-value evaluations: \(L\) initial evaluations and one local
proposal evaluation per replica per iteration. Swaps reuse cached values and incur
no additional policy evaluation. In exact-model mode, each evaluation includes one
finite-horizon dynamic program. In fixed-tape mode, each evaluation uses \(K\)
rollouts and at most \(KH\) simulator transitions; the exact observed transition
count is authoritative when episodes can terminate early.

Every comparison records value evaluations, simulator transitions, guide-training
transitions, optimizer updates, wall-clock time, CPU/GPU time, peak memory, and
failed pilot/tuning cost. The full cost of training the guide is charged to a guided
run unless a separately labeled amortization protocol was predeclared. Increasing
\(L\), \(I\), or \(K\) buys different forms of accuracy and cannot be reported under
one undifferentiated episode budget.

### 4.8 Limitations

The present method samples finite, complete, stationary tabular policies. A
time-dependent finite-horizon optimum can lie outside that class. Exact-model mode
requires transition and reward models; fixed-tape mode avoids that requirement but
targets a finite-\(K\) SAA posterior. High \(\beta\), sparse rewards, strong policy
correlations, and distant modes can still defeat single-site local moves even with
replica exchange. More temperatures improve communication only at additional cost,
and a learned guide can make proposals efficient without proving mixing. Finite
diagnostics cannot prove global exploration, and posterior marginal execution is
not equivalent to committing to a sampled deterministic policy. Claims are
therefore limited to the declared policy class, target semantics, execution
protocol, and diagnostics actually passed.

## 5. Complementary candidate A: direct expected-return ELBO

This is the semantic baseline. It is deliberately simple and must work before a
particle method is trusted.

For a proposal \(q_\theta\), define

\[
\mathcal L(\theta)
=
\mathbb E_{\pi\sim q_\theta}
\left[
  \beta J(\pi)+\log\mu(\pi)-\log q_\theta(\pi)
\right].
\tag{5}
\]

Then

\[
\mathcal L(\theta)
=\log Z_\beta-\operatorname{KL}(q_\theta\Vert p_\beta).
\tag{6}
\]

For each sampled policy, draw \(K\) independent tapes and compute

\[
\widehat J_K(\pi)
=\frac1K\sum_{k=1}^{K}G(\pi,\xi_k).
\tag{7}
\]

The Monte Carlo ELBO sample

\[
\widehat{\mathcal L}
=
\beta\widehat J_K(\pi)+\log\mu(\pi)-\log q_\theta(\pi)
\tag{8}
\]

is unbiased for (5). Increasing \(K\) reduces return-estimation variance without
changing the expected objective.

For discrete policy samples, a valid score-function gradient is

\[
\widehat g
=
\left(
\beta\widehat J_K(\pi)+\log\mu(\pi)-\log q_\theta(\pi)-b
\right)^{\mathrm{stop}}
\nabla_\theta\log q_\theta(\pi),
\tag{9}
\]

where \(b\) is independent of the current policy sample, or is a conditionally
valid leave-one-out baseline. The omitted constant `-1` has zero expectation
because \(\mathbb E_q[\nabla\log q]=0\).

The entire parenthesized learning signal in (9) is stop-gradient. Allowing gradients
through that coefficient adds product-rule terms and changes the estimator.

### 5.1 Full-policy versus lazy sampling

For exact tabular tests, sample every action assignment in the finite policy.

For a scalable lazy policy object, an action is sampled when a state is first queried
and then memoized. The same memoized policy is shared by all \(K\) tapes evaluating
that policy. Uninstantiated assignments may be integrated out only when a theorem
shows that their prior and proposal factors cancel or integrate to one. Until that
equivalence is proved and tested, lazy direct-ELBO training is experimental and the
full-policy tabular implementation remains the correctness oracle.

### 5.2 Role of Candidate A

Candidate A provides:

- an estimator whose expectation targets (1) for every \(K\ge1\);
- a low-complexity gradient oracle;
- an optimization baseline for small and medium tabular problems; and
- a diagnostic separating target errors from particle/resampling errors.

It may have high variance and may underrepresent multimodality. Those are efficiency
limitations, not permission to change the target.

## 6. Complementary candidate B: K-tape sample-average policy SMC

Candidate B approximates \(J(\pi)\) *inside each particle* before exponentiation.
It is a sample-average approximation (SAA), not an exact finite-\(K\) realization of
(1).

At the start of a sweep, draw \(K\) independent transition tapes

\[
\Xi_K=(\xi_1,\ldots,\xi_K).
\]

The same \(K\) tapes are shared across all policy particles. For a policy particle,
define

\[
\widehat J_K(\pi;\Xi_K)
=\frac1K\sum_{k=1}^{K}G(\pi,\xi_k)
\tag{10}
\]

and the conditional SAA target

\[
p_{\beta,K}(d\pi\mid\Xi_K)
\propto
\exp\{\beta\widehat J_K(\pi;\Xi_K)\}\mu(d\pi).
\tag{11}
\]

### 6.1 Particle state

One policy particle contains:

- \(K\) environment states, one per tape;
- one policy-action memo shared across all \(K\) tapes;
- accumulated average reward; and
- any proposal context required by the declared variational family.

If two tape replicas of the same particle reach the same state, both must use the
same memoized policy action. Distinct policy particles may assign different actions.

### 6.2 Learned full-support guide

Candidate B's scalable proposal is a separately trained stochastic guide
\(q_{\mathrm{PPO},\phi}\). It is used only to propose policy assignments; it is
not the prior, the posterior, or a replacement for the importance weights. Let
\(\mathcal A_\mu(s,m)\) be the prior-supported legal actions at state \(s\),
given the assignments \(m\) already stored in the policy memo, and let

\[
u(a\mid s,m)
=
\frac{\mathbf 1\{a\in\mathcal A_\mu(s,m)\}}
     {|\mathcal A_\mu(s,m)|}.
\]

After applying the same legal-action mask to both components, the SMC proposal is

\[
q_{\epsilon,\phi}(a\mid s,m)
=
(1-\epsilon)q_{\mathrm{PPO},\phi}(a\mid s,m)
+\epsilon u(a\mid s,m),
\qquad 0<\epsilon\le 1.
\]

Consequently,

\[
q_{\epsilon,\phi}(a\mid s,m)
\ge \frac{\epsilon}{|\mathcal A_\mu(s,m)|}
\]

for every prior-supported legal action. This lower bound is the support guarantee;
clipping a zero-support proposal after sampling is not an acceptable substitute.
The mixture probability must be evaluated in stable log space from the exact
\(q_{\mathrm{PPO},\phi}\), \(\epsilon\), and action mask that generated the sample.

The guide is frozen during an SMC sweep. The default protocol trains it before SMC
and freezes it for all reported sweeps. If a declared adaptive variant updates the
guide between sweeps, every sweep records and uses one fixed parameter snapshot;
weights from different proposal snapshots are never combined as though they came
from a single proposal.

The proposal is queried only when the policy memo first assigns an action to a
state. A revisit reuses the memoized action and incurs neither a new proposal draw
nor a new prior/proposal ratio. When several of the \(K\) replicas first expose the
same unassigned state in one simulated step, it generates one proposal event.

### 6.3 Incremental log weight

At simulated time \(t\), the reward increment is

\[
\Delta R_t
=\frac{\beta}{K}\sum_{k=1}^{K}r_{t,k}.
\tag{12}
\]

Let \(U_t\) be the deterministically ordered set of distinct, previously unassigned
states first exposed by the replicas at step \(t\). For each \(s\in U_t\), draw
\(a_s\sim q_{\epsilon,\phi}(\cdot\mid s,m_{<s})\), store it in the shared policy
memo, and update the particle log-weight by

\[
\Delta\log w_t
=
\frac{\beta}{K}\sum_{k=1}^{K}r_{t,k}
+
\sum_{s\in U_t}
\left[
\log\mu(a_s\mid s,m_{<s})
-\log q_{\epsilon,\phi}(a_s\mid s,m_{<s})
\right].
\tag{13}
\]

Equation (13) is the exact prior/proposal importance correction. Omitting its
denominator, substituting \(q_{\mathrm{PPO},\phi}\) for the actual mixture, or
applying the ratio once per replica changes the target. The transition mechanism
needs no additional ratio because, conditional on the fixed tapes, target and
proposal use the same simulator dynamics. Resampling copies the entire particle,
including every environment state, the complete policy memo, and all proposal
context.

### 6.4 What finite K means

For finite \(K\), (11) is random and generally differs from (1). In particular,
integrating over tape sets again produces a risk-sensitive tilt:

\[
\mathbb E_{\Xi_K}
\left[\exp\{\beta\widehat J_K(\pi;\Xi_K)\}\right]
\ne
\exp\{\beta J(\pi)\}.
\]

Candidate B is admissible only as an explicitly labeled, convergence-controlled SAA.
It must never be described as an unbiased pseudo-marginal implementation of (1).

Under A1 and finite \(|\Pi|=P\), Hoeffding's inequality and a union bound give

\[
\Pr\!\left[
\sup_{\pi\in\Pi}|\widehat J_K(\pi)-J(\pi)|>\varepsilon
\right]
\le
2P\exp\!\left(
-\frac{2K\varepsilon^2}{(g_+-g_-)^2}
\right).
\tag{14}
\]

On the event in (14), the exact and SAA posterior density ratio is bounded by

\[
e^{-2\beta\varepsilon}
\le
\frac{p_{\beta,K}(\pi\mid\Xi_K)}{p_\beta(\pi)}
\le
e^{2\beta\varepsilon},
\tag{15}
\]

which yields, for example, the loose total-variation bound

\[
\|p_{\beta,K}-p_\beta\|_{\mathrm{TV}}
\le
\min\{1,e^{2\beta\varepsilon}-1\}.
\tag{16}
\]

Thus Candidate B is consistent as \(K\to\infty\) under the stated assumptions, but
the required \(K\) grows with return range, inverse temperature, and policy-space
complexity.

### 6.5 Conditional SMC objective

For fixed tapes, let \(\widehat Z_{K,N}\) be the normalizing-constant estimator from
an \(N\)-particle sweep. With a frozen guide, proposal quality affects estimator
variance and particle degeneracy but not the conditional target; inference itself
does not require differentiating through SMC. An optional adaptive-proposal variant
may use the variational objective

\[
\mathcal F_{K,N}(\phi)
=
\mathbb E_{\Xi_K,\mathrm{SMC}_{\epsilon,\phi}}
\left[\log\widehat Z_{K,N}\right].
\tag{17}
\]

For fixed \(\Xi_K\), standard SMC unbiasedness of \(\widehat Z_{K,N}\) must be
proved for the exact proposal, propagation, coupling, and resampling scheme actually
used. Correlation from common tapes is allowed only if the proof's conditional-
unbiasedness requirements remain satisfied.

If the optional adaptive variant differentiates through categorical proposal
samples, it requires score-function terms. If \(L_j\) is the
causally valid suffix learning signal for proposal event \(j\), its multiplier must
be stopped:

\[
L_j^{\mathrm{stop}}
\nabla_\phi\log q_{\epsilon,\phi}(a_j\mid c_j).
\tag{18}
\]

The ordinary derivative of log weights through explicit `-log q` terms is separate
from (18). Omitting the stop in (18) adds an invalid product-rule term.

Resampling ancestry is also sampled from a \(\phi\)-dependent categorical
distribution. There are only three acceptable contracts:

1. include and validate the resampling score term;
2. use a construction with a proved unbiased alternative gradient; or
3. explicitly call the update a stopped-resampling surrogate and make no claim that
   it is an unbiased gradient of (17).

For initial validation, no-resampling importance sampling is preferred because it
isolates target and proposal-gradient correctness. Resampling is enabled only after
its separate gradient contract passes.

### 6.6 Anchor invariants

The following identities are mandatory:

- At \(N=1\), with no resampling, Candidate B's expected objective and gradient
  reduce to Candidate A's direct ELBO using the same \(K\) tapes and proposal
  family.
- With deterministic dynamics, all \(K\ge1\) give the same target.
- As \(K\to\infty\), exact finite-policy SAA posteriors converge to (1).
- Increasing \(N\) does not compensate for insufficient \(K\): \(N\) controls
  particle approximation, while \(K\) controls environment-expectation error.

### 6.7 Proposal cost and required ablations

Training \(q_{\mathrm{PPO},\phi}\) is part of the method cost. Every comparison that
uses the learned guide counts its proposal-training environment transitions,
optimizer updates, wall-clock time, accelerator time, and peak memory in addition
to the subsequent SMC cost. Hyperparameter-search and failed-training budgets are
recorded separately and are not silently assigned a zero cost. Amortized accounting
is permitted only in a separately labeled multi-instance study with a predeclared
reuse horizon; the primary single-instance comparison uses total, unamortized cost.

At a minimum, the following configurations use the same environment, evaluation
protocol, and total-budget accounting:

1. uniform-proposal policy SMC, obtained with \(\epsilon=1\);
2. guided policy SMC using the declared \(q_{\epsilon,\phi}\) and the correction in
   (13); and
3. a proposal-alone ablation that samples the same lazy, memoized deterministic
   policy from \(q_{\epsilon,\phi}\), then executes it without importance weighting
   or resampling.

The third configuration isolates what is already supplied by the learned guide from
what is added by corrected SMC. A conventional per-visit execution of
\(q_{\mathrm{PPO},\phi}\) may be reported as an additional diagnostic, but it is not
a substitute for the proposal-alone ablation because it has different policy
commitment semantics. These are protocol requirements, not empirical claims.

## 7. Theorem obligations

The following results require written proofs or precise citations to standard results
whose assumptions are checked line by line.

### T1. Gibbs target correctness

Prove normalization of (1), mode equivalence to expected-return maximization, and
the stated reward-scaling properties under A1 and the chosen \(\mu\).

### T2. Replica-exchange invariance and convergence conditions

For both choices in (RE-1), prove detailed balance of (RE-7), detailed balance of
(RE-9) with respect to the product target, and invariance of the cold marginal.
State the precise irreducibility and aperiodicity conditions needed for convergence.
Show that warm and overdispersed initialization alter only the initial law, not the
invariant law. For fixed tapes, the claim is conditional correctness for (RE-2),
not exactness for (1).

### T3. Direct-ELBO estimator and gradient

Prove that (8) is unbiased for (5) and that (9) is an unbiased gradient estimator,
including the exact admissibility conditions on baselines and lazy policy sampling.

### T4. K-tape SAA consistency

Prove almost-sure or in-probability convergence of \(p_{\beta,K}\) to \(p_\beta\).
For finite tabular spaces, (14)-(16) are sufficient. For larger policy classes, state
and prove the needed uniform law of large numbers or restrict claims to empirical
convergence.

### T5. Conditional normalizer correctness

For fixed tapes, prove that the implemented \(\widehat Z_{K,N}\) is unbiased for
the SAA normalizer. The proof must cover adaptive resampling, shared tape coupling,
lazy action instantiation, terminal states, and copied particle memory.

### T6. Gradient contract

State exactly which scalar objective the implemented gradient estimates. Prove the
categorical proposal score terms and temporal credit assignment. Either account for
resampling scores or explicitly delimit the stopped-resampling approximation. A
custom-autograd implementation must be tested against exhaustive expectation or
finite differences on tiny problems.

### T7. Lazy-policy equivalence

Prove that sampling an action only on first query and integrating unqueried action
assignments produces the same conditional target as sampling a complete policy in
advance. For K-tape evaluation, the policy memo must be shared across all tapes.

## 8. Exact validation suite

All tests run in float64 unless a test explicitly targets lower precision.

### 8.1 Oracle domains

#### O1. Deterministic one-step bandit

Use at least three actions with distinct rewards. Verify exact softmax posterior,
normalizer, action marginals, ELBO, and gradients.

#### O2. Stochastic one-step bandit

Use Bernoulli rewards with actions having distinct means and variances. Compute both

\[
\operatorname{softmax}(\beta\mathbb E[R_a])
\]

and

\[
\frac{\mathbb E[e^{\beta R_a}]}
     {\sum_b\mathbb E[e^{\beta R_b}]}.
\]

The implementation must match the first and demonstrably reject the second. This is
the primary order-of-operations tripwire.

#### O3. Stochastic revisit domain

Include a state that can transition back to itself. Enumerate deterministic policies
and verify that a policy's action is identical on every revisit and across all K
tapes. Separately evaluate per-visit marginal resampling so execution semantics cannot
be conflated with posterior semantics.

#### O4. Two-route stochastic grid

Use a tiny grid with one risky short route and one safer longer route. Enumerate all
deterministic policies and compute exact expected-return and entropic-risk rankings.
Choose parameters for which the rankings differ.

#### O5. Four-decision-state grid

Use four nonterminal decision states, four actions, and horizon 20. Enumerate all
\(4^4=256\) policies. This is large enough to exercise memoization, stochastic
transitions, absorbing states, and multimodality while remaining exact.

### 8.2 Deterministic unit tests

1. Transition probabilities sum to one and match the declared kernel.
2. Absorbing states remain fixed and receive the declared post-terminal reward.
3. A fixed tape and time index return the same successor for the same state and action.
4. Different tape identifiers are statistically independent.
5. Policy actions are memoized once per state and copied through resampling.
6. Each prior/proposal correction is applied once per policy-state assignment.
7. All K replicas of one policy use the same action assignment.
8. Resampling copies all K states, policy memory, and accumulated weights from
   the selected ancestor.
9. The guided mixture is normalized and obeys
   \(q_{\epsilon,\phi}(a\mid s,m)\ge
   \epsilon/|\mathcal A_\mu(s,m)|\) for every supported action.
10. Every importance denominator equals the actual mixture probability used for
    its proposal draw, including the declared action mask and \(\epsilon\).
11. Log-mean-exp and ESS calculations are stable for extreme finite logits.
12. NaN, infinite reward, zero particles, zero tapes, invalid probabilities, and
    unsupported actions fail before training starts.
13. Every replica-exchange proposal row has full support on legal alternative
    actions, and its recorded forward and reverse probabilities equal (RE-6).
14. Swap scheduling proposes only disjoint adjacent pairs and alternates parity
    without changing the random stream used by local moves.
15. Diagnostic collection is observational: enabling it cannot alter policies,
    accept/reject decisions, or retained samples under the same seed.

### 8.3 Replica-exchange sampler tests

On enumerable policy spaces, construct the local and swap transition kernels
directly. Verify that rows sum to one, (RE-7) and (RE-9) satisfy detailed balance,
and the product distribution in (RE-4) is invariant to float64 tolerance. Test
uniform and nonuniform guide rows, multiple guide strengths, nonuniform reference
mass in the mathematical oracle, and both exact-model and fixed-tape scores. The
implemented uniform-reference restriction must reject unsupported nonuniform
configurations rather than silently dropping \(\mu(\pi')/\mu(\pi)\).

For every oracle domain, compare retained cold samples with exact target
probabilities, action marginals, and bounded expectations using predeclared
simultaneous Monte Carlo uncertainty. Repeat from identical warm starts and
overdispersed starts. Verify that \(\beta_0=0\) has reference-measure marginals,
that changing initialization does not change the transition kernel, and that
cached-value and full-reevaluation implementations produce identical decisions
under the same random variates. Confirm the exact count \(L(I+1)\) of policy-value
evaluations and the observed simulator-transition count.

### 8.4 Objective-value tests

For O1-O5, compare:

- exact \(J(\pi)\);
- exact \(\log Z_\beta\);
- exact posterior probabilities;
- exact action marginals;
- exact posterior mean return;
- exact declared execution-policy return; and
- the best value attainable inside the chosen variational family.

For deterministic quantities, the default tolerance is `1e-10` absolute in float64.

### 8.5 Gradient tests

For small logits and every oracle domain:

1. enumerate all policy samples and tape outcomes where feasible;
2. compute the exact expected objective as a function of logits;
3. compare analytic/autodiff gradients with central finite differences;
4. compare the Monte Carlo gradient mean and confidence interval with the exact
   gradient; and
5. run a mutation test that removes the stop-gradient and confirm that the test
   fails.

Default deterministic acceptance criteria:

- maximum absolute gradient error \(\le10^{-7}\);
- relative \(\ell_2\) error \(\le10^{-6}\); and
- cosine similarity \(\ge 1-10^{-8}\).

Monte Carlo gradient tests must use a predeclared sample count and pass a simultaneous
99% confidence-region check. Merely obtaining the same gradient sign is insufficient.

### 8.6 Normalizer tests

For fixed small tape sets, enumerate the SAA normalizer exactly. Across repeated SMC
sweeps, verify

\[
\mathbb E[\widehat Z_{K,N}\mid\Xi_K]=Z_K(\Xi_K).
\]

The exact value must lie inside a predeclared 99% confidence interval, and the
estimated relative bias must be no larger than both three standard errors and 1%.
Run this test separately with no resampling, fixed-schedule resampling, and adaptive
resampling, and for both uniform and learned full-support proposals.

### 8.7 K-convergence tests

On O2-O5, test \(K\in\{1,2,4,8,16,32,64,128,\ldots\}\). For each K and many tape
sets, compute the exact SAA posterior and measure against (1):

- total-variation distance;
- maximum action-marginal error;
- normalizer error;
- posterior expected-return error; and
- probability of selecting the correct expected-return mode.

Do not require monotone sample-path improvement in K; require decreasing expected
error with uncertainty bands and agreement with the finite-K concentration analysis.
The selected production K is part of the experiment configuration and cannot be
changed after viewing benchmark outcomes.

### 8.8 End-to-end optimization tests

For each oracle domain and at least 20 independent training seeds:

- compare the primary sampler's cold distribution and bounded expectations with
  exact enumeration, including complete-policy mixing diagnostics;
- compare Candidate A with exact best-in-family optimization;
- compare Candidate B at the preselected K values;
- compare uniform-proposal SMC, guided SMC, and the proposal-alone condition under
  total-budget accounting;
- report convergence failures, not only successful seeds;
- verify that increasing evaluation sample count does not alter training; and
- verify checkpoint/resume equivalence.

No aggregate may silently discard failed, NaN, timed-out, or low-performing seeds.

## 9. Fail-fast go/no-go gates

Every gate is blocking. `NO-GO` means no cluster-scale run and no empirical claim
based on the affected method.

### Gate G0: target lock

**GO only if:** equation (1), \(\beta\), \(\mu\), horizon, terminal convention,
reward scale, and execution protocol are frozen in a versioned experiment manifest.

**NO-GO if:** the method is described only as "Bayesian RL" or "policy inference"
without an executable mathematical target.

### Gate G1: exact oracle

**GO only if:** O1-O5 dynamic programs and enumerations pass independent probability,
normalization, and return checks.

**NO-GO if:** any oracle depends on the implementation being validated.

### Gate G2: risk-target tripwire

**GO only if:** on O2 and O4, the exact target, the primary exact-model sampler,
and Candidate A match `exp(E[return])` and are statistically incompatible with
`E[exp(return)]`.

**NO-GO if:** the two targets were not deliberately separated by the test parameters,
or the implementation matches the entropic-risk target.

### Gate G3: direct-ELBO gradient

**GO only if:** exact value and gradient tolerances in Sections 8.4-8.5 pass, including
the stop-gradient mutation test.

**NO-GO if:** only loss values, learning curves, or final returns were checked.

### Gate G4: replica-exchange correctness and mixing

**GO only if:** Section 8.3 and T2 pass; exact-mode cold samples agree with (1);
fixed-tape cold samples agree with (RE-2); and the pilot meets its predeclared
complete-policy mixing thresholds. Every ladder edge must exchange configurations,
walkers must reach both endpoints and complete the required number of round trips,
action-indicator ESS must meet its minimum, and independently seeded warm and
overdispersed chains must agree within simultaneous uncertainty bounds.

**NO-GO if:** only return ESS or local acceptance is satisfactory, any ladder edge
isolates the cold replica, tapes are refreshed inside acceptance decisions, the
guide is changed without a valid adaptive-MCMC contract, or more iterations are
launched before the failed diagnostic is explained.

### Gate G5: conditional SMC correctness

**GO only if:** fixed-tape normalizer tests pass for every enabled resampling mode,
and T5-T7 are discharged.

**NO-GO if:** shared randomness or adaptive resampling is justified only heuristically.

### Gate G6: SMC gradient contract

**GO only if:** the implemented gradient matches the declared scalar objective or is
explicitly labeled as a stopped-resampling surrogate with finite-difference and
ablation evidence.

**NO-GO if:** a learning signal that should be stopped remains attached to autograd,
or omitted resampling scores are called unbiased.

### Gate G7: K adequacy

**GO only if:** a predeclared K meets all of the following on exact stochastic oracles:

- mean total-variation distance \(\le0.02\);
- 95% upper confidence bound on maximum action-marginal error \(\le0.05\);
- correct expected-return mode probability \(\ge0.95\); and
- no material conclusion changes when K is doubled.

Threshold changes require a written scientific justification made before large-scale
benchmark results are examined.

**NO-GO if:** K is selected only for runtime or K=1 is used in a stochastic domain
without proving equivalence.

### Gate G8: reproducibility and isolation

**GO only if:** identical manifests reproduce identical CPU traces where deterministic
execution is requested; evaluation settings do not alter training; and all RNG stream
seeds are recorded.

**NO-GO if:** changing report frequency, evaluation trajectories, or output path changes
the learned policy under the same training seed.

### Gate G9: numerical and support safety

**GO only if:** all probabilities are normalized and finite, proposal support covers
the reference measure, the full-support lower bound and exact mixture denominator are
tested, extreme-logit tests pass, and invalid configurations fail before the first
optimization step.

**NO-GO if:** categorical fallbacks silently assign residual or NaN mass to an action.

### Gate G10: end-to-end recovery

**GO only if:** on exact oracles, the primary sampler recovers the enumerated cold
target within Monte Carlo uncertainty, Candidate A reaches the best solution
available to its variational family, and Candidate B approaches its finite-policy
target as K increases, across the predeclared seed suite.

**NO-GO if:** success is demonstrated only by improvement over initialization.

### Gate G11: scale authorization

**GO only if:** every gate applicable to the method under test among G0-G10 passes,
a runtime/memory pilot succeeds, checkpoint/resume is tested,
and the full run matrix, seeds, stopping rule, and analysis plan are frozen.

**NO-GO if:** more compute is proposed as a remedy for unresolved target, gradient, or
environment-semantics failures.

## 10. Required experiment record

Every run must save, atomically where practical:

- target-specification version;
- source revision and dirty-tree status;
- complete environment instance and checksum;
- horizon, reference measure, execution protocol, and exact-model versus fixed-tape
  score semantics;
- \(\beta\), and, when applicable, \(K\), \(N\), and the tape-bank checksum;
- for replica exchange: the complete beta ladder, ladder construction, iteration
  count, burn-in, thinning, swap interval, initialization mode and policy checksum,
  guide strength, and every frozen guide probability row;
- the PPO proposal checkpoint, action mask, \(\epsilon\), and whether the guide was
  frozen or adapted between sweeps;
- local and swap proposal/acceptance counts, walker visits and round trips,
  action-occupancy diagnostics, retained complete policies, policy-value evaluation
  count, and simulator-transition count;
- guide/proposal training, sampler, and SMC simulator interactions, optimizer
  updates, wall-clock time, accelerator time, and peak memory as separate and total
  costs;
- all optimizer and resampling settings;
- all RNG seeds and stream names;
- dependency/runtime versions and device information;
- per-iteration objective diagnostics, ESS, resampling events, and wall time;
- checkpoints sufficient for exact resume;
- every evaluation return, not only aggregates; and
- explicit status for success, failure, timeout, or numerical abort.

Final comparisons must aggregate independent training seeds, report uncertainty and
failure counts, and keep algorithm budgets comparable. Hyperparameters chosen using
an oracle domain must not be described as untuned on that domain.

## 11. Recommended implementation order

1. Implement exact dynamic-programming and policy-enumeration oracles.
2. Implement one-temperature exact-model policy MH and verify its full transition
   matrix, detailed balance, and cold target on enumerable domains.
3. Add the temperature ladder, adjacent swaps, walker diagnostics, and warm versus
   overdispersed starts; pass G4 in exact-model mode.
4. Implement immutable time-indexed tape banks, validate (RE-2) by enumeration, and
   establish K adequacy through G7 before using fixed-tape results as an
   approximation to (1).
5. Add the frozen full-support guide to local proposals and repeat all detailed-
   balance and mixing tests at the guide strengths intended for experiments.
6. Profile the smallest representative run, verify exact resume, freeze diagnostic
   thresholds and retuning limits, and authorize only the next fail-fast pilot.
7. Maintain complementary Candidate A as an independent objective and gradient
   check; pass G0-G3 and its applicable G8-G10 requirements.
8. Maintain complementary Candidate B only after validating the exact finite-policy
   SAA posterior, no-resampling anchor, conditional normalizer, and resampling
   contract through G5-G7.
9. Add uniform-guide, proposal-alone, exact-versus-SAA, and initialization ablations
   with complete cost accounting.
10. Freeze manifests, seeds, stopping rules, failure handling, and analysis before
    authorizing scale through G11.

The replica-exchange chain is the primary corrected sampler, the direct ELBO is an
independent semantic and optimization check, and replicated SMC is a complementary
SAA candidate. The exact finite-policy posterior remains the ground truth. No
method is promoted because it is computationally elaborate; it must demonstrably
target the declared distribution under the stated semantics.
