# Expected-Return Policy Inference

Research code and LaTeX sources for Eden Havri's thesis on treating finite-
horizon control as Bayesian inference over complete deterministic policies.

The target distribution is

\[
p_\beta(\pi) \propto \mu(\pi)\exp\{\beta\,\mathbb E[G\mid\pi]\}.
\]

The expectation is taken **before** exponentiation. This distinction is
important: exponentiating one sampled return instead targets a different,
risk-sensitive objective.

## Current method

The primary corrected sampler is guided replica-exchange Metropolis--Hastings
over complete policies. A PPO policy supplies proposals and an initializer;
the Hastings correction preserves the declared Gibbs target. Exact dynamic-
programming returns are used when the transition model is available. A fixed-
tape sample-average variant is available for simulator-only studies and is
reported explicitly as an approximation.

Complementary implementations include direct expected-return ELBO training,
replicated policy SMC, single-chain policy MH, and a restricted policy bank.
The standard comparison set is REINFORCE with a learned value baseline,
PPO-Clip, discrete SAC, categorical CEM, and Double Q-learning. Random control
and finite-horizon dynamic programming are reported separately.

## Repository layout

- `mdp_inference/`: corrected methods, baselines, environments, evaluation,
  experiment runner, and artifact handling.
- `tests/`: semantic, exact-target, sampler, baseline, and reproducibility tests.
- `experiments/`: frozen maps and fail-fast JSONL manifests.
- `docs/`: method contract, competitor contract, experiment plan, domain study,
  and scaling plan.
- `latex/`: shared scientific content with paper and BGU thesis wrappers.
- `slurm/`: BGU cluster setup and capped-array scripts.
- `gocai/`: earlier compatibility prototype; it is not the primary experiment
  implementation.

## Local setup and checks

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
```

Run or audit one manifest row through the CLI:

```bash
python -m mdp_inference.cli audit experiments/cluster_canary_manifest.jsonl
python -m mdp_inference.cli run-manifest-row experiments/cluster_canary_manifest.jsonl 0
python -m mdp_inference.cli audit-tempering-chains results/medium-tempering-multichain-gate
```

Every run writes a self-contained result directory and creates `DONE` only
after its artifacts have been validated.

## Cluster workflow

Login nodes are used only for submission and monitoring. From the project root
on the cluster:

```bash
mkdir -p slurm/logs
sbatch slurm/setup_env.sbatch
sbatch --array=0-0 slurm/run_manifest_array.sbatch experiments/cluster_canary_manifest.jsonl
```

Inspect tests, result artifacts, `sacct`, and `jobstats` before releasing a
capped array. See `slurm/README.md` and `docs/experiment_plan.md` for the full
fail-fast protocol.
