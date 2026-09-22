# Slurm workflow

Do not run experiments on the login node. First run one interactive/single-task
smoke job, inspect its artifacts and accounting, then submit a capped array.

The September 2026 cluster guide recommends the `main` partition with
`--gpus=0` for ordinary CPU work. The scripts request a small local scratch
allocation and direct temporary files there; final artifacts remain under the
manifest's result directory. Measured pilots used one CPU core and under
300 MB RAM, so experiment rows request one core and 2 GB rather than reserving
idle resources.

The cluster account is `brafman`. From the project root, create the isolated
Python environment on a compute node once:

```bash
mkdir -p slurm/logs
sbatch slurm/setup_env.sbatch
```

Wait for this job and verify that its final `pytest` step passes before any
experiment job is submitted.

```bash
mkdir -p slurm/logs
sbatch --array=0-6%2 slurm/run_manifest_array.sbatch experiments/smoke_manifest.jsonl
```

After completion:

```bash
python -m mdp_inference.cli audit experiments/smoke_manifest.jsonl
jobstats <job-id>
```

Run the one-row canary before any array:

```bash
sbatch --array=0-0 slurm/run_manifest_array.sbatch experiments/cluster_canary_manifest.jsonl
```

Increase concurrency only after runtime, memory, output integrity, and resume
behavior have been checked. Credentials must never be stored in this tree.
