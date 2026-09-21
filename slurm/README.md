# Slurm workflow

Do not run experiments on the login node. First run one interactive/single-task
smoke job, inspect its artifacts and accounting, then submit a capped array.

```bash
mkdir -p slurm/logs
sbatch --array=0-6%2 slurm/run_manifest_array.sbatch experiments/smoke_manifest.jsonl
```

After completion:

```bash
python -m mdp_inference.cli audit experiments/smoke_manifest.jsonl
```

Increase concurrency only after runtime, memory, output integrity, and resume
behavior have been checked. Credentials must never be stored in this tree.
