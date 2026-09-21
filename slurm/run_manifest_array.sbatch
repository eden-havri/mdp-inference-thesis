#!/usr/bin/env bash
#SBATCH --job-name=mdpi
#SBATCH --account=brafman
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch --array=0-N%8 slurm/run_manifest_array.sbatch MANIFEST.jsonl" >&2
  exit 2
fi

manifest_path="$1"
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

mkdir -p slurm/logs
conda run --no-capture-output --prefix /home/havri/.conda/envs/mdpi \
  python -m mdp_inference.cli run-manifest-row "$manifest_path" "$task_index"
