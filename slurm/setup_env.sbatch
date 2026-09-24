#!/usr/bin/env bash
#SBATCH --job-name=mdpi-env
#SBATCH --account=brafman
#SBATCH --partition=main
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --tmp=2G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

environment_path=/home/havri/.conda/envs/mdpi
environment_python="$environment_path/bin/python"

if [[ ! -f pyproject.toml || ! -f requirements.txt ]]; then
  echo "submit this job from the project root" >&2
  exit 2
fi

if [[ ! -x "$environment_python" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    module load anaconda
  fi
  conda create --yes --prefix "$environment_path" python=3.11 pip numpy pytest
fi

# Install PyTorch from its CPU-only index first.  The subsequent requirements
# install sees this satisfying torch>=2.1 and therefore does not pull a GPU
# wheel from PyPI.
"$environment_python" -m pip install "torch>=2.1" \
  --index-url https://download.pytorch.org/whl/cpu
"$environment_python" -m pip install --requirement requirements.txt
"$environment_python" -m pip install --no-build-isolation --no-deps --editable .
"$environment_python" -c \
  'import matplotlib, numpy, PIL, pydantic, pytest, torch, tqdm'
"$environment_python" -m pytest -q
