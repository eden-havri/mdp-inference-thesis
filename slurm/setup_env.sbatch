#!/usr/bin/env bash
#SBATCH --job-name=mdpi-env
#SBATCH --account=brafman
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

environment_path=/home/havri/.conda/envs/mdpi

if [[ ! -x "$environment_path/bin/python" ]]; then
  conda create --yes --prefix "$environment_path" python=3.11 pip numpy pytest
fi

conda run --no-capture-output --prefix "$environment_path" \
  python -m pip install --upgrade 'setuptools>=68'
conda run --no-capture-output --prefix "$environment_path" \
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run --no-capture-output --prefix "$environment_path" \
  python -m pip install --no-deps --editable .
conda run --no-capture-output --prefix "$environment_path" \
  python -m pytest -q
