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

environment_path=/home/havri/.conda/envs/mdpi
environment_python="$environment_path/bin/python"

if [[ ! -x "$environment_python" ]]; then
  conda create --yes --prefix "$environment_path" python=3.11 pip numpy pytest
fi

if ! "$environment_python" -c 'import numpy, pytest, torch'; then
  "$environment_python" -m pip install torch \
    --index-url https://download.pytorch.org/whl/cpu
fi
"$environment_python" -m pip install --no-build-isolation --no-deps --editable .
"$environment_python" -m pytest -q
