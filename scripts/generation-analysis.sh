#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J generation-analysis
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/gen-analysis/%j.out
#SBATCH --partition=tnt,ai
#SBATCH --time=03:00:00

module load Miniconda3
conda activate vllm

if [ "$#" -ne 1 ]; then
  echo "Usage: $1 /path/to/stats.csv"
  exit 1
fi

python playground/generation-analysis.py --statspath $1