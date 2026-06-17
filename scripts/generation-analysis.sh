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

echo "Usage: $1 resultdir $2+ /paths/to/stats.csv"

python playground/generation-analysis.py --resultdir "$1" --runs "${@:2}"