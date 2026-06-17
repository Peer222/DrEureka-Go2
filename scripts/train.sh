#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --reservation=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1

#SBATCH -J train
#SBATCH -o slurm_logs/train/%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


if [ "$#" -ne 5 ]; then
    echo "Usage: $1 <environment> $2 <iterations> $3 <reward ["original", "eureka", "eureka_original"]> $4 <dr ["original", "eureka", "eureka_original", "off"] $5 seed>"
    exit 1
fi

module load Miniconda3

conda activate dr_eureka
cd /bigwork/nhwpduep/master_thesis/dr-eureka-go2/

export LD_LIBRARY_PATH=/bigwork/nhwpduep/.conda/envs/dr_eureka/lib:$LD_LIBRARY_PATH
# prevent different compilers used for torch and gym extension
export CXX=g++
export CC=gcc
rm -rf ~/.cache/torch_extensions

echo "Starting Gym..."
# /bigwork/nhwpduep/master_thesis/dr-eureka-go2/runs/eureka/2026-02-21_09:35:30_GL_Go2_Qwen-30BQ-nt/1/14_2026-02-22_10:16:49/checkpoints
export WANDB_MODE="offline"
python "$1/scripts/train.py" --iterations "$2" --headless --dr-config "$4" --reward-config "$3" --wandb-group "train/$1" --device "cuda:0" --no-wandb --seed "$5"
