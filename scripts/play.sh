#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1

#SBATCH -J play
#SBATCH -o slurm_logs/play/%j.out
#SBATCH --time=2:00:00

if [ "$#" -ne 2 ]; then
    echo "Usage: $1 <environment> $2 <run_checkpoint_dir>"
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
python "$1/scripts/play.py" --headless --run "$2" --dr-config eureka --num-rollouts 5 --load-reward
