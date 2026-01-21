#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1

#SBATCH -J globe-walking-go2
#SBATCH -o slurm_logs/globe-walking-go2/%j.out
#SBATCH --time=4-00:00:00

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <iterations>"
    exit 1
fi

module load Miniconda3

conda activate dr_eureka
cd /bigwork/nhwpduep/master_thesis/dr-eureka-go2/globe_walking_go2/

export LD_LIBRARY_PATH=/bigwork/nhwpduep/.conda/envs/dr_eureka/lib:$LD_LIBRARY_PATH
# prevent different compilers used for torch and gym extension
export CXX=g++
export CC=gcc
#rm -rf ~/.cache/torch_extensions

echo ""
pip list
echo ""

echo "Starting Gym..."
WANDB_MODE="offline" python -u scripts/train.py --iterations $1 --headless --dr-config off --reward-config eureka
