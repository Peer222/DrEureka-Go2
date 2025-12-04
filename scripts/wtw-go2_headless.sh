#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J wtw-go2
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH -o results/wtw-go2/slurm/%j.out
#SBATCH --partition=tnt
#SBATCH --time=24:00:00


module load Miniconda3
#source "/${home}/.bashrc"

conda activate dr_eureka

export LD_LIBRARY_PATH=/bigwork/nhwpduep/.conda/envs/dr_eureka/lib:$LD_LIBRARY_PATH

# prevent different compilers used for torch and gym extension
export CXX=g++
export CC=gcc
rm -rf ~/.cache/torch_extensions

cd /bigwork/nhwpduep/master_thesis/walk-these-ways-go2/scripts

# headless is set hard in file
python train.py
