#!/bin/bash
#SBATCH --partition=ai.h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH -J vllm-eureka
#SBATCH -o results/vllm/%j.out
#SBATCH --time=24:00:00

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

module load Miniconda3
#source "/${home}/.bashrc"

conda activate vllm

DATA_ROOT="/bigwork/nhwpduep/data/"
#MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
MODEL="openai/gpt-oss-20b"

HOST=0.0.0.0
PORT=8000

echo "Start server..."
CUDA_VISIBLE_DEVICES=1 VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --max_model_len 80000
