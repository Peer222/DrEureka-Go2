#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:2

#SBATCH -J vllm-dr
#SBATCH -o slurm_logs/vllm-dr/%j.out
#SBATCH --time=3-00:00:00

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <environment> <run_name>"
    exit 1
fi

module load Miniconda3
#source "/${home}/.bashrc"

conda activate vllm

# /project not accessible on compute nodes
DATA_ROOT="/bigwork/nhwpduep/master_thesis/models/" # "/project/NHWP25179/vllm/"
#MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
MODEL="openai/gpt-oss-20b"

HOST=0.0.0.0
PORT=8000

echo "Start server..."
CUDA_VISIBLE_DEVICES=1 VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --max_model_len 80000 &
VLLM_PID=$!
echo "Server starting ($VLLM_PID)..."

# Wait until server responds
echo "Waiting for VLLM server to be ready..."
until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
  sleep 10s
done

conda activate dr_eureka
cd /bigwork/nhwpduep/master_thesis/dr-eureka-go2/dr_eureka/

export LD_LIBRARY_PATH=/bigwork/nhwpduep/.conda/envs/dr_eureka/lib:$LD_LIBRARY_PATH
# prevent different compilers used for torch and gym extension
export CXX=g++
export CC=gcc
rm -rf ~/.cache/torch_extensions

echo ""
pip list
echo ""

echo "Starting Gym..."
CUDA_VISIBLE_DEVICES=0 python rapp.py env=$1 run_path=runs/$1/$2
CUDA_VISIBLE_DEVICES=0 python dr_eureka.py env=$1

kill -0 $VLLM_PID
