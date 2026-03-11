#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_3090:3

#SBATCH -J large-vllm-eureka
#SBATCH -o slurm_logs/large-vllm-eureka/%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


if [ "$#" -ne 2 ]; then
    echo "Usage: $1 <llm-type> [open-ai/gpt-oss-20b, Qwen/Qwen3-32B-AWQ, Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8] $2 <environment>"
    exit 1
fi

module load Miniconda3
conda activate vllm

# /project not accessible on compute nodes
DATA_ROOT="/bigwork/nhwpduep/master_thesis/models/" # "/project/NHWP25179/vllm/"
MODEL=$1 # "openai/gpt-oss-20b"

HOST=0.0.0.0
PORT=8000

echo "Start server..."
if [ "$MODEL" = "open-ai/gpt-oss-20b" ]; then
  CUDA_VISIBLE_DEVICES="1,2" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --tensor-parallel-size 2 &
elif [ "$MODEL" = "Qwen/Qwen3-32B-AWQ" ]; then
  CUDA_VISIBLE_DEVICES="1,2" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.96 --max-num-seqs 16 --tensor-parallel-size 2 &
else
  CUDA_VISIBLE_DEVICES="1,2" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.96 --max-num-seqs 16 --max-model-len 120000 --tensor-parallel-size 2 &
fi

VLLM_PID=$!
echo "Server starting ($VLLM_PID)..."

# Wait until server responds
echo "Waiting for VLLM server to be ready..."
until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
  sleep 10s
done

conda activate dr_eureka
cd /bigwork/nhwpduep/master_thesis/dr-eureka-go2/eureka/

export LD_LIBRARY_PATH=/bigwork/nhwpduep/.conda/envs/dr_eureka/lib:$LD_LIBRARY_PATH
# prevent different compilers used for torch and gym extension
export CXX=g++
export CC=gcc
rm -rf ~/.cache/torch_extensions

echo ""
pip list
echo ""

echo "Starting Gym..."
export WANDB_MODE="offline"
CUDA_VISIBLE_DEVICES=0 python eureka.py model=$MODEL env=$2

kill -0 $VLLM_PID
