#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_3090:2

#SBATCH -J vl-vllm-eureka-submitit
#SBATCH -o slurm_logs/vl-vllm-eureka-submitit/%j.out
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


if [ "$#" -ne 2 ]; then
    echo "Usage: $1 <llm-type> [Qwen/Qwen3-VL-30B-A3B-Thinking-FP8, Qwen/Qwen3.5-27B-FP8] $2 <environment>"
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
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.9 --max-model-len 78000 --max-num-seqs 16 --tensor-parallel-size 2 --allowed-local-media-path "/bigwork/nhwpduep/master_thesis/dr-eureka-go2/" &
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
python v-eureka.py model=$MODEL env=$2 use_submitit=1
