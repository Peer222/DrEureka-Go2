#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --reservation=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_3090:2

#SBATCH -J large-vllm-eureka-evolved-submitit-resume
#SBATCH -o slurm_logs/large-vllm-eureka-evolved-submitit-resume/%j.out
#SBATCH --time=6-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


if [ "$#" -ne 4 ]; then
    echo "Usage: $1 <llm-type> [open-ai/gpt-oss-20b, Qwen/Qwen3-32B-AWQ, Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, Qwen/Qwen3.5-27B-FP8] $2 <environment> $3 <seed> $4 </path/to/stats.csv>"
    exit 1
fi

if [[ $4 == /* ]]; then echo "absolute path provided"; else echo "absolute path to stats.csv required"; exit 1; fi

module load Miniconda3
conda activate vllm

# /project not accessible on compute nodes
DATA_ROOT="/bigwork/nhwpduep/master_thesis/models/" # "/project/NHWP25179/vllm/"
MODEL=$1

HOST=0.0.0.0
PORT=8000

echo "Start server..."
if [ "$MODEL" = "open-ai/gpt-oss-20b" ]; then
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed $3 --tensor-parallel-size 2 &
elif [ "$MODEL" = "Qwen/Qwen3-32B-AWQ" ]; then
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed $3 --gpu-memory-utilization 0.9 --max-num-seqs 16 --tensor-parallel-size 2 &
elif [ "$MODEL" = "Qwen/Qwen3.5-27B-FP8" ]; then
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed $3 --gpu-memory-utilization 0.9 --max-model-len 78000 --max-num-seqs 16 --tensor-parallel-size 2 --allowed-local-media-path "/bigwork/nhwpduep/master_thesis/dr-eureka-go2/" &
else
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed $3 --gpu-memory-utilization 0.96 --max-num-seqs 16 --max-model-len 120000 --tensor-parallel-size 2 &
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
python eureka-evolved.py model=$MODEL env=$2 resume=True stats_file=$4 use_submitit=1 seed=$3

kill -0 $VLLM_PID
