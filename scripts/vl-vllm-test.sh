#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J vl-vllmapi-test
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:2
#SBATCH -o slurm_logs/vl-vllmapi-test/%j.out
#SBATCH --partition=tnt
#SBATCH --time=01:00:00

module load Miniconda3
conda activate vllm

# /project not accessible on compute nodes
DATA_ROOT="/bigwork/nhwpduep/master_thesis/models/"
MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"

HOST=0.0.0.0
PORT=8000

echo "Start server..."
VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.9 --max-model-len 78000 --max-num-seqs 16 --tensor-parallel-size 2 --allowed-local-media-path "/bigwork/nhwpduep/master_thesis/dr-eureka-go2/examples/video_frames/" &
VLLM_PID=$!
echo "Server starting ($VLLM_PID)..."


# Wait until server responds
echo "Waiting for VLLM server to be ready..."
until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
  sleep 10s
done

conda activate dr_eureka
echo "Making request..."
cd /bigwork/nhwpduep/master_thesis/dr-eureka-go2
python playground/vl-vllmapi-test.py
