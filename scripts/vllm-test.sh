#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J vllm-test
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/vllm-test/%j.out
#SBATCH --partition=ai
#SBATCH --time=00:10:00

module load Miniconda3
#source "/${home}/.bashrc"

conda activate vllm

DATA_ROOT="/project/NHWP25179/vllm/"  # "/bigwork/nhwpduep/data/"
#MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
MODEL="openai/gpt-oss-20b"

HOST=0.0.0.0
PORT=8000

echo "Start server..."
CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --max_model_len 80000 &
VLLM_PID=$!
echo "Server starting..."
echo "$VLLM_PID"

echo "Sleeping for 7 minutes..."
sleep 7m

conda activate dr_eureka

echo "Making request..."
curl "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$DATA_ROOT$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"Explain quantum computing in simple terms.\"}
    ],
    \"max_tokens\": 200
  }"
echo "\nRequest completed..."

kill -0 $VLLM_PID
