#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J vllm-test
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:2
#SBATCH -o slurm_logs/vllm-test-multigpu/%j.out
#SBATCH --partition=tnt
#SBATCH --time=01:00:00

module load Miniconda3
conda activate vllm

if [ "$#" -ne 1 ]; then
  echo "Usage: $1 <llm-type> [open-ai/gpt-oss-20b, Qwen/Qwen3-32B-AWQ, Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8]"
  exit 1
fi

# /project not accessible on compute nodes
DATA_ROOT="/bigwork/nhwpduep/master_thesis/models/" # "/project/NHWP25179/vllm/"
MODEL=$1

HOST=0.0.0.0
PORT=8000

echo "Start server..."
if [ "$MODEL" = "open-ai/gpt-oss-20b" ]; then
  VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" TIKTOKEN_ENCODINGS_BASE="$DATA_ROOT$MODEL/encodings" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --tensor-parallel-size 2 &
elif [ "$MODEL" = "Qwen/Qwen3-32B-AWQ" ]; then
  VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.96 --max-num-seqs 16 --tensor-parallel-size 2 &
else
  VLLM_CACHE_ROOT="/bigwork/nhwpduep/.cache" vllm serve "$DATA_ROOT$MODEL" --host $HOST --port $PORT --seed 0 --gpu-memory-utilization 0.96 --max-num-seqs 16 --max-model-len 120000 --tensor-parallel-size 2 &
fi
VLLM_PID=$!
echo "Server starting ($VLLM_PID)..."

# Wait until server responds
echo "Waiting for VLLM server to be ready..."
until curl -s "http://$HOST:$PORT/v1/models" > /dev/null; do
  sleep 10s
done

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

curl -X 'POST' "http://$HOST:$PORT/tokenize" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d "{
    \"model\": \"$DATA_ROOT$MODEL\",
    \"prompt\": \"Here You can decode the tokens\"
  }"
echo "\n Request completed..."

# kill -0 $VLLM_PID
