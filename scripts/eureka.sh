#!/bin/bash
#SBATCH --partition=tnt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_3090:1

#SBATCH -J eureka
#SBATCH -o slurm_logs/eureka/%j.out
#SBATCH --time=24:00:00

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

module load Miniconda3

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
CUDA_VISIBLE_DEVICES=0 python eureka.py env=$1

kill -0 $VLLM_PID
