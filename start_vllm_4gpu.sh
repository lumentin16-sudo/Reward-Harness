#!/usr/bin/env bash
set -euo pipefail

# 修改成服务器上已经下载好的 Qwen3-8B 权重目录。
MODEL_PATH="/home/mao/models/Qwen3-8B"
SERVED_MODEL_NAME="Qwen/Qwen3-8B"
HOST="0.0.0.0"
PORT="8000"

export CUDA_VISIBLE_DEVICES=0,1

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 32

# 如果单卡显存不足，请删除上面的 exec 命令，改用下面的两卡张量并行：
# exec vllm serve "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT" \
#   --data-parallel-size 1 --tensor-parallel-size 2 --dtype bfloat16 \
#   --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 32
