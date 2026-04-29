#!/bin/bash
# 启动 vLLM server (7B),提供 OpenAI 兼容 API
# 在集群上建议 tmux session 里跑

set -e

MODEL_PATH=${MODEL_PATH:-"../models/Qwen2.5-7B-Instruct"} #../models/Qwen2.5-7B-Instruct
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
CUDA_DEVICES=${CUDA_DEVICES:-0}   # 默认用第 0 张卡

echo "Starting vLLM server (7B)..."
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "TP:     $TP_SIZE"
echo "GPU:    $CUDA_DEVICES"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \ 
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 10 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
