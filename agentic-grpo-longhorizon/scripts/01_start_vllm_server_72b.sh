#!/bin/bash
# 启动 vLLM server (72B AWQ)
# 双卡 A800 分离部署场景：
#   - GPU 0 (port 8000): 跑 policy，供 agent 决策
#   - GPU 1 (port 8001): 跑 user simulator，供 tau-bench 模拟用户
# 两张卡共用同一份模型权重文件，各加载到自己的显存中
# 在集群上建议开两个 tmux session 分别跑
#
# [16K 约束]
# max_model_len = 16384，主动截断阈值 14K tokens（≈49000 字符）
# 留 2K tokens 给 assistant 生成，避免 vLLM 400 / CUDA crash

set -e

MODEL_PATH=${MODEL_PATH:-"../models/Qwen2.5-72B-Instruct-AWQ"}
PORT=${PORT:-8001}               # policy 用 8000，user sim 用 8001
TP_SIZE=${TP_SIZE:-1}            # 单卡 A800 80GB 跑 72B AWQ，TP=1 足够
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}  # 16K 上下文 + AWQ，0.90 留安全余量
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}  # 16K 硬约束
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}        # policy 默认 4；user sim 上下文短，可设 6-10
CUDA_DEVICES=${CUDA_DEVICES:-2}  # GPU 0 和 1 当 policy，GPU 2 当 user sim

echo "Starting vLLM server (72B AWQ) with 16K context limit..."
echo "Model:    $MODEL_PATH"
echo "Port:     $PORT"
echo "TP:       $TP_SIZE"
echo "GPU:      $CUDA_DEVICES"
echo "MaxLen:   $MAX_MODEL_LEN"
echo "MaxSeqs:  $MAX_NUM_SEQS"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# 确保使用 agentrl conda 环境的 Python
#PYTHON=$(conda run -n agentrl which python)

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen/Qwen2.5-72B-Instruct-AWQ" \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --quantization awq \
    --enable-prefix-caching \
    --no-enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
