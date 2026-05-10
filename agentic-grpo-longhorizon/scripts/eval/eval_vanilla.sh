#!/bin/bash
# 批量评估 vanilla GRPO 的 4 个 checkpoint (hf_step_50/100/150/200)
#
# 前置条件：
#   1. 72B user sim 已在 port 8001 运行：
#      bash scripts/vllm_server/72b.sh
#   2. 当前目录为 agentic-grpo-longhorizon/
#
# 用法：
#      bash scripts/eval/eval_vanilla.sh

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/../.."
STEPS=(50 100 150 200)
SPLIT_FILE="experiments/sft_collect_airline/split.json"

# 如果 vanilla 有自己的 split.json，优先用它
if [ -f "experiments/vanilla/split.json" ]; then
    SPLIT_FILE="experiments/vanilla/split.json"
fi

echo "========================================"
echo "  Vanilla GRPO 批量评估"
echo "  Split file: ${SPLIT_FILE}"
echo "========================================"

for STEP in "${STEPS[@]}"; do
    MODEL_PATH="experiments/vanilla/hf_step_${STEP}"
    OUTPUT_DIR="experiments/vanilla/eval_step_${STEP}"
    CONFIG="configs/eval/vanilla_grpo/eval_vanilla_step${STEP}.yaml"

    echo ""
    echo "========================================"
    echo "  [$(date '+%H:%M:%S')] Evaluating ${MODEL_PATH}"
    echo "========================================"

    # 清理可能残留的 8000 端口进程
    echo "Cleaning up port 8000..."
    pkill -f "api_server.*port 8000" || true
    sleep 3

    mkdir -p "${OUTPUT_DIR}"

    # 启动 policy server (后台)
    echo "Starting policy server on port 8000 (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.82 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &

    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    # 等待 vLLM 启动完成（检测 /v1/models 是否 ready）
    echo "Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "vLLM is ready!"
            break
        fi
        sleep 5
        if [ $i -eq 60 ]; then
            echo "ERROR: vLLM failed to start within 5 minutes. Check ${OUTPUT_DIR}/vllm_server.log"
            kill $SERVER_PID || true
            exit 1
        fi
    done
    sleep 5  # 多留一点缓冲

    # 运行评估
    echo "Running eval..."
    python scripts/eval/eval_sft.py \
        --config "${CONFIG}" \
        --split-file "${SPLIT_FILE}"

    # 关闭 policy server
    echo "Stopping policy server..."
    kill $SERVER_PID || true
    sleep 3
    pkill -f "api_server.*port 8000" || true
    sleep 5

done

echo ""
echo "========================================"
echo "  所有评估已完成！"
echo "  结果目录："
for STEP in "${STEPS[@]}"; do
    echo "    experiments/vanilla/eval_step_${STEP}/split_eval_report.json"
done
echo "========================================"
