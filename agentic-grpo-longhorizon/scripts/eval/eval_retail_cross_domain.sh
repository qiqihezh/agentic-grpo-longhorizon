#!/bin/bash
# 跨域泛化评测：将在 airline 上训练的 PRM-Lite checkpoint 在 retail 上测试
#
# 说明：
#   - 模型在 airline 数据上训练，从未见过 retail 的任何 task 或 tool schema
#   - retail 有 115 个 test tasks（vs airline 的 50 个）
#   - 不传 --split-file，因为所有 retail task 对该模型都是 unseen
#   - 只报 overall 指标（无 covered/uncovered/unseen 分组）
#
# 前置条件：
#   1. 72B user sim 已在 port 8001 运行
#   2. HF checkpoint 已生成（python scripts/test/merge_fsdp_to_hf.py）
#
# 用法：
#   # 快速验证（10 tasks, 2 samples，约 10-15 分钟）
#   bash scripts/eval/eval_retail_cross_domain.sh tiny
#
#   # 完整评测（115 tasks, 4 samples，约 2-3 小时）
#   bash scripts/eval/eval_retail_cross_domain.sh full

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/../.."
MODE="${1:-tiny}"   # tiny | full

if [ "$MODE" = "tiny" ]; then
    STEPS=(250)
    echo "========================================"
    echo "  Retail 跨域泛化评测（TINY 快速验证）"
    echo "  10 tasks × 2 samples"
    echo "========================================"
else
    STEPS=(250)
    echo "========================================"
    echo "  Retail 跨域泛化评测（FULL 完整）"
    echo "  115 tasks × 4 samples per checkpoint"
    echo "========================================"
fi

for STEP in "${STEPS[@]}"; do
    MODEL_PATH="experiments/prm_lite/hf_step_${STEP}"

    if [ "$MODE" = "tiny" ]; then
        CONFIG="configs/eval/retail/eval_retail_tiny.yaml"
        OUTPUT_DIR="experiments/prm_lite/eval_retail_step${STEP}_tiny"
    else
        CONFIG="configs/eval/retail/eval_retail.yaml"
        OUTPUT_DIR="experiments/prm_lite/eval_retail_step${STEP}"
    fi

    echo ""
    echo "========================================"
    echo "  [$(date '+%H:%M:%S')] Evaluating ${MODEL_PATH} on retail"
    echo "========================================"

    # 检查 HF 模型是否存在
    if [ ! -f "${MODEL_PATH}/model.safetensors" ]; then
        echo "WARNING: HF model not found at ${MODEL_PATH}/model.safetensors"
        echo "Run python scripts/test/merge_fsdp_to_hf.py first."
        continue
    fi

    # 检查配置文件是否存在
    if [ ! -f "${CONFIG}" ]; then
        echo "Creating config ${CONFIG} ..."
        mkdir -p "$(dirname "$CONFIG")"
        cat > "${CONFIG}" <<EOF
# Auto-generated retail eval config for step ${STEP}
env:
  name: retail
  user_strategy: llm
  user_model: "Qwen/Qwen2.5-72B-Instruct-AWQ"
  user_provider: openai
  user_base_url: "http://localhost:8001/v1"
  task_split: test

policy:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  base_url: "http://localhost:8000/v1"
  temperature: 0.7
  top_p: 0.9
  max_tokens: 4096

eval:
  num_tasks: 115
  num_samples_per_task: 4
  max_turns: 30
  num_workers: 2

output:
  dir: "${OUTPUT_DIR}"
EOF
    fi

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

    # 等待 vLLM 启动完成
    echo "Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "vLLM is ready!"
            break
        fi
        sleep 5
        if [ $i -eq 60 ]; then
            echo "ERROR: vLLM failed to start within 5 minutes."
            kill $SERVER_PID || true
            exit 1
        fi
    done
    sleep 5

    # 运行评估（注意：不传 --split-file，因为 retail 没有 split.json，且跨域无需分组）
    echo "Running retail eval..."
    if [ "$MODE" = "tiny" ]; then
        python scripts/eval/eval_sft.py \
            --config "${CONFIG}" \
            --tiny
    else
        python scripts/eval/eval_sft.py \
            --config "${CONFIG}"
    fi

    # 关闭 policy server
    echo "Stopping policy server..."
    kill $SERVER_PID || true
    sleep 3
    pkill -f "api_server.*port 8000" || true
    sleep 5

done

echo ""
echo "========================================"
echo "  Retail 跨域评测已完成！"
echo "  结果目录："
for STEP in "${STEPS[@]}"; do
    if [ "$MODE" = "tiny" ]; then
        echo "    experiments/prm_lite/eval_retail_step${STEP}_tiny/eval_report.json"
    else
        echo "    experiments/prm_lite/eval_retail_step${STEP}/eval_report.json"
    fi
done
echo "========================================"
