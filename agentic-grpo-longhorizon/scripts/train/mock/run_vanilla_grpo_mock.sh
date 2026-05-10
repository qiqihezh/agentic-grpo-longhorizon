#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

# 注意: PyTorch 是 2.7.0+cu126(自带 CUDA 12.6 runtime),但系统 CUDA 工具链是 12.4
# CUDA_HOME 必须指向实际存在的系统目录,供 triton/deepspeed 找 ptxas
export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR

cd /workspace/agentic-grpo-longhorizon
mkdir -p experiments/vanilla_mock

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=mock_grpo
