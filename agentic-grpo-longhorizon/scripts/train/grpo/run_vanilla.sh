#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

# 注意: PyTorch 是 2.7.0+cu126(自带 CUDA 12.6 runtime),但系统 CUDA 工具链是 12.4
# CUDA_HOME 必须指向实际存在的系统目录,供 triton/deepspeed 找 ptxas
export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
export CUDA_VISIBLE_DEVICES=0,1
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR
export SWANLAB_RUN_ID="6ewzzw71dojehfhws9a85"
export SWANLAB_RESUME="true"

# expandable_segments disabled: incompatible with vLLM memory pool
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /workspace/agentic-grpo-longhorizon
mkdir -p experiments/vanilla

# nohup python -m verl.trainer.main_ppo \
#     --config-path=$(pwd)/configs \
#     --config-name=vanilla_grpo \
#     > experiments/vanilla/training.log 2>&1 &
# echo "Training PID: $!"

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=vanilla_grpo