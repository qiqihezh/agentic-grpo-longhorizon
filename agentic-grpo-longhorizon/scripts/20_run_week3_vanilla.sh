#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

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
export SWANLAB_RUN_ID="6ewzzw71dojehfhws9a85"
export SWANLAB_RESUME="true"

# expandable_segments disabled: incompatible with vLLM memory pool
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /workspace/agentic-grpo-longhorizon
mkdir -p experiments/week3_vanilla

# nohup python -m verl.trainer.main_ppo \
#     --config-path=$(pwd)/configs \
#     --config-name=week3_vanilla_grpo \
#     > experiments/week3_vanilla/training.log 2>&1 &
# echo "Training PID: $!"

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=week3_vanilla_grpo