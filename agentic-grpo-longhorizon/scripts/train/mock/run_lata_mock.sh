#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate agentrl

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
mkdir -p experiments/lata_mock

python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=mock_lata_3step
