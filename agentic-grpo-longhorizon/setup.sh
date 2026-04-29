#!/bin/bash
# 项目环境搭建脚本，建议在全新 conda env 里跑
# 用法: bash setup.sh

set -e

ENV_NAME="agentrl"
PYTHON_VERSION="3.10"

echo "=== [1/5] 创建 conda 环境: $ENV_NAME ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在，跳过创建步骤..."
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi
source activate $ENV_NAME || conda activate $ENV_NAME

echo "=== [2/5] 安装 PyTorch (CUDA 12.1) ==="
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/5] 安装项目依赖 ==="
pip install -r requirements.txt

echo "=== [4/5] 安装 τ-bench ==="
# τ-bench 建议放在项目同级目录,不要装进项目里避免污染
cd ..
if [ ! -d "tau-bench" ]; then
    git clone https://github.com/sierra-research/tau-bench
    #git clone https://ghproxy.net/https://github.com/sierra-research/tau-bench.git
fi
cd tau-bench
pip install -e .
cd ../agentic-grpo-longhorizon

echo "=== [5/5] 从 ModelScope 下载模型 ==="
# 国内网络,用 ModelScope 更快
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
# 72B 太大,先注释,需要时再下
 snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', cache_dir='./models')
"

echo "=== 搭建完成 ==="
echo "激活环境: conda activate $ENV_NAME"