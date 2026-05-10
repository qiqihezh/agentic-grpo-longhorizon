# 环境信息

本文件记录打包时的环境配置，供新服务器部署参考。

## 系统要求
- Linux (x86_64)
- CUDA 12.x
- Python 3.10+
- 2×A800 80GB (或同等显存 GPU)

## 安装步骤
1. 安装 conda/miniconda
2. 创建环境: `conda create -n agentrl python=3.10`
3. 激活环境: `conda activate agentrl`
4. 安装依赖: `pip install -r requirements.txt`
5. 安装 flash_attn: `pip install flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl` (如有)
6. 安装 tau-bench: `cd tau-bench && pip install -e .`
7. 安装 verl: `cd verl && pip install -e .`

## 模型下载
基座模型需要从 HuggingFace 下载：
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct-AWQ`

或从原服务器 `/workspace/models/` 复制（如有）。

## 项目结构
- `agentic-grpo-longhorizon/`: 主项目代码、配置、脚本、实验结果
- `tau-bench/`: τ-bench 环境（airline/retail）
- `verl/`: veRL 训练框架（含 Turn-Discount 和 LATA 实现）
