# Agentic GRPO for Long-Horizon Tasks

本仓库用于探索 **Agentic GRPO（Group Relative Policy Optimization）在长时程（long-horizon）决策任务上的应用**，覆盖 SFT 数据构建、LoRA 微调、多轮 Rollout 与策略优化等完整链路。

---

## 📁 仓库结构

| 目录 | 说明 |
|------|------|
| `agentic-grpo-longhorizon/` | **主项目**。包含配置、数据收集脚本、SFT/评测脚本、实验记录与项目文档 |
| `tau-bench/` | **评测环境**。基于 τ-bench 的航空（Airline）与零售（Retail）多轮交互任务环境 |
| `verl/` | **训练框架**。基于 [volcengine/verl](https://github.com/volcengine/verl) 修改的 RL 训练框架，支持多轮工具调用与 rollout |

---

## 🚀 快速开始

### 环境准备

```bash
cd agentic-grpo-longhorizon
conda create -n agentrl python=3.10 -y
conda activate agentrl
pip install -r requirements.txt

# 若使用 NVIDIA GPU，建议安装 flash-attn
pip install flash-attn --no-build-isolation
```

### 主要流程

| 阶段 | 脚本/说明 |
|------|----------|
| **基线评测** | `scripts/02_run_baseline_eval.py` |
| **SFT 数据采集** | `scripts/04_collect_sft_data.py` |
| **LoRA SFT 训练** | `scripts/05_sft_train.py` |
| **LoRA 合并** | `scripts/05b_merge_lora.py` |
| **SFT 后评测** | `scripts/05c_eval_sft.py` |
| **GRPO 训练** | 基于 `verl/` 框架，使用 `configs/week3_vanilla_grpo.yaml` 等配置 |

---

## 📂 agentic-grpo-longhorizon 目录详解

```
agentic-grpo-longhorizon/
├── configs/                    # 训练与评测配置
│   ├── baseline_airline.yaml
│   ├── sft_airline_lora.yaml
│   └── week3_vanilla_grpo.yaml
├── docs/                       # 阶段文档与实验记录
├── scripts/                    # 核心操作脚本
│   ├── 01_start_vllm_server_*.sh   # vLLM 服务启动
│   ├── 02_run_baseline_eval.py     # 基线评测
│   ├── 04_collect_sft_data.py      # SFT 数据收集
│   ├── 05_sft_train.py             # LoRA SFT
│   └── 20_run_week3_*.sh           # GRPO 训练入口
├── src/                        # 核心代码
│   ├── envs/                   # τ-bench 环境封装
│   ├── evaluation/             # 评测逻辑
│   ├── training/               # SFT/GRPO 训练相关
│   └── utils/                  # 工具函数
├── requirements.txt
└── setup.sh
```

---

## 🔧 tau-bench

多轮对话决策评测基准，包含：
- **Airline**：航班预订、改签、行李、客服转接等任务
- **Retail**：订单查询、修改、退换货等任务

详见 `tau-bench/README.md`。

---

## ⚙️ verl（修改版）

基于 [volcengine/verl](https://github.com/volcengine/verl) `release/v0.6.1` 分支的定制版本，主要改动：
- 支持多轮工具调用（multi-turn tool calling）的 rollout 逻辑
- agent loop 与 tool parser 的适配
- 与 τ-bench 环境的交互接口

---

## 📝 注意事项

- `experiments/`、`outputs/`、`swanlog/` 等实验输出目录已通过 `.gitignore` 排除，不会进入版本控制
- 大模型权重文件（`models/`）与 whl 包已排除，请按需自行下载或安装
- `verl/` 内部已移除原 `.git`，作为本仓库的普通子目录管理，以保留对框架的定制修改

---

## 📄 License

本项目代码遵循各自子目录的原许可证。`verl/` 部分遵循 volcengine/verl 的许可证，`tau-bench/` 部分遵循其原始许可证。
