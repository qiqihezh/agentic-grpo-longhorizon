# Agentic-GRPO-LongHorizon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **解决长链路多工具智能体中的 GRPO 训练崩溃问题**  
> 在 τ-bench airline（50 任务、多轮对话、多工具调用场景）上的系统性消融实验，通过创新的 **PRM-Lite + LATA 联合方案**，相较 Vanilla GRPO 基线实现 **+37% 的整体 pass^1 提升**。

---

## 🔥 核心结果

**最优检查点（step 250）：联合方案（PRM-Lite + LATA）达到 0.240 的整体 pass^1 —— 相较 Vanilla GRPO 基线（0.175）提升 +37%。**

| 指标 | Vanilla | Turn-Discount | PRM-Lite | LATA | **联合方案** | 相较 Vanilla |
|------|---------|---------------|----------|------|------------|-------------|
| **整体 pass^1** | 0.175 | 0.125 | 0.140 | 0.185 | **0.240** | **+37%** |
| **泛化能力** | 0.071 | 0.052 | 0.059 | 0.088 | **0.110** | **+55%** |
| **错误率** | 0.200 | 0.345 | 0.365 | 0.290 | **0.140** | **−30%** |
| **推理深度** (p50 tokens) | 72 | 245 | 169 | 183 | **313** | **+334%** |

> *泛化 pass^1 = (uncovered_seen × 24 + unseen × 10) / 34，排除训练集泄漏影响的核心指标。*

### 多维度对比

![消融实验对比](ablation_comparison.png)

### 训练轨迹

![训练轨迹](ablation_progression.png)

> **观察**：Turn-Discount 停滞（被动保护）；LATA 通过 √L 归一化持续增长；联合方案在 step 250 达到峰值（0.240），step 300 温和回落至 0.225。

---

## 🎯 问题与动机

标准的 **GRPO（Group Relative Policy Optimization）** 应用于长链路、多工具对话智能体（τ-bench airline，50 任务，40 训练 / 10 测试）时会发生灾难性的**训练崩溃**。我们识别出**三个根本原因**：

### 1. 群组奖励饱和（双向死锁）
结果奖励是二元的（0/1）。当 `group_size=8` 时，群组容易达到全 0 或全 1 状态 → `优势方差 → 0` → 梯度消失。

### 2. 训练集泄漏偏差
40 个训练任务中有 16 个是 *covered_seen*（有 72B 教师轨迹覆盖）。策略记住教师模式，covered 性能虚高，而 uncovered/unseen 几乎为零。

### 3. 每轮推理退化
线性长度归一化 `advantage / L` 惩罚长回复。策略学会**"以量补质"** —— 短推理 + 频繁工具试错 —— 导致 step 150 后崩溃。

> **关键发现**：训练验证奖励**不是**可靠代理。Turn-Discount 报告验证奖励 0.80，但真实评测仅 0.125 —— **6.4 倍差距**。

---

## 💡 方法

我们设计并验证了**四个消融实验**，每个针对特定的失效模式：

### 实验 1：Turn-Discounted Advantage
**思路**：通过指数衰减的 token 权重保护早期轮次推理。  
**机制**：`weight[t] = α^(L-1-t)`，其中 `α=1.05`，归一化至 `mean(weight)=1`。早期 token 获得更高优势，抑制晚期猜测。  
**结果**：成功阻止崩溃形状（回复长度 −23% vs Vanilla −63%），但评测仍然较低（0.125），因为缺乏质量引导。

### 实验 2：LATA — 长度感知轮次优势
**思路**：用 `1/√L` 替换线性 `1/L` 归一化，保留长推理的边际激励。  
**机制**：`advantage_token = A / sqrt(L)` 替代 `A / L`。当回复长度增长 4 倍时，每 token 梯度仅减半（Vanilla 中降至四分之一）。  
**结果**：持续提升（0.155 → 0.185 → 0.190），错误率从 0.345 降至 0.290，但无质量信号时可见天花板。

### 实验 3：PRM-Lite — 轻量级过程奖励
**思路**：用密集的基于规则的过程奖励打破群组饱和。  
**机制**：15 条手工规则（P1–P8 惩罚，B1–B7 奖励）提供连续的 `[-0.5, +0.5]` 信号。最终奖励 = `outcome + 0.3 × process_score`。  
**结果**：成功消除 score/min = 0/1 死锁，但信号被轨迹级线性归一化稀释 —— 错误率反而恶化至 0.365。

### 实验 4：联合方案 — PRM-Lite + LATA ⭐
**思路**：**PRM-Lite 提供局部质量信号；LATA 的 √L 归一化确保这些信号不被回复长度淹没。**  
**机制**：每轮过程分数的惩罚/奖励通过 `A/√L` 传播到单个 token，使策略能够学习*哪一轮错了*，而不仅仅是*整个轨迹是否成功*。  
**结果**：**0.240 整体表现** —— 超越所有单组件基线。错误率唯一持续下降（0.170 → 0.140 → 0.120）。Unseen 任务表现转正并稳定。

> **核心洞察**：价值不在于单独拥有过程奖励*或*更好的归一化 —— 而在于**信号传播**。PRM-Lite 生成局部信号；LATA 的 √L 提供传输通道。两者单独使用效果均不佳。

---

## 🌟 技术亮点

### 1. 信号传输理论（算法贡献）
消融报告实证证明了长链路智能体 GRPO 的**分解原理**：
- **信号源**（PRM-Lite）：15 条手工规则提供密集的每轮质量信号 `[-0.5, +0.5]`。
- **信号通路**（LATA）：`advantage / √L` 替代 `advantage / L`，防止回复长度稀释。
- **单独失效**：PRM-Lite 单独（0.140 整体，0.365 错误）— 信号被淹没。LATA 单独（0.185 整体）— 无信号源。**只有联合方案解锁 0.240**。

> 此分解是**模型无关的**，适用于 τ-bench 之外的任何长形式 RL 任务。

### 2. PRM-Lite v4-最优（可解释的过程奖励）
完全可解释、零可训练参数的过程奖励模型：
- **P1–P8 惩罚**：占位符（−0.05）、冗余（−0.03）、错误重复（−0.04）、无推理（−0.05）
- **B1–B7 奖励**：恢复（+0.05）、数据链（+0.08）、读取多样性（+0.01）、思考奖励（条件触发）
- **反黑客防御**：条件思考评分、基于 schema 的实体提取、n_tools > 8 时的长度惩罚

### 3. 内存高效的训练系统（工程）
- **Bypass 模式 + 融合内核 + TP=2** 将每步内存峰值从 **OOM 降至 73.2 GB**，在 **2×A800** 上实现 7B 策略 + 72B-AWQ 模拟器。
- **离线优先**：所有脚本注入 `HF_HUB_OFFLINE=1`，适用于隔离的 HPC 集群。
- **Render-Twice-Diff SFT**：一种模板无关的多轮工具调用损失掩码方法，避免 off-by-one token 错误。

---

## 📊 详细结果

### 逐步评测（N=4 样本/任务，max_tokens=4096）

| 实验 | Step | 整体 | 泛化 pass^1 | 错误率 | per_turn p50 | 备注 |
|-----------|------|---------|-------------|------------|--------------|-------|
| Vanilla | 200 | 0.175 | 0.071 | 0.200 | 72 | 崩溃基线 |
| Turn-Discount | 250 | 0.125 | 0.052 | 0.345 | 245 | 被动保护 |
| PRM-Lite | 250 | 0.140 | 0.059 | 0.365 | 169 | 信号阻塞 |
| LATA | 250 | 0.185 | 0.088 | 0.290 | 183 | √L 增益 |
| **联合方案** | **250** | **0.240** | **0.110** | **0.140** | **313** | **最优检查点** |

### 假设验证

| 假设 | 状态 | 证据 |
|-----------|--------|----------|
| H1: Turn-Discount 阻止推理崩溃 | ✅ 已验证 | 回复长度 −23%，无断崖 |
| H2: 联合方案打破群组饱和 | ✅ 已验证 | 300 步内 score/min 从未 0/1 |
| H3: 联合方案改善 OOD 泛化 | ✅ 已验证 | unseen 为正（0.15–0.175） |
| H4: LATA 优于 Turn-Discount | ✅ 已验证 | +0.060 整体，−0.055 错误 |
| H5: 联合方案 > max(单组件) | ✅ 已验证 | 0.240 > 0.185 > 0.140 > 0.125 |

---

## 🏗️ 项目结构

```raw
agentic-grpo-longhorizon/
├── configs/                    # 所有实验的 Hydra YAML 配置
│   ├── turn_discount.yaml
│   ├── prm_lite.yaml
│   ├── lata.yaml
│   ├── prm_lite_lata.yaml
│   └── eval/                   # 各实验评测配置
├── src/
│   ├── envs/                   # τ-bench 包装器与工具配置
│   │   ├── tau_bench_wrapper.py
│   │   ├── tau_bench_interaction.py   # PRM-Lite 规则引擎
│   │   └── tau_bench_tools.py
│   ├── evaluation/
│   │   └── pass_k_eval.py      # 独立 pass@k 评测器
│   ├── models/
│   │   └── vllm_policy.py      # vLLM 策略包装器
│   └── training/
│       └── sft_dataset.py      # SFT 数据收集
├── scripts/
│   ├── train/grpo/             # GRPO 训练启动脚本
│   │   ├── run_exp1_turn_discount.sh
│   │   ├── run_exp2_lata.sh
│   │   ├── run_exp3_prm_lite.sh
│   │   ├── run_exp4_prm_lite_lata.sh
│   │   └── run_vanilla.sh
│   ├── eval/                   # 独立评测启动脚本
│   │   ├── eval_exp1_turn_discount.sh
│   │   ├── eval_exp2_lata.sh
│   │   ├── eval_exp3_prm_lite.sh
│   │   └── eval_exp4_prm_lite_lata.sh
│   ├── train/sft/              # SFT 预热脚本
│   └── vllm_server/            # vLLM 服务启动脚本
├── docs/
│   └── ablation/
│       ├── ablation_diagnosis_report.md   # 完整诊断报告（≈800 行）
│       ├── ablation_plan.md               # 实验设计手册
│       ├── ablation_comparison.png
│       └── ablation_progression.png
├── experiments/                # 检查点、HF 导出、评测输出
├── requirements.txt
└── setup.sh                    # 一键环境搭建
```

---

## 🚀 快速开始

### 1. 环境搭建
（实验过程中改了不少源码，以本仓库的verl框架和benchmark为标准）

```bash
# 一键搭建（conda + PyTorch 2.7 + CUDA 12.6 + 依赖）
bash setup.sh
conda activate agentrl
cd agentic-grpo-longhorizon

# 或手动安装：
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd ../tau-bench && pip install -e .
cd ../verl && pip install -e .
cd agentic-grpo-longhorizon
```

### 2. 训练模型

```bash
# 示例：联合方案（PRM-Lite + LATA）
cd scripts/train/grpo
bash run_exp4_prm_lite_lata.sh

# 或：Vanilla GRPO 基线
bash run_vanilla.sh
```

### 3. 独立评测

```bash
# 自动评测 step 200/250/300 检查点
cd scripts/eval
bash eval_exp4_prm_lite_lata.sh
```

> **硬件**：2×A800（80GB）。GPU 0 运行 7B 策略 vLLM；GPU 1 运行 72B-AWQ 用户模拟器 vLLM。  
> **离线模式**：所有脚本注入 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，适用于隔离环境。

---

## 📚 文档

| 📄 文档 | 📝 内容 |
|----------|---------|
| [`docs/ablation/ablation_diagnosis_report.md`](docs/ablation/ablation_diagnosis_report.md) | **主报告**：训练曲线、评测数据、机制分析、假设验证 |
| [`docs/ablation/ablation_plan.md`](docs/ablation/ablation_plan.md) | 实验设计手册：代码实现、PRM-Lite 规则集、黑客风险分析 |
| [`docs/vanilla_grpo/vanilla_grpo_diagnosis.md`](docs/vanilla_grpo/vanilla_grpo_diagnosis.md) | Vanilla GRPO 崩溃诊断：三个根本原因、五个检查点分析 |
| [`../agentic-grpo-longhorizon-blog.md`](../agentic-grpo-longhorizon-blog.md) | 🆕 技术博客：从训练崩溃到稳定收敛的完整复盘（PRM-Lite + LATA） |

---

## 🛠️ 技术栈

- **训练框架**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **策略模型**: Qwen2.5-7B-Instruct
- **用户模拟器**: Qwen2.5-72B-Instruct-AWQ
- **评测基准**: [τ-bench](https://github.com/sierra-research/tau-bench) airline (50 任务)
- **推理引擎**: vLLM V1 with tool-call parsing (Hermes)
- **注意力机制**: FlashAttention-2

---

## 🙏 致谢

- [veRL](https://github.com/volcengine/verl) 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) 挑战性长链路智能体评测基准
- [Qwen](https://github.com/QwenLM/Qwen) 系列模型提供的强大基座策略

---

> **为什么重要**：大多数 RLHF/RLAIF 工作聚焦单轮问答或代码生成。本项目 tackle 更难的问题 —— **多轮、多工具、部分可观测的对话智能体** —— 在这里 vanilla GRPO 会灾难性失败。PRM-Lite + LATA 联合设计提供了一条有原理支撑、轻量且可解释的通往稳定训练的路径，无需昂贵的学习式奖励模型。
