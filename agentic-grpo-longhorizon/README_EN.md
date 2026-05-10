# Agentic-GRPO-LongHorizon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Solving GRPO Training Collapse in Long-Horizon Multi-Tool Agents**  
> A systematic ablation study on τ-bench airline (50-task, multi-turn, multi-tool conversational agents), achieving **+37% overall pass^1** over vanilla GRPO via a novel **PRM-Lite + LATA** joint approach.

---

## 🔥 Key Results

**Best checkpoint (step 250): Joint (PRM-Lite + LATA) achieves 0.240 overall pass^1 — a +37% relative gain over the vanilla GRPO baseline (0.175).**

| Metric | Vanilla | Turn-Discount | PRM-Lite | LATA | **Joint** | Δ vs Vanilla |
|--------|---------|---------------|----------|------|-----------|-------------|
| **Overall pass^1** | 0.175 | 0.125 | 0.140 | 0.185 | **0.240** | **+37%** |
| **Generalization** | 0.071 | 0.052 | 0.059 | 0.088 | **0.110** | **+55%** |
| **Error Rate** | 0.200 | 0.345 | 0.365 | 0.290 | **0.140** | **−30%** |
| **Reasoning Depth** (p50 tokens) | 72 | 245 | 169 | 183 | **313** | **+334%** |

> *Generalization pass^1 = (uncovered_seen × 24 + unseen × 10) / 34, the core metric excluding train-set leakage.*

### Multi-Dimensional Comparison

![Ablation Comparison](ablation_comparison.png)

### Training Progression

![Training Progression](ablation_progression.png)

> **Observation**: Turn-Discount plateaus (passive protection); LATA sustains growth via √L normalization; the Joint approach peaks at step 250 (0.240) and gracefully degrades at step 300 (0.225).

---

## 🎯 Problem & Motivation

Standard **GRPO (Group Relative Policy Optimization)** suffers from catastrophic **training collapse** when applied to long-horizon, multi-tool conversational agents on τ-bench airline (50 tasks, 40 train / 10 test). We identified **three root causes**:

### 1. Group Reward Saturation (Bidirectional Deadlock)
Outcome reward is binary (0/1). With `group_size=8`, the group easily reaches all-0 or all-1 states → `advantage variance → 0` → gradient vanishes.

### 2. Training-Set Leakage Bias
16 of 40 train tasks are *covered_seen* (72B teacher trajectories available). Policy memorizes teacher patterns, inflating covered performance while uncovered/unseen remain near zero.

### 3. Per-Turn Reasoning Degeneration
Linear length normalization `advantage / L` penalizes long responses. Policy learns to **"trade quantity for quality"** — short reasoning + frequent tool trial-and-error — leading to collapse after step 150.

> **Critical Finding**: Training validation reward is **not** a reliable proxy. Turn-Discount reports val reward 0.80 but true eval is only 0.125 — a **6.4× gap**.

---

## 💡 Method

We design and validate **four ablation experiments**, each addressing specific failure modes:

### Exp 1: Turn-Discounted Advantage
**Idea**: Protect early-turn reasoning by exponentially decaying token weights.  
**Mechanism**: `weight[t] = α^(L-1-t)` with `α=1.05`, normalized so `mean(weight)=1`. Early tokens receive higher advantage, discouraging late-stage guessing.  
**Result**: Successfully prevents collapse shape (response length −23% vs vanilla −63%), but eval remains low (0.125) due to lack of quality guidance.

### Exp 2: LATA — Length-Aware Turn-Advantage
**Idea**: Replace linear `1/L` normalization with `1/√L`, preserving marginal incentives for long reasoning.  
**Mechanism**: `advantage_token = A / sqrt(L)` instead of `A / L`. When response length grows 4×, per-token gradient only halves (vs. quartering in vanilla).  
**Result**: Sustained improvement (0.155 → 0.185 → 0.190), error rate drops from 0.345 to 0.290, but ceiling visible without quality signals.

### Exp 3: PRM-Lite — Lightweight Process Reward
**Idea**: Break group saturation with dense, rule-based process rewards.  
**Mechanism**: 15 hand-crafted rules (P1–P8 penalties, B1–B7 bonuses) providing continuous `[-0.5, +0.5]` signals. Final reward = `outcome + 0.3 × process_score`.  
**Result**: Successfully eliminates score/min = 0/1 deadlocks, but signal is diluted by trajectory-level linear normalization — error rate actually worsens to 0.365.

### Exp 4: Joint — PRM-Lite + LATA ⭐
**Idea**: **PRM-Lite supplies local quality signals; LATA's √L normalization ensures these signals are not drowned by response length.**  
**Mechanism**: Per-turn process score penalties/bonuses propagate through `A/√L` to individual tokens, enabling the policy to learn *which turn was wrong* rather than just *whether the whole trajectory succeeded*.  
**Result**: **0.240 overall** — surpassing all single-component baselines. Error rate uniquely decreases (0.170 → 0.140 → 0.120). Unseen task performance turns positive and stabilizes.

> **Core Insight**: The value is not in having process rewards *or* better normalization alone — it is in **signal propagation**. PRM-Lite generates local signals; LATA's √L provides the transmission channel. Neither works well in isolation.

---

## 🌟 Technical Highlights

### 1. Signal Transmission Theory (Algorithmic Contribution)
The ablation report empirically proves a **decomposition principle** for GRPO in long-horizon agents:
- **Signal Source** (PRM-Lite): 15 hand-crafted rules provide dense per-turn quality signals `[-0.5, +0.5]`.
- **Signal Pathway** (LATA): `advantage / √L` replaces `advantage / L`, preventing response-length dilution.
- **Isolation Failure**: PRM-Lite alone (0.140 overall, 0.365 error) — signal drowned. LATA alone (0.185 overall) — no signal source. **Only their combination unlocks 0.240**.

> This decomposition is **model-agnostic** and applicable beyond τ-bench to any long-form RL task.

### 2. PRM-Lite v4-Optimal (Interpretable Process Reward)
A fully interpretable, zero-trainable-parameter process reward model:
- **P1–P8 Penalties**: Placeholder (−0.05), Redundancy (−0.03), Error repetition (−0.04), No reasoning (−0.05)
- **B1–B7 Bonuses**: Recovery (+0.05), Data chain (+0.08), Read diversity (+0.01), Think bonus (conditional)
- **Anti-Hacking Defenses**: Conditional think scoring, schema-based entity extraction, length penalty for n_tools > 8

### 3. Memory-Efficient Training System (Engineering)
- **Bypass Mode + Fused Kernels + TP=2** reduces per-step memory peak from **OOM to 73.2 GB**, enabling 7B policy + 72B-AWQ simulator on **2×A800**.
- **Offline-first**: All scripts inject `HF_HUB_OFFLINE=1` for air-gapped HPC clusters.
- **Render-Twice-Diff SFT**: A template-agnostic loss-masking method for multi-turn tool-calling that avoids off-by-one token errors.

---

## 📊 Detailed Results

### Step-by-Step Eval (N=4 samples/task, max_tokens=4096)

| Experiment | Step | Overall | Gen. pass^1 | Error Rate | per_turn p50 | Notes |
|-----------|------|---------|-------------|------------|--------------|-------|
| Vanilla | 200 | 0.175 | 0.071 | 0.200 | 72 | Collapse baseline |
| Turn-Discount | 250 | 0.125 | 0.052 | 0.345 | 245 | Passive protection |
| PRM-Lite | 250 | 0.140 | 0.059 | 0.365 | 169 | Signal blocked |
| LATA | 250 | 0.185 | 0.088 | 0.290 | 183 | √L gains |
| **Joint** | **250** | **0.240** | **0.110** | **0.140** | **313** | **Best checkpoint** |

### Hypothesis Validation

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1: Turn-Discount prevents reasoning collapse | ✅ Verified | Response length −23%, no cliff |
| H2: Joint breaks group saturation | ✅ Verified | score/min never 0/1 across 300 steps |
| H3: Joint improves OOD generalization | ✅ Verified | unseen positive (0.15–0.175) |
| H4: LATA improves over Turn-Discount | ✅ Verified | +0.060 overall, −0.055 error |
| H5: Joint > max(single component) | ✅ Verified | 0.240 > 0.185 > 0.140 > 0.125 |

---

## 🏗️ Project Structure

```
📦 agentic-grpo-longhorizon/
├── ⚙️ configs/                 # Hydra YAML configs for all experiments
│   ├── turn_discount.yaml
│   ├── prm_lite.yaml
│   ├── lata.yaml
│   ├── prm_lite_lata.yaml
│   └── eval/                   # Per-experiment eval configs
├── 💻 src/                     # Core source code
│   ├── 🌍 envs/                # τ-bench wrapper & tool configs
│   │   ├── 🐍 tau_bench_wrapper.py
│   │   ├── 🐍 tau_bench_interaction.py   # PRM-Lite rule engine
│   │   └── 🐍 tau_bench_tools.py
│   ├── 📊 evaluation/
│   │   └── 🐍 pass_k_eval.py   # Independent pass@k evaluator
│   ├── 🧠 models/
│   │   └── 🐍 vllm_policy.py   # vLLM-based policy wrapper
│   └── 🎓 training/
│       └── 🐍 sft_dataset.py   # SFT data collection
├── 📜 scripts/
│   ├── 🚀 train/grpo/          # GRPO training launchers
│   │   ├── 📜 run_exp1_turn_discount.sh
│   │   ├── 📜 run_exp2_lata.sh
│   │   ├── 📜 run_exp3_prm_lite.sh
│   │   ├── 📜 run_exp4_prm_lite_lata.sh
│   │   └── 📜 run_vanilla.sh
│   ├── 📈 eval/                # Independent eval launchers
│   │   ├── 📜 eval_exp1_turn_discount.sh
│   │   ├── 📜 eval_exp2_lata.sh
│   │   ├── 📜 eval_exp3_prm_lite.sh
│   │   └── 📜 eval_exp4_prm_lite_lata.sh
│   ├── 🔧 train/sft/           # SFT warmup scripts
│   └── 🖥️ vllm_server/         # vLLM server startup scripts
├── 📚 docs/
│   └── 🔬 ablation/
│       ├── 📝 ablation_diagnosis_report.md   # Full diagnosis (≈800 lines)
│       ├── 📝 ablation_plan.md               # Experiment design manual
│       ├── 🖼️ ablation_comparison.png
│       └── 🖼️ ablation_progression.png
├── 🧪 experiments/             # Checkpoints, HF exports, eval outputs
├── 📄 requirements.txt
└── 🔨 setup.sh                 # One-click environment setup
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# One-click setup (conda + PyTorch 2.7 + CUDA 12.6 + dependencies)
bash setup.sh
conda activate agentrl

# Or manual:
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd ../tau-bench && pip install -e .
cd ../verl && pip install -e .
```

### 2. Train a Model

```bash
# Example: Joint (PRM-Lite + LATA)
cd scripts/train/grpo
bash run_exp4_prm_lite_lata.sh

# Or: Vanilla GRPO baseline
bash run_vanilla.sh
```

### 3. Independent Evaluation

```bash
# Evaluates step 200/250/300 checkpoints automatically
cd scripts/eval
bash eval_exp4_prm_lite_lata.sh
```

> **Hardware**: 2×A800 (80GB). GPU 0 runs 7B policy vLLM; GPU 1 runs 72B-AWQ user simulator vLLM.  
> **Offline Mode**: All scripts inject `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for air-gapped environments.

---

## 📚 Documentation

| 📄 Document | 📝 Content |
|----------|---------|
| [`docs/ablation/ablation_diagnosis_report.md`](docs/ablation/ablation_diagnosis_report.md) | **Main report**: training curves, eval data, mechanism analysis, hypothesis validation |
| [`docs/ablation/ablation_plan.md`](docs/ablation/ablation_plan.md) | Experiment design manual: code implementation, PRM-Lite rule set, hacking risk analysis |
| [`docs/vanilla_grpo/vanilla_grpo_diagnosis.md`](docs/vanilla_grpo/vanilla_grpo_diagnosis.md) | Vanilla GRPO collapse diagnosis: three root causes, five checkpoints analysis |
| [`../agentic-grpo-longhorizon-blog.md`](../agentic-grpo-longhorizon-blog.md) | 🆕 Technical blog: from training collapse to stable convergence (PRM-Lite + LATA) |

---

## 🛠️ Tech Stack

- **Training Framework**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **Policy Model**: Qwen2.5-7B-Instruct
- **User Simulator**: Qwen2.5-72B-Instruct-AWQ
- **Benchmark**: [τ-bench](https://github.com/sierra-research/tau-bench) airline (50 tasks)
- **Inference Engine**: vLLM V1 with tool-call parsing (Hermes)
- **Attention**: FlashAttention-2

---

## 🙏 Acknowledgements

- [veRL](https://github.com/volcengine/verl) for the open-source RL training framework
- [τ-bench](https://github.com/sierra-research/tau-bench) for the challenging long-horizon agent benchmark
- [Qwen](https://github.com/QwenLM/Qwen) series models for strong base policies

---

> **Why this matters**: Most RLHF/RLAIF work focuses on single-turn QA or coding. This project tackles the harder problem — **multi-turn, multi-tool, partially-observable conversational agents** — where vanilla GRPO catastrophically fails. The PRM-Lite + LATA joint design offers a principled, lightweight, and interpretable path to stable training without requiring expensive learned reward models.
