# Vanilla GRPO 工程优化记录

> **目标**：在 A800-80GB × 3 上完成 veRL + vLLM V1 + tau-bench airline 的多轮工具调用 GRPO 训练（500 steps）。
>
> **核心结论**：通过 TP=2 双 GPU Policy + `bypass_mode` + `use_fused_kernels` + 显存参数调优，将峰值控制在 73.2GB，稳定完成训练。

---

## 1. 硬件拓扑

| GPU | 用途 | 关键配置 |
|-----|------|---------|
| GPU 0 + GPU 1 | Policy（FSDP actor/ref + vLLM RolloutEngine） | `TP=2`, `CUDA_VISIBLE_DEVICES=0,1` |
| GPU 2 | 72B AWQ User Simulator（OpenAI API, port 8001） | `max_num_seqs=8` |

---

## 2. 核心工程优化

### 2.1 TP=2 双 GPU Policy

单卡 80GB 在长序列多轮场景下无法同时容纳 vLLM 池子 + FSDP all_gather + logits 峰值。启用 TP=2 将 Policy 分布到 GPU0+GPU1：

```yaml
actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 2
```

### 2.2 bypass_mode + calculate_log_probs（OOM 最终解法）

**问题**：FSDP `compute_log_prob` 创建 full logits `[micro_batch, seq_len, vocab_size]`，bf16 下 13.9GB，叠加 PyTorch softmax 输出 pd 同尺寸，峰值 27.8GB，单卡必爆。

**解法**：
1. vLLM 在 rollout 生成时计算 `log_probs`（采样需要，零成本）：
   actor_rollout_ref:
     rollout:
       calculate_log_probs: true
2. `bypass_mode` 直接用 rollout_log_probs 作为 old_log_probs，**完全跳过** FSDP actor 的 `compute_log_prob`：
   algorithm:
     rollout_correction:
       bypass_mode: true

**收益**：actor `compute_log_prob` 的 ~20-40GB 峰值降为 **0GB**。

**质量影响**：GRPO 同一步内 rollout→update，`π_rollout` 与 `π_old` 差异极小，clip 变为 `π_θ/π_rollout`，效果与标准 GRPO 几乎无差别。

### 2.3 use_fused_kernels（消除 logits 峰值）

`bypass` 只跳过 actor old_log_prob，**ref log_prob 和 actor update 仍需要 forward**。`use_fused_kernels=true` 调用 flash-attn 的 `cross_entropy_loss`，在 CUDA kernel 内直接计算 log_prob/entropy，**不创建完整 logits 张量**。

```yaml
actor_rollout_ref:
  model:
    use_fused_kernels: true
    fused_kernel_options:
      impl_backend: torch
  actor:
    use_fused_kernels: true
  ref:
    use_fused_kernels: true
```

**收益**：ref/actor update 的 logits+pd 峰值 **27.8GB → ~2GB**。

> **Qwen2.5 兼容性**：veRL 通过 `monkey_patch.py` 将 `forward_with_torch_backend` patch 到 `Qwen2ForCausalLM`，已验证支持。

### 2.4 数据集大小修正

`train.parquet` 仅 40 行（seen tasks），默认 `total_epochs=1` + `train_batch_size=4` = **10 steps 就结束**，不是 OOM/crash。注意修改epoch，grpo中steps和epoch是先到哪个阈值就结束。

```yaml
trainer:
  total_epochs: 50   # 40 × 50 / 4 = 500 steps
```

### 2.5 显存参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `gpu_memory_utilization` | **0.50** | vLLM 池子 40GB，PyTorch 可用 40GB |
| `max_num_seqs` | **12** | TP=2 下每卡分摊，KV cache 约 5-8GB |
| `ppo_micro_batch_size_per_gpu` | **2** | actor update 平衡速度与峰值 |
| `ref.log_prob_micro_batch_size_per_gpu` | **4** | fused kernel 下安全 |

---

## 3. 已应用的关键 Patch

| 文件 | 修改内容 |
|------|----------|
| `verl/trainer/ppo/ray_trainer.py` | bypass fallback：batch 有 `rollout_log_probs` 时强制跳过 `compute_log_prob` |
| `verl/trainer/ppo/rollout_corr_helper.py` | `open_dict(policy_loss_config)` 绕过 OmegaConf struct 限制 |
| `verl/workers/fsdp_workers.py` | `compute_ref_log_prob 返回 meta_info={"temperature": ...}（修复 _is_lora=True 路径 temperature 丢失）` |
| `verl/experimental/agent_loop/tool_parser.py` | `_repair_json()` 修复截断 JSON |
| `tau_bench/envs/airline/tools/send_certificate.py` | 用户已有最大证书数时返回字符串 |
| `verl/utils/tracking.py` | Swanlab resume 支持 `SWANLAB_RUN_ID` / `SWANLAB_RESUME` |

---

## 4. 稳定配置

```yaml
hydra:
  searchpath:
    - pkg://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: experiments/week3_vanilla/train.parquet
  val_files: experiments/week3_vanilla/val.parquet
  train_batch_size: 4
  max_prompt_length: 8192
  max_response_length: 12288
  return_raw_chat: True
  tool_config_path: configs/tool_config/tau_bench_airline_tools.yaml

actor_rollout_ref:
  model:
    path: experiments/week2_sft_lora_merged
    enable_gradient_checkpointing: true
    lora_rank: 16
    lora_alpha: 32
    use_fused_kernels: true
    fused_kernel_options:
      impl_backend: torch
    override_config:
      attn_implementation: flash_attention_2
  actor:
    fsdp_config:
      optimizer_offload: true
      param_offload: true
      model_dtype: bf16
    optim:
      lr: 5.0e-6
    use_kl_loss: True
    kl_loss_coef: 0.01
    kl_loss_type: low_var_kl
    ppo_mini_batch_size: 4
    ppo_micro_batch_size_per_gpu: 2
    use_fused_kernels: true
  rollout:
    name: "vllm"
    mode: async
    agent:
      default_agent_loop: tool_agent
    multi_turn:
      enable: True
      format: "hermes"
      max_user_turns: 15
      max_assistant_turns: 15
      tool_config_path: configs/tool_config/tau_bench_airline_tools.yaml
      interaction_config_path: configs/interaction_config/tau_bench_airline.yaml
    n: 8
    temperature: 0.7
    top_p: 0.9
    tensor_model_parallel_size: 2
    calculate_log_probs: true
    gpu_memory_utilization: 0.50
    free_cache_engine: true
    max_num_seqs: 12
    max_model_len: 24576
    max_num_batched_tokens: 24576
    log_prob_micro_batch_size_per_gpu: 2
  ref:
    fsdp_config:
      optimizer_offload: true
      param_offload: true
      model_dtype: bf16
    model:
      path: experiments/week2_sft_lora_merged
      override_config:
        attn_implementation: flash_attention_2
    log_prob_micro_batch_size_per_gpu: 4
    use_fused_kernels: true

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.01
  rollout_correction:
    bypass_mode: true

trainer:
  total_epochs: 50
  total_training_steps: 500
  save_freq: 50
  test_freq: 100
  default_local_dir: experiments/week3_vanilla/checkpoints
  logger: ['console', 'swanlab']
  project_name: agentic-grpo-longhorizon
  experiment_name: week3_vanilla_grpo_500step
  n_gpus_per_node: 2
  nnodes: 1
```

---

## 5. 性能基准

### 5.1 显存占用（Step 1-5 稳定）

| 指标 | 数值 | 余量 |
|------|------|------|
| Peak Memory Allocated | **73.2 GB** | ~6.8 GB |
| Peak Memory Reserved | **73.4 GB** | ~6.6 GB |

**峰值构成**：
- vLLM 池子：40GB
- FSDP all_gather：14GB
- 激活值（seq≈20000）：~12GB
- fused kernel 临时：~2GB
- 其他碎片：~5GB

### 5.2 单步耗时（前 5 步平均）

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Rollout (gen) | ~720s | **~86%** |
| Ref log_prob | ~27s | 3% |
| Actor update | ~93s | 11% |
| **单步总计** | **~840s** | — |

> 500 步预估：~117 小时（~4.9 天）。Rollout 是最大瓶颈，受 72B usersim 延迟和多轮对话（平均 27 turns）主导。

---

## 6. 快速启动

```bash
# 1. 启动 72B User Simulator（GPU 2）
bash scripts/01_start_vllm_server_72b.sh

# 2. 启动训练（GPU 0+1）
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
bash scripts/20_run_week3_vanilla.sh

# 3. SwanLab resume（如需接续）
export SWANLAB_RUN_ID="xxx"
export SWANLAB_RESUME="true"
```

---

## 7. Checklist

- [ ] `CUDA_VISIBLE_DEVICES=0,1`
- [ ] `tensor_model_parallel_size: 2`
- [ ] `calculate_log_probs: true`
- [ ] `bypass_mode: true`
- [ ] `use_fused_kernels: true`（model + actor + ref）
- [ ] `total_epochs: 50`
- [ ] `gpu_memory_utilization: 0.50`
- [ ] `max_num_seqs: 12`
- [ ] `log_prob_micro_batch_size_per_gpu: 4`（ref）/ `2`（actor ppo）
- [ ] 72B User Simulator 正常（`curl http://localhost:8001/v1/models`）
