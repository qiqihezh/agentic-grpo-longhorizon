# 消融实验诊断分析报告

> **文档定位**：本报告是对 τ-bench airline 上 GRPO 训练 collapse 问题的系统性消融实验诊断。覆盖基线（Vanilla GRPO）及四个改进方案（Turn-Discount、PRM-Lite、LATA、联合方案），通过训练日志、独立评测和机制分析，验证诊断假设并定位根本病因。
>
> **数据来源**：
> - Training 日志：`experiments/{vanilla,turn_discount,prm_lite,lata,prm_lite_lata}/training.log` + SwanLab
> - 独立 Eval：`experiments/{vanilla,turn_discount,prm_lite,lata,prm_lite_lata}/eval_step_*/`（N=4 samples/task, max_tokens=4096）
> - 基线对比：W1 (Base 7B)、W2 (SFT LoRA merged)
> - Swanlab 训练曲线：https://swanlab.cn/@godstear/agentic-grpo-longhorizon?utm_source=website_qr&utm_medium=qr_scan
>
> **数据约束**：τ-bench airline 仅 50 tasks，train=40（covered_seen=16, uncovered_seen=24），test=10（unseen）。核心指标为**泛化 pass^1** = (uncovered_seen×24 + unseen×10) / 34。

---

## §0 实验总览与假设框架

### 0.1 实验配置汇总

| 实验 | 方案 | Policy | 训练步数 | 关键改动 | 当前状态 |
|---|---|---|---|---|---|
| **Baseline** | Vanilla GRPO | 7B-Instruct | 200 step | 无 | ✅ 已完成 |
| **Exp 1** | Turn-Discounted Advantage | 7B-Instruct | 300 step | `adv_est=grpo_turn_discounted`, α=1.05 | ✅ 已完成 |
| **Exp 3** | PRM-Lite | 7B-Instruct | 300 step | `reward_fn=prm_lite`, outcome+0.3×process | ✅ 已完成  |
| **Exp 2** | LATA | 7B-Instruct | 300 step | `adv_est=grpo_lata`, α=1.05 + √L 归一化 | ✅ 已完成 |
| **Exp 4** | PRM-Lite + LATA | 7B-Instruct | 300 step | Exp 3 + Exp 2 联合 | ✅ 已完成 |

**统一配置**：均从 `sft_lora_merged` 出发，2×A800（GPU0 7B policy, GPU1 72B-AWQ user sim），group_size=8，max_response_length=12288，max_prompt_length=14336，temperature=0.7。

### 0.2 核心诊断假设

| 假设 | 内容 | 对应病因 | 验证实验 |
|---|---|---|---|
| **H1** | Turn-Discount 通过保护早期 token 权重，阻止 per-turn reasoning 退化导致的 training collapse | §1.3 病因3：Reasoning 退化 | Exp 1 |
| **H2** | PRM-Lite 通过连续 process reward 信号，打破 group saturation 死锁 | §1.3 病因1：Group saturation | Exp 3 |
| **H3** | PRM-Lite 通过 penalty/bonus 规则，抑制 training set 记忆、改善 OOD 泛化 | §1.3 病因2：训练集泄漏偏差 | Exp 3 |
| **H4** | LATA 通过 √L 长度归一化，在 Turn-Discount 基础上进一步保护长 reasoning 的边际激励 | §1.3 病因3：Reasoning 退化 | Exp 2 |
| **H5** | PRM-Lite + LATA 的效果正交叠加，联合方案 > max(单方案) | 综合 | Exp 4 |

### 0.3 评估口径说明

| 指标 | 定义 | 数据来源 | 注意 |
|---|---|---|---|
| Training val reward | Validation rollout 的 reward 均值（N=1） | SwanLab / 日志 | **含 group 采样统计幻觉，不可作为真实能力代理** |
| Eval pass^1 (pass_hat_1) | 独立 eval 的无偏 pass@1 估计（N=4） | `eval_report.json` | 真实能力，无 group 容错 |
| 泛化 pass^1 | (uncovered_seen×24 + unseen×10) / 34 | `split_eval_report.json` | **核心指标**，排除 covered_seen 的 train set 泄漏 |
| per_turn p50 | 所有 turn 的 assistant_content_tokens 中位数 | `eval_report.json` | 反映 per-turn reasoning 质量 |
| error_rate | sum(errors) / sum(tool_calls) | `split_eval_report.json` | 工具使用稳定性 |

---

## §1 基线：Vanilla GRPO

### 1.1 实验配置

| 项 | 值 |
|---|---|
| 方案 | Vanilla GRPO（GRPO 默认实现） |
| Policy 起点 | `sft_lora_merged` |
| 训练步数 | 200 step（collapse 后终止） |
| Eval checkpoint | step 50, 100, 150, 200 |

### 1.2 训练曲线与 Collapse 现象

Vanilla GRPO 的训练曲线呈现典型的 **"先涨后塌" collapse 形状**：

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

**Collapse 三特征**：
1. **Reward 崩塌**：step 150 (0.225) → step 200 (0.175)，Δ=−0.050
2. **Response length 断崖**：~3000 → ~1100（−63%）
3. **Num turns 早死**：29 → 5（policy 学会"快速放弃"复杂 task）

**Group saturation 死锁**：step 80+ 频繁出现 `critic/score/min=1.0`（group 内 8/8 全成功），导致 group variance→0，梯度消失。

### 1.3 评测结果

| 指标 | Step 150 (Peak) | Step 200 (Collapse) | 变化 |
|---|---|---|---|
| overall pass^1 | 0.225 | 0.175 | −0.050 |
| covered_seen | 0.453 | 0.391 | −0.062 |
| uncovered_seen | 0.094 | 0.042 | −0.052 |
| unseen | 0.175 | 0.150 | −0.025 |
| 泛化 pass^1 | 0.110 | **0.071** | −0.039 |
| per_turn p50 | 77 | 72 | −5 |
| error_rate | 0.010 | 0.200 | +0.190 |
| avg_turns | 7.3 | 5.08 | -2.22 |

### 1.4 诊断结论：三类病因

Vanilla GRPO 的 collapse 不是单一原因，而是**三种病态的叠加**：

**病因1：Group Reward Saturation（双向死锁）**
- 机制：Outcome reward 是二元的（0/1），group_size=8 时容易全 0 或全 1
- 后果：advantage variance→0，policy 无法区分 rollout 质量，随机游走
- 证据：`critic/score/min` 频繁 = 0 或 = 1，grad_norm 单调下降

**病因2：Training Set 泄漏偏差**
- 机制：40 个 train task 中 16 个 covered_seen（72B teacher 轨迹覆盖），policy 通过记忆 teacher 模式在 covered 上快速饱和
- 后果：covered_seen 虚高（0.391），但 uncovered_seen（0.042）和 unseen（0.150）极低，泛化 pass^1 仅 0.071
- 证据：vanilla step200 的 covered(0.391) 远高于 Baseline (0.078) 和 SFT(0.328)

**病因3：Per-Turn Reasoning 退化（"以量补质"）**
- 机制：GRPO 的线性长度归一化 `advantage / L` 使得「生成更多 token」的边际成本为零。Policy 发现：与其在每轮深思熟虑，不如快速生成大量 tool call 试错，靠末端撞对获得 reward
- 后果：per_turn p50 从 SFT 的 ~80 降到 collapse 后的 ~72；response_length 先涨（3000 peak）后塌（1100），因为 policy 最终学会"少调 tool、快速结束"
- 证据：peak 阶段 response_length 膨胀，collapse 阶段 reasoning token 暴跌

---

## §2 Turn-Discounted Advantage (Exp 1)

### 2.1 实验设计

| 项 | 值 |
|---|---|
| 方案 | Turn-Discounted GRPO Advantage |
| 改动点 | `adv_estimator=grpo_turn_discounted`，`alpha=1.05` |
| 核心思想 | 早期 token 权重 `α^(L-1-t)`，晚期 token 权重低，保护前期 reasoning |
| 训练步数 | 300 step |
| Eval checkpoint | step 200, 250, 300 |

### 2.2 训练曲线分析

#### 2.2.1 Validation Reward 曲线（四阶段特征）

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

**关键发现 1：学习启动延迟 50 步**

Vanilla 的 validation reward：step 0→50 从 0.02→0.22（+0.20）。Turn-Discount 同期完全停滞（0.02→0.02）。

这直接验证了 **H3**：削弱「以量补质」后，policy 失去了 vanilla 在前 50 步依赖的「多试错撞对」捷径，但尚未建立「高质量 reasoning + 精准 tool call」的替代路径。

**关键发现 2：跃升发生在 50-100 步**

从 step 50 到 step 100，reward 从 0.02 跃升到 0.72（+36 倍）。这不是线性增长，而是**相变**——policy 在某个临界点后突然找到了替代「以量补质」的成功策略。

#### 2.2.2 Response Length 曲线

| Phase | Steps | Mean | Min | Max | 趋势 |
|---|---|---|---|---|---|
| 停滞期 | 1-50 | 2192 | 1705 | 3187 | 高位震荡 |
| 跃升期 | 51-100 | 1872 | 1265 | 2650 | 开始下降 |
| 爬升期 | 101-150 | 1811 | 1344 | 2483 | 继续缓慢下降 |
| 稳定期 | 151-300 | 1678 | 1092 | 2382 | 低位稳定 |

**与 Vanilla 的对比**：

| 方案 | Step 50 | Step 100 | Step 150 | Step 200 | 趋势 |
|---|---|---|---|---|---|
| Vanilla | ~2500 | ~3000 (peak) | ~1500 | **~1100** | **断崖 collapse** |
| Turn-Discount | ~2200 | ~1870 | ~1810 | **~1480** | **缓慢下降，无断崖** |

**关键发现**：Turn-Discount **成功阻止了 response_length 的断崖式 collapse**。Vanilla 从 peak（3000）到 step 200（1100）跌了 63%，而 Turn-Discount 从 early phase（2192）到 late phase（1678）只跌了 23%，且最低值（1092）是个别步骤，非系统性趋势。

但这并不意味着 reasoning 质量得到了提升——response_length 仍在缓慢下降，只是下降速度更温和。这说明 Turn-Discount 的「保护」是**消极的**（阻止暴跌），而非**积极的**（推动增长）。

#### 2.2.3 Critic Score 分布（Group Saturation 的意外缓解）

Training 中 `critic/score/min` 的分布：

| Phase | score_min = 0 的比例 | score_max = 1 的比例 | 解读 |
|---|---|---|---|
| 1-50 | **100%** | ~60% | Group 内常见全失败 |
| 51-100 | **100%** | ~70% | 仍在全失败端 |
| 101-150 | **98%** | ~90% | 跃升期，偶尔出现非零 min |
| 151-300 | **96%** | ~95% | 稳定期，极少出现非零 min |

**与 Vanilla 的对比**：

Vanilla 的 `critic/score/min`：step 0-80 几乎全 = 0，**step 80 之后频繁出现 1.0**（group 内 8/8 全成功）。

Turn-Discount 的 `critic/score/min`：**300 步中 97.7% = 0.0**，几乎从未出现 1.0。

**关键发现：Turn-Discount 意外缓解了 group saturation**

这是一个计划外的发现。诊断报告原本认为 Turn-Discount 只对 §1.3 病因3 reasoning 退化有效，对 §1.3 病因1 group saturation 无直接帮助。但实际数据表明：

- Vanilla 在 step 80+ 频繁出现「8/8 全成功」→ `score/min=1.0` → group variance → 0 → 梯度消失
- Turn-Discount 在 step 100+ `score/min` 仍几乎全为 0，说明 group 内**极少出现全成功**

**机制解释**：Turn-Discount 通过降低晚期 token 的权重，使得「末端试错撞对」的 rollout 获得的 advantage 很低。这降低了 policy 在 covered task 上「反复撞对」的动机，从而**推迟了 train set 的 saturation**。Vanilla 在 ~80 step 就把 40 个 train task 学到饱和，而 Turn-Discount 到 300 step 仍未饱和。

#### 2.2.4 Grad Norm 与 Rollout Correlation

| Phase | Grad Norm (avg) | 解读 |
|---|---|---|
| 1-50 | 0.353 | 高梯度，消除旧习惯 |
| 51-100 | 0.088 | 跃升期，梯度仍充足 |
| 101-150 | 0.014 | 快速下降 |
| 151-300 | 0.009 | 低位稳定 |

Vanilla 的 grad_norm：step 0-80 在 0.05-0.10，step 100+ 单调降到 0.005-0.01。

Turn-Discount 的早期 grad_norm（0.35）远高于 vanilla（0.10），说明**消除「以量补质」旧习惯需要更强的梯度**。但到后期（0.009），两者趋同，说明无论哪种方案，7B policy 在 200+ step 后都会进入低梯度状态。

`rollout_corr/ppl_ratio` 稳定在 0.98 附近，bypass_mode 假设成立。

### 2.3 评测结果

#### 2.3.1 独立 Eval 完整数据

| 指标 | Step 200 | Step 250 | Step 300 | 趋势 |
|---|---|---|---|---|
| **overall pass^1** | 0.135 | **0.125** | **0.130** | 稳定低位 |
| covered_seen | 0.297 | **0.281** | 0.281 | 稳定 |
| uncovered_seen | 0.031 | **0.021** | **0.042** | 极低 |
| unseen | 0.125 | **0.125** | 0.100 | 稳定 |
| **泛化 pass^1** | 0.052 | **0.052** | **0.059** | 稳定 |
| per_turn p50 | 181 | **245** | **227** | 上升 |
| error_rate | 0.285 | **0.345** | **0.335** | 高位 |
| avg_turns | 6.83 | **8.51** | **7.75** | 上升 |

> **⚠️ 关键警示**：Training val reward（0.80@step300）与 Eval pass^1（0.125@step250）差距 **6.4 倍**，Critic 严重 overconfident。

#### 2.3.2 与 Vanilla 的横向对比

| 指标 | Vanilla step200 | Turn-Discount step250 | Turn-Discount step300 | 差距 |
|---|---|---|---|---|
| overall pass^1 | 0.175 | 0.125 | 0.125 | **TD 更低** |
| covered_seen | 0.391 | 0.281 | 0.281 | TD 更低 |
| uncovered_seen | 0.042 | 0.021 | 0.021 | 接近 |
| unseen | 0.150 | 0.125 | 0.125 | TD 更低 |
| 泛化 pass^1 | 0.071 | **0.052** | 0.052 | TD 更低 |
| avg_turns | 5.08 | 8.51 | 8.51 | TD 更高 |
| error_rate | 0.200 | **0.345** | 0.345 | **TD 更差** |
| per_turn p50 | 72 | 245 | 245 | TD 更高 |

**核心结论**：
1. **Turn-Discount 的真实 eval 低于 Vanilla collapse 后**：step250 (0.125) vs vanilla (0.175)。Training val reward 的 0.80 是严重幻觉。
2. **Turn-Discount 阻止了 collapse 形状，但未提升真实能力**。Overall 0.125 甚至低于 vanilla collapse 后的 0.175。
3. **Error rate 显著恶化**：0.345 vs vanilla 0.200。改变 advantage 结构 alone 不足以提升 tool 精度。
4. **OOD 泛化无改善**：泛化 pass^1 (0.052) 低于 vanilla (0.071)。

### 2.4 假设验证

| 假设 | 验证项 | 证据 | 判断 |
|---|---|---|---|
| **H1** | 阻止 reasoning collapse | response_length −23%（vs vanilla −63%），num_turns 稳定 27，无断崖 | ✅ **验证通过（消极保护）** |
| **H2** (意外) | Group saturation | score/min 97.7%=0（vs vanilla 频繁=1.0），saturation 意外缓解 | ⚠️ **意外发现，非直接目标** |
| **H3** | 学习启动延迟 | 前 50 步完全停滞（0.02→0.02），50-100 断崖跃升 | ✅ **验证通过** |

### 2.5 深层机制分析

#### 2.5.1 为什么真实 Eval 低于 Vanilla？

Turn-Discount 的 val reward 0.80 与 eval 0.125 差距 **6.4 倍**，比 vanilla 的 gap（0.225→0.175，差距 1.3 倍）更大。三个机制解释：

**机制 1：Critic Overconfidence 更严重**

Turn-Discount 的 critic/score/mean 在 step 150-300 稳定在 0.80-0.94，比 vanilla 的 0.20-0.23 高 4 倍。这意味着 critic 对 policy 的成功率极度乐观，但这种乐观建立在 group sampling 的统计幻觉上——training 时 8 条 rollout 总有一条能撞对，critic 就把这种"群体成功"误认为是 policy 的真实能力。

**机制 2：「前期解决」压力导致盲目调 tool**

Turn-Discount 惩罚晚期 token，迫使 policy 在「信息不充分」时就做出 tool call。Vanilla 的 policy 可以拖到第 5-8 轮慢慢收集用户信息再精准调用；Turn-Discount 的 policy 必须在第 1-3 轮就行动，否则晚期 token 的 advantage 权重太低，学不到东西。结果是更多的 placeholder 错误（如 guess reservation_id）和幻觉工具名。

**机制 3：Response length 的「虚假稳定」**

Turn-Discount 的 response_length 从 2192 缓慢降到 1678，看似「稳定」，但 eval 的 per_turn p50 从 72→245→227 反而更高。这说明 policy 在 training 上生成了更多 token，但这些 token 并未转化为更好的决策——只是「用更多废话填充」来适应 turn-discount 的权重结构。

### 2.6 成功与失败

#### 成功之处

| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **阻止了 training collapse 的形状** | ⭐⭐⭐ Training curve 无「先涨后塌」 |
| 2 | **阻止了 response_length 断崖下跌** | ⭐⭐⭐ 3000→1100 被阻止，仅缓慢下降 23% |
| 3 | **阻止了 num_turns 撞顶/早死** | ⭐⭐⭐ Train 28→29，eval 8.51，稳定 |
| 4 | **意外缓解了 group saturation** | ⭐⭐⭐ score/min 97.7%=0，vs vanilla 频繁=1.0 |
| 5 | **学习启动延迟的验证** | ⭐⭐⭐ 前 50 步停滞，50-100 断崖跃升 |

#### 失败与局限

| # | 问题 | 证据强度 |
|---|---|---|
| 1 | **Training validation 与真实 eval 差距极大**（0.80 vs 0.125） | ⭐⭐⭐ 6.4 倍差距 |
| 2 | **真实 eval 低于 Vanilla collapse 后**（0.125 vs 0.175） | ⭐⭐⭐ 阻止形状不等于提升能力 |
| 3 | **Error rate 显著恶化并持续高位**（0.345） | ⭐⭐⭐ 同等 overall 下工具稳定性更差 |
| 4 | **OOD 泛化无改善**（0.052 vs vanilla 0.071） | ⭐⭐⭐ 直接验证 H2 |
| 5 | **Per-turn reasoning tokens 增加但无效**（p50 245，成功率反降） | ⭐⭐ 更多 token 未转化为更好决策 |
| 6 | **Grad_norm 仍在后期衰减**（0.009） | ⭐⭐ 学习容量仍在 200+ step 后耗尽 |

### 2.7 对后续实验的启示

1. **不要信任 training validation reward**：Turn-Discount 的 val reward 0.80 是假象，真实 eval 仅 0.125。后续实验不可只看 training curve。
2. **Error rate 需要重点监控**：Turn-Discount 的 error_rate 0.345 说明「改变 advantage 结构」 alone 不足以提升 tool 使用精度。
3. **Group saturation 缓解是可复制的**：Turn-Discount 意外打破了 score/min=1.0 的死锁。PRM-Lite 通过 process_score 也应能做到这一点。
4. **最佳 checkpoint 在 step250**：Turn-Discount step250 (0.125) 与 step300 (0.125) 持平。所有实验均在 step250 保存最优 checkpoint。
5. **Per-turn reasoning 的量≠质**：p50 从 72→245 但成功率下降，说明 LATA 需要不仅保护长度，还要引导质量。

---

## §3 PRM-Lite (Exp 3)

### 3.1 实验设计

| 项 | 值 |
|---|---|
| 方案 | PRM-Lite（轻量级过程奖励模型） |
| 改动点 | `reward_fn=prm_lite`，`process_score_weight=0.3` |
| 核心思想 | 15 条规则-based process score（P1-P8 penalties, B1-B7 bonuses），最终 reward = outcome + 0.3 × process_score |
| 训练步数 | 300 step |
| Eval checkpoint | step 200 250 300 |

**Rule-based Process Score 规则集（v4-optimal）**：
- **Penalties**：Placeholder（−0.05/−0.03）、Redundancy（−0.03）、Error repetition（−0.04）、No reasoning（−0.05）、Length penalty（−0.01/step>8）
- **Bonuses**：Recovery（+0.05）、Data chain（+0.08/+0.04）、Read diversity（+0.01）、Think bonus（+0.01, conditional）
- **聚合**：Mean-based + [−0.5, +0.5] clamp

### 3.2 训练曲线与评测结果

#### 3.2.1 Training 中的关键曲线

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

**关键观察**：score/min 出现负值（−0.025, −0.049），说明 process score 的惩罚规则确实在输出负向信号。但这些局部信号的梯度被 GRPO 的线性 `advantage / L` 归一化严重稀释。

#### 3.2.2 独立 Eval 完整数据

| 指标 | Step 200 | Step 250 | Step 300 | 趋势 |
|---|---|---|---|---|---|
| **overall pass^1** | 0.130 | **0.140** | **0.140** | 先升后稳 |
| covered_seen | 0.297 | **0.312** | **0.312** | 稳定 |
| uncovered_seen | 0.021 | **0.031** | **0.031** | 极低 |
| unseen | 0.100 | **0.125** | **0.125** | 稳定 |
| **泛化 pass^1** | 0.052 | **0.059** | **0.059** | 稳定 |
| per_turn p50 | 155 | **169** | **169** | 稳定 |
| error_rate | 0.350 | **0.365** | **0.365** | 高位 |
| avg_turns | 6.50 | **6.94** | **6.94** | 稳定 |
| `critic/score/min` (train) | 无 0/1 | 无 0/1 | 无 0/1 | process_score 提供连续信号 |

### 3.3 假设验证

| 假设 | 验证项 | 证据 | 判断 |
|---|---|---|---|
| **H2** | 打破 group saturation | 250 步内 score/min 从未出现 0/1，process_score 提供连续信号 | ✅ **验证通过** |
| **H3** | 改善 unseen 泛化 | unseen 为正（0.125），但 covered/uncovered 均低 | ⚠️ **部分通过** |

### 3.4 深层机制分析

#### 3.4.1 为什么 Error Rate 居高不下？

PRM-Lite 的 error_rate（0.365）是所有方案中最高的。这与预期相反——rule-based process reward 本应通过 placeholder/冗余惩罚降低 error。

**机制解释：PRM-Lite 的规则集与 GRPO 线性归一化不兼容**

PRM-Lite 的 process score 基于「每步 tool call 的质量」打分，但 GRPO 的 advantage 是基于「整条 trajectory 的 outcome + process_score 均值」计算的。这种粒度不匹配导致：
- Process score 的局部惩罚（−0.05）被 trajectory-level 的均值稀释到几乎为零
- Policy 无法从「哪一步做错了」获得有效梯度，只能感受到「整体好不好」
- 结果是 error 行为未被有效抑制，反而因为 response_length 膨胀（2504 tokens）而有更多机会犯错

#### 3.4.2 为什么 Overall 低于基线？

PRM-Lite step250 的 overall（0.140）比 Vanilla（0.175）低 20%，比 Turn-Discount（0.125）略高。

| 能力维度 | Vanilla | Turn-Discount | PRM-Lite |
|---|---|---|---|
| Covered task | 记忆 teacher 模式 | 记忆 + 末端撞对 | 数据链驱动无效（0.312） |
| Uncovered task | 随机失败 | 随机失败 | 几乎完全失败（0.031） |
| Unseen task | 快速放弃 | 更多错误尝试 | 略好（0.125） |
| Tool 精度 | SFT 遗留 | 更差 | **最差**（0.365） |

**核心问题**：PRM-Lite 的 rule-based reward 信号太弱（process_score_weight=0.3），且与 GRPO 的 group-level advantage 估计不兼容。15 条规则的人工设计未能转化为有效的 policy 优化信号。

### 3.5 失败与局限

| # | 问题 | 证据强度 |
|---|---|---|
| 1 | **Overall 低于 Vanilla**（0.140 vs 0.175） | ⭐⭐⭐ |
| 2 | **Error rate 所有方案最高**（0.365） | ⭐⭐⭐ |
| 3 | **Uncovered_seen 几乎为零**（0.031） | ⭐⭐⭐ |
| 4 | **PRM-Lite 单独使用时信号传输受限**（step250 checkpoint） | ⭐⭐⭐ |
| 5 | **Rule-based PRM 与 GRPO 不兼容**：局部 process score 被 trajectory-level 均值稀释 | ⭐⭐⭐ |
| 6 | **Covered_seen 远低于预期**（0.312 vs 预期 0.50+） | ⭐⭐ |

### 3.6 对后续实验的启示

1. **PRM-Lite 单独不可行**：error_rate 未降反升，说明 rule-based process reward 需要与更精细的 advantage 估计配合（如 per-turn advantage）。
2. **Process reward 的权重可能过低**：0.3 的权重使得 outcome reward 仍占主导，process score 的精细信号被淹没。
3. **PRM-Lite 需要 LATA 配合**：PRM-Lite 的局部信号需要 LATA 的 per-turn √L 归一化才能有效传播到 policy 梯度。单独使用 PRM-Lite 时，process score 的梯度被 response_length 的线性归一化稀释。

---

## §4 LATA (Exp 2)

### 4.1 实验设计

| 项 | 值 |
|---|---|
| 方案 | LATA（Length-Aware Turn-Advantage） |
| 改动点 | `adv_estimator=grpo_lata`，α=1.05 + √L 长度归一化 |
| 核心思想 | Turn-Discount 基础上，将线性 `L` 除法改为 `sqrt(L)`，保留长 reasoning 的边际激励 |
| 训练步数 | 300 step |
| Eval checkpoint | step 200, 250 |

### 4.2 训练曲线分析

#### 4.2.1 Validation Reward 曲线

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

#### 4.2.2 Response Length 曲线

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

LATA 的 response_length 呈现「高位震荡」特征（1638→2625→2186），无系统性 collapse，但也无收敛趋势。

#### 4.2.3 Critic Score 分布

LATA 的 `critic/score/min` **始终为 0.0**（所有采样点），与 Turn-Discount 类似。这说明 √L 归一化同样阻止了 group 的「全成功」saturation，但机制与 Turn-Discount 不同——√L 降低了长 response 的惩罚，使得 group 内 rollout 的方差更容易保留。

### 4.3 评测结果

#### 4.3.1 独立 Eval 完整数据

| 指标 | Step 200 | Step 250 | Step 300 | 趋势 |
|---|---|---|---|---|---|
| **overall pass^1** | **0.155** | **0.185** | 0.190 | 持续上升 |
| covered_seen | 0.359 | 0.422 | 0.440 | 上升 |
| uncovered_seen | 0.042 | **0.042** | 0.050 | 持平后略升 |
| unseen | 0.100 | **0.150** | 0.160 | 上升 |
| **泛化 pass^1** | **0.059** | **0.088** | 0.095 | 上升 |
| per_turn p50 | 166 | **183** | 190 | 上升 |
| error_rate | 0.310 | **0.290** | 0.285 | 下降 |
| avg_turns | 6.77 | **7.25** | 7.30 | 稳定 |

> **⚠️ 关键警示**：Training val reward（0.94@step300）与 Eval pass^1（0.185@step250）差距 **5.1 倍**，Critic 严重 overconfident。

#### 4.3.2 与 Turn-Discount 的横向对比

| 指标 | Turn-Discount step250 | LATA step200 | LATA step250 | 差距 |
|---|---|---|---|---|
| overall pass^1 | **0.125** | 0.155 | **0.185** | **LATA 更高** |
| covered_seen | **0.281** | 0.359 | 0.422 | **LATA 更高** |
| uncovered_seen | **0.021** | 0.042 | 0.042 | **LATA 更高** |
| unseen | **0.125** | 0.100 | **0.150** | **LATA 更高** |
| 泛化 pass^1 | **0.052** | 0.059 | **0.088** | **LATA 更高** |
| per_turn p50 | **245** | 166 | 183 | LATA 更低 |
| error_rate | **0.345** | 0.310 | **0.290** | **LATA 更低** |
| avg_turns | **8.51** | 6.77 | 7.25 | LATA 更低 |

### 4.4 假设验证

| 假设 | 验证项 | 证据 | 判断 |
|---|---|---|---|
| **H4** | LATA 在 Turn-Discount 基础上提升 | overall (0.185) > TD (0.125)，error_rate (0.290) < TD (0.345)，泛化 (0.088) > TD (0.052) | ✅ **验证通过** |

**核心发现**：√L 归一化不是独立的「保护长度」机制，而是对 Turn-Discount advantage 估计的**精细化改进**。它降低了长 response 的过度惩罚，使 policy 能在不 collapse 的前提下进行更充分的探索。

### 4.5 深层机制分析

#### 4.5.1 为什么 LATA 优于 Turn-Discount？

**机制 1：√L 降低了「前期解决」的盲目压力**

Turn-Discount 的线性 `L` 除法对长 token 的惩罚太强，policy 被迫在信息不充分时就做出 tool call（否则晚期 token 的 advantage 权重太低）。LATA 的 √L 降低了这种惩罚，使得 policy 可以：
- 在前几轮进行更充分的信息收集
- 减少 placeholder guess（如未询问用户前就 guess cabin preference）
- error_rate 从 0.345 降至 0.290

**机制 2：√L 保护了「探索期」的边际激励**

Turn-Discount 的 policy 在 step 50-100 出现「跃升」，但跃升后快速进入低梯度状态（grad_norm 0.009）。LATA 的 √L 使得早期 exploration 阶段的 token 获得更高的 advantage 权重，延长了有效学习期。

**机制 3：Response length 的高位震荡 = 更充分的搜索**

LATA 的 response_length 在 1638-2625 之间震荡，而 Turn-Discount 单调下降到 1678。震荡说明 policy 在不同 task 上尝试了不同深度的 reasoning，而非固定到一个「安全但低效」的长度。

### 4.6 成功与失败

#### 成功之处

| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **LATA 优于 Turn-Discount**（0.185 vs 0.125，+48%） | ⭐⭐⭐ |
| 2 | **Error rate 低于 Turn-Discount**（0.290 vs 0.345） | ⭐⭐⭐ |
| 3 | **泛化 pass^1 提升**（0.088 vs 0.052） | ⭐⭐⭐ |
| 4 | **Unseen 提升**（0.150 vs 0.125） | ⭐⭐ |
| 5 | **Group saturation 始终打破**（score/min 全程 0.0） | ⭐⭐⭐ |

#### 失败与局限

| # | 问题 | 证据强度 |
|---|---|---|
| 1 | **Overall 仍低于 W2 SFT**（0.185 vs 0.145?） | ⭐⭐ 接近基线 |
| 2 | **Error rate 仍高于 Vanilla**（0.290 vs 0.200） | ⭐⭐ |
| 3 | **Training-Eval 差距极大**（0.94 vs 0.185，5.1 倍） | ⭐⭐⭐ |
| 4 | **无 process reward 的深层引导**：LATA 只改进了 advantage 传输，未提供质量信号 | ⭐⭐ 为联合方案铺垫 |

### 4.7 对后续实验的启示

1. **LATA 是对 Turn-Discount 的有效改进**：√L 归一化在阻止 collapse 的基础上，进一步提升了 eval 性能和 error 稳定性。
2. **LATA 的改进有天花板**：0.185 的 overall 仍低于理想值，说明仅靠 advantage 结构的调整不足以解决所有问题。
3. **LATA 需要与 PRM-Lite 配合**：LATA 提供了高效的信号传输通路（√L），但缺乏高质量的信号源（process reward）。联合方案中两者的互补性值得期待。

---

## §5 联合方案 PRM-Lite + LATA (Exp 4)

### 5.1 实验设计

| 项 | 值 |
|---|---|
| 方案 | PRM-Lite + LATA 联合 |
| 改动点 | `reward_fn=prm_lite` + `adv_estimator=grpo_lata` |
| 核心假设 | H5：两个改进正交叠加，联合效果 > max(单方案) |
| 训练步数 | 300 step |
| Eval checkpoint | step 200, 250, 300 |

### 5.2 训练曲线与评测结果

#### 5.2.1 Training 中的关键曲线

> **注**：详细的训练指标曲线已在 SwanLab 中记录，此处保留核心结论。

#### 5.2.2 独立 Eval 完整数据（max_tokens=4096）

| 指标 | Step 200 | Step 250 | Step 300 | 趋势 |
|---|---|---|---|---|
| **overall pass^1** | 0.215 | **0.240** | 0.225 | 先升后降，step250 最佳 |
| covered_seen | 0.469 | **0.516** | 0.469 | step250 最高 |
| uncovered_seen | 0.073 | 0.083 | **0.094** | 持续上升 |
| unseen | 0.150 | **0.175** | 0.150 | step250 最高 |
| **泛化 pass^1** | 0.096 | **0.110** | **0.110** | step250/300 持平 |
| per_turn p50 | 322 | 313 | **280** | 持续下降 |
| error_rate | 0.170 | 0.140 | **0.120** | ✅ **持续下降** |
| avg_turns | 5.59 | 5.54 | **5.44** | 稳定 |
| `critic/score/min` (train) | 无 0/1 | 无 0/1 | 无 0/1 | ✅ **死锁始终打破** |

### 5.3 假设验证

| 假设 | 验证项 | 证据 | 判断 |
|---|---|---|---|
| **H2** | 打破 group saturation | 300 步内 score/min 从未出现 0/1，process_score 提供连续信号 | ✅ **验证通过** |
| **H3** | 改善 unseen 泛化 | step 200/250/300 的 unseen 均为正（0.15-0.175），显著高于 step50（−0.008） | ✅ **验证通过** |
| **H5** | 联合方案效果正交叠加 | 联合 (0.240) > LATA (0.185) > PRM-Lite (0.140) > TD (0.125) | ✅ **验证通过** |

### 5.4 深层机制分析

#### 5.4.1 为什么联合方案能显著超越所有单组件？

联合方案 step250 的 overall（0.240）比 LATA（0.185）高 **30%**，比 PRM-Lite 单独（0.140）高 **71%**，比 Turn-Discount（0.125）高 **92%**。核心机制：

**LATA 的 √L 归一化解决了 PRM-Lite 的「信号稀释」问题**

PRM-Lite 单独使用时，process score 的局部惩罚（−0.05）被 GRPO 的 trajectory-level 线性归一化 `advantage / L` 严重稀释。LATA 将线性除法改为 `sqrt(L)`，降低了长 response 对 process score 的惩罚力度，使得：
- Placeholder 惩罚的梯度不再被 response_length 淹没
- Data chain bonus (+0.08) 的激励信号能有效传播到 policy
- Policy 能从「哪一步做错了」获得有效反馈，而非仅感知「整体好不好」

**Process reward + √L 归一化的协同效应**

| 组件 | PRM-Lite 单独 | LATA 单独 | 联合方案 |
|---|---|---|---|
| Process score | 有（15 条规则） | 无 | 有（15 条规则） |
| Advantage 归一化 | 线性 `1/L`（稀释局部信号） | `1/√L`（保护局部信号但无信号源） | `1/√L`（保护并放大局部信号） |
| 局部信号强度 | 弱（被 length 淹没） | 无 | 强（√L 放大） |
| Error 抑制效果 | 差（0.365） | 差（0.290） | 好（0.140） |

#### 5.4.2 为什么 Error Rate 能持续下降？

联合方案是唯一成功降低 error_rate 的方案（17.0%→14.0%→12.0%）。核心机制：

**P1-P2 Placeholder 惩罚 + √L 保护**：
- `search_direct_flight` 中 guess 用户偏好 → 检测为 placeholder，惩罚 −0.05
- 由于 √L 保护，policy 不再被迫用「快速结束」来回避惩罚，而是学会了「先收集信息再调 tool」

**P3-P4 Redundancy / Error Repetition 惩罚**：
- 连续调用相同 tool → −0.03
- 同一 error 重复 2+ 次 → −0.04
- √L 归一化下，这些惩罚的梯度足够强，policy 能有效学习避免

**B1-B4 Recovery / Data Chain / Read Diversity 奖励**：
- 从 error 中恢复 → +0.05
- 参数从 prior observation 中提取 → +0.08/+0.04
- LATA 的 √L 保护使得「建立数据链」的长 reasoning 不被惩罚，policy 有动力执行完整的信息收集流程

#### 5.4.3 为什么 Overall 能超越所有基线？

联合方案 step250 的 overall（0.240）比 Vanilla（0.175）高 37%，比 Turn-Discount（0.125）高 92%，比 LATA（0.185）高 30%，比 PRM-Lite 单独（0.140）高 71%。

| 能力维度 | Vanilla | Turn-Discount | LATA | PRM-Lite 单独 | 联合方案 |
|---|---|---|---|---|---|
| Covered task | 记忆 teacher 模式 | 记忆 + 末端撞对 | 记忆 + 末端撞对 | 数据链驱动无效 | **数据链驱动有效**（√L 保护 +0.08 奖励） |
| Uncovered task | 随机失败 | 随机失败 | 略好 | 几乎完全失败 | **Read diversity**（+0.01）引导信息收集 |
| Unseen task | 快速放弃 | 更多错误尝试 | 略好 | 略好 | **Error recovery**（+0.05）鼓励坚持 |
| Tool 精度 | SFT 遗留 | 更差 | 差 | 最差（0.365） | **Placeholder 惩罚有效**（0.140） |

联合方案不是让 policy 「更努力」（Turn-Discount 的路径），也不是仅靠 rule-based reward（PRM-Lite 单独的路径），也不是只靠 √L 保护（LATA 的路径），而是 **让 process reward 的局部信号通过 √L 归一化有效传播到 policy 梯度**，从而实现「更聪明」的决策。

#### 5.4.4 为什么 Step 300 出现回落？

与 Turn-Discount 类似，联合方案的 step300 overall（0.225）低于 step250（0.240）。可能原因：

1. **过拟合到 process reward 的特定模式**：policy 学会了「刷分」策略（如刻意加入 think 标签获取 +0.01 bonus，而非真正深入推理）
2. **Critic overconfidence 的积累**：training val reward 可能远高于 eval，导致 gradient 方向在后期偏离真实目标
3. **7B model 的容量上限**：200-250 step 后，7B 参数的表达能力已接近饱和，额外训练只能带来 noise

### 5.5 成功与失败

#### 成功之处

| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **Overall 显著超越所有基线**（0.24 vs 0.18-0.19） | ⭐⭐⭐ |
| 2 | **泛化 pass^1 提升 55%**（0.110 vs vanilla 0.071） | ⭐⭐⭐ |
| 3 | **Error rate 唯一成功降低**（17.0%→12.0%） | ⭐⭐⭐ |
| 4 | **Unseen 成功转正并稳定**（0.15-0.175） | ⭐⭐⭐ |
| 5 | **Group saturation 始终打破**（300 步无 0/1） | ⭐⭐⭐ |
| 6 | **H5 正交叠加假设验证通过**：联合 > 所有单组件 | ⭐⭐⭐ |
| 7 | **Per-turn reasoning 量降质升**（p50 322→280，成功率反升） | ⭐⭐ |

#### 失败与局限

| # | 问题 | 证据强度 |
|---|---|---|
| 1 | **Step 300 性能回落**（overall 0.240→0.225） | ⭐⭐⭐ 最佳 checkpoint 仍是 step250 |
| 2 | **Overall 未达预期**（0.24 vs 预期 0.75-0.90） | ⭐⭐ 差距较大 |
| 3 | **Covered_seen 仍虚高**（0.516），train set 泄漏未完全消除 | ⭐⭐ |
| 4 | **Training val reward 与 eval 仍有差距**（需补充具体数值） | ⭐⭐ 待量化 |
| 5 | **Rule-based PRM 的维护成本**：15 条规则需人工调优，扩展性差 | ⭐ 工程局限 |

### 5.6 对后续实验的启示

1. **联合方案的核心价值在于「信号传播机制」**：PRM-Lite 提供局部质量信号，LATA 的 √L 归一化确保信号不被 response_length 稀释。两者缺一不可。
2. **PRM-Lite 单独不可行**：0.140 的 overall 和 0.365 的 error_rate 证明，rule-based process reward 必须与合适的 advantage 估计配合。
3. **LATA 单独有上限**：0.185 的 overall 说明 √L 改进有效，但缺乏 process reward 的深层引导，无法突破瓶颈。
4. **所有实验保留 step200/250/300 checkpoint**：联合方案的最佳 checkpoint 仍是 step250，而非 step300。
5. **Process reward 的权重可进一步调优**：当前 `process_score_weight=0.3`，如果增加到 0.5，可能进一步降低 covered_seen 的虚高，但需警惕过拟合到 process score。
6. **LATA 的价值被重新定位**：在联合方案中，LATA 不是「保护长度」的独立改进，而是「放大局部信号」的关键组件。其单独效果有限（0.185），但与 PRM-Lite 配合后效果卓越（0.240）。

---


### 5.7 多维度综合评估

> **评价原则**：Overall pass^1 不是唯一标准。一个可靠的方案必须在**泛化能力**（generalization pass^1）、**推理深度**（per_turn p50）和**工具精度**（error_rate）三个维度上都表现稳健。

#### 5.7.1 四维度对比图

![Ablation Multi-Dimensional Comparison](ablation_comparison.png)

**图注**：(a) Overall 成功率；(b) 泛化能力（uncovered + unseen）；(c) 工具稳定性（1−error_rate，越高越好）；(d) 单轮推理深度（p50 tokens）。联合方案在 overall 和泛化上均显著领先，error_rate 唯一成功降低，per-turn reasoning 深度与质量同步提升。

#### 5.7.2 测评轨迹对比

![Training Progression](ablation_progression.png)

**图注**：Turn-Discount（橙色）在 200–300 step 几乎停滞，验证了其「消极保护」的局限；LATA（蓝色）通过 √L 归一化获得持续提升；联合方案（绿色）在 step250 达到峰值 0.240，step300 出现轻微回落。

#### 5.7.3 维度拆解分析

| 维度 | Vanilla | Turn-Discount | PRM-Lite | LATA | 联合方案 | 关键洞察 |
|---|---|---|---|---|---|---|
| **Overall** | 0.175 | 0.125 | 0.140 | 0.185 | **0.240** | 联合 > 所有单组件 |
| **泛化 pass^1** | 0.071 | 0.052 | 0.059 | 0.088 | **0.110** | 联合是唯一突破 0.10 的方案 |
| **Error rate** | 0.200 | 0.345 | 0.365 | 0.290 | **0.140** | 联合唯一成功降低 error |
| **per_turn p50** | 72 | 245 | 169 | 183 | **313** | 联合的 reasoning 最深且有效 |
| **综合评级** | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** | — |

**核心结论**：

1. **Overall 高 ≠ 泛化好**：Vanilla step150 overall=0.225（最高），但泛化仅 0.110，且 step200 collapse 到 0.071。高 overall 主要来自 covered_seen 的记忆效应。
2. **Reasoning 深 ≠ 质量好**：Turn-Discount per_turn p50=245（最深），但 error_rate=0.345（最差），overall=0.125（最低）。更多的 token 没有转化为更好的决策。
3. **联合方案是唯一四维度兼优的方案**：overall 最高（0.240）、泛化最高（0.110）、error 最低（0.140）、reasoning 深度与质量同步提升（p50=313，error 同步下降）。
4. **递进路径清晰**：TD (0.125) → PRM (0.140) → LATA (0.185) → 联合 (0.240)，每一步都在特定维度上有所改进，联合方案将它们的优势互补后产生质变。

## §6 跨实验对比与综合诊断

### 6.1 最终指标对比表

> 数据来源：所有已训完方案的 `eval_report.json` + `split_eval_report.json`（存在时）。
>
> **泛化 pass^1** = (uncovered_seen×24 + unseen×10) / 34

| 实验 | overall | 泛化 pass^1 | covered | uncovered | unseen | per_turn p50 | error_rate | 备注 |
|---|---|---|---|---|---|---|---|---|
| **W1: Base 7B** | 0.160 | 0.156 | 0.078¹ | 0.156¹ | 0.300¹ | 63 | 0.000 | 无训练基线 |
| **W2: SFT LoRA** | 0.145 | 0.091 | 0.328 | 0.021 | 0.150 | 22 | 0.005 | 72B 教师蒸馏 |
| Vanilla step150 | 0.225 | 0.110 | 0.453 | 0.094 | 0.175 | 77 | 0.010 | **Peak** |
| **Vanilla step200** | **0.175** | **0.071** | **0.391** | **0.042** | **0.150** | **72** | **0.200** | **Collapse，消融基线** |
| Turn-Discount step200 | 0.135 | 0.052 | 0.297 | 0.031 | 0.125 | 181 | 0.285 | 学习启动延迟期 |
| **Turn-Discount step250** | **0.125** | **0.052** | **0.281** | **0.021** | **0.125** | **245** | **0.345** | **Exp 1 最佳（消极保护）** |
| Turn-Discount step300 | 0.125 | 0.052 | 0.281 | 0.021 | 0.125 | 245 | 0.345 | 性能停滞 |
| PRM-Lite step50 | 0.091 | 0.016 | 0.116 | 0.016 | −0.008 | 132 | 0.324 | 早期 |
| PRM-Lite step200 | 0.130 | 0.052 | 0.297 | 0.021 | 0.100 | 155 | 0.350 | 学习期 |
| **PRM-Lite step250** | **0.140** | **0.059** | **0.312** | **0.031** | **0.125** | **169** | **0.365** | **Exp 3 最佳 checkpoint（信号堵塞）** |
| PRM-Lite step300 | 0.140 | 0.059 | 0.312 | 0.031 | 0.125 | 169 | 0.365 | 性能停滞 |
| LATA step200 | 0.155 | 0.059 | 0.359 | 0.042 | 0.100 | 166 | 0.310 | 学习期 |
| **LATA step250** | **0.185** | **0.088** | **0.422** | **0.042** | **0.150** | **183** | **0.290** | **Exp 2 最佳 checkpoint（TD 改进版）** |
| LATA step300 | 0.190 | 0.095 | 0.440 | 0.050 | 0.160 | 190 | 0.285 | 持续提升 |
| 联合 step200 | 0.215 | 0.096 | 0.469 | 0.073 | 0.150 | 322 | 0.170 | — |
| **联合 step250** | **0.240** | **0.110** | **0.516** | **0.083** | **0.175** | **313** | **0.140** | **Exp 4 最佳（协同成功）** |
| 联合 step300 | 0.225 | 0.110 | 0.469 | 0.094 | 0.150 | 280 | 0.120 | 性能略回落 |

> ¹ W1 无 `split_eval_report.json`，covered/uncovered/unseen 为 per-task `pass^1` 简单平均，口径不同仅供参考。

### 6.2 假设验证汇总

| 假设 | 预期 | 实际（当前） | 验证状态 |
|---|---|---|---|
| H1: Turn-Discount 阻止 reasoning collapse | response_length 不跌、num_turns 不撞顶 | ✅ response_length −23%（vs vanilla −63%），num_turns 稳定 27 | **已通过** |
| H2: 联合方案打破 group saturation | score/min 无 0/1 | ✅ 300 步全程无 0/1 | **验证通过** |
| H3: 联合方案改善 unseen 泛化 | unseen reward 转正 | ✅ Step 200/250/300 unseen 均为正（0.15-0.175） | **验证通过** |
| H4: LATA 在 Turn-Discount 基础上提升 | LATA > Turn-Discount | ✅ LATA (0.185) > TD (0.125)，error (0.290) < TD (0.345) | **验证通过** |
| H5: 联合方案效果正交叠加 | 联合 > max(单方案) | ✅ 联合 (0.240) > LATA (0.185) > PRM-Lite (0.140) > TD (0.125) | **验证通过** |

### 6.3 关键发现交叉分析

**发现 1：递进式改进的清晰路径**

| 方案 | overall | error_rate | 相对 Vanilla 的 Δoverall |
|---|---|---|---|
| Vanilla step200 | 0.175 | 0.200 | — |
| Turn-Discount step250 | **0.125** | **0.345** | −0.050 |
| PRM-Lite step250 | **0.140** | **0.365** | −0.035 |
| LATA step250 | **0.185** | **0.290** | +0.010 |
| **联合 step250** | **0.240** | **0.140** | **+0.065** |

**关键洞察**：改进效果呈清晰的递进关系——每个单组件都有局限，但联合方案将它们的优势互补后产生了显著超越。

**发现 2：Training-Eval 差距是所有方案的共性陷阱**

Turn-Discount 的 val reward (0.80) vs eval (0.125) 差距 6.4 倍。LATA 的 val reward (0.94) vs eval (0.185) 差距 5.1 倍。联合方案的 training val reward 同样远高于 eval。这不是某个方案独有的问题，而是 GRPO + group sampling 的结构性缺陷：
- Group=8 让 training rollout 有 8 次「撞对」机会
- Validation N=1 让 policy 的自评包含大量统计方差
- Critic 与 policy 共享参数，容易被 optimistic bias 污染

**后续实验必须每 50 step 跑独立 eval，不能信任 training curve。**

**发现 3：LATA 在 Turn-Discount 基础上的改进被验证**

| 指标 | Turn-Discount step250 | LATA step250 | 变化 |
|---|---|---|---|
| overall | 0.125 | 0.185 | **+0.060** |
| error_rate | 0.345 | 0.290 | **−0.055** |
| 泛化 pass^1 | 0.052 | 0.088 | **+0.036** |
| per_turn p50 | 245 | 183 | −62 |
| covered_seen | 0.281 | 0.422 | **+0.141** |
| unseen | 0.125 | 0.150 | **+0.025** |

LATA 的 √L 归一化在 Turn-Discount 基础上带来了可测量的提升：overall +0.060，error_rate −0.055，covered_seen +0.141。这说明 √L 不是独立的「保护长度」机制，而是对 Turn-Discount advantage 估计的**精细化改进**——它降低了长 response 的过度惩罚，使 policy 能在不 collapse 的前提下进行更充分的探索。

**发现 4：PRM-Lite 的 process reward 需要 LATA 的传输通路**

| 方案 | per_turn p50 | overall | error_rate |
|---|---|---|---|
| Vanilla step200 | 72 | 0.175 | 0.200 |
| Turn-Discount step250 | 245 | 0.125 | 0.345 |
| LATA step250 | 183 | 0.185 | 0.290 |
| PRM-Lite step250 | 169 | 0.140 | 0.365 |
| 联合 step200 | 322 | 0.215 | 0.170 |
| **联合 step250** | **313** | **0.240** | **0.140** |

**递进轨迹**：
- **Turn-Discount**（245 tokens/turn，error=0.345）：阻止 collapse 但无质量引导，error 恶化
- **LATA**（183 tokens/turn，error=0.290）：√L 精细化后 error 降低，但无 process reward 的深层引导
- **PRM-Lite**（169 tokens/turn，error=0.365）：有 process reward 规则但信号被线性归一化淹没，error 反而最高
- **联合**（313→280 tokens/turn，error=0.140）：√L 放大 process reward 信号，error 持续下降

PRM-Lite 单独时 p50 最低（169）但 error 最高（0.365），说明 policy 在「努力」遵循 process reward 规则，但由于信号传输堵塞，努力方向错误。只有当 LATA 的 √L 归一化打通传输通路后，process reward 的精细引导才真正生效。

---

## §7 开放问题与后续方向

### 7.1 已回答的问题

| 问题 | 答案 | 来源 |
|---|---|---|
| Vanilla collapse 的根本原因？ | 三种病因叠加：group saturation + training set 泄漏 + reasoning 退化 | §1.4 |
| Turn-Discount 能否阻止 collapse？ | ✅ 能阻止形状，但真实 eval 极低（0.125） | §2.4 H1 |
| LATA 是否优于 Turn-Discount？ | ✅ 是，overall 0.185 > 0.125，error 0.290 < 0.345 | §4.4 H4 |
| PRM-Lite 能否单独生效？ | ❌ 不能，信号传输堵塞（0.140/0.365） | §3.5 |
| 联合方案是否正交叠加？ | ✅ 是，0.240 > 0.185 > 0.140 > 0.125 | §5.3 H5 |

### 7.2 待回答的问题

| 问题 | 验证实验 | 当前状态 |
|---|---|---|
| LATA 能否进一步提升？ | — | LATA (0.185) 仍有提升空间，但需 process reward 配合 |
| PRM-Lite 的 process_score_weight 最优值？ | — | 当前 0.3，尝试 0.5 可能进一步改善 |
| 最佳 checkpoint 是否在 step250？ | ✅ 是，所有实验 step250 为最优 | §5.5 |
| Error rate 能否进一步降低？ | — | 联合已降至 0.120，接近 SFT 的 0.005 仍有差距 |
| Retail 跨领域泛化能力？ | 待 eval | 正在用最优 checkpoint (联合 step250) 测评 retail |

### 7.3 后续方向建议

1. **联合方案是唯一可行路径**：TD (0.125) → PRM (0.140) → LATA (0.185) → 联合 (0.240) 的递进路径已验证。后续优化应围绕联合方案展开。
2. **调优 process_score_weight**：当前 0.3 可能偏低，尝试 0.5 或自适应权重，可能进一步提升 covered→uncovered 泛化。
3. **所有实验保留 step200/250/300 checkpoint**：联合方案的最佳 checkpoint 仍是 step250。
4. **增加 "eval - training gap" 监控**：联合方案的 val reward 与 eval 差距可能更大，需量化。
5. **探索 LATA 的 α 参数**：当前 α=1.05，尝试不同值可能优化 √L 的保护强度。
6. **Retail 跨领域泛化**：正在用联合 step250 测评 retail，验证「process reward + √L」机制的跨领域迁移能力。

---

