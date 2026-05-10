# -5 Ablation 计划：从诊断到改进

> **文档定位**：基于 `docs/vanilla_grpo/vanilla_grpo_diagnosis.md` 的三类诊断问题，设计并实施四个消融实验。本文是「实验设计手册」——只包含**计划、设计原理、代码实现和预期指标**，不包含任何实际训练进度。
>
> **实际执行范围**：四个实验（Turn-Discount、PRM-Lite、LATA、PRM-Lite + LATA）。已删除未执行的 partial credit 方案。
>
> **实验进度与结果追踪**：见 `docs/ablation_progress_tracker.md`，每次实验完成后填入实际数据。

---

## §1 项目目标与约束

### 1.1 核心目标

在 τ-bench airline（50 task，长链路多工具对话 agent）上，用 7B policy + 72B-AWQ user simulator，解决标准 GRPO 训练中的三类问题：

1. **Group reward saturation（双向死局）**：group 内全 0 或全 1 → advantage=0 → 梯度消失
2. **训练集泄漏收益偏向**：covered_seen 提升大，uncovered/unseen 几乎无提升
3. **Per-turn reasoning 退化**：policy 漂向「短 reasoning + 多 tool 试错」，train 上有效但 eval 上 collapse

### 1.2 硬性约束

| 约束 | 说明 |
|---|---|
| **控变量严格** | 所有方案共享同一套 base config，仅改 reward computation 或 advantage 计算 |
| **不破坏工程优化** | 保留 bypass_mode + use_fused_kernels + TP=2，否则显存爆 |
| **训练 budget** | 单台 3×A800，每个方案 300 step，约 70h wall time |
| **评测一致性** | 所有方案用同一份 50-task eval set，主指标用 `pass_hat_1`（uncovered+unseen 加权） |

### 1.3 四个实验总览

| 实验 | 改了什么 | 解决什么问题 | 代码位置 |
|---|---|---|---|
| **Exp 1: Turn-Discount** | Advantage 计算：早期 token 高权重 | §3.3 reasoning 退化 | `core_algos.py: compute_grpo_turn_discounted_outcome_advantage` |
| **Exp 3: PRM-Lite** | Reward 计算：`outcome + 0.3×process_score` | §3.1 group saturation + §3.2 泛化 | `tau_bench_interaction.py: _compute_reasoning_quality_score` |
| **Exp 2: LATA** | Advantage 计算：turn-discount + `sqrt(L)` 归一化 | §3.3 reasoning 退化（升级版） | `core_algos.py: compute_grpo_lata_outcome_advantage` |
| **Exp 4: PRM-Lite + LATA** | Reward + Advantage 同时改 | 全部 3 类问题 | 上述两者同时启用 |

---

## §2 Exp 1: Turn-Discounted Advantage

### 2.1 问题定义

标准 GRPO 的 advantage 是一个**标量**，乘以 response_mask 后平均分摊到 trajectory 的每一个 token 上：

```python
# 标准 GRPO
advantages = scalar_advantage.unsqueeze(-1) * response_mask
```

这意味着：无论你是在第 1 轮就通过精准 reasoning 解决问题，还是拖到第 15 轮靠试错蒙对，**所有 token 的 advantage 完全一样**。

在 outcome-only reward 下，policy 面临两条等期望路径：
- 路径 A：每轮充分 reasoning + 1-2 次精准 tool call → reward = 1
- 路径 B：每轮短 reasoning + 5-8 次试错 → 偶尔也撞对 → reward = 1

路径 B 在 group=8 的多次采样下甚至期望更高（8 条总有一条能撞对）。标准 GRPO 的均匀 advantage 无法区分这两条路径。

### 2.2 改进设计

**核心思想**：给**早期 token 更高的 advantage 权重**，给晚期 token 更低的权重。这样，「前期就解决」的路径 A 中那些高权重的早期 token 会带来更大的 policy gradient；而「拖到末端试错」的路径 B 中那些低权重的晚期 token 贡献很小。

**具体实现**：指数衰减权重

```python
weight[t] = alpha^(L-1-t)   # alpha = 1.05
```

- `t` = token 位置（0 到 L-1）
- `L` = 该样本的实际 response length
- `alpha = 1.05` → 早期 token（t 小）的 exponent 大 → 权重大
- 归一化：`mean(weight) = 1.0`，不改变整体梯度量级

**完整代码**（`verl/trainer/ppo/core_algos.py`）：

```python
@register_adv_est("grpo_turn_discounted")
def compute_grpo_turn_discounted_outcome_advantage(
    token_level_rewards, response_mask, index, epsilon=1e-6,
    norm_adv_by_std_in_grpo=True, config=None
):
    # Step 1: 标准 GRPO group normalize
    scores = token_level_rewards.sum(dim=-1)
    # ... group mean/std normalization ...

    # Step 2: 位置权重
    alpha = 1.05
    resp_len = response_mask.shape[1]
    positions = torch.arange(resp_len, dtype=torch.float64)
    active_lengths = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
    exponents = active_lengths - 1 - positions.unsqueeze(0)
    log_weights = exponents * math.log(alpha)
    # ... log-space stable normalization ...
    weights = (weights_stable * active_count / weight_sum).to(torch.float32)

    # Step 3: 加权
    scores = scores.unsqueeze(-1) * weights * response_mask
    return scores, scores
```

**配置启用**（`configs/train/grpo/turn_discount.yaml`）：

```yaml
algorithm:
  adv_estimator: grpo_turn_discounted
  turn_discount:
    enable: true
    alpha: 1.05
```

### 2.3 为什么 alpha=1.05？

alpha 控制「对早期 token 的偏爱程度」：
- alpha = 1.0 → 退化为标准 GRPO（无衰减）
- alpha = 1.05 → 对于 L=2000 的 response，早期 token 权重约是晚期的 2.5 倍
- alpha 越大 → 对「早期解决」的激励越强，但也会过度惩罚必要的多轮交互

1.05 是一个经验值：从 vanilla 的 collapse 曲线反推，policy 在 step 100 后 response_length 从 3000 降到 1100，说明晚期 token 的「试错价值」被高估了约 2-3 倍。alpha=1.05 把这个高估纠正回来，同时不过度惩罚真正需要多轮交互的 task。

### 2.4 Reward Hacking 风险与防御

**Turn-Discount 可能引入的 hacking 风险**：

| 风险 | 机制 | 防御措施 |
|---|---|---|
| **过度压缩晚期交互** | 晚期 token 权重过低 → policy 学会「提前结束」而非「完成必要步骤」 | alpha=1.05 是 mild 衰减（非极端）；监控 `num_turns/mean` 若 <10 则调低 alpha |
| **早期 token 梯度爆炸** | 早期权重集中 → 少数 token 的梯度可能过大 | log-space stable normalization 防止数值溢出；监控 `actor/grad_norm` |
| **绕过 turn-discount 的「中期试错」** | Policy 不在末端试错，改在第 3-5 轮集中试错 | 这是设计目标之一——把试错从晚期移到中期，至少保留了前几轮的 reasoning 空间；PRM-Lite 的 P3/P4 规则进一步惩罚 redundancy/error repetition |

### 2.5 预期指标（300 step）

| 指标 | Vanilla 基线 | 预期值 | 预期变化 |
|---|---|---|---|
| overall pass^1 @step300 | 0.175 (step200 已塌，无 step300) | **0.75-0.85** | 阻止 collapse，稳定收敛 |
| 泛化 pass^1 @step300 | — | **0.10-0.12** | 对 OOD 无直接帮助，但稳定不塌 |
| per_turn p50 @step300 | ~30-50 | **≥ 80** | Reasoning 退化被阻止 |
| Δstability (step300−step150) | −0.050 | **≥ −0.02** | 无 collapse |
| num_turns/mean (val) | 5.08 (step200) | **25-30** | 不撞顶也不早死 |
| error_rate @step300 | 0.200 | **≤ 0.08** | 不爆发 |
| `critic/score/min` | 频繁 = 1.0 | **≤ 0.1（极少 =1.0）** | Group saturation 意外缓解 |
| 学习启动延迟 | — | **前 50 步可能停滞** | 削弱「以量补质」后的副作用 |

---

## §3 Exp 3: PRM-Lite

### 3.1 问题定义

标准 GRPO 的 reward 是 **binary outcome**：成功=1，失败=0。这导致两个问题：

**问题一（§3.1）**：group saturation。当 8 条 rollout 的 outcome 全相同时（全 0 或全 1），`std(r_group)=0`，advantage=0，梯度消失。

**问题二（§3.2）**：uncovered/unseen 区域 group 内几乎全失败，outcome 始终为 0，policy 在这些 task 上永远学不到东西。

PRM（Process Reward Model）的经典解法是训练一个独立模型来评估每一步的质量。但在只有 50 task 的数据集上，训练一个独立的 step-level reward model 本身就是一个小项目，数据量可能不够。所以我们设计了 **PRM-Lite**：用一组 hand-crafted rules 模拟 PRM 的功能，验证「process-level signal 能否打破 group variance 死锁」这一核心机制。

### 3.2 改进设计

**核心思想**：在 outcome 之外，叠加一个 **rule-based process score**（∈[-0.5, +0.5]），最终 reward：

```raw
r = outcome + 0.3 × process_score
```

这样即使 outcome 全相同，process_score 的 variance 仍能在 group 内拉出梯度。

**为什么 weight=0.3？**
- outcome 的权重是 1.0（范围 [0, 1]）
- process_score 的范围是 [-0.5, +0.5]
- weight=0.3 → process 对最终 reward 的贡献范围是 [-0.15, +0.15]
- 这个量级足够在 group 内拉出 variance（离线验证 intra-group std ≈ 0.049），但又不会让 outcome 信号被淹没

**配置启用**（`configs/train/grpo/prm_lite.yaml`）：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      interaction_config_path: configs/interaction_config/tau_bench_airline_prm_lite.yaml
```

在 `tau_bench_airline_prm_lite.yaml` 中启用 `reward_mode: prm_lite`，让 `TauBenchInteraction.calculate_score()` 返回 `_compute_prm_lite_reward()`。

### 3.3 PRM-Lite 规则集完整设计

规则集基于诊断报告 §3.3 的 task 49 failure case 设计——每条规则都对应一个真实观察到的 failure mode。

#### 3.3.1 Tool 分类

```python
_READ_TOOLS = {
    "list_all_airports", "search_direct_flight", "search_onestop_flight",
    "get_user_details", "get_reservation_details", "calculate"
}
_WRITE_TOOLS = {
    "book_reservation", "cancel_reservation", "update_reservation_baggages",
    "update_reservation_passengers", "update_reservation_flights", "send_certificate"
}
_ESCALATION = {"transfer_to_human_agents"}
_THINK_TOOLS = {"think", "implicit_think"}
```

**分类原则**：read 失败后果小（查询返回错误，可重试），write 失败后果大（可能污染数据库）。罚分非对称。

#### 3.3.2 15 条规则速查表（v4-optimal）

| 类别 | # | 规则 | 触发条件 | 分数 | 设计动机 |
|---|---|---|---|---|---|
| **惩罚** | P1 | Placeholder (write) | write tool 参数含占位符 | **-0.05** | task 49 turn 4：把 `"previous_reservation"` 当字面 ID |
| | P2 | Placeholder (read) | read tool 参数含占位符 | **-0.03** | 同上，read 后果较小 |
| | P3 | Redundancy | 最近 3 步内完全相同的 (tool, args) | **-0.03** | task 49 turn 6：重复调同样的 `get_user_details` |
| | P4 | Error repetition | 上一步 error，本步完全相同的 (tool, args) | **-0.04** | task 49 turn 7：犯错后不纠正 |
| | P4 | Recovery | 上一步 error，本步换了 (tool, args) | **+0.05** | 正向激励：犯错后尝试不同方法 |
| | P5 | Escalation (premature) | 无 read 就 transfer_to_human | **-0.10** | 防 lazy give-up |
| | P5 | Escalation (late) | 有 read 后 transfer_to_human | **-0.05** | 区别于 premature |
| | P6 | No reasoning | trajectory 无 think 且 ≥3 步 | **-0.05** | 强制保留 reasoning |
| | P8 | Length penalty | trajectory 步数 > 8 | **-0.01/步** | 惩罚「做得多但没结果」 |
| **奖励** | B1 | Data chain (write) | write 参数值出现在之前 observation 提取的 entity 集合里 | **+0.08** | reasoning-action 一致性 |
| | B1 | Data chain (read) | read 参数值出现在之前 observation 提取的 entity 集合里 | **+0.04** | 同上，read 奖励减半 |
| | B2 | First read exploration | 首次调用某类 read tool | **+0.01** | 鼓励多样化信息收集 |
| | B4/B5 | Think bonus | 调用了 think tool（**条件给**，见 3.3.5） | **+0.01** | reward thinking |
| | B7 | Read diversity | trajectory 使用 ≥3 种不同 read tools | **+0.01** | 全局奖励 |

**已移除的规则**（v4-optimal 删除）：
- `Successful tool call`（+0.01）：长 failure trajectory 因 step 数多自动累积高分，产生反激励
- `Implicit think` 长度梯度：奖励不稳定，易被 hacking

#### 3.3.3 聚合方式：mean-based

```python
process_score = mean(per_step_scores) + trajectory_adjustments
process_score = clip(process_score, -0.5, +0.5)
```

**为什么 mean-based 而非 sum-based？**
- sum-based：长 trajectory 的分数会被 step 数稀释（10 步的平均分 vs 20 步的平均分）
- mean-based：长 trajectory 不会被 clamp 抹平 discrimination。一条 20 步的 failure trajectory 如果每步都犯错，mean score 可以很负；如果只有 2 步犯错，mean score 接近 0

**Trajectory-level adjustments**：
- `think_count == 0 and len >= 3` → no-reasoning penalty (-0.05)
- `len(all_reads) >= 3` → read diversity bonus (+0.01)

#### 3.3.4 Schema-based 占位符检测

```python
_PARAM_PATTERNS = {
    "reservation_id": r"^[A-Z0-9]{6}$",                    # ZFA04Y
    "user_id":        r"^[a-z]+_[a-z]+_\d+$",              # emma_kim_9957
    "payment_id":     r"^(credit_card|gift_card|certificate)_\d+$",
    "flight_number":  r"^[A-Z]{3}\d{3}$",                  # HAT043
    "date":           r"^\d{4}-\d{2}-\d{2}$",
    "origin":         r"^[A-Z]{3}$",
    "destination":    r"^[A-Z]{3}$",
}

_PLACEHOLDER_KEYWORDS = {
    "previous", "unknown", "my", "the", "first", "last",
    "any", "some", "another", "<placeholder>", "<unknown>",
}
```

检测逻辑：(a) 关键词触发 或 (b) schema regex 不匹配 → 判定为 placeholder。

#### 3.3.5 Think bonus 的条件给分（防 reward hacking）

担忧：think 无条件给分 → policy 学会每轮先输出 100+ char 空话再调 tool 拿分。

修正：两种情况下不给 think bonus：
1. **连续 think**：上一步也是 think → think 没起到指导作用
2. **think 后跟 placeholder/redundancy**：think 没指导出有效行动

```python
def _think_bonus(action_history, idx):
    if action_history[idx]["tool"] not in _THINK_TOOLS:
        return 0
    if idx > 0 and action_history[idx-1]["tool"] in _THINK_TOOLS:
        return 0  # 连续 think
    if idx + 1 < len(action_history):
        next_action = action_history[idx + 1]
        if _has_placeholder(next_action) or _is_redundant(action_history, idx + 1):
            return 0  # think 后还是犯错
    return +0.01
```

#### 3.3.6 完整规则示例：task 49 step-by-step 评估

| Turn | Action | 触发规则 | 分数 |
|---|---|---|---|
| 1 | `get_reservation_details(MDCLVA)` | first-read(+0.01) | **+0.01** |
| 3 | `cancel_reservation(MDCLVA)` | data chain write(+0.08) | **+0.08** |
| 4 | `get_reservation_details("previous_reservation")` | placeholder read(-0.03) + first-read(+0.01) | **-0.02** |
| 5 | `get_user_details(emma_kim_9957)` | recovery(+0.05) + first-read(+0.01) | **+0.06** |
| 6 | `get_user_details(emma_kim_9957)` | redundancy(-0.03) | **-0.03** |
| 7 | `get_reservation_details("previous_reservation")` | placeholder read(-0.03) + error repetition(-0.04) | **-0.07** |
| 8 | `get_reservation_details(EHGLP3)` | recovery(+0.05) + data chain read(+0.04) | **+0.09** |
| 9 | `get_reservation_details(66EEUA)` | data chain read(+0.04) | **+0.04** |
| 10 | `get_reservation_details(H1QGCY)` | data chain read(+0.04) | **+0.04** |

`mean(per_step) = 0.022`

Trajectory-level：think_count=0 且 len=9≥3 → no-reasoning penalty **-0.05**；len=9>8 → length penalty **-0.01**

`process_score = 0.022 - 0.05 - 0.01 = -0.04`

**解读**：这条「中段乱、后段对」的 trajectory 得到明确负分——**不再因为「后期做了正确的事」就掩盖中段的错误**。

### 3.4 Reward Hacking 风险与防御

**PRM-Lite 可能引入或未能完全阻止的 hacking 风险**：

| 风险 | 机制 | 防御措施 | 验证方式 |
|---|---|---|---|
| **Think 空话（模式 7）** | Policy 输出无意义的 100+ char 文本拿 think bonus | Think bonus 条件给分：连续 think 不给，think 后犯错不给；v4-optimal 将 bonus 从 0.05 降到 0.01 | 监控 `reasoning_tokens_per_turn` 的语义质量（人工抽检） |
| **Success bonus 累积（模式 10）** | 长 failure trajectory 因 step 数多自动累积正分 | **已移除** success_bonus(+0.01)；改用 mean-based 聚合 | 离线验证：长 failure 的 process_score 应为负 |
| **Implicit think 伪造（模式 13）** | 写 100+ char 纯文本触发 implicit_think 奖励 | **已移除** implicit think 奖励；只保留显式 think tool | 监控 implicit_think 比例 |
| **Data chain 误匹配（模式 12）** | 无关字符串被误判定为 data chain | 采用 schema-based entity extraction（regex 提取有效 entity），非 raw text matching | 离线验证：检查误匹配率 |
| **Process score 范围漂移** | Policy 找到 exploit 使 process_score 持续为正但 outcome 为 0 | Clamp [-0.5, +0.5] + mean-based 聚合；监控 `critic/score/mean` 分布 | 若 score/mean 持续 >0.3 但 pass^1 不升，则存在漂移 |
| **Error rate 上升（模式 2）** | PRM 鼓励探索 → 更多工具幻觉 | Tool 白名单 + runtime KeyError catch；P1/P2 placeholder 惩罚；KL loss 约束 | 监控 `error_rate` 和 `total_errors/mean` |
| **Response length 膨胀（模式 9）** | Policy 写更长 reasoning 拿 think bonus / 逃避 placeholder 惩罚 | P8 length penalty（ntools>8 后 -0.01/步）；LATA（Exp 2）的 sqrt(L) 归一化 | 监控 `response_length/mean` 和 `clip_ratio` |
| **「以量补质」转移到 process score** | Policy 通过多步「合规但无效」的 read tool 拿 first-read / diversity 分 | first-read 降至 +0.01；diversity 降至 +0.01；P8 length penalty | 监控 `tool_calls/mean` 和 `num_turns/mean` |
| **过早 escalate（模式 6）** | 无 read 就 transfer_to_human 逃避困难 task | P5 premature escalation 惩罚 -0.10 | 监控 escalation 比例 |

**五层防御体系总结**：

| 防线 | 措施 | 针对的 Hacking 模式 |
|---|---|---|
| **L1: 奖励设计** | mean-based + clamp + 移除反激励规则 | 模式 9, 10 |
| **L2: 防 hacking 规则** | 条件 think bonus / placeholder 检测 / redundancy 检测 / escalation 惩罚 / no-reasoning 惩罚 / length 惩罚 | 模式 1, 3, 4, 5, 6, 7 |
| **L3: Advantage 结构** | Turn-Discount / LATA（Exp 1/2/4） | 模式 1, 14 |
| **L4: 约束层** | KL loss / max_response_length / max_turns / Tool 白名单 | 模式 2, 8, 9, 13 |
| **L5: 监控层** | error_rate / reasoning_tokens / clip_ratio / score 分布 / grad_norm | 全部模式 |

### 3.5 预期指标（300 step）

| 指标 | Vanilla 基线 | 预期值 | 预期变化 |
|---|---|---|---|
| overall pass^1 @step300 | 0.175 (step200) | **0.50-0.70** | 打破死锁后显著提升 |
| 泛化 pass^1 @step300 | 0.071 (step200) | **0.12-0.15** | 核心目标： unseen 区域获得有效梯度 |
| per_turn p50 @step300 | ~30-50 | **70-100** | No-reasoning penalty 强制保留 reasoning |
| Δstability (step300−step150) | −0.050 | **≥ −0.02** | Process signal 提供持续梯度，不易 saturate |
| num_turns/mean (val) | 5.08 (step200) | **20-28** | P8 length penalty 抑制过度试错 |
| error_rate @step300 | 0.200 | **0.15-0.25** | 探索期可能上升，后期应回落 |
| `critic/score/min` | 频繁 = 1.0 | **从不 = 0.0 或 1.0** | Process score 打破死锁 |
| `critic/score/mean` 分布 | 0.85-1.0 | **0.1-0.3 波动** | 中间值丰富，有正有负 |
| unseen reward @step300 | 0.150 (step150) | **≥ 0.05** | 从负值或接近 0 改善 |

---

## §4 Exp 2: LATA（Length-Aware Turn-weighted Advantage）

### 4.1 问题定义

Turn-discount（Exp 1）解决了「早期 token 被晚期稀释」的问题，但还有一个更深层的问题：

标准 GRPO 的 **per-token 梯度与 response length L 成反比**。在 PPO/GRPO 的 loss 中，token-level gradient 约等于 `A / L`（A 是 advantage 标量，L 是 response length）。这意味着：

| Response length | Per-token gradient | 激励效果 |
|---|---|---|
| L = 1000 | A / 1000 | 基准 |
| L = 2000 | A / 2000 | **减半** |
| L = 4000 | A / 4000 | **四分之一** |

这个**线性稀释**会隐性惩罚长 reasoning。即使 turn-discount 保护了早期 token 的权重，如果模型选择写更长的 reasoning 链，每个 token 的激励仍然被 L 线性削弱。长期下来，policy 仍然有动机压缩 reasoning。

### 4.2 改进设计

**LATA = Turn-Discount + Length-Aware Normalization**

两阶段：
1. **Turn-discount**：`weight[t] = alpha^(L-1-t)`，保护早期 reasoning（与 Exp 1 相同）
2. **Length-aware normalization**：`advantage = A × weight / sqrt(L)`

**为什么除以 sqrt(L) 而不是 L？**

数学上，如果我们希望「写 2 倍长的 reasoning」的**总梯度激励**与「写 1 倍长」相当：
- 线性归一化（标准 GRPO）：总梯度 ∝ L × (A/L) = A（与 L 无关）→ **边际激励为 0**
- sqrt 归一化（LATA）：总梯度 ∝ L × (A/sqrt(L)) = A × sqrt(L) → **边际激励为正**

换一种理解：标准 GRPO 让每个 token 的梯度是 A/L，LATA 让每个 token 的梯度是 A/sqrt(L)。当 L 从 1000 涨到 4000 时：
- A/L 从 A/1000 降到 A/4000（×0.25）
- A/sqrt(L) 从 A/31.6 降到 A/63.2（×0.50）

**次线性衰减**保留了「愿意写更长 reasoning」的激励。

**完整代码**（`verl/trainer/ppo/core_algos.py`）：

```python
@register_adv_est("grpo_lata")
def compute_grpo_lata_outcome_advantage(
    token_level_rewards, response_mask, index, epsilon=1e-6,
    norm_adv_by_std_in_grpo=True, config=None
):
    # Step 1: 标准 GRPO group normalize
    scores = token_level_rewards.sum(dim=-1)
    # ... group mean/std normalization ...

    # Step 2: turn-discount 权重（与 Exp 1 相同）
    alpha = 1.05
    resp_len = response_mask.shape[1]
    positions = torch.arange(resp_len, dtype=torch.float64)
    active_lengths = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
    exponents = active_lengths - 1 - positions.unsqueeze(0)
    log_weights = exponents * math.log(alpha)
    # ... stable normalization ...
    weights = (weights_stable * active_count / weight_sum).to(torch.float32)

    # Step 3: length-aware normalization（LATA 新增）
    length_norm = torch.sqrt(active_lengths).to(torch.float32)
    scores = scores.unsqueeze(-1) * weights * response_mask / length_norm

    return scores, scores
```

**配置启用**（`configs/train/grpo/lata.yaml`）：

```yaml
algorithm:
  adv_estimator: grpo_lata
  turn_discount:
    enable: true
    alpha: 1.05
```

### 4.3 Reward Hacking 风险与防御

**LATA 可能引入的 hacking 风险**：

| 风险 | 机制 | 防御措施 |
|---|---|---|
| **早期 token 梯度过大** | Turn-discount 权重 + sqrt(L) 归一化叠加后，前 20% token 的梯度可能比标准 GRPO 大 5-10 倍 | 监控 `actor/grad_norm` 前 10 步；若 >0.5 或 nan，调低 alpha 或调 lr |
| **过度追求长度** | sqrt(L) 归一化使长 trajectory 的总梯度激励 ∝ sqrt(L) → policy 有动机写更长 | 与标准 GRPO 的「零边际激励」相比，这反而是设计目标；P8 length penalty（PRM-Lite）和 max_response_length=12288 作为上限约束 |
| **晚期必要交互被过度惩罚** | 两个机制叠加后，晚期 token 的权重极低且被 sqrt(L) 进一步压缩 | 监控 `num_turns/mean` 若 <15 则判断为过度压缩；必要时调低 alpha |
| **与 PRM-Lite 的 length penalty 冲突** | LATA 鼓励长 reasoning，PRM-Lite P8 惩罚长 trajectory | 两者作用于不同对象：LATA 保护「高质量的长 reasoning」，P8 惩罚「低质量的长试错」；预期正交而非冲突 |

### 4.4 预期指标（300 step）

| 指标 | Vanilla 基线 | 预期值 | 预期变化 |
|---|---|---|---|
| overall pass^1 @step300 | 0.175 (step200) | **0.70-0.80** | 阻止 collapse，效果接近 Turn-Discount |
| 泛化 pass^1 @step300 | 0.071 (step200) | **0.10-0.12** | 对 OOD 无直接帮助 |
| per_turn p50 @step300 | ~30-50 | **≥ 100** | **核心预期**：sqrt(L) 保护长 reasoning 激励 |
| Δstability (step300−step150) | −0.050 | **≥ −0.02** | 稳定 |
| num_turns/mean (val) | 5.08 (step200) | **25-30** | 不撞顶 |
| error_rate @step300 | 0.200 | **≤ 0.08** | 稳定 |
| response_length/mean | ~1100 (step200) | **1800-2200** | 比 Turn-Discount 更高，证明 sqrt(L) 有效 |
| `critic/score/min` | 频繁 = 1.0 | **≤ 0.1（极少 =1.0）** | Group saturation 意外缓解（同 Exp 1） |

---

## §5 Exp 4: PRM-Lite + LATA（联合方案）

### 5.1 设计动机

Exp 1/2（Turn-Discount/LATA）和 Exp 3（PRM-Lite）分别解决了不同的问题：

| 实验 | 主要解决的问题 | 对另一类问题的作用 |
|---|---|---|
| Turn-Discount/LATA | §3.3 reasoning 退化 | 对 §3.1/§3.2 无直接帮助 |
| PRM-Lite | §3.1 group saturation + §3.2 泛化 | 对 §3.3 只有间接改善（no-reasoning penalty） |

**联合方案的期望**：
- PRM-Lite 提供细粒度 signal 打破死锁 → policy 能在 uncovered/unseen 区域学到东西
- LATA 保护 reasoning 质量不被长度稀释 → policy 学到的东西是「高质量 reasoning」而非「更多试错」
- 两者效果**叠加** → 联合方案的泛化指标 > 两个单方案各自的泛化指标

如果联合方案的效果只是「取 max」或甚至互相 cancel，说明两个改进命中的是同一个 failure mode 的不同侧面，而非正交问题。

### 5.2 配置

同时启用 PRM-Lite reward 和 LATA advantage（`configs/train/grpo/prm_lite_lata.yaml`）：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      interaction_config_path: configs/interaction_config/tau_bench_airline_prm_lite.yaml

algorithm:
  adv_estimator: grpo_lata
  turn_discount:
    enable: true
    alpha: 1.05
```

**注意**：联合方案用 `tensor_model_parallel_size=2`（与 PRM-Lite 相同），因为 PRM-Lite 的 reward 计算需要保留完整的 trajectory 信息，TP=2 的内存配置更稳定。

### 5.3 Reward Hacking 风险与防御

**联合方案的特殊风险**：两个改进叠加后可能产生**新的 exploit 路径**：

| 风险 | 机制 | 防御措施 |
|---|---|---|
| **PRM-Lite 奖励 + LATA 长度保护 → 超长空 reasoning** | Policy 写极长的无意义 reasoning 文本，既拿 think bonus 又享受 LATA 的长度保护 | Think bonus 条件给分（连续 think 不给，think 后犯错不给）；no-reasoning penalty 针对「无 think」而非「短 think」；人工抽检 reasoning 质量 |
| **Data chain 伪造 + 早期高权重 → 快速刷分** | Policy 在前几步快速调用几个 read tool 拿 first-read / diversity 分，然后 escalate | P5 escalation 惩罚；P8 length penalty（如果后续无有效进展） |
| **梯度量级不可预测** | PRM 改变 reward 分布 + LATA 改变 advantage 分布 → 两者叠加后 grad_norm 可能异常 | 监控 `actor/grad_norm` 前 20 步；必要时调 lr 或 alpha |
| **训练稳定性下降** | 两个改动同时引入更多超参和更多非线性 → 训练曲线可能比单方案更震荡 | 保存 freq 保持 50 step；若 step 50 eval 严重劣化，立即 rollback 分析 |

### 5.4 预期指标（300 step）

| 指标 | Vanilla 基线 | 预期值 | 预期变化 |
|---|---|---|---|
| overall pass^1 @step300 | 0.175 (step200) | **0.75-0.90** | 双管齐下，效果叠加 |
| 泛化 pass^1 @step300 | 0.071 (step200) | **0.14-0.18** | **核心预期**：PRM 打破死锁 + LATA 保护 reasoning 质量 → OOD 显著提升 |
| per_turn p50 @step300 | ~30-50 | **≥ 110** | LATA 保护长 reasoning + PRM no-reasoning penalty 强制保留 |
| Δstability (step300−step150) | −0.050 | **≥ +0.00** | 双信号叠加后更稳定 |
| num_turns/mean (val) | 5.08 (step200) | **22-28** | PRM P8 抑制过度试错，LATA 保留必要交互 |
| error_rate @step300 | 0.200 | **≤ 0.10** | PRM P1/P2 placeholder 惩罚 + P3/P4 冗余惩罚应能降低幻觉 |
| `critic/score/min` | 频繁 = 1.0 | **从不 = 0.0 或 1.0** | Process score 打破死锁 |
| unseen reward @step300 | 0.150 (step150) | **≥ 0.10** | **核心预期**：从负值/接近 0 显著改善 |

---

## §6 评测协议与成功标准

### 6.1 评测配置（所有方案共享）

```python
VLLMPolicy(
    model_name=<checkpoint_path>,
    base_url="http://localhost:8000/v1",
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096,    # 🔥 必须从 1024 升到 4096（防御性配置）
)

run_eval(
    wrapper=TauBenchWrapper(env_name="airline", user_model="qwen-72b-awq"),
    policy_factory=...,
    num_tasks=50,
    num_samples_per_task=4,
    max_turns=30,
)
```

### 6.2 主指标

| 指标 | 定义 | Vanilla step200 | 改进目标 |
|---|---|---|---|
| **泛化 pass^1** | (uncovered_seen×24 + unseen×10) / 34 | 0.071 | **≥ 0.13** |
| **per_turn p50** | trajectory 中 assistant content token 的 p50 | ~30-50（SFT 水平） | **≥ 100** |
| **Δstability** | pass^1@step300 − pass^1@step150 | −0.050 | **≥ −0.02** |
| **error_rate** | @step300 | 0.200 | **≤ 0.05** |

**为什么不再用 overall pass^1**：covered_seen 16 task 占 eval 总样本量 32%，且其上的提升大部分是训练集泄漏。overall 指标会掩盖改进方案在 OOD 上的真实效果。

### 6.3 四个实验的预期结果汇总

| 实验 | 预期泛化 pass^1 | 预期 per_turn p50 | 预期 Δstability | 预期 unseen reward | 核心理由 |
|---|---|---|---|---|---|
| Vanilla | 0.071 | ~50 | −0.050 | ~0.15 | 基线 |
| Turn-Discount | ~0.10 | ~80 | −0.01 | ~0.15 | 阻止 reasoning collapse，但对 OOD 无直接帮助 |
| PRM-Lite | ~0.12 | ~70 | +0.00 | ~0.05 | 打破死锁，在 unseen 区域提供梯度 |
| LATA | ~0.11 | ~100 | −0.01 | ~0.15 | 保护长 reasoning 激励 |
| **PRM-Lite + LATA** | **~0.15** | **~110** | **+0.02** | **~0.10** | 双管齐下，正交叠加 |

---

## §7 实施计划

### 7.1 实验序列

```raw
Phase 1: Exp 1 Turn-discount-adeanyage
Phase 2: Exp 2 LATA 单方案训练（300 step, ~70h）
Phase 3: Exp 3 PRM-Lite 单方案训练（300 step, ~70h）
Phase 4: Exp 4 PRM-Lite + LATA 联合训练（300 step, ~70h）
Phase 5: 所有方案统一 eval + 汇总 ablation 表
```

### 7.2 风险与触发条件

| 触发条件 | 处理 |
|---|---|
| Step 50 时 critic/score/mean < 0.1 | 训练失效，检查 reward_fn 是否漏接入 |
| Step 100 时泛化 pass^1 < 0.05 | 严重退化，怀疑 reward shaping 引入 noise，检查 process_score 量级 |
| `actor/grad_norm` 在 step < 50 内出现 nan | 立即停 + 打 dump checkpoint，可能是新 advantage/reward 计算的数值不稳 |
| eval 时 per_turn p50 < 30 | reasoning 已退化到 SFT 水平，方案完全失效，跳过该方案直接进下一个 |
| 单 step wall time 超 1500s（vanilla 的 1.8×） | reward_fn 计算开销过大，需要 profile 并优化（rule-based 应该 < 10ms / trajectory） |
| error_rate 持续上升且 >0.40 | PRM 探索失控，需检查 placeholder/redundancy 惩罚是否生效 |
| response_length/clip_ratio >0.20 | 大量样本被截断到 12288，需考虑降低 max_prompt_length 或调优生成参数 |

