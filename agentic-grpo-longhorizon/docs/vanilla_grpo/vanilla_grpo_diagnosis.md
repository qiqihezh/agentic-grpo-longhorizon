# Vanilla GRPO 诊断报告与消融实验动机

> **文档定位**：本文是项目的「诊断书 + 路线图」。前半部分基于 SFT warmup 后的 vanilla GRPO 完整训练，独立诊断 GRPO 在长链路多工具 agent 场景下的核心问题；后半部分汇报已完成的消融实验（Turn-Discount、PRM-Lite 初步）对这些问题的缓解效果，并引出剩余实验（LATA、PRM-Lite + LATA）的设计动机。
>
> **目标读者**：任何没有参与过本项目的人。读完本文应能独立回答：
> 1. 这个项目在解决什么问题？
> 2. 标准 GRPO 出了什么问题？
> 3. 四个消融实验各自改了什么、为什么改、期望解决什么问题？
> 4. 已做的实验效果如何？
> 5. Reward hacking 有哪些表现形式、为什么会发生、怎么防？

---

## §0 项目背景：我们为什么做这件事

### 0.1 任务：τ-bench Airline

τ-bench 是一个**长链路多工具对话 agent** 评测基准。Airline 子任务模拟航空公司客服场景：用户提出需求（如改签、退票、查询），policy（7B LLM）需要通过多轮对话调用工具（查航班、查预订、改乘客信息等）来解决问题。

关键难点：
- **长链路**：一个任务平均需要 10-30 轮交互，涉及 13 种工具
- **工具依赖**：后续 tool call 的参数必须从前面的 observation 中提取（如先 `get_user_details` 拿到 `reservation_id`，再用它 `get_reservation_details`）
- **用户 simulator**：用 72B-AWQ 模型扮演用户，对 policy 的回复进行自然语言回应，增加不确定性

### 0.2 方法：GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek-R1 背后的核心 RL 算法。与 PPO 不同，GRPO **不需要训练独立的 value network**，而是对每个 prompt 采样一个 group（如 8 条 rollout），用 group 内的 reward 分布估计 advantage：

```raw
A_i = (r_i - mean(r_group)) / std(r_group)
```

这很优雅，但有一个隐藏假设：**group 内的 reward 必须有足够的 variance**。如果 8 条 rollout 全是 0（全失败）或全是 1（全成功），`std → 0`，advantage 退化为 0，policy 学不到任何东西。

### 0.3 三阶段基线

| 阶段 | 方法 | 核心指标 pass^1 | 说明 |
|---|---|---|---|
| W1 | Base 7B（无训练） | 0.160 | 直接让 7B 模型上，自然行为 |
| W2 | SFT（监督微调） | 0.145 | 用 72B 教师的轨迹蒸馏 7B |
| W3 | Vanilla GRPO | 0.225 → **0.175** (塌) | 在 SFT 基础上用 GRPO 强化 |

一个反直觉的发现：**SFT 后性能反而比 base 下降了**（0.160 → 0.145）。这说明 7B 模型在模仿 72B 教师时，由于容量不足，发生了「模式压缩」——只学到了 tool call 的形式，丢掉了 reasoning 内容。GRPO 的目标就是把被压缩的 reasoning 「拉回来」，但 vanilla GRPO 在 200 步内就出现了 collapse。

---

## §1 实验设置

### 1.1 训练配置

| 项 | 值 | 说明 |
|---|---|---|
| 算法 | GRPO | `algorithm.adv_estimator=grpo`，`use_kl_loss=True`，`kl_loss_type=low_var_kl` |
| Policy 起点 | `experiments/sft_lora_merged` | W2 LoRA SFT checkpoint，r=16/α=32 |
| Group size | n=8 | 每个 prompt 采 8 条 rollout |
| Train batch | 4 prompt × 8 = 32 rollout/step | `train_batch_size=4`, `ppo_mini_batch_size=4` |
| KL coef | 0.01（固定） | `kl_coef=0.01`，`kl_loss_coef=0.01` |
| Actor lr | 5e-6 | LoRA Adam |
| 目标 step | 500 | **实际训到 225 step 终止**（见 §1.4）|
| Save / eval freq | 50 step | 5 个 eval checkpoint：50/100/150/200(/225) |

### 1.2 数据划分（关键约束）

τ-bench airline 官方只有 **50 个 task**。数据量小是本项目最核心的结构性约束。

| 划分 | task 数 | 说明 |
|---|---|---|
| **train.parquet** | 40 task | 训练集 |
| **eval（50 task）** | 16 covered_seen + 24 uncovered_seen + 10 unseen | 评测集 |
| **train ∩ eval** | 40 task（covered_seen 16 + uncovered_seen 24） | 训练与评测必然重合 |
| **真正 OOD** | 10 unseen task | 训练时完全未见过 |

**重要后果**：train 与 eval 重合 40/50 = 80%。covered_seen（在 SFT 阶段就被 72B 成功示教过的 task）上的提升**大概率包含训练集泄漏**，不能作为「GRPO 算法贡献」的证据。因此后续消融实验的核心指标统一用 **uncovered_seen + unseen**（「泛化指标」）。

### 1.3 终止依据：为什么 225 step 而非 500 step

1. **`actor/grad_norm` 衰减到接近 0**：从 step 0 的 0.05-0.10 单调降到 step 200 后的 0.005-0.01
2. **`critic/score/mean` 已 saturate**：step 80 之后稳定在 0.85-1.0 高位震荡
3. **eval pass_hat_1 在 step 150 触顶后回落**：step 150 = 0.225（peak），step 200 = 0.175（−0.050）

**vanilla GRPO 在该设置下 200 step 内即耗尽学习容量**。

---

## §2 Vanilla GRPO 的崩溃表现

### 2.1 五个 checkpoint 的 eval 指标

| Checkpoint | overall<br>pass^1 | covered_seen<br>pass^1 | uncovered_seen<br>pass^1 | unseen<br>pass^1 | avg_turns | error_rate |
|---|---|---|---|---|---|---|
| **Base 7B**（W1） | 0.160 | — | — | — | 12.30 | 0.000 |
| **SFT**（W2） | 0.145 | 0.328 | 0.021 | 0.150 | 14.43 | — |
| GRPO step50 | 0.220 | 0.469 | 0.073 | 0.175 | 9.38 | 0.016 |
| GRPO step100 | 0.205 | 0.484 | 0.042 | 0.150 | 7.69 | 0.078 |
| **GRPO step150**（peak） | **0.225** | 0.453 | **0.094** | 0.175 | 7.30 | 0.010 |
| **GRPO step200**（collapsed） | 0.175 | 0.391 | 0.042 | 0.150 | **5.08** | **0.200** |

三组核心观察：
1. **「pass^1 涨而泛化不涨」**：overall 从 0.145 涨到 0.225，但 covered_seen +0.13、uncovered_seen +0.07、unseen +0.025——收益主要来自训练集泄漏
2. **「先涨后塌」**：step 150 peak 后 step 200 劣化 −0.050，avg_turns 从 7.3→5.1（−30%），error_rate 从 0.010→0.200（×20）
3. **「train num_turns 涨 vs eval avg_turns 跌」**：训练时 num_turns 从 28 涨到 30 撞顶，评测时 avg_turns 从 12.3 单调跌到 5.08

### 2.2 Reasoning 长度的三阶段演化

| 阶段 | mean | p50 | p95 | =1024 截断率 | 解读 |
|---|---|---|---|---|---|
| **Base 7B** | 83 | 63 | 225 | 0.00% | 短解释、直接调 tool |
| **SFT** | **59** | **22** | 255 | 0.14% | 比 base 还短——p50=22 token 几乎只说 "Done." |
| **GRPO step150**（peak） | TBD | TBD | TBD | TBD | 早期把 reasoning 拉回 ~150-300 token |
| **GRPO step200**（collapsed） | TBD | TBD | TBD | TBD | 回退到 SFT 形态 |

**关键发现**：SFT 阶段 reasoning 比 base 更短（p50 22 vs 63）。72B 教师的成功路径是「长 reasoning + 准确 tool call」，7B 学生容量不够，被迫压缩 reasoning 只保留 tool call 框架。GRPO 在做 "anti-SFT-compression" 的工作，但这个工作不稳定，200 步内就回退了。

---

## §3 核心问题诊断（3 类）

### 3.1 Group Reward Saturation 与梯度消失（双向死局）

**现象**：
- `critic/score/mean`：step 0→80 从 0.1 涨到 0.85；step 80→225 在 0.5-1.0 高位震荡
- `critic/score/max`：step 50 之后几乎全程 = 1.0
- `critic/score/min`：step 0-80 几乎全 = 0；step 80 之后**频繁出现 1.0**（从 0/8 全失败变成 8/8 全成功）
- `actor/grad_norm`：step 100 之后单调衰减到 0.005-0.01

**机制**：

GRPO advantage 计算 `A_i = (r_i - mean(r_group)) / std(r_group)`。当 group 内 8 条 rollout 的 reward 同号集中（全 0 或全 1），`std → 0`，**advantage 退化为 0** —— 梯度消失。

这是**双向死局**：
- **低端**：group 内全失败 → std=0 → advantage=0
- **高端**：policy 把 40 个 train task 学到 saturate → group 内频繁 8/8 全成功 → std=0 → advantage=0

**两端通向同一个 failure mode：group variance → 0 → advantage → 0 → grad_norm → 0**。

> ⚠️ 结构性约束：train.parquet 仅 40 task，τ-bench airline 官方只有 50 task，无法扩大 train pool。所以「加数据」不是可选解法，必须从算法层面打破死锁。

---

### 3.2 GRPO 收益严重偏向训练集泄漏部分

**现象**（来自 §2.1）：

| 划分 | SFT pass^1 | GRPO peak (step150) | Δ |
|---|---|---|---|
| covered_seen（16 task，train 子集） | 0.328 | 0.453 | **+0.125** |
| uncovered_seen（24 task，train 子集） | 0.021 | 0.094 | +0.073 |
| unseen（10 task，OOD） | 0.150 | 0.175 | +0.025 |

**机制**：

- **covered_seen**：SFT 阶段就被 72B 成功示教过，policy 已有「成功路径模板」。GRPO 阶段 group 内大概率有成功样本 → advantage 正常 → pass^1 大涨
- **uncovered_seen**：SFT 阶段 72B 也搞不定，group 内大概率全失败 → advantage ≈ 0 → policy 学不到任何东西
- **unseen**：完全 OOD，+0.025 几乎是噪声水平

**结论**：GRPO 的收益**严重依赖 group 内有正样本**。在 group sparse 区域（uncovered_seen + unseen），vanilla GRPO 几乎无效。配合 §3.1 的 grad_norm 衰减证据，这是**比「reward 稀疏」更强的论点**：不是单纯「reward 信号少」，而是「**group 内方差结构性塌陷**」。

---

### 3.3 Per-turn Reasoning 退化驱动的训练曲线先涨后塌

**现象一：训练曲线「先涨后塌」**

overall pass^1：base 0.160 → SFT 0.145 → step50 0.220 → step100 0.205 → step150 **0.225 (peak)** → step200 **0.175 (−0.050)**

**现象二：train num_turns 撞顶 vs eval avg_turns 跌**

| 阶段 | step 0 | step 100 | step 200 | 趋势 |
|---|---|---|---|---|
| **Train** num_turns/mean | ~28 | ~30 | ~30 | 单调上升撞 30 上限 |
| **Eval** avg_turns | 12.30→14.43→9.38 | 7.69 | **5.08** | 单调下降 |

**现象三：response_length/mean 急剧缩短**

| step | 0 | 50 | 100 | 200 |
|---|---|---|---|---|
| `response_length/mean` | ~2500 | ~3000 (peak) | ~1500 | **~1100** |
| `response_length/clip_ratio` | 0-3% | **15.5% (peak)** | 1-5% | 0% |

**现象四（关键 case study）：Step 200 task 49 失败 trajectory**

一条 11 turn 的 trajectory：
- Turn 4：`get_reservation_details("previous_reservation")` → **把语义占位符当字面 ID**
- Turn 6：拿到 4 个 reservation_id 后，**重复调一次同样的 `get_user_details`**
- Turn 7：又调 `get_reservation_details("previous_reservation")` → **重蹈覆辙，没学到**

**统一机制**：以上四个现象由**一个底层问题**驱动 —— **per-turn reasoning 退化**。

| 表象 | 同一机制下的具体表现 |
|---|---|
| Train num_turns 撞 30（「以量补质」）| 单轮 reasoning 不充分 → 多调几次 tool 试错；group=8 让「撞够 30 turn 总能撞对一个」成立 |
| Eval avg_turns 跌到 5.08（trajectory 早死）| reasoning 不充分 + N=4 没有 group 试错空间 → tool call 出错 → trajectory 提前死 |
| response_length 从 3000→1100 | 单轮 content token 数被压缩，接近 SFT 形态 |
| Task 49 占位符错误 | policy 没有在 reason 「previous trip 是占位符，需要先查 reservation 列表」 |

**为什么会发生 reasoning 退化**：在 outcome-only reward + group sampling 下，policy 在两条路径之间被推向后者：

1. **每轮充分 reasoning + 准确 tool call**：1-2 次 tool call 解决 → reward = 1
2. **每轮短 reasoning + 多 tool call 试错**：3-5 次 tool call 偶尔也撞对 → reward = 1

两条路径在 outcome-only reward 下**期望相等**（甚至后者在 group=8 多次试错下期望更高），所以 policy 自然漂向「短 reasoning + 多 tool call」。这条路径在 train 上有效（covered_seen 上反复见过的 task 总能撞对），在 eval 上无效（OOD task 一旦 tool call 出错就没有补救机制）。

**Train num_turns 涨和 eval avg_turns 跌**不是行为分裂，而是**同一个 reasoning 退化机制在两种采样条件下的不同 endpoint**。

---

## §4 Reward Hacking：表现形式、成因与防御体系

> 本节系统梳理长链路 agent + GRPO 场景下 reward hacking 的所有已知风险点。所有条目均基于本项目实际观察（vanilla 训练日志、PRM-Lite 训练日志、eval trajectory）或从机制出发的严密推导，不胡编乱造。

### 4.1 什么是 Reward Hacking

Reward hacking 指 policy **找到一条「绕过任务真实目标、直接最大化 reward」的捷径**。在 outcome-only 的 RL 中，这表现为：policy 不做「正确的事」，而做「看起来容易拿到高分的事」。

在 τ-bench airline 中，hacking 的判定标准是：**trajectory 在训练时因 group sampling 或 reward 设计缺陷拿到正分，但在独立 eval（N=4，无 group 容错）时失败**。

---

### 4.2 已观察到的/可推导的 Hacking 模式（14 种）

#### 模式 1：「以量补质」——用更多 tool call 撞概率

**表现**：单轮 reasoning 极短，但频繁调用 tool（5-10 次），靠 brute-force 试错蒙对。

**成因**：
- outcome-only reward 不区分「1 次精准 call」和「10 次瞎蒙 call」
- group=8 的统计效应：即使单次成功率仅 15%，8 条 rollout 中至少一条成功的概率也高达 73%
- 训练集仅 40 task，covered_seen 上 policy 反复见过相同 task，「撞对」概率更高

**实际观察**：vanilla step 150 时 train num_turns/mean=30（撞顶），eval avg_turns=7.3；step 200 eval avg_turns 跌到 5.08，但 train num_turns 仍维持 30——说明 train 上的「成功」是靠 group 内多次试错撑出来的。

---

#### 模式 2：工具幻觉（Tool Hallucination）

**表现**：生成不存在的 tool 名，如 `get_payment_method_details`、`update_reservation_status`、`update_reservation_details`。

**成因**：
- Tool name 不在 model vocab 的显式约束中（vocab 里什么 token 都可以生成）
- 虽然工具层会 catch 为 `KeyError → Error: Unknown action`，但这个 error 反馈到 policy 时已经是 1-2 轮之后
- Outcome-only reward 不惩罚中间步骤的幻觉，只有最终失败时才给 0 分——而 group=8 可能让其他 rollout 成功，导致幻觉 rollout 的 advantage 并不显著为负

**实际观察**：Exp 3 (PRM-Lite) step 50 时 error_rate 从 25% 上升到 32.4%，出现 `update_reservation_details`、`update_user_payment_methods` 等幻觉。

---

#### 模式 3：占位符滥用（Placeholder Abuse）

**表现**：把自然语言描述直接当参数传入 tool，如 `"previous_reservation"`、`"my flight"`、`"the user"`。

**成因**：
- Tool schema 在训练时没有被显式约束到 policy 的生成过程中
- 72B user simulator 的自然语言回复中包含大量描述性文本，policy 可能误把这些文本当有效参数
- Outcome-only reward 不惩罚「语法上有效但语义上错误」的 tool call——只有等到 tool 执行失败（或 user 指出错误）时才反馈

**实际观察**：vanilla step 200 task 49 中，turn 4 和 turn 7 两次传入 `"previous_reservation"` 作为 `reservation_id`，直接导致后续推理链断裂。

---

#### 模式 4：重复调用同一工具（Redundancy）

**表现**：在已经获得所需信息的情况下，重复调用同一个 tool 且参数完全相同。

**成因**：
- 没有中间步骤惩罚时，「再试一次」是零成本策略
- Group sampling 下，冗余调用可能偶然改变状态（如 user simulator 的不同回复），使得某一条 rollout 成功
- Policy 可能学会「多 call 几次总能拿到想要的结果」

**实际观察**：vanilla step 200 task 49 turn 6，policy 在已经拿到 user details 后，又调了一次完全相同的 `get_user_details(emma_kim_9957)`。

---

#### 模式 5：犯错后不换策略（Error Repetition）

**表现**：上一步 tool call 返回 error，本步用完全相同的 tool 和参数再 call 一次，期望「这次能成功」。

**成因**：
- Outcome-only reward 不区分「犯错后纠正」和「犯错后重复」，policy 没有动力去分析 error 原因
- 若 group 内其他 rollout 成功，这条重复错误的 rollout 的 advantage 只是略低于均值，不构成强烈负反馈

**实际观察**：vanilla step 200 task 49 turn 7，turn 4 的占位符错误已经导致 error，turn 7 又重复了完全相同的错误调用。

---

#### 模式 6：过早 Escalate（Lazy Give-up）

**表现**：task 刚开始 1-2 轮，policy 还没尝试任何 read tool，就直接 `transfer_to_human_agents`。

**成因**：
- Escalate 在某些 task 上可能拿到 partial credit（如用户确实要求转人工）
- 在 outcome-only reward 下，「快速放弃」和「努力解决但最终失败」都是 0 分，但「快速放弃」消耗的 token 更少，policy 有动机选择成本更低的路径
- Group sampling 下，如果其他 7 条 rollout 成功，这条 escalate 的 rollout 的 advantage 接近 0，没有强烈负反馈

**防御**：PRM-Lite P5 规则对 premature escalation 惩罚 -0.10。

---

#### 模式 7：Think 空话（Think Hacking）

**表现**：Policy 学会在每次 tool call 前输出一段 100+ char 的「 reasoning」，内容空洞（如重复问题描述），目的是拿 think bonus。

**成因**：
- 如果 think 无条件给分，policy 会迅速发现「写废话 → 拿分 → 再调 tool」是最优策略
- 纯文本 reasoning 的质量难以自动评估，给分阈值（如 100 char）容易被 hack

**防御**：PRM-Lite 采用「条件给分」——连续 think 不给分，think 后跟 placeholder/redundancy 也不给分（详见 ablation_plan.md §3.3.5）。

---

#### 模式 8：训练集记忆（Template Memorization）

**表现**：Policy 在 covered_seen task 上重复 SFT 阶段学到的固定解题模板，不根据具体 task 变化调整。

**成因**：
- 40 task 的训练集太小，policy 有能力在 100 步内 memorize 所有 covered task 的成功路径
- GRPO 的 group sampling 在 covered task 上几乎总能产出成功样本，advantage 正常 → policy 被激励沿着固定模板走
- Eval 时若 task 细节略有变化（如不同的 reservation_id），模板化 policy 可能出错

**实际观察**：vanilla step 150 peak 时 covered_seen pass^1=0.453，但 uncovered_seen 仅 0.094——说明「成功」主要来自模板记忆。

---

#### 模式 9：Response Length 填充（Length Padding）

**表现**：Policy 在 reasoning 中写入大量无意义的重复文本或冗余解释，目的是「撑满」response 以获取潜在的 length-based 奖励或避免被截断。

**成因**：
- max_response_length=12288 提供了巨大的填充空间
- 如果存在任何与 length 正相关的奖励信号（如 implicit think 长度奖励），policy 会迅速学会填充
- 即使没有显式 length 奖励，更长的 response 在 group 内可能偶然包含更多关键词，触发出乎意料的正面反馈

**防御**：PRM-Lite P8 length penalty（ntools>8 后 -0.01/步）+ 已移除 implicit think 长度梯度奖励。

---

#### 模式 10：Success Bonus 累积（Long Failure Score Inflation）

**表现**：一条长 failure trajectory（如 15 步），每一步都拿 small success bonus，最终 mean score 反而为正。

**成因**：
- 如果存在「非 error、非 think 的 tool call 就给 +0.01」的规则，长 trajectory 的 step 数多，自动累积高分
- 即使最终 outcome=0，process_score 可能因累积而接近 0 甚至为正，削弱了 failure 的负反馈

**实际案例**：v4-fix 版本中，ntools≥15 的长 failure trajectory 平均 process_score 为 +0.03~+0.05（应为负值）。

**防御**：v4-optimal 彻底移除了 success_bonus(+0.01)。

---

#### 模式 11：User Simulator 随机性套利

**表现**：Policy 对同一 task 的多次 rollout 采用相同的「高风险高回报」策略（如直接 guess 一个 reservation_id），赌 user simulator 的随机回复中有某一次恰好配合。

**成因**：
- 72B user simulator 的回复具有随机性（temperature=0.7），同一 policy 行为在不同 run 中可能得到不同 outcome
- Group=8 的采样放大了这种随机性：即使策略本身不合理，8 次尝试中也可能偶然成功
- Policy 学会「 exploit simulator 的方差」而非「学习真实因果」

---

#### 模式 12：Data Chain 误匹配（False Data Chain）

**表现**：Tool 参数中的某个字符串恰好出现在之前 observation 的无关上下文中（如 observation 提到 "user has 2 reservations: ABC123 and DEF456"，policy 传入 "2" 作为 reservation_id），被误判定为 data chain。

**成因**：
- Data chain 检测基于字符串匹配（而非语义理解）
- Observation 文本中大量无关字符串增加了误匹配概率
- 如果 data chain 奖励过高，policy 会学会「把 observation 中的任意字符串塞进参数」

**防御**：PRM-Lite 采用 schema-based entity extraction（从 observation 中按 regex 提取有效 entity），而非 raw text matching，显著降低误匹配率。

---

#### 模式 13：Implicit Think 伪造（Fake Implicit Think）

**表现**：Policy 在不需要 reasoning 的 turn 中，故意写一段 100+ char 的纯文本，让系统误判为 implicit_think 并给分。

**成因**：
- Implicit think 的检测阈值是 100 char，门槛低
- 100 char 约等于 2-3 句话，policy 很容易生成无意义的填充文本达到阈值
- 若 implicit think 有正奖励，policy 会在每轮都「伪造」一段思考

**防御**：v4-optimal 完全移除了 implicit think 的奖励。只保留显式 `think` tool 的条件奖励。

---

#### 模式 14：晚期 Token 稀释利用

**表现**：Policy 故意在 trajectory 早期做少量 reasoning，然后拖到晚期进行大量无意义 tool call，利用标准 GRPO 中「所有 token advantage 相同」的缺陷。

**成因**：
- 标准 GRPO 的 advantage 是标量，均匀分摊到所有 token
- 晚期 token（试错）和早期 token（reasoning）对 policy gradient 的贡献完全相同
- Policy 没有动机在早期投入高质量的 reasoning

**防御**：Turn-Discount / LATA 给早期 token 更高权重，晚期 token 更低权重，逆转这一激励结构。

---

### 4.3 防御体系总览（五层防线）

| 防线 | 措施 | 针对的 Hacking 模式 | 实施位置 |
|---|---|---|---|
| **L1: 奖励设计层** | PRM-Lite 15 条规则提供 dense step-level feedback | 模式 1, 3, 4, 5, 6, 10 | `tau_bench_interaction.py` |
| **L1: 奖励设计层** | mean-based 聚合 + clamp [-0.5, +0.5] | 模式 9, 10 | `tau_bench_interaction.py` |
| **L1: 奖励设计层** | 移除 success_bonus / implicit think 奖励 | 模式 10, 13 | `tau_bench_interaction.py` |
| **L2: 防 Hacking 规则** | Think bonus 条件给分（连续 think 不给，think 后犯错不给） | 模式 7 | `tau_bench_interaction.py` |
| **L2: 防 Hacking 规则** | Schema-based placeholder 检测（P1/P2） | 模式 3 | `tau_bench_interaction.py` |
| **L2: 防 Hacking 规则** | Redundancy / Error repetition 检测（P3/P4） | 模式 4, 5 | `tau_bench_interaction.py` |
| **L2: 防 Hacking 规则** | Escalation penalty（P5） | 模式 6 | `tau_bench_interaction.py` |
| **L2: 防 Hacking 规则** | No-reasoning penalty + Length penalty（P6/P8） | 模式 1, 9 | `tau_bench_interaction.py` |
| **L3: Advantage 结构层** | Turn-Discount：早期 token 高权重 | 模式 1, 14 | `core_algos.py` |
| **L3: Advantage 结构层** | LATA：sqrt(L) 归一化保护长 reasoning | 模式 9, 14 | `core_algos.py` |
| **L4: 约束层** | KL loss（coef=0.01）限制 policy 偏离 ref | 模式 2, 7, 8, 13 | `ray_trainer.py` |
| **L4: 约束层** | max_response_length=12288（上限约束） | 模式 9 | `config.yaml` |
| **L4: 约束层** | max_turns=30（交互轮数上限） | 模式 1 | `config.yaml` |
| **L4: 约束层** | Tool 白名单 + runtime KeyError catch | 模式 2 | `tool_agent_loop.py` |
| **L5: 监控层** | error_rate 监控 | 模式 2, 3 | swanlab / 日志 |
| **L5: 监控层** | reasoning_tokens_per_turn 监控 | 模式 7, 9, 13 | swanlab / 日志 |
| **L5: 监控层** | response_length/clip_ratio 监控 | 模式 9 | swanlab / 日志 |
| **L5: 监控层** | group outcome 分布监控（score/min/mean/max） | 模式 1, 8, 11 | swanlab / 日志 |
| **L5: 监控层** | grad_norm 监控 | 模式 10（score inflation 导致梯度异常） | swanlab / 日志 |

### 4.4 防御效果评估（基于已有实验数据）

| Hacking 模式 | Vanilla | Turn-Discount | PRM-Lite (50步) | 判断 |
|---|---|---|---|---|
| 模式 1：以量补质 | ✅ 严重（num_turns 撞 30） | ✅ **缓解**（num_turns 稳定 27） | ⚠️ 待观察 | Turn-discount 有效 |
| 模式 2：工具幻觉 | ⚠️ 少量 | ⚠️ 少量 | ⚠️ **增加**（32.4% error_rate） | PRM 探索期副作用 |
| 模式 3：占位符滥用 | ✅ 严重（task 49） | ⚠️ 待观察 | ⚠️ 待观察 | P1/P2 规则应能惩罚 |
| 模式 7：Think 空话 | ⚠️ 待观察 | ⚠️ 待观察 | ⚠️ 待观察 | 条件给分设计中 |
| 模式 8：训练集记忆 | ✅ 严重 | ✅ **仍存在** | ⚠️ 待观察 | 数据量硬约束 |
| 模式 9：Length padding | ⚠️ 轻微 | ⚠️ 轻微 | ⚠️ **风险**（response_length 膨胀） | P8 + LATA 应能缓解 |
| 模式 10：Success bonus 累积 | N/A（vanilla 无 process score） | N/A | ✅ **已消除**（success_bonus 已移除） | v4-optimal 修复 |
| 模式 14：晚期 token 稀释 | ✅ 严重 | ✅ **缓解** | N/A（advantage 未改） | Turn-discount 有效 |

---

## §5 已做实验的验证结果

### 5.1 Exp 1: Turn-Discounted Advantage（已完成，300 step）

**改了什么**：标准 GRPO 中，trajectory 内所有 token 共享同一个标量 advantage。Turn-discount 给**早期 token 更高权重、晚期 token 更低权重**，用指数衰减 `weight[t] = alpha^(L-1-t)`（alpha=1.05），归一化后乘到 advantage 上。

**为什么改**：削弱「以量补质」路径的吸引力——如果拖到末端靠试错撞对，那些晚期 token 的 advantage 权重很低，policy 从中学不到多少；反之，如果早期就通过充分 reasoning 解决问题，早期 token 权重高，policy 会被强烈激励往这个方向走。

**实际效果**（对比 vanilla）：

| 指标 | Vanilla step200 | Turn-Discount step300 | 判断 |
|---|---|---|---|
| overall pass^1 | 0.175 (塌) | **0.80** (training val reward，非独立 eval) | ⚠️ 训练指标，不可直接对比 |
| response_length/mean | ~1100 | **~1678**（后期平均） | ✅ 未 collapse |
| num_turns/mean (val) | 5.08 | **27.0** | ✅ 未撞顶，也未早死 |
| `critic/score/min` | step80+ 频繁 = 1.0 | **97.7% = 0.0** | ✅ group saturation 意外缓解 |
| error_rate | 0.200 | **0.345**（独立 eval） | ⚠️ 高于 vanilla，但稳定无爆发 |

**关键发现**：
- **Turn-discount 成功阻止了 reasoning collapse**。response_length 没有再断崖下跌，num_turns 稳定在 25-29，训练曲线没有「先涨后塌」。
- **副作用：学习启动极慢**。step 0-50 reward 完全没动（0.02 → 0.02），直到 step 100 才跃升到 0.72。这是因为削弱「以量补质」后，policy 在早期找不到替代的成功路径。
- **对 §3.1/§3.2 作用有限**。turn-discount 主要解决 §3.3 reasoning 退化，对 group saturation 和 OOD 泛化没有直接帮助（虽然意外缓解了 group saturation）。

---

### 5.2 Exp 3: PRM-Lite（进行中，step 50/300）

**改了什么**：标准 GRPO 的 reward 只有 outcome（成功=1，失败=0）。PRM-Lite 在 outcome 之外叠加一个 **rule-based process score**（∈[-0.5, +0.5]），最终 reward = `outcome + 0.3 × process_score`。process score 由 15 条规则在 step-level 打分后取平均得到。

**为什么改**：打破 §3.1 的 group saturation 死锁和 §3.2 的训练集泄漏依赖。即使 8 条 rollout 的 outcome 全相同，process score 仍能在 group 内拉出 variance，提供有效的梯度信号。

**初步效果**（step 0 → step 50）：

| 指标 | step 0 | step 50 | 变化 |
|---|---|---|---|
| total val reward | 0.069 | 0.091 | ↑ +32% |
| seen reward | 0.089 | 0.116 | ↑ +30% |
| unseen reward | -0.013 | -0.008 | ↑ 改善但仍为**负** |
| reasoning_tokens/turn | 84.7 | 131.1 | ↑ +55% |
| error_rate | 25.0% | **32.4%** | ↑ 恶化 |
| `critic/score/min` | — | **[-0.053, -0.014]** | ✅ **从未出现 0.0 或 1.0** |

**关键发现**：
- **Group saturation 被非常显著地打破**。训练 50 步，`critic/score/min` 没有任何一步是 0.0 或 1.0（vanilla 中 step 80+ 频繁飚 1.0）。process_score 的连续负向惩罚确保了 group 内始终有 variance。
- **Unseen 泛化仍弱**。unseen reward 虽然相对改善 38%，但绝对值仍是负的。50 步可能还太早——作为对比，exp1 在 50 步时 reward 也是完全没动的。
- **副作用：error rate 上升 + response length 膨胀**。模型在生成更多 reasoning tokens（+55%），但工具幻觉也增加了（25% → 32.4%）。response_length 均值从 ~2700 涨到 ~3500，clip_ratio 经常在 6-15%。

---

## §6 对剩余实验的动机锚定

### 6.1 为什么还需要 LATA（Exp 2）

Turn-discount（Exp 1）缓解了 reasoning 退化，但有一个未被解决的细节问题：

标准 GRPO 中，**per-token 的梯度与 response length 成反比**。假设一个 trajectory 的 advantage 是 A，总 token 数是 L，那么每个 token 的梯度约是 A/L。这意味着：
- 写 1000 token 的 reasoning → 每个 token 的梯度是 A/1000
- 写 2000 token 的 reasoning → 每个 token 的梯度是 A/2000（减半）

这个**线性稀释**会隐性惩罚长 reasoning。即使 turn-discount 保护了早期 token 的权重，但如果模型选择写更长的 reasoning 链，每个 token 的激励仍然被 L 线性削弱。LATA 在 turn-discount 基础上增加 **sqrt(L) 归一化**，把 per-token 梯度从 A/L 变成 A/sqrt(L)，让长 reasoning 的激励衰减更慢（次线性）。

**LATA 的核心改进**（两阶段）：
1. **Turn-discount**：`weight[t] = alpha^(L-1-t)`，保护早期 reasoning
2. **Length-aware normalization**：`advantage = A × weight / sqrt(L)`，避免长 trajectory 被线性惩罚

### 6.2 为什么还需要联合方案（Exp 4: PRM-Lite + LATA）

Exp 1（Turn-discount）和 Exp 3（PRM-Lite）各自解决了不同的问题：
- Turn-discount → §3.3 reasoning 退化（主要）+ §3.1 group saturation（意外缓解）
- PRM-Lite → §3.1 group saturation + §3.2 训练集泄漏

但两者都有副作用：
- Turn-discount 学习启动慢（前 50 步没动静）
- PRM-Lite 导致 error rate 上升和 response length 膨胀

**联合方案的期望**：PRM-Lite 提供细粒度信号打破死锁，LATA 保护 reasoning 质量不被长度稀释，两者效果叠加而非互相 cancel。如果联合方案的效果优于两个单方案，就能证明两个改进是**正交**的——分别命中了不同的 failure mode。

---

## §7 四类实验与三类问题的最终映射

| 实验 | 对应问题 | 核心改动 | 状态 |
|---|---|---|---|
| **Exp 1: Turn-Discount** | §3.3 reasoning 退化 | Advantage 计算：早期 token 权重高，晚期低 | ✅ 已完成（300 step） |
| **Exp 3: PRM-Lite** | §3.1 group saturation + §3.2 训练集泄漏 | Reward 计算：outcome + 0.3×process_score | ⏳ 进行中（50/300） |
| **Exp 2: LATA** | §3.3 reasoning 退化（升级版） | Advantage 计算：turn-discount + sqrt(L) 归一化 | ⏳ 待启动 |
| **Exp 4: PRM-Lite + LATA** | 全部 3 类问题 | Reward + Advantage 同时改 | ⏳ 待启动 |

---

## §8 附录

### 8.1 工程数据自证：bypass_mode 的合理性

W3 工程优化采用 `bypass_mode=true`，跳过 FSDP actor `compute_log_prob`，直接用 vLLM rollout 的 `log_probs` 当 `old_log_probs`。这意味着 PPO ratio = π_θ / π_rollout，而非标准的 π_θ / π_old。

自证证据：

| 指标 | step 0 | step 100 | step 200 | 趋势 |
|---|---|---|---|---|
| `rollout_corr/k3_kl` | 0.005 | 0.0012 | 0.0008 | 单调收敛 |
| `rollout_corr/log_ppl_diff` | -0.025 | -0.008 | -0.005 | 收敛到 0 |
| `rollout_corr/ppl_ratio` | 0.975 | 0.992 | 0.995 | 收敛到 1 |

结论：π_rollout ≈ π_old 得到全程实证。这也解释了为什么 `actor/pg_clipfrac` 和 `actor/ppo_kl` 全程为 0——ratio ≈ 1，clip 永不触发。这是**已解决的工程现象**，不是训练 bug。
