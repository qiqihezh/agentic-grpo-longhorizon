# Agentic-GRPO-LongHorizon 保姆级源码剖析

> **目标读者**：准备用此项目投递暑期实习、需要经受互联网大厂面试官拷打的同学  
> **阅读建议**：建议配合源码文件逐节阅读，遇到代码块时切到对应文件对照  
> **前置知识**：需要了解 Transformer、LoRA、PPO/GRPO 基础概念、Python asyncio、PyTorch FSDP

---

## 目录

1. [项目立意：为什么要做这件事](#1-项目立意为什么要做这件事)
2. [整体架构：三阶段 Pipeline](#2-整体架构三阶段-pipeline)
3. [tau-bench 环境详解](#3-tau-bench-环境详解)
4. [veRL 框架核心机制](#4-verl-框架核心机制)
5. [Week 1：Baseline 评测体系](#5-week-1baseline-评测体系)
6. [Week 2：SFT 数据采集与训练](#6-week-2sft-数据采集与训练)
7. [Week 3：GRPO 训练适配层（项目最核心的工程创新）](#7-week-3grpo-训练适配层项目最核心的工程创新)
8. [配置文件与超参数设计哲学](#8-配置文件与超参数设计哲学)
9. [关键优化决策与踩坑记录](#9-关键优化决策与踩坑记录)
10. [完整数据流：从 Parquet 到 Policy 更新](#10-完整数据流从-parquet-到-policy-更新)
11. [面试常考点与回答思路](#11-面试常考点与回答思路)
12. [附录：文件索引速查](#12-附录文件索引速查)

---

## 1. 项目立意：为什么要做这件事

### 1.1 问题背景

大模型做工具调用（Tool Use）已经有很多工作了，但大多数集中在**单轮或短链路**场景（比如调用一次计算器、查一次天气）。真实业务中，客服、运维、数据分析等场景往往需要 **10-30 轮的长链路交互**，涉及多个工具的串行/并行调用、用户的反复确认、异常处理等。

**τ-bench**（由 Sierra Research 发布）是一个专门评估长链路工具调用能力的 benchmark，包含两个领域：
- **Airline**：航空客服（改签、退票、查订单、搜航班等 14 个工具）
- **Retail**：电商客服（改地址、换货、查库存等）

每个 task 都需要模型与 **LLM-based User Simulator** 进行多轮对话，最终完成一个复杂目标。评测标准非常严格：不是只看对话是否流畅，而是**比对数据库的最终状态**是否与专家标注的黄金轨迹一致。

### 1.2 技术挑战

在这个项目上应用 RL（特别是 GRPO）面临几大挑战：

| 挑战 | 说明 |
|------|------|
| **Reward 稀疏** | 只有 task 完成时才有 reward（0 或 1），中间 10-20 轮没有任何反馈 |
| **Long-horizon Credit Assignment** | 第 5 轮的错误可能导致第 20 轮失败，GRPO 的 outcome reward 很难归因 |
| **LLM User Simulator 开销大** | 需要本地部署 72B 模型做 user sim，每次 env.step 都是一次 LLM 推理 |
| **Tool 状态管理复杂** | 14 个工具共享一个内存数据库（航班、订单、用户），env 实例必须在多轮中保持状态一致 |
| **Context 爆炸** | 长链路 trajectory 可达 8K-16K tokens，容易触发截断/污染 |

### 1.3 项目的核心目标

本项目是一个**系统性工程**：从零开始在 τ-bench airline 子集上，搭建完整的 **Baseline → SFT Warmup → GRPO RL** 三阶段 pipeline，并在此过程中：
1. 解决 τ-bench 与 veRL 框架的异步适配问题
2. 实现长链路场景下的稳定 rollout
3. 产出可诊断的实验数据，为后续改进（reward shaping、PRM、turn-level advantage 等）奠基

---

## 2. 整体架构：三阶段 Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Agentic-GRPO-LongHorizon                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Week 1: Baseline                                                            │
│    Qwen2.5-7B-Instruct (zero-shot) ──► τ-bench airline 50 tasks             │
│    产出: pass^1=0.16, failure case 分析                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Week 2: SFT Warmup                                                          │
│    72B-AWQ best-of-16 采集 ──► LoRA SFT (r=16) ──► merge ──► 评测          │
│    产出: 45 条成功 trajectory, pass^1=0.21 (seen)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Week 3: Vanilla GRPO                                                        │
│    veRL + vLLM V1 + τ-bench adapter ──► 500 step GRPO training              │
│    产出: policy checkpoint, 诊断曲线 (reward/kl/length drift)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 技术栈全景

| 层级 | 组件 | 版本/说明 |
|------|------|-----------|
| **模型** | Policy | Qwen2.5-7B-Instruct + LoRA |
| | User Simulator | Qwen2.5-72B-Instruct-AWQ (vLLM 部署) |
| | Ref Model | Week 2 SFT merged checkpoint |
| **训练框架** | veRL | v0.6.1 release branch，支持 multi-turn + tool + interaction |
| **推理引擎** | vLLM | 0.9.2 (V1 engine，`VLLM_USE_V1=1`) |
| **环境** | τ-bench | airline 子集，14 个工具，50 个 tasks |
| **分布式** | Ray | 2.55.1，管理 AgentLoopWorker + vLLM server replica |
| **微调** | PEFT | LoRA r=16, alpha=32 |
| **数据格式** | Parquet / JSONL | veRL 要求 parquet；SFT 采集用 jsonl |

---

## 3. tau-bench 环境详解

### 3.1 核心抽象

τ-bench 的环境设计非常简洁，核心只有三个概念：

**`Action`**（`tau_bench/types.py`）
```python
class Action(BaseModel):
    name: str      # tool 名 或 "respond"
    kwargs: Dict[str, Any]
```

- `name="respond"` 是一个**伪动作**，表示 agent 向用户说话，`kwargs={"content": "..."}`
- `name="search_direct_flight"` 等表示调用真实业务工具

**`EnvResponse`**（`tau_bench/types.py`）
```python
class EnvResponse(BaseModel):
    observation: str   # 返回给模型的文本（tool 结果 或 user 回复）
    reward: float      # 只有 episode 结束时非零
    done: bool         # 是否结束
    info: EnvInfo      # 元数据
```

**`Env`**（`tau_bench/envs/base.py`）
```python
class Env(object):
    def reset(self, task_index) -> EnvResetResponse:
        # 重新加载数据库，初始化 user simulator
        
    def step(self, action: Action) -> EnvResponse:
        # 如果是 respond → 驱动 user simulator
        # 如果是 tool → 调用 tool.invoke(data, **kwargs)
        # 如果 done → 调用 calculate_reward()
```

### 3.2 Airline 环境的 14 个工具

| 工具名 | 类型 | 作用 | 副作用 |
|--------|------|------|--------|
| `search_direct_flight` | 读 | 按起止地+日期搜直飞 | 无 |
| `search_onestop_flight` | 读 | 搜经停航班 | 无 |
| `get_reservation_details` | 读 | 查订单详情 | 无 |
| `get_user_details` | 读 | 查用户资料（会员等级、支付方式） | 无 |
| `list_all_airports` | 读 | 列出机场代码 | 无 |
| `calculate` | 读 | 安全 eval 计算表达式 | 无 |
| `think` | 读 | 思考工具，返回空串 | 无 |
| `book_reservation` | 写 | 预订航班 | 扣款、写 reservations |
| `cancel_reservation` | 写 | 取消订单 | 退款、改 status |
| `update_reservation_flights` | 写 | 改签 | 价差计算、扣款 |
| `update_reservation_passengers` | 写 | 改乘客信息 | 写 reservations |
| `update_reservation_baggages` | 写 | 改行李数 | 费用计算 |
| `send_certificate` | 写 | 发补偿券 | 新增 certificate 支付方式 |
| `transfer_to_human_agents` | 终止 | 转人工 | **done=True** |

### 3.3 Reward 机制（非常严格）

τ-bench 的 reward 是 **outcome-based 0/1**：

1. **数据库状态匹配**：在 episode 结束时，把 agent 实际执行的所有 tool 动作（跳过 respond）在**全新 env 上重放**，比对最终数据库状态 hash 是否一致
2. **输出信息匹配**：如果 task 要求 agent 在对话中提及某些数值（如总价、退款金额），必须在某次 `respond` 中出现

这意味着：
- 即使对话很流畅，如果数据库改错了（比如少扣了行李费），reward=0
- 即使数据库对了，但如果没说 "refund of $150"，reward=0
- 只有一次性的最终 reward，没有 step reward

### 3.4 User Simulator 的工作原理

`tau_bench/envs/user.py` 中的 `LLMUserSimulationEnv`：

```python
def reset(self, instruction):
    self.messages = [
        {"role": "system", "content": f"You are a user... Instruction: {instruction}..."},
        {"role": "user", "content": "Hi! How can I help you today?"}
    ]
    return self.generate_next_message(self.messages)  # LLM 生成用户第一句话

def step(self, content: str):
    self.messages.append({"role": "user", "content": content})
    return self.generate_next_message(self.messages)
```

关键约束（在 system prompt 中硬编码）：
- 用户**不会一次性透露所有需求**，必须被 agent 询问后才逐步给出
- 用户**不能编造** instruction 中未提供的信息（如不记得 order id 就说不记得）
- 当目标达成时，LLM 会输出 `###STOP###`，`env.step()` 检测到后设置 `done=True`

### 3.5 本项目对 tau-bench 的改动

**注意**：项目本身没有直接修改 tau-bench 的源码（保持可复现性），而是通过**封装层**做适配。但以下情况除外：

根据 `docs/week3_vanilla_grpo_optimization.md` 记录：
- `tau_bench/envs/airline/tools/send_certificate.py`：原版在用户已有最大证书数时可能抛异常，项目做了防御性修复使其返回字符串而非崩溃

其他所有交互都通过 `src/envs/` 下的适配层完成，不动 tau-bench 本体。

---

## 4. veRL 框架核心机制

veRL 是字节跳动开源的 RL 训练框架，主打 **multi-turn + tool-use + async rollout**。本项目使用的是 `release/v0.6.1` 分支。

### 4.1 三大扩展点

veRL v0.6.1 为 tool-use 场景提供了三个可扩展抽象：

| 抽象 | 职责 | 生命周期 | 本项目实现 |
|------|------|----------|----------|
| `AgentLoopBase` | 一条 trajectory 的完整循环 | 每个 prompt 一个实例 | **复用官方 `ToolAgentLoop`** |
| `BaseTool` | 单个 tool 的 schema + 执行逻辑 | 全局单例，通过 `instance_id` 区分并发调用 | `TauBenchToolBase` + 14 个子类 |
| `BaseInteraction` | 模拟用户回复（非 tool 交互） | 全局单例，通过 `instance_id` 区分 | `TauBenchInteraction` |

### 4.2 ToolAgentLoop 状态机

```
PENDING ──► GENERATING ──► PROCESSING_TOOLS ──► GENERATING ──► INTERACTING ──► ... ──► TERMINATED
```

- **PENDING**：`tokenizer.apply_chat_template(raw_prompt, tools=tool_schemas)` 生成初始 prompt_ids
- **GENERATING**：调用 LLM server（vLLM）生成 assistant message
- **PROCESSING_TOOLS**：如果检测到 `tool_calls`，并发执行 tool（`tool.create()` → `tool.execute()` → `tool.release()`），将 tool response 编码为 token ids 追加到 prompt
- **INTERACTING**：如果没有 tool_calls 但有 interaction 配置，调用 `interaction.generate_response()` 获取 user reply，同样追加到 prompt
- **TERMINATED**：达到 max_turns、interaction 返回 `should_terminate=True`、或 LLM 输出终止符

**关键机制**：`response_mask`
- `response_mask=1`：该 token 是 LLM 生成的（参与 policy gradient 计算）
- `response_mask=0`：该 token 是 tool response 或 user reply（**不**参与 gradient，只作为 context）

### 4.3 Async Rollout 架构

```
RayPPOTrainer (driver)
    └── AgentLoopManager
            ├── LLM Server Replicas (vLLM/SGLang)
            └── AgentLoopWorker[0..N] (Ray remote actors)
                    └── asyncio.create_task(_run_agent_loop(...))
                            └── ToolAgentLoop.run()
```

- 每个 worker 内部用 `asyncio.create_task` 为 batch 里的每个 sample 创建独立 coroutine
- 同一条 trajectory 的所有操作（generate → tool → interact → generate...）都在**同一个 asyncio task** 内顺序执行
- **不同 trajectory 之间是并发的**

### 4.4 GRPO Advantage 计算

veRL 的 `compute_grpo_outcome_advantage()`（`verl/trainer/ppo/core_algos.py`）：

```python
def compute_grpo_outcome_advantage(batch):
    # 按 uid (group id) 分组
    for uid in unique_uids:
        rewards = batch[uid].rewards  # shape: (group_size,)
        mean = rewards.mean()
        std = rewards.std() + 1e-6
        advantages = (rewards - mean) / std
        returns = rewards  # GRPO 中 returns = raw rewards
```

- `uid` 来自 parquet 的 `extra_info`，相同 task_id 的 group_size 条 rollout 共享一个 uid
- Advantage 是**组内归一化**的：如果 4 条轨迹中有 1 条成功、3 条失败，成功的那条 advantage ≈ +1.3，失败的 ≈ -0.4
- **只更新 `response_mask=1` 的 token**

### 4.5 本项目对 veRL 的改动

根据 `docs/week3_vanilla_grpo_optimization.md` 的 "已应用的关键 Patch" 表格：

| 文件 | 修改内容 | 原因 |
|------|----------|------|
| `verl/trainer/ppo/ray_trainer.py` | bypass fallback：有 `rollout_log_probs` 时强制跳过 `compute_log_prob` | 消除 FSDP 重算 old_log_prob 的 20-40GB 显存峰值 |
| `verl/trainer/ppo/rollout_corr_helper.py` | `open_dict(policy_loss_config)` 绕过 OmegaConf struct 限制 | 配置系统 bugfix |
| `verl/workers/fsdp_workers.py` | `compute_ref_log_prob` 返回 meta_info 修复 `_is_lora=True` 路径 temperature 丢失 | LoRA + ref model 的 log_prob 计算 bug |
| `verl/experimental/agent_loop/tool_parser.py` | `_repair_json()` 修复截断 JSON | tool call 的 JSON 被截断时的解析修复 |
| `verl/utils/tracking.py` | Swanlab resume 支持 `SWANLAB_RUN_ID` / `SWANLAB_RESUME` | 实验管理集成 |

---

## 5. Week 1：Baseline 评测体系

### 5.1 核心文件

- `src/envs/tau_bench_wrapper.py`：同步评测封装
- `src/models/vllm_policy.py`：基于 OpenAI API 的 policy 实现
- `src/evaluation/pass_k_eval.py`：pass^k 评测框架
- `scripts/02_run_baseline_eval.py`：评测入口
- `scripts/03_analyze_failures.py`：失败归因分析

### 5.2 tau_bench_wrapper.py 逐行剖析

```python
class TauBenchWrapper:
    def run_single_task(self, task_idx, policy, max_turns=30) -> TrajectoryResult:
```

这是整个 Week 1/2 的**核心执行引擎**。它实现了一个简单的同步 turn-based 循环：

**初始化阶段**：
```python
system_content = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    ...
)
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": str(obs_res.observation)},
]
```

**【关键 Hack】Date Grounding**：Qwen2.5-7B 默认把不带年份的日期补全为 **2023**，但 τ-bench airline 的所有航班数据基于 **2024**。如果不注入这条 system message，模型搜航班永远返回 `[]`，task 不可能完成。

**循环体**：
```python
while not done and turn_idx < max_turns:
    assistant_msg = policy(messages)   # 调用 LLM
    messages.append(assistant_msg)
    
    if assistant_msg.get("tool_calls"):
        # 解析 tool_calls，构造 Action，调用 env.step()
        for tc in tool_calls:
            action = Action(name=tc["function"]["name"], 
                          kwargs=json.loads(tc["function"]["arguments"]))
            tool_result = env.step(action)
    else:
        # 无 tool_calls，构造 respond Action
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": ...})
        user_obs = env.step(action)
```

**终止判定**：
```python
success = total_reward >= 1.0   # tau-bench 原生 reward 可能 >1，这里截断
```

### 5.3 vllm_policy.py 逐行剖析

`VLLMPolicy` 是本项目的**"客户端 Policy"**，直接对接 vLLM server 的 OpenAI API。

**消息清洗**（`__call__` 方法）：
```python
for m in messages:
    # 修复 tau-bench 返回的 ['observation', '...'] 元组格式
    if isinstance(m, (list, tuple)) and len(m) == 2:
        role = "user" if m[0] in ["observation", "user"] else "assistant"
        content = str(m[1])
    
    # 过滤 Task 对象泄露
    if "task=Task(" in str(content): continue
    
    # 合并连续同角色消息（vLLM 不允许连续 user/user）
    # 但包含 tool_calls / tool_call_id 的消息不能合并
```

**主动截断**（`_truncate_messages`）：
```python
def _truncate_messages(self, messages, max_chars=35000):
    # 阈值 35000 字符 ≈ 11-12K content tokens
    # 策略：从头部丢弃旧消息，保留 system + 最近 3 条
    # 极端情况下对最后一条做内容级截断
```

为什么需要主动截断？因为 vLLM 的 `max_model_len=16384`，如果消息超过这个长度会报 `context length exceeded` 直接崩溃。与其让 vLLM 崩，不如在客户端**有策略地丢弃旧轮次**。

**污染检测**：
```python
FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]
```

如果 assistant 输出这些标记（意味着模型开始模仿训练数据中的模板格式而非标准 tool-calling 格式），整条 trajectory 被标记为 `was_truncated=True`，后续会被过滤出训练集。

**Tool Call 双通道解析**：
```python
if raw_msg.tool_calls:
    # 标准 OpenAI 格式
    final_tool_calls = [...]
elif "<tool_call>" in content:
    # 正则回退：从 content 中抠出 JSON
    matches = TOOL_CALL_PATTERN.findall(content)
```

### 5.4 pass_k_eval.py 评测指标

```python
@dataclass
class EvalReport:
    pass_at_1: float      # 任意一次成功（只要有 1 次成功就算过）
    pass_hat_1: float     # pass^1: 平均成功率（所有 task 成功率的平均）
    pass_hat_4: float     # pass^4: 连续 4 次都成功的比例（稳定性）
    pass_hat_8: float
```

`pass_at_k` 使用 HumanEval 的无偏估计器：
```python
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

**面试考点**：为什么用 `pass^k` 而不是简单平均？因为长链路任务有随机性，单次成功可能是运气，`pass^4` 更能反映模型的稳定能力。

### 5.5 analyze_failures.py 失败归因

启发式分类规则（按优先级）：
1. `traj.get("error")` → `EXCEPTION`
2. `num_turns >= 29` → `TIMEOUT_OR_LOOP`
3. 最后一条含 "sorry"/"cannot"/"unable" → `MODEL_REFUSED_OR_GAVEUP`
4. tool response 含 "error"/"invalid"/"not found" → `TOOL_CALL_ERROR`
5. `num_turns > 15` → `LONG_HORIZON_DRIFT`
6. 其余 → `UNKNOWN_FAILURE`

Baseline 结果（7B zero-shot）：
- 主要失败原因：`TOOL_CALL_ERROR`（参数错误、JSON 解析失败）和 `TIMEOUT_OR_LOOP`
- `pass^1 = 0.16`，说明 7B 模型在 zero-shot 下基本无法完成复杂航空客服任务

---

## 6. Week 2：SFT 数据采集与训练

### 6.1 核心文件

- `scripts/04_collect_sft_data.py`：Best-of-N 采集
- `scripts/04b_inspect_sft_dataset.py`：验证 loss mask
- `src/training/sft_dataset.py`：multi-turn loss mask 核心
- `scripts/05_sft_train.py`：LoRA SFT 训练
- `scripts/05b_merge_lora.py`：合并 adapter
- `scripts/05c_eval_sft.py`：SFT 后评测

### 6.2 SFT 数据采集策略

**为什么用 72B 采集、7B 训练？**
- 72B-AWQ 的 policy 能力远强于 7B，能产出更多成功 trajectory
- SFT 的本质是行为克隆（Behavior Cloning）：让 7B 模仿 72B 的成功轨迹
- 这是经典的 **Distillation** 思路

**温度采样策略**（`04_collect_sft_data.py`）：
```python
# best_of_n=16
# 第 1 次 temp=0.0 greedy（最稳定的轨迹）
# 第 2-16 次 temp=0.8（多样化）
temps = [0.0, 0.8, 0.8, ...]  # 16 个
```

**断点续跑**：每个 task 产出 `task_XXXX.jsonl` + `task_XXXX.meta.json`，如果文件存在则跳过。

**污染过滤**：
```python
is_contaminated = traj.was_contaminated_from_turn is not None
if is_contaminated:
    contaminated.append((traj, sample_idx, temp))
elif traj.success:
    successes.append((traj, sample_idx, temp))
```

被截断的轨迹进 `*_contaminated.jsonl`，**永不进入 `train.jsonl`**。

**Seen/Unseen 切分**：
```python
# 均匀分层：每 stride 取一个进 holdout
stride = len(task_ids) / holdout_size
holdout_ids = sorted({task_ids[int(i * stride)] for i in range(holdout_size)})
```

40 seen tasks 用于训练，10 unseen tasks 用于测试泛化性。

### 6.3 sft_dataset.py：multi-turn loss mask 核心

这是 Week 2 **技术含量最高的文件**。

**问题**：在 multi-turn trajectory 中，我们只希望模型在 **assistant turn** 上计算 loss，user/system/tool turn 应该被 mask 成 `-100`（PyTorch 的 `ignore_index`）。

**难点**：Qwen2.5 的 chat template 很复杂，assistant turn 不是简单的 `"role": "assistant"` 字符串，而是包含 `<|im_start|>assistant\n...<|im_end|>` 以及各种 tool_calls 渲染。手切 token 很容易 off-by-one。

**解决方案**："渲染两次取 diff"

```python
for ai in assistant_indices:
    # 第 1 次：渲染到 assistant turn 之前（含 generation prompt）
    prefix_text = tokenizer.apply_chat_template(
        messages[:ai], tools=tools, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    
    # 第 2 次：渲染到 assistant turn 之后（不含 generation prompt）
    with_assistant_text = tokenizer.apply_chat_template(
        messages[:ai+1], tools=tools, add_generation_prompt=False
    )
    with_assistant_ids = tokenizer(with_assistant_text, add_special_tokens=False)["input_ids"]
    
    # diff 区间就是本轮 assistant 的所有 token
    start = len(prefix_ids)
    end = len(with_assistant_ids)
    labels[start:end] = full_ids[start:end]
```

**为什么 robust**：不依赖任何硬编码的特殊 token 字符串，无论 chat template 怎么变，两次渲染的 diff 一定准确对应 assistant turn 的 token。

**Sanity check**：
```python
n_label_tokens = sum(1 for x in labels if x != IGNORE_INDEX)
if n_label_tokens < 5:
    return None  # 几乎没东西学，丢弃
```

### 6.4 SFT 训练细节

```python
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)
```

**为什么 target_modules 覆盖 MLP**：
- 只训练 attn 层的话，模型学会的只是"怎么组织 attention"，但 tool-use 需要大量的**逻辑推理**（算价格、比航班、改订单）
- MLP 层负责知识的存储和逻辑变换，对客服任务至关重要

**关键配置**：
- `remove_unused_columns=False`：HF Trainer 默认会删掉 dataset 中不认识的列，但 `labels` 会被误杀
- `gradient_checkpointing_kwargs={"use_reentrant": False}`：与 LoRA 兼容
- 不用 packing：multi-turn 样本之间不能 pack，会污染 loss mask

### 6.5 SFT 结果

- 采集：50 tasks × 16 samples = 800 rollouts
- 成功 task 数：19 / 50（覆盖率 38%）
- 成功 trajectory：67 条
- 进训练集：45 条（过滤污染后）
- SFT 后 seen task pass^1 ≈ 0.21（baseline 0.16，提升 5pt）

---

## 7. Week 3：GRPO 训练适配层（项目最核心的工程创新）

这是整个项目**工程难度最高、面试最容易被深挖**的部分。你需要对以下每个文件的每一行都烂熟于心。

### 7.1 核心文件

- `src/envs/tau_bench_context.py`：contextvar 定义
- `src/envs/tau_bench_tools.py`：14 个 tool 的 veRL 适配
- `src/envs/tau_bench_interaction.py`：user simulator 的 veRL 适配
- `scripts/10_gen_tool_config.py`：生成 tool config YAML
- `scripts/11_build_grpo_parquet.py`：生成 GRPO 训练/验证数据
- `configs/week3_mock_grpo.yaml`：mock 配置
- `configs/week3_vanilla_grpo.yaml`：正式训练配置

### 7.2 核心设计问题：env 怎么共享？

veRL 的 `ToolAgentLoop` 架构中：
- `BaseInteraction` 和 `BaseTool` 是**全局单例**
- 但它们需要通过 `instance_id` 区分并发调用
- **问题是**：veRL 的 `instance_id` 在 Interaction 和 Tool 之间**不是同一套**，无法通过 instance_id 反查

**方案对比**（来自 `week3_verl_rollout_design_merged.md`）：

| 维度 | 方案 A (Tool-centric) | 方案 B (Interaction-centric) | 方案 C (Hybrid) | **方案 I (ContextVar)** |
|------|----------------------|----------------------------|----------------|------------------------|
| 符合 veRL 设计 | ✗ | ✗ | ✓ 但 instance_id 不通 | ✓ |
| 14 个独立 schema | ✗ | ✓ | ✓ | ✓ |
| 并发隔离 | ✗ | ✗ | ✗ | ✓ |
| 代码量 | 低 | 高 | 中 | 中 |

**最终决策**：采用 **方案 I — Python `contextvars.ContextVar`**

### 7.3 tau_bench_context.py

```python
CURRENT_TAU_ENV: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "current_tau_env", default=None
)
CURRENT_TAU_STATE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "current_tau_state", default=None
)
```

**为什么用 ContextVar？**

Python 的 `contextvars` 在 `asyncio.create_task()` 时有特殊语义：
- 创建新 task 时，context 是**浅拷贝**
- 不同 task 修改各自的 contextvar **互不影响**
- 同一个 task 内的所有代码（包括 await 切换）共享同一个 context

这正好匹配 veRL 的执行模型：
- 每条 trajectory = 一个 `asyncio.create_task()`
- `Interaction.start_interaction()` 里 `set(env)`
- 后续的 `Tool.execute()` 在同一个 task 里 `get()` → 拿到正确的 env
- `group_size=4` 的 4 条 rollout = 4 个独立 task，天然隔离

### 7.4 tau_bench_tools.py 逐行剖析

**基类 `TauBenchToolBase`**：
```python
class TauBenchToolBase(BaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        env = CURRENT_TAU_ENV.get()
        state = CURRENT_TAU_STATE.get()
        
        # Fail loud: 宁可崩 batch，不让带毒 trajectory 进 GRPO
        if env is None or state is None:
            raise RuntimeError("...")
        
        action = Action(name=self.name, kwargs=parameters)
        step_res = env.step(action)
        
        state["total_reward"] += step_res.reward
        state["num_tool_calls"] += 1
        
        return (
            ToolResponse(text=step_res.observation),
            0.0,  # step reward = 0（Week 3 锁定 outcome reward）
            {"inc_reward": step_res.reward, "done": step_res.done}
        )
```

**14 个静态子类**：
```python
class TauBench_book_reservation_Tool(TauBenchToolBase): pass
class TauBench_calculate_Tool(TauBenchToolBase): pass
# ... 共 14 个
```

**为什么必须静态定义？**
- veRL 用 `hydra.utils.instantiate()` 按 `class_name` 加载 tool
- 如果 14 个 tool 共用同一个类名，veRL 只会加载一个 schema，模型只能看到 1 个 function
- 早期设计用 `type()` 动态生成，但 **cloudpickle**（Ray 跨进程序列化）对动态类支持不稳定，会 `AttributeError`
- 14 行样板代码换 100% pickle 可靠性

**一致性校验**：
```python
def verify_tool_classes_match_env():
    # 启动时检查静态类名与 env.tools_info 完全对齐
    # 如果 tau-bench 升级加了新 tool，立刻报错提示更新
```

### 7.5 tau_bench_interaction.py 逐行剖析

**`start_interaction`**：
```python
async def start_interaction(self, instance_id, task_id=0, **kwargs):
    env = get_env(..., task_index=task_id_int)
    env.reset(task_index=task_id_int)
    state = make_initial_state(task_id_int)
    
    # 关键：绑定到当前 asyncio task 的 context
    CURRENT_TAU_ENV.set(env)
    CURRENT_TAU_STATE.set(state)
    
    self._instance_dict[instance_id] = {"env": env, "state": state}
    return instance_id
```

**`generate_response`**（核心交互方法）：
```python
async def generate_response(self, instance_id, messages, **kwargs):
    # Defensive re-set：重新 set contextvar，防止状态机跨 turn 时丢失
    CURRENT_TAU_ENV.set(env)
    CURRENT_TAU_STATE.set(state)
    
    assistant_content = _extract_latest_assistant_content(messages)
    
    # 污染检测
    if _has_forbidden_token(assistant_content):
        state["contaminated"] = True
        state["done"] = True
        return (True, "", 0.0, {"contaminated": True, ...})
    
    # 正常路径：驱动 user simulator
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content})
    step_res = env.step(action)
    
    # 终止判定
    total_turns = state["num_user_turns"] + state["num_tool_calls"]
    if is_done or total_turns >= self.max_turns:
        final_score = 1.0 if state["total_reward"] >= 1.0 else 0.0
        return (True, "", final_score, {"num_turns": total_turns, ...})
    
    # 继续交互
    return (False, user_reply, 0.0, {"turn": total_turns, ...})
```

**关键设计点**：
1. **污染检测双边一致**：`FORBIDDEN_TEMPLATE_TOKENS` 与 `vllm_policy.py` 保持一致
2. **Outcome reward 二元化**：`total_reward >= 1.0 ? 1.0 : 0.0`
3. **Defensive re-set**：每次 `generate_response` 都重新 set contextvar，这是防御式编程
4. **Fail loud**：entry 找不到直接 `RuntimeError`，不返回错误字符串

### 7.6 数据格式：GRPO Parquet

`scripts/11_build_grpo_parquet.py` 生成的 parquet schema：

| 列名 | 类型 | 说明 |
|------|------|------|
| `prompt` | list[dict] | 只含 system message（日期锚定），user msg 由 Interaction 生成 |
| `extra_info` | dict | `{"index": ..., "task_id": ..., "split": "seen", "interaction_kwargs": {"name": "tau_bench_airline", "task_id": ...}}` |
| `data_source` | str | `"tau_bench_airline"` |
| `ability` | str | `"tau_bench_airline"` |
| `reward_model.ground_truth` | str | `""`（placeholder） |

**关键**：GRPO 是 **on-policy** 的，parquet 里只有 prompt，不含 trajectory。实际的 user message 和 tool 交互在 rollout 时由 `TauBenchInteraction` 实时生成。

---

## 8. 配置文件与超参数设计哲学

### 8.1 week3_vanilla_grpo.yaml 关键参数

```yaml
data:
  train_batch_size: 4          # 每 step 4 条 prompt
  max_prompt_length: 8192
  max_response_length: 12288   # 给足长链路空间

actor_rollout_ref:
  actor:
    optim:
      lr: 5.0e-6               # 很小，防止 SFT checkpoint 被冲偏
    kl_loss_coef: 0.01
    kl_loss_type: low_var_kl
  rollout:
    n: 8                       # group_size=8，GRPO 组内归一化
    temperature: 0.7
    multi_turn:
      enable: True
      format: "hermes"         # Qwen2.5 tool parser
      max_user_turns: 15
      max_assistant_turns: 15
    tensor_model_parallel_size: 2   # 双卡 TP
    gpu_memory_utilization: 0.50
    max_num_seqs: 12

algorithm:
  adv_estimator: grpo
```

### 8.2 超参数设计理由

| 参数 | 取值 | 理由 |
|------|------|------|
| `lr=5e-6` | 低 | SFT 后的 warm-start，学习率太高会灾难性遗忘 |
| `group_size=8` | 8 | 50 tasks 的 outcome reward 方差大，需要较大 group 来稳定 advantage 估计 |
| `max_response_length=12288` | 12K | airline 长链路 trajectory 可达 8-10K，留余量 |
| `max_turns=15` | 15 | 训练时限制，防止少数超长 trajectory 拖垮 throughput；评测时用 30 |
| `TP=2` | 双卡 | 单卡 80GB 无法同时容纳 vLLM pool + FSDP all_gather + logits 峰值 |
| `bypass_mode=true` | 开 | vLLM 生成时计算 log_probs，跳过 actor 重算，消除 20-40GB 峰值 |

---

## 9. 关键优化决策与踩坑记录

### 9.1 OOM 终极解法：bypass_mode + fused_kernels

**问题**：FSDP 的 `compute_log_prob` 会创建 full logits，bf16 下峰值 27.8GB，单卡必爆。

**解法**：
1. `calculate_log_probs: true`：vLLM 生成时零成本附带计算 log_probs
2. `bypass_mode: true`：直接将 vLLM 的 log_probs 作为 `old_log_probs`，**完全跳过 actor 重算**
3. `use_fused_kernels: true`：调用 flash-attn 的 `cross_entropy_loss`，在 CUDA kernel 内直接计算，不创建完整 logits

**效果**：actor `compute_log_prob` 的 ~20-40GB 峰值降为 **0GB**；ref/actor update 的 logits 峰值 **27.8GB → ~2GB**。

### 9.2 日期锚定（Date Grounding）

**问题**：Qwen2.5-7B 默认把 "May 15" 补全为 2023，但 airline 数据是 2024 的，导致搜航班永远返回 `[]`。

**解法**：在 system prompt 中强制注入：
```
The current date is 2024-05-15 (Wednesday).
When users mention dates without specifying the year,
always assume they refer to 2024.
```

这个 trick 让 pass^1 从接近 0 提升到 0.16。

### 9.3 主动截断策略

**问题**：vLLM `max_model_len=16384`，长链路容易超限崩溃。

**解法**：
- 客户端预截断（35000 字符阈值，≈11-12K tokens）
- 策略是**丢弃旧轮次**而非截断内容，保留 system + 最近 3 条
- 一旦触发截断，`was_truncated=True`，整条 trajectory 标记为污染

### 9.4 AWQ 崩溃 → awq_marlin

**问题**：Native AWQ kernel 在 72B/16K+ 下有 `CUDA illegal memory access`。

**解法**：改用 `awq_marlin` 量化格式，同时禁用 `enable_chunked_prefill` 和 `prefix_caching`。

### 9.5 contextvar 跨线程降级预案

**风险**：如果 ToolAgentLoop 内部用 `asyncio.to_thread` 或 `loop.run_in_executor` 把 tool 调到另一个线程，contextvar 会丢失。

**检测**：`grep -n "run_in_executor\|to_thread" tool_agent_loop.py`

**结论**：v0.6.1 的 `ToolAgentLoop` 没有跨线程调用，contextvar 方案安全。

---

## 10. 完整数据流：从 Parquet 到 Policy 更新

```
[Parquet 数据源]
  prompt: [{"role":"system", "content":"The current date is 2024-05-15..."}]
  extra_info: {interaction_kwargs: {name:"tau_bench_airline", task_id: 0}}

        ↓ RLHFDataset.__getitem__()
        ↓ (return_raw_chat=True)

[DataProto (batch_dict)]
  batch: {input_ids, attention_mask, position_ids}
  non_tensor_batch: {
    raw_prompt: [messages list],
    interaction_kwargs: {name, task_id},
    extra_info: {...},
    index: 0,
    uid: "task_0"   # group id
  }

        ↓ RayPPOTrainer.fit()
        ↓ _get_gen_batch()

[Gen Batch]
        ↓ repeat(n=8)  [group_size]

        ↓ async_rollout_manager.generate_sequences()

[AgentLoopManager]
        ↓ chunk() → AgentLoopWorker[i].generate_sequences.remote(chunk)

[AgentLoopWorker]
        ↓ asyncio.create_task(_run_agent_loop(...)) for each sample

[ToolAgentLoop.run() — 状态机]
  PENDING: apply_chat_template(raw_prompt, tools=14_schemas) → prompt_ids
  GENERATING: vLLM server.generate(prompt_ids) → assistant_msg
  
  ├─ has tool_calls?
  │  YES → _call_tool() → TauBenchToolBase.execute()
  │         → CURRENT_TAU_ENV.get() → env.step(Action(tool_name, params))
  │         → ToolResponse(text=observation) 追加到 prompt_ids
  │
  └─ no tool_calls?
     → TauBenchInteraction.generate_response()
     → env.step(Action("respond", {"content": ...}))
     → user_reply 追加到 prompt_ids
  
  LOOP until done or max_turns
  
  TERMINATED: calculate_score() → 1.0 if total_reward >= 1.0 else 0.0

[AgentLoopOutput]
  prompt_ids, response_ids, response_mask(1=LLM生成, 0=tool/user),
  reward_score, num_turns, extra_fields

        ↓ _postprocess()

[Output DataProto]
  batch: {
    prompts, responses, response_mask, input_ids, attention_mask, position_ids,
    rm_scores: [0,0,...,reward_score,0,0]  # reward 放在 response 末尾位置
  }

        ↓ compute_reward()
        ↓ NaiveRewardManager 发现已有 rm_scores → 直接复用

        ↓ compute_advantage() — GRPO
        ↓ 按 uid 分组，advantage = (r - mean) / (std + ε)

        ↓ actor_rollout_wg.update_actor()
        ↓ PPO/GRPO loss，只更新 response_mask=1 的 token
```

---

## 11. 面试常考点与回答思路

### Q1：为什么要用 GRPO 而不是 PPO？

**答**：
1. PPO 需要训练一个 Critic 模型来估计 V(s)，在长链路场景下 Critic 的估值非常不准（sparse reward + long horizon）
2. GRPO 不需要 Critic，直接用组内归一化的 outcome reward 作为 advantage，实现更简单
3. 在 τ-bench 这种只有 0/1 reward 的场景，GRPO 的组内比较比绝对值估计更稳定
4. 我们的资源有限（2×A800），去掉 Critic 可以省一张卡

### Q2：contextvar 方案的原理是什么？为什么能保证并发安全？

**答**：
1. Python 的 `contextvars.ContextVar` 在 `asyncio.create_task()` 时会**浅拷贝**父任务的 context
2. veRL 的 `ToolAgentLoop` 为每条 trajectory 创建一个独立的 `asyncio.create_task()`
3. 同一条 trajectory 内，`Interaction.start_interaction()` set 的 env，后续 `Tool.execute()` 在同一个 task 里 get，一定能读到
4. 不同 trajectory 之间，各自的 set 操作互不影响，因为 context 是 task-local 的
5. 我们在本地写过一个最小测试验证过这个语义（4 个并发 task 各 set 不同值，互不干扰）

### Q3：为什么 tool 要定义 14 个静态子类？一个基类不行吗？

**答**：
1. veRL 的 `initialize_tools_from_config` 按 `class_name` 实例化 tool，同一个 `class_name` 只会实例化一次
2. 如果 14 个 tool 共用一个类，veRL 只会加载一个 schema，模型在 function calling 时只能看到 1 个 function
3. 早期我们尝试过 `type()` 动态生成，但 Ray 用的 cloudpickle 对动态类序列化不稳定，会 `AttributeError`
4. 最终方案：14 个空静态类 + `verify_tool_classes_match_env()` 启动时校验，保证 100% pickle 安全

### Q4：SFT 的 loss mask 怎么做的？为什么不用 TRL SFTTrainer？

**答**：
1. multi-turn trajectory 中，我们只对 assistant turn 计算 loss，user/system/tool turn 要 mask 成 -100
2. Qwen2.5 的 chat template 很复杂，手切 token 容易 off-by-one
3. 我们的方案是"渲染两次取 diff"：对每轮 assistant turn，分别渲染 `messages[:i]`（加 generation prompt）和 `messages[:i+1]`（不加），取 token diff 作为 label 区间
4. 这个做法不依赖硬编码的特殊 token，无论 chat template 怎么变都对得上
5. 不用 TRL SFTTrainer 是因为它对 multi-turn + tool_calls 的 loss mask 是黑盒，无法保证正确性

### Q5：bypass_mode 解决了什么问题？原理是什么？

**答**：
1. 问题：FSDP 的 `compute_log_prob` 需要创建 full logits 张量，bf16 下 7B 模型在 8K 长度时峰值达 27.8GB，单卡 80GB 不够
2. 原理：vLLM 在生成时已经计算了每个 token 的 log_prob（这是采样概率的自然副产品，零额外开销）
3. `bypass_mode` 把这些 log_probs 缓存下来，直接作为 PPO 的 `old_log_probs`，**完全跳过 actor 的 forward 重算**
4. 效果：actor `compute_log_prob` 的显存峰值从 ~20-40GB 降到 **0GB**
5. 配合 `use_fused_kernels`（flash-attn 的 fused cross_entropy），训练阶段的 logits 峰值从 27.8GB 降到 ~2GB

### Q6：Reward 为什么只有 0/1？中间步骤没有任何反馈，模型怎么知道哪里错了？

**答**：
1. 这是 τ-bench 原生设计：最终比对数据库状态 hash，完全一致才给 1，否则 0
2. 这个设计很严格，但也很 realistic：真实客服场景里，用户不会每轮都告诉你"做得对/错"
3. **当前项目（Week 3）确实没有解决这个问题**，这正是我们诊断阶段要暴露的"reward 稀疏"问题
4. 后续改进方向（Week 4+）：
   - Step-level reward shaping（如 tool 调用成功给 small positive）
   - Turn-level advantage（而非 episode-level）
   - PRM（Process Reward Model）评估每步质量

### Q7：你们项目的创新点是什么？

**答**：
1. **工程层面**：首次将 τ-bench 长链路 tool-use 场景接入 veRL 框架，解决了 async multi-turn rollout 中的 env 状态共享问题（contextvar 方案）
2. **系统层面**：搭建了完整的 Baseline → SFT → GRPO 三阶段 pipeline，每个阶段都有严格的评测和数据质量保障（污染检测、loss mask 验证、pass^k 指标）
3. **优化层面**：针对 2×A800 的资源约束，通过 bypass_mode + fused_kernels + TP=2 的组合，在显存极限下跑通了 GRPO
4. **诊断层面**：我们不是为了刷分而训练，而是系统性地记录 vanilla GRPO 的 4 类核心问题（reward 稀疏、credit misassignment、长度漂移、KL 失控），为后续改进提供数据基础

---

## 12. 附录：文件索引速查

### 12.1 项目核心源码

| 文件 | 职责 | 对应章节 |
|------|------|----------|
| `src/envs/tau_bench_context.py` | contextvar 定义（CURRENT_TAU_ENV / CURRENT_TAU_STATE） | 7.3 |
| `src/envs/tau_bench_tools.py` | 14 个 tool 的 veRL 适配 | 7.4 |
| `src/envs/tau_bench_interaction.py` | user simulator 的 veRL 适配 | 7.5 |
| `src/envs/tau_bench_wrapper.py` | 同步评测封装（Week 1/2） | 5.2 |
| `src/models/vllm_policy.py` | OpenAI API policy 实现 | 5.3 |
| `src/training/sft_dataset.py` | multi-turn loss mask 核心 | 6.3 |
| `src/evaluation/pass_k_eval.py` | pass^k 评测框架 | 5.4 |

### 12.2 脚本文件

| 文件 | 职责 | 对应章节 |
|------|------|----------|
| `scripts/02_run_baseline_eval.py` | Week 1 baseline 评测 | 5 |
| `scripts/03_analyze_failures.py` | 失败归因分析 | 5.5 |
| `scripts/04_collect_sft_data.py` | Week 2 SFT 数据采集 | 6.2 |
| `scripts/04b_inspect_sft_dataset.py` | 验证 loss mask | 6 |
| `scripts/05_sft_train.py` | LoRA SFT 训练 | 6.4 |
| `scripts/05b_merge_lora.py` | 合并 LoRA adapter | 6 |
| `scripts/05c_eval_sft.py` | SFT 后评测 | 6 |
| `scripts/10_gen_tool_config.py` | 生成 tool config YAML | 7 |
| `scripts/11_build_grpo_parquet.py` | 生成 GRPO parquet | 7.6 |
| `scripts/20_run_week3_vanilla.sh` | 启动 GRPO 训练 | 8 |

### 12.3 配置文件

| 文件 | 职责 |
|------|------|
| `configs/baseline_airline.yaml` | Week 1 baseline 配置 |
| `configs/sft_collect_airline.yaml` | Week 2 数据采集配置 |
| `configs/sft_airline_lora.yaml` | Week 2 SFT 训练配置 |
| `configs/week3_mock_grpo.yaml` | Week 3 mock 验证配置 |
| `configs/week3_vanilla_grpo.yaml` | Week 3 正式训练配置 |
| `configs/interaction_config/tau_bench_airline.yaml` | veRL Interaction 配置 |
| `configs/tool_config/tau_bench_airline_tools.yaml` | 14 个 tool schema |

### 12.4 文档文件

| 文件 | 职责 |
|------|------|
| `week2.md` | Week 2 操作手册 |
| `week3_pipeline_manual.md` | Week 3 使用手册 |
| `week3_verl_rollout_design_merged.md` | Week 3 设计决策文档 |
| `docs/week2_collect_sft_optimization.md` | Week 2 优化经验 |
| `docs/week3_vanilla_grpo_optimization.md` | Week 3 优化经验 |

---

> **最后提醒**：面试时，如果面试官深挖代码细节，建议打开 IDE 指着源码讲。特别是 `tau_bench_interaction.py` 的 `generate_response`、`tau_bench_tools.py` 的 `execute`、`sft_dataset.py` 的 `build_supervised_example` 这三个函数，是最高频的考点。

> **祝实习顺利！**
