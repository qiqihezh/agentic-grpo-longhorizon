# Week 2 SFT 轨迹采集：工程设计与关键决策

> 本文档记录在使用 **Qwen2.5-72B-Instruct-AWQ** 于 τ-bench airline 领域进行监督微调（SFT）轨迹采集过程中的工程细节、踩坑记录与优化决策。
>
> **目标**：在 2×A800 的严格显存与稳定性约束下，采集高质量的工具调用轨迹，用于训练 Qwen2.5-7B-Instruct 的 LoRA 检查点（作为 GRPO 的 warm-start）。

---

## 核心决策速览

| 决策项 | 最终取值 | 决策理由 |
|--------|----------|----------|
| 上下文长度上限 | **16K**（原 32K） | Native AWQ 在 >16K 时 CUDA 崩溃；`awq_marlin` 仅在 16K 以内稳定 |
| 量化格式 | **`awq_marlin`** | Native AWQ 在长上下文下存在非法内存访问 bug |
| Best-of-N 采样 | **16**（原 8） | 补偿因截断过滤而丢弃的长轨迹 |
| 主动截断阈值 | **35,000 字符**（≈11-12K tokens） | 为生成预留余量；被截断的轨迹标记为**污染**并丢弃 |
| 每步最大生成长度 | **768**（原 512） | 防止复杂 JSON tool-call 被截断引发的级联故障 |
| 双卡部署策略 | Policy @ GPU0 (4 seqs) + UserSim @ GPU1 (8 seqs) | 消除单卡 seq-slot 争抢；不对称上限匹配各自负载特征 |
| 训练/保留集划分 | **40 seen / 10 unseen** | 最大化训练数据量，同时保留泛化评估 holdout |

---

## 1. 项目背景

本文档对应一个 agentic-RL 项目的 **Week 2** 工程迭代：

- **环境**：[τ-bench](https://github.com/sierra-research/tau-bench) airline（50 个任务、有状态工具、基于 LLM 的用户模拟器）。
- **策略模型**：Qwen2.5-72B-Instruct-AWQ（权重约 40 GB）。
- **核心挑战**：72B 模型同时充当 **Agent 策略** 和 **τ-bench 用户模拟器**，需要在两张 A800 上各部署一个 vLLM 实例，既要避免 CUDA 崩溃/OOM，又要保证轨迹质量。
- **硬约束**：单机 2×A800-80GB，不允许多机占用，不允许独占数周。

---

## 2. 硬件与部署架构

### 2.1 环境配置
- **GPU**：2× NVIDIA A800-SXM4-80GB
- **Conda 环境**：`agentrl`（Python 3.10, torch 2.4.0+cu121, vLLM dev, autoawq 0.2.9）
- **模型路径**：`../models/Qwen2.5-72B-Instruct-AWQ`

### 2.2 双卡分离部署

| 角色 | GPU | 端口 | `max_num_seqs` | 典型上下文长度 | 16K 下每序列 KV Cache |
|------|-----|------|----------------|----------------|----------------------|
| **Policy** | 0 | 8000 | 4 | 8K–14K | ~5 GB |
| **User Sim** | 1 | 8001 | 8 | 2K–4K | ~1.25 GB |

**为什么不对称？**  
Policy 请求携带完整的工具调用历史（8K–14K），4 条并发序列已占用约 20 GB KV Cache。User-sim 请求是短回复，8 条序列仍有余量。分离部署消除了单卡 seq-slot 争抢，稳定了吞吐。

### 2.3 启动命令

```bash
# Policy（GPU 0）— 上下文长，KV Cache 压力大
CUDA_DEVICES=0 PORT=8000 \
  bash scripts/01_start_vllm_server_72b.sh

# User Sim（GPU 1）— 上下文短，KV Cache 压力小
CUDA_DEVICES=1 PORT=8001 MAX_NUM_SEQS=8 \
  bash scripts/01_start_vllm_server_72b.sh
```

---

## 3. 量化格式与稳定性

### 3.1 从 32K 到 16K 的踩坑历程

| `max_model_len` | 量化格式 | 结果 | 根因 |
|-----------------|----------|------|------|
| 32K | `awq` | ❌ CUDA illegal memory access | Native AWQ kernel 在超长上下文下有 bug |
| 24K | `awq` | ❌ 同上 | KV Cache 仅 17% 时即崩溃，非 OOM |
| 16K | `awq_marlin` | ✅ 稳定 | Marlin kernel 内存访问更安全 |
| 16K | `enforce_eager=True` | ❌ 无效 | 不是 eager mode 的问题 |
| 16K | `enable_chunked_prefill=False` | ✅ 保留 | 降低 prefill 阶段的不确定性 |

**最终选择**：`awq_marlin` + `enable_chunked_prefill=False`。

### 3.2 16K 下的显存占用分析

```raw
A800 80GB × 0.90 util = 72GB 可用
├── AWQ Marlin 权重:          ~40 GB  (含 fp16 scales/zeros)
├── CUDA context/workspace:    ~3 GB
├── Prefix caching 索引:       ~1 GB
└── KV Cache block pool:       ~22 GB (预分配)
    └── 16K × 80层 × 8 KV heads × 128 dim × 2 bytes ≈ 5 GB/seq
        4 seqs (policy)  = 20 GB
        8 seqs (user sim)= 峰值 40GB，但平均远低于（user sim 上下文通常 2-4K）

nvidia-smi 实际观察: ~68-70 GB
├── 预分配: ~68 GB（启动时）
└── 运行时增量: +1-2 GB
    └── CUDA graph capture (bs=1,2,3,4 各一套)
    └── Forward activation 峰值
    └── PyTorch allocator 碎片
```

---

## 4. 上下文长度策略：主动截断

### 4.1 设计哲学

**截断不是救命稻草，而是质量过滤器。**  
一旦某条 trajectory 触发了截断阈值，**整条 trajectory** 即被标记为**污染**并丢弃，即使最终回合本可以 `success=True`。

### 4.2 三层防护设计

**第一层：采集时主动截断**
- **触发阈值**：**35,000 字符**（≈**11-12K tokens**，Qwen 约 2.5-3.0 字符/token + tool metadata overhead）。
- **截断策略**：保留 `system` + 前 2 条（首轮需求）+ 最近交互，逐步丢弃旧轮次。
- **污染标记**：`VLLMPolicy.was_truncated = True`（一旦触发永久置位）。
- **过滤规则**：`TrajectoryResult.was_contaminated_from_turn` 一旦非 None，整条 trajectory 进入 `*_contaminated.jsonl`，**永不进入 `train.jsonl`**。

**第二层：数量兜底**
- `best_of_n`：**8 → 16**
- `temperatures`：`[0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]`（分层采样：4×greedy + 4×轻度探索 + 4×标准多样化 + 4×激进探索）
- 短 trajectory 仍有 15 次额外机会补充成功样本。

**第三层：训练对齐**
- SFT `max_length`：**16384**（让 7B 模型在训练时见过接近上限的长样本）。
- Week 3 rollout 保持同样的 16K 约束，避免训练/ serving 分布偏移。

### 4.3 阈值设定

| 阈值 | 取值 | 用途 |
|------|------|------|
| vLLM `max_model_len` | 16K | 硬约束，避开 CUDA crash |
| 主动截断触发线 | 35,000 chars（≈11-12K tokens） | 留 4-5K 给 assistant 生成 + 安全余量 |
| 允许进入 `train.jsonl` 的最大长度 | 约 12K-13K | `raw_messages` tokenize 后的实际上限 |

**实测数据**：50 task × best-of-16 = 800 次 rollout，仅产生 **1 条**污染轨迹（task_0009，在 turn 23 触发截断），污染率 0.125%。

---

## 5. `max_tokens=512` 级联故障（关键优化）

### 5.1 故障链条

```raw
max_tokens=512 过紧
      ↓
复杂 tool-call JSON 在 512 处被截断（如 book_reservation 带 3-5 人乘客信息）
      ↓
json.loads 失败 → policy 返回"无 tool_calls"的截断文本
      ↓
τ-bench 将其视为普通 respond 处理（非 tool call）
      ↓
user sim 看到不完整的废话/半截 JSON → 困惑 → 继续追问
      ↓
交互延长 5-10 轮（本该结束的 trajectory 持续下去）
      ↓
input 更快膨胀到 14K → 触发主动截断
      ↓
was_truncated=True → 整条 trajectory 标记为污染 → 作废
```

**后果**：本该成功的一条 trajectory，因为 `max_tokens` 不够，变成了污染数据并被丢弃。

### 5.2 具体场景

**场景 1：复杂 tool call JSON**
```json
{"name": "book_reservation", "arguments": {"user_id": "user_123", "passengers": [{"first_name": "John", ...}, {"first_name": "Jane", ...}, {"first_name": "Bob", ...}]}}
```
这段 JSON 约 **320-380 tokens**。512 勉强装下，但乘客再多（4-5 人）或名字较长 → **截断**。

**场景 2：CoT + Tool Call**
```text
<think>用户要改签，我需要：1. 查原订单 2. 找可用航班 3. 计算差价...</think>
<tool_call>{"name": "search_reservation", ...}</tool_call>
```
CoT 思考 + tool call 可能轻松超过 512 tokens。

### 5.3 优化决策

| `max_tokens` | 覆盖场景 | 级联风险 | 决策 |
|--------------|----------|----------|------|
| 512 | 简单 tool call | **高**（复杂 JSON/CoT 易截断） | ❌ 弃用 |
| **768** | 复杂 tool call + 短 CoT | **低**（余量 300+ tokens） | ✅ **采用** |
| 1024 | 超长 tool call + 长 CoT | 无 | 可行但收益递减，多占 256 tokens KV cache |

**为什么选 768 而不是 1024？**
- 消除了 95% 的截断风险
- 只比 512 多 256 tokens，对并发度影响 < 2%
- 11K input + 768 = 11.8K < 16K，仍有 4K+ 余量

### 5.4 后续硬化措施（未实施，留作参考）

1. **截断检测**：在 `vllm_policy.py` 中检测 `<tool_call>` 存在但 `</tool_call>` 缺失 → 主动标记失败。
2. **长度门控**：如果生成内容长度 > 90% `max_tokens` 且 tool-call 解析失败 → 返回 `[GENERATION_TRUNCATED]`，让 `TauBenchWrapper` 感知并终止。

---

## 6. 并发优化：`num_workers` 与 `max_num_seqs` 的匹配

### 6.1 核心关系

```raw
num_workers    = 同时有多少个 task 在跑（ThreadPoolExecutor 并发度）
max_num_seqs   = vLLM 单卡能同时处理的 request 数上限

理想: max_num_seqs(policy) ≥ num_workers
      max_num_seqs(user_sim) ≥ num_workers（通常有余量）
```

**一个 worker 的交互链（单 trajectory）：**
```raw
Worker A: policy req → wait → user_sim req → wait → policy req → wait → ...
```
任意时刻一个 worker 只发 1 个 request，但 4 个 workers 的 peak 并发 = 4。

### 6.2 为什么可以不对称？

| | Policy (GPU0) | User Sim (GPU1) |
|---|---|---|
| 典型上下文 | 8K-14K | 2K-4K |
| 每序列 KV cache @16K | ~5 GB | ~1.25 GB |
| 4 seqs KV cache | 20 GB | 5 GB |
| 安全上限 | **4 seqs** | **8+ seqs** |

**最终配置**：
```yaml
num_workers: 4
# policy 端: max_num_seqs=4（默认，刚好匹配）
# user_sim 端: max_num_seqs=8（环境变量覆盖，有余量）
```

### 6.3 性能表现

| 配置 | 并发 task | 实际采集时间 |
|------|-----------|-------------|
| 初始 (2 seqs, 2 workers) | 2 | ~8-10 h（预估） |
| 中间 (4 seqs, 3 workers) | 3 | ~5-6 h（预估） |
| **最终 (4/8 seqs, 4 workers)** | **4** | **~7.2 h** |

**实际耗时**：25,913 秒（≈ **7.2 小时**），50 task × 16 samples = 800 rollouts 全部完成。

---

## 7. 数据质量控制机制

### 7.1 四层过滤规则

1. **Context 超限**：`[CONTEXT_LENGTH_EXCEEDED]` → 立即 abort trajectory（不浪费采样）。
2. **主动截断污染**：`was_contaminated_from_turn != None` → 整条作废，不进 train。
3. **其他 API 错误**：`[POLICY_API_ERROR]` → 返回 fake assistant message，允许 trajectory 继续（收集诊断信息）。
4. **Success 筛选**：只有 `success=True` 且 **未被污染** 的 trajectory 进 `train.jsonl`。

### 7.2 输出文件结构

```raw
experiments/week2_sft_collect_airline/
├── task_0000.jsonl                 # 干净的 success trajectory（进 train.jsonl）
├── task_0000_contaminated.jsonl    # 被截断污染的 trajectory（诊断用，不进 train）
├── task_0000.meta.json             # 该 task 的统计（success 数、污染数）
├── train.jsonl                     # 合并后的 SFT 训练数据（45 条）
├── holdout_train.jsonl             # 10 个 unseen task 的数据（22 条）
├── split.json                      # seen/unseen task 切分（40/10）
├── summary.json                    # 全局统计
└── collect_config.yaml             # 采集参数快照
```

### 7.3 Checkpoint Resume
- 按 task 粒度：`task_XXXX.meta.json` 存在则跳过。
- **重跑前必须删除空文件**：`rm task_*.jsonl task_*.meta.json task_*_contaminated.jsonl`

### 7.4 日期锚定（关键修复）

Qwen2.5 默认把无年份日期补成 2023，但 τ-bench airline 数据基于 2024。

```python
SYSTEM_PROMPT = (
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, "
    "always assume they refer to 2024."
)
```

---

## 8. 代码修改文件清单

### `src/models/vllm_policy.py`
- 恢复 `_truncate_messages`，阈值 **35,000 字符**（≈11-12K tokens）。
- 增加 `self.was_truncated` 标记（一旦触发永久置 True）。
- 错误分类：`[CONTEXT_LENGTH_EXCEEDED]` 直接 abort vs `[POLICY_API_ERROR]` 返回 fake message。
- 消息清洗：处理 `['observation', '...']` 元组格式、过滤 `task=Task(` 泄露、合并连续同角色消息。
- Tool call 解析：原生解析 + `<tool_call>` 正则回退。

### `src/envs/tau_bench_wrapper.py`
- `TrajectoryResult` 增加 `was_contaminated_from_turn: Optional[int]`。
- `run_single_task` 每次调用 policy 后检测 `policy.was_truncated`。
- 日期锚定 system prompt 注入。

### `scripts/04_collect_sft_data.py`
- `best_of_n=16`（从 8 提升）。
- 污染 trajectory 输出到 `task_XXXX_contaminated.jsonl`。
- 合并 `train.jsonl` 时**不合并** contaminated 文件。
- `summary.json` 增加 `total_contaminated_trajectories`。
- Tiny 模式支持（2 task, best_of=2）。

### `configs/sft_collect_airline.yaml`
- `best_of_n: 16`
- `temperatures`: `[0.0×4, 0.5×4, 0.8×4, 1.0×4]`
- `num_workers: 4`
- `max_turns: 30`
- `holdout_size: 10`（40 seen / 10 unseen）

### `configs/sft_airline_lora.yaml`
- `max_length: 16384`（上下文对齐）
- LoRA r=16, alpha=32, lr=1e-4, 3 epochs

---

## 9. 关键 Bug 与修复历史

| 现象 | 根因 | 修复 |
|------|------|------|
| `Engine process failed to start` | `max_model_len` 超过 KV Cache 容量 | 降低 `max_model_len` |
| `ValueError: max seq len > KV cache` | `gpu_memory_utilization` 太低 | 调高利用率 |
| `404 model not found` | `--served-model-name` 与客户端配置不匹配 | 对齐名称 |
| `CUDA illegal memory access` | Native AWQ kernel bug @ 24K+ | 改用 `awq_marlin` + 16K |
| 日期解析成 2023 | Qwen 默认日期 | 注入 system prompt 锚定 2024-05-15 |
| 消息格式 `['observation', '...']` | τ-bench 返回元组 | 清洗逻辑兼容 tuple 格式 |
| 连续同角色消息 400 | vLLM 不允许连续 user/user | 合并连续同角色消息 |

---

## 10. 操作 SOP

### 10.1 启动前
```bash
# 清掉旧进程
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5

# 清掉之前的失败/空文件（否则 checkpoint 跳过）
rm -f experiments/week2_sft_collect_airline/task_*.jsonl
rm -f experiments/week2_sft_collect_airline/task_*.meta.json
rm -f experiments/week2_sft_collect_airline/task_*_contaminated.jsonl
```

### 10.2 启动 vLLM（双卡）
```bash
# GPU 0 → policy
cd agentic-grpo-longhorizon
CUDA_DEVICES=0 PORT=8000 \
  bash scripts/01_start_vllm_server_72b.sh

# GPU 1 → user sim（另起终端/tmux）
CUDA_DEVICES=1 PORT=8001 MAX_NUM_SEQS=8 \
  bash scripts/01_start_vllm_server_72b.sh
```

### 10.3 Tiny 验证（必做）
```bash
python scripts/04_collect_sft_data.py \
  --config configs/sft_collect_airline.yaml --tiny
```

验证项：
1. vLLM 16K 是否稳定（无 CUDA crash）。
2. `task_0000.meta.json` 中 `num_contaminated` 是否为 0。
3. `train.jsonl` 是否只包含干净数据。

### 10.4 全量采集
```bash
python scripts/04_collect_sft_data.py \
  --config configs/sft_collect_airline.yaml
```

### 10.5 采后检查
1. **长度分布直方图**：`train.jsonl` 中 trajectory token 数应在 8K-13K，无 14K+ 长尾。
2. **Task 覆盖率**：某些 task 为 0 条 → Week 3 诊断素材。
3. **抽样人工检查**：随机抽 3-5 条，看截断边界附近是否有异常行为。

---

## 11. 实际采集结果

| 指标 | 数值 | 备注 |
|------|------|------|
| 尝试 task 数 | 50 | τ-bench airline test split 全部 |
| 成功 task 数 | **19** | 覆盖率 **38%** |
| 成功轨迹总数 | **67** | 50×16 = 800 次 rollout |
| 进训练集 | **45** | `train.jsonl`（40 个 seen task 产出和10 个 unseen task 产出） |
| 进 holdout 集 | **22** | `holdout_train.jsonl`（seen task和unseen task只是task id ，与成功采样轨迹的task没有关系） |
| 污染轨迹 | **1** | task_0009，turn 23 触发截断 |
| 单 task 最高成功数 | **12** | task 43 |
| 平均成功/task | **1.34** | 67 / 50 |
| 采集耗时 | **~7.2 h** | 25,913 秒 |

**失败 task 分布**：31 个 task 零成功，主要集中在需要复杂多步推理或长交互链的场景（Week 3 GRPO 的优化目标）。

---

## 12. 关键决策一句话总结

1. **16K 不是妥协，是主动设计**：用 35K 字符截断阈值 + 污染标记，换来 CUDA 稳定性。
2. **best_of_16 是数量兜底的必要代价**：长 trajectory 被过滤后，短 trajectory 需要更多采样机会。
3. **双卡分离 + 不对称 max_seqs**：policy 端 4 seqs（KV cache 瓶颈），user sim 端 8 seqs（上下文短，有余量）。
4. **截断 ≠ 救命稻草，是质量过滤器**：被截断碰过的 trajectory 永不进训练集（实际仅 1 条）。
5. **awq_marlin 是唯一稳定的量化选项**：native AWQ 在 72B/16K+ 下有 kernel bug。
6. **40 seen / 10 unseen**：最大化训练数据量（45 条），同时保留泛化评估 holdout（22 条）。
