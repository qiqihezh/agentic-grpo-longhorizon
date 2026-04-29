# Week 3 veRL Multi-Turn Rollout Pipeline 设计

> **文档定位**:Week 3 正式启动 GRPO 训练前的内部技术决策文档。锁定 τ-bench → veRL 的抽象映射、接口契约、数据流设计。不包含 Week 3 后续的诊断发现——那些应写入单独的 `vanilla_grpo_diagnosis.md`。
>
> **前置依赖**:Week 1 baseline(72B-user pass^1=0.160)、Week 2 SFT checkpoint(overall pass^1=0.145,见 PROGRESS.md Week 2)、`src/envs/tau_bench_wrapper.py` 当前版本、`src/models/vllm_policy.py` 带 FORBIDDEN_TEMPLATE_TOKENS 补丁的版本。
>
> **文档范围**:本文覆盖"5-step mock GRPO 跑通"前的所有设计决策。实际训练参数调优、诊断观察写入 Week 3 后续文档。

---

## §1 需求与约束

### 1.1 来自项目宪章的硬性要求

| 需求项 | 来源 | 具体要求 |
|---|---|---|
| RL 训练框架 | PROJECT.md §2.1 | veRL(主力),不切 OpenRLHF/LLaMA-Factory |
| Policy 模型 | PROJECT.md §2.1 | Qwen2.5-7B-Instruct + Week 2 SFT LoRA(merged) |
| User simulator | PROJECT.md §2.1 | Qwen2.5-72B-Instruct-AWQ 本地部署 |
| 推理引擎 | 本次决策会话 | policy 走 vLLM(与 W1/W2 对齐),user sim 不变 |
| 训练目标 | PROJECT.md §1.1 | airline 子集上的 vanilla GRPO + 后续改进 |
| 评测指标 | PROJECT.md §2.1 | pass^1 / pass^4 / pass^8 / turn_efficiency |
| 资源 | PROJECT.md §1.4 | 2×A800(GPU0 user sim,GPU1 policy+training) |

### 1.2 来自 τ-bench 的环境约束

`tau_bench.envs.get_env(env_name, task_index, ...)` 返回一个**有状态**的 env 实例:

- `env.reset(task_index)` → 返回 `EnvResetResponse(observation, info)`,其中 `observation` 是用户第一条消息
- `env.step(Action)` 同时处理两类动作:
  - `Action(name=<tool_name>, kwargs=<tool_args>)` → tool 调用,返回 `EnvStepResponse(observation=tool_result, reward, done, info)`
  - `Action(name=RESPOND_ACTION_NAME, kwargs={"content": ...})` → 触发 user simulator 回复,返回 `EnvStepResponse(observation=user_reply, reward, done, info)`
- `env.tools_info` 暴露 13 个 airline tool 的 OpenAI function schema
- **env 状态(订单、数据库、对话历史)绑定在实例内**,跨 task 不能复用

### 1.3 来自资源的硬约束

- **单节点 2×A800** 同时承载:72B user sim (GPU0) + 7B policy rollout (GPU1) + LoRA 训练(GPU1,与 rollout 分时或 colocate)
- **vLLM 稳定区间**(Week 2 血泪教训):`max-model-len=16384`,`max-num-seqs=2-4`,禁用 awq_marlin/prefix-caching/chunked-prefill
- **不能独占多卡数周**,所有 GRPO 配置必须在单 A800 policy + LoRA 上能跑
- **5-step mock GRPO 必须在 < 2 小时内完成**(否则 Week 3 Day 2-3 时间窗口撑不住)

### 1.4 Week 3 的核心交付物(对设计的反向约束)

Week 3 的目标是**暴露并记录 vanilla GRPO 的 4 类核心问题**(reward 稀疏 / long-horizon credit misassignment / 长度-entropy 漂移 / KL 失控)。为此 rollout 设计必须:

- **信号干净**:reward 严格等于 τ-bench 原生 outcome reward(0/1),不引入任何 shaping(Week 4 改进时再加)
- **可诊断**:每条 trajectory 必须记录 `num_turns` / `num_tool_calls` / 污染标记,供后续分析
- **可复现**:相同 seed + 相同 checkpoint 下 rollout 结果必须一致(影响 ablation 可信度)
- **与 W1/W2 评测可比**:rollout 期间的 policy 行为分布必须能用 `pass_k_eval.run_eval` 验证(至少在每 N step 做一次)

---

## §2 τ-bench → veRL 抽象映射(三方案 rubric)

### 2.1 veRL 的三个扩展点(v0.6.1)

| 抽象 | 负责什么 | 生命周期 |
|---|---|---|
| `AgentLoopBase` | 一条 trajectory 的完整循环 | 每个 prompt 一个实例 |
| `BaseTool` | 单个 tool 的 schema + 执行逻辑 | 全局单例,通过 `instance_id` 区分并发调用 |
| `BaseInteraction` | 模拟用户回复(非 tool 交互) | 全局单例,通过 `instance_id` 区分 |

veRL 已提供 `ToolAgentLoop`(走 Tool 分支 + 可选 Interaction 分支),**本项目不自己写 AgentLoop**,复用官方实现。

### 2.2 三个候选映射方案

**方案 A:Tool-centric(把一切塞进 Tool)**

- `TauBenchTool` 持有 env 实例,`execute` 处理 tool 调用
- 用户回复也通过 Tool 返回(每次 assistant 回复触发 `TauBenchTool.execute(name="respond", ...)`)
- 不用 Interaction

**方案 B:Interaction-centric(把 env 塞进 Interaction)**

- `TauBenchInteraction` 持有 env 实例,`generate_response` 通过 `env.step(RESPOND_ACTION)` 生成用户回复
- 13 个 tool 每个都是独立 `BaseTool` 子类,但**共享 env**——需要跨对象访问同一个 env 实例

**方案 C:Hybrid(Tool 做工具、Interaction 做用户,通过 instance_id 共享 env)**

- `TauBenchInteraction` 持有 env 实例,绑定到 `instance_id`
- `TauBenchTool`(13 个动态生成的子类)通过 `instance_id` 向 Interaction 反查 env
- 所有 `env.step()` 的调用方都能访问同一个 env

### 2.3 Rubric 评分

| 维度 | 方案 A | 方案 B | 方案 C |
|---|---|---|---|
| **符合 veRL 原生设计** | ✗ 违反 "Tool 是 tool,Interaction 是 user" 的抽象 | ✗ Tool 之间需要共享状态,veRL 没这个机制 | ✓ 完全对齐官方示例(gsm8k_interaction + gsm8k_tool) |
| **Function calling schema 正确** | ✗ 只有一个 Tool,模型看不到 13 个独立 function | ✓ 13 个独立 schema | ✓ 13 个独立 schema |
| **env 状态一致性** | ✓ env 只有 Tool 持有 | ✗ Tool 之间 + Interaction 要共享 | ✓ env 归 Interaction,Tool 反查 |
| **代码量** | 低 | 高(要解决共享 env 问题) | 中(反查机制 ~30 行) |
| **τ-bench contract 保留** | ✗ `RESPOND_ACTION_NAME` 被伪装成 tool,模型可能误调 | ✓ 用户回复走 Interaction 自然路径 | ✓ 同 B |
| **调试复杂度** | 低 | 高 | 中 |

### 2.4 决策:采用方案 I(contextvars-based env 共享)

**新增候选方案 I**,与原方案 A/B/C 并列评分。在 rubric 上 I 胜出。

**方案 I 核心设计**:
- Interaction 持有 env 实例(每条 trajectory 独立 env)
- Tool 通过**`contextvars.ContextVar`** 读当前 trajectory 的 env,不需要 instance_id 反查
- contextvar 由 `Interaction.start_interaction` 和 `Interaction.generate_response` 在进入时 `set`,确保 Tool 看到正确的 env

**为什么 I 可行**:
1. **Python contextvars 在 asyncio 中的语义**:`asyncio.create_task()` 创建新 task 时,context 是**浅拷贝**;不同 task 修改各自的 contextvar 互不影响(已通过最小测试实地验证)
2. **ToolAgentLoop 的执行模型**:一条 trajectory 的整个生命周期(pending → generating → processing_tools → interacting → ...)**都在同一个 `asyncio.create_task` 里顺序执行**,contextvar 值贯穿整个循环
3. **同 task_id 的 group_size 条 rollout**:veRL 用 `asyncio.create_task(agent_loop.run(...))` 为每条 rollout 创建独立 task,contextvar **天然隔离**

**Rubric 评分更新(方案 I 加入)**:

| 维度 | 方案 A | 方案 B | 方案 C(旧) | **方案 I(新)** |
|---|---|---|---|---|
| 符合 veRL 原生设计 | ✗ | ✗ | ✓ 但依赖 instance_id 反查(实际不成立) | ✓ 用 veRL 不反对的 contextvar |
| Function calling schema 正确 | ✗ | ✓ | ✓ | ✓ |
| env 状态一致性 | ✓ | ✗ | 声称可以,实际机制不通 | ✓ asyncio 原生隔离 |
| group_size>1 下的并发 env | ✗(Tool 无法区分) | ✗(同上) | ✗(instance_id 共享失败) | ✓ contextvar per-task |
| 代码量 | 低 | 高 | 中 | 中(少了 ENV_REGISTRY 的并发锁) |
| τ-bench contract 保留 | ✗ | ✓ | ✓ | ✓ |
| 调试复杂度 | 低 | 高 | 中(但反查机制失败后难查) | 中(contextvar 行为可测试) |

**决策:采用方案 I**,锁定。

---

## §3 关键接口契约

### 3.1 TauBenchToolBase 与工厂函数

**关键变更**:
- 移除 `ENV_REGISTRY` 反查逻辑(原设计里 `ENV_REGISTRY.get(instance_id)` 在当前 ToolAgentLoop 架构下拿不到有效 env)
- Tool 改为从 `contextvars.ContextVar` 读 env
- `create` 和 `release` 都变成 no-op(env 生命周期完全由 Interaction 管)

```python
# src/envs/tau_bench_context.py
import contextvars

CURRENT_TAU_ENV: contextvars.ContextVar = contextvars.ContextVar(
    "current_tau_env", default=None
)
# 用于追踪本轮 trajectory 的累计 reward 和 turn count,方便 Interaction 在 finalize 时读取
CURRENT_TAU_STATE: contextvars.ContextVar = contextvars.ContextVar(
    "current_tau_state", default=None
)
```

```python
# src/envs/tau_bench_tools.py
from typing import Any
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from tau_bench.types import Action

from src.envs.tau_bench_context import CURRENT_TAU_ENV, CURRENT_TAU_STATE

class TauBenchToolBase(BaseTool):
    """
    τ-bench 13 个 airline tool 的基类。通过 contextvar 读 env,不持有任何状态。
    子类由 _make_tau_bench_tool_class(tool_schema) 动态生成。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # self.name 已由 BaseTool.__init__ 设置为 tool_schema.function.name

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        # env 生命周期归 Interaction,Tool.create 是 no-op
        # verl v0.6.1 要求 create 返回 (instance_id, ToolResponse)
        from uuid import uuid4
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        env = CURRENT_TAU_ENV.get()
        state = CURRENT_TAU_STATE.get()
        if env is None or state is None:
            # 理论上不应该发生:Interaction.start_interaction 一定先跑
            return (
                ToolResponse(text="[ERROR] TauBench env not bound to current context"),
                0.0,
                {"error": "no_env_in_context"},
            )

        try:
            action = Action(name=self.name, kwargs=parameters)
            step_res = env.step(action)
        except Exception as e:
            # env.step 自身抛异常(比如 parameters schema 不匹配)
            return (
                ToolResponse(text=f"[TOOL_ERROR] {type(e).__name__}: {e}"),
                0.0,
                {"error": "env_step_exception"},
            )

        obs = str(getattr(step_res, "observation", ""))
        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))

        # 更新 state(这个 state 对象绑定在当前 trajectory 的 contextvar 里)
        state["total_reward"] += inc_reward
        state["num_tool_calls"] += 1
        if is_done:
            state["done"] = True

        return (
            ToolResponse(text=obs),
            0.0,  # step-level reward 留给后续 reward shaping 方案,Week 3 不用
            {"inc_reward": inc_reward, "done": is_done},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0  # 所有 reward 通过 Interaction.generate_response 的返回值传回

    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # no-op,env 在 Interaction.finalize_interaction 里 release


def _make_tau_bench_tool_class(schema: OpenAIFunctionToolSchema) -> type[TauBenchToolBase]:
    """为每个 τ-bench tool schema 动态创建一个子类。"""
    cls_name = f"TauBench_{schema.function.name}_Tool"
    return type(cls_name, (TauBenchToolBase,), {})
```

**为什么不直接用 `TauBenchToolBase` 作为单一类**:veRL 的 `initialize_tools_from_config` 按 class_name 实例化 tool,**同一个 class_name 只会实例化一次**。如果 13 个 tool 共用一个 class_name,只会加载一个 tool 的 schema,模型只能看到 1 个 function。必须 13 个独立类。

**静态生成 vs 动态生成**:动态生成(`_make_tau_bench_tool_class` + 在 `__init__.py` 里导出)可以,但 Hydra 的 `instantiate` 需要**通过模块路径 + 类名**找到类。动态生成的类如果没绑定到一个模块级的名字(`sys.modules[...]`),Hydra 找不到。

**实现约定**:提供一个模块 `src/envs/tau_bench_tools.py`,在 import 时调用 `_generate_all_tau_bench_tool_classes()`,把所有 13 个类作为模块级属性挂上去:

```python
# src/envs/tau_bench_tools.py 末尾
def _generate_all_tau_bench_tool_classes():
    """
    启动时从 airline env 的 tools_info 拿 13 个 schema,动态生成 13 个子类,
    挂到 sys.modules[__name__] 上,让 Hydra instantiate 能按 class_name 找到。
    
    调用时机: scripts/10_gen_tool_config.py 启动前 import 这个模块时执行。
    """
    import sys
    from tau_bench.envs import get_env
    env = get_env(env_name="airline", user_strategy="llm", user_model="dummy",
                  user_provider="openai", task_split="test", task_index=0)
    module = sys.modules[__name__]
    for tool_schema_dict in env.tools_info:
        schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        cls = _make_tau_bench_tool_class(schema)
        setattr(module, cls.__name__, cls)

_generate_all_tau_bench_tool_classes()
```

**兜底方案**:如果动态生成 + Hydra 找类有问题(常见坑:pickle / inspect 找不到类定义),**降级为手写 13 个空子类**:

```python
class TauBench_get_reservation_details_Tool(TauBenchToolBase): pass
class TauBench_book_reservation_Tool(TauBenchToolBase): pass
# ... 其余 11 个
```

代价:13 行样板代码。Week 3 Day 1 先试动态,硬阻塞就手写。

### 3.2 TauBenchInteraction(持有 env,做 user simulation)

**接口签名**(继承 `verl.interactions.base.BaseInteraction`):

```python
# src/envs/tau_bench_interaction.py
import logging
import uuid
from typing import Any, Optional

from verl.interactions.base import BaseInteraction
from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME

from src.envs.tau_bench_context import CURRENT_TAU_ENV, CURRENT_TAU_STATE

logger = logging.getLogger(__name__)

FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]


def _has_forbidden_token(content: str) -> bool:
    return any(tok in content for tok in FORBIDDEN_TEMPLATE_TOKENS)


class TauBenchInteraction(BaseInteraction):
    """
    τ-bench 的 user simulator 适配层。
    持有 env 实例(per-trajectory),通过 contextvar 把 env 暴露给 Tool。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_name = config.get("env_name", "airline")
        self.user_model = config.get("user_model", "Qwen/Qwen2.5-72B-Instruct-AWQ")
        self.user_base_url = config.get("user_base_url", "http://localhost:8001/v1")
        self.user_strategy = config.get("user_strategy", "llm")
        self.user_provider = config.get("user_provider", "openai")
        self.task_split = config.get("task_split", "test")
        self.max_turns = config.get("max_turns", 30)
        # instance_id -> {env, state_dict}; 用于 finalize 时清理
        self._instance_dict: dict[str, dict] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        task_id: int = 0,                # 从 dataset 的 interaction_kwargs.task_id 来
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        env = get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=int(task_id),
        )
        env.reset(task_index=int(task_id))

        state = {
            "total_reward": 0.0,
            "num_tool_calls": 0,
            "num_user_turns": 0,
            "done": False,
            "contaminated": False,
            "task_id": int(task_id),
        }

        # 关键:绑定到当前 asyncio task 的 contextvar
        # asyncio.create_task 会 fork context,同一个 trajectory 后续的 Tool.execute
        # 会看到这里 set 的值;不同 trajectory 之间天然隔离
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        self._instance_dict[instance_id] = {"env": env, "state": state}

        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict],
        **kwargs,
    ) -> tuple[bool, str, float, dict]:
        """
        被触发的时机:assistant 刚刚输出了一条没有 tool_calls 的 message。
        需要用 env.step(RESPOND_ACTION) 驱动 user simulator 回复。
        """
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return True, "", 0.0, {"error": "no_instance"}
        env = entry["env"]
        state = entry["state"]

        # Defensive: 重新 set contextvar,保证本函数结束前 CURRENT_TAU_ENV 可用
        # (ToolAgentLoop 的状态机已经保证同 trajectory 顺序执行,这里是防御式写法)
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        # 提取最新一条 assistant content
        assistant_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                assistant_content = m.get("content", "") or ""
                break

        # §3.3 污染检测
        if _has_forbidden_token(assistant_content):
            state["contaminated"] = True
            state["done"] = True
            # reward=0 + terminate(§3.3 锁定,Week 3 不引入 shaping)
            return True, "", 0.0, {
                "contaminated": True,
                "reason": "forbidden_template_token",
                "total_reward": state["total_reward"],
                "num_turns": state["num_user_turns"] + state["num_tool_calls"],
            }

        # 正常路径:触发 user reply
        try:
            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content})
            step_res = env.step(action)
        except Exception as e:
            logger.warning(f"env.step(RESPOND) failed: {e}")
            state["done"] = True
            return True, "", 0.0, {"error": "respond_exception", "reason": str(e)}

        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        state["total_reward"] += inc_reward
        state["num_user_turns"] += 1

        total_turns = state["num_user_turns"] + state["num_tool_calls"]
        if is_done or total_turns >= self.max_turns:
            state["done"] = True
            final_score = 1.0 if state["total_reward"] >= 1.0 else 0.0
            return True, "", final_score, {
                "total_reward": state["total_reward"],
                "num_turns": total_turns,
                "num_tool_calls": state["num_tool_calls"],
                "num_user_turns": state["num_user_turns"],
            }

        user_reply = str(getattr(step_res, "observation", ""))
        return False, user_reply, 0.0, {
            "turn": total_turns,
            "num_tool_calls": state["num_tool_calls"],
        }

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return 0.0
        return 1.0 if entry["state"]["total_reward"] >= 1.0 else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
        # contextvar 不需要显式清理,asyncio task 结束时自动释放
```

### 3.3 env 生命周期 + contextvar 语义

#### 3.3.1 env 生命周期

| 事件 | 触发函数 | 操作 |
|---|---|---|
| Trajectory 开始 | `Interaction.start_interaction(request_id, task_id=..)` | `get_env() + env.reset()`,`CURRENT_TAU_ENV.set(env)` |
| Assistant 有 tool_calls | `Tool.execute(instance_id, parameters)` | `CURRENT_TAU_ENV.get()` → `env.step(Action(name, kwargs))` |
| Assistant 无 tool_calls | `Interaction.generate_response(request_id, messages)` | 重新 `set` + `env.step(Action("respond", content))` |
| Trajectory 结束 | `Interaction.finalize_interaction(request_id)` | 清 `_instance_dict`,contextvar 随 asyncio task 死亡 |

#### 3.3.2 contextvar 并发语义(验证过)

**测试代码**(已在 Claude computer 环境实地跑过并通过):

```python
# 模拟 4 条并发 rollout(同一个 task_id=38,group_size=4)
import asyncio, contextvars

ENV = contextvars.ContextVar("env", default=None)

async def agent_loop(tid):
    ENV.set(f"env_instance_{tid}_{id(object())}")  # 模拟 start_interaction
    for _ in range(3):
        await asyncio.sleep(0.01)
        # 模拟 Tool.execute 读取 contextvar
        assert ENV.get().startswith(f"env_instance_{tid}")
    return True

async def main():
    tasks = [asyncio.create_task(agent_loop(38)) for _ in range(4)]
    await asyncio.gather(*tasks)  # 4 条并发,contextvar 互不干扰

asyncio.run(main())
```

**语义保证**:
- `asyncio.create_task` 创建新 task 时 **fork context**,每个 task 有自己的 contextvar 副本
- 同一个 task 内 `set` 的值对该 task 的所有后续代码可见
- 不同 task 的 `set` 互不影响
- **验证结论**:group_size=4 下 4 条 rollout 的 Tool.execute 能正确拿到各自 Interaction.start_interaction 里 set 的 env

#### 3.3.3 污染处理(锁定)

| 阶段 | 污染 reward | 终止行为 | 实现位置 |
|---|---|---|---|
| Week 3 mock + vanilla GRPO | 0.0 | terminate | `Interaction.generate_response` 的污染分支 |
| Week 4 改进方案 1 | 可切 -0.5 | terminate | 改 `FORBIDDEN_REWARD` 常量 |

`FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]`,与 `vllm_policy.py` 的 Week 2 补丁保持一致。

### 3.4 数据格式:Week 3 训练集 parquet

**关键变更**:取消 `traj_uid` 列(veRL repeat 机制让这个方案不成立)。parquet 每行对应一个 task,`rollout.n=4` 由 veRL 在运行时展开。

**parquet schema**:

| 列名 | 类型 | 示例值 | 说明 |
|---|---|---|---|
| `prompt` | list[dict] | `[{"role":"system","content":"<date_grounding_prompt>"}]` | 只含 system,user message 由 Interaction 生成 |
| `extra_info` | dict | `{"index":<row_index>, "task_id":<int>, "split":"seen"\|"unseen", "interaction_kwargs":{"name":"tau_bench_airline","task_id":<int>}}` | veRL 从这里拿 sample_index、task_id、interaction 路由 |
| `tools_kwargs` | dict | `{}` | 空即可,Tool 不需要 per-task 参数(env 从 contextvar 拿) |
| `ability` | str | `"tau_bench_airline"` | veRL 的 sanity 字段,可作为 agent_name 的 fallback |

**`extra_info.index`** 是 veRL 计算 `sample_index` 用的(agent_loop.py line 643),必须是整数。一般用 `range(len(parquet))`。

**`agent_name`** 的来源:
- 方式 A(新版):parquet 有 `agent_name` 列,每行值为 `"tool_agent"`
- 方式 B(推荐,Week 3 用):在 hydra config 里设 `actor_rollout_ref.rollout.agent.default_agent_name="tool_agent"`

选 B,避免 parquet 多一列。具体字段名以 v0.6.1 config schema 为准,Day 1 确认。

**训练集大小**:
- `experiments/week3_mock/train.parquet` — seen 40 task,每行一个 task,共 40 行
- `experiments/week3_mock/val.parquet` — 评测时跑 50 task(overall),共 50 行
- `rollout.n=4` 下,实际 rollout 数 = 40 × 4 = 160 条/step

**系统 prompt**(写在 `prompt` 列):

```python
SYSTEM_PROMPT = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, "
    "always assume they refer to 2024. "
    "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
)
```

与 Week 1/2 的 `tau_bench_wrapper.py` 里 `system_content` **完全一致**,保持跨 Week 可比。

**生成脚本约定** `scripts/11_build_grpo_parquet.py`:

```python
# 伪代码
import pandas as pd
seen = [0, 1, 2, 4, ...]   # 40 个,从 metadata.json 读
train_rows = []
for idx, tid in enumerate(seen):
    train_rows.append({
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
        "extra_info": {
            "index": idx,
            "task_id": tid,
            "split": "seen",
            "interaction_kwargs": {"name": "tau_bench_airline", "task_id": tid},
        },
        "tools_kwargs": {},
        "ability": "tau_bench_airline",
    })
pd.DataFrame(train_rows).to_parquet("experiments/week3_mock/train.parquet")
```

---

## §4 配置文件设计

### 4.1 `configs/tool_config/tau_bench_airline_tools.yaml`

**动态生成而非手写**:Week 3 Day 1 启动脚本会先调用 `env.tools_info` 拿到 13 个 schema,自动生成这份 yaml:

```yaml
tools:
  - class_name: "src.envs.tau_bench_tools.TauBench_get_reservation_details_Tool"
    config:
      type: native
    tool_schema:
      type: function
      function:
        name: get_reservation_details
        description: "..."  # 从 env.tools_info 拷贝
        parameters: {...}
  # ... 其余 12 个 tool
```

**生成器**:`scripts/10_gen_tool_config.py`(Week 3 Day 1 产出)。

### 4.2 `configs/interaction_config/tau_bench_airline.yaml`

```yaml
interaction:
  - name: "tau_bench_airline"
    class_name: "src.envs.tau_bench_interaction.TauBenchInteraction"
    config:
      env_name: "airline"
      user_model: "Qwen/Qwen2.5-72B-Instruct-AWQ"
      user_base_url: "http://localhost:8001/v1"
      task_split: "test"  # airline 官方只有 test split,50 task 全用
      max_turns: 30
```

### 4.3 主 GRPO 配置(5-step mock,Week 3 Day 2 验收用)

```yaml
# configs/week3_mock_grpo.yaml
hydra:
  searchpath:
    - pkg://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: experiments/week3_mock/train.parquet
  val_files: experiments/week3_mock/val.parquet
  train_batch_size: 16
  max_prompt_length: 4096
  max_response_length: 12288  # airline trajectory 可能长,给足空间
  return_raw_chat: True        # ToolAgentLoop 需要

actor_rollout_ref:
  model:
    path: experiments/week2_sft_lora/merged/        # Week 2 SFT 产物
  actor:
    optim:
      lr: 5.0e-6                                    # PROJECT.md 锁定
    use_kl_loss: True
    kl_loss_coef: 0.01                              # PROJECT.md 锁定
    kl_loss_type: low_var_kl
    ppo_mini_batch_size: 16
    ppo_micro_batch_size_per_gpu: 2
  rollout:
    name: "vllm"                                    # 本次决策锁定
    multi_turn:
      enable: True
      format: "hermes"                              # 对齐 Qwen2.5 tool parser
      max_user_turns: 15
      max_assistant_turns: 15
    tool_kwargs:
      tools_config_file: configs/tool_config/tau_bench_airline_tools.yaml
    interaction_kwargs:
      interaction_config_file: configs/interaction_config/tau_bench_airline.yaml
    n: 4                                            # group_size=4 (§5 锁定)
    temperature: 0.7                                # 和 W1/W2 对齐
    top_p: 0.9
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.55                    # 给训练留显存
    max_num_seqs: 4                                 # Week 2 踩过的坑

algorithm:
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.01

trainer:
  total_epochs: 1
  total_training_steps: 5                           # Mock: 5 step
  save_freq: -1                                     # mock 不 save
  test_freq: -1                                     # mock 不 eval
  logger: ['console']                               # mock 不传 swanlab
  project_name: agentic-grpo-longhorizon
  experiment_name: week3_mock_5step
  n_gpus_per_node: 1
  nnodes: 1
```

### 4.4 vLLM ↔ SGLang 降级条件

**继续用 vLLM**,但 design 层面预留降级口。触发切换的信号:

- Mock GRPO 5 step 里有 3 step 以上因 tool_call parsing 失败报错
- rollout 时频繁出现 `tool_calls` 字段为 None 但 content 里有 `<tool_call>` 标签(意味着 veRL 的 vLLM backend 没 unpack tool_calls)
- Qwen2.5 的 hermes parser 和 veRL 的 parsed 字段对不上,导致 Tool.execute 拿不到正确 parameters

**降级操作**:
1. `actor_rollout_ref.rollout.name` 从 `"vllm"` 改为 `"sglang"`
2. 启动独立的 SGLang server(端口 8002,不动 GPU0 的 72B-AWQ vLLM)
3. 重装依赖:`pip install "sglang[all]==0.4.6"`(与 veRL v0.6.x 兼容)
4. 其他 config 不动

Week 3 Day 1-2 **不提前切**,按 vLLM 跑。只有硬阻塞才切。

---

## §5 验收里程碑

### Milestone 1:Day 1 完成——静态检查通过

| 检查项 | 通过标准 |
|---|---|
| veRL v0.6.1 装成功 | `import verl; print(verl.__version__)` 输出 0.6.1 |
| TauBenchToolBase 单元测试 | `TauBench_get_reservation_details_Tool(config, schema).execute(instance_id, {"reservation_id": "H8Q05L"})` 返回合法 ToolResponse |
| TauBenchInteraction 单元测试 | `start_interaction` 成功注册 env,`generate_response` 能触发 user reply |
| parquet 生成 | `experiments/week3_mock/train.parquet` 存在,40 行,每行有 interaction_kwargs.task_id |
| tool_config.yaml 自动生成 | 13 个 class_name 齐全 |

### Milestone 2:Day 2 完成——5-step mock GRPO 不崩

| 检查项 | 通过标准 |
|---|---|
| 训练启动 | `python -m verl.trainer.main_ppo --config-path configs --config-name week3_mock_5step` 不报 ImportError/ConfigError |
| Rollout 成功 | 每个 step 至少产出 16 条 trajectory(train_batch_size),每条 num_turns ≥ 1 |
| Reward 有值 | 16 条里至少 1 条 reward=1.0(basic sanity,SFT checkpoint 在 covered task 上应能成功) |
| loss 有值 | `actor/loss` 不为 NaN |
| KL 有值 | `actor/kl_loss` 在 [0, 0.5] 区间 |
| 5 step 在 2 小时内完成 | wall time < 120 min |

### Milestone 3:Day 3 完成——wandb 曲线可诊断

| 检查项 | 通过标准 |
|---|---|
| swanlab 日志齐全 | actor/loss, actor/kl_loss, rollout/reward_mean, rollout/response_length, rollout/num_turns 都有 |
| 无 NaN | 所有 metric 在 5 step 内无 NaN |
| num_turns 合理 | 平均 5-15 turn,不全是 1 turn(否则 tool 没调用) |
| 能保存 checkpoint | step 5 时 `save_freq=5` 能产出可 load 的 LoRA |

**Milestone 3 通过后,Day 4 开启 500 step 正式 vanilla GRPO,进入 `vanilla_grpo_diagnosis.md` 的产出阶段。**

---

## §6 风险与降级预案

### 6.1 vLLM multi-turn + tool parsing 不稳定

**表现**:rollout 时 `tool_calls` 字段为 None 但 content 里有 `<tool_call>` 标签。

**根因**:veRL 的 vLLM backend(截至 v0.6.1)对 hermes parser 的支持不如 SGLang 完善(参考 veRL issue #344 / #3195)。

**降级**:按 §4.4 切 SGLang,成本 1-2 天(不含 baseline 重测,因为 rollout backend 不影响 pass^k 评测)。

### 6.2 72B-AWQ user sim 被 GRPO rollout 压崩

**表现**:GPU0 的 72B vLLM server OOM / 响应超时。

**根因**:GRPO 的 group_size=4 会让 user sim 的 QPS 翻 4 倍,`max-num-seqs=4` 可能不够。

**降级**:
- Plan A:`max-num-seqs=2`,牺牲 rollout throughput,换稳定
- Plan B:group_size 从 4 降到 2(对 GRPO variance 不利,但 Week 3 诊断 reward 稀疏问题反而**放大信号**,可接受)
- Plan C(硬阻塞时):user sim 切到 32B 版本(需要重测 W1 baseline,放弃)

### 6.3 rollout 吞吐过低导致 5 step 无法在 2 小时内完成

**表现**:Milestone 2 wall time > 120 min。

**根因**:max_response_length=12288 太大,或 max_turns=15 让单条轨迹过长。

**降级**:
- Plan A:`max_response_length=8192`(放弃 Week 2 发现的 5 条长轨迹所在的那类 task)
- Plan B:`max_turns=10`(会让部分 task 无法完成,但 Week 3 诊断阶段可接受——这本身就是"长链路"问题的信号)
- Plan C:`train_batch_size=8`,用 step 数换时间

### 6.4 KL 初始爆炸

**表现**:step 1-2 的 `kl_loss` > 1.0。

**根因**:SFT checkpoint 和 base 7B 的 policy distribution 已经偏离较大,reference model 应该用 SFT checkpoint 而非 base。

**降级**:确认 `actor_rollout_ref.ref.model.path` 指向 SFT merged checkpoint 而非 base。如果已经指向,降 `kl_loss_coef` 到 0.001 先跑通,后续调。

### 6.5 env 实例状态撕裂(ENV_REGISTRY 的锁失效)

**表现**:同一个 instance_id 下 Tool 和 Interaction 看到的 env 状态不一致。

**根因**:veRL 的 AgentLoop 可能在 coroutine 调度上跨线程(未确认)。

**降级**:`_EnvRegistry` 的锁粒度从 method-level 提升到 instance_id-level(用 per-instance 的 asyncio.Lock),成本 < 20 行代码。

### 6.6 contextvar 边界情况与降级

#### 6.6.1 场景 A:跨线程调用

**风险**:ToolAgentLoop 内部某个步骤通过 `asyncio.to_thread` 或 `loop.run_in_executor` 把 Tool.execute 调到另一个线程时,contextvar 可能丢失(asyncio.to_thread 会保留 context,但线程池可能不会)。

**检测**:v0.6.1 的 `tool_agent_loop.py` 查一遍是否有跨线程 tool 调用;`grep -n "run_in_executor\|to_thread" tool_agent_loop.py`

**降级**:如果发现跨线程,回退到"方案 C 变体":用 `threading.local()` + 从 parameters 里嵌入一个 context_id(通过 hack 在 tools_kwargs 里动态注入)。

#### 6.6.2 场景 B:ToolAgentLoop 实例跨 trajectory 复用

**风险**:contextvar 在 trajectory A 的 start_interaction 里 set,之后 trajectory B 的 Tool.execute 读到 A 的 env。

**检测**:在 `start_interaction` 里打印 `id(env)`,在 `Tool.execute` 里打印 `id(CURRENT_TAU_ENV.get())`,Day 2 的 mock GRPO 5 step 输出里对照。

**降级**:加一个"ownership check"——state 里存 `request_id`,Tool.execute 验证 `state["request_id"] == instance_id`(不一致则抛错,退到 ENV_REGISTRY 方案)。

#### 6.6.3 场景 C:asyncio task cancellation 导致 finalize_interaction 未调用

**风险**:`_instance_dict` 泄漏,长训可能 OOM。

**缓解**:`_instance_dict` 里只存 env 和 state 两个引用,一条 trajectory 不到 1KB,500 step × 40 task × 4 group = 80K 条,极端情况下最多 80MB,可接受。

**降级**:加一个周期性 sweep,清理超过 max_idle_time 的 entry。

---

## §7 明确排除的事(不做清单)

**设计决策的边界声明**——以下事项 Week 3 不做,避免 scope creep:

1. **不自己实现 AgentLoopBase**:复用 veRL 的 ToolAgentLoop,即使它在未来 v0.5.x 会被重构也接受
2. **不把 Week 2 轨迹转 parquet 作为 offline 数据**:GRPO 是 on-policy
3. **不做 reward shaping**:Week 3 的 reward 严格等于 τ-bench outcome reward(污染轨迹 reward=0)
4. **不做 turn-level advantage**:这是 Week 4 改进方案 2 的内容
5. **不做 PRM**:超出 6-8 周 scope,列入 Week 5 "可选"项
6. **不测 retail**:Week 3 只关注 airline
7. **不换 Qwen2.5 家族**:即使 base → SFT 出现 behavior cloning 反向,也用这个 checkpoint(Week 2 已决策)
8. **不改 `pass_k_eval.run_eval`**:Week 3 的 validation 完全复用 W1/W2 的评测代码
9. **不预判 Week 3 诊断结果**:§5 的 Milestone 只验证 pipeline 工作,不预判 `vanilla_grpo_diagnosis.md` 的内容

---

## 附录 A:锁定参数表

| 参数 | 值 | 来源 | 可变更条件 |
|---|---|---|---|
| **RL 框架版本** | veRL v0.6.1 | §1.1 | 硬阻塞才切 v0.4.3+(不切 v0.5.x) |
| **Rollout backend** | vLLM | §1.1 + §4.4 | 按 §6.1 切 SGLang |
| **Policy model** | Qwen2.5-7B-Instruct + Week 2 SFT LoRA merged | PROJECT.md | 不变 |
| **Ref model** | Week 2 SFT LoRA merged(同 policy 初始值) | §6.4 | 不变 |
| **User sim** | Qwen2.5-72B-Instruct-AWQ @ port 8001 | §1.1 | 按 §6.2 切 32B |
| **Group size** | 4 | 本次决策(从 PROJECT.md 的 8 下调) | 显存不足降到 2 |
| **Train batch size** | 16 | §4.3 | 时间不够降到 8 |
| **PPO mini batch size** | 16 | §4.3 | 固定 = train_batch_size |
| **PPO micro batch/GPU** | 2 | §4.3 | 显存不足降到 1 |
| **Actor lr** | 5e-6 | PROJECT.md §2.3 | 不变 |
| **KL loss coef** | 0.01 | PROJECT.md §2.3 | 按 §6.4 临时降到 0.001 |
| **KL loss type** | low_var_kl | §4.3 | 不变 |
| **Max prompt length** | 4096 | §4.3 | 不变 |
| **Max response length** | 12288 | §4.3 | 按 §6.3 降到 8192 |
| **Max turns(训练)** | 15 | §4.3 | 按 §6.3 降到 10 |
| **Max turns(评测)** | 30 | §4.2 | 不变(与 W1/W2 对齐) |
| **Rollout temperature** | 0.7 | §4.3 | 不变 |
| **Rollout top_p** | 0.9 | §4.3 | 不变 |
| **污染处理** | reward=0 + terminate | §3.3 | Week 4 改进方案 1 可切 -0.5 |
| **Task split** | airline test,50 task | §4.2 | 不变 |
| **Mock GRPO steps** | 5 | §4.3 | 不变 |
| **Vanilla GRPO steps** | 500 | PROJECT.md Week 3 | 不变 |
| **Vanilla GRPO eval 间隔** | 每 50 step | PROJECT.md | 不变 |
| **env 共享机制** | contextvars 隐式共享 | §2.4 | 硬阻塞时切方案 C 变体(§6.6) |
| **env_registry 模块** | `src/envs/tau_bench_context.py` | §3.3 | 不变 |
| **parquet 列** | 不含 `traj_uid` | §3.4 | 不变 |
| **污染处理实现位置** | 仅 Interaction.generate_response | §3.3 | 不变 |

## 附录 B:Week 3 Day 1-3 可执行命令

```bash
# ===== Day 1: 环境 + 配置 =====
# 1. 装 veRL
pip install verl==0.6.1
pip install -U vllm==0.7.3  # 避开 0.8.x 的 tool parser regression,同时 > 0.7.x bug

# 2. 生成 tool config(先启 72B server + 7B server 后运行)
python scripts/10_gen_tool_config.py \
    --env airline \
    --task-id 0 \
    --output configs/tool_config/tau_bench_airline_tools.yaml

# 3. 生成训练 parquet
python scripts/11_build_grpo_parquet.py \
    --seen-task-ids-from experiments/week2_sft_data/metadata.json \
    --output-train experiments/week3_mock/train.parquet \
    --output-val experiments/week3_mock/val.parquet

# 4. 单元测试(不依赖 veRL trainer)
pytest tests/test_tau_bench_tool.py -v
pytest tests/test_tau_bench_interaction.py -v
pytest tests/test_env_registry.py -v

# ===== Day 2: Mock GRPO 5 step =====
# 前置: GPU0 72B-AWQ server + GPU1 7B-SFT server 已起(复用 W2 脚本)
# bash scripts/01_start_vllm_server_72b.sh      # tmux session: user_sim
# bash scripts/01_start_vllm_server_7b_sft.sh   # tmux session: policy

# Mock GRPO(会接管 GPU1,需要先停 7B server)
python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs \
    --config-name=week3_mock_5step \
    2>&1 | tee experiments/week3_mock/training.log

# ===== Day 3: Milestone 验证 =====
# 1. 看日志里是否有 NaN
grep -E "nan|NaN" experiments/week3_mock/training.log || echo "[PASS] no NaN"

# 2. 验证 checkpoint 可 load
python scripts/12_verify_checkpoint.py \
    --path experiments/week3_mock/checkpoints/step_5

# 3. 快速 sanity eval(4 task × 2 sample,走 Mock checkpoint)
python scripts/02_run_baseline_eval.py \
    --config configs/eval_sft_airline.yaml \
    --tiny \
    --policy-path experiments/week3_mock/checkpoints/step_5
```

## 附录 C:下一份文档的锚点

Week 3 完成后需要产出的文档:
- `docs/vanilla_grpo_diagnosis.md`(Week 3 Day 4-7 产出)——诊断 4 类核心问题(reward 稀疏 / credit misassignment / 长度-entropy 漂移 / KL 失控)
- 本文档与 `vanilla_grpo_diagnosis.md` 的边界:**本文档定义的是 pipeline 怎么搭,`vanilla_grpo_diagnosis.md` 记录的是跑起来之后观察到什么**

## 附录 D:文档修订条件

本文档在以下条件下需要更新:
1. veRL v0.6.1 的 `BaseTool` / `BaseInteraction` / `ToolAgentLoop` 签名与 §3 记录的不一致(以实际源码为准,更新 §3)
2. Milestone 2 不通过且根因是 §6 未列出的新问题(增加 §6.7+)
3. 污染处理从 reward=0 改为其他策略(更新 §3.3 表格 + 附录 A)
4. Rollout backend 从 vLLM 切 SGLang(更新 §4.3 config + §4.4 状态)

**不更新的情况**:Week 4 改进方案的参数、诊断阶段的观察、post-hoc 的性能数字——这些归属于其他文档。
