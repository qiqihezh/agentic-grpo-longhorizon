"""
Per-trajectory 共享的 context 变量,用于 TauBenchTool 和 TauBenchInteraction 之间传递 env 实例。

设计要点(见 design doc §2.4 / §3.3):
- veRL ToolAgentLoop 为每条 trajectory 用 asyncio.create_task 创建独立 coroutine
- asyncio 会为每个 task fork 一份 context,contextvar 修改互不干扰
- 同一条 trajectory 内,Interaction.start_interaction 先 set,后续 Tool.execute 在同一个 task 里读取
- group_size=G 下 G 条同 task_id 的 rollout 各自 set 一个 env 实例,天然隔离

已实地验证(见 design doc §3.3.2 测试代码)。
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

# 当前 trajectory 的 τ-bench env 实例
# 生命周期: Interaction.start_interaction 里 set → Tool.execute 读 → trajectory 结束自然释放
CURRENT_TAU_ENV: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "current_tau_env", default=None
)

# 当前 trajectory 的累计状态(reward, turn count, done flag 等)
# Interaction 和 Tool 都会读写
CURRENT_TAU_STATE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "current_tau_state", default=None
)


def make_initial_state(task_id: int) -> dict:
    """构造 trajectory 级 state 字典的标准初值。Interaction.start_interaction 调用。"""
    return {
        "task_id": int(task_id),
        "total_reward": 0.0,
        "num_tool_calls": 0,
        "num_user_turns": 0,
        "done": False,
        "contaminated": False,
    }
