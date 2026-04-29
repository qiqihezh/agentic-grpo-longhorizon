"""
TauBenchToolBase + 14 个 τ-bench airline tool 的静态子类。

设计要点(见 design doc §3.1,已根据 review 修订):
- 14 个独立子类,veRL 按 class_name 实例化时一个 schema 对应一个类
- 静态定义在本文件中,避免 cloudpickle 序列化动态类的坑(Ray/FSDP 多进程场景)
- 启动时有一致性校验: verify_tool_classes_match_env() 检查静态类名与 env.tools_info 完全对齐
- Tool 本身 stateless, env 通过 contextvar 从当前 asyncio task 获取

关键修订(相对初稿):
1. 动态 type() 生成 → 静态 class 定义(pickle-safe)
2. "no env in context" 从返回错误字符串 → raise RuntimeError(fail loud)
3. Tool error 格式 "[TOOL_ERROR] XXX" → "Error: XXX"(对齐 τ-bench 原生)
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from src.envs.tau_bench_context import CURRENT_TAU_ENV, CURRENT_TAU_STATE

logger = logging.getLogger(__name__)


# ============================================================================
# 基类
# ============================================================================

class TauBenchToolBase(BaseTool):
    """
    τ-bench tool 的基类。14 个子类都继承它,不重写任何方法。
    self.name 由 BaseTool.__init__ 从 tool_schema.function.name 设置,
    Tool.execute 用 self.name 构造 τ-bench Action。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """veRL 每次 tool call 都会新建一个 instance_id,create 是 no-op。"""
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """
        1. 从 contextvar 拿当前 trajectory 的 env(Interaction.start_interaction 设置)
        2. 调 env.step(Action(tool_name, parameters))
        3. observation 回传给模型,inc_reward 累计到 state

        step-level reward 永远返回 0.0: Week 3 锁定 reward 走 Interaction 的 outcome
        (Week 4 改进方案若引入 step reward,改这里)
        """
        env = CURRENT_TAU_ENV.get()
        state = CURRENT_TAU_STATE.get()

        # 【修订 1】Fail loud: 宁可整个 batch 崩,也不让带毒 trajectory 进 GRPO
        if env is None or state is None:
            raise RuntimeError(
                f"[CRITICAL] TauBench env/state missing from contextvar for tool "
                f"'{self.name}'. This means Interaction.start_interaction did not "
                f"run in the same asyncio task as this Tool.execute call. "
                f"Check veRL ToolAgentLoop wiring (expected: single asyncio task "
                f"per trajectory, contextvar fork on asyncio.create_task)."
            )

        from tau_bench.types import Action

        try:
            action = Action(name=self.name, kwargs=parameters)
            step_res = env.step(action)
        except Exception as e:
            # env.step 自身异常(参数 schema 不匹配 / env 已 done 被再 step / 等)
            # 【修订 3】Error 格式对齐 τ-bench 原生: 一句自然语言,无前缀
            # (τ-bench 自己的 tool error 样例: "Unknown action update_reservation_insurance")
            err_msg = f"Error: {type(e).__name__}: {e}"
            logger.warning(f"[Tool.execute] {self.name} raised: {err_msg}")
            return (
                ToolResponse(text=err_msg),
                0.0,
                {"error": "env_step_exception", "tool": self.name, "detail": str(e)},
            )

        obs = str(getattr(step_res, "observation", ""))
        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))

        state["total_reward"] += inc_reward
        state["num_tool_calls"] += 1
        if is_done:
            state["done"] = True

        return (
            ToolResponse(text=obs),
            0.0,  # step reward = 0 (Week 3)
            {"inc_reward": inc_reward, "done": is_done, "tool": self.name},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


# ============================================================================
# 静态子类定义(14 个 airline tool,一个 schema 一个类)
#
# 为什么要静态定义而不是 type() 动态生成:
#   - cloudpickle(Ray/FSDP 跨进程传对象时用的)对 "动态 type() 创建且挂到模块"的类
#     序列化是 best-effort,在某些 Python / cloudpickle 版本组合下会 AttributeError
#   - 14 行样板代码换 100% pickle 可靠性,划算
#
# 若未来 τ-bench 加/减 tool,启动时 verify_tool_classes_match_env() 会报错,
# 提示手工更新本文件。
# ============================================================================

class TauBench_book_reservation_Tool(TauBenchToolBase): pass
class TauBench_calculate_Tool(TauBenchToolBase): pass
class TauBench_cancel_reservation_Tool(TauBenchToolBase): pass
class TauBench_get_reservation_details_Tool(TauBenchToolBase): pass
class TauBench_get_user_details_Tool(TauBenchToolBase): pass
class TauBench_list_all_airports_Tool(TauBenchToolBase): pass
class TauBench_search_direct_flight_Tool(TauBenchToolBase): pass
class TauBench_search_onestop_flight_Tool(TauBenchToolBase): pass
class TauBench_send_certificate_Tool(TauBenchToolBase): pass
class TauBench_think_Tool(TauBenchToolBase): pass
class TauBench_transfer_to_human_agents_Tool(TauBenchToolBase): pass
class TauBench_update_reservation_baggages_Tool(TauBenchToolBase): pass
class TauBench_update_reservation_flights_Tool(TauBenchToolBase): pass
class TauBench_update_reservation_passengers_Tool(TauBenchToolBase): pass


# ============================================================================
# 一致性校验
# ============================================================================

# airline 14 个 tool 的权威列表(与上面静态类一一对应)
AIRLINE_TOOL_NAMES = [
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "think",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
]


def get_tool_class_by_name(tool_name: str) -> type[TauBenchToolBase]:
    """根据 tool_name 查对应的类。scripts/10_gen_tool_config.py 用。"""
    cls_name = f"TauBench_{tool_name}_Tool"
    module = sys.modules[__name__]
    if not hasattr(module, cls_name):
        raise KeyError(
            f"No tool class {cls_name} defined in {__name__}. "
            f"If τ-bench added a new tool '{tool_name}', add a static class here."
        )
    return getattr(module, cls_name)


def verify_tool_classes_match_env(
    env_name: str = "airline",
    task_index: int = 0,
) -> list[str]:
    """
    验证本模块定义的 14 个静态类和 env.tools_info 一一对应,不多不少。
    应在 scripts/10_gen_tool_config.py 启动时调用,不一致立刻报错。

    Returns:
        env.tools_info 里的 tool 名列表(调用方可用来生成 yaml)。
    Raises:
        AssertionError: 静态类名和 env tool 名不一致时。
    """
    from tau_bench.envs import get_env

    env = get_env(
        env_name=env_name,
        user_strategy="llm",
        user_model="dummy",
        user_provider="openai",
        task_split="test",
        task_index=task_index,
    )
    env_tool_names = sorted(t["function"]["name"] for t in env.tools_info)
    static_tool_names = sorted(AIRLINE_TOOL_NAMES)

    missing_in_static = set(env_tool_names) - set(static_tool_names)
    extra_in_static = set(static_tool_names) - set(env_tool_names)

    errors = []
    if missing_in_static:
        errors.append(
            f"Tools in env but no static class: {sorted(missing_in_static)}"
        )
    if extra_in_static:
        errors.append(
            f"Static classes but not in env: {sorted(extra_in_static)}"
        )

    if errors:
        raise AssertionError(
            "Static tool classes out of sync with τ-bench env:\n  " +
            "\n  ".join(errors) +
            f"\nUpdate AIRLINE_TOOL_NAMES and static class definitions in "
            f"{__file__}."
        )

    # 还要验证每个静态类都是 TauBenchToolBase 的子类
    for tn in AIRLINE_TOOL_NAMES:
        cls = get_tool_class_by_name(tn)
        assert issubclass(cls, TauBenchToolBase), f"{cls} is not TauBenchToolBase subclass"

    logger.info(
        f"Tool class <-> env consistency check PASSED: {len(env_tool_names)} tools matched."
    )
    return env_tool_names
