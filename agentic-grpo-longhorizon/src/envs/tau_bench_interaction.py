"""
TauBenchInteraction: veRL BaseInteraction 的 τ-bench 实现。

职责:
1. 每条 trajectory 创建独立的 τ-bench env 实例(task_id 从 interaction_kwargs 传入)
2. 把 env 和 state 绑到当前 asyncio task 的 contextvar(让 Tool 能读到)
3. 驱动 user simulator(通过 env.step(RESPOND_ACTION))
4. 检测污染 trajectory(assistant 输出含 forbidden template token),直接终止

不做的事:
- 不做 reward shaping(Week 3 锁定: outcome reward 0/1,污染也是 0)
- 不做 step-level / turn-level score(留给 Week 4 改进方案)
- 不做 PRM(超出项目 scope)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from src.envs.tau_bench_context import (
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    make_initial_state,
)

logger = logging.getLogger(__name__)


# 与 src/models/vllm_policy.py 的 FORBIDDEN_TEMPLATE_TOKENS 保持一致
# (Week 2 SFT 采集阶段验证过: assistant 输出这些 token 意味着长 context format drift)
FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]


def _has_forbidden_token(content: str) -> bool:
    if not content:
        return False
    return any(tok in content for tok in FORBIDDEN_TEMPLATE_TOKENS)


def _extract_latest_assistant_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


class TauBenchInteraction(BaseInteraction):
    """τ-bench user simulator 在 veRL 侧的适配层。"""

    def __init__(self, config: dict):
        super().__init__(config)
        # config 字段都来自 configs/interaction_config/tau_bench_airline.yaml
        self.env_name: str = config.get("env_name", "airline")
        self.user_strategy: str = config.get("user_strategy", "llm")
        self.user_model: str = config.get(
            "user_model", "Qwen/Qwen2.5-72B-Instruct-AWQ"
        )
        self.user_provider: str = config.get("user_provider", "openai")
        self.user_base_url: str = config.get(
            "user_base_url", "http://localhost:8001/v1"
        )
        self.task_split: str = config.get("task_split", "test")
        self.max_turns: int = int(config.get("max_turns", 30))

        # instance_id -> {"env": ..., "state": ...}
        # 仅用于 finalize_interaction 能找到 env 做清理
        # 核心 env 访问走 contextvar(见 design §3.3)
        self._instance_dict: dict[str, dict] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        task_id: int = 0,
        **kwargs,
    ) -> str:
        """
        ToolAgentLoop 里每条 trajectory 开始时调用一次。
        在这里创建 env 实例并绑定到 contextvar。

        Args:
            instance_id: ToolAgentLoop 生成的 request_id(trajectory-unique uuid)。
                如果为 None 则自己生成。
            task_id: 从 parquet 的 extra_info.interaction_kwargs.task_id 读出来的 task index
                (0 ~ 49 for airline)。
        """
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        # 延迟 import,避免单测时强依赖 tau_bench 包
        from tau_bench.envs import get_env

        task_id_int = int(task_id)
        env = get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=task_id_int,
        )
        # τ-bench 的 reset 在 get_env 里已经调过一次,但显式再 reset 一遍稳妥
        env.reset(task_index=task_id_int)

        state = make_initial_state(task_id_int)

        # 关键: 绑定到当前 asyncio task 的 context
        # 同一个 coroutine 后续的 Tool.execute 会读到这里 set 的 env
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        # 备份引用: finalize 时清理用,以及 generate_response 里 defensive re-set
        self._instance_dict[instance_id] = {"env": env, "state": state}

        logger.debug(
            f"[start_interaction] instance={instance_id[:8]} task_id={task_id_int} "
            f"env_id={id(env)}"
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        被 ToolAgentLoop 在 AgentState.INTERACTING 触发(assistant 输出了不带 tool_calls 的 message)。

        Returns:
            (should_terminate, user_response_content, reward, metadata)
            - should_terminate: True 则本 trajectory 结束
            - user_response_content: 返回给模型的 user reply(空串 = terminate 时不需要)
            - reward: 本 turn 的 reward(终止时是 final outcome reward,否则 0)
            - metadata: 诊断用(num_turns, contaminated, error 等)
        """
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            # 【修订 1】Fail loud: start_interaction 必须先于 generate_response,
            # 不满足说明 ToolAgentLoop 生命周期被破坏,继续跑会产生带毒 trajectory
            raise RuntimeError(
                f"[CRITICAL] TauBenchInteraction.generate_response called for "
                f"instance_id={instance_id} but no corresponding entry found. "
                f"Either start_interaction was never called, or this Interaction "
                f"instance is different from the one that handled start_interaction. "
                f"Check veRL Interaction lifecycle."
            )

        env = entry["env"]
        state = entry["state"]

        # Defensive re-set: ToolAgentLoop 的状态机在同一个 coroutine 内顺序执行,
        # 理论上 start_interaction 里 set 的值一直有效,但重新 set 一遍无副作用。
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        assistant_content = _extract_latest_assistant_content(messages)

        # 污染检测: Week 3 锁定 reward=0 + terminate(§3.3)
        if _has_forbidden_token(assistant_content):
            state["contaminated"] = True
            state["done"] = True
            logger.info(
                f"[generate_response] FORBIDDEN_TOKEN detected in task {state['task_id']}, "
                f"terminating with reward=0"
            )
            return (
                True,
                "",
                0.0,
                {
                    "contaminated": True,
                    "reason": "forbidden_template_token",
                    "total_reward": state["total_reward"],
                    "num_turns": state["num_user_turns"] + state["num_tool_calls"],
                    "task_id": state["task_id"],
                },
            )

        # 正常路径: 驱动 user simulator
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        try:
            action = Action(
                name=RESPOND_ACTION_NAME,
                kwargs={"content": assistant_content},
            )
            step_res = env.step(action)
        except Exception as e:
            # env 内部 exception(一般是 user simulator 返回异常,或 env 已 done 被重复 step)
            logger.warning(
                f"[generate_response] env.step(RESPOND) failed for task "
                f"{state['task_id']}: {type(e).__name__}: {e}"
            )
            state["done"] = True
            return (
                True,
                "",
                0.0,
                {
                    "error": "respond_exception",
                    "reason": f"{type(e).__name__}: {e}",
                    "task_id": state["task_id"],
                },
            )

        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        state["total_reward"] += inc_reward
        state["num_user_turns"] += 1

        total_turns = state["num_user_turns"] + state["num_tool_calls"]

        # 终止条件: env 说 done / 超 max_turns
        if is_done or total_turns >= self.max_turns:
            state["done"] = True
            # Week 3 outcome reward: >= 1.0 记 1,否则 0
            final_score = 1.0 if state["total_reward"] >= 1.0 else 0.0
            return (
                True,
                "",
                final_score,
                {
                    "total_reward": state["total_reward"],
                    "num_turns": total_turns,
                    "num_tool_calls": state["num_tool_calls"],
                    "num_user_turns": state["num_user_turns"],
                    "task_id": state["task_id"],
                    "reason": "done" if is_done else "max_turns",
                },
            )

        # 继续交互: 返回 user reply
        user_reply = str(getattr(step_res, "observation", ""))
        return (
            False,
            user_reply,
            0.0,
            {
                "turn": total_turns,
                "num_tool_calls": state["num_tool_calls"],
                "task_id": state["task_id"],
            },
        )

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Turn-level score(Week 3 不用,返回 outcome)"""
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return 0.0
        return 1.0 if entry["state"]["total_reward"] >= 1.0 else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Trajectory 结束时清理 _instance_dict 避免内存泄漏"""
        self._instance_dict.pop(instance_id, None)
        # contextvar 随 asyncio task 死亡自动释放,不需要显式 reset
