"""
TauBenchInteraction: veRL BaseInteraction 的 τ-bench 实现。

职责:
1. 每条 trajectory 创建独立的 τ-bench env 实例(task_id 从 interaction_kwargs 传入)
2. 把 env 和 state 绑到当前 asyncio task 的 contextvar(让 Tool 能读到)
3. 驱动 user simulator(通过 env.step(RESPOND_ACTION))
4. 检测污染 trajectory(assistant 输出含 forbidden template token),直接终止

W4 新增:
- reward_mode 配置化切换: binary(原始) / partial_credit(未执行) / prm_lite(Exp 3)
- 通过 interaction config 的 reward_mode 字段控制,不改代码即可切换实验
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from src.envs.tau_bench_context import (
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    CURRENT_ASSISTANT_CONTENT,
    make_initial_state,
)


def record_assistant_content(content: str) -> None:
    """ToolAgentLoop 在调用 tool.execute 前记录当前 turn 的 assistant content。
    TauBenchTool.execute 读取此值存入 action_history，用于 cheap reasoning 检测。"""
    CURRENT_ASSISTANT_CONTENT.set(content)

logger = logging.getLogger(__name__)


# 与 src/models/vllm_policy.py 的 FORBIDDEN_TEMPLATE_TOKENS 保持一致
# (SFT 采集阶段验证过: assistant 输出这些 token 意味着长 context format drift)
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


# ============================================================================
# Reward computation functions (W4-W5 ablation)
# ============================================================================

# PRM-Lite v3: tool categories based on tau-bench airline actual tools
_READ_TOOLS = frozenset({
    "list_all_airports", "search_direct_flight", "search_onestop_flight",
    "get_user_details", "get_reservation_details", "calculate",
})
_WRITE_TOOLS = frozenset({
    "book_reservation", "cancel_reservation", "update_reservation_baggages",
    "update_reservation_passengers", "update_reservation_flights", "send_certificate",
})
_ESCALATION_TOOLS = frozenset({"transfer_to_human_agents"})
_THINK_TOOLS = frozenset({"think", "implicit_think"})

# Schema-based parameter validation patterns (from tau_bench_airline_tools.yaml)
_PARAM_PATTERNS = {
    "reservation_id": re.compile(r'^[A-Z0-9]{6}$'),
    "user_id": re.compile(r'^[a-z]+_[a-z]+_[0-9]+$'),
    "payment_id": re.compile(r'^(credit_card|gift_card|certificate)_[0-9]+$'),
    "flight_number": re.compile(r'^[A-Z]{3}[0-9]{3}$'),
    "origin": re.compile(r'^[A-Z]{3}$'),
    "destination": re.compile(r'^[A-Z]{3}$'),
    "date": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
}

_PLACEHOLDER_KEYWORDS = frozenset({
    "previous", "unknown", "placeholder", "none", "null", "n/a",
    "any", "some", "first", "last", "default", "example", "sample",
    "test", "dummy", "temp", "temporary",
})


def _param_str(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False).lower()


def _is_placeholder_param(field_name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lower = value.lower()
    if any(kw in lower for kw in _PLACEHOLDER_KEYWORDS):
        return True
    for key, pattern in _PARAM_PATTERNS.items():
        if key in field_name.lower():
            if not pattern.match(value):
                return True
            return False
    return False


def _has_placeholder(params: dict) -> bool:
    for field_name, value in params.items():
        if _is_placeholder_param(field_name, value):
            return True
    return False


def _is_redundant(action_history: list[dict], current_tool: str, current_params: dict, window: int = 3) -> bool:
    current_sig = (current_tool, _param_str(current_params))
    for prev in action_history[-window:]:
        prev_tool = prev.get("tool", "")
        if prev_tool in _THINK_TOOLS:
            continue
        prev_sig = (prev_tool, prev.get("param_str", ""))
        if prev_sig == current_sig:
            return True
    return False


def _compute_binary_reward(state: dict) -> float:
    """W3 原始: outcome >= 1.0 → 1, 否则 0"""
    return 1.0 if state["total_reward"] >= 1.0 else 0.0


def _compute_partial_credit_reward(state: dict) -> float:
    """Partial-Credit: 成功=1.0, 失败=基于参与度+区分性信号的 0~0.3 partial credit"""
    if state["total_reward"] >= 1.0:
        return 1.0
    turn_ratio = min(state["num_user_turns"] / 8.0, 1.0)
    tool_ratio = min(state["num_tool_calls"] / 5.0, 1.0)
    surrender = 0.3 if state.get("transferred_to_human", False) else 0.0

    # tool call 成功率: 区分"有效探索"和"无效重复"
    history = state.get("action_history", [])
    if history:
        error_count = sum(1 for a in history if a.get("done") and a["inc_reward"] <= 0)
        success_ratio = 1.0 - (error_count / len(history))
    else:
        success_ratio = 0.0

    # 三个信号加权: 参与度(0.3) + 工具使用(0.3) + 成功率(0.4)
    raw = 0.3 * turn_ratio + 0.3 * tool_ratio + 0.4 * success_ratio
    score = 0.3 * raw - surrender
    return max(0.0, min(score, 0.3))


def _compute_reasoning_quality_score(action_history: list[dict]) -> float:
    """
    PRM-Lite v4-optimal: rule-based per-step reasoning quality scorer.
    Hyperparameters tuned via offline grid search on vanilla eval data.
    Key design principles:
    1. Mean-based (not sum-based) to avoid long-trajectory penalty flattening
    2. Penalize "busy but wrong" (length penalty + no implicit_think reward)
    3. Reward signal concentrated in high-value actions (data chain, recovery)
    4. Force reasoning (no-reasoning penalty strengthened)
    """
    if not action_history:
        return 0.0

    per_step_scores = []

    for i, action in enumerate(action_history):
        tool = action["tool"]
        params = action.get("parameters", {})
        pstr = action.get("param_str", "")
        score = 0.0

        # --- Core Penalties ---

        # P1: Placeholder penalty (schema-based)
        if tool not in _THINK_TOOLS and _has_placeholder(params):
            score += (-0.05 if tool in _WRITE_TOOLS else -0.03)

        # P2: Redundancy (same tool+params in last 3 steps)
        if tool not in _THINK_TOOLS and _is_redundant(action_history[:i], tool, params, window=3):
            score -= 0.03

        # P3: Error repetition vs recovery
        if i >= 1 and tool not in _THINK_TOOLS:
            prev = action_history[i - 1]
            if prev.get("is_error", False):
                prev_sig = (prev.get("tool", ""), prev.get("param_str", ""))
                curr_sig = (tool, pstr)
                if curr_sig == prev_sig:
                    score -= 0.04
                else:
                    score += 0.05

        # P4: Escalation penalty (layered: did we even try to gather info?)
        if tool in _ESCALATION_TOOLS:
            has_done_read = any(
                prev.get("tool") in _READ_TOOLS
                for prev in action_history[:i]
            )
            score += (-0.10 if not has_done_read else -0.05)

        # --- Positive Incentives ---

        # B1: Data chain (parameter value appeared in previous extracted entities)
        if i >= 1 and tool not in _THINK_TOOLS and params:
            seen_entities = set()
            for prev in action_history[:i]:
                for ent_list in prev.get("extracted_entities", {}).values():
                    seen_entities.update(ent_list)
            used = any(isinstance(v, str) and v in seen_entities for v in params.values())
            if used:
                score += (0.08 if tool in _WRITE_TOOLS else 0.04)

        # B2: First read exploration (diverse info gathering)
        if tool in _READ_TOOLS:
            seen_reads = set(
                prev["tool"] for prev in action_history[:i]
                if prev.get("tool") in _READ_TOOLS
            )
            if tool not in seen_reads:
                score += 0.01

        # B3: Removed — success_bonus caused anti-incentive in long failures

        # B4/B5: Think bonus with anti-hacking (tuned to +0.01 via grid search)
        if tool in _THINK_TOOLS:
            # (a) consecutive think: no bonus
            if i >= 1 and action_history[i - 1].get("tool") in _THINK_TOOLS:
                pass
            # (b) think is the last step: no bonus (didn't guide any action)
            elif i == len(action_history) - 1:
                pass
            # (c) think followed by placeholder/redundancy: no bonus
            elif i + 1 < len(action_history):
                next_action = action_history[i + 1]
                next_tool = next_action.get("tool", "")
                next_params = next_action.get("parameters", {})
                if _has_placeholder(next_params) or _is_redundant(
                    action_history[:i + 1], next_tool, next_params, window=3
                ):
                    pass
                else:
                    score += 0.01
            else:
                score += 0.01

        # B6: Implicit think reward REMOVED (grid search found it inflates long-failure scores)

        # P9: Cheap reasoning penalty — assistant content too short before tool call
        # 直接打击"没想清楚就调 tool"的敷衍行为 (SFT baseline 中 66.7% turns 受此罚)
        # 注意: content=="" 表示未记录(旧数据/测试)，不触发惩罚; 0 < len(content) < 30 才罚
        if tool not in _THINK_TOOLS:
            content = action.get("content", "")
            if 0 < len(content) < 30:
                score -= 0.02

        per_step_scores.append(score)

    mean_score = sum(per_step_scores) / len(per_step_scores) if per_step_scores else 0.0

    # --- Trajectory-level adjustments (not averaged) ---

    # P5: No reasoning penalty (strengthened to -0.05 via grid search)
    think_count = sum(1 for a in action_history if a["tool"] in _THINK_TOOLS)
    if think_count == 0 and len(action_history) >= 3:
        mean_score -= 0.05

    # B7: Read tool diversity bonus (reduced to +0.01 via grid search)
    all_reads = set(a["tool"] for a in action_history if a.get("tool") in _READ_TOOLS)
    if len(all_reads) >= 3:
        mean_score += 0.01

    # P8: Length penalty — penalize excessively long trajectories
    # v4-optimal grid search: threshold=8, per_step=-0.01
    length_threshold = 8
    length_penalty_per_step = -0.01
    if len(action_history) > length_threshold:
        mean_score += length_penalty_per_step * (len(action_history) - length_threshold)

    return float(max(-0.5, min(0.5, mean_score)))


def _compute_prm_lite_reward(state: dict) -> float:
    """Exp 3 v4: outcome + 0.3 * process (un-normalized, process as independent signal)"""
    outcome = 1.0 if state["total_reward"] >= 1.0 else 0.0
    history = state.get("action_history", [])
    process_score = _compute_reasoning_quality_score(history)
    return outcome + 0.3 * process_score


_REWARD_FUNCTIONS = {
    "binary": _compute_binary_reward,
    "partial_credit": _compute_partial_credit_reward,
    "prm_lite": _compute_prm_lite_reward,
}


class TauBenchInteraction(BaseInteraction):
    """τ-bench user simulator 在 veRL 侧的适配层。"""

    def __init__(self, config: dict):
        super().__init__(config)
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

        # W4: reward mode switching
        self.reward_mode: str = config.get("reward_mode", "binary")
        if self.reward_mode not in _REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward_mode '{self.reward_mode}'. "
                f"Valid: {list(_REWARD_FUNCTIONS.keys())}"
            )
        self._compute_reward = _REWARD_FUNCTIONS[self.reward_mode]
        logger.info(f"[TauBenchInteraction] reward_mode={self.reward_mode}")

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

        # [W5 PRM-Lite v3] 记录 implicit think: assistant 纯文本回复 > 100 chars
        if assistant_content and len(assistant_content) > 100:
            # 防御：避免同一 assistant message 被重复记录
            last_action = state["action_history"][-1] if state["action_history"] else None
            is_duplicate = (
                last_action
                and last_action.get("tool") == "implicit_think"
                and last_action.get("content", "") == assistant_content[:300]
            )
            if not is_duplicate:
                state["action_history"].append({
                    "tool": "implicit_think",
                    "parameters": {},
                    "param_str": "",
                    "inc_reward": 0,
                    "done": False,
                    "is_error": False,
                    "extracted_entities": {},
                    "content": assistant_content[:300],
                })

        # 污染检测: 锁定 reward=0 + terminate(§3.3)
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
            final_score = self._compute_reward(state)
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
                    "reward_mode": self.reward_mode,
                    "transferred_to_human": state.get("transferred_to_human", False),
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

    async def calculate_score(self, instance_id: str, **kwargs) -> float | dict:
        """Turn-level score: uses configured reward_mode.

        W5 conditional PRM: returns dict with score, outcome_score, process_score
        so that trainer can decide whether to apply PRM based on group std.
        """
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return {"score": 0.0, "outcome_score": 0.0, "process_score": 0.0}
        state = entry["state"]
        outcome = 1.0 if state["total_reward"] >= 1.0 else 0.0
        process = _compute_reasoning_quality_score(state.get("action_history", []))

        if self.reward_mode == "binary":
            score = outcome
        else:
            score = outcome + 0.3 * process

        return {"score": score, "outcome_score": outcome, "process_score": process}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Trajectory 结束时清理 _instance_dict 避免内存泄漏"""
        self._instance_dict.pop(instance_id, None)
        # contextvar 随 asyncio task 死亡自动释放,不需要显式 reset
