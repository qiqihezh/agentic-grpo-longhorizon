#!/usr/bin/env python3
"""
PRM-Lite 超参网格搜索（纯 CPU，零 GPU）

目标：在不占用 GPU 的前提下，通过离线验证数据找出最优规则权重组合。

评价指标（多目标优化）：
1. intra_group_std: saturated group 的 process_score 标准差 → 越高越好（GRPO 需要 variance）
2. bad_behavior_score: 含 placeholder/redundancy 的 trajectory 平均分数 → 越负越好
3. length_correlation: ntools=2 与 ntools>=15 的分数差 → 越高越好（防止长 failure 自动高分）
4. short_success_score: ntools=2 的 success trajectory 平均分数 → 越高越好（保护正确短路径）

综合得分 = intra_group_std * 2.0 + (-bad_behavior_score) * 1.5 + length_correlation * 1.0 + short_success_score * 0.5
"""
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

_PROJECT_ROOT = Path(__file__).resolve().parent
while not (_PROJECT_ROOT / "src").is_dir():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from src.envs.tau_bench_interaction import _compute_reasoning_quality_score

_PARAM_PATTERNS = {
    "reservation_id": re.compile(r'^[A-Z0-9]{6}$'),
    "user_id": re.compile(r'^[a-z]+_[a-z]+_[0-9]+$'),
    "payment_id": re.compile(r'^(credit_card|gift_card|certificate)_[0-9]+$'),
    "flight_number": re.compile(r'^[A-Z]{3}[0-9]{3}$'),
    "origin": re.compile(r'^[A-Z]{3}$'),
    "destination": re.compile(r'^[A-Z]{3}$'),
    "date": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
}


def _extract_entities(obs_str: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for ent_name, pattern in _PARAM_PATTERNS.items():
        for m in pattern.finditer(obs_str):
            val = m.group(0)
            if val not in out[ent_name]:
                out[ent_name].append(val)
    return dict(out)


def trajectory_to_action_history(raw_messages: list[dict]) -> list[dict]:
    history = []
    tool_results = {}
    for msg in raw_messages:
        if msg.get("role") == "tool":
            tool_results[msg.get("tool_call_id", "")] = msg
    for msg in raw_messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                tc_id = tc.get("id", "")
                result_msg = tool_results.get(tc_id, {})
                obs = result_msg.get("content", "")
                is_error = str(obs).startswith("Error:")
                history.append({
                    "tool": func.get("name", ""),
                    "parameters": args,
                    "param_str": json.dumps(args, sort_keys=True, ensure_ascii=False).lower(),
                    "is_error": is_error,
                    "extracted_entities": _extract_entities(str(obs)),
                    "content": "",
                })
        elif len(content) > 100 and not tool_calls:
            if not history or history[-1]["tool"] != "implicit_think":
                history.append({
                    "tool": "implicit_think",
                    "parameters": {},
                    "param_str": "{}",
                    "is_error": False,
                    "extracted_entities": {},
                    "content": content,
                })
    return history


@dataclass
class Weights:
    placeholder_write: float = -0.05
    placeholder_read: float = -0.03
    redundancy: float = -0.03
    error_repetition: float = -0.04
    recovery: float = 0.05
    escalation_premature: float = -0.10
    escalation_late: float = -0.05
    data_chain_write: float = 0.08
    data_chain_read: float = 0.04
    first_read: float = 0.01
    think_bonus: float = 0.05
    implicit_think_medium: float = 0.03
    implicit_think_long: float = 0.01
    no_reasoning_penalty: float = -0.03
    diversity_bonus: float = 0.03
    # 新增：length penalty
    length_penalty_threshold: int = 999  # 不启用
    length_penalty_per_step: float = 0.0
    # 新增：excessive exploration penalty
    excessive_explore_threshold: int = 999
    excessive_explore_penalty: float = 0.0


def compute_score_with_weights(history: list[dict], w: Weights) -> float:
    if not history:
        return 0.0

    from src.envs.tau_bench_interaction import _THINK_TOOLS, _READ_TOOLS, _WRITE_TOOLS, _ESCALATION_TOOLS, _has_placeholder, _is_redundant

    per_step_scores = []
    tool_type_counts = defaultdict(int)

    for i, action in enumerate(history):
        tool = action["tool"]
        params = action.get("parameters", {})
        pstr = action.get("param_str", "")
        score = 0.0

        # P1: Placeholder
        if tool not in _THINK_TOOLS and _has_placeholder(params):
            score += (w.placeholder_write if tool in _WRITE_TOOLS else w.placeholder_read)

        # P2: Redundancy
        if tool not in _THINK_TOOLS and _is_redundant(history[:i], tool, params, window=3):
            score += w.redundancy

        # P3/P4: Error repetition vs recovery
        if i >= 1 and tool not in _THINK_TOOLS:
            prev = history[i - 1]
            if prev.get("is_error", False):
                prev_sig = (prev.get("tool", ""), prev.get("param_str", ""))
                curr_sig = (tool, pstr)
                score += (w.error_repetition if curr_sig == prev_sig else w.recovery)

        # P5: Escalation
        if tool in _ESCALATION_TOOLS:
            has_read = any(p.get("tool") in _READ_TOOLS for p in history[:i])
            score += (w.escalation_premature if not has_read else w.escalation_late)

        # B1: Data chain
        if i >= 1 and tool not in _THINK_TOOLS and params:
            seen = set()
            for prev in history[:i]:
                for lst in prev.get("extracted_entities", {}).values():
                    seen.update(lst)
            if any(isinstance(v, str) and v in seen for v in params.values()):
                score += (w.data_chain_write if tool in _WRITE_TOOLS else w.data_chain_read)

        # B2: First read
        if tool in _READ_TOOLS:
            seen_reads = set(p["tool"] for p in history[:i] if p.get("tool") in _READ_TOOLS)
            if tool not in seen_reads:
                score += w.first_read

        # B4/B5: Think bonus
        if tool in _THINK_TOOLS:
            if (i >= 1 and history[i - 1].get("tool") in _THINK_TOOLS) or (i == len(history) - 1):
                pass
            elif i + 1 < len(history):
                nxt = history[i + 1]
                if not (_has_placeholder(nxt.get("parameters", {})) or _is_redundant(history[:i + 1], nxt.get("tool", ""), nxt.get("parameters", {}), window=3)):
                    score += w.think_bonus
            else:
                score += w.think_bonus

        # B6: Implicit think
        if tool == "implicit_think":
            cl = len(action.get("content", ""))
            if 100 < cl < 300:
                score += w.implicit_think_medium
            elif 300 <= cl < 600:
                score += w.implicit_think_long

        # Excessive exploration penalty
        if tool in _READ_TOOLS:
            tool_type_counts[tool] += 1
            if tool_type_counts[tool] > w.excessive_explore_threshold:
                score += w.excessive_explore_penalty

        per_step_scores.append(score)

    mean_score = sum(per_step_scores) / len(per_step_scores) if per_step_scores else 0.0

    # P6: No reasoning penalty
    think_count = sum(1 for a in history if a["tool"] in _THINK_TOOLS)
    if think_count == 0 and len(history) >= 3:
        mean_score += w.no_reasoning_penalty

    # B7: Read diversity
    all_reads = set(a["tool"] for a in history if a.get("tool") in _READ_TOOLS)
    if len(all_reads) >= 3:
        mean_score += w.diversity_bonus

    # Length penalty
    if len(history) > w.length_penalty_threshold:
        mean_score += w.length_penalty_per_step * (len(history) - w.length_penalty_threshold)

    return float(max(-0.5, min(0.5, mean_score)))


def evaluate_weights(eval_data: dict, w: Weights) -> dict:
    """在 eval 数据上评估一组权重，返回多个指标。"""
    all_scores = []
    success_scores = []
    failure_scores = []
    bad_behavior_scores = []
    ntools2_success = []
    ntools15plus_failure = []
    task_scores = defaultdict(list)

    for task in eval_data.get("per_task_results", []):
        tid = task["task_id"]
        for traj in task["trajectories"]:
            history = trajectory_to_action_history(traj.get("raw_messages", []))
            if not history:
                continue
            score = compute_score_with_weights(history, w)
            success = traj.get("success", False)
            ntools = len(history)

            all_scores.append(score)
            task_scores[tid].append(score)

            if success:
                success_scores.append(score)
                if ntools == 2:
                    ntools2_success.append(score)
            else:
                failure_scores.append(score)
                if ntools >= 15:
                    ntools15plus_failure.append(score)

            # Bad behavior detection
            has_bad = False
            for i, a in enumerate(history):
                if a["tool"] not in {"think", "implicit_think"}:
                    for k, v in a.get("parameters", {}).items():
                        if isinstance(v, str):
                            if v.lower() in {"previous", "unknown", "placeholder", "none", "null", "n/a", "any", "some", "first", "last", "default", "example", "sample", "test", "dummy", "temp", "temporary"}:
                                has_bad = True
                            if k in {"reservation_id", "user_id", "flight_number"}:
                                if k == "reservation_id" and not re.match(r'^[A-Z0-9]{6}$', v):
                                    has_bad = True
                                if k == "user_id" and not re.match(r'^[a-z]+_[a-z]+_[0-9]+$', v):
                                    has_bad = True
                if i >= 1:
                    prev = history[i - 1]
                    if prev.get("tool") == a["tool"] and prev.get("param_str") == a.get("param_str"):
                        has_bad = True
            if has_bad:
                bad_behavior_scores.append(score)

    # Intra-group variance (saturated groups)
    saturated_stds = []
    for tid, scores in task_scores.items():
        if len(scores) >= 2:
            std = np.std(scores)
            saturated_stds.append(std)

    intra_group_std = np.mean(saturated_stds) if saturated_stds else 0.0
    bad_behavior_mean = np.mean(bad_behavior_scores) if bad_behavior_scores else 0.0
    length_corr = (np.mean(ntools2_success) if ntools2_success else 0.0) - (np.mean(ntools15plus_failure) if ntools15plus_failure else 0.0)
    short_success_mean = np.mean(ntools2_success) if ntools2_success else 0.0

    # 综合得分
    composite = (
        intra_group_std * 2.0 +
        (-bad_behavior_mean) * 1.5 +
        length_corr * 1.0 +
        short_success_mean * 0.5
    )

    return {
        "intra_group_std": intra_group_std,
        "bad_behavior_mean": bad_behavior_mean,
        "length_correlation": length_corr,
        "short_success_mean": short_success_mean,
        "success_mean": np.mean(success_scores) if success_scores else 0.0,
        "failure_mean": np.mean(failure_scores) if failure_scores else 0.0,
        "composite": composite,
        "n_evaluated": len(all_scores),
    }


def main():
    print("=" * 90)
    print("  PRM-Lite v4-fix 超参网格搜索（纯 CPU）")
    print("=" * 90)

    # 加载 eval 数据
    eval_path = Path("experiments/vanilla/eval_step_150/eval_report.json")
    with open(eval_path) as f:
        eval_data = json.load(f)

    # 定义搜索空间
    search_space = {
        "think_bonus": [0.05, 0.03, 0.02, 0.01],
        "implicit_think_medium": [0.03, 0.02, 0.01, 0.00],
        "implicit_think_long": [0.01, 0.00],
        "data_chain_write": [0.08, 0.06, 0.04],
        "recovery": [0.05, 0.04, 0.03],
        "no_reasoning_penalty": [-0.03, -0.05],
        "diversity_bonus": [0.03, 0.02, 0.01],
        "length_penalty_threshold": [999, 12, 10, 8],
        "length_penalty_per_step": [0.0, -0.005, -0.01],
    }

    # 生成所有组合（可能会很多，做渐进式搜索）
    # 先搜索小空间：think + implicit + length penalty
    print("\n🔍 Phase 1: 搜索 think/implicit/length penalty（约 144 种组合）")
    phase1_keys = ["think_bonus", "implicit_think_medium", "implicit_think_long", "length_penalty_threshold", "length_penalty_per_step"]
    phase1_values = [search_space[k] for k in phase1_keys]

    results = []
    base = Weights()

    for combo in product(*phase1_values):
        w = Weights(
            think_bonus=combo[0],
            implicit_think_medium=combo[1],
            implicit_think_long=combo[2],
            length_penalty_threshold=combo[3],
            length_penalty_per_step=combo[4],
        )
        metrics = evaluate_weights(eval_data, w)
        results.append((w, metrics))

    # 按 composite 排序
    results.sort(key=lambda x: x[1]["composite"], reverse=True)

    print(f"\n  已评估 {len(results)} 种组合")
    print("\n  🏆 Top 10 组合：")
    print("-" * 90)
    print(f"{'rank':>4} {'think':>5} {'imp_m':>5} {'imp_l':>5} {'len_th':>6} {'len_pn':>6} | {'intra':>6} {'bad':>6} {'len_cr':>7} {'short':>6} | {'composite':>9}")
    print("-" * 90)

    for rank, (w, m) in enumerate(results[:10], 1):
        print(
            f"{rank:>4} {w.think_bonus:>5.2f} {w.implicit_think_medium:>5.2f} {w.implicit_think_long:>5.2f} "
            f"{w.length_penalty_threshold:>6} {w.length_penalty_per_step:>6.3f} | "
            f"{m['intra_group_std']:>6.3f} {m['bad_behavior_mean']:>6.3f} {m['length_correlation']:>7.3f} {m['short_success_mean']:>6.3f} | "
            f"{m['composite']:>9.3f}"
        )

    # Phase 2: 基于 top3，搜索 data_chain + recovery + diversity
    print("\n" + "=" * 90)
    print("🔍 Phase 2: 基于 top3 搜索 data_chain/recovery/diversity/no_reasoning")
    print("=" * 90)

    top3 = results[:3]
    phase2_keys = ["data_chain_write", "recovery", "diversity_bonus", "no_reasoning_penalty"]
    phase2_values = [search_space[k] for k in phase2_keys]

    results2 = []
    for w_base, _ in top3:
        for combo in product(*phase2_values):
            w = Weights(
                think_bonus=w_base.think_bonus,
                implicit_think_medium=w_base.implicit_think_medium,
                implicit_think_long=w_base.implicit_think_long,
                length_penalty_threshold=w_base.length_penalty_threshold,
                length_penalty_per_step=w_base.length_penalty_per_step,
                data_chain_write=combo[0],
                recovery=combo[1],
                diversity_bonus=combo[2],
                no_reasoning_penalty=combo[3],
            )
            metrics = evaluate_weights(eval_data, w)
            results2.append((w, metrics))

    results2.sort(key=lambda x: x[1]["composite"], reverse=True)

    print(f"\n  已评估 {len(results2)} 种组合")
    print("\n  🏆 Top 10 组合（Phase 2）：")
    print("-" * 110)
    print(f"{'rank':>4} {'think':>5} {'imp_m':>5} {'imp_l':>5} {'len_th':>6} {'len_pn':>6} {'dc_w':>5} {'recov':>5} {'div':>4} {'no_r':>5} | {'intra':>6} {'bad':>6} {'len_cr':>7} {'short':>6} | {'composite':>9}")
    print("-" * 110)

    for rank, (w, m) in enumerate(results2[:10], 1):
        print(
            f"{rank:>4} {w.think_bonus:>5.2f} {w.implicit_think_medium:>5.2f} {w.implicit_think_long:>5.2f} "
            f"{w.length_penalty_threshold:>6} {w.length_penalty_per_step:>6.3f} "
            f"{w.data_chain_write:>5.2f} {w.recovery:>5.2f} {w.diversity_bonus:>4.2f} {w.no_reasoning_penalty:>5.2f} | "
            f"{m['intra_group_std']:>6.3f} {m['bad_behavior_mean']:>6.3f} {m['length_correlation']:>7.3f} {m['short_success_mean']:>6.3f} | "
            f"{m['composite']:>9.3f}"
        )

    # 最优解详情
    best_w, best_m = results2[0]
    print("\n" + "=" * 90)
    print("  🥇 最优权重组合")
    print("=" * 90)
    for field in Weights.__dataclass_fields__:
        val = getattr(best_w, field)
        print(f"  {field:30s}: {val}")
    print("-" * 90)
    print(f"  intra_group_std:        {best_m['intra_group_std']:+.3f}")
    print(f"  bad_behavior_mean:      {best_m['bad_behavior_mean']:+.3f}")
    print(f"  length_correlation:     {best_m['length_correlation']:+.3f}")
    print(f"  short_success_mean:     {best_m['short_success_mean']:+.3f}")
    print(f"  success_mean:           {best_m['success_mean']:+.3f}")
    print(f"  failure_mean:           {best_m['failure_mean']:+.3f}")
    print(f"  composite:              {best_m['composite']:+.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
