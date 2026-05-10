#!/usr/bin/env python3
"""
PRM-Lite v4-revised 快速离线验证
支持传入任意 eval json，对比 SFT / vanilla / 其他 checkpoint
新增 cheap reasoning 分析（从 raw_messages 直接计算 assistant content 长度）
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent
while not (_PROJECT_ROOT / "src").is_dir():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from src.envs.tau_bench_interaction import _compute_reasoning_quality_score


def trajectory_to_action_history(raw_messages):
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
                    if not isinstance(args, dict):
                        args = {}
                except:
                    args = {}
                tc_id = tc.get("id", "")
                result_msg = tool_results.get(tc_id, {})
                obs = result_msg.get("content", "")
                history.append({
                    "tool": func.get("name", ""),
                    "parameters": args,
                    "param_str": json.dumps(args, sort_keys=True, ensure_ascii=False).lower(),
                    "is_error": str(obs).startswith("Error:"),
                    "extracted_entities": {},
                    "content": content,  # 记录 assistant content 用于 cheap reasoning 分析
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


def analyze_cheap_reasoning(raw_messages):
    """分析每个 assistant turn 的 reasoning 充分性。
    返回 (num_short_reasoning, num_total_reasoning_turns)
    short = content < 30 chars 且包含 tool_calls（即敷衍一下就调 tool）
    """
    short_count = 0
    total_reasoning = 0
    for msg in raw_messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            total_reasoning += 1
            if len(content) < 30:
                short_count += 1
    return short_count, total_reasoning


def validate(path: Path, label: str):
    if not path.exists():
        print(f"⚠️  {label}: {path} 不存在")
        return

    with open(path) as f:
        data = json.load(f)

    success_scores, failure_scores = [], []
    ntools_dist = defaultdict(lambda: {"success": [], "failure": []})
    cheap_stats = {"success": [], "failure": []}

    for task in data.get("per_task_results", []):
        for traj in task.get("trajectories", []):
            raw = traj.get("raw_messages", [])
            history = trajectory_to_action_history(raw)
            if not history:
                continue

            score = _compute_reasoning_quality_score(history)
            ntools = len([a for a in history if a["tool"] not in {"think", "implicit_think"}])
            succ = traj.get("success", False)

            if succ:
                success_scores.append(score)
                ntools_dist[ntools]["success"].append(score)
            else:
                failure_scores.append(score)
                ntools_dist[ntools]["failure"].append(score)

            # cheap reasoning
            short, total = analyze_cheap_reasoning(raw)
            if total > 0:
                ratio = short / total
                if succ:
                    cheap_stats["success"].append(ratio)
                else:
                    cheap_stats["failure"].append(ratio)

    print(f"\n{'='*80}")
    print(f"  {label}: {path.name}")
    print(f"{'='*80}")

    if success_scores:
        arr = np.array(success_scores)
        print(f"  success : n={len(arr):3d} mean={arr.mean():+.3f} std={arr.std():.3f} min={arr.min():+.3f} max={arr.max():+.3f}")
    if failure_scores:
        arr = np.array(failure_scores)
        print(f"  failure : n={len(arr):3d} mean={arr.mean():+.3f} std={arr.std():.3f} min={arr.min():+.3f} max={arr.max():+.3f}")
    if success_scores and failure_scores:
        sep = np.mean(success_scores) - np.mean(failure_scores)
        print(f"  SEP     = {sep:+.3f}")

    # Length correlation
    print(f"\n  Length correlation:")
    for n in sorted(ntools_dist.keys()):
        s_scores = ntools_dist[n]["success"]
        f_scores = ntools_dist[n]["failure"]
        all_scores = s_scores + f_scores
        if all_scores:
            print(f"    ntools={n:2d}: n={len(all_scores):3d} mean={np.mean(all_scores):+.3f} (succ={len(s_scores)} fail={len(f_scores)})")

    # Cheap reasoning
    print(f"\n  Cheap reasoning (content<30 chars before tool call):")
    for k in ["success", "failure"]:
        vals = cheap_stats[k]
        if vals:
            print(f"    {k:8s}: mean_short_ratio={np.mean(vals):.3f} (n={len(vals)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()
    validate(args.eval_json, args.label or args.eval_json.name)


if __name__ == "__main__":
    main()
