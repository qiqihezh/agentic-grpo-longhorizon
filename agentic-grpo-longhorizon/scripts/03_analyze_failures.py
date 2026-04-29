"""
分析 baseline 的失败轨迹,为后续 reward design 提供 insight
用法:
    python scripts/03_analyze_failures.py \
        --report experiments/week1_baseline_airline/eval_report.json \
        --n 20
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def classify_failure(traj: dict) -> str:
    """启发式失败归因"""
    msgs = traj["raw_messages"]
    num_turns = traj["num_turns"]
    
    if traj.get("error"):
        return "EXCEPTION"
    if num_turns >= 29:
        return "TIMEOUT_OR_LOOP"
    
    # 看最后一条 assistant message
    last_assistant = None
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            last_assistant = m
            break
    
    if last_assistant is None:
        return "NO_ASSISTANT_RESPONSE"
    
    content = (last_assistant.get("content") or "").lower()
    if "sorry" in content or "cannot" in content or "unable" in content:
        return "MODEL_REFUSED_OR_GAVEUP"
    
    # 工具调用错误(通过 tool response 里有 error 关键字判断)
    for m in msgs:
        if m.get("role") == "tool":
            tc = (m.get("content") or "").lower()
            if "error" in tc or "invalid" in tc or "not found" in tc:
                return "TOOL_CALL_ERROR"
    
    # 多轮后仍没完成
    if num_turns > 15:
        return "LONG_HORIZON_DRIFT"
    
    return "UNKNOWN_FAILURE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--n", type=int, default=20, help="dump 前 N 条失败轨迹")
    args = parser.parse_args()
    
    with open(args.report) as f:
        report = json.load(f)
    
    # 收集所有失败轨迹
    failures = []
    for task in report["per_task_results"]:
        for traj in task["trajectories"]:
            if not traj["success"]:
                traj["_task_id"] = task["task_id"]
                failures.append(traj)
    
    print(f"\n总轨迹数: {sum(len(t['trajectories']) for t in report['per_task_results'])}")
    print(f"失败轨迹数: {len(failures)}")
    
    # 失败分类统计
    categories = Counter(classify_failure(t) for t in failures)
    print("\n=== 失败类型分布 ===")
    for cat, cnt in categories.most_common():
        pct = cnt / len(failures) * 100
        print(f"  {cat:30s} {cnt:4d} ({pct:5.1f}%)")
    
    # dump 前 N 条详细 case
    out_dir = Path(args.report).parent / "failure_cases"
    out_dir.mkdir(exist_ok=True)
    for i, t in enumerate(failures[:args.n]):
        cat = classify_failure(t)
        fname = f"case_{i:03d}_task{t['_task_id']}_{cat}.json"
        with open(out_dir / fname, "w") as f:
            json.dump(t, f, indent=2, ensure_ascii=False)
    
    print(f"\n前 {args.n} 条失败 case 已 dump 到 {out_dir}/")
    print("\n建议人工审读其中每种类型各 3-5 条,归纳 reward shaping 切入点")


if __name__ == "__main__":
    main()