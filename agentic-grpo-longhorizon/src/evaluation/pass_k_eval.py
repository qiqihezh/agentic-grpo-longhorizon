"""
pass^k 评测
对每个 task 独立采样 k 次,计算 pass^k (至少一次成功的比例)
同时统计 turn efficiency 和 tool call accuracy
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult


@dataclass
class EvalReport:
    env_name: str
    num_tasks: int
    num_samples_per_task: int
    pass_at_1: float           # 任意一次成功
    pass_hat_1: float          # pass^1: 平均成功率
    pass_hat_4: float          # pass^4: 连续 4 次都成功的比例(稳定性)
    pass_hat_8: float
    avg_turns: float
    avg_tool_calls: float
    error_rate: float          # trajectory 异常中止的比例
    per_task_results: list[dict]


def run_eval(
    wrapper: TauBenchWrapper,
    policy_factory,                    # callable -> policy instance (thread-safe)
    num_tasks: Optional[int] = None,
    num_samples_per_task: int = 4,
    max_turns: int = 30,
    num_workers: int = 4,
    output_dir: str = "experiments/baseline",
) -> EvalReport:
    """
    policy_factory: 每个 worker 线程自己 new 一个 policy,避免并发问题
    """
    if num_tasks is None:
        num_tasks = wrapper.get_num_tasks()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个 (task_idx, sample_idx) 是一个独立 job
    jobs = [(t, s) for t in range(num_tasks) for s in range(num_samples_per_task)]
    results: dict[int, list[TrajectoryResult]] = {t: [] for t in range(num_tasks)}
    
    def _run_one(task_idx: int, sample_idx: int) -> tuple[int, TrajectoryResult]:
        policy = policy_factory()
        # 给同一个 task 不同 sample 设不同 temperature seed
        # (vLLM server 端已经有采样随机性,这里主要是逻辑标记)
        traj = wrapper.run_single_task(task_idx, policy, max_turns=max_turns)
        return task_idx, traj
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_run_one, t, s) for t, s in jobs]
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Eval"):
            task_idx, traj = fut.result()
            results[task_idx].append(traj)
    
    # 计算 pass^k
    # pass^k 定义: 对同一个 task 采样 n 次,估计"连续 k 次都成功"的概率
    # 用 unbiased estimator (HumanEval 里的 pass@k 公式)
    import numpy as np
    
    def pass_at_k(n: int, c: int, k: int) -> float:
        """n: 总采样数, c: 成功数, k: pass^k 的 k"""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    per_task = []
    pass_1_list, pass_4_list, pass_8_list = [], [], []
    pass_at_1_list = []  # 任意一次成功
    all_turns, all_tool_calls, all_errors = [], [], []
    
    for t in range(num_tasks):
        trajs = results[t]
        n = len(trajs)
        c = sum(1 for tr in trajs if tr.success)
        
        p1 = pass_at_k(n, c, 1)
        p4 = pass_at_k(n, c, 4) if n >= 4 else None
        p8 = pass_at_k(n, c, 8) if n >= 8 else None
        
        pass_1_list.append(p1)
        pass_at_1_list.append(1.0 if c > 0 else 0.0)
        if p4 is not None: pass_4_list.append(p4)
        if p8 is not None: pass_8_list.append(p8)
        
        for tr in trajs:
            all_turns.append(tr.num_turns)
            all_tool_calls.append(tr.num_tool_calls)
            all_errors.append(1.0 if tr.error else 0.0)
        
        per_task.append({
            "task_id": t,
            "success_count": c,
            "total_samples": n,
            "pass^1": p1,
            "avg_turns": np.mean([tr.num_turns for tr in trajs]),
            "trajectories": [tr.to_dict() for tr in trajs],
        })
    
    report = EvalReport(
        env_name=wrapper.env_name,
        num_tasks=num_tasks,
        num_samples_per_task=num_samples_per_task,
        pass_at_1=float(np.mean(pass_at_1_list)),
        pass_hat_1=float(np.mean(pass_1_list)),
        pass_hat_4=float(np.mean(pass_4_list)) if pass_4_list else 0.0,
        pass_hat_8=float(np.mean(pass_8_list)) if pass_8_list else 0.0,
        avg_turns=float(np.mean(all_turns)),
        avg_tool_calls=float(np.mean(all_tool_calls)),
        error_rate=float(np.mean(all_errors)),
        per_task_results=per_task,
    )
    
    # 保存
    with open(output_dir / "eval_report.json", "w") as f:
        #json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
    
    # 打印摘要
    print(f"\n=== Eval Report: {wrapper.env_name} ===")
    print(f"Tasks: {num_tasks} × Samples: {num_samples_per_task}")
    print(f"pass@1 (any success): {report.pass_at_1:.3f}")
    print(f"pass^1 (avg success): {report.pass_hat_1:.3f}")
    print(f"pass^4 (stability):   {report.pass_hat_4:.3f}")
    print(f"pass^8 (stability):   {report.pass_hat_8:.3f}")
    print(f"Avg turns:       {report.avg_turns:.2f}")
    print(f"Avg tool calls:  {report.avg_tool_calls:.2f}")
    print(f"Error rate:      {report.error_rate:.3f}")
    
    return report