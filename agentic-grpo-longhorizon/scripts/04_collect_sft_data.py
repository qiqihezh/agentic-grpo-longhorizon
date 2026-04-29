"""
Week 2 Day 1-2: 用 Qwen2.5-72B-Instruct-AWQ 在 τ-bench airline 上 best-of-N 采集 SFT 数据

设计思路（对应 PROJECT.md §3 Week 2 + §6.4 复用约定）:
1. 72B-AWQ 此时不再当 user simulator，而是当 policy 跑 trajectory 采集
   user simulator 仍然用 72B-AWQ（同一个 server，复用 8001 端口）
2. 每个 task 跑 best_of=16 次:
   - 第 1 次 temp=0.0 greedy（保证至少有一条最稳定的 trajectory）
   - 第 2-16 次 temp=0.8 多样化采样
3. 过滤策略 (a)：只保留 success=True 的 trajectory
4. 过滤策略 (b)：被截断污染的 trajectory 进 contaminated 桶，永不进 train.jsonl
5. 断点续跑：每个 task 一个 jsonl，文件存在则跳过

用法:
    bash scripts/01_start_vllm_server_72b.sh   # GPU0 跑 72B-AWQ
    python scripts/04_collect_sft_data.py --config configs/sft_collect_airline.yaml
    # tiny 模式
    python scripts/04_collect_sft_data.py --config configs/sft_collect_airline.yaml --tiny
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import yaml
from tqdm import tqdm

from src.envs.tau_bench_wrapper import TauBenchWrapper, TrajectoryResult
from src.models.vllm_policy import VLLMPolicy

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def make_policy(cfg: dict, temperature: float) -> VLLMPolicy:
    """每次创建一个 policy 实例。VLLMPolicy 持有 OpenAI client，复用是 thread-safe 的，
    但每个采样的 temperature 不同，所以这里按 temperature 构造一个新的 wrapper。"""
    return VLLMPolicy(
        model_name=cfg["policy"]["model_name"],
        base_url=cfg["policy"]["base_url"],
        api_key="EMPTY",
        temperature=temperature,
        top_p=cfg["policy"]["top_p"],
        max_tokens=cfg["policy"]["max_tokens"],
    )


def collect_one_task(
    task_idx: int,
    wrapper: TauBenchWrapper,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """对一个 task 跑 best_of_n 次，返回该 task 的统计信息"""
    out_file = output_dir / f"task_{task_idx:04d}.jsonl"
    meta_file = output_dir / f"task_{task_idx:04d}.meta.json"
    contaminated_file = output_dir / f"task_{task_idx:04d}_contaminated.jsonl"

    # 断点续跑
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    best_of_n = cfg["collect"]["best_of_n"]
    max_turns = cfg["collect"]["max_turns"]
    temps = cfg["collect"]["temperatures"]
    assert len(temps) == best_of_n, f"temperatures 长度 {len(temps)} 必须等于 best_of_n {best_of_n}"

    successes: list[tuple[TrajectoryResult, int, float]] = []
    contaminated: list[tuple[TrajectoryResult, int, float]] = []
    all_attempts_summary = []

    for sample_idx in range(best_of_n):
        temp = temps[sample_idx]
        policy = make_policy(cfg, temperature=temp)
        try:
            traj = wrapper.run_single_task(task_idx, policy, max_turns=max_turns)
        except Exception as e:
            all_attempts_summary.append({
                "sample_idx": sample_idx, "temperature": temp,
                "success": False, "reward": 0.0,
                "num_turns": 0, "num_tool_calls": 0,
                "error": f"OUTER_EXCEPTION: {type(e).__name__}: {e}",
            })
            continue

        # [污染标记] 一旦截断，无论 success 与否都进 contaminated 桶
        is_contaminated = traj.was_contaminated_from_turn is not None

        all_attempts_summary.append({
            "sample_idx": sample_idx, "temperature": temp,
            "success": traj.success, "reward": traj.reward,
            "num_turns": traj.num_turns, "num_tool_calls": traj.num_tool_calls,
            "error": traj.error,
            "was_contaminated": is_contaminated,
            "contaminated_from_turn": traj.was_contaminated_from_turn,
        })

        if is_contaminated:
            contaminated.append((traj, sample_idx, temp))
        elif traj.success:
            successes.append((traj, sample_idx, temp))

    # 写出成功的 trajectory（干净数据 → train.jsonl）
    with open(out_file, "w") as f:
        for traj, sample_idx, temp in successes:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "sample_idx": sample_idx,
                "temperature": temp,
                "success": traj.success,
                "reward": traj.reward,
                "num_turns": traj.num_turns,
                "num_tool_calls": traj.num_tool_calls,
                "messages": traj.raw_messages,
            }, ensure_ascii=False) + "\n")

    # 写出被污染的 trajectory（诊断用，永不进 train.jsonl）
    with open(contaminated_file, "w") as f:
        for traj, sample_idx, temp in contaminated:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "sample_idx": sample_idx,
                "temperature": temp,
                "success": traj.success,
                "reward": traj.reward,
                "num_turns": traj.num_turns,
                "num_tool_calls": traj.num_tool_calls,
                "was_contaminated_from_turn": traj.was_contaminated_from_turn,
                "messages": traj.raw_messages,
            }, ensure_ascii=False) + "\n")

    meta = {
        "task_id": task_idx,
        "best_of_n": best_of_n,
        "num_successes": len(successes),
        "num_contaminated": len(contaminated),
        "any_success": len(successes) > 0,
        "attempts": all_attempts_summary,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tiny", action="store_true",
                        help="smoke test: 只跑 2 个 task，best_of=2")
    parser.add_argument("--task-range", type=str, default=None,
                        help="只跑指定 task 范围，如 '0:10' 或 '5,8,11'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.tiny:
        cfg["collect"]["best_of_n"] = 2
        cfg["collect"]["temperatures"] = [0.0, 0.8]
        cfg["output"]["dir"] = cfg["output"]["dir"] + "_tiny"

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 config 快照（方便事后追溯用了什么参数采集的）
    with open(output_dir / "collect_config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    wrapper = TauBenchWrapper(
        env_name=cfg["env"]["name"],
        user_strategy=cfg["env"]["user_strategy"],
        user_model=cfg["env"]["user_model"],
        user_provider=cfg["env"]["user_provider"],
        user_base_url=cfg["env"].get("user_base_url"),
        task_split=cfg["env"]["task_split"],
    )

    # 决定要跑哪些 task
    if args.task_range:
        if ":" in args.task_range:
            lo, hi = args.task_range.split(":")
            task_ids = list(range(int(lo), int(hi)))
        else:
            task_ids = [int(x) for x in args.task_range.split(",")]
    else:
        total = wrapper.get_num_tasks()
        if args.tiny:
            task_ids = list(range(2))
        else:
            task_ids = list(range(total))

    print(f"=== SFT 数据采集 ===")
    print(f"env:        {cfg['env']['name']}")
    print(f"policy:     {cfg['policy']['model_name']} @ {cfg['policy']['base_url']}")
    print(f"user sim:   {cfg['env']['user_model']} @ {cfg['env'].get('user_base_url')}")
    print(f"tasks:      {len(task_ids)} (total available: {wrapper.get_num_tasks()})")
    print(f"best_of_n:  {cfg['collect']['best_of_n']}")
    print(f"output:     {output_dir}")

    t0 = time.time()
    num_workers = cfg["collect"]["num_workers"]
    all_meta: list[dict] = []

    # ThreadPoolExecutor: 每个 task 一个 worker
    # 注意：同一个 task 内的 best_of_n 是串行的（共享 task state），不能并行
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(collect_one_task, t, wrapper, cfg, output_dir): t
            for t in task_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Collecting"):
            t = futures[fut]
            try:
                meta = fut.result()
                all_meta.append(meta)
            except Exception as e:
                print(f"[ERROR] task {t} 整体失败: {e}")

    # 汇总报告
    all_meta.sort(key=lambda x: x["task_id"])
    num_tasks_with_success = sum(1 for m in all_meta if m["any_success"])
    total_successes = sum(m["num_successes"] for m in all_meta)
    total_contaminated = sum(m["num_contaminated"] for m in all_meta)
    dropped_tasks = [m["task_id"] for m in all_meta if not m["any_success"]]

    summary = {
        "env": cfg["env"]["name"],
        "policy_model": cfg["policy"]["model_name"],
        "user_sim_model": cfg["env"]["user_model"],
        "best_of_n": cfg["collect"]["best_of_n"],
        "num_tasks_attempted": len(all_meta),
        "num_tasks_with_success": num_tasks_with_success,
        "task_coverage_rate": num_tasks_with_success / max(len(all_meta), 1),
        "total_success_trajectories": total_successes,
        "total_contaminated_trajectories": total_contaminated,
        "avg_successes_per_task": total_successes / max(len(all_meta), 1),
        "avg_contaminated_per_task": total_contaminated / max(len(all_meta), 1),
        "dropped_tasks": dropped_tasks,
        "elapsed_seconds": time.time() - t0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------- seen / unseen 切分 -------
    holdout_size = cfg["collect"].get("holdout_size", 0)
    if holdout_size > 0 and len(task_ids) > holdout_size:
        # 均匀分层：每 stride 取一个进 holdout
        stride = len(task_ids) / holdout_size
        holdout_ids = sorted({task_ids[int(i * stride)] for i in range(holdout_size)})
        # 如果因为 round 撞车不够 holdout_size，从尾部补
        i = -1
        while len(holdout_ids) < holdout_size:
            cand = task_ids[i]
            if cand not in holdout_ids:
                holdout_ids.append(cand)
            i -= 1
        holdout_ids = sorted(set(holdout_ids))[:holdout_size]
        seen_ids = [t for t in task_ids if t not in holdout_ids]
    else:
        holdout_ids = []
        seen_ids = list(task_ids)

    # 把切分写到 split.json，评测脚本读这个
    split_file = output_dir / "split.json"
    with open(split_file, "w") as f:
        json.dump({
            "seen_task_ids": seen_ids,
            "unseen_task_ids": holdout_ids,
            "split_strategy": "stratified_by_stride",
            "total_tasks": len(task_ids),
        }, f, indent=2)

    # 合并 trajectory：seen 进 train.jsonl，unseen 进 holdout_train.jsonl（备用）
    # [关键] 只合并 task_XXXX.jsonl（干净的），不合并 *_contaminated.jsonl
    train_file = output_dir / "train.jsonl"
    holdout_file = output_dir / "holdout_train.jsonl"
    n_train_traj = 0
    n_holdout_traj = 0
    with open(train_file, "w") as f_train, open(holdout_file, "w") as f_hold:
        for t in task_ids:
            tf = output_dir / f"task_{t:04d}.jsonl"
            if not tf.exists():
                continue
            target_f = f_train if t in seen_ids else f_hold
            with open(tf) as fin:
                for line in fin:
                    target_f.write(line)
                    if t in seen_ids:
                        n_train_traj += 1
                    else:
                        n_holdout_traj += 1

    summary["seen_task_ids"] = seen_ids
    summary["unseen_task_ids"] = holdout_ids
    summary["n_train_trajectories"] = n_train_traj
    summary["n_holdout_trajectories"] = n_holdout_traj
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=== 采集完成 ===")
    print(f"task 覆盖率:        {summary['task_coverage_rate']:.2%} "
          f"({num_tasks_with_success}/{len(all_meta)})")
    print(f"成功 trajectory 总数: {total_successes}")
    print(f"污染 trajectory 总数: {total_contaminated}")
    print(f"平均每 task 成功条数: {summary['avg_successes_per_task']:.2f}")
    print(f"平均每 task 污染条数: {summary['avg_contaminated_per_task']:.2f}")
    print(f"被 drop 的 task:     {dropped_tasks if len(dropped_tasks) < 20 else f'{len(dropped_tasks)} tasks'}")
    print(f"耗时:                {summary['elapsed_seconds']:.1f}s "
          f"({summary['elapsed_seconds']/60:.1f}min)")
    print()
    print(f"--- seen/unseen 切分 ---")
    print(f"seen task ({len(seen_ids)}):    {seen_ids[:10]}{'...' if len(seen_ids) > 10 else ''}")
    print(f"unseen task ({len(holdout_ids)}): {holdout_ids}")
    print(f"SFT 训练数据 → {train_file}  ({n_train_traj} 条，已过滤污染)")
    print(f"holdout 备份 → {holdout_file}  ({n_holdout_traj} 条，不进 SFT)")
    print(f"切分配置 → {split_file}  (评测脚本会读这个)")


if __name__ == "__main__":
    main()
