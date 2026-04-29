"""
Week 1 主脚本: 评测 vanilla Qwen2.5-7B-Instruct 在 τ-bench airline 上的 baseline
用法:
    # Step 1: 先启动 vLLM server(另一个 tmux session)
    bash scripts/01_start_vllm_server.sh
    
    # Step 2: 跑评测
    python scripts/run_baseline_eval.py --config configs/baseline_airline.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
from pathlib import Path
import os

from src.envs.tau_bench_wrapper import TauBenchWrapper
from src.models.vllm_policy import VLLMPolicy
from src.evaluation.pass_k_eval import run_eval
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tiny", action="store_true", 
                        help="快速 smoke test: 只跑 2 个任务,每个采 2 次")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # tiny 模式: 在 4090 上快速验证代码通不通
    if args.tiny:
        cfg["eval"]["num_tasks"] = 2
        cfg["eval"]["num_samples_per_task"] = 2
        cfg["eval"]["num_workers"] = 2
        cfg["output"]["dir"] += "_tiny"
    
    wrapper = TauBenchWrapper(
        env_name=cfg["env"]["name"],
        user_strategy=cfg["env"]["user_strategy"],
        user_model=cfg["env"]["user_model"],
        user_provider=cfg["env"]["user_provider"],
        user_base_url=cfg["env"].get("user_base_url"),
        task_split=cfg["env"]["task_split"],
    )
    
    # policy_factory: 每次调用新建一个 policy
    # VLLMPolicy 本身是 thread-safe 的(OpenAI client 内部处理),这里简单复用
    shared_policy = VLLMPolicy(**cfg["policy"])
    
    def policy_factory():
        return shared_policy
    
    report = run_eval(
        wrapper=wrapper,
        policy_factory=policy_factory,
        num_tasks=cfg["eval"]["num_tasks"],
        num_samples_per_task=cfg["eval"]["num_samples_per_task"],
        max_turns=cfg["eval"]["max_turns"],
        num_workers=cfg["eval"]["num_workers"],
        output_dir=cfg["output"]["dir"],
    )
    
    print(f"\n报告已保存到: {cfg['output']['dir']}/eval_report.json")


if __name__ == "__main__":
    main()