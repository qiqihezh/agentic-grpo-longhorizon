"""
Generate veRL tool_config YAML from τ-bench env.tools_info.

Usage:
    python scripts/10_gen_tool_config.py \
        --env airline \
        --output configs/tool_config/tau_bench_airline_tools.yaml

Design: §4.1 — dynamic generation, not hand-written.
Produces one entry per tool with class_name pointing to the dynamically
generated TauBench_<name>_Tool class in srcs.evns.tau_bench_tools.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_tool_schemas(env_name: str = "airline", task_index: int = 0) -> list[dict]:
    from tau_bench.envs import get_env

    env = get_env(
        env_name=env_name,
        user_strategy="human",  # 不需要 LLM，只要 tools_info
        user_model="dummy",
        user_provider="openai",
        task_split="test",
        task_index=task_index,
    )
    return env.tools_info


def build_tool_config(env_name: str, schemas: list[dict]) -> dict:
    tools = []
    for schema in schemas:
        func = schema.get("function", schema)
        name = func["name"]
        cls_name = f"src.envs.tau_bench_tools.TauBench_{name}_Tool"
        tools.append({
            "class_name": cls_name,
            "config": {"type": "native"},
            "tool_schema": schema,
        })
    return {"tools": tools}


def main():
    parser = argparse.ArgumentParser(description="Generate veRL tool config from τ-bench")
    parser.add_argument("--env", default="airline", help="τ-bench env name")
    parser.add_argument("--task-id", type=int, default=0, help="task index for schema extraction")
    parser.add_argument("--output", default="configs/tool_config/tau_bench_airline_tools.yaml")
    args = parser.parse_args()

    schemas = get_tool_schemas(args.env, args.task_id)
    print(f"Extracted {len(schemas)} tool schemas from {args.env} env")

    config = build_tool_config(args.env, schemas)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Written to {output_path}")
    for t in config["tools"]:
        print(f"  - {t['class_name']}")


if __name__ == "__main__":
    main()
