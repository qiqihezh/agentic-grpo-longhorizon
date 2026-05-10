#!/usr/bin/env python3
"""
PRM-Lite v3 离线诊断脚本

用途：不上线训练，验证 scoring 规则是否正确触发、signal 分布是否合理。

两层验证：
1. 单元测试（零 GPU，5 分钟）：手工构造典型 case，验证每条规则的触发逻辑
2. 小规模 Rollout（单卡，1-2 小时）：用实际 policy 跑 20 task × 2 samples，dump action_history

用法：
    # 仅跑单元测试
    python scripts/diagnose_prm_lite.py

    # 跑单元测试 + 小规模 rollout（需要提供 checkpoint 路径和启动 vLLM）
    python scripts/diagnose_prm_lite.py --model-path experiments/vanilla/hf_step_150 --num-tasks 20 --num-samples 2
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent
while not (_PROJECT_ROOT / "src").is_dir():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from dataclasses import dataclass

from src.envs.tau_bench_interaction import _compute_reasoning_quality_score


# ============================================================================
# 第一层：单元测试（零 GPU）
# ============================================================================

TEST_CASES = []


def register_test(name: str):
    def decorator(fn):
        TEST_CASES.append((name, fn))
        return fn
    return decorator


def _make_action(tool: str, params: dict, is_error: bool = False, entities: dict = None, content: str = "") -> dict:
    """构造标准化的 action_history 条目"""
    import json
    return {
        "tool": tool,
        "parameters": params,
        "param_str": json.dumps(params, sort_keys=True, ensure_ascii=False).lower(),
        "inc_reward": 0,
        "done": False,
        "is_error": is_error,
        "extracted_entities": entities or {},
        "content": content,
    }


@register_test("P1: Placeholder detection (read tool)")
def test_placeholder_read():
    """task 49 典型错误：get_reservation_details({"reservation_id": "previous_reservation"})"""
    hist = [_make_action("get_reservation_details", {"reservation_id": "previous_reservation"})]
    score = _compute_reasoning_quality_score(hist)
    assert score < -0.03, f"Expected negative for read placeholder, got {score:.3f}"
    return score


@register_test("P1: Placeholder detection (write tool, more severe)")
def test_placeholder_write():
    """write tool 占位符惩罚应更重"""
    hist = [_make_action("book_reservation", {"reservation_id": "previous_reservation"})]
    score = _compute_reasoning_quality_score(hist)
    assert score <= -0.07, f"Expected <= -0.07 for write placeholder, got {score:.3f}"
    return score


@register_test("P1: Schema-based validation (valid reservation_id)")
def test_placeholder_schema_valid():
    """合法的 reservation_id 不应被误判为占位符"""
    hist = [_make_action("get_reservation_details", {"reservation_id": "ZFA04Y"})]
    score = _compute_reasoning_quality_score(hist)
    assert score == 0.0, f"Expected 0 for valid ID, got {score:.3f}"
    return score


@register_test("P2: Redundancy detection (same tool+params within window=3)")
def test_redundancy():
    """连续两次 get_user_details(emma_kim_9957) → 冗余"""
    hist = [
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score < -0.03, f"Expected negative for redundancy, got {score:.3f}"
    return score


@register_test("P2: Non-redundancy (different params)")
def test_non_redundancy():
    """search_flight(EWR, DTW, 05-15) 和 search_flight(EWR, DTW, 05-16) 不是冗余"""
    hist = [
        _make_action("search_direct_flight", {"origin": "EWR", "destination": "DTW", "date": "2024-05-15"}),
        _make_action("search_direct_flight", {"origin": "EWR", "destination": "DTW", "date": "2024-05-16"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score == 0.0, f"Expected 0 for different params, got {score:.3f}"
    return score


@register_test("P3: Recovery bonus (different tool after error)")
def test_recovery():
    """上一步 error，本步换了 tool → +0.03"""
    hist = [
        _make_action("get_reservation_details", {"reservation_id": "BAD001"}, is_error=True),
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score > 0.01, f"Expected positive for recovery, got {score:.3f}"
    return score


@register_test("P3: Error repetition penalty (same tool+args after error)")
def test_error_repetition():
    """上一步 error，本步完全重复 → -0.06"""
    hist = [
        _make_action("get_reservation_details", {"reservation_id": "BAD001"}, is_error=True),
        _make_action("get_reservation_details", {"reservation_id": "BAD001"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score < -0.05, f"Expected <= -0.05 for error repetition, got {score:.3f}"
    return score


@register_test("P4: Premature surrender (no read before transfer)")
def test_premature_surrender():
    """第 1 步就 transfer → -0.15"""
    hist = [_make_action("transfer_to_human_agents", {"summary": "help"})]
    score = _compute_reasoning_quality_score(hist)
    assert score == -0.15, f"Expected -0.15 for premature surrender, got {score:.3f}"
    return score


@register_test("P4: Mild surrender (did read before transfer)")
def test_mild_surrender():
    """至少做过一次 read 再 transfer → -0.05"""
    hist = [
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
        _make_action("transfer_to_human_agents", {"summary": "help"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score == -0.05, f"Expected -0.05 for mild surrender, got {score:.3f}"
    return score


@register_test("B1: Data chain bonus (write tool uses historical data)")
def test_data_chain_write():
    """write tool 使用之前 observation 中提取的 reservation_id → +0.05"""
    hist = [
        _make_action("get_user_details", {"user_id": "emma_kim_9957"},
                     entities={"user_id": ["emma_kim_9957"], "reservation_id": ["ZFA04Y"]}),
        _make_action("cancel_reservation", {"reservation_id": "ZFA04Y"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score > 0.04, f"Expected > 0.04 for write data chain, got {score:.3f}"
    return score


@register_test("B1: Data chain bonus (read tool uses historical data)")
def test_data_chain_read():
    """read tool 使用之前数据 → +0.02"""
    hist = [
        _make_action("get_user_details", {"user_id": "emma_kim_9957"},
                     entities={"user_id": ["emma_kim_9957"]}),
        _make_action("get_reservation_details", {"reservation_id": "ZFA04Y"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    # ZFA04Y 不在 extracted_entities 中，所以不应触发 data chain
    assert score == 0.0, f"Expected 0 for no data chain, got {score:.3f}"
    return score


@register_test("B2/B3: Think bonus (single think)")
def test_think_bonus():
    """单次 think → +0.02"""
    hist = [
        _make_action("think", {"thought": "I need to find the user's reservations first."}),
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score > 0.01, f"Expected positive for think, got {score:.3f}"
    return score


@register_test("B2/B3: Consecutive think filtering")
def test_consecutive_think():
    """连续两次 think，第二次不应给分"""
    hist = [
        _make_action("implicit_think", {}, content="Let me analyze the situation..."),
        _make_action("implicit_think", {}, content="I should consider all options..."),
        _make_action("get_user_details", {"user_id": "emma_kim_9957"}),
    ]
    score = _compute_reasoning_quality_score(hist)
    # 第一次 +0.02，第二次 0，总计 +0.02
    assert 0.01 < score < 0.03, f"Expected ~0.02 for single think bonus, got {score:.3f}"
    return score


@register_test("Clamp range")
def test_clamp():
    """大量惩罚不应超出 clamp 下限"""
    hist = [
        _make_action("book_reservation", {"reservation_id": "previous_reservation"}),  # -0.08
        _make_action("book_reservation", {"reservation_id": "previous_reservation"}),  # -0.08 + redundancy -0.04
        _make_action("transfer_to_human_agents", {"summary": "help"}),  # -0.15
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score >= -0.5, f"Expected >= -0.5 (clamp), got {score:.3f}"
    return score


@register_test("Scale: process_score magnitude for bad trajectory")
def test_bad_trajectory_magnitude():
    """一条充满错误的 trajectory 应该拿到显著的负分"""
    hist = [
        _make_action("get_reservation_details", {"reservation_id": "previous_reservation"}, is_error=True),  # placeholder + error
        _make_action("get_reservation_details", {"reservation_id": "previous_reservation"}),  # placeholder + repetition
        _make_action("transfer_to_human_agents", {"summary": "help"}),  # premature surrender
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score <= -0.2, f"Expected strongly negative for bad traj, got {score:.3f}"
    return score


@register_test("Scale: process_score magnitude for good trajectory")
def test_good_trajectory_magnitude():
    """一条正确的 trajectory 应该拿到显著的正分"""
    hist = [
        _make_action("get_user_details", {"user_id": "emma_kim_9957"},
                     entities={"user_id": ["emma_kim_9957"], "reservation_id": ["ZFA04Y"]}),
        _make_action("get_reservation_details", {"reservation_id": "ZFA04Y"}),
        _make_action("update_reservation_flights", {"reservation_id": "ZFA04Y", "cabin": "economy", "flights": [{"flight_number": "HAT001", "date": "2024-05-15"}], "payment_id": "credit_card_123"},
                     entities={"reservation_id": ["ZFA04Y"], "flight_number": ["HAT001"]}),
    ]
    score = _compute_reasoning_quality_score(hist)
    assert score >= 0.05, f"Expected positive for good traj, got {score:.3f}"
    return score


def run_unit_tests():
    print("=" * 60)
    print("  PRM-Lite v3 单元测试")
    print("=" * 60)
    passed = 0
    failed = 0
    scores = []
    for name, fn in TEST_CASES:
        try:
            score = fn()
            scores.append((name, score))
            print(f"  ✅ {name}: {score:+.3f}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    print("=" * 60)
    print(f"  结果: {passed}/{passed+failed} 通过")
    print("=" * 60)

    if scores:
        min_score = min(s for _, s in scores)
        max_score = max(s for _, s in scores)
        print(f"\n  process_score 范围: [{min_score:+.3f}, {max_score:+.3f}]")
        print(f"  跨度: {max_score - min_score:.3f}")
    return failed == 0


# ============================================================================
# 第二层：小规模 Rollout 诊断（需要 GPU + vLLM）
# ============================================================================

def run_rollout_diagnosis(model_path: str, num_tasks: int, num_samples: int):
    """
    用实际 policy 跑少量 rollout，收集 action_history 和 process_score 分布。
    前置条件：vLLM server 已在 port 8000 运行，72B user sim 已在 port 8001 运行。
    """
    import numpy as np
    from src.envs.tau_bench_wrapper import TauBenchWrapper
    from src.models.vllm_policy import VLLMPolicy
    from src.evaluation.pass_k_eval import run_eval

    print("\n" + "=" * 60)
    print("  PRM-Lite v3 Rollout 诊断")
    print(f"  Model: {model_path}")
    print(f"  Tasks: {num_tasks} × Samples: {num_samples}")
    print("=" * 60)

    wrapper = TauBenchWrapper(
        env_name="airline",
        user_strategy="llm",
        user_model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        user_provider="openai",
        user_base_url="http://localhost:8001/v1",
        task_split="test",
    )

    policy = VLLMPolicy(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    report = run_eval(
        wrapper=wrapper,
        policy_factory=lambda: policy,
        num_tasks=num_tasks,
        num_samples_per_task=num_samples,
        max_turns=30,
        num_workers=2,
        output_dir=f"experiments/prm_lite_diagnose_{Path(model_path).name}",
    )

    # 从 per_task_results 中提取 action_history 并重新计算 process_score
    all_process_scores = []
    success_scores = []
    failure_scores = []
    group_stds = []

    for task_result in report.per_task_results:
        task_process_scores = []
        for traj in task_result["trajectories"]:
            # raw_messages 中没有 action_history，但 eval_report 的 trajectories 不包含 action_history
            # 需要重新跑 wrapper.run_single_task 来收集 action_history
            pass

    print("\n[注意] Rollout 诊断需要修改 wrapper 以导出 action_history。")
    print("建议：先跑单元测试确认规则逻辑，再上线观察训练曲线。")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRM-Lite v3 离线诊断")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Policy checkpoint 路径（如需 rollout 诊断）")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()

    # 第一层：单元测试（必跑）
    ok = run_unit_tests()
    if not ok:
        print("\n⚠️  单元测试有失败，请先修复 scoring 逻辑再上线。")
        return

    # 第二层：Rollout 诊断（可选）
    if args.model_path:
        run_rollout_diagnosis(args.model_path, args.num_tasks, args.num_samples)
    else:
        print("\n💡 提示：如需 rollout 诊断，加 --model-path experiments/xxx/hf_step_N")


if __name__ == "__main__":
    main()
