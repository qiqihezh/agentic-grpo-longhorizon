"""
Unit tests for PRM-Lite v4-optimal (_compute_reasoning_quality_score).
Run: python src/envs/tests/test_prm_lite_v4.py

v4-optimal changes from v4-fix (tuned via offline grid search):
- think_bonus: 0.05 -> 0.01
- implicit_think reward: REMOVED (was +0.03/+0.01)
- no_reasoning_penalty: -0.03 -> -0.05
- diversity_bonus: 0.03 -> 0.01
- NEW: length_penalty (ntools > 8, -0.01 per extra step)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.envs.tau_bench_interaction import (
    _compute_reasoning_quality_score,
    _compute_prm_lite_reward,
)


def _make_action(tool, params=None, param_str=None, is_error=False, extracted_entities=None, content=""):
    return {
        "tool": tool,
        "parameters": params or {},
        "param_str": param_str or ("{}" if not params else str(sorted(params.items()))),
        "is_error": is_error,
        "extracted_entities": extracted_entities or {},
        "content": content,
    }


def test_empty_history():
    assert _compute_reasoning_quality_score([]) == 0.0


def test_placeholder_write():
    # placeholder write(-0.05)
    history = [_make_action("book_reservation", params={"reservation_id": "my_trip"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.05)) < 1e-6


def test_placeholder_read():
    # placeholder read(-0.03) + first-read(+0.01) = -0.02
    history = [_make_action("get_reservation_details", params={"reservation_id": "previous_reservation"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.02)) < 1e-6


def test_no_placeholder_valid_id():
    # first-read(+0.01)
    history = [_make_action("get_reservation_details", params={"reservation_id": "ABC123"})]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_redundancy_same_tool_params():
    # step0: first-read(+0.01) = +0.01
    # step1: redundancy(-0.03) = -0.03
    # mean = -0.01
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}, param_str='{"user_id": "john_doe_123"}'),
        _make_action("get_user_details", params={"user_id": "john_doe_123"}, param_str='{"user_id": "john_doe_123"}'),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.01)) < 1e-6


def test_error_repetition():
    # step0: placeholder read(-0.03) + first-read(+0.01) = -0.02 (error)
    # step1: placeholder read(-0.03) + error-repetition(-0.04) = -0.07
    # mean = -0.045
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}, is_error=True),
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.045)) < 1e-6


def test_recovery_different_tool():
    # step0: placeholder read(-0.03) + first-read(+0.01) = -0.02 (error)
    # step1: recovery(+0.05) + first-read(+0.01) = +0.06
    # mean = 0.02
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "BAD"}, is_error=True),
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.02) < 1e-6


def test_escalation_premature_no_read():
    # escalation(-0.10)
    history = [_make_action("transfer_to_human_agents")]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.10)) < 1e-6


def test_escalation_late_with_read():
    # step0: first-read(+0.01) = +0.01
    # step1: escalation with read(-0.05) = -0.05
    # mean = -0.02
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("transfer_to_human_agents"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.02)) < 1e-6


def test_data_chain_write():
    # step0: first-read(+0.01) = +0.01
    # step1: data-chain-write(+0.08) = +0.08
    # mean = 0.045
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"},
                     extracted_entities={"reservation_id": ["ABC123"]}),
        _make_action("cancel_reservation", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.045) < 1e-6


def test_first_read_exploration():
    # step0: first-read(+0.01) = +0.01
    # step1: first-read(+0.01) = +0.01
    # mean = 0.01
    history = [
        _make_action("get_user_details", params={"user_id": "john_doe_123"}),
        _make_action("get_reservation_details", params={"reservation_id": "ABC123"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_read_diversity_bonus():
    # 3 different read tools
    # per-step: each gets first-read +0.01 = +0.01
    # mean before adjustments = 0.01
    # no-reasoning penalty = -0.05, diversity bonus = +0.01
    # final = -0.03
    history = [
        _make_action("get_user_details"),
        _make_action("get_reservation_details"),
        _make_action("search_direct_flight"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.03)) < 1e-6


def test_successful_tool_call():
    # B3 removed — no success_bonus. Only first-read(+0.01) applies.
    history = [_make_action("get_user_details", is_error=False)]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_think_valid():
    # step0: think(+0.01)
    # step1: first-read(+0.01) = +0.01
    # mean = 0.01
    history = [
        _make_action("think"),
        _make_action("get_user_details"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.01) < 1e-6


def test_think_consecutive():
    # step0: think(+0.01)
    # step1: consecutive think -> 0
    # mean = 0.005
    history = [
        _make_action("think"),
        _make_action("think"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.005) < 1e-6


def test_think_last_step():
    # step0: first-read(+0.01) = +0.01
    # step1: last-step think -> 0
    # mean = 0.005
    history = [
        _make_action("get_user_details"),
        _make_action("think"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - 0.005) < 1e-6


def test_think_followed_by_placeholder():
    # step0: think followed by placeholder -> 0
    # step1: placeholder write(-0.05)
    # mean = -0.025
    history = [
        _make_action("think"),
        _make_action("book_reservation", params={"reservation_id": "my_trip"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.025)) < 1e-6


def test_implicit_think_no_reward():
    # implicit_think reward REMOVED in v4-optimal
    history = [_make_action("implicit_think", content="x" * 150)]
    score = _compute_reasoning_quality_score(history)
    assert score == 0.0


def test_implicit_think_long_no_reward():
    # implicit_think reward REMOVED in v4-optimal
    history = [_make_action("implicit_think", content="x" * 400)]
    score = _compute_reasoning_quality_score(history)
    assert score == 0.0


def test_no_reasoning_penalty():
    # step0: first-read(+0.01) = +0.01
    # step1: first-read(+0.01) = +0.01
    # step2: (write, no first-read) = 0.0
    # mean before penalty = 0.0067
    # no-reasoning penalty = -0.05
    # final = -0.0433
    history = [
        _make_action("get_user_details"),
        _make_action("get_reservation_details"),
        _make_action("book_reservation"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.043333333333333335)) < 1e-6


def test_clamp_bounds():
    history = [
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
        _make_action("transfer_to_human_agents"),
    ]
    score = _compute_reasoning_quality_score(history)
    assert score >= -0.5


def test_length_penalty():
    # 10 identical transfer_to_human_agents (ntools=10 > 8)
    # step0: escalation(-0.10)
    # step1-9: escalation(-0.10) + redundancy(-0.03) = -0.13 each
    # mean = (-0.10 + 9*-0.13) / 10 = -0.127
    # no-reasoning penalty = -0.05
    # length penalty = -0.01 * (10-8) = -0.02
    # final = -0.197
    history = [_make_action("transfer_to_human_agents") for _ in range(10)]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.197)) < 1e-6


def test_prm_lite_reward_v4_formula():
    state = {
        "total_reward": 1.0,
        "action_history": [
            _make_action("think"),
            _make_action("get_user_details"),
        ]
    }
    reward = _compute_prm_lite_reward(state)
    assert reward > 1.0 and reward < 1.1


def test_task49_like_trajectory():
    history = [
        _make_action("get_reservation_details", params={"reservation_id": "MDCLVA"},
                     extracted_entities={"reservation_id": ["MDCLVA"]}),
        _make_action("cancel_reservation", params={"reservation_id": "MDCLVA"}),
        _make_action("get_reservation_details", params={"reservation_id": "previous_reservation"}),
        _make_action("get_user_details", params={"user_id": "emma_kim_9957"}),
        _make_action("get_user_details", params={"user_id": "emma_kim_9957"}),
        _make_action("get_reservation_details", params={"reservation_id": "previous_reservation"}),
    ]
    score = _compute_reasoning_quality_score(history)
    assert abs(score - (-0.043333333333333335)) < 1e-6


if __name__ == "__main__":
    import traceback
    tests = [
        test_empty_history,
        test_placeholder_write,
        test_placeholder_read,
        test_no_placeholder_valid_id,
        test_redundancy_same_tool_params,
        test_error_repetition,
        test_recovery_different_tool,
        test_escalation_premature_no_read,
        test_escalation_late_with_read,
        test_data_chain_write,
        test_first_read_exploration,
        test_read_diversity_bonus,
        test_successful_tool_call,
        test_think_valid,
        test_think_consecutive,
        test_think_last_step,
        test_think_followed_by_placeholder,
        test_implicit_think_no_reward,
        test_implicit_think_long_no_reward,
        test_no_reasoning_penalty,
        test_clamp_bounds,
        test_length_penalty,
        test_prm_lite_reward_v4_formula,
        test_task49_like_trajectory,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
