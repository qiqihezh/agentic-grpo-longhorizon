#!/usr/bin/env python3
"""
最小化测试：验证 Qwen2.5 是否支持 use_fused_kernels
不启动完整训练，只加载模型做一次 forward
"""
import torch
from transformers import AutoModelForCausalLM
from verl.models.transformers.monkey_patch import apply_monkey_patch

MODEL_PATH = "/workspace/models/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"

def test_fused_kernels(backend: str):
    print(f"\n{'='*60}")
    print(f"Testing use_fused_kernels=True, backend='{backend}'")
    print(f"{'='*60}")

    # 1. 加载模型
    print(f"[1/4] Loading model from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=DEVICE,
    )
    model.eval()
    print(f"      model_type: {model.config.model_type}")

    # 2. 应用 monkey patch（关键步骤）
    print(f"[2/4] Applying monkey patch with use_fused_kernels=True ...")
    apply_monkey_patch(
        model=model,
        use_remove_padding=True,
        ulysses_sp_size=1,
        use_fused_kernels=True,
        fused_kernels_backend=backend,
    )

    # 3. 构造 dummy 输入
    print(f"[3/4] Preparing dummy input ...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)

    # 4. Forward
    print(f"[4/4] Running forward ...")
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                temperature=1.0,
                use_cache=False,
            )

        # 检查输出
        has_log_probs = hasattr(output, "log_probs")
        has_entropy = hasattr(output, "entropy")
        has_logits = hasattr(output, "logits")

        print(f"      ✓ forward succeeded")
        print(f"      output has log_probs: {has_log_probs}")
        print(f"      output has entropy:   {has_entropy}")
        print(f"      output has logits:    {has_logits}")

        if has_log_probs and output.log_probs is not None:
            print(f"      log_probs shape:      {output.log_probs.shape}")
        if has_entropy and output.entropy is not None:
            print(f"      entropy shape:        {output.entropy.shape}")

        if has_log_probs and has_entropy:
            print(f"\n{'='*60}")
            print(f"  RESULT: ✓ Qwen2.5 SUPPORTS use_fused_kernels with '{backend}' backend")
            print(f"{'='*60}")
            return True
        else:
            print(f"\n{'='*60}")
            print(f"  RESULT: ✗ Qwen2.5 does NOT return log_probs/entropy")
            print(f"          use_fused_kernels will FAIL")
            print(f"{'='*60}")
            return False

    except Exception as e:
        print(f"      ✗ forward FAILED: {type(e).__name__}: {e}")
        print(f"\n{'='*60}")
        print(f"  RESULT: ✗ use_fused_kernels with '{backend}' backend is BROKEN")
        print(f"{'='*60}")
        return False

    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 测试 torch backend（推荐，依赖 flash-attn）
    result_torch = test_fused_kernels("torch")

    # 测试 triton backend
    result_triton = test_fused_kernels("triton")

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"  torch backend:   {'✓ SUPPORTED' if result_torch else '✗ NOT SUPPORTED'}")
    print(f"  triton backend:  {'✓ SUPPORTED' if result_triton else '✗ NOT SUPPORTED'}")
    print(f"{'='*60}")

    if result_torch or result_triton:
        print("\n配置建议:")
        print("  actor_rollout_ref:")
        print("    model:")
        print("      fused_kernel_options:")
        print(f"        impl_backend: {'torch' if result_torch else 'triton'}")
        print("    actor:")
        print("      use_fused_kernels: true")
        print("    ref:")
        print("      use_fused_kernels: true")
