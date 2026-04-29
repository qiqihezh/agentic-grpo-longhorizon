"""
合并 LoRA adapter 到 base 模型，输出一个完整的可独立加载模型
后续 vLLM 启动直接指向合并后的目录即可

用法:
    python scripts/05b_merge_lora.py \
        --base ../models/Qwen2.5-7B-Instruct \
        --adapter experiments/week2_sft_lora \
        --out experiments/week2_sft_lora_merged
"""
import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base 模型路径")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 路径")
    parser.add_argument("--out", required=True, help="合并后保存路径")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"加载 base: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    print(f"加载 adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("合并中…")
    merged = model.merge_and_unload()
    print(f"保存到: {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)

    # 复制 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.out)

    # 复制 chat_template 等可能的额外文件
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt", "chat_template.json"]:
        src = os.path.join(args.base, fname)
        dst = os.path.join(args.out, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    print("\n✅ 完成。下一步:")
    print(f"  bash scripts/01_start_vllm_server_7b_sft.sh   # 改 MODEL_PATH={args.out}")
    print(f"  python scripts/05c_eval_sft.py --config configs/eval_sft_airline.yaml")


if __name__ == "__main__":
    main()
