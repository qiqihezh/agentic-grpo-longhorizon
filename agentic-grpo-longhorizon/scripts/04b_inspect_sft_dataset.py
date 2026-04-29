"""
Dry-run 测试 SFT dataset 的 loss mask 是否正确
打印一条样本，把 label != IGNORE_INDEX 的 token 高亮出来

用法:
    python scripts/04b_inspect_sft_dataset.py \
        --train-jsonl experiments/week2_sft_collect_airline/train.jsonl \
        --tokenizer ../models/Qwen2.5-7B-Instruct \
        --num-show 2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from transformers import AutoTokenizer

from src.training.sft_dataset import TrajectorySFTDataset, IGNORE_INDEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="path to Qwen2.5-7B-Instruct or HF id")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-show", type=int, default=2,
                        help="打印几条样本看 loss mask")
    args = parser.parse_args()

    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    ds = TrajectorySFTDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        tools=None,  # 不传 tools，靠 messages 里 assistant 的 tool_calls 字段渲染
        max_length=args.max_length,
    )

    print(f"\n数据集大小: {len(ds)}")
    if len(ds) == 0:
        print("⚠️  数据集为空，检查 jsonl 文件和渲染逻辑")
        return

    print(f"\n=== 打印前 {args.num_show} 条样本的 loss mask 情况 ===\n")
    for i in range(min(args.num_show, len(ds))):
        ex = ds.examples[i]
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        print(f"--- 样本 {i} (task_id={ex.get('task_id')}) ---")
        print(f"总 token: {len(input_ids)}, label token: {ex['n_label_tokens']} "
              f"({ex['n_label_tokens']/len(input_ids):.1%})")

        # 把 token 解码成可读字符串，标注哪些有 label
        # 用 [TRAIN] / [MASK] 标签
        decoded_chunks = []
        cur_mode = None
        cur_buf = []
        for tok_id, lab in zip(input_ids, labels):
            mode = "TRAIN" if lab != IGNORE_INDEX else "MASK"
            if mode != cur_mode:
                if cur_buf:
                    txt = tokenizer.decode(cur_buf, skip_special_tokens=False)
                    decoded_chunks.append((cur_mode, txt))
                cur_mode = mode
                cur_buf = [tok_id]
            else:
                cur_buf.append(tok_id)
        if cur_buf:
            txt = tokenizer.decode(cur_buf, skip_special_tokens=False)
            decoded_chunks.append((cur_mode, txt))

        # 截断显示，避免刷屏
        for mode, txt in decoded_chunks:
            preview = txt if len(txt) < 200 else txt[:100] + "...[truncated]..." + txt[-100:]
            tag = "🟢[TRAIN]" if mode == "TRAIN" else "⚫[MASK]"
            print(f"  {tag} {repr(preview)}")
        print()

    # 整体统计
    total_tok = sum(e["n_total_tokens"] for e in ds.examples)
    total_lab = sum(e["n_label_tokens"] for e in ds.examples)
    print(f"=== 整体统计 ===")
    print(f"总 token:        {total_tok}")
    print(f"总 label token:  {total_lab} ({total_lab/total_tok:.1%})")
    print(f"平均每样本:       {total_tok/len(ds):.0f} total, "
          f"{total_lab/len(ds):.0f} label")
    print()
    print("✅ 检查清单:")
    print("  1. [TRAIN] 段是否只覆盖 assistant 的回复内容（包括 <tool_call>...</tool_call>）？")
    print("  2. [MASK] 段是否覆盖 system / user / tool 的全部内容？")
    print("  3. [TRAIN] 段是否包含 <|im_end|> 这个结束 token？（应该包含，让模型学会停止）")


if __name__ == "__main__":
    main()
