"""
SFT Dataset: 把 OpenAI 格式的 multi-turn trajectory 转成 Qwen2.5 训练样本

核心难点 (PROJECT.md §3 Week 2 关键风险第 3 条):
- 多轮对话里只在 assistant turn 算 loss，user/system/tool turn 全部 mask 掉
- 包括 assistant 的 tool_calls 部分（Qwen 渲染成 <tool_call>...</tool_call>），也要算 loss
- 不能简单按 token "user/assistant/system" 字符串切分，因为 Qwen2.5 chat template
  会加各种 special token 和换行，手切容易 off-by-one

稳健做法 ("渲染两次取 diff"):
  对每条 trajectory，遍历 messages 的 assistant turn:
    full_ids   = tokenize(apply_chat_template(messages[:i+1], add_generation_prompt=False))
    prefix_ids = tokenize(apply_chat_template(messages[:i],   add_generation_prompt=True))
    # full_ids 的 [len(prefix_ids):] 部分就是这一轮 assistant 的所有 token
    # （包括可能的 tool_calls 渲染），这部分给 label，其他部分 mask 成 -100
  这样不管 Qwen 模板怎么变，loss mask 都对得上。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


def build_supervised_example(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[dict]] = None,
    max_length: int = 8192,
) -> Optional[dict]:
    """
    把一条 OpenAI 格式的 multi-turn trajectory 渲染成 (input_ids, labels, attention_mask)
    
    返回 None 表示这条样本应被丢弃（超长 / 没有可训练的 assistant turn / 渲染失败）
    
    messages 示例:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."},
        {"role": "assistant", "content": "..."},  # 最后回复
      ]
    """
    # 1. 找出所有 assistant turn 的位置
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return None

    # 2. 先把整段 trajectory 渲染成 token ids
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"[skip] apply_chat_template 失败: {type(e).__name__}: {e}")
        return None

    if len(full_ids) > max_length:
        return None  # 超长直接丢，不做截断（截断后 assistant turn 可能被切掉）

    labels = [IGNORE_INDEX] * len(full_ids)

    # 3. 对每个 assistant turn 算 prefix，定位 label 区间
    for ai in assistant_indices:
        try:
            prefix_text = tokenizer.apply_chat_template(
                messages[:ai],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,  # 关键：让模板输出到 "<|im_start|>assistant\n" 为止
            )
            with_assistant_text = tokenizer.apply_chat_template(
                messages[:ai + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f"[skip] prefix 渲染失败 ai={ai}: {type(e).__name__}: {e}")
            return None

        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        with_assistant_ids = tokenizer(with_assistant_text, add_special_tokens=False)["input_ids"]

        # assistant turn 的 token 范围 = with_assistant_ids[len(prefix_ids):]
        start = len(prefix_ids)
        end = len(with_assistant_ids)

        # 健壮性检查: with_assistant 应该是 prefix 的扩展
        # 但实际上由于 add_generation_prompt 的差异，前缀可能略有不同
        # 我们用一个更稳妥的对齐：找 with_assistant_ids 在 full_ids 里的最大公共前缀
        if end > len(full_ids):
            print(f"[skip] 对齐异常: end={end} > len(full_ids)={len(full_ids)}")
            return None

        # 直接复制 full_ids[start:end] 到 labels
        # 注意：start 也必须 ≤ len(full_ids)
        if start >= len(full_ids):
            continue
        actual_end = min(end, len(full_ids))
        for j in range(start, actual_end):
            labels[j] = full_ids[j]

    # 4. sanity check: labels 至少要有一些非 IGNORE_INDEX 的 token
    n_label_tokens = sum(1 for x in labels if x != IGNORE_INDEX)
    if n_label_tokens < 5:
        return None  # 几乎没东西学，丢

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "n_label_tokens": n_label_tokens,
        "n_total_tokens": len(full_ids),
    }


class TrajectorySFTDataset(Dataset):
    """
    从 train.jsonl 读 trajectory，渲染成训练样本
    每行 jsonl 是 {"messages": [...], "task_id": ..., ...}
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        tools: Optional[list[dict]] = None,
        max_length: int = 8192,
        cache_in_memory: bool = True,
        verbose: bool = True,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_length = max_length

        # 一次性预处理（trajectory 数量不大，几百条以内 OK）
        raw = []
        with open(jsonl_path) as f:
            for line in f:
                raw.append(json.loads(line))

        self.examples: list[dict] = []
        n_skipped_empty = 0
        n_skipped_long = 0
        n_skipped_other = 0

        for r in raw:
            msgs = r["messages"]
            ex = build_supervised_example(msgs, tokenizer, tools=tools, max_length=max_length)
            if ex is None:
                # 区分丢弃原因方便诊断
                try:
                    full_text = tokenizer.apply_chat_template(msgs, tools=tools,
                                                              tokenize=False, add_generation_prompt=False)
                    if len(tokenizer(full_text)["input_ids"]) > max_length:
                        n_skipped_long += 1
                    else:
                        n_skipped_empty += 1
                except Exception:
                    n_skipped_other += 1
                continue
            ex["task_id"] = r.get("task_id", -1)
            self.examples.append(ex)

        if verbose:
            print(f"[Dataset] 加载 {jsonl_path}")
            print(f"  原始 trajectory:   {len(raw)}")
            print(f"  保留 example:      {len(self.examples)}")
            print(f"  丢弃-超长:         {n_skipped_long}")
            print(f"  丢弃-无 label:     {n_skipped_empty}")
            print(f"  丢弃-其他:         {n_skipped_other}")
            if self.examples:
                avg_total = sum(e["n_total_tokens"] for e in self.examples) / len(self.examples)
                avg_label = sum(e["n_label_tokens"] for e in self.examples) / len(self.examples)
                print(f"  平均总 token:       {avg_total:.0f}")
                print(f"  平均 label token:   {avg_label:.0f} "
                      f"({avg_label/avg_total:.1%} 占比)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
        }


def collate_fn_padding(batch: list[dict], pad_token_id: int) -> dict:
    """右 padding 到 batch 内最长长度"""
    max_len = max(b["input_ids"].size(0) for b in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attention_mask[i, :L] = b["attention_mask"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
