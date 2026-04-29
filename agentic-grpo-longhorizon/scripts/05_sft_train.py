"""
Week 2 Day 3-4: LoRA SFT 训练
基于 transformers + peft，不直接用 TRL SFTTrainer
理由：TRL SFTTrainer 对 multi-turn + tool_calls 的 loss mask 处理是黑盒，
我们自己的 TrajectorySFTDataset 已经把 labels 算好了，直接用 HF Trainer 就行。

关键设计:
1. LoRA r=16, alpha=32（PROJECT.md §2.3 规定）
2. target_modules 覆盖 q_proj, k_proj, v_proj, o_proj + gate/up/down_proj
   （Qwen2.5 全 linear，比只跑 attn 效果好）
3. bf16 训练（A800 / 4090 都支持）
4. gradient checkpointing 开（7B 单卡 24GB 也能跑）
5. 不用 packing：multi-turn 样本之间不能 pack，会污染 loss mask

用法:
    python scripts/05_sft_train.py --config configs/sft_airline_lora.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from functools import partial

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from src.training.sft_dataset import TrajectorySFTDataset, collate_fn_padding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------- Tokenizer -------
    print(f"加载 tokenizer: {cfg['model']['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        # PROJECT.md §2.1 历史踩坑：Qwen 的 pad_token 必须显式设
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Qwen 的 padding_side 训练时用 right
    tokenizer.padding_side = "right"

    # ------- Dataset -------
    train_ds = TrajectorySFTDataset(
        jsonl_path=cfg["data"]["train_jsonl"],
        tokenizer=tokenizer,
        tools=None,
        max_length=cfg["data"]["max_length"],
    )
    if len(train_ds) == 0:
        raise RuntimeError("训练集为空，先跑 04_collect_sft_data.py 并确认 04b_inspect_sft_dataset.py")

    eval_ds = None
    if cfg["data"].get("eval_jsonl"):
        eval_ds = TrajectorySFTDataset(
            jsonl_path=cfg["data"]["eval_jsonl"],
            tokenizer=tokenizer,
            tools=None,
            max_length=cfg["data"]["max_length"],
        )

    # ------- Model + LoRA -------
    print(f"加载 base 模型: {cfg['model']['name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_impl", "flash_attention_2"),
        trust_remote_code=True,
    )
    model.config.use_cache = False  # gradient checkpointing 必须

    # 准备 LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.05),
        bias="none",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------- Training Args -------
    output_dir = cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)
    # 保存 config 快照
    with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["train"]["num_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum_steps"],
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type=cfg["train"].get("lr_scheduler", "cosine"),
        warmup_ratio=cfg["train"].get("warmup_ratio", 0.03),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["train"].get("logging_steps", 5),
        save_strategy=cfg["train"].get("save_strategy", "epoch"),
        save_total_limit=cfg["train"].get("save_total_limit", 3),
        eval_strategy="epoch" if eval_ds is not None else "no",
        # 关键：report_to 设 "none" 关掉 HF 自带的 wandb/tensorboard 自动连接
        # swanlab 用 callback 方式手动接入（见下面 callbacks=...）
        report_to="none",
        run_name=cfg["train"].get("run_name", "sft_airline_lora"),
        seed=cfg["train"].get("seed", 42),
        remove_unused_columns=False,  # ⚠️ 必须 False，否则 labels 会被 Trainer 删掉
        dataloader_num_workers=0,
    )

    # ------- swanlab callback -------
    callbacks = []
    logger_cfg = cfg.get("logger", {})
    use_swanlab = logger_cfg.get("backend", "swanlab") == "swanlab"
    if use_swanlab:
        try:
            from swanlab.integration.transformers import SwanLabCallback
        except ImportError as e:
            raise ImportError(
                "需要安装 swanlab: pip install swanlab\n"
                "如果只想看 stdout 不要 logger，把 config 里 logger.backend 设成 'none'"
            ) from e

        swanlab_cb = SwanLabCallback(
            project=logger_cfg.get("project", "agentic-grpo-longhorizon"),
            experiment_name=cfg["train"].get("run_name", "w2_sft_airline_lora"),
            description=logger_cfg.get("description", "Week 2 SFT warmup on airline (option D)"),
            config={
                "model": cfg["model"]["name_or_path"],
                "lora_r": cfg["lora"]["r"],
                "lora_alpha": cfg["lora"]["alpha"],
                "lr": cfg["train"]["lr"],
                "num_epochs": cfg["train"]["num_epochs"],
                "eff_batch_size": cfg["train"]["per_device_batch_size"] * cfg["train"]["grad_accum_steps"],
                "max_length": cfg["data"]["max_length"],
                "n_train_examples": len(train_ds),
            },
        )
        callbacks.append(swanlab_cb)
        print(f"[logger] swanlab enabled, project={logger_cfg.get('project', 'agentic-grpo-longhorizon')}")
    else:
        print("[logger] disabled, only stdout + train_summary.json")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=partial(collate_fn_padding, pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    # ------- Train -------
    train_result = trainer.train()
    trainer.save_model(output_dir)  # 保存 LoRA adapter
    tokenizer.save_pretrained(output_dir)

    # 写出 training summary
    summary = {
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "n_train_examples": len(train_ds),
        "n_eval_examples": len(eval_ds) if eval_ds else 0,
    }
    with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== SFT 训练完成 ===")
    print(f"LoRA adapter 保存在: {output_dir}")
    print(f"final train loss: {train_result.training_loss:.4f}")
    print()
    print("下一步:")
    print(f"  1. 合并 LoRA: python scripts/05b_merge_lora.py "
          f"--base {cfg['model']['name_or_path']} "
          f"--adapter {output_dir} "
          f"--out {output_dir}_merged")
    print(f"  2. 启动 vLLM 加载合并后的模型，跑 scripts/05c_eval_sft.py")


if __name__ == "__main__":
    main()