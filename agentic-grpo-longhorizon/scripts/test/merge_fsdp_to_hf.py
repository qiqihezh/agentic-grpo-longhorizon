#!/usr/bin/env python3
"""
Standalone script to merge veRL FSDP checkpoints to HuggingFace format.
Does NOT require GPU and does NOT import verl (avoids triton issues).

Usage:
    python scripts/test/merge_fsdp_to_hf.py \
        --actor-dir experiments/vanilla/checkpoints/global_step_200/actor \
        --output-dir experiments/vanilla/hf_step_200
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from safetensors.torch import save_file

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def merge_fsdp_checkpoint(actor_dir: str, output_dir: str, merge_lora: bool = True):
    actor_path = Path(actor_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Read FSDP config
    with open(actor_path / "fsdp_config.json") as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config["world_size"]
    print(f"World size: {world_size}")

    # 2. Load all rank shards
    shards = []
    for rank in range(world_size):
        pt_path = actor_path / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"Loading {pt_path.name} ...")
        shards.append(torch.load(pt_path, map_location="cpu", weights_only=False))

    # 3. Merge state dicts
    merged = {}
    keys = list(shards[0].keys())

    for key in keys:
        tensors = [shard.pop(key) for shard in shards]
        is_dt = isinstance(tensors[0], DTensor)

        if is_dt:
            local_tensors = [t._local_tensor.bfloat16() for t in tensors]
            placements = tuple(tensors[0].placements)
            if hasattr(tensors[0], 'device_mesh') and tensors[0].device_mesh.mesh_dim_names:
                mesh_dim_names = tensors[0].device_mesh.mesh_dim_names
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
            else:
                placements = placements[-1:]

            if len(placements) == 1 and placements[0].is_shard():
                dim = placements[0].dim
                merged[key] = torch.cat(local_tensors, dim=dim).contiguous()
            elif all(p.is_replicate() for p in placements):
                merged[key] = local_tensors[0]
            else:
                raise NotImplementedError(f"Unsupported placements for {key}: {placements}")
        else:
            # heuristic: cat along dim 0 for non-DTensor FSDP shards
            merged[key] = torch.cat([t.bfloat16() for t in tensors], dim=0)

    del shards

    # 4. Separate LoRA and base weights
    lora_A = {}
    lora_B = {}
    base_weights = {}
    lora_rank = None
    lora_alpha = None

    for k, v in merged.items():
        if "lora_A" in k:
            new_k = k.replace(".default.weight", ".weight")
            new_k = new_k.replace("base_model.model.", "")
            new_k = new_k.replace(".lora_A.weight", ".weight")
            lora_A[new_k] = v
            if lora_rank is None:
                lora_rank = v.shape[0]
        elif "lora_B" in k:
            new_k = k.replace(".default.weight", ".weight")
            new_k = new_k.replace("base_model.model.", "")
            new_k = new_k.replace(".lora_B.weight", ".weight")
            lora_B[new_k] = v
        else:
            # base layer weight/bias
            new_k = k.replace("base_model.model.", "")
            new_k = new_k.replace(".base_layer.weight", ".weight")
            new_k = new_k.replace(".base_layer.bias", ".bias")
            base_weights[new_k] = v

    # 5. Merge LoRA into base if requested
    if merge_lora and lora_A:
        # Read lora_alpha from existing adapter_config if available
        adapter_config_path = actor_path / "lora_adapter" / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                cfg = json.load(f)
            lora_alpha = cfg.get("lora_alpha", lora_rank * 2)
        else:
            lora_alpha = lora_rank * 2
        scale = lora_alpha / lora_rank
        print(f"Merging LoRA (r={lora_rank}, alpha={lora_alpha}, scale={scale}) into base weights ...")

        for k in lora_A:
            if k not in lora_B:
                raise ValueError(f"Missing lora_B for {k}")
            if k not in base_weights:
                raise ValueError(f"Missing base weight for {k}")
            # W_merged = W_base + scale * lora_B @ lora_A
            delta = scale * (lora_B[k] @ lora_A[k])
            base_weights[k] = base_weights[k] + delta.to(base_weights[k].dtype)

        print(f"LoRA merged. Total merged keys: {len(lora_A)}")
    elif lora_A:
        # Save LoRA adapter separately (same as checkpoint's lora_adapter/)
        lora_out = out_path / "lora_adapter"
        lora_out.mkdir(exist_ok=True)
        lora_params = {}
        # Reconstruct PEFT format keys
        for k in lora_A:
            peft_k = k.replace(".weight", ".lora_A.weight")
            lora_params[peft_k] = lora_A[k]
        for k in lora_B:
            peft_k = k.replace(".weight", ".lora_B.weight")
            lora_params[peft_k] = lora_B[k]
        save_file(lora_params, lora_out / "adapter_model.safetensors")
        if (actor_path / "lora_adapter" / "adapter_config.json").exists():
            shutil.copy2(actor_path / "lora_adapter" / "adapter_config.json", lora_out / "adapter_config.json")
        print(f"LoRA adapter saved to {lora_out}")

    # 6. Save model weights
    print(f"Saving merged model to {out_path} ...")
    save_file(base_weights, out_path / "model.safetensors")

    # 7. Copy HF config & tokenizer
    hf_src = actor_path / "huggingface"
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "merges.txt", "vocab.json"]:
        src = hf_src / fname
        if src.exists():
            shutil.copy2(src, out_path / fname)

    print(f"Done! Output at {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-merge-lora", action="store_true",
                        help="Keep LoRA adapter separate (do not merge into base)")
    args = parser.parse_args()
    merge_fsdp_checkpoint(args.actor_dir, args.output_dir, merge_lora=not args.no_merge_lora)
