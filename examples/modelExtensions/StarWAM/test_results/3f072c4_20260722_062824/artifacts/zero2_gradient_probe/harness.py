#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import safe_get_full_grad

from starVLA.training.gradient_probe import collect_gradient_probe


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = torch.nn.Linear(8, 8, bias=False)
        self.vae = torch.nn.Linear(8, 8, bias=False)
        self.vae.requires_grad_(False)


class TinyWAM(torch.nn.Module):
    model_family = "mot_wam"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.action_expert = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.dit(inputs).square().mean() + self.action_expert(inputs).square().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(20260721)
    model = TinyWAM()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-4,
    )
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "steps_per_print": 1000,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        optimizer=optimizer,
        config=config,
    )

    boundary_pattern = []
    raw_groups = None
    zero_groups = None
    for micro_step in range(2):
        inputs = torch.full((2, 8), float(micro_step + 1), device=engine.device, dtype=torch.bfloat16)
        loss = engine(inputs)
        engine.backward(loss)
        is_boundary = bool(engine.is_gradient_accumulation_boundary())
        boundary_pattern.append(is_boundary)
        if is_boundary:
            raw_groups = collect_gradient_probe(engine.module, chunk_size=4)
            zero_groups = collect_gradient_probe(
                engine.module,
                chunk_size=4,
                gradient_getter=safe_get_full_grad,
            )
        engine.step()

    assert boundary_pattern == [False, True], boundary_pattern
    assert raw_groups is not None
    assert zero_groups is not None
    assert sum(int(group["gradient_elements"]) for group in raw_groups.values()) == 0
    for required_group in ("backbone.dit", "action_expert"):
        group = zero_groups[required_group]
        assert group["gradient_elements"] == group["trainable_parameter_elements"], group
        assert group["finite_gradient_ratio"] == 1.0, group
        assert group["nonzero_gradient_elements"] > 0, group
        assert group["gradient_l2_norm"] > 0.0, group
    frozen = zero_groups["backbone.vae"]
    assert frozen["trainable_parameter_elements"] == 0, frozen
    assert frozen["gradient_elements"] == 0, frozen

    local_result = {
        "rank": rank,
        "world_size": world_size,
        "boundary_pattern": boundary_pattern,
        "raw_gradient_elements": sum(int(group["gradient_elements"]) for group in raw_groups.values()),
        "zero_groups": zero_groups,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        assert all(item["boundary_pattern"] == [False, True] for item in gathered)
        assert all(item["zero_groups"] == gathered[0]["zero_groups"] for item in gathered)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "PASS", "ranks": gathered}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "output": str(output), "world_size": world_size}))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
