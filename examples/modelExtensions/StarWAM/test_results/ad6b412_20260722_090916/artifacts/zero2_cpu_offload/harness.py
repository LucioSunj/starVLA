#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader, TensorDataset

from starVLA.training.gradient_probe import collect_gradient_probe


class TinyModel(torch.nn.Module):
    model_family = "mot_wam"

    def __init__(self) -> None:
        super().__init__()
        self.action_expert = torch.nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.action_expert(inputs).square().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=2, step_scheduler_with_optimizer=False)
    torch.manual_seed(20260721)
    model = TinyModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-2,
        fused=False,
    )
    dataloader = DataLoader(TensorDataset(torch.ones(4, 8)), batch_size=1)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    zero_optimizer = model.optimizer
    cpu_optimizer = zero_optimizer.optimizer
    assert isinstance(cpu_optimizer, DeepSpeedCPUAdam), type(cpu_optimizer)
    group = cpu_optimizer.param_groups[0]
    assert tuple(group["betas"]) == (0.9, 0.95), group
    assert group["eps"] == 1.0e-8, group
    assert group["weight_decay"] == 1.0e-2, group
    assert cpu_optimizer.adam_w_mode, cpu_optimizer

    before = model.module.action_expert.weight.detach().clone()
    boundaries = []
    probe = None
    for micro_step, (inputs,) in zip(range(2), dataloader):
        inputs = inputs.to(device=model.device, dtype=model.module.action_expert.weight.dtype)
        with accelerator.autocast():
            loss = model(inputs)
        model.backward(loss)
        boundary = bool(model.is_gradient_accumulation_boundary())
        boundaries.append(boundary)
        if boundary:
            from deepspeed.utils import safe_get_full_grad

            probe = collect_gradient_probe(
                model.module,
                chunk_size=4,
                gradient_getter=safe_get_full_grad,
            )
        model.step()

    assert boundaries == [False, True], boundaries
    assert probe is not None
    action_group = probe["action_expert"]
    assert action_group["finite_gradient_ratio"] == 1.0, action_group
    assert action_group["nonzero_gradient_elements"] > 0, action_group
    assert not torch.equal(before, model.module.action_expert.weight.detach())

    local = {
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "boundaries": boundaries,
        "optimizer": type(cpu_optimizer).__name__,
        "adamw_mode": bool(cpu_optimizer.adam_w_mode),
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "weight_decay": group["weight_decay"],
        "gradient_l2_norm": action_group["gradient_l2_norm"],
        "finite_gradient_ratio": action_group["finite_gradient_ratio"],
        "nonzero_gradient_elements": action_group["nonzero_gradient_elements"],
    }
    gathered = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "PASS", "ranks": gathered}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "world_size": accelerator.num_processes}))
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
