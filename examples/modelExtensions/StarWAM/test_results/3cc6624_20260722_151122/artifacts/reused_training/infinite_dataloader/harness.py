from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from starVLA.training.train_starvla import _infinite_dataloader
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.num_processes < 2:
        raise RuntimeError(f"Harness requires at least two ranks, got {accelerator.num_processes}")

    reset_calls = 0

    def forbidden_reset(*_args, **_kwargs):
        nonlocal reset_calls
        reset_calls += 1
        raise AssertionError("legacy TrainerUtils._reset_dataloader was called")

    TrainerUtils._reset_dataloader = staticmethod(forbidden_reset)

    dataset = TensorDataset(torch.arange(16, dtype=torch.int64))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
    dataloader = accelerator.prepare(dataloader)
    steps_per_rank_epoch = len(dataloader)
    if steps_per_rank_epoch <= 0:
        raise RuntimeError("Prepared dataloader is empty")

    iterator = _infinite_dataloader(dataloader)
    target_steps = 2 * steps_per_rank_epoch
    local_batches: list[list[int]] = []
    collective_values: list[int] = []

    for step in range(target_steps):
        (batch,) = next(iterator)
        local_batches.append([int(value) for value in batch.cpu().tolist()])
        collective = torch.tensor(step + 1, device=accelerator.device, dtype=torch.int64)
        dist.all_reduce(collective, op=dist.ReduceOp.SUM)
        expected = (step + 1) * accelerator.num_processes
        if int(collective.item()) != expected:
            raise AssertionError(
                f"collective mismatch at step {step}: {int(collective.item())} != {expected}"
            )
        collective_values.append(int(collective.item()))

    local_step_count = torch.tensor([target_steps], device=accelerator.device, dtype=torch.int64)
    gathered_step_counts = accelerator.gather(local_step_count).cpu().tolist()
    if len(set(int(value) for value in gathered_step_counts)) != 1:
        raise AssertionError(f"rank step counts differ: {gathered_step_counts}")
    if reset_calls != 0:
        raise AssertionError(f"legacy reset called {reset_calls} times")

    payload = {
        "status": "PASS",
        "world_size": accelerator.num_processes,
        "steps_per_rank_epoch": steps_per_rank_epoch,
        "epochs_completed": 2,
        "steps_per_rank": target_steps,
        "gathered_step_counts": [int(value) for value in gathered_step_counts],
        "legacy_reset_calls": reset_calls,
        "rank": accelerator.process_index,
        "local_batches": local_batches,
        "collective_values": collective_values,
    }
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, sort_keys=True))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
