"""Bounded-memory gradient evidence for StarWAM GPU smoke tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch


_FUNCTIONAL_PREFIXES = (
    "backbone.dit",
    "backbone.vae",
    "backbone.text_encoder",
    "shared_dit",
    "action_expert",
    "proprio_encoder",
    "action_encoder",
    "action_decoder",
    "action_embedder",
    "action_proj_out",
    "state_encoder",
    "feature_projector",
    "feature_timestep_embedding",
    "mot",
)


def _parameter_group(name: str) -> str:
    normalized = name
    wrapper_prefixes = ("module.", "_orig_mod.")
    while any(normalized.startswith(prefix) for prefix in wrapper_prefixes):
        for prefix in wrapper_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
    if normalized.startswith("wam."):
        normalized = normalized[len("wam.") :]
    for prefix in _FUNCTIONAL_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}."):
            return prefix
    return normalized.split(".", 1)[0]


def collect_gradient_probe(
    model: torch.nn.Module,
    *,
    chunk_size: int = 1_048_576,
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate finite/nonzero gradient evidence by functional module.

    Gradients are inspected in chunks so a probe cannot allocate a second
    full-size mask for a multi-billion-parameter tensor.
    """
    if chunk_size <= 0:
        raise ValueError("gradient probe chunk_size must be positive")

    groups: dict[str, dict[str, float | int | None]] = {}
    for name, parameter in model.named_parameters():
        group_name = _parameter_group(name)
        group = groups.setdefault(
            group_name,
            {
                "parameter_tensors": 0,
                "parameter_elements": 0,
                "trainable_parameter_tensors": 0,
                "trainable_parameter_elements": 0,
                "gradient_tensors": 0,
                "gradient_elements": 0,
                "finite_gradient_elements": 0,
                "nonzero_gradient_elements": 0,
                "gradient_l2_norm": 0.0,
                "finite_gradient_ratio": None,
                "nonzero_gradient_ratio": None,
            },
        )
        group["parameter_tensors"] += 1
        group["parameter_elements"] += parameter.numel()
        if parameter.requires_grad:
            group["trainable_parameter_tensors"] += 1
            group["trainable_parameter_elements"] += parameter.numel()

        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        flat = gradient.detach().reshape(-1)
        group["gradient_tensors"] += 1
        group["gradient_elements"] += flat.numel()
        squared_norm = 0.0
        for start in range(0, flat.numel(), chunk_size):
            values = flat[start : start + chunk_size]
            finite_mask = torch.isfinite(values)
            finite_count = int(finite_mask.sum().item())
            group["finite_gradient_elements"] += finite_count
            if finite_count:
                finite_values = values[finite_mask].float()
                group["nonzero_gradient_elements"] += int(torch.count_nonzero(finite_values).item())
                squared_norm += float(finite_values.square().sum().item())
        group["gradient_l2_norm"] += squared_norm

    for group in groups.values():
        gradient_elements = int(group["gradient_elements"])
        finite_elements = int(group["finite_gradient_elements"])
        group["gradient_l2_norm"] = math.sqrt(float(group["gradient_l2_norm"]))
        if gradient_elements:
            group["finite_gradient_ratio"] = finite_elements / gradient_elements
        if finite_elements:
            group["nonzero_gradient_ratio"] = int(group["nonzero_gradient_elements"]) / finite_elements
    return groups


def write_gradient_probe(
    model: torch.nn.Module,
    output_path: str | Path,
    *,
    model_family: str,
    optimizer_step: int,
    chunk_size: int = 1_048_576,
) -> dict:
    """Write one atomic JSON gradient-probe artifact and return its payload."""
    payload = {
        "model_family": str(model_family),
        "optimizer_step": int(optimizer_step),
        "groups": collect_gradient_probe(model, chunk_size=chunk_size),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, output_path)
    return payload
