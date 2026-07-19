"""Backward-compatible model loss selection and logging helpers."""

from __future__ import annotations

from typing import Any

import torch


def resolve_model_loss(output: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    """Select exactly one differentiable objective and collect detached metrics.

    Multi-objective frameworks own their weighting and return ``total_loss``.
    Legacy StarVLA frameworks continue to return only ``action_loss``.
    Component metrics are never summed here, which prevents auxiliary losses
    from being counted twice.
    """
    total_loss = output.get("total_loss")
    if total_loss is None:
        total_loss = output.get("action_loss")
    if not isinstance(total_loss, torch.Tensor):
        raise TypeError("Framework output must contain Tensor `total_loss` or `action_loss`.")
    if total_loss.numel() != 1:
        raise ValueError(f"Optimization loss must be scalar, got shape={tuple(total_loss.shape)}")

    metrics: dict[str, float] = {}
    nested = output.get("loss_metrics", {})
    if nested is not None:
        if not isinstance(nested, dict):
            raise TypeError("Framework `loss_metrics` must be a mapping.")
        for key, value in nested.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)):
                metrics[str(key)] = float(value)

    if "total_loss" not in metrics:
        metrics["total_loss"] = float(total_loss.detach().float().item())
    if "action_loss" not in metrics and isinstance(output.get("action_loss"), torch.Tensor):
        action_loss = output["action_loss"]
        if action_loss.numel() == 1:
            metrics["action_loss"] = float(action_loss.detach().float().item())
    return total_loss, metrics
