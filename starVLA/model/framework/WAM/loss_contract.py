"""Loss-contract validation for StarWAM framework adapters."""

from __future__ import annotations

import math
from typing import Any

import torch


def validated_loss_metrics(
    model_family: str,
    framework_config: Any,
    total_loss: torch.Tensor,
    metrics: dict[str, Any],
) -> dict[str, float]:
    """Validate StarWAM's owned weighting and return detached components."""
    action_loss = float(metrics.get("loss_action", 0.0))
    video_loss = float(metrics.get("loss_video", 0.0))
    reported_total = float(metrics.get("loss_total", total_loss.detach().float().item()))
    lambda_action = float(framework_config.loss_lambda_action)
    lambda_video = float(framework_config.loss_lambda_video)

    if model_family == "feature_conditioned_action_model":
        if not math.isclose(video_loss, 0.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise RuntimeError(f"Feature-conditioned StarWAM must report video_loss=0, got {video_loss}")
        expected_total = lambda_action * action_loss
    else:
        expected_total = lambda_action * action_loss + lambda_video * video_loss
    if not math.isclose(reported_total, expected_total, rel_tol=1.0e-4, abs_tol=1.0e-5):
        raise RuntimeError(
            "StarWAM returned an inconsistent total loss: "
            f"reported={reported_total}, expected={expected_total} "
            f"(lambda_action={lambda_action}, lambda_video={lambda_video})."
        )

    loss_metrics = {
        "action_loss": action_loss,
        "video_loss": video_loss,
        "total_loss": reported_total,
    }
    loss_metrics.update(
        {key: float(value) for key, value in metrics.items() if key not in {"loss_action", "loss_video", "loss_total"}}
    )
    return loss_metrics
