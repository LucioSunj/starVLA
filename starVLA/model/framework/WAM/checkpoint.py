"""Checkpoint payload normalization for StarWAM imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def strip_checkpoint_prefixes(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "_orig_mod.", "wam.")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        normalized[stripped] = value
    return normalized


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Read native StarWAM, DeepSpeed, safetensors, or wrapper state."""
    path = Path(path)
    if path.is_dir():
        candidates = (
            "model.pt",
            "pytorch_model.bin",
            "model.bin",
            "mp_rank_00_model_states.pt",
            "pytorch_model/mp_rank_00_model_states.pt",
        )
        selected = next((path / name for name in candidates if (path / name).is_file()), None)
        if selected is None:
            shards = sorted(path.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No supported StarWAM checkpoint found under {path}")
            from safetensors.torch import load_file

            state: dict[str, torch.Tensor] = {}
            for shard in shards:
                state.update(load_file(str(shard), device="cpu"))
            return strip_checkpoint_prefixes(state)
        path = selected

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload: Any = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "module", "state_dict"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dict")
    return strip_checkpoint_prefixes(payload)
