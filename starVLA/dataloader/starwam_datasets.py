"""StarWAM tensor-dict dataloader for StarVLA training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.model.framework.WAM.config import get_starwam_config


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _roots(starwam_config: Any) -> list[str]:
    data = starwam_config.data
    roots = list(data.dataset_dirs) if data.dataset_dirs else ([data.root] if data.root else [])
    return [str(root) for root in roots if root]


def _prepare_text_caches(starwam_config: Any) -> None:
    if starwam_config.data.dataset_type != "lerobot":
        return
    roots = _roots(starwam_config)
    from starwam.builder import _ensure_text_caches

    _ensure_text_caches(starwam_config, roots)


def _update_embedded_data_path(cfg: Any, key: str, value: str) -> None:
    raw = cfg.unwrap() if hasattr(cfg, "unwrap") else cfg
    OmegaConf.update(
        raw,
        f"framework.starwam.resolved.data.{key}",
        value,
        merge=False,
        force_add=True,
    )


def _resolve_statistics_paths(cfg: Any, starwam_config: Any) -> tuple[str | None, str | None]:
    data = starwam_config.data
    need_action = bool(data.normalize_actions)
    need_state = bool(data.normalize_states)
    action_path = getattr(data, "action_stats_path", None)
    state_path = getattr(data, "state_stats_path", None) or action_path
    if not action_path and need_action:
        action_path = str(Path(cfg.output_dir) / "starwam_source_statistics.json")
        data.action_stats_path = action_path
        _update_embedded_data_path(cfg, "action_stats_path", action_path)
    if not state_path and need_state:
        state_path = action_path or str(Path(cfg.output_dir) / "starwam_source_statistics.json")
        data.state_stats_path = state_path
        _update_embedded_data_path(cfg, "state_stats_path", state_path)
    return action_path, state_path


def _prepare_statistics(cfg: Any, starwam_config: Any) -> None:
    data = starwam_config.data
    if data.dataset_type != "lerobot":
        return
    need_action = bool(data.normalize_actions)
    need_state = bool(data.normalize_states)
    if not need_action and not need_state:
        return

    roots = _roots(starwam_config)
    if not roots:
        raise ValueError("StarWAM LeRobot normalization requires data.root or data.dataset_dirs.")

    from starwam.builder import _load_or_compute_lerobot_stats

    action_path, state_path = _resolve_statistics_paths(cfg, starwam_config)

    if action_path and state_path == action_path:
        _load_or_compute_lerobot_stats(
            Path(action_path),
            roots,
            data.action_key,
            data.state_key,
            need_action,
            need_state,
        )
        return
    if need_action:
        _load_or_compute_lerobot_stats(
            Path(action_path),
            roots,
            data.action_key,
            data.state_key,
            True,
            False,
        )
    if need_state:
        _load_or_compute_lerobot_stats(
            Path(state_path),
            roots,
            data.action_key,
            data.state_key,
            False,
            True,
        )


def prepare_starwam_data_artifacts(cfg: Any) -> None:
    """Prepare text embeddings and normalization statistics on rank 0."""
    starwam_config = get_starwam_config(cfg)
    _resolve_statistics_paths(cfg, starwam_config)
    if _is_main_process():
        _prepare_text_caches(starwam_config)
        _prepare_statistics(cfg, starwam_config)
    if dist.is_initialized():
        dist.barrier()


def _json_stats(stats: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    if not stats:
        return {}
    return {
        key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value for key, value in stats.items()
    }


def save_starwam_statistics(starwam_config: Any, output_dir: str | Path, normalization_key: str) -> None:
    """Copy StarWAM action/state statistics into a StarVLA checkpoint bundle."""
    from starwam.data.lerobot import load_lerobot_stats

    data = starwam_config.data
    action_path = getattr(data, "action_stats_path", None)
    state_path = getattr(data, "state_stats_path", None) or action_path
    action_stats = None
    state_stats = None
    if action_path and Path(action_path).is_file():
        loaded = load_lerobot_stats(action_path)
        action_stats = loaded.get("action")
        if state_path == action_path:
            state_stats = loaded.get("state")
    if state_stats is None and state_path and Path(state_path).is_file():
        state_stats = load_lerobot_stats(state_path).get("state")

    payload = {
        normalization_key: {
            "action": _json_stats(action_stats),
            "state": _json_stats(state_stats),
        }
    }
    output_path = Path(output_dir) / "dataset_statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, output_path)


def make_starwam_dataloader(cfg: Any, model: Any = None) -> DataLoader:
    from starwam.builder import build_dataset

    prepare_starwam_data_artifacts(cfg)
    starwam_config = get_starwam_config(cfg)
    text_dim = 4096
    if model is not None and hasattr(model, "wam"):
        text_dim = int(getattr(getattr(model.wam.backbone, "info", None), "text_dim", text_dim))
    dataset = build_dataset(starwam_config, is_training=True, text_dim=text_dim)

    data_cfg = cfg.datasets.vla_data
    num_workers = int(data_cfg.get("num_workers", starwam_config.training.num_workers))
    kwargs = {
        "batch_size": int(data_cfg.per_device_batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 1))
        kwargs["timeout"] = int(data_cfg.get("worker_timeout", 120))

    if _is_main_process():
        save_starwam_statistics(
            starwam_config,
            cfg.output_dir,
            str(cfg.framework.starwam.get("normalization_key", "starwam")),
        )
    if dist.is_initialized():
        dist.barrier()
    dataloader = DataLoader(dataset, **kwargs)
    if len(dataloader) == 0:
        raise ValueError(
            "StarWAM DataLoader has zero batches. Increase the dataset size or reduce "
            f"per_device_batch_size={kwargs['batch_size']} (drop_last=True)."
        )
    return dataloader
