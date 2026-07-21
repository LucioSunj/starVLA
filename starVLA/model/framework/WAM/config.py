"""Configuration bridge between StarVLA run configs and StarWAM recipes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

STARWAM_REVISION = "a7b05c8da8f8c88bf4888aa737b938af3b67ab48"


def require_starwam() -> None:
    """Raise an actionable error only when the StarWAM integration is selected."""
    if importlib.util.find_spec("starwam") is None:
        raise RuntimeError(
            "framework.name='StarWAM' requires the optional StarWAM dependency. "
            "Install it with `pip install -e '.[wam]'` or, for this workspace, "
            "`pip install -e '../../WorldActionModels/StarWAM[train]'`."
        )


def detect_starwam_revision() -> str:
    """Return the actual installed/editable StarWAM revision when discoverable."""
    spec = importlib.util.find_spec("starwam")
    if spec is not None and spec.origin:
        source_root = Path(spec.origin).resolve().parent.parent
        if (source_root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except OSError:
                pass

    try:
        direct_url = metadata.distribution("starwam").read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            commit_id = payload.get("vcs_info", {}).get("commit_id")
            if commit_id:
                return str(commit_id)
    except (metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        pass
    return STARWAM_REVISION


def _plain(value: Any) -> Any:
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _plain(item) for key, item in vars(value).items()}
    return value


def _select(cfg: Any, path: str, default: Any = None) -> Any:
    raw = cfg.unwrap() if hasattr(cfg, "unwrap") else cfg
    if OmegaConf.is_config(raw):
        return OmegaConf.select(raw, path, default=default)
    current = raw
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _setdefault(cfg: DictConfig, path: str, value: Any) -> None:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            OmegaConf.update(cfg, path, value, merge=False, force_add=True)
            return
        current = current[part]


def _resolve_recipe_path(recipe: str, config_path: str | Path | None) -> Path:
    path = Path(recipe).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser().resolve().parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _materialize_starwam_config(mapping: dict[str, Any]):
    require_starwam()
    from starwam.config import StarWAMConfig, _from_dict

    return _from_dict(StarWAMConfig, mapping)


def get_starwam_config(cfg: Any):
    """Return the resolved StarWAM dataclass stored in a host config."""
    resolved = _select(cfg, "framework.starwam.resolved")
    if resolved is None:
        raise ValueError(
            "StarWAM config has not been resolved. Call prepare_starwam_host_config() "
            "before building the framework or dataloader."
        )
    return _materialize_starwam_config(_plain(resolved))


def is_starwam_config(cfg: Any) -> bool:
    return _select(cfg, "framework.name") == "StarWAM"


def select_config(cfg: Any, path: str, default: Any = None) -> Any:
    """Select a value from OmegaConf, tracked config, mapping, or namespace."""
    return _select(cfg, path, default)


def prepare_starwam_host_config(
    cfg: DictConfig,
    *,
    config_path: str | Path | None = None,
) -> DictConfig:
    """Resolve a StarWAM recipe and project its runtime settings into StarVLA.

    The resolved dataclass is embedded under ``framework.starwam.resolved`` so
    saved StarVLA checkpoints do not depend on the original recipe path.
    Explicit StarVLA fields win over projected recipe defaults.
    """
    if not is_starwam_config(cfg):
        return cfg

    require_starwam()
    from starwam.config import config_to_dict, load_config
    from starwam.taxonomy import validate_taxonomy
    from starwam.utils.config_cli import apply_overrides

    starwam_node = OmegaConf.select(cfg, "framework.starwam")
    if starwam_node is None:
        raise ValueError("framework.name='StarWAM' requires `framework.starwam`.")

    resolved = OmegaConf.select(cfg, "framework.starwam.resolved")
    recipe = OmegaConf.select(cfg, "framework.starwam.recipe")
    overrides = list(OmegaConf.select(cfg, "framework.starwam.overrides", default=[]) or [])
    if resolved is not None:
        starwam_cfg = _materialize_starwam_config(_plain(resolved))
        resolved_recipe = recipe
    else:
        if not recipe:
            raise ValueError("StarWAM requires `framework.starwam.recipe` or an embedded `resolved` config.")
        recipe_path = _resolve_recipe_path(str(recipe), config_path)
        starwam_cfg = load_config(recipe_path)
        resolved_recipe = str(recipe_path)
    if overrides:
        starwam_cfg = apply_overrides(starwam_cfg, overrides)

    model_family = validate_taxonomy(starwam_cfg)
    action_representation = str(starwam_cfg.taxonomy.action_representation)
    supported_pairs = {
        "mot_wam": "token_action",
        "shared_dit_wam": "token_action",
        "feature_conditioned_action_model": "action_head",
    }
    expected_representation = supported_pairs[model_family]
    if action_representation != expected_representation:
        raise NotImplementedError(
            "The StarVLA StarWAM adapter supports only MoT/token_action, "
            "Shared-DiT/token_action, and Feature-conditioned/action_head. "
            f"Got {model_family}/{action_representation}; latent-action/LAPA is out of scope."
        )
    training = starwam_cfg.training
    data = starwam_cfg.data
    inference = starwam_cfg.inference

    _setdefault(cfg, "seed", training.seed)
    _setdefault(cfg, "run_root_dir", str(Path(training.output_dir).parent))
    _setdefault(cfg, "run_id", Path(training.output_dir).name)
    run_root = OmegaConf.select(cfg, "run_root_dir")
    run_id = OmegaConf.select(cfg, "run_id")
    if not run_root or not run_id:
        raise ValueError("StarWAM training requires non-empty StarVLA `run_root_dir` and `run_id` values.")
    training.output_dir = str(Path(str(run_root)) / str(run_id))
    _setdefault(cfg, "is_debug", False)
    _setdefault(cfg, "wandb_entity", None)
    _setdefault(cfg, "wandb_project", training.wandb_project)

    _setdefault(cfg, "datasets.vla_data.dataset_py", "starwam_datasets")
    _setdefault(cfg, "datasets.vla_data.data_mix", starwam_cfg.taxonomy.preset or model_family)
    _setdefault(cfg, "datasets.vla_data.per_device_batch_size", training.batch_size)
    _setdefault(cfg, "datasets.vla_data.num_workers", training.num_workers)
    _setdefault(cfg, "datasets.vla_data.pin_memory", True)
    _setdefault(cfg, "datasets.vla_data.persistent_workers", training.num_workers > 0)
    _setdefault(cfg, "datasets.vla_data.prefetch_factor", 1)

    _setdefault(cfg, "trainer.max_train_steps", training.max_steps)
    _setdefault(cfg, "trainer.num_epochs", training.num_epochs)
    _setdefault(cfg, "trainer.num_warmup_steps", training.warmup_steps)
    _setdefault(cfg, "trainer.warmup_ratio", training.warmup_ratio)
    _setdefault(cfg, "trainer.save_interval", training.save_every)
    _setdefault(cfg, "trainer.eval_interval", training.eval_every)
    _setdefault(cfg, "trainer.logging_frequency", training.log_every)
    _setdefault(cfg, "trainer.gradient_clipping", training.max_grad_norm)
    _setdefault(cfg, "trainer.gradient_accumulation_steps", training.gradient_accumulation_steps)
    _setdefault(cfg, "trainer.mixed_precision", training.mixed_precision)
    _setdefault(cfg, "trainer.training_strategy", training.strategy)
    _setdefault(cfg, "trainer.wandb_enabled", training.wandb_enabled)
    _setdefault(cfg, "trainer.freeze_modules", "")
    _setdefault(cfg, "trainer.save_format", "pt")
    _setdefault(cfg, "trainer.learning_rate.base", training.learning_rate)
    _setdefault(cfg, "trainer.optimizer.name", "AdamW")
    _setdefault(cfg, "trainer.optimizer.betas", [0.9, 0.95])
    _setdefault(cfg, "trainer.optimizer.eps", 1.0e-8)
    _setdefault(cfg, "trainer.optimizer.weight_decay", training.weight_decay)
    _setdefault(cfg, "trainer.optimizer.fused", False)

    effective_strategy = str(OmegaConf.select(cfg, "trainer.training_strategy"))
    if effective_strategy != "full":
        raise NotImplementedError(
            "StarVLA's StarWAM adapter currently supports only training.strategy='full'; "
            f"got {effective_strategy!r}. LoRA and staged training remain native-StarWAM-only."
        )
    freeze_modules = OmegaConf.select(cfg, "trainer.freeze_modules", default="")
    if freeze_modules not in (None, "", []):
        raise NotImplementedError(
            "StarWAM owns its full-strategy trainable parameter set; "
            "trainer.freeze_modules is not supported for this framework."
        )

    scheduler = str(training.lr_scheduler_type).lower()
    if scheduler == "cosine":
        _setdefault(cfg, "trainer.lr_scheduler_type", "cosine_with_min_lr")
        effective_lr = float(OmegaConf.select(cfg, "trainer.learning_rate.base"))
        _setdefault(cfg, "trainer.scheduler_specific_kwargs.min_lr", effective_lr * 0.01)
    elif scheduler == "cosine_with_min_lr":
        _setdefault(cfg, "trainer.lr_scheduler_type", scheduler)
        _setdefault(cfg, "trainer.scheduler_specific_kwargs.min_lr", training.min_lr)
    else:
        _setdefault(cfg, "trainer.lr_scheduler_type", "constant")
        _setdefault(cfg, "trainer.scheduler_specific_kwargs", {})

    _setdefault(cfg, "trainer.eval_num_inference_steps", training.eval_num_inference_steps)
    _setdefault(
        cfg,
        "trainer.eval_action_num_inference_steps",
        training.eval_action_num_inference_steps or inference.action_num_inference_steps,
    )
    _setdefault(cfg, "trainer.eval_max_samples", training.eval_max_samples)
    _setdefault(cfg, "trainer.eval_compute_video_psnr", training.eval_compute_video_psnr)
    _setdefault(cfg, "trainer.gradient_probe_enabled", False)
    _setdefault(cfg, "trainer.gradient_probe_step", 1)
    _setdefault(cfg, "trainer.gradient_probe_chunk_size", 1_048_576)

    OmegaConf.update(
        cfg,
        "framework.starwam.resolved",
        config_to_dict(starwam_cfg),
        merge=False,
        force_add=True,
    )
    runtime_revision = detect_starwam_revision()
    source_revision = OmegaConf.select(cfg, "framework.starwam.source_revision") or runtime_revision
    OmegaConf.update(
        cfg,
        "framework.starwam.source_revision",
        source_revision,
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "framework.starwam.runtime_revision",
        runtime_revision,
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "framework.starwam.pinned_revision",
        STARWAM_REVISION,
        merge=False,
        force_add=True,
    )
    if resolved_recipe:
        OmegaConf.update(
            cfg,
            "framework.starwam.resolved_recipe",
            str(resolved_recipe),
            merge=False,
            force_add=True,
        )
    _setdefault(cfg, "framework.starwam.init_checkpoint", None)
    _setdefault(cfg, "framework.starwam.strict_init", False)
    _setdefault(cfg, "framework.starwam.normalization_key", "starwam")

    # Keep data-derived fields visible in the host snapshot for diagnostics.
    _setdefault(cfg, "datasets.vla_data.action_norm_mode", data.action_norm_mode)
    _setdefault(cfg, "datasets.vla_data.state_norm_mode", data.state_norm_mode)
    return cfg
