"""StarVLA adapter for StarWAM's joint world-action model families."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.WAM.checkpoint import load_checkpoint_state
from starVLA.model.framework.WAM.config import get_starwam_config, require_starwam, select_config
from starVLA.model.framework.WAM.loss_contract import validated_loss_metrics
from starVLA.model.framework.WAM.runtime import (
    TextConditioner,
    compose_camera_views,
    denormalize_actions,
    normalize_vector,
    tensor_stats,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("StarWAM")
class StarWAMFramework(baseframework):
    """Expose StarWAM models through StarVLA's train and policy contracts."""

    def __init__(
        self,
        config: Any,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        require_starwam()
        from starwam.builder import build_framework as build_starwam_framework

        self.config = config
        self.starwam_config = get_starwam_config(config)
        mixed_precision = str(
            select_config(
                config,
                "trainer.mixed_precision",
                getattr(self.starwam_config.training, "mixed_precision", "bf16"),
            )
        ).lower()
        dtype = dtype or {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision, torch.float32)
        if str(device).startswith("cpu"):
            dtype = torch.float32
        self.wam = build_starwam_framework(self.starwam_config, device=str(device), dtype=dtype)
        self.action_horizon = int(self.starwam_config.framework.chunk_size)
        self.model_family = str(self.starwam_config.taxonomy.model_family)
        self.normalization_key = str(config.framework.starwam.get("normalization_key", "starwam"))
        self._text_conditioner = TextConditioner(self.starwam_config, self.wam)
        self._action_stats = None
        self._state_stats = None
        self._stats_initialized = False

        init_checkpoint = config.framework.starwam.get("init_checkpoint", None)
        if init_checkpoint:
            self._load_init_checkpoint(
                init_checkpoint,
                strict=bool(config.framework.starwam.get("strict_init", False)),
            )
            config.framework.starwam.init_checkpoint = None
            config.framework.starwam.imported_checkpoint = str(init_checkpoint)

    def _load_init_checkpoint(self, path: str | Path, *, strict: bool) -> None:
        state = load_checkpoint_state(path)
        model_keys = set(self.wam.state_dict())
        matched = model_keys.intersection(state)
        if not matched:
            raise RuntimeError(f"StarWAM checkpoint {path} matched zero model parameters")
        required_prefixes = {
            "mot_wam": ("mot.", "action_expert."),
            "shared_dit_wam": ("shared_dit.",),
            "feature_conditioned_action_model": ("action_expert.", "feature_projector."),
        }.get(self.model_family, ())
        missing_critical = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in matched)]
        if missing_critical:
            raise RuntimeError(f"StarWAM checkpoint {path} is missing critical parameter groups: {missing_critical}")
        result = self.wam.load_state_dict(state, strict=strict)
        if not strict and result.missing_keys:
            print(f"[StarVLA/StarWAM] missing init keys (first 20): {result.missing_keys[:20]}")
        if not strict and result.unexpected_keys:
            print(f"[StarVLA/StarWAM] unexpected init keys (first 20): {result.unexpected_keys[:20]}")

    def configure_training(self, cfg: Any) -> None:
        freeze_modules = select_config(cfg, "trainer.freeze_modules", "")
        if freeze_modules not in (None, "", []):
            raise NotImplementedError(
                "StarWAM owns its full-strategy trainable parameter set; "
                "trainer.freeze_modules is not supported for this framework."
            )
        strategy = str(
            select_config(
                cfg,
                "trainer.training_strategy",
                getattr(self.starwam_config.training, "strategy", "full"),
            )
        )
        if strategy != "full":
            raise NotImplementedError(
                f"StarVLA's StarWAM adapter currently supports training.strategy='full', got {strategy!r}. "
                "Use StarWAM's native trainer for lora/staged runs."
            )

        model = self.wam
        model.requires_grad_(False)
        if hasattr(model, "mot"):
            model.mot.requires_grad_(True)
        elif hasattr(model, "shared_dit"):
            model.shared_dit.requires_grad_(True)
        elif hasattr(model, "action_expert"):
            model.action_expert.requires_grad_(True)
            if bool(getattr(model.config, "feature_condition_train_backbone", False)):
                model.backbone.get_dit().requires_grad_(True)
        elif hasattr(model, "backbone"):
            model.backbone.get_dit().requires_grad_(True)

        containers = [model]
        if hasattr(model, "shared_dit"):
            containers.append(model.shared_dit)
        for container in containers:
            for name in (
                "action_encoder",
                "action_decoder",
                "action_embedder",
                "action_proj_out",
                "state_encoder",
                "proprio_encoder",
                "feature_projector",
                "feature_timestep_embedding",
            ):
                module = getattr(container, name, None)
                if module is not None:
                    module.requires_grad_(True)

    def forward(self, examples: dict[str, Any], **_: Any) -> dict[str, Any]:
        if not isinstance(examples, dict):
            raise TypeError("StarWAM training expects a tensor dict from dataset_py='starwam_datasets'.")
        total_loss, metrics = self.wam.training_step(examples)
        loss_metrics = validated_loss_metrics(
            self.model_family,
            self.starwam_config.framework,
            total_loss,
            metrics,
        )
        return {"total_loss": total_loss, "loss_metrics": loss_metrics}

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        compute_module = self.wam.shared_dit if self.model_family == "shared_dit_wam" else self.wam
        parameter = next(compute_module.parameters())
        return parameter.device, parameter.dtype

    def _sampled_video_frames(self) -> int:
        num_frames = int(self.starwam_config.data.num_frames)
        ratio = max(1, int(self.starwam_config.data.action_freq_ratio))
        return len(range(0, num_frames, ratio))

    def _normalized_state(self, state: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        proprio_dim = getattr(self.starwam_config.framework, "proprio_dim", None)
        if not proprio_dim:
            return None
        if state is None:
            raise ValueError(f"{self.model_family} requires a {int(proprio_dim)}-D `state` input for policy inference.")
        value = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, -1)[:, : int(proprio_dim)]
        if value.shape[-1] < int(proprio_dim):
            raise ValueError(f"State dim {value.shape[-1]} is smaller than proprio_dim={proprio_dim}")
        if bool(getattr(self.starwam_config.data, "normalize_states", False)):
            self._ensure_stats()
            if self._state_stats is None:
                raise FileNotFoundError("StarWAM state statistics are unavailable for policy normalization.")
            value = normalize_vector(value, self.starwam_config.data.state_norm_mode, self._state_stats)
        return value.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: Any) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        device, dtype = self._model_device_dtype()
        inference_steps = int(kwargs.get("num_inference_steps") or self.starwam_config.inference.num_inference_steps)
        action_steps = int(
            kwargs.get("action_num_inference_steps") or self.starwam_config.inference.action_num_inference_steps
        )
        base_seed = kwargs.get("seed", self.starwam_config.inference.seed)
        predictions = []
        for index, example in enumerate(examples):
            image = compose_camera_views(example["image"], self.starwam_config.data).to(device=device, dtype=dtype)
            context, context_mask = self._text_conditioner.encode([str(example["lang"])], device=device, dtype=dtype)
            proprio = self._normalized_state(example.get("state"), device, dtype)
            prediction = self.wam.infer_action(
                input_image=image,
                context=context,
                context_mask=context_mask,
                action_horizon=self.action_horizon,
                num_inference_steps=inference_steps,
                action_num_inference_steps=action_steps,
                seed=None if base_seed is None else int(base_seed) + index,
                proprio=proprio,
                num_video_frames=self._sampled_video_frames(),
            )
            predictions.append(prediction[0].detach().float().cpu().numpy())
        return {"normalized_actions": np.stack(predictions, axis=0)}

    @torch.inference_mode()
    def evaluate_batch(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, float]:
        was_training = self.wam.training
        self.wam.eval()
        try:
            return self._evaluate_batch_impl(batch, **kwargs)
        finally:
            self.wam.train(was_training)

    def _evaluate_batch_impl(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, float]:
        device, dtype = self._model_device_dtype()
        video = batch["video"]
        target = batch["action"]
        proprio = batch.get("proprio")
        inference_steps = int(kwargs.get("num_inference_steps", 4))
        action_steps = int(kwargs.get("action_num_inference_steps", 4))
        base_seed = int(kwargs.get("seed", self.starwam_config.inference.seed))
        predictions = []
        psnrs = []
        compute_video = bool(kwargs.get("compute_video", False))
        can_infer_video = self.model_family in {"mot_wam", "shared_dit_wam"}
        sample_count = min(video.shape[0], max(int(kwargs.get("max_samples", video.shape[0])), 1))
        for index in range(sample_count):
            common = {
                "input_image": video[index : index + 1, :, 0].to(device=device, dtype=dtype),
                "context": batch["context"][index : index + 1].to(device=device, dtype=dtype),
                "context_mask": (
                    batch["context_mask"][index : index + 1].to(device=device)
                    if batch.get("context_mask") is not None
                    else None
                ),
                "action_horizon": target.shape[1],
                "num_inference_steps": inference_steps,
                "action_num_inference_steps": action_steps,
                "seed": base_seed + index,
                "proprio": (
                    proprio[index : index + 1, 0].to(device=device, dtype=dtype) if proprio is not None else None
                ),
                "num_video_frames": video.shape[2],
            }
            pred = self.wam.infer_action(**common)
            predictions.append(pred.detach().float().cpu())
            if compute_video and can_infer_video:
                joint = self.wam.infer_joint(**common)
                predicted_video = joint["video"].detach().float().cpu()
                target_video = video[index : index + 1, :, : predicted_video.shape[2]].detach().float().cpu()
                mse_video = (predicted_video - target_video).square().mean().clamp_min(1.0e-12)
                psnrs.append(10.0 * torch.log10(mse_video.new_tensor(4.0) / mse_video))

        prediction = torch.cat(predictions, dim=0)
        error = (prediction - target[:sample_count].detach().float().cpu()).square()
        padding = batch.get("action_is_pad")
        if padding is not None:
            keep = (~padding[:sample_count].detach().cpu()).unsqueeze(-1).expand_as(error)
            mse = error[keep].mean() if keep.any() else error.new_zeros(())
        else:
            mse = error.mean()
        metrics = {"action_mse": float(mse.item()), "mse_score": float(mse.item())}
        if psnrs:
            metrics["video_psnr"] = float(torch.stack(psnrs).mean().item())
        return metrics

    def _ensure_stats(self) -> None:
        if self._stats_initialized:
            return
        attached = getattr(self, "norm_stats", None)
        if attached and self.normalization_key in attached:
            block = attached[self.normalization_key]
            self._action_stats = tensor_stats(block.get("action"))
            self._state_stats = tensor_stats(block.get("state"))

        from starwam.data.lerobot import load_lerobot_stats

        data_cfg = self.starwam_config.data
        action_path = getattr(data_cfg, "action_stats_path", None)
        state_path = getattr(data_cfg, "state_stats_path", None) or action_path
        if self._action_stats is None and action_path and Path(action_path).is_file():
            raw = load_lerobot_stats(action_path)
            self._action_stats = tensor_stats(raw.get("action"))
            if self._state_stats is None and state_path == action_path:
                self._state_stats = tensor_stats(raw.get("state"))
        if self._state_stats is None and state_path and Path(state_path).is_file():
            self._state_stats = tensor_stats(load_lerobot_stats(state_path).get("state"))
        self._stats_initialized = True

    def uses_framework_action_unnormalization(self) -> bool:
        return True

    def unnormalize_actions(self, actions: np.ndarray, **_: Any) -> np.ndarray:
        self._ensure_stats()
        if not bool(getattr(self.starwam_config.data, "normalize_actions", False)):
            return np.asarray(actions, dtype=np.float32)
        if self._action_stats is None:
            raise FileNotFoundError("StarWAM action statistics are unavailable for policy un-normalization.")
        tensor = torch.as_tensor(actions, dtype=torch.float32)
        return denormalize_actions(tensor, self.starwam_config.data.action_norm_mode, self._action_stats).numpy()

    def get_policy_metadata(self) -> dict[str, Any]:
        data_cfg = self.starwam_config.data
        is_robotwin = str(data_cfg.concat_multi_camera) == "robotwin"
        return {
            "action_chunk_size": self.action_horizon,
            "available_unnorm_keys": [self.normalization_key],
            "default_unnorm_key": self.normalization_key,
            "action_layout": "native_qpos" if is_robotwin else "default",
            "camera_order": list(data_cfg.video_keys or [data_cfg.video_key]),
            "camera_layout": str(data_cfg.concat_multi_camera),
            "framework_preprocesses_images": True,
            "training_obs_image_size": list(data_cfg.video_size),
            "state_required": bool(getattr(self.starwam_config.framework, "proprio_dim", None)),
            "model_family": self.model_family,
        }
