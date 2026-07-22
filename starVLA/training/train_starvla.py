# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA's trainer uses native PyTorch, Accelerate, and DeepSpeed while keeping
the loop explicit and easy to modify.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if-else chains).
"""

# Standard Library
import argparse
import json
import logging
import os
import time
from math import ceil
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.dataloader.starwam_datasets import prepare_starwam_data_artifacts
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.framework.WAM.config import is_starwam_config, prepare_starwam_host_config
from starVLA.training.gradient_probe import write_gradient_probe
from starVLA.training.loss_utils import resolve_model_loss
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = logging.getLogger(__name__)


def _get_state_dict_for_save(accelerator: Accelerator, model: torch.nn.Module):
    """Materialize full weights only on ranks required by the backend."""
    deepspeed_config = getattr(accelerator, "deepspeed_config", None) or {}
    zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage")
    requires_all_processes = accelerator.distributed_type == DistributedType.FSDP or (
        accelerator.distributed_type == DistributedType.DEEPSPEED and str(zero_stage) == "3"
    )
    if not accelerator.is_main_process and not requires_all_processes:
        return None
    return accelerator.get_state_dict(model)


def create_accelerator(cfg) -> Accelerator:
    """Create Accelerate after config resolution so accumulation/precision are honored."""
    mixed_precision = str(cfg.trainer.get("mixed_precision", "no")).lower()
    gradient_accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    is_starwam = is_starwam_config(cfg)
    plugin_kwargs = {}
    if is_starwam:
        plugin_kwargs = {
            "zero_stage": 2,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": cfg.trainer.get("gradient_clipping", None),
        }
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(**plugin_kwargs),
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        # WAM schedules are expressed in global optimizer steps. We gate
        # scheduler.step() ourselves, so Accelerate must not multiply it by
        # world size when wrapping the scheduler.
        step_scheduler_with_optimizer=not is_starwam,
    )
    accelerator.print(accelerator.state)
    return accelerator


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir, model=None) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
        model=model,
    )

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    if is_starwam_config(cfg):
        optimizer_name = str(cfg.trainer.optimizer.get("name", "AdamW")).lower()
        if optimizer_name not in {"adamw", "torch_adamw"}:
            raise NotImplementedError(f"StarWAM integration currently supports AdamW only, got {optimizer_name!r}.")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("StarWAM freeze strategy produced zero trainable parameters.")
        param_groups = [
            {
                "params": trainable,
                "lr": cfg.trainer.learning_rate.base,
                "name": "starwam_trainable",
            }
        ]
    else:
        param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=bool(cfg.trainer.optimizer.get("fused", True)),
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    if is_starwam_config(cfg):
        lr_scheduler = _build_starwam_scheduler(optimizer, cfg)
    else:
        sched_kwargs = {key: value for key, value in cfg.trainer.scheduler_specific_kwargs.items()}
        lr_scheduler = get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=sched_kwargs,
        )

    return optimizer, lr_scheduler


def _build_starwam_scheduler(optimizer, cfg):
    """Mirror StarWAM's warmup + cosine scheduler exactly."""
    max_steps = max(int(cfg.trainer.max_train_steps), 1)
    warmup_steps = max(int(cfg.trainer.num_warmup_steps), 0)
    scheduler_type = str(cfg.trainer.lr_scheduler_type).strip().lower()
    if scheduler_type in {"cosine", "cosine_with_min_lr"}:
        warmup_steps = min(warmup_steps, max_steps - 1)
        cfg.trainer.num_warmup_steps = warmup_steps
        min_lr = float(cfg.trainer.scheduler_specific_kwargs.get("min_lr", 0.0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(max_steps - warmup_steps, 1),
            eta_min=min_lr,
        )
        if warmup_steps == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        scheduler_specific_kwargs={key: value for key, value in cfg.trainer.scheduler_specific_kwargs.items()},
    )


def resolve_training_schedule(cfg, dataloader: DataLoader, accelerator: Accelerator) -> None:
    """Materialize epoch-based StarWAM schedules before optimizer creation."""
    if not is_starwam_config(cfg):
        return
    if cfg.trainer.max_train_steps is None:
        dataset_size = len(dataloader.dataset)
        per_device_batch = int(cfg.datasets.vla_data.per_device_batch_size)
        world_size = max(int(accelerator.num_processes), 1)
        grad_accum = max(int(accelerator.gradient_accumulation_steps), 1)
        micro_steps_per_epoch = max(ceil(dataset_size / (per_device_batch * world_size)), 1)
        cfg.trainer.max_train_steps = max(
            ceil(micro_steps_per_epoch / grad_accum) * int(cfg.trainer.num_epochs),
            1,
        )
    if cfg.trainer.num_warmup_steps is None:
        cfg.trainer.num_warmup_steps = int(
            int(cfg.trainer.max_train_steps) * float(cfg.trainer.get("warmup_ratio", 0.0))
        )


def _infinite_dataloader(dataloader):
    """Repeat complete dataloader epochs without bypassing shard finalization.

    ``yield from`` lets Accelerate's ``DataLoaderShard`` finish its end-of-epoch
    synchronization before a fresh iterator is created. Catching
    ``StopIteration`` outside the shard can skip that synchronization and hang
    a distributed StarWAM job at the epoch boundary.
    """
    while True:
        yield from dataloader


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._gradient_probe_written = False

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        if is_starwam_config(self.config):
            (
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.lr_scheduler,
            ) = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.lr_scheduler,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        wandb_disabled = os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if not bool(self.config.trainer.get("wandb_enabled", True)) or wandb_disabled:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # WAM checkpoints need the entire embedded recipe to be self-contained.
        if is_starwam_config(self.config):
            OmegaConf.save(full_cfg, output_dir / "config.yaml", resolve=True)
            logger.info(f"📝 Self-contained config saved at {output_dir / 'config.yaml'}")
        # Other frameworks preserve the compact accessed-only snapshot.
        elif isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        state_dict = _get_state_dict_for_save(self.accelerator, self.model)
        if self.accelerator.is_main_process:
            assert state_dict is not None
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig) and not is_starwam_config(self.config):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        if is_starwam_config(self.config):
            self.vla_iter = _infinite_dataloader(self.vla_train_dataloader)
        else:
            self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            optimizer_stepped = step_metrics.pop("_optimizer_step")

            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if (
                optimizer_stepped
                and self.config.trainer.eval_interval > 0
                and self.completed_steps % self.config.trainer.eval_interval == 0
            ):
                eval_batch = batch_vla if is_starwam_config(self.config) else None
                step_metrics = self.eval_action_model(step_metrics, examples=eval_batch)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if optimizer_stepped:
                self._log_metrics(step_metrics)

            if (
                optimizer_stepped
                and self.config.trainer.save_interval > 0
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict | None = None, examples=None) -> dict:
        """Run simple action-eval on current batch and attach score to metrics."""
        if examples is None:
            examples = self._get_next_batch()
        unwrapped = self.accelerator.unwrap_model(self.model)
        framework_metrics = unwrapped.evaluate_batch(
            examples,
            num_inference_steps=int(self.config.trainer.get("eval_num_inference_steps", 4)),
            action_num_inference_steps=int(self.config.trainer.get("eval_action_num_inference_steps", 4)),
            compute_video=bool(self.config.trainer.get("eval_compute_video_psnr", False)),
            max_samples=int(self.config.trainer.get("eval_max_samples", 4)),
            seed=int(getattr(self.config, "seed", 42)),
        )
        if framework_metrics is not None:
            if self.accelerator.is_main_process:
                step_metrics.update(framework_metrics)
            del examples
            return step_metrics

        actions = [example["action"] for example in examples]
        output_dict = unwrapped.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            if is_starwam_config(self.config):
                trainer = self.config.trainer
                self.accelerator.print(
                    "StarWAM effective values: "
                    f"batch={self.config.datasets.vla_data.per_device_batch_size} "
                    f"accumulation={self.accelerator.gradient_accumulation_steps} "
                    f"lr={trainer.learning_rate.base} "
                    f"weight_decay={trainer.optimizer.weight_decay} "
                    f"epochs={trainer.num_epochs} max_steps={trainer.max_train_steps} "
                    f"warmup={trainer.num_warmup_steps} precision={trainer.mixed_precision} "
                    f"save_interval={trainer.save_interval} scheduler={trainer.lr_scheduler_type} "
                    f"min_lr={trainer.scheduler_specific_kwargs.get('min_lr', None)} "
                    f"fused_adamw={trainer.optimizer.fused}"
                )

    def _maybe_write_starwam_gradient_probe(self) -> None:
        if self._gradient_probe_written or not is_starwam_config(self.config):
            return
        trainer_cfg = self.config.trainer
        if not bool(trainer_cfg.get("gradient_probe_enabled", False)):
            return
        target_step = int(trainer_cfg.get("gradient_probe_step", 1))
        if target_step <= 0:
            raise ValueError("trainer.gradient_probe_step must be positive")
        if self.completed_steps + 1 != target_step:
            return

        self._gradient_probe_written = True
        unwrapped = self.accelerator.unwrap_model(self.model)
        zero_partitioned = any(
            hasattr(parameter, "_hp_mapping") or hasattr(parameter, "ds_id")
            for parameter in unwrapped.parameters()
        )
        gradient_getter = None
        if zero_partitioned:
            # ZeRO partitions gradients during backward and clears ``param.grad``.
            # Its public helper reconstructs one parameter at a time and requires
            # every data-parallel rank to participate in the collective.
            from deepspeed.utils import safe_get_full_grad

            gradient_getter = safe_get_full_grad
        output_path = Path(self.config.output_dir) / "gradient_probe.json"
        write_gradient_probe(
            unwrapped,
            output_path,
            model_family=str(getattr(unwrapped, "model_family", "unknown")),
            optimizer_step=target_step,
            chunk_size=int(trainer_cfg.get("gradient_probe_chunk_size", 1_048_576)),
            gradient_getter=gradient_getter,
            write_output=self.accelerator.is_main_process,
        )
        if self.accelerator.is_main_process:
            logger.info("StarWAM gradient probe saved at %s", output_path)

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        optimizer_stepped = False
        is_deepspeed_engine = all(
            hasattr(self.model, attribute)
            for attribute in ("backward", "step", "is_gradient_accumulation_boundary")
        )
        if is_deepspeed_engine:
            with self.accelerator.autocast():
                output_dict = self.model(batch_vla)
                total_loss, loss_metrics = resolve_model_loss(output_dict)

            # Accelerate's DeepSpeed backward wrapper calls engine.step()
            # immediately. Drive the engine directly so the gradient probe is
            # captured before DeepSpeed clips, updates, and clears gradients.
            self.model.backward(total_loss)
            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            if optimizer_stepped:
                self._maybe_write_starwam_gradient_probe()
            self.model.step()
            if optimizer_stepped:
                self.lr_scheduler.step()
        else:
            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    output_dict = self.model(batch_vla)
                    total_loss, loss_metrics = resolve_model_loss(output_dict)

                self.accelerator.backward(total_loss)

                optimizer_stepped = bool(self.accelerator.sync_gradients)
                if optimizer_stepped:
                    self._maybe_write_starwam_gradient_probe()
                    if self.config.trainer.gradient_clipping is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                    # Step, schedule, and clear gradients only at a real optimizer
                    # boundary. This is explicit rather than relying on an
                    # AcceleratedOptimizer to silently no-op on micro-batches.
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

        metrics = {f"loss/{key}": value for key, value in loss_metrics.items()}
        if "action_loss" in loss_metrics:
            metrics["action_dit_loss"] = loss_metrics["action_loss"]
        metrics["_optimizer_step"] = optimizer_stepped
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        state_dict = _get_state_dict_for_save(self.accelerator, self.model)
        if self.accelerator.is_main_process:
            assert state_dict is not None
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = prepare_starwam_host_config(cfg, config_path=cfg.get("config_yaml", None))
    accelerator = create_accelerator(cfg)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    if is_starwam_config(cfg):
        prepare_starwam_data_artifacts(cfg)
        mixed_precision = str(cfg.trainer.get("mixed_precision", "bf16")).lower()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision, torch.float32)
        vla = build_framework(cfg, device=accelerator.device, dtype=dtype)
    else:
        vla = build_framework(cfg)
    vla.configure_training(cfg)
    vla_train_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        model=vla,
    )
    resolve_training_schedule(cfg, vla_train_dataloader, accelerator)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
