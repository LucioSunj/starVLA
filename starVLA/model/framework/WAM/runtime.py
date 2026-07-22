"""Observation, text, and normalization adapters for StarWAM policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _to_chw_float(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(image))
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Camera image must be HWC or CHW, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer camera channel axis from shape={tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and tensor.min() < 0:
        tensor = (tensor + 1.0) * 127.5
    elif not tensor.numel() or tensor.max() <= 1.0:
        tensor = tensor * 255.0
    return tensor.clamp(0.0, 255.0).contiguous()


def _resize(frame: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(frame.shape[-2:]) == tuple(size):
        return frame
    return F.interpolate(frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]


def compose_camera_views(images: Any, data_config: Any) -> torch.Tensor:
    """Build StarWAM's training-contract camera tensor ``[1, 3, H, W]`` in ``[-1, 1]``."""
    views = list(images) if isinstance(images, (list, tuple)) else [images]
    layout = str(getattr(data_config, "concat_multi_camera", "horizontal"))
    expected = list(getattr(data_config, "video_keys", None) or [getattr(data_config, "video_key", "image")])
    if len(views) != len(expected):
        raise ValueError(
            f"StarWAM camera count mismatch: recipe expects {len(expected)} views in order {expected}, got {len(views)}."
        )
    frames = [_to_chw_float(image) for image in views]
    if layout == "robotwin":
        if len(frames) != 3:
            raise ValueError("RoboTwin camera layout requires [head, left_wrist, right_wrist].")
        top = _resize(frames[0], (256, 320))
        left = _resize(frames[1], (128, 160))
        right = _resize(frames[2], (128, 160))
        frame = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
    else:
        size = tuple(int(v) for v in data_config.video_size)
        frames = [_resize(frame, size) for frame in frames]
        if len(frames) == 1:
            frame = frames[0]
        elif layout == "horizontal":
            frame = torch.cat(frames, dim=-1)
        elif layout == "vertical":
            frame = torch.cat(frames, dim=-2)
        else:
            raise ValueError(f"Unsupported StarWAM camera layout: {layout!r}")
    return (frame / 255.0 * 2.0 - 1.0).unsqueeze(0).contiguous()


def normalize_vector(values: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    dim = int(values.shape[-1])
    if mode == "zscore":
        mean = stats["mean"][:dim].to(values.device, values.dtype)
        std = stats["std"][:dim].to(values.device, values.dtype).clamp_min(1.0e-6)
        return ((values - mean) / std).clamp(-5.0, 5.0)
    if mode == "minmax":
        value_min = stats["min"][:dim].to(values.device, values.dtype)
        value_max = stats["max"][:dim].to(values.device, values.dtype)
        return (2.0 * (values - value_min) / (value_max - value_min).clamp_min(1.0e-6) - 1.0).clamp(-5.0, 5.0)
    raise ValueError(f"Unsupported StarWAM normalization mode: {mode!r}")


def denormalize_actions(actions: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    dim = int(actions.shape[-1])
    if mode == "zscore":
        mean = stats["mean"][:dim].to(actions.device, actions.dtype)
        std = stats["std"][:dim].to(actions.device, actions.dtype).clamp_min(1.0e-6)
        return actions * std + mean
    if mode == "minmax":
        value_min = stats["min"][:dim].to(actions.device, actions.dtype)
        value_max = stats["max"][:dim].to(actions.device, actions.dtype)
        return (actions.clamp(-1.0, 1.0) + 1.0) * 0.5 * (value_max - value_min).clamp_min(1.0e-6) + value_min
    raise ValueError(f"Unsupported StarWAM normalization mode: {mode!r}")


def tensor_stats(raw: dict[str, Any] | None) -> dict[str, torch.Tensor] | None:
    if not raw:
        return None
    return {
        key: value.detach().cpu().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value).float()
        for key, value in raw.items()
        if key != "mask"
    }


class TextConditioner:
    """Lazily load text caches or a StarWAM text encoder for policy inference."""

    def __init__(self, starwam_config: Any, model: torch.nn.Module) -> None:
        self.config = starwam_config
        self.model = model
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._encoder = None

    def encode(
        self,
        instructions: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = []
        masks = []
        for instruction in instructions:
            context, mask = self._encode_one(str(instruction))
            contexts.append(context)
            masks.append(mask)
        return torch.cat(contexts, dim=0).to(device=device, dtype=dtype), torch.cat(masks, dim=0).to(device=device)

    def _encode_one(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if instruction in self._cache:
            return self._cache[instruction]

        from starwam.data.lerobot import (
            DEFAULT_TEXT_CACHE_ENCODER_ID,
            DEFAULT_TEXT_PROMPT,
            format_text_prompt,
            load_text_cache,
            save_text_cache,
            text_cache_path,
        )

        data_cfg = self.config.data
        text_len = int(getattr(data_cfg, "text_len", 128))
        template = getattr(data_cfg, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
        encoder_id = getattr(data_cfg, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
        cache_dir = getattr(data_cfg, "text_embedding_cache_dir", None)
        text_dim = int(getattr(getattr(self.model.backbone, "info", None), "text_dim", 4096))
        cache_path = text_cache_path(cache_dir, instruction, text_len, template, encoder_id) if cache_dir else None

        if cache_path is not None and Path(cache_path).is_file():
            context, mask = load_text_cache(cache_path, text_len, text_dim)
            context, mask = context.unsqueeze(0), mask.unsqueeze(0)
        else:
            if self._encoder is None:
                from starwam.backbone import build_text_encoder

                parameter = next(self.model.parameters())
                encoder_dtype = torch.bfloat16 if parameter.device.type == "cuda" else torch.float32
                self._encoder = build_text_encoder(
                    self.config.backbone,
                    text_len=text_len,
                    device=str(parameter.device),
                    dtype=encoder_dtype,
                )
            prompt = format_text_prompt(instruction, template)
            with torch.no_grad():
                context, mask = self._encoder.encode([prompt])
            if cache_path is not None:
                save_text_cache(cache_path, context[0], mask[0], prompt, instruction)

        cached = (context.detach().cpu(), mask.detach().cpu().bool())
        self._cache[instruction] = cached
        return cached
