from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WAM.StarWAM import StarWAMFramework
from starVLA.model.framework.WAM.runtime import compose_camera_views


TEST_ROOT = Path("/home/amax/data0/SJ/test-runs/starwam-integration/d187bb1_20260722_140418")
PRIOR_ROOT = Path("/home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824")
CONFIG_PATH = PRIOR_ROOT / "train/SW-L-MW_20260722_063933/config.full.yaml"
GOLDEN_PATH = (
    Path("/home/amax/SJ/projects/Embodied-World/starVLA/examples/modelExtensions/StarWAM/test_results")
    / "3f072c4_20260722_062824/artifacts/SW-L-MW/golden_input_libero.npz"
)
NATIVE_CHECKPOINT = Path(
    "/home/amax/data0/SJ/models/starwam_ckpts/starwam-libero/mot/starwam_wan225b_mot.pt"
)
OUTPUT_DIR = TEST_ROOT / "artifacts/parity/SW-L-MW"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
SEED = 42
INFERENCE_STEPS = 2
ACTION_STEPS = 2
ACTION_ATOL = 1.0e-4
ACTION_RTOL = 1.0e-3


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    difference = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    relative = difference / expected.abs().clamp_min(1.0e-12)
    failed = difference > tolerance
    return {
        "shape_equal": list(actual.shape) == list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_diff": float(difference.max().item()),
        "max_rel_diff": float(relative.max().item()),
        "failed_elements": int(torch.count_nonzero(failed).item()),
        "total_elements": int(difference.numel()),
        "atol": atol,
        "rtol": rtol,
    }


def _load_native_rollout_module():
    path = "/home/amax/SJ/projects/Embodied-World/StarWAM/examples/libero/rollout.py"
    spec = importlib.util.spec_from_file_location("native_starwam_libero_rollout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_training_tensor(client_images: list[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    from starwam.data.lerobot import _resize_frames

    frames = []
    for image in client_images:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        frames.append(_resize_frames(tensor, size).float()[0] / 255.0)
    return (torch.cat(frames, dim=-1) * 2.0 - 1.0).unsqueeze(0).contiguous()


def _native_phase(
    config: Any,
    canonical: torch.Tensor,
    state: np.ndarray,
    instruction: str,
) -> dict[str, Any]:
    from starwam.eval.policy import StarwamPolicy, _denormalize_action, _normalize_vector

    policy = StarwamPolicy(
        str(config.framework.starwam.resolved_recipe),
        str(NATIVE_CHECKPOINT),
        overrides=list(config.framework.starwam.overrides),
        device=DEVICE,
        num_inference_steps=INFERENCE_STEPS,
        seed=SEED,
    )
    context, context_mask = policy._encode_context(instruction)
    proprio_dim = int(policy.config.framework.proprio_dim)
    proprio = torch.as_tensor(state[:proprio_dim], dtype=torch.float32).view(1, proprio_dim)
    if policy._state_stats is not None:
        proprio = _normalize_vector(proprio, policy.config.data.state_norm_mode, policy._state_stats)
    proprio = proprio.to(device=policy.device, dtype=policy.dtype)
    normalized = policy.model.infer_action(
        input_image=canonical.to(device=policy.device, dtype=policy.dtype),
        context=context,
        context_mask=context_mask,
        action_horizon=policy.action_horizon,
        num_inference_steps=INFERENCE_STEPS,
        action_num_inference_steps=ACTION_STEPS,
        seed=SEED,
        proprio=proprio,
        num_video_frames=policy._sampled_video_frame_count(),
    ).detach().float().cpu()
    final = normalized
    if policy._action_stats is not None:
        final = _denormalize_action(final, policy.config.data.action_norm_mode, policy._action_stats)
    result = {
        "context": context.detach().cpu(),
        "context_mask": context_mask.detach().cpu().bool(),
        "proprio": proprio.detach().float().cpu(),
        "normalized": normalized,
        "final": final.detach().float().cpu(),
        "action_horizon": policy.action_horizon,
        "num_video_frames": policy._sampled_video_frame_count(),
    }
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _starvla_phase(
    config: Any,
    client_images: list[np.ndarray],
    state: np.ndarray,
    instruction: str,
) -> dict[str, Any]:
    wrapper_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    wrapper_config.framework.starwam.init_checkpoint = str(NATIVE_CHECKPOINT)
    wrapper_config.framework.starwam.strict_init = True
    wrapper = StarWAMFramework(wrapper_config, device=DEVICE, dtype=DTYPE).to(DEVICE).eval()
    device, dtype = wrapper._model_device_dtype()
    context, context_mask = wrapper._text_conditioner.encode([instruction], device=device, dtype=dtype)
    proprio = wrapper._normalized_state(state, device, dtype)
    prediction = wrapper.predict_action(
        [{"image": client_images, "lang": instruction, "state": state}],
        num_inference_steps=INFERENCE_STEPS,
        action_num_inference_steps=ACTION_STEPS,
        seed=SEED,
    )["normalized_actions"]
    final = wrapper.unnormalize_actions(prediction)
    result = {
        "context": context.detach().cpu(),
        "context_mask": context_mask.detach().cpu().bool(),
        "proprio": None if proprio is None else proprio.detach().float().cpu(),
        "normalized": torch.from_numpy(prediction),
        "final": torch.from_numpy(final),
        "metadata": wrapper.get_policy_metadata(),
        "action_horizon": wrapper.action_horizon,
        "num_video_frames": wrapper._sampled_video_frames(),
    }
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(CONFIG_PATH)
    resolved = config.framework.starwam.resolved
    golden = np.load(GOLDEN_PATH)
    raw_primary = np.asarray(golden["primary"], dtype=np.uint8)
    raw_wrist = np.asarray(golden["wrist"], dtype=np.uint8)
    state = np.asarray(golden["state"], dtype=np.float32)
    instruction = str(golden["instruction"].item())
    client_images = [
        np.ascontiguousarray(raw_primary[::-1, ::-1]),
        np.ascontiguousarray(raw_wrist[::-1, ::-1]),
    ]

    starvla_pixels = compose_camera_views(client_images, resolved.data)
    training_pixels = _canonical_training_tensor(
        client_images,
        tuple(int(value) for value in resolved.data.video_size),
    )
    canonical_equal = bool(torch.equal(starvla_pixels, training_pixels))
    canonical_difference = (starvla_pixels - training_pixels).abs()

    native_rollout = _load_native_rollout_module()
    pil_pixels, _ = native_rollout._obs_to_image(
        {
            "agentview_image": raw_primary,
            "robot0_eye_in_hand_image": raw_wrist,
        },
        resolved,
    )
    pil_difference = (starvla_pixels - pil_pixels).abs()

    native = _native_phase(config, starvla_pixels, state, instruction)
    starvla = _starvla_phase(config, client_images, state, instruction)

    text_equal = bool(
        torch.equal(native["context"], starvla["context"])
        and torch.equal(native["context_mask"], starvla["context_mask"])
    )
    state_comparison = _comparison(starvla["proprio"], native["proprio"], atol=1.0e-6, rtol=1.0e-6)
    normalized_comparison = _comparison(
        starvla["normalized"], native["normalized"], atol=ACTION_ATOL, rtol=ACTION_RTOL
    )
    final_comparison = _comparison(starvla["final"], native["final"], atol=ACTION_ATOL, rtol=ACTION_RTOL)

    gated_pass = all(
        [
            canonical_equal,
            text_equal,
            state_comparison["allclose"],
            normalized_comparison["allclose"],
            final_comparison["allclose"],
            native["action_horizon"] == starvla["action_horizon"] == 32,
            native["num_video_frames"] == starvla["num_video_frames"] == 9,
        ]
    )
    result = {
        "status": "PASS" if gated_pass else "FAIL",
        "case": "SW-L-MW",
        "canonical_contract": "StarVLA compose_camera_views == StarWAM training _resize_frames",
        "checkpoint": str(NATIVE_CHECKPOINT),
        "checkpoint_sha256": _sha256_file(NATIVE_CHECKPOINT),
        "golden_input": str(GOLDEN_PATH),
        "golden_input_sha256": _sha256_file(GOLDEN_PATH),
        "golden_array_sha256": {
            "primary": _sha256_array(raw_primary),
            "wrist": _sha256_array(raw_wrist),
            "state": _sha256_array(state),
        },
        "instruction": instruction,
        "seed": SEED,
        "num_inference_steps": INFERENCE_STEPS,
        "action_num_inference_steps": ACTION_STEPS,
        "pixels": {
            "status": "PASS" if canonical_equal else "FAIL",
            "shape": list(starvla_pixels.shape),
            "torch_equal": canonical_equal,
            "max_abs_diff": float(canonical_difference.max().item()),
            "failed_elements": int(torch.count_nonzero(canonical_difference).item()),
        },
        "pil_rollout_diagnostic": {
            "status": "DIAGNOSTIC_ONLY",
            "torch_equal": bool(torch.equal(starvla_pixels, pil_pixels)),
            "max_abs_diff": float(pil_difference.max().item()),
            "mean_abs_diff": float(pil_difference.mean().item()),
            "failed_elements": int(torch.count_nonzero(pil_difference).item()),
        },
        "text": {
            "status": "PASS" if text_equal else "FAIL",
            "context_equal": bool(torch.equal(native["context"], starvla["context"])),
            "mask_equal": bool(torch.equal(native["context_mask"], starvla["context_mask"])),
            "context_shape": list(native["context"].shape),
        },
        "state": {"status": "PASS" if state_comparison["allclose"] else "FAIL", **state_comparison},
        "normalized_action": {
            "status": "PASS" if normalized_comparison["allclose"] else "FAIL",
            **normalized_comparison,
        },
        "final_action": {
            "status": "PASS" if final_comparison["allclose"] else "FAIL",
            **final_comparison,
        },
        "metadata": starvla["metadata"],
        "action_horizon": starvla["action_horizon"],
        "num_video_frames": starvla["num_video_frames"],
    }
    torch.save(
        {
            "native_normalized": native["normalized"],
            "starvla_normalized": starvla["normalized"],
            "native_final": native["final"],
            "starvla_final": starvla["final"],
        },
        OUTPUT_DIR / "action_outputs.pt",
    )
    (OUTPUT_DIR / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if gated_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
