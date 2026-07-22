from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WAM.runtime import compose_camera_views


TEST_ROOT = Path("/home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824")
DATASET_ROOT = Path(
    "/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/"
    "libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
)
CONFIG_PATH = TEST_ROOT / "train/SW-L-MW_20260722_063933/config.full.yaml"
OUTPUT_DIR = TEST_ROOT / "artifacts/parity/SW-L-MW"
EPISODE = 15
TIMESTEP = 0


def decode_frame(path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
    raise IndexError(f"frame {frame_index} not found in {path}")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_name = f"episode_{EPISODE:06d}.mp4"
    primary_path = DATASET_ROOT / "videos/chunk-000/observation.images.image" / video_name
    wrist_path = DATASET_ROOT / "videos/chunk-000/observation.images.wrist_image" / video_name
    parquet_path = DATASET_ROOT / "data/chunk-000" / f"episode_{EPISODE:06d}.parquet"

    # Dataset videos hold the camera orientation seen by training. Reversing both axes
    # reconstructs the simulator-facing raw array; both native and StarVLA client paths
    # then apply the same documented 180-degree LIBERO rotation before model resize.
    training_primary = decode_frame(primary_path, TIMESTEP)
    training_wrist = decode_frame(wrist_path, TIMESTEP)
    raw_primary = np.ascontiguousarray(training_primary[::-1, ::-1])
    raw_wrist = np.ascontiguousarray(training_wrist[::-1, ::-1])

    row = pq.read_table(parquet_path, columns=["task_index", "observation.state"]).slice(TIMESTEP, 1).to_pydict()
    task_index = int(row["task_index"][0])
    state = np.asarray(row["observation.state"][0], dtype=np.float32)
    tasks = {
        int(record["task_index"]): str(record["task"])
        for record in (
            json.loads(line) for line in (DATASET_ROOT / "meta/tasks.jsonl").read_text().splitlines() if line
        )
    }
    instruction = tasks[task_index]

    spec = importlib.util.spec_from_file_location(
        "native_starwam_libero_rollout",
        "/home/amax/SJ/projects/Embodied-World/StarWAM/examples/libero/rollout.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to import native StarWAM LIBERO adapter")
    native_rollout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_rollout)

    resolved = OmegaConf.load(CONFIG_PATH).framework.starwam.resolved
    native_image, _ = native_rollout._obs_to_image(
        {
            "agentview_image": raw_primary,
            "robot0_eye_in_hand_image": raw_wrist,
        },
        resolved,
    )
    client_primary = np.ascontiguousarray(raw_primary[::-1, ::-1])
    client_wrist = np.ascontiguousarray(raw_wrist[::-1, ::-1])
    starvla_image = compose_camera_views([client_primary, client_wrist], resolved.data)

    difference = (native_image - starvla_image).abs()
    pixel_equal = bool(torch.equal(native_image, starvla_image))
    np.savez_compressed(
        OUTPUT_DIR / "golden_input_libero.npz",
        primary=raw_primary,
        wrist=raw_wrist,
        state=state,
        instruction=np.asarray(instruction),
    )
    metadata = {
        "benchmark": "LIBERO",
        "dataset_root": str(DATASET_ROOT),
        "episode": EPISODE,
        "task_index": task_index,
        "instruction": instruction,
        "environment_seed": None,
        "environment_seed_status": "not present in archived LeRobot episode metadata",
        "observation_timestep": TIMESTEP,
        "raw_primary_shape": list(raw_primary.shape),
        "raw_wrist_shape": list(raw_wrist.shape),
        "raw_primary_dtype": str(raw_primary.dtype),
        "raw_wrist_dtype": str(raw_wrist.dtype),
        "state_dim": int(state.size),
        "raw_primary_sha256": sha256_array(raw_primary),
        "raw_wrist_sha256": sha256_array(raw_wrist),
        "state_sha256": sha256_array(state),
    }
    (OUTPUT_DIR / "golden_input_libero.json").write_text(json.dumps(metadata, indent=2) + "\n")

    result = {
        "case": "SW-L-MW",
        "checkpoint": "/home/amax/data0/SJ/models/starwam_ckpts/starwam-libero/mot/starwam_wan225b_mot.pt",
        "config": str(CONFIG_PATH),
        "golden_input": str(OUTPUT_DIR / "golden_input_libero.npz"),
        "status": "FAIL" if not pixel_equal else "PASS",
        "first_failed_layer": None if pixel_equal else "pixels",
        "pixels": {
            "status": "PASS" if pixel_equal else "FAIL",
            "native_shape": list(native_image.shape),
            "starvla_shape": list(starvla_image.shape),
            "native_dtype": str(native_image.dtype),
            "starvla_dtype": str(starvla_image.dtype),
            "torch_equal": pixel_equal,
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "failed_elements": int(torch.count_nonzero(difference).item()),
            "total_elements": int(difference.numel()),
            "native_path": "StarWAM examples/libero/rollout.py::_obs_to_image (PIL bilinear)",
            "starvla_path": "StarVLA compose_camera_views (torch bilinear, training-matched)",
        },
        "text": {"status": "NOT_RUN_DUE_PIXEL_FAILURE"},
        "state": {"status": "NOT_RUN_DUE_PIXEL_FAILURE"},
        "normalized_action": {"status": "NOT_RUN_DUE_PIXEL_FAILURE"},
        "final_action": {"status": "NOT_RUN_DUE_PIXEL_FAILURE"},
        "service_action": {"status": "NOT_RUN_DUE_PIXEL_FAILURE"},
    }
    (OUTPUT_DIR / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if pixel_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
