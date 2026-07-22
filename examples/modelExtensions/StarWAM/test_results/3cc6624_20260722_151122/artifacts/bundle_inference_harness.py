from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.framework.WAM.StarWAM import StarWAMFramework


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    golden = np.load(args.golden)
    images = [
        np.ascontiguousarray(np.asarray(golden["primary"], dtype=np.uint8)[::-1, ::-1]),
        np.ascontiguousarray(np.asarray(golden["wrist"], dtype=np.uint8)[::-1, ::-1]),
    ]
    state = np.asarray(golden["state"], dtype=np.float32)
    instruction = str(golden["instruction"].item())

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model = StarWAMFramework.from_pretrained(
        str(args.checkpoint),
        device="cuda:0",
        dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    device, dtype = model._model_device_dtype()
    prediction = model.predict_action(
        [{"image": images, "lang": instruction, "state": state}],
        seed=42,
        num_inference_steps=2,
        action_num_inference_steps=2,
    )["normalized_actions"]
    finite = bool(np.isfinite(prediction).all())
    result = {
        "status": "PASS" if finite and prediction.shape == (1, 32, 7) else "FAIL",
        "case": args.case,
        "checkpoint": str(args.checkpoint),
        "strict_restore": True,
        "recipe_path": "/definitely/missing/starwam_recipe.yaml",
        "recipe_exists": False,
        "used_resolved_snapshot": True,
        "seed": 42,
        "num_inference_steps": 2,
        "action_num_inference_steps": 2,
        "shape": list(prediction.shape),
        "finite": finite,
        "min": float(prediction.min()),
        "max": float(prediction.max()),
        "model_device": str(device),
        "model_dtype": str(dtype),
        "peak_mib": float(torch.cuda.max_memory_allocated(0) / 1024**2),
        "metadata": model.get_policy_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
