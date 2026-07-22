# StarWAM integration gate decision — `ad6b412`

## Scope and frozen inputs

- StarVLA: `ad6b4122a852d9c6c178d721461d420762efaec1`, branch `codex/starwam-integration`.
- StarWAM: `a7b05c8da8f8c88bf4888aa737b938af3b67ab48`, detached and clean.
- External run root: `/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916`.
- Full environment: Python 3.11.15, PyTorch 2.6.0+cu124, CUDA 12.4, NCCL 2.21.5, Accelerate 1.5.2, DeepSpeed 0.16.9, Transformers 4.57.0, OpenCV 4.11.0, Matplotlib 3.11.1.
- `starwam` imported from `/home/amax/SJ/projects/Embodied-World/StarWAM/starwam/__init__.py`.
- The full environment manifests were reused from the immediately preceding `3f072c4` run because the environment was unchanged; revisions and asset availability were revalidated for this run.
- No StarVLA or StarWAM runtime source was changed during this execution. Large checkpoints, caches, raw datasets, and videos remain outside the repository.

## Gate decision

| Gate | Decision | Evidence / reason |
|---|---|---|
| G0 | **PASS** | A1 checks pass; targeted integration tests 41 passed; full `tests/` 74 passed; no failures, errors, or skips; no-StarWAM lazy import test passes. |
| G1 | **BLOCKED** | Four Wan cases complete two real optimizer steps on two GPUs. Both mandatory Cosmos cases are blocked by unavailable gated Cosmos assets (HTTP 401), so the six-case matrix is incomplete. |
| G2 | **FAIL** | All four generated Wan `steps_2` checkpoints strict-load with 0 missing and 0 unexpected keys, including from relocated bundles. However, restored SW-L-SW fails actual inference with a CPU/CUDA device mismatch. |
| G3 | **PASS** | With the approved canonical StarVLA/StarWAM training tensor-resize contract, real native SW-L-MW parity is exact for pixels, state, normalized action, and final action. Scope is one available native MoT/LIBERO checkpoint. |
| G4 | **FAIL** | Real RoboTwin policy-server checks pass, but the real LIBERO runner fails before rollout because it passes `pathlib.Path` to a LIBERO wrapper expecting a string. RoboTwin rollout remains blocked by an incomplete simulator checkout/assets. |

**Recommendation: DO NOT MERGE.** G2 and G4 contain reproducible code failures, and G1 cannot be completed without the two Cosmos assets. G3 confirms that the selected PyTorch/torchvision-style training resize is not the parity blocker.

## Assets

| Asset | Result |
|---|---|
| `WAN22_ROOT` | `/home/amax/data0/SJ/models/Wan2.2-TI2V-5B` (validated) |
| `WAN_ACTION_DIT_INIT` | `/home/amax/data0/SJ/preprocessed/starwam_action_dit_init_wan22.pt` (validated) |
| native StarWAM checkpoint | `/home/amax/data0/SJ/models/starwam_ckpts/starwam-libero/mot/starwam_wan225b_mot.pt`, SHA-256 `d24edea01579880327cfd9dc84d24adab82e420dca9652e614ad697bc8cc5378` |
| `COSMOS_ROOT` | **BLOCKED** — gated Hugging Face repository returned HTTP 401 |
| `COSMOS_ACTION_DIT_INIT` | **BLOCKED** — requires unavailable Cosmos backbone |
| four LIBERO datasets | Present at the exact paths recorded in `dataset_directories_revalidated.txt` |
| `ROBOTWIN_DATA` | Present at the exact path recorded in `dataset_directories_revalidated.txt` |
| RoboTwin simulator | **BLOCKED** — the requested checkout is empty; the discovered FastWAM copy lacks `assets/`, `assets/embodiments`, `assets/objects`, and `task_config/`, and its runtime is missing `pytorch_kinematics` |

No asset path was guessed. The `assets.txt`, `asset_revalidation.txt`, `wan-action-init-validation.json`, and `wan22-integrity.txt` files contain the exact evidence.

## G0 details

- A1: ruff, compileall, TOML parsing, and diff checks pass (`logs/g0_a1.log`).
- `tests/test_starwam_integration.py`: **41 passed, 0 failed, 0 errors, 0 skipped** (`logs/g0_starwam_integration.log`). No StarWAM, Transformers, trainer, OpenCV, or Matplotlib absence skip occurred.
- Full `tests/`: **74 passed, 0 failed, 0 errors, 0 skipped** (`logs/g0_all_tests.log`).
- Separate no-StarWAM environment: legacy ACT and lightweight framework imports pass; requesting WAM raises the expected actionable `.[wam]` error (`logs/g0_no_starwam.log`).
- Two-rank `_infinite_dataloader` harness traversed two full epochs: 8 steps on each rank, a collective on every step, no hang, and `legacy_reset_calls=0`.
- Two-rank ZeRO-2 harness observed boundary pattern `[false, true]`; no optimizer/scheduler/clip/zero-grad action occurred on the intermediate micro-step. The real runs advanced the scheduler only at the two optimizer boundaries.

## G1 case matrix

All completed cases used bf16 ZeRO-2 on two GPUs and retained `gradient_probe_enabled=true`, `gradient_probe_step=1`, and `save_interval=1`. Each completed run produced `config.yaml`, `config.full.yaml`, `dataset_statistics.json`, `gradient_probe.json`, `steps_1`, `steps_2`, and `final_model`. All reported losses are finite.

| Case | Exit | Peak GPU MiB (GPU0/GPU1) | Step 1 `(action, video, total)` | Step 2 `(action, video, total)` | Gradient probe at first boundary | Checkpoints / parity | Run path |
|---|---:|---:|---|---|---|---|---|
| SW-L-MW | 0 | 36,936 / 36,882 | `(1.447847605, 0.163085938, 1.610933542)` | `(7.583916664, 0.279296875, 7.863213539)` | action expert 1.135181; backbone DiT 0.553603; proprio 0.023536; finite/nonzero | steps 1/2 + final present; strict 0/0; native parity PASS | `/home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824/train/SW-L-MW_20260722_063933` |
| SW-L-MC | N/A | N/A | BLOCKED | BLOCKED | BLOCKED | Cosmos assets unavailable | N/A |
| SW-L-SW | 0 | 36,461 / 36,480 | `(2.183520317, 0.225585938, 2.409106255)` | `(27.896472931, 0.474609375, 28.371082306)` | shared DiT 5.000910; finite/nonzero; backbone DiT frozen | steps 1/2 + final present; strict 0/0; restored inference FAIL | `/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-L-SW_20260722_092000` |
| SW-L-SC | N/A | N/A | BLOCKED | BLOCKED | BLOCKED | Cosmos assets unavailable | N/A |
| SW-L-FW | 0 | 32,131 / 30,468 | `(1.264688253, 0, 1.264688253)` | `(11.166011810, 0, 11.166011810)` | action expert 0.462537; backbone 0.058717; feature projector 0.075468; proprio 0.001034; finite/nonzero | steps 1/2 + final present; strict 0/0; bundle inference PASS | `/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-L-FW_20260722_092000` |
| SW-R-MW | 0 | 39,153 / 39,152 | `(1.469900131, 0.447265625, 1.917165756)` | `(10.304623604, 1.078125000, 11.382748604)` | action expert 0.107077; backbone DiT 0.059759; proprio 0.004765; finite/nonzero | steps 1/2 + final present; strict 0/0; live policy server PASS | `/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-R-MW_20260722_092000` |

For MoT and Shared-DiT, `total = lambda_action * action_loss + lambda_video * video_loss` exactly at both recorded steps. Feature-conditioned video loss is exactly zero, satisfying `abs(video_loss) <= 1e-8`.

The formal SW-R-MW run used per-device batch 4, accumulation 128, world size 2, and therefore effective global batch 1024; this was not silently reduced after memory observations. Its four-shard official text precompute completed with 921,032 real `.pt` cache entries (about 904 GB), and dataset statistics were computed from 27,500 parquet files. A prior single-rank cache attempt was deliberately interrupted as too slow and is retained as an attempt log.

The real-case probes cover the required trainable functional groups. The separate two-rank probe explicitly verifies a frozen VAE has zero trainable/gradient elements. The text encoder is not registered as a trainable model group. The archived real-case probe format does not independently enumerate frozen VAE/text-encoder parameters; this limitation is not promoted to a pass claim.

SW-L-MW training and its GPU trace were reused from `3f072c4`: `ad6b412` changed only the test plan, test names, and a runtime docstring, leaving that training behavior unchanged. The reuse is recorded in `reused_unaffected_artifacts.txt`.

## G2 restore and relocation

| Case | Original `steps_2` strict load | Relocated bundle strict load | Actual bundle inference |
|---|---|---|---|
| SW-L-MW | PASS, 3300 keys, missing 0, unexpected 0 | PASS; missing recipe path; resolved snapshot used | PASS, finite `[1,32,7]` |
| SW-L-SW | PASS, 1664 keys, missing 0, unexpected 0 | PASS; missing recipe path; resolved snapshot used | **FAIL**, CPU/CUDA mismatch |
| SW-L-FW | PASS, 1657 keys, missing 0, unexpected 0 | PASS; missing recipe path; resolved snapshot used | PASS, finite `[1,32,7]` |
| SW-R-MW | PASS, 3300 keys, missing 0, unexpected 0 | PASS; missing recipe path; resolved snapshot used | PASS through live server, finite `[1,32,14]` |

Each relocated bundle set its original recipe to `/definitely/missing/starwam_recipe.yaml`, proving recovery used the bundled resolved snapshot rather than the original recipe location.

A real native StarWAM/DeepSpeed MoT checkpoint was imported strictly: 3300 keys, 0 missing, 0 unexpected (`artifacts/SW-L-MW/native_checkpoint_import.json`).

First critical G2 stack (`logs/SW-L-SW_bundle_inference.log`):

```text
StarWAM.py:194 predict_action
shared_dit_wam.py:190 infer_action -> shared_dit_wam.py:280 infer_joint
causal_wan.py:238 forward -> causal_wan.py:175 _build_video_time_mod
self.time_embedding(t_emb) -> torch.nn.functional.linear
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

Reproduction uses the archived bundle and the same fixed golden input:

```bash
cd /home/amax/SJ/projects/Embodied-World/starVLA
PYTHONPATH=/home/amax/SJ/projects/Embodied-World/starVLA:/home/amax/SJ/projects/Embodied-World/StarWAM \
  /home/amax/data0/SJ/envs/starvla-starwam-py311/bin/python \
  -c 'import numpy as np; from starVLA.model.framework import get_model; p="/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/artifacts/bundles/SW-L-SW"; g=np.load("examples/modelExtensions/StarWAM/test_results/3f072c4_20260722_062824/artifacts/SW-L-MW/golden_input_libero.npz"); m=get_model("starwam", pretrained_path=p); m.predict_action(images=[g["primary"],g["wrist"]], instruction="pick up the black bowl between the plate and the ramekin and place it on the plate", state=g["state"], seed=42, num_inference_steps=2, action_num_inference_steps=2)'
```

No `.to(cuda)` workaround or source edit was applied after observing this failure.

## G3 native parity and resize decision

The canonical image contract follows the user-approved StarVLA training pipeline: tensor-domain PyTorch resize, matching StarWAM training `_resize_frames`. FastWAM's PIL-based native rollout resize is retained only as a diagnostic and is not the canonical integration contract.

Using the same real native checkpoint, original image arrays, instruction, state, seed 42, and two video/action inference steps:

- Pixels: exact `torch.equal`, max absolute difference 0, 0 differing elements, shape `[1,3,224,448]`.
- Text context/mask: exact.
- State: max absolute difference 0 at tolerance `1e-6`.
- Normalized action: max absolute difference 0 at `atol=1e-4`, `rtol=1e-3`.
- Final action: max absolute difference 0 at `atol=1e-4`, `rtol=1e-3`.
- Action horizon 32; video frames 9.
- PIL rollout diagnostic: max absolute pixel difference 0.513967693, mean 0.010353229, 230,030 differing elements. This demonstrates why changing the canonical path to PIL would create mismatch.

Full machine-readable evidence is in `artifacts/SW-L-MW/parity.json`. Native parity could not be expanded to Shared-DiT, Feature-conditioned, RoboTwin, or Cosmos because matching published native checkpoints/golden inputs were unavailable; this scope limitation is explicit.

## Policy server

The real SW-R-MW relocated bundle was served and queried:

- Metadata: action horizon 32; model family `mot_wam`; three cameras in head/left/right order; server preprocessing enabled; action layout `native_qpos`.
- Response: finite float32 action tensor `[1,32,14]`; framework z-score unnormalization applied.
- Client contract: `client_must_not_reorder=true`; no legacy joint reorder was performed.
- Negative paths correctly reject missing 14-D state, two cameras instead of three, unknown unnormalization key, missing statistics at startup, and 13-D statistics for a 14-D request.

See `artifacts/SW-R-MW/policy_server_result.json` and the associated server/client logs. A wrong-count camera order is detectable; same-count semantic misordering cannot be inferred from anonymous arrays, so metadata and the declared client contract are the guardrail.

## G4 rollout

LIBERO assets and simulator functionality were validated independently: a real `libero_goal` BDDL task reset, accepted an init state, rendered 64x64 images, and stepped successfully. The formal StarVLA runner reached environment construction but failed before episode execution:

```text
eval_libero.py:110 eval_libero
eval_libero.py:236 _get_libero_env -> OffScreenRenderEnv(**env_args)
/home/amax/SJ/libero-plus/libero/libero/envs/env_wrapper.py:207
if "_view_" in bddl_file_name and "_initstate_" in bddl_file_name:
TypeError: argument of type 'PosixPath' is not iterable
```

Reproduction evidence is in `logs/SW-L-MW_rollout_attempt2.log`. The preceding CLI attempt stopped earlier because that compatible LIBERO environment lacked the optional `tyro` CLI parser; the direct invocation then exposed the actual runner incompatibility. No monkeypatch or source change was used to claim a rollout.

RoboTwin rollout is **BLOCKED**, not passed: the available simulator checkout/assets are incomplete. Policy-server inference alone is not reported as a simulator rollout.

## Skips, overrides, and retained first-attempt evidence

- Test-suite skips: **0** targeted, **0** full-suite.
- Mandatory matrix skips: none reported as pass; SW-L-MC and SW-L-SC are explicitly BLOCKED.
- Resource configuration: two physical GPUs; bf16; ZeRO-2 with CPU optimizer offload and no parameter offload. SW-R-MW effective global batch remained 1024 (4 per device × 2 ranks × 128 accumulation).
- Retained non-code/harness attempts: Shared-DiT launcher quoting error; SW-R-MW single-rank cache interruption; two initial no-StarWAM harness assertion/module mistakes; policy missing-stats exception-type overconstraint; LIBERO environment compatibility attempts; initial CLI `tyro` absence. These are preserved in logs and are not counted as product failures or passes.
- Stop rule: after the first critical restored-inference device mismatch, no performance benchmark was promoted. Compatibility diagnostics already in flight were retained. No code was modified to conceal the failure.

## Artifact index

- Machine/environment: `manifest/`
- Complete logs and first stacks: `logs/`
- Small configs, statistics, probes, strict-restore results, policy result, and parity result: `artifacts/`
- Per-GPU memory traces: `stats/`
- Large external checkpoints and caches: `/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916`
