# StarWAM framework integration

This integration keeps StarWAM as an external Apache-2.0 dependency. StarVLA
owns the run directory, trainer, distributed data loading, checkpoints, and
policy server; the adapter owns a complete StarWAM `WAMModel` and does not turn
its action stream into a regular StarVLA action head.

Pinned source revisions:

- StarVLA integration base: `a060fd9`; the feature branch is synchronized with
  `starVLA_dev` through `c00b8e1`.
- StarWAM: `a7b05c8da8f8c88bf4888aa737b938af3b67ab48`.

## Installation

For a reproducible install from GitHub:

```bash
pip install -e '.[wam]'
```

For this workspace, use the sibling checkout while developing:

```bash
pip install -e '.[wam]'
pip install --no-deps -e '../StarWAM[train]'
```

Install the backbone-specific packages documented by StarWAM as well. In
particular, Wan and Cosmos recipes require their local pretrained model trees;
the recipe placeholders must be overridden before training.

## Configuration

The thin StarVLA config points to one StarWAM recipe:

```yaml
framework:
  name: StarWAM
  starwam:
    recipe: /path/to/starwam_recipe.yaml
    overrides: []
    init_checkpoint: null
    strict_init: false
datasets:
  vla_data:
    dataset_py: starwam_datasets
```

The effective precedence is:

1. StarVLA command-line fields.
2. Fields explicitly present in the thin StarVLA YAML.
3. Values projected from the resolved StarWAM recipe.

Use `framework.starwam.overrides` for StarWAM-native fields, for example:

```bash
python -m starVLA.training.train_starvla \
  --config_yaml examples/modelExtensions/StarWAM/train_files/libero_mot_wan22.yaml \
  --framework.starwam.overrides='["backbone.pretrained_model_id=/models/Wan2.2-TI2V-5B","data.dataset_dirs=[/data/libero]","data.text_embedding_cache_dir=/runs/cache","data.action_stats_path=/runs/stats.json","data.state_stats_path=/runs/stats.json"]'
```

The complete resolved recipe and StarWAM revision are embedded in
`config.yaml`. A saved wrapper checkpoint therefore does not need the original
recipe file. `dataset_statistics.json` is copied into the same run bundle and
is used by the policy server for StarWAM min-max or z-score action recovery.

## Supported recipes

The `train_files` directory contains the supported matrix:

- LIBERO MoT with Wan2.2 or Cosmos-Predict2.
- LIBERO Shared-DiT with Wan2.2 or Cosmos-Predict2.
- LIBERO Feature-conditioned with Wan2.2.
- RoboTwin MoT with Wan2.2.

MoT and Shared-DiT return their already weighted `total_loss`. StarVLA
backpropagates that tensor once and logs raw `action_loss` and auxiliary
`video_loss`; it never adds the components again. Feature-conditioned models
report `video_loss=0`.

Only `training.strategy=full` is supported. LoRA, staged unfreezing, LAPA, and
latent-action tokenizers fail at configuration time rather than silently using
the wrong trainable parameter set.

## Checkpoints and serving

`framework.starwam.init_checkpoint` imports a native StarWAM or DeepSpeed model
payload. Prefixes `module.`, `model.`, `_orig_mod.`, and `wam.` are adapted.
An import fails if no tensors match or if a critical MoT, Shared-DiT, action
expert, or feature projector group is absent.

Regular StarVLA checkpoints contain the complete wrapper state and restore
strictly with `baseframework.from_pretrained`. The policy server delegates
camera composition, state normalization, action horizon, and action
unnormalization to the StarWAM adapter. RoboTwin metadata marks actions as
`native_qpos`, so the legacy client permutation is skipped.

## Verification

The staged commands, evidence requirements, failure triage, and merge/release
gates are defined in the [test execution plan](TEST_EXECUTION_PLAN.md).

Run the CPU integration and existing regression tests with the optional WAM
environment and the full StarVLA training dependencies (including DeepSpeed)
installed:

```bash
PYTHONPATH=.:../StarWAM pytest -q \
  tests/test_starwam_integration.py \
  tests/test_single_process_dist_safety.py \
  tests/test_robocasa_tabletop_interface.py
```

The GPU merge gate is two bf16 ZeRO-2 optimizer steps, inference, save, and
strict restore for every thin config. Verify finite loss/gradients and expected
trainable modules on at least one multi-GPU run for each family. Offline policy
parity uses the same checkpoint, input, and seed and requires action chunks at
`atol=1e-4, rtol=1e-3`. Full 50-trial rollouts are release validation rather
than a code-review gate; record the recipe snapshot, commit revisions, GPU
model/count, CUDA stack, and per-task results.

Enable `trainer.gradient_probe_enabled=true` during GPU smoke runs. The first
optimizer boundary writes `gradient_probe.json` before clipping, grouped by
StarWAM functional module with finite/nonzero ratios and gradient norms.
