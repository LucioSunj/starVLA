"""CPU contracts for the optional StarWAM framework integration."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from starVLA.model.framework.WAM.checkpoint import load_checkpoint_state, strip_checkpoint_prefixes
from starVLA.model.framework.WAM.config import (
    STARWAM_REVISION,
    prepare_starwam_host_config,
    require_starwam,
    select_config,
)
from starVLA.model.framework.WAM.loss_contract import validated_loss_metrics
from starVLA.model.framework.WAM.runtime import (
    compose_camera_views,
    denormalize_actions,
    normalize_vector,
)
from starVLA.training.loss_utils import resolve_model_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
STARWAM_ROOT = REPO_ROOT.parent / "StarWAM"
_FRAMEWORK_DEPS_AVAILABLE = importlib.util.find_spec("transformers") is not None
_TRAINER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("accelerate", "transformers", "torchvision", "wandb")
)


def _starwam_available() -> bool:
    return importlib.util.find_spec("starwam") is not None


def _import_starvla_example(module_name: str):
    """Avoid the sibling StarWAM `examples` package shadowing this repo."""
    removed = [entry for entry in sys.path if Path(entry or ".").resolve() == STARWAM_ROOT]
    for entry in removed:
        sys.path.remove(entry)
    for name in list(sys.modules):
        if name == "examples" or name.startswith("examples."):
            del sys.modules[name]
    try:
        return importlib.import_module(module_name)
    finally:
        for entry in reversed(removed):
            sys.path.append(entry)


def _write_recipe(
    path: Path,
    *,
    family: str = "mot_wam",
    representation: str = "token_action",
    framework_type: str = "mot",
    max_steps: int | None = None,
) -> Path:
    recipe = {
        "taxonomy": {
            "package": "starwam",
            "model_family": family,
            "action_representation": representation,
            "conditioning": "first_frame",
        },
        "framework": {
            "type": framework_type,
            "action_dim": 7,
            "chunk_size": 8,
            "proprio_dim": 8,
            "loss_lambda_video": 0.5,
            "loss_lambda_action": 2.0,
        },
        "training": {
            "output_dir": str(path.parent / "native-output"),
            "batch_size": 3,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.02,
            "num_epochs": 4,
            "max_steps": max_steps,
            "warmup_steps": None,
            "warmup_ratio": 0.1,
            "mixed_precision": "bf16",
            "save_every": 9,
            "eval_every": 0,
            "log_every": 2,
            "strategy": "full",
            "num_workers": 0,
            "wandb_enabled": False,
        },
        "data": {
            "dataset_type": "synthetic",
            "num_frames": 9,
            "video_size": [12, 10],
            "action_freq_ratio": 2,
        },
    }
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return path


def _host_config(recipe: Path):
    return OmegaConf.create(
        {
            "run_id": "host-run",
            "run_root_dir": str(recipe.parent / "runs"),
            "framework": {
                "name": "StarWAM",
                "starwam": {
                    "recipe": str(recipe),
                    "overrides": ["training.batch_size=5"],
                    "init_checkpoint": None,
                    "strict_init": False,
                },
            },
            "datasets": {
                "vla_data": {
                    "dataset_py": "starwam_datasets",
                    "per_device_batch_size": 7,
                }
            },
            "trainer": {"learning_rate": {"base": 3.0e-4}},
        }
    )


def test_lightweight_modules_do_not_import_starwam() -> None:
    code = """
import sys
import types
import torch

class BlockStarWAM:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'starwam' or fullname.startswith('starwam.'):
            raise AssertionError(f'unexpected eager import: {fullname}')
        return None

sys.meta_path.insert(0, BlockStarWAM())
transformers = types.ModuleType('transformers')
class PretrainedConfig:
    pass
class PreTrainedModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
transformers.PretrainedConfig = PretrainedConfig
transformers.PreTrainedModel = PreTrainedModel
sys.modules['transformers'] = transformers
torchvision = types.ModuleType('torchvision')
torchvision.transforms = types.SimpleNamespace(ToTensor=lambda: None)
sys.modules['torchvision'] = torchvision
import starVLA.model.framework.WAM.config
import starVLA.model.framework.WAM.runtime
import starVLA.model.framework.WAM.StarWAM
import starVLA.training.loss_utils
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_optional_dependency_error_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "starwam" else object())
    with pytest.raises(RuntimeError, match=r"pip install -e.*wam"):
        require_starwam()


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_recipe_projection_precedence_and_self_contained_snapshot(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path / "recipe.yaml", max_steps=99)
    host_cfg = _host_config(recipe)
    host_cfg.trainer.max_train_steps = None
    cfg = prepare_starwam_host_config(host_cfg, config_path=tmp_path / "host.yaml")

    assert cfg.datasets.vla_data.per_device_batch_size == 7
    assert cfg.trainer.learning_rate.base == pytest.approx(3.0e-4)
    assert cfg.trainer.optimizer.weight_decay == pytest.approx(0.02)
    assert cfg.trainer.scheduler_specific_kwargs.min_lr == pytest.approx(3.0e-6)
    assert cfg.trainer.max_train_steps is None
    assert cfg.framework.starwam.resolved.training.batch_size == 5
    assert cfg.framework.starwam.resolved.training.max_steps == 99
    assert cfg.framework.starwam.resolved.training.output_dir == str(tmp_path / "runs" / "host-run")
    assert cfg.framework.starwam.source_revision == STARWAM_REVISION
    assert cfg.framework.starwam.resolved_recipe == str(recipe)
    assert cfg.trainer.gradient_probe_enabled is False
    assert cfg.trainer.gradient_probe_step == 1
    assert cfg.trainer.gradient_probe_chunk_size == 1_048_576

    snapshot = tmp_path / "config.yaml"
    OmegaConf.save(cfg, snapshot, resolve=True)
    recipe.unlink()
    restored_snapshot = OmegaConf.load(snapshot)
    restored_snapshot.framework.starwam.overrides = ["training.batch_size=11"]
    restored = prepare_starwam_host_config(restored_snapshot, config_path=snapshot)
    assert restored.framework.starwam.resolved.taxonomy.model_family == "mot_wam"
    assert restored.framework.starwam.resolved.training.batch_size == 11
    assert restored.datasets.vla_data.per_device_batch_size == 7


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_latent_action_recipe_fails_at_configuration_time(tmp_path: Path) -> None:
    recipe = _write_recipe(
        tmp_path / "latent.yaml",
        family="shared_dit_wam",
        representation="latent_action",
        framework_type="shared_dit",
    )
    with pytest.raises(NotImplementedError, match="latent-action/LAPA"):
        prepare_starwam_host_config(_host_config(recipe))


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_lora_and_manual_freezing_fail_before_model_construction(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path / "recipe.yaml")
    lora_cfg = _host_config(recipe)
    lora_cfg.trainer.training_strategy = "lora"
    with pytest.raises(NotImplementedError, match="LoRA and staged"):
        prepare_starwam_host_config(lora_cfg)

    freeze_cfg = _host_config(recipe)
    freeze_cfg.trainer.freeze_modules = "wam.backbone"
    with pytest.raises(NotImplementedError, match="freeze_modules"):
        prepare_starwam_host_config(freeze_cfg)


def test_select_config_supports_namespace_and_mapping() -> None:
    cfg = SimpleNamespace(trainer=SimpleNamespace(mixed_precision="fp16"))
    assert select_config(cfg, "trainer.mixed_precision") == "fp16"
    assert select_config({"trainer": {"mixed_precision": "bf16"}}, "trainer.mixed_precision") == "bf16"


def test_total_loss_is_selected_without_readding_auxiliary_components() -> None:
    action = torch.tensor(2.0, requires_grad=True)
    video = torch.tensor(3.0, requires_grad=True)
    total = 0.25 * action + 2.0 * video
    selected, metrics = resolve_model_loss(
        {
            "total_loss": total,
            "action_loss": action,
            "loss_metrics": {
                "action_loss": action.detach(),
                "video_loss": video.detach(),
            },
        }
    )
    assert selected is total
    selected.backward()
    assert action.grad.item() == pytest.approx(0.25)
    assert video.grad.item() == pytest.approx(2.0)
    assert metrics["action_loss"] == pytest.approx(2.0)
    assert metrics["video_loss"] == pytest.approx(3.0)


def test_legacy_action_loss_fallback() -> None:
    action_loss = torch.tensor(1.25, requires_grad=True)
    selected, metrics = resolve_model_loss({"action_loss": action_loss})
    assert selected is action_loss
    assert metrics == {"total_loss": 1.25, "action_loss": 1.25}


@pytest.mark.parametrize(
    ("family", "lambda_action", "lambda_video", "action", "video"),
    [
        ("mot_wam", 1.0, 1.0, 2.0, 3.0),
        ("mot_wam", 0.0, 2.0, 2.0, 3.0),
        ("shared_dit_wam", 3.0, 0.0, 2.0, 3.0),
        ("feature_conditioned_action_model", 0.5, 0.0, 2.0, 0.0),
    ],
)
def test_starwam_loss_weighting_contract(
    family: str,
    lambda_action: float,
    lambda_video: float,
    action: float,
    video: float,
) -> None:
    expected = lambda_action * action + lambda_video * video
    config = SimpleNamespace(loss_lambda_action=lambda_action, loss_lambda_video=lambda_video)
    metrics = validated_loss_metrics(
        family,
        config,
        torch.tensor(expected),
        {"loss_action": action, "loss_video": video, "loss_total": expected},
    )
    assert metrics["total_loss"] == pytest.approx(expected)


def test_feature_conditioned_rejects_nonzero_video_loss() -> None:
    config = SimpleNamespace(loss_lambda_action=1.0, loss_lambda_video=0.0)
    with pytest.raises(RuntimeError, match="video_loss=0"):
        validated_loss_metrics(
            "feature_conditioned_action_model",
            config,
            torch.tensor(1.0),
            {"loss_action": 1.0, "loss_video": 0.1, "loss_total": 1.0},
        )


def _reference_resize(frame: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
    from starwam.data.lerobot import _resize_frames

    return _resize_frames(tensor, size).float()[0] / 255.0


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_libero_camera_composition_matches_starwam_pixels() -> None:
    rng = np.random.default_rng(7)
    primary = rng.integers(0, 256, (19, 23, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (17, 29, 3), dtype=np.uint8)
    data = SimpleNamespace(
        concat_multi_camera="horizontal",
        video_keys=["primary", "wrist"],
        video_key="primary",
        video_size=[12, 10],
    )
    actual = compose_camera_views([primary, wrist], data)
    expected = torch.cat(
        [_reference_resize(primary, (12, 10)), _reference_resize(wrist, (12, 10))],
        dim=-1,
    )
    expected = (expected * 2.0 - 1.0).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_robotwin_camera_composition_matches_starwam_pixels() -> None:
    rng = np.random.default_rng(11)
    head = rng.integers(0, 256, (31, 47, 3), dtype=np.uint8)
    left = rng.integers(0, 256, (23, 37, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (29, 41, 3), dtype=np.uint8)
    data = SimpleNamespace(
        concat_multi_camera="robotwin",
        video_keys=["head", "left", "right"],
        video_key="head",
        video_size=[384, 320],
    )
    actual = compose_camera_views([head, left, right], data)
    top = _reference_resize(head, (256, 320))
    bottom = torch.cat(
        [_reference_resize(left, (128, 160)), _reference_resize(right, (128, 160))],
        dim=-1,
    )
    expected = (torch.cat([top, bottom], dim=-2) * 2.0 - 1.0).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_action_and_state_normalization_roundtrip(mode: str) -> None:
    stats = {
        "min": torch.tensor([-2.0, 1.0]),
        "max": torch.tensor([2.0, 5.0]),
        "mean": torch.tensor([0.5, 3.0]),
        "std": torch.tensor([0.5, 2.0]),
    }
    values = torch.tensor([[0.0, 3.0]])
    normalized = normalize_vector(values, mode, stats)
    restored = denormalize_actions(normalized, mode, stats)
    torch.testing.assert_close(restored, values)


def test_normalization_clipping() -> None:
    stats = {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0])}
    assert normalize_vector(torch.tensor([[100.0]]), "zscore", stats).item() == 5.0


def test_gradient_probe_reports_functional_trainable_and_frozen_groups(tmp_path: Path) -> None:
    import json

    from starVLA.training.gradient_probe import collect_gradient_probe, write_gradient_probe

    class TinyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Linear(2, 2, bias=False)
            self.vae = torch.nn.Linear(2, 2, bias=False)
            self.vae.requires_grad_(False)

    class TinyWAM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = TinyBackbone()
            self.action_expert = torch.nn.Linear(2, 2, bias=False)

    class TinyWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wam = TinyWAM()

    model = TinyWrapper()
    with torch.no_grad():
        model.wam.backbone.dit.weight.fill_(1.0)
        model.wam.action_expert.weight.fill_(2.0)
    loss = model.wam.backbone.dit.weight.square().sum() + model.wam.action_expert.weight.square().sum()
    loss.backward()

    groups = collect_gradient_probe(model, chunk_size=2)
    assert groups["backbone.dit"]["finite_gradient_ratio"] == 1.0
    assert groups["backbone.dit"]["nonzero_gradient_ratio"] == 1.0
    assert groups["action_expert"]["gradient_l2_norm"] > 0
    assert groups["backbone.vae"]["trainable_parameter_elements"] == 0
    assert groups["backbone.vae"]["gradient_elements"] == 0

    output = tmp_path / "gradient_probe.json"
    payload = write_gradient_probe(
        model,
        output,
        model_family="mot_wam",
        optimizer_step=1,
        chunk_size=2,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_checkpoint_prefix_adaptation(tmp_path: Path) -> None:
    weight = torch.arange(4, dtype=torch.float32)
    payload = {"module": {"model._orig_mod.wam.mot.weight": weight, "metadata": "ignored"}}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    state = load_checkpoint_state(path)
    assert list(state) == ["mot.weight"]
    torch.testing.assert_close(state["mot.weight"], weight)
    stripped = strip_checkpoint_prefixes({"wam.action.weight": weight})
    assert list(stripped) == ["action.weight"]
    torch.testing.assert_close(stripped["action.weight"], weight)


@pytest.mark.skipif(not _FRAMEWORK_DEPS_AVAILABLE, reason="transformers is not installed")
def test_init_checkpoint_rejects_zero_match_and_missing_critical_groups(tmp_path: Path) -> None:
    from starVLA.model.framework.WAM.StarWAM import StarWAMFramework

    class TinyMoT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mot = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Linear(2, 2)

    framework = object.__new__(StarWAMFramework)
    torch.nn.Module.__init__(framework)
    framework.wam = TinyMoT()
    framework.model_family = "mot_wam"

    zero_match = tmp_path / "zero.pt"
    torch.save({"unrelated.weight": torch.ones(2, 2)}, zero_match)
    with pytest.raises(RuntimeError, match="matched zero"):
        framework._load_init_checkpoint(zero_match, strict=False)

    missing_action_expert = tmp_path / "missing_action.pt"
    torch.save(
        {
            "mot.weight": framework.wam.mot.weight.detach().clone(),
            "mot.bias": framework.wam.mot.bias.detach().clone(),
        },
        missing_action_expert,
    )
    with pytest.raises(RuntimeError, match="action_expert"):
        framework._load_init_checkpoint(missing_action_expert, strict=False)


@pytest.mark.skipif(not _starwam_available(), reason="StarWAM optional dependency is not installed")
def test_synthetic_starwam_batch_contract_and_shared_dit_proprio_validation() -> None:
    from starwam.builder import build_dataset
    from starwam.config import StarWAMConfig
    from starwam.wam.shared_dit_wam import SharedDiTWAM

    config = StarWAMConfig()
    config.data.dataset_type = "synthetic"
    config.data.num_frames = 9
    config.data.action_freq_ratio = 2
    config.data.video_size = [12, 10]
    config.framework.action_dim = 7
    config.framework.chunk_size = 8
    config.framework.proprio_dim = 8
    sample = build_dataset(config, is_training=True, text_dim=16)[0]

    assert sample["video"].shape == (3, 5, 12, 10)
    assert sample["action"].shape == (8, 7)
    assert sample["context"].shape == (77, 16)
    assert sample["image_is_pad"].shape == (5,)
    assert sample["action_is_pad"].shape == (8,)
    assert sample["proprio"].shape == (8, 8)

    shared = object.__new__(SharedDiTWAM)
    torch.nn.Module.__init__(shared)
    shared.state_dim = 8
    shared.action_tokens_per_state = 4
    with pytest.raises(ValueError, match=r"requires sample\['proprio'\]"):
        shared._state_from_sample({"action_is_pad": sample["action_is_pad"]}, sample["action"].unsqueeze(0))


def test_thin_config_matrix_is_complete() -> None:
    from starVLA.model.framework.share_tools import apply_config_compat

    config_dir = REPO_ROOT / "examples" / "modelExtensions" / "StarWAM" / "train_files"
    expected = {
        "libero_mot_wan22.yaml": ("starwam_libero_mot_wan22_5b.yaml", "mot_wam"),
        "libero_mot_cosmos.yaml": ("starwam_libero_mot_cosmos_predict2.yaml", "mot_wam"),
        "libero_shared_dit_wan22.yaml": (
            "starwam_libero_shared_dit_wan22_5b.yaml",
            "shared_dit_wam",
        ),
        "libero_shared_dit_cosmos.yaml": (
            "starwam_libero_shared_dit_cosmos_predict2.yaml",
            "shared_dit_wam",
        ),
        "libero_feature_conditioned_wan22.yaml": (
            "starwam_libero_feature_conditioned_wan22_5b.yaml",
            "feature_conditioned_action_model",
        ),
        "robotwin_mot_wan22.yaml": ("starwam_robotwin_mot_wan22_5b.yaml", "mot_wam"),
    }
    assert {path.name for path in config_dir.glob("*.yaml")} == set(expected)
    for filename, (recipe_name, family) in expected.items():
        config_path = config_dir / filename
        config = apply_config_compat(OmegaConf.load(config_path))
        assert Path(config.framework.starwam.recipe).name == recipe_name
        if STARWAM_ROOT.is_dir():
            resolved = (config_dir / config.framework.starwam.recipe).resolve()
            assert resolved.is_file()
        if _starwam_available():
            resolved_config = prepare_starwam_host_config(config, config_path=config_path)
            assert resolved_config.framework.starwam.resolved.taxonomy.model_family == family


class _FakePolicyClient:
    def __init__(self, metadata: dict, action_dim: int) -> None:
        self.metadata = metadata
        self.action_dim = action_dim
        self.requests = []

    def get_server_metadata(self) -> dict:
        return self.metadata

    def predict_action(self, request: dict) -> dict:
        self.requests.append(request)
        action = np.arange(self.action_dim, dtype=np.float32)
        actions = np.stack([action, action], axis=0)[None]
        return {"data": {"actions": actions}}


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed")
def test_libero_client_leaves_raw_pixels_for_framework_preprocessing() -> None:
    model2libero_interface = _import_starvla_example("examples.simBenchmarks.LIBERO.eval_files.model2libero_interface")

    fake = _FakePolicyClient(
        {
            "action_chunk_size": 2,
            "framework_preprocesses_images": True,
            "state_required": True,
        },
        action_dim=7,
    )
    with mock.patch.object(model2libero_interface, "WebsocketClientPolicy", return_value=fake):
        client = model2libero_interface.ModelClient(action_ensemble=False, image_size=(224, 224))
    image = np.zeros((31, 47, 3), dtype=np.uint8)
    client.step({"image": [image, image], "lang": "pick", "state": np.zeros(8)}, step=0)
    sent = fake.requests[-1]["examples"][0]
    assert np.asarray(sent["image"][0]).shape == (31, 47, 3)
    assert np.asarray(sent["state"]).shape == (8,)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="OpenCV is not installed")
def test_robotwin_client_preserves_raw_pixels_and_native_action_order() -> None:
    model2robotwin_interface = _import_starvla_example(
        "examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface"
    )

    fake = _FakePolicyClient(
        {
            "action_chunk_size": 2,
            "framework_preprocesses_images": True,
            "action_layout": "native_qpos",
            "state_required": True,
        },
        action_dim=14,
    )
    with mock.patch.object(model2robotwin_interface, "WebsocketClientPolicy", return_value=fake):
        client = model2robotwin_interface.ModelClient(policy_ckpt_path="unused", action_ensemble=False)
    images = [np.zeros((31, 47, 3), dtype=np.uint8) for _ in range(3)]
    action = client.step({"image": images, "lang": "move", "state": np.zeros(14)}, step=0)
    sent = fake.requests[-1]["examples"][0]
    assert np.asarray(sent["image"][0]).shape == (31, 47, 3)
    assert np.asarray(sent["state"]).shape == (14,)
    np.testing.assert_array_equal(action, np.arange(14, dtype=np.float32))


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("cv2") is None,
    reason="client regression dependencies are not installed",
)
def test_benchmark_clients_do_not_add_state_to_legacy_policy_requests() -> None:
    model2libero_interface = _import_starvla_example("examples.simBenchmarks.LIBERO.eval_files.model2libero_interface")
    libero_fake = _FakePolicyClient(
        {"action_chunk_size": 2, "framework_preprocesses_images": False},
        action_dim=7,
    )
    with mock.patch.object(model2libero_interface, "WebsocketClientPolicy", return_value=libero_fake):
        libero = model2libero_interface.ModelClient(action_ensemble=False, image_size=(16, 16))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    libero.step({"image": [image], "lang": "pick", "state": np.zeros(8)}, step=0)
    assert "state" not in libero_fake.requests[-1]["examples"][0]

    model2robotwin_interface = _import_starvla_example(
        "examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface"
    )
    robotwin_fake = _FakePolicyClient(
        {"action_chunk_size": 2, "framework_preprocesses_images": False},
        action_dim=14,
    )
    with mock.patch.object(model2robotwin_interface, "WebsocketClientPolicy", return_value=robotwin_fake):
        robotwin = model2robotwin_interface.ModelClient(
            policy_ckpt_path="unused",
            action_ensemble=False,
            image_size=[16, 16],
        )
    robotwin.step(
        {"image": [image, image, image], "lang": "move", "state": np.zeros(14)},
        step=0,
    )
    assert "state" not in robotwin_fake.requests[-1]["examples"][0]


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_legacy_action_only_framework_completes_one_training_step() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class LegacyActionFramework(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, target):
            return {"action_loss": (self.weight - target).square()}

    class SingleProcessAccelerator:
        num_processes = 1
        gradient_accumulation_steps = 1
        sync_gradients = True

        @staticmethod
        def accumulate(model):
            return nullcontext()

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def backward(loss):
            loss.backward()

        @staticmethod
        def clip_grad_norm_(parameters, max_norm):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    model = LegacyActionFramework()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    cfg = OmegaConf.create(
        {
            "framework": {"name": "Legacy"},
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {"gradient_clipping": 1.0},
        }
    )
    trainer = VLATrainer(
        cfg,
        model,
        [torch.tensor(0.0)],
        optimizer,
        scheduler,
        SingleProcessAccelerator(),
    )
    before = model.weight.detach().clone()
    metrics = trainer._train_step(torch.tensor(0.0))

    assert model.weight.item() < before.item()
    assert metrics["action_dit_loss"] == pytest.approx(1.0)
    assert metrics["loss/total_loss"] == pytest.approx(1.0)
    assert metrics["_optimizer_step"] is True
    assert model.weight.grad is None


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_optimizer_lifecycle_waits_for_accumulation_boundary() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class LegacyActionFramework(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, target):
            return {"action_loss": (self.weight - target).square()}

    class ToggleAccelerator:
        num_processes = 1
        gradient_accumulation_steps = 2

        def __init__(self) -> None:
            self.sync_gradients = False
            self.clip_calls = 0

        @staticmethod
        def accumulate(model):
            return nullcontext()

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def backward(loss):
            loss.backward()

        def clip_grad_norm_(self, parameters, max_norm):
            self.clip_calls += 1
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    model = LegacyActionFramework()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scheduler_step = mock.patch.object(scheduler, "step", wraps=scheduler.step)
    cfg = OmegaConf.create(
        {
            "framework": {"name": "Legacy"},
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {"gradient_clipping": 1.0},
        }
    )
    accelerator = ToggleAccelerator()
    trainer = VLATrainer(cfg, model, [torch.tensor(0.0)], optimizer, scheduler, accelerator)

    with scheduler_step as step:
        before = model.weight.detach().clone()
        micro_metrics = trainer._train_step(torch.tensor(0.0))
        assert torch.equal(model.weight.detach(), before)
        assert model.weight.grad is not None
        assert step.call_count == 0
        assert accelerator.clip_calls == 0
        assert micro_metrics["_optimizer_step"] is False

        accelerator.sync_gradients = True
        boundary_metrics = trainer._train_step(torch.tensor(0.0))
        assert model.weight.item() < before.item()
        assert model.weight.grad is None
        assert step.call_count == 1
        assert accelerator.clip_calls == 1
        assert boundary_metrics["_optimizer_step"] is True


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_zero2_engine_updates_and_probes_only_at_accumulation_boundary(tmp_path: Path) -> None:
    from starVLA.training import train_starvla

    class FakeDeepSpeedEngine(torch.nn.Module):
        model_family = "mot_wam"

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.micro_steps = 0
            self.events: list[str] = []

        def forward(self, target):
            return {"action_loss": (self.weight - target).square()}

        def backward(self, loss):
            self.events.append("backward")
            (loss / 2).backward()

        def is_gradient_accumulation_boundary(self) -> bool:
            return (self.micro_steps + 1) % 2 == 0

        def step(self) -> None:
            self.events.append("engine_step")
            if self.is_gradient_accumulation_boundary():
                self.events.append("clip")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.events.append("optimizer_step")
                with torch.no_grad():
                    self.weight.add_(self.weight.grad, alpha=-0.1)
                self.events.append("zero_grad")
                self.weight.grad = None
            self.micro_steps += 1

    class FakeAccelerator:
        num_processes = 2
        gradient_accumulation_steps = 2
        is_main_process = True

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def accumulate(model):
            raise AssertionError("ZeRO-2 must not enter Accelerate accumulate/no_sync")

        @staticmethod
        def backward(loss):
            raise AssertionError("ZeRO-2 must use engine.backward to preserve the pre-clip probe")

        @staticmethod
        def clip_grad_norm_(parameters, max_norm):
            raise AssertionError("DeepSpeed must own clipping at the optimizer boundary")

        @staticmethod
        def unwrap_model(model):
            return model

    model = FakeDeepSpeedEngine()
    optimizer = mock.Mock()
    scheduler = mock.Mock()
    cfg = OmegaConf.create(
        {
            "framework": {"name": "StarWAM"},
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "output_dir": str(tmp_path),
            "trainer": {
                "gradient_clipping": 1.0,
                "gradient_probe_enabled": True,
                "gradient_probe_step": 1,
                "gradient_probe_chunk_size": 2,
            },
        }
    )
    trainer = train_starvla.VLATrainer(cfg, model, [torch.tensor(0.0)], optimizer, scheduler, FakeAccelerator())

    def record_probe(*args, **kwargs):
        model.events.append("gradient_probe")
        return {}

    with mock.patch.object(train_starvla, "write_gradient_probe", side_effect=record_probe) as probe:
        before = model.weight.detach().clone()
        micro_metrics = trainer._train_step(torch.tensor(0.0))
        assert torch.equal(model.weight.detach(), before)
        assert model.weight.grad is not None
        assert micro_metrics["_optimizer_step"] is False
        assert "optimizer_step" not in model.events
        assert "clip" not in model.events
        assert "zero_grad" not in model.events
        scheduler.step.assert_not_called()

        boundary_metrics = trainer._train_step(torch.tensor(0.0))

    assert boundary_metrics["_optimizer_step"] is True
    assert model.weight.item() < before.item()
    assert model.weight.grad is None
    assert model.events.index("gradient_probe") < model.events.index("clip")
    assert model.events.index("gradient_probe") < model.events.index("optimizer_step")
    assert model.events.index("gradient_probe") < model.events.index("zero_grad")
    probe.assert_called_once()
    scheduler.step.assert_called_once_with()
    optimizer.step.assert_not_called()
    optimizer.zero_grad.assert_not_called()


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_starwam_dataloader_completes_epochs_without_legacy_reset() -> None:
    from starVLA.training import train_starvla

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.config = OmegaConf.create({"framework": {"name": "StarWAM"}})
    trainer.vla_train_dataloader = [1, 2]
    trainer._create_data_iterators()

    with mock.patch.object(
        train_starvla.TrainerUtils,
        "_reset_dataloader",
        side_effect=AssertionError("StarWAM must not bypass DataLoaderShard epoch finalization"),
    ) as reset:
        assert [trainer._get_next_batch() for _ in range(5)] == [1, 2, 1, 2, 1]
    reset.assert_not_called()


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_starwam_accelerator_uses_global_step_scheduler_semantics() -> None:
    from starVLA.training import train_starvla

    cfg = OmegaConf.create(
        {
            "framework": {"name": "StarWAM"},
            "trainer": {
                "mixed_precision": "bf16",
                "gradient_accumulation_steps": 3,
                "gradient_clipping": 1.0,
            },
        }
    )
    fake_accelerator = SimpleNamespace(print=lambda *_: None, state="ready")
    with (
        mock.patch.object(train_starvla, "DeepSpeedPlugin", return_value="zero2") as plugin,
        mock.patch.object(train_starvla, "Accelerator", return_value=fake_accelerator) as factory,
    ):
        assert train_starvla.create_accelerator(cfg) is fake_accelerator

    plugin.assert_called_once_with(
        zero_stage=2,
        gradient_accumulation_steps=3,
        gradient_clipping=1.0,
    )
    factory.assert_called_once_with(
        deepspeed_plugin="zero2",
        gradient_accumulation_steps=3,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )


@pytest.mark.skipif(not _FRAMEWORK_DEPS_AVAILABLE, reason="transformers is not installed")
def test_full_freeze_strategy_and_unsupported_strategies() -> None:
    from starVLA.model.framework.WAM.StarWAM import StarWAMFramework

    class FakeWAM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.mot = torch.nn.Linear(2, 2)
            self.proprio_encoder = torch.nn.Linear(2, 2)

    framework = object.__new__(StarWAMFramework)
    torch.nn.Module.__init__(framework)
    framework.wam = FakeWAM()
    framework.starwam_config = SimpleNamespace(training=SimpleNamespace(strategy="full"))
    framework.configure_training(OmegaConf.create({"trainer": {"training_strategy": "full"}}))

    assert all(parameter.requires_grad for parameter in framework.wam.mot.parameters())
    assert all(parameter.requires_grad for parameter in framework.wam.proprio_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in framework.wam.backbone.parameters())

    for strategy in ("lora", "staged"):
        with pytest.raises(NotImplementedError, match=r"currently supports.*full"):
            framework.configure_training(OmegaConf.create({"trainer": {"training_strategy": strategy}}))
    with pytest.raises(NotImplementedError, match="freeze_modules"):
        framework.configure_training(
            OmegaConf.create({"trainer": {"training_strategy": "full", "freeze_modules": "wam.backbone"}})
        )


@pytest.mark.skipif(not _FRAMEWORK_DEPS_AVAILABLE, reason="transformers is not installed")
def test_wrapper_checkpoint_strict_roundtrip_is_deterministic(tmp_path: Path) -> None:
    import json

    from starVLA.model.framework import base_framework as base_module

    class TinyFramework(base_module.baseframework):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 2)

        def predict_action(self, examples, **kwargs):
            values = torch.as_tensor(examples, dtype=torch.float32)
            return {"normalized_actions": self.projection(values).detach().numpy()}

    torch.manual_seed(17)
    original = TinyFramework()
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "steps_1_pytorch_model.pt"
    torch.save(original.state_dict(), checkpoint)
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "StarWAM"},
                "trainer": {"pretrained_checkpoint": None},
            }
        ),
        run_dir / "config.yaml",
    )
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps({"starwam": {"action": {}, "state": {}}}),
        encoding="utf-8",
    )

    with mock.patch.object(base_module, "build_framework", side_effect=lambda cfg, **kwargs: TinyFramework()):
        restored = base_module.baseframework.from_pretrained(str(checkpoint))
    inputs = np.asarray([[0.25, -0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(
        restored.predict_action(inputs)["normalized_actions"],
        original.predict_action(inputs)["normalized_actions"],
        rtol=0.0,
        atol=0.0,
    )
    assert restored.norm_stats == {"starwam": {"action": {}, "state": {}}}


@pytest.mark.skipif(not _FRAMEWORK_DEPS_AVAILABLE, reason="transformers is not installed")
def test_policy_wrapper_preserves_legacy_normalization_path() -> None:
    from deployment.model_server import policy_wrapper

    class FakeFramework:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def get_policy_metadata(self):
            return {}

        def uses_framework_action_unnormalization(self):
            return False

        def predict_action(self, examples, **kwargs):
            return {"normalized_actions": np.ones((1, 2, 3), dtype=np.float32)}

    processor = SimpleNamespace(
        unnorm_key="legacy",
        available_unnorm_keys=["legacy"],
        action_keys=["action"],
        state_keys=["state"],
        unapply_actions=lambda actions: actions + 10.0,
    )
    config = {"framework": {"name": "Legacy", "action_model": {"action_horizon": 2}}}
    with (
        mock.patch.object(policy_wrapper, "read_mode_config", return_value=(config, {"legacy": {}})),
        mock.patch.object(
            policy_wrapper.baseframework,
            "from_pretrained",
            return_value=FakeFramework(),
        ),
        mock.patch.object(policy_wrapper.PolicyServerWrapper, "_get_processor", return_value=processor),
    ):
        wrapper = policy_wrapper.PolicyServerWrapper("unused.pt", device="cpu")
        result = wrapper.predict_action([{"image": [], "lang": "test"}])
        metadata = wrapper.metadata
    np.testing.assert_array_equal(result["actions"], np.full((1, 2, 3), 11.0, dtype=np.float32))
    assert metadata["action_chunk_size"] == 2


@pytest.mark.skipif(not _FRAMEWORK_DEPS_AVAILABLE, reason="transformers is not installed")
def test_policy_wrapper_delegates_starwam_metadata_and_unnormalization() -> None:
    from deployment.model_server import policy_wrapper

    class FakeStarWAM:
        def eval(self):
            return self

        def get_policy_metadata(self):
            return {
                "action_chunk_size": 4,
                "available_unnorm_keys": ["starwam"],
                "default_unnorm_key": "starwam",
                "action_layout": "native_qpos",
            }

        def uses_framework_action_unnormalization(self):
            return True

        def predict_action(self, examples, **kwargs):
            return {"normalized_actions": np.ones((1, 4, 2), dtype=np.float32)}

        def unnormalize_actions(self, actions, **kwargs):
            return actions * 3.0

    config = {"framework": {"name": "StarWAM"}}
    fake = FakeStarWAM()
    with (
        mock.patch.object(policy_wrapper, "read_mode_config", return_value=(config, {"starwam": {}})),
        mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=fake) as loader,
    ):
        wrapper = policy_wrapper.PolicyServerWrapper("unused.pt", device="cpu")
        result = wrapper.predict_action([{"image": [], "lang": "test"}])
    loader.assert_called_once_with("unused.pt", device="cpu", dtype=None)
    np.testing.assert_array_equal(result["actions"], np.full((1, 4, 2), 3.0, dtype=np.float32))
    assert wrapper.metadata["action_layout"] == "native_qpos"


@pytest.mark.skipif(not _TRAINER_DEPS_AVAILABLE, reason="full StarVLA trainer dependencies are not installed")
def test_starwam_schedule_and_optimizer_use_effective_batch_and_trainable_set() -> None:
    from starVLA.training.train_starvla import resolve_training_schedule, setup_optimizer_and_scheduler

    cfg = OmegaConf.create(
        {
            "framework": {"name": "StarWAM"},
            "datasets": {"vla_data": {"per_device_batch_size": 5}},
            "trainer": {
                "max_train_steps": None,
                "num_epochs": 2,
                "num_warmup_steps": None,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine_with_min_lr",
                "scheduler_specific_kwargs": {"min_lr": 1.0e-6},
                "learning_rate": {"base": 1.0e-4},
                "optimizer": {
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "eps": 1.0e-8,
                    "fused": False,
                },
            },
        }
    )
    accelerator = SimpleNamespace(num_processes=2, gradient_accumulation_steps=4)
    dataloader = SimpleNamespace(dataset=range(100))
    resolve_training_schedule(cfg, dataloader, accelerator)
    assert cfg.trainer.max_train_steps == 6
    assert cfg.trainer.num_warmup_steps == 0

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trainable = torch.nn.Linear(2, 2)
            self.frozen = torch.nn.Linear(2, 2)
            self.frozen.requires_grad_(False)

    model = TinyModel()
    optimizer, _ = setup_optimizer_and_scheduler(model, cfg)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.trainable.parameters()}

    legacy_cfg = OmegaConf.create({"framework": {"name": "Legacy"}, "trainer": {"max_train_steps": None}})
    resolve_training_schedule(legacy_cfg, SimpleNamespace(), accelerator)
    assert legacy_cfg.trainer.max_train_steps is None
