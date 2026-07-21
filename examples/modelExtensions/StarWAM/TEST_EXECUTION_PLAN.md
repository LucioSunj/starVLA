# StarWAM 集成 StarVLA 测试执行计划

本文档是 StarWAM 三类模型接入 StarVLA 后的可执行测试手册。它不仅列出要测什么，
也规定测试顺序、命令模板、证据、判定标准、失败分流和发布前归档要求。

适用基线：

- StarVLA 集成分支：`codex/starwam-integration`。
- StarVLA 分支基线：`a060fd97ddb7e4163d4e4fb17271e4bde74ea3c7`。
- StarWAM 固定版本：`a7b05c8da8f8c88bf4888aa737b938af3b67ab48`。
- 测试对象：MoT、Shared-DiT、Feature-conditioned。
- 明确不在本轮范围内：LAPA、latent-action tokenizer、LoRA、staged unfreezing。

测试必须在候选提交上执行。不能只记录分支名，也不能用有未记录修改的工作区结果作为合并
或发布证据。

## 1. 验证目标

本轮验证回答以下问题：

1. StarWAM 未安装时，StarVLA 的旧框架是否仍可导入和训练。
2. 六份薄配置是否解析为正确的 StarWAM recipe，CLI、StarVLA 显式字段和 recipe 的
   优先级是否符合约定。
3. StarWAM 的 tensor-dict 数据是否未经旧 PIL collate 改写，并保持视频、动作、文本、
   mask、padding 和 proprio 的契约。
4. `video_loss` 是否只作为 auxiliary metric 出现一次，Trainer 是否只反传 StarWAM 已经
   加权的 `total_loss`。
5. 三个模型族的冻结集合、优化器参数、梯度累积、bf16、scheduler、保存频率和
   global-step 语义是否正确。
6. 原生 StarWAM checkpoint、DeepSpeed payload 和 StarVLA wrapper checkpoint 是否可以按
   约定加载，且关键权重缺失时会失败。
7. StarVLA policy server 的相机拼接、state 归一化、动作反归一化、action horizon 和
   RoboTwin 原生 qpos 顺序是否正确。
8. 在同一 checkpoint、输入和随机种子下，StarVLA adapter 与原生 StarWAM 的动作块是否
   数值一致。
9. LIBERO 和 RoboTwin 的闭环评测是否能够完成，并且只有在两端 rollout 控制参数一致时
   才比较成功率。
10. 旧 action-only framework、旧 normalization 路径和旧 benchmark client 是否无回归。

本文所称“两步训练”均指两个 optimizer/global step，不是两个 micro-batch。梯度累积为
`N` 时，每个 rank 至少要完成 `2 * N` 个 micro-batch。

## 2. 测试门禁

| 门禁 | 内容 | 合并要求 | 发布要求 |
| --- | --- | --- | --- |
| G0 | 静态检查、CPU 单元测试、旧路径回归 | 必须通过 | 必须通过 |
| G1 | 六份真实 recipe 的 bf16 ZeRO-2 两个 optimizer step | 必须通过 | 必须通过 |
| G2 | 保存、严格恢复、固定 seed 推理和原生 checkpoint 导入 | 必须通过 | 必须通过 |
| G3 | 原生 StarWAM 与 StarVLA 离线动作一致性 | 必须通过 | 必须通过 |
| G4 | LIBERO/RoboTwin 小规模闭环 smoke | 至少每个 benchmark 一项 | 必须通过完整矩阵 |
| G5 | 50-trial 或 benchmark 原生 episode 数的正式评测 | 不阻塞纯代码评审 | 必须归档 |

以下失败不能豁免：loss 重复加权、非有限 loss/gradient、rank hang、wrapper 严格恢复失败、
像素不一致、动作维度或顺序错误、旧 framework 回归。

Cosmos 没有对应发布 checkpoint 时，可以不设置绝对成功率门槛，但真实权重构建、前后向、
checkpoint、离线推理和小规模 rollout 仍必须通过。

## 3. 配方矩阵与预期契约

| ID | 薄配置 | 模型族/骨干 | 输出动作 | 输入视频最终几何 | `video_loss` |
| --- | --- | --- | --- | --- | --- |
| SW-L-MW | `libero_mot_wan22.yaml` | MoT/Wan2.2 | `[B,32,7]` | `[B,3,9,224,448]` | 参与总损失 |
| SW-L-MC | `libero_mot_cosmos.yaml` | MoT/Cosmos | `[B,32,7]` | `[B,3,9,320,640]` | 参与总损失 |
| SW-L-SW | `libero_shared_dit_wan22.yaml` | Shared-DiT/Wan2.2 | `[B,32,7]` | `[B,3,9,160,320]` | 参与总损失 |
| SW-L-SC | `libero_shared_dit_cosmos.yaml` | Shared-DiT/Cosmos | `[B,32,7]` | `[B,3,9,320,640]` | 参与总损失 |
| SW-L-FW | `libero_feature_conditioned_wan22.yaml` | Feature-conditioned/Wan2.2 | `[B,32,7]` | `[B,3,9,224,448]` | 必须为 0 |
| SW-R-MW | `robotwin_mot_wan22.yaml` | MoT/Wan2.2 | `[B,32,14]` | `[B,3,9,384,320]` | 参与总损失 |

所有配方还应满足：

- `context` 为 `[B,text_len,text_dim]`，`context_mask` 为 `[B,text_len]`。
- `action_is_pad` 为 `[B,32]`，`image_is_pad` 为 `[B,9]`。
- proprio 为 `[B,32,8]`（LIBERO）或 `[B,32,14]`（RoboTwin）。
- LIBERO 相机顺序为 primary、wrist，横向拼接。
- RoboTwin 相机顺序为 head、left wrist、right wrist，布局为上方 head、下方左右 wrist。
- MoT/Shared-DiT 使用
  `total = lambda_action * action_loss + lambda_video * video_loss`。
- Feature-conditioned 使用 `total = lambda_action * action_loss`，且报告的
  `video_loss` 绝对值不超过 `1e-8`。

## 4. 环境与资产准备

### 4.1 冻结候选版本

从 StarVLA 仓库根目录执行：

```bash
export STARVLA_ROOT=/absolute/path/to/starVLA
export STARWAM_ROOT=/absolute/path/to/StarWAM
cd "${STARVLA_ROOT}"

test "$(git branch --show-current)" = "codex/starwam-integration"
git merge-base --is-ancestor c00b8e18c1b5ba6228fd68a8f38acca8df4e938a HEAD
test "$(git -C "${STARWAM_ROOT}" rev-parse HEAD)" = \
  a7b05c8da8f8c88bf4888aa737b938af3b67ab48
test -z "$(git -C "${STARWAM_ROOT}" status --porcelain)"
test -z "$(git status --porcelain)"

git rev-parse HEAD
git status --short --branch
git diff --check
```

正式 GPU 门禁开始前，StarVLA 候选改动必须已经形成 commit。记录 commit、分支、
`git status` 和 StarWAM revision。测试过程中不更新任何一个仓库，也不修改 StarWAM。

### 4.2 Python 环境

在已有 StarVLA 完整训练环境中安装可选依赖：

```bash
cd "${STARVLA_ROOT}"
python -m pip install -e '.[wam]'
python -m pip install --no-deps -e "${STARWAM_ROOT}[train]"
python -m pip install pytest ruff

export PYTHONPATH="${STARVLA_ROOT}:${STARWAM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
```

`pip install -e '.[wam]'` 用于安装固定依赖，随后 sibling editable 安装用于验证本地固定
revision。最终归档必须证明 `import starwam` 实际指向 `${STARWAM_ROOT}`，而不是另一个
site-packages 副本。

```bash
python -c 'import pathlib, starwam; print(pathlib.Path(starwam.__file__).resolve())'
python -c 'import torch, accelerate, transformers; print(torch.__version__, accelerate.__version__, transformers.__version__)'
accelerate env
nvidia-smi
```

CPU 门禁环境必须具有 StarWAM、Transformers、OpenCV、Matplotlib 和完整 StarVLA trainer
依赖。否则 `pytest -ra` 中的 skip 必须被视为环境未准备完成，而不是通过。

### 4.3 模型与数据路径

执行者需要定义：

```bash
export WAN22_ROOT=/models/Wan2.2-TI2V-5B
export COSMOS_ROOT=/models/Cosmos-Predict2-2B-Video2World
export WAN_ACTION_DIT_INIT=/models/preprocessed/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export COSMOS_ACTION_DIT_INIT=/models/preprocessed/starwam_action_dit_init_cosmos_predict2_notimeproj.pt

export LIBERO_SPATIAL_DATA=/data/libero_spatial_lerobot
export LIBERO_OBJECT_DATA=/data/libero_object_lerobot
export LIBERO_GOAL_DATA=/data/libero_goal_lerobot
export LIBERO_10_DATA=/data/libero_10_lerobot
export ROBOTWIN_DATA=/data/robotwin2.0

export TEST_RUN_ROOT=/runs/starwam-integration/$(date +%Y%m%d_%H%M%S)
mkdir -p "${TEST_RUN_ROOT}/logs" "${TEST_RUN_ROOT}/artifacts"
```

Cosmos recipe 使用的 LIBERO 数据版本必须与原 recipe 注释一致。若实际目录是 no-noop 版本，
四个 `LIBERO_*_DATA` 都必须指向同一数据版本系列，不能混用普通版和 no-noop 版。

每个 case 使用独立的 text cache 目录。Wan 和 Cosmos 的文本编码器 ID 不同，禁止共享缓存。
每个 case 也使用独立 stats 文件，以便审计其数据源和维度。

### 4.4 分布式配置

本计划默认使用：

```text
starVLA/config/deepseeds/zero2.yaml
```

该配置使用 Accelerate 的内联 ZeRO-2 设置，StarVLA 可以按 recipe 注入 gradient
accumulation。若平台必须使用 `deepspeed_zero2.yaml` 或外部 DeepSpeed JSON，必须确认外部
配置没有把 `gradient_accumulation_steps` 固定为与 recipe 不同的值，并在日志中保存最终
DeepSpeed 配置。

单机测试至少使用两张 GPU。正式结果记录 GPU 型号、数量、显存、driver、CUDA、PyTorch、
NCCL、Accelerate 和 DeepSpeed 版本。

## 5. 阶段 A：静态检查与 CPU 合同测试

### A1. 静态检查

```bash
cd "${STARVLA_ROOT}"

ruff check \
  deployment/model_server/policy_wrapper.py \
  deployment/model_server/tools/websocket_policy_client.py \
  examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py \
  examples/simBenchmarks/Robotwin/eval_files/model2robotwin_interface.py \
  starVLA/dataloader/starwam_datasets.py \
  starVLA/model/framework/WAM \
  starVLA/training/loss_utils.py \
  starVLA/training/train_starvla.py \
  tests/test_starwam_integration.py

python -m compileall -q \
  deployment/model_server \
  examples/simBenchmarks/LIBERO/eval_files \
  examples/simBenchmarks/Robotwin/eval_files \
  starVLA tests

python -c 'import pathlib, tomllib; tomllib.loads(pathlib.Path("pyproject.toml").read_text())'
git diff --check
```

通过标准：全部返回 0，无语法错误、TOML 错误或 whitespace error。

### A2. StarWAM 集成测试

```bash
cd "${STARVLA_ROOT}"
USE_TF=0 USE_FLAX=0 PYTHONPATH="${STARVLA_ROOT}:${STARWAM_ROOT}" \
  pytest -q -ra tests/test_starwam_integration.py \
  2>&1 | tee "${TEST_RUN_ROOT}/logs/cpu_starwam_integration.log"
```

当前候选基线为 32 个通过项。后续增加测试时数量可以增长，因此真正门槛是 0 failed、
0 error，并且没有因为 StarWAM、Transformers、trainer、OpenCV 或 Matplotlib 缺失而产生的
skip。

该文件必须覆盖：

| 合同 | 对应验证 |
| --- | --- |
| 可选依赖 | 轻量模块不提前导入 StarWAM；缺依赖错误可操作 |
| 配置 | 优先级、override、self-contained snapshot、六份薄配置 |
| 不支持项 | LAPA、LoRA、staged、手工 freeze 在建模前失败 |
| loss | `total_loss` 优先、旧 `action_loss` fallback、video loss 不重复相加 |
| 数据 | synthetic shape、future frame 窗口、mask、padding、Shared-DiT proprio |
| 图像 | LIBERO 横拼、RoboTwin 三相机布局逐像素一致 |
| normalization | min-max、z-score、clip 和往返 |
| checkpoint | prefix 适配、零匹配失败、关键组缺失失败、wrapper 严格恢复 |
| serving | metadata、framework 反归一化、旧 normalization 路径 |
| 回归 | 旧 action-only framework 单步训练和旧 client 行为 |

### A3. StarVLA 现有测试

在包含 DeepSpeed 的完整训练环境中执行完整测试集：

```bash
USE_TF=0 USE_FLAX=0 PYTHONPATH="${STARVLA_ROOT}:${STARWAM_ROOT}" \
  pytest -q -ra tests \
  2>&1 | tee "${TEST_RUN_ROOT}/logs/cpu_all_tests.log"
```

只有最小本地环境、暂时没有 DeepSpeed 时，可以先运行以下非替代性检查：

```bash
pytest -q \
  tests/test_single_process_dist_safety.py::SingleProcessDistSafetyTest::test_trainer_utils_safe_without_process_group \
  tests/test_single_process_dist_safety.py::SingleProcessDistSafetyTest::test_train_starvla_prepare_data_safe_without_process_group \
  tests/test_robocasa_tabletop_interface.py
```

最小检查不能替代完整测试。`train_starvlm.py` 和 `train_starvla_cotrain.py` 的单进程测试在
缺少 DeepSpeed 的环境中可能在 import 阶段失败；应在完整环境重跑，而不是把这类环境失败
归因于本次 WAM 修改。

### A4. 无 StarWAM 环境回归

另建一个不安装 `starwam` 的轻量环境，至少验证旧框架注册和轻量模块导入。选择
`framework.name: StarWAM` 时必须只得到带安装命令的错误；不选择 StarWAM 时不能因为
optional dependency 缺失而失败。

通过标准：旧 framework import 成功，StarWAM 选择路径的异常明确包含 `.[wam]` 或 sibling
editable 安装提示。

## 6. 阶段 B：真实数据预检

这一步可由首次 GPU smoke 的 rank 0 完成，但在模型前向开始前必须检查以下内容。

### B1. cache 与统计量

1. 只有 rank 0 生成 text cache 和 action/state statistics。
2. 其他 rank 在 barrier 后继续，不重复写同一文件。
3. stats JSON 可解析，包含 `action` 和 `state`，维度分别为 7/8（LIBERO）或 14/14
   （RoboTwin）。
4. StarVLA run 根目录生成 `dataset_statistics.json`，顶层包含 `starwam`。
5. recipe 内解析后的 stats 路径也写入 `config.yaml` 的
   `framework.starwam.resolved.data`。
6. 同一 case 第二次启动时复用 cache/stats，不应重复完整编码或统计。

stats 或 cache 写入中断后，不得手工把半成品当作成功结果。删除对应 case 的临时产物并由
rank 0 重建。

### B2. 首批数据

对每个 case 保存首个 batch 的 shape、dtype、数值范围和 padding 比例。不得保存包含敏感
数据的完整原始轨迹，只保存必要诊断信息或经过批准的固定 fixture。

通过标准：

- `video` 是 float tensor，布局为 `[B,3,9,H,W]`，不是 PIL list。
- `action`、`context`、`context_mask`、`action_is_pad`、`image_is_pad` 和 `proprio` 均存在。
- `video` 的 camera order、最终几何和表 3 一致。
- action/state normalization 模式分别为 LIBERO min-max、RoboTwin z-score。
- Shared-DiT 缺少 proprio 时立刻报错，不能用零向量静默替代。
- DataLoader 为 `shuffle=True`、`drop_last=True`；worker 大于 0 时启用 persistent worker 和
  recipe 对齐的 prefetch。
- 每个 rank 至少有一个完整 batch；`drop_last=True` 后长度为 0 必须失败。

## 7. 阶段 C：六配方 GPU/ZeRO-2 冒烟

### C1. override 模板

LIBERO MoT/Wan 示例：

```bash
CASE_ID=SW-L-MW
CONFIG=examples/modelExtensions/StarWAM/train_files/libero_mot_wan22.yaml
RUN_ID="${CASE_ID}_$(date +%Y%m%d_%H%M%S)"
OVERRIDES="[\"backbone.pretrained_model_id=${WAN22_ROOT}\",\"framework.action_expert_init_from=${WAN_ACTION_DIT_INIT}\",\"data.dataset_dirs=[${LIBERO_SPATIAL_DATA},${LIBERO_OBJECT_DATA},${LIBERO_GOAL_DATA},${LIBERO_10_DATA}]\",\"data.text_embedding_cache_dir=${TEST_RUN_ROOT}/cache/${CASE_ID}\",\"data.action_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\",\"data.state_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\"]"
```

Shared-DiT 和 Feature-conditioned 不传 `framework.action_expert_init_from`：

```bash
OVERRIDES="[\"backbone.pretrained_model_id=${WAN22_ROOT}\",\"data.dataset_dirs=[${LIBERO_SPATIAL_DATA},${LIBERO_OBJECT_DATA},${LIBERO_GOAL_DATA},${LIBERO_10_DATA}]\",\"data.text_embedding_cache_dir=${TEST_RUN_ROOT}/cache/${CASE_ID}\",\"data.action_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\",\"data.state_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\"]"
```

RoboTwin MoT 使用：

```bash
OVERRIDES="[\"backbone.pretrained_model_id=${WAN22_ROOT}\",\"framework.action_expert_init_from=${WAN_ACTION_DIT_INIT}\",\"data.dataset_dirs=[${ROBOTWIN_DATA}]\",\"data.text_embedding_cache_dir=${TEST_RUN_ROOT}/cache/${CASE_ID}\",\"data.action_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\",\"data.state_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\"]"
```

Cosmos 两项将骨干替换为 `${COSMOS_ROOT}`。MoT/Cosmos 还必须使用
`${COSMOS_ACTION_DIT_INIT}`。每次启动前把最终 `OVERRIDES` 原样写入 case manifest。

### C2. 两 optimizer step 命令

```bash
set -o pipefail
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch \
  --config_file starVLA/config/deepseeds/zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.starwam.overrides="${OVERRIDES}" \
  --run_root_dir "${TEST_RUN_ROOT}/train" \
  --run_id "${RUN_ID}" \
  --trainer.max_train_steps 2 \
  --trainer.save_interval 1 \
  --trainer.eval_interval 1 \
  --trainer.eval_max_samples 1 \
  --trainer.eval_num_inference_steps 2 \
  --trainer.eval_action_num_inference_steps 2 \
  --trainer.eval_compute_video_psnr false \
  --trainer.logging_frequency 1 \
  --trainer.gradient_probe_enabled true \
  --trainer.gradient_probe_step 1 \
  --trainer.wandb_enabled false \
  2>&1 | tee "${TEST_RUN_ROOT}/logs/${CASE_ID}_train.log"
```

依次执行六个 ID。正式门禁不能因为 OOM 静默降低 batch 或 accumulation。可以用更小 batch
做故障定位，但该结果只能标记为 diagnostic。若目标硬件确实需要改变 batch，必须同步调整
accumulation 以维持 recipe 的 global batch，并把原值、最终值和原因写入 manifest；是否接受
该偏差由合并负责人明确签字。

不要给 RoboTwin 正式 rollout 外层加 wall-clock timeout。训练 smoke 可以由作业调度器设置
合理超时；超时退出属于失败，不能当作资源跳过。

### C3. 每个作业的通过标准

日志必须出现 `StarWAM effective values`，且最终值满足：

- `max_steps=2`、`precision=bf16`、`fused_adamw=False`。
- batch、accumulation、learning rate、weight decay 与 recipe 或已批准 override 一致。
- scheduler 按 optimizer step 前进，两个 global step 后只前进两次。
- warmup 被限制在有效的两步区间内，不能按 micro-batch 重复前进。
- 两个 step 的 `total_loss`、`action_loss`、`video_loss`、action MSE 和 gradient 都有限。
- MoT/Shared-DiT 的已报告总损失在 `rtol=1e-4, atol=1e-5` 内满足加权公式。
- Feature-conditioned 的 `video_loss` 为 0，且没有 video loss 梯度分支。
- 每次 optimizer boundary 才执行 clip、step、scheduler step 和 zero-grad。
- 多 rank 无 NCCL hang、barrier hang、重复 cache 写入或 dataloader worker timeout。

输出目录必须包含：

```text
<run>/config.full.yaml
<run>/config.yaml
<run>/dataset_statistics.json
<run>/gradient_probe.json
<run>/summary.jsonl
<run>/checkpoints/steps_1_pytorch_model.pt
<run>/checkpoints/steps_2_pytorch_model.pt
<run>/final_model/pytorch_model.pt
```

`config.yaml` 必须包含完整 `framework.starwam.resolved`、`source_revision`、
`runtime_revision`、`pinned_revision` 和实际 stats/cache 路径。原 recipe 路径可以保留用于审计，
但恢复不能依赖它仍然存在。三个 revision 字段在本门禁中都应等于固定的 StarWAM revision；
任何不一致都必须停止测试并确认实际 import 来源。

### C4. 梯度探针

只看 `requires_grad=True` 不足以证明计算图连通。首个 optimizer boundary 在 clip 之前记录以下
参数组的 `grad_norm`、有限元素比例和非零元素比例，写入 `gradient_probe.json`：

| 模型族 | 必须有有限非零梯度 | 必须保持冻结 |
| --- | --- | --- |
| MoT | video/action expert 的 MoT 层、proprio/action 小头 | VAE、text encoder |
| Shared-DiT | shared DiT、action/state encoder/decoder/projection | VAE、text encoder |
| Feature-conditioned | ActionDiT、feature projector、recipe 指定的 backbone DiT | VAE、text encoder |

探针应按参数名前缀聚合，避免保存完整 gradient。某一参数恰好为零不构成失败，但每个“必须
有梯度”的功能组都至少要有一个参数得到有限非零梯度。冻结组出现 gradient 直接失败。

上述 smoke 命令通过 `trainer.gradient_probe_enabled=true` 在第一个 optimizer boundary、梯度裁剪
之前自动写出 `gradient_probe.json`。探针按功能模块聚合参数量、梯度元素量、L2 norm、有限元素
比例和非零元素比例，并分块扫描大张量以限制额外显存；不能用“loss 在变”替代梯度组验证。

### C5. 可选 video PSNR

在至少一个 MoT 和一个 Shared-DiT case 上，把
`trainer.eval_compute_video_psnr=true` 再跑一个 optimizer step。`video_psnr` 必须存在且有限。
Feature-conditioned 不应要求该指标。

## 8. 阶段 D：checkpoint 与自包含恢复

### D1. wrapper 严格恢复

对六个 `steps_2_pytorch_model.pt` 分别调用 `baseframework.from_pretrained`。加载必须是
`strict=True` 语义，0 missing key、0 unexpected key。

固定一份 raw image、language、state 和 seed，比较保存前与恢复后的：

- normalized action chunk；
- framework 反归一化后的 action chunk；
- metadata。

动作比较阈值为 `atol=1e-4, rtol=1e-3`。shape、dtype、camera order、horizon 和 metadata
字段要求完全一致。

### D2. 脱离原 recipe 路径恢复

为每个 case 建立独立 bundle：

```text
<bundle>/config.yaml
<bundle>/dataset_statistics.json
<bundle>/checkpoints/steps_2_pytorch_model.pt
```

将 `config.yaml` 中的 `framework.starwam.recipe` 改为一个确定不存在的路径，但保留
`framework.starwam.resolved`。从 bundle checkpoint 恢复并推理。成功表示 snapshot 自包含；
如果仍尝试读取原 recipe，则失败。

测试只能修改 bundle 副本，不能改训练原始产物。

### D3. 原生 StarWAM/DeepSpeed 导入矩阵

| payload | 设置 | 预期 |
| --- | --- | --- |
| 原生 state dict | `strict_init=true`，同架构 | 完全加载 |
| `model_state_dict`/`state_dict`/`module` 包装 | `strict_init=false` | 正确提取 |
| `module.`、`model.`、`_orig_mod.`、`wam.` 前缀 | `strict_init=false` | 递归去前缀 |
| DeepSpeed `mp_rank_00_model_states.pt` | directory path | 正确发现并加载 |
| safetensors 单文件或 shards | file/directory | 正确加载 |
| 完全无匹配权重 | 任意 | 必须失败 |
| 缺 MoT/Shared-DiT/action expert/feature projector 关键组 | non-strict | 必须失败 |

对一个真实发布的原生 StarWAM checkpoint 至少执行一次，而不能只依赖合成小模型单测。

## 9. 阶段 E：离线原生实现一致性

这是模型集成的核心数值门禁。它与 simulator 成功率分开执行，以免环境控制差异掩盖模型
问题。

### E1. 固定 golden input

每个 benchmark 至少保存一个获准归档的输入：

- LIBERO：primary、wrist 两张原始 `uint8 HWC` 图，8 维 state，instruction。
- RoboTwin：head、left wrist、right wrist 三张原始 `uint8 HWC` 图，14 维 native qpos，
  instruction。
- 输入记录 task、episode、environment seed 和观测时刻。

两条路径都从同一份原始数组开始，禁止一边读取 simulator、一边读取训练 batch。

### E2. 分层比较

按以下顺序比较，第一层不一致时停止，不继续解释后续 action 差异：

1. **像素层**：比较 StarVLA `compose_camera_views` 与原生 StarWAM benchmark adapter 的模型
   输入。要求 shape 相同且 `torch.equal`，即 `max_abs_diff=0`。
2. **文本层**：比较 prompt、encoder ID、cache key、context 和 mask。固定 cache 时要求 tensor
   一致；重新编码时至少要求 prompt、mask 和数值 tolerance 一致。
3. **state 层**：比较裁剪维度、顺序和 normalization 后的 proprio，要求
   `atol=1e-6, rtol=1e-6`。
4. **模型层**：同一原生 checkpoint、bf16、inference step、action step 和 seed，比较
   `infer_action` 的 normalized chunk。
5. **动作层**：比较 min-max/z-score 反归一化后的 chunk。
6. **服务层**：比较 policy server 返回的 action 与直接 framework 调用。

第 4 至 6 层统一门槛为 `atol=1e-4, rtol=1e-3`，同时要求最大绝对误差、最大相对误差和
失败元素数量写入 `parity.json`。

原生 `starwam.eval.policy.StarwamPolicy.predict_chunk` 返回已经反归一化的动作，而 StarVLA
`StarWAMFramework.predict_action` 返回 normalized action。因此 normalized 层应直接比较两端
的 `wam.infer_action`，最终动作层再把 `StarwamPolicy.predict_chunk` 与
`StarWAMFramework.unnormalize_actions` 比较，不能把两个不同层级直接做 allclose。

### E3. checkpoint 选择

原生一致性优先使用同一个原生 StarWAM checkpoint：

- 原生路径直接加载该 checkpoint。
- StarVLA 路径通过 `framework.starwam.init_checkpoint` 导入该 checkpoint。

StarVLA wrapper checkpoint 另做“保存前对保存后”的 D1 比较。除非先明确剥离 `wam.` 前缀，
不要把 wrapper state 直接交给旧原生 loader 后声称是同一 checkpoint 测试。

### E4. 训练 loss 一致性补充

在一份固定 tensor batch 上重置 Python、NumPy、CPU CUDA RNG，再分别调用原生 WAM 和
StarVLA wrapper。`total_loss`、action component 和 video component 应在
`atol=1e-5, rtol=1e-4` 内一致。该测试验证 wrapper 没有改变噪声、timestep 或加权逻辑。

## 10. 阶段 F：policy server

### F1. 启动

```bash
export CKPT="${TEST_RUN_ROOT}/train/<RUN_ID>/checkpoints/steps_2_pytorch_model.pt"
export PORT=10093

CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  --idle_timeout 600 \
  2>&1 | tee "${TEST_RUN_ROOT}/logs/policy_server.log"
```

启动后先读取 websocket handshake metadata，再发送 golden input。响应必须为
`status=ok`，`data.actions` shape 为 `[1,32,7]` 或 `[1,32,14]`，且所有元素有限。

### F2. metadata 判定

两类 benchmark 都要求：

- `action_chunk_size=32`；
- `available_unnorm_keys=["starwam"]`；
- `default_unnorm_key="starwam"`；
- `framework_preprocesses_images=true`；
- `state_required=true`；
- `camera_order` 与 recipe 的 `video_keys` 一致；
- `model_family` 与薄配置一致。

LIBERO 额外要求 `camera_layout=horizontal` 和 `action_layout=default`。RoboTwin 额外要求
`camera_layout=robotwin` 和 `action_layout=native_qpos`。

RoboTwin 客户端必须保持原始 14 维 qpos 顺序，不能执行旧 StarVLA 的
`[0..5,12,6..11,13]` 重排。服务返回动作与客户端实际提交给环境的动作必须逐维比较。

### F3. 错误路径

还要主动验证：

- 缺 state 时，三族都给出维度明确的错误。
- 相机数量不符时，server 返回 error，不静默丢弃视角。
- 目前 raw image list 本身不携带相机名称，server 无法自动判断“数量正确但顺序错误”。该
  情况通过 metadata、golden input 和 E2 的逐像素比较判定，不能写成已由 server 拦截。
- 错误 `unnorm_key` 被拒绝。
- bundle 缺 `dataset_statistics.json` 时启动失败。
- action/state stats 维度小于 recipe 要求时推理失败。

## 11. 阶段 G：仿真闭环

### G1. 比较前先对齐 rollout 控制

只有下列参数一致时，才允许把两套驱动的 success count 当作模型一致性证据：

| 控制项 | 必须对齐的内容 |
| --- | --- |
| 环境 | simulator commit、依赖、assets、renderer |
| 初始条件 | task、initial-state index、environment seed |
| 模型随机性 | diffusion seed、fixed/per-episode 策略 |
| 推理 | video steps、action steps、bf16/fp32 |
| 控制 | num-steps-wait、replan-steps、action chunk 使用方式 |
| episode | max steps、done 判定、gripper 后处理 |
| 观测 | 相机顺序、旋转、resize、state 顺序 |

当前 StarVLA LIBERO/RoboTwin clients 按 `action_chunk_size=32` 缓存完整 action chunk；原生
StarWAM 的常用设置分别可能是 LIBERO `replan_steps=10`、RoboTwin
`replan_steps=24`。直接用不同 replan 策略比较成功率没有解释力。确定性 parity run 应把原生
driver 设为 32，或先给 StarVLA client 增加等价的可配置 replan 策略。正式复现论文/发布
结果时，两端则都应使用目标配方的 replan 设置。

LIBERO 两套 driver 的默认 max steps 也不同。进行 paired comparison 时，应把原生
`--max-steps` 设为 StarVLA 当前 suite 值：spatial 220、object 280、goal 300、libero_10 520。

### G2. LIBERO 快速 smoke

终端 1 启动 F1 server。终端 2：

```bash
export LIBERO_HOME=/absolute/path/to/LIBERO
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${STARVLA_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

python examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.pretrained-path "${CKPT}" \
  --args.task-suite-name libero_goal \
  --args.num-trials-per-task 1 \
  --args.max-tasks 1 \
  --args.seed 42 \
  --args.video-out-path "${TEST_RUN_ROOT}/libero_smoke"
```

对 MoT/Wan、Shared-DiT/Wan、Feature-conditioned/Wan 各完成至少一个 episode。Cosmos 两项
各完成至少一次一个 task/一个 trial 的小规模 rollout。smoke 门槛是无异常完成、动作有限、
视频可解码；一两个 episode 的 success/failure 不作为模型质量门槛。

原生 StarWAM paired run 使用 `${STARWAM_ROOT}/examples/libero/rollout.py`，并显式设置同一
recipe、checkpoint、suite、task、trial、seed、wait、replan 和 max steps。先从 StarVLA
snapshot 导出实际 resolved recipe，避免漏掉训练时 override：

```bash
export RUN_DIR="${TEST_RUN_ROOT}/train/<RUN_ID>"
python -c 'import os; from omegaconf import OmegaConf; c=OmegaConf.load(os.environ["RUN_DIR"] + "/config.yaml"); OmegaConf.save(c.framework.starwam.resolved, os.environ["RUN_DIR"] + "/resolved_starwam_recipe.yaml", resolve=True)'
```

例如确定性比较：

```bash
python "${STARWAM_ROOT}/examples/libero/rollout.py" \
  --config "${RUN_DIR}/resolved_starwam_recipe.yaml" \
  --checkpoint /absolute/path/to/native_starwam_checkpoint \
  --libero-home "${LIBERO_HOME}" \
  --task-suite-name libero_goal \
  --task-id 0 \
  --num-trials 1 \
  --num-steps-wait 10 \
  --replan-steps 32 \
  --max-steps 300 \
  --seed 42 \
  --fixed-seed \
  --device cuda:0
```

### G3. LIBERO 发布矩阵

Wan 三族分别运行 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，每个 task
50 trials。与原生 StarWAM 做 paired comparison 时，每个 task 的成功次数差不超过 1。

同时报告：

- 每 task successes/trials；
- 每 suite micro success rate；
- 四 suite 总 micro 和 macro average；
- integration 与 native 的逐 task 差值；
- crash、invalid action、server reconnect 和超时次数。

任何一端 episode 未完成时，不得缩小分母或只比较共同完成部分。应先修复执行失败后重跑
同一完整 task。

### G4. RoboTwin 快速 smoke

StarVLA 自带 runner：

```bash
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin
export STARVLA_PYTHON=/absolute/path/to/starvla-env/bin/python
export ROBOTWIN_PYTHON=/absolute/path/to/robotwin-env/bin/python
export CKPT="${TEST_RUN_ROOT}/train/<SW-R-MW_RUN_ID>/checkpoints/steps_2_pytorch_model.pt"

cd "${STARVLA_ROOT}/examples/simBenchmarks/Robotwin/eval_files"
bash start_eval.sh \
  --mode demo_clean \
  --name starwam_integration_smoke \
  --ckpt "${CKPT}" \
  --seed 0 \
  adjust_bottle
```

第三方 RoboTwin checkout 必须包含 `policy_ckpt_path` patch，server metadata 必须显示
`native_qpos`。检查 server log、eval log 和 RoboTwin `_result.txt`。RoboTwin 当前 runner
通常对一个 task-setting 执行 100 episodes，因此这里的“快速”是缩小到一个 task，而不一定
是短 wall time；不要在中途截断并把不完整结果标记为通过。

原生 StarWAM 常用 RoboTwin 评测为 50 tasks、`demo_clean` 和 `demo_randomized`、每个
task-setting 100 episodes、`instruction_type=unseen`、`replan_steps=24`。发布比较前应让
StarVLA 与原生 adapter 的 replan 策略一致；未对齐前，结果只记为闭环可运行证据，不能套用
“每 task 成功数差不超过 1”的 parity 门槛。

RoboTwin 使用其原生 episode 数作为正式分母。若发布流程另要求 50 trials，必须通过
RoboTwin 配置明确限制为相同的前 50 个 initial states，并在两端使用完全相同的选择规则。

## 12. 阶段 H：旧路径回归

除 A3 外，在具备一个旧 action-only 小 checkpoint 的环境执行：

1. 旧 framework 单个 optimizer step，确认仍从 `action_loss` 反传。
2. 旧 checkpoint 通过 policy server 启动并返回原有 horizon。
3. 旧模型继续使用 `PolicyNormProcessor`，不调用 WAM 的反归一化方法。
4. LIBERO/RoboTwin/Robocasa 客户端在 metadata 未声明 `state_required` 时不额外发送 state。
5. 非 `native_qpos` RoboTwin checkpoint 仍执行旧动作重排。
6. 旧 config 保存仍使用 accessed-config 行为；只有 WAM 强制保存完整 resolved recipe。

通过标准：loss、请求 payload、metadata、action shape 和 normalization 与集成前基线一致。

## 13. 失败分流

| 症状 | 首先检查 | 主要归属 | 处理 |
| --- | --- | --- | --- |
| 旧 framework import 失败 | 是否提前 import `starwam` | registry/lazy import | G0 失败 |
| recipe 找不到 | 薄配置相对路径、工作目录、snapshot | WAM config bridge | G0 失败 |
| override 未生效 | `config.yaml` 与 effective-values log | CLI/config precedence | G0/G1 失败 |
| stats/cache 多 rank 冲突 | rank 0 log、barrier、临时文件 | dataloader preparation | G1 失败 |
| dataloader 为 0 | 数据长度、batch、world size、drop-last | data config | G1 失败 |
| total loss 不满足公式 | lambda、原生 metrics、Trainer 是否重加 | loss contract | 立即停止 |
| loss/gradient NaN/Inf | 首个异常 step、bf16、输入范围、scheduler | model/trainer | G1 失败 |
| scheduler 过快 | micro-step 与 sync-gradients 次数 | accumulation lifecycle | G1 失败 |
| 只有一个 rank 前进 | cache barrier、worker timeout、NCCL log | distributed data/train | G1 失败 |
| strict restore missing key | wrapper 前缀、保存范围、config family | checkpoint | G2 失败 |
| pixel mismatch | resize 插值、相机顺序、旋转、值域 | runtime adapter | G3 失败 |
| normalized action mismatch | seed、steps、text/state、checkpoint 层级 | inference adapter | G3 失败 |
| direct 一致但 server 不一致 | stats key、反归一化、dtype | policy wrapper | G3 失败 |
| offline 一致但 rollout 分歧 | replan/wait/max-step/gripper/seed | benchmark adapter | 先对齐控制 |
| RoboTwin 关节错位 | `action_layout` 和 client permutation | RoboTwin adapter | 立即停止 |

发生失败时保留第一个失败 case 的完整日志和最小输入，不要先批量重跑。修复后从最早受影响
阶段开始重跑，而不是只重跑最终 rollout。

## 14. 结果归档

推荐目录：

```text
<TEST_RUN_ROOT>/
  manifest/
    source.txt
    environment.txt
    pip-freeze.txt
    accelerate-env.txt
    nvidia-smi.txt
    decision.md
  logs/
    cpu_*.log
    <CASE>_train.log
    <CASE>_server.log
    <CASE>_rollout.log
  artifacts/<CASE>/
    config.yaml
    config.full.yaml
    dataset_statistics.json
    overrides.txt
    batch_contract.json
    gradient_probe.json
    checkpoint_restore.json
    parity.json
    metadata.json
  rollout/
    libero_results.csv
    robotwin_results.csv
    failures/
```

`source.txt` 至少记录：

```text
starvla_revision=
starvla_branch=
starvla_status=
starwam_revision=
starwam_import_path=
test_plan_revision=
```

每个 case 的结果至少记录：

```text
case_id=
config=
resolved_recipe=
backbone_path_or_revision=
dataset_paths_and_revisions=
world_size=
per_device_batch=
gradient_accumulation=
effective_global_batch=
precision=
optimizer_steps=
checkpoint=
exit_code=
gate_result=PASS|FAIL|BLOCKED
```

大 checkpoint、原始数据和 rollout 视频不提交到源码仓库。归档位置应由项目实验存储管理，
源码仓库只保留结果摘要、manifest 和可追溯链接。

## 15. 执行顺序与签字

严格按以下顺序执行：

1. 冻结候选 commit 和环境。
2. A1 静态检查。
3. A2 集成 CPU 测试。
4. A3 完整旧测试，A4 无 WAM 回归。
5. B 真实数据预检。
6. C 六配方 GPU/ZeRO-2 smoke 和梯度探针。
7. D checkpoint 导入、保存、自包含严格恢复。
8. E 原生实现离线一致性。
9. F policy server 协议和错误路径。
10. G 小规模 simulator smoke。
11. H 旧 action-only 训练与服务回归。
12. 合并后执行 G 的正式发布矩阵并归档。

停止条件：任何核心合同失败、非有限数值、分布式 hang、严格恢复失败、像素/动作超 tolerance
或动作顺序错误时，停止后续性能评测。性能结果不能覆盖功能正确性失败。

最终 `decision.md` 应由训练执行者和集成评审者分别确认：

```text
[ ] G0 静态、CPU、旧路径回归通过
[ ] G1 六份 recipe 两个 bf16 ZeRO-2 optimizer step 通过
[ ] G2 原生导入、保存、自包含严格恢复通过
[ ] G3 离线像素、state、normalized/final action 一致
[ ] G4 LIBERO 和 RoboTwin 小规模闭环通过
[ ] Cosmos 真实权重构建、前后向、checkpoint、rollout 通过
[ ] 所有偏差、skip、资源 override 和已知限制均有记录
[ ] 发布 benchmark 结果与硬件/软件配置已归档
```
