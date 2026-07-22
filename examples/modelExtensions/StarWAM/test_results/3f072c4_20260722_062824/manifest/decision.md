# StarWAM 集成候选复测决策

- 执行时间：2026-07-22 UTC
- 正式 `TEST_RUN_ROOT`：`/home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824`
- 初始冻结候选：StarVLA `c3bbb7c9524cefbde4e36b5bbae5336ab7589bc9`
- 本轮正式复测候选：StarVLA `3f072c4717ca0c1402d39d5d0031ca58df597600`
- StarWAM：`a7b05c8da8f8c88bf4888aa737b938af3b67ab48`（detached、始终未修改）
- 正式运行前源码状态：两个仓库均 clean
- 最终结论：**不建议合并**

初始候选的结果已归档在 `c3bbb7c_20260721_074033`。该候选暴露的问题经过独立提交修复后，本报告重新冻结并验证 `3f072c4`；不能把本报告的通过项倒推为 `c3bbb7c` 已通过。

## Gate 结论

| Gate | 结果 | 依据 |
| --- | --- | --- |
| G0 | **PASS** | A1 全部 exit 0；集成测试 41 passed、完整 tests 74 passed，均 0 skip；无 StarWAM 环境和三个两卡 harness 通过。 |
| G1 | **BLOCKED** | SW-L-MW 完成两个 optimizer/global step；两个 Cosmos case 缺官方 gated 资产；其余三个 Wan case 在 G3 首层像素失败后按硬停止条件未再运行。 |
| G2 | **BLOCKED** | SW-L-MW strict restore、自包含 bundle、真实原生 checkpoint strict import 均通过；其余五个 case 没有 steps_2，不能把单 case 证据升级为总门禁通过。 |
| G3 | **FAIL** | 同一真实 LIBERO 原始输入的像素层不一致：shape 相同但 `torch.equal=false`，`max_abs_diff=0.5139676929`。按计划未继续文本/state/action/service 层。 |
| G4 | **BLOCKED** | 实际 checkpoint metadata 与 CPU client 合同通过；因 G3 像素硬失败，未启动真实 server action 推理或 rollout，未伪造成功。 |

## 候选修复范围

从初始冻结点到本轮正式候选的功能修复如下：

- `3c93550`：相邻 StarWAM checkout 的 recipe 路径解析。
- `d7a0d86`：移除 LIBERO rollout 遗留调试代码。
- `eac3cf9`：ZeRO-2 accumulation boundary，仅在 optimizer boundary 执行 clip/step/scheduler/zero-grad。
- `4e4bdf7`：通过 DeepSpeed 官方接口读取 ZeRO-2 分片梯度。
- `3f072c4`：ZeRO-2/DDP 非主 rank 不再重复 materialize 完整 state dict；ZeRO-3/FSDP 仍保持全 rank collective。

`cd782f7` 只归档初始候选测试结果，不改变运行时行为。

## G0

### A1 与 CPU tests

| 项目 | Exit code | 结果 |
| --- | ---: | --- |
| ruff、compileall、TOML、git diff --check | 0 | PASS |
| `tests/test_starwam_integration.py` | 0 | 41 passed，0 failed/error/skip |
| 完整 `tests` | 0 | 74 passed，0 failed/error/skip |
| 独立无 StarWAM 环境 | 0 | `starwam_spec=None`；旧 ACT/轻量模块可导入；选择 StarWAM 时提示 `.[wam]`/`--no-deps` |

不存在因 StarWAM、Transformers、trainer、OpenCV 或 Matplotlib 缺失而 skip 的项目。本轮正式 tests skip 总数为 **0**；环境缺失没有被解释为代码通过。

### 两卡正确性 harness

- `_infinite_dataloader`：world size 2，完整遍历 2 个 DataLoader epoch，每 rank 8 step，gathered `[8,8]`，每步 collective，旧 `_reset_dataloader` 调用 0，无 hang，exit 0。
- ZeRO-2 gradient probe：两个 rank 的 accumulation boundary 均为 `[false,true]`；原始 `param.grad` 在分片下为空，但 `safe_get_full_grad` 得到有限非零梯度；冻结 VAE 参数 0 trainable、0 gradient，exit 0。
- ZeRO-2 CPU optimizer：两个 rank 均为 `DeepSpeedCPUAdam`、`adamw_mode=true`、betas `(0.9,0.95)`、eps `1e-8`、weight decay `0.01`；boundary `[false,true]`，有限非零梯度并发生权重更新，exit 0。

相关结果见 `artifacts/infinite_dataloader/result.json`、`artifacts/zero2_gradient_probe/result.json` 和 `artifacts/zero2_cpu_offload/result.json`。

## 环境、硬件与资产

- Python 3.11.15
- PyTorch 2.6.0+cu124；CUDA runtime 12.4；NCCL 2.21.5
- Accelerate 1.5.2；DeepSpeed 0.16.9；Transformers 4.57.0
- OpenCV 4.11.0；Matplotlib 3.11.1
- 8× NVIDIA GeForce RTX 4090，约 48 GiB/卡；正式训练使用 GPU 0/1，GPU 4–7 被无关既有作业占用且未干预
- `WANDB_MODE=disabled`

真实资产：

| 变量/资产 | 状态与路径 |
| --- | --- |
| WAN22_ROOT | VERIFIED：`/home/amax/data0/SJ/models/Wan2.2-TI2V-5B` |
| WAN_ACTION_DIT_INIT | VERIFIED：`/home/amax/data0/SJ/preprocessed/starwam_action_dit_init_wan22.pt` |
| 原生 StarWAM checkpoint | VERIFIED：`/home/amax/data0/SJ/models/starwam_ckpts/starwam-libero/mot/starwam_wan225b_mot.pt` |
| COSMOS_ROOT | BLOCKED：官方 `nvidia/Cosmos-Predict2-2B-Video2World` gated，下载 HTTP 401；未猜测替代路径 |
| COSMOS_ACTION_DIT_INIT | BLOCKED：未获得，与 Cosmos case 一并阻塞 |
| 四个 LIBERO 数据目录 | VERIFIED：`/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/*_no_noops_lerobot` |
| ROBOTWIN_DATA | VERIFIED：`/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/repo-runs-performance-assets/robotwin2.0-fastwam/robotwin2.0` |

结构、哈希与路径复核见本目录 `manifest/assets.txt`、`asset_revalidation.txt`、`wan22-integrity.txt`、`wan-action-init-validation.json` 和 `checkpoints.sha256`。

## G1 六个 case

`N/A` 和 `NOT_RUN` 不是 exit code 0。G3 像素层失败后，严格执行“像素不一致立即停止后续性能评测”，因此没有继续消耗 GPU 运行余下 Wan case。

| Case | Exit code | GPU/显存峰值 | Loss | 梯度 | Checkpoint | Parity | 结果/产物 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| SW-L-MW | 0 | 2 GPU；36,936 / 36,882 MiB | step1 action 1.4478476、video 0.16308594、total 1.61093354；step2 action 7.58391666、video 0.27929688、total 7.86321354；均有限且 total 精确等于两项和 | 必需 action expert、backbone DiT、proprio encoder 均 finite/nonzero；真实 probe 未列出冻结 VAE/text 组，故不把冻结组误报为完整实模证据 | config/full、stats、probe、steps_1、steps_2、final_model 全部存在 | **FAIL：pixels** | 外部 run：`train/SW-L-MW_20260722_063933`；入库摘要：`artifacts/SW-L-MW` |
| SW-L-MC | N/A | N/A | N/A | N/A | N/A | N/A | **BLOCKED**：COSMOS_ROOT / COSMOS_ACTION_DIT_INIT 缺失 |
| SW-L-SW | N/A | N/A | N/A | N/A | N/A | N/A | **NOT_RUN**：G3 首个关键失败后的硬停止 |
| SW-L-SC | N/A | N/A | N/A | N/A | N/A | N/A | **BLOCKED**：COSMOS_ROOT / COSMOS_ACTION_DIT_INIT 缺失 |
| SW-L-FW | N/A | N/A | N/A | N/A | N/A | N/A | **NOT_RUN**：G3 首个关键失败后的硬停止 |
| SW-R-MW | N/A | N/A | N/A | N/A | N/A | N/A | **NOT_RUN**：G3 首个关键失败后的硬停止 |

### SW-L-MW optimizer 与产物

- 正式值：bf16、ZeRO-2、world size 2、per-device batch 4、accumulation 16、effective global batch 128、max steps 2、save interval 1、gradient probe step 1。
- 只有 CPU optimizer offload；parameter offload 关闭。AdamW 语义/超参、模型、精度、scheduler 和 global batch 未改变。这是显式资源 override，只能作为功能证据，不能当作默认 recipe 性能证据。
- 日志只报告两个 optimizer/global step；对应 LR 为 `5.05e-5` 和 `1e-6`。两卡 boundary harness 与单测确认 accumulation 中间 micro-step 不执行 optimizer/scheduler/clip/zero-grad，probe 在第一次 boundary、clip 前执行。
- checkpoint 大小：steps_1 = 24,083,916,506 bytes；steps_2 = 24,083,916,506 bytes；final_model = 24,083,890,074 bytes。
- `gradient_probe.json`：action expert norm 1.13518125、backbone DiT norm 0.55360258、proprio encoder norm 0.02353618；finite ratio 均为 1.0，nonzero ratio 均约 0.999。

## G2 checkpoint 与恢复

SW-L-MW 的可执行项全部通过：

- `baseframework.from_pretrained` 严格恢复 steps_2：3,300 keys，0 missing，0 unexpected，bf16，action horizon 32。
- 独立 bundle 把 recipe 改为 `/definitely/missing/starwam_recipe.yaml`，仍通过 resolved snapshot 严格恢复：0 missing，0 unexpected。
- 真实原生 checkpoint 通过 `strict_init=true` 导入：3,300 keys，0 missing，0 unexpected。
- 单测覆盖 state dict 包装、递归前缀、DeepSpeed directory/safetensors 发现、零匹配和关键组缺失拒绝。

但其余五个 case 没有 steps_2，且 D1 保存前/恢复后动作比较在 G3 像素硬失败后未继续，因此总 G2 为 BLOCKED，而不是 PASS。

## G3 原生一致性首个失败

Golden input 来自真实 LIBERO LeRobot episode 15、timestep 0：两张 512×512 `uint8 HWC` 图、8D state、instruction。数据集元信息没有 environment seed，报告为 unknown，未伪造。归档文件：

- `artifacts/SW-L-MW/golden_input_libero.npz`，SHA-256 `551858ae5806a52ba636e370f040d0e3366e60a3475c3c9ce3965bbb208eb0f6`
- `artifacts/SW-L-MW/golden_input_libero.json`

同一原始数组分别经过原生 StarWAM LIBERO benchmark adapter 和 StarVLA `compose_camera_views`：

| 指标 | 结果 |
| --- | ---: |
| shape | 两端均 `[1,3,224,448]` |
| dtype | 两端均 float32 |
| `torch.equal` | false |
| max absolute difference | 0.5139676928520203 |
| mean absolute difference | 0.010353229939937592 |
| 不同元素 | 230,030 / 301,056 |

根因定位为 resize 实现不一致：原生 `examples/libero/rollout.py::_resize_rgb` 使用 PIL bilinear；StarVLA wrapper 使用 PyTorch bilinear，并与 StarWAM 训练 dataset 的 `_resize_frames` 一致。由于门禁要求与原生 benchmark adapter 逐像素一致，本轮 G3 仍必须判 FAIL。没有通过改变 resize、选择 224×224 假输入或绕过原生 adapter 来伪造通过。

按 E2 “第一层不一致时停止”，text、state、normalized action、final action 和 service action 均标为 `NOT_RUN_DUE_PIXEL_FAILURE`。正式失败是数值断言式 exit 2，没有 Python exception traceback；首个失败现场完整保存在 `artifacts/SW-L-MW/parity.json` 和 `logs/SW-L-MW_parity.log`。

复现：

```bash
cd /home/amax/SJ/projects/Embodied-World
PYTHONPATH="$PWD/starVLA:$PWD/StarWAM" \
  /home/amax/data0/SJ/envs/starvla-starwam-py311/bin/python \
  starVLA/examples/modelExtensions/StarWAM/test_results/3f072c4_20260722_062824/artifacts/SW-L-MW/pixel_parity_harness.py
```

预期复现 exit code 为 2，并重写外部 `TEST_RUN_ROOT` 下的 parity 结果。

## G4 policy server 与 rollout

- 实际 SW-L-MW strict restore metadata：chunk 32、`available_unnorm_keys=[starwam]`、default `starwam`、camera order primary/wrist、horizontal、framework preprocess true、state required、model family `mot_wam`、LIBERO action layout `default`。
- 完整 CPU tests 验证 RoboTwin metadata 为 `native_qpos` 时客户端逐维保持原始 14D qpos，不执行旧 `[0..5,12,6..11,13]` 重排。
- 因 G3 首层像素失败，真实 server action、错误路径矩阵、LIBERO/RoboTwin rollout 均未继续；这不是成功，也不是模拟器缺失的替代说法。

所以 G4 为 BLOCKED。没有报告 rollout success rate。

## 资源 override、诊断失败与首个堆栈

正式 SW-L-MW 前的诊断尝试全部保留在外部旧 root：`/home/amax/data0/SJ/test-runs/starwam-integration/4e4bdf7_20260721_115340`。

1. 四卡标准 GPU AdamW 在第一次 optimizer step 分配 state 时 OOM；没有把 micro-batch 数当 optimizer step。
2. 四卡 CPU optimizer offload 进入 step 1 后，旧保存路径让每 rank materialize 完整 state dict，主机 RAM 低于 1 GiB、8 GiB swap 用满，人工停止保护其他作业。
3. 两卡 CPU offload 完成 steps_1/steps_2 后，rank 0 在 final_model 前被 SIGKILL；该现场定位并由 `3f072c4` 修复。
4. 修复后正式两卡 run exit 0，final_model 完整产生。

首个异常堆栈的关键尾部：

```text
train_starvla.py:_train_step -> deepspeed.runtime.engine.step
  -> zero.stage_1_and_2._optimizer_step -> torch.optim.AdamW.step
  -> _single_tensor_adamw
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.61 GiB.
GPU 0 total 47.37 GiB, free 2.90 GiB.
```

旧保存路径的首个关键终止：

```text
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
rank 0 exitcode -9
traceback: Signal 9 (SIGKILL) received by PID 1945412
```

正式训练复现使用 `artifacts/SW-L-MW/run.sh`。该脚本明确记录 CPU optimizer offload 资源 override，并保持 global batch 128。

## Skip 与产物边界

- 正式 integration/full tests：skip 0。
- Cosmos 两项：资产 BLOCKED，不是 skip。
- 三个未运行 Wan case、G3 下游比较和 G4：由首个关键像素失败触发硬停止，不是 skip。
- 未把约 24 GB checkpoint 提交进 Git；报告保留绝对路径、大小、SHA-256、配置、日志和恢复结果。实际 checkpoint 仍在外部 `TEST_RUN_ROOT`。

## 合并建议

**不建议合并。** G0 和 SW-L-MW 的两步训练/恢复证明路径、checkpoint 和核心 ZeRO-2 修复有效，但核心数值门禁 G3 在像素第一层明确失败，且 G1/G2/G4 不完整。应先统一原生 StarWAM LIBERO rollout 与训练/StarVLA 的 resize 合同，重新冻结新候选，然后从 G0 开始重跑六 case、全量 strict restore、原生 action parity、真实 server 错误路径和 rollout；在此之前不能批准合并。
