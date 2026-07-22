# StarWAM 集成门禁决策 — 3cc6624

## 范围与冻结输入

- StarVLA：3cc6624225d6efc2b18752bbaf7c66744290ffcf，分支 codex/starwam-integration。
- StarWAM：a7b05c8da8f8c88bf4888aa737b938af3b67ab48，detached 且干净。
- 外部运行目录：/home/amax/data0/SJ/test-runs/starwam-integration/3cc6624_20260722_151122。
- 完整环境：Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4、NCCL 2.21.5、Accelerate 1.5.2、DeepSpeed 0.16.9、Transformers 4.57.0、OpenCV 4.11.0、Matplotlib 3.11.1。
- starwam 实际导入：/home/amax/SJ/projects/Embodied-World/StarWAM/starwam/__init__.py。
- 本轮候选包含三组修复：d187bb1 修复 Shared-DiT 推理设备选择和 LIBERO Path 兼容；7d7de91 修复两个 RoboTwin launcher 的仓库根目录；3cc6624 让 RoboTwin 客户端使用 server metadata 的归一化键。
- 每个候选在测试期间均冻结；源码测试结束后才把小型结果归档到本目录。checkpoint、cache、数据集、simulator 和视频未提交。
- 四份 Wan 训练、strict restore、parity 和 LIBERO 证据来自 d187bb1 之前已完成且不受后续 RoboTwin-only 改动影响的运行；3cc6624 重新执行了 G0、RoboTwin G4，并补跑 SW-L-MW/SW-L-FW relocated bundle 推理。复用边界见 manifest/reused_evidence.txt。

## 门禁结论

| 门禁 | 结论 | 依据 |
|---|---|---|
| G0 | **PASS** | A1 全部通过；集成测试 46 passed；完整 tests 79 passed；0 failure、0 error、0 skip；无 StarWAM 环境懒加载和旧 framework 导入通过。 |
| G1 | **BLOCKED** | 四份 Wan case 在两 GPU bf16 ZeRO-2 下完成两个真实 optimizer step；SW-L-MC 与 SW-L-SC 因官方 gated Cosmos 权重不可得而未运行，六 case 矩阵不完整。 |
| G2 | **BLOCKED** | 四份可用 Wan steps_2 均 strict 0/0、脱离原 recipe 恢复并完成实际推理；真实原生 checkpoint 导入 strict 0/0。两份 Cosmos checkpoint 不存在；历史训练还未保留保存前的 in-memory action 基线，不能声称六 case 的完整 save-before/restore-after 对照。 |
| G3 | **PASS** | 使用同一真实原生 MoT/Wan checkpoint、原始图像、文本、state、seed 和推理步数，canonical tensor resize、state、normalized/final action 全部精确一致。范围为一份可用的原生 LIBERO checkpoint。 |
| G4 | **PASS** | 合并级 smoke：LIBERO 1 task × 1 episode 正常结束；RoboTwin 1 task × 2 完整 episode 正常 exit 0，metadata、反归一化键和 native_qpos 顺序正确。未声称发布矩阵或成功率。 |

**明确结论：暂不建议合并。** 当前已覆盖的 Wan 路径没有剩余已知代码失败，但 G1/G2 是“必须通过”的合并门禁；两份 Cosmos case 仍 BLOCKED，且 G2 的六 case 完整基线比较未完成。

## 真实资产

| 资产 | 状态 |
|---|---|
| WAN22_ROOT | /home/amax/data0/SJ/models/Wan2.2-TI2V-5B |
| WAN_ACTION_DIT_INIT | /home/amax/data0/SJ/preprocessed/starwam_action_dit_init_wan22.pt |
| 原生 StarWAM checkpoint | /home/amax/data0/SJ/models/starwam_ckpts/starwam-libero/mot/starwam_wan225b_mot.pt；SHA-256 d24edea01579880327cfd9dc84d24adab82e420dca9652e614ad697bc8cc5378 |
| COSMOS_ROOT | **BLOCKED**：本机 HF token 无效，官方 gated 仓库返回 401；本地目录只有 README/ref，没有权重 blob |
| COSMOS_ACTION_DIT_INIT | **BLOCKED**：依赖缺失的 Cosmos backbone |
| LIBERO spatial | /data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot |
| LIBERO object | /data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_object_no_noops_lerobot |
| LIBERO goal | /data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_goal_no_noops_lerobot |
| LIBERO 10 | /data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_10_no_noops_lerobot |
| ROBOTWIN_DATA | /data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/repo-runs-performance-assets/robotwin2.0-fastwam/robotwin2.0 |
| RoboTwin simulator/assets | /data0/CWJ/vla_3d_geometry_project/extern/robotwin；真实 assets、task_config 和 policy_ckpt_path patch 均存在 |

没有猜测路径；精确记录见 manifest/assets.txt 和 manifest/cosmos_blocked.txt。

## G0

- A1 的 ruff、compileall、TOML 和 git diff 检查：PASS，见 logs/g0_a1.log。
- tests/test_starwam_integration.py：46 passed，0 failed/error/skipped，见 logs/g0_starwam_integration.log。
- 完整 tests：79 passed，0 failed/error/skipped，见 logs/g0_all_tests.log。
- 独立无 StarWAM 环境：legacy ACT/lightweight import 通过；请求 StarWAM 时给出可执行的可选依赖错误，见 logs/g0_no_starwam.log。
- 两 rank 外部 harness 遍历两个完整 DataLoader epoch：每 rank 8 step、每步 collective、步数一致、legacy_reset_calls=0，见 artifacts/reused_training/infinite_dataloader/result.json。
- 两 rank ZeRO-2 boundary pattern 为 [false,true]；中间 micro-step 未执行 optimizer.step、scheduler.step、clip、zero_grad；两个 global boundary 后 scheduler 只前进两次。
- 独立两 rank probe 证明冻结 VAE 为 0 trainable/0 gradient；见 artifacts/reused_training/zero2_gradient_probe/result.json。

## G1 六 case

所有完成项保留 gradient_probe_enabled=true、gradient_probe_step=1、save_interval=1。loss 全部有限；MoT/Shared-DiT 在记录精度下严格满足 total=action_loss+video_loss；Feature-conditioned 的 video_loss 两步均为 0。

| Case | 状态/exit | GPU 峰值 MiB (rank0/rank1) | Step 1 (action, video, total) | Step 2 (action, video, total) | 首个 boundary 梯度 L2 |
|---|---|---:|---|---|---|
| SW-L-MW | PASS_AVAILABLE / 0 | 36,936 / 36,882 | (1.447847605, 0.163085938, 1.610933542) | (7.583916664, 0.279296875, 7.863213539) | action 1.135181；backbone 0.553603；proprio 0.023536 |
| SW-L-MC | **BLOCKED** / N/A | N/A | Cosmos 资产缺失 | Cosmos 资产缺失 | N/A |
| SW-L-SW | PASS_AVAILABLE / 0 | 36,461 / 36,480 | (2.183520317, 0.225585938, 2.409106255) | (27.896472931, 0.474609375, 28.371082306) | shared_dit 5.000910；unused backbone 冻结 |
| SW-L-SC | **BLOCKED** / N/A | N/A | Cosmos 资产缺失 | Cosmos 资产缺失 | N/A |
| SW-L-FW | PASS_AVAILABLE / 0 | 32,131 / 30,468 | (1.264688253, 0, 1.264688253) | (11.166011810, 0, 11.166011810) | action 0.462537；backbone 0.058717；feature projector 0.075468；proprio 0.001034 |
| SW-R-MW | PASS_AVAILABLE / 0 | 39,153 / 39,152 | (1.469900131, 0.447265625, 1.917165756) | (10.304623604, 1.078125000, 11.382748604) | action 0.107077；backbone 0.059759；proprio 0.004765 |

每个完成项均重新确认存在 config.yaml、config.full.yaml、dataset_statistics.json、gradient_probe.json、summary.jsonl、steps_1、steps_2 和 final_model。大文件仍在以下外部目录：

- SW-L-MW：/home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824/train/SW-L-MW_20260722_063933
- SW-L-SW：/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-L-SW_20260722_092000
- SW-L-FW：/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-L-FW_20260722_092000
- SW-R-MW：/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/train/SW-R-MW_20260722_092000

LIBERO 三项有效 global batch 为 4 × 2 ranks × 16 accumulation = 128；RoboTwin 为 4 × 2 × 128 = 1024。没有因 OOM 静默降低 batch。Feature-conditioned 的 feature_timestep_embedding 在该首批次梯度为 0；必需的 ActionDiT、feature projector 和 recipe 指定 backbone 组均为有限非零。历史 real-run probe JSON 没有单独持久化 captured_before_clip 布尔位，也没有逐 case 枚举 VAE/text encoder；当前控制流/回归测试和独立两 rank frozen-VAE probe覆盖这一合同，但该归档格式限制不被隐瞒。

## G2 restore、relocation 与推理

| Case | strict steps_2 | relocated bundle | 固定 seed 实际推理 |
|---|---|---|---|
| SW-L-MW | 3300 keys，missing 0，unexpected 0 | recipe=/definitely/missing/starwam_recipe.yaml；resolved snapshot 使用成功 | PASS，有限 [1,32,7]，min -1.65625，max 2.65625 |
| SW-L-MC | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| SW-L-SW | 1664 keys，missing 0，unexpected 0 | 同上 | PASS，有限 [1,32,7]，min -18.875，max 10.5625；compute module cuda:0/bf16 |
| SW-L-SC | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| SW-L-FW | 1657 keys，missing 0，unexpected 0 | 同上 | PASS，有限 [1,32,7]，min -13.9375，max 8.75 |
| SW-R-MW | 3300 keys，missing 0，unexpected 0 | 同上 | PASS，policy server 返回有限 [1,32,14] |

实际推理机器结果：artifacts/SW-L-MW_bundle_inference.json、artifacts/SW-L-SW_bundle_inference.json、artifacts/SW-L-FW_bundle_inference.json、artifacts/reused_training/SW-R-MW/policy_server_result.json。

真实原生 StarWAM checkpoint 导入：3300 keys、missing 0、unexpected 0、strict_init=true，见 artifacts/reused_training/SW-L-MW/native_checkpoint_import.json。

## G3 数值一致性和 resize 决策

canonical 合同继续沿用 StarVLA/StarWAM 训练路径的 PyTorch tensor resize；FastWAM/StarWAM rollout 的 PIL bilinear 只保留为诊断。

- 像素：torch.equal=true，max_abs_diff=0，failed_elements=0，shape [1,3,224,448]。
- 文本 context/mask：精确一致。
- state：max_abs_diff=0，atol=rtol=1e-6。
- normalized action：[1,32,7]，max_abs_diff=0，failed_elements=0，atol=1e-4、rtol=1e-3。
- final action：[1,32,7]，max_abs_diff=0，failed_elements=0，同一 tolerance。
- action horizon 32；video frames 9。
- PIL rollout 诊断：max_abs_diff=0.513967693，mean_abs_diff=0.010353229，230,030 个元素不同；因此不把 canonical 训练 resize 改成 PIL。

完整结果见 artifacts/parity/SW-L-MW/parity.json。

## Policy server 与 G4

Policy server 对真实 SW-R-MW relocated bundle 的验证：

- metadata：action_chunk_size=32、available/default key=starwam、相机顺序 head/left/right、camera_layout=robotwin、action_layout=native_qpos、state_required=true。
- 有效响应：finite float32 [1,32,14]，framework z-score 反归一化。
- 错误路径：缺 14-D state、相机数为 2、错误 unnorm_key、缺统计文件和 13-D stats 均被拒绝。
- RoboTwin 客户端从 metadata 将 null key 解析为 starwam；没有执行旧的 [0..5,12,6..11,13] 重排，动作按 native qpos 原顺序提交。

LIBERO：

- libero_goal，1 task × 1 episode，client exit 0，0 success；server 在 client 正常结束后手动停止，exit 130。
- Path 类型失败未复现；0 success 只反映两步训练 checkpoint，不作为 smoke 失败。
- 证据：artifacts/libero_rollout.json、logs/libero_rollout_client.log、logs/libero_rollout_server.log。

RoboTwin：

- adjust_bottle/demo_clean，显式 ROBOTWIN_TEST_NUM=2；两次均完整运行 400/400 step，launcher exit 0，0/2 success。
- 产生两个可解码 mp4 和 _result.txt；视频留在外部目录，归档仅保留 result 文本。
- GPU 3 峰值 44,272 MiB；关键错误扫描 0 matches。
- 证据：artifacts/robotwin_rollout.json、artifacts/robotwin_bounded_result.txt、logs/robotwin_launcher_bounded.log 和 stats/robotwin_bounded_nvidia_smi.csv。
- 第一次命令漏设 ROBOTWIN_TEST_NUM，落回 100；完成 7 个 episode 后被中止并完整保留，但不计作 PASS。随后两 episode bounded run 完成其配置分母并正常 exit 0。
- G4 只判定合并级闭环可运行；50-task/双 setting/原生 episode 数发布矩阵未运行，也未将 0/2 success 解释为性能结论。

## skips、override 与复用

- pytest skip：集成 0，完整 tests 0。
- SW-L-MC/SW-L-SC 是明确 BLOCKED，不计作 skip 或 pass。
- 训练资源：两物理 GPU、bf16、ZeRO-2、CPU optimizer offload、无 parameter offload；有效 global batch 未更改。
- RoboTwin override：ROBOTWIN_TEST_NUM=2；PATH 前置 /data0/CWJ/vla_3d_geometry_project/envs/starvla/bin 以使用该环境已有 ffmpeg。
- LIBERO server exit 130 是 client 完成后的人工关停，不是 episode 失败。
- 训练/G2/G3/LIBERO 复用理由见 manifest/reused_evidence.txt；候选 3cc6624 自身的 G0 和 RoboTwin 均重新执行。
- 没有把环境缺失、被中断的 100-episode 尝试或 Cosmos 缺失写成代码通过。

## 首个失败现场与修复

1. Shared-DiT 恢复后推理：旧候选 ad6b412 在 self.time_embedding 处报 RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu。根因是 adapter 从未使用的 CPU/fp32 backbone 首参数推断设备；d187bb1 改为读取 shared_dit compute module。当前实际推理 PASS。
2. LIBERO：旧候选把 pathlib.Path 传入只接受字符串的 wrapper，首栈 TypeError: argument of type 'PosixPath' is not iterable。d187bb1 在构造环境前转换为 str；真实 episode PASS。
3. RoboTwin launcher：旧 REPO_ROOT 少退一级，尝试打开 starVLA/examples/deployment/...，报 Errno 2。7d7de91 修正两个脚本；证据在 artifacts/prior_failures/robotwin_attempt0_bad_repo_root。
4. RoboTwin ffmpeg：使用绝对 Python 路径不会自动把环境 bin 放入 PATH，首栈 FileNotFoundError: ffmpeg。此为运行环境问题；正式命令显式前置 PATH，源码未为此修改。
5. RoboTwin unnorm key：旧模板硬编码 new_embodiment，server 只有 starwam，首栈 KeyError: unnorm_key='new_embodiment' not in ['starwam']。3cc6624 让模板使用 null、客户端读取 default_unnorm_key；正式 rollout PASS。

前三个旧源码失败的原始日志分别保留在此前 ad6b412 归档和本目录 artifacts/prior_failures；正式候选日志没有 NaN/Inf、rank hang、严格恢复失败、像素差异或动作顺序错误。

## 复现命令

当前 SW-L-MW relocated bundle 推理：

~~~bash
cd /home/amax/data0/SJ/test-runs/starwam-integration/3cc6624_20260722_151122
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/amax/SJ/projects/Embodied-World/starVLA:/home/amax/SJ/projects/Embodied-World/StarWAM   /home/amax/data0/SJ/envs/starvla-starwam-py311/bin/python artifacts/bundle_inference_harness.py   --case SW-L-MW   --checkpoint /home/amax/data0/SJ/test-runs/starwam-integration/3f072c4_20260722_062824/artifacts/bundles/SW-L-MW/checkpoints/steps_2_pytorch_model.pt   --golden /home/amax/SJ/projects/Embodied-World/starVLA/examples/modelExtensions/StarWAM/test_results/3f072c4_20260722_062824/artifacts/SW-L-MW/golden_input_libero.npz   --output artifacts/SW-L-MW_bundle_inference.json
~~~

当前 RoboTwin bounded smoke：

~~~bash
cd /home/amax/SJ/projects/Embodied-World/starVLA
ROBOTWIN_TEST_NUM=2 CUDA_VISIBLE_DEVICES=3 ROBOTWIN_PATH=/home/amax/data0/SJ/test-runs/starwam-integration/3cc6624_20260722_151122/artifacts/robotwin_runtime STARVLA_PYTHON=/home/amax/data0/SJ/envs/starvla-starwam-py311/bin/python ROBOTWIN_PYTHON=/data0/CWJ/vla_3d_geometry_project/envs/starvla/bin/python ROBOTWIN_LOG_ROOT=/home/amax/data0/SJ/test-runs/starwam-integration/3cc6624_20260722_151122/logs/robotwin_rollout_bounded bash examples/simBenchmarks/Robotwin/eval_files/start_eval.sh   -m demo_clean -n starwam_integration_bounded   -c /home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916/artifacts/bundles/SW-R-MW/checkpoints/steps_2_pytorch_model.pt   -s 0 -j 1 -p 10122 adjust_bottle
~~~

历史失败可在对应旧 commit 的临时 worktree 使用相同命令重现；第一关键栈已原样归档，不需要修改当前源码来重现。

## 产物索引

- 门禁表：manifest/gates.tsv
- 六 case 表：manifest/cases.tsv
- 环境、revision、资产与复用：manifest/
- 完整小型日志和首栈：logs/、artifacts/prior_failures/
- 配置、statistics、gradient probe、strict restore、policy、parity：artifacts/
- GPU trace：stats/
- 大 checkpoint/cache/raw data/video：仍位于上述外部 TEST_RUN_ROOT 和资产目录。
