# StarWAM 集成候选测试决策

- 执行时间：2026-07-21 UTC
- TEST_RUN_ROOT：/runs/starwam-integration/c3bbb7c_20260721_074033
- StarVLA：c3bbb7c9524cefbde4e36b5bbae5336ab7589bc9
- StarWAM：a7b05c8da8f8c88bf4888aa737b938af3b67ab48（detached）
- 最终源码状态：StarVLA clean；StarWAM clean
- 最终结论：**不建议合并**

## Gate 结论

| Gate | 结果 | 结论依据 |
| --- | --- | --- |
| G0 | **FAIL** | A1 ruff/TOML 命令失败；A2/A3 均有同一薄配置路径失败。 |
| G1 | **BLOCKED** | 六个 case 均在启动前被真实模型/初始化权重缺失和薄配置错误路径阻塞。 |
| G2 | **BLOCKED** | 没有任何 steps_2 checkpoint，也没有真实原生 StarWAM checkpoint。 |
| G3 | **BLOCKED** | 没有同一真实 checkpoint/golden input，不能生成合法 parity.json。 |
| G4 | **BLOCKED** | 无可启动 server 的 checkpoint；clean RoboTwin checkout 还缺 policy_ckpt_path patch。 |

G0 是合并必过门禁且已经失败；G1–G4 未形成正式证据，不能以诊断测试替代。

## G0 结果

### A1 静态检查

| 检查 | Exit code | 结果 |
| --- | ---: | --- |
| ruff | 1 | FAIL：eval_libero.py 共 5 项（2×F841、2×RUF059、1×E501）。 |
| compileall | 0 | PASS；通过 PYTHONPYCACHEPREFIX 把字节码写到 TEST_RUN_ROOT。 |
| Python 3.10 tomllib 命令 | 1 | FAIL：ModuleNotFoundError: No module named tomllib。 |
| git diff --check | 0 | PASS。 |

项目声明 Python >=3.10，但测试计划中的 tomllib 命令实际要求 Python 3.11+。辅助诊断证明 TOML 本身有效：Python 3.10 + tomli 和 Python 3.14 + tomllib 均解析成功；规定命令的非零 exit code 仍保留为失败。

### A2/A3/A4

| 项目 | 结果 | Skip |
| --- | --- | ---: |
| tests/test_starwam_integration.py | 34 passed, 1 failed | 0 |
| 完整 tests（补齐 requirements 中的 pyzmq 后） | 67 passed, 1 failed | 0 |
| 无 StarWAM 独立环境 | PASS：旧 ACT 注册/轻量模块导入成功，选择 StarWAM 返回含 .[wam] 的安装提示 | N/A |
| 核心合同诊断子集 | 21 passed | 0 |

完整 tests 第一次收集因 ModuleNotFoundError: zmq 中止（1 error）；requirements.txt 明确声明 pyzmq，在隔离 venv 补装后重跑得到上表最终结果。该初始环境错误单独保留，未解释为代码通过。

首个关键代码堆栈：

    tests/test_starwam_integration.py:538
      prepare_starwam_host_config(config, config_path=config_path)
    starVLA/model/framework/WAM/config.py:178
      starwam_cfg = load_config(recipe_path)
    StarWAM/starwam/config.py:243
      FileNotFoundError: Config file not found:
      /home/amax/SJ/projects/WorldActionModels/StarWAM/examples/libero/configs/recipes/
      starwam_libero_mot_wan22_5b.yaml

六份薄配置和测试常量解析到固定的 /home/amax/SJ/projects/WorldActionModels/StarWAM，而本次用户指定布局为同一 WORK_ROOT 下的 starVLA 与 StarWAM。该目录不存在，导出的 STARWAM_ROOT 不参与这段解析。

复现：

    cd /home/amax/SJ/projects/Embodied-World/starVLA
    export STARVLA_ROOT=$PWD
    export STARWAM_ROOT=/home/amax/SJ/projects/Embodied-World/StarWAM
    USE_TF=0 USE_FLAX=0 PYTHONPATH=$STARVLA_ROOT:$STARWAM_ROOT /runs/starwam-integration/c3bbb7c_20260721_074033/venv/bin/pytest -q -ra tests/test_starwam_integration.py

## 真实资产

所有要求的资产变量最初均未设置。只导出了经过结构和元数据校验的数据路径。

| 变量 | 状态 |
| --- | --- |
| WAN22_ROOT | BLOCKED：唯一同名候选只有 3 个 safetensors shard；缺 index/single file、VAE、T5 和 tokenizer。 |
| COSMOS_ROOT | BLOCKED：未发现完整目录或 HF cache。 |
| WAN_ACTION_DIT_INIT | BLOCKED：规定 payload 不存在；只发现未获准替代的 *_untrained.pt。 |
| COSMOS_ACTION_DIT_INIT | BLOCKED：未发现。 |
| 四个 LIBERO_*_DATA | VERIFIED：同一 libero_mujoco3.3.2/*_no_noops_lerobot 系列，7D action/8D state。 |
| ROBOTWIN_DATA | VERIFIED：LeRobot v2.1，27,500 episodes，14D state/action，native qpos 顺序。 |
| 原生 StarWAM checkpoint | BLOCKED：未发现；扫描到的 DeepSpeed payload 属于无关 StarVLA geometry runs。 |

精确路径和 metadata SHA-256：

- manifest/verified-data-assets.env
- manifest/assets.tsv
- manifest/dataset-metadata.sha256

## G1 六个 case

没有 case 被启动；exit_code=N/A 表示启动前 BLOCKED，不是 exit code 0。没有静默修改 batch、accumulation 或有效 global batch，也没有 OOM 后降配。

| Case | Exit code | 显存峰值 | Loss | 梯度 | Checkpoint | Parity | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SW-L-MW | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |
| SW-L-MC | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |
| SW-L-SW | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |
| SW-L-SC | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |
| SW-L-FW | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |
| SW-R-MW | N/A | N/A | N/A | N/A | N/A | N/A | BLOCKED |

每项的阻塞链和要求/实际产物见 artifacts/CASE/case_manifest.txt。以下正式产物均未生成：config.yaml、config.full.yaml、dataset_statistics.json、gradient_probe.json、steps 1/2 checkpoint、final_model。

### 两 GPU _infinite_dataloader harness

此项是独立诊断，不替代 G1：

- world size：2（GPU 0/1）
- 每 rank 每 epoch：4 step
- 完整 epoch：2
- 每 rank 总 step：8
- gathered rank step：[8, 8]
- 每步 NCCL all-reduce：PASS
- _reset_dataloader 调用：0
- rank hang：无
- exit code：0

脚本：/runs/starwam-integration/c3bbb7c_20260721_074033/infinite_dataloader_harness.py

结果：artifacts/dataloader_harness/result.json

日志末尾有 harness 未显式销毁 process group 的 PyTorch 警告；两个 rank 已正常退出，结果完整。候选 dataloader 合同本身通过该诊断。

## G2/G3

- 六个 steps_2_pytorch_model.pt 均不存在，无法执行 strict restore、missing/unexpected key 检查或脱离 recipe 恢复。
- 没有真实原生 StarWAM/DeepSpeed checkpoint，不能执行原生导入门禁。
- 没有合法训练 checkpoint 与批准的同源 golden input，不能比较像素、state、normalized/final action。
- 未创建伪造的 parity.json。
- CPU 合同诊断覆盖 prefix 适配、零匹配/关键组拒绝、tiny wrapper strict roundtrip、像素组合、normalization 与 metadata，共 21 项通过，但不升级 G2/G3 状态。

## G4 / policy server 与 rollout

- 没有 checkpoint，policy server 未启动；metadata、action horizon、真实反归一化和 server 错误路径未形成运行证据。
- CPU 合同确认 RoboTwin client 在 action_layout=native_qpos 时不执行旧 [0..5,12,6..11,13] 重排。
- clean LIBERO reference checkout 存在，但当前可导入环境指向另一个有 20,969 项既有修改的 checkout，未使用。
- clean RoboTwin reference revision c3ddfa8b97d5519efa828b075999bd0006778e5e 存在，但未发现要求的 policy_ckpt_path patch。
- 因此 LIBERO/RoboTwin rollout 均为 BLOCKED；没有伪造 smoke 成功或 success rate。

## 环境、硬件与 override

- Python 3.10.20
- PyTorch 2.6.0+cu124；CUDA 12.4；NCCL 2.21.5；cuDNN 9.1
- Accelerate 1.5.2；DeepSpeed 0.16.9；Transformers 4.57.0
- 8× NVIDIA GeForce RTX 4090，约 48 GiB/卡；driver 580.159.03
- 使用现有完整 StarVLA Conda 环境创建 --system-site-packages 隔离 venv。
- CUDA_HOME=/home/amax/CWJ/vla_3d_geometry_project/envs/starvla，否则 DeepSpeed import 失败。
- WANDB_MODE=disabled。
- compileall 的 pycache、HF/Matplotlib/Torch/Triton cache 均重定向到 TEST_RUN_ROOT。
- A3 为执行完整 tests 补装了 requirements 明确列出的 pyzmq==27.1.0。
- harness 使用 CUDA_VISIBLE_DEVICES=0,1；未对正式训练做资源 override。
- A2/A3 最终 skip 总数均为 0。

完整环境证据：

- manifest/source.txt
- manifest/environment.txt
- manifest/pip-freeze.txt
- manifest/accelerate-env.txt
- manifest/nvidia-smi.txt
- manifest/gpus.csv

## 关键日志

- logs/a1_ruff.log
- logs/a1_toml.log
- logs/cpu_starwam_integration.log
- logs/cpu_all_tests.log（初始 pyzmq 环境错误）
- logs/cpu_all_tests_rerun1.log
- logs/cpu_no_starwam.log
- logs/diagnostic_core_contracts.log
- logs/infinite_dataloader_harness.log

## 合并建议

**不建议合并。** 最低门禁 G0 已经失败，六份真实 recipe 的 G1 训练、G2 恢复、G3 parity 和 G4 server/rollout 也均未获得正式证据。至少应先修复/消除薄配置对固定 WorldActionModels/StarWAM 布局的依赖、使规定 A1 命令在声明的 Python 版本范围内成立、清除 ruff 错误，并提供完整 Wan/Cosmos、两份初始化 payload 和一个真实原生 StarWAM checkpoint 后，从 G0 重新执行。
