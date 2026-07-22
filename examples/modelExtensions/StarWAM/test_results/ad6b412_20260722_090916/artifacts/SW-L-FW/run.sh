#!/usr/bin/env bash
set -uo pipefail

export STARVLA_ROOT=/home/amax/SJ/projects/Embodied-World/starVLA
export STARWAM_ROOT=/home/amax/SJ/projects/Embodied-World/StarWAM
export TEST_RUN_ROOT=/home/amax/data0/SJ/test-runs/starwam-integration/ad6b412_20260722_090916
export WAN22_ROOT=/home/amax/data0/SJ/models/Wan2.2-TI2V-5B
export WAN_ACTION_DIT_INIT=/home/amax/data0/SJ/preprocessed/starwam_action_dit_init_wan22.pt
export LIBERO_SPATIAL_DATA=/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
export LIBERO_OBJECT_DATA=/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_object_no_noops_lerobot
export LIBERO_GOAL_DATA=/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_goal_no_noops_lerobot
export LIBERO_10_DATA=/data0/FastWAM-PERF-002-storage/root-migration-20260716T0311Z/Datasets/fastwam/libero_mujoco3.3.2/libero_10_no_noops_lerobot

export CUDA_HOME=/home/amax/miniconda3/pkgs/cuda-nvcc-12.4.131-0
export PATH=/home/amax/data0/SJ/envs/starvla-starwam-py311/bin:${CUDA_HOME}/bin:${PATH}
export PYTHONPATH=${STARVLA_ROOT}:${STARWAM_ROOT}:${PYTHONPATH:-}
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=${TEST_RUN_ROOT}/cache/xdg
export HF_HOME=${TEST_RUN_ROOT}/cache/huggingface
export TORCH_HOME=${TEST_RUN_ROOT}/cache/torch
export TRITON_CACHE_DIR=${TEST_RUN_ROOT}/cache/triton
export LIBRARY_PATH=${TEST_RUN_ROOT}/artifacts/cuda-link
export LD_LIBRARY_PATH=${TEST_RUN_ROOT}/artifacts/cuda-link:/home/amax/data0/SJ/envs/starvla-starwam-py311/lib/python3.11/site-packages/nvidia/curand/lib:/home/amax/data0/SJ/envs/starvla-starwam-py311/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/home/amax/data0/SJ/envs/starvla-starwam-py311/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export TORCH_EXTENSIONS_DIR=${TEST_RUN_ROOT}/cache/torch_extensions
export MAX_JOBS=8

CASE_ID=SW-L-FW
RUN_ID=SW-L-FW_20260722_092000
CONFIG=examples/modelExtensions/StarWAM/train_files/libero_feature_conditioned_wan22.yaml
OVERRIDES="[\"backbone.pretrained_model_id=${WAN22_ROOT}\",\"training.batch_size=4\",\"training.gradient_accumulation_steps=16\",\"data.dataset_dirs=['${LIBERO_SPATIAL_DATA}','${LIBERO_OBJECT_DATA}','${LIBERO_GOAL_DATA}','${LIBERO_10_DATA}']\",\"data.text_embedding_cache_dir=${TEST_RUN_ROOT}/cache/${CASE_ID}\",\"data.action_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\",\"data.state_stats_path=${TEST_RUN_ROOT}/stats/${CASE_ID}.json\"]"

mkdir -p "${TEST_RUN_ROOT}/manifest/${CASE_ID}" "${TEST_RUN_ROOT}/stats" "${TEST_RUN_ROOT}/cache/${CASE_ID}" "${TEST_RUN_ROOT}/train"
printf '%s\n' "${OVERRIDES}" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/overrides.json"
printf '%s\n' "${RUN_ID}" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/run_id.txt"
printf '%s\n' "2" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/world_size.txt"
printf '%s\n' "4 * 2 * 16 = 128" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/effective_global_batch.txt"
printf '%s\n' "RESOURCE OVERRIDE: GPUs 4-7 remain occupied by an unrelated existing run. Default four-GPU GPU-optimizer attempts OOMed at AdamW state creation; four-rank CPU-offload duplicated full state dicts and exhausted host RAM; the pre-fix two-rank run completed steps_1 and steps_2 but was SIGKILLed before final_model. Candidate 3f072c4 fixes the verified code bug by materializing ZeRO-2 state dicts only on rank 0 while preserving all-rank collection for ZeRO-3/FSDP. This formal run uses the required minimum two GPUs, batch=4 and accumulation=16, preserving effective global batch 128. CPU optimizer offload only; parameter offload remains disabled. BF16, ZeRO stage 2, model, AdamW semantics/hyperparameters, scheduler, and global batch are unchanged. The dedicated two-GPU harness validates DeepSpeedCPUAdam adamw_mode, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01, boundaries, finite nonzero gradients, and updates. Functional evidence only, not default-recipe performance evidence." > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/resource_override.txt"
printf '%s\n' "${TEST_RUN_ROOT}/artifacts/zero2_cpu_offload_2gpu.yaml" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/accelerate_config.txt"
printf '%s\n' "${TEST_RUN_ROOT}/artifacts/zero2_cpu_offload_result.json" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/offload_validation.txt"

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > "${TEST_RUN_ROOT}/stats/${CASE_ID}_nvidia_smi.csv" &
MONITOR_PID=$!
vmstat -t 1 > "${TEST_RUN_ROOT}/stats/${CASE_ID}_vmstat.txt" &
HOST_MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" "${HOST_MONITOR_PID}" 2>/dev/null || true' EXIT

cd "${STARVLA_ROOT}"
set +e
accelerate launch \
  --config_file "${TEST_RUN_ROOT}/artifacts/zero2_cpu_offload_2gpu.yaml" \
  --num_processes 2 \
  --main_process_port 29646 \
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
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
kill "${HOST_MONITOR_PID}" 2>/dev/null || true
wait "${HOST_MONITOR_PID}" 2>/dev/null || true
trap - EXIT
printf '%s\n' "${TRAIN_EXIT}" > "${TEST_RUN_ROOT}/manifest/${CASE_ID}/exit_code.txt"
exit "${TRAIN_EXIT}"
