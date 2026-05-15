#!/usr/bin/env bash
set -euo pipefail

# This job runs inside a Docker container where the visible NIC is `eth0`.
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# Optional
# export WANDB_MODE=disabled

playground_root=../starVLA_playground
logging_backend=${LOGGING_BACKEND:-tensorboard}
Framework_name=QwenPI_v3
freeze_module_list=''
base_vlm=${playground_root}/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_starvla_playground.yaml
libero_data_root=${playground_root}/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal
run_root_dir=${playground_root}/Checkpoints
run_id=qwenpi_v3_libero_goal_qwen35_08b_8gpu
num_processes=8

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --logging_backend ${logging_backend} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity yelinhe
