#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL="${MODEL:-convnext_base}"
DATA="${DATA:-fgvc}"
DATASET="${DATASET:-cars}"
DATA_PATH="${DATA_PATH:-/home/dataset/fgvc/cars}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL}/${DATA}/${DATASET}}"

TUNING_METHOD="${TUNING_METHOD:-loca}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
if [[ -z "${LORA_TARGET:-}" ]]; then
  case "$MODEL" in
    convnext*)
      LORA_TARGET='["pwconv1", "pwconv2", "downsample", "dwconv", "stem"]'
      ;;
    mambavision*)
      LORA_TARGET='["patch_embed", "conv_down", "downsample", "conv1", "conv2", "in_proj", "x_proj", "dt_proj", "out_proj", "conv1d_x", "conv1d_z", "qkv", "proj"]'
      ;;
    mambaout*)
      LORA_TARGET='["stem", "stages", "blocks", "conv", "fc", "proj"]'
      ;;
    swin*)
      LORA_TARGET='["qkv", "proj", "fc1", "fc2", "downsample"]'
      ;;
    resnet*)
      LORA_TARGET='["conv1", "conv2", "conv3", "downsample", "fc"]'
      ;;
    *)
      LORA_TARGET='["conv", "qkv", "proj", "fc1", "fc2", "head"]'
      ;;
  esac
fi

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INPUT_SIZE="${INPUT_SIZE:-224}"
SEED="${SEED:-0}"
USE_AMP="${USE_AMP:-False}"

SV_LAYERS="${SV_LAYERS:-}"
SV_INTERVAL="${SV_INTERVAL:-1}"
SV_INCLUDE_SPATIAL="${SV_INCLUDE_SPATIAL:-True}"

COMMON_ARGS=(
  --model "$MODEL"
  --tuning_method "$TUNING_METHOD"
  --type loca
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_target "$LORA_TARGET"
  --data "$DATA"
  --dataset "$DATASET"
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_epochs "$WARMUP_EPOCHS"
  --update_freq "$UPDATE_FREQ"
  --num_workers "$NUM_WORKERS"
  --input_size "$INPUT_SIZE"
  --seed "$SEED"
  --use_amp "$USE_AMP"
  --wandb False
  --save_ckpt True
  --auto_resume True
  --sv_interval "$SV_INTERVAL"
  --sv_include_spatial "$SV_INCLUDE_SPATIAL"
)

if [[ -n "$SV_LAYERS" ]]; then
  COMMON_ARGS+=(--sv_layers "$SV_LAYERS")
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" main.py "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" main.py "${COMMON_ARGS[@]}"
fi
