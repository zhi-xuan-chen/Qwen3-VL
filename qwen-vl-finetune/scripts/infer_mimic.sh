#!/bin/bash
# MIMIC-CXR 测试集推理
# 支持原版 Qwen 与自定义视觉编码器两种模型

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ========== 可修改参数 ==========
# 数据路径
TEST_JSON="${TEST_JSON:-/jhcnas4/XR/MIMIC-CXR/labels/qwen/mimic_qwen_test.json}"
DATA_PATH="${DATA_PATH:-/jhcnas4/XR/MIMIC-CXR/imgs-1024}"

# Checkpoint 路径（必填）
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/jhcnas5/chenzhixuan/checkpoints/XRay-VLP/experiments/RG/mimic/qwen4b/checkpoint-xxx}"

# 自定义视觉编码器：不填则使用原版 Qwen；填 ark_base / rad_dino 等则使用自定义 vision
CUSTOM_VISION_ENCODER="${CUSTOM_VISION_ENCODER:-}"
# our_ 开头时需指定预训练路径
CUSTOM_VISION_PRETRAINED_PATH="${CUSTOM_VISION_PRETRAINED_PATH:-}"

# 输出（使用自定义视觉编码器时，自动在路径中加入 encoder 名称）
OUTPUT_FILE="${OUTPUT_FILE:-results/mimic_test_predictions.jsonl}"

# 生成参数
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# 自定义视觉编码器时，将 encoder 名加入输出路径
# our 开头：用预训练路径最后一段；否则用 encoder 名
if [ -n "${CUSTOM_VISION_ENCODER}" ]; then
  if [[ "${CUSTOM_VISION_ENCODER}" == our* ]]; then
    CUSTOM_VISION_MODEL_NAME=$(basename "${CUSTOM_VISION_PRETRAINED_PATH%/}")
    [ -z "${CUSTOM_VISION_MODEL_NAME}" ] && CUSTOM_VISION_MODEL_NAME="${CUSTOM_VISION_ENCODER}"
  else
    CUSTOM_VISION_MODEL_NAME="${CUSTOM_VISION_ENCODER}"
  fi
  OUT_DIR=$(dirname "${OUTPUT_FILE}")
  OUT_BASE=$(basename "${OUTPUT_FILE}" .jsonl)
  OUTPUT_FILE="${OUT_DIR}/${OUT_BASE}_${CUSTOM_VISION_MODEL_NAME}.jsonl"
fi

# ========== 原版 Qwen 推理 ==========
if [ -z "${CUSTOM_VISION_ENCODER}" ]; then
  echo "Running inference with standard Qwen (LoRA)..."
  python infer_mimic.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_json "${TEST_JSON}" \
    --data_path "${DATA_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}
else
  # ========== 自定义视觉编码器推理 ==========
  echo "Running inference with custom vision encoder: ${CUSTOM_VISION_ENCODER}"
  EXTRA=""
  if [[ "${CUSTOM_VISION_ENCODER}" == our* ]] && [ -n "${CUSTOM_VISION_PRETRAINED_PATH}" ]; then
    EXTRA="--custom_vision_pretrained_path ${CUSTOM_VISION_PRETRAINED_PATH}"
  fi
  python infer_mimic.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_json "${TEST_JSON}" \
    --data_path "${DATA_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --custom_vision_encoder "${CUSTOM_VISION_ENCODER}" \
    ${EXTRA} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}
fi
