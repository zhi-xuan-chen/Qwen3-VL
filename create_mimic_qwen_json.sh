#!/bin/bash
# 将 MIMIC-CXR CSV 转为 Qwen JSON，使用相对路径，一次性处理 train/val/test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CSV_PATH="/jhcnas4/XR/MIMIC-CXR/labels/mimic_engineered_merge_with_split.csv"
OUTPUT_DIR="/jhcnas4/XR/MIMIC-CXR/labels/qwen"

for SPLIT in train validate test; do
  OUT_NAME="${SPLIT/validate/valid}"  # validate -> valid
  echo "Processing split: $SPLIT"
  python create_mimic_qwen_json.py \
    --csv_path "$CSV_PATH" \
    --output_path "${OUTPUT_DIR}/mimic_qwen_${OUT_NAME}.json" \
    --split "$SPLIT" \
    --use_relative_path
done

echo "Done. Outputs: ${OUTPUT_DIR}/mimic_qwen_train.json, mimic_qwen_valid.json, mimic_qwen_test.json"
