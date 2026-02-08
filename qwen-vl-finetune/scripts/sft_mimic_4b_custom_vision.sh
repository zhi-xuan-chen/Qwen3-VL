#!/bin/bash
# MIMIC-CXR 放射报告生成微调：Qwen3-VL-4B + 自定义视觉 encoder (rad_dino) + SimpleVisionMerger + LoRA
# 计划第三步、第四步：替换 model.visual 为 qwenvl.vision 中的 encoder，关闭 DeepStack

# GPU 配置
export CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1

# Weights & Biases
export WANDB_PROJECT="mimic-cxr"
export WANDB_ENTITY=""

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct

# 自定义视觉 encoder：rad_dino / gloria / chexzero / chess / our_ark_base / our_ark_large 等（见 qwenvl/vision/utils.py）
CUSTOM_VISION_ENCODER=ark_base
# 仅 our_ 开头模型需要：预训练权重路径（our_ark_base / our_ark_large 必填）
CUSTOM_VISION_PRETRAINED_PATH=""

# our_ 开头时必须指定预训练路径，否则报错退出
if [[ "${CUSTOM_VISION_ENCODER}" == our* ]] && [ -z "${CUSTOM_VISION_PRETRAINED_PATH}" ]; then
    echo "Error: ${CUSTOM_VISION_ENCODER} 必须设置 CUSTOM_VISION_PRETRAINED_PATH"
    exit 1
fi

# Training hyperparameters
lr=1e-6
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset
datasets=mimic_train

# Model name：our 开头用预训练路径最后一段，否则用 encoder 名
if [[ "${CUSTOM_VISION_ENCODER}" == our* ]]; then
    CUSTOM_VISION_MODEL_NAME=$(basename "${CUSTOM_VISION_PRETRAINED_PATH%/}")
    [ -z "${CUSTOM_VISION_MODEL_NAME}" ] && CUSTOM_VISION_MODEL_NAME="${CUSTOM_VISION_ENCODER}"
else
    CUSTOM_VISION_MODEL_NAME="${CUSTOM_VISION_ENCODER}"
fi

# Output configuration：run_name 与 output_dir 末尾均含 qwen + 上述 model name
run_name="qwen3vl-mimic-4b-${CUSTOM_VISION_MODEL_NAME}"
output_dir=/jhcnas5/chenzhixuan/checkpoints/XRay-VLP/experiments/RG/mimic/qwen_${CUSTOM_VISION_MODEL_NAME}

# Training arguments：仅 our_ 开头且路径非空时传预训练权重
extra_custom=""
if [[ "${CUSTOM_VISION_ENCODER}" == our* ]] && [ -n "${CUSTOM_VISION_PRETRAINED_PATH}" ]; then
    extra_custom="--custom_vision_pretrained_path ${CUSTOM_VISION_PRETRAINED_PATH}"
fi

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --custom_vision_encoder ${CUSTOM_VISION_ENCODER} \
    ${extra_custom} \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
cd "$(dirname "${BASH_SOURCE[0]}")/.."
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
