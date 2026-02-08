#!/bin/bash
# MIMIC-CXR 放射报告生成微调：Qwen3-VL-4B + LoRA + 投影层
# 使用前请确保已运行 create_mimic_qwen_json.sh 生成 mimic_qwen_train.json

# GPU 配置（在脚本内直接修改）
export CUDA_VISIBLE_DEVICES=1,2
NPROC_PER_NODE=2

# Weights & Biases
export WANDB_PROJECT="qwen3-vl-xr"
export WANDB_ENTITY="zchenhi"  # 可选，填你的 wandb 用户名或 team 名

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct

# Training hyperparameters（LoRA 推荐 1e-4~2e-5，2e-7 过小导致 loss 几乎不降）
lr=1e-6
batch_size=8
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset: MIMIC-CXR 训练集
datasets=mimic_train

# Output configuration
run_name="qwen3vl-mimic-4b"
output_dir=/jhcnas5/chenzhixuan/checkpoints/XRay-VLP/experiments/RG/mimic/qwen4b

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
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
    --report_to wandb"

# Launch training
cd "$(dirname "${BASH_SOURCE[0]}")/.."
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
