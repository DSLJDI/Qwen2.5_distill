#!/bin/bash

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 获取脚本所在的绝对路径
PROJECT_DIR=$(dirname "$(realpath "$0")")

# --- 定义路径 ---
TRAIN_DATA="$PROJECT_DIR/data/train_data.json"
VAL_DATA="$PROJECT_DIR/data/test.data.json"
OUTPUT_DIR="$PROJECT_DIR/cot_distill_results_7B_model"
ORIGINAL_MODEL="Qwen/Qwen2.5-7B-Instruct"

# --- 创建目录 ---
mkdir -p "$OUTPUT_DIR/"

# --- 启动训练 ---
torchrun --nproc_per_node=1 --master_port 29512 \
    $PROJECT_DIR/train.py \
    --model_name_or_path "$ORIGINAL_MODEL" \
    --train_dataset_path "$TRAIN_DATA" \
    --val_dataset_path "$VAL_DATA" \
    --prompt_path "$PROJECT_DIR/data/gen_reasoning_prompt.txt" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --deepspeed "$PROJECT_DIR/configs/deepspeed_config.json" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --train_max_length 4300 \
    --fp16 \
    --gradient_checkpointing \
    > "$OUTPUT_DIR/training.log" 2> "$OUTPUT_DIR/error.log"
