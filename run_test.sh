#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_DIR=$(dirname "$(realpath "$0")")

TEST_DATA="$PROJECT_DIR/data/test.csv"
OUTPUT_DIR="$PROJECT_DIR/cot_distill_results_v6_1w2_7B_model/final_checkpoint"
PROMPT_PATH="$PROJECT_DIR/data/prompt.txt"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

python $PROJECT_DIR/test_model.py \
    --test_dataset_path "$TEST_DATA" \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2 \
    --input_cloumn_name "input" \
    --output_cloumn_name "model_output" \
    > "$OUTPUT_DIR/test.log" 2>&1
