import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


# 数据处理函数
def dataset_jsonl_transfer(origin_path, new_path, prompt: str = ""):
    """
    将原始数据转换为jsonl格式，方便后续处理
    """
    messages = []
    try:
        with open(origin_path, "r", encoding="utf-8") as f:
            datas = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {origin_path}. It might be empty or malformed.")
        return

    random.shuffle(datas)
    for data in datas:
        message = {
            "instruction": prompt,
            "input": data.get("noisy_text", ""),
            "output": f"<think>{data.get('think', '')}</think> \n 最终输出:{data.get('output', '')}"  # noqa E231
        }
        messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


# 数据预处理函数
def process_func(example, tokenizer, max_length=2048, prompt: str = ""):
    """
    处理单个数据样本，将其转换为Qwen模型的输入格式
    """
    system_prompt = "<|im_start|>system\n你是一个专业的ASR文本的修复助手。<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{prompt}{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{example['output']}<|im_end|>\n"

    instruction = tokenizer(system_prompt + user_prompt, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_model(args):
    print("使用全量微调模式...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用 fp16
    )
    return model


def load_lora_model(args):
    print("使用LoRA微调模式...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        revision="main",
        resume_download=True,
        torch_dtype=torch.float16,  # 确保这里也指定fp16
        use_cache=False,
    )
    # LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # noqa: E501  针对Qwen模型常用LoRA目标模块
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数量
    return model


def load_tokenizer(args):
    """
    加载模型分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='right'
    )
    # Qwen tokenizer 需要 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    if os.environ.get('LOCAL_RANK') is not None:
        torch.distributed.init_process_group(backend="nccl")
    # --- 1. 解析参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--train_max_length", type=int, default=2048)
    # LoRA 相关参数
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA的缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA的Dropout率")

    args = parser.parse_args()

    # --- 2. 加载模型和分词器 ---
    tokenizer = load_tokenizer(args)

    # --- 3. 加载模型 ---
    if args.use_lora:
        model = load_lora_model(args)
    else:
        model = load_model(args)

    model.config.use_cache = False

    # --- 4. 准备数据集 ---
    # 仅主进程执行数据转换
    if int(os.environ.get("RANK", 0)) == 0:
        print("Preparing dataset...")
        prompt_content = open(args.prompt_path, "r", encoding="utf-8").read()
        train_jsonl_path = os.path.join(args.output_dir, "train_formatted.jsonl")
        val_jsonl_path = os.path.join(args.output_dir, "val_formatted.jsonl")
        dataset_jsonl_transfer(args.train_dataset_path, train_jsonl_path, prompt_content)
        dataset_jsonl_transfer(args.val_dataset_path, val_jsonl_path, prompt_content)

    # 所有进程等待主进程完成数据准备
    torch.distributed.barrier()

    train_jsonl_path = os.path.join(args.output_dir, "train_formatted.jsonl")
    val_jsonl_path = os.path.join(args.output_dir, "val_formatted.jsonl")

    train_dataset = Dataset.from_json(train_jsonl_path)
    eval_dataset = Dataset.from_json(val_jsonl_path)

    prompt_content = open(args.prompt_path, "r", encoding="utf-8").read()
    train_dataset = train_dataset.map(
        process_func,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.train_max_length, "prompt": prompt_content},
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        process_func,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.train_max_length, "prompt": prompt_content},
        remove_columns=eval_dataset.column_names
    )

    # --- 5. 设置训练参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        deepspeed=args.deepspeed,
        # 从args中获取所有训练相关参数
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        fp16=True,
        gradient_checkpointing=False,
        report_to=args.report_to,
    )

    # --- 6. 初始化 Trainer 并开始训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("Starting training...")
    trainer.train()

    print("Training finished. Saving final model...")
    # Trainer 会根据是否是PeftModel自动保存LoRA适配器或完整模型
    trainer.save_model(os.path.join(args.output_dir, "final_checkpoint"))
    print(f"Final model saved to {os.path.join(args.output_dir, 'final_checkpoint')}")


if __name__ == "__main__":
    main()
