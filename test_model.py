import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from util.evaluation import evaluate_cot_ability, load_finetuned_lora, load_finetuned_models, response_split


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def eval(args):
    # 加载模型
    if args.use_lora:
        model, tokenizer = load_finetuned_lora(args)
    else:
        model, tokenizer = load_finetuned_models(args)

    model.eval()

    # 准备数据集
    df = pd.read_csv(args.test_dataset_path)
    dataset = TextDataset(df[args.input_cloumn_name].tolist())

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    final_results = []
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="评估进度"):
            # 直接在单个进程中进行评估
            batch_results = evaluate_cot_ability(model, tokenizer, batch_texts, args)
            final_results.extend(batch_results)

    # 直接处理和保存结果
    outputs = []
    for result in final_results:
        _, output = response_split(result["full_response"])
        outputs.append(output)

    df[args.output_cloumn_name] = outputs
    df.to_csv(args.test_dataset_path, index=False, encoding="utf-8-sig")
    print(f"评估完成，结果已保存至 {args.test_dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_path", required=True)
    parser.add_argument("--output_dir", default="./cot_distill_results_v6_1w")
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_lora", action='store_true', help="是否使用LoRA微调")
    parser.add_argument("--input_cloumn_name", default="input", help="作为模型输入的那一列的列名")
    parser.add_argument("--output_cloumn_name", default="model_output", help="模型输出的那一列的列名")
    args = parser.parse_args()
    eval(args)
