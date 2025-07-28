import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_finetuned_models(args):
    print("加载蒸馏后的学生模型...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False,
    )

    student_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    return student_model, student_tokenizer


def load_finetuned_lora(args):
    print("加载蒸馏后的学生模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False,
    )

    student_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    student_model = PeftModel.from_pretrained(
        base_model,
        args.output_dir,
        is_trainable=False
    )
    student_model = student_model.merge_and_unload()
    return student_model, student_tokenizer


# 7. 评估函数
def evaluate_cot_ability(model, tokenizer, test_questions, args):
    model.eval()
    results = []
    with torch.inference_mode():
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            full_system_prompt = f.read().strip()

        for question in tqdm(test_questions, desc="评估思考过程能力"):
            question = str(question).strip()
            messages = [
                {"role": "system", "content": "你是一个专业的ASR文本的修复助手。"},
                {"role": "user", "content": full_system_prompt + question}
            ]

            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=3048,
                temperature=0.6,
                top_p=0.95,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"问题: {question}")
            print(f"思考过程: {response}" if response else "无思考过程")
            results.append({
                "question": question,
                "reasoning": response,
                "full_response": response
            })

        return results


def response_split(response):
    if "最终输出:" in response:
        parts = response.split("最终输出:")
        reasoning = parts[0].strip()
        output = parts[1].strip()
    else:
        reasoning = response
        output = ""
    return reasoning, output
