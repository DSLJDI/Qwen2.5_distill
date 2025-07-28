# Qwen2.5_distill
## 项目概述

本项目基于Qwen2.5模型，能够进行指令微调和全量微调。

## 数据准备

<details>
  <summary>1) 你在clone本仓库后，在仓库主文件夹下新建/data文件夹</summary>

  <summary>2) 在/data文件夹下新建train_data.json和val_data.json文件，json文件的数据组织格式如下：</summary>

  ```json
  [
    {
        "noisy_text": "...",
        "think": "...",
        "output": {
            "output": "...",
            "if_complete": "是或否"
        }
    },
    ...
  ]
  ```

<summary>3) 在/data文件夹下新建prompt.txt文件，文件内容如下：</summary>

```bash
你负责对 ASR(语音转文本) 的输出文本做修复。

我们的场景是 MOBA 类型的游戏《王者荣耀》，玩家可以在游戏中通过语音与别的玩家交流，语音被 ASR 模块转换为文本。需要注意的是，尽管文本是《王者荣耀》场景下的文本，但文本内容也会存在与游戏无关为用户闲聊的情况，且由于以下原因 ASR 文本并不完美，需要修复。

第一，玩家发言包含一些不流利的因素，导致 ASR 预测出的文本不便于其他玩家理解，需要流利化：
...

```
</details>

## 训练设置
<details>
  <summary>1) 设置代码环境</summary>
  <br>

  - 我们的环境如下：

  ```
  CUDA 12.1

  Python 3.12

  PyTorch 2.2
  ```

  - 你需要安装的包在 [requirements.txt](https://git.woa.com/dslleisu/asr_llm/blob/master/requirements.txt)

  ```bash
pip install -r requirements.txt
  ```

</details>

<details>
  <summary>2) 下载Qwen2.5系列的预训练模型</summary>
  <br>

  - 你需要从Hugging Face下载Qwen2.5系列的预训练模型，下载链接为：[Qwen2.5](https://huggingface.co/Qwen/)
  - 建议3B以上的模型直接下载到本地，避免占用本地缓存空间（对于3B以下模型，可以直接采用Hugging Face加载的方式如"Qwen/Qwen2.5-0.5B-Instruct"）.

</details>
<details>
  <summary>3) 配置训练参数</summary>

  ```bash
  # 路径配置
  TRAIN_DATA="你的训练数据路径"
  VAL_DATA="你的验证数据路径"
  OUTPUT_DIR="模型输出路径"
  ORIGINAL_MODEL="预训练模型路径"
  prompt_path="prompt文件路径"

  # 训练参数
  fp16=true                  # 开启FP16训练
  use_lora=true              # 启用LoRA微调
  gradient_checkpointing=false # LoRA微调时关闭梯度检查点
  ```

</details>

## 测试设置

<details>
  <summary>1) 设置代码环境</summary>
  <br>

  - 和训练部分一致。


</details>

<details>
  <summary>2) 生成蒸馏模型的权重参数</summary>
  <br>

  - 如果采用DeepSpeed进行训练，你需要在训练完成后，执行以下代码，生成蒸馏模型的权重参数，生成的权重参数会存储在模型输出路径下的final_checkpoint文件夹中。

  ```bash
python OUTPUT_DIR/final_checkpoint/zero_to_fp32.py OUTPUT_DIR/final_checkpoint OUTPUT_DIR/final_checkpoint
  ```

</details>

<details>
  <summary>3) 配置测试参数</summary>

  ```bash
  # 路径配置
  TEST_DATA="你的测试文件路径，要求为CSV文件"
  OUTPUT_DIR="模型输出路径下的final_checkpoint文件夹"
  prompt_path="prompt文件路径"

  # 训练参数
  input_cloumn_name                 # 用于测试的数据所在的列名
  output_cloumn_name              # 模型的输出结果存储的列名
  use_lora=true              # 启用LoRA微调
  ```

</details>

## 脚本

接下来超级简单，只需要运行：

 - 对于训练
```
sh run_train.sh
```

 - 对于测试
```
sh run_test.sh
```

