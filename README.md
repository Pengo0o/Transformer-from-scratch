# Transformer 文本摘要 - CNN/DailyMail

基于 Transformer 的新闻文本摘要模型，在 CNN/DailyMail 数据集上训练。

## 📁 项目结构

```
home_work/
├── src/
│   ├── model.py          # Transformer 模型定义
│   ├── trainer.py        # 训练器
│   └── tester.py         # 测试器（推理和评估）
├── dataset/
│   ├── dataset.py        # CNN/DailyMail 数据集加载
│   ├── train/            # 训练集（287,113 样本）
│   ├── validation/       # 验证集（13,368 样本）
│   └── test/             # 测试集（11,490 样本）
├── scripts/
│   └── run.sh            # 快速运行脚本
├── results/              # 模型保存目录
├── config.yaml           # 配置文件（YAML格式）
├── main.py               # 主程序（训练+测试）
├── requirements.txt      # Python 依赖
└── README.md             # 本文件
```

## 🔧 环境配置

### 安装依赖

```bash

conda create -n transformer python=3.10
conda activate transformer
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

需要的包：
- torch >= 2.0.0
- datasets >= 2.14.0
- tqdm >= 4.65.0
- numpy >= 1.24.0
- rouge-score >= 0.1.2
- nltk >= 3.8
- matplotlib >= 3.7.0
- pyyaml >= 6.0

## 🚀 快速开始

### 1. 训练模型

```bash
python main.py --mode train --config config.yaml
```

从检查点恢复训练：

```bash
python main.py --mode train --resume results/latest_model.pt
```

### 2. 测试模型（生成摘要）

```bash

# 执行 ROUGE 评估
python main.py --mode test --evaluate --num_samples 1000

# 保存测试结果
python main.py --mode test --save_results --show_examples

# 使用指定检查点
python main.py --mode test --checkpoint results/best_model.pt --show_examples
```

### 3. 完整流程（使用脚本）

```bash
bash scripts/run.sh
```

## ⚙️ 配置说明

所有配置都在 `config.yaml` 文件中管理，包括：

### 数据配置
- `dataset_path`: 数据集路径
- `max_vocab_size`: 最大词汇量（默认 50,000）
- `src_max_len`: 源序列最大长度（默认 512）
- `tgt_max_len`: 目标序列最大长度（默认 150）

### 模型配置
- `d_model`: 嵌入维度（默认 256）
- `num_heads`: 注意力头数（默认 8）
- `num_layers`: Encoder/Decoder 层数（默认 4）
- `d_ff`: 前馈网络维度（默认 1024）
- `dropout`: Dropout 率（默认 0.1）

### 训练配置
- `batch_size`: 批次大小（默认 8）
- `num_epochs`: 训练轮数（默认 10）
- `learning_rate`: 学习率（默认 0.0001）
- `warmup_steps`: 预热步数（默认 4000）
- `label_smoothing`: 标签平滑（默认 0.1）

### 测试配置
- `batch_size`: 测试批次大小（默认 16）
- `decode_method`: 解码方法（`greedy` 或 `beam_search`）
- `beam_width`: Beam Search 宽度（默认 5）
- `max_generate_len`: 最大生成长度（默认 150）

可以直接编辑 `config.yaml` 来调整这些参数。

## 📊 数据集信息

**CNN/DailyMail 3.0.0**

- **数据集链接**: https://huggingface.co/datasets/cnn_dailymail
- **任务**: 新闻文本摘要（Abstractive Summarization）
- **格式**: 
  - 输入（article）: 新闻文章全文
  - 输出（highlights）: 文章摘要
- **数据量**:
  - 训练集: 287,113 条
  - 验证集: 13,368 条
  - 测试集: 11,490 条


## 🏗️ 模型架构

标准 Transformer Encoder-Decoder 架构：

### Encoder
- Multi-Head Self-Attention
- Position-wise Feed-Forward
- Layer Normalization
- Residual Connection
- Positional Encoding (正弦余弦)

### Decoder
- Masked Multi-Head Self-Attention
- Multi-Head Cross-Attention
- Position-wise Feed-Forward
- Layer Normalization
- Residual Connection
- Positional Encoding

### 关键设计
- **位置编码**: 使用正弦余弦函数，支持任意长度序列
- **缩放**: Embedding 乘以 √d_model 平衡位置编码
- **Mask**: 
  - Padding mask: 屏蔽填充位置
  - No-peek mask: 防止 Decoder 看到未来信息
- **标签平滑**: label_smoothing=0.1 提高泛化
- **学习率调度**: Warmup + Decay（Transformer 原论文策略）


## 📝 许可证

MIT License
