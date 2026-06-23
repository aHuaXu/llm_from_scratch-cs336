# CS336: Language Modeling from Scratch (Stanford Spring 2025)

[Stanford CS336](https://stanford-cs336.github.io/) 五个作业的个人实现 by **aHuaXu**。

本仓库是一个 monorepo，把课程的 5 个 assignment 合并到了一起，并完整保留了各自的 git 提交历史。每个子目录是一个独立的 `uv` 项目，拥有自己的 `.venv`、`pyproject.toml` 和 `uv.lock`。

## 作业总览

| Assignment | 主题 | 内容简介 | 子 README |
|---|---|---|---|
| **1 — Basics** | 从零实现 LM | 手写 Transformer（Linear / Embedding / RMSNorm / SwiGLU / RoPE / MHA）、BPE 分词器、AdamW 优化器，并在 TinyStories / OpenWebText 上训练 | [assignment1-basics](./assignment1-basics/README.md) |
| **2 — Systems** | 分布式训练 | 基于 `torch.distributed` 实现 DDP（naive → flat → overlap → bucketed）、optimizer state sharding，复用 assignment 1 的 LM | [cs336_assignment2](./cs336_assignment2/README.md) |
| **3 — Scaling** | 缩放律 | IsoFLOP 分析、用 `scipy.optimize.curve_fit` 拟合 power-law，预测最优模型/数据规模 | [assignment3-scaling](./assignment3-scaling/README.md) |
| **4 — Data** | 数据质量与过滤 | WARC/WET 解析、语言识别（fastText）、去重（MinHash）、PII 检测、毒性过滤，并训练 LM 验证数据质量 | [assignment4-data](./assignment4-data/README.md) |
| **5 — Alignment** | 对齐（SFT + GRPO） | 在 Qwen2.5-Math-1.5B 上做 SFT 与 GRPO（DeepSeek R1-Zero 风格），用 vLLM 推理，在 GSM8K / MMLU / AlpacaEval 上评测 | [assignment5-alignment](./assignment5-alignment/README.md) |

> 每个作业的完整说明见各子目录里的官方 handout PDF（`cs336_spring2025_assignmentX_*.pdf`）。

## 环境与依赖

- **Python**：3.11+（assignment 3、5 用 3.12）
- **框架**：PyTorch（~2.6 / ~2.7，按作业而定）
- **关键库**：einops、jaxtyping、tiktoken、wandb、transformers、vLLM、flash-attn
- **包管理**：所有作业统一用 [`uv`](https://github.com/astral-sh/uv)

## 快速开始

每个作业是独立环境，先 `cd` 进对应目录再操作：

```sh
cd assignment1-basics        # 或其他作业目录

uv sync                      # 安装依赖（uv run 也会自动同步）
uv run pytest                # 跑测试
uv run -m cs336_basics.train_script   # 运行（示例：assignment 1 训练）
```

测试采用 `tests/adapters.py` 桥接模式：测试从 adapters 导入，adapters 再导入你的实现，补全 `adapters.py` 即可把实现接到测试上。

### Assignment 5 特别说明

`flash-attn` 需要分两步装：

```sh
cd assignment5-alignment
uv sync --no-install-package flash-attn
uv sync
```

## 数据下载

Assignment 1 需手动下载 TinyStories 与 OpenWebText 子集：

```sh
cd assignment1-basics/data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
```

- Assignment 4：用 `get_assets.sh` 下载测试 fixtures。
- Assignment 5：从 HuggingFace 下载 Qwen2.5-Math-1.5B 到 `data/models/`。

## 目录结构

```
cs336/
├── assignment1-basics/      # 1. Basics：从零实现 Transformer LM
│   ├── cs336_basics/        #    源码：Transformer、BPE、AdamW、训练
│   └── tests/
├── cs336_assignment2/       # 2. Systems：分布式训练
│   ├── cs336-basics/        #    assignment 1 的官方参考实现
│   └── cs336_systems/       #    源码：DDP、all-reduce、bucketed sync
├── assignment3-scaling/     # 3. Scaling Laws
│   └── cs336_scaling/       #    源码：IsoFLOP 分析、power-law 拟合
├── assignment4-data/        # 4. Data Quality & Filtering
│   ├── cs336-basics/        #    官方参考实现
│   └── cs336_data/          #    源码：去重、抽取、langid、PII、质量、毒性
├── assignment5-alignment/   # 5. Alignment：SFT + GRPO
│   ├── cs336_alignment/     #    源码：SFT、GRPO、vLLM 封装、评测
│   └── scripts/
├── CLAUDE.md                # 仓库说明（给 Claude Code 的上下文）
└── README.md
```