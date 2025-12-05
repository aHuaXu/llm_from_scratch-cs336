import random
from typing import List, Optional, Callable, Tuple, Literal

from datasets import Dataset, load_dataset
import pandas as pd
import re
import os


class PromptDataset:
    """
    管理 GRPO 任务的 Prompt 数据集（支持纯问题、R1-Zero Prompt、自定义模板）
    核心能力：加载 Prompt、格式化 Prompt、批次采样
    """

    def __init__(
        self,
        # raw_question: List[str],  # 原始数据（纯问题/R1-Zero 原始文本/自定义内容）
        # raw_ground_truths: List[str],
        prompt_type: Literal[
            "question", "r1_zero", "custom"
        ] = "r1_zero",  # prompt 类型：raw/r1_zero/custom
        prompt_template: Optional[Callable[[str], str]] = None,  # 自定义格式化模板
        r1_zero_template_path: str = "./prompts/r1_zero.prompt",
        dataset_id: str = "openai/gsm8k",
        dataset_type: Literal["train", "test"] = "train",
    ):
        """
        Args:
            raw_question: 原始数据列表（如 ["什么是GRPO？", "USER: xxx ASSISTANT: "]）
            prompt_type: Prompt 类型，可选：
                - "question": 纯问题
                - "r1_zero": R1-Zero 格式的 Prompt
                - "custom": 自定义 Prompt（需传入 prompt_template）
            prompt_template: 自定义格式化函数（输入 raw_question 单条数据，输出格式化 Prompt）
        """
        # assert len(raw_question) == len(raw_ground_truths), "raw_question and raw_ground_truths must have the same length"
        # self.raw_question = raw_question
        # self.raw_ground_truths = raw_ground_truths
        self.prompt_type = prompt_type
        self.prompt_template = prompt_template
        self.r1_zero_template_path = r1_zero_template_path
        self.r1_zero_template = self._load_r1_zero_template()
        self.dataset_id = dataset_id
        self.dataset_type = dataset_type
        self.dataset = self._load_dataset()

        # 预处理：将原始数据转为最终用于采样的 Prompt 列表
        self._preprocess_prompts()
        print(
            f"✅ PromptDataset 初始化完成：共加载 {len(self.dataset)} 条数据，类型：{prompt_type}"
        )

    def _load_dataset(self) -> Dataset:
        """从 Hugging Face 加载数据集"""
        """从本地加载 gsm8k 数据集（绕过网络问题）"""

       # 本地 parquet 文件路径（根据你手动下载的路径调整）
        data_dir = "./data/gsm8k"
        train_file = os.path.join(data_dir, "train-00000-of-00001.parquet")
        test_file = os.path.join(data_dir, "test-00000-of-00001.parquet")

        # 校验文件是否存在
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(
                f"本地数据集文件不存在！请确认以下文件已下载：\n"
                f"- {train_file}\n"
                f"- {test_file}\n"
                f"下载地址：https://huggingface.co/datasets/openai/gsm8k/tree/main/data"
            )
        if self.dataset_type == "train":
            df = pd.read_parquet(train_file)  # pandas 读取本地 parquet
        elif self.dataset_type == "test":
            df = pd.read_parquet(test_file)
        else:
            raise ValueError(f"无效的 dataset_type：{self.dataset_type}，仅支持 train/test")

        # pandas DataFrame 转 Hugging Face Dataset（关键：绕开 LocalFileSystem）
        dataset = Dataset.from_pandas(df)
        
        ANSWER_PATTERN = r"(?i)(?:the answer is|answer:|final answer:|####)\s*([+-]?\d+(?:\.\d+)?)"

        def extract_ground_truth(example):
            match = re.search(ANSWER_PATTERN, example["answer"])
            example["ground_truth"] = float(match.group(1)) if match else None
            return example

        dataset = dataset.map(extract_ground_truth, batched=False)
        return dataset

    def _load_r1_zero_template(self) -> str:
        """从文件读取 R1-Zero 模板文本"""
        try:
            with open(self.r1_zero_template_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            print(f"✅ 成功加载 R1-Zero 模板：{self.r1_zero_template_path}")
            return template
        except FileNotFoundError:
            raise ValueError(f"R1-Zero 模板文件不存在：{self.r1_zero_template_path}")
        except Exception as e:
            raise RuntimeError(f"读取 R1-Zero 模板失败：{e}")

    def _preprocess_prompts(self):
        """预处理原始数据，生成最终的 Prompt 列表"""

        def gen_prompt(example):
            if self.prompt_type == "question":
                example["prompt"] = example["question"]
            elif self.prompt_type == "r1_zero":
                example["prompt"] = self._format_r1_zero_prompt(example["question"])
            elif self.prompt_type == "custom":
                if self.prompt_template is None:
                    raise ValueError(
                        "使用 custom 类型时必须传入 prompt_template 函数！"
                    )
                example["prompt"] = self.prompt_template(example["question"])
            else:
                raise ValueError(
                    f"不支持的 prompt_type：{self.prompt_type}，可选：question/r1_zero/custom"
                )
            return example

        self.dataset = self.dataset.map(gen_prompt, batched=False)

    def _format_r1_zero_prompt(self, question: str) -> str:
        """核心：将原始问题填充到 R1-Zero 模板的 {question} 占位符"""
        # 替换占位符，生成完整 R1-Zero Prompt
        return self.r1_zero_template.replace("{question}", question)

    def sample_batch(self, batch_size: int) -> Tuple[List[str], List[str]]:
        """
        从 Prompt 列表中采样批次数据（对齐 GRPO 伪代码的 Sample Db 步骤）
        Args:
            batch_size: 批次大小
        Returns:
            Tuple[List[str], List[str]].
                questions: List[str] 采样后的 Prompt 批次
                ground_truths: List[str] 采样后的 Ground Truth 批次
        """
        batch_size = min(batch_size, len(self.dataset))
        # 无放回采样（保证批次多样性，若需放回可改用 random.choices）
        one_batch = self.dataset.select(
            random.sample(range(len(self.dataset)), k=batch_size)
        )
        return one_batch["prompt"], one_batch["ground_truth"]


if __name__ == "__main__":
    prompt_dataset = PromptDataset()
    prompts, answers = prompt_dataset.sample_batch(10)
    for prompt, answer in zip(prompts, answers):
        print(prompt)
        print(answer)
        print("--------------------------------")
