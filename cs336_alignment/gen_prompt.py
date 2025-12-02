import random
from typing import List, Optional, Callable


class PromptDataset:
    """
    管理 GRPO 任务的 Prompt 数据集（支持纯问题、R1-Zero Prompt、自定义模板）
    核心能力：加载 Prompt、格式化 Prompt、批次采样
    """
    def __init__(
        self,
        raw_question: List[str],  # 原始数据（纯问题/R1-Zero 原始文本/自定义内容）
        raw_ground_truths: List[str],  
        prompt_type: str = "r1_zero",  # prompt 类型：raw/r1_zero/custom
        prompt_template: Optional[Callable[[str], str]] = None,  # 自定义格式化模板
        r1_zero_template_path: str = "./prompts/r1_zero.prompt",
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
        assert len(raw_question) == len(raw_ground_truths), "raw_question and raw_ground_truths must have the same length"
        self.raw_question = raw_question
        self.raw_ground_truths = raw_ground_truths
        self.prompt_type = prompt_type
        self.prompt_template = prompt_template
        self.r1_zero_template_path = r1_zero_template_path
        self.r1_zero_template = self._load_r1_zero_template()

        # 预处理：将原始数据转为最终用于采样的 Prompt 列表
        self.prompts = self._preprocess_prompts()
        print(f"✅ PromptDataset 初始化完成：共加载 {len(self.prompts)} 条 Prompt，类型：{prompt_type}")


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

    def _preprocess_prompts(self) -> List[str]:
        """预处理原始数据，生成最终的 Prompt 列表"""
        if self.prompt_type == "question":
            return self.raw_question
        elif self.prompt_type == "r1_zero":
            return [self._format_r1_zero_prompt(ques) for ques in self.raw_question]
        elif self.prompt_type == "custom":
            # 自定义 Prompt → 用传入的模板函数格式化
            if self.prompt_template is None:
                raise ValueError("使用 custom 类型时必须传入 prompt_template 函数！")
            return [self.prompt_template(raw) for raw in self.raw_question]
        else:
            raise ValueError(f"不支持的 prompt_type：{self.prompt_type}，可选：question/r1_zero/custom")

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
        batch_size = min(batch_size, len(self.prompts))
        # 无放回采样（保证批次多样性，若需放回可改用 random.choices）
        indices = random.sample(range(len(self.prompts)), k=batch_size)
        questions = [self.prompts[i] for i in indices]
        ground_truths = [self.raw_ground_truths[i] for i in indices]
        return questions, ground_truths

    def add_prompts(self, new_prompts: List[str]):
        """动态添加新的 Prompt 到数据集（可选扩展）"""
        self.raw_question.extend(new_prompts)
        self.prompts = self._preprocess_prompts()  # 重新预处理

    def get_all_prompts(self) -> List[str]:
        """获取所有格式化后的 Prompt（用于验证/测试）"""
        return self.prompts