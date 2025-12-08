import os
from typing import List, Callable, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from .vllm_wrapper import VLLMWrapper
from .gen_prompt import PromptDataset
from .drgrpo_grader import r1_zero_reward_fn
import vllm

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"  # 子进程内屏蔽其他 GPU
os.environ["VLLM_ATTENTION_BACKEND"] = "torch"

model_path = os.path.abspath("./data/models/Qwen2.5-Math-1.5B")
print(model_path)
default_vllm_wrapper = VLLMWrapper(
    model_id=model_path,
    device="cuda:0",
    seed=666,
)
default_dataset = PromptDataset(
    dataset_type="test"
)
default_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    min_tokens=4,
    max_tokens=1024,
    stop=["\n"],
    include_stop_str_in_output=True
)

def evaluate_vllm(
    vllm_model: VLLMWrapper = default_vllm_wrapper,
    reward_fn: Callable[[str, str], dict[str, float]] = r1_zero_reward_fn,
    prompt_dataset: PromptDataset = default_dataset,
    eval_sampling_params: SamplingParams = default_sampling_params,
) -> Tuple[float, float]:
    """
        Evaluate a language model on a list of prompts,
        compute evaluation metrics, and serialize results to disk.
    """
    prompts, _, ground_truths = prompt_dataset.sample_batch(64, False)

    if len(prompts) != len(ground_truths):
        raise ValueError("len(prompts) != len(ground_truths)")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    format_correct_num, result_correct_num = 0, 0
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for idx, (output, label) in enumerate(tqdm(zip(outputs, ground_truths), total=len(outputs))):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(output, label)
        print(f"prompt: {prompt!r}, output: {generated_text!r}, label: {label!r},"
              f"reward: {reward_dict}")
        format_correct_num += reward_dict["format_reward"]
        result_correct_num += reward_dict["reward"]

    return format_correct_num / len(prompts), result_correct_num / len(prompts)

if __name__ == "__main__":
    evaluate_vllm()