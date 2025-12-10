import os
from typing import List, Callable, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from .vllm_wrapper import VLLMWrapper
from .gen_prompt import PromptDataset
from .drgrpo_grader import r1_zero_reward_fn
from .init import env_init, eval_device
from .sft_helper import get_model, print_all_gpu_memory

env_init()

model_path = os.path.abspath("./data/models/Qwen2.5-Math-1.5B")

default_test_dataset = PromptDataset(
    dataset_type="test"
)
default_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    min_tokens=4,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True
)

def evaluate_vllm(
    vllm_model: VLLMWrapper,
    reward_fn: Callable[[str, str], dict[str, float]] = r1_zero_reward_fn,
    prompt_dataset: PromptDataset = default_test_dataset,
    eval_sampling_params: SamplingParams = default_sampling_params,
) -> Tuple[float, float]:
    """
        Evaluate a language model on a list of prompts,
        compute evaluation metrics, and serialize results to disk.
    """
    total_prompts_num = 0
    format_correct_num, result_correct_num = 0, 0
    for prompts, _, ground_truths in prompt_dataset.evaluate_batch(10000):
        if len(prompts) != len(ground_truths):
            raise ValueError("len(prompts) != len(ground_truths)")

        total_prompts_num += len(prompts)

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = vllm_model.generate(prompts, eval_sampling_params)
        for idx, (output, label) in enumerate(tqdm(zip(outputs, ground_truths), total=len(outputs))):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            reward_dict = reward_fn(generated_text, label)
            print(f"prompt: {prompt!r},\noutput: {generated_text!r},\nlabel: {label!r},"
              f"\nreward: {reward_dict}")
            format_correct_num += reward_dict["format_reward"]
            result_correct_num += reward_dict["reward"]

    return format_correct_num / total_prompts_num, result_correct_num / total_prompts_num

if __name__ == "__main__":
    policy, _, inf_vllm = get_model(model_path="./data/models/Qwen2.5-Math-1.5B-test")
    print_all_gpu_memory("before load policy")
    inf_vllm.load_policy_into_vllm(policy)
    print_all_gpu_memory("after load policy")
    format_accuracy, accuracy = evaluate_vllm(
        inf_vllm
    )

    print(f"format_accuracy: {format_accuracy}, accuracy: {accuracy}")