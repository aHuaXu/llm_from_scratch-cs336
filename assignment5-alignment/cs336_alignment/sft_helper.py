from typing import List, Dict, Tuple
import torch
from torchtyping import TensorType
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import PaddingStrategy
from .vllm_wrapper import VLLMWrapper
from .init import train_device, eval_device
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings.
            Then the returned dictionary should have the following keys:

            - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              the tokenized prompt and output strings, with the final token sliced off.

            - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              shifted input ids, i.e., the input ids without the first token.

            - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1)
              a mask on the response tokens in the labels.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("len(prompt_strs) != len(output_strs)")

    # tokenize
    prompt_tokens = tokenizer(prompt_strs, padding=False, truncation=False, return_tensors=None)
    output_tokens = tokenizer(output_strs, padding=False, truncation=False, return_tensors=None)

    batch_size = len(prompt_strs)
    # concatenated_ids = torch.cat([prompt_tokens["input_ids"], output_tokens["input_ids"]], dim=1)
    concatenated_ids = []
    for i in range(batch_size):
        concatenated_ids.append(prompt_tokens["input_ids"][i] + output_tokens["input_ids"][i])

    padded = tokenizer.pad(
        {"input_ids": concatenated_ids},
        padding=PaddingStrategy.LONGEST,
        return_tensors="pt",
    )
    padded_input_ids = padded["input_ids"]
    max_seq_len = padded_input_ids.shape[1]

    input_ids = padded_input_ids[:, :-1] # (batch_size, max_seq_len-1)
    labels = padded_input_ids[:, 1:] # (batch_size, max_seq_len-1)

    response_mask = torch.zeros_like(input_ids)
    for i in range(batch_size):
        prompt_len = len(prompt_tokens["input_ids"][i])
        output_end = prompt_len + len(output_tokens["input_ids"][i]) - 1
        response_mask[i, prompt_len-1:output_end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
            prediction.

    Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    """
    lsm = torch.logsumexp(logits, dim=-1, keepdim=True)

    log_probs = logits - lsm
    probs = torch.exp(log_probs)

    return -torch.sum(log_probs * probs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.
    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ(xt | x<t).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    model_output = model(input_ids)
    logics = model_output.logits

    # log(softmax(logics))
    logics_probs = logics - torch.logsumexp(logics, dim=-1, keepdim=True)

    # log[softmax(logics)]_y
    label_probs = torch.gather(logics_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    res = {'log_probs': label_probs}
    if return_token_entropy:
        # res['token_entropy'] = compute_entropy(logics)
        res['token_entropy'] = -torch.sum(logics_probs * torch.exp(logics_probs), dim=-1)
    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to the sum.
    """
    assert normalize_constant != 0.0
    return torch.sum(tensor * mask, dim=dim)/normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    """
    # also normalize in the batch
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant*policy_log_probs.shape[0])

    microbatch_loss = loss/gradient_accumulation_steps
    microbatch_loss.backward()

    metadata = {
        "log_probs": policy_log_probs.detach(),
        "response_mask": response_mask.detach(),
        "raw_loss": loss.detach(),
        "microbatch_loss": microbatch_loss.detach(),
    }
    return microbatch_loss, metadata

def print_gpu_memory(device_id=0, operate_type: str = ""):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    print(f"After {operate_type} GPU {device_id} 总显存: {mem_info.total/1024**3:.2f} GB")
    print(f"已用显存: {mem_info.used/1024**3:.2f} GB")
    print(f"可用显存: {mem_info.free/1024**3:.2f} GB")

def print_all_gpu_memory(operate_type: str = ""):
    train_physical_device, eval_physical_device = torch.device("cuda:7"), torch.device("cuda:6")
    print_gpu_memory(train_physical_device.index, operate_type)
    print_gpu_memory(eval_physical_device.index, operate_type)

def get_model(
    model_path: str = "./data/models/Qwen2.5-Math-1.5B",
    no_inf: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, VLLMWrapper]:
    # print_all_gpu_memory("init")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # print_all_gpu_memory("load policy")
    # init vllm
    inf_vllm = VLLMWrapper(
        model_id=model_path,
        device=eval_device
    ) if not no_inf else None
    # print_all_gpu_memory("init vllm")

    return policy, tokenizer, inf_vllm

def save_policy(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dir_path: str = "./data/models/Qwen2.5-Math-1.5B-test",
):
    policy.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)
