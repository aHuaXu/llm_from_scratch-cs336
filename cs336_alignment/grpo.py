import torch
from typing import Callable, List, Dict, Tuple


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys"reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
        response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
        response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rollout_batch_size = len(rollout_responses)
    
    raw_rewards = torch.Tensor([reward_fn(response, ground_truth)['reward'] for response, ground_truth in zip(rollout_responses, repeated_ground_truths)])
    
    rewards_grouped = raw_rewards.reshape(-1, group_size)
    rewards_grouped_mean = rewards_grouped.mean(dim=-1, keepdim=True)
    
    if normalize_by_std:
        rewards_grouped_std = rewards_grouped.std(dim=-1, keepdim=True)
        advantages = (rewards_grouped - rewards_grouped_mean) / (rewards_grouped_std + advantage_eps)
    else:
        advantages = rewards_grouped - rewards_grouped_mean
        
    metadata: Dict[str, float] = {
        "rewards_grouped_mean": rewards_grouped_mean.mean().item(),
        "rewards_grouped_max": rewards_grouped.max().item(),
        "rewards_grouped_min": rewards_grouped.min().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_max": advantages.max().item(),
        "advantages_min": advantages.min().item(),
    }
    
    return advantages.flatten(), raw_rewards, metadata