import torch
from typing import Callable, List, Dict, Tuple, Literal, Optional


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
    # for res, gt in zip(rollout_responses, repeated_ground_truths):
    #     reward = reward_fn(res, gt)
    #     print(f"response: {res}\ntruth:{gt}\nreward: {reward}")
    
    raw_rewards = torch.Tensor([reward_fn(response, ground_truth)['reward'] for response, ground_truth in zip(rollout_responses, repeated_ground_truths)])
    
    rewards_grouped = raw_rewards.reshape(-1, group_size)
    rewards_grouped_mean = rewards_grouped.mean(dim=-1, keepdim=True)
    
    if normalize_by_std:
        rewards_grouped_std = rewards_grouped.std(dim=-1, keepdim=True)
        advantages = (rewards_grouped - rewards_grouped_mean) / (rewards_grouped_std + advantage_eps)
        # print(f"rewards_grouped_mean: {rewards_grouped_mean}, rewards_grouped_std: {rewards_grouped_std}, advantages: {advantages}")
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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    Implementation tips:
    • Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    Implementation tips:
    • Broadcast advantages over sequence_length.
    """
    log_probs_div = torch.exp(policy_log_probs) / torch.exp(old_log_probs)
    clipped_log_probs_div = torch.clamp(log_probs_div, min=1-cliprange, max=1+cliprange)
    
    lhs = advantages * log_probs_div
    rhs = advantages * clipped_log_probs_div
    
    loss = torch.min(lhs, rhs)
    
    metadata: Dict[str, torch.Tensor] = {
        "is_clipped": (loss != lhs),
        "clipped_ratio": (loss != lhs).float().mean().item(),
    }
    
    return -loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for"reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        old_log_probs: Required for"grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for"grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    Implementation tips:
    • Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss.
    • Perform argument checks (see assertion pattern above).
    • Aggregate any returned metadata into a single dict.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for loss_type='no_baseline'"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for loss_type='reinforce_with_baseline'"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert old_log_probs is not None, "old_log_probs is required for loss_type='grpo_clip'"
        assert cliprange is not None, "cliprange is required for loss_type='grpo_clip'"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")
    
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange: Clip parameter ϵ for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
              this so we can log it.
            - metadata: Dict with metadata from the underlying loss call, and any other statistics you
              might want to log.

    Implementation tips:
    • You should call loss.backward() in this function. Make sure to adjust for gradient
      accumulation.
    """
    loss_tensor, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)

    loss = masked_mean(loss_tensor, response_mask)/gradient_accumulation_steps
    loss.backward()

    metadata['loss'] = loss
    return loss, metadata