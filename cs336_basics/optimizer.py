from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
# import torch.jit as jit


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate for this parameter group
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get optimizer state associated with parameter p
                t = state.get("t", 0)  # Get iteration number from state, defaulting to 0
                grad = p.grad.data  # Get gradient of loss with respect to parameter p
                p.data -= lr / math.sqrt(t + 1) * grad  # Update parameter using adaptive learning rate
                state["t"] = t + 1  # Increment iteration counter

        return loss
    
"""
AdamW Optimizer Algorithm

This implements the AdamW (Adam with Weight Decay) optimization algorithm as described in
"Decoupled Weight Decay Regularization" by Loshchilov & Hutter (2017).

Algorithm:
1. Initialize parameters θ, first moment vector m = 0, second moment vector v = 0
2. For each iteration t = 1, ..., T:
   a. Sample batch of data B_t
   b. Compute gradient: g = ∇_θ L(θ; B_t)
   c. Update biased first moment estimate: m = β₁m + (1-β₁)g
   d. Update biased second moment estimate: v = β₂v + (1-β₂)g²
   e. Compute bias-corrected learning rate: α_t = α * √(1-β₂^t) / (1-β₁^t)
   f. Update parameters: θ = θ - α_t * m / (√v + ε)
   g. Apply weight decay: θ = θ - α * λ * θ

Parameters:
- α (lr): Learning rate
- β₁ (beta1): Exponential decay rate for first moment estimates (default: 0.9)
- β₂ (beta2): Exponential decay rate for second moment estimates (default: 0.999)
- ε (eps): Small constant for numerical stability (default: 1e-8)
- λ (weight_decay): Weight decay coefficient (default: 1e-8)
"""
# @jit.script
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    # 1. warn up
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)))/2
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float=5) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad.data for p in parameters if p.grad is not None]

    total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    
    for t in range(20):
        opt.zero_grad()  # Reset gradients for all learnable parameters
        loss = (weights**2).mean()  # Compute scalar loss value (L2 regularization)
        print(loss.cpu().item())
        loss.backward()  # Run backward pass to compute gradients
        opt.step()  # Execute optimizer step to update parameters