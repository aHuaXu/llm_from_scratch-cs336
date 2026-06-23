import os
import random
from typing import BinaryIO, IO

import torch

def log_softmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    log_exp_x = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return x - log_exp_x

def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (torch.Tensor): The input tensor of shape (batch_size, vocab_size).
        targets (torch.Tensor): The target tensor of shape (batch_size,).

    Returns:
        torch.Tensor: The average cross-entropy loss across examples.
    """

    # this method has loss, and lead to wrong result
    # softmax_inputs = softmax(inputs, -1)
    # gather_inputs = softmax_inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # return torch.mean(-torch.log(softmax_inputs))

    """
    • Subtract the largest element for numerical stability.
    • Cancel out log and exp whenever possible.
    """
    log_softmax_inputs = log_softmax(inputs, dim=-1)
    gather_inputs = log_softmax_inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.mean(-gather_inputs)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(data, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    data = torch.load(src)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    return data['iteration']