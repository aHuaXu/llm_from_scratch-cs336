import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# it cannot run one epoch data
def get_batch(
    dataset: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    # [0, n-m-1]
    assert batch_size <= n - context_length - 1

    start_indices = torch.tensor(
        random.sample(range(n - context_length), batch_size),
        # device=device,
        dtype=torch.int64
    )
    offset = torch.arange(context_length, dtype=torch.int)

    x_indices = start_indices.unsqueeze(1) + offset

    x, y = dataset[x_indices], dataset[x_indices+1]
    return x.to(device), y.to(device)


class DatasetForTransformer(Dataset):
    def __init__(
        self,
        data_path: str,
        context_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.int,
    ):
        self.context_length = context_length
        self.data = np.memmap(data_path, mode="r", dtype=dtype)
        self.device = device

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx+self.context_length]
        y = self.data[idx+1:idx+self.context_length+1]
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device)
        )