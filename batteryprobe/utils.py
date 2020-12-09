"""Utility functions."""


import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def pad_and_pack(data):
    """Pad and pack a list of tensors of variable size."""
    lengths = [el.shape[0] for el in data]
    # If tensors are numpy arrays
    if all(isinstance(element, np.ndarray) for element in data):
        data = [torch.from_numpy(el) for el in data]
    data = pad_sequence(data, batch_first=True, padding_value=-999)
    data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
    return data


# pylint: disable=C0103
def masked_l1(inputs, targets):
    """Masked L1 loss."""
    mask = (inputs != -999)
    return torch.abs(inputs - targets).sum() / mask.sum()
