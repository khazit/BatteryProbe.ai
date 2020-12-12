"""Utility functions."""


from random import randint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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


# pylint: disable=R0914
def plot_sample(dataset, target_col, n=1, model=None):
    """Plot n samples from a given dataset.

    Args:
        dataset (torch.utils.data.DataLoader): dataset
        target_col (int): Target column
        n (int): number of samples to plot
        model (torch.nn.Module): model to use for predictions (optionnal)
    """
    count = 0
    for (inputs, time, context), labels in dataset:
        with torch.no_grad():
            if model is not None:
                outputs = model(inputs.float(), time.float(), context.float())
                outputs, _ = pad_packed_sequence(outputs,
                    batch_first=True, padding_value=-999)
                outputs = outputs.detach().numpy()

        inputs, inputs_len = pad_packed_sequence(inputs,
            batch_first=True, padding_value=-999)
        time, time_len = pad_packed_sequence(time,
            batch_first=True, padding_value=-999)
        labels, labels_len = pad_packed_sequence(labels,
            batch_first=True, padding_value=-999)

        fig, ax = plt.subplots(figsize=(8, 5))
        idx = randint(0, dataset.batch_size-1)
        date = mdate.epoch2num(time[idx])

        ax.plot_date(
            date[:inputs_len[idx]],
            inputs[idx, :inputs_len[idx], target_col],
            "bo-", label="Inputs"
        )
        ax.plot_date(
            date[inputs_len[idx]:time_len[idx]],
            labels[idx, :labels_len[idx], target_col],
            "ro-", label="Targets"
        )
        if model is not None:
            ax.plot_date(
                date[inputs_len[idx]:time_len[idx]],
                outputs[idx, :labels_len[idx], target_col],
                "k--", label="Predictions"
            )

        date_fmt = '%d/%m %H:%M:%S'
        date_formatter = mdate.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
        plt.ylim([0, 120])
        plt.legend()
        plt.show()

        count += 1
        if count >= n:
            break
