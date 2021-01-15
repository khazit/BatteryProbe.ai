"""Utility functions."""


from random import randint

from tqdm import tqdm
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
def masked_l1(outputs, targets):
    """Masked L1 loss."""
    mask = (outputs != -999)
    return torch.abs(outputs - targets).sum() / mask.sum()


def masked_nllloss(outputs, targets):
    """Masked NLL Loss."""
    # Center between mean and stds
    i = outputs.shape[2] // 2
    # Means
    means = outputs[:, :, :i]
    means = means[means != -999]
    # Stds
    stds = outputs[:, :, i:]
    stds = torch.clip(stds[stds != -999], min=0.01)
    # Targets
    targets = targets[:, :, :i]
    targets = targets[targets != -999]
    # Norm dist
    norm_dist = torch.distributions.Normal(means, stds)
    return -norm_dist.log_prob(targets).mean() + torch.abs(means - targets).mean()


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
            if model.use_std:
                # Confidence interval (99.7%)
                ci = outputs[idx, :labels_len[idx], 2*target_col + 1]
                ax.fill_between(
                    date[inputs_len[idx]:time_len[idx]],
                    (outputs[idx, :labels_len[idx], target_col] - ci),
                    (outputs[idx, :labels_len[idx], target_col] + ci),
                    color='b', alpha=.1)

        date_fmt = '%d/%m %H:%M:%S'
        date_formatter = mdate.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
        plt.ylim([0, 1.2])
        plt.ylabel("SoC (%)")
        plt.legend()
        plt.show()

        count += 1
        if count >= n:
            break


def diagnose(model, dataset, criterion, target_col, n=6):
    """Diagnose a model on a given dataset.

    1. Plot n worst predictions (based on loss)
    2. Plot a hist of loss values

    Args:
        model (torch.nn.Module): Model to diagnose
        dataset (torch.utils.data.DataLoader): dataset
        criterion (torch.nn.Loss): loss
        target_col (int): Index of the target column
        n (int): Number of predictions to plot
    """
    # Round n to closest even number
    n = n + 1 if n % 2 == 1 else n
    # Useful variables
    bad_losses = [0 for _ in range(n)]
    bad_elements = [None for _ in range(n)]
    pbar = tqdm(dataset)
    hist = []

    with torch.no_grad():
        for _, ((inputs, time, context), labels) in enumerate(pbar):
            # Compute predictions
            outputs = model(inputs.float(), time.float(), context.float())

            # Unpack tensors
            inputs, inputs_len = pad_packed_sequence(inputs,
                batch_first=True, padding_value=-999)
            time, _ = pad_packed_sequence(time,
                batch_first=True, padding_value=-999)
            context, _ = pad_packed_sequence(context,
                batch_first=True, padding_value=-999)
            labels, labels_len = pad_packed_sequence(labels,
                batch_first=True, padding_value=-999)
            outputs, _ = pad_packed_sequence(outputs,
                batch_first=True, padding_value=-999)

            # Compute loss for every element in batch
            for j in range(inputs.shape[0]):
                loss = criterion(
                    outputs[j, :, target_col],
                    labels[j, :, target_col],
                )
                # Add loss to hist list
                hist.append(loss.numpy().item())
                # If loss is one of the n worst, add to list
                smallest = min(bad_losses)
                if loss > smallest:
                    idx = bad_losses.index(smallest)
                    bad_losses[idx] = loss
                    bad_elements[idx] = (
                        inputs[j], time[j], context[j],
                        outputs[j], labels[j],
                        inputs_len[j], labels_len[j],
                    )

    fig, ax = plt.subplots(n // 2, 2, figsize=(16, n*2))
    date_fmt = '%d/%m %H:%M:%S'
    date_formatter = mdate.DateFormatter(date_fmt)
    for row in range(n // 2):
        for col in range(2):
            x, time, context, outputs, y, len_i, len_o = bad_elements[row*2+col]
            date = mdate.epoch2num(time)

            ax[row, col].plot_date(
                date[:len_i],
                x[:len_i, target_col],
                "bo-", label="Inputs"
            )
            ax[row, col].plot_date(
                date[len_i:len_i+len_o],
                y[:len_o, target_col],
                "ro-", label="Targets"
            )
            ax[row, col].plot_date(
                date[len_i:len_i+len_o],
                outputs[:len_o, target_col],
                "k--", label="Predictions"
            )
            context = np.round(context[0].numpy(), 3)
            ax[row, col].set_title(f"Loss: {bad_losses[row*2+col].numpy():.4f}\nContext: {context}")

            ax[row, col].xaxis.set_major_formatter(date_formatter)
            ax[row, col].set_ylim([0, 120])
            plt.setp(ax[row, col].get_xticklabels(), rotation=20, ha='right')

    fig.tight_layout()
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(hist, bins=20)
    plt.ylabel("Loss")
    plt.show()
