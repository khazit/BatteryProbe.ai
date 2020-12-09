"""Define the training and evaluation loops."""


import torch
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from batteryprobe.utils import masked_l1


def evaluate(model, dataset, target_col):
    """Evaluate a model on a dataset given a target feature.

    Args:
        model (nn.Module): Model to evaluate.
        dataset (torch.utils.data.Dataset): Evaluation dataset.
        target_col (int): Target feature.

    Returns:
        (int) L1 score.
    """
    loss = masked_l1
    running_loss = 0
    pbar = tqdm(dataset)
    with torch.no_grad():
        for i, ((inputs, context), labels) in enumerate(pbar):
            outputs = model(inputs.float(), context.float())

            # Pad packed labels and outputs
            pad_labels, _ = pad_packed_sequence(labels,
                batch_first=True, padding_value=-999)
            pad_outputs, _ = pad_packed_sequence(outputs,
                    batch_first=True, padding_value=-999)

            # Compute loss for this batch
            running_loss += loss(
                pad_outputs[:, :, target_col],
                pad_labels[:, :, target_col]
            )
            pbar.set_description(f"Loss {running_loss / (i+1):.5f}")
    return running_loss.numpy() / (i+1)
