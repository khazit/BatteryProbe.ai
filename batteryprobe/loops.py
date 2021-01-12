"""Define the training and evaluation loops."""

import logging
from os.path import join

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from batteryprobe.utils import masked_l1, masked_nllloss

# pylint: disable=R0914, R0915


def train(model, datasets, params):
    """Train a model on a given dataset."""
    # Prepare training
    patience = 0
    best_loss = 9999
    idx_dataloader = 0
    if params["use_std"]:
        criterion = masked_nllloss
    else:
        criterion = masked_l1
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    train_dls, val_dl = datasets
    if not params["debug"]:
        writer = SummaryWriter(params["log_dir"])

    for epoch in range(params["n_epochs"]):
        ########### Training ##########
        running_loss = 0
        pbar = tqdm(train_dls[idx_dataloader])
        for i, ((inputs, time, context), labels) in enumerate(pbar):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs.float(), time.float(), context.float())

            # Backward pass
            pad_labels, _ = pad_packed_sequence(labels, batch_first=True, padding_value=-999)
            pad_outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=-999)
            loss = criterion(pad_outputs, pad_labels.float())
            loss.backward()
            optimizer.step()

            # Update progress bar
            running_loss += loss.item()
            pbar.set_description(f"Epoch #{epoch+1} - Loss = {running_loss / (i+1):.5f}")

        ########### Validation ##########
        val_running_loss = 0
        pbar = tqdm(val_dl)
        with torch.no_grad():
            for j, ((inputs, time, context), labels) in enumerate(pbar):
                # Evaluate
                outputs = model(inputs.float(), time.float(), context.float())

                # Compute loss
                pad_labels, _ = pad_packed_sequence(labels, batch_first=True, padding_value=-999)
                pad_outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=-999)
                loss = criterion(pad_outputs, pad_labels.float())

                # Update progress bar
                val_running_loss += loss  # MSE per batch
                pbar.set_description(f"Validation loss = {val_running_loss / (j+1):.5f}")

        ########### Callbacks at the end of each epoch ##########
        # Tensorboard
        if params["debug"]:
            continue
        writer.add_scalars("loss", {
            "train": running_loss / (i+1),
            "val": val_running_loss / (j+1),
        }, epoch+1)
        # Keep best model
        if (val_running_loss / (j+1)) < best_loss:
            logging.info(
                f"Validation loss improved from {best_loss} to {(val_running_loss / (j+1))}"
            )
            best_loss = (val_running_loss / (j+1))
            torch.save(model.state_dict(), join(params["log_dir"], "model.pt"))
            patience = 0
        else:
            logging.info(
                f"Val loss did not improve. Patience: {patience} (max: {params['max_patience']})")
            patience += 1
        # Early stopping
        if patience >= params["max_patience"]:
            logging.info("Triggered early stopping.")
            idx_dataloader += 1
            if idx_dataloader == len(train_dls):
                break
            patience = 0
            logging.info(
                f"Using next dataloader w/ bounds {params['label_bounds'][idx_dataloader]}")

    if not params["debug"]:
        writer.add_hparams(
            {k:v.__str__() if isinstance(v, list) else v for k, v in params.items()},
            {"val_loss": best_loss}
        )
    logging.info("Training done.")


def evaluate(model, dataset, target_col, use_std=False):
    """Evaluate a model on a dataset given a target feature.

    Args:
        model (nn.Module): Model to evaluate.
        dataset (torch.utils.data.Dataset): Evaluation dataset.
        target_col (int): Target feature.

    Returns:
        (int) L1 score.
    """
    if use_std:
        loss = masked_nllloss
    else:
        loss = masked_l1
    running_loss = 0
    pbar = tqdm(dataset)
    with torch.no_grad():
        for i, ((inputs, time, context), labels) in enumerate(pbar):
            outputs = model(inputs.float(), time.float(), context.float())

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
