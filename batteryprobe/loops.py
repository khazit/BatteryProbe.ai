"""Define the training and evaluation loops."""


import torch
from torch import nn
from tqdm import tqdm


def evaluate(model, dataset, target_col):
    """"""
    loss = nn.L1Loss()
    running_loss = 0
    pbar = tqdm(dataset)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            out = model(inputs, labels[:, :, 0])
            running_loss += loss(
                out[:, :, target_col],
                labels[:, :, target_col]
            )  # loss per batch
            pbar.set_description(f"Loss {running_loss / (i+1):.5f}")
    return running_loss.numpy() / (i+1)
