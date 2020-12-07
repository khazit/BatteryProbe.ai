"""Descibe the models architecture."""


import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from batteryprobe.utils import pad_and_pack


class AutoRegressive(nn.Module):
    """AutoRegressive model for time series forecasting."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(13, 32, batch_first=True)
        self.dense = nn.Linear(32, 13)

    # pylint: disable=C0103
    def forward(self, x, out_steps=None):
        """Forward step.

        Args:
            x (torch.Tensor): inputs
            out_steps (torch.Tensor): length of the desired output from each element of the batch
        """
        predictions = []
        x, warmup_state = self.warmup(x)
        predictions.append(x)

        # Loop over every element of a batch
        batch = []
        for i, element in enumerate(x):
            # State corresponding to a single element in a batch
            state = (
                warmup_state[0][:, i, :][None, :],
                warmup_state[1][:, i, :][None, :],
            )

            # Add first prediction from warmup
            timestamps = [element[None, None, :]]

            # Predict values
            for _ in range(out_steps[i]-1):
                x, state = self.lstm(
                    element[None, None, :], state
                )
                x = self.dense(x)
                timestamps.append(x)
            batch.append(torch.cat(timestamps, 1)[0])

        # Pack and pad sequence
        return pad_and_pack(batch)

    def _warmup(self, x):
        x, state = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True, padding_value=-999)
        x = torch.stack([x[i, length-1] for i, length in enumerate(lengths)])
        x = self.dense(x)
        return x, state
