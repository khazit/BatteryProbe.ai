"""Describe the models architecture."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from batteryprobe.utils import pad_and_pack


# pylint: disable=C0103
class AutoRegressive(nn.Module):
    """AutoRegressive model.

    Attributes:
        in_size (int): Size in input of the model.
        out_size (init): Size in output of the model.
        lstm (nn.LSTM): The lstm layer.
        dense (nn.Linear): The dense layer.

    Args:
        params(dict): Parameters dict.
    """

    def __init__(self, params):
        self.params = params
        super().__init__()
        self.in_size = len(self.params["features"]) + len(self.params["context"])
        self.out_size = len(self.params["features"])
        self.lstm = nn.LSTM(self.in_size, 64, num_layers=params["lstm_layers"], batch_first=True)
        self.dense = nn.Linear(64, self.out_size)

    # pylint: disable=W0613
    def forward(self, x, time, context):
        """Forward pass.

        Args:
            x (torch.Tensor): inputs
            time (torch.nn.utils.rnn.PackedSequence): Timestamps
            context (torch.nn.utils.rnn.PackedSequence): Context
        """
        x, warmup_state = self._warmup(x)

        context, lengths_context = pad_packed_sequence(
            context,
            batch_first=True,
            padding_value=-999
        )
        # Loop over every element of a batch
        batch = []
        for i, element in enumerate(x):
            # State corresponding to a single element in a batch
            state = (  # (num_layers, batch, hidden_size)
                torch.unsqueeze(warmup_state[0][:, i, :], 1),
                torch.unsqueeze(warmup_state[1][:, i, :], 1),
            )

            # Add first prediction from warmup
            timestamps = [element[None, None, :]]

            # Predict values
            # pylint: disable=C0103
            for t in range(lengths_context[i] - 1):
                # The order here is important. Inputs features should be first
                in_tensor = torch.cat([element, context[i, t, :]], axis=-1)[None, None, :]
                x, state = self.lstm(in_tensor, state)
                x = self.dense(x)
                timestamps.append(x)
            batch.append(torch.cat(timestamps, 1)[0])

        # Pack and pad sequence
        return pad_and_pack(batch)

    def _warmup(self, x):
        x, state = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True, padding_value=-999)
        x = torch.stack([x[i, length - 1] for i, length in enumerate(lengths)])
        x = self.dense(x)
        return x, state


class Baseline(nn.Module):
    """Baseline model.

    Use a linear slope of the inputs to predict the target column.
    """
    def __init__(self, target_col):
        super().__init__()
        self.target_col = target_col

    def forward(self, x, time, context):
        """Forward pass."""
        x, len_x = pad_packed_sequence(x,
            batch_first=True, padding_value=-999)
        context, len_context = pad_packed_sequence(context,
            batch_first=True, padding_value=-999)
        time, len_time = pad_packed_sequence(
            time, batch_first=True, padding_value=-999)

        slopes = torch.zeros(x.shape[0])
        beta = torch.zeros(x.shape[0])
        res = -999 * torch.ones((x.shape[0], context.shape[1], x.shape[2]))

        # Loop over every element of a batch
        for i in range(x.shape[0]):
            # Absolute to relative
            time[i, :len_time[i]] = time[i, :len_time[i]] - time[i, 0]
            # Compute slope
            slopes[i] = x[i, len_x[i]-1, self.target_col] - x[i, 0, self.target_col]
            slopes[i] = slopes[i] / time[i, len_x[i]-1]
            # Compute beta
            beta[i] =  x[i, 0, self.target_col]
            # Compute res
            res[i, :len_context[i], self.target_col] = slopes[i] \
                * time[i, len_x[i]:len_time[i]] + beta[i]

        # Pack sequences
        res = pack_padded_sequence(res,
            len_context, batch_first=True, enforce_sorted=False)
        return res
