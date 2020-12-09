"""Describe the models architecture."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from batteryprobe.utils import pad_and_pack


class AutoRegressive(nn.Module):
    """
    AutoRegressive model.

    Attributes:
        in_size (int): Size in input of the model.
        out_size (init): Size in output of the model.
        lstm (nn.LSTM): The lstm model.
        dense (nn.Linear): The dense model.

    Args:
        params(dict): Parameters dict.
    """

    def __init__(self, params):
        self.params = params
        super(AutoRegressive, self).__init__()
        self.in_size = len(self.params["features"]) + len(self.params["context"])
        self.out_size = len(self.params["features"])
        self.lstm = nn.LSTM(self.in_size, 32, batch_first=True)
        self.dense = nn.Linear(32, self.out_size)

    def forward(self, x, context):
        """Forward step.

        Args:
            x (torch.Tensor): inputs
            context (torch.nn.utils.rnn.PackedSequence): The context
        """
        predictions = []
        x, warmup_state = self._warmup(x)
        predictions.append(x)

        context, lengths_context = pad_packed_sequence(context, batch_first=True, padding_value=-999)
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
