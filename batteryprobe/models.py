"""Describe the models architecture."""

import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from batteryprobe.utils import pad_and_pack


# pylint: disable=C0103
class Baseline(nn.Module):
    """Baseline model.

    Use a linear slope of the inputs to predict the target column.
    """

    def __init__(self, target_col, use_std = False):
        super().__init__()
        self.use_std = use_std
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
            if self.use_std:
                res[i, :len_context[i], 2 * self.target_col + 1] = 0

        # Pack sequences
        res = pack_padded_sequence(res,
            len_context, batch_first=True, enforce_sorted=False)
        return res


class Time2Vec(nn.Module):
    """Time2Vec module.

    Implements: https://arxiv.org/abs/1907.05321
    """
    def __init__(self, k, periodic_activation_function=torch.sin):
        super().__init__()
        self.k = k
        self.periodic_activation_fn = periodic_activation_function
        omega = torch.Tensor(k)
        self.omega = nn.Parameter(omega)
        phi = torch.Tensor(k)
        self.phi = nn.Parameter(phi)

        # Initialization phase and frequency
        bound = 1
        init.uniform_(self.omega, -bound, bound)
        init.uniform_(self.phi, -bound, bound)

    def forward(self, t):
        """Forward pass.

        Args:
            t (torch.Tensor): (batch_size, sequence_len, 1)

        Returns:
            (torch.Tensor): (batch_size, sequence_len, k)
        """
        return self.periodic_activation_fn(t @ self.omega[None, :] + self.phi)


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
        # pylint: disable=C0301
        if params["use_std"]:
            k = 2
        else:
            k = 1
        self.in_size = k * len(self.params["features"]) + len(self.params["context"]) + params["t2v_k"]
        self.out_size = k * len(self.params["features"])  # Outputs = Means + Stds
        self.t2v = Time2Vec(k=params["t2v_k"])
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
        # Embed time and add it to inputs
        x, time_embedding = self.prepare_input(x, time, self.params["use_std"])

        # Pass into warmup
        x, warmup_state = self._warmup(x)

        # Unpack context for outputs prediction
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
                in_tensor = torch.cat(
                    [element, context[i, t, :], time_embedding[i, t, :]],
                    axis=-1
                )[None, None, :]
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

    def prepare_input(self, x, time, use_std):
        """ Prepare input by adding time embedding and std values to input feature"""
        time_embedding = self.embed_time(x, time)
        x, x_len = pad_packed_sequence(x,
            batch_first=True, padding_value=-999
        )
        inputs_time_embedding = pad_sequence([
            time_embedding[i, :length] for i, length in enumerate(x_len)],
            batch_first=True, padding_value=-999
        )
        # Add std and time embeddings to inputs
        if use_std:
            x = torch.cat(
                (
                    x,  # inputs
                    torch.ones(x.shape[0], x.shape[1], len(self.params["features"])),  # std
                    inputs_time_embedding  # time embeddings
                ), 2
            )
        else:
            x = torch.cat(
                (
                    x,  # inputs
                    inputs_time_embedding  # time embeddings
                ), 2
            )
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        return x, time_embedding

    def embed_time(self, x, time):
        """Use Time2Vec layer to compute a time embedding and add it to input features."""
        time, _ = pad_packed_sequence(time,
            batch_first=True, padding_value=-999
        )
        time_embedding = self.t2v(torch.unsqueeze(time, -1))

        return time_embedding
