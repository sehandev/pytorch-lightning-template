# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom


class CustomModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        num_layer,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=d_model,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')
        self.rnn.bias.data.zero_()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)

        rnn_output, _ = self.rnn(x)
        return rnn_output
