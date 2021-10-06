# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom


class CustomModel(nn.Module):
    def __init__(
        self,
        model_option,
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=model_option['input_size'],
            hidden_size=model_option['d_model'],
            num_layers=model_option['num_layer'],
            batch_first=True,
            dropout=model_option['dropout'],
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')
        self.rnn.bias.data.zero_()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)

        rnn_output, _ = self.rnn(x)
        return rnn_output
