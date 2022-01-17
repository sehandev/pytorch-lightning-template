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

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, model_option['d_model']),
            nn.ReLU(),
            nn.Linear(model_option['d_model'], 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, model_option['d_model']),
            nn.ReLU(),
            nn.Linear(model_option['d_model'], 28 * 28),
        )

    def forward(self, x):
        # x: (batch_size, 1, 28, 28)

        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x, x_hat
