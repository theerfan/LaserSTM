import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define the Fourier layer
class FourierLayer(nn.Module):
    def __init__(self, channels, modes):
        super(FourierLayer, self).__init__()
        self.channels = channels
        self.modes = modes
        self.scale = 1 / (channels * modes)
        self.fc_r = nn.Linear(modes, channels)

    def forward(self, x):
        # Apply the Fourier Transform
        x_ft = torch.fft.rfft(x)

        # Filter the Fourier modes
        x_ft = x_ft[:, : self.modes]

        # Apply the learned linear transform R
        x_ft = self.fc_r(x_ft)

        # Apply the Inverse Fourier Transform
        x = torch.fft.irfft(x_ft, n=self.channels)

        # Apply the non-linear activation function
        return F.gelu(x)


# Combine everything into the Neural Operator
class NeuralOperator(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lifted_channels: int,
        output_channels: int,
        num_layers: int,
        modes: int,
    ):
        super(NeuralOperator, self).__init__()
        self.upscale_nn = nn.Linear(input_channels, lifted_channels)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer(lifted_channels, modes) for _ in range(num_layers)]
        )
        self.downscale_nn = nn.Linear(lifted_channels, output_channels)

    def forward(self, x):
        x = self.upscale_nn(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.downscale_nn(x)
        return x


# Example instantiation of the Neural Operator
input_channels = 1  # Assuming input a(x) has one channel
lifted_channels = 64  # Example lifted dimension
output_channels = 1  # Assuming output u(x) has one channel
num_layers = 4  # Number of Fourier layers as shown in the diagram
modes = 16  # Number of Fourier modes to keep

neural_operator = NeuralOperator(
    input_channels, lifted_channels, output_channels, num_layers, modes
)
