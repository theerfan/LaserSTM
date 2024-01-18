import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Fourier layer
class FourierLayer(nn.Module):
    def __init__(self, n_channels, n_modes):
        super(FourierLayer, self).__init__()
        self.n_channels = n_channels
        self.modes = n_modes
        self.scale = 1 / (n_channels * n_modes)
        self.fc_r = nn.Linear(n_modes, n_channels)

    def forward(self, x):
        # Apply the Fourier Transform
        x_ft = torch.fft.rfft(x)

        # Filter the Fourier modes
        x_ft = x_ft[:, : self.modes]

        # Apply the learned linear transform R
        x_ft = self.fc_r(x_ft)

        # Apply the Inverse Fourier Transform
        x = torch.fft.irfft(x_ft, n=self.n_channels)

        # Apply the non-linear activation function
        return F.gelu(x)


# Combine everything into the Neural Operator
class NeuralOperator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lifted_dim: int,
        output_dim: int,
        num_layers: int,
        n_modes: int,
        is_slice: bool = False,
    ):
        super(NeuralOperator, self).__init__()
        self.upscale_nn = nn.Linear(input_dim, lifted_dim)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer(lifted_dim, n_modes) for _ in range(num_layers)]
        )
        self.downscale_nn = nn.Linear(lifted_dim, output_dim)

        self.is_slice = is_slice

    def forward(self, x):
        if not self.is_slice:
            x = x[:, -1, :]
        else:
            pass
        
        x = self.upscale_nn(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.downscale_nn(x)
        return x


# # Example instantiation of the Neural Operator
# input_channels = 1  # Assuming input a(x) has one channel
# lifted_channels = 64  # Example lifted dimension
# output_channels = 1  # Assuming output u(x) has one channel
# num_layers = 4  # Number of Fourier layers as shown in the diagram
# modes = 16  # Number of Fourier modes to keep

# neural_operator = NeuralOperator(
#     input_channels, lifted_channels, output_channels, num_layers, modes
# )
