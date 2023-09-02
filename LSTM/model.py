import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    # basic one with two linear layers and final output with sigmoid
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            batch_first=True,
            dropout=0,
            num_layers=num_layers,
        )
        self.fc1 = nn.Linear(lstm_hidden_size, linear_layer_size)
        self.fc2 = nn.Linear(linear_layer_size, linear_layer_size)
        self.fc3 = nn.Linear(linear_layer_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # hidden state
        h_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(
            x.device
        )  # Modified line
        # cell state
        c_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(
            x.device
        )  # Modified line

        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)

        return out
