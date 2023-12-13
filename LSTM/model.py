import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
        LSTM_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            batch_first=True,
            dropout=LSTM_dropout,
            num_layers=num_layers,
        )
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, input_size),
            nn.Sigmoid(),
        )

        print(
            f"hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}"
        )

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None
    ):
        # hidden state
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(
                x.device
            )
        # cell state
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(
                x.device
            )

        out_1, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.linear(out_1[:, -1, :])

        return out


class TridentLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
        LSTM_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden_size,
            batch_first=True,
            dropout=LSTM_dropout,
            num_layers=num_layers,
        )
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, input_size),
            nn.Sigmoid(),
        )

        print(
            f"hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}"
        )