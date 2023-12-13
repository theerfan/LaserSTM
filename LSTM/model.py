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
        shg_lower_factor: int = 4,
        **kwargs,
    ):
        super(TridentLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.shg_size = 1892
        self.sfg_size = 348

        self.lstm = nn.LSTM(
            input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            dropout=LSTM_dropout,
            num_layers=num_layers,
        )

        self.scale_up = nn.Linear(lstm_hidden_size, input_size)

        self.fc_sfg = nn.Sequential(
            nn.Linear(2 * self.sfg_size, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, 2 * self.sfg_size),
            nn.Sigmoid(),
        )

        self.fc_shg = nn.Sequential(
            nn.Linear(2 * self.shg_size, linear_layer_size // shg_lower_factor),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size // shg_lower_factor, 2 * self.shg_size),
            nn.Sigmoid(),
        )

        print(
            f"Trident LSTM: hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}"
        )

    # TODO: Re-write this and make it clear
    def separate_shg_sfg(self, fields: torch.Tensor):
        # [shg1_int, shg2_int, sfg_int, shg1_phase, shg2_phase, sfg_phase]
        shg1 = torch.cat(
            (fields[:, 0:1892], fields[:, 1892 * 2 + 348 : 1892 * 3 + 348]), dim=1
        )
        shg2 = torch.cat(
            (fields[:, 1892 : 1892 * 2], fields[:, 1892 * 3 + 348 : 1892 * 4 + 348]),
            dim=1,
        )
        sfg = torch.cat(
            (
                fields[:, 1892 * 2 : 1892 * 2 + 348],
                fields[:, 1892 * 4 + 348 : 1892 * 4 + 2 * 348],
            ),
            dim=1,
        )

        return shg1, shg2, sfg

    def forward(self, x):
        # Forward through LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        lstm_out = self.scale_up(lstm_out)

        shg1, shg2, sfg = self.separate_shg_sfg(lstm_out)

        # Forward through the fully connected layers
        out1 = self.fc_shg(shg1)
        out2 = self.fc_shg(shg2)
        out3 = self.fc_sfg(sfg)

        return torch.cat((out1, out2, out3), dim=1)
