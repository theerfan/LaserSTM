import torch
import torch.nn as nn

from typing import Tuple

from LSTM.custom_lstms import LayerNormLSTMCell, LSTMLayer


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
        LSTM_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        has_fc_dropout: bool = True,
        bidirectional: int = 0,
        layernorm: int = 0,
        **kwargs,
    ):
        super().__init__()

        bidirectional = bool(bidirectional)
        layernorm = bool(layernorm)
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden_size

        if layernorm:
            self.lstm = LSTMLayer(
                LayerNormLSTMCell,
                input_size,
                lstm_hidden_size,
            )
        else:
            self.lstm = nn.LSTM(
                input_size,
                lstm_hidden_size,
                batch_first=True,
                dropout=LSTM_dropout,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

            # doing it like this so it won't be saved in the state dict
            if bidirectional:
                self.num_layers *= 2
                lstm_hidden_size *= 2

        if has_fc_dropout:
            self.linear = nn.Sequential(
                nn.Linear(lstm_hidden_size, linear_layer_size),
                # NOTE: This used to be relu, but we're trying different things
                nn.Sigmoid(),
                nn.Dropout(fc_dropout),
                nn.Linear(linear_layer_size, linear_layer_size),
                nn.Tanh(),
                nn.Dropout(fc_dropout),
                nn.Linear(linear_layer_size, input_size),
                nn.Sigmoid(),
            )
        else:
            print("No FC dropout!")
            self.linear = nn.Sequential(
                nn.Linear(lstm_hidden_size, linear_layer_size),
                # NOTE: This used to be relu, but we're trying different things
                nn.Sigmoid(),
                nn.Linear(linear_layer_size, linear_layer_size),
                nn.Tanh(),
                nn.Linear(linear_layer_size, input_size),
                nn.Sigmoid(),
            )

        print(
            f"hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}, bidirectional: {bidirectional}, LayerNorm: {layernorm}"
        )




    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None
    ):
        out_1, (hn, cn) = self.lstm(x)
        out = self.linear(out_1[:, -1, :])

        return out


class BlindTridentLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
        LSTM_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        shg_lower_factor: int = 4,
        bidirectional: int = 0,
        **kwargs,
    ):
        super(BlindTridentLSTM, self).__init__()

        bidirectional = bool(bidirectional)
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
            bidirectional=bidirectional,
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
            f"Blind Trident LSTM: hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}, bidirectional: {bidirectional}"
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
        bidirectional: int = 0,
        **kwargs,
    ):
        super(TridentLSTM, self).__init__()
        bidirectional = bool(bidirectional)
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
            bidirectional=bidirectional,
        )

        self.fc_sfg = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, 2 * self.sfg_size),
            nn.Sigmoid(),
        )

        self.fc_shg = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size // shg_lower_factor),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size // shg_lower_factor, 2 * self.shg_size),
            nn.Sigmoid(),
        )

        print(
            f"Trident LSTM: hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}, bidirectional: {bidirectional}"
        )

    def recombine_shg_sfg(
        self, shg1: torch.Tensor, shg2: torch.Tensor, sfg: torch.Tensor
    ) -> torch.Tensor:
        # Get the appropriate device
        device = shg1.device
        # Create an empty tensor with the appropriate size
        total_length = 1892 * 4 + 2 * 348
        fields = torch.zeros((shg1.shape[0], total_length), dtype=shg1.dtype).to(device)

        # Place segments from shg1, shg2, sfg into their original positions
        fields[:, 0:1892] = shg1[:, 0:1892]
        fields[:, 1892 : 1892 * 2] = shg2[:, 0:1892]
        fields[:, 1892 * 2 : 1892 * 2 + 348] = sfg[:, 0:348]
        fields[:, 1892 * 2 + 348 : 1892 * 3 + 348] = shg1[:, 1892:]
        fields[:, 1892 * 3 + 348 : 1892 * 4 + 348] = shg2[:, 1892:]
        fields[:, 1892 * 4 + 348 : 1892 * 4 + 2 * 348] = sfg[:, 348:]

        return fields

    def forward(self, x):
        # Forward through LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        # Forward through the fully connected layers
        shg1 = self.fc_shg(lstm_out)
        shg2 = self.fc_shg(lstm_out)
        sfg = self.fc_sfg(lstm_out)

        out = self.recombine_shg_sfg(shg1, shg2, sfg)

        return out


class BidentLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 1024,
        linear_layer_size: int = 4096,
        num_layers: int = 1,
        LSTM_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        shg_lower_factor: int = 2,
        bidirectional: int = 0,
        **kwargs,
    ):
        super(TridentLSTM, self).__init__()
        bidirectional = bool(bidirectional)
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
            bidirectional=bidirectional,
        )

        self.fc_sfg = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size, 2 * self.sfg_size),
            nn.Sigmoid(),
        )

        self.fc_shg = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_layer_size // shg_lower_factor),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(linear_layer_size // shg_lower_factor, 4 * self.shg_size),
            nn.Sigmoid(),
        )

        print(
            f"Trident LSTM: hidden_size: {lstm_hidden_size}, linear size: {linear_layer_size}, n_layers: {num_layers}, LSTM dropout: {LSTM_dropout}, fc dropout: {fc_dropout}, bidirectional: {bidirectional}"
        )

    def recombine_shg_sfg(
        self, shg: torch.Tensor, sfg: torch.Tensor
    ) -> torch.Tensor:
        # Get the appropriate device
        device = shg.device
        # Create an empty tensor with the appropriate size
        total_length = 1892 * 4 + 2 * 348
        fields = torch.zeros((shg.shape[0], total_length), dtype=shg.dtype).to(device)

        # shg1 is the first half of shg
        shg1 = shg[:, 0 : shg.shape[1] // 2]
        # shg2 is the second half of shg
        shg2 = shg[:, shg.shape[1] // 2 :]

        # Place segments from shg1, shg2, sfg into their original positions
        fields[:, 0:1892] = shg1[:, 0:1892]
        fields[:, 1892 : 1892 * 2] = shg2[:, 0:1892]
        fields[:, 1892 * 2 : 1892 * 2 + 348] = sfg[:, 0:348]
        fields[:, 1892 * 2 + 348 : 1892 * 3 + 348] = shg1[:, 1892:]
        fields[:, 1892 * 3 + 348 : 1892 * 4 + 348] = shg2[:, 1892:]
        fields[:, 1892 * 4 + 348 : 1892 * 4 + 2 * 348] = sfg[:, 348:]

        return fields

    def forward(self, x):
        # Forward through LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        # Forward through the fully connected layers
        sfg = self.fc_sfg(lstm_out)
        shg = self.fc_shg(lstm_out)

        out = self.recombine_shg_sfg(shg, sfg)

        return out
