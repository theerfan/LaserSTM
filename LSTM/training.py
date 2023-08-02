from train_predict_utils import (
    CustomSequence,
    train,
    predict,
    weighted_MSE,
    pearson_corr,
    single_pass,
)
import torch
import torch.nn as nn
from functools import partial


class LSTMModel_1(nn.Module):
    # basic one with two linear layers and final output with sigmoid
    def __init__(
        self, input_size, lstm_hidden_size=1024, linear_layer_size=4096, num_layers=1
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

    def forward(self, x):
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


def main_train():
    # The data that is currently here is the V2 data (reIm)
    data_dir = "/u/scratch/t/theerfan/JackData"
    train_dataset = CustomSequence(
        data_dir, range(90), file_batch_size=1, model_batch_size=512
    )
    val_dataset = CustomSequence(
        data_dir, [90], file_batch_size=1, model_batch_size=512
    )

    # (SHG1, SHG2) + SFG * 2
    # (1892 * 2 + 348) * 2
    model = LSTMModel_1(input_size=8264)

    train(
        model,
        train_dataset,
        num_epochs=10,
        val_dataset=val_dataset,
        use_gpu=True,
        data_parallel=True,
        out_dir=".",
        model_name="model",
        verbose=1,
        save_checkpoints=True,
        custom_loss=weighted_MSE,
    )

    test_dataset = CustomSequence(
        data_dir, range(91, 99), file_batch_size=1, model_batch_size=512, test_mode=True
    )

    predict(
        model,
        # model_param_path="model_epoch_2.pth",
        test_dataset=test_dataset,
        use_gpu=True,
        data_parallel=False,
        output_dir=".",
        output_name="all_preds.npy",
        verbose=1,
    )


def dev_test_losses():
    data_dir = "processed_dataset"
    test_dataset = CustomSequence(
        data_dir, [1], file_batch_size=1, model_batch_size=512
    )

    # (SHG1, SHG2) + SFG * 2
    # (1892 * 2 + 348) * 2
    model = LSTMModel_1(input_size=8264)
    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    normalized_mse_loss, last_mse_loss = single_pass(
        model, test_dataset, "cpu", optimizer, mse, verbose=False
    )

    print(normalized_mse_loss, last_mse_loss)

    normalized_equal_mse_loss, last_equal_mse_loss = single_pass(
        model, test_dataset, "cpu", optimizer, weighted_MSE, verbose=False
    )

    print(normalized_equal_mse_loss, last_equal_mse_loss)


if __name__ == "__main__":
    main_train()