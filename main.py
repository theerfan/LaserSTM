import argparse
from typing import Callable

import torch

from LSTM.model import LSTMModel_1
from train_utils.train_predict_utils import (
    CustomSequence,
    pearson_corr,
    predict,
    single_pass,
    train,
    weighted_MSE,
)
from Transformer.model import TransformerModel


def dev_test_losses():
    data_dir = "processed_dataset"
    test_dataset = CustomSequence(
        data_dir, [1], file_batch_size=1, model_batch_size=512
    )

    # (SHG1, SHG2) + SFG * 2
    # (1892 * 2 + 348) * 2
    model = LSTMModel_1(input_size=8264)
    # mse = nn.MSELoss()
    mse = pearson_corr

    optimizer = torch.optim.Adam(model.parameters())

    normalized_mse_loss, last_mse_loss = single_pass(
        model, test_dataset, "cpu", optimizer, mse, verbose=False
    )

    print(normalized_mse_loss, last_mse_loss)

    normalized_equal_mse_loss, last_equal_mse_loss = single_pass(
        model, test_dataset, "cpu", optimizer, weighted_MSE, verbose=False
    )

    print(normalized_equal_mse_loss, last_equal_mse_loss)


def main_train(
    model: torch.nn.Module,
    data_dir: str,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
):
    # The data that is currently here is the V2 data (reIm)
    train_dataset = CustomSequence(
        data_dir, [0], file_batch_size=1, model_batch_size=512
    )
    val_dataset = CustomSequence(data_dir, [0], file_batch_size=1, model_batch_size=512)

    # (SHG1, SHG2) + SFG * 2
    # (1892 * 2 + 348) * 2 = 8264
    model = TransformerModel(
        n_features=8264,
        n_predict=8264,
        n_head=8,
        n_hidden=2048,
        n_enc_layers=6,
        n_dec_layers=6,
        dropout=0.1,
    )

    train(
        model,
        train_dataset,
        num_epochs=num_epochs,
        val_dataset=val_dataset,
        use_gpu=True,
        data_parallel=True,
        out_dir=output_dir,
        model_name="model",
        verbose=1,
        save_checkpoints=True,
        custom_loss=custom_loss,
        epoch_save_interval=epoch_save_interval,
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
        output_dir=output_dir,
        output_name="all_preds.npy",
        verbose=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test the model.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model to use for training."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory."
    )
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs for training."
    )
    parser.add_argument(
        "--custom_loss", type=str, required=True, help="Custom loss function name."
    )

    parser.add_argument(
        "--epoch_save_interval", type=int, default=1, help="Epoch save interval."
    )

    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")

    loss_dict = {
        "weighted_MSE": weighted_MSE,
        "pearson_corr": pearson_corr,
    }

    args = parser.parse_args()

    if args.model == "LSTM":
        model = LSTMModel_1(input_size=8264)
    elif args.model == "Transformer":
        model = TransformerModel(
            n_features=8264,
            n_predict=8264,
            n_head=8,
            n_hidden=2048,
            n_enc_layers=6,
            n_dec_layers=6,
            dropout=0.1,
        )
    else:
        raise ValueError("Model not supported.")

    custom_loss = loss_dict[args.custom_loss]

    main_train(
        model,
        args.data_dir,
        args.num_epochs,
        custom_loss,
        args.epoch_save_interval,
        args.output_dir,
    )
