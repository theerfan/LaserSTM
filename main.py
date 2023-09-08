import logging

import argparse

import torch.nn as nn

from LSTM.model import LSTMModel
from LSTM.training import (
    predict,
    test_train_lstm,
    tune_train_lstm,
)
from LSTM.utils import (
    CustomSequence,
    pearson_corr,
    weighted_MSE,
    pseudo_energy_loss,
    area_under_curve_loss,
    wrapped_MSE,
    wrapped_BCE,
)

from Transformer.model import TransformerModel

from GAN.training import gan_train
from typing import Callable

from functools import partial

logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
)


def main_lstm(
    args: dict,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    custom_loss: Callable,
):
    model = LSTMModel(input_size=8264)
    if args.do_prediction == 1:
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        predict(
            model,
            model_param_path=args.model_param_path,
            test_dataset=test_dataset,
            use_gpu=True,
            data_parallel=False,
            output_dir=args.output_dir,
            output_name="all_preds.npy",
            verbose=1,
        )
    else:
        # This assumes that `tune_train` and `train_model` have the same signature
        # (as in required arguments)
        if args.tune_train == 1:
            function = tune_train_lstm
            print_str = f"Tune train mode for model {args.model}"
        else:
            function = test_train_lstm
            print_str = f"Training mode for model {args.model}"

        print(print_str)
        logging.info(print_str)

        function(
            model,
            args.num_epochs,
            custom_loss,
            args.epoch_save_interval,
            args.output_dir,
            train_dataset,
            val_dataset,
            test_dataset,
            args.verbose,
        )


def main_gan(
    args: dict,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    custom_loss: Callable,
):
    gan_train(
        input_dim=8264,
        hidden_dim=128,
        output_dim=8264,
        num_epochs=args.num_epochs,
        train_set=train_dataset,
        lr=0.001,
    )


def test_energy_stuff():
    val_dataset = CustomSequence(".", [0], file_batch_size=1, model_batch_size=512)
    gen = val_dataset[0]
    X, y = next(gen)
    pseudo_energy_loss(y, y)
    pass


if __name__ == "__main__":
    # test_energy_stuff()
    parser = argparse.ArgumentParser(description="Train and test the model.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model to use for training."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--custom_loss", type=str, default="MSE", help="Custom loss function name."
    )

    # Add arguments for the shg and sfg weight losses
    parser.add_argument(
        "--shg_weight", type=float, default=None, help="Weight for the SHG loss."
    )

    parser.add_argument(
        "--sfg_weight", type=float, default=None, help="Weight for the SFG loss."
    )

    parser.add_argument(
        "--epoch_save_interval", type=int, default=1, help="Epoch save interval."
    )

    parser.add_argument(
        "--tune_train",
        type=int,
        default=0,
        help="Whether to do hyperparameter tuning or not.",
    )

    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")

    parser.add_argument(
        "--do_prediction",
        type=int,
        default=0,
        help="Whether to do prediction or not.",
    )

    parser.add_argument(
        "--model_param_path",
        type=str,
        default="model.pth",
        help="Path to the model parameters.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Whether to print the progress or not.",
    )

    loss_dict = {
        "weighted_MSE": weighted_MSE,
        "pearson_corr": pearson_corr,
        "MSE": wrapped_MSE,
        "BCE": wrapped_BCE,
        "pseudo_energy": pseudo_energy_loss,
        "area_under_curve": area_under_curve_loss,
    }

    args = parser.parse_args()

    def custom_loss(y_real, y_pred, shg_weight=None, sfg_weight=None):
        return loss_dict[args.custom_loss](
            y_real, y_pred, shg_weight=shg_weight, sfg_weight=sfg_weight
        )

    if args.shg_weight is not None and args.sfg_weight is not None:
        # Do a partial function application
        custom_loss = partial(
            custom_loss, shg_weight=args.shg_weight, sfg_weight=args.sfg_weight
        )
    else:
        pass

    # The data that is currently here is the V2 data (reIm)
    train_dataset = CustomSequence(
        args.data_dir, range(0, 90), file_batch_size=1, model_batch_size=512
    )
    val_dataset = CustomSequence(
        args.data_dir, [90], file_batch_size=1, model_batch_size=512
    )

    test_dataset = CustomSequence(
        args.data_dir,
        range(91, 99),
        file_batch_size=1,
        model_batch_size=512,
        test_mode=True,
    )

    if args.model == "LSTM":
        main_lstm(args, train_dataset, val_dataset, test_dataset, custom_loss)
    elif args.model == "Transformer":
        model = TransformerModel(
            n_features=8264,
            n_predict=8264,
            n_head=2,
            n_hidden=128,
            n_enc_layers=2,
            n_dec_layers=2,
            dropout=0.1,
        )
    elif args.model == "GAN":
        pass
    else:
        raise ValueError("Model not supported.")
