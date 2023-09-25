import logging

import argparse

import torch.nn as nn

from LSTM.utils import (
    CustomSequence,
    pearson_corr,
    weighted_MSE,
    pseudo_energy_loss,
    area_under_curve_loss,
    wrapped_MSE,
    wrapped_BCE,
)

from LSTM.main_fn import main_lstm
from GAN.main_fn import main_gan
from Transformer.main_fn import main_transformer
from FNO.main_fn import main_NFO

from Transformer.model import TransformerModel

from functools import partial

logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
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
        main_transformer(args, train_dataset, val_dataset, test_dataset, custom_loss)
    elif args.model == "GAN":
        main_gan(args, train_dataset, val_dataset, test_dataset, custom_loss)
    elif args.model == "NFO":
        main_NFO(args, train_dataset, val_dataset, test_dataset, custom_loss)
    else:
        raise ValueError("Model not supported.")
