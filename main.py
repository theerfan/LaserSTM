import argparse
from typing import Callable

import ray
import torch
import torch.nn as nn
from ray import tune

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


# Tune wrapper for hyperparameter tuning (main_train wrapped in this)
def ray_train_model(config):
    model = config["model"]
    num_epochs = config["num_epochs"]
    shg1_weight = config["shg1_weight"]
    shg2_weight = config["shg2_weight"]
    sfg_weight = config["sfg_weight"]
    custom_loss = config["custom_loss"]

    def tuned_custom_loss(y_pred, y_real):
        return custom_loss(y_pred, y_real, shg1_weight, shg2_weight, sfg_weight)

    epoch_save_interval = config["epoch_save_interval"]
    output_dir = config["output_dir"]
    train_dataset = config["train_dataset"]
    val_dataset = config["val_dataset"]
    test_dataset = config["test_dataset"]
    verbose = config["verbose"]

    trained_model, train_losses, val_losses, all_test_preds = train_model(
        model,
        num_epochs,
        tuned_custom_loss,
        epoch_save_interval,
        output_dir,
        train_dataset,
        val_dataset,
        test_dataset,
        verbose,
    )

    val_loss = torch.mean(torch.tensor(val_losses).flatten())

    # Report the test loss back to Ray Tune
    tune.report(loss=val_loss.item())


def tune_train(
    model: torch.nn.Module,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    verbose: int = 1,
):
    # Initialize Ray
    ray.init()

    # Specify the hyperparameter search space
    config = {
        "shg1_weight": tune.uniform(0, 1),
        "shg2_weight": tune.uniform(0, 1),
        "sfg_weight": tune.uniform(0, 1),
        "model": model,
        "num_epochs": num_epochs,
        "custom_loss": custom_loss,
        "epoch_save_interval": epoch_save_interval,
        "output_dir": output_dir,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "verbose": verbose,
    }

    # Ensure the sum of hyperparameters equals 1 using a constraint
    constraint = "shg1_weight + shg2_weight + sfg_weight <= 1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Run the experiments
    analysis = tune.run(
        ray_train_model,
        config=config,
        resources_per_trial={device: device_count},
        num_samples=100,  # Number of hyperparameter combinations to try
        stop={"loss": 0.01},  # Stop trials if the loss goes below this threshold
        constraint=constraint,
    )

    # Print the best hyperparameters
    print("Best hyperparameters found were: ", analysis.best_config)


# (SHG1, SHG2) + SFG * 2
# (1892 * 2 + 348) * 2 = 8264


def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    verbose: int = 1,
):
    trained_model, train_losses, val_losses = train(
        model,
        train_dataset,
        num_epochs=num_epochs,
        val_dataset=val_dataset,
        use_gpu=True,
        data_parallel=True,
        out_dir=output_dir,
        model_name="model",
        verbose=verbose,
        save_checkpoints=True,
        custom_loss=custom_loss,
        epoch_save_interval=epoch_save_interval,
    )

    all_test_preds = predict(
        model,
        # model_param_path="model_epoch_2.pth",
        test_dataset=test_dataset,
        use_gpu=True,
        data_parallel=False,
        output_dir=output_dir,
        output_name="all_preds.npy",
        verbose=verbose,
    )

    return trained_model, train_losses, val_losses, all_test_preds


if __name__ == "__main__":
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
        "verbose",
        type=int,
        default=1,
        help="Whether to print the progress or not.",
    )

    loss_dict = {
        "weighted_MSE": weighted_MSE,
        "pearson_corr": pearson_corr,
        "MSE": nn.MSELoss,
    }

    args = parser.parse_args()

    if args.model == "LSTM":
        model = LSTMModel_1(input_size=8264)
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
    else:
        raise ValueError("Model not supported.")

    custom_loss = loss_dict[args.custom_loss]

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

    if args.do_prediction == 1:
        print(f"Prediction only mode for model {args.model}")
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
            function = tune_train
            print_str = f"Tune train mode for model {args.model}"
        else:
            function = train_model
            print_str = f"Training mode for model {args.model}"

        print(print_str)

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
