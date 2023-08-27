import argparse
from typing import Callable

import torch
import torch.nn as nn

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

import ray
from ray import tune

# Define the training function for Ray Tune
def train_model(config):
    shg1_weight = config["shg1_weight"]
    shg2_weight = config["shg2_weight"]
    sfg_weight = config["sfg_weight"]

    model = LSTMModel_1(input_size=8264)
    optimizer = torch.optim.Adam(model.parameters())

    # Assuming you have a train function like in the provided code
    train_loss = train(
        model,
        "/u/scratch/t/theerfan/JackData/train",
        10,
        optimizer,
        lambda y_pred, y_real: weighted_MSE(
            y_pred, y_real, shg1_weight, shg2_weight, sfg_weight
        ),
    )

    # Evaluate the model on the test dataset
    test_loss = single_pass(
        model,
        "/u/scratch/t/theerfan/JackData/test",
        "cpu",
        optimizer,
        lambda y_pred, y_real: weighted_MSE(
            y_pred, y_real, shg1_weight, shg2_weight, sfg_weight
        ),
    )

    # Report the test loss back to Ray Tune
    tune.report(loss=test_loss)


def tune_train():
    # Initialize Ray
    ray.init()

    # Specify the hyperparameter search space
    config = {
        "shg1_weight": tune.uniform(0, 1),
        "shg2_weight": tune.uniform(0, 1),
        "sfg_weight": tune.uniform(0, 1),
    }

    # Ensure the sum of hyperparameters equals 1 using a constraint
    constraint = "shg1_weight + shg2_weight + sfg_weight <= 1"

    # Run the experiments
    analysis = tune.run(
        train_model,
        config=config,
        resources_per_trial={"cpu": 2},
        num_samples=100,  # Number of hyperparameter combinations to try
        stop={"loss": 0.01},  # Stop trials if the loss goes below this threshold
        constraint=constraint,
    )

    # Print the best hyperparameters
    print("Best hyperparameters found were: ", analysis.best_config)


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


# (SHG1, SHG2) + SFG * 2
# (1892 * 2 + 348) * 2 = 8264


def do_the_prediction(model, model_param_path, output_dir, test_dataset, verbose=1):
    if verbose:
        print("Initialized the dataset")

    predict(
        model,
        model_param_path=model_param_path,
        test_dataset=test_dataset,
        use_gpu=True,
        data_parallel=False,
        output_dir=output_dir,
        output_name="all_preds.npy",
        verbose=1,
    )


def main_train(
    model: torch.nn.Module,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    config: dict = None,
):

    if config is not None:
        shg1_weight = config["shg1_weight"]
        shg2_weight = config["shg2_weight"]
        sfg_weight = config["sfg_weight"]
        def tuned_custom_loss(y_pred, y_real):
            return custom_loss(y_pred, y_real, shg1_weight, shg2_weight, sfg_weight)
    else:
        tuned_custom_loss = None

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
        custom_loss=tuned_custom_loss or custom_loss,
        epoch_save_interval=epoch_save_interval,
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
        do_the_prediction(
            model,
            args.model_param_path,
            args.data_dir,
            args.output_dir,
        )
    else:
        if args.tune_train == 1:
            print(f"Tune train mode for model {args.model}")
            tune_train()
        else:
            print(f"Training mode for model {args.model}")
            main_train(
                model,
                args.data_dir,
                args.num_epochs,
                custom_loss,
                args.epoch_save_interval,
                args.output_dir,
            )
