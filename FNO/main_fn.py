import numpy as np
import torch
import torch.nn as nn
from neuralop import H1Loss, LpLoss
from neuralop import Trainer
from LSTM.utils import CustomSequence
from LSTM.training import train, predict
from FNO.model import FNO_wrapper
from Analysis.analyze_reim import do_analysis
from typing import Tuple
import logging
import os


def main_FNO(
    args: dict,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    custom_loss: callable = None,
):
    # x.shape = (100, 10, 8264)
    # (timesteps, spatial points, features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iniiate FNO model with explicit parameters
    # n_modes: tuple of ints, number of modes in each dimension (if 1d data, then (n_modes,) is going to work)
    model = FNO_wrapper(
        n_modes=(16,), hidden_channels=64, in_channels=10, out_channels=10
    )
    model = model.to(device)

    # adjust the "relative" position of the file,``
    # since we get the "absolute" index of the file as input
    testset_starting_point = test_dataset.file_indexes[0]

    if args.do_prediction == 1:
        pass
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        predict(
            model,
            model_param_path=args.model_param_path,
            test_dataset=test_dataset,
            output_dir=args.output_dir,
            output_name="all_preds.npy",
            verbose=args.verbose,
            model_save_name=args.model_save_name,
            batch_size=args.batch_size,
        )
        do_analysis(
            output_dir=args.output_dir,
            data_directory=args.data_dir,
            model_save_name=args.model_save_name,
            file_idx=testset_starting_point - args.analysis_file,
            item_idx=args.analysis_example,
        )
    else:
        # Loss function
        custom_loss = custom_loss or H1Loss()

        model_save_name = args.model_save_name
        num_epochs = args.num_epochs
        output_dir = args.output_dir

        trained_model, train_losses, val_losses = train(
            model,
            train_dataset,
            num_epochs=num_epochs,
            val_dataset=val_dataset,
            data_parallel=True,
            out_dir=output_dir,
            model_save_name=model_save_name,
            verbose=args.verbose,
            save_checkpoints=True,
            custom_loss=custom_loss,
            epoch_save_interval=args.epoch_save_interval,
            batch_size=args.batch_size,
            model_param_path=args.model_param_path,
        )

        last_model_name = f"{model_save_name}_epoch_{num_epochs}"

        # In predict we use the path of the model that was trained the latest

        all_test_preds = predict(
            model,
            model_param_path=os.path.join(output_dir, last_model_name + ".pth"),
            test_dataset=test_dataset,
            output_dir=output_dir,
            output_name="all_preds.npy",
            verbose=args.verbose,
            model_save_name=last_model_name,
            batch_size=args.batch_size,
        )

        ## automatically analyze the results
        do_analysis(
            output_dir=args.output_dir,
            data_directory=args.data_dir,
            model_save_name=args.model_save_name + f"_epoch_{args.num_epochs}",
            file_idx=testset_starting_point - args.analysis_file,
            item_idx=args.analysis_example,
        )

    return model, train_losses, val_losses, all_test_preds


# Returns the normalized loss and the last loss
# Over a single pass of the dataset
def NFO_single_pass(
    model: nn.Module,
    dataset: CustomSequence,
    device: torch.device,
    optimizer: torch.optim,
    loss_fn: nn.Module,
    verbose: bool = False,
) -> Tuple[float, float]:
    data_len = len(dataset)
    pass_loss = 0
    pass_len = 0

    for i, sample_generator in enumerate(dataset):
        # Putting this here becasue enumerator blah blah
        if i == data_len:
            break

        if verbose:
            log_str = f"Processing batch {(i+1)} / {data_len}"
            print(log_str)
            logging.info(log_str)

        for X, y in sample_generator:
            X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            pass_len += X.size(0)
            pass_loss += loss.item() * X.size(0)

    return pass_loss / pass_len, loss
