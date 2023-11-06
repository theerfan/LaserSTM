import numpy as np
import torch
import torch.nn as nn
from neuralop import H1Loss, LpLoss
from neuralop.models import FNO
from neuralop import Trainer
from LSTM.utils import CustomSequence
from LSTM.training import train
from typing import Tuple
import logging  


def main_NFO(
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
    model = FNO(n_modes=(16,), hidden_channels=64, in_channels=10, out_channels=10)
    model = model.to(device)

    # Loss function
    custom_loss = custom_loss or H1Loss()

    trained_model, train_losses, val_losses = train(
        model,
        train_dataset,
        num_epochs=args.num_epochs,
        val_dataset=val_dataset,
        data_parallel=True,
        out_dir=args.output_dir,
        model_save_name="NFO_model",
        verbose=args.verbose,
        save_checkpoints=True,
        custom_loss=custom_loss,
        epoch_save_interval=args.epoch_save_interval,
        custom_single_pass=NFO_single_pass,
    )


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