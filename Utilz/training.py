import os
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from Utilz.data import CustomSequence, X_Dataset, Y_Dataset
from torch.utils.data import DataLoader

from Analysis.analyze_reim import do_analysis

import logging

import time
import matplotlib.pyplot as plt

from functools import partial

import h5py

logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
)


# Returns the normalized loss and the last loss
# Over a single pass of the dataset
def default_single_pass(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    loss_fn: Callable,
    verbose: bool = True,
    max_norm: float = 1.0,
) -> Tuple[float, float]:
    pass_loss = 0
    pass_len = len(dataloader)
    torch.autograd.set_detect_anomaly(True)
    for i, (X_batch, y_batch) in enumerate(dataloader):
        if verbose:
            print(f"On batch {i} out of {pass_len}")

        if optimizer is not None:
            optimizer.zero_grad()

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        logging.info(f"Loss: {loss.mean().item()}")

        if optimizer is not None:
            # If the loss is a scalar,
            if len(loss.shape) == 0:
                loss.backward()
                optimizer.step()
            else:
                for l in loss:
                    l.backward(retain_graph=True)
                optimizer.step()
        
            # do gradient clipping because we ran into exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        pass_loss += loss.mean().item()

    return pass_loss / pass_len, loss.mean().item()


def train(
    model: nn.Module,
    train_dataset: CustomSequence,
    num_epochs: int = 10,
    val_dataset: CustomSequence = None,
    data_parallel: bool = True,
    out_dir: str = ".",
    model_save_name: str = "model",
    verbose: bool = True,
    save_checkpoints: bool = True,
    custom_loss: Callable = None,
    epoch_save_interval: int = 1,
    custom_single_pass: Callable = None,
    batch_size: int = 200,
    model_param_path: str = None,
    learning_rate: float = 1e-4,
    shuffle: bool = True,
    test_criterion: Callable = None,
    max_norm: float = 1.0,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, optimizer_params = load_model_params(model, model_param_path, device)

    # if model param path obeys the naming convention, we can extract the epoch number
    # and start from there
    if model_param_path is not None:
        try:
            epoch_start = int(model_param_path.split("_")[-1].split(".")[0])
        except ValueError:
            epoch_start = 0
    else:
        epoch_start = 0

    if device == "cpu":
        Warning("GPU not available, using CPU instead.")
    elif data_parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            log_str = f"Using {torch.cuda.device_count()} GPUs!"
            print(log_str)
            logging.info(log_str)
        else:
            Warning("Data parallelism not available, using single GPU instead.")
    else:
        pass
    model.to(device)

    # Check if the output directory exists, if not, create it
    os.makedirs(out_dir, exist_ok=True)

    # default to MSE loss if no custom loss is provided
    train_criterion = custom_loss or nn.MSELoss()
    # if no test criterion is provided, use the train criterion
    test_criterion = test_criterion or train_criterion

    # TODO: Other optimizers for time series?
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_params is not None:
        optimizer.load_state_dict(optimizer_params)

    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=2, verbose=True
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0.0001, last_epoch=-1
    )
    train_losses = []
    val_losses = [] if val_dataset is not None else None

    single_pass_fn = custom_single_pass or default_single_pass

    shuffle = bool(shuffle)
    if shuffle:
        print("Everyday I'm shuffling!")
        logging.info("Everyday I'm shuffling!")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    initiation_time = time.time()
    epoch_start_time = time.time()

    # Train
    for epoch in range(num_epochs):
        if verbose:
            log_str = f"Epoch {epoch_start + epoch + 1} of {epoch_start + num_epochs}"
            print(log_str)
            logging.info(log_str)

        model.train()

        train_loss, last_train_loss = single_pass_fn(
            model, train_dataloader, optimizer, train_criterion, max_norm
        )
        train_losses.append(train_loss)

        # Validation
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_loss, _ = single_pass_fn(
                    model, val_dataloader, None, test_criterion, verbose=False
                )
                val_losses.append(val_loss)
            # Update the learning rate if we're not improving
            scheduler_plateau.step(val_loss)
            scheduler_cosine.step()
        else:
            pass

        # Erfan: Maybe delete the older checkpoints after saving the new one?
        # (So you wouldn't have terabytes of checkpoints just sitting there)
        # approved.
        if save_checkpoints and epoch % epoch_save_interval == 0:
            checkpoint_path = os.path.join(
                out_dir, f"{model_save_name}_epoch_{epoch_start + epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    # save model architecture
                    "model": model,
                    # this was for older versions of saving
                    # "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": last_train_loss,
                },
                checkpoint_path,
            )

            # Save these things at checkpoints
            np.save(
                os.path.join(
                    out_dir,
                    f"{model_save_name}_epoch_{epoch_start + epoch + 1}_train_losses.npy",
                ),
                np.array(train_losses),
            )

            # valuation
            if val_dataset is not None:
                np.save(
                    os.path.join(
                        out_dir,
                        f"{model_save_name}_epoch_{epoch_start + epoch + 1}_val_losses.npy",
                    ),
                    np.array(val_losses),
                )
            else:
                pass
        else:
            pass

        if verbose:
            log_str = f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}"
            if val_loss is not None:
                log_str += f"Val Loss={val_loss:.18f}"

            this_epoch_time = time.time() - epoch_start_time
            log_str += f"\nEpoch time: {this_epoch_time} seconds"
            # Calculate the remaining time to finish training all epochs
            eta = (num_epochs - epoch - 1) * (this_epoch_time)
            # turn it into a specific date and time
            eta = time.strftime(
                "%a, %d %b %Y %H:%M:%S", time.localtime(eta + initiation_time)
            )
            log_str += f"\nEstimated time to finish: {eta}"
            print(log_str)
            logging.info(log_str)
            # reset start time
            epoch_start_time = time.time()
        else:
            pass

    # Save everything at the end
    train_losses = np.array(train_losses)
    np.save(os.path.join(out_dir, "train_losses.npy"), train_losses)
    if val_dataset is not None:
        val_losses = np.array(val_losses)
        np.save(os.path.join(out_dir, "val_losses.npy"), val_losses)
    else:
        pass

    torch.save(model.state_dict(), os.path.join(out_dir, f"{model_save_name}.pth"))

    return model, train_losses, val_losses


def load_model_params(
    model: nn.Module, model_param_path: str, device: torch.device
) -> (nn.Module, dict):
    optimizer_state = None
    # Load model parameters if path is provided
    if model_param_path is not None:
        print(f"Loading pre-trained model parameters from {model_param_path}")
        # replace the .pth extension with .pt to load the "clean" model
        params = torch.load(model_param_path, map_location=device)

        try:
            if "model" in params:
                model = params["model"]
            elif "model_state_dict" in params:
                model.load_state_dict(params["model_state_dict"])
            else:
                raise ValueError(
                    "The model parameter path provided does not contain a valid model!"
                )

            if "optimizer_state_dict" in params:
                optimizer_state = params["optimizer_state_dict"]

        except RuntimeError:
            # remove the 'module.' prefix from the keys if it's a DataParallel model
            params = {k.replace("module.", ""): v for k, v in params.items()}
            model = torch.nn.DataParallel(model)
            model.load_state_dict(params)
    else:
        pass

    return model, optimizer_state


def predict(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    output_dir: str = ".",
    output_name: str = "all_preds.h5",
    verbose: bool = True,
    model_save_name: str = "",
    batch_size: int = None,
    is_slice: bool = True,
    crystal_length: int = 100,
    load_model: bool = True,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_model:
        model, _ = load_model_params(model, model_param_path, device)

    if device == "gpu":
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    # We do this to have the same length for both dataloader and dataset
    # for "analysis" purposes
    batch_size = batch_size or test_dataset._num_samples_per_file
    # If #samples_per_file is not divisible by batch_size, find the nearest smaller batch size that is
    while test_dataset._num_samples_per_file % batch_size != 0:
        batch_size -= 1
    print(f"To confirm - Batch size: {batch_size}")

    testset_starting_point = test_dataset.file_indexes[0]

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)
    model.eval()
    current_preds = []
    final_shape = None

    if model_save_name != "":
        model_save_name = f"{model_save_name}_"
    file_save_name = f"{model_save_name}{output_name}"

    start_time = time.time()

    with h5py.File(os.path.join(output_dir, file_save_name), "w") as h5_file:
        with torch.no_grad():
            for j, (X_batch, y_batch) in enumerate(test_dataloader):
                X_batch = X_batch.to(device)
                if verbose:
                    print(f"On batch {j} out of {len(test_dataloader)}")

                pred = one_predict_pass(
                    model, X_batch, batch_size, final_shape, is_slice, crystal_length
                )

                current_preds.append(pred.squeeze().cpu().numpy())

                if (
                    len(current_preds) * batch_size
                    == test_dataset._num_samples_per_file
                ):
                    print("adding something to all_preds!!")
                    h5_file.create_dataset(
                        f"dataset_{testset_starting_point + j}",
                        data=np.concatenate(current_preds, axis=0),
                    )
                    current_preds = []

    end_time = time.time()

    # TODO: This could be a way of trying to stop the process from getting killed for some unknown reason!
    # del model

    # print elapsed time in seconds
    print(f"Elapsed time: {end_time - start_time} seconds")


def one_predict_pass(
    model,
    X_batch,
    batch_size,
    final_shape,
    is_slice,
    crystal_length,
    return_all_preds: bool = False,
):
    preds_list = []
    if final_shape is None:
        final_shape = X_batch.shape[-1]
    if is_slice:
        expected_size = (batch_size, 10, 8264)
        for i in range(crystal_length):  # need to predict 100 times
            assert (
                X_batch.size() == expected_size
            ), f"Tensor size should be {expected_size}, but got {X_batch.size()}"
            # run an inference
            pred = model(X_batch)
            if return_all_preds:
                preds_list.append(pred)
            # pop the first element of every x in the batch
            X_batch = X_batch[:, 1:, :]
            # add the inferences to the end of every x in the batch
            X_batch = torch.cat((X_batch, torch.reshape(pred, (-1, 1, final_shape))), 1)
    else:
        pred = model(X_batch)

    if return_all_preds:
        return preds_list
    else:
        return pred


def tune_and_train(
    model: torch.nn.Module,
    model_save_name: str,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    verbose: int = 1,
    custom_single_pass: Callable = default_single_pass,
    data_dir: str = ".",
    analysis_file_idx: int = 90,
    analysis_item_idx: int = 15,
    batch_size: int = 200,
    model_param_path: str = None,
    crystal_length: int = 100,
    is_slice: bool = True,
    model_dict: dict = None,
    learning_rate: float = 1e-4,
    shuffle: int = 1,
    max_norm: float = 1.0,
):
    stacked_layers_combinations = [1, 2]
    lstm_hidden_size_combinations = [1024, 2048, 4096]
    mlp_hidden_size_multiplier_combinations = [1, 2, 4]

    def time_domain_mse(y_real, y_pred):
        mse = nn.MSELoss()
        batch_size = y_real.shape[0]
        losses = np.zeros((batch_size))
        for i in range(batch_size):
            (
                sfg_time_true,
                sfg_time_pred,
                shg1_time_true,
                shg1_time_pred,
                shg2_time_true,
                shg2_time_pred,
            ) = do_analysis(
                ".",
                "/mnt/oneterra/SFG_reIm_h5/",
                model_save_name,
                analysis_file_idx,
                analysis_item_idx,
                ".",
                100,
                y_pred[i].cpu().numpy(),
                y_real[i].cpu().numpy(),
                return_vals=True,
            )
            # turn the numpy arrays into tensors
            sfg_time_true = torch.from_numpy(sfg_time_true.real)
            sfg_time_pred = torch.from_numpy(sfg_time_pred.real)
            shg1_time_true = torch.from_numpy(shg1_time_true.real)
            shg1_time_pred = torch.from_numpy(shg1_time_pred.real)
            shg2_time_true = torch.from_numpy(shg2_time_true.real)
            shg2_time_pred = torch.from_numpy(shg2_time_pred.real)

            # calculate the losses
            losses[i] = (
                mse(sfg_time_true, sfg_time_pred)
                + mse(shg1_time_true, shg1_time_pred)
                + mse(shg2_time_true, shg2_time_pred)
            )

        return torch.from_numpy(losses)

    init_model_save_name = model_save_name
    results = {}

    for num_layers in stacked_layers_combinations:
        for lstm_hidden_size in lstm_hidden_size_combinations:
            for mlp_hidden_size_multiplier in mlp_hidden_size_multiplier_combinations:
                combo_str = f"nlayers_{num_layers}_lstmhs_{lstm_hidden_size}_mlphs_{mlp_hidden_size_multiplier}"

                model_save_name = f"{init_model_save_name}_{combo_str}"

                model_dict["lstm_hidden_size"] = lstm_hidden_size
                model_dict["num_layers"] = num_layers
                model_dict["linear_layer_size"] = (
                    lstm_hidden_size * mlp_hidden_size_multiplier
                )
                # initialize a new model to train with the new hyperparameters
                model = type(model)(**model_dict)

                # Train the model with the training dataset
                # This assumes a train function is available and works similarly to the one in main.py
                # We'll also need to modify the train function to accept the loss function with hyperparameters
                trained_model, train_losses, val_losses = train(
                    model,
                    train_dataset,
                    num_epochs=num_epochs,
                    val_dataset=val_dataset,
                    data_parallel=True,
                    out_dir=output_dir,
                    model_save_name=model_save_name,
                    verbose=verbose,
                    save_checkpoints=True,
                    custom_loss=custom_loss,
                    epoch_save_interval=epoch_save_interval,
                    batch_size=batch_size,
                    model_param_path=model_param_path,
                    learning_rate=learning_rate,
                    shuffle=shuffle,
                    test_criterion=time_domain_mse,
                    max_norm=max_norm
                )

                # select model with the lowest validation loss
                best_val_loss_idx = np.argmin(val_losses)
                best_iter_model_name = f"{model_save_name}_epoch_{best_val_loss_idx}"
                combo_str = f"{combo_str}_epoch_{best_val_loss_idx}"

                results[combo_str] = val_losses[best_val_loss_idx]

                predict(
                    model,
                    model_param_path=os.path.join(
                        output_dir, best_iter_model_name + ".pth"
                    ),
                    test_dataset=test_dataset,
                    output_dir=output_dir,
                    verbose=verbose,
                    model_save_name=model_save_name + f"_epoch_{num_epochs}",
                    is_slice=is_slice,
                    crystal_length=crystal_length,
                    load_model=False,
                )

    # Find the best hyperparameters based on test loss (returns the key)
    best_hyperparameters = min(results, key=results.get)

    # find best model's save name from the best hyperparameters
    model_save_name = f"{model_save_name}_{best_hyperparameters}"

    log_str = f"Best hyperparameters: {best_hyperparameters}"
    print(log_str)
    logging.info(log_str)

    log_str = f"Val loss: {results[best_hyperparameters]}"
    print(log_str)
    logging.info(log_str)

    log_str = f"Results: {results}"
    print(log_str)
    logging.info(log_str)

    do_analysis(
        output_dir=output_dir,
        data_directory=data_dir,
        model_save_name=model_save_name,
        file_idx=analysis_file_idx,
        item_idx=analysis_item_idx,
    )

    return best_hyperparameters, results


# (SHG1, SHG2) + SFG * 2
# (1892 * 2 + 348) * 2 = 8264


def train_and_test(
    model: torch.nn.Module,
    model_save_name: str,
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    verbose: int = 1,
    custom_single_pass: Callable = default_single_pass,
    data_dir: str = ".",
    analysis_file_idx: int = 90,
    analysis_item_idx: int = 15,
    batch_size: int = 200,
    model_param_path: str = None,
    crystal_length: int = 100,
    is_slice: bool = True,
    model_dict: dict = None,
    learning_rate: float = 1e-4,
    shuffle: int = 1,
    max_norm: float = 1.0,
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray, np.ndarray]:
    trained_model, train_losses, val_losses = train(
        model,
        train_dataset,
        num_epochs=num_epochs,
        val_dataset=val_dataset,
        data_parallel=True,
        out_dir=output_dir,
        model_save_name=model_save_name,
        verbose=verbose,
        save_checkpoints=True,
        custom_loss=custom_loss,
        epoch_save_interval=epoch_save_interval,
        custom_single_pass=custom_single_pass,
        batch_size=batch_size,
        model_param_path=model_param_path,
        learning_rate=learning_rate,
        shuffle=shuffle,
        max_norm=max_norm
    )

    if model_param_path is not None:
        try:
            epoch_start = int(model_param_path.split("_")[-1].split(".")[0])
        except ValueError:
            epoch_start = 0
    else:
        epoch_start = 0

    # select model with the lowest validation loss
    best_val_loss_idx = np.argmin(val_losses) + epoch_start
    last_model_name = f"{model_save_name}_epoch_{best_val_loss_idx}"

    # In predict we use the path of the model that was trained the latest

    predict(
        model,
        model_param_path=os.path.join(output_dir, last_model_name + ".pth"),
        test_dataset=test_dataset,
        output_dir=output_dir,
        verbose=verbose,
        model_save_name=last_model_name,
        crystal_length=crystal_length,
        is_slice=is_slice,
        # If it's a slice, it will need smaller batch size
        load_model=True,
    )

    do_analysis(
        output_dir=output_dir,
        data_directory=data_dir,
        model_save_name=model_save_name + f"_epoch_{best_val_loss_idx}",
        file_idx=analysis_file_idx,
        item_idx=analysis_item_idx,
    )

    return trained_model, train_losses, val_losses


# Erfan-Jack Meeting:
# 1. The sfg signal didn't look so good when the input pulse was a bit complicated
# Re-proportion loss function to make sfg more important (1892 * 2 for shg, 348 for sfg)

# Time-Frequency Representations: If you are working with time-frequency representations like spectrograms, you can use loss functions that operate directly on these representations. For instance, you can use the spectrogram difference as a loss, or you can use perceptual loss functions that take into account human perception of audio signals. [introduces a lot of time]
# Wasserstein Loss: The Wasserstein loss, also known as Earth Mover's Distance (EMD), is a metric used in optimal transport theory. It measures the minimum cost of transforming one distribution into another. It has been applied in signal processing tasks, including time and frequency domain analysis, to capture the structure and shape of signals.
# 3. Check out the intPhEn part again and see what else can be done [Ae^{i\phi}}]
# (it's more natural to the optics domain, energy is proportional to the intensity,
# it's only 3 values for 2shg, sfg, but it's really important for how strong the non-linear process is)
# [ you get jump disconts in 0 to 2pi, we also have to note that phase is only important when we have intensity]
# phase unwrapping (np.unwrap) [add the inverse relation of phase/intensity to the loss function]
# (real intensity and set threshold for it to not affect the phase too much)
# or could cut the phase vector shorter than the intensity vector


################## [These are some parts of the code I actively dislike]

def funky_predict(
    model: nn.Module,
    model_param_path: str = None,
    x_dataset: X_Dataset = None,
    y_dataset: Y_Dataset = None,
    output_dir: str = ".",
    output_name: str = "all_preds.h5",
    verbose: bool = True,
    model_save_name: str = "",
    x_batch_size: int = None,
    y_batch_size: int = None,
    is_slice: bool = True,
    crystal_length: int = 100,
    load_model: bool = True,
    analysis_file_idx: int = 90,
    analysis_item_idx: int = 15,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_model:
        model, _ = load_model_params(model, model_param_path, device)
    else:
        model = model.to(device)

    x_dataloader = DataLoader(x_dataset, batch_size=x_batch_size)
    y_dataloader = DataLoader(y_dataset, batch_size=y_batch_size)

    testset_starting_point = x_dataset.file_indexes[0]

    model.to(device)
    model.eval()
    current_preds = []
    final_shape = None

    if model_save_name != "":
        model_save_name = f"{model_save_name}_"
    file_save_name = f"{model_save_name}{output_name}"

    start_time = time.time()

    our_idx = (
        analysis_file_idx - testset_starting_point
    ) * x_dataset._num_samples_per_file + analysis_item_idx

    with torch.no_grad():
        for j, (X_batch, y_batch) in enumerate(zip(x_dataloader, y_dataloader)):
            if verbose:
                print(f"On batch {j} out of {len(y_dataloader)}")
            # since batch size is 1 here.
            if j == our_idx:
                X_batch = X_batch.to(device)

                if final_shape is None:
                    final_shape = X_batch.shape[-1]

                if is_slice:
                    for i in range(crystal_length):  # need to predict 100 times
                        pred = model(X_batch)

                        if i == 0 or i == 1 or i == 50 or i == 99:
                            do_analysis(
                                ".",
                                "/mnt/oneterra/SFG_reIm_h5/",
                                model_save_name,
                                0,
                                0,
                                ".",
                                100,
                                pred[-1].cpu().numpy(),
                                y_batch[i].cpu().numpy(),
                                f"analysis-scaled-file{analysis_file_idx}-item-{analysis_item_idx}-slice-{i}.jpg",
                            )
                            a = 12

                        X_batch = X_batch[:, 1:, :]  # pop first

                        # add to last
                        X_batch = torch.cat(
                            (X_batch, torch.reshape(pred, (-1, 1, final_shape))), 1
                        )
                else:
                    pred = model(X_batch)
                current_preds.append(pred.squeeze().cpu().numpy())

            else:
                pass

    end_time = time.time()

    # TODO: This could be a way of trying to stop the process from getting killed for some unknown reason!
    # del model

    # print elapsed time in seconds
    print(f"Elapsed time: {end_time - start_time} seconds")


def predict_timing(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    output_dir: str = ".",
    output_name: str = "all_preds.h5",
    verbose: bool = True,
    model_save_name: str = "",
    batch_size: int = None,
    is_slice: bool = True,
    crystal_length: int = 100,
    load_model: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    
    if load_model:
        model, _ = load_model_params(model, model_param_path, device)

    batch_size = batch_size or test_dataset._num_samples_per_file
    while test_dataset._num_samples_per_file % batch_size != 0:
        batch_size -= 1

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)
    model.eval()
    final_shape = None

    if model_save_name != "":
        model_save_name = f"{model_save_name}_"

    start_time = time.time()

    with torch.no_grad():
        for j, (X_batch, y_batch) in enumerate(test_dataloader):
            X_batch = X_batch.to(device)

            pred = one_predict_pass(
                model, X_batch, batch_size, final_shape, is_slice, crystal_length
            )

    end_time = time.time()

    # print elapsed time in seconds
    print(f"Elapsed time: {end_time - start_time} seconds for {len(test_dataloader)} batches of size {batch_size}")

def time_alan_code(test_dataset: CustomSequence, load_in_gpu: bool = True):
    class LSTMModel_Alan(nn.Module): 
        # basic one with two linear layers and final output with sigmoid
        def __init__(self, input_size, hidden_size=1024, num_layers=1):
            super().__init__()
            self.input_size = input_size
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size,
                                batch_first=True, dropout=0, num_layers=num_layers)
            self.fc1 = nn.Linear(hidden_size, 4000)
            self.fc2 = nn.Linear(4000, 4000)
            self.fc3 = nn.Linear(4000, input_size)
            self.relu = nn.ReLU()
            self.initialize_weights()

        def forward(self, x):
            # hidden state
            h_0 = torch.zeros(self.num_layers * 1, x.size(0),
                            self.hidden_size).to(x.device)  # Modified line
            # cell state
            c_0 = torch.zeros(self.num_layers * 1, x.size(0),
                            self.hidden_size).to(x.device)  # Modified line

            out, (hn, cn) = self.lstm(x, (h_0, c_0))
            out = self.fc1(out[:, -1, :])
            out = self.relu(out)
            out = self.fc3(out)
            out = torch.sigmoid(out)

            return out
        
        def initialize_weights(self):
            # Initialize LSTM weights and biases
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
            
                # Initialize linear layers
                nn.init.xavier_uniform_(self.fc1.weight)
                self.fc1.bias.data.fill_(0)
                nn.init.xavier_uniform_(self.fc2.weight)
                self.fc2.bias.data.fill_(0)
                nn.init.xavier_uniform_(self.fc3.weight)
                self.fc3.bias.data.fill_(0)

    
    print("Timing for Alan's model")

    alan_model = LSTMModel_Alan(8264, 1024, 1)

    if load_in_gpu:
        print("Timing for CUDA")
        predict_timing(model=alan_model, test_dataset=test_dataset, device="cuda")
    else:
        print("Timing for CPU")
        predict_timing(model=alan_model, test_dataset=test_dataset, device="cpu")
