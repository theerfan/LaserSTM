import os
import time
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from LSTM.utils import CustomSequence
from torch.utils.data import DataLoader

from Analysis.analyze_reim import do_analysis

import logging

logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
)


# Returns the normalized loss and the last loss
# Over a single pass of the dataset
def LSTM_single_pass(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    loss_fn: nn.Module,
    verbose: bool = True,
) -> Tuple[float, float]:
    pass_loss = 0
    pass_len = len(dataloader)
    for i, (X_batch, y_batch) in enumerate(dataloader):
        if verbose:
            print(f"On batch {i} out of {pass_len}")

        if optimizer is not None:
            optimizer.zero_grad()

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        pass_loss += loss.item() * X_batch.size(0)

    return pass_loss / pass_len, loss


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
    custom_loss=None,
    epoch_save_interval: int = 1,
    custom_single_pass: Callable = None,
    batch_size: int = 200,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    criterion = custom_loss or nn.MSELoss()
    # TODO: Other optimizers for time series?
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )
    train_losses = []
    val_losses = [] if val_dataset is not None else None

    single_pass_fn = custom_single_pass or LSTM_single_pass

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Train
    for epoch in range(num_epochs):
        if verbose:
            log_str = f"Epoch {epoch + 1} of {num_epochs}"
            print(log_str)
            logging.info(log_str)

        model.train()

        train_loss, last_train_loss = single_pass_fn(
            model, train_dataloader, optimizer, criterion
        )
        train_losses.append(train_loss)

        # Validation
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_loss, _ = single_pass_fn(model, val_dataloader, None, criterion)
                val_losses.append(val_loss)
            # Update the learning rate if we're not improving
            scheduler.step(val_loss)
        else:
            pass

        # Erfan: Maybe delete the older checkpoints after saving the new one?
        # (So you wouldn't have terabytes of checkpoints just sitting there)
        # approved.
        if save_checkpoints and epoch % epoch_save_interval == 0:
            checkpoint_path = os.path.join(out_dir, f"{model_save_name}_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": last_train_loss.item(),
                },
                checkpoint_path,
            )

            # Save these things at checkpoints
            np.save(
                os.path.join(out_dir, f"{model_save_name}_epoch_{epoch+1}_train_losses.npy"),
                np.array(train_losses),
            )

            # valuation
            if val_dataset is not None:
                np.save(
                    os.path.join(
                        out_dir, f"{model_save_name}_epoch_{epoch+1}_val_losses.npy"
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
            print(log_str)
            logging.info(log_str)
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


def predict(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    output_dir: str = ".",
    output_name: str = "all_preds.npy",
    verbose: bool = True,
    model_save_name: str = "",
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model parameters if path is provided
    if model_param_path is not None:
        params = torch.load(model_param_path, map_location=device)
        if isinstance(params, dict) and "model_state_dict" in params:
            params = params["model_state_dict"]
        try:
            model.load_state_dict(params)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(params)

    if device == "gpu":
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    # We do this to have the same length for both dataloader and dataset
    # for "analysis" purposes
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.get_num_samples_per_file())

    model.to(device)
    model.eval()
    all_preds = None
    final_shape = None
    # i = 0
    with torch.no_grad():
        for j, (X_batch, _) in enumerate(test_dataloader):
            if verbose:
                print(f"On batch {j} out of {len(test_dataloader)}")
            X_batch = X_batch.to(torch.float32)

            if final_shape is None:
                final_shape = X_batch.shape[-1]

            for _ in range(100):  # need to predict 100 times
                pred = model(X_batch)
                X_batch = X_batch[:, 1:, :]  # pop first

                # add to last
                X_batch = torch.cat(
                    (X_batch, torch.reshape(pred, (-1, 1, final_shape))), 1
                )

            # Keep all_preds on GPU instead of sending it back to CPU at "each" iteration
            # Erfan TODO: Best if we know the value of *pred.squeeze().shape beforehand
            if all_preds is None:
                all_preds = torch.zeros(
                    (len(test_dataset), *pred.squeeze().shape), device=device
                )
            else:
                pass
            all_preds[j] = pred.squeeze()

    # And then we do the concatenation here and send it back to CPU
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    np.save(os.path.join(output_dir, f"{model_save_name}_{output_name}"), all_preds)
    return all_preds


def tune_train_lstm(
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
    data_dir: str = ".",
    batch_size: int = 200,
):

    # Generate possible values for each hyperparameter with a step size of 0.2
    possible_values = np.arange(0.1, 1.1, 0.2)  # Include 1.0 as a possible value

    # Generate all combinations where the sum of the hyperparameters equals 1
    combinations = [
        (np.round(shg, 3), np.around(sfg, 3))
        for shg in possible_values
        for sfg in possible_values
        if np.isclose(shg + sfg, 1.0)
    ]

    results = {}

    for combo in combinations:
        shg_weight, sfg_weight = combo

        if verbose:
            print(f"Weights combination -> SHG: {shg_weight} , SFG: {sfg_weight}")

        model_save_name = f"{model_save_name}_{shg_weight}_{sfg_weight}"

        def current_loss(y_pred, y_real):
            return custom_loss(
                y_pred,
                y_real,
                shg_weight=shg_weight,
                sfg_weight=sfg_weight,
            )

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
            custom_loss=current_loss,
            epoch_save_interval=epoch_save_interval,
            batch_size=batch_size
        )

        results[combo] = val_losses.flatten().mean()

        all_test_preds = predict(
            model,
            model_param_path=os.path.join(
                output_dir, f"{model_save_name}_epoch_{num_epochs}.pth"
            ),
            test_dataset=test_dataset,
            data_parallel=False,
            output_dir=output_dir,
            output_name="all_preds.npy",
            verbose=verbose,
            model_save_name=model_save_name + f"_epoch_{num_epochs}",
        )

    # Find the best hyperparameters based on test loss
    best_hyperparameters = min(results, key=results.get)

    log_str = f"Best hyperparameters: {best_hyperparameters}"
    print(log_str)
    logging.info(log_str)

    log_str = f"Val loss: {results[best_hyperparameters]}"
    print(log_str)
    logging.info(log_str)

    log_str = f"Results: {results}"
    print(log_str)
    logging.info(log_str)

    return best_hyperparameters, results


# (SHG1, SHG2) + SFG * 2
# (1892 * 2 + 348) * 2 = 8264


def test_train_lstm(
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
    custom_single_pass: Callable = LSTM_single_pass,
    data_dir: str = ".",
    analysis_file_idx: int = 90,
    analysis_item_idx: int = 15,
    batch_size: int = 200,
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
        batch_size=batch_size
    )

    last_model_name = f"{model_save_name}_epoch_{num_epochs}"

    all_test_preds = predict(
        model,
        model_param_path=os.path.join(output_dir, last_model_name + ".pth"),
        test_dataset=test_dataset,
        data_parallel=False,
        output_dir=output_dir,
        output_name="all_preds.npy",
        verbose=verbose,
        model_save_name=last_model_name,
    )

    ## automatically analyze the results
    # adjust the "relative" position of the file, 
    # since we get the "absolute" index of the file as input
    testset_starting_point = test_dataset.file_indexes[0]

    do_analysis(
        output_dir=output_dir,
        data_directory=data_dir,
        model_save_name=model_save_name + f"_epoch_{num_epochs}",
        file_idx=testset_starting_point - analysis_file_idx,
        item_idx=analysis_item_idx,
    )

    return trained_model, train_losses, val_losses, all_test_preds


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
