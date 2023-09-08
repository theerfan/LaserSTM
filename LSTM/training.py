import os
import time
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from LSTM.utils import CustomSequence

import logging

logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
)


# Returns the normalized loss and the last loss
# Over a single pass of the dataset
def single_pass(
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
    # Erfan: This is ChatGPT-improved, let's see if it manages batches correctly
    # Also, TODO: Make it such that it goes through a list of hyperparameters
    for i, sample_generator in enumerate(dataset):
        if verbose:
            log_str = f"Processing batch {(i+1) / data_len}"
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


def train(
    model: nn.Module,
    train_dataset: CustomSequence,
    num_epochs: int = 10,
    val_dataset: CustomSequence = None,
    use_gpu: bool = True,
    data_parallel: bool = True,
    out_dir: str = ".",
    model_name: str = "model",
    verbose: bool = True,
    save_checkpoints: bool = True,
    custom_loss=None,
    epoch_save_interval: int = 1,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if use_gpu:
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
    else:
        pass
    model.to(device)

    # Check if the output directory exists, if not, create it
    os.makedirs(out_dir, exist_ok=True)

    criterion = custom_loss or nn.MSELoss()
    # TODO: Other optimizers for time series?
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    if val_dataset is not None:
        val_losses = []

    # Train
    for epoch in range(num_epochs):
        if verbose:
            log_str = f"Epoch {epoch + 1} of {num_epochs}"
            print(log_str)
            logging.info(log_str)

        model.train()

        train_loss, last_train_loss = single_pass(
            model, train_dataset, device, optimizer, criterion
        )
        train_losses.append(train_loss)

        # Validation
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_loss, _ = single_pass(model, val_dataset, device, None, criterion)
                val_losses.append(val_loss)
        else:
            # For formatting purposes, but it basically means that it's nan
            val_loss = -1

        # Erfan: Maybe delete the older checkpoints after saving the new one?
        # (So you wouldn't have terabytes of checkpoints just sitting there)
        # approved.
        if save_checkpoints and epoch % epoch_save_interval == 0:
            checkpoint_path = os.path.join(out_dir, f"{model_name}_epoch_{epoch+1}.pth")
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
            np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
            if val_dataset is not None:
                np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))
            else:
                pass
        else:
            pass

        if verbose:
            log_str = f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}, Val Loss={val_loss:.18f}"
            print(log_str)
            logging.info(log_str)
        else:
            pass

    # Save everything at the end
    np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
    if val_dataset is not None:
        np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))
    else:
        pass
    torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}.pth"))
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    return model, train_losses, val_losses


def predict(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    use_gpu: bool = True,
    data_parallel: bool = False,
    output_dir: str = ".",
    output_name: str = "all_preds.npy",
    verbose: bool = True,
) -> np.ndarray:

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if not use_gpu:
        log_str = "Warning: GPU not available, using CPU instead."
        print(log_str)
        logging.info(log_str)

    # Load model parameters if path is provided
    if model_param_path is not None:
        params = torch.load(model_param_path, map_location=device)
        # remove the 'module.' prefix from the keys
        params = {k.replace("module.", ""): v for k, v in params.items()}
        model.load_state_dict(params, strict=False)
    else:
        pass

    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    if data_parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            log_str = (
                "Warning: Data parallelism not available, using single GPU instead."
            )
            print(log_str)
            logging.info(log_str)
    else:
        try:
            model = model.module
        except AttributeError:
            pass

    model = model.to(device)
    model.eval()

    all_preds = []
    final_shape = None

    if verbose:
        log_str = "Finished loading the model, starting prediction."
        print(log_str)
        logging.info(log_str)
        whole_start = time.time()

    dataset_len = len(test_dataset)

    with torch.no_grad():
        for j in range(dataset_len):
            sample_generator = test_dataset[j]

            if verbose:
                this_batch_start = time.time()
                this_batch_elapsed = this_batch_start - whole_start
                log_str_2 = f"Processing batch {(j+1)} / {len(test_dataset)} at time {this_batch_elapsed}"
                print(log_str_2)
                logging.info(log_str_2)

            counter = 0
            for X, y in sample_generator:
                if verbose:
                    now_time = time.time()
                    elapsed = now_time - this_batch_start
                    this_batch_start = now_time

                    log_str_3 = f"Processing sample {(counter+1)} / n at time {elapsed}"
                    print(log_str_3)
                    logging.info(log_str_3)

                    counter += 1
                X, y = X.to(torch.float32).to(device), y.to(torch.float32).to(device)

                if final_shape is None:
                    final_shape = X.shape[-1]

                for _ in range(100):
                    pred = model(X)
                    X = X[:, 1:, :]  # pop first

                    # add to last
                    X = torch.cat((X, torch.reshape(pred, (-1, 1, final_shape))), 1)

                all_preds.append(pred.squeeze())

            if verbose:
                log_str_4 = f"Finished processing samples in {j} batch."
                print(log_str_4)
                logging.info(log_str_4)

        if verbose:
            log_str_5 = "Finished processing all batches."
            print(log_str_5)
            logging.info(log_str_5)

    all_preds = torch.stack(all_preds, dim=0).cpu().numpy()
    np.save(os.path.join(output_dir, f"{output_name}"), all_preds)
    return all_preds


def tune_train_lstm(
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
            use_gpu=True,
            data_parallel=True,
            out_dir=output_dir,
            model_name="model" + str(combo),
            verbose=verbose,
            save_checkpoints=True,
            custom_loss=current_loss,
            epoch_save_interval=epoch_save_interval,
        )

        results[combo] = val_losses.flatten().mean()

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
    num_epochs: int,
    custom_loss: Callable,
    epoch_save_interval: int,
    output_dir: str,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    verbose: int = 1,
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray, np.ndarray]:
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


# Erfan-Jack Meeting:
# 1. The sfg signal didn't look so good when the input pulse was a bit complicated
# Re-proportion loss function to make sfg more important (1892 * 2 for shg, 348 for sfg)
# 2. Instead of MSE, adding some convolution based-loss function might be better
# Look up Pierson-correlation coefficient: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# [e.g, overkill rn]:
# Time-Frequency Representations: If you are working with time-frequency representations like spectrograms, you can use loss functions that operate directly on these representations. For instance, you can use the spectrogram difference as a loss, or you can use perceptual loss functions that take into account human perception of audio signals. [introduces a lot of time]
# Wasserstein Loss: The Wasserstein loss, also known as Earth Mover's Distance (EMD), is a metric used in optimal transport theory. It measures the minimum cost of transforming one distribution into another. It has been applied in signal processing tasks, including time and frequency domain analysis, to capture the structure and shape of signals.
# 3. Check out the intPhEn part again and see what else can be done [Ae^{i\phi}}]
# (it's more natural to the optics domain, energy is proportional to the intensity,
# it's only 3 values for 2shg, sfg, but it's really important for how strong the non-linear process is)
# [ you get jump disconts in 0 to 2pi, we also have to note that phase is only important when we have intensity]
# phase unwrapping (np.unwrap) [add the inverse relation of phase/intensity to the loss function]
# (real intensity and set threshold for it to not affect the phase too much)
# or could cut the phase vector shorter than the intensity vector
