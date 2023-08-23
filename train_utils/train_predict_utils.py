import torch
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef
import numpy as np
import os
from torch.utils import data

from typing import Tuple


def add_prefix(lst: list, prefix="X"):
    """
    Add prefix to list of file names
    @param lst: list of file names
    @param prefix: prefix to add
    @return: list of file names with prefix
    """
    return [prefix + "_" + str(i) + ".npy" for i in lst]


class CustomSequence(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        file_idx: list,
        file_batch_size: int,
        model_batch_size: int,
        test_mode: bool = False,
        train_prefix: str = "X_new",
        val_prefix: str = "y_new",
    ):
        """
        Custom PyTorch dataset for loading data
        @param data_dir: directory containing data
        @param file_idx: list of file indices to load
        @param file_batch_size: number of files to load at once
        @param model_batch_size: number of samples to load at once to feed into model
        @param test_mode: whether to load data for testing
        """
        self.Xnames = add_prefix(file_idx, train_prefix)
        self.ynames = add_prefix(file_idx, val_prefix)
        self.file_batch_size = file_batch_size
        self.model_batch_size = model_batch_size
        self.test_mode = test_mode
        self.data_dir = data_dir

    def __len__(self):
        return int(np.ceil(len(self.Xnames) / float(self.file_batch_size)))

    def __getitem__(self, idx):
        batch_x = self.Xnames[
            idx * self.file_batch_size : (idx + 1) * self.file_batch_size
        ]
        batch_y = self.ynames[
            idx * self.file_batch_size : (idx + 1) * self.file_batch_size
        ]

        data = []
        labels = []

        for x, y in zip(batch_x, batch_y):
            if self.test_mode:
                # Every 100th sample
                temp_x = np.load(os.path.join(self.data_dir, x))[::100]
                # Every 100th sample, starting from 100th sample
                temp_y = np.load(os.path.join(self.data_dir, y))[99:][::100]
            else:
                temp_x = np.load(os.path.join(self.data_dir, x))
                temp_y = np.load(os.path.join(self.data_dir, y))

            data.extend(temp_x)
            labels.extend(temp_y)

        for i in range(0, len(data), self.model_batch_size):
            data_batch = data[i : i + self.model_batch_size]
            labels_batch = labels[i : i + self.model_batch_size]

            data_tensor = torch.tensor(np.array(data_batch))
            label_tensor = torch.tensor(np.array(labels_batch))

            yield data_tensor, label_tensor


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
            print(f"Processing batch {(i+1) / data_len}")

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
):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if use_gpu:
        if device == "cpu":
            Warning("GPU not available, using CPU instead.")
        elif data_parallel:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print("Using", torch.cuda.device_count(), "GPUs!")
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
            print("Epoch", epoch + 1, "of", num_epochs)

        model.train()

        train_loss, last_train_loss = single_pass(
            model, train_dataset, device, optimizer, criterion
        )

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
        else:
            pass

        # Erfan: As far as I understand, validation loss is used for hyperparameter tuning
        # are we doing that here? Or are we planning to do that later?
        # Answer: Inshallah we'll do hyperparameter turning later.
        # Validation
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_loss, _ = single_pass(model, val_dataset, device, None, criterion)
        else:
            pass

        train_losses.append(train_loss)
        np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))

        if val_dataset is not None:
            val_losses.append(val_loss)
            np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))
        else:
            pass

        if verbose:
            if val_dataset is not None:
                print(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}, Val Loss={val_loss:.18f}"
                )
            else:
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}")
        else:
            pass

    torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}.pth"))
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
):
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if not use_gpu:
        print("Warning: GPU not available, using CPU instead.")

    # Load model parameters if path is provided
    if model_param_path is not None:
        params = torch.load(model_param_path, map_location=device)
        # remove the 'module.' prefix from the keys
        params = {
            k.replace("module.", ""): v for k, v in params.items()
        }
        model.load_state_dict(params, strict=False)
    else:
        pass

    # Check if the output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    if data_parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            print("Warning: Data parallelism not available, using single GPU instead.")
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
        print("Finished loading the model, starting prediction.")

    with torch.no_grad():
        for j, sample_generator in enumerate(test_dataset):

            if verbose:
                print(f"Processing batch {(j+1) / len(test_dataset)}")

            counter = 0
            for X, y in sample_generator:
                if verbose:
                    print(f"Processing sample {(counter+1) / len(sample_generator)}")
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

    all_preds = torch.stack(all_preds, dim=0).cpu().numpy()
    np.save(os.path.join(output_dir, f"{output_name}"), all_preds)


# `:,` is there because we want to keep the batch dimension
def re_im_sep(fields, detach=False):
    shg1 = fields[:, 0:1892] + fields[:, 1892 * 2 + 348 : 1892 * 3 + 348] * 1j
    shg2 = fields[:, 1892 : 1892 * 2] + fields[:, 1892 * 3 + 348 : 1892 * 4 + 348] * 1j
    sfg = (
        fields[:, 1892 * 2 : 1892 * 2 + 348]
        + fields[:, 1892 * 4 + 348 : 1892 * 4 + 2 * 348] * 1j
    )

    if detach:
        shg1 = shg1.detach().numpy()
        shg2 = shg2.detach().numpy()
        sfg = sfg.detach().numpy()
    else:
        pass
    return shg1, shg2, sfg


# `:,` is there because we want to keep the batch dimension
def re_im_sep_vectors(fields, detach=False):
    shg1_real = fields[:, 0:1892]
    shg1_complex = fields[:, 1892 * 2 + 348 : 1892 * 3 + 348]

    shg2_real = fields[:, 1892 : 1892 * 2]
    shg2_complex = fields[:, 1892 * 3 + 348 : 1892 * 4 + 348]
    sfg_real = fields[:, 1892 * 2 : 1892 * 2 + 348]
    sfg_complex = fields[:, 1892 * 4 + 348 : 1892 * 4 + 2 * 348]

    if detach:
        shg1_real = shg1_real.detach().numpy()
        shg1_complex = shg1_complex.detach().numpy()
        shg2_real = shg2_real.detach().numpy()
        shg2_complex = shg2_complex.detach().numpy()
        sfg_real = sfg_real.detach().numpy()
        sfg_complex = sfg_complex.detach().numpy()
    else:
        pass
    return shg1_real, shg1_complex, shg2_real, shg2_complex, sfg_real, sfg_complex


# This is a custom loss function that gives different weights
# to the different parts of the signal
def weighted_MSE(y_pred, y_real, shg1_weight=1, shg2_weight=1, sfg_weight=1):
    (
        shg1_real_pred,
        shg1_complex_pred,
        shg2_real_pred,
        shg2_complex_pred,
        sfg_real_pred,
        sfg_complex_pred,
    ) = re_im_sep_vectors(y_pred)

    (
        shg1_real_real,
        shg1_complex_real,
        shg2_real_real,
        shg2_complex_real,
        sfg_real_real,
        sfg_complex_real,
    ) = re_im_sep_vectors(y_real)

    mse = nn.MSELoss()

    shg1_loss = 0.5 * (
        mse(shg1_real_pred, shg1_real_real) + mse(shg1_complex_pred, shg1_complex_real)
    )
    shg2_loss = 0.5 * (
        mse(shg2_real_pred, shg2_real_real) + mse(shg2_complex_pred, shg2_complex_real)
    )

    sfg_loss = 0.5 * (
        mse(sfg_real_pred, sfg_real_real) + mse(sfg_complex_pred, sfg_complex_real)
    )

    return shg1_weight * shg1_loss + shg2_weight * shg2_loss + sfg_weight * sfg_loss


def pearson_corr(y_pred, y_real, shg1_weight=1, shg2_weight=1, sfg_weight=1):
    (
        shg1_real_pred,
        shg1_complex_pred,
        shg2_real_pred,
        shg2_complex_pred,
        sfg_real_pred,
        sfg_complex_pred,
    ) = re_im_sep_vectors(y_pred)

    (
        shg1_real_real,
        shg1_complex_real,
        shg2_real_real,
        shg2_complex_real,
        sfg_real_real,
        sfg_complex_real,
    ) = re_im_sep_vectors(y_real)

    # Calculate the Pearson correlation coefficient for each signal
    shape = shg1_real_pred.shape
    shg1_coeffs = torch.zeros(shape)
    shg2_coeffs = torch.zeros(shape)
    sfg_coeffs = torch.zeros(shape)

    pearson_fn = PearsonCorrCoef().to(y_pred.get_device())

    for i in range(shape[0]):
        shg1_coeffs[i] = 0.5 * (
            pearson_fn(shg1_real_pred[i], shg1_real_real[i])
            + pearson_fn(shg1_complex_pred[i], shg1_complex_real[i])
        )
        shg2_coeffs[i] = 0.5 * (
            pearson_fn(shg2_real_pred[i], shg2_real_real[i])
            + pearson_fn(shg2_complex_pred[i], shg2_complex_real[i])
        )
        sfg_coeffs[i] = 0.5 * (
            pearson_fn(sfg_real_pred[i], sfg_real_real[i])
            + pearson_fn(sfg_complex_pred[i], sfg_complex_real[i])
        )
    # Calculate the mean of the coefficients
    shg1_corr = torch.mean(shg1_coeffs)
    shg2_corr = torch.mean(shg2_coeffs)
    sfg_corr = torch.mean(sfg_coeffs)

    return shg1_weight * shg1_corr + shg2_weight * shg2_corr + sfg_weight * sfg_corr


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
