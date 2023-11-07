import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchmetrics.regression import PearsonCorrCoef

from typing import Iterable

freq_vectors_shg = np.load("Data/shg_freq_domain_ds.npy")
freq_vectors_sfg = np.load("Data/sfg_freq_domain_ds.npy")

# TODO: Make sure the scalings are correct (ask Jack)
domain_spacing_shg = (
    freq_vectors_shg[1] - freq_vectors_shg[0]
)  # * 1e12  # scaled to be back in Hz
domain_spacing_sfg = freq_vectors_sfg[1] - freq_vectors_sfg[0]  # * 1e12


# Adding prefixes to file names we're trying to iterate
# because we have multiple versions of the data
def add_prefix(lst: list, prefix="X"):
    return [prefix + "_" + str(i) + ".npy" for i in lst]


class CustomSequence(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        file_indexes: Iterable,
        test_mode: bool = False,
        train_prefix: str = "X_new",
        val_prefix: str = "y_new",
        crystal_length: int = 100,
    ):
        self.Xnames = add_prefix(file_indexes, train_prefix)
        self.ynames = add_prefix(file_indexes, val_prefix)
        self.file_indexes = file_indexes
        self.test_mode = test_mode
        self.data_dir = data_dir
        self.crystal_length = crystal_length
        self.current_file_idx = None
        self.current_data = None
        self.current_labels = None

    # NOTE: We always assume that we have 10000 examples per file.
    def get_num_samples_per_file(self):
        return 10_000

    def load_data_for_file_index(self, file_idx):
        if file_idx != self.current_file_idx:
            # Clear memory of previously loaded data
            self.current_data = None
            self.current_labels = None
            # Load new data
            self.current_data = np.load(
                os.path.join(self.data_dir, self.Xnames[file_idx])
            )
            self.current_labels = np.load(
                os.path.join(self.data_dir, self.ynames[file_idx])
            )
            self.current_file_idx = file_idx
        else:
            pass

    def __len__(self):
        # Assuming every file has the same number of samples, otherwise you need a more dynamic way
        return len(self.Xnames) * self.get_num_samples_per_file()

    def __getitem__(self, idx):
        # Compute file index and sample index within that file
        num_samples_per_file = self.get_num_samples_per_file()
        file_idx = idx // num_samples_per_file
        sample_idx = idx % num_samples_per_file

         # Load data if not already in memory or if file index has changed
        self.load_data_for_file_index(file_idx)

       # Get the specific sample from the currently loaded data
        data = self.current_data[sample_idx]
        labels = self.current_labels[sample_idx]

        # In test mode, we only care about the first thing that goes
        # into the crystal and the thing that comes out. (All steps of crystal at once)
        if self.test_mode:
            data = data[:: self.crystal_length]
            labels = labels[self.crystal_length - 1::][:: self.crystal_length]

        # Each file is about 3 GB, just move directly into GPU memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

        return data_tensor, labels_tensor


def area_under_curve_diff(
    real_pred: torch.Tensor,
    complex_pred: torch.Tensor,
    real_real: torch.Tensor,
    complex_real: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    pred_magnitude = torch.trapezoid(real_pred, dx=spacing, dim=-1) + torch.trapezoid(
        complex_pred, dx=spacing, dim=-1
    )
    real_magnitude = torch.trapezoid(real_real, dx=spacing, dim=-1) + torch.trapezoid(
        complex_real, dx=spacing, dim=-1
    )
    return 0.5 * (pred_magnitude - real_magnitude) ** 2


# Erfan: Total sum of areas under the curve for both real and predicted fields
# Could also add it to the loss function
# Could split it into the 6 different sections, get the total area under the curve of each
# and give different "importance" to each section

# Here the SFG is should be more much important than the SHG (actual values of them really differ)
# Min-Max scaling (hopefully no bad spikes)

# TODO: Also do energy using L2 norm of the complex values (different version)


def area_under_curve_loss(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_spacing: float = domain_spacing_shg,
    sfg_spacing: float = domain_spacing_sfg,
    shg_weight: float = 1,
    sfg_weight: float = 1,
):
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

    # Calculate the area under the curve for each signal
    shg1_auc = area_under_curve_diff(
        shg1_real_pred,
        shg1_complex_pred,
        shg1_real_real,
        shg1_complex_real,
        shg_spacing,
    )
    shg2_auc = area_under_curve_diff(
        shg2_real_pred,
        shg2_complex_pred,
        shg2_real_real,
        shg2_complex_real,
        shg_spacing,
    )
    sfg_auc = area_under_curve_diff(
        sfg_real_pred, sfg_complex_pred, sfg_real_real, sfg_complex_real, sfg_spacing
    )

    # Calculate the mean of the coefficients
    shg1_auc = torch.mean(shg1_auc)
    shg2_auc = torch.mean(shg2_auc)
    sfg_auc = torch.mean(sfg_auc)

    return shg_weight * (shg1_auc + shg2_auc) + sfg_weight * sfg_auc


# Function to calculate pseudo-energy for a given set of tensors
def calculate_pseudo_energy_diff(
    real_pred: torch.Tensor,
    complex_pred: torch.Tensor,
    real_real: torch.Tensor,
    complex_real: torch.Tensor,
    spacing: float,
):
    pred_magnitude = torch.sum(torch.square(real_pred), dim=-1) + torch.sum(
        torch.square(complex_pred), dim=-1
    )

    real_magnitude = torch.sum(torch.square(real_real), dim=-1) + torch.sum(
        torch.square(complex_real), dim=-1
    )

    return 0.5 * (pred_magnitude - real_magnitude) ** 2 * spacing


# Convert from a + bi to A * exp(i * theta) to get the energy from the amplitude
def pseudo_energy_loss(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_spacing: float = domain_spacing_shg,
    sfg_spacing: float = domain_spacing_sfg,
    shg_weight: float = 1,
    sfg_weight: float = 1,
):
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

    # Calculate energies
    shg1_energy_diff = calculate_pseudo_energy_diff(
        shg1_real_pred,
        shg1_complex_pred,
        shg1_real_real,
        shg1_complex_real,
        shg_spacing,
    )
    shg2_energy_diff = calculate_pseudo_energy_diff(
        shg2_real_pred,
        shg2_complex_pred,
        shg2_real_real,
        shg2_complex_real,
        shg_spacing,
    )
    sfg_energy_diff = calculate_pseudo_energy_diff(
        sfg_real_pred, sfg_complex_pred, sfg_real_real, sfg_complex_real, sfg_spacing
    )

    # Calculate the mean of the energy differences in the batches
    shg1_energy_diff = torch.mean(shg1_energy_diff)
    shg2_energy_diff = torch.mean(shg2_energy_diff)
    sfg_energy_diff = torch.mean(sfg_energy_diff)

    return (
        shg_weight * (shg1_energy_diff + shg2_energy_diff)
        + sfg_weight * sfg_energy_diff
    )


# batch size, time steps, features
# 512, 10, 8264
# (10, 8264) pass through the LSTM
# Changes all (10, 8264) at the same time
# We care about the most recent item for the difference PINN loss function
# We could get creative and use the previous ones to try to calculate a more accurate version of the dA_i/dz


# `:,` is there because we want to keep the batch dimension
def re_im_sep(fields: torch.Tensor, detach=False):
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
def re_im_sep_vectors(fields: torch.Tensor, detach=False):
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


# This is a wrapper for the MSE and BCE losses just to make them have the same
# signature as the custom loss functions
def wrapped_MSE(y_pred: torch.Tensor, y_real: torch.Tensor, **kwargs) -> torch.Tensor:
    return nn.MSELoss()(y_pred, y_real)


def wrapped_BCE(y_pred: torch.Tensor, y_real: torch.Tensor, **kwargs) -> torch.Tensor:
    return nn.BCELoss()(y_pred, y_real)


# This is a custom loss function that gives different weights
# to the different parts of the signal
def weighted_MSE(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
) -> torch.Tensor:
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

    return shg_weight * (shg1_loss + shg2_loss) + sfg_weight * sfg_loss


def pearson_corr(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
):
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

    return (shg_weight * shg1_corr + shg2_corr) + sfg_weight * sfg_corr
