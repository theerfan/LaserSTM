import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchmetrics.regression import PearsonCorrCoef

freq_vectors_shg = np.load("../Data/shg_freq_domain_ds.npy")
freq_vectors_sfg = np.load("../Data/sfg_freq_domain_ds.npy")

domain_spacing_shg = (
    freq_vectors_shg[1] - freq_vectors_shg[0]
) * 1e12  # scaled to be back in Hz
domain_spacing_sfg = (freq_vectors_sfg[1] - freq_vectors_sfg[0]) * 1e12


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


# Erfan: Total sum of areas under the curve for both real and predicted fields
# Could also add it to the loss function
# Could split it into the 6 different sections, get the total area under the curve of each
# and give different "importance" to each section

# Here the SFG is should be more much important than the SHG (actual values of them really differ)
# Min-Max scaling (hopefully no bad spikes)

# TODO: Also do energy using L2 norm of the complex values (different version)


def total_area_under_curve(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_spacing: float = domain_spacing_shg,
    sfg_spacing: float = domain_spacing_sfg,
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
    shape = shg1_real_pred.shape
    shg1_auc = torch.zeros(shape)
    shg2_auc = torch.zeros(shape)
    sfg_auc = torch.zeros(shape)

    for i in range(shape[0]):
        shg1_auc[i] = 0.5 * (
            torch.trapezoid(shg1_real_pred[i], dx=shg_spacing)
            - torch.trapezoid(shg1_real_real[i], dx=shg_spacing)
            + torch.trapezoid(shg1_complex_pred[i], dx=shg_spacing)
            - torch.trapezoid(shg1_complex_real[i], dx=shg_spacing)
        )

        shg2_auc[i] = 0.5 * (
            torch.trapezoid(shg2_real_pred[i], dx=shg_spacing)
            - torch.trapezoid(shg2_real_real[i], dx=shg_spacing)
            + torch.trapezoid(shg2_complex_pred[i], dx=shg_spacing)
            - torch.trapezoid(shg2_complex_real[i], dx=shg_spacing)
        )

        sfg_auc[i] = 0.5 * (
            torch.trapezoid(sfg_real_pred[i], dx=sfg_spacing)
            - torch.trapezoid(sfg_real_real[i], dx=sfg_spacing)
            + torch.trapezoid(sfg_complex_pred[i], dx=sfg_spacing)
            - torch.trapezoid(sfg_complex_real[i], dx=sfg_spacing)
        )

    # Calculate the mean of the coefficients
    shg1_auc = torch.mean(shg1_auc)
    shg2_auc = torch.mean(shg2_auc)
    sfg_auc = torch.mean(sfg_auc)

    return shg1_auc + shg2_auc + sfg_auc


# Convert from a + bi to A * exp(i * theta) to get the energy from the amplitude
def total_energy(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_spacing: float = domain_spacing_shg,
    sfg_spacing: float = domain_spacing_sfg,
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
    shape = shg1_real_pred.shape
    shg1_energy = torch.zeros(shape)
    shg2_energy = torch.zeros(shape)
    sfg_energy = torch.zeros(shape)

    for i in range(shape[0]):
        # Convert from a + bi to r * exp(i * theta)
        shg1_energy[i] = (
            0.5
            * (
                torch.sqrt(
                    torch.sum(torch.square(shg1_real_pred[i]))
                    + torch.sum(torch.square(shg1_complex_pred[i]))
                )
                - torch.sqrt(
                    torch.sum(torch.square(shg1_real_real[i]))
                    + torch.sum(torch.square(shg1_complex_real[i]))
                )
            )
            * shg_spacing
        )

        shg2_energy[i] = (
            0.5
            * (
                torch.sqrt(
                    torch.sum(torch.square(shg2_real_pred[i]))
                    + torch.sum(torch.square(shg2_complex_pred[i]))
                )
                - torch.sqrt(
                    torch.sum(torch.square(shg2_real_real[i]))
                    + torch.sum(torch.square(shg2_complex_real[i]))
                )
            )
            * shg_spacing
        )

        sfg_energy[i] = (
            0.5
            * (
                torch.sqrt(
                    torch.sum(torch.square(sfg_real_pred[i]))
                    + torch.sum(torch.square(sfg_complex_pred[i]))
                )
                - torch.sqrt(
                    torch.sum(torch.square(sfg_real_real[i]))
                    + torch.sum(torch.square(sfg_complex_real[i]))
                )
            )
            * sfg_spacing
        )

    # Calculate the mean of the coefficients
    shg1_energy = torch.mean(shg1_energy)
    shg2_energy = torch.mean(shg2_energy)
    sfg_energy = torch.mean(sfg_energy)

    return shg1_energy + shg2_energy + sfg_energy


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


# This is a custom loss function that gives different weights
# to the different parts of the signal
def weighted_MSE(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg1_weight: float = 1,
    shg2_weight: float = 1,
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

    return shg1_weight * shg1_loss + shg2_weight * shg2_loss + sfg_weight * sfg_loss


def pearson_corr(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg1_weight: float = 1,
    shg2_weight: float = 1,
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

    return shg1_weight * shg1_corr + shg2_weight * shg2_corr + sfg_weight * sfg_corr