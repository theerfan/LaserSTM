from torchmetrics.regression import PearsonCorrCoef
import torch
import torch.nn as nn
import numpy as np
import os

from Analysis.analyze_reim import do_analysis
from Analysis.util import get_intensity
import matplotlib.pyplot as plt

freq_vectors_shg = np.load("Data/shg_freq_domain_ds.npy")
freq_vectors_sfg = np.load("Data/sfg_freq_domain_ds.npy")

# TODO: Make sure the scalings are correct (ask Jack)
domain_spacing_shg = (
    freq_vectors_shg[1] - freq_vectors_shg[0]
) * 1e-12  # scaled to be back in Hz
domain_spacing_sfg = (freq_vectors_sfg[1] - freq_vectors_sfg[0]) * 1e-12


def get_intensity(field: torch.Tensor) -> float:
    """
    Returns the intensity of a field
    """
    return torch.abs(field) ** 2


def calc_energy_expanded(
    field: torch.Tensor, domain_spacing: torch.Tensor, spot_area: float
) -> float:
    return torch.sum(get_intensity(field)) * domain_spacing * spot_area


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
    **kwargs,
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


# Convert from a + bi to A * exp(i * theta) to get the energy from the amplitude
def pseudo_energy_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    shg_spacing: float = domain_spacing_shg,
    sfg_spacing: float = domain_spacing_sfg,
    shg_weight: float = 1,
    sfg_weight: float = 1,
    **kwargs,
):
    combined_pred_shg1, combined_pred_shg2, combined_pred_sfg = re_im_combined(y_pred)
    combined_true_shg1, combined_true_shg2, combined_true_sfg = re_im_combined(y_true)

    pred_shg1_energy = calc_energy_expanded(combined_pred_shg1, shg_spacing, 1)
    pred_shg2_energy = calc_energy_expanded(combined_pred_shg2, shg_spacing, 1)
    pred_sfg_energy = calc_energy_expanded(combined_pred_sfg, sfg_spacing, 1)

    true_shg1_energy = calc_energy_expanded(combined_true_shg1, shg_spacing, 1)
    true_shg2_energy = calc_energy_expanded(combined_true_shg2, shg_spacing, 1)
    true_sfg_energy = calc_energy_expanded(combined_true_sfg, sfg_spacing, 1)

    shg1_diff = true_shg1_energy - pred_shg1_energy
    shg2_diff = true_shg2_energy - pred_shg2_energy
    sfg_diff = true_sfg_energy - pred_sfg_energy

    return shg_weight * (shg1_diff + shg2_diff) + sfg_weight * sfg_diff


## Normalized MSE
def normalized_weighted_MSE(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    # if the y_pred has a second dimension of 1, squeeze it
    if y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)

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

    mse = nn.MSELoss(reduction=reduction)

    shg1_pred = torch.cat((shg1_real_pred, shg1_complex_pred), dim=1)
    shg1_real = torch.cat((shg1_real_real, shg1_complex_real), dim=1)

    shg2_pred = torch.cat((shg2_real_pred, shg2_complex_pred), dim=1)
    shg2_real = torch.cat((shg2_real_real, shg2_complex_real), dim=1)

    sfg_pred = torch.cat((sfg_real_pred, sfg_complex_pred), dim=1)
    sfg_real = torch.cat((sfg_real_real, sfg_complex_real), dim=1)

    shg1_numerator = torch.sum(torch.square(shg1_pred - shg1_real), dim=1)
    shg1_denominator = torch.sum(torch.square(shg1_real), dim=1)

    shg2_numerator = torch.sum(torch.square(shg2_pred - shg2_real), dim=1)
    shg2_denominator = torch.sum(torch.square(shg2_real), dim=1)

    sfg_numerator = torch.sum(torch.square(sfg_pred - sfg_real), dim=1)
    sfg_denominator = torch.sum(torch.square(sfg_real), dim=1)

    shg1_loss_mean = torch.mean(torch.sqrt(shg1_numerator / shg1_denominator))
    shg2_loss_mean = torch.mean(torch.sqrt(shg2_numerator / shg2_denominator))
    sfg_loss_mean = torch.mean(torch.sqrt(sfg_numerator / sfg_denominator))

    loss_val = (
        shg_weight * (shg1_loss_mean + shg2_loss_mean) + sfg_weight * sfg_loss_mean
    )

    if loss_val > 1:
        print("Loss value is greater than 1:", loss_val)

    return loss_val


# This is a custom loss function that gives different weights
# to the different parts of the signal
def weighted_MSE(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    # if the y_pred has a second dimension of 1, squeeze it
    if y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)

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

    mse = nn.MSELoss(reduction=reduction)

    shg1_loss = 0.5 * (
        mse(shg1_real_pred, shg1_real_real) + mse(shg1_complex_pred, shg1_complex_real)
    )
    shg1_loss_mean = torch.mean(shg1_loss, dim=-1)

    shg2_loss = 0.5 * (
        mse(shg2_real_pred, shg2_real_real) + mse(shg2_complex_pred, shg2_complex_real)
    )
    shg2_loss_mean = torch.mean(shg2_loss, dim=-1)

    sfg_loss = 0.5 * (
        mse(sfg_real_pred, sfg_real_real) + mse(sfg_complex_pred, sfg_complex_real)
    )
    sfg_loss_mean = torch.mean(sfg_loss, dim=-1)

    loss_val = (
        shg_weight * (shg1_loss_mean + shg2_loss_mean) + sfg_weight * sfg_loss_mean
    )

    if loss_val > 1:
        print("Loss value is greater than 1:", loss_val)

    return loss_val


def pearson_corr(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
    **kwargs,
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


# batch size, time steps, features
# 512, 10, 8264
# (10, 8264) pass through the LSTM
# Changes all (10, 8264) at the same time
# We care about the most recent item for the difference PINN loss function
# We could get creative and use the previous ones to try to calculate a more accurate version of the dA_i/dz


# `:,` is there because we want to keep the batch dimension
def re_im_combined(fields: torch.Tensor, detach=False):
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


def wMSE_and_energy(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    shg_weight: float = 1,
    sfg_weight: float = 1,
    **kwargs,
) -> torch.Tensor:
    return weighted_MSE(
        y_pred, y_real, shg_weight, sfg_weight, **kwargs
    ) + pseudo_energy_loss(
        y_pred, y_real, shg_weight=shg_weight, sfg_weight=sfg_weight, **kwargs
    )


def time_domain_based_MSE_metric(output_dir: str, data_dir: str, model_save_name: str):
    npz_file_path = f"{output_dir}/{model_save_name}_time_domain_MSE_errors.npz"

    # First, attempt to load the pre-calculated MSE values
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path)
        SFG_MSE_errors = data["SFG_MSE_errors"]
        SHG1_MSE_errors = data["SHG1_MSE_errors"]
        SHG2_MSE_errors = data["SHG2_MSE_errors"]
    else:
        # Data loading or MSE metrics calculation code will go here
        SFG_MSE_errors, SHG1_MSE_errors, SHG2_MSE_errors = calculate_MSE_metrics(
            data_dir, model_save_name, output_dir
        )

        # Convert the list of errors to a numpy array
        SFG_MSE_errors = np.array(SFG_MSE_errors)
        SHG1_MSE_errors = np.array(SHG1_MSE_errors)
        SHG2_MSE_errors = np.array(SHG2_MSE_errors)

        # Save the errors to a file inside the output directory with the model name
        np.savez_compressed(
            npz_file_path,
            SFG_MSE_errors=SFG_MSE_errors,
            SHG1_MSE_errors=SHG1_MSE_errors,
            SHG2_MSE_errors=SHG2_MSE_errors,
        )

    # Proceed to visualize the results
    visualize_MSE_errors(
        SFG_MSE_errors, SHG1_MSE_errors, SHG2_MSE_errors, model_save_name
    )


def calculate_MSE_metrics(data_dir, model_save_name, output_dir):
    # Assumed imports and preceding for-loop for the original code's MSE metric load or calculations
    mse = nn.MSELoss(reduction="mean")
    SFG_MSE_errors, SHG1_MSE_errors, SHG2_MSE_errors = [], [], []

    for file_idx in range(91, 100):
        for example_idx in range(0, 100):
            print(f"File: {file_idx}, Example: {example_idx}")
            (
                sfg_time_true,
                sfg_time_pred,
                shg1_time_true,
                shg1_time_pred,
                shg2_time_true,
                shg2_time_pred,
            ) = do_analysis(
                output_dir,
                data_dir,
                model_save_name,
                file_idx,
                example_idx,
                return_vals=True,
            )

            sfg_time_true = torch.tensor(sfg_time_true)
            sfg_time_pred = torch.tensor(sfg_time_pred)
            shg1_time_true = torch.tensor(shg1_time_true)
            shg1_time_pred = torch.tensor(shg1_time_pred)
            shg2_time_true = torch.tensor(shg2_time_true)
            shg2_time_pred = torch.tensor(shg2_time_pred)

            sfg_time_true_intensity = get_intensity(sfg_time_true)
            sfg_time_pred_intensity = get_intensity(sfg_time_pred)
            shg1_time_true_intensity = get_intensity(shg1_time_true)
            shg1_time_pred_intensity = get_intensity(shg1_time_pred)
            shg2_time_true_intensity = get_intensity(shg2_time_true)
            shg2_time_pred_intensity = get_intensity(shg2_time_pred)

            SFG_MSE_errors.append(mse(sfg_time_true_intensity, sfg_time_pred_intensity))
            SHG1_MSE_errors.append(
                mse(shg1_time_true_intensity, shg1_time_pred_intensity)
            )
            SHG2_MSE_errors.append(
                mse(shg2_time_true_intensity, shg2_time_pred_intensity)
            )

    # convert the list of errors to a numpy array
    SFG_MSE_errors = np.array(SFG_MSE_errors)
    SHG1_MSE_errors = np.array(SHG1_MSE_errors)
    SHG2_MSE_errors = np.array(SHG2_MSE_errors)

    return SFG_MSE_errors, SHG1_MSE_errors, SHG2_MSE_errors


def visualize_MSE_errors(
    SFG_MSE_errors, SHG1_MSE_errors, SHG2_MSE_errors, model_save_name
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    datasets = [
        (SFG_MSE_errors, "SFG Intensity MSE Errors"),
        (SHG1_MSE_errors, "SHG1 Intensity MSE Errors"),
        (SHG2_MSE_errors, "SHG2 Intensity MSE Errors"),
    ]
    for ax, (data, title) in zip(axs, datasets):
        ax.hist(data, bins=20)
        ax.set_title(title)
        mean = np.mean(data)
        std = np.std(data)
        # Text formatting in scientific notation
        text_str = f"Mean: {mean:.3e}\nStd: {std:.3e}"
        ax.text(
            0.95,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"{model_save_name}_time_domain_MSE_errors.png")
