import numpy as np
import pickle
import os
import matplotlib

import matplotlib.pyplot as plt
from Analysis.util import (
    get_intensity,
    get_phase,
    re_im_sep,
    change_domain_and_adjust_energy
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def intensity_phase_plot(
    domains,
    fields,
    labels,
    colors,
    domain_type,
    xlims=None,
    ylabel=None,
    normalize=False,
    legend=False,
    offsets=None,
    save_format="pdf",
    save_name=None,
    save=True,
    plot_show=True,
    plot_hold=False,
):
    """
    Plot intensity and phase of a field
    """
    if domain_type == "time":
        factor = 1e12
        xlabel = "time (ps)"
    elif domain_type == "wavelength":
        factor = 1e9
        xlabel = "wavelength (nm)"
    elif domain_type == "frequency" or domain_type == "freq":
        factor = 1e-12
        xlabel = "frequency (THz)"
    for ii in range(len(domains)):
        domains[ii] = domains[ii] * factor

    intensities = [get_intensity(field) for field in fields]
    phases = [np.unwrap(get_phase(field)) for field in fields]

    ylabel2 = "Phase (rad)"
    # TODO: get clear working so that can call function multiple times and have all plots appear
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), clear=False)

    axs.set_xlabel(xlabel)

    for ii in range(len(intensities)):
        if normalize:
            intensity = intensities[ii] / np.max(intensities[ii])
            ylabel1 = "Norm. Intensity (a.u.)"
        else:
            ylabel1 = "Fluence (J/m^2)"
            #             warnings.warn("Using default intensity units (J/m^2)")
            intensity = intensities[ii]

        if ylabel is None:
            ylabel1 = ylabel1
        else:
            ylabel1 = ylabel
        if offsets is not None:
            offset = offsets[ii]
        axs.plot(
            domains[ii],
            intensity + offset,
            color=colors[ii],
            label=labels[ii],
            alpha=0.6,
        )
    if legend:
        plt.legend()

    if xlims is not None:
        axs.set_xlim(xlims[0], xlims[1])

    axs.set_ylabel(ylabel1, color="black")
    ax2 = axs.twinx()

    for ii in range(len(intensities)):
        ax2.plot(
            domains[ii], phases[ii], color=colors[ii], linestyle="dashed", alpha=0.6
        )

    ax2.set_ylabel(ylabel2, color="black")

    if save:
        if save_name is None:
            raise ValueError("Save name is not specified")
        if "." + save_format not in save_name:
            save_name += "." + save_format
        plt.savefig(
            save_name,
            bbox_inches="tight",
            dpi=300,
            transparent=True,
            format=save_format,
        )
        
    if plot_show:
        plt.show()
    else:
        if ~plot_hold:
            plt.close()


saved_model_output_dir = "/home/ubuntu/oneterra/outputs/10-26-2023"  # directory from model training
data_directory = "/home/ubuntu/oneterra/SFG_reIm_version1"  # directory from preprocessing

model_name = "model-0.9-0.1_epoch_10"  # model name from training

train_losses = np.load(os.path.join(saved_model_output_dir, f"{model_name}_train_losses.npy"))
val_losses = np.load(os.path.join(saved_model_output_dir, f"{model_name}_val_losses.npy"))

with open(
    os.path.join(data_directory, "scaler.pkl"), "rb"
) as file:  # can use scaler.pkl or scaler_bckkup.pkl
    scaler = pickle.load(file)

all_preds = np.load(os.path.join(saved_model_output_dir, f"{model_name}_all_preds.npy"))
all_preds_trans = np.zeros(all_preds.shape)
for ii in range(all_preds.shape[0]):
    all_preds_trans[ii] = scaler.inverse_transform(all_preds[ii])
# all_preds_trans = np.load(
#     os.path.join(saved_model_output_dir, "all_preds_transformed.npy")
# )
 
freq_vectors_shg1 = np.load("Data/shg_freq_domain_ds.npy")
freq_vectors_shg2 = freq_vectors_shg1  # these are equivalent here
freq_vectors_sfg = np.load("Data/sfg_freq_domain_ds.npy")

domain_spacing_1 = (
    freq_vectors_shg1[1] - freq_vectors_shg1[0]
) * 1e12  # scaled to be back in Hz
domain_spacing_2 = (freq_vectors_shg2[1] - freq_vectors_shg2[0]) * 1e12
domain_spacing_3 = (freq_vectors_sfg[1] - freq_vectors_sfg[0]) * 1e12

factors_freq = {
    "beam_area": 400e-6**2 * np.pi,
    "grid_spacing": [domain_spacing_1, domain_spacing_2, domain_spacing_3],
    "domain_spacing_1": domain_spacing_1,
    "domain_spacing_2": domain_spacing_2,
    "domain_spacing_3": domain_spacing_3,
}  # beam radius 400 um (and circular beam)


y_90 = np.load(
    os.path.join(data_directory, "y_new_90.npy")
)  # these are used to compare to the predictions


y_90_trans = scaler.inverse_transform(y_90)

sfg_original_freq = np.load("Data/sfg_original_freq_vector.npy")
sfg_original_time = np.load("Data/sfg_original_time_vector.npy")
sfg_original_time_ds = sfg_original_time[1] - sfg_original_time[0]


ii = (
    8  # use this to select one of the examples (should be 100 in total to choose among)
)

all_preds_trans = all_preds_trans[0]

y_pred_trans = all_preds_trans[ii]
y_true_trans = y_90_trans[99:][::100][ii]

y_pred_trans_shg1, y_pred_trans_shg2, y_pred_trans_sfg = re_im_sep(y_pred_trans)
y_true_trans_shg1, y_true_trans_shg2, y_true_trans_sfg = re_im_sep(y_true_trans)


sfg_freq_to_time_direct_pred, sfg_freq_to_time_pred = change_domain_and_adjust_energy(
    freq_vectors_sfg,
    y_pred_trans_sfg,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_3,
    true_domain_spacing=sfg_original_time_ds,
)
sfg_freq_to_time_direct_true, sfg_freq_to_time_true = change_domain_and_adjust_energy(
    freq_vectors_sfg,
    y_true_trans_sfg,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_3,
    true_domain_spacing=sfg_original_time_ds,
)


shg1_freq_to_time_direct_pred, shg1_freq_to_time_pred = change_domain_and_adjust_energy(
    freq_vectors_shg1,
    y_pred_trans_shg1,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_1,
    true_domain_spacing=sfg_original_time_ds,
)
shg1_freq_to_time_direct_true, shg1_freq_to_time_true = change_domain_and_adjust_energy(
    freq_vectors_shg1,
    y_true_trans_shg1,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_1,
    true_domain_spacing=sfg_original_time_ds,
)

shg2_freq_to_time_direct_pred, shg2_freq_to_time_pred = change_domain_and_adjust_energy(
    freq_vectors_shg2,
    y_pred_trans_shg2,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_2,
    true_domain_spacing=sfg_original_time_ds,
)
shg2_freq_to_time_direct_true, shg2_freq_to_time_true = change_domain_and_adjust_energy(
    freq_vectors_shg2,
    y_true_trans_shg2,
    sfg_original_freq,
    "freq",
    beam_area=factors_freq["beam_area"],
    domain_spacing=domain_spacing_2,
    true_domain_spacing=sfg_original_time_ds,
)


# training and validation error
plt.scatter(
    range(1, train_losses.shape[0] + 1, 1), train_losses, label="Train Loss", alpha=0.6
)
plt.scatter(
    range(1, val_losses.shape[0] + 1, 1), val_losses, label="Validation Loss", alpha=0.6
)
plt.title("MSE Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()


# plots frequency domain for all three fields (prediction vs true) normalized (first three) and non-normalized (next three)
print("------- Normalized True vs Prediction Frequency Domain --------")
print("*** SFG ***")

intensity_phase_plot(
    [freq_vectors_sfg, freq_vectors_sfg],
    [y_true_trans_sfg, y_pred_trans_sfg],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=True,
    legend=True,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_pfg1.pdf",
    
    plot_show=True,
    plot_hold=False,
)

print("*** SHG1 ***")
intensity_phase_plot(
    [freq_vectors_shg1, freq_vectors_shg1],
    [y_true_trans_shg1, y_pred_trans_shg1],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=True,
    legend=False,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_pfg2.pdf",
    
    plot_show=True,
    plot_hold=False,
)

print("*** SHG2 ***")
intensity_phase_plot(
    [freq_vectors_shg2, freq_vectors_shg2],
    [y_true_trans_shg2, y_pred_trans_shg2],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=True,
    legend=False,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_pfg3.pdf",
    plot_show=True,
    plot_hold=False,
)

print("------- Non-normalized True vs Prediction Frequency Domain --------")
print("*** SFG ***")
intensity_phase_plot(
    [freq_vectors_sfg, freq_vectors_sfg],
    [y_true_trans_sfg, y_pred_trans_sfg],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=False,
    legend=True,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_pfg4.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG1 ***")
intensity_phase_plot(
    [freq_vectors_shg1, freq_vectors_shg1],
    [y_true_trans_shg1, y_pred_trans_shg1],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=False,
    legend=False,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_pfg5.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG2 ***")
intensity_phase_plot(
    [freq_vectors_shg2, freq_vectors_shg2],
    [y_true_trans_shg2, y_pred_trans_shg2],
    ["true", "pred"],
    ["red", "black"],
    "freq",
    normalize=False,
    legend=False,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_pfg6.pdf",
    plot_show=True,
    plot_hold=False,
)


# plots time domain for all three fields (prediction vs true) normalized (first three) and non-normalized (next three)
print("------- Normalized True vs Prediction Time Domain --------")
print("*** SFG ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [sfg_freq_to_time_true, sfg_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    xlims=[-15, 15],
    normalize=True,
    legend=True,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_ptd1.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG1 ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [shg1_freq_to_time_true, shg1_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    normalize=True,
    legend=False,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_ptd2.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG2 ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [shg2_freq_to_time_true, shg2_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    normalize=True,
    legend=False,
    offsets=[0, 0.2],
    save_format="pdf",
    save_name=model_name + "_ptd3.pdf",
    plot_show=True,
    plot_hold=False,
)

print("------- Non-normalized True vs Prediction Frequency Domain --------")
print("*** SFG ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [sfg_freq_to_time_true, sfg_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    xlims=[-15, 15],
    normalize=False,
    legend=True,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_ptd4.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG1 ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [shg1_freq_to_time_true, shg1_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    normalize=False,
    legend=False,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_ptd5.pdf",
    plot_show=True,
    plot_hold=False,
)
print("*** SHG2 ***")
intensity_phase_plot(
    [sfg_original_time, sfg_original_time],
    [shg2_freq_to_time_true, shg2_freq_to_time_pred],
    ["true", "pred"],
    ["red", "black"],
    "time",
    normalize=False,
    legend=False,
    offsets=[0, 0],
    save_format="pdf",
    save_name=model_name + "_ptd6.pdf",
    plot_show=True,
    plot_hold=False,
)