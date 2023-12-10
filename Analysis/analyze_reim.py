import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from Analysis.util import (
    get_intensity,
    get_phase,
    re_im_sep,
    change_domain_and_adjust_energy,
)


def save_figure(save_name, save_format, save_dir):
    if save_name is None:
        raise ValueError("Save name is not specified")
    if "." + save_format not in save_name:
        save_name += "." + save_format
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(
        os.path.join(save_dir, save_name),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
        format=save_format,
    )


def intensity_phase_plot(
    domains,
    fields,
    labels,
    colors,
    domain_type,
    xlims=None,
    y_label=None,
    normalize=False,
    offsets=None,
    save_format="jpg",
    save_name=None,
    plot_show=True,
    plot_hold=False,
    save_dir="",
):
    """
    Plot intensity and phase of a field
    """
    if domain_type == "time":
        factor = 1e12
        x_label = "time (ps)"
    elif domain_type == "wavelength":
        factor = 1e9
        x_label = "wavelength (nm)"
    elif domain_type == "frequency" or domain_type == "freq":
        factor = 1e-12
        x_label = "frequency (THz)"

    # NOTE: Modifying the domains itself will change the original domains in the calling scope!
    # Which was causing our problems
    factored_domains = [domain * factor for domain in domains]

    intensities = [get_intensity(field) for field in fields]
    phases = [np.unwrap(get_phase(field)) for field in fields]

    y_label_2 = "Phase (rad)"
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), clear=False)

    axs.set_xlabel(x_label)

    for i in range(len(intensities)):
        if normalize:
            intensity = intensities[i] / np.max(intensities[i])
            y_label_1 = "Norm. Intensity (a.u.)"
        else:
            y_label_1 = "Fluence (J/m^2)"
            # warnings.warn("Using default intensity units (J/m^2)")
            intensity = intensities[i]

        if y_label is None:
            y_label_1 = y_label_1
        else:
            y_label_1 = y_label
        if offsets is not None:
            offset = offsets[i]
        axs.plot(
            factored_domains[i],
            intensity + offset,
            color=colors[i],
            label=labels[i],
            alpha=0.6,
        )
    plt.legend()

    if xlims is not None:
        axs.set_xlim(xlims[0], xlims[1])

    axs.set_ylabel(y_label_1, color="black")
    axs2 = axs.twinx()

    for i in range(len(intensities)):
        axs2.plot(
            factored_domains[i],
            phases[i],
            color=colors[i],
            linestyle="dashed",
            alpha=0.6,
        )

    axs2.set_ylabel(y_label_2, color="black")

    save_figure(save_name, save_format, save_dir)

    if plot_show:
        plt.show()
    else:
        if ~plot_hold:
            plt.close()


def do_analysis(
    output_dir: str,  # directory from model training
    data_directory: str,  # directory from preprocessing
    model_save_name: str,  # model name from training
    file_idx: int,  # on which file to do analysis
    item_idx: int,  # which example of the file to do analysis
    fig_save_dir: str = None,  # where to save the figures
    crystal_length: int = 100,  # length of the crystal
):
    if fig_save_dir is None:
        fig_save_dir = os.path.join(
            model_save_name, f"figures_file{file_idx}_item{item_idx}"
        )

    # Loading files for scaling
    with open(
        os.path.join(data_directory, "scaler.pkl"), "rb"
    ) as file:  # can use scaler.pkl or scaler_bckkup.pkl
        scaler = pickle.load(file)

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

    # Loading the single file out of the test dataset
    y_true = np.load(
        os.path.join(data_directory, f"y_new_{file_idx}.npy")
    )  # these are used to compare to the predictions

    y_true = scaler.inverse_transform(y_true)

    sfg_original_freq = np.load("Data/sfg_original_freq_vector.npy")
    sfg_original_time = np.load("Data/sfg_original_time_vector.npy")
    sfg_original_time_ds = sfg_original_time[1] - sfg_original_time[0]

    ### The part where we load the predictions

    # the output file from the "predict" function
    # this has a shape of (files, predictions, channels)
    all_preds = np.load(os.path.join(output_dir, f"{model_save_name}_all_preds.npy"))

    # Transform this using the scaler, then get which file
    # and then get the example using its index
    y_pred_trans = scaler.inverse_transform(all_preds[file_idx])[item_idx]
    # y_pred_trans = np.array([scaler.inverse_transform(item) for item in all_preds])[file_idx][item_idx]

    ###

    # Get the transformed value of the real item in the dataset
    y_true_trans = y_true[crystal_length - 1 :][::crystal_length][item_idx]

    y_pred_trans_shg1, y_pred_trans_shg2, y_pred_trans_sfg = re_im_sep(y_pred_trans)
    y_true_trans_shg1, y_true_trans_shg2, y_true_trans_sfg = re_im_sep(y_true_trans)

    (
        sfg_freq_to_time_direct_pred,
        sfg_freq_to_time_pred,
    ) = change_domain_and_adjust_energy(
        freq_vectors_sfg,
        y_pred_trans_sfg,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_3,
        true_domain_spacing=sfg_original_time_ds,
    )
    (
        sfg_freq_to_time_direct_true,
        sfg_freq_to_time_true,
    ) = change_domain_and_adjust_energy(
        freq_vectors_sfg,
        y_true_trans_sfg,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_3,
        true_domain_spacing=sfg_original_time_ds,
    )

    (
        shg1_freq_to_time_direct_pred,
        shg1_freq_to_time_pred,
    ) = change_domain_and_adjust_energy(
        freq_vectors_shg1,
        y_pred_trans_shg1,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_1,
        true_domain_spacing=sfg_original_time_ds,
    )
    (
        shg1_freq_to_time_direct_true,
        shg1_freq_to_time_true,
    ) = change_domain_and_adjust_energy(
        freq_vectors_shg1,
        y_true_trans_shg1,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_1,
        true_domain_spacing=sfg_original_time_ds,
    )

    (
        shg2_freq_to_time_direct_pred,
        shg2_freq_to_time_pred,
    ) = change_domain_and_adjust_energy(
        freq_vectors_shg2,
        y_pred_trans_shg2,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_2,
        true_domain_spacing=sfg_original_time_ds,
    )
    (
        shg2_freq_to_time_direct_true,
        shg2_freq_to_time_true,
    ) = change_domain_and_adjust_energy(
        freq_vectors_shg2,
        y_true_trans_shg2,
        sfg_original_freq,
        "freq",
        beam_area=factors_freq["beam_area"],
        domain_spacing=domain_spacing_2,
        true_domain_spacing=sfg_original_time_ds,
    )

    freq_vectors_sfg_list = [freq_vectors_sfg, freq_vectors_sfg]
    freq_vectors_shg1_list = [freq_vectors_shg1, freq_vectors_shg1]
    freq_vectors_shg2_list = [freq_vectors_shg2, freq_vectors_shg2]

    fields_sfg_list = [y_true_trans_sfg, y_pred_trans_sfg]
    fields_shg1_list = [y_true_trans_shg1, y_pred_trans_shg1]
    fields_shg2_list = [y_true_trans_shg2, y_pred_trans_shg2]

    colors_list = ["red", "black"]
    labels_list = ["true", "pred"]

    # plots frequency domain for all three fields (prediction vs true) normalized (first three) and non-normalized (next three)
    print("------- Normalized True vs Prediction Frequency Domain --------")

    print("*** SFG ***")
    intensity_phase_plot(
        freq_vectors_sfg_list,
        fields_sfg_list,
        labels_list,
        colors_list,
        "freq",
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_pfg1.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    print("*** SHG1 ***")
    intensity_phase_plot(
        freq_vectors_shg1_list,
        fields_shg1_list,
        labels_list,
        colors_list,
        "freq",
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_pfg2.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    print("*** SHG2 ***")
    intensity_phase_plot(
        freq_vectors_shg2_list,
        fields_shg2_list,
        labels_list,
        colors_list,
        "freq",
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_pfg3.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    print("------- Non-normalized True vs Prediction Frequency Domain --------")
    print("*** SFG ***")
    intensity_phase_plot(
        freq_vectors_sfg_list,
        fields_sfg_list,
        labels_list,
        colors_list,
        "freq",
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg4.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
    print("*** SHG1 ***")
    intensity_phase_plot(
        freq_vectors_shg1_list,
        fields_shg1_list,
        labels_list,
        colors_list,
        "freq",
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg5.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
    print("*** SHG2 ***")
    intensity_phase_plot(
        freq_vectors_shg2_list,
        fields_shg2_list,
        labels_list,
        colors_list,
        "freq",
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg6.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    sfg_time_vector_list = [sfg_original_time, sfg_original_time]
    shg1_time_vector_list = [sfg_original_time, sfg_original_time]
    shg2_time_vector_list = [sfg_original_time, sfg_original_time]

    sfg_freq_to_time_list = [sfg_freq_to_time_true, sfg_freq_to_time_pred]
    shg1_freq_to_time_list = [shg1_freq_to_time_true, shg1_freq_to_time_pred]
    shg2_freq_to_time_list = [shg2_freq_to_time_true, shg2_freq_to_time_pred]

    # plots time domain for all three fields (prediction vs true) normalized (first three) and non-normalized (next three)
    print("------- Normalized True vs Prediction Time Domain --------")
    print("*** SFG ***")
    intensity_phase_plot(
        sfg_time_vector_list,
        sfg_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        xlims=[-15, 15],
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_ptd1.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    print("*** SHG1 ***")
    intensity_phase_plot(
        shg1_time_vector_list,
        shg1_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_ptd2.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
    print("*** SHG2 ***")
    intensity_phase_plot(
        shg2_time_vector_list,
        shg2_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=True,
        offsets=[0, 0.2],
        save_format="jpg",
        save_name=model_save_name + "_ptd3.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )

    print("------- Non-normalized True vs Prediction Frequency Domain --------")
    print("*** SFG ***")
    intensity_phase_plot(
        sfg_time_vector_list,
        sfg_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        xlims=[-15, 15],
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd4.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
    print("*** SHG1 ***")
    intensity_phase_plot(
        shg1_time_vector_list,
        shg1_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd5.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
    print("*** SHG2 ***")
    intensity_phase_plot(
        shg2_time_vector_list,
        shg2_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=False,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd6.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
    )
