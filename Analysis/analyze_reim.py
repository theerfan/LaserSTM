import numpy as np
import pickle
import os
import h5py

import matplotlib.pyplot as plt
from Analysis.util import (
    get_intensity,
    get_phase,
    re_im_combined,
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
    save=False,
    axs=None,
    fig=None,
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

    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), clear=False)

    if normalize:
        y_label_1 = "Norm. Intensity (a.u.)"
        intensities = [intensity / np.max(intensity) for intensity in intensities]
    else:
        y_label_1 = "Fluence (J/m^2)"

    axs.set_xlabel(x_label)

    for i, intensity in enumerate(intensities):
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

    axs.legend()

    if xlims is not None:
        axs.set_xlim(xlims[0], xlims[1])

    axs.set_ylabel(y_label_1, color="black")
    axs2 = axs.twinx()

    for i in range(len(phases)):
        axs2.plot(
            factored_domains[i],
            phases[i],
            color=colors[i],
            linestyle="dashed",
            alpha=0.6,
        )

    axs2.set_ylabel(y_label_2, color="black")

    if save:
        save_figure(save_name, save_format, save_dir)

    return fig


def plot_a_bunch_of_fields(
    freq_vectors_sfg_list,
    fields_sfg_list,
    freq_vectors_shg1_list,
    fields_shg1_list,
    freq_vectors_shg2_list,
    fields_shg2_list,
    sfg_time_vector_list,
    sfg_freq_to_time_list,
    shg1_time_vector_list,
    shg1_freq_to_time_list,
    shg2_time_vector_list,
    shg2_freq_to_time_list,
    labels_list,
    colors_list,
    model_save_name,
    fig_save_dir,
    file_save_name=None,
    normalize=False,
):
    nrows = 2
    ncols = 3

    plt.figure()

    new_fig, new_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15))

    # Flatten the array of axes if it's 2D
    if nrows > 1 and ncols > 1:
        new_axs = new_axs.flatten()
    else:  # 1D array of axes
        new_axs = new_axs.reshape(-1)

    print("------- True vs Prediction Frequency Domain --------")
    print("*** SFG ***")
    fig_pfg4 = intensity_phase_plot(
        freq_vectors_sfg_list,
        fields_sfg_list,
        labels_list,
        colors_list,
        "freq",
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg4.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[0],
    )

    print("*** SHG1 ***")
    fig_pfg5 = intensity_phase_plot(
        freq_vectors_shg1_list,
        fields_shg1_list,
        labels_list,
        colors_list,
        "freq",
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg5.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[1],
    )

    print("*** SHG2 ***")
    fig_pfg6 = intensity_phase_plot(
        freq_vectors_shg2_list,
        fields_shg2_list,
        labels_list,
        colors_list,
        "freq",
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_pfg6.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[2],
    )

    print("------- True vs Prediction Time Domain --------")

    print("*** SFG ***")
    fig_ptd4 = intensity_phase_plot(
        sfg_time_vector_list,
        sfg_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        xlims=[-15, 15],
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd4.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[3],
    )

    print("*** SHG1 ***")
    fig_ptd5 = intensity_phase_plot(
        shg1_time_vector_list,
        shg1_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd5.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[4],
    )

    print("*** SHG2 ***")
    fig_ptd6 = intensity_phase_plot(
        shg2_time_vector_list,
        shg2_freq_to_time_list,
        labels_list,
        colors_list,
        "time",
        normalize=normalize,
        offsets=[0, 0],
        save_format="jpg",
        save_name=model_save_name + "_ptd6.jpg",
        plot_show=True,
        plot_hold=False,
        save_dir=fig_save_dir,
        axs=new_axs[5],
    )

    new_fig.tight_layout()
    plt.show()

    file_save_name = (
        file_save_name
        or model_save_name + f"_All_{'normalized' if normalize else 'orig'}.jpg"
    )

    save_figure(file_save_name, "jpg", fig_save_dir)


def do_analysis(
    output_dir: str,  # directory from model training
    data_directory: str,  # directory from preprocessing
    model_save_name: str,  # model name from training
    file_idx: int,  # on which file to do analysis
    item_idx: int,  # which example of the file to do analysis
    fig_save_dir: str = None,  # where to save the figures
    crystal_length: int = 100,  # length of the crystal
    y_pred_trans_item: np.ndarray = None,
    y_true_trans_item: np.ndarray = None,
    file_save_name: str = None,
    return_vals: bool = False,
    labels_list: list = None,
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
    )  # * 1e12  # scaled to be back in Hz
    domain_spacing_2 = freq_vectors_shg2[1] - freq_vectors_shg2[0]  # * 1e12
    domain_spacing_3 = freq_vectors_sfg[1] - freq_vectors_sfg[0]  # * 1e12

    factors_freq = {
        "beam_area": 400e-6**2 * np.pi,
        "grid_spacing": [domain_spacing_1, domain_spacing_2, domain_spacing_3],
        "domain_spacing_1": domain_spacing_1,
        "domain_spacing_2": domain_spacing_2,
        "domain_spacing_3": domain_spacing_3,
    }  # beam radius 400 um (and circular beam)

    sfg_original_freq = np.load("Data/sfg_original_freq_vector.npy")
    sfg_original_time = np.load("Data/sfg_original_time_vector.npy")
    sfg_original_time_ds = sfg_original_time[1] - sfg_original_time[0]

    # Loading the single file out of the test dataset
    # these are used to compare to the predictions

    if y_true_trans_item is None:
        with h5py.File(os.path.join(data_directory, "y_new_data.h5"), "r") as file:
            dataset = file[f"dataset_{file_idx}"]
            # get all the data from the file
            y_true = dataset[:]
            # then scale it back to the original values
            y_true = scaler.inverse_transform(y_true)

        # Get the transformed value of the real item in the dataset
        # The first part slices out everything before the output of the first crystal,
        # then it jumps at iterations of the size of crystal_length to get the output
        # of the next crystal, then it selects one of those outputs in there.
        y_true_trans_all = y_true[crystal_length - 1 :][::crystal_length]
        y_true_trans_item = y_true_trans_all[item_idx]
    else:
        y_true_trans_item = y_true_trans_item[None, :]
        y_true_trans_item = scaler.inverse_transform(y_true_trans_item)
        y_true_trans_item = y_true_trans_item.squeeze()

    ### The part where we load the predictions

    if y_pred_trans_item is None:
        # the output file from the "predict" function
        # this has a shape of (files, predictions, channels)
        with h5py.File(
            os.path.join(output_dir, f"{model_save_name}_all_preds.h5"), "r"
        ) as file:
            y_preds = file[f"dataset_{file_idx}"]
            y_preds_loaded = y_preds[:]
        y_preds_trans = scaler.inverse_transform(y_preds_loaded)

        # Transform this using the scaler, then get which file
        # and then get the example using its index
        y_pred_trans_item = y_preds_trans[item_idx]
    else:
        y_pred_trans_item = y_pred_trans_item[None, :]
        y_pred_trans_item = scaler.inverse_transform(y_pred_trans_item)
        y_pred_trans_item = y_pred_trans_item.squeeze()

    # combine the vectors into a complex vector
    y_pred_trans_shg1, y_pred_trans_shg2, y_pred_trans_sfg = re_im_combined(
        y_pred_trans_item
    )
    y_true_trans_shg1, y_true_trans_shg2, y_true_trans_sfg = re_im_combined(
        y_true_trans_item
    )

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

    if return_vals:
        return (
            sfg_freq_to_time_true,
            sfg_freq_to_time_pred,
            shg1_freq_to_time_true,
            shg1_freq_to_time_pred,
            shg2_freq_to_time_true,
            shg2_freq_to_time_pred,
        )
    else:
        pass

    # required lists for plotting the frequency domain
    freq_vectors_sfg_list = [freq_vectors_sfg, freq_vectors_sfg]
    freq_vectors_shg1_list = [freq_vectors_shg1, freq_vectors_shg1]
    freq_vectors_shg2_list = [freq_vectors_shg2, freq_vectors_shg2]

    fields_sfg_list = [y_true_trans_sfg, y_pred_trans_sfg]
    fields_shg1_list = [y_true_trans_shg1, y_pred_trans_shg1]
    fields_shg2_list = [y_true_trans_shg2, y_pred_trans_shg2]

    # required lists for plotting the time domain
    sfg_time_vector_list = [sfg_original_time, sfg_original_time]
    shg1_time_vector_list = [sfg_original_time, sfg_original_time]
    shg2_time_vector_list = [sfg_original_time, sfg_original_time]

    sfg_freq_to_time_list = [sfg_freq_to_time_true, sfg_freq_to_time_pred]
    shg1_freq_to_time_list = [shg1_freq_to_time_true, shg1_freq_to_time_pred]
    shg2_freq_to_time_list = [shg2_freq_to_time_true, shg2_freq_to_time_pred]

    colors_list = ["red", "black"]
    if labels_list is None:
        labels_list = ["true", "pred"]

    print("Normalized plots")

    # Draw normalized plots
    plot_a_bunch_of_fields(
        freq_vectors_sfg_list,
        fields_sfg_list,
        freq_vectors_shg1_list,
        fields_shg1_list,
        freq_vectors_shg2_list,
        fields_shg2_list,
        sfg_time_vector_list,
        sfg_freq_to_time_list,
        shg1_time_vector_list,
        shg1_freq_to_time_list,
        shg2_time_vector_list,
        shg2_freq_to_time_list,
        labels_list,
        colors_list,
        model_save_name,
        fig_save_dir,
        file_save_name=file_save_name,
        normalize=True,
    )

    print("Original plots")

    # Draw non-normalized plots
    plot_a_bunch_of_fields(
        freq_vectors_sfg_list,
        fields_sfg_list,
        freq_vectors_shg1_list,
        fields_shg1_list,
        freq_vectors_shg2_list,
        fields_shg2_list,
        sfg_time_vector_list,
        sfg_freq_to_time_list,
        shg1_time_vector_list,
        shg1_freq_to_time_list,
        shg2_time_vector_list,
        shg2_freq_to_time_list,
        labels_list,
        colors_list,
        model_save_name,
        fig_save_dir,
        file_save_name=file_save_name,
        normalize=False,
    )
