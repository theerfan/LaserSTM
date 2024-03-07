# # #
# import numpy as np
# import os

# # direct = "/mnt/oneterra/SFG_reIm_version1_reduced/"
# direct = "/mnt/oneterra/SFG_reIm_version1/"
# # direct = "/mnt/oneterra/outputs/04-12-2023"

# def get_npy_shape(npy_file_path):
#     with open(npy_file_path, 'rb') as f:
#         version = np.lib.format.read_magic(f)
#         shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
#         return shape, dtype

# print(get_npy_shape(os.path.join(direct, "X_new_72.npy")))

# for i in range(0, 10):
#     fname = os.path.join(direct, f"y_new_{i}.npy")
#     shape = get_npy_shape(fname)
#     print("Shape of the npy file:", shape)
# #


###
# def test_energy_stuff():
#     val_dataset = CustomSequence(".", [0], test_mode=True)
#     gen = val_dataset[0]
#     X, y = next(gen)
#     pseudo_energy_loss(y, y)
#     pass
###


# from Analysis.analyze_reim import do_analysis

# do_analysis(
#     "/mnt/oneterra/outputs/06-12-2023",
#     "/mnt/oneterra/SFG_reIm_version1/",
#     "LSTM_100_epoch_66",
#     91,
#     0,
#     15
# )

# import os
# import re

# def delete_specific_files(directory):
#     # Regular expression to match files of the format X_{number}.npy or y_{number}.npy
#     pattern = re.compile(r'^(X|y)_[0-9]+\.npy$')

#     # Iterate over the files in the specified directory
#     for filename in os.listdir(directory):
#         # Check if the file matches the pattern and does not contain 'new'
#         if pattern.match(filename) and 'new' not in filename:
#             file_path = os.path.join(directory, filename)
#             os.remove(file_path)  # Delete the file
#             print(f"Deleted: {file_path}")

# # Specify the directory to operate on
# directory = '/mnt/oneterra/SFG_reIm_version1'
# delete_specific_files(directory)


# import numpy as np
# import h5py
# import os

# # Directory containing the .npy files
# npy_dir = '/mnt/oneterra/SFG_reIm_version1/'

# # Path for the new HDF5 file
# hdf5_path = '/mnt/oneterra/SFG_reIm_h5/X_new_data.h5'

# # Create the directory for the HDF5 file if it doesn't exist
# os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

# # Open an HDF5 file in write mode
# with h5py.File(hdf5_path, 'w') as hdf5_file:
#     # Loop through the file numbers
#     for i in range(100):  # 0 to 99
#         # Construct the .npy file path
#         npy_file_path = os.path.join(npy_dir, f'X_new_{i}.npy')

#         # Load the .npy file
#         data = np.load(npy_file_path)

#         # Save the data to the HDF5 file, indexed by file number
#         hdf5_file.create_dataset(f'dataset_{i}', data=data)

# print(f'All .npy files have been saved to {hdf5_path}')


# import numpy as np
# import matplotlib.pyplot as plt

# first_train_losses = np.load("/mnt/twoterra/outputs/21-02-2024/LSTM_200_anew_epoch_103_train_losses.npy")
# first_val_losses = np.load("/mnt/twoterra/outputs/21-02-2024/LSTM_200_anew_epoch_103_val_losses.npy")

# model_path = "/mnt/twoterra/outputs/28-02-2024/"
# model_name = "LSTM_200_anew_epoch_187"

# train_losses = np.load(f"{model_path}{model_name}_train_losses.npy")
# val_losses = np.load(f"{model_path}{model_name}_val_losses.npy")

# # # # put the first train losses at the beginning of the train losses
# train_losses = np.concatenate((first_train_losses, train_losses))
# val_losses = np.concatenate((first_val_losses, val_losses))

# train_losses = train_losses[10:]
# val_losses = val_losses[10:]

# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.legend()
# plt.show()
# plt.savefig(f"{model_name}_losses.png")

# # import pickle

# # # Replace 'your_file.pkl' with the path to your pickle file
# # file_path = 'your_file.pkl'

# # with open(file_path, 'rb') as file:
# #     # Load (unpickle) the contents of the file
# #     data = pickle.load(file)

# # # Now, you can print or inspect 'data' to see what's inside the pickle file
# # print(data)

# ###

# import h5py
# import numpy as np

# #  Path for the new HDF5 file
# x_hdf5_path = '/mnt/oneterra/SFG_reIm_h5/X_new_data.h5'
# y_hdf5_path = '/mnt/oneterra/SFG_reIm_h5/y_new_data.h5'

# Open an HDF5 file in read mode
# with h5py.File(x_hdf5_path, 'r') as hdf5_file:
#     # Get the keys of the datasets
#     keys = list(hdf5_file.keys())

#     # Get the first dataset
#     first_dataset = hdf5_file[keys[0]]

#     # Get the shape of the first dataset
#     first_dataset_shape = first_dataset.shape

#     # Get the dtype of the first dataset
#     first_dataset_dtype = first_dataset.dtype

#     print(f'First dataset shape: {first_dataset_shape}')
#     print(f'First dataset dtype: {first_dataset_dtype}')

# # Open an HDF5 file in read mode
# with h5py.File(y_hdf5_path, 'r') as hdf5_file:
#     # Get the keys of the datasets
#     keys = list(hdf5_file.keys())

#     # Get the first dataset
#     first_dataset = hdf5_file[keys[0]]

#     # Get the shape of the first dataset
#     first_dataset_shape = first_dataset.shape

#     # Get the dtype of the first dataset
#     first_dataset_dtype = first_dataset.dtype

#     print(f'First dataset shape: {first_dataset_shape}')
#     print(f'First dataset dtype: {first_dataset_dtype}')


# Go through each x dataset, then use only the last element in the second dimension and save it to a new dataset called x_reduced

# Open an HDF5 file in read mode
# with h5py.File(x_hdf5_path, 'r') as hdf5_file:
#     # Create a new HDF5 file in write mode
#     with h5py.File('/mnt/oneterra/SFG_reIm_h5/X_new_reduced_data.h5', 'w') as hdf5_file_reduced:
#         # Loop through the datasets
#         for key in hdf5_file.keys():
#             # Get the dataset
#             dataset = hdf5_file[key]

#             print(f'Processing dataset: {key}')

#             # Perform reduction (taking the last element in the second dimension)
#             reduced_data = dataset[:, -1, :]

#             # Create a new dataset in the new HDF5 file with reduced data
#             hdf5_file_reduced.create_dataset(key, data=reduced_data)
#
# import h5py

# Get the keys of the datasets from the original HDF5 file
# import logging

# # Configure logging
# logging.basicConfig(filename='dataset_processing.log',
#                     level=logging.INFO,
#                     format='%(asctime)s %(levelname)s: %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')

# # Get the keys of the datasets from the original HDF5 file
# with h5py.File(x_hdf5_path, 'r') as hdf5_file:
#     keys = list(hdf5_file.keys())

# # Process each dataset one by one
# for key in keys:
#     logging.info(f'Processing dataset: {key}')

#     # Open the original HDF5 file to read the current dataset
#     with h5py.File(x_hdf5_path, 'r') as hdf5_file:
#         dataset = hdf5_file[key]

#         # Perform reduction (taking the last element in the second dimension)
#         reduced_data = dataset[:, -1, :]

#     # Open the reduced HDF5 file to write the processed data
#     with h5py.File('/mnt/oneterra/SFG_reIm_h5/X_new_reduced_data.h5', 'a') as hdf5_file_reduced:
#         # Create a new dataset in the reduced HDF5 file with reduced data
#         hdf5_file_reduced.create_dataset(key, data=reduced_data)

#     logging.info(f'Completed dataset: {key}')


import numpy as np

# path =  '/mnt/oneterra/outputs/07-02-2024/LSTM_120_epoch_42_time_domain_MSE_errors.npz'

path = "/mnt/twoterra/outputs/14-02-2024/LSTM_200_epoch_125_time_domain_MSE_errors.npz"

f = np.load(path)
sfg, shg1, shg2 = f["SFG_MSE_errors"], f["SHG1_MSE_errors"], f["SHG2_MSE_errors"]


def get(f_idx, e_idx):
    idx = (f_idx - 91) * 100 + e_idx
    return np.array([sfg[idx], shg1[idx], shg2[idx]])


# means = np.array([np.mean(sfg), np.mean(shg1), np.mean(shg2)])


# def norm_dist(f_idx, e_idx):
#     return (get(f_idx, e_idx) - means) / means


# def find_percentile_of_value(arr, value):
#     # Count how many values are less than the value
#     less_than_count = np.sum(arr < value)

#     # Count how many values are equal to the value
#     equal_count = np.sum(arr == value)

#     # Calculate the rank of the value
#     rank = less_than_count + 0.5 * equal_count

#     # Calculate the percentile rank
#     percentile_rank = (rank / arr.size) * 100

#     return percentile_rank


# def calculate_percentile_cutoffs(arr, percentiles_list):
#     # Sort the array just in case it's not sorted, though for percentile calculation it's not a necessary step
#     sorted_arr = np.sort(arr)

#     # Calculate the percentiles
#     percentiles = np.percentile(sorted_arr, percentiles_list)

#     return percentiles


# file_idx = 94
# example_idx = 15
# print("For model loaded from", path)
# # print(
# #     f"For file {file_idx} and example {example_idx} the normalized distance from the mean is:"
# # )
# # print(norm_dist(94, 15))
# # print(
# #     "The percentile rank of the SFG value is:",
# #     find_percentile_of_value(sfg, sfg[(file_idx - 91) * 100 + example_idx]),
# # )
# # print(
# #     "The percentile rank of the SHG1 value is:",
# #     find_percentile_of_value(shg1, shg1[(file_idx - 91) * 100 + example_idx]),
# # )
# # print(
# #     "The percentile rank of the SHG2 value is:",
# #     find_percentile_of_value(shg2, shg2[(file_idx - 91) * 100 + example_idx]),
# # )


# percentiles = [20, 40, 60, 80]
# for p, cutoff1, cutoff2, cutoff3 in zip(
#     percentiles,
#     calculate_percentile_cutoffs(sfg, percentiles),
#     calculate_percentile_cutoffs(shg1, percentiles),
#     calculate_percentile_cutoffs(shg2, percentiles),
# ):
#     # print the cutoffs in scientific notation
#     print(f"Percentile {p} cutoffs:")
#     print(f"SFG: {cutoff1:.3e}")
#     print(f"SHG1: {cutoff2:.3e}")
#     print(f"SHG2: {cutoff3:.3e}")
#     print()


def find_percentile_indices(errors):
    """
    Find indices of samples within each 10-percentile range for a given error array.

    Parameters:
    - errors: numpy array of errors.

    Returns:
    - A dictionary where keys are percentile ranges (as strings) and
      values are lists of indices within those ranges.
    """
    percentile_indices = {}
    for i in range(0, 100, 10):
        # Define percentile range
        low_percentile = np.percentile(errors, i)
        high_percentile = np.percentile(errors, i + 10)

        # Find indices of samples within the current percentile range
        indices = np.where((errors >= low_percentile) & (errors < high_percentile))[0]

        # Add indices to the dictionary
        percentile_range = f"{i}-{i+10}"
        percentile_indices[percentile_range] = indices

    return percentile_indices


def select_random_index_per_percentile(percentile_indices):
    """
    Select a random index from each percentile range.

    Parameters:
    - percentile_indices: Dictionary of percentile ranges to indices.

    Returns:
    - A dictionary with the same keys but a single, randomly selected index as values.
    """
    selected_indices = {}
    for percentile_range, indices in percentile_indices.items():
        if len(indices) > 0:
            selected_indices[percentile_range] = np.random.choice(indices)
        else:
            selected_indices[percentile_range] = (
                None  # No index available for this range
            )
    return selected_indices


def select_very_good_example(
    sfg_percentiles,
    shg1_percentiles,
    shg2_percentiles,
    percentile_indices="90-100",
    return_all=False,
):
    """
    Selects an example which is in the selected percentile range for all three error arrays.
    """
    sfg_indices = sfg_percentiles[percentile_indices]
    shg1_indices = shg1_percentiles[percentile_indices]
    shg2_indices = shg2_percentiles[percentile_indices]

    shg_indices = np.intersect1d(shg1_indices, shg2_indices)
    common_indices = np.intersect1d(sfg_indices, shg_indices)

    if len(common_indices) > 0:
        if return_all:
            return common_indices
        else:
            return np.random.choice(common_indices)
    else:
        return None  # No common index available for this range


def reverse_get_formula(idx):
    """
    Reverse engineers the 'get' function formula to find f_idx and e_idx.

    Parameters:
    - idx: Combined index as used in the 'get' function.

    Returns:
    - Tuple of (f_idx, e_idx)
    """
    f_idx = idx // 100 + 91  # Integer division to reverse-engineer f_idx
    e_idx = idx % 100  # Remainder gives e_idx
    return (f_idx, e_idx)


# Find percentile indices for each error array
sfg_percentiles = find_percentile_indices(sfg)
shg1_percentiles = find_percentile_indices(shg1)
shg2_percentiles = find_percentile_indices(shg2)

# Assuming the percentile_indices dictionaries are already computed
sfg_selected = select_random_index_per_percentile(sfg_percentiles)
shg1_selected = select_random_index_per_percentile(shg1_percentiles)
shg2_selected = select_random_index_per_percentile(shg2_percentiles)


# Assuming the `get` function and previously defined functions are available in the context
def print_example_indices_for_all_percentiles(selected_indices, errors_name):
    """
    Prints example indices and corresponding errors for all percentiles for a given error array.

    Parameters:
    - selected_indices: Dictionary of selected indices per percentile.
    - errors_name: Name of the error array (for printing purposes).
    """
    for percentile_range, idx in selected_indices.items():
        if idx is not None:
            f_idx, e_idx = reverse_get_formula(idx)
            example_errors = get(f_idx, e_idx)
            print(
                f"{errors_name} {percentile_range}th percentile example: f_idx={f_idx}, e_idx={e_idx}, errors={example_errors}"
            )
        else:
            print(
                f"No {errors_name} index available for {percentile_range}th percentile."
            )


def print_good_indices_for_all_percentiles(
    sfg_percentiles, shg1_percentiles, shg2_percentiles
):
    for i in range(0, 100, 10):
        key = f"{i}-{i+10}"
        idx = select_very_good_example(
            sfg_percentiles, shg1_percentiles, shg2_percentiles, key
        )
        if idx is not None:
            f_idx, e_idx = reverse_get_formula(idx)
            example_errors = get(f_idx, e_idx)
            print(
                f"{key}th percentile example: f_idx={f_idx}, e_idx={e_idx}, errors={example_errors}"
            )
        else:
            print(f"No index available for {key}th percentile.")


def plot_error_bars_for_all_percentiles(
    sfg_percentiles, shg1_percentiles, shg2_percentiles
):
    import matplotlib.pyplot as plt
    # Setup the figure and axis for the plot
    # plt.figure(figsize=(10, 6))

    percentiles_range = list(range(0, 100, 10))
    n_percentiles = len(percentiles_range)
    means = []
    std_devs = []
    max_errors = []

    fig, axes = plt.subplots(1, n_percentiles, figsize=(15, 5))

    for i, idx in enumerate(percentiles_range):
        key = f"{idx}-{idx+10}"
        indexes = select_very_good_example(
            sfg_percentiles, shg1_percentiles, shg2_percentiles, key, return_all=True
        )
        errors = np.array([get(*reverse_get_formula(idx)) for idx in indexes])

        # Calculate mean, standard deviation (mean error), and max error
        mean_error = np.mean(errors)
        std_dev = np.std(errors)
        max_error = np.max(errors) - mean_error

        # Save the calculations for plotting
        means.append(mean_error)
        std_devs.append(std_dev)
        max_errors.append(max_error)

        # Plot the error bars and means as circles
        current_percentile = [percentiles_range[i]]
        current_mean = [means[i]]
        current_std_dev = [[std_devs[i]], [max_errors[i]]]
        axes[i].errorbar(
            current_percentile,
            current_mean,
            yerr=current_std_dev,
            fmt="o",
            linestyle="-",
            ecolor="r",
            capsize=5,
            capthick=2,
            marker="o",
            markersize=5,
            label="Mean & Errors" if i == 0 else "",
        )

        # Calculate min and max y values for this subplot
        y_min = mean_error - std_dev
        y_max = mean_error + max(max_error, std_dev)

        # Optionally, add some padding to min and max y values
        padding = 0.1 * (y_max - y_min)
        y_min -= padding
        y_max += padding

        # Set y-axis limits for this subplot
        axes[i].set_ylim(y_min, y_max)

        axes[i].set_xticks([]) 
        axes[i].text(0.5, -0.1, key, ha='center', va='top', transform=axes[i].transAxes)
        # axes[i].set_xticklabels([key], rotation=45, ha='center')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(
        wspace=0.8
    )  # Adjust this value as needed to increase horizontal spacing

    # Adjust figure-level attributes and display the plot
    fig.suptitle("Error Bars for Each Percentile")
    axes[n_percentiles // 2].set_xlabel("Percentiles")
    fig.text(0.04, 0.5, "Error Values", va="center", rotation="vertical")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.show()

    plt.savefig("error_bars.png")


# Function calls for sfg, shg1, and shg2
# print_example_indices_for_all_percentiles(sfg_selected, "SFG")
# print_example_indices_for_all_percentiles(shg1_selected, "SHG1")
# print_example_indices_for_all_percentiles(shg2_selected, "SHG2")

# print_good_indices_for_all_percentiles(sfg_percentiles, shg1_percentiles, shg2_percentiles)
