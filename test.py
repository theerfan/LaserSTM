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
# # import matplotlib.pyplot as plt

# # # first_train_losses = np.load("/mnt/oneterra/outputs/23-12-2023/LSTM_100_epoch_55_train_losses.npy")
# # # first_val_losses = np.load("/mnt/oneterra/outputs/23-12-2023/LSTM_100_epoch_55_val_losses.npy")

# train_losses = np.load("/mnt/oneterra/outputs/03-01-2024/LSTM_bi_epoch_26_train_losses.npy")
# val_losses = np.load("/mnt/oneterra/outputs/03-01-2024/LSTM_bi_epoch_26_val_losses.npy")

# # # # put the first train losses at the beginning of the train losses
# # # train_losses = np.concatenate((first_train_losses, train_losses))
# # # val_losses = np.concatenate((first_val_losses, val_losses))

# train_losses = train_losses[2:]
# val_losses = val_losses[2:]

# # plt.plot(train_losses, label="Train Loss")
# # plt.plot(val_losses, label="Val Loss")
# # plt.legend()
# # plt.show()
# # plt.savefig("losses.png")

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

path =  '/mnt/oneterra/outputs/07-02-2024/LSTM_120_epoch_42_time_domain_MSE_errors.npz'
# path = '/mnt/twoterra/outputs/14-02-2024/LSTM_200_epoch_125_time_domain_MSE_errors.npz'

f = np.load(path)
sfg, shg1, shg2 = f['SFG_MSE_errors'], f['SHG1_MSE_errors'], f['SHG2_MSE_errors']

def get(f_idx, e_idx):
    idx = (f_idx - 91) * 100 + e_idx
    return np.array([sfg[idx], shg1[idx], shg2[idx]])

means = np.array([np.mean(sfg), np.mean(shg1), np.mean(shg2)])

def norm_dist(f_idx, e_idx):
    return (get(f_idx, e_idx) - means) / means

print(norm_dist(91, 0))