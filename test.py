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

# first_train_losses = np.load("/mnt/oneterra/outputs/04-12-2023/LSTM_100_epoch_55_train_losses.npy")
# first_val_losses = np.load("/mnt/oneterra/outputs/04-12-2023/LSTM_100_epoch_55_val_losses.npy")

# train_losses = np.load("/mnt/oneterra/outputs/04-12-2023/LSTM_100_cont_epoch_25_train_losses.npy")
# val_losses = np.load("/mnt/oneterra/outputs/04-12-2023/LSTM_100_cont_epoch_25_val_losses.npy")

# # put the first train losses at the beginning of the train losses
# train_losses = np.concatenate((first_train_losses, train_losses))
# val_losses = np.concatenate((first_val_losses, val_losses))

# train_losses = train_losses[10:]
# val_losses = val_losses[10:]

# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.legend()
# plt.show()
# plt.savefig("losses.png")

import pickle

# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'your_file.pkl'

with open(file_path, 'rb') as file:
    # Load (unpickle) the contents of the file
    data = pickle.load(file)

# Now, you can print or inspect 'data' to see what's inside the pickle file
print(data)
