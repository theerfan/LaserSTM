# #
# import numpy as np
# import os

# # direct = "/mnt/oneterra/SFG_reIm_version1_reduced/"
# direct = "/mnt/oneterra/SFG_reIm_version1/"
# # direct = "/mnt/oneterra/outputs/04-12-2023"

# def get_npy_shape(npy_file_path):
#     with open(npy_file_path, 'rb') as f:
#         version = np.lib.format.read_magic(f)
#         shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
#         return shape

# print(get_npy_shape(os.path.join(direct, "y_new_0.npy")))

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



from Analysis.analyze_reim import do_analysis

do_analysis(
    "/mnt/oneterra/outputs/06-12-2023",
    "/mnt/oneterra/SFG_reIm_version1/",
    "LSTM_100_epoch_66",
    91,
    0,
    15
)

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
