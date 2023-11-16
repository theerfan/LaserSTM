# #
# import numpy as np
# import os

# direct = "/mnt/oneterra/SFG_reIm_version1_reduced/"
# direct = "/mnt/oneterra/SFG_reIm_version1"

# def get_npy_shape(npy_file_path):
#     with open(npy_file_path, 'rb') as f:
#         version = np.lib.format.read_magic(f)
#         shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
#         return shape

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
    "/mnt/oneterra/outputs/15-11-2023/",
    "/mnt/oneterra/SFG_reIm_version1/",
    "LSTM_hyper_shg_0.3_sfg_0.7_epoch_10",
    0,
    15
)