import numpy as np
import os

direct = "/mnt/oneterra/SFG_reIm_version1/"

def get_npy_shape(npy_file_path):
    with open(npy_file_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        return shape

for i in range(0, 100):
    fname = os.path.join(direct, f"X_new_{i}.npy")
    shape = get_npy_shape(fname)
    print("Shape of the npy file:", shape)