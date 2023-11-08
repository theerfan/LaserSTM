# from LSTM.utils import re_im_sep, re_im_sep_vectors
import os
import numpy as np
import pypret
import matplotlib.pyplot as plt

# `:,` is there because we want to keep the batch dimension
def re_im_sep(fields: np.array, detach=False):
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

data_dir = "/mnt/oneterra/SFG_reIm_version1"

file_dir = os.path.join(data_dir, "X_new_0.npy")

X = np.load(file_dir)

X_0 = X[0]

shg1, shg2, sfg = re_im_sep(X_0)

time = np.linspace(-500e-15, 500e-15, 128)

central_wavelength = 800e-9

pulse = pypret.Pulse(time, central_wavelength)

pulse.set_field(shg1)

pypret.PulsePlot(pulse)
plt.savefig("frog1.jpg")