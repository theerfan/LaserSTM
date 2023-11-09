# from LSTM.utils import re_im_sep, re_im_sep_vectors
import os
import numpy as np
import pypret
import matplotlib.pyplot as plt
from scipy.constants import c

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
shg1_0, shg2_0, sfg_0 = shg1[0], shg2[0], sfg[0]

time = np.linspace(-500e-15, 500e-15, 128)
central_freq = 1025e-9 / 2
w_0 = 2 * np.pi * (central_freq / 2) / c
dt = time[1] - time[0]

# create simulation grid
ft = pypret.FourierTransform(shg1_0.shape[0], dt=dt, w0=w_0)
# instantiate a pulse object, central wavelength 800 nm
pulse = pypret.Pulse(ft, central_freq)
pulse.spectrum = shg1_0

pypret.PulsePlot(pulse)
plt.savefig("frog1.jpg")


# simulate a frog measurement
delay = np.linspace(-500e-15, 500e-15, 128)  # delay in s
pnps = pypret.PNPS(pulse, "frog", "shg")
# calculate the measurement trace
pnps.calculate(pulse.spectrum, delay)
original_spectrum = pulse.spectrum
# and plot it
pypret.MeshDataPlot(pnps.trace)
plt.savefig("frog2.jpg")
