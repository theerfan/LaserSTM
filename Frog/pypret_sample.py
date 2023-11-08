import numpy as np
import pypret
import matplotlib.pyplot as plt

# create simulation grid
ft = pypret.FourierTransform(256, dt=5.0e-15)
# instantiate a pulse object, central wavelength 800 nm
pulse = pypret.Pulse(ft, 800e-9)
# create a random pulse with time-bandwidth product of 2.
pypret.random_pulse(pulse, 2.0)
# plot the pulse
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

# # and do the retrieval
# ret = pypret.Retriever(pnps, "copra", verbose=True, maxiter=300)
# # start with a Gaussian spectrum with random phase as initial guess
# pypret.random_gaussian(pulse, 50e-15, phase_max=0.0)
# # now retrieve from the synthetic trace simulated above
# ret.retrieve(pnps.trace, pulse.spectrum)
# # and print the retrieval results
# ret.result(original_spectrum)
