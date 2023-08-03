
import numpy as np
from scipy.interpolate import interp1d


def get_phase(field):
    return np.arctan2(np.imag(field), np.real(field))

def resample_method1(input_domain, target_domain, input_vector):
    try:
        f = interp1d(input_domain, input_vector)
        resampled_vector = f(target_domain)
        return resampled_vector
    except ValueError:
        print(
            "Likely the target wavelength vector is outside the bounds of the input vector (only interpolation\n"
        )

def fft(field):
    """fft with shift

    Shifting values so that initial time is 0.
    Then perform FFT, then shift 0 back to center.

    field: 1xN numpy array

    return a 1xN numpy array"""
    return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(field)))


def ifft(field):
    """ifft with shift

    Shifting values so that initial time is 0.
    Then perform IFFT, then shift 0 back to center.

    field: 1xN numpy array

    return a 1xN numpy array"""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(field)))


def get_intensity(field):
    """
    Returns the intensity of a field
    """
    return np.abs(field) ** 2


def energy_match(field, energy):
    return np.sqrt(
        energy * get_intensity(field) / np.sum(get_intensity(field))
    ) * np.exp(1j * get_phase(field))


def calc_energy_expanded(field, domain_spacing, spot_area):
    return np.sum(get_intensity(field)) * domain_spacing * spot_area


def energy_match_expanded(field, energy, domain_spacing, spot_area):
    norm_E = np.sum(get_intensity(field)) * domain_spacing * spot_area
    return np.sqrt(energy / norm_E) * field



class UNITS:
    def __init__(self, mScale=0, sScale=0):
        self.m = 10**mScale
        self.mm = 10 ** (-3 * self.m)
        self.um = 10 ** (-6 * self.m)
        self.nm = 10 ** (-9 * self.m)

        self.s = 10**sScale
        self.ns = 10 ** (-9 * self.s)
        self.ps = 10 ** (-12 * self.s)
        self.fs = 10 ** (-15 * self.s)

        self.J = (self.m**2) / (self.s**2)
        self.mJ = 10 ** (-3 * self.J)
        self.uJ = 10 ** (-6 * self.J)
