import numpy as np

#import mpmath
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import os
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import interpolate
from scipy import ndimage
from scipy.signal import savgol_filter
import csv

c = 299792458.0

def concatenate_arrays(*arrays):
    return np.concatenate(arrays,axis=None)

def split_and_concatenate(fields, scaling_factors, normalize_energy=False):
    energy_out = np.array([])
    intensity_out = np.array([])
    phase_out = np.array([])
    scaling_factors["beam_area"] = (400 * 1e-6) ** 2 * (np.pi)
    for i, field in enumerate(fields):
        intensity = get_intensity(field)
        phase = get_phase(field)

        energy = calc_energy_expanded(
            field, scaling_factors["grid_spacing"][i], scaling_factors["beam_area"]
        )

        if normalize_energy:
            energy = energy / 25e-6
        else:
            pass

        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)
        else:
            pass

        energy_out = concatenate_arrays(energy_out, energy)
        intensity_out = concatenate_arrays(intensity_out, intensity)
        phase_out = concatenate_arrays(phase_out, phase / np.pi)

    return concatenate_arrays(intensity_out, phase_out, energy_out).astype(np.float32)

def create_necessary_folders_and_files():
    # create output path
    if not os.path.exists("output"):
        os.makedirs("output")


def to_angular(vec):
    """
    To angular frequency
    """
    return 2 * np.pi * vec


def to_freq_vector(time_vector):
    """
    Time vec to freq vec
    """
    return np.fft.fftshift(np.fft.fftfreq(n=time_vector.shape[0], d=(time_vector[1] - time_vector[0])))


def get_phase(field):
    return np.arctan2(np.imag(field), np.real(field))


def super_gaus(x, sigma, x0, pow=2, amp=1):
    return amp * np.exp((-1 * ((x - x0) ** 2 / (2 * sigma ** 2)) ** pow).astype(float))


def resample_method1(input_domain, target_domain, input_vector):
    try:
        f = interp1d(input_domain, input_vector)
        resampled_vector = f(target_domain)
        return resampled_vector
    except ValueError:
        print("Likely the target wavelength vector is outside the bounds of the input vector (only interpolation\n")


def interpolate_Efield(efield_fd, freq_vector_old, freq_vector_new):
    """
    Interpolates Efield from the dazzler-set sampling rate and vector length defined
    by freq_vector_old to the new sampling rate and vector length defined by freq_vector_new.
    This interpolation is necessary to preserve phase information and frequency bandwidth.

    efield_fd: 1xN complex numpy array
    freq_vector_old: 1xN numpy array that was defined by self.input_freq_vector
    freq_vector_new: 1xN numpy array that was defined in genGrids and has a set length of 2**15 and a spacing of 16.5 fs

    return a 1xN numpy array of the input field with the sample sampling rate and length as freq_vector_new

    SPECIAL NOTATION: It might seem like breaking down exp(i*phase) into it's real and imaginary components and then
    interpolating is redundant. HOWEVER, if you just interpolate the phase and then do exp(i (interpolated phase))
    there will be odd gaussian shaped growths on the exp(i phase) graph. To check this, plot exp(i (interpolated
    phase)) against frequency
    """
    real = np.real(efield_fd)
    imag = np.imag(efield_fd)
    real_new = scipy.interpolate.griddata(freq_vector_old, real, freq_vector_new, method='linear')
    imag_new = scipy.interpolate.griddata(freq_vector_old, imag, freq_vector_new, method='linear')
    efield_new = real_new + 1j * imag_new
    return efield_new


def reorder(vec):
    vec_reordered = np.copy(vec)
    vec_reordered[0:vec.shape[0] // 2] = vec[vec.shape[0] // 2:]
    vec_reordered[vec.shape[0] // 2:] = vec[0:vec.shape[0] // 2]
    return vec_reordered


def convert_to_wavelength(I_freq, phase_freq, freq_vec, wavelength_vector_limits, sample_points=0):
    """
    This function converts the frequency domain signal into a wavelength domain.
    Defaults to a wavelength vector that goes from 200nm below to 200nm above the central
    wavelength. This should be adjusted if width of distribution is very large.
    """
    if sample_points == 0:
        sample_points = len(freq_vec)
    wavelength_vector = np.linspace(wavelength_vector_limits[0], wavelength_vector_limits[1], num=sample_points)
    I_freq_interp = interp1d(2 * np.pi * freq_vec, I_freq)
    I_wavelength = (2 * np.pi * c / (wavelength_vector ** 2)) * I_freq_interp(2 * np.pi * c / wavelength_vector)
    phase_freq_interp = interp1d(2 * np.pi * freq_vec, phase_freq)
    phase_wavelength = phase_freq_interp(2 * np.pi * c / wavelength_vector)
    return wavelength_vector, I_wavelength, phase_wavelength


def inten_phase_plot(domain, field, xlabel="time (s)", y1label="Norm. Intensity", normalize=True, xlims=None):
    fig, ax = plt.subplots()
    if normalize:
        factor = np.max(np.abs(field) ** 2)
    else:
        factor = 1
    ax.plot(domain, np.abs(field) ** 2 / factor, color="red")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(y1label, color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(domain, np.unwrap(get_phase(field)), color="blue")
    ax2.set_ylabel("Phase (Arb.)", color="blue", fontsize=14)
    if xlims is None:
        x_lim = [domain[0], domain[-1]]
    else:
        x_lim = [xlims[0], xlims[1]]
    plt.xlim(x_lim[0], x_lim[1])
    plt.show()


def spec_phase_plot2(domain, field, xlabel="frequency (Hz)", y1label="Norm. Intensity", normalize=True, xlims=None,
                     shift_domain=True):
    fig, ax = plt.subplots()
    if shift_domain:
        shifted_domain = np.fft.fftshift(domain)
    else:
        shifted_domain = domain
    if normalize:
        factor = np.max(np.abs(field) ** 2)
    else:
        factor = 1
    ax.plot(shifted_domain, (np.abs(field) ** 2) / factor, color="red")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(y1label, color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(shifted_domain, (np.unwrap((get_phase(field)))), color="blue")
    ax2.set_ylabel("Phase (Arb.)", color="blue", fontsize=14)

    if xlims is None:
        x_lim = [shifted_domain[0], shifted_domain[-1]]
    else:
        x_lim = [xlims[0], xlims[1]]
    plt.xlim(x_lim[0], x_lim[1])
    plt.show()


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


def get_freq_vector(time_vector):
    """get frequency vector from time vector

    time_vector: 1xN numpy array

    return a 1xN numpy array"""
    return np.fft.fftshift(np.fft.fftfreq(n=len(time_vector), d=time_vector[1] - time_vector[0]))


def exp(x):
    """
    quick return for exponential function
    """
    return np.exp(x)


def ln(x):
    """
    quick return for exponential function
    """
    return np.log(x)


def sqrt(x):
    """
    quick return for sqrt function
    """
    return np.sqrt(x)


# def sech(x):
#     """
#     quick return for sech function
#     """
#     return mpmath.sech(x)


# def acosh(x):
#     """
#     quick return for acosh function
#     """
#     return mpmath.acosh(x)


def rotate_peak_indices(array):
    """
    Rotates peak so that it can be centered at f or t = 0 following a
    fourier transformation. This is acheived when desired_pk_indx = mid_pt_indx
    np_array: 1xN numpy array with a local maximum
    desired_pk_indx: integer index

    return a 1xN numpy array rotated to place the peak at desired_pk_indx
    """
    ind = np.argmax(array)
    return np.roll(array, array.shape[0] // 2 - ind)


def find_peak_positions(x, y):
    """
    Finds peaks in a 1D array. Returns the indices of the peaks.
    """
    peaks = find_peaks(y / np.max(y), threshold=0.5)  # find peaks
    position = x[peaks[0]]
    if len(position) == 1:
        return x[0]
    return np.asarray(x[peaks[0]])


# Doesn't work well for fourier transformed data... DONT USE
def find_fwhm(x, y):
    print("Depricated function. Use fwhm() instead.")
    y = y / np.max(y)
    peaks = find_peaks(y / np.max(y), threshold=0.5)
    dx = x[1] - x[0]
    pw = peak_widths(y, peaks[0], rel_height=0.5)[0]
    if len(pw) == 1:
        return dx * pw[0]
    return np.asarray(dx * pw)


'''
find, agrlimit, limit and fwhm taken from pypret package: https://github.com/ncgeib/pypret/blob/d47deb675640439df7c8b7c08d71f45ecea3c568/pypret/lib.py#L193
Can update these later to our own functions if need be
'''


def calculate_bandwidth(y, x):
    """
    Calculates the bandwidth of a spectral intensity.
    Treats as single pulse so looks at bandwidth of entire spectrum (FWHM)
    """
    y = get_intensity(y)
    y = y / np.max(y)
    # find starting position
    peak_position = np.argmax(y)  # defines peak, one point will be to left and one point will be to the right
    end = 0
    start = 0
    for i in range(1, peak_position):
        if y[i] >= 0.5 > y[i - 1]:
            if np.abs(y[i] - 0.5) < np.abs(y[i - 1] - 0.5):
                start = i
            else:
                start = i - 1
            break
    # find ending position
    for i in range(len(y) - 2, peak_position, -1):
        if y[i] >= 0.5 > y[i + 1]:
            if np.abs(y[i] - 0.5) < np.abs(y[i + 1] - 0.5):
                end = i
            else:
                end = i + 1
            break
    return x[end] - x[start]


def find(x, condition, n=1):
    """ Return the index of the nth element that fulfills the condition.
    """
    search_n = 1
    for i in range(len(x)):
        if condition(x[i]):
            if search_n == n:
                return i
            search_n += 1
    return -1


def arglimit(y, threshold=1e-3, padding=0.0, normalize=True):
    """ Returns the first and last index where `y >= threshold * max(abs(y))`.
    """
    t = np.abs(y)
    if normalize:
        t /= np.max(t)

    idx1 = find(t, lambda x: x >= threshold)
    if idx1 == -1:
        idx1 = 0
    idx2 = find(t[::-1], lambda x: x >= threshold)
    if idx2 == -1:
        idx2 = t.shape[0] - 1
    else:
        idx2 = t.shape[0] - 1 - idx2

    return idx1, idx2


def limit(x, y=None, threshold=1e-3, padding=0.25, extend=True):
    """ Returns the maximum x-range where the y-values are sufficiently large.
    Parameters
    ----------
    x : array_like
        The x values of the graph.
    y : array_like, optional
        The y values of the graph. If `None` the maximum range of `x` is
        used. That is only useful if `padding > 0`.
    threshold : float
        The threshold relative to the maximum of `y` of values that should be
        included in the bracket.
    padding : float
        The relative padding on each side in fractions of the bracket size.
    extend : bool, optional
        Signals if the returned range can be larger than the values in ``x``.
        Default is `True`.
    Returns
    -------
    xl, xr : float
        Lowest and biggest value of the range.
    """
    if y is None:
        x1, x2 = np.min(x), np.max(x)
        if not extend:
            return x1, x2
    else:
        idx1, idx2 = arglimit(y, threshold=threshold)
        x1, x2 = sorted([x[idx1], x[idx2]])

    # calculate the padding
    if padding != 0.0:
        pad = (x2 - x1) * padding
        x1 -= pad
        x2 += pad

    if not extend:
        x1 = max(x1, np.min(x))
        x2 = min(x2, np.max(x))

    return x1, x2


def fwhm(x, y):
    """ Calculates the full width at half maximum of the distribution described
        by (x, y).
    """
    xl, xr = limit(x, y, threshold=0.5, padding=0.0)
    return np.abs(xr - xl)


def get_intensity(field):
    """
    Returns the intensity of a field
    """
    return np.abs(field) ** 2


def freq_bw_to_wavelength(bw_freq, center, center_domain):
    """
    Converts a frequency bandwidth to a wavelength bandwidth
    """
    if center_domain == 'frequency':
        return bw_freq * c / center ** 2
    elif center_domain == 'wavelength':
        return bw_freq * center ** 2 / c
    else:
        raise ValueError('center_domain must be either frequency or wavelength')


def wavelength_bw_to_frequency(bw_wavelength, center, center_domain):
    """
    Converts a frequency bandwidth to a wavelength bandwidth
    """
    if center_domain == 'wavelength':
        return bw_wavelength * c / center ** 2
    elif center_domain == 'frequency':
        return bw_wavelength * center ** 2 / c
    else:
        raise ValueError('center_domain must be either frequency or wavelength')


def energy_renormalization(intensity1, intensity2):
    """
    Renormalizes the energy of field2 to match field1
    """
    energy1 = np.sum(intensity1)
    energy2 = np.sum(intensity2)
    return intensity2 * energy1 / energy2


def energy_match(field, energy):
    return np.sqrt(energy * get_intensity(field) / np.sum(get_intensity(field))) * np.exp(1j * get_phase(field))

def calc_energy_expanded(field, domain_spacing, spot_area):
    return np.sum(get_intensity(field)) * domain_spacing * spot_area
def energy_match_expanded(field, energy, domain_spacing, spot_area):
    norm_E = np.sum(get_intensity(field)) * domain_spacing * spot_area
    return np.sqrt(energy / norm_E) * field
def frequency_to_wavelength(field_frequency, frequency, wavelength_vector_limits, central_wavelength=None,
                            sample_points=0, wavelength_vector=None):
    """
    Converts a frequency domain field to a wavelength domain field
    Normally assumes frequency vector with 0 centered field so takes central wavelength as input to adjust
    frequency vector
    Wavelength limits must be within the frequency vector limits when converted to wavelength
    """
    if central_wavelength is None:
        frequency_vec = frequency
    else:
        frequency_vec = frequency + c / central_wavelength
    if wavelength_vector is None and wavelength_vector_limits is None:
        raise ValueError('Must provide either wavelength_vector or wavelength_vector_limits')
    elif wavelength_vector is None:
        if sample_points == 0:
            sample_points = len(frequency_vec)

        # check wavelength limits
        if wavelength_vector_limits[0] < c / frequency_vec[-1]:
            raise ValueError('Lower wavelength limit is too small')
        if wavelength_vector_limits[1] > c / frequency_vec[0]:
            raise ValueError('Upper wavelength limit is too large')
        wavelength_vector = np.linspace(wavelength_vector_limits[0], wavelength_vector_limits[1], num=sample_points)
    else:
        if wavelength_vector[0] < c / frequency_vec[-1]:
            raise ValueError('Lower wavelength limit on input vector is too small')
        if wavelength_vector[-1] > c / frequency_vec[0]:
            raise ValueError('Upper wavelength limit on input vector is too large')

    spectrum = get_intensity(field_frequency)
    spectrum_freq_interp = interp1d(2 * np.pi * frequency_vec, spectrum)
    phase = np.unwrap(get_phase(field_frequency))
    spectrum_wavelength = (2 * np.pi * c / (wavelength_vector ** 2)) * spectrum_freq_interp(
        2 * np.pi * c / wavelength_vector)
    ph_freq_interp = interp1d(2 * np.pi * frequency_vec, phase)
    ph_wavelength = ph_freq_interp(2 * np.pi * c / wavelength_vector)

    # # check for negative values
    if np.any(spectrum_wavelength < 0):
        spectrum_wavelength = spectrum_wavelength + np.abs(np.min(spectrum_wavelength))

    return wavelength_vector, np.sqrt(spectrum_wavelength) * np.exp(1j * ph_wavelength)


def wavelength_to_frequency(field_wavelength, wavelength, frequency_vector):
    """
    Converts a wavelength domain field to a frequency domain field

    """

    # check wavelength limits
    # if wavelength[0] < c / frequency_vector[-1]:
    #     raise ValueError('Lower wavelength limit is too small')
    # if wavelength[1] > c / frequency_vector[0]:
    #     raise ValueError('Upper wavelength limit is too large')
    central_wavelength = calculate_com(field_wavelength, wavelength)
    angfreq_vector = 2 * np.pi * (frequency_vector + c / central_wavelength)
    angfreq_endpoints = [2 * np.pi * c / wavelength[-1], 2 * np.pi * c / wavelength[0]]
    indices = [np.argmin(np.abs(angfreq_vector - angfreq_endpoints[0])) + 1,
               np.argmin(np.abs(angfreq_vector - angfreq_endpoints[1])) - 1]
    spectrum_wavelength = get_intensity(field_wavelength)
    spectrum_wavelength_interp = interp1d(wavelength, spectrum_wavelength)
    phase = get_phase(field_wavelength)
    spectrum_angfreq = (2 * np.pi * c / (angfreq_vector[indices[0]:indices[1]] ** 2)) * spectrum_wavelength_interp(
        2 * np.pi * c / angfreq_vector[indices[0]:indices[1]])
    phase_wavelength_interp = interp1d(wavelength, phase)
    phase_angfreq = phase_wavelength_interp(2 * np.pi * c / angfreq_vector[indices[0]:indices[1]])
    field_angfreq = np.sqrt(spectrum_angfreq) * np.exp(1j * phase_angfreq)
    field_freq = np.ones(len(frequency_vector), dtype=complex) * field_angfreq[0]
    field_freq[indices[0]:indices[1]] = field_angfreq

    return field_freq


def calculate_energy(field, domain):
    '''
    Calculates the energy of a field.
    For frequency, should multiply by 2pi to get energy per unit frequency
    For wavelength, don't need any adjustment
    '''
    return np.sum(np.abs(field) ** 2 * (domain[1] - domain[0]))


def smooth(x, y):
    # resample to lots more points - needed for the smoothed curves
    x_smooth = np.linspace(x.min(), x.max(), len(x))

    # spline - always goes through all the data points x/y
    y_spline = interpolate.spline(x, y, x_smooth)

    spl = interpolate.UnivariateSpline(x, y)

    sigma = 2
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
    y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)


def calculate_peak_power(beam_area, pulse_width, pulse_energy=None, average_power=None, rep_rate=None):
    """
    Calculates the peak power of a beam.
    """
    if pulse_energy is None and average_power is not None and rep_rate is not None:
        pulse_energy = average_power / rep_rate
    elif pulse_energy is not None:
        pass
    else:
        raise ValueError('Either pulse_energy or average_power and rep_rate must be specified')

    return pulse_energy / (pulse_width * beam_area)


def calculate_average_power(pulse_energy, rep_rate):
    """
    Calculates the average power of a beam.
    """
    return pulse_energy * rep_rate


def calculate_peak_intensity(peak_power, spot_size):
    """
    Calculates the peak intensity of a beam.
    """
    return peak_power / spot_size


def calculate_fluence(pulse_energy, spot_size):
    """
    Calculates the fluence of a beam.
    """
    return pulse_energy / spot_size


def calculate_beam_area(radius):
    """
    Calculates the area of a beam.
    """
    return np.pi * radius ** 2


def calculate_pulse_energy(avg_power, rep_rate):
    """
    Calculates the pulse energy of a beam.
    """
    return avg_power / rep_rate


def calculate_com(field, domain):
    """
    Calculates the center of mass of a field.
    """
    return np.dot(np.abs(field) ** 2, domain) / np.sum(np.abs(field) ** 2)


def dopant_ion_concentration_to_density(dopant_ion_concentration, mass_density_dopant, substrate_molar_mass):
    """
    Calculates the density of dopant ions in a material. Assumes in the mass density of the material is 1 g/cm^3.
    Conversion from rp photonics. https://www.rp-photonics.com/doping_concentration.html
    Example: Yb:KGW is Yb3+:KGd(WO4)2, with a Yb3+ mass density of 6.5 g/cm^3.molar mass of KGW is about 692.0203 .
    """
    # TODO: make more general for mass or density
    return dopant_ion_concentration * mass_density_dopant / (substrate_molar_mass * 1.66e-24)


class UNITS:
    def __init__(self, mScale=0, sScale=0):
        self.m = 10 ** mScale
        self.mm = 10 ** (-3 * self.m)
        self.um = 10 ** (-6 * self.m)
        self.nm = 10 ** (-9 * self.m)

        self.s = 10 ** sScale
        self.ns = 10 ** (-9 * self.s)
        self.ps = 10 ** (-12 * self.s)
        self.fs = 10 ** (-15 * self.s)

        self.J = (self.m ** 2) / (self.s ** 2)
        self.mJ = 10 ** (-3 * self.J)
        self.uJ = 10 ** (-6 * self.J)
