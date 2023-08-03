import numpy as np
from scipy.interpolate import interp1d
from scipy import signal


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


def re_im_sep(fields):
    shg1 = fields[0:1892] + fields[1892 * 2 + 348 : 1892 * 3 + 348] * 1j
    shg2 = fields[1892 : 1892 * 2] + fields[1892 * 3 + 348 : 1892 * 4 + 348] * 1j
    sfg = (
        fields[1892 * 2 : 1892 * 2 + 348]
        + fields[1892 * 4 + 348 : 1892 * 4 + 2 * 348] * 1j
    )

    return shg1, shg2, sfg


def change_domains(domain, field, new_domain, domain_type):
    padded_vector = np.pad(field, (1, 1), mode="constant")
    window = signal.windows.tukey(int(len(padded_vector) // 1))
    padded_vector = window * padded_vector

    # Extend old domain to match new domain
    alt_domain = np.append([new_domain[0]], domain)
    alt_domain = np.append(alt_domain, [new_domain[-1]])

    # Resample padded vector using new domain
    resampled_vector = resample_method1(alt_domain, new_domain, padded_vector)

    if domain_type == "freq":
        out_direct = ifft(field)
        out = ifft(resampled_vector)
    elif domain_type == "time":
        out_direct = fft(field)
        out = fft(resampled_vector)
    else:
        print("field type not supported")

    return out_direct, out


def change_domain_and_adjust_energy(
    domain,
    field,
    new_domain,
    domain_type,
    beam_area,
    domain_spacing,
    true_domain_spacing,
):
    out_direct, out = change_domains(domain, field, new_domain, domain_type)
    domain_spacing_calc = domain[1] - domain[0]
    pulse_energy_calc = calc_energy_expanded(field, domain_spacing_calc, beam_area)
    out_direct = energy_match_expanded(
        out_direct, pulse_energy_calc, domain_spacing, beam_area
    )
    out = energy_match_expanded(out, pulse_energy_calc, true_domain_spacing, beam_area)

    print(
        "direct, and resampled energies: ",
        calc_energy_expanded(out_direct, domain_spacing, beam_area),
        calc_energy_expanded(out, true_domain_spacing, beam_area),
    )

    return out_direct, out


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
