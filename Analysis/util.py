import numpy as np
from scipy.interpolate import interp1d
from scipy import signal


def get_phase(field):
    return np.arctan2(np.imag(field), np.real(field))


def intrepolate_vector(input_domain, target_domain, input_vector, method="linear"):
    try:
        f = interp1d(input_domain, input_vector, kind=method)
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


def calc_energy_expanded(
    field: np.ndarray, domain_spacing: np.ndarray, spot_area: float
) -> float:
    return np.sum(get_intensity(field)) * domain_spacing * spot_area


def normalize_expanded_energy(
    field: np.ndarray, energy: np.ndarray, domain_spacing: float, spot_area: float
) -> np.ndarray:
    norm_E = calc_energy_expanded(field, domain_spacing, spot_area)
    return np.sqrt(energy / norm_E) * field


def re_im_combined(fields):
    shg1 = fields[0:1892] + fields[1892 * 2 + 348 : 1892 * 3 + 348] * 1j
    shg2 = fields[1892 : 1892 * 2] + fields[1892 * 3 + 348 : 1892 * 4 + 348] * 1j
    sfg = (
        fields[1892 * 2 : 1892 * 2 + 348]
        + fields[1892 * 4 + 348 : 1892 * 4 + 2 * 348] * 1j
    )

    return shg1, shg2, sfg


# domain_type is the original one
def change_domains(domain, field, new_domain, domain_type):
    # pad the "downsampled" version with zeros so we could do extrapolation
    padded_vector = np.pad(field, (1, 1), mode="constant")
    # tukey filter smoothes out the wavefunction
    # TODO: Is doing the window here necessary?
    window = signal.windows.tukey(int(len(padded_vector) // 1))
    padded_vector = window * padded_vector

    # Extend old domain to match new domain
    # This is to make sure the domains of the stretched (padded) vector
    # match what it should be in reality
    alt_domain = np.concatenate(([new_domain[0]], domain, [new_domain[-1]]))

    # Resample padded vector using new domain
    resampled_vector = intrepolate_vector(alt_domain, new_domain, padded_vector)

    if domain_type == "freq":
        out_direct = ifft(field)
        out = ifft(resampled_vector)
    elif domain_type == "time":
        out_direct = fft(field)
        out = fft(resampled_vector)
    else:
        out_direct = field
        out = resampled_vector
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

    # Calculate the energy for the original field
    pulse_energy_calc = calc_energy_expanded(field, domain_spacing_calc, beam_area)

    # then normalize the energies of the resampled and non-resampled vectors from the
    # time domain to the energy of the field because they should be the same
    out_direct = normalize_expanded_energy(
        out_direct, pulse_energy_calc, domain_spacing, beam_area
    )
    out = normalize_expanded_energy(
        out, pulse_energy_calc, true_domain_spacing, beam_area
    )

    print(
        "direct, and resampled energies: ",
        calc_energy_expanded(out_direct, domain_spacing, beam_area),
        calc_energy_expanded(out, true_domain_spacing, beam_area),
    )

    return out_direct, out
