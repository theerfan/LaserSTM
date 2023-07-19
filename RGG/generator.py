import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from math import factorial


def taylor_expansion(
    init_phases: List[float], omega: float, omega_0: float = 0
) -> float:
    n = len(init_phases)
    total_phase = init_phases[0]
    for i in range(1, n):
        total_phase += init_phases[i] * (omega - omega_0) / factorial(i)
    return total_phase


# Generate a single gaussian distribution in an array of size n as input
def generate_gaussian(
    output_size: int,
    mean: float,
    std: float,
    init_phases: List[float],
    energy: float = 1,
) -> dict:
    frequencies = np.random.normal(mean, std, output_size)
    phases = np.array([taylor_expansion(init_phases, omega) for omega in frequencies])
    return {
        "frequencies": frequencies,
        "phases": phases,
        "energy": energy,
    }


# Visualize the gaussian distribution
def visualize_gaussian(gaussian: np.ndarray):
    plt.hist(gaussian, bins=200)
    plt.show()


# Generate combination of gaussian distributions
# Input: size of the array, list of [mean], [std], [amplitude] and the number
# of gaussian distributions
# Output: array of size n with the combination of gaussian distributions
def generate_gaussian_combination(
    output_size: int,
    means: List[float],
    stds: List[float],
    energies: List[float],
) -> dict:
    # Make sure the length of the list are the same
    assert len(means) == len(stds) == len(energies)
    n = len(means)
    total_gaussian = {
        "frequencies": np.zeros(output_size),
        "phases": np.zeros(output_size),
        "energy": 0,
    }
    for i in range(n):
        sign = np.random.choice([-1, 1])
        current_gaussian = generate_gaussian(
            output_size, means[i], stds[i], energies[i]
        )
        total_gaussian["frequencies"] += sign * current_gaussian["frequencies"]
        total_gaussian["phases"] += sign * current_gaussian["phases"]
        total_gaussian["energy"] += sign * current_gaussian["energy"]

    if min(total_gaussian["frequencies"]) < 0:
        total_gaussian["frequencies"] -= min(total_gaussian["frequencies"])
    else:
        pass

    if min(total_gaussian["energies"]) < 0:
        total_gaussian["energies"] -= min(total_gaussian["energies"])
    else:
        pass

    return total_gaussian


a = generate_gaussian_combination(10000, [0, 0], [1, 1], [1, 1])
visualize_gaussian(a)
