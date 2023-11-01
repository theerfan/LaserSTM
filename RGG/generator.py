import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from math import factorial

# Taylor expansion of the spectral phase
def taylor_expansion(
    init_phases: List[float], omega: float, omega_0: float = 0
) -> float:
    n = len(init_phases)
    total_phase = 0
    for i in range(n):
        total_phase += init_phases[i] * (omega - omega_0) ** i / factorial(i)
    return total_phase


# Generate a single gaussian distribution in an array of size n as input
def generate_gaussian(
    output_size: int,
    mean: float,
    std: float,
    energy: float = 1,
) -> dict:
    frequencies = np.random.normal(mean, std, output_size)
    return {
        "frequencies": frequencies,
        # "phases": phases,
        "energy": energy,
    }


# Visualize the gaussian distribution
def visualize_gaussian(gaussian: np.ndarray):
    # plt.hist(, bins=200, label="w", color="blue")
    # plt.hist(, bins=200, label="E", color="green")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Create the vertical histogram on the first axes
    axs[0].hist(gaussian["frequencies"], bins=200, alpha=0.5, label='Freq (V)', color='blue')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Occurance')
    axs[0].legend(loc='upper right')

    # Create the horizontal histogram on the second axes
    axs[1].hist(gaussian["phases"], bins=200, alpha=0.5, label='Phases (H)', color='green', orientation='horizontal')
    axs[1].set_xlabel('Occurance')
    axs[1].set_ylabel('Value')
    axs[1].legend(loc='upper right')

    axs[1].text(0.95, 0.0, f'Energy = {gaussian["energy"]}', horizontalalignment='right',
         verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.show()
    plt.savefig("gaussian.png")


# Fix the minimum value of the gaussian distribution to 0
def fix_min(gaussian: np.ndarray):
    if mini := np.min(gaussian) < 0:
        gaussian -= mini
    else:
        pass
    return gaussian


# Generate combination of gaussian distributions
# Input: size of the array, list of [mean], [std], [amplitude] and the number
# of gaussian distributions
# Output: array of size n with the combination of gaussian distributions
def generate_gaussian_combination(
    output_size: int,
    means: List[float],
    stds: List[float],
    energies: List[float],
    init_phases: List[float],
) -> dict:
    # Make sure the length of the list are the same
    assert len(means) == len(stds)
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
        total_gaussian["energy"] += sign * current_gaussian["energy"]

    total_gaussian["frequencies"] = fix_min(total_gaussian["frequencies"])
    total_gaussian["energy"] = fix_min(total_gaussian["energy"])

    total_gaussian["phases"] = np.array(
        [
            taylor_expansion(init_phases, omega)
            for omega in total_gaussian["frequencies"]
        ]
    )

    return total_gaussian


a = generate_gaussian_combination(10000, means=[0.2, 0.1], stds=[2, 1], energies=[0.1, 1], init_phases=[0.1, 0.2])
visualize_gaussian(a)
