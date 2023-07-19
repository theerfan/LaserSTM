import numpy as np
import matplotlib.pyplot as plt

from typing import List

# Generate a single gaussian distribution in an array of size n as input
def generate_gaussian(
    output_size: int, mean: float, std: float, amplitude: float = 1
) -> np.ndarray:
    return amplitude * np.random.normal(mean, std, output_size)


# Visualize the gaussian distribution
def visualize_gaussian(gaussian: np.ndarray):
    plt.hist(gaussian, bins=200)
    plt.show()


# Generate combination of gaussian distributions
# Input: size of the array, list of [mean], [std], [amplitude] and the number
# of gaussian distributions
# Output: array of size n with the combination of gaussian distributions
def generate_gaussian_combination(
    output_size: int, mean: float, std: float, amplitude: float, n: int
) -> np.ndarray:
    gaussian = np.zeros(output_size)
    for i in range(n):
        sign = np.random.choice([-1, 1])
        gaussian += sign * generate_gaussian(output_size, mean[i], std[i], amplitude[i])

    if min(gaussian) < 0:
        gaussian -= min(gaussian)
    else:
        pass
    return gaussian


a = generate_gaussian_combination(10000, [0, 0], [1, 1], [1, 1], 2)
visualize_gaussian(a)
