"""Horizontal and Vertical Lines"""

import matplotlib.pylab as plt
from typing import Tuple
import numpy as np


def add_lines(
    array: np.array,
    width_range: Tuple[int, int],
    num_lines: int,
    rng_obj: np.random.Generator,
):
    for y_pos in rng.integers(0, SIZE, size=num_lines):
        thickness = max(rng.integers(*width_range, size=1)[0] // 2, 1)
        start_pos, length = rng.integers(1, SIZE, size=2)
        array[(y_pos - thickness) : (y_pos), start_pos : (start_pos + length)] = (
            rng.random()
        )
    for x_pos in rng.integers(0, SIZE, size=num_lines):
        thickness = max(rng.integers(*width_range, size=1)[0] // 2, 1)
        start_pos, length = rng.integers(1, SIZE, size=2)
        array[start_pos : (start_pos + length), (x_pos - thickness) : (x_pos)] = (
            rng.random()
        )
    return image


SIZE = 500
rng = np.random.default_rng(1701)

fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    axis_rng = np.random.default_rng(rng.integers(1, 1e6, size=1)[0])
    image = np.ones(shape=(SIZE, SIZE))
    image = add_lines(image, (20, 40), SIZE // 40, rng_obj=axis_rng)
    image = add_lines(image, (4, 10), SIZE // 20, rng_obj=axis_rng)
    image = add_lines(image, (1, 3), SIZE // 10, rng_obj=axis_rng)
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("out/day1.png")
plt.show()
