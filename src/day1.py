"""Horizontal and Vertical Lines"""

import matplotlib.pylab as plt
from einops import rearrange
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
    return array


def generate_gray_image(rng_obj: np.random.Generator):
    array = np.ones(shape=(SIZE, SIZE))
    array = add_lines(array, (20, 40), SIZE // 40, rng_obj=rng_obj)
    array = add_lines(array, (4, 10), SIZE // 20, rng_obj=rng_obj)
    array = add_lines(array, (1, 3), SIZE // 10, rng_obj=rng_obj)
    return array


def repeat_channels(array: np.array, offset: int = 100):
    array_red = array.copy()
    array_green = np.roll(np.roll(array.copy(), offset, axis=0), offset, axis=1)
    array_blue = np.roll(np.roll(array.copy(), -offset, axis=0), -offset, axis=1)
    array = rearrange([array_red, array_green, array_blue], "b h w -> h w b")
    return array


SIZE = 500

rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    axis_rng = np.random.default_rng(rng.integers(1, 1e6, size=1)[0])
    image = generate_gray_image(axis_rng)
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("out/day1.png")
plt.show()

rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    axis_rng = np.random.default_rng(rng.integers(1, 1e6, size=1)[0])
    image = generate_gray_image(axis_rng)
    image = repeat_channels(image, offset=5)
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("out/day1_chrom.png")
plt.show()
