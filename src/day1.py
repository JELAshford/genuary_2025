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
    for y_pos in rng_obj.integers(0, SIZE, size=num_lines):
        thickness = max(rng_obj.integers(*width_range, size=1)[0] // 2, 1)
        start_pos, length = rng_obj.integers(1, SIZE, size=2)
        array[(y_pos - thickness) : (y_pos), start_pos : (start_pos + length)] = (
            rng_obj.random()
        )
    for x_pos in rng_obj.integers(0, SIZE, size=num_lines):
        thickness = max(rng_obj.integers(*width_range, size=1)[0] // 2, 1)
        start_pos, length = rng_obj.integers(1, SIZE, size=2)
        array[start_pos : (start_pos + length), (x_pos - thickness) : (x_pos)] = (
            rng_obj.random()
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


def draw_images(subplot_shape: Tuple[int, int], image_func, rng_obj, save_name: str):
    fig, axs = plt.subplots(*subplot_shape, figsize=(10, 10), squeeze=False)
    for ax in axs.flatten():
        ax.imshow(image_func(rng_obj), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name)
    plt.show()


SIZE = 500

# Grayscale lines
draw_images(
    (1, 1),
    generate_gray_image,
    np.random.default_rng(1701),
    "out/day1.png",
)

# Grayscale lines with chromatic shift
draw_images(
    (1, 1),
    lambda r: repeat_channels(generate_gray_image(r), offset=5),
    np.random.default_rng(1701),
    "out/day1_chrom.png",
)
